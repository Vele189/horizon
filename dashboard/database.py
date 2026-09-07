"""The dashboard's only route to the warehouse.

Two things make this layer worth its own module rather than a helper at the
top of ``app.py``: where the connection string comes from, and what happens
when the database on the other end is asleep.

Where the connection string comes from
--------------------------------------

Nowhere in this file, and nowhere in the repository. The deployed app reads
``DATABASE_URL`` from **Streamlit secrets**, which Streamlit Community Cloud
supplies from its own encrypted store and which never enters git;
``.streamlit/secrets.toml`` is git-ignored and only ``.streamlit/secrets.toml.example``
is committed. A test parses this module and fails on any string literal that
looks like a connection string, so "never hardcoded" is enforced rather than
remembered.

Locally there are no Streamlit secrets, and the fallback is ``config.py`` —
still the only module in the project that reads the process environment. It is
asked for ``SERVING_DATABASE_URL`` rather than ``DATABASE_URL``, because those
two names mean different databases on a development machine: ``DATABASE_URL``
is the Docker container holding bronze, silver and gold, and the dashboard has
no business reading bronze. Neon is the serving tier, and the dashboard is a
serving-tier reader in both places it runs. The full order is in
:func:`resolve_database_url`.

What happens when the database is asleep
----------------------------------------

Neon scales compute to zero after five minutes idle and resumes on the next
query, so a visitor arriving after a quiet afternoon is the *normal* case, not
an edge case. It shows up in two different ways and both are handled here.

**The wait.** Resuming costs roughly 1.2 s on top of the connection (§Promotion
to Neon in the README has the measurements). That is a delay, not an error, and
it needs to look like one: every cached query declares a spinner, which
Streamlit shows only on a cache miss — precisely the path that can be slow.

**The dead socket.** This is the failure the ticket is really about. A pooled
connection opened before the compute suspended is not gracefully closed; it is
a file descriptor pointing at nothing, and the next ``execute`` on it raises
``OperationalError`` from inside the driver — which, unhandled, is a red
traceback on a public URL. Three things stop that:

* ``pool_pre_ping`` — SQLAlchemy tests a pooled connection before lending it
  out and transparently replaces a dead one. This catches most of it.
* ``pool_recycle`` below Neon's idle timeout — a connection is discarded before
  it is old enough to have been suspended under it, so pre-ping usually has
  nothing to find.
* An explicit retry, because the first two are not sufficient. Pre-ping's own
  probe can fail while compute is still coming back, and that raises the same
  exception the query would have. So a transient failure disposes the pool —
  dropping every stale socket, not just the one that failed — waits, and tries
  again.

The retry is deliberately narrow. Only connection-shaped failures are retried;
a syntax error or a missing column is raised on the first attempt, because
retrying a query that cannot succeed turns a clear error into a slow one.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping

import pandas as pd
import streamlit as st
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ConfigError, get_settings, mask_secret, split_database_url  # noqa: E402

__all__ = [
    "CACHE_TTL_SECONDS",
    "CACHE_TTL_REFERENCE_SECONDS",
    "GOLD_SCHEMA",
    "DashboardConfigError",
    "Source",
    "WarehouseUnreachable",
    "clear_caches",
    "connection_label",
    "get_engine",
    "resolve_database_url",
    "run_query",
    "warehouse_status",
]

log = logging.getLogger(__name__)

# The only schema promoted to Neon. Named here so a view cannot reach for a
# layer that does not exist on the serving side.
GOLD_SCHEMA: Final[str] = "gold_marts"

# The secret the deployed app reads, and the local variable it falls back to.
# Both are names, not values; the values live in Streamlit secrets and .env.
SECRET_KEY: Final[str] = "DATABASE_URL"
SECRET_KEY_SERVING: Final[str] = "SERVING_DATABASE_URL"

# How long a query result stays good.
#
# The marts change only when `serving/promote.py` runs, which is a manual step
# taken at most once a day — so staleness is not the constraint. Neon's free
# plan is: 100 compute-hours a month, and compute stays awake for five minutes
# after each query. A query that misses the cache therefore costs a *five
# minute* minimum of the allowance no matter how fast it runs, which makes the
# meter count wake-ups rather than queries. Six hours between refreshes is four
# wake-ups a day, twenty minutes of compute, about ten hours a month — a tenth
# of the allowance spent on keeping the dashboard current, and the rest left
# for people actually looking at it. Streamlit's cache is per process and
# Community Cloud runs one, so this is four wake-ups a day in total, not four
# per visitor.
#
# The cost of the long TTL is that a promotion is invisible for up to six
# hours, so the sidebar carries a button that clears these caches.
CACHE_TTL_SECONDS: Final[int] = 6 * 60 * 60

# Dimensions — the fifteen cities, the date spine — change when the project
# changes, not when it runs.
CACHE_TTL_REFERENCE_SECONDS: Final[int] = 24 * 60 * 60

# Neon suspends after five minutes. Recycling below that means the pool never
# hands out a connection old enough to have been suspended under it.
POOL_RECYCLE_SECONDS: Final[int] = 240

# One attempt to find the socket dead, one for the resume, one for bad luck.
COLD_START_ATTEMPTS: Final[int] = 3
COLD_START_BACKOFF_SECONDS: Final[float] = 1.5

SPINNER_MESSAGE: Final[str] = "Reading the warehouse — Neon resumes from idle in about a second…"


class DashboardConfigError(RuntimeError):
    """No connection string was found, in any of the places one may live."""


class WarehouseUnreachable(RuntimeError):
    """The warehouse did not answer, after the cold-start retries were spent."""


@dataclass(frozen=True)
class Source:
    """Where the connection string came from. For the sidebar, and for tests."""

    url: str
    origin: str

    @property
    def masked(self) -> str:
        return mask_secret(self.url)

    @property
    def host(self) -> str:
        return split_database_url(self.url, name=self.origin).host


def _secret(key: str) -> str | None:
    """Read one Streamlit secret, tolerating the absence of a secrets file.

    ``st.secrets`` raises rather than returning empty when no ``secrets.toml``
    exists anywhere, which is the ordinary state of a development machine. That
    is a missing file, not a missing secret, so it is caught and treated as
    "not set" — otherwise every local run would fail before reaching the .env
    that does have the answer.
    """
    try:
        value = st.secrets.get(key)
    except Exception:  # noqa: BLE001 - any secrets failure means "not configured"
        return None
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def resolve_database_url() -> Source:
    """Find the serving warehouse, and say where the answer came from.

    In order, stopping at the first that is set:

    1. ``DATABASE_URL`` in Streamlit secrets. What Community Cloud injects, and
       the only database that exists in the deployed environment.
    2. ``SERVING_DATABASE_URL`` in Streamlit secrets, so a deployment can use
       the same variable name the promotion script does.
    3. ``SERVING_DATABASE_URL`` from ``config.py`` — Neon, on a development
       machine, which is the target the dashboard must be developed against.
    4. ``DATABASE_URL`` from ``config.py``, **only** when ``ENVIRONMENT`` is
       ``serving``. That is the project's existing switch for "this machine's
       DATABASE_URL is the serving tier"; honouring it here avoids inventing a
       second convention that could disagree with the first.

    Raises:
        DashboardConfigError: None of the four is set, with the fix for both
            the deployed and the local case spelled out.
    """
    for key, origin in ((SECRET_KEY, "Streamlit secrets"), (SECRET_KEY_SERVING, "Streamlit secrets")):
        found = _secret(key)
        if found:
            return Source(found, f"{key} ({origin})")

    try:
        settings = get_settings()
    except ConfigError as exc:  # pragma: no cover - config refuses to build
        raise DashboardConfigError(str(exc)) from exc

    if settings.serving_database_url:
        return Source(settings.serving_database_url, f"{SECRET_KEY_SERVING} (.env)")

    if settings.environment == "serving" and settings.database_url:
        return Source(settings.database_url, f"{SECRET_KEY} (.env, ENVIRONMENT=serving)")

    raise DashboardConfigError(
        "No serving database is configured.\n\n"
        "Deployed: set DATABASE_URL in the app's Streamlit secrets "
        "(Manage app → Settings → Secrets).\n"
        "Locally: set SERVING_DATABASE_URL in .env to the Neon connection "
        "string — see .env.example and .streamlit/secrets.toml.example."
    )


@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    """One engine per process, shared by every session.

    Cached as a *resource* rather than data: an Engine owns sockets and threads,
    so it must be handed out as the same object rather than copied per session,
    and it must not be serialised.
    """
    source = resolve_database_url()
    settings = get_settings()
    log.info("dashboard connecting to %s", source.masked)
    return create_engine(
        source.url,
        future=True,
        pool_pre_ping=True,
        pool_recycle=POOL_RECYCLE_SECONDS,
        # A public dashboard should hold almost nothing open against a database
        # billed by compute time. Streamlit renders one script run at a time per
        # session, so concurrency here is visitors, not queries per visitor.
        pool_size=2,
        max_overflow=3,
        connect_args={
            "connect_timeout": settings.request_connect_timeout_seconds,
            # Identifies dashboard traffic in Neon's monitoring, so a compute
            # spike can be attributed to visitors rather than to a promotion.
            "application_name": "horizon-dashboard",
        },
    )


def _is_transient(exc: Exception) -> bool:
    """Is this the database being asleep, or the query being wrong?

    Only the former is worth retrying. ``OperationalError`` and
    ``InterfaceError`` are the driver saying it could not talk to the server —
    a suspended compute, a dropped socket, a refused connection. A
    ``ProgrammingError`` for a column that does not exist is none of those and
    will fail identically on every attempt.

    ``connection_invalidated`` is checked as well: SQLAlchemy sets it when the
    pool discards a connection mid-statement, which is exactly what a Neon
    suspension looks like from inside a query.
    """
    if isinstance(exc, (OperationalError, InterfaceError)):
        return True
    return isinstance(exc, DBAPIError) and bool(exc.connection_invalidated)


def _execute(sql: str, params: Mapping[str, Any] | None) -> pd.DataFrame:
    """Run one query, surviving a cold start rather than reporting it."""
    engine = get_engine()
    last: Exception | None = None

    for attempt in range(1, COLD_START_ATTEMPTS + 1):
        try:
            with engine.connect() as connection:
                return pd.read_sql_query(text(sql), connection, params=dict(params or {}))
        except Exception as exc:  # noqa: BLE001 - re-raised below unless transient
            if not _is_transient(exc):
                raise
            last = exc
            log.warning(
                "warehouse unreachable on attempt %d/%d (%s); disposing pool and retrying",
                attempt,
                COLD_START_ATTEMPTS,
                type(exc).__name__,
            )
            # Dispose rather than let the pool heal one connection at a time:
            # if compute suspended, *every* pooled socket is dead, and a
            # per-connection recovery would pay the same failure again on the
            # next checkout.
            engine.dispose()
            if attempt < COLD_START_ATTEMPTS:
                time.sleep(COLD_START_BACKOFF_SECONDS * attempt)

    raise WarehouseUnreachable(
        f"The serving warehouse did not answer after {COLD_START_ATTEMPTS} attempts."
    ) from last


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=SPINNER_MESSAGE)
def _query_marts(sql: str, key: tuple[tuple[str, Any], ...]) -> pd.DataFrame:
    return _execute(sql, dict(key))


@st.cache_data(ttl=CACHE_TTL_REFERENCE_SECONDS, show_spinner=SPINNER_MESSAGE)
def _query_reference(sql: str, key: tuple[tuple[str, Any], ...]) -> pd.DataFrame:
    return _execute(sql, dict(key))


def run_query(
    sql: str,
    params: Mapping[str, Any] | None = None,
    *,
    reference: bool = False,
) -> pd.DataFrame:
    """Run a read query against the serving warehouse, cached.

    Args:
        sql: A ``select``. Bind parameters as ``:name``; never format values in.
        params: Values for those binds.
        reference: True for dimension tables, which get the longer TTL because
            they change when the project changes rather than when it runs.

    Returns:
        A DataFrame. Streamlit hands back a copy, so a view may modify it
        without corrupting the cache for the next visitor.

    Raises:
        WarehouseUnreachable: The database did not answer within the retries.
    """
    # A dict is not reliably hashable as a cache key across Streamlit versions,
    # and two dicts with the same pairs in a different order must not become two
    # cache entries. Sorted pairs make the key canonical.
    key = tuple(sorted((params or {}).items()))
    fetch = _query_reference if reference else _query_marts
    return fetch(sql, key)


def clear_caches() -> None:
    """Drop cached query results, keeping the engine.

    What the sidebar's refresh button calls after a promotion. The engine is
    deliberately kept: the data is stale, the connection is not.
    """
    _query_marts.clear()
    _query_reference.clear()
    warehouse_status.clear()


def connection_label() -> str:
    """The host the dashboard is reading, with the credentials removed."""
    return resolve_database_url().host


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=SPINNER_MESSAGE)
def warehouse_status() -> dict[str, Any]:
    """One round trip that answers "is it up, and how fresh is it".

    Deliberately a single query rather than one per fact. It runs on the first
    render of a session, so it is the query that pays the cold start — making
    it two would pay it twice for the same information.
    """
    frame = _execute(
        f"""
        select
            current_database()                                        as database,
            now()                                                     as server_time,
            (select count(*) from {GOLD_SCHEMA}.dim_cities)           as cities,
            (select max(date_key)
               from {GOLD_SCHEMA}.fact_weather_observations)          as latest_observation,
            (select max(forecast_date)
               from {GOLD_SCHEMA}.fact_ml_predictions)                as latest_forecast
        """,
        None,
    )
    return frame.iloc[0].to_dict()
