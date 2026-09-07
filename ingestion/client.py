"""HTTP client for the Open-Meteo historical archive (ERA5).

One public function — :func:`fetch_observations` — takes a city, a date range,
and a grain, and returns one validated :class:`ArchiveResponse`. Chunking a
30-year backfill into requests is ING-02's job; landing rows is ING-04's. This
module owns exactly one thing: getting a single request to succeed, or failing
in a way that says why.

Why this file is more defensive than a wrapper around ``requests.get``:

*   **The backfill is long.** Fifteen cities × thirty years of daily data plus
    twenty-four months of hourly is hundreds of requests running for hours. At
    that length, a transient 502 is not a possibility but a certainty, and a
    run that dies at hour three with no diagnosis costs a day.
*   **Silence is worse than failure.** Open-Meteo's defaults are local time and
    metric units. A request that forgets ``timezone=UTC`` still returns 200 and
    still looks like weather — it just has the day boundaries shifted, which
    would poison a 30-year climatological baseline in a way no downstream test
    would obviously catch. So every unit and the timezone are asked for
    explicitly *and* verified on the way back.
*   **A hung socket is not an error.** ``requests`` applies no timeout unless
    told to. A connection that opens and then stops sending would block the
    run forever, with no traceback to show for it.

Error model — one family, so ING-02 catches :class:`ArchiveError` and nothing
else::

    ArchiveError
    ├── ArchiveRetryableError     retried with backoff, then re-raised
    │   ├── ArchiveTransportError connection refused, timeout, truncated body
    │   ├── ArchiveServerError    5xx
    │   └── ArchiveRateLimited    429, carries Retry-After
    ├── ArchiveRequestError       4xx — a bug in the request, never retried
    └── ArchiveResponseError      200, but not the data we asked for

Usage::

    from ingestion.client import fetch_observations

    response = fetch_observations("london", date(2023, 1, 1), date(2023, 12, 31))
    for row in response.rows():
        ...

Run ``python ingestion/client.py --city london --year 2023`` to fetch one
city-year and print the provenance and unit table for eyeball comparison
against https://open-meteo.com/en/docs/historical-weather-api.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Final, Iterator, Literal, Mapping

import requests
from requests.adapters import HTTPAdapter
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tenacity.wait import wait_base

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cities import City, get_city  # noqa: E402
from config import Settings, get_settings  # noqa: E402

__all__ = [
    "ArchiveError",
    "ArchiveRateLimited",
    "ArchiveRequestError",
    "ArchiveResponse",
    "ArchiveResponseError",
    "ArchiveRetryableError",
    "ArchiveServerError",
    "ArchiveTransportError",
    "DAILY_UNITS",
    "DAILY_VARIABLES",
    "HOURLY_UNITS",
    "HOURLY_VARIABLES",
    "Grain",
    "LONG_RATE_LIMIT_WINDOWS",
    "build_session",
    "fetch_observations",
    "parse_payload",
    "request_url",
]

log = logging.getLogger(__name__)

Grain = Literal["daily", "hourly"]

# ---------------------------------------------------------------------------
# What we ask for, and what it must come back as
# ---------------------------------------------------------------------------
# Each mapping is variable name -> the unit string the API is expected to
# report for it. Both halves were read off a live response on 2026-09-07 and
# diffed against the archive documentation; the variable names match the
# columns of bronze_raw.observations_daily / _hourly one for one, so a row can
# be traced back to the request that produced it.
#
# The unit strings are not decoration. They are asserted on every response, so
# the day Open-Meteo changes a default the run stops rather than quietly
# landing kilometres per hour into a column commented "km/h".

DAILY_UNITS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "temperature_2m_max": "°C",
        "temperature_2m_min": "°C",
        "temperature_2m_mean": "°C",
        "apparent_temperature_max": "°C",
        "apparent_temperature_min": "°C",
        "apparent_temperature_mean": "°C",
        "precipitation_sum": "mm",
        "rain_sum": "mm",
        "snowfall_sum": "cm",  # centimetres, not millimetres — the one trap here
        "precipitation_hours": "h",
        "wind_speed_10m_max": "km/h",
        "wind_speed_10m_mean": "km/h",
        "wind_gusts_10m_max": "km/h",
        "wind_direction_10m_dominant": "°",
        "surface_pressure_mean": "hPa",
        "pressure_msl_mean": "hPa",
        "relative_humidity_2m_mean": "%",
        "dew_point_2m_mean": "°C",
        "cloud_cover_mean": "%",
        "shortwave_radiation_sum": "MJ/m²",
        "weather_code": "wmo code",
    }
)

HOURLY_UNITS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "temperature_2m": "°C",
        "apparent_temperature": "°C",
        "relative_humidity_2m": "%",
        "dew_point_2m": "°C",
        "surface_pressure": "hPa",
        "pressure_msl": "hPa",
        "wind_speed_10m": "km/h",
        "wind_gusts_10m": "km/h",
        "wind_direction_10m": "°",
        "precipitation": "mm",
        "cloud_cover": "%",
        "weather_code": "wmo code",
    }
)

# Derived rather than written twice, so the request list and the unit contract
# can never drift apart.
DAILY_VARIABLES: Final[tuple[str, ...]] = tuple(DAILY_UNITS)
HOURLY_VARIABLES: Final[tuple[str, ...]] = tuple(HOURLY_UNITS)

_UNITS_BY_GRAIN: Final[Mapping[str, Mapping[str, str]]] = MappingProxyType(
    {"daily": DAILY_UNITS, "hourly": HOURLY_UNITS}
)

# Every unit-bearing parameter is sent explicitly. These are Open-Meteo's
# current defaults, which is exactly why they are stated: a default that
# changes upstream is a silent data corruption, a parameter that changes
# upstream is a loud one.
_UNIT_PARAMS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "timeformat": "iso8601",
        # Coastal cities — Singapore, Lagos, Sydney, Cape Town — sit next to
        # sea cells whose temperature series is materially different. "land"
        # is the default, but it is the parameter that decides whether Lagos
        # is Lagos or the Bight of Benin.
        "cell_selection": "land",
    }
)

# `elevation` is deliberately not sent. Omitting it leaves Open-Meteo's 90 m
# DEM downscaling in place; passing the city's configured elevation would
# override that and change the values returned. Whatever elevation actually
# answered is recorded per row as api_elevation_m.

# The ERA5 archive begins here. Anything earlier is a caller bug, not a
# request worth spending a round trip on.
ARCHIVE_START: Final[dt.date] = dt.date(1940, 1, 1)

# Retry-After is attacker-ish input in the sense that matters here: a proxy
# misconfiguration returning "Retry-After: 86400" must not park the backfill
# for a day. Honour the header, but not past this.
MAX_RETRY_AFTER_SECONDS: Final[float] = 300.0

# What to wait when a 429 arrives with no Retry-After header at all. Measured
# on 2026-09-07: Open-Meteo rate-limits on *weighted* API calls rather than
# HTTP requests, and answers an overrun with a bare 429 whose body reads
# "Minutely API request limit exceeded. Please try again in one minute." No
# header, so the ordinary exponential backoff applied — and 2s, 6s, 10s, 16s
# never spans the minute the server is actually asking for, which burns every
# attempt for nothing. A rate limit is not a transient blip to feel out
# gradually; the server has stated its window, so wait it out.
RATE_LIMIT_FALLBACK_SECONDS: Final[float] = 60.0

# The body also says *which* allowance was spent — minutely, hourly, or daily —
# and that changes what to do about it. A minute is worth waiting out inside
# the request. An hour is not: the backfill's manifest makes resuming free, so
# four sixty-second retries only delay the inevitable stop by four minutes and
# teach the caller nothing. Observed during the ING-05 backfill, which spent
# the hourly allowance and then burned four attempts discovering it.
_RATE_LIMIT_WINDOWS: Final[tuple[str, ...]] = ("minutely", "hourly", "daily")

#: Windows too long to wait out inside a single request.
LONG_RATE_LIMIT_WINDOWS: Final[frozenset[str]] = frozenset({"hourly", "daily"})

# Ceiling on the exponential backoff between ordinary retries.
MAX_BACKOFF_SECONDS: Final[float] = 60.0

# Open-Meteo asks non-commercial users to identify themselves so abuse can be
# traced to a project rather than an IP.
USER_AGENT: Final[str] = (
    "climate-volatility-risk-engine/0.1 (+https://github.com/; contact via repo)"
)

# How much of an unexpected response body to quote in an exception message.
_BODY_EXCERPT_CHARS: Final[int] = 500


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ArchiveError(RuntimeError):
    """Base class for every failure reaching the caller from this module."""


class ArchiveRetryableError(ArchiveError):
    """A failure worth trying again. Raised after the retries are spent."""


class ArchiveTransportError(ArchiveRetryableError):
    """The request never completed: refused, timed out, or truncated."""


class ArchiveServerError(ArchiveRetryableError):
    """The API returned 5xx."""

    def __init__(self, message: str, *, status_code: int, url: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.url = url


class ArchiveRateLimited(ArchiveRetryableError):
    """The API returned 429.

    ``retry_after`` is the parsed header in seconds, or ``None`` when the
    response omitted it or sent something unparseable — in which case the wait
    strategy falls back to :data:`RATE_LIMIT_FALLBACK_SECONDS` rather than to
    exponential backoff. Open-Meteo sends no header, so this is the usual path
    rather than the exotic one.

    ``limit_window`` is which allowance the body says was spent — ``minutely``,
    ``hourly``, ``daily``, or ``None`` when it does not say. Only a minutely
    limit is worth waiting out inside the request.
    """

    def __init__(
        self,
        message: str,
        *,
        url: str,
        retry_after: float | None = None,
        limit_window: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = 429
        self.url = url
        self.retry_after = retry_after
        self.limit_window = limit_window

    @property
    def waitable(self) -> bool:
        """Can this clear within a request's retry budget?"""
        return self.limit_window not in LONG_RATE_LIMIT_WINDOWS


class ArchiveRequestError(ArchiveError):
    """The API returned a non-retryable 4xx.

    A bad coordinate, an unknown variable, or a date outside the archive will
    fail identically on every attempt, so this is raised on the first one. The
    response body is carried in the message because Open-Meteo's ``reason``
    field is genuinely diagnostic — it names the offending parameter.
    """

    def __init__(self, message: str, *, status_code: int, url: str, body: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.url = url
        self.body = body


class ArchiveResponseError(ArchiveError):
    """A 200 that is not the data that was asked for.

    Wrong units, a non-UTC offset, a missing variable, or a row count that does
    not match the requested range. Never retried: repeating the request would
    produce the same wrong answer.
    """


# ---------------------------------------------------------------------------
# The parsed response
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchiveResponse:
    """One validated archive response, with its provenance attached."""

    city_id: str
    grain: Grain
    start: dt.date
    end: dt.date
    #: The exact request URL, query string included. Bronze stores this per row
    #: so any landed observation can be replayed against the API.
    url: str
    #: The ERA5 grid cell that actually answered, which is not the coordinate
    #: that was asked for — London's 51.5074/-0.1278 resolves to
    #: 51.4938/-0.1630 at 16 m.
    latitude: float
    longitude: float
    elevation_m: float | None
    times: tuple[dt.datetime, ...]
    values: Mapping[str, tuple[float | int | None, ...]]
    units: Mapping[str, str]
    generation_time_ms: float | None
    #: The decoded response, untouched. ING-03 archives this as gzip to disk.
    payload: Mapping[str, Any]

    def __len__(self) -> int:
        return len(self.times)

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(self.values)

    def null_counts(self) -> dict[str, int]:
        """Nulls per variable — how a partial grid cell announces itself."""
        return {
            name: sum(1 for v in series if v is None)
            for name, series in self.values.items()
        }

    def rows(self) -> Iterator[dict[str, Any]]:
        """One dict per timestamp, keyed to match the bronze columns.

        Carries the provenance this module knows about. ``ingested_at`` and
        ``batch_id`` belong to the run, not the response, and are added by the
        loader.
        """
        for index, observation_time in enumerate(self.times):
            row: dict[str, Any] = {
                "city_id": self.city_id,
                "observation_time": observation_time,
                "source_url": self.url,
                "api_latitude": self.latitude,
                "api_longitude": self.longitude,
                "api_elevation_m": self.elevation_m,
            }
            for name, series in self.values.items():
                row[name] = series[index]
            yield row


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def build_session() -> requests.Session:
    """A session with connection pooling and urllib3's own retries disabled.

    Retrying is tenacity's job and only tenacity's: two independent retry
    layers would multiply into far more requests than either intends, and
    urllib3's layer cannot see a 429's Retry-After the way the wait strategy
    below does.

    Reuse one session across a backfill. Reconnecting and renegotiating TLS on
    every one of several hundred requests is a measurable share of the runtime.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    adapter = HTTPAdapter(pool_connections=4, pool_maxsize=4, max_retries=0)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


def _rate_limit_window(body: str) -> str | None:
    """Which allowance a 429 body says was spent, if it says."""
    lowered = body.lower()
    for window in _RATE_LIMIT_WINDOWS:
        if f"{window} api request limit" in lowered:
            return window
    return None


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header, in either of its two legal forms.

    RFC 9110 permits delay-seconds or an HTTP-date. Returns ``None`` for an
    absent or unparseable header so the caller can fall back to backoff rather
    than guess.
    """
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None

    try:
        return max(0.0, float(int(value)))
    except ValueError:
        pass

    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    return max(0.0, (when - dt.datetime.now(dt.timezone.utc)).total_seconds())


class _wait_archive(wait_base):
    """Exponential backoff with jitter, except on 429, which is told what to do.

    Blind exponential backoff on a rate limit is the wrong answer twice over:
    it can retry sooner than the server allows, earning another 429, and it can
    wait far longer than needed once the window has reset. When the server
    states a delay, use it.

    Jitter matters because the backfill walks the same fifteen cities in a
    loop. Without it, everything that failed against one upstream blip retries
    in lockstep and reproduces the blip.
    """

    def __init__(self, backoff_seconds: float) -> None:
        self._exponential = wait_exponential_jitter(
            initial=backoff_seconds,
            max=MAX_BACKOFF_SECONDS,
            exp_base=2,
            jitter=backoff_seconds,
        )

    def __call__(self, retry_state: RetryCallState) -> float:
        outcome = retry_state.outcome
        exc = outcome.exception() if outcome is not None else None
        if isinstance(exc, ArchiveRateLimited):
            stated = (
                exc.retry_after
                if exc.retry_after is not None
                else RATE_LIMIT_FALLBACK_SECONDS
            )
            return min(stated, MAX_RETRY_AFTER_SECONDS)
        return self._exponential(retry_state)


def _log_retry(retry_state: RetryCallState) -> None:
    outcome = retry_state.outcome
    exc = outcome.exception() if outcome is not None else None
    log.warning(
        "archive request failed (attempt %d), retrying in %.1fs: %s",
        retry_state.attempt_number,
        retry_state.upcoming_sleep,
        exc,
    )


def _is_retryable(exc: BaseException) -> bool:
    """Everything retryable except a rate limit that cannot clear in time.

    An hourly or daily allowance will not come back inside a request's retry
    budget. Failing immediately hands the caller a clear reason to stop and
    resume later, which for a backfill with a manifest costs nothing; retrying
    would spend four minutes arriving at the same place.
    """
    if isinstance(exc, ArchiveRateLimited):
        return exc.waitable
    return isinstance(exc, ArchiveRetryableError)


def _retrying(settings: Settings) -> Retrying:
    return Retrying(
        stop=stop_after_attempt(settings.max_retry_attempts),
        wait=_wait_archive(float(settings.retry_backoff_seconds)),
        retry=retry_if_exception(_is_retryable),
        before_sleep=_log_retry,
        # Surface the real failure rather than tenacity's RetryError wrapper,
        # so a caller can tell a rate limit from a dead connection.
        reraise=True,
    )


# ---------------------------------------------------------------------------
# One request
# ---------------------------------------------------------------------------


def _body_excerpt(response: requests.Response) -> str:
    """The response body, reduced to something safe to put in a message."""
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict) and payload.get("reason"):
        text = str(payload["reason"])
    else:
        text = response.text or "<empty body>"
    text = " ".join(text.split())
    if len(text) > _BODY_EXCERPT_CHARS:
        text = text[:_BODY_EXCERPT_CHARS] + "…"
    return text


def _raise_for_status(response: requests.Response) -> None:
    status = response.status_code
    url = response.url

    if status == 429:
        retry_after = _parse_retry_after(response.headers.get("Retry-After"))
        body = _body_excerpt(response)
        window = _rate_limit_window(body)
        stated = (
            f"Retry-After {retry_after:.0f}s"
            if retry_after is not None
            else f"no Retry-After header, {window or 'unstated'} window"
        )
        raise ArchiveRateLimited(
            f"HTTP 429 rate limited by {url} ({stated}): {body}",
            url=url,
            retry_after=retry_after,
            limit_window=window,
        )

    if status >= 500:
        raise ArchiveServerError(
            f"HTTP {status} from {url}: {_body_excerpt(response)}",
            status_code=status,
            url=url,
        )

    if status >= 400:
        body = _body_excerpt(response)
        raise ArchiveRequestError(
            f"HTTP {status} from {url}: {body}. This will fail identically on "
            "every retry — the request itself is wrong.",
            status_code=status,
            url=url,
            body=body,
        )

    if status != 200:
        raise ArchiveResponseError(
            f"HTTP {status} from {url}, expected 200: {_body_excerpt(response)}"
        )


def _get(
    session: requests.Session,
    url: str,
    params: dict[str, str],
    timeout: tuple[float, float],
) -> tuple[dict[str, Any], str, bytes]:
    """Perform one attempt, translating every failure into an ArchiveError.

    Returns the decoded body, the URL that produced it, and the undecoded
    bytes. The URL comes from the response rather than being rebuilt, so
    ``source_url`` is the exact string that was sent, redirects and parameter
    encoding included. The bytes are what ING-03 archives: re-serialising the
    decoded object would already be a transformation, and the whole point of
    the archive is to hold something no code of ours has touched.
    """
    try:
        response = session.get(url, params=params, timeout=timeout)
    except requests.exceptions.Timeout as exc:
        connect_s, read_s = timeout
        raise ArchiveTransportError(
            f"timed out requesting {url} "
            f"(connect {connect_s}s, read {read_s}s): {exc}"
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise ArchiveTransportError(f"could not reach {url}: {exc}") from exc
    except requests.exceptions.RequestException as exc:
        raise ArchiveTransportError(f"request to {url} failed: {exc}") from exc

    _raise_for_status(response)

    try:
        payload = response.json()
    except ValueError as exc:
        # A 200 whose body will not decode is a truncated or proxied response,
        # not a malformed API — worth another attempt.
        raise ArchiveTransportError(
            f"could not decode JSON from {response.url}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ArchiveResponseError(
            f"{response.url} returned {type(payload).__name__}, expected a JSON object."
        )

    # Belt and braces: the archive signals errors with HTTP 400, but the flag
    # is documented as part of the response body and costs nothing to honour.
    if payload.get("error"):
        raise ArchiveRequestError(
            f"{response.url} returned error=true: {payload.get('reason', payload)}",
            status_code=response.status_code,
            url=response.url,
            body=str(payload.get("reason", "")),
        )

    return payload, str(response.url), response.content


# ---------------------------------------------------------------------------
# Validation of a 200
# ---------------------------------------------------------------------------
# parse_payload is public because it has two callers, not one: the live path
# below, and ING-03's replay rebuilding bronze from archived payloads. Sharing
# it is the point — a parsing bug fixed here is fixed for replay by
# construction, rather than fixed twice and drifting.


def _parse_timestamp(raw: str, grain: Grain) -> dt.datetime:
    """Interpret an API timestamp as UTC.

    Open-Meteo returns naive local strings and states the offset separately —
    which is exactly why the offset is checked to be zero before this runs. The
    result is timezone-aware, matching the timestamptz columns in bronze and
    the midnight-UTC / top-of-hour check constraints on them.
    """
    try:
        if grain == "daily":
            parsed = dt.datetime.strptime(raw, "%Y-%m-%d")
        else:
            parsed = dt.datetime.fromisoformat(raw)
    except (TypeError, ValueError) as exc:
        raise ArchiveResponseError(
            f"unparseable {grain} timestamp {raw!r} in response."
        ) from exc
    if parsed.tzinfo is not None:
        return parsed.astimezone(dt.timezone.utc)
    return parsed.replace(tzinfo=dt.timezone.utc)


def _expected_count(start: dt.date, end: dt.date, grain: Grain) -> int:
    days = (end - start).days + 1
    return days if grain == "daily" else days * 24


def parse_payload(
    payload: Mapping[str, Any],
    *,
    city_id: str,
    grain: Grain,
    start: dt.date,
    end: dt.date,
    url: str,
) -> ArchiveResponse:
    expected_units = _UNITS_BY_GRAIN[grain]

    offset = payload.get("utc_offset_seconds")
    if offset != 0:
        raise ArchiveResponseError(
            f"{city_id}: response reports utc_offset_seconds={offset!r} "
            f"(timezone {payload.get('timezone')!r}) despite timezone=UTC being "
            "requested. Every timestamp would be shifted and the daily "
            "aggregates would be computed over the wrong day boundaries."
        )

    block = payload.get(grain)
    units = payload.get(f"{grain}_units")
    if not isinstance(block, dict) or not isinstance(units, dict):
        raise ArchiveResponseError(
            f"{city_id}: response has no usable {grain!r} / {grain}_units block. "
            f"Top-level keys: {sorted(payload)}."
        )

    raw_times = block.get("time")
    if not isinstance(raw_times, list):
        raise ArchiveResponseError(f"{city_id}: {grain}.time is missing or not a list.")

    expected = _expected_count(start, end, grain)
    if len(raw_times) != expected:
        raise ArchiveResponseError(
            f"{city_id}: expected {expected} {grain} timestamps for "
            f"{start}..{end}, got {len(raw_times)}. The API silently returned a "
            "different range than the one requested."
        )

    times = tuple(_parse_timestamp(raw, grain) for raw in raw_times)
    if times and (times[0].date() != start or times[-1].date() != end):
        raise ArchiveResponseError(
            f"{city_id}: requested {start}..{end} but the response spans "
            f"{times[0].date()}..{times[-1].date()}."
        )

    values: dict[str, tuple[float | int | None, ...]] = {}
    for name, expected_unit in expected_units.items():
        series = block.get(name)
        if series is None:
            raise ArchiveResponseError(
                f"{city_id}: {grain} variable {name!r} was requested but is "
                f"absent from the response. Present: {sorted(block)}."
            )
        if not isinstance(series, list) or len(series) != len(times):
            length = len(series) if isinstance(series, list) else "not a list"
            raise ArchiveResponseError(
                f"{city_id}: {grain}.{name} has length {length}, expected "
                f"{len(times)} to match the time axis."
            )

        actual_unit = units.get(name)
        if actual_unit != expected_unit:
            raise ArchiveResponseError(
                f"{city_id}: {grain}.{name} came back in {actual_unit!r}, not "
                f"the {expected_unit!r} this client requests and the bronze "
                "schema documents. Refusing to land it — the values would be "
                "numerically wrong under a correct-looking column name."
            )

        for value in series:
            if value is not None and not isinstance(value, (int, float)):
                raise ArchiveResponseError(
                    f"{city_id}: {grain}.{name} contains a non-numeric value "
                    f"{value!r}."
                )
        values[name] = tuple(series)

    latitude = payload.get("latitude")
    longitude = payload.get("longitude")
    if not isinstance(latitude, (int, float)) or not isinstance(
        longitude, (int, float)
    ):
        raise ArchiveResponseError(
            f"{city_id}: response is missing the grid cell coordinates that "
            "bronze records as provenance."
        )

    elevation = payload.get("elevation")

    return ArchiveResponse(
        city_id=city_id,
        grain=grain,
        start=start,
        end=end,
        url=url,
        latitude=float(latitude),
        longitude=float(longitude),
        elevation_m=float(elevation) if isinstance(elevation, (int, float)) else None,
        times=times,
        values=MappingProxyType(values),
        units=MappingProxyType({name: units[name] for name in values}),
        generation_time_ms=payload.get("generationtime_ms"),
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Building the request
# ---------------------------------------------------------------------------


def _resolve(
    city: City | str, start: dt.date, end: dt.date, grain: Grain
) -> tuple[City, Grain]:
    """Check the arguments and resolve the city, before any network call.

    A round trip spent learning that ``end`` precedes ``start`` is a wasted
    one, and during a backfill it is a wasted one out of hundreds.
    """
    resolved_city = city if isinstance(city, City) else get_city(str(city))

    if grain not in _UNITS_BY_GRAIN:
        raise ValueError(f"grain must be 'daily' or 'hourly', got {grain!r}.")
    if not isinstance(start, dt.date) or not isinstance(end, dt.date):
        raise ValueError(
            f"start and end must be dates, got {type(start).__name__} and "
            f"{type(end).__name__}."
        )
    # datetime is a subclass of date; accepting one silently would truncate the
    # time component and misreport the range in source_url.
    if isinstance(start, dt.datetime) or isinstance(end, dt.datetime):
        raise ValueError("start and end must be dates, not datetimes.")
    if end < start:
        raise ValueError(f"end {end} precedes start {start}.")
    if start < ARCHIVE_START:
        raise ValueError(
            f"start {start} precedes the ERA5 archive, which begins "
            f"{ARCHIVE_START}."
        )
    today = dt.datetime.now(dt.timezone.utc).date()
    if end > today:
        raise ValueError(
            f"end {end} is in the future (today is {today} UTC). The archive "
            "also trails the present by several days; the API states its exact "
            "cut-off in the 400 it returns for a range beyond it."
        )
    return resolved_city, grain


def _build_params(
    city: City, start: dt.date, end: dt.date, grain: Grain
) -> dict[str, str]:
    return {
        "latitude": f"{city.lat:.6f}",
        "longitude": f"{city.lon:.6f}",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        grain: ",".join(_UNITS_BY_GRAIN[grain]),
        **_UNIT_PARAMS,
    }


def request_url(
    city: City | str,
    start: dt.date,
    end: dt.date,
    grain: Grain = "daily",
    *,
    settings: Settings | None = None,
) -> str:
    """The exact URL :func:`fetch_observations` would request.

    Prepared through ``requests`` rather than assembled by hand, so it is the
    same string by construction rather than by careful maintenance. Two callers
    need it without making the request: ING-03's replay, which rebuilds bronze
    from archived payloads and must reproduce the ``source_url`` those rows
    would have carried, and any log line that wants to say what it is about to
    fetch.
    """
    resolved_settings = settings if settings is not None else get_settings()
    resolved_city, grain = _resolve(city, start, end, grain)
    prepared = requests.Request(
        "GET",
        resolved_settings.openmeteo_base_url,
        params=_build_params(resolved_city, start, end, grain),
    ).prepare()
    return str(prepared.url)


# ---------------------------------------------------------------------------
# The public entry point
# ---------------------------------------------------------------------------


def fetch_observations(
    city: City | str,
    start: dt.date,
    end: dt.date,
    grain: Grain = "daily",
    *,
    on_payload: Callable[[bytes, str], None] | None = None,
    session: requests.Session | None = None,
    settings: Settings | None = None,
) -> ArchiveResponse:
    """Fetch one city's observations for one date range at one grain.

    Args:
        city: A :class:`~cities.City`, or a city id resolved through
            ``config/cities.yml``. Coordinates are never passed in directly —
            the registry is the only source of them.
        start: First day of the range, inclusive.
        end: Last day, inclusive. Both are calendar dates in UTC.
        grain: ``"daily"`` (30-year baseline) or ``"hourly"`` (trailing
            24 months, storm dynamics).
        on_payload: Called with ``(raw_bytes, url)`` once the request has
            succeeded and **before** the response is parsed. This is the seam
            ING-03's archival hangs off: a parsing bug found on day seven must
            not cost a re-pull of thirty years, which it would if the payload
            only reached disk after the code that has the bug in it. It runs
            outside the retry loop, so failed attempts are never archived, and
            an exception from it propagates — a response fetched and then
            dropped on the floor is worse than a loud failure.
        session: Reuse one across a backfill. A private session is created and
            closed per call when omitted, which is fine for one-off use and
            wasteful for hundreds of requests.
        settings: Override the process configuration. Intended for tests.

    Returns:
        A validated :class:`ArchiveResponse`.

    Raises:
        ValueError: The arguments cannot describe a valid request — checked
            before any network call, because a round trip to learn that
            ``end`` precedes ``start`` is a wasted one.
        ArchiveRequestError: The API rejected the request with a 4xx. Raised on
            the first attempt, with the response body in the message.
        ArchiveRetryableError: Timeouts, 5xx, or rate limits that survived
            ``MAX_RETRY_ATTEMPTS``.
        ArchiveResponseError: A 200 whose units, timezone, or shape do not
            match what was asked for.
    """
    resolved_settings = settings if settings is not None else get_settings()
    resolved_city, grain = _resolve(city, start, end, grain)
    params = _build_params(resolved_city, start, end, grain)
    timeout = (
        float(resolved_settings.request_connect_timeout_seconds),
        float(resolved_settings.request_timeout_seconds),
    )

    owns_session = session is None
    active = session if session is not None else build_session()
    log.debug(
        "fetching %s %s %s..%s from %s",
        resolved_city.id,
        grain,
        start,
        end,
        resolved_settings.openmeteo_base_url,
    )
    try:
        payload, url, raw = _retrying(resolved_settings)(
            _get, active, resolved_settings.openmeteo_base_url, params, timeout
        )
    finally:
        if owns_session:
            active.close()

    if on_payload is not None:
        on_payload(raw, url)

    return parse_payload(
        payload,
        city_id=resolved_city.id,
        grain=grain,
        start=start,
        end=end,
        url=url,
    )


# ---------------------------------------------------------------------------
# Manual verification against the API documentation
# ---------------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> int:
    from cities import load_cities

    registry = load_cities()
    parser = argparse.ArgumentParser(
        description=(
            "Fetch one city-year and print its provenance, units, and coverage "
            "for comparison against the Open-Meteo archive documentation."
        )
    )
    parser.add_argument("--city", default=registry.ids[0], choices=registry.ids)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--grain", default="daily", choices=("daily", "hourly"))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    )

    city = registry[args.city]
    start = dt.date(args.year, 1, 1)
    end = dt.date(args.year, 12, 31)

    try:
        response = fetch_observations(city, start, end, args.grain)
    except (ArchiveError, ValueError) as exc:
        print(f"FAILED  {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    expected = _expected_count(start, end, args.grain)
    print(f"{city.name}, {city.country} — {args.grain} {start}..{end}\n")
    print(f"  requested     {city.lat:.4f}, {city.lon:.4f} at {city.elevation_m:.0f} m")
    print(
        f"  grid cell     {response.latitude:.4f}, {response.longitude:.4f} at "
        f"{response.elevation_m:.0f} m"
        if response.elevation_m is not None
        else f"  grid cell     {response.latitude:.4f}, {response.longitude:.4f}"
    )
    print(f"  timezone      {response.payload.get('timezone')} "
          f"(utc_offset_seconds={response.payload.get('utc_offset_seconds')})")
    print(f"  rows          {len(response)} of {expected} expected")
    print(f"  first / last  {response.times[0].isoformat()} / "
          f"{response.times[-1].isoformat()}")
    generated = response.generation_time_ms
    print(f"  generated in  {generated:.1f} ms" if generated is not None
          else "  generated in  <not reported>")
    print(f"  source_url    {response.url}\n")

    nulls = response.null_counts()
    width = max(len(name) for name in response.variables)
    print(f"  {'variable'.ljust(width)}  {'unit':8} {'nulls':>6}  sample")
    print("  " + "-" * (width + 32))
    for name in response.variables:
        series = response.values[name]
        sample = next((v for v in series if v is not None), None)
        print(
            f"  {name.ljust(width)}  {response.units[name]:8} {nulls[name]:6d}  "
            f"{sample}"
        )

    missing = [name for name, count in nulls.items() if count == len(response)]
    print(
        f"\n  {len(response.variables)} variables, all units as documented."
        + (f" Entirely null: {missing}" if missing else " No variable entirely null.")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
