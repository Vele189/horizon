"""Teleconnection indices from NOAA, stamped with when they became knowable.

Every feature this project has ever fed a model is one city's own history.
These are the first that are not: the ENSO signal (ONI), the North Atlantic
Oscillation, the Arctic Oscillation and the Indian Ocean Dipole are the
dominant large-scale drivers of seasonal temperature anomalies, they are free
from NOAA, and they should help most in the tropical cities where the seven-day
model is weakest.

**The whole difficulty is dating them.** An index value is stamped with a
nominal period -- "January 2024" -- and that is not when it existed. It is
computed once the period has ended and published some days after that. Joining
on the nominal month reads the future, and it reads the future in the way that
is hardest to notice: the feature is correctly named, correctly typed,
correctly joined, and answers a question about days that had not happened yet.
No existing test in this repository would catch it. Every metric would simply
improve.

So nothing here stores a value without also storing:

``publication_date``
    The earliest date the value could be read. Derived, per index, as the end
    of the last month it covers plus a stated lag. `config/teleconnections.yml`
    carries the lags and the observations they were calibrated against.

``vintage_at``
    When *this* pipeline first saw this value. Indices are revised -- the ONI's
    base period shifts every five years, and NOAA restates history when it does
    -- so "the value for January 2024" is not one number, it is a series of
    numbers with dates. A model scoring a day in 2024 must read the value as it
    stood then, not as it stands now.

The vintage table is append-on-change: a re-run that finds the same number
writes nothing, and a re-run that finds a different one writes a new row beside
the old rather than over it.

Usage::

    python ingestion/teleconnections.py             # fetch and report
    python ingestion/teleconnections.py --write     # and land it in bronze
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import logging
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import pandas as pd
import yaml
from sqlalchemy import Engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings  # noqa: E402
from ingestion.client import build_session  # noqa: E402
from ingestion.loader import engine_from_settings  # noqa: E402

__all__ = [
    "REGISTRY_PATH",
    "TELECONNECTION_TABLE",
    "Index",
    "covered_span",
    "fetch_index",
    "load_registry",
    "parse_cpc_monthly",
    "parse_cpc_seasonal",
    "parse_psl_monthly",
    "publication_date",
    "changed_rows",
    "latest_vintage",
    "read_vintages",
    "write_vintages",
]

log = logging.getLogger(__name__)

REGISTRY_PATH: Final[Path] = (
    Path(__file__).resolve().parent.parent / "config" / "teleconnections.yml"
)
TELECONNECTION_TABLE: Final[str] = "bronze_raw.teleconnection_indices"
SCHEMA_SQL: Final[Path] = Path(__file__).resolve().parent / "teleconnections.sql"

#: The three-letter season labels the CPC uses, in order, and the month each
#: one is centred on. ``DJF`` is centred on January, which is the fact the
#: publication rule turns on.
SEASONS: Final[tuple[str, ...]] = (
    "DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ",
    "JJA", "JAS", "ASO", "SON", "OND", "NDJ",
)

#: PSL writes its build date into the file. Real provenance, so it is read
#: rather than assumed: `Created Sat Jul 25 13:56:30 MDT 2026`.
_PSL_CREATED = re.compile(
    r"^Created\s+\w+\s+(\w+)\s+(\d+)\s+[\d:]+\s+\w+\s+(\d{4})\s*$", re.M
)


class TeleconnectionError(RuntimeError):
    """Raised when a feed cannot be parsed into dated, usable values.

    A silently empty parse is the worst outcome available here: the ablation
    would report the indices as worthless and the conclusion would be about
    the parser.
    """


@dataclass(frozen=True)
class Index:
    """One index, and everything needed to date it."""

    id: str
    name: str
    driver: str
    url: str
    format: str
    covers_months: int
    centred: bool
    publication_lag_days: int
    description: str
    missing_value: float | None = None

    def __post_init__(self) -> None:
        if self.covers_months < 1:
            raise TeleconnectionError(f"{self.id}: covers_months must be >= 1.")
        if self.centred and self.covers_months % 2 == 0:
            # A centred window needs a middle month to be centred on.
            raise TeleconnectionError(
                f"{self.id}: a centred window must cover an odd number of months, "
                f"got {self.covers_months}."
            )
        if self.publication_lag_days < 0:
            raise TeleconnectionError(
                f"{self.id}: a negative publication lag would read the future."
            )


def load_registry(path: Path = REGISTRY_PATH) -> tuple[Index, ...]:
    """The indices, validated on load. Mirrors `cities.load_cities()`."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or "indices" not in document:
        raise TeleconnectionError(f"{path} has no `indices` key.")
    return tuple(
        Index(
            id=str(raw["id"]),
            name=str(raw["name"]),
            driver=str(raw["driver"]),
            url=str(raw["url"]),
            format=str(raw["format"]),
            covers_months=int(raw["covers_months"]),
            centred=bool(raw.get("centred", False)),
            publication_lag_days=int(raw["publication_lag_days"]),
            description=str(raw.get("description", "")).strip(),
            missing_value=(
                float(raw["missing_value"]) if "missing_value" in raw else None
            ),
        )
        for raw in document["indices"]
    )


# ---------------------------------------------------------------------------
# Dating
# ---------------------------------------------------------------------------


def _month_end(year: int, month: int) -> dt.date:
    return dt.date(year, month, calendar.monthrange(year, month)[1])


def _shift_month(year: int, month: int, by: int) -> tuple[int, int]:
    index = (year * 12 + (month - 1)) + by
    return index // 12, index % 12 + 1


def covered_span(index: Index, nominal: dt.date) -> tuple[dt.date, dt.date]:
    """The first and last month an index value actually summarises.

    For a monthly index this is the nominal month twice over. For a centred
    window it straddles: the ONI labelled January covers December through
    February, so its last covered month is *February*, one month after the
    label. That single month is the difference between a correct join and a
    leak, and it is why this is a function rather than a subtraction at the
    call site.
    """
    if not index.centred:
        first = nominal.replace(day=1)
        start_year, start_month = _shift_month(
            first.year, first.month, -(index.covers_months - 1)
        )
        return dt.date(start_year, start_month, 1), _month_end(
            first.year, first.month
        )

    wing = (index.covers_months - 1) // 2
    start_year, start_month = _shift_month(nominal.year, nominal.month, -wing)
    end_year, end_month = _shift_month(nominal.year, nominal.month, wing)
    return dt.date(start_year, start_month, 1), _month_end(end_year, end_month)


def publication_date(index: Index, nominal: dt.date) -> dt.date:
    """The earliest date this value may be read.

    End of the last covered month, plus the index's stated lag. Everything
    downstream joins on this and never on the nominal period.
    """
    _, last_covered = covered_span(index, nominal)
    return last_covered + dt.timedelta(days=index.publication_lag_days)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_cpc_monthly(index: Index, body: str) -> pd.DataFrame:
    """``YYYY MM value``, one row a month. NAO and AO."""
    rows: list[tuple[dt.date, float]] = []
    for line in body.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            year, month, value = int(parts[0]), int(parts[1]), float(parts[2])
        except ValueError:
            continue
        if not 1 <= month <= 12:
            continue
        rows.append((dt.date(year, month, 1), value))
    return _frame(index, rows)


def parse_cpc_seasonal(index: Index, body: str) -> pd.DataFrame:
    """``SEAS YR TOTAL ANOM``, one row a season. The ONI.

    The nominal date is the season's **centre** month, so ``DJF 1950`` becomes
    1950-01-01. Storing it as December would misdate the value by two months in
    the direction that leaks.
    """
    rows: list[tuple[dt.date, float]] = []
    for line in body.splitlines():
        parts = line.split()
        if len(parts) != 4 or parts[0] not in SEASONS:
            continue
        try:
            year, value = int(parts[1]), float(parts[3])
        except ValueError:
            continue
        # The label's year is the year the season *ends* in for DJF and NDJ
        # wrap-arounds; CPC labels DJF 1950 as the winter ending in Feb 1950,
        # so the centre month is January of that year. For every other season
        # the centre is inside the labelled year.
        centre_month = SEASONS.index(parts[0]) + 1
        rows.append((dt.date(year, centre_month, 1), value))
    return _frame(index, rows)


def parse_psl_monthly(index: Index, body: str) -> pd.DataFrame:
    """A year then twelve values a line, with a sentinel for months not yet in.

    NOAA PSL's long-format table. The sentinel is ``-9999`` and dropping it is
    not optional: carried through, the DMI would read as a catastrophic
    negative dipole for every month of the current year that has not happened.
    """
    rows: list[tuple[dt.date, float]] = []
    sentinel = index.missing_value
    for line in body.splitlines():
        parts = line.split()
        if len(parts) != 13:
            continue
        try:
            year = int(parts[0])
            values = [float(part) for part in parts[1:]]
        except ValueError:
            continue
        if not 1800 <= year <= 2200:
            continue
        for month, value in enumerate(values, start=1):
            if sentinel is not None and value == sentinel:
                continue
            rows.append((dt.date(year, month, 1), value))
    return _frame(index, rows)


PARSERS: Final[Mapping[str, Any]] = {
    "cpc_monthly": parse_cpc_monthly,
    "cpc_seasonal": parse_cpc_seasonal,
    "psl_monthly": parse_psl_monthly,
}


def _frame(index: Index, rows: Sequence[tuple[dt.date, float]]) -> pd.DataFrame:
    """Dated values, with every row carrying the span it covers and its date."""
    if not rows:
        raise TeleconnectionError(
            f"{index.id}: parsed no values. A silently empty feed would be "
            f"reported by the ablation as an index that does not help."
        )
    frame = pd.DataFrame(rows, columns=["nominal_period", "value"])
    spans = [covered_span(index, nominal) for nominal in frame["nominal_period"]]
    frame["covers_start"] = [start for start, _ in spans]
    frame["covers_end"] = [end for _, end in spans]
    frame["publication_date"] = [
        publication_date(index, nominal) for nominal in frame["nominal_period"]
    ]
    frame["index_id"] = index.id
    return frame.sort_values("nominal_period").reset_index(drop=True)


def source_created_at(body: str) -> dt.date | None:
    """The build date a feed states about itself, when it states one.

    PSL writes one; CPC does not. Real provenance beats an assumption, so it is
    read where available and left null where not, rather than being
    back-filled with the fetch time and quietly becoming a different fact.
    """
    match = _PSL_CREATED.search(body)
    if match is None:
        return None
    month, day, year = match.groups()
    try:
        return dt.datetime.strptime(f"{month} {day} {year}", "%b %d %Y").date()
    except ValueError:
        return None


def fetch_index(index: Index, session=None) -> tuple[pd.DataFrame, dt.date | None]:
    """Download and parse one index. Returns its values and its stated build date."""
    parser = PARSERS.get(index.format)
    if parser is None:
        raise TeleconnectionError(
            f"{index.id}: unknown format {index.format!r}; "
            f"have {sorted(PARSERS)}."
        )
    owned = session is None
    session = session or build_session()
    try:
        response = session.get(index.url, timeout=60)
        response.raise_for_status()
        body = response.text
    finally:
        if owned:
            session.close()
    return parser(index, body), source_created_at(body)


# ---------------------------------------------------------------------------
# Vintages
# ---------------------------------------------------------------------------


def apply_schema(engine: Engine) -> None:
    """Apply the idempotent DDL."""
    with engine.begin() as connection:
        connection.execute(text(SCHEMA_SQL.read_text()))


def read_vintages(engine: Engine) -> pd.DataFrame:
    """Everything landed so far, newest vintage last."""
    frame = pd.read_sql(
        text(
            f"""
            select index_id, nominal_period, vintage_at, covers_start,
                   covers_end, publication_date, publication_is_estimated,
                   value, source_created_at
            from {TELECONNECTION_TABLE}
            order by index_id, nominal_period, vintage_at
            """
        ),
        engine,
    )
    for column in ("nominal_period", "covers_start", "covers_end",
                   "publication_date", "source_created_at"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column]).dt.date
    return frame


def latest_vintage(vintages: pd.DataFrame) -> pd.DataFrame:
    """The current value of each period: the last row per index and period."""
    if vintages.empty:
        return vintages
    return (
        vintages.sort_values("vintage_at")
        .groupby(["index_id", "nominal_period"], as_index=False)
        .last()
    )


#: Below this, two values are the same number written differently.
#:
#: The feeds print four decimals at most, and a float round-trip through
#: Postgres and pandas can move the last bit. Without a tolerance every re-run
#: would record a "revision" that is really a representation, and the vintage
#: table would fill with noise that hides the real restatements.
REVISION_TOLERANCE: Final[float] = 5e-7


def changed_rows(fresh: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """The rows worth writing: new periods, and periods whose value moved.

    This is what makes the table append-on-change rather than append-always. A
    daily re-run of an unchanged 76-year series should write nothing at all.
    """
    if existing.empty:
        return fresh
    current = latest_vintage(existing).set_index(["index_id", "nominal_period"])
    keys = pd.MultiIndex.from_arrays(
        [fresh["index_id"], fresh["nominal_period"]]
    )
    previous = pd.Series(
        current["value"].reindex(keys).to_numpy(), index=fresh.index, dtype="float64"
    )
    moved = previous.isna() | ((fresh["value"] - previous).abs() > REVISION_TOLERANCE)
    return fresh.loc[moved]


def write_vintages(
    engine: Engine,
    fresh: pd.DataFrame,
    *,
    index: Index,
    vintage_at: dt.datetime,
    created: dt.date | None,
    first_ingest: bool,
) -> int:
    """Append the rows that are new or revised. Returns how many.

    ``first_ingest`` decides `publication_is_estimated`. On the very first run
    every period predates this pipeline, so every publication date comes from
    the lag rule and is an estimate. After that, a period appearing for the
    first time is one we watched arrive, and its publication date is the lag
    rule *or* the observation, whichever is later -- the observation cannot be
    earlier than the rule without the rule being wrong, and taking the later of
    the two keeps the column conservative either way.
    """
    if fresh.empty:
        return 0
    batch = uuid.uuid4()
    rows = [
        {
            "index_id": row.index_id,
            "nominal_period": row.nominal_period,
            "vintage_at": vintage_at,
            "covers_start": row.covers_start,
            "covers_end": row.covers_end,
            "publication_date": (
                row.publication_date
                if first_ingest
                # Otherwise the later of the rule and the observation. For a
                # period appearing for the first time that keeps the
                # conservative rule when the rule is slower than the feed. For
                # a *revision* it is the observation date, which is the point:
                # the restated number did not exist until it was restated, so
                # a day in 2024 must keep reading the 2024 vintage.
                else max(row.publication_date, vintage_at.date())
            ),
            "publication_is_estimated": first_ingest,
            "value": float(row.value),
            "source_url": index.url,
            "source_created_at": created,
            "batch_id": batch,
        }
        for row in fresh.itertuples()
    ]
    statement = text(
        f"""
        insert into {TELECONNECTION_TABLE} (
            index_id, nominal_period, vintage_at, covers_start, covers_end,
            publication_date, publication_is_estimated, value,
            source_url, source_created_at, batch_id
        ) values (
            :index_id, :nominal_period, :vintage_at, :covers_start, :covers_end,
            :publication_date, :publication_is_estimated, :value,
            :source_url, :source_created_at, :batch_id
        )
        on conflict (index_id, nominal_period, vintage_at) do nothing
        """
    )
    with engine.begin() as connection:
        connection.execute(statement, rows)
    return len(rows)


def _report(landed: Sequence[Mapping[str, Any]]) -> str:
    return pd.DataFrame(landed).to_string(index=False)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch NOAA teleconnection indices into bronze, with vintages."
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        default=None,
        help="Index ids to fetch (default: every index in the registry).",
    )
    parser.add_argument(
        "--write", action="store_true", help="Land the rows. Default reports only."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    get_settings()  # fail fast on a bad environment, as every entry point does

    registry = load_registry()
    wanted = set(args.indices) if args.indices else {index.id for index in registry}
    unknown = wanted - {index.id for index in registry}
    if unknown:
        print(f"unknown indices: {sorted(unknown)}", file=sys.stderr)
        return 1

    vintage_at = dt.datetime.now(dt.timezone.utc)
    engine = engine_from_settings()
    session = build_session()
    landed: list[dict[str, Any]] = []
    try:
        if args.write:
            apply_schema(engine)
            existing = read_vintages(engine)
        else:
            try:
                existing = read_vintages(engine)
            except Exception:  # noqa: BLE001 - an absent table is a dry run, not a failure
                existing = pd.DataFrame()
        first_ingest = existing.empty

        for index in registry:
            if index.id not in wanted:
                continue
            values, created = fetch_index(index, session)
            fresh = changed_rows(values, existing)
            row = {
                "index": index.id,
                "parsed": len(values),
                "first": values["nominal_period"].min(),
                "last": values["nominal_period"].max(),
                "covers_to": values["covers_end"].max(),
                "publishes": values["publication_date"].max(),
                "source_built": created or "-",
                "new_or_revised": len(fresh),
            }
            if args.write and not fresh.empty:
                row["written"] = write_vintages(
                    engine,
                    fresh,
                    index=index,
                    vintage_at=vintage_at,
                    created=created,
                    first_ingest=first_ingest,
                )
            landed.append(row)
    finally:
        session.close()
        engine.dispose()

    print(_report(landed))
    if not args.write:
        print("\n(no --write: nothing landed)")
        return 0
    print(f"\nvintage {vintage_at:%Y-%m-%d %H:%M:%SZ}, first ingest: {first_ingest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
