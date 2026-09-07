"""Loads archived payloads into the bronze landing tables.

Reads from disk, never from the network. The network is ING-01's business and
the archive is ING-03's; by the time a row reaches here the payload has already
been fetched, verified, and written to ``data/raw``. That separation is what
lets a parsing fix be re-run against thirty years of history in minutes.

**Bronze's job is faithful landing, not cleaning.** Nothing here converts a
unit, shifts a timezone, fills a gap, or deduplicates a row. Every one of those
is a decision, and a decision belongs in dbt where it is SQL, version
controlled, and covered by tests — not buried in a Python loader where it is
invisible to anyone reading the models. What this module adds to a row is the
two things the payload cannot know: when it was ingested, and which run it
belonged to. The request URL is not among them — it was measured at over 80%
of the row payload for one of a few hundred distinct strings, and
:func:`ingestion.archive.source_url_for` derives it from the archive instead.

The one place that judgement is exercised is null handling, and it is exercised
in the direction of doing nothing. A null in the payload is a null in the
warehouse. Coercing it to zero would turn "this grid cell reports no snowfall
data" into "it did not snow", which is a different claim and a wrong one; and a
zero cannot be distinguished from a measurement afterwards. Dropping the row
would silently shorten a series that downstream code counts on being complete.

Usage::

    from ingestion.loader import load_archive

    result = load_archive()          # everything on disk, into bronze
    print(result.rows, result.batch_id)

Run ``python ingestion/loader.py --dry-run`` to see what would be loaded, or
``--city london --grain daily`` to load one slice.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterable, Iterator, Mapping, Sequence

import pandas as pd
import psycopg2
from sqlalchemy import Connection, Engine, create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Settings, get_settings  # noqa: E402
from ingestion import archive  # noqa: E402
from ingestion.client import (  # noqa: E402
    DAILY_VARIABLES,
    HOURLY_VARIABLES,
    ArchiveResponse,
    Grain,
)
from ingestion.planner import Manifest, WorkUnit  # noqa: E402

__all__ = [
    "BRONZE_SCHEMA",
    "TABLE_BY_GRAIN",
    "LoadError",
    "LoadResult",
    "build_frame",
    "engine_from_settings",
    "load_archive",
    "load_response",
    "load_unit",
]

log = logging.getLogger(__name__)

BRONZE_SCHEMA: Final[str] = "bronze_raw"

TABLE_BY_GRAIN: Final[Mapping[str, str]] = {
    "daily": "observations_daily",
    "hourly": "observations_hourly",
}

#: The metadata every bronze row carries, plus the three grid-cell columns
#: recording which ERA5 cell actually answered.
#:
#: ``source_url`` is not among them. §5.2 asked for it per row; it measured 712
#: bytes on a daily row and 466 on an hourly one — over 80% of the payload,
#: roughly 300 MB across the backfill — for one of a few hundred distinct
#: strings. :func:`ingestion.archive.source_url_for` derives it instead.
PROVENANCE_COLUMNS: Final[tuple[str, ...]] = (
    "city_id",
    "observation_time",
    "ingested_at",
    "batch_id",
    "api_latitude",
    "api_longitude",
    "api_elevation_m",
)

_VARIABLES_BY_GRAIN: Final[Mapping[str, tuple[str, ...]]] = {
    "daily": DAILY_VARIABLES,
    "hourly": HOURLY_VARIABLES,
}

#: Rows per COPY. One city-year is 365 daily or 8 784 hourly rows, so this
#: keeps a unit to a single round trip while bounding memory if a caller ever
#: hands over a much wider window.
CHUNK_ROWS: Final[int] = 20_000


def columns_for(grain: Grain) -> tuple[str, ...]:
    """Every column this loader writes, in a fixed order."""
    return (*PROVENANCE_COLUMNS, *_VARIABLES_BY_GRAIN[grain])


class LoadError(RuntimeError):
    """The load cannot proceed, and retrying would not help."""


@dataclass(frozen=True)
class LoadResult:
    """What one load run did."""

    batch_id: uuid.UUID
    ingested_at: dt.datetime
    units: int = 0
    rows: int = 0
    seconds: float = 0.0
    by_table: dict[str, int] = field(default_factory=dict)

    @property
    def rows_per_second(self) -> float:
        return self.rows / self.seconds if self.seconds else 0.0


# ---------------------------------------------------------------------------
# Rows to frame
# ---------------------------------------------------------------------------


def build_frame(
    response: ArchiveResponse,
    *,
    batch_id: uuid.UUID,
    ingested_at: dt.datetime,
) -> pd.DataFrame:
    """One parsed response as a frame whose columns match the bronze table.

    Built with ``dtype=object`` deliberately. Left to itself pandas would widen
    any column holding a null to ``float64`` and replace the null with ``NaN``
    — so ``weather_code`` would arrive as ``51.0`` and, worse, a missing value
    would land as the float NaN rather than as SQL NULL. Holding everything as
    objects means the values that reach ``COPY`` are the values that came out
    of the JSON, and Postgres does the one cast that should happen: the
    declared column type.
    """
    columns = columns_for(response.grain)
    rows = [
        {**row, "ingested_at": ingested_at, "batch_id": batch_id}
        for row in response.rows()
    ]
    frame = pd.DataFrame(rows, columns=list(columns), dtype=object)

    if len(frame) != len(response):
        raise LoadError(
            f"{response.city_id}: frame has {len(frame)} rows, response has "
            f"{len(response)}."
        )
    return frame


# ---------------------------------------------------------------------------
# Frame to Postgres
# ---------------------------------------------------------------------------


def _csv_buffer(rows: Iterable[Sequence[Any]]) -> io.StringIO:
    """Render rows as CSV for ``COPY``. None is the only unquoted empty field.

    Postgres reads an unquoted empty CSV field as NULL and a quoted one as an
    empty string, so the two must not render alike. ``QUOTE_MINIMAL`` — the
    default — writes both as nothing at all, which would silently turn any
    empty string into a NULL. ``QUOTE_NOTNULL`` quotes everything that is not
    None, which is precisely the distinction Postgres is looking for.

    This is the whole null-preservation mechanism: nothing substitutes a
    sentinel that would then need substituting back.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, quoting=csv.QUOTE_NOTNULL, lineterminator="\n")
    for row in rows:
        writer.writerow(
            [None if value is None or value is pd.NA else value for value in row]
        )
    buffer.seek(0)
    return buffer


def _copy_insert(table, conn, keys: list[str], data_iter) -> int:
    """A ``DataFrame.to_sql`` insert method that uses ``COPY``.

    pandas' own inserters send one multi-row ``INSERT`` per chunk, which for
    436 000 rows is minutes of round trips and parameter binding. ``COPY FROM
    STDIN`` is the interface Postgres provides for exactly this and moves the
    same rows in seconds. ``to_sql`` documents ``method`` as the hook for
    swapping the mechanism without giving up the frame, which is what this is.
    """
    dbapi = conn.connection.dbapi_connection
    qualified = (
        f'"{table.schema}"."{table.name}"' if table.schema else f'"{table.name}"'
    )
    column_list = ", ".join(f'"{key}"' for key in keys)
    with dbapi.cursor() as cursor:
        try:
            cursor.copy_expert(
                f"COPY {qualified} ({column_list}) FROM STDIN WITH (FORMAT csv)",
                _csv_buffer(data_iter),
            )
        except psycopg2.DataError as exc:
            # Deliberately not coerced. A smallint column refusing "98.0" means
            # the API changed how it represents an integer, and bronze landing
            # it quietly as 98 would hide that. The payload is already archived,
            # so nothing is lost by stopping — only delayed.
            raise LoadError(
                f"{qualified} refused a value COPY could not cast. Bronze does "
                f"not convert types, so this means the payload's representation "
                f"changed upstream: {exc}"
            ) from exc
        return cursor.rowcount


def _assert_columns_exist(
    connection: Connection, table: str, columns: Sequence[str]
) -> None:
    """Fail with the offending column name rather than a COPY parse error."""
    present = {
        row[0]
        for row in connection.execute(
            text(
                "select column_name from information_schema.columns "
                "where table_schema = :schema and table_name = :table"
            ),
            {"schema": BRONZE_SCHEMA, "table": table},
        )
    }
    if not present:
        raise LoadError(
            f"{BRONZE_SCHEMA}.{table} does not exist. Run "
            "`python ingestion/apply_schema.py` first."
        )
    missing = [column for column in columns if column not in present]
    if missing:
        raise LoadError(
            f"{BRONZE_SCHEMA}.{table} has no column(s) {missing}. The client "
            "requests variables the schema has nowhere to land."
        )


def load_response(
    response: ArchiveResponse,
    connection: Connection,
    *,
    batch_id: uuid.UUID,
    ingested_at: dt.datetime,
    verify_columns: bool = True,
) -> int:
    """Land one parsed response. Returns the number of rows written.

    Appends. Bronze permits duplicates by design — re-ingesting a window
    inserts a second copy and silver takes the most recent ``ingested_at`` per
    ``(city_id, observation_time)``. A loader that deduplicated here would be
    making that decision twice, in two languages, and only one of them tested.
    """
    table = TABLE_BY_GRAIN[response.grain]
    frame = build_frame(response, batch_id=batch_id, ingested_at=ingested_at)
    if verify_columns:
        _assert_columns_exist(connection, table, list(frame.columns))

    frame.to_sql(
        table,
        connection,
        schema=BRONZE_SCHEMA,
        if_exists="append",
        index=False,
        chunksize=CHUNK_ROWS,
        method=_copy_insert,
    )
    log.debug("loaded %d rows into %s.%s", len(frame), BRONZE_SCHEMA, table)
    return len(frame)


def load_unit(
    unit: WorkUnit,
    connection: Connection,
    *,
    batch_id: uuid.UUID,
    ingested_at: dt.datetime,
    root: Path | str | None = None,
    settings: Settings | None = None,
    verify_columns: bool = True,
) -> int:
    """Replay one archived unit from disk and land it. No network."""
    response = archive.replay_unit(unit, root, settings=settings)
    if len(response) != unit.expected_rows:
        raise LoadError(
            f"{unit.key}: archived payload holds {len(response)} rows, the "
            f"window covers {unit.expected_rows}."
        )
    return load_response(
        response,
        connection,
        batch_id=batch_id,
        ingested_at=ingested_at,
        verify_columns=verify_columns,
    )


# ---------------------------------------------------------------------------
# A whole run
# ---------------------------------------------------------------------------


def engine_from_settings(settings: Settings | None = None) -> Engine:
    resolved = settings if settings is not None else get_settings()
    return create_engine(resolved.require_database_url(), future=True)


def load_archive(
    root: Path | str | None = None,
    *,
    units: Iterable[WorkUnit] | None = None,
    grain: Grain | None = None,
    city_id: str | None = None,
    engine: Engine | None = None,
    manifest: Manifest | None = None,
    batch_id: uuid.UUID | None = None,
    settings: Settings | None = None,
) -> LoadResult:
    """Load archived units into bronze, one transaction per unit.

    Per unit rather than per run, deliberately. One transaction around the
    whole backfill would hold hundreds of thousands of rows and every lock
    until the end, and a failure at unit four hundred would roll back four
    hundred units of work. Per unit, a failure costs one unit and everything
    before it is committed and recorded.

    Args:
        root: Archive root. Defaults to ``DATA_RAW_DIR``.
        units: Which units to load. Defaults to everything on disk.
        grain, city_id: Narrow the default discovery.
        engine: A SQLAlchemy engine. One is built from ``DATABASE_URL`` and
            disposed at the end when omitted.
        manifest: Recorded **after** each unit's transaction commits, so a
            manifest entry always means the rows are really in the warehouse.
        batch_id: Groups every row this run writes, so a bad run can be deleted
            with one ``delete from ... where batch_id = ...``.
    """
    resolved_settings = settings if settings is not None else get_settings()
    resolved_batch = batch_id or uuid.uuid4()
    # One timestamp for the whole run, not one per row. Silver's dedup orders
    # by ingested_at desc within (city_id, observation_time); rows that arrived
    # in the same run should tie rather than be ordered by how long the COPY
    # took to reach them.
    ingested_at = dt.datetime.now(dt.timezone.utc)

    selected = list(
        archive.archived_units(
            root, grain=grain, city_id=city_id, settings=resolved_settings
        )
        if units is None
        else units
    )

    owns_engine = engine is None
    active = engine if engine is not None else engine_from_settings(resolved_settings)

    started = time.perf_counter()
    loaded_units = 0
    loaded_rows = 0
    by_table: dict[str, int] = {}
    verified: set[str] = set()

    try:
        for unit in selected:
            table = TABLE_BY_GRAIN[unit.grain]
            with active.begin() as connection:
                rows = load_unit(
                    unit,
                    connection,
                    batch_id=resolved_batch,
                    ingested_at=ingested_at,
                    root=root,
                    settings=resolved_settings,
                    verify_columns=table not in verified,
                )
            verified.add(table)

            # Only after the commit above. Recording first would let a crash
            # leave a manifest saying rows are present that are not, and
            # nothing downstream would report the hole.
            if manifest is not None:
                manifest.record(unit, rows=rows)

            loaded_units += 1
            loaded_rows += rows
            by_table[table] = by_table.get(table, 0) + rows
            log.info("loaded %s (%d rows)", unit.key, rows)
    finally:
        if owns_engine:
            active.dispose()

    return LoadResult(
        batch_id=resolved_batch,
        ingested_at=ingested_at,
        units=loaded_units,
        rows=loaded_rows,
        seconds=time.perf_counter() - started,
        by_table=by_table,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _pending(
    root: Path | str | None,
    grain: Grain | None,
    city_id: str | None,
    manifest: Manifest | None,
    settings: Settings,
) -> Iterator[WorkUnit]:
    for unit in archive.archived_units(
        root, grain=grain, city_id=city_id, settings=settings
    ):
        if manifest is not None and manifest.is_complete(unit):
            continue
        yield unit


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Load archived payloads into the bronze landing tables."
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--grain", choices=tuple(TABLE_BY_GRAIN), default=None)
    parser.add_argument("--city", dest="city_id", default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--all",
        action="store_true",
        help="load every archived unit, including ones the manifest already covers",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would be loaded"
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level, format="%(levelname)-8s %(name)s  %(message)s"
    )
    root = args.root or settings.data_raw_dir
    manifest = (
        None
        if args.all
        else Manifest(args.manifest or settings.ingest_manifest_path)
    )

    units = list(_pending(root, args.grain, args.city_id, manifest, settings))
    if not units:
        print(f"nothing to load from {root}")
        return 0

    rows = sum(u.expected_rows for u in units)
    print(f"{len(units)} archived unit(s), {rows:,} rows, from {root}")
    if args.dry_run:
        for unit in units:
            print(f"  {unit.key:44} {unit.expected_rows:>7,} rows")
        return 0

    try:
        result = load_archive(
            root,
            units=units,
            manifest=manifest,
            settings=settings,
        )
    except Exception as exc:
        print(f"FAILED  {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"\n  batch_id     {result.batch_id}")
    print(f"  ingested_at  {result.ingested_at.isoformat()}")
    for table, count in sorted(result.by_table.items()):
        print(f"  {BRONZE_SCHEMA}.{table:22} {count:>9,} rows")
    print(
        f"  {result.rows:,} rows in {result.seconds:.1f}s "
        f"({result.rows_per_second:,.0f} rows/s)"
    )
    print(
        f"\n  Undo this run with:\n"
        f"    delete from {BRONZE_SCHEMA}.<table> where batch_id = "
        f"'{result.batch_id}';"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
