"""Runs the backfill: plan, fetch, archive, load, record. Resumably.

This is the only module that drives the others. The planner decides what to
request, the client makes the request, the archive holds the payload, and the
loader lands it — this ties them into one loop that can be killed at any moment
and picked up again without losing or re-fetching work.

Per unit, in this order and no other:

1.  If the payload is already on disk, **no request is made.** The archive is
    the fetch cache as well as the replay source, so a run interrupted between
    fetching and loading costs nothing to resume.
2.  Otherwise fetch it, writing the payload to disk before it is parsed.
3.  Load it into bronze in its own transaction.
4.  Record the unit in the manifest, *after* that transaction commits.

Step 4 last is the whole resumability story. A manifest entry means the rows
are in the warehouse; anything less would let a crash leave the next run
skipping a window whose rows are missing, with nothing downstream reporting the
hole.

**The free tier will not do this in one sitting.** The daily backfill is ~26 000
weighted API calls against an allowance of 10 000 a day (see the planner). The
run therefore stops when its budget is spent rather than earning a 429 per
remaining unit, and says so. Resuming tomorrow is a re-invocation with no
arguments.

Usage::

    python ingestion/backfill.py --grain daily            # run it
    python ingestion/backfill.py --grain daily --dry-run  # what it would do
    python ingestion/backfill.py --report                 # what landed

Ctrl-C once finishes the unit in flight and stops cleanly. Twice gives up
immediately, which may leave the unit in flight to be re-fetched.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import signal
import sys
import time
import types
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Iterable, Sequence

from sqlalchemy import Engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cities import City, load_cities  # noqa: E402
from config import Settings, get_settings  # noqa: E402
from ingestion import archive  # noqa: E402
from ingestion.client import (  # noqa: E402
    ArchiveError,
    ArchiveRateLimited,
    Grain,
    build_session,
    fetch_observations,
)
from ingestion.loader import (  # noqa: E402
    BRONZE_SCHEMA,
    TABLE_BY_GRAIN,
    engine_from_settings,
    load_unit,
)
from ingestion.planner import (  # noqa: E402
    DAILY_QUOTA_CALLS,
    Manifest,
    _add_months,
    Plan,
    WorkUnit,
    plan_backfill,
)

__all__ = [
    "NEON_STORAGE_BUDGET_BYTES",
    "STORM_DYNAMICS_COLUMNS",
    "BackfillResult",
    "CityCoverage",
    "TableSize",
    "bronze_coverage",
    "column_population",
    "run_backfill",
    "table_size",
]

log = logging.getLogger("ingestion.backfill")

#: Consecutive failures before the run gives up. A single unit failing is
#: worth stepping over; four in a row means something systemic and continuing
#: only burns quota.
MAX_CONSECUTIVE_FAILURES: Final[int] = 4

#: A gap wider than this in a city's daily series wants an explanation.
MAX_ACCEPTABLE_GAP_DAYS: Final[int] = 7


@dataclass
class BackfillResult:
    """What one invocation did. Every number here is logged on exit."""

    batch_id: uuid.UUID
    started_at: dt.datetime
    grains: tuple[str, ...] = ()
    units_planned: int = 0
    units_completed: int = 0
    requests_made: int = 0
    units_from_cache: int = 0
    weight_spent: float = 0.0
    rows_loaded: int = 0
    seconds: float = 0.0
    stopped_because: str = "complete"
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def units_remaining(self) -> int:
        return self.units_planned - self.units_completed

    def summary(self) -> str:
        return (
            f"{self.units_completed}/{self.units_planned} units, "
            f"{self.rows_loaded:,} rows, {self.requests_made} requests "
            f"({self.weight_spent:,.0f} weighted calls), "
            f"{self.units_from_cache} from cache, "
            f"{_humanise(self.seconds)} wall clock — {self.stopped_because}"
        )


# ---------------------------------------------------------------------------
# Stopping politely
# ---------------------------------------------------------------------------


class _Interruptible:
    """Turns SIGINT into a flag checked between units.

    The default KeyboardInterrupt lands wherever the interpreter happens to be
    — quite possibly mid-COPY or between the warehouse commit and the manifest
    write, which is the one window where a crash costs correctness rather than
    time. Catching the signal and finishing the unit in flight closes it. A
    second press restores the default handler, so an operator who really means
    it is never trapped.
    """

    def __init__(self) -> None:
        self.requested = False
        self._previous = None

    def __enter__(self) -> "_Interruptible":
        self._previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle)
        return self

    def _handle(self, signum: int, frame: types.FrameType | None) -> None:
        self.requested = True
        signal.signal(signal.SIGINT, self._previous)
        log.warning(
            "interrupt received — finishing the unit in flight, then stopping. "
            "Press Ctrl-C again to stop immediately."
        )

    def __exit__(self, *exc_info: object) -> None:
        if self._previous is not None:
            signal.signal(signal.SIGINT, self._previous)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run_backfill(
    *,
    grains: Sequence[Grain] = ("daily",),
    cities: Iterable[City] | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
    chunk_months: int | None = None,
    hourly_months: int | None = None,
    max_weight: float = DAILY_QUOTA_CALLS,
    max_units: int | None = None,
    root: Path | str | None = None,
    manifest: Manifest | None = None,
    engine: Engine | None = None,
    settings: Settings | None = None,
) -> BackfillResult:
    """Fetch, archive, and land every pending unit until the budget runs out.

    Args:
        max_weight: Weighted API calls this invocation may spend. Once it is
            gone the run stops *fetching* but keeps walking the queue, because
            a later unit already on disk costs nothing to land and stopping
            outright would strand work already paid for.
        max_units: Stop after this many units. For smoke tests and for
            deliberately interrupting a run.
    """
    resolved_settings = settings if settings is not None else get_settings()
    resolved_manifest = (
        manifest
        if manifest is not None
        else Manifest(resolved_settings.ingest_manifest_path)
    )
    plan = plan_backfill(
        grains=grains,
        cities=cities,
        start=start,
        end=end,
        chunk_months=chunk_months,
        hourly_months=hourly_months,
        manifest=resolved_manifest,
        settings=resolved_settings,
    )

    result = BackfillResult(
        batch_id=uuid.uuid4(),
        started_at=dt.datetime.now(dt.timezone.utc),
        grains=tuple(grains),
        units_planned=len(plan.units),
    )
    if not plan.units:
        log.info("nothing pending — the manifest covers every planned window")
        return result

    log.info(
        "starting: %d pending unit(s), %s weighted calls to spend, batch %s",
        len(plan.units),
        f"{max_weight:,.0f}",
        result.batch_id,
    )

    owns_engine = engine is None
    active_engine = engine or engine_from_settings(resolved_settings)
    session = build_session()
    started = time.perf_counter()
    consecutive_failures = 0
    budget_exhausted = False
    ingested_at = dt.datetime.now(dt.timezone.utc)
    verified_tables: set[str] = set()

    try:
        with _Interruptible() as interrupt:
            for unit in plan.units:
                if interrupt.requested:
                    result.stopped_because = "interrupted"
                    break
                if max_units is not None and result.units_completed >= max_units:
                    result.stopped_because = "unit limit reached"
                    break

                cached = archive.exists(unit, root, settings=resolved_settings)
                if not cached and (
                    budget_exhausted
                    or result.weight_spent + unit.weight > max_weight
                ):
                    if not budget_exhausted:
                        budget_exhausted = True
                        result.stopped_because = "daily quota budget spent"
                        log.warning(
                            "no more fetching: %s costs %.0f calls and only "
                            "%.0f of the budget remain",
                            unit.key,
                            unit.weight,
                            max_weight - result.weight_spent,
                        )
                    # Keep walking rather than breaking. A later unit already on
                    # disk from an interrupted run costs nothing to land, and
                    # stopping here would strand work already paid for.
                    continue

                try:
                    rows = _run_one(
                        unit,
                        cached=cached,
                        session=session,
                        engine=active_engine,
                        manifest=resolved_manifest,
                        batch_id=result.batch_id,
                        ingested_at=ingested_at,
                        root=root,
                        settings=resolved_settings,
                        verify_columns=TABLE_BY_GRAIN[unit.grain]
                        not in verified_tables,
                    )
                except ArchiveRateLimited as exc:
                    # The budget is gone whatever the local counter says.
                    result.failures.append((unit.key, str(exc)))
                    result.stopped_because = "rate limited by the API"
                    log.error("rate limited on %s; stopping", unit.key)
                    break
                except (ArchiveError, Exception) as exc:  # noqa: BLE001
                    consecutive_failures += 1
                    result.failures.append((unit.key, f"{type(exc).__name__}: {exc}"))
                    log.error("failed %s: %s", unit.key, exc)
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        result.stopped_because = (
                            f"{consecutive_failures} consecutive failures"
                        )
                        break
                    continue

                consecutive_failures = 0
                verified_tables.add(TABLE_BY_GRAIN[unit.grain])
                result.units_completed += 1
                result.rows_loaded += rows
                if cached:
                    result.units_from_cache += 1
                else:
                    result.requests_made += 1
                    result.weight_spent += unit.weight
                    _pace(plan, unit, interrupt)
    finally:
        session.close()
        if owns_engine:
            active_engine.dispose()
        result.seconds = time.perf_counter() - started

    log.info("finished: %s", result.summary())
    return result


def _run_one(
    unit: WorkUnit,
    *,
    cached: bool,
    session,
    engine: Engine,
    manifest: Manifest,
    batch_id: uuid.UUID,
    ingested_at: dt.datetime,
    root: Path | str | None,
    settings: Settings,
    verify_columns: bool,
) -> int:
    """Fetch if needed, load, then record. Returns rows landed."""
    if cached:
        log.info("%s: already archived, no request made", unit.key)
    else:
        log.info("%s: fetching %.0f weighted calls", unit.key, unit.weight)
        fetch_observations(
            unit.city_id,
            unit.start,
            unit.end,
            unit.grain,
            on_payload=archive.writer_for(unit, root, settings=settings),
            session=session,
            settings=settings,
        )

    with engine.begin() as connection:
        rows = load_unit(
            unit,
            connection,
            batch_id=batch_id,
            ingested_at=ingested_at,
            root=root,
            settings=settings,
            verify_columns=verify_columns,
        )

    # Only now. The rows are committed, so the manifest can honestly say so.
    manifest.record(unit, rows=rows)
    log.info("%s: %d rows landed and recorded", unit.key, rows)
    return rows


def _pace(plan: Plan, unit: WorkUnit, interrupt: _Interruptible) -> None:
    """Wait out the unit's share of the minutely allowance, interruptibly."""
    remaining = plan.delay_for(unit)
    deadline = time.monotonic() + remaining
    while not interrupt.requested:
        left = deadline - time.monotonic()
        if left <= 0:
            return
        time.sleep(min(left, 0.5))


# ---------------------------------------------------------------------------
# What actually landed
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CityCoverage:
    """One city's daily series in bronze, as the warehouse sees it."""

    city_id: str
    rows: int
    distinct_days: int
    first_day: dt.date | None
    last_day: dt.date | None
    #: Missing days at the widest hole, not the distance between the two rows
    #: either side of it: consecutive days are a gap of zero.
    largest_gap_days: int
    expected_days: int

    @property
    def completeness(self) -> float:
        """Distinct observations landed over observations the range asked for."""
        return self.distinct_days / self.expected_days if self.expected_days else 0.0

    @property
    def within_one_percent(self) -> bool:
        return abs(1.0 - self.completeness) <= 0.01

    @property
    def has_unexplained_gap(self) -> bool:
        return self.largest_gap_days > MAX_ACCEPTABLE_GAP_DAYS


def bronze_coverage(
    engine: Engine,
    *,
    grain: Grain = "daily",
    start: dt.date | None = None,
    end: dt.date | None = None,
    batch_ids: Sequence[uuid.UUID] | None = None,
) -> list[CityCoverage]:
    """Row counts, date spans, and the largest gap per city.

    Gaps are measured over *distinct* observation times, because bronze is
    append-only: a re-ingested window is two rows for the same day and would
    otherwise read as denser coverage than there is.
    """
    table = TABLE_BY_GRAIN[grain]
    step = dt.timedelta(days=1) if grain == "daily" else dt.timedelta(hours=1)

    # The acceptance report wants every row ever landed and passes nothing.
    # Asking "what did *this* run land" — or inspecting one run while another is
    # still writing — needs the filter.
    scoped = batch_ids is not None
    batches = [str(b) for b in (batch_ids or ())]
    predicate = "where batch_id = any(cast(:batches as uuid[]))" if scoped else ""

    with engine.connect() as connection:
        rows = connection.execute(
            text(
                f"""
                with observed as (
                    select distinct city_id, observation_time
                    from {BRONZE_SCHEMA}.{table} {predicate}
                ),
                stepped as (
                    select city_id, observation_time,
                           observation_time - lag(observation_time) over (
                               partition by city_id order by observation_time
                           ) as step
                    from observed
                )
                select o.city_id,
                       (select count(*) from {BRONZE_SCHEMA}.{table} t
                        where t.city_id = o.city_id
                          and (:scoped = false
                               or t.batch_id = any(cast(:batches as uuid[]))))   as rows,
                       count(*)                                  as distinct_times,
                       min(o.observation_time)                   as first_time,
                       max(o.observation_time)                   as last_time,
                       coalesce(max(s.step), interval '0')       as largest_step
                from observed o
                join stepped s
                  on s.city_id = o.city_id
                 and s.observation_time = o.observation_time
                group by o.city_id
                order by o.city_id
                """
            ),
            {"scoped": scoped, "batches": batches},
        ).fetchall()

    resolved_end = end or dt.datetime.now(dt.timezone.utc).date()
    coverage: list[CityCoverage] = []
    for city_id, row_count, distinct_times, first, last, largest_step in rows:
        first_day = first.date() if first else None
        last_day = last.date() if last else None
        # Measured against the range that was *asked for*, not against the
        # city's own last row. Clipping to last_day would report a city holding
        # only 1995 as 100% complete — 365 days observed out of 365 expected —
        # which is precisely the state this gate exists to catch.
        window_start = start or first_day
        expected = 0
        if window_start:
            span = (resolved_end - window_start).days + 1
            expected = span if grain == "daily" else span * 24
        # A step of one unit is contiguous; anything more is a gap of the
        # difference. Reported in days for both grains so one threshold reads
        # the same way.
        gap = max(largest_step - step, dt.timedelta(0))
        coverage.append(
            CityCoverage(
                city_id=city_id,
                rows=row_count,
                distinct_days=distinct_times,
                first_day=first_day,
                last_day=last_day,
                largest_gap_days=gap.days,
                expected_days=expected,
            )
        )
    return coverage


#: The columns the Storm Dynamics view reads. A row present but null in these
#: is a row that feeds nothing, so the acceptance report names them.
STORM_DYNAMICS_COLUMNS: Final[tuple[str, ...]] = (
    "temperature_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "surface_pressure",
)

#: Neon's free plan. Bronze never goes there — only gold marts are promoted —
#: but the hourly table is the largest thing the pipeline builds, so it is the
#: number worth checking a design against.
NEON_STORAGE_BUDGET_BYTES: Final[int] = 500 * 1000 * 1000


@dataclass(frozen=True)
class TableSize:
    """One table's footprint, as Postgres accounts for it."""

    table: str
    rows: int
    total_bytes: int
    table_bytes: int
    index_bytes: int
    dead_rows: int = 0
    #: (column, average bytes) for the widest column, and the row's total
    #: payload — enough to say where the space actually goes.
    widest_column: tuple[str, float] | None = None
    payload_bytes: float = 0.0

    @property
    def bytes_per_row(self) -> float:
        return self.total_bytes / self.rows if self.rows else 0.0

    @property
    def dead_share(self) -> float:
        """Rows deleted or updated but not yet vacuumed.

        ``pg_total_relation_size`` counts them, so a table that has been loaded
        and cleared a few times reads far larger than it is — 31 MB against a
        true 12 MB, in the first measurement taken here.
        """
        total = self.rows + self.dead_rows
        return self.dead_rows / total if total else 0.0

    def share_of_neon_budget(self) -> float:
        return self.total_bytes / NEON_STORAGE_BUDGET_BYTES

    def project(self, rows: int) -> int:
        """This table at ``rows`` rows, at the density measured here."""
        return round(self.bytes_per_row * rows)


def table_size(engine: Engine, grain: Grain) -> TableSize:
    """Measure a bronze table including its indexes and TOAST."""
    table = TABLE_BY_GRAIN[grain]
    qualified = f"{BRONZE_SCHEMA}.{table}"
    with engine.connect() as connection:
        rows = connection.execute(
            text(f"select count(*) from {qualified}")
        ).scalar_one()
        total, relation, indexes = connection.execute(
            text(
                "select pg_total_relation_size(:t), pg_table_size(:t), "
                "pg_indexes_size(:t)"
            ),
            {"t": qualified},
        ).one()
        dead = connection.execute(
            text(
                "select coalesce(n_dead_tup, 0) from pg_stat_user_tables "
                "where schemaname = :s and relname = :r"
            ),
            {"s": BRONZE_SCHEMA, "r": table},
        ).scalar() or 0

        widest: tuple[str, float] | None = None
        payload = 0.0
        if rows:
            columns = connection.execute(
                text(
                    "select column_name from information_schema.columns "
                    "where table_schema = :s and table_name = :r "
                    "order by ordinal_position"
                ),
                {"s": BRONZE_SCHEMA, "r": table},
            ).scalars().all()
            averages = connection.execute(
                text(
                    "select "
                    + ", ".join(
                        f'avg(pg_column_size("{c}")) as "{c}"' for c in columns
                    )
                    + f" from {qualified}"
                )
            ).one()
            sizes = {c: float(v or 0.0) for c, v in zip(columns, averages)}
            payload = sum(sizes.values())
            widest = max(sizes.items(), key=lambda kv: kv[1])

    return TableSize(
        table=qualified,
        rows=rows,
        total_bytes=total,
        table_bytes=relation,
        index_bytes=indexes,
        dead_rows=dead,
        widest_column=widest,
        payload_bytes=payload,
    )


def column_population(
    engine: Engine, grain: Grain, columns: Sequence[str]
) -> dict[str, tuple[int, int]]:
    """Per column: rows populated, rows null."""
    table = f"{BRONZE_SCHEMA}.{TABLE_BY_GRAIN[grain]}"
    selects = ", ".join(
        f'count("{c}") as "{c}_filled", '
        f'count(*) - count("{c}") as "{c}_null"'
        for c in columns
    )
    with engine.connect() as connection:
        row = connection.execute(text(f"select {selects} from {table}")).one()
    values = dict(zip(row._fields, row))
    return {c: (values[f"{c}_filled"], values[f"{c}_null"]) for c in columns}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _humanise(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def _configure_logging(settings: Settings) -> Path:
    """Console plus a per-run file under LOG_DIR, so a long run leaves a record."""
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = settings.log_dir / f"backfill-{stamp}.log"
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(path, encoding="utf-8")],
        force=True,
    )
    return path


def _bytes(count: float) -> str:
    for unit, size in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if count >= size:
            return f"{count / size:.1f} {unit}"
    return f"{count:.0f} B"


def _print_report(
    engine: Engine,
    grain: Grain,
    expected_start: dt.date,
    expected_end: dt.date | None = None,
) -> int:
    coverage = bronze_coverage(
        engine, grain=grain, start=expected_start, end=expected_end
    )
    registry = load_cities()
    present = {c.city_id for c in coverage}
    missing = [city.id for city in registry if city.id not in present]

    print(f"bronze_raw.{TABLE_BY_GRAIN[grain]} — {len(present)}/{len(registry)} cities\n")
    width = max((len(c.city_id) for c in coverage), default=10)
    print(
        f"  {'city'.ljust(width)}  {'rows':>8}  {'days':>7}  {'first':10}  "
        f"{'last':10}  {'cover':>6}  gap"
    )
    print("  " + "-" * (width + 56))
    for entry in sorted(coverage, key=lambda c: c.city_id):
        flag = ""
        if not entry.within_one_percent:
            flag += " <1%!"
        if entry.has_unexplained_gap:
            flag += f" gap {entry.largest_gap_days}d!"
        print(
            f"  {entry.city_id.ljust(width)}  {entry.rows:>8,}  "
            f"{entry.distinct_days:>7,}  {str(entry.first_day):10}  "
            f"{str(entry.last_day):10}  {entry.completeness:>5.1%}  "
            f"{entry.largest_gap_days:>3}d{flag}"
        )

    print()
    if missing:
        print(f"  MISSING {len(missing)} cities: {missing}")
    off = [c.city_id for c in coverage if not c.within_one_percent]
    if off:
        print(f"  outside 1% of expected: {off}")
    gapped = [
        f"{c.city_id} ({c.largest_gap_days}d)"
        for c in coverage
        if c.has_unexplained_gap
    ]
    if gapped:
        print(f"  gaps over {MAX_ACCEPTABLE_GAP_DAYS} days: {gapped}")
    if not (missing or off or gapped):
        print("  all cities present, within 1%, no gap over "
              f"{MAX_ACCEPTABLE_GAP_DAYS} days")

    total_rows = sum(c.rows for c in coverage)
    print(f"\n  {total_rows:,} rows total, {expected_start}..{expected_end or 'now'}")

    # The columns the downstream view actually reads. A row present but null in
    # these is a row that feeds nothing.
    checked = STORM_DYNAMICS_COLUMNS if grain == "hourly" else (
        "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
        "wind_speed_10m_max",
    )
    unpopulated = []
    if total_rows:
        print()
        for column, (filled, nulls) in column_population(
            engine, grain, checked
        ).items():
            share = filled / (filled + nulls) if filled + nulls else 0.0
            print(f"  {column:24} {filled:>9,} populated  {nulls:>7,} null  "
                  f"{share:6.2%}")
            if nulls:
                unpopulated.append(column)

    if unpopulated:
        print(
            f"\n  columns carrying nulls: {unpopulated}. Bronze preserves nulls "
            "rather than\n  filling them, so these are a finding to explain, "
            "not necessarily a defect."
        )

    size = table_size(engine, grain)
    print(
        f"\n  {size.table}: {_bytes(size.total_bytes)} "
        f"({_bytes(size.table_bytes)} table + {_bytes(size.index_bytes)} indexes)"
    )
    if size.rows:
        print(f"  {size.bytes_per_row:.0f} bytes/row including indexes")
    if size.dead_share > 0.1:
        print(
            f"  {size.dead_rows:,} dead rows ({size.dead_share:.0%}) are counted "
            f"in that figure — run VACUUM FULL {size.table} for the true size"
        )
    if size.widest_column and size.payload_bytes:
        column, average = size.widest_column
        print(
            f"  widest column is {column} at {average:.0f} B, "
            f"{average / size.payload_bytes:.0%} of the row payload"
        )
    print(
        f"  {size.share_of_neon_budget():.1%} of Neon's "
        f"{_bytes(NEON_STORAGE_BUDGET_BYTES)} free-plan allowance "
        "— bronze stays local, this is the yardstick only"
    )

    # The table is rarely full when someone asks how big it will be.
    per_city = max((c.expected_days for c in coverage), default=0)
    expected_total = per_city * len(load_cities())
    if size.rows and expected_total > size.rows:
        projected = size.project(expected_total)
        print(
            f"  projected at {expected_total:,} rows "
            f"({len(load_cities())} cities complete): {_bytes(projected)}, "
            f"{projected / NEON_STORAGE_BUDGET_BYTES:.0%} of that allowance"
        )

    return 1 if (missing or off or gapped or unpopulated) else 0


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the historical backfill.")
    parser.add_argument("--grain", choices=("daily", "hourly"), default="daily")
    parser.add_argument("--city", action="append", dest="cities", metavar="CITY_ID")
    parser.add_argument("--start", type=dt.date.fromisoformat, default=None)
    parser.add_argument("--end", type=dt.date.fromisoformat, default=None)
    parser.add_argument("--chunk-months", type=int, default=None)
    parser.add_argument(
        "--hourly-months",
        type=int,
        default=None,
        help="how far the hourly grain reaches back from the anchor "
             "(default: INGEST_HOURLY_MONTHS)",
    )
    parser.add_argument(
        "--anchor",
        type=dt.date.fromisoformat,
        default=None,
        dest="anchor",
        help="the date the trailing hourly window ends at; same thing as "
             "--end, named for what it does. Pass it for a reproducible plan — "
             "left to the default, 'the trailing 24 months' names a different "
             "window tomorrow",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=DAILY_QUOTA_CALLS,
        help="weighted API calls this invocation may spend (default: one day's free tier)",
    )
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--report", action="store_true", help="report bronze coverage, then exit"
    )
    args = parser.parse_args(argv)
    if args.anchor is not None and args.end is not None and args.anchor != args.end:
        parser.error("--anchor and --end are the same date; pass only one")
    args.end = args.end or args.anchor

    settings = get_settings()
    registry = load_cities()
    selected = None
    if args.cities:
        try:
            selected = [registry[city_id] for city_id in args.cities]
        except Exception as exc:  # CityConfigError carries the known ids
            print(f"FAILED  {exc}", file=sys.stderr)
            return 1

    if args.report:
        logging.basicConfig(level=settings.log_level)
        engine = engine_from_settings(settings)
        try:
            from ingestion.planner import BACKFILL_START, archive_end_date

            start = args.start
            if start is None:
                start = (
                    BACKFILL_START
                    if args.grain == "daily"
                    else _add_months(
                        args.end or archive_end_date(),
                        -(args.hourly_months or settings.ingest_hourly_months),
                    )
                )
            return _print_report(engine, args.grain, start, args.end)
        finally:
            engine.dispose()

    manifest = Manifest(args.manifest or settings.ingest_manifest_path)

    if args.dry_run:
        logging.basicConfig(level=settings.log_level)
        plan = plan_backfill(
            grains=(args.grain,),
            cities=selected,
            start=args.start,
            end=args.end,
            chunk_months=args.chunk_months,
            hourly_months=args.hourly_months,
            manifest=manifest,
            settings=settings,
        )
        cached = sum(
            1 for u in plan.units if archive.exists(u, args.root, settings=settings)
        )
        print(
            f"{len(plan.units)} pending unit(s), {plan.expected_rows:,} rows, "
            f"{plan.weight:,.0f} weighted calls"
        )
        print(f"  {len(plan.skipped)} already in the manifest")
        print(f"  {cached} already archived — no request needed")
        print(f"  budget {args.max_weight:,.0f} calls covers "
              f"{len(plan.within_daily_quota())} of them")
        print(f"  paced runtime ~{_humanise(plan.estimated_seconds)}")
        return 0

    log_path = _configure_logging(settings)
    print(f"logging to {log_path}\n")
    result = run_backfill(
        grains=(args.grain,),
        cities=selected,
        start=args.start,
        end=args.end,
        chunk_months=args.chunk_months,
        hourly_months=args.hourly_months,
        max_weight=args.max_weight,
        max_units=args.max_units,
        root=args.root,
        manifest=manifest,
        settings=settings,
    )

    print(f"\n  batch_id     {result.batch_id}")
    print(f"  units        {result.units_completed} of {result.units_planned}")
    print(f"  requests     {result.requests_made} "
          f"({result.units_from_cache} served from the archive)")
    print(f"  weighted     {result.weight_spent:,.0f} API calls")
    print(f"  rows         {result.rows_loaded:,}")
    print(f"  wall clock   {_humanise(result.seconds)}")
    print(f"  stopped      {result.stopped_because}")
    if result.failures:
        print(f"\n  {len(result.failures)} failure(s):")
        for key, message in result.failures[:10]:
            print(f"    {key}: {message[:140]}")
    if result.units_remaining:
        print(
            f"\n  {result.units_remaining} unit(s) still pending. Re-run this "
            "command to continue;\n  the manifest makes it resume rather than "
            "restart."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
