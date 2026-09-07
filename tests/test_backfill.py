"""Tests for the backfill runner.

The runner is the only module that drives the others, so what is tested here is
ordering and stopping: that a unit already on disk costs no request, that the
manifest is written only after the warehouse commit, and that every way of
stopping — budget, interrupt, rate limit, repeated failure — leaves a state the
next invocation can resume from.

Network is scripted. The database is real, inside its own batch, deleted
afterwards; the runner commits per unit by design, so a wrapping transaction
would be testing something the runner does not do.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("pandas")
sqlalchemy = pytest.importorskip("sqlalchemy")
requests = pytest.importorskip("requests")

from sqlalchemy import text  # noqa: E402

from cities import load_cities  # noqa: E402
from config import ConfigError, Settings, get_settings  # noqa: E402
from ingestion import archive, backfill  # noqa: E402
from ingestion.backfill import (  # noqa: E402
    MAX_ACCEPTABLE_GAP_DAYS,
    MAX_CONSECUTIVE_FAILURES,
    _Interruptible,
    _pace,
    bronze_coverage,
    run_backfill,
)
from ingestion.loader import BRONZE_SCHEMA, TABLE_BY_GRAIN, engine_from_settings  # noqa: E402
from ingestion.planner import Manifest, Plan, plan_backfill  # noqa: E402
from http_fixtures import daily_payload, responds, session_for  # noqa: E402

CITY = "london"
START = dt.date(2020, 1, 1)
END = dt.date(2022, 12, 31)
UNITS_PER_CITY = 3


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "raw"


@pytest.fixture
def manifest(tmp_path: Path) -> Manifest:
    return Manifest(tmp_path / "manifest.jsonl")


@pytest.fixture
def settings(root: Path, tmp_path: Path) -> Settings:
    return dataclasses.replace(
        get_settings(),
        openmeteo_base_url="https://archive-api.test/v1/archive",
        data_raw_dir=root,
        ingest_manifest_path=tmp_path / "manifest.jsonl",
        max_retry_attempts=1,
    )


@pytest.fixture(scope="module")
def engine():
    try:
        get_settings().require_database_url()
    except ConfigError as exc:
        pytest.skip(f"no DATABASE_URL: {exc}")
    built = engine_from_settings()
    try:
        with built.connect() as connection:
            connection.execute(text("select 1"))
    except Exception as exc:  # noqa: BLE001
        built.dispose()
        pytest.skip(f"database unreachable: {exc}")
    yield built
    built.dispose()


@pytest.fixture
def cleanup(engine):
    """Delete whatever a run wrote, however it ended."""
    batches: list = []
    yield batches
    with engine.begin() as connection:
        for batch_id in batches:
            for table in TABLE_BY_GRAIN.values():
                connection.execute(
                    text(f"delete from {BRONZE_SCHEMA}.{table} "
                         "where batch_id = :batch"),
                    {"batch": str(batch_id)},
                )


@pytest.fixture
def instant(monkeypatch):
    """Drop the politeness delay; pacing has its own test."""
    monkeypatch.setattr(Plan, "delay_for", lambda self, unit: 0.0)


def units_for(cities=(CITY,), settings=None, manifest=None):
    registry = load_cities()
    return plan_backfill(
        grains=("daily",),
        cities=[registry[c] for c in cities],
        start=START,
        end=END,
        manifest=manifest,
        settings=settings,
    ).units


def script_for(units) -> list:
    """One well-formed response per unit, in order."""
    return [
        responds(json_body=daily_payload(unit.days, unit.start)) for unit in units
    ]


@pytest.fixture
def scripted(monkeypatch):
    """Install a scripted session and hand back the adapter that recorded it."""
    holder: dict = {}

    def install(script: list):
        session, adapter = session_for(script)
        monkeypatch.setattr(backfill, "build_session", lambda: session)
        holder["adapter"] = adapter
        return adapter

    install.holder = holder  # type: ignore[attr-defined]
    return install


def run(settings, manifest, root, cleanup, **kwargs):
    result = run_backfill(
        grains=("daily",),
        cities=[load_cities()[CITY]],
        start=START,
        end=END,
        root=root,
        manifest=manifest,
        settings=settings,
        **kwargs,
    )
    cleanup.append(result.batch_id)
    return result


def landed(engine, batch_id) -> int:
    with engine.connect() as connection:
        return connection.execute(
            text(f"select count(*) from {BRONZE_SCHEMA}.observations_daily "
                 "where batch_id = :batch"),
            {"batch": str(batch_id)},
        ).scalar()


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_a_run_fetches_archives_loads_and_records(
    settings, manifest, root, engine, cleanup, scripted, instant
) -> None:
    pending = units_for(settings=settings, manifest=manifest)
    adapter = scripted(script_for(pending))

    result = run(settings, manifest, root, cleanup, engine=engine)

    assert result.units_completed == UNITS_PER_CITY
    assert result.requests_made == UNITS_PER_CITY
    assert len(adapter.calls) == UNITS_PER_CITY
    assert result.rows_loaded == sum(u.expected_rows for u in pending)
    assert result.stopped_because == "complete"

    # Every stage left its trace.
    assert all(archive.exists(u, root) for u in pending)
    assert len(manifest) == UNITS_PER_CITY
    assert landed(engine, result.batch_id) == result.rows_loaded


def test_weight_spent_is_the_sum_of_what_was_fetched(
    settings, manifest, root, engine, cleanup, scripted, instant
) -> None:
    pending = units_for(settings=settings, manifest=manifest)
    scripted(script_for(pending))
    result = run(settings, manifest, root, cleanup, engine=engine)
    assert result.weight_spent == pytest.approx(sum(u.weight for u in pending))


def test_nothing_pending_makes_no_request(
    settings, manifest, root, engine, cleanup, scripted, instant
) -> None:
    pending = units_for(settings=settings, manifest=manifest)
    for unit in pending:
        manifest.record(unit, rows=unit.expected_rows)
    adapter = scripted([])

    result = run(settings, manifest, root, cleanup, engine=engine)
    assert result.units_planned == 0
    assert adapter.calls == []


# ---------------------------------------------------------------------------
# The archive is the fetch cache
# ---------------------------------------------------------------------------


def test_an_archived_unit_costs_no_request(
    settings, manifest, root, engine, cleanup, scripted, instant
) -> None:
    """A run killed between fetching and loading must not re-pay for the fetch."""
    pending = units_for(settings=settings, manifest=manifest)
    first = pending[0]
    archive.write(first, json.dumps(daily_payload(first.days, first.start)).encode(),
                  root)
    adapter = scripted(script_for(pending[1:]))

    result = run(settings, manifest, root, cleanup, engine=engine)

    assert result.units_completed == UNITS_PER_CITY
    assert result.units_from_cache == 1
    assert result.requests_made == UNITS_PER_CITY - 1
    assert len(adapter.calls) == UNITS_PER_CITY - 1
    assert result.weight_spent == pytest.approx(
        sum(u.weight for u in pending[1:])
    ), "a cached unit must not be billed"
    assert landed(engine, result.batch_id) == sum(u.expected_rows for u in pending)


def test_a_cached_unit_is_loaded_even_when_the_budget_is_gone(
    settings, manifest, root, engine, cleanup, scripted, instant
) -> None:
    """It costs nothing, so refusing it would strand work already paid for."""
    pending = units_for(settings=settings, manifest=manifest)
    for unit in pending:
        archive.write(
            unit, json.dumps(daily_payload(unit.days, unit.start)).encode(), root
        )
    adapter = scripted([])

    result = run(settings, manifest, root, cleanup, engine=engine, max_weight=0.0)

    assert result.units_completed == UNITS_PER_CITY
    assert result.requests_made == 0
    assert adapter.calls == []


# ---------------------------------------------------------------------------
# Ordering: the manifest is the last thing written
# ---------------------------------------------------------------------------


def test_the_manifest_is_not_written_when_the_load_fails(
    settings, manifest, root, engine, cleanup, scripted, instant, monkeypatch
) -> None:
    """A manifest entry must never claim rows the warehouse does not have."""
    pending = units_for(settings=settings, manifest=manifest)
    scripted(script_for(pending))

    def refuse(*args, **kwargs):
        raise RuntimeError("warehouse unavailable")

    monkeypatch.setattr(backfill, "load_unit", refuse)
    result = run(settings, manifest, root, cleanup, engine=engine)

    assert result.units_completed == 0
    assert len(manifest) == 0
    # The payload is on disk regardless, so the retry costs no quota.
    assert archive.exists(pending[0], root)


def test_a_recorded_unit_is_skipped_next_time(
    settings, manifest, root, engine, cleanup, scripted, instant
) -> None:
    pending = units_for(settings=settings, manifest=manifest)
    scripted(script_for(pending))
    run(settings, manifest, root, cleanup, engine=engine)

    scripted([])
    again = run(settings, Manifest(manifest.path), root, cleanup, engine=engine)
    assert again.units_planned == 0


# ---------------------------------------------------------------------------
# Every way of stopping
# ---------------------------------------------------------------------------


def test_the_budget_stops_the_run_before_it_is_exceeded(
    settings, manifest, root, engine, cleanup, scripted, instant
) -> None:
    pending = units_for(settings=settings, manifest=manifest)
    scripted(script_for(pending))
    budget = pending[0].weight + pending[1].weight

    result = run(settings, manifest, root, cleanup, engine=engine, max_weight=budget)

    assert result.units_completed == 2
    assert result.weight_spent <= budget
    assert result.stopped_because == "daily quota budget spent"
    assert result.units_remaining == 1


def test_a_unit_limit_stops_the_run(
    settings, manifest, root, engine, cleanup, scripted, instant
) -> None:
    pending = units_for(settings=settings, manifest=manifest)
    scripted(script_for(pending))
    result = run(settings, manifest, root, cleanup, engine=engine, max_units=1)
    assert result.units_completed == 1
    assert result.stopped_because == "unit limit reached"


def test_a_rate_limit_stops_the_run_immediately(
    settings, manifest, root, engine, cleanup, scripted, instant
) -> None:
    """The budget is gone whatever the local counter believes."""
    pending = units_for(settings=settings, manifest=manifest)
    scripted(
        [
            responds(json_body=daily_payload(pending[0].days, pending[0].start)),
            responds(429, text="Minutely API request limit exceeded."),
        ]
    )
    result = run(settings, manifest, root, cleanup, engine=engine)

    assert result.units_completed == 1
    assert result.stopped_because == "rate limited by the API"
    assert len(result.failures) == 1


def test_repeated_failures_stop_the_run(
    settings, manifest, root, engine, cleanup, scripted, instant
) -> None:
    scripted([responds(400, json_body={"error": True, "reason": "nope"})] * 8)
    result = run_backfill(
        grains=("daily",),
        cities=[load_cities()[c] for c in ("london", "tokyo")],
        start=START,
        end=END,
        root=root,
        manifest=manifest,
        settings=settings,
        engine=engine,
    )
    cleanup.append(result.batch_id)
    assert result.units_completed == 0
    assert len(result.failures) == MAX_CONSECUTIVE_FAILURES
    assert "consecutive failures" in result.stopped_because


def test_one_failure_does_not_stop_the_run(
    settings, manifest, root, engine, cleanup, scripted, instant
) -> None:
    pending = units_for(settings=settings, manifest=manifest)
    scripted(
        [
            responds(400, json_body={"error": True, "reason": "one bad window"}),
            *script_for(pending[1:]),
        ]
    )
    result = run(settings, manifest, root, cleanup, engine=engine)

    assert result.units_completed == UNITS_PER_CITY - 1
    assert len(result.failures) == 1
    assert result.stopped_because == "complete"


# ---------------------------------------------------------------------------
# Interruption
# ---------------------------------------------------------------------------


def test_an_interrupt_finishes_the_unit_in_flight(
    settings, manifest, root, engine, cleanup, scripted, instant, monkeypatch
) -> None:
    """Stopping mid-unit is the one window where a crash costs correctness."""
    pending = units_for(settings=settings, manifest=manifest)
    scripted(script_for(pending))

    real_pace = backfill._pace
    seen: list = []

    def interrupt_after_first(plan, unit, interrupt):
        seen.append(unit)
        if len(seen) == 1:
            interrupt.requested = True
        return real_pace(plan, unit, interrupt)

    monkeypatch.setattr(backfill, "_pace", interrupt_after_first)
    result = run(settings, manifest, root, cleanup, engine=engine)

    assert result.units_completed == 1
    assert result.stopped_because == "interrupted"
    # The unit that was in flight is complete in every store, not half-done.
    assert archive.exists(pending[0], root)
    assert manifest.is_complete(pending[0])
    assert landed(engine, result.batch_id) == pending[0].expected_rows


def test_an_interrupted_run_resumes_where_it_stopped(
    settings, manifest, root, engine, cleanup, scripted, instant, monkeypatch
) -> None:
    pending = units_for(settings=settings, manifest=manifest)
    scripted(script_for(pending))

    real_pace = backfill._pace
    seen: list = []

    def interrupt_after_first(plan, unit, interrupt):
        seen.append(unit)
        if len(seen) == 1:
            interrupt.requested = True
        return real_pace(plan, unit, interrupt)

    monkeypatch.setattr(backfill, "_pace", interrupt_after_first)
    first = run(settings, manifest, root, cleanup, engine=engine)
    assert first.units_completed == 1

    monkeypatch.setattr(backfill, "_pace", real_pace)
    scripted(script_for(pending[1:]))
    second = run(settings, Manifest(manifest.path), root, cleanup, engine=engine)

    assert second.units_planned == UNITS_PER_CITY - 1
    assert second.units_completed == UNITS_PER_CITY - 1
    assert second.requests_made == UNITS_PER_CITY - 1
    total = landed(engine, first.batch_id) + landed(engine, second.batch_id)
    assert total == sum(u.expected_rows for u in pending)


def test_the_signal_handler_sets_a_flag_rather_than_raising() -> None:
    with _Interruptible() as interrupt:
        assert not interrupt.requested
        interrupt._handle(2, None)
        assert interrupt.requested


def test_pacing_returns_early_when_an_interrupt_arrives(tmp_path) -> None:
    """A 30-second wait must not be 30 seconds of ignoring Ctrl-C."""
    plan = plan_backfill(
        grains=("daily",),
        cities=[load_cities()[CITY]],
        start=START,
        end=END,
        manifest=Manifest(tmp_path / "manifest.jsonl"),
    )
    unit = plan.units[0]

    with _Interruptible() as interrupt:
        interrupt.requested = True
        started = time.monotonic()
        _pace(plan, unit, interrupt)
        assert time.monotonic() - started < 0.5
        assert plan.delay_for(unit) > 1.0


# ---------------------------------------------------------------------------
# Coverage reporting
# ---------------------------------------------------------------------------


def test_coverage_reports_a_complete_city(
    settings, manifest, root, engine, cleanup, scripted, instant
) -> None:
    pending = units_for(settings=settings, manifest=manifest)
    scripted(script_for(pending))
    run(settings, manifest, root, cleanup, engine=engine)

    entry = next(
        c for c in bronze_coverage(
            engine, grain="daily", start=START, batch_ids=cleanup
        )
        if c.city_id == CITY
    )
    assert entry.first_day == START
    assert entry.last_day == END
    assert entry.largest_gap_days == 0
    assert not entry.has_unexplained_gap


def test_coverage_finds_a_gap(
    settings, manifest, root, engine, cleanup, scripted, instant
) -> None:
    """A missing middle year is exactly what a half-finished backfill leaves."""
    pending = units_for(settings=settings, manifest=manifest)
    keep = [pending[0], pending[2]]
    for unit in keep:
        archive.write(
            unit, json.dumps(daily_payload(unit.days, unit.start)).encode(), root
        )
    scripted([])
    result = run_backfill(
        grains=("daily",), cities=[load_cities()[CITY]], start=START, end=END,
        root=root, manifest=manifest, settings=settings, engine=engine,
        max_weight=0.0,
    )
    cleanup.append(result.batch_id)
    assert result.units_completed == 2, "both cached units land despite no budget"

    entry = next(
        c for c in bronze_coverage(
            engine, grain="daily", start=START, batch_ids=cleanup
        )
        if c.city_id == CITY
    )
    assert entry.largest_gap_days == 365  # all of 2021, which is not a leap year
    assert entry.has_unexplained_gap
    assert entry.largest_gap_days > MAX_ACCEPTABLE_GAP_DAYS


def test_coverage_counts_distinct_days_not_rows(
    settings, manifest, root, engine, cleanup, scripted, instant, tmp_path
) -> None:
    """Bronze is append-only; a re-ingest would otherwise read as denser."""
    pending = units_for(settings=settings, manifest=manifest)
    scripted(script_for(pending) + script_for(pending))
    first = run(settings, manifest, root, cleanup, engine=engine)
    second = run(settings, Manifest(tmp_path / "second.jsonl"), root, cleanup,
                 engine=engine)
    assert second.units_completed == UNITS_PER_CITY

    entry = next(
        c for c in bronze_coverage(
            engine, grain="daily", start=START, batch_ids=cleanup
        )
        if c.city_id == CITY
    )
    assert entry.rows >= first.rows_loaded * 2
    assert entry.distinct_days == sum(u.expected_rows for u in pending)


def test_completeness_is_a_ratio_of_distinct_days_to_the_window() -> None:
    from ingestion.backfill import CityCoverage

    full = CityCoverage(CITY, 365, 365, START, END, 0, 365)
    assert full.completeness == 1.0
    assert full.within_one_percent

    short = CityCoverage(CITY, 300, 300, START, END, 20, 365)
    assert not short.within_one_percent
    assert short.has_unexplained_gap
