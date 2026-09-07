"""Proof that running ingestion twice does not duplicate data.

This matters beyond tidiness. OPS-01's orchestration runs ingestion on every
invocation, so a loader that is not idempotent means every ``make run``
corrupts the warehouse a little further.

The guarantee has two layers, and both are tested here because neither is
sufficient alone:

1.  **Run level — the manifest.** A unit recorded as landed is not planned
    again, so a re-run makes no requests and writes no rows. This is the layer
    ``make run`` relies on, and it is exact: the row count after a re-run is
    the same number, not a similar one.
2.  **Row level — append-only bronze, deduplicated in silver (DBT-02).** There
    is one window where layer 1 cannot hold: a crash between the warehouse
    commit and the manifest write leaves rows that nothing has recorded, and
    the next run lands them again. That direction is deliberate — duplicates
    are recoverable and holes are silent — and it is why deduplication is
    deferred to SQL rather than relied on here.

The last section runs the actual silver dedup over deliberately duplicated
bronze, so "deferred to DBT-02" is a demonstrated claim rather than a promise.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("pandas")
pytest.importorskip("sqlalchemy")

from sqlalchemy import text  # noqa: E402

from cities import load_cities  # noqa: E402
from ingestion import archive  # noqa: E402
from ingestion.backfill import duplication, run_backfill  # noqa: E402
from ingestion.loader import BRONZE_SCHEMA, TABLE_BY_GRAIN  # noqa: E402
from ingestion.planner import Manifest, plan_backfill  # noqa: E402
from http_fixtures import daily_payload, responds  # noqa: E402

CITIES = ("london", "tokyo")
START = dt.date(2020, 1, 1)
END = dt.date(2022, 12, 31)
UNITS_PER_CITY = 3
TABLE = f"{BRONZE_SCHEMA}.{TABLE_BY_GRAIN['daily']}"


def planned(settings, manifest, cities=CITIES):
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
    return [responds(json_body=daily_payload(u.days, u.start)) for u in units]


def ingest(settings, manifest, root, cleanup, cities=CITIES, **kwargs):
    result = run_backfill(
        grains=("daily",),
        cities=[load_cities()[c] for c in cities],
        start=START,
        end=END,
        root=root,
        manifest=manifest,
        settings=settings,
        **kwargs,
    )
    cleanup.append(result.batch_id)
    return result


def fingerprint(engine, batch_ids, city_id: str | None = None) -> tuple:
    """Enough of a city's landed rows to notice any change at all.

    Row count alone would miss a row replaced by a different one, so this also
    pins the id range and the sum of a value column.
    """
    predicate = "where batch_id = any(cast(:batches as uuid[]))"
    parameters: dict = {"batches": [str(b) for b in batch_ids]}
    if city_id:
        predicate += " and city_id = :city"
        parameters["city"] = city_id
    with engine.connect() as connection:
        return tuple(
            connection.execute(
                text(
                    "select count(*), min(id), max(id), "
                    "count(distinct observation_time), "
                    "coalesce(sum(temperature_2m_max)::numeric(20,3), 0) "
                    f"from {TABLE} {predicate}"
                ),
                parameters,
            ).one()
        )


def rows_for(engine, batch_ids, city_id: str) -> int:
    return fingerprint(engine, batch_ids, city_id)[0]


# ---------------------------------------------------------------------------
# Layer 1: a full re-run is a no-op
# ---------------------------------------------------------------------------


def test_a_full_rerun_produces_identical_row_counts(
    settings, manifest, root, engine, cleanup, scripted, instant
) -> None:
    """The acceptance criterion, stated as an equality rather than a tolerance."""
    units = planned(settings, manifest)
    scripted(script_for(units))
    first = ingest(settings, manifest, root, cleanup)
    assert first.units_completed == UNITS_PER_CITY * len(CITIES)

    before = fingerprint(engine, cleanup)

    # A second process, reading the manifest from disk as the next `make run`
    # would.
    adapter = scripted([])
    second = ingest(settings, Manifest(manifest.path), root, cleanup)

    assert second.units_planned == 0
    assert second.rows_loaded == 0
    assert adapter.calls == []
    assert fingerprint(engine, cleanup) == before


def test_a_rerun_makes_no_requests_and_no_writes(
    settings, manifest, root, engine, cleanup, scripted, instant
) -> None:
    units = planned(settings, manifest)
    scripted(script_for(units))
    ingest(settings, manifest, root, cleanup)

    adapter = scripted([])
    for _ in range(3):
        again = ingest(settings, Manifest(manifest.path), root, cleanup)
        assert (again.requests_made, again.rows_loaded) == (0, 0)
    assert adapter.calls == []


def test_ten_reruns_do_not_drift(
    settings, manifest, root, engine, cleanup, scripted, instant
) -> None:
    """`make run` is invoked far more often than the backfill changes."""
    units = planned(settings, manifest)
    scripted(script_for(units))
    ingest(settings, manifest, root, cleanup)
    expected = fingerprint(engine, cleanup)

    scripted([])
    for _ in range(10):
        ingest(settings, Manifest(manifest.path), root, cleanup)
    assert fingerprint(engine, cleanup) == expected


def test_a_rerun_leaves_no_duplicate_observations(
    settings, manifest, root, engine, cleanup, scripted, instant
) -> None:
    units = planned(settings, manifest)
    scripted(script_for(units))
    ingest(settings, manifest, root, cleanup)
    scripted([])
    ingest(settings, Manifest(manifest.path), root, cleanup)

    for city in CITIES:
        counted = duplication(engine, "daily", city_id=city, batch_ids=cleanup)
        assert counted.duplicate_rows == 0, city


# ---------------------------------------------------------------------------
# A partial re-run touches only what it re-runs
# ---------------------------------------------------------------------------


def test_rerunning_one_city_leaves_the_others_untouched(
    settings, manifest, root, engine, cleanup, scripted, instant
) -> None:
    """The blast radius of a re-ingest is exactly the window re-ingested."""
    units = planned(settings, manifest)
    scripted(script_for(units))
    ingest(settings, manifest, root, cleanup)

    untouched_before = fingerprint(engine, cleanup, "tokyo")
    london_before = fingerprint(engine, cleanup, "london")

    # Forget one city's history, as a targeted re-ingest would.
    kept = [e for e in Manifest(manifest.path).entries if e.city_id != "london"]
    manifest.path.write_text(
        "".join(e.as_json() + "\n" for e in kept), encoding="utf-8"
    )

    reduced = Manifest(manifest.path)
    london_units = planned(settings, reduced, cities=("london",))
    assert len(london_units) == UNITS_PER_CITY
    scripted(script_for(london_units))
    again = ingest(settings, reduced, root, cleanup, cities=("london",))
    assert again.units_completed == UNITS_PER_CITY

    assert fingerprint(engine, cleanup, "tokyo") == untouched_before
    assert fingerprint(engine, cleanup, "london") != london_before
    assert rows_for(engine, cleanup, "london") == london_before[0] * 2


def test_a_partial_rerun_duplicates_only_the_window_it_covers(
    settings, manifest, root, engine, cleanup, scripted, instant
) -> None:
    units = planned(settings, manifest, cities=("london",))
    scripted(script_for(units))
    ingest(settings, manifest, root, cleanup, cities=("london",))

    # Re-ingest one year of the three.
    one_year = units[1]
    kept = [
        e
        for e in Manifest(manifest.path).entries
        if not (e.start == one_year.start and e.end == one_year.end)
    ]
    manifest.path.write_text(
        "".join(e.as_json() + "\n" for e in kept), encoding="utf-8"
    )
    scripted(script_for([one_year]))
    again = ingest(settings, Manifest(manifest.path), root, cleanup,
                   cities=("london",))
    assert again.units_completed == 1

    counted = duplication(engine, "daily", city_id="london", batch_ids=cleanup)
    assert counted.duplicate_rows == one_year.expected_rows
    assert counted.distinct_observations == sum(u.expected_rows for u in units)


# ---------------------------------------------------------------------------
# Layer 2: the window layer 1 cannot cover
# ---------------------------------------------------------------------------


def test_a_lost_manifest_duplicates_rather_than_leaving_a_hole(
    settings, manifest, root, engine, cleanup, scripted, instant
) -> None:
    """A crash between the warehouse commit and the manifest write.

    The manifest is written last on purpose. Writing it first would let the
    same crash leave a manifest claiming rows that are not there — a hole no
    downstream model would report. Duplicates are the recoverable direction.
    """
    units = planned(settings, manifest, cities=("london",))
    scripted(script_for(units))
    first = ingest(settings, manifest, root, cleanup, cities=("london",))
    expected_observations = first.rows_loaded

    manifest.path.unlink()

    scripted(script_for(units))
    second = ingest(settings, Manifest(manifest.path), root, cleanup,
                    cities=("london",))
    assert second.units_completed == UNITS_PER_CITY

    counted = duplication(engine, "daily", city_id="london", batch_ids=cleanup)
    assert counted.rows == expected_observations * 2
    # Not one observation lost, and not one gained.
    assert counted.distinct_observations == expected_observations


def test_the_archive_means_a_lost_manifest_costs_no_quota(
    settings, manifest, root, engine, cleanup, scripted, instant
) -> None:
    """Re-landing is free; re-fetching would not be."""
    units = planned(settings, manifest, cities=("london",))
    scripted(script_for(units))
    ingest(settings, manifest, root, cleanup, cities=("london",))

    manifest.path.unlink()
    adapter = scripted([])
    second = ingest(settings, Manifest(manifest.path), root, cleanup,
                    cities=("london",))

    assert second.units_completed == UNITS_PER_CITY
    assert second.units_from_cache == UNITS_PER_CITY
    assert second.weight_spent == 0.0
    assert adapter.calls == []


def test_bronze_permits_the_duplicate_by_design(engine) -> None:
    """A unique key on the natural key would reject a legitimate re-ingest."""
    with engine.connect() as connection:
        definitions = connection.execute(
            text(
                "select indexdef from pg_indexes "
                "where schemaname = :s and tablename = :t"
            ),
            {"s": BRONZE_SCHEMA, "t": "observations_daily"},
        ).scalars().all()
    natural = [d for d in definitions if "(city_id, observation_time" in d]
    assert natural, definitions
    assert all("UNIQUE" not in d.upper() for d in natural)


# ---------------------------------------------------------------------------
# The deferral to DBT-02, demonstrated
# ---------------------------------------------------------------------------


DEDUP_SQL = f"""
    select id, city_id, observation_time, ingested_at
    from (
        select id, city_id, observation_time, ingested_at,
               row_number() over (
                   partition by city_id, observation_time
                   order by ingested_at desc, id desc
               ) as rn
        from {TABLE}
        where batch_id = any(cast(:batches as uuid[]))
    ) ranked
    where rn = 1
"""


def test_silver_dedup_collapses_duplicated_bronze(
    settings, manifest, root, engine, cleanup, scripted, instant
) -> None:
    """The strategy is only sound if the SQL it defers to actually works."""
    units = planned(settings, manifest, cities=("london",))
    scripted(script_for(units))
    first = ingest(settings, manifest, root, cleanup, cities=("london",))

    manifest.path.unlink()
    scripted(script_for(units))
    ingest(settings, Manifest(manifest.path), root, cleanup, cities=("london",))

    batches = {"batches": [str(b) for b in cleanup]}
    with engine.connect() as connection:
        total = connection.execute(
            text(f"select count(*) from {TABLE} "
                 "where batch_id = any(cast(:batches as uuid[]))"),
            batches,
        ).scalar()
        deduped = connection.execute(text(DEDUP_SQL), batches).fetchall()

    assert total == first.rows_loaded * 2
    assert len(deduped) == first.rows_loaded
    assert len({r.observation_time for r in deduped}) == first.rows_loaded


def test_silver_dedup_keeps_the_most_recent_ingest(
    settings, manifest, root, engine, cleanup, scripted, instant
) -> None:
    """Which copy survives is the whole point: the newest, not an arbitrary one."""
    units = planned(settings, manifest, cities=("london",))[:1]
    scripted(script_for(units))
    ingest(settings, manifest, root, cleanup, cities=("london",))

    manifest.path.unlink()
    scripted(script_for(units))
    second = ingest(settings, Manifest(manifest.path), root, cleanup,
                    cities=("london",), max_units=1)

    batches = {"batches": [str(b) for b in cleanup]}
    with engine.connect() as connection:
        surviving = connection.execute(text(DEDUP_SQL), batches).fetchall()
        newest = connection.execute(
            text(f"select max(ingested_at) from {TABLE} "
                 "where batch_id = any(cast(:batches as uuid[]))"),
            batches,
        ).scalar()

    assert second.units_completed == 1
    assert {r.ingested_at for r in surviving} == {newest}


def test_dedup_does_not_reach_across_cities(
    settings, manifest, root, engine, cleanup, scripted, instant
) -> None:
    """It partitions by city as well as time; two cities share every date."""
    units = planned(settings, manifest)
    scripted(script_for(units))
    first = ingest(settings, manifest, root, cleanup)

    with engine.connect() as connection:
        deduped = connection.execute(
            text(DEDUP_SQL), {"batches": [str(b) for b in cleanup]}
        ).fetchall()

    assert len(deduped) == first.rows_loaded
    assert len({r.city_id for r in deduped}) == len(CITIES)


# ---------------------------------------------------------------------------
# Everything upstream of the warehouse is idempotent too
# ---------------------------------------------------------------------------


def test_rearchiving_a_payload_is_byte_identical(root: Path) -> None:
    """Otherwise a re-run would rewrite the archive and look like a change."""
    from ingestion.planner import WorkUnit

    unit = WorkUnit(city_id="london", grain="daily", start=START,
                    end=dt.date(2020, 1, 3))
    raw = json.dumps(daily_payload(3, START)).encode("utf-8")
    first = archive.write(unit, raw, root).read_bytes()
    second = archive.write(unit, raw, root).read_bytes()
    assert first == second


def test_recording_the_same_unit_twice_is_visible_but_harmless(
    manifest, settings
) -> None:
    """The manifest is append-only; coverage is what matters, not line count."""
    units = planned(settings, manifest, cities=("london",))
    for unit in units:
        manifest.record(unit, rows=unit.expected_rows)
        manifest.record(unit, rows=unit.expected_rows)

    reopened = Manifest(manifest.path)
    assert len(reopened) == len(units) * 2
    assert all(reopened.is_complete(u) for u in units)
    assert planned(settings, reopened, cities=("london",)) == ()
