"""Tests for the reconciliation report.

Gap arithmetic is the kind of code that looks obviously right and is off by a
day. These tests insert known holes and check that each is found, measured, and
placed in the category that explains it.

Rows are inserted directly rather than through the loader, because what is
under test is the arithmetic over what landed rather than the landing, and every test uses a
synthetic city id so the real warehouse, which holds a backfill in progress,
cannot influence the result.
"""

from __future__ import annotations

import datetime as dt
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("sqlalchemy")

from sqlalchemy import text  # noqa: E402

from cities import City  # noqa: E402
from ingestion.client import ARCHIVE_START  # noqa: E402
from ingestion.loader import BRONZE_SCHEMA  # noqa: E402
from ingestion.planner import Manifest, WorkUnit  # noqa: E402
from ingestion.reconcile import (  # noqa: E402
    API_LIMITATION,
    ARCHIVE_BOUNDARY,
    CATEGORIES,
    NOT_INGESTED,
    UNEXPLAINED,
    _intersect,
    _subtract,
    reconcile,
    to_markdown,
)

START = dt.date(2020, 1, 1)
END = dt.date(2022, 12, 31)
DAYS = (END - START).days + 1  # 1096, one leap year


@pytest.fixture
def city() -> City:
    """A city the registry has never heard of, so bronze holds no rows for it."""
    return City(
        id=f"recon_{uuid.uuid4().hex[:8]}",
        name="Reconciliation Test",
        country="Testland",
        country_code="ZZ",
        region="Europe",
        lat=51.5,
        lon=-0.1,
        elevation_m=10.0,
        timezone="UTC",
        koppen="Cfb",
        season_model="four_season",
    )


@pytest.fixture
def landed(engine, city):
    """Insert observation days for the synthetic city; remove them afterwards."""
    inserted: list[str] = []

    def insert(*spans: tuple[dt.date, dt.date], times: int = 1) -> None:
        rows = []
        for span_start, span_end in spans:
            day = span_start
            while day <= span_end:
                rows.append(day)
                day += dt.timedelta(days=1)
        with engine.begin() as connection:
            for _ in range(times):
                connection.execute(
                    text(
                        f"insert into {BRONZE_SCHEMA}.observations_daily "
                        "(city_id, observation_time, batch_id) "
                        "values (:city, :when, :batch)"
                    ),
                    [
                        {
                            "city": city.id,
                            "when": dt.datetime.combine(
                                day, dt.time(), tzinfo=dt.timezone.utc
                            ),
                            "batch": str(uuid.uuid4()),
                        }
                        for day in rows
                    ],
                )
        inserted.append(city.id)

    yield insert
    with engine.begin() as connection:
        connection.execute(
            text(f"delete from {BRONZE_SCHEMA}.observations_daily "
                 "where city_id = :city"),
            {"city": city.id},
        )


def report_for(engine, city, manifest, **kwargs):
    return reconcile(
        engine,
        grain="daily",
        start=START,
        end=END,
        cities=[city],
        manifest=manifest,
        **kwargs,
    )


def only(report):
    assert len(report.cities) == 1
    return report.cities[0]


def cover(manifest, city, start: dt.date, end: dt.date, rows: int | None = None):
    unit = WorkUnit(city_id=city.id, grain="daily", start=start, end=end)
    manifest.record(unit, rows=unit.expected_rows if rows is None else rows)
    return unit


# ---------------------------------------------------------------------------
# Interval arithmetic
# ---------------------------------------------------------------------------


def test_intersect() -> None:
    a = (dt.date(2020, 1, 1), dt.date(2020, 6, 30))
    b = (dt.date(2020, 4, 1), dt.date(2020, 9, 30))
    assert _intersect(a, b) == (dt.date(2020, 4, 1), dt.date(2020, 6, 30))
    assert _intersect(a, (dt.date(2021, 1, 1), dt.date(2021, 2, 1))) is None
    # Touching at a single day still intersects on that day.
    assert _intersect(a, (dt.date(2020, 6, 30), dt.date(2021, 1, 1))) == (
        dt.date(2020, 6, 30),
        dt.date(2020, 6, 30),
    )


def test_subtract_removes_the_middle() -> None:
    whole = (dt.date(2020, 1, 1), dt.date(2020, 12, 31))
    cut = (dt.date(2020, 4, 1), dt.date(2020, 6, 30))
    assert _subtract(whole, [cut]) == [
        (dt.date(2020, 1, 1), dt.date(2020, 3, 31)),
        (dt.date(2020, 7, 1), dt.date(2020, 12, 31)),
    ]


def test_subtract_is_inclusive_at_both_ends() -> None:
    """Off-by-one here would silently mis-size every gap in the report."""
    whole = (dt.date(2020, 1, 1), dt.date(2020, 1, 10))
    assert _subtract(whole, [whole]) == []
    assert _subtract(whole, [(dt.date(2020, 1, 1), dt.date(2020, 1, 1))]) == [
        (dt.date(2020, 1, 2), dt.date(2020, 1, 10))
    ]
    assert _subtract(whole, [(dt.date(2020, 1, 10), dt.date(2020, 1, 10))]) == [
        (dt.date(2020, 1, 1), dt.date(2020, 1, 9))
    ]


def test_subtract_handles_several_cuts_in_any_order() -> None:
    whole = (dt.date(2020, 1, 1), dt.date(2020, 12, 31))
    cuts = [
        (dt.date(2020, 7, 1), dt.date(2020, 7, 31)),
        (dt.date(2020, 3, 1), dt.date(2020, 3, 31)),
    ]
    assert _subtract(whole, cuts) == _subtract(whole, list(reversed(cuts)))
    assert len(_subtract(whole, cuts)) == 3


def test_subtract_of_a_disjoint_cut_changes_nothing() -> None:
    whole = (dt.date(2020, 1, 1), dt.date(2020, 6, 30))
    assert _subtract(whole, [(dt.date(2021, 1, 1), dt.date(2021, 1, 2))]) == [whole]


# ---------------------------------------------------------------------------
# Expected against actual
# ---------------------------------------------------------------------------


def test_a_complete_city_reconciles(engine, city, manifest, landed) -> None:
    landed((START, END))
    cover(manifest, city, START, END)
    entry = only(report_for(engine, city, manifest))

    assert entry.expected == entry.actual == entry.distinct == DAYS
    assert entry.delta == 0
    assert entry.gaps == ()
    assert entry.reconciled


def test_a_city_with_no_rows_is_reported_missing(engine, city, manifest) -> None:
    """The mid-backfill case, and the one that first crashed this code."""
    report = report_for(engine, city, manifest)
    entry = only(report)

    assert report.missing_cities == (city.id,)
    assert (entry.actual, entry.distinct) == (0, 0)
    assert entry.delta == -DAYS
    assert entry.gap_count == 1
    assert entry.gaps[0].days == DAYS


def test_duplicates_do_not_count_as_coverage(engine, city, manifest, landed) -> None:
    landed((START, END), times=2)
    cover(manifest, city, START, END)
    entry = only(report_for(engine, city, manifest))

    assert entry.actual == DAYS * 2
    assert entry.distinct == DAYS
    assert entry.duplicates == DAYS
    assert entry.delta == 0


def test_rows_outside_the_range_are_a_surplus_not_coverage(
    engine, city, manifest, landed
) -> None:
    """Cairo and London really do carry extra hourly rows from an earlier run."""
    landed((START, END), (dt.date(2019, 1, 1), dt.date(2019, 12, 31)))
    cover(manifest, city, START, END)
    entry = only(report_for(engine, city, manifest))

    assert entry.distinct == DAYS
    assert entry.delta == 0
    assert entry.outside_range == 365
    assert entry.first == dt.date(2019, 1, 1)


# ---------------------------------------------------------------------------
# Gaps, and the category that explains each
# ---------------------------------------------------------------------------


def test_a_missing_middle_year_is_one_gap(engine, city, manifest, landed) -> None:
    landed(
        (START, dt.date(2020, 12, 31)),
        (dt.date(2022, 1, 1), END),
    )
    cover(manifest, city, START, dt.date(2020, 12, 31))
    cover(manifest, city, dt.date(2022, 1, 1), END)
    entry = only(report_for(engine, city, manifest))

    assert entry.gap_count == 1
    gap = entry.gaps[0]
    assert (gap.start, gap.end) == (dt.date(2021, 1, 1), dt.date(2021, 12, 31))
    assert gap.days == 365
    assert entry.delta == -365


def test_a_gap_the_manifest_never_covered_is_not_ingested(
    engine, city, manifest, landed
) -> None:
    landed((START, dt.date(2020, 12, 31)))
    cover(manifest, city, START, dt.date(2020, 12, 31))
    entry = only(report_for(engine, city, manifest))

    assert entry.gap_count == 1
    assert entry.gaps[0].category == NOT_INGESTED
    assert entry.unexplained == ()


def test_a_gap_inside_a_completed_unit_is_unexplained(
    engine, city, manifest, landed
) -> None:
    """Fetched, recorded complete, and missing anyway. Should never happen."""
    landed((START, dt.date(2020, 12, 31)), (dt.date(2022, 1, 1), END))
    cover(manifest, city, START, END)  # claims the whole range landed
    entry = only(report_for(engine, city, manifest))

    assert entry.gap_count == 1
    assert entry.gaps[0].category == UNEXPLAINED
    assert entry.unexplained == entry.gaps
    assert not entry.reconciled


def test_a_range_before_the_archive_is_an_archive_boundary(
    engine, city, manifest, landed
) -> None:
    landed((dt.date(1940, 1, 1), dt.date(1940, 12, 31)))
    report = reconcile(
        engine, grain="daily",
        start=dt.date(1935, 1, 1), end=dt.date(1940, 12, 31),
        cities=[city], manifest=manifest,
    )
    entry = only(report)
    assert entry.gap_count == 1
    gap = entry.gaps[0]
    assert gap.category == ARCHIVE_BOUNDARY
    assert gap.end == ARCHIVE_START - dt.timedelta(days=1)


def test_a_short_unit_is_an_api_limitation(engine, city, manifest, landed) -> None:
    """A completed unit whose recorded row count falls short of its window."""
    landed((START, dt.date(2020, 12, 31)), (dt.date(2022, 1, 1), END))
    cover(manifest, city, START, dt.date(2020, 12, 31))
    cover(manifest, city, dt.date(2021, 1, 1), dt.date(2021, 12, 31), rows=0)
    cover(manifest, city, dt.date(2022, 1, 1), END)
    entry = only(report_for(engine, city, manifest))

    assert entry.gap_count == 1
    assert entry.gaps[0].category == API_LIMITATION
    assert entry.unexplained == ()


def test_short_gaps_are_not_listed(engine, city, manifest, landed) -> None:
    """One missing day in a thirty-year series is noise."""
    landed(
        (START, dt.date(2020, 6, 1)),
        (dt.date(2020, 6, 4), END),   # two days missing
    )
    cover(manifest, city, START, END)
    assert only(report_for(engine, city, manifest)).gaps == ()
    wider = report_for(engine, city, manifest, min_gap_days=1)
    assert only(wider).gap_count == 1
    assert only(wider).gaps[0].days == 2


def test_every_gap_lands_in_exactly_one_category(
    engine, city, manifest, landed
) -> None:
    landed((dt.date(2021, 1, 1), dt.date(2021, 12, 31)))
    cover(manifest, city, dt.date(2021, 1, 1), dt.date(2021, 12, 31))
    report = report_for(engine, city, manifest)

    grouped = report.by_category()
    assert sum(len(v) for v in grouped.values()) == len(report.gaps)
    assert set(grouped) == set(CATEGORIES)
    assert all(g.category in CATEGORIES for g in report.gaps)


def test_gap_observations_scale_with_the_grain(engine, city, manifest) -> None:
    report = report_for(engine, city, manifest)
    gap = only(report).gaps[0]
    assert gap.observations == gap.days
    assert gap.grain == "daily"


# ---------------------------------------------------------------------------
# The artefact
# ---------------------------------------------------------------------------


def test_markdown_carries_the_columns_the_ticket_asks_for(
    engine, city, manifest, landed
) -> None:
    landed((START, dt.date(2020, 12, 31)))
    cover(manifest, city, START, dt.date(2020, 12, 31))
    rendered = to_markdown(report_for(engine, city, manifest))

    assert "| city | expected | actual | distinct | delta | gaps |" in rendered
    assert f"`{city.id}`" in rendered
    for category in CATEGORIES:
        assert category in rendered
    assert "| city | from | to | days |" in rendered
    assert "2021-01-01" in rendered


def test_markdown_states_the_verdict(engine, city, manifest, landed) -> None:
    landed((START, END))
    cover(manifest, city, START, END)
    clean = to_markdown(report_for(engine, city, manifest))
    assert "No gap is unexplained" in clean

    cover(manifest, city, dt.date(2019, 1, 1), dt.date(2019, 12, 31))
    wider = reconcile(
        engine, grain="daily", start=dt.date(2019, 1, 1), end=END,
        cities=[city], manifest=manifest,
    )
    assert "remain unexplained" in to_markdown(wider)


def test_accepted_notes_are_rendered(engine, city, manifest) -> None:
    rendered = to_markdown(
        report_for(engine, city, manifest),
        {NOT_INGESTED: "Accepted: the backfill is still running."},
    )
    assert "Accepted: the backfill is still running." in rendered


def test_the_report_knows_whether_it_reconciles(
    engine, city, manifest, landed
) -> None:
    landed((START, END))
    cover(manifest, city, START, END)
    assert report_for(engine, city, manifest).reconciled

    with_hole = reconcile(
        engine, grain="daily", start=dt.date(2019, 1, 1), end=END,
        cities=[city], manifest=Manifest(manifest.path),
    )
    assert not with_hole.reconciled
