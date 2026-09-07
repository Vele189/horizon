"""Tests for the backfill planner.

Entirely offline. The planner's whole job is arithmetic and bookkeeping — what
to request, in what order, and what has already landed — so there is nothing
here that needs the network. The one thing worth stating as a test rather than
a comment is Open-Meteo's billing model, because every default in the module
is derived from it.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cities import load_cities  # noqa: E402
from config import Settings, get_settings  # noqa: E402
from ingestion.client import ARCHIVE_START, DAILY_VARIABLES, HOURLY_VARIABLES  # noqa: E402
from ingestion.planner import (  # noqa: E402
    BACKFILL_START,
    DAILY_QUOTA_CALLS,
    HOURLY_BACKFILL_MONTHS,
    HOURLY_QUOTA_CALLS,
    MINUTELY_QUOTA_CALLS,
    Manifest,
    ManifestEntry,
    WorkUnit,
    _add_months,
    _merge,
    api_call_weight,
    archive_end_date,
    delay_seconds_for,
    plan_backfill,
    windows,
)

CITY = "london"
YEAR_START = dt.date(2020, 1, 1)
YEAR_END = dt.date(2024, 12, 31)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return dataclasses.replace(
        get_settings(),
        ingest_chunk_months=12,
        request_delay_seconds=1.0,
        ingest_manifest_path=tmp_path / "manifest.jsonl",
    )


@pytest.fixture
def manifest(tmp_path: Path) -> Manifest:
    return Manifest(tmp_path / "manifest.jsonl")


@pytest.fixture
def one_city():
    return [load_cities()[CITY]]


def plan(settings, manifest, one_city, **overrides):
    kwargs = dict(
        grains=("daily",),
        cities=one_city,
        start=YEAR_START,
        end=YEAR_END,
        manifest=manifest,
        settings=settings,
    )
    kwargs.update(overrides)
    return plan_backfill(**kwargs)


def unit(start: dt.date, end: dt.date, grain: str = "daily") -> WorkUnit:
    return WorkUnit(city_id=CITY, grain=grain, start=start, end=end)


# ---------------------------------------------------------------------------
# Open-Meteo's billing model, which every default below depends on
# ---------------------------------------------------------------------------


def test_the_documented_worked_examples() -> None:
    """From the pricing page: 2 weeks x 15 variables is 1.5 calls, 4 weeks 3.0."""
    assert api_call_weight(days=14, variables=15) == pytest.approx(1.5)
    assert api_call_weight(days=28, variables=15) == pytest.approx(3.0)


def test_a_small_request_still_costs_one_call() -> None:
    """Both factors are floored at 1, so there is no fractional bargain."""
    assert api_call_weight(days=1, variables=1) == 1.0
    assert api_call_weight(days=14, variables=10) == 1.0
    assert api_call_weight(days=7, variables=3) == 1.0


def test_weight_is_linear_in_days_above_the_floor() -> None:
    """Which is why chunk size is quota-neutral: ten units cost what one does."""
    whole = api_call_weight(days=3650, variables=21)
    split = sum(api_call_weight(days=365, variables=21) for _ in range(10))
    assert split == pytest.approx(whole)


def test_below_the_floor_splitting_costs_more() -> None:
    """And why the planner does not recommend windows under a fortnight."""
    whole = api_call_weight(days=140, variables=21)
    split = sum(api_call_weight(days=1, variables=21) for _ in range(140))
    assert split > whole


def test_a_city_year_of_daily_data_is_not_one_api_call() -> None:
    """The premise of the whole module: 55 calls, not 1."""
    weight = api_call_weight(days=365, variables=len(DAILY_VARIABLES))
    assert 54 < weight < 56


def test_the_observed_429_is_consistent_with_the_formula() -> None:
    """Measured 2026-09-07: 1, 2, 5 and 10-year requests, then a refusal.

    Four HTTP requests is nowhere near 600/min, but their weighted cost crosses
    600 on the fourth — which is where the fifth was refused.
    """
    weights = [
        api_call_weight(days=365 * years, variables=len(DAILY_VARIABLES))
        for years in (1, 2, 5, 10)
    ]
    assert sum(weights[:3]) < MINUTELY_QUOTA_CALLS
    assert sum(weights) > MINUTELY_QUOTA_CALLS


@pytest.mark.parametrize("bad", [(0, 10), (10, 0), (-1, 10)])
def test_weight_rejects_nonsense(bad) -> None:
    with pytest.raises(ValueError):
        api_call_weight(days=bad[0], variables=bad[1])


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


def test_twelve_month_windows_are_calendar_years() -> None:
    got = list(windows(dt.date(2020, 1, 1), dt.date(2022, 12, 31), 12))
    assert got == [
        (dt.date(2020, 1, 1), dt.date(2020, 12, 31)),
        (dt.date(2021, 1, 1), dt.date(2021, 12, 31)),
        (dt.date(2022, 1, 1), dt.date(2022, 12, 31)),
    ]


def test_windows_are_contiguous_and_never_overlap() -> None:
    got = list(windows(dt.date(2020, 3, 15), dt.date(2023, 8, 2), 7))
    assert got[0][0] == dt.date(2020, 3, 15)
    assert got[-1][1] == dt.date(2023, 8, 2)
    for (_, end), (start, _) in zip(got, got[1:]):
        assert start == end + dt.timedelta(days=1)


def test_the_last_window_is_clipped_to_the_end() -> None:
    got = list(windows(dt.date(2020, 1, 1), dt.date(2021, 6, 15), 12))
    assert got[-1] == (dt.date(2021, 1, 1), dt.date(2021, 6, 15))


def test_a_single_day_range_is_one_window() -> None:
    day = dt.date(2023, 5, 5)
    assert list(windows(day, day, 12)) == [(day, day)]


def test_leap_day_falls_inside_its_year(one_city) -> None:
    got = dict(windows(dt.date(2024, 1, 1), dt.date(2024, 12, 31), 12))
    assert got[dt.date(2024, 1, 1)] == dt.date(2024, 12, 31)
    assert unit(dt.date(2024, 1, 1), dt.date(2024, 12, 31)).expected_rows == 366


def test_windows_reject_a_reversed_or_zero_range() -> None:
    with pytest.raises(ValueError, match="precedes"):
        list(windows(dt.date(2021, 1, 1), dt.date(2020, 1, 1), 12))
    with pytest.raises(ValueError, match="at least 1"):
        list(windows(YEAR_START, YEAR_END, 0))


@pytest.mark.parametrize(
    ("start", "months", "expected"),
    [
        (dt.date(2020, 1, 31), 1, dt.date(2020, 2, 29)),   # clamped, leap year
        (dt.date(2021, 1, 31), 1, dt.date(2021, 2, 28)),   # clamped, common year
        (dt.date(2020, 12, 1), 1, dt.date(2021, 1, 1)),    # year rollover
        (dt.date(2026, 9, 2), -24, dt.date(2024, 9, 2)),   # backwards
        (dt.date(2020, 6, 15), 0, dt.date(2020, 6, 15)),
    ],
)
def test_month_arithmetic(start, months, expected) -> None:
    assert _add_months(start, months) == expected


# ---------------------------------------------------------------------------
# Work units
# ---------------------------------------------------------------------------


def test_unit_key_is_readable_and_unique() -> None:
    assert unit(YEAR_START, dt.date(2020, 12, 31)).key == (
        "london/daily/2020-01-01/2020-12-31"
    )


def test_unit_row_counts_match_the_grain() -> None:
    window = (dt.date(2023, 1, 1), dt.date(2023, 1, 10))
    assert unit(*window).expected_rows == 10
    assert unit(*window, grain="hourly").expected_rows == 240


def test_hourly_costs_less_than_daily_for_the_same_range() -> None:
    """Twelve variables against twenty-one; the row count is not what is billed."""
    window = (dt.date(2023, 1, 1), dt.date(2023, 12, 31))
    assert unit(*window, grain="hourly").weight < unit(*window).weight
    assert unit(*window, grain="hourly").expected_rows > unit(*window).expected_rows


def test_a_reversed_unit_is_rejected() -> None:
    with pytest.raises(ValueError, match="precedes"):
        unit(dt.date(2023, 5, 1), dt.date(2023, 4, 1))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_the_plan_is_deterministic(settings, manifest, one_city) -> None:
    first = plan(settings, manifest, one_city)
    second = plan(settings, manifest, one_city)
    assert [u.key for u in first.units] == [u.key for u in second.units]


def test_the_full_plan_is_deterministic(settings, manifest) -> None:
    end = dt.date(2026, 9, 2)
    kwargs = dict(end=end, manifest=manifest, settings=settings)
    assert [u.key for u in plan_backfill(**kwargs)] == [
        u.key for u in plan_backfill(**kwargs)
    ]


def test_order_is_city_major_then_chronological(settings, manifest) -> None:
    """A city completes as early as possible, rather than all at once at the end."""
    registry = load_cities()
    units = plan_backfill(
        grains=("daily",),
        cities=[registry["london"], registry["tokyo"]],
        start=YEAR_START,
        end=YEAR_END,
        manifest=manifest,
        settings=settings,
    ).units
    assert [u.city_id for u in units] == ["london"] * 5 + ["tokyo"] * 5
    london = [u for u in units if u.city_id == "london"]
    assert london == sorted(london, key=lambda u: u.start)


def test_city_order_follows_the_registry(settings, manifest) -> None:
    units = plan_backfill(
        grains=("daily",), start=YEAR_START, end=YEAR_END,
        manifest=manifest, settings=settings,
    ).units
    seen = list(dict.fromkeys(u.city_id for u in units))
    assert seen == list(load_cities().ids)


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


def test_an_absent_manifest_is_empty_not_an_error(tmp_path: Path) -> None:
    empty = Manifest(tmp_path / "nope" / "manifest.jsonl")
    assert len(empty) == 0
    assert not empty.is_complete(unit(YEAR_START, YEAR_END))


def test_recording_appends_one_line_per_unit(manifest) -> None:
    manifest.record(unit(dt.date(2020, 1, 1), dt.date(2020, 12, 31)), rows=366)
    manifest.record(unit(dt.date(2021, 1, 1), dt.date(2021, 12, 31)), rows=365)
    lines = manifest.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["rows"] for line in lines] == [366, 365]
    assert manifest.completed_rows() == 731


def test_a_record_survives_a_new_process(manifest) -> None:
    landed = unit(dt.date(2020, 1, 1), dt.date(2020, 12, 31))
    manifest.record(landed, rows=366)
    reopened = Manifest(manifest.path)
    assert len(reopened) == 1
    assert reopened.is_complete(landed)


def test_a_manifest_entry_round_trips(manifest) -> None:
    entry = manifest.record(unit(YEAR_START, dt.date(2020, 12, 31)), rows=366)
    assert ManifestEntry.from_json(entry.as_json()) == entry


def test_a_truncated_last_line_costs_one_unit_not_the_run(manifest) -> None:
    """What a killed run leaves behind. Refusing to start would be worse."""
    manifest.record(unit(dt.date(2020, 1, 1), dt.date(2020, 12, 31)), rows=366)
    with manifest.path.open("a", encoding="utf-8") as handle:
        handle.write('{"city_id":"london","grain":"daily","start":"2021-01')
    reopened = Manifest(manifest.path)
    assert len(reopened) == 1
    assert reopened.is_complete(unit(dt.date(2020, 1, 1), dt.date(2020, 12, 31)))
    assert not reopened.is_complete(unit(dt.date(2021, 1, 1), dt.date(2021, 12, 31)))


def test_a_record_missing_a_field_is_ignored(manifest) -> None:
    manifest.path.parent.mkdir(parents=True, exist_ok=True)
    manifest.path.write_text(
        json.dumps({"city_id": "london", "grain": "daily", "start": "2020-01-01"})
        + "\n",
        encoding="utf-8",
    )
    assert len(Manifest(manifest.path)) == 0


def test_blank_lines_are_skipped(manifest) -> None:
    manifest.record(unit(YEAR_START, dt.date(2020, 12, 31)), rows=366)
    with manifest.path.open("a", encoding="utf-8") as handle:
        handle.write("\n\n")
    assert len(Manifest(manifest.path)) == 1


def test_coverage_merges_adjacent_windows(manifest) -> None:
    manifest.record(unit(dt.date(2020, 1, 1), dt.date(2020, 12, 31)), rows=366)
    manifest.record(unit(dt.date(2021, 1, 1), dt.date(2021, 12, 31)), rows=365)
    assert manifest.coverage(CITY, "daily") == (
        (dt.date(2020, 1, 1), dt.date(2021, 12, 31)),
    )


def test_coverage_keeps_a_gap_apart(manifest) -> None:
    manifest.record(unit(dt.date(2020, 1, 1), dt.date(2020, 12, 31)), rows=366)
    manifest.record(unit(dt.date(2022, 1, 1), dt.date(2022, 12, 31)), rows=365)
    assert len(manifest.coverage(CITY, "daily")) == 2


def test_coverage_is_per_city_and_per_grain(manifest) -> None:
    manifest.record(unit(YEAR_START, dt.date(2020, 12, 31)), rows=366)
    landed = unit(YEAR_START, dt.date(2020, 12, 31))
    assert manifest.is_complete(landed)
    assert not manifest.is_complete(dataclasses.replace(landed, grain="hourly"))
    assert not manifest.is_complete(dataclasses.replace(landed, city_id="tokyo"))


def test_a_partly_covered_unit_is_not_complete(manifest) -> None:
    manifest.record(unit(dt.date(2020, 1, 1), dt.date(2020, 6, 30)), rows=182)
    assert not manifest.is_complete(unit(dt.date(2020, 1, 1), dt.date(2020, 12, 31)))


def test_merge_handles_overlap_and_containment() -> None:
    merged = _merge(
        [
            (dt.date(2020, 1, 1), dt.date(2020, 6, 30)),
            (dt.date(2020, 3, 1), dt.date(2020, 9, 30)),
            (dt.date(2020, 4, 1), dt.date(2020, 5, 1)),
        ]
    )
    assert merged == [(dt.date(2020, 1, 1), dt.date(2020, 9, 30))]


def test_compact_merges_records_without_losing_coverage(manifest) -> None:
    for year in (2020, 2021, 2022):
        manifest.record(
            unit(dt.date(year, 1, 1), dt.date(year, 12, 31)), rows=365
        )
    removed = manifest.compact()
    assert removed == 2
    reopened = Manifest(manifest.path)
    assert len(reopened) == 1
    assert reopened.completed_rows() == 1095
    assert reopened.coverage(CITY, "daily") == (
        (dt.date(2020, 1, 1), dt.date(2022, 12, 31)),
    )


def test_compact_leaves_a_gap_as_two_records(manifest) -> None:
    manifest.record(unit(dt.date(2020, 1, 1), dt.date(2020, 12, 31)), rows=366)
    manifest.record(unit(dt.date(2022, 1, 1), dt.date(2022, 12, 31)), rows=365)
    manifest.compact()
    assert len(Manifest(manifest.path)) == 2


# ---------------------------------------------------------------------------
# Re-running skips what has landed
# ---------------------------------------------------------------------------


def test_a_fresh_plan_has_nothing_skipped(settings, manifest, one_city) -> None:
    built = plan(settings, manifest, one_city)
    assert len(built.units) == 5
    assert built.skipped == ()


def test_completed_units_move_from_pending_to_skipped(
    settings, manifest, one_city
) -> None:
    first = plan(settings, manifest, one_city)
    for landed in first.units[:2]:
        manifest.record(landed, rows=landed.expected_rows)

    second = plan(settings, Manifest(manifest.path), one_city)
    assert len(second.units) == 3
    assert [u.key for u in second.skipped] == [u.key for u in first.units[:2]]
    assert second.total_units == first.total_units


def test_a_completed_plan_is_empty(settings, manifest, one_city) -> None:
    for landed in plan(settings, manifest, one_city).units:
        manifest.record(landed, rows=landed.expected_rows)
    assert plan(settings, Manifest(manifest.path), one_city).units == ()


def test_halving_the_chunk_still_skips_what_landed(
    settings, manifest, one_city
) -> None:
    """Shrinking the chunk after a 429 must not re-fetch two years of history."""
    for landed in plan(settings, manifest, one_city).units[:2]:
        manifest.record(landed, rows=landed.expected_rows)

    halved = plan(settings, Manifest(manifest.path), one_city, chunk_months=6)
    assert len(halved.skipped) == 4
    assert all(u.start >= dt.date(2022, 1, 1) for u in halved.units)


def test_growing_the_chunk_re_fetches_a_partly_covered_window(
    settings, manifest, one_city
) -> None:
    """The honest limit of coverage matching, stated rather than hidden."""
    for landed in plan(settings, manifest, one_city).units[:1]:
        manifest.record(landed, rows=landed.expected_rows)

    grown = plan(settings, Manifest(manifest.path), one_city, chunk_months=24)
    assert grown.skipped == ()
    assert len(grown.units) == 3


# ---------------------------------------------------------------------------
# Chunk size and pacing are configurable
# ---------------------------------------------------------------------------


def test_chunk_size_comes_from_configuration(settings, manifest, one_city) -> None:
    six = dataclasses.replace(settings, ingest_chunk_months=6)
    built = plan(six, manifest, one_city)
    assert built.chunk_months == 6
    assert len(built.units) == 10


def test_an_explicit_chunk_size_overrides_configuration(
    settings, manifest, one_city
) -> None:
    assert len(plan(settings, manifest, one_city, chunk_months=60).units) == 1


def test_the_default_chunk_is_a_calendar_year() -> None:
    assert get_settings().ingest_chunk_months == 12


def test_the_default_delay_is_not_zero() -> None:
    assert get_settings().request_delay_seconds > 0


def test_the_delay_paces_against_the_minutely_budget() -> None:
    """A one-second delay between city-years would spend 3 300 calls a minute."""
    year = api_call_weight(days=365, variables=len(DAILY_VARIABLES))
    assert delay_seconds_for(year, floor=1.0) > 1.0
    calls_per_minute = 60.0 / delay_seconds_for(year, floor=1.0) * year
    assert calls_per_minute <= MINUTELY_QUOTA_CALLS


def test_the_delay_paces_against_the_hourly_budget_too() -> None:
    """The limit the first real backfill run actually hit.

    Half of 600 calls a minute is 18 000 an hour against an allowance of 5 000.
    Pacing on the minutely bar alone cleared it on every request and still
    collected "Hourly API request limit exceeded" nineteen minutes in.
    """
    year = api_call_weight(days=365, variables=21)
    delay = delay_seconds_for(year, floor=1.0)
    calls_per_hour = 3600.0 / delay * year
    assert calls_per_hour <= HOURLY_QUOTA_CALLS
    minutely_only = 60.0 * year / (MINUTELY_QUOTA_CALLS * 0.5)
    assert delay > minutely_only, "the hourly limit is the binding one"


def test_the_configured_delay_is_a_floor_not_a_ceiling() -> None:
    assert delay_seconds_for(1.0, floor=3.0) == 3.0
    assert delay_seconds_for(600.0, floor=1.0) > 3.0


def test_a_zero_floor_still_paces_a_heavy_unit() -> None:
    assert delay_seconds_for(api_call_weight(365, 21), floor=0.0) > 0


def test_the_plan_reports_a_delay_per_unit(settings, manifest, one_city) -> None:
    built = plan(settings, manifest, one_city)
    assert all(built.delay_for(u) >= settings.request_delay_seconds for u in built)
    assert built.estimated_seconds == pytest.approx(
        sum(built.delay_for(u) for u in built.units)
    )


# ---------------------------------------------------------------------------
# Cost and quota
# ---------------------------------------------------------------------------


def test_the_plan_totals_its_units(settings, manifest, one_city) -> None:
    built = plan(settings, manifest, one_city)
    assert built.weight == pytest.approx(sum(u.weight for u in built.units))
    assert built.expected_rows == sum(u.expected_rows for u in built.units)
    assert built.expected_rows == 1827  # five years, one leap


def test_the_daily_quota_prefix_fits_inside_the_allowance(
    settings, manifest
) -> None:
    built = plan_backfill(
        end=dt.date(2026, 9, 2), manifest=manifest, settings=settings
    )
    within = built.within_daily_quota()
    assert 0 < len(within) < len(built.units)
    assert sum(u.weight for u in within) <= DAILY_QUOTA_CALLS


def test_the_full_backfill_does_not_fit_in_one_day_of_free_quota(
    settings, manifest
) -> None:
    """Recorded as a test because it is the reason the manifest exists."""
    built = plan_backfill(
        end=dt.date(2026, 9, 2), manifest=manifest, settings=settings
    )
    assert built.quota_days > 1


def test_planned_row_counts_match_the_proposal(settings, manifest) -> None:
    """§5.1 sizes bronze at ~164k daily and ~263k hourly rows."""
    built = plan_backfill(
        end=dt.date(2026, 9, 2), manifest=manifest, settings=settings
    )
    daily = sum(u.expected_rows for u in built.units if u.grain == "daily")
    hourly = sum(u.expected_rows for u in built.units if u.grain == "hourly")
    assert 160_000 < daily < 180_000
    assert 255_000 < hourly < 275_000


# ---------------------------------------------------------------------------
# Ranges and validation
# ---------------------------------------------------------------------------


def test_daily_reaches_back_thirty_years_and_hourly_does_not(
    settings, manifest
) -> None:
    end = dt.date(2026, 9, 2)
    built = plan_backfill(
        cities=[load_cities()[CITY]], end=end, manifest=manifest, settings=settings
    )
    daily = [u for u in built.units if u.grain == "daily"]
    hourly = [u for u in built.units if u.grain == "hourly"]
    assert daily[0].start == BACKFILL_START
    assert hourly[0].start == _add_months(end, -HOURLY_BACKFILL_MONTHS)
    assert daily[-1].end == hourly[-1].end == end


def test_the_hourly_window_is_anchored_not_relative_to_today(
    settings, manifest
) -> None:
    """A plan must name the same window tomorrow as it does today.

    Left to the default the anchor is the archive edge, which moves. Passing it
    explicitly is what makes "the trailing 24 months" a reproducible statement
    rather than a description of when the command happened to run.
    """
    anchor = dt.date(2026, 9, 2)
    built = plan_backfill(
        grains=("hourly",), cities=[load_cities()[CITY]], end=anchor,
        manifest=manifest, settings=settings,
    )
    assert built.units[0].start == dt.date(2024, 9, 2)
    assert built.units[-1].end == anchor
    # Same anchor, same plan — whatever the clock says.
    again = plan_backfill(
        grains=("hourly",), cities=[load_cities()[CITY]], end=anchor,
        manifest=manifest, settings=settings,
    )
    assert [u.key for u in built.units] == [u.key for u in again.units]


def test_the_hourly_reach_is_configurable(settings, manifest) -> None:
    anchor = dt.date(2026, 9, 2)
    for months, expected in ((24, dt.date(2024, 9, 2)), (6, dt.date(2026, 3, 2))):
        built = plan_backfill(
            grains=("hourly",), cities=[load_cities()[CITY]], end=anchor,
            hourly_months=months, manifest=manifest, settings=settings,
        )
        assert built.units[0].start == expected


def test_the_hourly_reach_comes_from_configuration(settings, manifest) -> None:
    twelve = dataclasses.replace(settings, ingest_hourly_months=12)
    built = plan_backfill(
        grains=("hourly",), cities=[load_cities()[CITY]],
        end=dt.date(2026, 9, 2), manifest=manifest, settings=twelve,
    )
    assert built.units[0].start == dt.date(2025, 9, 2)


def test_the_default_hourly_reach_is_two_years() -> None:
    assert get_settings().ingest_hourly_months == HOURLY_BACKFILL_MONTHS == 24


def test_a_zero_month_hourly_reach_is_rejected(settings, manifest) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        plan_backfill(
            grains=("hourly",), hourly_months=0, manifest=manifest,
            settings=settings,
        )


def test_hourly_is_two_years_not_thirty(settings, manifest) -> None:
    """Thirty years of hourly is ~4 million rows for no analytical benefit."""
    anchor = dt.date(2026, 9, 2)
    hourly = plan_backfill(
        grains=("hourly",), end=anchor, manifest=manifest, settings=settings
    )
    daily = plan_backfill(
        grains=("daily",), end=anchor, manifest=manifest, settings=settings
    )
    assert 255_000 < hourly.expected_rows < 270_000
    thirty_years_hourly = daily.expected_rows * 24
    assert thirty_years_hourly > 4_000_000
    assert hourly.expected_rows < thirty_years_hourly / 15


def test_hourly_never_starts_before_the_daily_range(settings, manifest) -> None:
    built = plan_backfill(
        cities=[load_cities()[CITY]],
        start=dt.date(2026, 1, 1),
        end=dt.date(2026, 9, 2),
        manifest=manifest,
        settings=settings,
    )
    assert min(u.start for u in built.units) == dt.date(2026, 1, 1)


def test_the_end_date_stops_short_of_the_archive_edge() -> None:
    today = dt.date(2026, 9, 7)
    assert archive_end_date(today) < today


def test_a_start_before_the_archive_is_rejected(settings, manifest) -> None:
    with pytest.raises(ValueError, match=str(ARCHIVE_START)):
        plan_backfill(
            start=dt.date(1939, 1, 1), manifest=manifest, settings=settings
        )


def test_an_unknown_grain_is_rejected(settings, manifest) -> None:
    with pytest.raises(ValueError, match="monthly"):
        plan_backfill(grains=("monthly",), manifest=manifest, settings=settings)


def test_every_planned_unit_is_a_legal_client_request(settings, manifest) -> None:
    """The planner must not emit a window the client would refuse."""
    built = plan_backfill(
        end=dt.date(2026, 9, 2), manifest=manifest, settings=settings
    )
    today = dt.datetime.now(dt.timezone.utc).date()
    for u in built.units:
        assert u.start >= ARCHIVE_START
        assert u.start <= u.end <= today
        assert u.grain in ("daily", "hourly")


def test_variable_counts_match_the_client(settings, manifest) -> None:
    """Weight is charged on variables, so a drift here mis-prices every plan."""
    year = (dt.date(2023, 1, 1), dt.date(2023, 12, 31))
    assert unit(*year).weight == pytest.approx(
        api_call_weight(365, len(DAILY_VARIABLES))
    )
    assert unit(*year, grain="hourly").weight == pytest.approx(
        api_call_weight(365, len(HOURLY_VARIABLES))
    )
