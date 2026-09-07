"""Tests for the date spine and the season resolution.

Two failures this guards against, both quiet:

*   **A global month lookup.** `case month when 12 then 'winter'` passes review
    and is wrong for a third of this city set. It does not merely mislabel the
    five southern cities — it puts their hottest month in the same cohort as
    Moscow's coldest, so a seasonal aggregate averages summer and winter
    together and reports something near the annual mean with a season's name
    on it.
*   **Grouping a climatology by `day_of_year`.** 1 March is day 60 in a common
    year and 61 in a leap year, so the grouping mixes 1 March with 29 February
    and shifts every day after February by one in three years out of four — a
    systematic error that reads as a seasonal signal.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("sqlalchemy")

from sqlalchemy import text  # noqa: E402

from cities import load_cities  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DBT_DIR = REPO_ROOT / "dbt_analytics"
GOLD = "gold_marts"
FOUR = ("winter", "spring", "summer", "autumn")


def query(engine, sql: str, **params):
    with engine.connect() as connection:
        return connection.execute(text(sql), params).fetchall()


@pytest.fixture
def spine(engine):
    rows = query(engine, f"select count(*), min(date_day), max(date_day) from {GOLD}.dim_date")
    if not rows or rows[0][0] == 0:
        pytest.skip("dim_date not built")
    return rows[0]


# ---------------------------------------------------------------------------
# The spine
# ---------------------------------------------------------------------------


def test_the_spine_has_no_gaps(engine, spine) -> None:
    count, first, last = spine
    assert count == (last - first).days + 1, "a missing day would hide observations"


def test_the_spine_covers_every_observation(engine, spine) -> None:
    """A left join against a short spine drops rows without a word."""
    for grain in ("daily", "hourly"):
        uncovered = query(
            engine,
            f"""select count(*) from (
                    select distinct (observation_time at time zone 'UTC')::date as d
                    from silver_staging.stg_observations_{grain}
                ) observed
                where d not in (select date_day from {GOLD}.dim_date)""",
        )[0][0]
        assert uncovered == 0, grain


def test_the_spine_starts_at_the_backfill_start(spine) -> None:
    from ingestion.planner import BACKFILL_START

    assert spine[1] == BACKFILL_START


def test_the_spine_reaches_past_today(spine) -> None:
    """Headroom, so a spine rebuilt less often than the data still covers it."""
    assert spine[2] > dt.date.today()


def test_each_day_appears_once(engine) -> None:
    duplicates = query(
        engine,
        f"select count(*) from (select date_day from {GOLD}.dim_date "
        "group by date_day having count(*) > 1) d",
    )[0][0]
    assert duplicates == 0


# ---------------------------------------------------------------------------
# Leap years
# ---------------------------------------------------------------------------


def test_29_february_exists_in_leap_years_only(engine) -> None:
    years = {
        r[0]
        for r in query(
            engine,
            f"select distinct year from {GOLD}.dim_date where month_day = '02-29'",
        )
    }
    assert years, "no leap days in the spine"
    for year in years:
        assert year % 4 == 0 and (year % 100 != 0 or year % 400 == 0), year
    assert 2024 in years
    assert 2025 not in years


def test_29_february_is_flagged(engine) -> None:
    flagged = query(
        engine,
        f"select is_leap_day, is_leap_year, day_of_year, day_of_year_common "
        f"from {GOLD}.dim_date where date_day = date '2024-02-29'",
    )
    if not flagged:
        pytest.skip("2024 not in the spine")
    row = flagged[0]
    assert row.is_leap_day is True
    assert row.is_leap_year is True
    assert row.day_of_year == 60
    assert row.day_of_year_common == 59, "shares 28 February so nothing after shifts"


def test_raw_day_of_year_is_not_leap_stable(engine) -> None:
    """The trap, asserted as a fact rather than described in a comment."""
    values = {
        r[0]
        for r in query(
            engine, f"select distinct day_of_year from {GOLD}.dim_date where month_day = '03-01'"
        )
    }
    assert values == {60, 61}, "1 March takes two day_of_year values"


def test_the_common_day_of_year_is_leap_stable(engine) -> None:
    drifting = query(
        engine,
        f"select month_day from {GOLD}.dim_date group by month_day "
        "having count(distinct day_of_year_common) > 1",
    )
    assert not drifting, f"{len(drifting)} month-days drift across years"


def test_month_day_is_the_stable_climatology_key(engine) -> None:
    counts = query(
        engine,
        f"select count(distinct month_day) from {GOLD}.dim_date",
    )[0][0]
    assert counts == 366, "365 days plus 29 February"


# ---------------------------------------------------------------------------
# Seasons, per hemisphere and per regime
# ---------------------------------------------------------------------------


def test_dim_date_carries_both_answers_and_neither_as_the_season(engine) -> None:
    """A single `season` column on a date dimension is the bug, not the feature."""
    columns = {
        r[0]
        for r in query(
            engine,
            "select column_name from information_schema.columns "
            f"where table_schema = '{GOLD}' and table_name = 'dim_date'",
        )
    }
    if not columns:
        pytest.skip("dim_date not built")
    assert {"season_northern", "season_southern"} <= columns
    assert "season" not in columns


@pytest.mark.parametrize(
    ("month", "northern", "southern"),
    [(12, "winter", "summer"), (1, "winter", "summer"), (7, "summer", "winter"),
     (4, "spring", "autumn"), (10, "autumn", "spring")],
)
def test_the_two_hemispheres_are_six_months_apart(
    engine, month: int, northern: str, southern: str
) -> None:
    rows = query(
        engine,
        f"select distinct season_northern, season_southern from {GOLD}.dim_date "
        "where month = :m",
        m=month,
    )
    if not rows:
        pytest.skip("dim_date not built")
    assert len(rows) == 1
    assert (rows[0].season_northern, rows[0].season_southern) == (northern, southern)


def test_december_is_summer_in_the_south_and_winter_in_the_north(engine) -> None:
    """The checklist's assertion, against the real city set."""
    rows = query(
        engine,
        f"""select s.city_id, c.hemisphere, s.season
            from {GOLD}.dim_city_season s
            join {GOLD}.dim_cities c using (city_id)
            where s.month = 12 and c.season_model = 'four_season'""",
    )
    if not rows:
        pytest.skip("dim_city_season not built")
    by_hemisphere = {"north": set(), "south": set()}
    for row in rows:
        by_hemisphere[row.hemisphere].add(row.season)
    assert by_hemisphere["north"] == {"winter"}
    assert by_hemisphere["south"] == {"summer"}


def test_every_city_has_a_season_for_every_month(engine) -> None:
    count = query(engine, f"select count(*) from {GOLD}.dim_city_season")[0][0]
    if not count:
        pytest.skip("dim_city_season not built")
    assert count == len(load_cities()) * 12 == 180


def test_tropical_cities_are_not_given_a_thermal_season(engine) -> None:
    """Lagos with a winter would join a northern-winter cohort in every aggregate."""
    wrong = query(
        engine,
        f"""select city_id, month, season from {GOLD}.dim_city_season
            where season_model <> 'four_season' and season in
            ('winter', 'spring', 'summer', 'autumn')""",
    )
    assert not wrong, wrong


def test_lagos_gets_wet_and_dry(engine) -> None:
    seasons = {
        r[0]
        for r in query(
            engine, f"select distinct season from {GOLD}.dim_city_season where city_id = 'lagos'"
        )
    }
    if not seasons:
        pytest.skip("dim_city_season not built")
    assert seasons == {"wet", "dry"}


def test_singapore_is_told_it_has_no_season(engine) -> None:
    """Af, within 1.4 degrees of the equator: no thermal cycle, no dry season."""
    seasons = {
        r[0]
        for r in query(
            engine,
            f"select distinct season from {GOLD}.dim_city_season where city_id = 'singapore'",
        )
    }
    if not seasons:
        pytest.skip("dim_city_season not built")
    assert seasons == {"year_round"}


def test_the_season_regimes_match_the_registry(engine) -> None:
    rows = query(
        engine, f"select distinct city_id, season_model from {GOLD}.dim_city_season"
    )
    if not rows:
        pytest.skip("dim_city_season not built")
    registry = {c.id: c.season_model for c in load_cities()}
    assert {r.city_id: r.season_model for r in rows} == registry


# ---------------------------------------------------------------------------
# The macro is the only place the logic lives
# ---------------------------------------------------------------------------


def test_the_regime_aware_entry_point_is_what_models_call() -> None:
    """Reaching for the four-season macro directly is how Lagos gets a winter."""
    model = (DBT_DIR / "models" / "marts" / "dim_city_season.sql").read_text(
        encoding="utf-8"
    )
    assert "season_for(" in model
    assert "meteorological_season(" not in model


def test_the_season_macro_covers_every_configured_regime() -> None:
    macro = (DBT_DIR / "macros" / "seasons.sql").read_text(encoding="utf-8")
    for model in {c.season_model for c in load_cities()}:
        assert f"'{model}'" in macro, model
