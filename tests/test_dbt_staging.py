"""Tests for the silver staging models.

The dbt tests already assert the data; these assert the things dbt cannot,
plus one integration run so a broken model fails `pytest` and not only
`dbt build`.

Worth stating plainly, because it is the reason two of the dbt tests exist:
uniqueness proves *one* row survives per key and says nothing about *which*.
Reversing the sort order to `ingested_at asc` leaves every uniqueness test
green and silently serves the oldest copy of every observation. Verified by
mutation — with the order flipped, `unique_combination_of_columns`,
`unique_id` and `assert_staging_loses_no_observation` all pass, and only
`assert_daily_keeps_the_newest_ingest` fails, on 731 rows.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dbt_analytics.dbt_env import dbt_environment  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DBT_DIR = REPO_ROOT / "dbt_analytics"
MACRO = DBT_DIR / "macros" / "deduplicate_observations.sql"
STAGING = DBT_DIR / "models" / "staging"
GRAINS = ("daily", "hourly")



def arguments_of(test: dict, name: str) -> dict:
    """Arguments of a generic test, as dbt 1.12 nests them.

    dbt deprecated top-level arguments in favour of an `arguments` block, so
    the shape these tests read changed under them. Reading through one helper
    means the next such change is one edit rather than four.
    """
    return test[name].get("arguments", test[name])

@pytest.fixture(scope="module")
def macro() -> str:
    return MACRO.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def staging_yml() -> dict:
    return yaml.safe_load((STAGING / "_staging.yml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The deduplication itself
# ---------------------------------------------------------------------------


def test_it_does_not_use_qualify(macro: str) -> None:
    """QUALIFY is Snowflake, BigQuery and DuckDB. Postgres has no such clause."""
    assert "qualify" not in macro.lower().replace("qualify clause", "")


def test_it_uses_a_row_number_subquery_filtered_to_rank_one(macro: str) -> None:
    assert "row_number() over" in macro
    assert re.search(r"where\s+_dedup_rank\s*=\s*1", macro)


def test_it_partitions_by_the_natural_key(macro: str) -> None:
    assert re.search(
        r"partition by\s+city_id,\s*observation_time", macro
    ), "partitioning by anything else would collapse rows that are not duplicates"


def test_it_orders_newest_ingest_first(macro: str) -> None:
    assert re.search(r"order by\s+ingested_at desc", macro)


def test_it_breaks_ties_deterministically(macro: str) -> None:
    """The loader stamps one ingested_at per run, so ties are reachable."""
    assert re.search(r"order by\s+ingested_at desc,\s*id desc", macro)


def test_the_rank_column_does_not_leak(macro: str) -> None:
    """A stray _dedup_rank in silver would propagate into every mart."""
    final_select = macro.rsplit("from ranked", 1)[0].rsplit("select", 1)[1]
    assert "_dedup_rank" not in final_select


def test_columns_are_taken_from_the_relation_not_a_list(macro: str) -> None:
    """Bronze has already lost a column once; a hand-maintained list would rot."""
    assert "adapter.get_columns_in_relation" in macro


def test_introspection_is_guarded_by_execute(macro: str) -> None:
    """Unguarded, the column list is silently empty during dbt's parse pass."""
    assert "{%- if execute -%}" in macro or "{% if execute %}" in macro


# ---------------------------------------------------------------------------
# Both grains, and their tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grain", GRAINS)
def test_a_model_exists_for_each_grain(grain: str) -> None:
    model = STAGING / f"stg_observations_{grain}.sql"
    assert model.is_file()
    body = model.read_text(encoding="utf-8")
    assert "deduplicate_observations" in body
    assert f"observations_{grain}" in body


@pytest.mark.parametrize("grain", GRAINS)
def test_each_model_is_uniqueness_tested_on_the_natural_key(
    staging_yml, grain: str
) -> None:
    model = next(
        m for m in staging_yml["models"] if m["name"] == f"stg_observations_{grain}"
    )
    combos = [
        arguments_of(t, "unique_combination_of_columns")["combination_of_columns"]
        for t in model["tests"]
        if "unique_combination_of_columns" in t
    ]
    assert ["city_id", "observation_time"] in combos


@pytest.mark.parametrize("grain", GRAINS)
def test_each_model_records_its_duplicate_count(staging_yml, grain: str) -> None:
    """The ticket asks for before and after in the description, not in a commit."""
    model = next(
        m for m in staging_yml["models"] if m["name"] == f"stg_observations_{grain}"
    )
    description = model["description"]
    assert "Duplicates removed" in description
    numbers = re.findall(r"[\d,]{4,}", description)
    assert len(numbers) >= 2, "before and after should both be stated"


@pytest.mark.parametrize("grain", GRAINS)
def test_each_grain_asserts_the_newest_copy_survived(grain: str) -> None:
    test_file = DBT_DIR / "tests" / f"assert_{grain}_keeps_the_newest_ingest.sql"
    assert test_file.is_file()
    body = test_file.read_text(encoding="utf-8")
    assert "max(ingested_at)" in body
    assert f"stg_observations_{grain}" in body


def test_a_narrowed_partition_would_be_caught() -> None:
    """Partitioning by city_id alone yields a unique result that is also wrong."""
    body = (DBT_DIR / "tests" / "assert_staging_loses_no_observation.sql").read_text(
        encoding="utf-8"
    )
    assert "distinct city_id, observation_time" in body
    for grain in GRAINS:
        assert grain in body


# ---------------------------------------------------------------------------
# It actually builds
# ---------------------------------------------------------------------------


def test_the_staging_layer_builds_and_all_its_tests_pass(engine) -> None:
    """The integration check: a broken model fails pytest, not only dbt build."""
    pytest.importorskip("dbt.cli.main")
    if not (DBT_DIR / "profiles.yml").exists():
        pytest.skip("profiles.yml not generated; run dbt_env.py --write-profile")

    result = subprocess.run(
        [
            sys.executable, "-m", "dbt.cli.main", "build",
            "--project-dir", str(DBT_DIR),
            "--select", "path:models/staging",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, **dbt_environment()},
        check=False,
    )
    summary = re.search(r"Done\. PASS=(\d+) WARN=(\d+) ERROR=(\d+)", result.stdout)
    assert summary, result.stdout[-3000:]
    passed, warned, errored = (int(g) for g in summary.groups())
    assert errored == 0, result.stdout[-3000:]
    assert warned == 0, result.stdout[-3000:]
    assert passed >= 12, f"expected both models and their tests, got {passed}"


# ---------------------------------------------------------------------------
# Unit assertions (DBT-03)
# ---------------------------------------------------------------------------

UNITS = DBT_DIR / "macros" / "units.sql"
RANGE_TEST = DBT_DIR / "macros" / "accepted_range.sql"


@pytest.fixture(scope="module")
def units() -> str:
    return UNITS.read_text(encoding="utf-8")


def ranges_for(staging_yml, model_name: str) -> dict[str, dict]:
    """column -> the accepted_range arguments declared for it."""
    model = next(m for m in staging_yml["models"] if m["name"] == model_name)
    found: dict[str, dict] = {}
    for column in model.get("columns", []):
        for test in column.get("tests", []) or []:
            if isinstance(test, dict) and "accepted_range" in test:
                found[column["name"]] = arguments_of(test, "accepted_range")
    return found


def test_conversions_live_in_macros_not_inline_sql(units: str) -> None:
    """A factor typed into six schema entries is six chances to invert it."""
    assert "{% macro cm_to_mm" in units
    assert "{% macro kmh_to_ms" in units
    assert "* 10.0" in units
    assert "/ 3.6" in units


def test_the_only_conversion_applied_is_the_one_the_source_forces() -> None:
    """Silver asserts; it does not convert what ING-01 already verified."""
    model = (STAGING / "stg_observations_daily.sql").read_text(encoding="utf-8")
    assert "cm_to_mm('snowfall_sum')" in model
    assert "snowfall_sum_mm" in model
    # No other unit macro is applied to stored values.
    assert "kmh_to_ms" not in model


@pytest.mark.parametrize(
    ("column", "low", "high"),
    [
        ("temperature_2m_max", -90, 60),
        ("temperature_2m_min", -90, 60),
        ("temperature_2m_mean", -90, 60),
        ("dew_point_2m_mean", -90, 60),
        ("pressure_msl_mean", 850, 1085),
        ("relative_humidity_2m_mean", 0, 100),
        ("cloud_cover_mean", 0, 100),
        ("precipitation_hours", 0, 24),
        ("wind_direction_10m_dominant", 0, 360),
    ],
)
def test_daily_ranges_match_the_specification(
    staging_yml, column: str, low: int, high: int
) -> None:
    declared = ranges_for(staging_yml, "stg_observations_daily")[column]
    assert (declared["min_value"], declared["max_value"]) == (low, high)


@pytest.mark.parametrize(
    "column", ["precipitation_sum", "rain_sum", "snowfall_sum", "snowfall_sum_mm",
               "shortwave_radiation_sum"]
)
def test_accumulations_are_non_negative(staging_yml, column: str) -> None:
    assert ranges_for(staging_yml, "stg_observations_daily")[column]["min_value"] == 0


def test_surface_pressure_is_not_held_to_the_sea_level_floor(staging_yml) -> None:
    """Johannesburg sits at 1753 m and reads 822 hPa; its MSL pressure is 998.

    The 850 hPa floor is a *mean sea level* bound. Applying it to
    surface_pressure fails on altitude, on correct data.
    """
    daily = ranges_for(staging_yml, "stg_observations_daily")
    assert daily["pressure_msl_mean"]["min_value"] == 850
    assert daily["surface_pressure_mean"]["min_value"] < 850
    assert daily["surface_pressure_mean"]["max_value"] == 1085

    hourly = ranges_for(staging_yml, "stg_observations_hourly")
    assert hourly["surface_pressure"]["min_value"] < hourly["pressure_msl"]["min_value"]


@pytest.mark.parametrize(
    ("model", "column"),
    [
        ("stg_observations_daily", "wind_speed_10m_max"),
        ("stg_observations_daily", "wind_speed_10m_mean"),
        ("stg_observations_daily", "wind_gusts_10m_max"),
        ("stg_observations_hourly", "wind_speed_10m"),
        ("stg_observations_hourly", "wind_gusts_10m"),
    ],
)
def test_wind_is_bounded_in_metres_per_second(staging_yml, model, column) -> None:
    """Stored km/h, bounded 0-120 m/s through the macro rather than restated."""
    declared = ranges_for(staging_yml, model)[column]
    assert (declared["min_value"], declared["max_value"]) == (0, 120)
    # The macro call, not rendered arithmetic — which is the checklist's
    # requirement that conversions are macros rather than inline SQL.
    assert declared["expression"] == "{{ kmh_to_ms('" + column + "') }}"
    assert "/ 3.6" not in declared["expression"]


def test_a_naive_kmh_bound_would_pass_today_and_break_later(engine) -> None:
    """Why the conversion is there, stated as a fact about the data.

    The largest gust on record here is 119.9 km/h — 0.1 under a bound of 120
    read as km/h. That bound looks correct until an ordinary winter storm, and
    then fails on data that is fine. The conversion protects against a false
    alarm, not a missed one, which is the opposite of the usual reason.
    """
    from sqlalchemy import text

    with engine.connect() as connection:
        peak = connection.execute(
            text("select max(wind_gusts_10m) from silver_staging.stg_observations_hourly")
        ).scalar()
    if peak is None:
        pytest.skip("no hourly wind data landed yet")
    assert peak < 120, "a naive km/h bound has not yet been exceeded"
    assert peak / 3.6 < 120, "and the m/s bound has far more headroom"


def test_the_range_test_ignores_nulls(engine) -> None:
    """Bronze preserves nulls; a test that failed on them would forbid that."""
    body = RANGE_TEST.read_text(encoding="utf-8")
    assert "is not null" in body


def test_the_range_test_refuses_to_be_vacuous() -> None:
    body = RANGE_TEST.read_text(encoding="utf-8")
    assert "raise_compiler_error" in body
    assert "min_value is none and max_value is none" in body


@pytest.mark.parametrize(
    "name",
    [
        "assert_daily_temperature_extremes_are_ordered",
        "assert_rain_does_not_exceed_precipitation",
        "assert_dew_point_does_not_exceed_temperature",
    ],
)
def test_cross_field_physics_is_asserted(name: str) -> None:
    """Bounds pass on swapped columns; these are what catch that."""
    assert (DBT_DIR / "tests" / f"{name}.sql").is_file()


def test_tolerances_are_compared_in_numeric_not_real() -> None:
    """In float4, 23.1 - 23.0 is 0.10000038, and a 0.1 tolerance fails on it.

    That is not hypothetical: this test's absence cost 14 false failures on
    correct Singapore data.
    """
    for name in (
        "assert_dew_point_does_not_exceed_temperature",
        "assert_rain_does_not_exceed_precipitation",
    ):
        body = (DBT_DIR / "tests" / f"{name}.sql").read_text(encoding="utf-8")
        assert "::numeric" in body, name
