"""Tests for the daily fact table.

The fact is selected from silver rather than joined to the dimensions. An
inner join would enforce referential integrity by *dropping* rows it could not
match, and a fact silently missing a city looks exactly like a city with no
weather. The relationships tests assert the same property and fail loudly —
verified by planting an orphan city, which trips both the foreign key test and
the row-count test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("sqlalchemy")

from sqlalchemy import text  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DBT_DIR = REPO_ROOT / "dbt_analytics"
MODEL = DBT_DIR / "models" / "marts" / "fact_weather_observations.sql"
GOLD = "gold_marts"
FACT = f"{GOLD}.fact_weather_observations"

#: The measures the ticket requires, by the family it names them in.
REQUIRED_MEASURES = {
    "temperature min": "temperature_2m_min",
    "temperature max": "temperature_2m_max",
    "temperature mean": "temperature_2m_mean",
    "precipitation": "precipitation_sum",
    "wind speed": "wind_speed_10m_max",
    "wind gust": "wind_gusts_10m_max",
    "pressure": "pressure_msl_mean",
    "humidity": "relative_humidity_2m_mean",
}


def query(engine, sql: str, **params):
    with engine.connect() as connection:
        return connection.execute(text(sql), params).fetchall()


@pytest.fixture(scope="module")
def fact_doc() -> dict:
    marts = yaml.safe_load(
        (DBT_DIR / "models" / "marts" / "_marts.yml").read_text(encoding="utf-8")
    )
    return next(m for m in marts["models"] if m["name"] == "fact_weather_observations")


@pytest.fixture
def columns(engine) -> set[str]:
    found = {
        r[0]
        for r in query(
            engine,
            "select column_name from information_schema.columns "
            f"where table_schema = '{GOLD}' and table_name = 'fact_weather_observations'",
        )
    }
    if not found:
        pytest.skip("fact_weather_observations not built")
    return found


# ---------------------------------------------------------------------------
# Grain and keys
# ---------------------------------------------------------------------------


def test_the_grain_is_one_row_per_city_day(engine, columns) -> None:
    duplicates = query(
        engine,
        f"select count(*) from (select city_id, date_key from {FACT} "
        "group by city_id, date_key having count(*) > 1) d",
    )[0][0]
    assert duplicates == 0


def test_date_key_is_the_utc_date_of_the_observation(engine, columns) -> None:
    """A timezone slipping in between silver and here would put two
    observations on one date_key for the cities furthest from Greenwich."""
    wrong = query(
        engine,
        f"select count(*) from {FACT} "
        "where date_key <> (observation_time at time zone 'UTC')::date",
    )[0][0]
    assert wrong == 0


def test_both_keys_are_not_null(engine, columns) -> None:
    nulls = query(
        engine,
        f"select count(*) from {FACT} where city_id is null or date_key is null",
    )[0][0]
    assert nulls == 0


def test_every_city_key_resolves_to_the_dimension(engine, columns) -> None:
    orphans = query(
        engine,
        f"select distinct city_id from {FACT} "
        f"where city_id not in (select city_id from {GOLD}.dim_cities)",
    )
    assert not orphans, orphans


def test_every_date_key_resolves_to_the_spine(engine, columns) -> None:
    orphans = query(
        engine,
        f"select count(distinct date_key) from {FACT} "
        f"where date_key not in (select date_day from {GOLD}.dim_date)",
    )[0][0]
    assert orphans == 0


def test_the_yaml_declares_both_relationships(fact_doc) -> None:
    """The checklist asks for relationships tests, not merely for valid data."""
    declared = {}
    for column in fact_doc["columns"]:
        for test in column.get("tests", []) or []:
            if isinstance(test, dict) and "relationships" in test:
                declared[column["name"]] = test["relationships"]
    assert declared["city_id"]["to"] == "ref('dim_cities')"
    assert declared["city_id"]["field"] == "city_id"
    assert declared["date_key"]["to"] == "ref('dim_date')"
    assert declared["date_key"]["field"] == "date_day"


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("family", "column"), sorted(REQUIRED_MEASURES.items()))
def test_the_required_measures_are_present(columns, family: str, column: str) -> None:
    assert column in columns, f"no column for {family}"


def test_the_grid_cell_provenance_travels_with_the_measures(columns) -> None:
    """dim_cities holds what was asked for; these are what answered."""
    assert {"api_latitude", "api_longitude", "api_elevation_m"} <= columns


def test_lineage_survives_into_gold(columns) -> None:
    """So a mart row can be traced to the ingestion run that produced it."""
    assert {"ingested_at", "batch_id"} <= columns


def test_the_measures_are_not_all_null(engine, columns) -> None:
    """A column present but never populated is worse than an absent one."""
    selects = ", ".join(f"count({c})" for c in REQUIRED_MEASURES.values())
    counts = query(engine, f"select {selects} from {FACT}")[0]
    total = query(engine, f"select count(*) from {FACT}")[0][0]
    if total == 0:
        pytest.skip("fact is empty")
    for column, populated in zip(REQUIRED_MEASURES.values(), counts):
        assert populated > 0, column


# ---------------------------------------------------------------------------
# Size and materialisation
# ---------------------------------------------------------------------------


def test_the_fact_carries_every_silver_row(engine, columns) -> None:
    """Asserted against silver, not against a constant.

    The backfill runs across days, so a hardcoded 173,520 would be red for
    most of its life and would teach everyone to ignore it.
    """
    silver, fact = query(
        engine,
        "select (select count(*) from silver_staging.stg_observations_daily), "
        f"(select count(*) from {FACT})",
    )[0]
    assert silver == fact


def test_the_completed_size_is_the_one_the_proposal_expects(engine, columns) -> None:
    """15 cities x 11,568 days = 173,520 once the backfill finishes.

    The proposal's ~164,000 assumed a round thirty years; the configured range
    runs 1995-01-01 to the archive edge, which is longer. Asserted as a
    per-city rate so it holds mid-backfill.
    """
    rows = query(
        engine,
        f"""select city_id, count(*) as days,
                   max(date_key) - min(date_key) + 1 as span
            from {FACT} group by city_id""",
    )
    if not rows:
        pytest.skip("fact is empty")
    complete = [r for r in rows if r.days > 11_000]
    assert complete, "no city has finished backfilling yet"
    for row in complete:
        assert row.days == row.span, f"{row.city_id} has a gap"
        assert 11_000 < row.days < 12_000
    assert len(rows) * 11_568 <= 173_520


def test_it_is_a_table(engine) -> None:
    kind = query(
        engine,
        "select table_type from information_schema.tables "
        f"where table_schema = '{GOLD}' and table_name = 'fact_weather_observations'",
    )
    if not kind:
        pytest.skip("fact not built")
    assert kind[0][0] == "BASE TABLE"


def test_there_is_a_unique_index_on_the_grain(engine) -> None:
    definitions = [
        r[0]
        for r in query(
            engine,
            "select indexdef from pg_indexes where schemaname = :s "
            "and tablename = 'fact_weather_observations'",
            s=GOLD,
        )
    ]
    if not definitions:
        pytest.skip("fact not built")
    on_grain = [d for d in definitions if "(city_id, date_key)" in d]
    assert on_grain, definitions
    assert any("UNIQUE" in d.upper() for d in on_grain)


def test_the_index_is_declared_in_the_model(engine) -> None:
    body = MODEL.read_text(encoding="utf-8")
    assert "indexes" in body
    assert "'unique': True" in body


def test_the_grain_index_serves_a_dashboard_query(engine, columns) -> None:
    """One city, one year — the shape the dashboard actually issues."""
    plan = "\n".join(
        r[0]
        for r in query(
            engine,
            f"""explain (costs off) select date_key, temperature_2m_max from {FACT}
                where city_id = 'london'
                  and date_key between date '2020-01-01' and date '2020-12-31'""",
        )
    )
    if "london" not in plan.lower():
        pytest.skip("london not landed yet")
    assert "Index Scan" in plan, plan


# ---------------------------------------------------------------------------
# It reads silver, not the dimensions
# ---------------------------------------------------------------------------


def test_it_selects_from_silver_without_joining_the_dimensions() -> None:
    """An inner join would enforce integrity by dropping unmatched rows.

    That is the wrong failure: a fact silently missing a city is
    indistinguishable from a city with no weather. The relationships tests
    assert the same property and fail loudly instead.
    """
    sql = "\n".join(
        line
        for line in MODEL.read_text(encoding="utf-8").splitlines()
        if not line.strip().startswith("--")
    ).lower()
    assert "ref('stg_observations_daily')" in sql
    assert "join" not in sql
    assert "dim_cities" not in sql
    assert "dim_date" not in sql
