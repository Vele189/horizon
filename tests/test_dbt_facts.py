"""Tests for the daily fact table.

The fact is selected from silver rather than joined to the dimensions. An
inner join would enforce referential integrity by *dropping* rows it could not
match, and a fact silently missing a city looks exactly like a city with no
weather. The relationships tests assert the same property and fail loudly,
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



def arguments_of(test: dict, name: str) -> dict:
    """Arguments of a generic test, as dbt 1.12 nests them.

    dbt deprecated top-level arguments in favour of an `arguments` block, so
    the shape these tests read changed under them. Reading through one helper
    means the next such change is one edit rather than four.
    """
    return test[name].get("arguments", test[name])

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
                declared[column["name"]] = arguments_of(test, "relationships")
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
    """One city, one year: the shape the dashboard actually issues."""
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


# ---------------------------------------------------------------------------
# The hourly fact (DBT-08)
# ---------------------------------------------------------------------------

HOURLY = f"{GOLD}.fact_weather_hourly"
HOURLY_MODEL = DBT_DIR / "models" / "marts" / "fact_weather_hourly.sql"


@pytest.fixture
def hourly(engine):
    rows = query(engine, f"select count(*) from {HOURLY}")
    if not rows or rows[0][0] == 0:
        pytest.skip("fact_weather_hourly not built")
    return rows[0][0]


def test_the_hourly_grain_is_one_row_per_city_hour(engine, hourly) -> None:
    duplicates = query(
        engine,
        f"select count(*) from (select city_id, observation_hour from {HOURLY} "
        "group by city_id, observation_hour having count(*) > 1) d",
    )[0][0]
    assert duplicates == 0


def test_the_hourly_fact_covers_twenty_four_months(engine, hourly) -> None:
    """Silver holds more; the fact must hold exactly what it claims.

    Two cities carry an extra eight months from an archival sample that took a
    calendar year rather than the anchored window. Unfiltered they would
    quietly widen a table documented as trailing-24-months, and a per-city
    average over "the window" would cover different windows per city.
    """
    spans = query(
        engine,
        f"""select city_id,
                   (max(observation_hour) at time zone 'UTC')::date
                     - (min(observation_hour) at time zone 'UTC')::date as days
            from {HOURLY} group by city_id""",
    )
    assert len(spans) == 15
    for row in spans:
        assert 729 <= row.days <= 732, f"{row.city_id} spans {row.days} days"


def test_the_window_filter_actually_removes_rows(engine, hourly) -> None:
    silver = query(
        engine, "select count(*) from silver_staging.stg_observations_hourly"
    )[0][0]
    assert hourly < silver, "the out-of-window rows are still present"
    assert 255_000 < hourly < 275_000


def test_the_row_count_is_about_263_000(engine, hourly) -> None:
    assert 260_000 <= hourly <= 266_000, hourly


def test_every_hourly_city_key_resolves(engine, hourly) -> None:
    orphans = query(
        engine,
        f"select distinct city_id from {HOURLY} "
        f"where city_id not in (select city_id from {GOLD}.dim_cities)",
    )
    assert not orphans


# --- pressure tendency -----------------------------------------------------


def test_tendency_uses_a_range_frame_not_lag() -> None:
    """`lag(pressure, 3)` counts rows, not hours.

    One missing hour makes it reach four hours back and report the difference
    as a three-hour change: a fabricated storm signal from data that merely
    had a hole.
    """
    body = HOURLY_MODEL.read_text(encoding="utf-8")
    assert "range between interval '3 hours' preceding" in body
    assert "range between interval '24 hours' preceding" in body
    # The comment explains why lag is wrong, so strip comments before checking.
    sql = "\n".join(
        line for line in body.splitlines() if not line.strip().startswith("--")
    ).lower()
    assert "lag(" not in sql


def test_tendency_is_computed_on_sea_level_pressure() -> None:
    """Surface pressure carries elevation; Johannesburg reads 822 against London's 1013."""
    body = HOURLY_MODEL.read_text(encoding="utf-8")
    assert "pressure_msl - first_value(pressure_msl)" in body
    assert "surface_pressure - first_value" not in body


@pytest.mark.parametrize("hours", [3, 24])
def test_the_tendency_spans_the_hours_it_claims(engine, hourly, hours: int) -> None:
    """Recomputed by joining on the timestamp, so the two methods must agree."""
    disagreements = query(
        engine,
        f"""select count(*) from {HOURLY} as current_hour
            join {HOURLY} as earlier
              on earlier.city_id = current_hour.city_id
             and earlier.observation_hour = current_hour.observation_hour
                 - interval '{hours} hours'
            where abs(coalesce(current_hour.pressure_tendency_{hours}h, 0)
                      - coalesce(current_hour.pressure_msl - earlier.pressure_msl, 0))
                  > 0.001""",
    )[0][0]
    assert disagreements == 0


@pytest.mark.parametrize(("column", "expected"), [("3h", 3), ("24h", 24)])
def test_tendency_is_null_only_at_the_start_of_each_series(
    engine, hourly, column: str, expected: int
) -> None:
    wrong = query(
        engine,
        f"""select city_id, count(*) filter (where pressure_tendency_{column} is null) as n
            from {HOURLY} where pressure_msl is not null
            group by city_id having count(*) filter
                (where pressure_tendency_{column} is null) <> {expected}""",
    )
    assert not wrong, wrong


def test_the_tendency_finds_real_storms(engine, hourly) -> None:
    """A tendency that is arithmetically right can still be meaningless.

    The deepest 24-hour falls should land where explosive cyclogenesis happens,
    and should reach past the ~-24 hPa that defines it.
    """
    deepest = query(
        engine,
        f"""select city_id, pressure_tendency_24h, pressure_msl from {HOURLY}
            order by pressure_tendency_24h asc limit 5""",
    )
    assert all(r.pressure_tendency_24h < -24 for r in deepest), deepest
    assert "reykjavik" in {r.city_id for r in deepest}, "the North Atlantic storm track"


def test_falling_pressure_goes_with_stronger_wind(engine, hourly) -> None:
    """Weak but correctly signed; gusts follow the gradient, not the tendency."""
    correlation = query(
        engine,
        f"select corr(pressure_tendency_24h, wind_gusts_10m) from {HOURLY} "
        "where pressure_tendency_24h is not null",
    )[0][0]
    assert correlation is not None
    assert correlation < 0, correlation


# --- size ------------------------------------------------------------------


def test_gold_fits_inside_the_neon_allowance(engine, hourly) -> None:
    """Gold is the only layer promoted to Neon, so this budget is real here.

    Bronze's version of this check is a yardstick; this one is the constraint.
    """
    from ingestion.backfill import NEON_STORAGE_BUDGET_BYTES

    total = query(
        engine,
        """select coalesce(sum(pg_total_relation_size(
               format('%I.%I', schemaname, relname))), 0)
           from pg_stat_user_tables where schemaname = 'gold_marts'""",
    )[0][0]
    assert total > 0
    assert total < NEON_STORAGE_BUDGET_BYTES * 0.5, (
        f"gold is {total / 1e6:.0f} MB of a "
        f"{NEON_STORAGE_BUDGET_BYTES / 1e6:.0f} MB allowance"
    )


def test_the_hourly_fact_is_the_largest_object(engine, hourly) -> None:
    """Which is why it is capped at 24 months rather than thirty years."""
    sizes = query(
        engine,
        """select relname, pg_total_relation_size(
               format('%I.%I', schemaname, relname)) as bytes
           from pg_stat_user_tables where schemaname = 'gold_marts'
           order by bytes desc""",
    )
    assert sizes[0].relname == "fact_weather_hourly"


def test_thirty_years_hourly_would_not_fit(engine, hourly) -> None:
    """The reason for the 24-month cap, as arithmetic rather than assertion."""
    from ingestion.backfill import NEON_STORAGE_BUDGET_BYTES

    bytes_now = query(
        engine, f"select pg_total_relation_size('{HOURLY}')"
    )[0][0]
    per_row = bytes_now / hourly
    thirty_years = 15 * 11_568 * 24
    assert thirty_years > 4_000_000
    assert per_row * thirty_years > NEON_STORAGE_BUDGET_BYTES


def test_there_is_a_unique_index_on_the_hourly_grain(engine, hourly) -> None:
    definitions = [
        r[0]
        for r in query(
            engine,
            "select indexdef from pg_indexes where schemaname = :s "
            "and tablename = 'fact_weather_hourly'",
            s=GOLD,
        )
    ]
    on_grain = [d for d in definitions if "(city_id, observation_hour)" in d]
    assert on_grain, definitions
    assert any("UNIQUE" in d.upper() for d in on_grain)
