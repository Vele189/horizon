"""Tests for the city dimension.

The dimension is built from `config/cities.yml`, never from observation data.
That direction is the point: a dimension inferred from what happened to land
would list fourteen cities during a backfill and would quietly lose one whose
ingestion failed. The registry says what the set *is*; the facts say what has
been observed of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("sqlalchemy")

from sqlalchemy import text  # noqa: E402

from cities import load_cities  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DBT_DIR = REPO_ROOT / "dbt_analytics"
MODEL = DBT_DIR / "models" / "marts" / "dim_cities.sql"
GOLD = "gold_marts"

REQUIRED_COLUMNS = (
    "city_id", "name", "country", "region", "latitude", "longitude",
    "elevation_m", "timezone", "koppen", "hemisphere",
)



def arguments_of(test: dict, name: str) -> dict:
    """Arguments of a generic test, as dbt 1.12 nests them.

    dbt deprecated top-level arguments in favour of an `arguments` block, so
    the shape these tests read changed under them. Reading through one helper
    means the next such change is one edit rather than four.
    """
    return test[name].get("arguments", test[name])

@pytest.fixture(scope="module")
def marts_yml() -> dict:
    return yaml.safe_load(
        (DBT_DIR / "models" / "marts" / "_marts.yml").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def dim_cities_doc(marts_yml) -> dict:
    return next(m for m in marts_yml["models"] if m["name"] == "dim_cities")


@pytest.fixture
def dimension(engine):
    with engine.connect() as connection:
        rows = connection.execute(
            text(f"select * from {GOLD}.dim_cities order by city_id")
        ).fetchall()
    if not rows:
        pytest.skip("dim_cities not built")
    return rows


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_one_row_per_city(dimension) -> None:
    registry = load_cities()
    assert len(dimension) == len(registry) == 15
    assert {r.city_id for r in dimension} == set(registry.ids)


@pytest.mark.parametrize("column", REQUIRED_COLUMNS)
def test_the_required_columns_are_present(engine, column: str) -> None:
    with engine.connect() as connection:
        columns = {
            row[0]
            for row in connection.execute(
                text(
                    "select column_name from information_schema.columns "
                    "where table_schema = :s and table_name = 'dim_cities'"
                ),
                {"s": GOLD},
            )
        }
    if not columns:
        pytest.skip("dim_cities not built")
    assert column in columns


def test_it_is_a_table_not_a_view(engine) -> None:
    """Gold is materialised: the dashboard reads it over a serverless link."""
    with engine.connect() as connection:
        kind = connection.execute(
            text(
                "select table_type from information_schema.tables "
                "where table_schema = :s and table_name = 'dim_cities'"
            ),
            {"s": GOLD},
        ).scalar()
    if kind is None:
        pytest.skip("dim_cities not built")
    assert kind == "BASE TABLE"


def test_city_id_is_unique_and_present(dimension) -> None:
    ids = [r.city_id for r in dimension]
    assert len(ids) == len(set(ids))
    assert all(ids)


# ---------------------------------------------------------------------------
# hemisphere is derived, in both languages, and they agree
# ---------------------------------------------------------------------------


def test_hemisphere_is_derived_in_sql_not_carried_over() -> None:
    body = MODEL.read_text(encoding="utf-8")
    assert "case when latitude >= 0" in body
    # The seeded column must not simply be selected through.
    assert "\n    hemisphere,\n" not in body


def test_hemisphere_matches_the_sign_of_the_latitude(dimension) -> None:
    for row in dimension:
        expected = "north" if row.latitude >= 0 else "south"
        assert row.hemisphere == expected, row.city_id


def test_both_derivations_agree(engine, dimension) -> None:
    """Python computes it from lat >= 0 and seeds it; SQL recomputes it.

    Neither is authoritative, which is the point: a disagreement means one has
    drifted, and a silent drift inverts summer and winter for the five southern
    cities in every season mapping downstream.
    """
    with engine.connect() as connection:
        disagreements = connection.execute(
            text(
                f"""select d.city_id, d.hemisphere, s.hemisphere
                    from {GOLD}.dim_cities d
                    join silver_staging.stg_cities s using (city_id)
                    where d.hemisphere <> s.hemisphere"""
            )
        ).fetchall()
    assert not disagreements


def test_the_split_is_ten_north_five_south(dimension) -> None:
    """Decision D2's envelope, asserted where the dashboard will read it."""
    counts = {"north": 0, "south": 0}
    for row in dimension:
        counts[row.hemisphere] += 1
    assert counts == {"north": 10, "south": 5}


# ---------------------------------------------------------------------------
# Built from the registry, not from the facts
# ---------------------------------------------------------------------------


def test_it_reads_the_registry_not_the_observations() -> None:
    body = MODEL.read_text(encoding="utf-8")
    assert "ref('stg_cities')" in body
    assert "observations" not in body


def test_every_registry_field_survives_into_the_dimension(dimension) -> None:
    registry = {c.id: c for c in load_cities()}
    for row in dimension:
        city = registry[row.city_id]
        assert row.name == city.name
        assert row.country == city.country
        assert row.region == city.region
        assert row.timezone == city.timezone
        assert row.koppen == city.koppen
        assert row.latitude == pytest.approx(city.lat)
        assert row.longitude == pytest.approx(city.lon)
        assert row.elevation_m == pytest.approx(city.elevation_m)


def test_the_dimension_holds_cities_with_no_facts_yet(engine, dimension) -> None:
    """The normal state of a multi-day backfill, and not an error.

    A dimension built from landed data would silently shrink to whatever had
    arrived. This asserts the opposite property: the dimension is complete even
    when the facts are not.
    """
    with engine.connect() as connection:
        observed = {
            row[0]
            for row in connection.execute(
                text("select distinct city_id from silver_staging.stg_observations_daily")
            )
        }
    assert {r.city_id for r in dimension} >= observed
    assert len(dimension) == 15


# ---------------------------------------------------------------------------
# The framing
# ---------------------------------------------------------------------------


def test_the_description_explains_gridded_reanalysis(dim_cities_doc) -> None:
    """The original draft called this dim_weather_stations.

    Open-Meteo serves ERA5: a physical model reconciled with observations onto
    a regular grid. There is no station, no instrument history, no siting
    metadata and no station id to join on, and using station language would
    misdescribe the source to anyone who knows the domain.
    """
    description = dim_cities_doc["description"].lower()
    assert "reanalysis" in description
    assert "grid" in description
    assert "station" in description, "the framing it replaces should be named"


def test_the_description_distinguishes_asked_from_answered(dim_cities_doc) -> None:
    """The coordinates here are the request; api_* on the facts is the reply."""
    description = dim_cities_doc["description"]
    assert "api_latitude" in description


def test_region_is_constrained(dim_cities_doc) -> None:
    region = next(c for c in dim_cities_doc["columns"] if c["name"] == "region")
    values = next(
        arguments_of(t, "accepted_values")["values"] for t in region["tests"]
        if isinstance(t, dict) and "accepted_values" in t
    )
    assert set(values) == {c.region for c in load_cities()}
