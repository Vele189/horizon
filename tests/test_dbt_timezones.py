"""Tests for UTC normalisation and the timezone traps in cities.yml.

Four of the fifteen cities exist partly to break naive timezone handling, and
each has an assertion here that a plausible shortcut would fail:

*   **Phoenix** keeps standard time all year on a US longitude. Inferring a
    zone from a country or a longitude gives it America/Denver, and every
    reading from March to November lands an hour out — a shift small enough to
    read as weather.
*   **Delhi** is UTC+05:30. `observation_time + interval '5 hours'` looks like
    a conversion and is wrong by half an hour.
*   **Sydney** and **Auckland** run daylight saving on the southern calendar,
    so a hardcoded northern one is not merely wrong but inverted.
*   **Reykjavik** is UTC+0 year round, which makes it the city where a broken
    conversion looks correct.

The conclusion those tests reach is that local time is a one-way view: at a
fall-back, two instants share one wall-clock reading, so UTC is what silver
stores and what everything joins on.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("sqlalchemy")

from sqlalchemy import text  # noqa: E402

from cities import load_cities  # noqa: E402
from dbt_analytics.export_cities import FIELDS, SEED, render  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DBT_DIR = REPO_ROOT / "dbt_analytics"
SILVER = "silver_staging"

TRAPS = {
    "phoenix": "America/Phoenix",
    "reykjavik": "Atlantic/Reykjavik",
    "delhi": "Asia/Kolkata",
    "sydney": "Australia/Sydney",
    "auckland": "Pacific/Auckland",
}


# ---------------------------------------------------------------------------
# The timezone reaches the warehouse, and stays in step with cities.yml
# ---------------------------------------------------------------------------


def test_the_seed_matches_the_yaml() -> None:
    """Two copies of the same list is one copy and one liability."""
    assert SEED.read_text(encoding="utf-8") == render()


def test_the_staleness_check_is_runnable() -> None:
    result = subprocess.run(
        [sys.executable, "dbt_analytics/export_cities.py", "--check"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_the_seed_carries_the_timezone() -> None:
    assert "timezone" in FIELDS
    header = SEED.read_text(encoding="utf-8").splitlines()[0]
    assert "timezone" in header.split(",")


def test_stg_cities_exposes_the_timezone() -> None:
    body = (DBT_DIR / "models" / "staging" / "stg_cities.sql").read_text(
        encoding="utf-8"
    )
    assert "timezone" in body
    assert "ref('cities')" in body


@pytest.mark.parametrize(("city_id", "zone"), sorted(TRAPS.items()))
def test_each_trap_city_keeps_its_configured_zone(city_id: str, zone: str) -> None:
    assert load_cities()[city_id].timezone == zone


def test_every_configured_zone_is_known_to_postgres(engine) -> None:
    """Python's zoneinfo and Postgres's tz database are different databases."""
    with engine.connect() as connection:
        known = {
            row[0] for row in connection.execute(text("select name from pg_timezone_names"))
        }
    missing = [c.id for c in load_cities() if c.timezone not in known]
    assert not missing, f"Postgres does not know the zone for {missing}"


# ---------------------------------------------------------------------------
# Nothing naive, nothing in the future
# ---------------------------------------------------------------------------


def test_every_silver_timestamp_carries_a_zone(engine) -> None:
    with engine.connect() as connection:
        naive = connection.execute(
            text(
                "select table_name, column_name, data_type "
                "from information_schema.columns "
                "where table_schema = :s "
                "and data_type in ('timestamp without time zone', "
                "'time without time zone')"
            ),
            {"s": SILVER},
        ).fetchall()
    assert not naive, f"naive timestamps in silver: {naive}"


def test_the_observation_columns_are_timestamptz(engine) -> None:
    with engine.connect() as connection:
        types = dict(
            connection.execute(
                text(
                    "select table_name || '.' || column_name, data_type "
                    "from information_schema.columns "
                    "where table_schema = :s "
                    "and column_name in ('observation_time', 'ingested_at')"
                ),
                {"s": SILVER},
            ).fetchall()
        )
    assert types, "no staging models found"
    assert set(types.values()) == {"timestamp with time zone"}, types


@pytest.mark.parametrize("grain", ["daily", "hourly"])
def test_no_observation_is_in_the_future(engine, grain: str) -> None:
    with engine.connect() as connection:
        ahead = connection.execute(
            text(
                f"select count(*) from {SILVER}.stg_observations_{grain} "
                "where observation_time > now() + interval '1 day'"
            )
        ).scalar()
    assert ahead == 0


# ---------------------------------------------------------------------------
# The four traps, against real landed rows
# ---------------------------------------------------------------------------


def offsets_for(engine, city_id: str, zone: str) -> list:
    with engine.connect() as connection:
        return [
            row[0]
            for row in connection.execute(
                text(
                    f"select distinct (observation_time at time zone '{zone}') "
                    f"at time zone 'UTC' - observation_time "
                    f"from {SILVER}.stg_observations_hourly where city_id = :c"
                ),
                {"c": city_id},
            )
        ]


def test_phoenix_never_shifts(engine) -> None:
    """Across two years of US transitions, one offset and only one."""
    found = offsets_for(engine, "phoenix", "America/Phoenix")
    if not found:
        pytest.skip("phoenix hourly rows not landed yet")
    assert len(found) == 1, f"Phoenix took {len(found)} offsets: {found}"


def test_portland_does_shift(engine) -> None:
    """The counterweight: 'Phoenix never shifts' proves nothing on its own.

    A pipeline that skipped timezone conversion entirely would satisfy Phoenix
    perfectly. Portland is on the same longitude and does observe DST.
    """
    found = offsets_for(engine, "portland", "America/Los_Angeles")
    if not found:
        pytest.skip("portland hourly rows not landed yet")
    assert len(found) == 2, f"Portland took {len(found)} offsets: {found}"


def test_reykjavik_is_utc(engine) -> None:
    """The city where a broken conversion looks correct."""
    import datetime as dt

    found = offsets_for(engine, "reykjavik", "Atlantic/Reykjavik")
    if not found:
        pytest.skip("reykjavik hourly rows not landed yet")
    assert found == [dt.timedelta(0)]


def test_delhi_keeps_its_half_hour(engine) -> None:
    """`+ interval '5 hours'` passes review and is wrong for 1.4 billion people."""
    with engine.connect() as connection:
        total, off_the_half = connection.execute(
            text(
                f"""select count(*),
                       count(*) filter (where extract(minute from
                           (observation_time at time zone 'Asia/Kolkata')) <> 30)
                    from {SILVER}.stg_observations_hourly where city_id = 'delhi'"""
            )
        ).one()
    if not total:
        pytest.skip("delhi hourly rows not landed yet")
    assert off_the_half == 0, f"{off_the_half} of {total} Delhi rows are not on :30"


def test_sydney_runs_daylight_saving_on_the_southern_calendar(engine) -> None:
    """January is high summer in Sydney (+11); July is winter (+10).

    A hardcoded northern calendar inverts this, adding an hour exactly where
    one should be subtracted — an error of two hours, not one.
    """
    import datetime as dt

    with engine.connect() as connection:
        by_month = dict(
            connection.execute(
                text(
                    f"""select extract(month from observation_time)::int,
                           min((observation_time at time zone 'Australia/Sydney')
                               at time zone 'UTC' - observation_time)
                        from {SILVER}.stg_observations_hourly
                        where city_id = 'sydney'
                          and extract(month from observation_time) in (1, 7)
                        group by 1"""
                )
            ).fetchall()
        )
    if not by_month:
        pytest.skip("sydney hourly rows not landed yet")
    assert by_month[1] == dt.timedelta(hours=11), "southern summer should be +11"
    assert by_month[7] == dt.timedelta(hours=10), "southern winter should be +10"


def test_local_time_is_lossy_only_at_fall_back(engine) -> None:
    """The reason the macro is one-way, measured rather than asserted.

    Two instants share one wall-clock reading at a fall-back, so the reverse
    conversion cannot recover which. That is clocks, not a bug — but it must be
    ten rows, not ten thousand.
    """
    with engine.connect() as connection:
        failures = connection.execute(
            text(
                f"""select observations.city_id, count(*)
                    from {SILVER}.stg_observations_hourly as observations
                    join {SILVER}.stg_cities as cities
                      on cities.city_id = observations.city_id
                    where ((observations.observation_time at time zone cities.timezone)
                            at time zone cities.timezone)
                          <> observations.observation_time
                    group by 1"""
            )
        ).fetchall()
    for city_id, count in failures:
        assert count <= 4, f"{city_id} lost {count} rows; expected at most one per transition"


def test_the_conversion_macro_is_one_way() -> None:
    """Documented as such, so nobody reaches for the inverse."""
    macro = (DBT_DIR / "macros" / "to_local_time.sql").read_text(encoding="utf-8")
    assert "one way" in macro.lower()
    assert "at time zone" in macro
