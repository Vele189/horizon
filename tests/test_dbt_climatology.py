"""Tests for the leakage-safe rolling climatology.

The requirement this model is shaped around is that the baseline labelling an
observation must exclude that observation's own year. Without it the
observation contributes to its own mean and inflates its own sigma, every
Z-score comes out too small, and a model trained on those labels is scoring
against a target that has already seen its own answer.

Measured here rather than argued: the exclusion moves the mean by 0.037 °C and
sigma by 0.07%, which is invisible in a spot check — and changes the count of
|Z| > 2.5 days from 740 to 976. The leaky baseline misses a quarter of the
extremes, because the shift is small everywhere and the events live in the tail
where small shifts decide membership.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("sqlalchemy")

from sqlalchemy import text  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DBT_DIR = REPO_ROOT / "dbt_analytics"
GOLD = "gold_marts"
CLIMATOLOGY = f"{GOLD}.fact_climatology"


def query(engine, sql: str, **params):
    with engine.connect() as connection:
        return connection.execute(text(sql), params).fetchall()


@pytest.fixture
def built(engine):
    rows = query(engine, f"select count(*) from {CLIMATOLOGY}")
    if not rows or rows[0][0] == 0:
        pytest.skip("fact_climatology not built")
    return rows[0][0]


@pytest.fixture
def complete_cities(engine, built):
    """Cities with a full reference period; the partial ones prove nothing."""
    return [
        r[0]
        for r in query(
            engine,
            f"""select city_id from {GOLD}.fact_weather_observations group by 1
                having count(distinct extract(year from date_key)) >= 25""",
        )
    ]


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------


def test_the_power_sum_sigma_matches_postgres(engine, built) -> None:
    """sigma is recovered from Sn/Sx/Sx2 so the exclusion can be a subtraction.

    That identity is easy to get subtly wrong and the result still looks like a
    number, so it is checked against Postgres's own aggregate on the case where
    nothing is excluded.
    """
    result = query(
        engine,
        f"""with windows as (
              select gs.d as target_day,
                     ((gs.d - 1 + o.off) % 366 + 366) % 366 + 1 as source_day
              from generate_series(1,366) gs(d), generate_series(-7,7) o(off)),
            direct as (
              select f.city_id, w.target_day,
                     avg(f.temperature_2m_mean::numeric)::double precision as mu,
                     stddev_samp(f.temperature_2m_mean::numeric)::double precision as sd
              from {GOLD}.fact_weather_observations f
              join {GOLD}.dim_date d on d.date_day = f.date_key
              join windows w on w.source_day = d.climatology_day
              where f.temperature_2m_mean is not null
              group by 1,2)
            select count(*),
                   max(abs(direct.mu - c.mean_temperature_2m_mean_all_years)),
                   max(abs(direct.sd - c.stddev_temperature_2m_mean_all_years))
            from direct join {CLIMATOLOGY} c
              on c.city_id = direct.city_id and c.climatology_day = direct.target_day""",
    )[0]
    compared, mu_error, sd_error = result
    assert compared > 10_000
    assert mu_error == 0
    assert sd_error < 1e-12, sd_error


def test_the_exclusion_arithmetic_balances(engine, built) -> None:
    broken = query(
        engine,
        f"select count(*) from {CLIMATOLOGY} "
        "where observations <> observations_all_years - excluded_observations",
    )[0][0]
    assert broken == 0


def test_the_exclusion_removes_the_year_when_it_contributed(engine, built) -> None:
    """The failure this model exists to prevent, asserted directly."""
    unexcluded = query(
        engine,
        f"select count(*) from {CLIMATOLOGY} where excluded_observations > 0 "
        "and observations >= observations_all_years",
    )[0][0]
    assert unexcluded == 0


def test_the_exclusion_is_a_no_op_only_at_the_edges(engine, built) -> None:
    """2026 reaches only to the archive edge, so it is absent from later windows.

    Distinguishing "nothing to exclude" from "exclusion broken" is why
    excluded_observations is carried rather than inferred from a difference.
    """
    rows = query(
        engine,
        f"""select distinct for_year from {CLIMATOLOGY}
            where excluded_observations = 0 and observations_all_years > 0""",
    )
    years = {r[0] for r in rows}
    latest = query(engine, f"select max(for_year) from {CLIMATOLOGY}")[0][0]
    assert years <= {latest}, f"unexpected years with nothing excluded: {years}"


def test_the_exclusion_is_configurable() -> None:
    model = (DBT_DIR / "models" / "marts" / "fact_climatology.sql").read_text(
        encoding="utf-8"
    )
    assert "climatology_exclude_own_year" in model
    assert "var('climatology_exclude_own_year', true)" in model


# ---------------------------------------------------------------------------
# What the leakage actually costs
# ---------------------------------------------------------------------------


def test_leakage_barely_moves_the_baseline_and_badly_moves_the_tail(
    engine, built
) -> None:
    """Both halves matter, and only together.

    Small enough to survive a spot check; large enough to change a quarter of
    the extreme-day labels. Build this in from the start, as the ticket says,
    rather than patch it once the metrics look suspicious.
    """
    shift = query(
        engine,
        f"""select avg(abs(mean_temperature_2m_mean - mean_temperature_2m_mean_all_years))
            from {CLIMATOLOGY} where excluded_observations > 0""",
    )[0][0]
    assert shift is not None
    assert shift < 0.2, "the per-day shift really is small"

    safe, leaky = query(
        engine,
        f"""with z as (
              select (f.temperature_2m_mean - c.mean_temperature_2m_mean)
                       / nullif(c.stddev_temperature_2m_mean, 0) as z_safe,
                     (f.temperature_2m_mean - c.mean_temperature_2m_mean_all_years)
                       / nullif(c.stddev_temperature_2m_mean_all_years, 0) as z_leaky
              from {GOLD}.fact_weather_observations f
              join {GOLD}.dim_date d on d.date_day = f.date_key
              join {CLIMATOLOGY} c on c.city_id = f.city_id
               and c.month_day = d.month_day and c.for_year = d.year
              where c.stddev_temperature_2m_mean is not null)
            select count(*) filter (where abs(z_safe) > 2.5),
                   count(*) filter (where abs(z_leaky) > 2.5) from z""",
    )[0]
    assert safe > leaky, "the leaky baseline understates extremes"
    assert (safe - leaky) / safe > 0.1, (
        f"leaky misses {safe - leaky} of {safe} extreme days"
    )


# ---------------------------------------------------------------------------
# Sigma
# ---------------------------------------------------------------------------


def test_sigma_is_never_zero(engine, built) -> None:
    """~450 observations in a fortnight cannot be bit-identical."""
    zeros = query(
        engine, f"select count(*) from {CLIMATOLOGY} where stddev_temperature_2m_mean = 0"
    )[0][0]
    assert zeros == 0


def test_a_null_sigma_means_the_exclusion_emptied_the_window(engine, built) -> None:
    """Null is honest, not a bug — and only for that one reason.

    A city with a single reference year has no leakage-free baseline at all.
    Returning null says so; falling back to the all-years value would silently
    reintroduce exactly the leakage this model removes.
    """
    wrong = query(
        engine,
        f"select count(*) from {CLIMATOLOGY} "
        "where stddev_temperature_2m_mean is null and observations > 1",
    )[0][0]
    assert wrong == 0

    nulls = query(
        engine,
        f"""select city_id, count(distinct for_year) from {CLIMATOLOGY}
            where stddev_temperature_2m_mean is null group by 1""",
    )
    for city_id, years in nulls:
        reference = query(
            engine,
            f"select max(reference_years) from {CLIMATOLOGY} where city_id = :c",
            c=city_id,
        )[0][0]
        assert reference == 1, f"{city_id} has {reference} reference years"


def test_tropical_cities_have_the_tightest_distributions(
    engine, complete_cities
) -> None:
    """The sanity check, on the cities that have a full record.

    Note this is the *within-window* sigma — variability around the seasonal
    curve, which is what a Z-score should be measured against. It ranks
    differently from the sigma of all days pooled; see the test below.
    """
    if not {"singapore", "lagos"} <= set(complete_cities):
        pytest.skip("the tropical cities have not finished backfilling")
    sigmas = dict(
        query(
            engine,
            f"""select city_id, avg(stddev_temperature_2m_mean) from {CLIMATOLOGY}
                where city_id = any(:c) group by 1""",
            c=complete_cities,
        )
    )
    tropical = {c: s for c, s in sigmas.items() if c in ("singapore", "lagos")}
    others = {c: s for c, s in sigmas.items() if c not in tropical}
    assert all(s < 1.0 for s in tropical.values()), tropical
    if others:
        assert min(others.values()) > 2.0 * max(tropical.values()), sigmas


def test_singapore_is_the_least_variable_city_overall(engine, complete_cities) -> None:
    """The checklist's sanity check, on the quantity it is actually about.

    Singapore has the smallest spread of daily temperature *pooled across the
    year* — but Lagos has the smaller within-window sigma, because Lagos has a
    3.4 °C seasonal swing against Singapore's 1.6 °C while being marginally
    steadier around it. Both are correct measurements of different things, and
    the climatology needs the second.
    """
    if "singapore" not in complete_cities:
        pytest.skip("singapore has not finished backfilling")
    pooled = dict(
        query(
            engine,
            f"""select city_id, stddev_samp(temperature_2m_mean)
                from {GOLD}.fact_weather_observations
                where city_id = any(:c) group by 1""",
            c=complete_cities,
        )
    )
    assert min(pooled, key=pooled.get) == "singapore", pooled


def test_moscow_is_the_most_variable_once_it_lands(engine, complete_cities) -> None:
    """The other half of the sanity check, pending its data.

    Skips rather than passes while Moscow is still backfilling — a check that
    silently passes on absent data is worse than one that says it is waiting.
    """
    if "moscow" not in complete_cities:
        pytest.skip("moscow has not finished backfilling; cannot check yet")
    sigmas = dict(
        query(
            engine,
            f"""select city_id, avg(stddev_temperature_2m_mean) from {CLIMATOLOGY}
                where city_id = any(:c) group by 1""",
            c=complete_cities,
        )
    )
    assert max(sigmas, key=sigmas.get) == "moscow", sigmas


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


def test_the_window_is_fifteen_days_wide(engine, built) -> None:
    per_year = query(
        engine,
        f"""select max(observations_all_years::numeric / nullif(reference_years, 0))
            from {CLIMATOLOGY} where reference_years > 5""",
    )[0][0]
    assert per_year is not None
    assert 14 <= float(per_year) <= 15.5, per_year


def test_the_window_wraps_the_year_boundary(engine, built) -> None:
    """A non-circular window would halve the data exactly where winter extremes are."""
    rows = query(
        engine,
        f"""select b.city_id, b.month_day, b.observations, m.observations
            from {CLIMATOLOGY} b
            join {CLIMATOLOGY} m
              on m.city_id = b.city_id and m.for_year = b.for_year
             and m.month_day = '07-01'
            where b.month_day in ('01-01', '12-31') and m.observations > 0""",
    )
    assert rows
    for city_id, month_day, boundary, midyear in rows:
        assert float(boundary) >= float(midyear) * 0.8, (
            f"{city_id} {month_day}: {boundary} vs {midyear}"
        )


def test_the_leap_day_is_its_own_row(engine, built) -> None:
    """Keyed on month_day, so 29 February does not collide with 1 March."""
    cities = query(
        engine, f"select count(distinct city_id) from {CLIMATOLOGY} where month_day = '02-29'"
    )[0][0]
    total = query(engine, f"select count(distinct city_id) from {CLIMATOLOGY}")[0][0]
    assert cities == total


def test_the_leap_day_window_is_not_short(engine, built) -> None:
    """Its own sample is a quarter the size; its window is not.

    Eight leap years in thirty-two, but the +/-7 day window around 29 February
    is drawn from 22 February to 7 March in every year, leap or not. An
    implementation that filtered the window to leap years would show roughly a
    quarter here.
    """
    rows = query(
        engine,
        f"""select leap.city_id, leap.observations, prior.observations
            from {CLIMATOLOGY} leap
            join {CLIMATOLOGY} prior
              on prior.city_id = leap.city_id and prior.for_year = leap.for_year
             and prior.month_day = '02-28'
            where leap.month_day = '02-29' and prior.observations > 0""",
    )
    assert rows
    for city_id, leap_day, prior_day in rows:
        assert float(leap_day) >= float(prior_day) * 0.8, (
            f"{city_id}: {leap_day} vs {prior_day}"
        )


def test_the_smoothing_width_is_configurable() -> None:
    window = (DBT_DIR / "models" / "intermediate" / "int_climatology_window.sql").read_text(
        encoding="utf-8"
    )
    assert "climatology_smoothing_days" in window
    assert "% 366" in window, "the window must wrap"


def test_every_calendar_day_has_a_baseline(engine, built) -> None:
    days = query(engine, f"select count(distinct month_day) from {CLIMATOLOGY}")[0][0]
    assert days == 366
