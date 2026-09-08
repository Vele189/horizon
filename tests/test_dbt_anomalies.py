"""Tests for the Z-score anomaly flags.

Both tails matter. `z > 2.5` reads naturally, passes review, and silently
discards every cold extreme. Phoenix would lose 125 of its 145 flagged days,
and a warm-only project would report the coldest city in the set as one of the
calmest. The flag is on `abs(z)`, and the direction is asserted against the
sign so an inverted branch cannot flag the right days and label them backwards.
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
ANOMALIES = f"{GOLD}.fact_weather_anomalies"
THRESHOLD = 2.5


def query(engine, sql: str, **params):
    with engine.connect() as connection:
        return connection.execute(text(sql), params).fetchall()


@pytest.fixture
def built(engine):
    rows = query(engine, f"select count(*) from {ANOMALIES}")
    if not rows or rows[0][0] == 0:
        pytest.skip("fact_weather_anomalies not built")
    return rows[0][0]


@pytest.fixture
def full_record(engine, built):
    """Cities whose baseline is the full reference period."""
    return [
        r[0]
        for r in query(
            engine,
            f"""select city_id from {ANOMALIES}
                where z_temperature_2m_mean is not null
                group by 1 having avg(baseline_observations) > 300""",
        )
    ]


# ---------------------------------------------------------------------------
# The score
# ---------------------------------------------------------------------------


def test_z_is_the_departure_over_sigma(engine, built) -> None:
    """Recomputed from the stored components, so the formula cannot drift."""
    wrong = query(
        engine,
        f"""select count(*) from {ANOMALIES}
            where z_temperature_2m_mean is not null
              and abs(z_temperature_2m_mean
                      - (temperature_2m_mean - mean_temperature_2m_mean)
                        / nullif(stddev_temperature_2m_mean, 0)) > 1e-9""",
    )[0][0]
    assert wrong == 0


def test_the_baseline_is_the_leakage_safe_one(engine, built) -> None:
    """Scoring against the all-years normal is the failure DBT-09 exists to stop."""
    leaky = query(
        engine,
        f"select count(*) from {ANOMALIES} where excludes_own_year is not true",
    )[0][0]
    assert leaky == 0


def test_the_standardisation_has_unit_variance(engine, full_record) -> None:
    """The check that catches a sigma over the wrong window or grouping.

    All of those still produce a plausible column of numbers, and all show up
    here as a spread that is not one.
    """
    if not full_record:
        pytest.skip("no city has a full reference period yet")
    for city_id, spread in query(
        engine,
        f"""select city_id, stddev_samp(z_temperature_2m_mean) from {ANOMALIES}
            where z_temperature_2m_mean is not null and city_id = any(:c) group by 1""",
        c=full_record,
    ):
        assert 0.9 < float(spread) < 1.1, f"{city_id}: sd(Z) = {spread}"


# ---------------------------------------------------------------------------
# Both tails
# ---------------------------------------------------------------------------


def test_the_flag_is_on_the_absolute_value(engine, built) -> None:
    mismatched = query(
        engine,
        f"""select count(*) from {ANOMALIES}
            where z_temperature_2m_mean is not null
              and is_anomaly <> (abs(z_temperature_2m_mean) > {THRESHOLD})""",
    )[0][0]
    assert mismatched == 0


def test_cold_anomalies_exist_at_all(engine, built) -> None:
    """The half a warm-only project loses."""
    cold = query(
        engine, f"select count(*) from {ANOMALIES} where anomaly_direction = 'cold'"
    )[0][0]
    assert cold > 0


def test_both_tails_are_represented_in_every_direction_column(engine, built) -> None:
    directions = {
        r[0]
        for r in query(engine, f"select distinct anomaly_direction from {ANOMALIES}")
    }
    assert {"hot", "cold", "none"} <= directions


def test_the_direction_matches_the_sign(engine, built) -> None:
    """An inverted branch flags the right days and labels every one backwards.

    Every count-based test passes on that, and Moscow's January lands in the
    heatwave column.
    """
    wrong = query(
        engine,
        f"""select count(*) from {ANOMALIES}
            where (anomaly_direction = 'hot'  and z_temperature_2m_mean <= 0)
               or (anomaly_direction = 'cold' and z_temperature_2m_mean >= 0)
               or (anomaly_direction = 'hot'  and departure_c <= 0)
               or (anomaly_direction = 'cold' and departure_c >= 0)""",
    )[0][0]
    assert wrong == 0


def test_neither_tail_dominates_absurdly(engine, built) -> None:
    """Individual cities skew (Phoenix cold, Cairo hot) but not the whole set."""
    hot, cold = query(
        engine,
        f"""select count(*) filter (where anomaly_direction = 'hot'),
                   count(*) filter (where anomaly_direction = 'cold')
            from {ANOMALIES}""",
    )[0]
    assert hot and cold
    assert 0.25 < hot / (hot + cold) < 0.75, f"hot {hot}, cold {cold}"


# ---------------------------------------------------------------------------
# Unknown is not ordinary
# ---------------------------------------------------------------------------


def test_an_unscored_day_carries_null_flags(engine, built) -> None:
    """Calling it `none` would assert the day was ordinary on no evidence.

    It would also pad the denominator of every anomaly rate with days that
    could never have been flagged.
    """
    wrong = query(
        engine,
        f"""select count(*) from {ANOMALIES} where z_temperature_2m_mean is null
            and (is_anomaly is not null or anomaly_direction is not null)""",
    )[0][0]
    assert wrong == 0


def test_unscored_days_are_exactly_the_single_year_cities(engine, built) -> None:
    unscored = {
        r[0]
        for r in query(
            engine,
            f"select distinct city_id from {ANOMALIES} where z_temperature_2m_mean is null",
        )
    }
    for city_id in unscored:
        years = query(
            engine,
            f"""select count(distinct year) from {GOLD}.fact_weather_anomalies
                where city_id = :c""",
            c=city_id,
        )[0][0]
        assert years == 1, f"{city_id} has {years} years but is unscored"


# ---------------------------------------------------------------------------
# The rate, and the outliers
# ---------------------------------------------------------------------------


def test_the_overall_rate_is_plausible(engine, built) -> None:
    """1.24% for a normal distribution; real residuals have fatter tails."""
    rate = query(
        engine,
        f"""select count(*) filter (where is_anomaly)::numeric
                   / nullif(count(*) filter (where z_temperature_2m_mean is not null), 0)
            from {ANOMALIES}""",
    )[0][0]
    assert rate is not None
    assert 0.005 < float(rate) < 0.03, f"anomaly rate {float(rate):.2%}"


def test_every_full_record_city_is_in_range(engine, full_record) -> None:
    """The per-city comparison the ticket asks for, on comparable cities."""
    if not full_record:
        pytest.skip("no city has a full reference period yet")
    rates = dict(
        query(
            engine,
            f"""select city_id, count(*) filter (where is_anomaly)::numeric / count(*)
                from {ANOMALIES}
                where z_temperature_2m_mean is not null and city_id = any(:c)
                group by 1""",
            c=full_record,
        )
    )
    for city_id, rate in rates.items():
        assert 0.005 < float(rate) < 0.03, f"{city_id}: {float(rate):.2%}"


def test_a_high_rate_is_explained_by_a_small_baseline(engine, built) -> None:
    """Tokyo sits at 3.90% on a baseline of 45 rather than 455.

    Only four years have backfilled, so leave-one-out leaves three, and a sigma
    from 45 points is noisy enough to push the tail. That is a sample-size
    artefact rather than a data defect, and it resolves as the backfill
    completes, so it is asserted as an explanation rather than tolerated as an
    exception.
    """
    outliers = query(
        engine,
        f"""select city_id, avg(baseline_observations) as baseline,
                   count(*) filter (where is_anomaly)::numeric / count(*) as rate,
                   stddev_samp(z_temperature_2m_mean) as sd_of_z
            from {ANOMALIES} where z_temperature_2m_mean is not null
            group by 1 having count(*) filter (where is_anomaly)::numeric / count(*) > 0.03""",
    )
    for city_id, baseline, rate, spread in outliers:
        assert float(baseline) < 300, (
            f"{city_id} at {float(rate):.2%} has a full baseline of {baseline}"
        )
        assert float(spread) > 1.05, f"{city_id} over-disperses as expected"


def test_a_skewed_city_skews_the_direction_it_should(engine, full_record) -> None:
    """Cairo runs hot and Phoenix cold, and the residual skew says why.

    Desert heat has a radiative ceiling while cold outbreaks are sharp, so
    Phoenix's residuals are the only left-skewed ones in the set. A model that
    flagged Phoenix mostly hot would be the surprising result, not this.
    """
    skews = dict(
        query(
            engine,
            f"""select city_id, avg(power(z_temperature_2m_mean, 3)) from {ANOMALIES}
                where z_temperature_2m_mean is not null and city_id = any(:c) group by 1""",
            c=full_record,
        )
    )
    balance = dict(
        query(
            engine,
            f"""select city_id,
                       count(*) filter (where anomaly_direction = 'hot')::numeric
                       / nullif(count(*) filter (where is_anomaly), 0)
                from {ANOMALIES} where city_id = any(:c) group by 1""",
            c=full_record,
        )
    )
    for city_id, skew in skews.items():
        share_hot = balance.get(city_id)
        if share_hot is None or skew is None:
            continue
        if abs(float(skew)) > 0.3:
            expected_hot = float(skew) > 0
            assert (float(share_hot) > 0.5) == expected_hot, (
                f"{city_id}: skew {float(skew):+.2f}, {float(share_hot):.0%} hot"
            )


def test_a_warming_trend_runs_through_the_record(engine, full_record) -> None:
    """Positive corr(year, Z) everywhere, +0.05 to +0.38.

    Real signal, not artefact, but it means the flag conflates "unusual for
    this day of year" with "warmer than the thirty-year mean because the
    climate has warmed". ML-05 will need to make that distinction deliberately,
    so it is recorded here rather than discovered there.
    """
    if not full_record:
        pytest.skip("no city has a full reference period yet")
    trends = dict(
        query(
            engine,
            f"""select city_id, corr(year::float, z_temperature_2m_mean) from {ANOMALIES}
                where z_temperature_2m_mean is not null and city_id = any(:c) group by 1""",
            c=full_record,
        )
    )
    assert all(t is not None and t > 0 for t in trends.values()), trends


# ---------------------------------------------------------------------------
# Moscow, the cold branch's strongest case
# ---------------------------------------------------------------------------


def test_moscow_has_cold_anomalies(engine, built) -> None:
    """Skips rather than passes while Moscow is still backfilling.

    A check that silently passes on absent data is worse than one that says it
    is waiting, and this is the specific check the ticket names, so a false
    green here would be the worst kind.
    """
    landed = query(
        engine,
        f"""select count(*), count(distinct year) from {ANOMALIES}
            where city_id = 'moscow' and z_temperature_2m_mean is not null""",
    )[0]
    rows, years = landed
    if rows == 0:
        pytest.skip("moscow has not backfilled yet; cannot check the cold branch")
    if years < 5:
        pytest.skip(f"moscow has only {years} reference years; baseline too thin")

    cold, hot = query(
        engine,
        f"""select count(*) filter (where anomaly_direction = 'cold'),
                   count(*) filter (where anomaly_direction = 'hot')
            from {ANOMALIES} where city_id = 'moscow'""",
    )[0]
    assert cold > 0, "Moscow must show cold anomalies"
    assert hot > 0, "and hot ones too; a one-sided result means a broken branch"


def test_moscow_is_the_most_variable_once_it_lands(engine, full_record) -> None:
    if "moscow" not in full_record:
        pytest.skip("moscow has not finished backfilling")
    sigmas = dict(
        query(
            engine,
            f"""select city_id, avg(stddev_temperature_2m_mean) from {ANOMALIES}
                where city_id = any(:c) group by 1""",
            c=full_record,
        )
    )
    assert max(sigmas, key=sigmas.get) == "moscow", sigmas


# ---------------------------------------------------------------------------
# The macro is where the logic lives
# ---------------------------------------------------------------------------


def test_the_threshold_is_configurable() -> None:
    model = (DBT_DIR / "models" / "marts" / "fact_weather_anomalies.sql").read_text(
        encoding="utf-8"
    )
    assert "anomaly_z_threshold" in model


def test_the_direction_macro_handles_null_before_comparing() -> None:
    """A null z must yield null, not fall through to `none`."""
    macro = (DBT_DIR / "macros" / "anomaly.sql").read_text(encoding="utf-8")
    body = macro.split("anomaly_direction")[-1]
    assert "is null then null" in body
    assert body.index("is null") < body.index("'hot'")
