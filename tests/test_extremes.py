"""Tests for the peaks-over-threshold fit and the return periods it implies.

The failures worth catching here are quiet ones. A return period computed from
an undeclustered exceedance rate is too short by the mean cluster length --
about a factor of two on this data -- and looks like an interesting finding
rather than a bug. A shape parameter reported without its interval implies the
tail is bounded when the data cannot say. A single time trend fitted to folded
``abs(Z)`` reports a change in the *composition* of the tail as a change in its
*width*, and does it with a small p-value.

Each of those has a test below that would fail if the guard were removed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("scipy")

from scipy.stats import genpareto  # noqa: E402

from machine_learning.extremes import (  # noqa: E402
    MIN_CLUSTERS,
    POT_QUANTILE,
    RUN_SEPARATION_DAYS,
    build_extremes,
    decluster,
    fit_city,
    fit_directional_trend,
    fit_nonstationary,
    moving_threshold,
    fit_stationary,
    return_level,
    return_period,
)


def series(z: list[float], start: str = "2000-01-01") -> pd.DataFrame:
    """Consecutive daily Z-scores, one city."""
    return pd.DataFrame(
        {
            "city_id": "test",
            "date_key": pd.date_range(start, periods=len(z), freq="D"),
            "z_temperature_2m_mean": z,
        }
    )


# ---------------------------------------------------------------------------
# Declustering
# ---------------------------------------------------------------------------


def test_a_run_of_exceedances_is_one_event():
    """Five consecutive hot days are one heatwave, and the peak represents it.

    This is the whole reason the module exists. Without it the exceedance rate
    is five-fold here, and every return period computed from it is divided by
    five.
    """
    frame = series([0, 0, 3.0, 3.5, 3.2, 4.0, 3.1, 0, 0])
    peaks = decluster(frame, 2.5, separation=3)

    assert len(peaks) == 1
    assert peaks.iloc[0]["z_temperature_2m_mean"] == 4.0


def test_a_quiet_gap_longer_than_the_separation_splits_the_event():
    """Two heatwaves a fortnight apart are two events, not one.

    The mirror of the test above, and the reason the separation is a parameter:
    a rule that merged everything would understate the rate as badly as no rule
    overstates it.
    """
    quiet = [0.0] * 10
    frame = series([3.0, 3.5, *quiet, 3.2, 3.9])
    peaks = decluster(frame, 2.5, separation=3)

    assert len(peaks) == 2
    assert list(peaks["z_temperature_2m_mean"]) == [3.5, 3.9]


def test_the_gap_is_measured_in_days_and_not_in_rows():
    """A record with a hole in it must not merge across the hole.

    Rows are not days. If the frame is missing a fortnight of observations, two
    exceedances on either side of the gap are adjacent *rows* and distant
    *days*, and a `diff()` on the index rather than on the date would call them
    one event.
    """
    frame = pd.DataFrame(
        {
            "city_id": "test",
            "date_key": pd.to_datetime(["2000-01-01", "2000-02-01"]),
            "z_temperature_2m_mean": [3.0, 3.5],
        }
    )
    assert len(decluster(frame, 2.5, separation=3)) == 2


def test_declustering_never_invents_events():
    """The peaks are a subset of the exceedances, on real data's shape.

    Stated as a property rather than a count because it is the invariant the
    warehouse constraint also enforces: clusters <= exceedances <= observations.
    """
    rng = np.random.default_rng(0)
    frame = series(list(rng.normal(size=4000)))
    threshold = float(frame["z_temperature_2m_mean"].quantile(0.95))
    peaks = decluster(frame, threshold, separation=RUN_SEPARATION_DAYS)
    raw = int((frame["z_temperature_2m_mean"] > threshold).sum())

    assert 0 < len(peaks) <= raw
    assert set(peaks.index).issubset(set(frame.index))


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [-0.2, 0.0, 0.15])
def test_the_fit_recovers_a_tail_it_was_given(shape):
    """Simulate from a known generalised Pareto, fit it back.

    A maximum likelihood routine that returned plausible-looking numbers for
    the wrong distribution would pass every other test in this file, because
    every other test checks a relationship between the parameters rather than
    the parameters themselves.
    """
    scale = 0.5
    sample = genpareto.rvs(
        shape, loc=0.0, scale=scale, size=4000, random_state=11
    )
    fitted_shape, fitted_scale, _ = fit_stationary(sample)

    assert fitted_shape == pytest.approx(shape, abs=0.05)
    assert fitted_scale == pytest.approx(scale, abs=0.05)


def test_a_level_and_its_period_are_inverses():
    """`return_level` and `return_period` must round-trip.

    They are written separately -- one solves for the level, one for the
    probability -- and a sign error in either would be invisible on its own.
    """
    terms = {
        "threshold": 2.0,
        "shape": -0.08,
        "scale": 0.45,
        "exceedance_rate": 265 / 11569,
    }
    for years in (2.0, 10.0, 50.0):
        level = return_level(years, **terms)
        assert return_period(level, **terms) == pytest.approx(years, rel=1e-6)


def test_a_bounded_tail_has_no_period_beyond_its_endpoint():
    """Past where a negative-shape tail stops, the answer is `inf`, not a number.

    "One in forty thousand years" about a day the fitted model calls impossible
    would be worse than saying nothing, and it is the number a direct
    transcription of the formula produces.
    """
    terms = {"threshold": 2.0, "shape": -0.25, "scale": 0.5,
             "exceedance_rate": 0.02}
    endpoint = 2.0 + 0.5 / 0.25

    assert np.isfinite(return_period(endpoint - 0.1, **terms))
    assert return_period(endpoint + 0.1, **terms) == float("inf")


def test_the_period_uses_the_declustered_rate():
    """Halving the rate must double the period, and this is the failure mode.

    Passing the raw exceedance fraction instead of the declustered one is the
    single error that would make every number in the mart wrong in the same
    direction, and it would still typecheck, still be finite, and still look
    reasonable. Pinned as arithmetic.
    """
    terms = {"threshold": 2.0, "shape": 0.0, "scale": 0.5}
    slow = return_period(3.5, exceedance_rate=0.01, **terms)
    fast = return_period(3.5, exceedance_rate=0.02, **terms)

    assert slow == pytest.approx(2 * fast)


# ---------------------------------------------------------------------------
# The intervals
# ---------------------------------------------------------------------------


def test_the_interval_brackets_the_point_estimate():
    """A bootstrap interval that excludes its own point estimate is a bug."""
    rng = np.random.default_rng(3)
    z = np.concatenate([rng.normal(size=6000), rng.normal(3.2, 0.6, size=200)])
    tail = fit_city(series(list(z)), "test", samples=200)

    assert tail is not None
    low, high = tail.shape_interval
    assert low <= tail.shape <= high
    low, high = tail.scale_interval
    assert low <= tail.scale <= high


def test_bounded_means_the_whole_interval_and_not_the_estimate():
    """The claim "this tail is bounded" is about the interval.

    A negative point estimate whose interval crosses zero is not evidence of a
    hottest possible day, and seven of the eleven fitted cities are in exactly
    that position. Collapsing this to `shape < 0` is the simplification the
    warehouse constraint also refuses.
    """
    rng = np.random.default_rng(5)
    z = np.concatenate([rng.normal(size=6000), rng.normal(3.2, 0.6, size=200)])
    tail = fit_city(series(list(z)), "test", samples=200)

    assert tail is not None
    row = tail.as_dict()
    assert row["tail_is_bounded"] == (tail.shape_interval[1] < 0)
    if tail.shape < 0 <= tail.shape_interval[1]:
        assert not row["tail_is_bounded"]


# ---------------------------------------------------------------------------
# The trend, and why it is fitted per direction
# ---------------------------------------------------------------------------


def _two_sided_record(warm_growth: float, cold_growth: float) -> pd.DataFrame:
    """A record whose warm tail widens and whose cold tail narrows.

    Built so the two directions move in *opposite* ways by construction: the
    warm departures are scaled up over time and the cold ones scaled down.
    Anything that folds the two together is handed a mixture whose composition
    inverts, which is the situation the real data is in.
    """
    rng = np.random.default_rng(17)
    days = 11000
    years = np.arange(days) / 365.25
    base = rng.normal(size=days)
    scaled = np.where(
        base > 0,
        base * np.exp(warm_growth * years),
        base * np.exp(-cold_growth * years),
    )
    return series(list(scaled), start="1996-01-01")


def _drifting_record(drift: float = 0.02) -> pd.DataFrame:
    """A record that only *moves*: a linear location trend, constant variance.

    This is the shape of the real problem. Z is referenced to a baseline
    computed over the whole record, so a warming city sits below its own
    baseline early and above it late -- the departures slide without the tail
    changing width at all.

    Every trend fitted to this record should come back as drift and nothing
    else. A method that reports a width change here is reporting the movement
    twice under two names.
    """
    rng = np.random.default_rng(23)
    days = 11000
    years = np.arange(days) / 365.25
    return series(list(rng.normal(size=days) + drift * years), start="1996-01-01")


def test_the_two_directions_are_reported_separately():
    """A widening warm tail and a narrowing cold one come out as both.

    The folded fit cannot report both -- it has one scale parameter -- so
    whatever it says is a weighted average of two opposite truths, and it says
    it with a p-value.
    """
    frame = _two_sided_record(warm_growth=0.05, cold_growth=0.05)

    warm = fit_directional_trend(frame, "warm")
    cold = fit_directional_trend(frame, "cold")

    assert warm["converged"] and cold["converged"]
    assert warm["scale_trend_per_year"] > 0
    assert cold["scale_trend_per_year"] < 0
    assert warm["p_value"] < 0.01
    assert cold["p_value"] < 0.01


def test_the_moving_threshold_recovers_the_drift_it_was_given():
    """The threshold's slope is the location trend, and is estimated as one.

    Pinned against an injected value rather than a relationship, because this
    number is reported to a reader as "how fast this city's extremes are
    moving" and an estimator that was merely monotone in the truth would
    satisfy every other test here.
    """
    frame = _drifting_record(drift=0.02)
    years = (
        frame["date_key"] - frame["date_key"].min()
    ).dt.days.to_numpy(dtype=float) / 365.25

    _, slope = moving_threshold(
        years, frame["z_temperature_2m_mean"].to_numpy(dtype=float)
    )
    assert slope == pytest.approx(0.02, abs=0.004)


def test_a_record_that_only_moves_is_not_reported_as_one_that_widened():
    """The test this module's design exists to pass.

    `_drifting_record` has a rigorously constant variance: nothing about its
    tail gets wider. Against a *fixed* threshold the cold-side fit calls it
    significantly narrowing at p = 0.002, because the distribution slides out
    of the region above a bar that never moved. With the threshold fitted as a
    line in time, the movement lands in the threshold slope where it belongs
    and neither width trend is significant.

    Both halves are asserted. Dropping the second would let a future change
    quietly reintroduce the confound while still passing a test named for it.
    """
    frame = _drifting_record(drift=0.02)

    for direction in ("warm", "cold"):
        fitted = fit_directional_trend(frame, direction)
        expected = 0.02 if direction == "warm" else -0.02
        assert fitted["threshold_slope_per_year"] == pytest.approx(
            expected, abs=0.005
        ), f"{direction}: the drift did not land in the threshold"
        assert fitted["p_value"] > 0.05, (
            f"{direction}: a constant-width record was reported as widening"
        )


def test_a_fixed_threshold_would_have_got_that_wrong():
    """The failure the moving threshold removes, pinned so it stays removed.

    Fits the same drifting record the old way -- a threshold at the marginal
    quantile, held still -- and asserts it reaches the wrong conclusion. If
    someone reverts `fit_directional_trend` to a fixed bar, the test above goes
    red and this one goes green, and the pair says exactly what changed.
    """
    frame = _drifting_record(drift=0.02)
    cold = frame.assign(z=-frame["z_temperature_2m_mean"])
    threshold = float(cold["z"].quantile(POT_QUANTILE))
    peaks = decluster(cold, threshold, column="z")
    years = (
        peaks["date_key"] - peaks["date_key"].min()
    ).dt.days.to_numpy(dtype=float) / 365.25

    fixed = fit_nonstationary((peaks["z"] - threshold).to_numpy(), years)

    assert fixed["p_value"] < 0.05
    assert fixed["scale_trend_per_year"] < 0


def test_the_composition_shift_is_recorded():
    """The warm share of exceedances, early half against late.

    The evidence for splitting the trend, carried on the row beside the trends
    themselves so a reader meeting two disagreeing numbers can see why one
    would have been neither.
    """
    tail = fit_city(_drifting_record(drift=0.03), "test", samples=100)

    assert tail is not None
    assert tail.warm_share_late > tail.warm_share_early


def test_a_direction_must_be_named():
    with pytest.raises(ValueError, match="warm.*cold"):
        fit_directional_trend(series([0.0] * 100), "up")


# ---------------------------------------------------------------------------
# Refusing to fit
# ---------------------------------------------------------------------------


def test_a_short_record_is_skipped_by_a_rule_and_not_by_name():
    """Too little tail means no fit, and the reason is stated.

    Sydney has eighteen scored days. It is excluded because it has fewer than
    `MIN_CLUSTERS` declustered exceedances, which is a rule that will also
    exclude the next short city without anyone editing a list.
    """
    rng = np.random.default_rng(7)
    short = series(list(rng.normal(size=200)))
    assert fit_city(short, "short") is None

    fits, skipped = build_extremes(short.assign(city_id="short"), samples=50)
    assert fits.empty
    assert "short" in skipped
    assert str(MIN_CLUSTERS) in skipped["short"]


def test_a_period_shorter_than_the_spacing_between_events_is_not_answered():
    """Below the threshold the GPD has nothing to say, and says so.

    `return_level` for a period so short that fewer than one event is expected
    would extrapolate the tail model back under its own threshold, where it was
    never fitted.
    """
    level = return_level(
        0.01, threshold=2.0, shape=0.0, scale=0.5, exceedance_rate=0.02
    )
    assert np.isnan(level)


# ---------------------------------------------------------------------------
# The warehouse
# ---------------------------------------------------------------------------

EXTREMES = "gold_marts.fact_extreme_value"
PERIODS = "gold_marts.fact_anomaly_return_periods"


@pytest.fixture
def fitted(engine):
    """The fitted tails, or skip. Requires `extremes.py --write` to have run."""
    sqlalchemy = pytest.importorskip("sqlalchemy")
    with engine.connect() as connection:
        try:
            frame = pd.read_sql(
                sqlalchemy.text(f"select * from {EXTREMES} order by city_id"),
                connection,
            )
        except Exception:  # noqa: BLE001 - an absent mart is a skip, not a failure
            pytest.skip(f"{EXTREMES} not built")
    if frame.empty:
        pytest.skip(f"{EXTREMES} is empty")
    return frame


def test_the_sql_and_the_python_agree_on_the_same_formula(engine, fitted):
    """The exponent is written twice, so it is checked against itself.

    `return_period` lives in Python for the fit and in a dbt macro for the
    mart, because the fit is maximum likelihood and the mart is a join. Two
    implementations of one formula is a standing invitation to drift: a
    tolerance widened on one side, a sign fixed on one side, a near-zero shape
    handled on one side. Nothing else in the project would notice.

    Compared on the warehouse's own rows rather than on invented ones, so the
    shapes, scales and magnitudes are the ones actually in service.
    """
    sqlalchemy = pytest.importorskip("sqlalchemy")
    with engine.connect() as connection:
        try:
            rows = pd.read_sql(
                sqlalchemy.text(
                    f"""
                    select city_id, abs_z, tail_threshold, tail_shape,
                           tail_scale, tail_exceedance_rate, return_period_years
                      from {PERIODS}
                     where return_period_years is not null
                     order by abs_z desc
                     limit 500
                    """
                ),
                connection,
            )
        except Exception:  # noqa: BLE001
            pytest.skip(f"{PERIODS} not built")
    if rows.empty:
        pytest.skip(f"{PERIODS} is empty")

    for row in rows.itertuples():
        expected = return_period(
            float(row.abs_z),
            threshold=float(row.tail_threshold),
            shape=float(row.tail_shape),
            scale=float(row.tail_scale),
            exceedance_rate=float(row.tail_exceedance_rate),
        )
        assert float(row.return_period_years) == pytest.approx(expected, rel=1e-6), (
            f"{row.city_id} at |Z| {row.abs_z:.2f}: "
            f"SQL says {row.return_period_years:.3f} years, Python says "
            f"{expected:.3f}"
        )


def test_the_stored_rate_is_the_declustered_one(fitted):
    """Every fitted city's rate must be below its raw exceedance fraction.

    The single error that would make every number in the mart wrong in the same
    direction while remaining finite and plausible. Also enforced as a check
    constraint on the table; asserted here too because a constraint added later
    would not have caught a row written earlier.
    """
    raw = fitted["exceedances"] / fitted["observations"]
    assert (fitted["clusters"] <= fitted["exceedances"]).all()
    assert (fitted["exceedance_rate"] <= raw + 1e-12).all()
    # And it must actually be doing something: weather persists, so a rate
    # identical to the raw fraction everywhere would mean declustering ran and
    # collapsed nothing.
    assert (fitted["mean_cluster_days"] > 1.5).all()


def test_a_bounded_tail_is_claimed_only_on_the_whole_interval(fitted):
    """`tail_is_bounded` follows the interval, not the point estimate.

    Seven of the eleven fitted cities have a negative or near-zero shape whose
    bootstrap interval crosses zero. Reading those as "bounded" would put a
    hottest-possible-day claim on the map for cities where the data does not
    support one.
    """
    bounded = fitted["shape_high"] < 0
    assert (fitted["tail_is_bounded"] == bounded).all()
    assert (fitted.loc[fitted["tail_is_bounded"], "shape"] < 0).all()
