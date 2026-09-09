"""Tests for the short-horizon gust nowcast.

The defect this file exists to catch is a target that reaches backwards. Every
feature is a trailing window ending at hour *t* and the target is a forward
window starting at *t+1*; the target is built by reversing the series, taking a
trailing maximum and reversing back, which is three chances to be off by one in
a direction that would make the model look excellent and mean nothing.

A leak of one hour at a 72-hour horizon moves no metric enough to notice. It is
checked as arithmetic on a series whose answer is known by construction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("xgboost")

from machine_learning.features import (  # noqa: E402
    NOWCAST_WINDOWS,
    FeatureError,
    build_hourly_features,
    hourly_feature_columns,
)
from machine_learning.nowcast import (  # noqa: E402
    SPLIT_BOUNDARIES,
    baseline_frame,
    climatology,
    purged_split,
    skill,
)

HOURS = 400


def hourly(gusts: list[float] | None = None, city: str = "test") -> pd.DataFrame:
    """One city, consecutive hours, every required column present."""
    count = len(gusts) if gusts is not None else HOURS
    rng = np.random.default_rng(4)
    return pd.DataFrame(
        {
            "city_id": city,
            "observation_hour": pd.date_range(
                "2024-09-02", periods=count, freq="h", tz="UTC"
            ),
            "pressure_msl": 1013 + rng.normal(0, 5, count),
            "pressure_tendency_3h": rng.normal(0, 1, count),
            "pressure_tendency_24h": rng.normal(0, 3, count),
            "wind_speed_10m": rng.uniform(5, 30, count),
            "wind_gusts_10m": (
                np.asarray(gusts, dtype=float)
                if gusts is not None
                else rng.uniform(10, 60, count)
            ),
            "wind_direction_10m": rng.integers(0, 360, count),
            "temperature_2m": rng.normal(15, 5, count),
            "dew_point_2m": rng.normal(8, 4, count),
            "relative_humidity_2m": rng.integers(30, 100, count),
            "precipitation": rng.uniform(0, 2, count),
            "cloud_cover": rng.integers(0, 100, count),
        }
    )


# ---------------------------------------------------------------------------
# The boundary between what is known and what is being predicted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("horizon", [24, 48, 72])
def test_the_target_is_exactly_the_next_n_hours(horizon):
    """`peak_gust_ahead` at t is max(gusts) over t+1 .. t+horizon.

    Checked against a directly computed maximum rather than against another
    rolling expression, so a shared off-by-one cannot satisfy both sides.
    """
    frame = hourly()
    built = build_hourly_features(frame, horizon)
    gusts = frame["wind_gusts_10m"].to_numpy()
    hours = frame["observation_hour"]

    for row in built.sample(30, random_state=1).itertuples():
        origin = int(hours.searchsorted(row.observation_hour))
        expected = gusts[origin + 1 : origin + 1 + horizon].max()
        assert row.peak_gust_ahead == pytest.approx(expected), (
            f"origin {origin}: target {row.peak_gust_ahead} != {expected}"
        )


@pytest.mark.parametrize("horizon", [24, 72])
def test_hour_t_is_a_feature_and_not_part_of_its_own_answer(horizon):
    """Change hour t+1 and the target moves; change hour t and it does not.

    The sharpest statement of the boundary. A target built with `shift(0)`
    instead of `shift(-1)` would include the origin hour, which is a feature,
    and the model would be handed a component of its own answer.
    """
    base = hourly([20.0] * HOURS)
    built = build_hourly_features(base, horizon)
    origin = 200
    target_before = built.loc[built.index[0] + 0, "peak_gust_ahead"]
    assert target_before is not None  # the frame is non-empty

    spike_at_origin = base.copy()
    spike_at_origin.loc[origin, "wind_gusts_10m"] = 200.0
    at_origin = build_hourly_features(spike_at_origin, horizon)

    spike_after = base.copy()
    spike_after.loc[origin + 1, "wind_gusts_10m"] = 200.0
    after = build_hourly_features(spike_after, horizon)

    row_at = at_origin.loc[at_origin["observation_hour"] == base.loc[origin, "observation_hour"]]
    row_after = after.loc[after["observation_hour"] == base.loc[origin, "observation_hour"]]

    assert float(row_at["peak_gust_ahead"].iloc[0]) == 20.0, (
        "a spike at hour t leaked into t's own target"
    )
    assert float(row_after["peak_gust_ahead"].iloc[0]) == 200.0, (
        "a spike at hour t+1 is missing from t's target"
    )


@pytest.mark.parametrize("horizon", [24, 72])
def test_no_feature_can_see_past_the_origin(horizon):
    """Spiking a future hour must move the target and no feature at all.

    Every input is a trailing window ending at t. This asserts it from the data
    rather than from reading the code: perturb the future, and if any feature
    column changes, some window is looking forward.
    """
    base = hourly()
    origin = 200
    moved = base.copy()
    moved.loc[origin + 1 :, "wind_gusts_10m"] += 25.0
    moved.loc[origin + 1 :, "pressure_msl"] -= 15.0

    stamp = base.loc[origin, "observation_hour"]
    before = build_hourly_features(base, horizon)
    after = build_hourly_features(moved, horizon)
    row_before = before.loc[before["observation_hour"] == stamp].iloc[0]
    row_after = after.loc[after["observation_hour"] == stamp].iloc[0]

    for column in hourly_feature_columns():
        assert row_before[column] == pytest.approx(row_after[column]), (
            f"{column} changed when only future hours moved"
        )
    assert row_before["peak_gust_ahead"] != row_after["peak_gust_ahead"]


@pytest.mark.parametrize("horizon", [24, 48, 72])
def test_persistence_spans_the_same_length_as_the_target(horizon):
    """The baseline must answer the same question, backwards.

    A persistence baseline over a different span is not a weaker opponent, it
    is a different one, and the skill number computed against it would not mean
    what the card says it means.
    """
    frame = hourly()
    built = build_hourly_features(frame, horizon)
    gusts = frame["wind_gusts_10m"].to_numpy()
    hours = frame["observation_hour"]

    for row in built.sample(20, random_state=2).itertuples():
        origin = int(hours.searchsorted(row.observation_hour))
        expected = gusts[origin - horizon + 1 : origin + 1].max()
        assert row.persistence_gust == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Encodings
# ---------------------------------------------------------------------------


def test_wind_direction_is_circular():
    """359 degrees and 1 degree are two degrees apart, not 358.

    Fed as a raw bearing, a tree learns a split at north that has no physical
    meaning. The failure costs a little accuracy, breaks no other test, and is
    invisible in a feature importance table -- which is why it is asserted.
    """
    frame = hourly()
    frame.loc[100, "wind_direction_10m"] = 359
    frame.loc[101, "wind_direction_10m"] = 1
    frame.loc[102, "wind_direction_10m"] = 180

    built = build_hourly_features(frame, 24).set_index("observation_hour")
    stamps = frame.set_index(frame.index)["observation_hour"]
    north_a = built.loc[stamps[100]]
    north_b = built.loc[stamps[101]]
    south = built.loc[stamps[102]]

    def distance(left, right) -> float:
        return float(
            np.hypot(
                left["wind_direction_sin"] - right["wind_direction_sin"],
                left["wind_direction_cos"] - right["wind_direction_cos"],
            )
        )

    assert distance(north_a, north_b) < 0.1
    assert distance(north_a, south) > 1.9


def test_the_rolling_windows_need_their_full_history():
    """A 72-hour mean over eleven hours is not a 72-hour mean.

    `min_periods` equals the window, so partial windows are null and those rows
    are dropped rather than being filled with a statistic computed over
    whatever happened to be there.
    """
    frame = hourly()
    built = build_hourly_features(frame, 24)
    first_kept = built["observation_hour"].min()
    warmup = frame["observation_hour"].iloc[max(NOWCAST_WINDOWS) - 2]

    assert first_kept > warmup


def test_a_frame_missing_a_column_is_refused():
    """Silently absent features are worse than a crash: the matrix still fits."""
    frame = hourly().drop(columns=["pressure_tendency_3h"])
    with pytest.raises(FeatureError, match="pressure_tendency_3h"):
        build_hourly_features(frame, 24)


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("horizon", [24, 72])
def test_the_purge_removes_every_origin_whose_answer_crosses_a_fold(horizon):
    """No training row's target may reach into validation, and so on.

    The leak is small, real, and flatters exactly the rows used for early
    stopping. Asserted as the property rather than as a row count: the last
    training origin plus its horizon must not reach the first validation hour.
    """
    frame = build_hourly_features(
        hourly(list(np.linspace(10, 60, 20000))), horizon
    )
    split = purged_split(frame, horizon)
    gap = pd.Timedelta(hours=horizon)

    first, second = (pd.Timestamp(bound, tz="UTC") for bound in SPLIT_BOUNDARIES)
    assert split.train["observation_hour"].max() + gap < first
    assert split.validation["observation_hour"].max() + gap < second
    assert split.validation["observation_hour"].min() >= first
    assert split.test["observation_hour"].min() >= second


def test_the_folds_do_not_overlap():
    """Chronological, and disjoint. A row in two folds is a row scored twice."""
    frame = build_hourly_features(hourly(list(np.linspace(10, 60, 20000))), 24)
    split = purged_split(frame, 24)

    stamps = [set(part["observation_hour"]) for part in
              (split.train, split.validation, split.test)]
    assert not stamps[0] & stamps[1]
    assert not stamps[1] & stamps[2]
    assert not stamps[0] & stamps[2]


# ---------------------------------------------------------------------------
# The baselines
# ---------------------------------------------------------------------------


def test_the_climatology_never_sees_the_test_period():
    """A baseline fitted on everything is a baseline that cheated.

    A weak baseline built from the whole record flatters the model twice: it is
    wrong, and it is unfairly informed. Asserted by moving the test rows a long
    way and checking the baseline does not follow.
    """
    frame = build_hourly_features(hourly(list(np.linspace(10, 60, 20000))), 24)
    split = purged_split(frame, 24)

    by_city, _ = climatology(split.train)
    inflated = frame.copy()
    test_rows = inflated["observation_hour"] >= pd.Timestamp(
        SPLIT_BOUNDARIES[1], tz="UTC"
    )
    inflated.loc[test_rows, "peak_gust_ahead"] += 500.0

    by_city_after, _ = climatology(purged_split(inflated, 24).train)
    pd.testing.assert_series_equal(by_city, by_city_after)


def test_a_city_month_absent_from_training_falls_back_to_that_city():
    """Not to the global mean, which would compare cities against each other.

    Two years means some city-months are thin, and a fallback that reached for
    the pooled average would answer "how windy is a city on average" with a
    number about fifteen different places.
    """
    frame = build_hourly_features(hourly(list(np.linspace(10, 60, 20000))), 24)
    split = purged_split(frame, 24)
    train = split.train.copy()
    # Remove one month from training entirely, so the test rows in it have no
    # seasonal baseline to look up.
    absent = int(split.test["month"].iloc[0])
    train = train.loc[train["month"] != absent]

    scored = baseline_frame(split.test, train)
    by_city, _ = climatology(train)
    orphans = scored.loc[scored["month"] == absent]

    assert not orphans.empty
    assert orphans["baseline_climatology_month"].notna().all()
    assert (
        orphans["baseline_climatology_month"]
        == orphans["city_id"].map(by_city)
    ).all()


def test_skill_is_a_fractional_error_reduction():
    """Positive is better, zero is level, negative is worse."""
    assert skill(8.0, 10.0) == pytest.approx(0.2)
    assert skill(10.0, 10.0) == pytest.approx(0.0)
    assert skill(12.0, 10.0) == pytest.approx(-0.2)
    assert np.isnan(skill(8.0, 0.0))
