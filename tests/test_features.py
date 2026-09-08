"""Tests for the feature matrix, and for the claim that it cannot see forward.

The claim is the whole point. A leaked feature does not fail a test that was
not written for it: it raises PR-AUC, passes review, and is found, if it is
found at all, by someone asking why the model is so good. So the central test here
does not inspect the formulas. It perturbs the *future*, rebuilds, and asserts
that every row at or before the cut is bit-identical. Any window that reaches
forward by a single day moves those rows, whatever it is called and however it
is written.

A test that cannot fail proves nothing, so one of these deliberately builds a
centred window and asserts the same check catches it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from machine_learning.features import (  # noqa: E402
    ANOMALY_COUNT_WINDOW,
    LAGS,
    PASSTHROUGH_COLUMNS,
    PRESSURE,
    ROLLING_WINDOWS,
    TEMPERATURE,
    TRAILING_Z_WINDOW,
    WARMUP_DAYS,
    FeatureError,
    build_features,
    day_of_year_common,
    drop_warmup,
    feature_columns,
    load_features,
    missing_report,
)

GOLD = "gold_marts"
TRAILING_Z = f"{TEMPERATURE}_z_trailing{TRAILING_Z_WINDOW}"
ANOMALY_COUNT = f"anomaly_days_trailing{ANOMALY_COUNT_WINDOW}"
ANOMALY_SCORED = f"anomaly_days_scored{ANOMALY_COUNT_WINDOW}"


def synthetic(
    city_id: str = "alpha",
    *,
    days: int = 200,
    start: str = "2001-01-01",
    seed: int = 11,
    anomaly_every: int = 13,
) -> pd.DataFrame:
    """A gold-shaped frame with a seasonal signal and noise.

    Deliberately not a ramp: on a monotone series a window that reaches one day
    forward still produces plausible numbers, and an equality check against a
    smooth function can pass on the wrong window by coincidence.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    season = 12 * np.sin(2 * np.pi * np.arange(days) / 365.0)
    return pd.DataFrame(
        {
            "city_id": city_id,
            "date_key": dates,
            TEMPERATURE: 15 + season + rng.normal(scale=3.0, size=days),
            PRESSURE: 1013 + rng.normal(scale=8.0, size=days),
            "z_temperature_2m_mean": rng.normal(size=days),
            "is_anomaly": pd.array(
                [index % anomaly_every == 0 for index in range(days)], dtype="boolean"
            ),
            "latitude": 51.5,
            "elevation_m": 11.0,
        }
    )


def future_perturbed(frame: pd.DataFrame, cut: pd.Timestamp) -> pd.DataFrame:
    """The same frame with everything after ``cut`` replaced by nonsense."""
    changed = frame.copy()
    future = changed["date_key"] > cut
    changed.loc[future, TEMPERATURE] += 40.0
    changed.loc[future, PRESSURE] -= 60.0
    changed.loc[future, "z_temperature_2m_mean"] *= -5.0
    changed.loc[future, "is_anomaly"] = ~changed.loc[future, "is_anomaly"]
    return changed


# --------------------------------------------------------------------------
# The leakage guarantee
# --------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [40, 90, 150, 199])
def test_no_feature_sees_the_future(offset: int) -> None:
    """Rewriting every day after t changes no row at or before t.

    The cut points span the series rather than sitting in the middle: a
    30-day window that peeks forward is invisible at a cut 150 days in if the
    comparison only looks at the first row.
    """
    frame = synthetic(days=200)
    cut = frame["date_key"].iloc[offset]

    baseline = build_features(frame)
    rebuilt = build_features(future_perturbed(frame, cut))

    past = baseline.loc[baseline["date_key"] <= cut].reset_index(drop=True)
    past_rebuilt = rebuilt.loc[rebuilt["date_key"] <= cut].reset_index(drop=True)

    assert len(past) == offset + 1
    pd.testing.assert_frame_equal(past, past_rebuilt)


def test_the_leakage_check_catches_a_forward_window() -> None:
    """The test above can fail. A centred window is what failing looks like.

    Without this, a future-perturbation test that compared the wrong rows,
    or compared nothing, would be indistinguishable from a clean matrix.
    """
    frame = synthetic(days=200)
    cut = frame["date_key"].iloc[90]

    def centred(source: pd.DataFrame) -> pd.Series:
        return (
            source.groupby("city_id")[TEMPERATURE]
            .transform(lambda s: s.rolling(7, center=True, min_periods=7).mean())
            .reset_index(drop=True)
        )

    honest = centred(frame)
    leaked = centred(future_perturbed(frame, cut))
    past = frame["date_key"].reset_index(drop=True) <= cut

    assert not honest[past].equals(leaked[past]), (
        "a centred window did not move any past row, so the comparison this "
        "test protects is not actually comparing anything"
    )


def test_one_city_cannot_move_another() -> None:
    """Windows are per city. Interleaved rows must not bleed across the group."""
    alpha = synthetic("alpha", days=120, seed=1)
    beta = synthetic("beta", days=120, seed=2)
    together = pd.concat([alpha, beta], ignore_index=True).sample(
        frac=1.0, random_state=3
    )

    baseline = build_features(together)
    beta_wrecked = together.copy()
    is_beta = beta_wrecked["city_id"] == "beta"
    beta_wrecked.loc[is_beta, TEMPERATURE] += 100.0
    beta_wrecked.loc[is_beta, PRESSURE] += 100.0
    rebuilt = build_features(beta_wrecked)

    for frame in (baseline, rebuilt):
        assert set(frame["city_id"]) == {"alpha", "beta"}
    pd.testing.assert_frame_equal(
        baseline.loc[baseline["city_id"] == "alpha"].reset_index(drop=True),
        rebuilt.loc[rebuilt["city_id"] == "alpha"].reset_index(drop=True),
    )


def test_the_label_source_is_not_a_model_input() -> None:
    """``is_anomaly`` at t is carried for ML-02, and must never be fed to it."""
    assert "is_anomaly" not in feature_columns()
    assert "is_anomaly" in PASSTHROUGH_COLUMNS
    # The coverage denominator is a fact about the backfill, not the weather.
    assert ANOMALY_SCORED not in feature_columns()
    assert ANOMALY_SCORED in PASSTHROUGH_COLUMNS


# --------------------------------------------------------------------------
# The windows compute what they say they compute
# --------------------------------------------------------------------------


def test_lags_reach_exactly_as_far_back_as_they_claim() -> None:
    frame = synthetic(days=60)
    built = build_features(frame)
    for lag in LAGS:
        expected = frame[TEMPERATURE].shift(lag)
        pd.testing.assert_series_equal(
            built[f"{TEMPERATURE}_lag{lag}"],
            expected.rename(f"{TEMPERATURE}_lag{lag}"),
            check_names=True,
        )


def test_rolling_windows_end_on_the_row_they_label() -> None:
    frame = synthetic(days=90)
    built = build_features(frame)
    for window in ROLLING_WINDOWS:
        for position in (window - 1, window, 60, 89):
            span = frame[TEMPERATURE].iloc[position - window + 1 : position + 1]
            assert len(span) == window
            rolled_mean = built[f"{TEMPERATURE}_roll{window}_mean"]
            rolled_var = built[f"{TEMPERATURE}_roll{window}_var"]
            assert rolled_mean.iloc[position] == pytest.approx(span.mean())
            assert rolled_var.iloc[position] == pytest.approx(span.var(ddof=1))


def test_a_partial_window_is_null_rather_than_a_shorter_average() -> None:
    """min_periods is the window. A 30-day mean over 11 days is a different
    statistic, and letting it into the same column makes the feature's meaning
    depend on how far into the series the row sits."""
    built = build_features(synthetic(days=90))
    for window in ROLLING_WINDOWS:
        column = built[f"{TEMPERATURE}_roll{window}_mean"]
        assert column.iloc[: window - 1].isna().all()
        assert column.iloc[window - 1 :].notna().all()


def test_the_trailing_z_does_not_score_a_day_against_itself() -> None:
    """The baseline is t-30 .. t-1, exclusive of t.

    Including t pulls the mean 1/30 of the way towards it and inflates σ by its
    own deviation, so an extreme day scores systematically closer to zero, the
    same self-labelling ``fact_climatology`` excludes a whole year to avoid.
    """
    frame = synthetic(days=90)
    built = build_features(frame)
    position = 60
    prior = frame[TEMPERATURE].iloc[position - TRAILING_Z_WINDOW : position]
    assert len(prior) == TRAILING_Z_WINDOW
    expected = (frame[TEMPERATURE].iloc[position] - prior.mean()) / prior.std(ddof=1)
    assert built[TRAILING_Z].iloc[position] == pytest.approx(expected)

    # And the self-inclusive version it is not.
    inclusive_span = frame[TEMPERATURE].iloc[
        position - TRAILING_Z_WINDOW + 1 : position + 1
    ]
    inclusive = (
        frame[TEMPERATURE].iloc[position] - inclusive_span.mean()
    ) / inclusive_span.std(ddof=1)
    assert built[TRAILING_Z].iloc[position] != pytest.approx(inclusive)


def test_pressure_tendency_spans_the_days_it_names() -> None:
    frame = synthetic(days=40)
    built = build_features(frame)
    assert built["pressure_tendency_24h"].iloc[10] == pytest.approx(
        frame[PRESSURE].iloc[10] - frame[PRESSURE].iloc[9]
    )
    assert built["pressure_tendency_72h"].iloc[10] == pytest.approx(
        frame[PRESSURE].iloc[10] - frame[PRESSURE].iloc[7]
    )


# --------------------------------------------------------------------------
# Calendar gaps: the row-window trap
# --------------------------------------------------------------------------


def test_a_gap_nulls_the_windows_that_span_it_rather_than_reaching_further() -> None:
    """``rolling(7)`` counts rows. Over a hole that is eight calendar days.

    This is the trap ``fact_weather_hourly`` avoids with a ``RANGE`` frame, at
    the daily grain: a missing day makes a row-counted window report an
    eight-day mean as a seven-day one, which is a fabricated number from data
    that merely had a hole.
    """
    frame = synthetic(days=90)
    hole = frame["date_key"].iloc[50]
    with_gap = frame.loc[frame["date_key"] != hole].reset_index(drop=True)

    built = build_features(with_gap)
    assert len(built) == len(with_gap)
    assert (built["date_key"] == hole).sum() == 0

    day_after = built.loc[built["date_key"] == hole + pd.Timedelta(days=1)].iloc[0]
    assert pd.isna(day_after[f"{TEMPERATURE}_roll7_mean"])
    assert pd.isna(day_after[f"{TEMPERATURE}_roll30_mean"])
    assert pd.isna(day_after[ANOMALY_COUNT])

    # Seven clear days later the window has cleared the hole again.
    recovered = built.loc[built["date_key"] == hole + pd.Timedelta(days=7)].iloc[0]
    assert not pd.isna(recovered[f"{TEMPERATURE}_roll7_mean"])


def test_a_lag_across_a_gap_is_null_not_the_previous_row() -> None:
    frame = synthetic(days=60)
    hole = frame["date_key"].iloc[30]
    with_gap = frame.loc[frame["date_key"] != hole].reset_index(drop=True)
    built = build_features(with_gap)

    day_after = built.loc[built["date_key"] == hole + pd.Timedelta(days=1)].iloc[0]
    assert pd.isna(day_after[f"{TEMPERATURE}_lag1"])
    assert pd.isna(day_after["pressure_tendency_24h"])
    # Three days on, t-3 lands in the hole and t-1 does not: the lag is null
    # rather than quietly returning the reading from four days back.
    later = built.loc[built["date_key"] == hole + pd.Timedelta(days=3)].iloc[0]
    assert not pd.isna(later[f"{TEMPERATURE}_lag1"])
    assert pd.isna(later[f"{TEMPERATURE}_lag3"])
    assert pd.isna(later["pressure_tendency_72h"])


def test_a_gap_shows_up_outside_the_warmup_flag() -> None:
    """``is_warmup`` and ``has_missing_feature`` answer different questions."""
    frame = synthetic(days=90)
    hole = frame["date_key"].iloc[60]
    built = build_features(frame.loc[frame["date_key"] != hole])
    day_after = built.loc[built["date_key"] == hole + pd.Timedelta(days=1)].iloc[0]
    assert not day_after["is_warmup"]
    assert day_after["has_missing_feature"]


# --------------------------------------------------------------------------
# Nulls at the start of a series
# --------------------------------------------------------------------------


def test_warmup_rows_are_flagged_and_kept() -> None:
    frame = synthetic(days=120)
    built = build_features(frame)
    assert len(built) == len(frame)
    assert int(built["is_warmup"].sum()) == WARMUP_DAYS
    assert built["history_days"].iloc[0] == 0
    assert not built["is_warmup"].iloc[WARMUP_DAYS]
    # The first row past the warm-up is complete, which is what WARMUP_DAYS
    # claims. If a longer window is ever added and this constant is not
    # updated, this is where it shows.
    assert not built["has_missing_feature"].iloc[WARMUP_DAYS]


def test_dropping_the_warmup_takes_saying_so() -> None:
    built = build_features(synthetic(days=120))
    trimmed = drop_warmup(built)
    assert len(trimmed) == len(built) - WARMUP_DAYS
    assert not trimmed["is_warmup"].any()
    assert list(trimmed.index) == list(range(len(trimmed)))


def test_the_missing_report_separates_arithmetic_from_data() -> None:
    frame = synthetic(days=120)
    hole = frame["date_key"].iloc[80]
    built = build_features(frame.loc[frame["date_key"] != hole])
    report = missing_report(built).set_index("feature")

    assert report.loc[f"{TEMPERATURE}_lag1", "null_warmup"] == 1
    assert report.loc[f"{TEMPERATURE}_lag1", "null_after_warmup"] == 1
    totals = report["null_warmup"] + report["null_after_warmup"]
    assert (report["null_total"] == totals).all()


# --------------------------------------------------------------------------
# Unknown is not "ordinary"
# --------------------------------------------------------------------------


def test_an_unscored_day_is_not_counted_as_a_quiet_one() -> None:
    """A null flag lowers the denominator, never the numerator.

    ``fact_weather_anomalies`` leaves 1 095 city-days with a null flag: the
    cities holding one reference year, where leave-one-year-out leaves nothing
    to score against. Folding those into "not an anomaly" would report a quiet
    month that was never actually measured.
    """
    frame = synthetic(days=90, anomaly_every=10)
    frame.loc[frame.index[40:50], "is_anomaly"] = pd.NA

    built = build_features(frame)
    row = built.iloc[60]
    window = frame.iloc[60 - ANOMALY_COUNT_WINDOW + 1 : 61]

    assert row[ANOMALY_COUNT] == pytest.approx(
        float(window["is_anomaly"].fillna(False).sum())
    )
    assert row[ANOMALY_SCORED] == pytest.approx(
        float(window["is_anomaly"].notna().sum())
    )
    assert row[ANOMALY_SCORED] < ANOMALY_COUNT_WINDOW

    # Clear of the unscored stretch, every day in the window is scored again.
    assert built.iloc[85][ANOMALY_SCORED] == ANOMALY_COUNT_WINDOW


def test_a_city_with_no_baseline_keeps_its_rows() -> None:
    """A null climatological Z is a null feature, not a dropped city-day."""
    frame = synthetic(days=90)
    frame["z_temperature_2m_mean"] = np.nan
    frame["is_anomaly"] = pd.array([pd.NA] * len(frame), dtype="boolean")

    built = build_features(frame)
    assert len(built) == len(frame)
    assert built["z_temperature_2m_mean"].isna().all()
    assert (built[ANOMALY_COUNT].dropna() == 0).all()
    assert (built[ANOMALY_SCORED].dropna() == 0).all()


# --------------------------------------------------------------------------
# The seasonal encoding
# --------------------------------------------------------------------------


def test_the_year_closes_on_itself() -> None:
    """31 December is one day from 1 January, and the encoding has to agree."""
    frame = synthetic(days=400, start="2001-06-01")
    built = build_features(frame).set_index("date_key")

    unit = built["day_of_year_sin"] ** 2 + built["day_of_year_cos"] ** 2
    assert unit.between(0.999999, 1.000001).all()

    def step(first: str, second: str) -> float:
        a = built.loc[pd.Timestamp(first), ["day_of_year_sin", "day_of_year_cos"]]
        b = built.loc[pd.Timestamp(second), ["day_of_year_sin", "day_of_year_cos"]]
        return float(np.hypot(*(a.to_numpy() - b.to_numpy())))

    year_end = step("2001-12-31", "2002-01-01")
    mid_year = step("2001-07-01", "2001-07-02")
    assert year_end == pytest.approx(mid_year, rel=1e-6)


def test_a_leap_year_does_not_shift_the_seasonal_phase() -> None:
    """Raw day-of-year would put 1 March at 60 in 2003 and 61 in 2004.

    That is a one-day phase shift in three years out of four, which a model
    reads as a real difference between leap and common years.
    """
    dates = pd.Series(pd.to_datetime(["2003-03-01", "2004-03-01", "2004-02-29"]))
    common = day_of_year_common(dates)
    assert common.iloc[0] == common.iloc[1] == 60
    assert common.iloc[2] == 59  # 29 February folds onto 28


# --------------------------------------------------------------------------
# The shape of the contract
# --------------------------------------------------------------------------


def test_the_matrix_carries_exactly_what_it_declares() -> None:
    built = build_features(synthetic(days=60))
    assert list(built.columns) == list(PASSTHROUGH_COLUMNS) + list(feature_columns())
    assert len(set(feature_columns())) == len(feature_columns())
    assert not set(feature_columns()) & set(PASSTHROUGH_COLUMNS)


def test_a_repeated_city_day_is_rejected() -> None:
    """A duplicate makes a seven-day window span six days without saying so."""
    frame = synthetic(days=40)
    doubled = pd.concat([frame, frame.iloc[[10]]], ignore_index=True)
    with pytest.raises(FeatureError, match="duplicated city-days"):
        build_features(doubled)


def test_a_missing_column_is_rejected_rather_than_skipped() -> None:
    frame = synthetic(days=40).drop(columns=[PRESSURE])
    with pytest.raises(FeatureError, match=PRESSURE):
        build_features(frame)


def test_input_order_does_not_change_the_output() -> None:
    frame = synthetic(days=80)
    shuffled = frame.sample(frac=1.0, random_state=17)
    pd.testing.assert_frame_equal(build_features(frame), build_features(shuffled))


# --------------------------------------------------------------------------
# Against the warehouse
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gold(engine):
    from sqlalchemy import text

    with engine.connect() as connection:
        rows = connection.execute(
            text(f"select count(*) from {GOLD}.fact_weather_observations")
        ).scalar()
    if not rows:
        pytest.skip("fact_weather_observations not built")
    return rows


@pytest.fixture(scope="module")
def built(engine, gold):
    return load_features(engine)


def test_the_matrix_is_one_row_per_gold_city_day(built, gold) -> None:
    """Same row count as the fact table, or something was quietly dropped."""
    assert len(built) == gold
    assert not built.duplicated(subset=["city_id", "date_key"]).any()


def test_every_null_outside_the_warmup_has_a_name(engine, built) -> None:
    """The only unexplained nulls should be none.

    Outside the warm-up, a null feature means one of two things: a hole in the
    series, or a city whose leave-one-year-out baseline left nothing. Anything
    else is a defect, and this is where it surfaces.
    """
    from sqlalchemy import text

    report = missing_report(built).set_index("feature")
    unexplained = report.loc[
        (report["null_after_warmup"] > 0) & (report.index != "z_temperature_2m_mean")
    ]
    assert unexplained.empty, unexplained.to_string()

    with engine.connect() as connection:
        unscored = connection.execute(
            text(
                f"select count(*) from {GOLD}.fact_weather_anomalies "
                "where z_temperature_2m_mean is null"
            )
        ).scalar()
    assert int(report.loc["z_temperature_2m_mean", "null_total"]) == unscored


def test_the_warmup_is_one_window_per_city(built) -> None:
    assert int(built["is_warmup"].sum()) == WARMUP_DAYS * built["city_id"].nunique()


def test_day_of_year_common_agrees_with_dim_date(engine, gold) -> None:
    """The second derivation, checked against the first.

    ``dim_date`` already solves the leap-year shift and this recomputes it, so
    that ``build_features`` stays a pure function of the rows it is handed.
    Two derivations are only safe while something asserts they agree.
    """
    from sqlalchemy import text

    with engine.connect() as connection:
        rows = connection.execute(
            text(f"select date_day, day_of_year_common from {GOLD}.dim_date")
        ).fetchall()
    frame = pd.DataFrame(rows, columns=["date_day", "day_of_year_common"])
    frame["date_day"] = pd.to_datetime(frame["date_day"])
    derived = day_of_year_common(frame["date_day"])
    mismatched = frame.loc[derived.to_numpy() != frame["day_of_year_common"].to_numpy()]
    assert mismatched.empty, mismatched.head().to_string()


def test_a_slice_agrees_with_the_full_build(engine, built) -> None:
    """A windowed request is padded, so it is not silently all warm-up.

    Building features from exactly the rows asked for would give the first
    thirty days of the slice the nulls of a series that begins there, except
    the series does not begin there, the request does.
    """
    city = sorted(built["city_id"].unique())[0]
    whole = built.loc[built["city_id"] == city]
    start = whole["date_key"].min() + pd.Timedelta(days=400)
    end = start + pd.Timedelta(days=120)

    sliced = load_features(engine, cities=[city], start=start, end=end)
    expected = whole.loc[
        whole["date_key"].between(start, end)
    ].reset_index(drop=True)

    assert len(sliced) == len(expected)
    assert not sliced["is_warmup"].any()

    # Every feature agrees to the bit, and so does the warm-up flag: the
    # padding is what makes that true. `history_days` is the one column that
    # cannot: it counts from the padded window, so on a slice it is a lower
    # bound, and asserting that keeps it from quietly becoming something else.
    shared = [column for column in expected.columns if column != "history_days"]
    pd.testing.assert_frame_equal(sliced[shared], expected[shared])
    assert (sliced["history_days"] <= expected["history_days"]).all()
    assert sliced["history_days"].min() >= WARMUP_DAYS
