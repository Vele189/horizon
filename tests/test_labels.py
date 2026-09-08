"""Tests for the forward-window target.

Looking forward is correct here and nowhere else, so the tests are about the
*boundary* rather than about the direction. The central one puts a single
anomaly into an otherwise quiet series and asserts it labels exactly the seven
rows before it: not the day itself, not the eighth day back. An off-by-one at
either end does not fail anything on its own: it produces a slightly different
positive rate and a model quietly answering a different question.

The second thing being proved is that the label cannot be read off the feature
matrix. Exact reconstruction is the wrong measure, because on floating-point columns
every value is unique, so a check for "some function maps this column to the
label" passes vacuously. Rank AUC asks whether a column alone could order the
city-days with every positive first, and answers 1.0 for a leaked label.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from machine_learning.features import (  # noqa: E402
    FeatureError,
    drop_warmup,
    feature_columns,
)
from machine_learning.labels import (  # noqa: E402
    FORWARD_COUNT,
    HAZARD_DAY,
    HAZARD_EVENT,
    HORIZON_DAYS,
    LABEL,
    LABEL_COLUMNS,
    SCORED_COUNT,
    WINDOW_END,
    WINDOW_START,
    build_labels,
    compose_weekly,
    drop_unlabelled,
    person_periods,
    positive_rate_by_city,
    positives,
    single_feature_auc,
    training_frame,
)

#: The daily |Z| > 2.5 rate a normal distribution would give. The ticket's
#: 3-6% expectation is this number carried through the window; the measured
#: daily rate is higher because real residuals have fatter tails, and
#: `test_the_positive_rate_is_the_daily_rate_carried_forward` proves that is
#: where the whole excess comes from.
GAUSSIAN_DAILY_RATE = 0.0124


def quiet(days: int = 60, city_id: str = "alpha", start: str = "2001-01-01"):
    """A series with weather in it and not a single anomaly.

    The weather is deliberately noisy and deliberately irrelevant: the label is
    a function of the flag alone, so anything that makes it move with
    temperature is a defect these tests should see.
    """
    rng = np.random.default_rng(5)
    dates = pd.date_range(start, periods=days, freq="D")
    return pd.DataFrame(
        {
            "city_id": city_id,
            "date_key": dates,
            "temperature_2m_mean": 15 + rng.normal(scale=4.0, size=days),
            "pressure_msl_mean": 1013 + rng.normal(scale=8.0, size=days),
            "is_anomaly": pd.array([False] * days, dtype="boolean"),
        }
    )


# --------------------------------------------------------------------------
# The boundary
# --------------------------------------------------------------------------


def test_a_single_anomaly_labels_exactly_the_seven_days_before_it() -> None:
    """One anomaly, and the exact set of rows it makes positive.

    This is the whole specification of the module in one assertion. Day *t*
    itself is not positive: it is a feature, and a window that included it
    would hand the classifier its own answer through
    ``anomaly_days_trailing30``. The eighth day back is not positive either,
    because the window is seven days and not eight.
    """
    frame = quiet(days=60)
    marked = 30
    frame.loc[frame.index[marked], "is_anomaly"] = True

    labels = build_labels(frame)
    positive = set(labels.index[positives(labels[LABEL])])

    assert positive == set(range(marked - WINDOW_END, marked - WINDOW_START + 1))
    assert positive == set(range(23, 30))
    assert marked not in positive, "a day must not label itself"
    assert marked - WINDOW_END - 1 not in positive, "the window reached too far"


@pytest.mark.parametrize("offset", range(WINDOW_START, WINDOW_END + 1))
def test_every_offset_inside_the_window_labels_the_row(offset: int) -> None:
    frame = quiet(days=60)
    frame.loc[frame.index[30 + offset], "is_anomaly"] = True
    labels = build_labels(frame)
    assert labels.loc[30, LABEL] is not pd.NA and bool(labels.loc[30, LABEL])


@pytest.mark.parametrize("offset", [0, WINDOW_END + 1, WINDOW_END + 2])
def test_no_offset_outside_the_window_labels_the_row(offset: int) -> None:
    """Offset 0 is today, a feature. Offset 8 is next week's problem."""
    frame = quiet(days=60)
    frame.loc[frame.index[30 + offset], "is_anomaly"] = True
    labels = build_labels(frame)
    assert labels.loc[30, LABEL] is not pd.NA and not bool(labels.loc[30, LABEL])


def test_the_forward_count_counts_the_window_and_nothing_else() -> None:
    frame = quiet(days=60)
    for position in (28, 31, 34, 38):  # t-2, t+1, t+4, t+8 relative to row 30
        frame.loc[frame.index[position], "is_anomaly"] = True
    labels = build_labels(frame)
    # Of those four, only 31 and 34 fall in 31..37.
    assert labels.loc[30, FORWARD_COUNT] == 2
    assert labels.loc[30, SCORED_COUNT] == HORIZON_DAYS


def test_the_label_ignores_the_weather() -> None:
    """The target is a function of the flag alone.

    Which is what makes "no feature can reconstruct the label" a property of
    the construction rather than a hope: temperature and pressure are not
    inputs to it, so no amount of rewriting them can move it.
    """
    frame = quiet(days=90)
    frame.loc[frame.index[[20, 55]], "is_anomaly"] = True
    baseline = build_labels(frame)

    wrecked = frame.copy()
    wrecked["temperature_2m_mean"] += 60.0
    wrecked["pressure_msl_mean"] *= -1.0
    pd.testing.assert_frame_equal(baseline, build_labels(wrecked))


# --------------------------------------------------------------------------
# The end of the series, and other windows that cannot be closed
# --------------------------------------------------------------------------


def test_the_last_week_of_a_quiet_series_is_unlabelled() -> None:
    """Their label is unknowable, so it is null rather than negative.

    It falls out of the same rule as everything else: at the end of the record
    fewer than seven forward days are scored, so a ``False`` cannot be earned.
    """
    labels = build_labels(quiet(days=60))
    assert labels[LABEL].isna().sum() == HORIZON_DAYS
    assert labels[LABEL].tail(HORIZON_DAYS).isna().all()
    assert labels[LABEL].head(60 - HORIZON_DAYS).notna().all()
    assert list(labels[SCORED_COUNT].tail(HORIZON_DAYS)) == [6, 5, 4, 3, 2, 1, 0]


def test_a_positive_survives_a_window_that_cannot_be_closed() -> None:
    """An anomaly that happened is not un-happened by a gap beside it.

    So a city whose record ends in an anomalous week loses fewer than seven
    rows, and the ones it keeps are all positive.
    """
    frame = quiet(days=60)
    frame.loc[frame.index[57], "is_anomaly"] = True
    labels = build_labels(frame)

    # Rows 50..56 see the anomaly at 57 and are positive despite short windows.
    assert positives(labels.loc[50:56, LABEL]).all()
    assert (labels.loc[50:56, SCORED_COUNT] < HORIZON_DAYS).any()
    # Row 57 is unknown too, and for the boundary reason rather than the tail
    # one: its own anomaly is not in its own window, and the two days it can
    # see are quiet but too few to earn a negative.
    assert list(labels.index[labels[LABEL].isna()]) == [57, 58, 59]
    assert labels.loc[57, FORWARD_COUNT] == 0


def test_a_negative_needs_all_seven_days_scored() -> None:
    """A hole in the window is not a quiet day."""
    frame = quiet(days=60)
    with_gap = frame.loc[frame["date_key"] != frame["date_key"].iloc[40]]
    labels = build_labels(with_gap).set_index("date_key")

    hole = frame["date_key"].iloc[40]
    for back in range(WINDOW_START, WINDOW_END + 1):
        row = labels.loc[hole - pd.Timedelta(days=back)]
        assert pd.isna(row[LABEL]), f"t-{back} closed a window over the hole"
        assert row[SCORED_COUNT] == HORIZON_DAYS - 1
    # Eight days back the window clears the hole entirely.
    assert not bool(labels.loc[hole - pd.Timedelta(days=8), LABEL])


def test_an_unscored_day_cannot_make_a_week_quiet() -> None:
    """A null flag lowers the denominator; it never writes a negative."""
    frame = quiet(days=60)
    frame.loc[frame.index[35], "is_anomaly"] = pd.NA
    labels = build_labels(frame)
    assert labels.loc[28:34, LABEL].isna().all()
    assert (labels.loc[28:34, SCORED_COUNT] == HORIZON_DAYS - 1).all()
    assert not bool(labels.loc[27, LABEL])


def test_a_city_with_no_baseline_contributes_no_labels() -> None:
    frame = quiet(days=90)
    frame["is_anomaly"] = pd.array([pd.NA] * len(frame), dtype="boolean")
    labels = build_labels(frame)
    assert len(labels) == len(frame)
    assert labels[LABEL].isna().all()
    assert (labels[SCORED_COUNT] == 0).all()


def test_unlabellable_rows_are_kept_until_someone_says_otherwise() -> None:
    labels = build_labels(quiet(days=60))
    assert len(labels) == 60
    trimmed = drop_unlabelled(labels)
    assert len(trimmed) == 60 - HORIZON_DAYS
    assert trimmed[LABEL].notna().all()
    assert list(trimmed.index) == list(range(len(trimmed)))


def test_cities_do_not_label_each_other() -> None:
    alpha = quiet(days=60, city_id="alpha")
    beta = quiet(days=60, city_id="beta")
    beta.loc[beta.index[30], "is_anomaly"] = True
    together = pd.concat([alpha, beta], ignore_index=True).sample(
        frac=1.0, random_state=2
    )
    labels = build_labels(together)
    quiet_city = positives(labels.loc[labels["city_id"] == "alpha", LABEL])
    marked_city = positives(labels.loc[labels["city_id"] == "beta", LABEL])
    assert not quiet_city.any()
    assert marked_city.sum() == HORIZON_DAYS


def test_a_repeated_city_day_is_rejected() -> None:
    frame = quiet(days=40)
    with pytest.raises(FeatureError, match="duplicated city-days"):
        build_labels(pd.concat([frame, frame.iloc[[10]]], ignore_index=True))


def test_the_label_is_not_a_feature() -> None:
    assert not set(LABEL_COLUMNS) & set(feature_columns())
    assert LABEL not in feature_columns()


# --------------------------------------------------------------------------
# No feature reconstructs the label
# --------------------------------------------------------------------------


def test_the_reconstruction_check_catches_a_leaked_label() -> None:
    """The AUC test can fail, and this is what failing looks like.

    Two leaks, because they fail differently: the label copied outright scores
    exactly 1.0, and a single day of the window, the shape an accidental
    ``shift(-1)`` would take, scores far above anything an honest feature
    reaches.
    """
    rng = np.random.default_rng(3)
    frame = quiet(days=400)
    marks = rng.choice(400, size=40, replace=False)
    frame.loc[frame.index[marks], "is_anomaly"] = True

    labels = drop_unlabelled(build_labels(frame))
    # The label copied outright, a count taken over the label's own window
    # (the shape a stray `shift(-1)` in features.py would produce) and noise.
    labels["copied"] = positives(labels[LABEL]).astype(float)
    labels["window_count"] = labels[FORWARD_COUNT].astype(float)
    labels["noise"] = rng.normal(size=len(labels))

    scored = single_feature_auc(
        labels, ["copied", "window_count", "noise"]
    ).set_index("feature")
    assert scored.loc["copied", "auc"] == pytest.approx(1.0)
    assert scored.loc["window_count", "auc"] > 0.9
    assert abs(scored.loc["noise", "auc"] - 0.5) < 0.15


# --------------------------------------------------------------------------
# Against the warehouse
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def frame(engine):
    from sqlalchemy import text

    with engine.connect() as connection:
        rows = connection.execute(
            text("select count(*) from gold_marts.fact_weather_anomalies")
        ).scalar()
    if not rows:
        pytest.skip("fact_weather_anomalies not built")
    built = training_frame(engine)
    if not built[LABEL].notna().any():
        pytest.skip("no labelled rows yet")
    return built


@pytest.fixture(scope="module")
def labelled(frame):
    return drop_unlabelled(frame)


def test_features_and_labels_line_up_row_for_row(engine, frame) -> None:
    from sqlalchemy import text

    with engine.connect() as connection:
        gold_rows = connection.execute(
            text("select count(*) from gold_marts.fact_weather_observations")
        ).scalar()
    assert len(frame) == gold_rows
    assert not frame.duplicated(subset=["city_id", "date_key"]).any()


def test_no_feature_reconstructs_the_label(labelled) -> None:
    """No column alone orders the city-days with the positives first.

    The strongest is ``anomaly_days_trailing30`` at 0.643, which is not a leak
    but the persistence signal the proposal names as baseline one: a city that
    has been anomalous lately is more likely to be anomalous next week. 0.90 is
    the reconstruction bound; 0.75 is a regression guard with deliberate
    headroom over what is measured, so a future feature that quietly carries
    the answer fails here rather than in a suspiciously good metric.
    """
    scored = single_feature_auc(drop_warmup(labelled))
    strongest = scored.iloc[0]
    assert strongest["auc"] < 0.90, scored.head().to_string()
    assert strongest["auc"] < 0.75, scored.head().to_string()
    assert (scored["rows"] > 1000).all()


def test_the_positive_rate_is_the_daily_rate_carried_forward(labelled) -> None:
    """6.93%, against a ticket expecting 3-6%. The gap is accounted for.

    Under independence a daily rate *p* gives a weekly rate of 1-(1-p)⁷.
    Anomalies cluster, so the observed rate is always below that, and the
    ratio between them is what the window construction controls. Feed the same
    arithmetic the Gaussian tail rate the 3-6% expectation was drawn from, and
    it lands inside the band. The excess is entirely that real residuals have
    fatter tails than a normal, which ``fact_weather_anomalies`` already
    measured at 1.65% a day against a theoretical 1.24%.
    """
    daily = labelled["is_anomaly"].fillna(False).astype(bool).mean()
    observed = float(positives(labelled[LABEL]).mean())
    independent = 1 - (1 - daily) ** HORIZON_DAYS

    assert observed < independent, "clustering can only reduce the weekly rate"
    clustering = observed / independent
    assert 0.55 < clustering < 0.75

    gaussian = (1 - (1 - GAUSSIAN_DAILY_RATE) ** HORIZON_DAYS) * clustering
    assert 0.03 <= gaussian <= 0.06, (
        f"the ticket's 3-6% is the Gaussian-tail band; this build gives "
        f"{gaussian:.2%} from it and {observed:.2%} from the measured daily "
        f"rate of {daily:.2%}."
    )
    assert 0.03 <= observed <= 0.10


def test_every_city_rate_is_its_own_daily_rate_carried_forward(labelled) -> None:
    """Rates run 4.6% to 15.3%. The spread is not a defect, and this says why.

    Every city sits at the same fraction of its own independence bound, so the
    variation between them is the variation in their daily anomaly rates and
    nothing else. Tokyo is highest because only four years of it have
    backfilled, so its σ is noisy and it flags 3.9% of days, which is recorded
    against the ``fact_weather_anomalies`` finding rather than tolerated here.
    """
    rows = []
    for city_id, group in labelled.groupby("city_id"):
        daily = group["is_anomaly"].fillna(False).astype(bool).mean()
        observed = float(positives(group[LABEL]).mean())
        independent = 1 - (1 - daily) ** HORIZON_DAYS
        rows.append((city_id, daily, observed, observed / independent))

    for city_id, daily, observed, ratio in rows:
        assert 0.40 <= ratio <= 0.85, (
            f"{city_id} sits at {ratio:.2f} of its independence bound "
            f"({observed:.2%} observed, {daily:.2%} daily). Every other city "
            "is between 0.49 and 0.72, so this one clusters differently"
        )

    hottest = max(rows, key=lambda row: row[2])
    assert hottest[0] == "tokyo", f"expected tokyo to lead, got {hottest[0]}"
    assert hottest[1] > 0.03, "tokyo's rate should track its 3.9% daily flag rate"


def test_the_base_rate_climbs_through_the_chronological_split(labelled) -> None:
    """A finding for ML-03, recorded so it cannot be met by surprise.

    The proposal splits on time: train to 2018, validate to 2021, test after.
    The positive rate is not the same in the three periods: it roughly
    doubles, then doubles again. That is the warming trend expressed through a
    climatology whose baseline spans the whole record, which
    ``fact_weather_anomalies`` already found as a positive corr(year, Z) in
    every city; here it lands directly on the target.

    A model trained at one base rate and scored at another is miscalibrated
    before it starts, and the proposal asks for a Brier score and a calibration
    curve. This test fails if the shift disappears, so the reasoning gets
    revisited rather than silently invalidated.
    """
    years = labelled["date_key"].dt.year
    train = labelled.loc[years <= 2018, LABEL]
    test = labelled.loc[years >= 2022, LABEL]
    if len(train) < 1000 or len(test) < 1000:
        pytest.skip("not enough of the record backfilled to compare eras")

    train_rate = float(positives(train).mean())
    test_rate = float(positives(test).mean())
    assert test_rate > train_rate * 1.5, (
        f"train {train_rate:.2%} against test {test_rate:.2%}: the "
        "non-stationarity this test records has gone, and ML-03's calibration "
        "argument needs rewriting"
    )


def test_the_tail_loses_at_most_one_horizon_per_city(frame) -> None:
    """Seven rows per city, unless the record ends in an anomalous week.

    Cairo ends quiet and loses all seven. Lagos and Singapore end anomalous, so
    their last days are positive on a window that never closes, and they lose
    two and one. Both are the same rule.
    """
    for city_id, group in frame.groupby("city_id"):
        ordered = group.sort_values("date_key")
        if ordered[LABEL].notna().sum() == 0:
            continue  # a city with no baseline at all; covered elsewhere
        tail = ordered.tail(HORIZON_DAYS)
        unlabelled = int(tail[LABEL].isna().sum())
        assert unlabelled <= HORIZON_DAYS
        assert ordered[LABEL].head(len(ordered) - HORIZON_DAYS).notna().all()
        if unlabelled < HORIZON_DAYS:
            assert tail["is_anomaly"].fillna(False).astype(bool).any(), (
                f"{city_id} kept {HORIZON_DAYS - unlabelled} tail rows without "
                "an anomaly to justify them"
            )


def test_the_per_city_report_accounts_for_every_row(frame) -> None:
    report = positive_rate_by_city(frame)
    assert report["rows"].sum() == len(frame)
    assert (report["labelled"] + report["unlabelled"] == report["rows"]).all()
    assert (report["positives"] <= report["labelled"]).all()


# ---------------------------------------------------------------------------
# Discrete-time hazard (ML-13)
# ---------------------------------------------------------------------------


def test_a_city_day_stops_contributing_once_it_fails() -> None:
    """What makes a hazard a hazard rather than seven copies of the week.

    Day five's row exists only for the city-days that reached day five without
    an anomaly, so ``h_5`` is conditional on having survived. A frame that kept
    all seven rows regardless would be seven correlated copies of the weekly
    label and would compose to nonsense.
    """
    quiet = _series("alpha", "2020-01-01", [False] * 20)
    one_hit = quiet.copy()
    one_hit.loc[one_hit["date_key"] == pd.Timestamp("2020-01-04"), "is_anomaly"] = True

    periods = person_periods(one_hit)
    day = periods.loc[periods["date_key"] == pd.Timestamp("2020-01-01")]
    # The anomaly is three days out, so days 1..3 exist and the row stops there.
    assert list(day[HAZARD_DAY]) == [1, 2, 3]
    assert list(day[HAZARD_EVENT]) == [False, False, True]

    # A city-day whose whole window is quiet contributes all seven.
    untouched = periods.loc[periods["date_key"] == pd.Timestamp("2020-01-05")]
    assert list(untouched[HAZARD_DAY]) == [1, 2, 3, 4, 5, 6, 7]
    assert not untouched[HAZARD_EVENT].any()


def test_survival_has_to_be_known_and_not_merely_unflagged() -> None:
    """The same rule the weekly label's negative follows.

    A gap in the record does not establish that nothing happened in it. If day
    t+2 was never scored then day t+3's row cannot claim the event had not
    happened yet, so it does not exist.
    """
    holed = _series("alpha", "2020-01-01", [False] * 20)
    holed.loc[holed["date_key"] == pd.Timestamp("2020-01-03"), "is_anomaly"] = pd.NA

    periods = person_periods(holed)
    day = periods.loc[periods["date_key"] == pd.Timestamp("2020-01-01")]
    # t+2 is unknown, so the row for t+2 cannot exist and nothing after it can
    # claim to have survived it either.
    assert list(day[HAZARD_DAY]) == [1]


def test_the_empirical_hazards_compose_to_the_observed_weekly_rate() -> None:
    """The identity the whole ticket rests on, checked exactly.

    Composed from the *empirical* hazards -- events over at-risk rows at each
    day -- the product telescopes to the share of city-days that survived all
    seven, which is one minus the weekly rate. It is an identity rather than an
    approximation, so it is asserted to floating-point rather than to a
    tolerance, and it tests the reshaping and the composition rather than two
    fitted models agreeing.
    """
    rng = np.random.default_rng(11)
    flags = rng.random(4000) < 0.03
    frame = _series("alpha", "2015-01-01", list(flags))

    labels = build_labels(frame)
    labelled = labels.loc[labels[LABEL].notna()]
    # Over the *same* city-days, which is what makes it an identity. The two
    # populations differ at the end of the record on purpose: the weekly label
    # needs all seven days known, while a person-period row needs only its own
    # day, so the last week of a series contributes hazards and no label. That
    # is the right behaviour for training -- those rows are real observations --
    # and it means the telescoping product only closes exactly on the city-days
    # both agree about.
    periods = person_periods(frame).merge(
        labelled[["city_id", "date_key"]], on=["city_id", "date_key"], how="inner"
    )

    hazards = periods.groupby(HAZARD_DAY)[HAZARD_EVENT].mean().to_numpy()
    composed = compose_weekly(hazards)
    observed = float(positives(labelled[LABEL]).mean())

    assert composed == pytest.approx(observed, abs=1e-9), (
        f"composed {composed:.9f} against observed {observed:.9f}; the "
        "person-period reshaping and the weekly label disagree"
    )


def test_composing_the_wrong_number_of_days_is_an_error() -> None:
    """A short product silently understates the week and raises nothing."""
    with pytest.raises(ValueError, match="hazards to compose"):
        compose_weekly([0.01] * (HORIZON_DAYS - 1))
    assert compose_weekly([0.0] * HORIZON_DAYS) == pytest.approx(0.0)
    assert compose_weekly([1.0] + [0.0] * (HORIZON_DAYS - 1)) == pytest.approx(1.0)


def test_the_purge_covers_the_hazards_reach() -> None:
    """The acceptance's "purge extended to cover the full hazard horizon".

    The hazard's outcomes are days t+1 .. t+7, the same span the weekly label
    already reaches, so the existing purge covers it. Asserting it is what makes
    that a checked coincidence rather than a silent one: lengthen the hazard
    horizon past the label's and this fails, instead of a training row quietly
    acquiring an outcome from the validation period.
    """
    from machine_learning.evaluation import HAZARD_REACH_DAYS, PURGE_DAYS

    assert PURGE_DAYS >= HAZARD_REACH_DAYS


def test_no_training_person_period_reaches_into_a_later_split() -> None:
    """The leak the purge exists to prevent, asserted on the reshaped frame.

    A person-period row's outcome is its own day of the horizon, so the
    furthest a training row reaches is its date plus ``horizon_day``. Every one
    of those has to land before the next split opens.
    """
    from machine_learning.evaluation import split_frame

    rng = np.random.default_rng(3)
    frame = _series("alpha", "1995-01-01", list(rng.random(11000) < 0.02))
    parts = split_frame(person_periods(frame))

    ordered = ("train", "validation", "test")
    for earlier, later in zip(ordered, ordered[1:]):
        before, after = parts[earlier], parts[later]
        if before.empty or after.empty:
            continue
        reach = (
            before["date_key"] + pd.to_timedelta(before[HAZARD_DAY], unit="D")
        ).max()
        assert reach < after["date_key"].min(), (
            f"a {earlier} person-period's outcome falls on {reach.date()}, "
            f"inside {later}, which starts {after['date_key'].min().date()}"
        )


def test_reshaping_never_splits_a_city_day_across_two_splits() -> None:
    """The trap ML-13 names: split by city-day, never by person-period row.

    Reshaping and then splitting would put day 1 of a city-day in training and
    day 4 of the same city-day in test, and the leak would be invisible -- both
    frames would still be in date order and both would still look purged.
    """
    from machine_learning.evaluation import split_frame

    rng = np.random.default_rng(5)
    frame = _series("alpha", "1995-01-01", list(rng.random(11000) < 0.02))
    parts = split_frame(person_periods(frame))

    seen: dict[tuple, str] = {}
    for name, part in parts.items():
        for key in part.groupby(["city_id", "date_key"]).groups:
            assert seen.setdefault(key, name) == name, (
                f"{key} appears in both {seen[key]} and {name}"
            )


def _series(city: str, start: str, flags: list) -> pd.DataFrame:
    """A gold-shaped frame with the given anomaly flags, one per day."""
    return pd.DataFrame(
        {
            "city_id": city,
            "date_key": pd.date_range(start, periods=len(flags), freq="D"),
            "is_anomaly": pd.array(flags, dtype="boolean"),
        }
    )
