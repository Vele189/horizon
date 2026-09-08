"""Tests for the split, the metrics, and the two baselines.

These numbers are the target the model will be measured against, so they have
to be right before anything is trained. Two things are being defended here; the
split itself is defended in ``test_split.py``.

The **metrics** must be the ones the model will be scored with. Average
precision for a constant prediction is the base rate by construction, which
makes a constant a free self-check on the implementation, and one test uses it
that way.

The **fit** must never see the future. Baselines are fitted on train only;
tests mutate the validation and test periods and require the fitted parameters
to come back unchanged.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from ml_fixtures import labelled_span, scored_population, spanning  # noqa: E402

from machine_learning.baselines import (  # noqa: E402
    METRICS_SCHEMA_VERSION,
    SMOOTHING_GRID,
    BaseRateReference,
    ClimatologyBaseline,
    PersistenceBaseline,
    build_metrics,
    metrics_path,
    write_metrics,
)
from machine_learning.evaluation import (  # noqa: E402
    ANOMALY_THRESHOLD,
    ANOMALY_THRESHOLDS,
    EMBARGO_DAYS,
    PERSISTENCE_FLAG,
    PURGE_DAYS,
    Score,
    add_persistence_signal,
    base_rate,
    evaluation_frame,
    lift_over,
    reflag,
    score,
    split_frame,
)
from machine_learning.features import (  # noqa: E402
    ANOMALY_COUNT_WINDOW,
    build_features,
    feature_columns,
    on_daily_calendar,
    require_grain,
    trailing_anomaly_counts,
)
from machine_learning.labels import (  # noqa: E402
    HORIZON_DAYS,
    LABEL,
    positives,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# The metrics
# --------------------------------------------------------------------------


def test_a_constant_prediction_scores_exactly_the_base_rate() -> None:
    """Average precision for a constant is the positive rate, by construction.

    Which makes it a free check on the implementation: if this drifts, the
    metric is not the one every other number in the file assumes.
    """
    labels = pd.Series(
        pd.array([True] * 30 + [False] * 170, dtype="boolean")
    ).sample(frac=1.0, random_state=1)
    result = score(labels, np.full(len(labels), 0.4))
    assert result.base_rate == pytest.approx(0.15)
    assert result.pr_auc == pytest.approx(0.15)
    assert result.lift == pytest.approx(1.0)
    assert result.brier == pytest.approx(0.15 * 0.6**2 + 0.85 * 0.4**2)


def test_a_perfect_ranking_scores_one() -> None:
    labels = pd.Series(pd.array([True, True, False, False, False], dtype="boolean"))
    assert score(labels, [0.9, 0.8, 0.3, 0.2, 0.1]).pr_auc == pytest.approx(1.0)


def test_scoring_refuses_what_it_cannot_average() -> None:
    labels = pd.Series(pd.array([True, False, None], dtype="boolean"))
    with pytest.raises(ValueError, match="unlabelled rows reached scoring"):
        score(labels, [0.1, 0.2, 0.3])

    clean = pd.Series(pd.array([True, False], dtype="boolean"))
    with pytest.raises(ValueError, match="2 labels against 3 predictions"):
        score(clean, [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="probabilities in"):
        score(clean, [0.1, 1.4])
    with pytest.raises(ValueError, match="NaN or inf"):
        score(clean, [0.1, np.nan])
    with pytest.raises(ValueError, match="one class is missing"):
        score(pd.Series(pd.array([False, False], dtype="boolean")), [0.1, 0.2])


def test_the_base_rate_helper_counts_nulls_as_negative_nowhere() -> None:
    labels = pd.Series(pd.array([True, False, None, True], dtype="boolean"))
    # positives() is null-safe; base_rate over a frame that still has nulls is
    # a lower bound, which is why score() refuses them outright.
    assert base_rate(labels) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# The persistence signal
# --------------------------------------------------------------------------


def test_the_persistence_signal_uses_the_feature_matrix_counter() -> None:
    """One implementation of "anomaly days behind me", not two.

    ``anomaly_days_trailing30`` in the feature matrix and the 7-day window this
    baseline needs come from the same function, so they cannot drift on what a
    null flag means or on what a window over a hole is worth.
    """
    frame = spanning(days=500)
    features = build_features(frame)
    calendar = on_daily_calendar(
        require_grain(frame, ("city_id", "date_key", "is_anomaly"))
    )
    flagged, _ = trailing_anomaly_counts(calendar, 30)
    rebuilt = calendar.assign(flagged=flagged)
    rebuilt = rebuilt.loc[rebuilt["is_observed"]].reset_index(drop=True)

    pd.testing.assert_series_equal(
        features["anomaly_days_trailing30"],
        rebuilt["flagged"].rename("anomaly_days_trailing30"),
    )


def test_the_persistence_signal_never_looks_forward() -> None:
    frame = spanning(days=800)
    cut = frame["date_key"].iloc[400]

    baseline = add_persistence_signal(frame)
    changed = frame.copy()
    future = changed["date_key"] > cut
    changed.loc[future, "is_anomaly"] = ~changed.loc[future, "is_anomaly"]
    rebuilt = add_persistence_signal(changed)

    past = baseline.loc[baseline["date_key"] <= cut, PERSISTENCE_FLAG]
    past_rebuilt = rebuilt.loc[rebuilt["date_key"] <= cut, PERSISTENCE_FLAG]
    pd.testing.assert_series_equal(past, past_rebuilt)


def test_an_unclosable_persistence_window_is_unknown_not_quiet() -> None:
    frame = spanning(days=200)
    frame["is_anomaly"] = pd.array([False] * 200, dtype="boolean")
    hole = frame["date_key"].iloc[100]
    with_gap = frame.loc[frame["date_key"] != hole].reset_index(drop=True)

    signal = add_persistence_signal(with_gap).set_index("date_key")[PERSISTENCE_FLAG]
    for ahead in range(1, HORIZON_DAYS):
        assert pd.isna(signal.loc[hole + pd.Timedelta(days=ahead)])
    assert signal.loc[hole + pd.Timedelta(days=HORIZON_DAYS)] == False  # noqa: E712
    # The first six days of the record cannot close a seven-day window either.
    assert signal.iloc[: HORIZON_DAYS - 1].isna().all()


# --------------------------------------------------------------------------
# Fitting on train, and only on train
# --------------------------------------------------------------------------


@pytest.fixture
def parts():
    return split_frame(labelled_span())


def test_a_baseline_refuses_to_predict_before_it_is_fitted(parts) -> None:
    for baseline in (BaseRateReference(), PersistenceBaseline(), ClimatologyBaseline()):
        with pytest.raises(RuntimeError, match="has not been fitted"):
            baseline.predict(parts["test"])


@pytest.mark.parametrize(
    "factory", [BaseRateReference, PersistenceBaseline, ClimatologyBaseline]
)
def test_rewriting_the_future_does_not_move_a_fitted_baseline(parts, factory) -> None:
    """The parameters are a function of the training split alone."""
    fitted = factory().fit(parts["train"])
    wrecked = parts["train"].copy()

    later = pd.concat([parts["validation"], parts["test"]], ignore_index=True)
    later[LABEL] = pd.array([True] * len(later), dtype="boolean")
    refitted = factory().fit(wrecked)

    assert json.dumps(fitted.params(), sort_keys=True) == json.dumps(
        refitted.params(), sort_keys=True
    )


def test_the_tuning_never_consults_the_test_split(parts) -> None:
    tuned = ClimatologyBaseline.tuned(parts["train"], parts["validation"])
    wrecked_test = parts["test"].copy()
    wrecked_test[LABEL] = pd.array([True] * len(wrecked_test), dtype="boolean")
    again = ClimatologyBaseline.tuned(parts["train"], parts["validation"])
    assert tuned.smoothing == again.smoothing
    assert tuned.cell_rates == again.cell_rates


def test_the_smoothing_grid_reaches_its_limit() -> None:
    """A parameter chosen at the edge of the grid is clipped, not tuned.

    The first version of this grid stopped at 100 and validation Brier was
    still improving there. The limit is now in the grid, so the search either
    finds an interior optimum or reports the honest degenerate answer.
    """
    assert np.isinf(SMOOTHING_GRID[-1])
    assert SMOOTHING_GRID == tuple(sorted(SMOOTHING_GRID))


def test_the_smoothing_limit_is_exactly_the_city_rate(parts) -> None:
    """At infinity the week cells collapse, and they collapse cleanly.

    Approaching the limit with a large finite pseudo-count leaves a
    rounding-sized week term that still breaks ties, and on this data it
    breaks them the wrong way. The limit is computed, not approached.
    """
    limit = ClimatologyBaseline(smoothing=float("inf")).fit(parts["train"])
    predicted = limit.predict(parts["test"])
    expected = parts["test"]["city_id"].map(limit.city_rates).to_numpy(dtype=float)
    np.testing.assert_allclose(predicted, expected)
    assert len(set(np.round(predicted, 12))) == parts["test"]["city_id"].nunique()


def test_the_persistence_cells_are_the_rule_it_claims(parts) -> None:
    """Two probabilities, each the observed rate of its own cell in train."""
    fitted = PersistenceBaseline().fit(parts["train"])
    train = parts["train"]
    after = positives(train.loc[train[PERSISTENCE_FLAG].eq(True), LABEL]).mean()
    quiet = positives(train.loc[train[PERSISTENCE_FLAG].eq(False), LABEL]).mean()

    assert fitted.after_anomaly == pytest.approx(after)
    assert fitted.after_quiet == pytest.approx(quiet)
    assert fitted.after_anomaly > fitted.after_quiet, (
        "persistence predicts more anomalies after an anomalous week; if this "
        "inverts the rule is not persistence any more"
    )
    assert fitted.cell_rows["when_unknown"] == 0, (
        "a row could not be judged, which means the signal was computed after "
        "the population was trimmed"
    )
    predicted = fitted.predict(parts["test"])
    assert set(np.unique(predicted)) <= {
        fitted.after_anomaly,
        fitted.after_quiet,
        fitted.when_unknown,
    }


# --------------------------------------------------------------------------
# metrics.json
# --------------------------------------------------------------------------


def test_a_population_without_the_signal_is_refused() -> None:
    """The order is enforced, not documented. Getting it wrong is silent."""
    frame = labelled_span(days=4000).drop(columns=[PERSISTENCE_FLAG])
    with pytest.raises(ValueError, match="computed on the whole record"):
        build_metrics(frame=frame)


def test_the_payload_is_strict_json(tmp_path) -> None:
    """No bare Infinity, which Python writes happily and no strict parser reads."""
    payload = build_metrics(frame=labelled_span())
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
    assert "Infinity" not in encoded
    assert json.loads(encoded) == json.loads(
        write_metrics(payload, tmp_path / "metrics.json").read_text()
    )


def test_the_payload_records_what_it_was_computed_on(tmp_path) -> None:
    payload = build_metrics(frame=labelled_span())
    assert payload["schema_version"] == METRICS_SCHEMA_VERSION
    assert payload["label"]["horizon_days"] == HORIZON_DAYS
    assert payload["split"]["purge_days"] == PURGE_DAYS
    assert payload["split"]["embargo_days"] == EMBARGO_DAYS
    assert set(payload["baselines"]) == {"base_rate", "persistence", "climatology"}
    snapshot = payload["snapshot"]
    assert snapshot["rows"] > 0
    assert snapshot["cities"] == len(snapshot["rows_by_city"])
    for entry in payload["baselines"].values():
        assert entry["fitted_on"] == "train"
        for split_name in ("train", "validation", "test"):
            assert 0.0 <= entry[split_name]["pr_auc"] <= 1.0
            assert 0.0 <= entry[split_name]["brier"] <= 1.0


def test_two_runs_agree_to_the_last_digit() -> None:
    """No seed, no ordering dependence, nothing to explain in a diff."""
    frame = labelled_span()
    first = build_metrics(frame=frame)
    second = build_metrics(frame=frame.sample(frac=1.0, random_state=9))
    first.pop("generated_at")
    second.pop("generated_at")
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


# --------------------------------------------------------------------------
# Against the warehouse
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def population(engine):
    from sqlalchemy import text

    with engine.connect() as connection:
        rows = connection.execute(
            text("select count(*) from gold_marts.fact_weather_anomalies")
        ).scalar()
    if not rows:
        pytest.skip("fact_weather_anomalies not built")
    frame = evaluation_frame(engine)
    parts = split_frame(frame)
    if any(part.empty for part in parts.values()):
        pytest.skip("not enough of the record backfilled to fill every split")
    return frame


def test_the_scored_population_has_no_missing_feature(population) -> None:
    """Baseline and model must be scored on identical rows.

    A baseline ignores features, so nothing stops it scoring a row the model
    cannot use, and comparing the two would then be comparing different test
    sets. This is where a city that breaks that shows up. It used to hold by
    luck; ``evaluation_frame`` now enforces it, because the widening backfill
    landed nineteen rows whose windows span a hole in a discontinuous record.
    """
    assert not population["has_missing_feature"].any()
    assert population[LABEL].notna().all()
    assert not population["is_warmup"].any()


def test_every_population_row_is_scorable_by_persistence(population) -> None:
    """The rule has a cell for "could not be judged", and it is now in use.

    The persistence signal is *not* a model input, so a null in it is not a
    reason to drop a row the model can read perfectly well; it is a reason the
    rule has three cells rather than two. Until the backfill widened, the third
    cell was empty and this could be written as "the signal is complete". It no
    longer is: a city whose record has holes has windows that span them. What
    has to hold is the weaker and more useful thing -- that every row in the
    population comes out of the baseline as a probability, so no row is scored
    by the model and skipped by its yardstick.
    """
    fitted = PersistenceBaseline().fit(split_frame(population)["train"])
    predicted = fitted.predict(population)
    assert np.isfinite(predicted).all()
    assert predicted.min() >= 0.0 and predicted.max() <= 1.0

    unknown = population[PERSISTENCE_FLAG].isna()
    if unknown.any():
        # Confined to cities whose record is not continuous. A full city cannot
        # produce one: the population starts thirty days into the record, so a
        # seven-day window always closes.
        assert set(population.loc[unknown, "city_id"]) < set(
            population["city_id"]
        )


def test_the_no_skill_reference_scores_the_base_rate(population) -> None:
    parts = split_frame(population)
    fitted = BaseRateReference().fit(parts["train"])
    result = score(parts["test"][LABEL], fitted.predict(parts["test"]))
    assert result.pr_auc == pytest.approx(result.base_rate, abs=1e-9)
    assert result.lift == pytest.approx(1.0, abs=1e-9)


def test_persistence_is_the_bar_the_model_has_to_clear(population) -> None:
    parts = split_frame(population)
    fitted = PersistenceBaseline().fit(parts["train"])
    result = score(parts["test"][LABEL], fitted.predict(parts["test"]))
    assert result.lift > 1.3, (
        "persistence has lost its edge on test; the baseline in metrics.json "
        "no longer describes this data and must be rebuilt"
    )
    floor = BaseRateReference().fit(parts["train"])
    no_skill = score(parts["test"][LABEL], floor.predict(parts["test"]))
    assert result.brier < no_skill.brier


def test_the_week_of_year_signal_does_not_survive_the_split(population) -> None:
    """A finding, guarded so it cannot quietly stop being true.

    Fitted and scored inside the training period the (city, week) climatology
    is worth about 2.3x no-skill, so the seasonal structure is real and the
    baseline is not broken. Carried across the split it is worth *less than
    nothing*: the unsmoothed version ranks below random, because the anomaly
    mix flips from mostly cold to mostly hot and hot extremes fall in different
    weeks than cold ones. Validation therefore shrinks the week term away
    entirely, and what survives is a per-city rate.

    **And on twelve cities that rate has no skill either.** It used to clear
    the base rate by a little; it now scores 0.9955x it on test, which is
    chance. :func:`test_the_per_city_rate_does_not_survive_the_split_either`
    measures why, and the consequence is ML-09's whole argument: two of the
    three baselines are worth exactly 1.00x, so persistence is the only
    reference in this file that means anything.
    """
    parts = split_frame(population)
    raw = ClimatologyBaseline(smoothing=0.0).fit(parts["train"])
    in_sample = score(parts["train"][LABEL], raw.predict(parts["train"]))
    out_of_sample = score(parts["test"][LABEL], raw.predict(parts["test"]))

    assert in_sample.lift > 1.8, "the week signal was never there to begin with"
    assert out_of_sample.lift < 1.0, (
        f"the raw week climatology now ranks above random out of sample "
        f"({out_of_sample.lift:.2f}x); the regime-shift finding recorded in "
        "the build log needs revisiting"
    )

    tuned = ClimatologyBaseline.tuned(parts["train"], parts["validation"])
    assert np.isinf(tuned.smoothing), (
        "validation no longer wants the week term shrunk away entirely"
    )
    collapsed = score(parts["test"][LABEL], tuned.predict(parts["test"]))
    assert collapsed.lift == pytest.approx(1.0, abs=0.05), (
        f"the collapsed climatology scores {collapsed.lift:.4f}x the base rate "
        "on test. It was chance to within a twentieth when this was written, "
        "and both directions are news: skill means the per-city ordering has "
        "started surviving the split, and anti-skill means it has inverted."
    )


def test_the_per_city_rate_does_not_survive_the_split_either(population) -> None:
    """Why the climatology baseline has stopped being a baseline.

    Shrinkage collapses it to "this city's own historical rate", which is only
    a predictor if the cities keep their order. Across this split they do not:
    Reykjavik is the *most* anomalous city in training and the least in test,
    Singapore the second least and the most. The rank correlation between the
    two periods is about zero, so a per-city rate ranks the test period no
    better than a coin.

    This is the same regime shift the label's base rate shows, seen from
    another angle, and it is the reason ML-09 moved the reporting onto
    persistence: a reference predictor that has quietly become chance is worse
    than no reference, because it still produces a lift.
    """
    parts = split_frame(population)
    rates = {
        name: positives(parts[name][LABEL]).groupby(parts[name]["city_id"]).mean()
        for name in ("train", "test")
    }
    paired = pd.DataFrame(rates).dropna()
    if len(paired) < 5:
        pytest.skip(f"only {len(paired)} cities in both splits; a rank means little")

    agreement = paired["train"].corr(paired["test"], method="spearman")
    assert abs(agreement) < 0.5, (
        f"the per-city anomaly rate now carries across the split "
        f"(Spearman {agreement:+.3f}); the climatology baseline has become a "
        "predictor again and the finding above needs revisiting"
    )


def test_the_committed_metrics_match_a_fresh_run(population) -> None:
    """The target was fixed in advance, against a snapshot, and it says which.

    The backfill is mid-flight, so these numbers will change. A test that
    silently passed on a rebuilt file would defeat the point of committing it,
    and one that failed on every new city would be noise. So it compares the
    snapshot first and **skips with a reason** when the warehouse has moved on:
    the file is not wrong, it is stale, and it has to be rebuilt and
    re-committed before a model is compared against it.
    """
    path = metrics_path()
    if not path.exists():
        pytest.skip(f"{path} not written yet; run baselines.py --write")
    committed = json.loads(path.read_text())

    fresh = build_metrics(frame=population)
    if committed["snapshot"] != fresh["snapshot"]:
        pytest.skip(
            f"metrics.json was fixed against {committed['snapshot']['rows']} rows "
            f"to {committed['snapshot']['last_date']}; the warehouse now holds "
            f"{fresh['snapshot']['rows']} to {fresh['snapshot']['last_date']}. "
            "Re-run `python machine_learning/baselines.py --write` and commit "
            "the result BEFORE comparing a model against it."
        )

    for name, entry in fresh["baselines"].items():
        for split_name in ("train", "validation", "test"):
            for metric in ("pr_auc", "brier", "base_rate"):
                assert entry[split_name][metric] == pytest.approx(
                    committed["baselines"][name][split_name][metric], rel=1e-9
                ), f"{name}/{split_name}/{metric} drifted from the committed target"


# ---------------------------------------------------------------------------
# Threshold sensitivity, and persistence as the yardstick (ML-09)
# ---------------------------------------------------------------------------

#: Words a README sentence can use to name the metric it is claiming about.
METRIC_WORDS = {
    "pr_auc": ("pr-auc", "ranking", "ranks", "average precision"),
    "brier": ("brier", "calibration", "calibrated"),
    "f1": ("f1",),
}

#: Words that scope a claim to a threshold, so it is not claiming all of them.
QUALIFIERS = ("threshold", "|z|", "2.0", "2.5", "3.0", "at some", "not all")

CLAIM = re.compile(
    r"\bbeat(?:s|en|ing)?\b.*"
    r"\b(?:baselines?|persistence|climatology|no-skill)\b",
    re.IGNORECASE,
)

#: A sentence ends at a full stop followed by whitespace. Not at any full stop:
#: the qualifiers this scanner looks for are thresholds, and "2.0" carries one.
#: Splitting on every period truncates "at |Z| > 2.0 it beats the climatology"
#: to "0 it beats the climatology", losing the very words that scope the claim
#: and reporting a properly qualified sentence as a bare boast.
SENTENCE_END = re.compile(r"(?<=[.])\s+")


def flattened(text: str) -> str:
    """Text with its line breaks forgotten; the README wraps at 79 columns."""
    return " ".join(text.split())


def claims_in(text: str) -> list[str]:
    """Every sentence in the README that claims a baseline was beaten."""
    return [
        sentence.strip()
        for sentence in SENTENCE_END.split(flattened(text))
        if CLAIM.search(sentence)
    ]


def metrics_named(sentence: str) -> set[str]:
    lowered = sentence.lower()
    return {
        metric
        for metric, words in METRIC_WORDS.items()
        if any(word in lowered for word in words)
    }


def test_reflagging_at_the_shipped_threshold_reproduces_the_warehouse(
    engine, population
) -> None:
    """The middle point of the sweep is the shipped pipeline, not a copy of it.

    The sweep re-derives ``is_anomaly`` in Python rather than rebuilding the
    mart, because rebuilding writes a threshold nobody chose into the warehouse
    every other model reads. That is only sound if the transcription is exact,
    and "exact" is checkable: at 2.5 it has to reproduce dbt's own column row
    for row, nulls included. If it does not, every point of the sweep is
    measuring a second implementation of the flag rather than the flag.
    """
    from machine_learning.features import gold_frame

    gold = gold_frame(engine)
    rebuilt = reflag(gold, ANOMALY_THRESHOLD)["is_anomaly"]
    # Compared as nullable booleans on both sides. The warehouse column arrives
    # as `object` holding None and reflag builds a `boolean` holding pd.NA;
    # those are the same three states spelled two ways, and normalising here
    # keeps the assertion about the flag rather than about the driver.
    assert rebuilt.equals(gold["is_anomaly"].astype("boolean")), (
        "reflag() disagrees with fact_weather_anomalies at the shipped "
        "threshold, so the sweep is not a sweep of this pipeline"
    )
    assert rebuilt.isna().sum() == gold["is_anomaly"].isna().sum()

    # And the claim that actually matters: the whole population built through
    # the sweep's path is the population built through the shipped one, label
    # included. The dtype above is a detail; this is the guarantee.
    swept = evaluation_frame(engine, threshold=ANOMALY_THRESHOLD)
    assert len(swept) == len(population)
    assert swept[LABEL].equals(population[LABEL])


def test_a_higher_threshold_flags_a_subset(engine, population) -> None:
    """Monotone, and null-preserving. An unscored day is not a quiet day."""
    from machine_learning.features import gold_frame

    gold = gold_frame(engine)
    loose = reflag(gold, 2.0)["is_anomaly"]
    tight = reflag(gold, 3.0)["is_anomaly"]

    assert tight.isna().equals(loose.isna())
    both = loose.notna()
    assert not (tight[both].fillna(False) & ~loose[both].fillna(False)).any(), (
        "a day flagged at |Z| > 3.0 was not flagged at 2.0"
    )
    assert tight[both].sum() < loose[both].sum(), "the sweep is not moving anything"


def test_the_threshold_moves_the_features_and_not_only_the_label() -> None:
    """The sweep is a refit, not a rescore, and this is where that is shown.

    ``anomaly_days_trailing30`` counts flagged days, so it moves with the
    threshold; so does the persistence signal. Re-scoring one fixed feature
    matrix against three answer keys would be a much weaker experiment wearing
    the same name, and it would flatter the model, because the baseline built
    from the same flag would not have moved with it.
    """
    gold = spanning(days=2000)
    loose = scored_population(reflag(gold, 1.0))
    tight = scored_population(reflag(gold, 3.0))

    counts = f"anomaly_days_trailing{ANOMALY_COUNT_WINDOW}"
    assert counts in feature_columns()
    assert loose[counts].sum() > tight[counts].sum(), (
        "the trailing-anomaly feature did not move with the threshold"
    )
    assert (
        loose[PERSISTENCE_FLAG].fillna(False).sum()
        > tight[PERSISTENCE_FLAG].fillna(False).sum()
    )


def test_lift_over_reads_the_same_way_round_on_both_metrics() -> None:
    """Above one is better, for a score and for a loss.

    Brier is a loss, so its ratio is inverted. Getting that wrong produces a
    number that is still a number, still near one, and still plausible, while
    ranking every predictor backwards on calibration.
    """
    better = Score(rows=10, positives=2, base_rate=0.2, pr_auc=0.4,
                   brier=0.05, lift=2.0)
    worse = Score(rows=10, positives=2, base_rate=0.2, pr_auc=0.2,
                  brier=0.10, lift=1.0)

    against = lift_over(better, worse)
    assert against["pr_auc"] == pytest.approx(2.0)
    assert against["brier"] == pytest.approx(2.0)

    reversed_ = lift_over(worse, better)
    assert reversed_["pr_auc"] == pytest.approx(0.5)
    assert reversed_["brier"] == pytest.approx(0.5)

    # A reference that scores zero is not a comparison; inf says so, where a
    # nan would be dropped silently by every aggregation downstream.
    empty = Score(rows=10, positives=0, base_rate=0.0, pr_auc=0.0,
                  brier=0.0, lift=0.0)
    assert lift_over(better, empty)["pr_auc"] == float("inf")


def test_every_committed_score_is_reported_against_persistence() -> None:
    """The reporting shape ML-09 exists to fix, asserted on the committed file.

    Not "the baselines carry it" but "every scored entry carries it". A block
    that reported lift over chance for a new method and lift over persistence
    only for the old ones would be the exact failure this ticket names.
    """
    path = metrics_path()
    if not path.exists():
        pytest.skip("no metrics.json; run baselines.py --write first")
    payload = json.loads(path.read_text())

    for name, entry in payload["baselines"].items():
        assert entry["reference"] == "persistence", name
        for split in ("train", "validation", "test"):
            against = entry[split]["lift_over_persistence"]
            assert set(against) == {"pr_auc", "brier"}, name
            if name == "persistence":
                assert against["pr_auc"] == pytest.approx(1.0)
                assert against["brier"] == pytest.approx(1.0)

    evaluation = payload.get("evaluation")
    if evaluation is None:
        pytest.skip("run evaluate.py --write first")
    for row in evaluation["summary"]:
        assert "lift_over_persistence" in row, row["predictor"]
        assert "lift" in row, "the base-rate lift is still reported beside it"
    for row in evaluation["per_city"]:
        if "model_unweighted_pr_auc" not in row:
            continue
        assert "model_unweighted_vs_persistence" in row, row["city_id"]
        assert "persistence_vs_persistence" in row


def test_the_threshold_sweep_records_every_point_and_what_survives() -> None:
    path = metrics_path()
    if not path.exists():
        pytest.skip("no metrics.json; run baselines.py --write first")
    block = json.loads(path.read_text()).get("threshold_sensitivity")
    if block is None:
        pytest.skip("run `evaluate.py --thresholds --write` first")

    assert block["thresholds"] == list(ANOMALY_THRESHOLDS)
    assert block["shipped_threshold"] == ANOMALY_THRESHOLD
    shipped = [point for point in block["points"] if point["is_shipped"]]
    assert len(shipped) == 1, "the sweep does not contain the shipped threshold"

    # The base rate has to move with the threshold, or the sweep swept nothing.
    rates = [point["base_rate"]["test"] for point in block["points"]]
    assert rates == sorted(rates, reverse=True), rates
    assert rates[0] > rates[-1]

    holds = block["holds_at_every_threshold"]
    for metric in ("pr_auc", "brier", "f1"):
        expected = all(
            point["verdict"][f"model_{point['recommended_variant']}"][
                "beats_every_baseline"
            ][metric]
            for point in block["points"]
        )
        assert holds[metric] is expected, metric


def test_the_readme_claims_are_true_at_every_threshold_or_are_qualified() -> None:
    """The acceptance's third bullet, enforced rather than reviewed.

    Every README sentence claiming a baseline was beaten is checked against the
    sweep. A sentence that names its metrics has to hold for those metrics at
    all three thresholds; a sentence that names none is claiming all of them
    and has to hold for all of them; a sentence that names a threshold has
    scoped itself and is left alone.

    The failure this prevents is the ordinary one: a true sentence written at
    2.5, left in place while a re-run moves the answer, and read by everyone
    afterwards as though it had been checked.
    """
    path = metrics_path()
    if not path.exists():
        pytest.skip("no metrics.json; run baselines.py --write first")
    block = json.loads(path.read_text()).get("threshold_sensitivity")
    if block is None:
        pytest.skip("run `evaluate.py --thresholds --write` first")
    holds = block["holds_at_every_threshold"]

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    sentences = claims_in(readme)
    assert sentences, "no claim of the form 'beats the baselines' found at all"

    unsupported = []
    for sentence in sentences:
        if any(word in sentence.lower() for word in QUALIFIERS):
            continue
        named = metrics_named(sentence) or set(METRIC_WORDS)
        failing = sorted(name for name in named if not holds[name])
        if failing:
            unsupported.append(f"{failing}: {sentence}")
    assert not unsupported, (
        "the README claims a baseline is beaten on a metric that does not "
        "survive every threshold in the sweep, and does not say so: "
        + " | ".join(unsupported)
    )


def test_the_readme_carries_the_sweeps_own_verdict() -> None:
    """Generated, not typed, so it cannot outlive the run that justified it."""
    path = metrics_path()
    if not path.exists():
        pytest.skip("no metrics.json; run baselines.py --write first")
    block = json.loads(path.read_text()).get("threshold_sensitivity")
    if block is None:
        pytest.skip("run `evaluate.py --thresholds --write` first")

    readme = flattened((REPO_ROOT / "README.md").read_text(encoding="utf-8"))
    assert flattened(block["statement"]) in readme, (
        "the README's threshold verdict no longer matches metrics.json, which "
        f"now says: {block['statement']}"
    )


def test_the_claim_scanner_would_notice_an_unqualified_boast() -> None:
    """A scanner that finds nothing passes everything."""
    planted = (
        "The model beats every baseline. It also beats persistence on F1. "
        "At |Z| > 2.0 it beats the climatology baseline."
    )
    found = claims_in(planted)
    assert len(found) == 3
    assert metrics_named(found[0]) == set()
    assert metrics_named(found[1]) == {"f1"}
    assert any(word in found[2].lower() for word in QUALIFIERS)


def test_no_block_another_module_writes_is_dropped_by_a_rebuild(tmp_path) -> None:
    """The carry list has to name every block, and it silently did not.

    ``write_metrics`` keeps the downstream blocks when the snapshot has not
    moved, and drops them when it has, which is right: a model block describing
    a different warehouse is worse than no model block. What it cannot do is
    forget one. ML-09's ``threshold_sensitivity`` was missing from that tuple
    for exactly as long as it took to write this, and the loss is invisible --
    the file stays valid, every number left in it stays correct, and a block
    that has been dropped looks identical to one whose module was never run.

    So the tuple is checked against the keys the file actually acquires, taken
    from the committed file rather than from a second list that could drift
    from it in the same way.
    """
    path = metrics_path()
    if not path.exists():
        pytest.skip("no metrics.json; run baselines.py --write first")
    committed = json.loads(path.read_text())

    written_elsewhere = set(committed) - {
        "schema_version", "generated_at", "label", "split", "snapshot", "baselines",
    }
    assert written_elsewhere, "nothing downstream has been recorded yet"

    destination = tmp_path / "metrics.json"
    destination.write_text(json.dumps(committed, indent=2, sort_keys=True) + "\n")
    # Rewritten with the same snapshot, which is the case where everything must
    # survive. build_metrics is not re-run: the payload is the committed one.
    write_metrics(committed, destination)
    rewritten = json.loads(destination.read_text())

    lost = sorted(written_elsewhere - set(rewritten))
    assert not lost, (
        f"a rebuild of the baselines dropped {lost}. Add them to the "
        "`downstream` tuple in write_metrics; a block missing from it "
        "disappears without an error."
    )
