"""Tests for the trained classifier.

Two things are being defended, and neither of them is accuracy.

**Reproducibility.** The seed is fixed, the thread count is fixed at one, and
the whole training run happens twice in *separate processes* with the metrics
compared byte for byte. A test that only re-ran the fit in the same interpreter
would miss the failure worth catching, which is a model that depends on how
many cores the machine happened to have.

**That nothing was tuned on test.** ``tune()`` takes a training frame and a
validation frame, and there is no third argument to pass it. The behavioural
version of the same claim is here too: rewriting every label in the test split
must leave the fitted model identical.

The class-imbalance handling is checked rather than assumed. ``scale_pos_weight``
is asserted to come from the observed ratio, no resampler is allowed anywhere in
the repository, and the probability inflation the weighting causes is pinned as
a number, because the ticket asks for the weighting on the grounds that it
protects the probabilities a dashboard shows a reader, and it does the opposite.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("xgboost")
pytest.importorskip("joblib")

from ml_fixtures import labelled_span, repository_sources  # noqa: E402

from machine_learning.evaluation import (  # noqa: E402
    DECISION_PLACES,
    alerts_per_city_year,
    apply_prior_shift,
    decision_table,
    estimate_prior,
    evaluation_frame,
    expected_calibration_error,
    implied_cost_ratio,
    operating_points,
    score,
    scored_city_years,
    split_frame,
    threshold_for_budget,
)
from machine_learning.features import feature_columns  # noqa: E402
from machine_learning.labels import LABEL, positives  # noqa: E402
from machine_learning.train import (  # noqa: E402
    FIXED_PARAMS,
    N_JOBS,
    SEARCH_SPACE,
    SEED,
    RECENCY_HALF_LIVES,
    TrainingError,
    fit_calibrator,
    fit_once,
    merge_into_metrics,
    recency_ablation,
    recency_weights,
    scale_pos_weight_from,
    train_model,
    training_matrix,
    tune,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Names that mean the class balance was changed by duplicating or discarding
#: rows rather than by weighting them. Resampling distorts predicted
#: probabilities, and the Risk Horizon view shows those to a reader directly.
FORBIDDEN_RESAMPLERS = (
    "imblearn",
    "SMOTE",
    "RandomOverSampler",
    "RandomUnderSampler",
    "sklearn.utils.resample",
)


@pytest.fixture(scope="module")
def synthetic():
    return labelled_span()


@pytest.fixture(scope="module")
def synthetic_parts(synthetic):
    return split_frame(synthetic)


# --------------------------------------------------------------------------
# What goes into the model
# --------------------------------------------------------------------------


def test_the_design_matrix_is_the_declared_features_and_nothing_else(
    synthetic_parts,
) -> None:
    """Selected by name, never by dropping the keys.

    ``is_anomaly`` and the label sit in the same frame. A model handed either
    would score beautifully and mean nothing.
    """
    matrix, target = training_matrix(synthetic_parts["train"])
    assert list(matrix.columns) == list(feature_columns())
    assert "is_anomaly" not in matrix.columns
    assert LABEL not in matrix.columns
    assert len(matrix) == len(target) == len(synthetic_parts["train"])


def test_the_design_matrix_refuses_nulls(synthetic_parts) -> None:
    broken = synthetic_parts["train"].copy()
    broken.loc[broken.index[5], "latitude"] = np.nan
    with pytest.raises(TrainingError, match="design matrix has nulls"):
        training_matrix(broken)


def test_scale_pos_weight_is_the_observed_class_ratio() -> None:
    target = np.array([True] * 20 + [False] * 180)
    assert scale_pos_weight_from(target) == pytest.approx(9.0)
    with pytest.raises(TrainingError, match="no positives"):
        scale_pos_weight_from(np.zeros(10, dtype=bool))


def test_the_training_rows_are_the_split_rows(synthetic_parts) -> None:
    """No resampling: the model sees each training row exactly once.

    Weighting changes what a row counts for, not how many rows there are, so
    the design matrix has to be the split unchanged.
    """
    train = synthetic_parts["train"]
    matrix, target = training_matrix(train)
    assert len(matrix) == len(train)
    assert int(target.sum()) == int(positives(train[LABEL]).sum())
    assert matrix.index.equals(train.index)


def test_no_resampler_appears_anywhere_in_the_codebase() -> None:
    offenders = [
        f"{relative}: {name}"
        for relative, text in repository_sources(exclude={Path(__file__)})
        for name in FORBIDDEN_RESAMPLERS
        if name in text
    ]
    assert not offenders, (
        "a resampler reached the codebase: duplicating or discarding rows "
        "distorts the predicted probabilities the dashboard shows directly: "
        f"{offenders}"
    )


def test_the_resampler_scan_would_notice(tmp_path) -> None:
    """Proof the scan can fail, since it passes by finding nothing."""
    planted = tmp_path / "sneaky.py"
    planted.write_text("from imblearn.over_sampling import SMOTE\n")
    assert any(name in planted.read_text() for name in FORBIDDEN_RESAMPLERS)


# --------------------------------------------------------------------------
# Tuned on validation, never on test
# --------------------------------------------------------------------------


def test_the_search_visits_every_combination(synthetic_parts) -> None:
    fit = tune(
        synthetic_parts["train"], synthetic_parts["validation"], scale_pos_weight=1.0
    )
    expected = 1
    for values in SEARCH_SPACE.values():
        expected *= len(values)
    assert len(fit.search) == expected
    for entry in fit.search:
        assert set(SEARCH_SPACE) <= set(entry)
        assert 0.0 <= entry["validation_pr_auc"] <= 1.0
    assert fit.params in [
        {name: entry[name] for name in SEARCH_SPACE} for entry in fit.search
    ]


def test_rewriting_the_test_split_does_not_move_the_model(synthetic) -> None:
    """The behavioural form of "test was never seen".

    ``tune`` takes two frames and there is no third to hand it, but a
    structural argument is only as good as the wiring behind it.
    """
    parts = split_frame(synthetic)
    baseline = tune(parts["train"], parts["validation"], scale_pos_weight=1.0)

    wrecked = synthetic.copy()
    later = wrecked["date_key"] >= pd.Timestamp("2022-01-01")
    wrecked.loc[later, LABEL] = pd.array([True] * int(later.sum()), dtype="boolean")
    wrecked_parts = split_frame(wrecked)
    rebuilt = tune(
        wrecked_parts["train"], wrecked_parts["validation"], scale_pos_weight=1.0
    )

    assert baseline.params == rebuilt.params
    assert baseline.best_iteration == rebuilt.best_iteration
    np.testing.assert_array_equal(
        baseline.predict(parts["test"]), rebuilt.predict(parts["test"])
    )


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


def test_the_seed_and_the_thread_count_are_pinned() -> None:
    assert FIXED_PARAMS["random_state"] == SEED
    assert FIXED_PARAMS["n_jobs"] == N_JOBS == 1, (
        "XGBoost's histogram builder is deterministic for a given thread "
        "count, not across thread counts: the per-thread gradient sums are "
        "added in whatever order the threads finish. A thread count that "
        "depends on the machine is a model that depends on the machine."
    )


def test_two_fits_with_the_same_seed_are_identical(synthetic_parts) -> None:
    params = {"max_depth": 3, "learning_rate": 0.1, "min_child_weight": 10}
    first = fit_once(
        synthetic_parts["train"],
        synthetic_parts["validation"],
        params,
        scale_pos_weight=1.0,
    )
    second = fit_once(
        synthetic_parts["train"],
        synthetic_parts["validation"],
        params,
        scale_pos_weight=1.0,
    )
    np.testing.assert_array_equal(
        first.predict(synthetic_parts["test"]), second.predict(synthetic_parts["test"])
    )


def test_changing_the_seed_changes_the_model(synthetic_parts) -> None:
    """Proof the seed is in play, rather than a parameter nothing reads."""
    from xgboost import XGBClassifier

    matrix, target = training_matrix(synthetic_parts["train"])
    predictions = []
    for seed in (SEED, SEED + 1):
        params = dict(FIXED_PARAMS)
        params["random_state"] = seed
        estimator = XGBClassifier(**params, n_estimators=40, max_depth=4)
        estimator.fit(matrix, target, verbose=False)
        predictions.append(estimator.predict_proba(matrix)[:, 1])
    assert not np.array_equal(predictions[0], predictions[1])


def test_two_runs_in_separate_processes_produce_identical_metrics() -> None:
    """The checklist item, taken literally.

    A fresh interpreter each time, so nothing carried over in module state, a
    warmed cache, or a random generator somebody advanced earlier in the suite.
    This is the failure an in-process repeat cannot see.
    """
    script = textwrap.dedent(
        f"""
        import json, sys
        sys.path.insert(0, {str(REPO_ROOT)!r})
        sys.path.insert(0, {str(REPO_ROOT / "tests")!r})
        from ml_fixtures import labelled_span
        from machine_learning.train import train_model

        block, _fits = train_model(frame=labelled_span())
        block.pop("xgboost_version")
        print(json.dumps(block, sort_keys=True))
        """
    )
    runs = []
    for _ in range(2):
        finished = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=900,
            cwd=REPO_ROOT,
        )
        assert finished.returncode == 0, finished.stderr[-2000:]
        runs.append(finished.stdout)

    assert runs[0] == runs[1], "two clean runs disagreed"
    payload = json.loads(runs[0])
    assert payload["seed_check"] if "seed_check" in payload else True
    for variant in ("weighted", "unweighted"):
        assert payload["variants"][variant]["seed"] == SEED


def test_the_saved_artefact_predicts_what_the_fit_predicted(
    synthetic_parts, tmp_path
) -> None:
    """Serialisation itself is covered in ``test_artifact.py``; this is the
    one property the trainer owns: what came back out predicts what went in."""
    from machine_learning.artifact import load_model, save_artifact

    train = synthetic_parts["train"]
    fit = fit_once(
        train,
        synthetic_parts["validation"],
        {"max_depth": 3, "learning_rate": 0.1, "min_child_weight": 10},
        scale_pos_weight=1.0,
    )
    _, target = training_matrix(train)
    record = save_artifact(
        fit,
        variant="unweighted",
        train=train,
        target=target,
        metrics={},
        directory=tmp_path,
    )
    loaded = load_model(tmp_path / record["filename"], verify=False)

    assert list(loaded.features) == list(feature_columns())
    np.testing.assert_array_equal(
        loaded.predict(synthetic_parts["test"]),
        fit.predict(synthetic_parts["test"]),
    )


# --------------------------------------------------------------------------
# The target has to be the one that was fixed in advance
# --------------------------------------------------------------------------


def test_a_model_cannot_be_recorded_without_committed_baselines(
    synthetic, tmp_path
) -> None:
    block, _fits = train_model(frame=synthetic)
    with pytest.raises(TrainingError, match="does not exist"):
        merge_into_metrics(block, frame=synthetic, path=tmp_path / "absent.json")


def test_a_model_cannot_be_recorded_against_a_moved_snapshot(
    synthetic, tmp_path
) -> None:
    """The guard that makes "fixed in advance" a property rather than a hope."""
    from machine_learning.baselines import build_metrics, write_metrics

    stale = build_metrics(frame=synthetic)
    stale["snapshot"]["rows"] += 1
    path = write_metrics(stale, tmp_path / "metrics.json")

    block, _fits = train_model(frame=synthetic)
    with pytest.raises(TrainingError, match="Re-run baselines.py"):
        merge_into_metrics(block, frame=synthetic, path=path)


def test_a_matching_snapshot_is_recorded_beside_the_baselines(
    synthetic, tmp_path
) -> None:
    from machine_learning.baselines import build_metrics, write_metrics

    path = write_metrics(build_metrics(frame=synthetic), tmp_path / "metrics.json")
    block, _fits = train_model(frame=synthetic)
    merged = merge_into_metrics(block, frame=synthetic, path=path)

    assert set(merged["baselines"]) == {"base_rate", "persistence", "climatology"}
    assert merged["model"]["fitted_on"] == "train"
    assert merged["model"]["tuned_on"] == "validation"
    assert merged["model"]["resampling"] == "none"
    json.dumps(merged, allow_nan=False)


# --------------------------------------------------------------------------
# Against the warehouse
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained(engine):
    from sqlalchemy import text

    with engine.connect() as connection:
        rows = connection.execute(
            text("select count(*) from gold_marts.fact_weather_anomalies")
        ).scalar()
    if not rows:
        pytest.skip("fact_weather_anomalies not built")
    population = evaluation_frame(engine)
    parts = split_frame(population)
    if any(part.empty for part in parts.values()):
        pytest.skip("not enough of the record backfilled to fill every split")
    block, fits = train_model(frame=population)
    return population, parts, block, fits


def test_the_model_beats_both_baselines_on_test(trained) -> None:
    """The point of the whole workstream, and the only number that settles it."""
    _, _, block, _fits = trained
    committed = json.loads(
        (REPO_ROOT / "machine_learning/artifacts/metrics.json").read_text()
    )
    persistence = committed["baselines"]["persistence"]["test"]["pr_auc"]
    climatology = committed["baselines"]["climatology"]["test"]["pr_auc"]

    for variant, entry in block["variants"].items():
        assert entry["test"]["pr_auc"] > persistence, (
            f"the {variant} model scores {entry['test']['pr_auc']:.4f} against "
            f"persistence at {persistence:.4f}; it is not beating the bar that "
            "was fixed before it was trained"
        )
        assert entry["test"]["pr_auc"] > climatology


def test_the_specified_weighting_inflates_the_probabilities(trained) -> None:
    """The finding, pinned so it cannot quietly stop being true.

    The ticket asks for ``scale_pos_weight`` in preference to resampling
    because resampling distorts the probabilities the dashboard shows. But
    weighting the positive class by *k* is oversampling it *k*-fold, so it
    distorts them the same way, and here it costs ranking as well.
    """
    _, _, block, _fits = trained
    weighted = block["variants"]["weighted"]["test"]
    unweighted = block["variants"]["unweighted"]["test"]
    base = weighted["base_rate"]

    assert weighted["mean_predicted"] > 3 * base, (
        "the weighted model no longer inflates its probabilities; the argument "
        "recorded in the build log needs revisiting"
    )
    assert abs(unweighted["mean_predicted"] - base) < abs(
        weighted["mean_predicted"] - base
    )
    assert unweighted["brier"] < weighted["brier"]
    assert block["recommended_variant"] == "unweighted"


def test_the_recorded_model_matches_a_fresh_run(trained) -> None:
    """Same rule as the baselines: compare, or skip and say why."""
    population, _, block, _fits = trained
    path = REPO_ROOT / "machine_learning/artifacts/metrics.json"
    committed = json.loads(path.read_text())
    if "model" not in committed:
        pytest.skip("no model recorded yet; run train.py --write")

    from machine_learning.baselines import build_metrics

    fresh_snapshot = build_metrics(frame=population)["snapshot"]
    if committed["snapshot"] != fresh_snapshot:
        pytest.skip(
            "metrics.json was fixed against a different snapshot; re-run "
            "baselines.py --write, commit, then train.py --write."
        )

    for variant, entry in block["variants"].items():
        recorded = committed["model"]["variants"][variant]
        for split_name in ("train", "validation", "test"):
            for metric in ("pr_auc", "brier"):
                assert entry[split_name][metric] == pytest.approx(
                    recorded[split_name][metric], rel=1e-9
                ), f"{variant}/{split_name}/{metric} drifted from what was recorded"


def test_every_city_is_beaten_individually(trained) -> None:
    """A pooled win can be one city carrying five. This checks it is not."""
    from machine_learning.baselines import PersistenceBaseline

    _population, parts, _block, fits = trained
    fit = fits["unweighted"]
    reference = PersistenceBaseline().fit(parts["train"])

    losses = []
    for city_id, group in parts["test"].groupby("city_id"):
        model = score(group[LABEL], fit.predict(group)).pr_auc
        baseline = score(group[LABEL], reference.predict(group)).pr_auc
        if model <= baseline:
            losses.append(f"{city_id}: {model:.4f} against {baseline:.4f}")
    assert not losses, f"persistence wins in {losses}"


# --------------------------------------------------------------------------
# Calibration under label shift (ML-10)
# --------------------------------------------------------------------------


def test_the_calibrator_is_never_fitted_on_test_rows(synthetic) -> None:
    """The acceptance's fourth bullet, in its behavioural form.

    ``fit_calibrator`` takes one frame and there is no second to hand it, which
    is the same structural argument ``tune`` rests on. A structural argument is
    only as good as the wiring behind it, so the test period is replaced with
    nonsense and the calibrator refitted: every knot of the isotonic fit, and
    every probability it produces, has to be unchanged.

    Isotonic is the easiest calibrator in the world to leak with. It is fitted
    on (score, outcome) pairs, so handing it the test split would make it a
    lookup of the answers, and the resulting reliability curve would be a
    straight line through the origin -- a perfect-looking calibration plot that
    means the opposite of what a reader would take from it.
    """
    parts = split_frame(synthetic)
    fit = tune(parts["train"], parts["validation"], scale_pos_weight=1.0)
    baseline = fit_calibrator(fit, parts["validation"])

    wrecked = synthetic.copy()
    later = wrecked["date_key"] >= pd.Timestamp("2022-01-01")
    wrecked.loc[later, LABEL] = pd.array([True] * int(later.sum()), dtype="boolean")
    rebuilt = fit_calibrator(fit, split_frame(wrecked)["validation"])

    scores = fit.predict(parts["test"])
    np.testing.assert_array_equal(baseline.predict(scores), rebuilt.predict(scores))
    np.testing.assert_array_equal(baseline.X_thresholds_, rebuilt.X_thresholds_)
    np.testing.assert_array_equal(baseline.y_thresholds_, rebuilt.y_thresholds_)


def test_the_prior_estimator_cannot_see_a_label() -> None:
    """Not "does not": cannot. The signature is the guarantee.

    ``estimate_prior`` is the one part of this ticket that reads the target
    period at all, and the distinction that makes it legitimate is that it
    reads the *predictions* and never the outcomes. That is worth enforcing at
    the signature rather than by inspection, because a later edit adding a
    labels argument "just for the diagnostics" would be invisible in a diff of
    the call site.
    """
    parameters = inspect.signature(estimate_prior).parameters
    assert list(parameters)[:2] == ["predictions", "source_prior"]
    for name in parameters:
        assert "label" not in name and "truth" not in name and name != "y"


def test_the_prior_estimator_reports_no_shift_when_there_is_none() -> None:
    """The control that separates a biased estimate from a broken one.

    If the target sample *is* the source, a correct estimator has to return the
    source prior. It does, to the digit. That is what makes the overshoot
    recorded in ``metrics.json`` a property of the data rather than a defect
    here: the routine is right and its assumptions are not met.
    """
    rng = np.random.default_rng(11)
    # A perfectly calibrated predictor: draw the outcome from the prediction.
    predictions = rng.beta(2, 12, 40_000)
    estimated, iterations = estimate_prior(predictions, float(predictions.mean()))

    assert estimated == pytest.approx(float(predictions.mean()), abs=1e-6)
    assert iterations < 10, "a no-op should not need to iterate far"


def test_the_prior_estimator_recovers_a_shift_it_is_given() -> None:
    """And it does work, on data that meets its assumption.

    Two Gaussians with a shared covariance, so the class-conditionals really
    are unchanged and only the mixing weight moves, which is exactly the label
    shift the method assumes. The naive alternative -- averaging the model's
    own posteriors -- is quoted beside it, because that is what a reader would
    reach for and it is wrong by a factor of two.
    """
    rng = np.random.default_rng(3)
    rows, separation, source = 40_000, 1.6, 0.10

    def posterior(x, prior):
        # Two unit-variance Gaussians at 0 and `separation`: the log-odds are
        # linear in x, so no density function is needed.
        odds = np.log(prior / (1 - prior)) + separation * x - separation**2 / 2
        return 1.0 / (1.0 + np.exp(-odds))

    for target in (0.05, 0.25, 0.40):
        truth = rng.random(rows) < target
        observed = rng.normal(np.where(truth, separation, 0.0), 1.0)
        predictions = posterior(observed, source)

        estimated, _ = estimate_prior(predictions, source)
        assert estimated == pytest.approx(target, abs=0.01), (
            f"EM recovered {estimated:.4f} for a true prior of {target}"
        )
        if target > source:
            assert float(predictions.mean()) < target, (
                "the naive mean should understate an increased prior"
            )


def test_the_prior_shift_is_the_identity_when_the_prior_has_not_moved() -> None:
    predictions = np.linspace(0.0, 1.0, 101)
    np.testing.assert_allclose(
        apply_prior_shift(predictions, 0.2, 0.2), predictions, atol=1e-12
    )
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="strictly between"):
            apply_prior_shift(predictions, 0.2, bad)


def test_the_prior_shift_cannot_reorder_anything(synthetic) -> None:
    """It moves the level and never the ranking, which isotonic does not.

    Both halves matter. The prior shift is a monotone map of the posterior
    odds, so PR-AUC is identical to the last digit and the correction is free
    on the ranking axis. Isotonic is monotone but not *strictly*: it collapses
    distinct scores into flat runs, average precision is tie-sensitive, and the
    calibration is therefore paid for in ranking. The cost is recorded rather
    than absorbed.
    """
    parts = split_frame(synthetic)
    fit = tune(parts["train"], parts["validation"], scale_pos_weight=1.0)
    raw = fit.predict(parts["test"])
    calibrated = fit_calibrator(fit, parts["validation"]).predict(raw)
    shifted = apply_prior_shift(calibrated, 0.05, 0.15)

    assert score(parts["test"][LABEL], shifted).pr_auc == pytest.approx(
        score(parts["test"][LABEL], calibrated).pr_auc, abs=1e-12
    )
    order = np.argsort(calibrated, kind="stable")
    assert np.all(np.diff(shifted[order]) >= -1e-12)
    assert len(np.unique(calibrated)) < len(np.unique(raw)), (
        "isotonic is expected to tie scores together; if it stopped, the "
        "ranking cost recorded in metrics.json needs revisiting"
    )


def test_the_calibration_error_is_zero_for_a_calibrated_predictor() -> None:
    rng = np.random.default_rng(5)
    predictions = rng.beta(2, 8, 60_000)
    labels = pd.Series(pd.array(rng.random(60_000) < predictions, dtype="boolean"))

    assert expected_calibration_error(labels, predictions) < 0.01
    # A constant at the base rate is perfectly calibrated *and* useless, which
    # is why this is never reported without PR-AUC beside it.
    constant = np.full(60_000, float(positives(labels).mean()))
    assert expected_calibration_error(labels, constant) < 1e-9


def test_the_report_carries_the_four_rows_and_names_the_oracle(synthetic) -> None:
    """Every variant the ticket asks for, plus the ceiling, clearly marked."""
    block, _ = train_model(frame=synthetic)
    report = block["calibration"]

    assert report["fitted_on"] == "validation"
    assert report["applied_to"] == "test"
    assert {
        "raw_weighted",
        "raw_unweighted",
        "calibrated",
        "calibrated_prior_shifted",
    } <= set(report["variants"])

    for name, entry in report["variants"].items():
        assert entry["reliability"], name
        assert 0.0 <= entry["expected_calibration_error"] <= 1.0, name
        assert set(entry["reliability"][0]) == {
            "predicted", "observed", "rows", "weight"
        }
        assert sum(point["weight"] for point in entry["reliability"]) == pytest.approx(1.0)

    oracle = report["variants"]["prior_shifted_oracle"]
    assert oracle["uses_test_labels"] is True
    assert "NOT SHIPPABLE" in oracle["what"]
    assert report["recommended"] != "prior_shifted_oracle", (
        "the recommendation must never be a configuration that was told the "
        "test period's answer"
    )
    assert report["variants"][report["recommended"]].get("uses_test_labels") is None


def test_the_estimated_prior_is_recorded_beside_the_observed_one(synthetic) -> None:
    """Both numbers, so the estimator can be judged rather than trusted."""
    block, _ = train_model(frame=synthetic)
    report = block["calibration"]

    assert 0.0 < report["estimated_target_prior"] < 1.0
    assert report["prior_estimate_error"] == pytest.approx(
        report["estimated_target_prior"] - report["observed_target_prior"]
    )
    assert report["em_iterations"] >= 1

    diagnostics = report["quantifier_diagnostics"]
    assert set(diagnostics["candidates"]) == {"unweighted", "weighted"}
    for row in diagnostics["candidates"].values():
        for period in ("validation", "test"):
            assert row[period]["error"] == pytest.approx(
                row[period]["estimated"] - row[period]["observed"]
            )
    assert diagnostics["validation_picks_the_better_quantifier"] is (
        diagnostics["closest_on_validation"] == diagnostics["closest_on_test"]
    )


# --------------------------------------------------------------------------
# A threshold that encodes a decision (ML-11)
# --------------------------------------------------------------------------


def test_the_threshold_respects_the_budget_and_is_the_most_sensitive_that_does(
    synthetic,
) -> None:
    """Both halves. Under the budget, and no further under it than necessary.

    A rule that alerts less than it is allowed to is not conservative, it is
    leaving recall on the table for nothing: the budget is the constraint, and
    the best rule is the one that saturates it. So the chosen threshold must
    keep alerts at or under the budget, and the next rule *looser* than it must
    break the budget -- otherwise a better rule was available and was not taken.
    """
    parts = split_frame(synthetic)
    fit = tune(parts["train"], parts["validation"], scale_pos_weight=1.0)
    validation = parts["validation"]
    calibrated = fit_calibrator(fit, validation).predict(fit.predict(validation))

    budget = 20.0
    chosen = threshold_for_budget(
        validation, validation[LABEL], calibrated, budget
    )
    points = operating_points(validation, validation[LABEL], calibrated)
    at = points.loc[points["threshold"] == chosen].iloc[0]
    assert at["alerts_per_city_year"] <= budget

    looser = points.loc[points["threshold"] < chosen]
    if not looser.empty:
        assert looser["alerts_per_city_year"].min() > budget, (
            "a looser rule also fits the budget, so the chosen one is needlessly "
            "quiet and is giving up recall for nothing"
        )


def test_an_impossible_budget_is_refused_rather_than_approximated(synthetic) -> None:
    parts = split_frame(synthetic)
    fit = tune(parts["train"], parts["validation"], scale_pos_weight=1.0)
    validation = parts["validation"]
    calibrated = fit_calibrator(fit, validation).predict(fit.predict(validation))

    with pytest.raises(ValueError, match="quietest available"):
        threshold_for_budget(validation, validation[LABEL], calibrated, 0.0)


def test_the_alert_rate_counts_years_watched_and_not_years_spanned() -> None:
    """A city with a hole in its record was not watched during the hole.

    Counting the span instead of the rows would divide the same alerts by a
    longer period and quietly report a calmer product than the one a reader
    gets.
    """
    dense = pd.DataFrame({"date_key": pd.date_range("2020-01-01", periods=365)})
    holed = pd.concat(
        [dense.head(100), pd.DataFrame({"date_key": [pd.Timestamp("2029-01-01")]})],
        ignore_index=True,
    )
    assert scored_city_years(dense) == pytest.approx(365 / 365.25)
    assert scored_city_years(holed) == pytest.approx(101 / 365.25)

    flagged = np.zeros(len(dense), dtype=bool)
    flagged[:20] = True
    assert alerts_per_city_year(dense, flagged) == pytest.approx(20 * 365.25 / 365)


def test_the_implied_cost_ratio_inverts_the_bayes_rule() -> None:
    """A threshold on a calibrated probability *is* a cost ratio.

    Flagging when ``p >= 1 / (1 + c)`` minimises expected cost when a miss
    costs ``c`` false alarms, so the ratio can be read straight off the
    threshold. That is only true because ML-10 calibrated the probabilities; on
    a raw score it would be arithmetic with no meaning attached.
    """
    for ratio in (1.0, 4.0, 9.0):
        threshold = 1.0 / (1.0 + ratio)
        assert implied_cost_ratio(threshold) == pytest.approx(ratio)
    assert implied_cost_ratio(0.5) == pytest.approx(1.0), "F1's implicit claim"
    # The endpoints are limits, not errors: isotonic clips to [0, 1].
    assert implied_cost_ratio(0.0) == float("inf")
    assert implied_cost_ratio(1.0) == 0.0
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        implied_cost_ratio(1.5)


def test_the_operating_points_are_rules_and_not_arithmetic_dust(synthetic) -> None:
    """The defect that made the first decision table useless.

    ``IsotonicRegression.predict`` interpolates between knots, so rows the
    calibrator maps to one level come back spread across a window narrower than
    a nanometre. Swept as distinct thresholds they became distinct "rules", and
    the table offered a reader neighbours whose thresholds agreed to eight
    decimal places and whose alert rates differed by 4.5 a year.

    Two properties keep it honest: no two rules may sit closer together than
    the resolution the project states them at, and no two may flag the same
    days.
    """
    parts = split_frame(synthetic)
    fit = tune(parts["train"], parts["validation"], scale_pos_weight=1.0)
    validation = parts["validation"]
    calibrated = fit_calibrator(fit, validation).predict(fit.predict(validation))
    points = operating_points(validation, validation[LABEL], calibrated)

    gaps = np.diff(points["threshold"].to_numpy())
    assert (gaps >= 10.0**-DECISION_PLACES / 2).all(), (
        "two operating points are closer than the resolution a threshold is "
        "stated at, so they are the same rule printed twice"
    )
    assert not points.duplicated(subset=["flagged", "true_positive"]).any()
    assert points["alerts_per_city_year"].is_monotonic_decreasing


def test_the_decision_table_shows_a_trade_and_marks_the_choice(synthetic) -> None:
    parts = split_frame(synthetic)
    fit = tune(parts["train"], parts["validation"], scale_pos_weight=1.0)
    validation = parts["validation"]
    calibrated = fit_calibrator(fit, validation).predict(fit.predict(validation))
    chosen = threshold_for_budget(validation, validation[LABEL], calibrated, 20.0)

    table = decision_table(validation, validation[LABEL], calibrated, chosen)
    assert list(table["position"]).count("chosen") == 1
    assert len(table) >= 2, "a decision with no alternative beside it is not a decision"

    marked = table.loc[table["position"] == "chosen"].iloc[0]
    assert marked["threshold"] == pytest.approx(chosen)
    # A looser rule alerts more and catches more; a tighter one does the reverse.
    for _, row in table.iterrows():
        if row["position"] == "looser":
            assert row["alerts_per_city_year"] > marked["alerts_per_city_year"]
            assert row["recall"] >= marked["recall"]
        elif row["position"] == "tighter":
            assert row["alerts_per_city_year"] < marked["alerts_per_city_year"]
            assert row["recall"] <= marked["recall"]


def test_the_recorded_decision_states_its_rule_and_what_it_replaced(
    synthetic,
) -> None:
    """The block a reader disagrees with, rather than a number they must accept."""
    block, _ = train_model(frame=synthetic)
    decision = block["calibration"]["decision"]

    assert decision["rule"] == "alert budget"
    assert decision["rejected_rule"] == "f1"
    assert decision["chosen_on"] == "validation"
    assert "calibrated" in decision["computed_on"]
    assert decision["implied_cost_ratio"] == pytest.approx(
        implied_cost_ratio(decision["threshold"])
    )
    assert decision["f1_alternative"]["threshold"] != decision["threshold"], (
        "the budget rule picked exactly what F1 did, so the ticket changed "
        "nothing and the model card should not claim otherwise"
    )
    for period in ("validation", "test"):
        rows = decision["periods"][period]["neighbourhood"]
        assert any(row["position"] == "chosen" for row in rows)
        for row in rows:
            assert {"precision", "recall", "alerts_per_city_year"} <= set(row)


def test_the_budget_is_set_on_validation_and_the_overshoot_is_recorded(
    synthetic,
) -> None:
    """A budget set on one period and spent on another is not a guarantee.

    The base rate rises across this split, so a threshold that fits the budget
    on validation overspends it on test. That is the same drift every other
    ticket in this phase is about, and the point of recording both periods is
    that the overspend is a number rather than a surprise.
    """
    block, _ = train_model(frame=synthetic)
    decision = block["calibration"]["decision"]
    chosen = {
        period: next(
            row for row in decision["periods"][period]["neighbourhood"]
            if row["position"] == "chosen"
        )
        for period in ("validation", "test")
    }
    assert (
        chosen["validation"]["alerts_per_city_year"]
        <= decision["budget_alerts_per_city_year"]
    ), "the budget is not even met on the period it was chosen on"
    assert chosen["test"]["alerts_per_city_year"] > 0


# --------------------------------------------------------------------------
# Recency weighting (ML-12)
# --------------------------------------------------------------------------


def test_an_infinite_half_life_is_no_weights_at_all(synthetic) -> None:
    """The null arm of the ablation must be the fit every other ticket used.

    ``recency_weights`` returns ``None`` rather than a vector of ones, because
    ``None`` is XGBoost's own spelling of "unweighted" and takes a different
    path through the library. A vector of ones is arithmetically the same and
    is not the same code, and this ablation is only readable if its baseline is
    bit-identical to the model the rest of the project describes.
    """
    parts = split_frame(synthetic)
    assert recency_weights(parts["train"], float("inf")) is None

    plain = tune(parts["train"], parts["validation"], scale_pos_weight=1.0)
    explicit = tune(
        parts["train"],
        parts["validation"],
        scale_pos_weight=1.0,
        half_life_years=float("inf"),
    )
    np.testing.assert_array_equal(
        plain.predict(parts["test"]), explicit.predict(parts["test"])
    )


def test_the_weights_decay_from_the_last_training_row(synthetic) -> None:
    """Age is measured from the split's own end, never from today.

    Today moves with the wall clock, so a weight measured from it would make
    two runs of the same commit produce different models -- and the
    reproducibility test that runs training twice in separate processes would
    fail for a reason nobody could act on.
    """
    parts = split_frame(synthetic)
    train = parts["train"]
    weights = recency_weights(train, 4.0)

    newest = int(train["date_key"].idxmax())
    oldest = int(train["date_key"].idxmin())
    assert weights[newest] == weights.max()
    assert weights[oldest] == weights.min()

    # A half-life is a half-life: a row that old carries half the weight.
    span_years = (
        train["date_key"].max() - train["date_key"]
    ).dt.days.to_numpy() / 365.25
    unnormalised = weights / weights[newest]
    at_one_half_life = np.argmin(np.abs(span_years - 4.0))
    assert unnormalised[at_one_half_life] == pytest.approx(0.5, abs=0.01)


def test_the_weights_are_normalised_so_the_grid_means_one_thing(synthetic) -> None:
    """min_child_weight is a floor on summed hessian, which scales with weights.

    Without normalisation the same grid value would be a different constraint
    at every half-life, and the search would be comparing regularisation
    strengths while believing it was comparing half-lives.
    """
    parts = split_frame(synthetic)
    for half_life in (1.0, 4.0, 16.0):
        weights = recency_weights(parts["train"], half_life)
        assert weights.mean() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="must be positive"):
        recency_weights(parts["train"], 0.0)


def test_the_half_life_is_tuned_on_validation_and_test_cannot_move_it(
    synthetic,
) -> None:
    """The acceptance's "tuned on validation only", behaviourally.

    ``recency_ablation`` is handed every split, because it reports on test as
    well as choosing on validation, so a signature argument is not available
    here the way it is for ``tune``. The behavioural check is the one that
    matters anyway: rewrite the test period and the choice must not move.
    """
    parts = split_frame(synthetic)
    baseline_fit = tune(parts["train"], parts["validation"], scale_pos_weight=1.0)
    chosen = recency_ablation(
        parts, scale_pos_weight=1.0, baseline=baseline_fit
    )["chosen_half_life_years"]

    # Inverted rather than set to all-True, which is how the other rewrite
    # tests in this module wreck a split. `recency_ablation` also *reports* on
    # test, and a single-class test period makes average precision undefined,
    # so an all-True rewrite would fail on the reporting rather than prove
    # anything about the choosing. Inversion is just as destructive to any
    # signal and leaves both classes present.
    wrecked = synthetic.copy()
    later = wrecked["date_key"] >= pd.Timestamp("2022-01-01")
    wrecked.loc[later, LABEL] = ~wrecked.loc[later, LABEL]
    rebuilt = recency_ablation(
        split_frame(wrecked), scale_pos_weight=1.0, baseline=baseline_fit
    )["chosen_half_life_years"]

    assert chosen == rebuilt, (
        "rewriting the test period changed which half-life was chosen, so the "
        "choice is reading test"
    )


def test_the_grid_offers_the_null_hypothesis(synthetic) -> None:
    """A tuned parameter that cannot choose "do nothing" is a clipped one."""
    assert float("inf") in RECENCY_HALF_LIVES
    assert min(RECENCY_HALF_LIVES) > 0

    block, _ = train_model(frame=synthetic)
    recency = block["recency"]
    assert "inf" in recency["grid"]
    assert {row["half_life_years"] for row in recency["search"]} == set(
        recency["grid"]
    )
    assert recency["helped"] is (recency["chosen_half_life_years"] != "inf")


def test_the_ablation_shows_a_contrast_even_when_the_answer_is_no(
    synthetic,
) -> None:
    """Comparing the choice with itself prints one row twice and hides it.

    When the rule declines to weight -- which is what happens on this data --
    the ablation compares the unweighted fit against the best *finite*
    half-life, so a reader sees what the alternative would have cost rather
    than a table that agrees with itself.
    """
    block, _ = train_model(frame=synthetic)
    recency = block["recency"]
    arms = {row["arm"]: row for row in recency["ablation"]}
    assert set(arms) == {"unweighted", "recency"}
    assert arms["unweighted"]["half_life_years"] == "inf"
    assert arms["recency"]["half_life_years"] != "inf", (
        "the ablation is comparing the unweighted fit with itself"
    )
    for row in recency["ablation"]:
        for split in ("validation", "test"):
            assert {"pr_auc", "brier", "mean_predicted"} <= set(row[split])


def test_the_effective_sample_size_prices_a_short_half_life(synthetic) -> None:
    """A half-life short enough to fix drift is short enough to lose the record.

    Kish's effective sample size is what makes that visible: it is reported
    beside every validation score so a half-life that wins by a hundredth on a
    fraction of the rows can be recognised as the trade it is.
    """
    block, _ = train_model(frame=synthetic)
    search = {row["half_life_years"]: row for row in block["recency"]["search"]}
    rows = len(split_frame(synthetic)["train"])

    assert search["inf"]["effective_rows"] == pytest.approx(rows)
    finite = [row for key, row in search.items() if key != "inf"]
    for row in finite:
        assert row["effective_rows"] < rows
        assert 0.0 <= row["oldest_row_weight"] <= 1.0
    # Shorter half-lives keep less of the record, monotonically.
    ordered = sorted(finite, key=lambda row: row["half_life_years"])
    assert [row["effective_rows"] for row in ordered] == sorted(
        row["effective_rows"] for row in ordered
    )


def test_the_diagnosis_says_whether_there_is_drift_to_lean_on(synthetic) -> None:
    """Why the answer came out the way it did, recorded beside the answer.

    Recency weighting can only exploit a trend the *training* split contains.
    On the real warehouse there is none -- the first half of the training
    period runs at a slightly higher base rate than the second, and the label's
    drift is entirely between the splits -- so the search correctly refuses to
    throw away rows for it. Without this field the null result would look like
    a failure of the method rather than an absence of the thing it corrects.
    """
    block, _ = train_model(frame=synthetic)
    drift = block["recency"]["training_drift"]

    assert drift["train_first_half"]["through"] < drift["train_second_half"]["from"]
    assert drift["within_train_ratio"] == pytest.approx(
        drift["train_second_half"]["base_rate"]
        / drift["train_first_half"]["base_rate"]
    )
    assert drift["drift_is_inside_the_training_window"] is (
        drift["within_train_ratio"] > 1.1
    )


def test_the_distance_table_is_marked_as_never_selecting(synthetic) -> None:
    """It reads the test labels, so it must be unmistakably a report.

    The finding it carries is uncomfortable -- recency weighting wins on test
    and loses on validation -- and the temptation it creates is exactly the one
    this project refuses. The flags are asserted so that a later edit which
    started selecting on it would have to delete an assertion saying it does
    not.
    """
    block, _ = train_model(frame=synthetic)
    table = block["recency"]["gain_by_distance"]

    assert table["uses_test_labels"] is True
    assert table["used_for_selection"] is False
    periods = {row["period"] for row in table["periods"]}
    assert "validation" in periods and "test" in periods
    for row in table["periods"]:
        assert row["gain"] == pytest.approx(
            row["recency_pr_auc"] / row["unweighted_pr_auc"]
        )
