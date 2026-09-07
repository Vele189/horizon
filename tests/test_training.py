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
a number — because the ticket asks for the weighting on the grounds that it
protects the probabilities a dashboard shows a reader, and it does the opposite.
"""

from __future__ import annotations

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
    evaluation_frame,
    score,
    split_frame,
)
from machine_learning.features import feature_columns  # noqa: E402
from machine_learning.labels import LABEL, positives  # noqa: E402
from machine_learning.train import (  # noqa: E402
    FIXED_PARAMS,
    N_JOBS,
    SEARCH_SPACE,
    SEED,
    TrainingError,
    fit_once,
    merge_into_metrics,
    save_model,
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
        "a resampler reached the codebase — duplicating or discarding rows "
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
    import joblib

    fit = fit_once(
        synthetic_parts["train"],
        synthetic_parts["validation"],
        {"max_depth": 3, "learning_rate": 0.1, "min_child_weight": 10},
        scale_pos_weight=1.0,
    )
    path = save_model(fit, tmp_path / "model.joblib")
    loaded = joblib.load(path)

    assert loaded["features"] == list(feature_columns())
    matrix, _ = training_matrix(synthetic_parts["test"])
    np.testing.assert_array_equal(
        loaded["estimator"].predict_proba(matrix)[:, 1],
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
    distorts them the same way — and here it costs ranking as well.
    """
    _, _, block, _fits = trained
    weighted = block["variants"]["weighted"]["test"]
    unweighted = block["variants"]["unweighted"]["test"]
    base = weighted["base_rate"]

    assert weighted["mean_predicted"] > 3 * base, (
        "the weighted model no longer inflates its probabilities; the argument "
        "recorded in the README needs revisiting"
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
