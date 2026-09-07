"""Tests for model serialisation and the metadata that makes it usable.

A pickle is a liability without a record beside it. The estimator will accept a
matrix whose columns are in the wrong order and return a confident probability
for every row, so the tests here are mostly about the ways a loaded model can
be wrong while looking right:

* the **feature order** — selected by name rather than trusted, and a bare
  array refused outright because its order cannot be checked against anything;
* the **tree range** — early stopping leaves 68 trees in a booster that scores
  on 18, and a pickle that loses that range is a different model under the same
  name;
* the **file itself** — verified against the SHA-256 in the sidecar, so an
  artefact that changed after the metrics were written cannot be loaded under
  them.

The size gate is here too. Committing the model is deliberate — Streamlit
Community Cloud cannot reach the warehouse to retrain — and git cannot express
"only while it is small", so a test does.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("xgboost")
pytest.importorskip("joblib")

from ml_fixtures import labelled_span  # noqa: E402

from machine_learning.artifact import (  # noqa: E402
    ARTIFACT_FORMAT_VERSION,
    MAX_COMMITTED_BYTES,
    ArtifactError,
    artifact_name,
    fingerprint,
    git_provenance,
    load_model,
    save_artifact,
    tracked_artifacts,
)
from machine_learning.baselines import metrics_path  # noqa: E402
from machine_learning.evaluation import split_frame  # noqa: E402
from machine_learning.features import feature_columns  # noqa: E402
from machine_learning.train import train_model, training_matrix  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def trained():
    frame = labelled_span()
    parts = split_frame(frame)
    block, fits = train_model(frame=frame)
    return parts, block, fits


@pytest.fixture(scope="module")
def saved(trained, tmp_path_factory):
    parts, block, fits = trained
    directory = tmp_path_factory.mktemp("artifacts")
    _, target = training_matrix(parts["train"])
    record = save_artifact(
        fits["unweighted"],
        variant="unweighted",
        train=parts["train"],
        target=target,
        metrics=block["variants"]["unweighted"],
        baselines={"persistence": {"test": {"pr_auc": 0.2, "brier": 0.12}}},
        snapshot={"rows": len(parts["train"]), "last_date": "2025-01-01"},
        recommended=True,
        directory=directory,
    )
    return directory, record, fits["unweighted"], parts


# --------------------------------------------------------------------------
# The filename carries a version, and the version means something
# --------------------------------------------------------------------------


def test_the_filename_carries_a_version_and_a_fingerprint(saved) -> None:
    _, record, _, _ = saved
    name = record["filename"]
    assert name.startswith("model-unweighted-v")
    assert name.endswith(".joblib")
    assert f"-v{ARTIFACT_FORMAT_VERSION}-" in name
    assert record["fingerprint"] in name
    assert len(record["fingerprint"]) == 12
    assert artifact_name("unweighted", record["fingerprint"]) == name


def test_the_fingerprint_is_derived_from_what_determines_the_model() -> None:
    """Same inputs, same id. A timestamp would change when nothing had."""
    base = {"features": ["a", "b"], "seed": 42, "window": {"rows": 10}}
    assert fingerprint(base) == fingerprint(dict(reversed(list(base.items()))))
    assert fingerprint(base) != fingerprint({**base, "seed": 43})
    assert fingerprint(base) != fingerprint({**base, "features": ["b", "a"]})
    assert fingerprint(base) != fingerprint({**base, "window": {"rows": 11}})


def test_retraining_the_same_model_produces_the_same_filename(
    trained, tmp_path
) -> None:
    parts, block, fits = trained
    _, target = training_matrix(parts["train"])
    kwargs = dict(
        variant="unweighted",
        train=parts["train"],
        target=target,
        metrics=block["variants"]["unweighted"],
        directory=tmp_path,
    )
    first = save_artifact(fits["unweighted"], **kwargs)
    second = save_artifact(fits["unweighted"], **kwargs)
    assert first["filename"] == second["filename"]
    assert first["fingerprint"] == second["fingerprint"]
    # And nothing accumulated: one model, one file.
    assert len(list(tmp_path.glob("*.joblib"))) == 1


def test_a_superseded_artefact_is_pruned(trained, tmp_path) -> None:
    parts, block, fits = trained
    _, target = training_matrix(parts["train"])
    common = dict(
        variant="unweighted",
        train=parts["train"],
        target=target,
        metrics=block["variants"]["unweighted"],
        directory=tmp_path,
    )
    old = save_artifact(fits["unweighted"], snapshot={"rows": 1}, **common)
    new = save_artifact(fits["unweighted"], snapshot={"rows": 2}, **common)

    assert old["filename"] != new["filename"]
    assert not (tmp_path / old["filename"]).exists()
    assert (tmp_path / new["filename"]).exists()


# --------------------------------------------------------------------------
# The sidecar records what the ticket asks it to
# --------------------------------------------------------------------------


def test_the_metadata_records_the_training_window(saved) -> None:
    _, record, _, parts = saved
    window = record["training_window"]
    assert window["start"] == str(parts["train"]["date_key"].min().date())
    assert window["end"] == str(parts["train"]["date_key"].max().date())
    assert window["rows"] == len(parts["train"])
    assert 0 < window["positives"] < window["rows"]
    assert window["positive_rate"] == pytest.approx(
        window["positives"] / window["rows"]
    )


def test_the_metadata_records_the_feature_list_in_order(saved) -> None:
    _, record, fit, _ = saved
    assert record["features"] == list(feature_columns())
    assert record["features"] == list(fit.feature_names)
    assert record["feature_count"] == len(feature_columns())


def test_the_metadata_records_the_hyperparameters_and_the_seed(saved) -> None:
    from machine_learning.train import SEED

    _, record, fit, _ = saved
    hyper = record["hyperparameters"]
    assert hyper["seed"] == SEED
    assert hyper["n_estimators"] == fit.best_iteration + 1
    assert hyper["scale_pos_weight"] == fit.scale_pos_weight
    for name in fit.params:
        assert hyper[name] == fit.params[name]


def test_the_metadata_carries_the_baseline_comparison(saved) -> None:
    """Carried, not referenced. Somebody deciding whether to deploy should not
    need two other blocks in their head to find out whether 0.35 is good."""
    _, record, _, _ = saved
    comparison = record["baseline_comparison"]["persistence"]
    mine = record["metrics"]["test"]["pr_auc"]
    assert comparison["pr_auc"] == 0.2
    assert comparison["pr_auc_delta"] == pytest.approx(mine - 0.2)
    assert comparison["beaten_on_pr_auc"] == (mine > 0.2)
    # Lower Brier is better, so a positive delta is an improvement.
    theirs = 0.12
    assert comparison["brier_delta"] == pytest.approx(
        theirs - record["metrics"]["test"]["brier"]
    )


def test_the_metadata_records_provenance(saved) -> None:
    _, record, _, _ = saved
    assert set(record["libraries"]) >= {"xgboost", "numpy", "pandas", "joblib"}
    assert "created_at" in record
    assert record["sha256"] and len(record["sha256"]) == 64
    assert record["bytes"] > 0


def test_the_git_record_says_whether_the_tree_was_clean() -> None:
    """A hash with no dirty flag implies a provenance it does not have.

    The artefact must exist before it can be committed, so a model committed in
    the same change that made it was trained from a dirty tree. That is the
    normal case and it is recorded rather than hidden.
    """
    provenance = git_provenance(REPO_ROOT)
    assert set(provenance) == {"commit", "short_commit", "branch", "dirty"}
    if provenance["commit"] is not None:
        assert len(provenance["commit"]) == 40
        assert provenance["short_commit"] == provenance["commit"][:12]
        assert isinstance(provenance["dirty"], bool)


def test_provenance_outside_a_repository_is_null_not_missing(tmp_path) -> None:
    provenance = git_provenance(tmp_path)
    assert set(provenance) == {"commit", "short_commit", "branch", "dirty"}


# --------------------------------------------------------------------------
# The loader validates before it predicts
# --------------------------------------------------------------------------


def test_the_loader_round_trips_the_predictions(saved) -> None:
    directory, record, fit, parts = saved
    loaded = load_model(directory / record["filename"], verify=False)

    assert loaded.features == tuple(feature_columns())
    assert loaded.n_estimators == fit.best_iteration + 1
    assert loaded.fingerprint == record["fingerprint"]
    np.testing.assert_array_equal(
        loaded.predict(parts["test"]), fit.predict(parts["test"])
    )


def test_a_missing_feature_is_named_not_guessed(saved) -> None:
    directory, record, _, parts = saved
    loaded = load_model(directory / record["filename"], verify=False)
    short = parts["test"].drop(columns=["latitude", "day_of_year_sin"])

    with pytest.raises(ArtifactError, match="feature\\(s\\) missing"):
        loaded.predict(short)
    with pytest.raises(ArtifactError, match="latitude"):
        loaded.predict(short)


def test_shuffled_columns_are_reordered_rather_than_mis_mapped(saved) -> None:
    """The failure this whole module exists to prevent.

    Handing an estimator the right columns in the wrong order produces a
    probability for every row and no error at all. Selecting by name in the
    recorded order makes that impossible rather than merely detectable.
    """
    directory, record, fit, parts = saved
    loaded = load_model(directory / record["filename"], verify=False)

    test = parts["test"]
    shuffled = test.loc[:, list(reversed(test.columns))]
    np.testing.assert_array_equal(
        loaded.predict(shuffled), fit.predict(test)
    )

    # And the mis-mapping the reordering avoids is real: fed the same columns
    # backwards as a bare matrix, the estimator answers differently.
    backwards = test.loc[:, list(reversed(list(feature_columns())))]
    scrambled = loaded.estimator.predict_proba(backwards.to_numpy())[:, 1]
    assert not np.allclose(scrambled, fit.predict(test))


def test_a_bare_array_is_refused(saved) -> None:
    directory, record, _, parts = saved
    loaded = load_model(directory / record["filename"], verify=False)
    matrix, _ = training_matrix(parts["test"])

    with pytest.raises(ArtifactError, match="expected a DataFrame"):
        loaded.predict(matrix.to_numpy())


def test_nulls_are_refused_rather_than_scored(saved) -> None:
    directory, record, _, parts = saved
    loaded = load_model(directory / record["filename"], verify=False)
    holed = parts["test"].copy()
    holed.loc[holed.index[0], "latitude"] = np.nan

    with pytest.raises(ArtifactError, match="nulls in"):
        loaded.predict(holed)


def test_a_lost_tree_range_is_caught(saved) -> None:
    """Early stopping keeps trees the model does not use.

    A booster that has lost the range excluding them scores with all of them
    and is a different model under the same filename — the same failure that
    made the first SHAP values explain a model nobody runs.
    """
    import joblib

    directory, record, _, _ = saved
    payload = joblib.load(directory / record["filename"])
    payload["n_estimators"] = payload["n_estimators"] + 5
    broken = directory / "broken.joblib"
    joblib.dump(payload, broken)

    with pytest.raises(ArtifactError, match="rounds but the estimator"):
        load_model(broken, verify=False)


def test_an_unknown_format_version_is_refused(saved) -> None:
    import joblib

    directory, record, _, _ = saved
    payload = joblib.load(directory / record["filename"])
    payload["format_version"] = ARTIFACT_FORMAT_VERSION + 1
    future = directory / "future.joblib"
    joblib.dump(payload, future)

    with pytest.raises(ArtifactError, match="format version"):
        load_model(future, verify=False)


def test_a_file_that_changed_under_its_sidecar_is_refused(saved, tmp_path) -> None:
    directory, record, _, _ = saved
    sidecar = tmp_path / "metrics.json"
    artefact = tmp_path / record["filename"]
    artefact.write_bytes((directory / record["filename"]).read_bytes())
    sidecar.write_text(
        json.dumps(
            {
                "model": {
                    "recommended_variant": "unweighted",
                    "artifacts": {"unweighted": {**record, "sha256": "0" * 64}},
                }
            }
        )
    )
    with pytest.raises(ArtifactError, match="does not match the sidecar"):
        load_model(metrics=sidecar)


def test_a_missing_artefact_says_how_to_rebuild_it(tmp_path) -> None:
    sidecar = tmp_path / "metrics.json"
    sidecar.write_text(
        json.dumps(
            {
                "model": {
                    "recommended_variant": "unweighted",
                    "artifacts": {
                        "unweighted": {"filename": "gone.joblib", "sha256": "x"}
                    },
                }
            }
        )
    )
    with pytest.raises(ArtifactError, match="train.py --write"):
        load_model(metrics=sidecar)

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"model": {}}))
    with pytest.raises(ArtifactError, match="records no artefacts"):
        load_model(metrics=empty)


# --------------------------------------------------------------------------
# The size gate
# --------------------------------------------------------------------------


def test_every_tracked_model_is_small_enough_to_belong_in_git() -> None:
    """git cannot say "only while it is small", so this does.

    Committing the model is deliberate: Streamlit Community Cloud cannot reach
    the warehouse it is trained from, so an artefact outside the repository
    means the deployed dashboard has no model. Past the threshold that trade
    stops paying — git history is forever and a binary does not diff — and the
    file belongs in a release asset instead.
    """
    oversized = [
        f"{path.relative_to(REPO_ROOT)}: {path.stat().st_size:,} bytes"
        for path in tracked_artifacts(REPO_ROOT)
        if path.exists() and path.stat().st_size > MAX_COMMITTED_BYTES
    ]
    assert not oversized, (
        f"tracked artefacts over {MAX_COMMITTED_BYTES:,} bytes: {oversized}"
    )


def test_only_versioned_artefacts_are_tracked() -> None:
    """An unversioned ``model.joblib`` would be overwritten in place, and its
    history would be a sequence of indistinguishable binaries."""
    for path in tracked_artifacts(REPO_ROOT):
        assert path.name.startswith("model-"), path
        assert f"-v{ARTIFACT_FORMAT_VERSION}-" in path.name, path


# --------------------------------------------------------------------------
# Against the committed sidecar
# --------------------------------------------------------------------------


def test_the_committed_sidecar_and_artefacts_agree() -> None:
    path = metrics_path()
    if not path.exists():
        pytest.skip("metrics.json not written yet")
    payload = json.loads(path.read_text())
    artefacts = payload.get("model", {}).get("artifacts")
    if not artefacts:
        pytest.skip("no artefacts recorded yet; run train.py --write")

    recommended = payload["model"]["recommended_variant"]
    assert artefacts[recommended]["recommended"] is True
    assert sum(entry["recommended"] for entry in artefacts.values()) == 1

    for variant, record in artefacts.items():
        beside = path.parent / record["filename"]
        if not beside.exists():
            pytest.skip(f"{record['filename']} not on disk; run train.py --write")
        assert record["variant"] == variant
        assert beside.stat().st_size == record["bytes"]
        assert hashlib.sha256(beside.read_bytes()).hexdigest() == record["sha256"]
        assert record["features"] == list(feature_columns())


def test_the_recommended_model_loads_from_the_sidecar_alone() -> None:
    """No "latest" pointer to go stale: metrics.json is the index."""
    if not metrics_path().exists():
        pytest.skip("metrics.json not written yet")
    payload = json.loads(metrics_path().read_text())
    if not payload.get("model", {}).get("artifacts"):
        pytest.skip("no artefacts recorded yet; run train.py --write")

    model = load_model()
    assert model.metadata["recommended"] is True
    assert model.features == tuple(feature_columns())
    assert model.path.exists()
