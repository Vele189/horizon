"""Writes and reads the model artefact, with the metadata that makes it usable.

A ``.joblib`` with no record of what it was trained on is a file nobody can
audit and nobody should deploy. Six months on, "which features, in what order,
over what window, scoring what against which baseline" are not recoverable from
the pickle — the estimator will happily accept a matrix with the columns in the
wrong order and return confident nonsense.

So every artefact is written with a **fingerprint** in its own filename and a
full record in ``metrics.json``:

* the **training window** and its row and positive counts;
* the **feature list, in order** — the order is the part that matters, because
  it is the part a caller can get wrong silently;
* the hyperparameters, the seed, and the number of rounds early stopping kept;
* the metrics on every split, and the baselines they are measured against;
* the library versions, the git commit, and whether the tree was dirty;
* the file's own SHA-256 and size.

**The fingerprint is derived, not stamped.** It hashes exactly the things that
determine the model — window, features, hyperparameters, seed, library version,
data snapshot — so two artefacts with the same name *are* the same model and a
retrain that changes nothing produces no diff. A timestamp would change when
nothing had.

**On the git hash.** It records the commit the working tree was on when the
model was trained, and whether that tree was clean. It cannot be the commit
that contains the model: the artefact has to exist before it can be committed.
``git_dirty: true`` in a committed sidecar is the normal case and is recorded
rather than hidden — the alternative is a hash that implies a provenance it
does not have.

**On the tree range.** The estimator stops early, so the booster carries 68
trees and the model uses 18. ``predict_proba`` knows that through the
``best_iteration`` it was pickled with; :func:`load_model` asserts it survived
the round trip, because a pickle that lost it scores with every tree and is a
different model wearing the same name.

Usage::

    from machine_learning.artifact import load_model

    model = load_model()               # the recommended variant
    risk = model.predict(frame)        # validates the features first
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings  # noqa: E402

__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "MAX_COMMITTED_BYTES",
    "ArtifactError",
    "LoadedModel",
    "artifact_dir",
    "artifact_name",
    "fingerprint",
    "git_provenance",
    "load_model",
    "save_artifact",
    "tracked_artifacts",
]

#: Bumped when the *shape* of what is pickled changes, so a loader can refuse a
#: file it does not understand instead of unpickling it and hoping. It is in
#: the filename as well as in the payload: a directory listing should be enough
#: to see that two artefacts are not the same kind of thing.
ARTIFACT_FORMAT_VERSION: Final[int] = 1

#: The size at which a model stops being something to commit.
#:
#: Committing it at all is deliberate. Streamlit Community Cloud cannot reach
#: the warehouse this model is trained from, so an artefact outside the
#: repository means the deployed dashboard has no model — and a 271 KB file
#: that makes a clone runnable is worth more than the tidiness of an empty
#: artifacts directory. Two megabytes is where that stops being true: git
#: history is forever and a binary does not diff, so a model that large belongs
#: in a release asset instead. A test enforces it on tracked files.
MAX_COMMITTED_BYTES: Final[int] = 2 * 1024 * 1024


class ArtifactError(RuntimeError):
    """Raised when an artefact cannot be trusted to be what it claims.

    Every case is one where continuing would produce a confident number: a
    feature list that does not match, a file whose contents have changed under
    its fingerprint, or an estimator that lost the tree range it was scored on.
    """


def artifact_dir() -> Path:
    return get_settings().model_artifact_dir


def fingerprint(payload: Mapping[str, Any]) -> str:
    """A short, deterministic id for the inputs that determine a model.

    Over a canonical JSON encoding, so key order in the caller cannot change
    the answer. Twelve hex characters is 48 bits — ample for telling apart the
    handful of models a project like this produces, and short enough to read
    out of a filename.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


def artifact_name(variant: str, digest: str) -> str:
    """``model-<variant>-v<format>-<fingerprint>.joblib``."""
    return f"model-{variant}-v{ARTIFACT_FORMAT_VERSION}-{digest}.joblib"


def git_provenance(root: Path | None = None) -> dict[str, Any]:
    """The commit the model was trained from, and whether the tree was clean.

    Every field is ``None`` outside a repository rather than absent, so a
    reader can tell "not a git checkout" from "this key is new".
    """
    root = root or Path(__file__).resolve().parent.parent

    def run(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *args],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "short_commit": commit[:12] if commit else None,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        # True when the tree had uncommitted changes at training time, which is
        # the normal case for a model committed in the same change that made
        # it. Recorded rather than hidden.
        "dirty": None if status is None else bool(status),
    }


def _library_versions() -> dict[str, str]:
    import joblib
    import sklearn
    import xgboost

    return {
        "xgboost": xgboost.__version__,
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }


def _training_window(train: pd.DataFrame, target: np.ndarray) -> dict[str, Any]:
    return {
        "start": str(train["date_key"].min().date()),
        "end": str(train["date_key"].max().date()),
        "rows": int(len(train)),
        "positives": int(target.sum()),
        "positive_rate": float(target.mean()),
        "cities": sorted(str(city) for city in train["city_id"].unique()),
    }


def save_artifact(
    fit,
    *,
    variant: str,
    train: pd.DataFrame,
    target: np.ndarray,
    metrics: Mapping[str, Any],
    baselines: Mapping[str, Any] | None = None,
    snapshot: Mapping[str, Any] | None = None,
    recommended: bool = False,
    directory: Path | None = None,
    prune: bool = True,
) -> dict[str, Any]:
    """Write one fitted variant and return the metadata describing it.

    Args:
        fit: The fitted variant, from ``train.py``.
        variant: ``"weighted"`` or ``"unweighted"``.
        train: The training split it was fitted on, for the window record.
        target: That split's labels, for the class counts.
        metrics: The variant's scores on every split.
        baselines: The committed baselines, so the sidecar carries the
            comparison rather than pointing at it.
        snapshot: The warehouse snapshot, which is part of the fingerprint —
            the same hyperparameters over different data are a different model.
        recommended: Whether this is the variant to deploy.
        directory: Where to write. Defaults to the configured artifacts dir.
        prune: Delete superseded artefacts of the same variant and format.
            Git history keeps them; a working tree accumulating one binary per
            retrain does not help anybody.

    Returns:
        The metadata block, including the filename and the file's own digest.
    """
    import joblib

    directory = Path(directory) if directory is not None else artifact_dir()
    directory.mkdir(parents=True, exist_ok=True)

    described = fit.describe()
    window = _training_window(train, target)
    identity = {
        "format": ARTIFACT_FORMAT_VERSION,
        "variant": variant,
        "features": list(fit.feature_names),
        "params": described["params"],
        "scale_pos_weight": fit.scale_pos_weight,
        "n_estimators": described["n_estimators"],
        "seed": described["seed"],
        "window": window,
        "libraries": _library_versions(),
        "snapshot": dict(snapshot) if snapshot else None,
    }
    digest = fingerprint(identity)
    destination = directory / artifact_name(variant, digest)

    payload = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "variant": variant,
        "estimator": fit.estimator,
        # Duplicated inside the pickle as well as in metrics.json, because a
        # loader handed a bare file must be able to validate features without
        # the sidecar beside it.
        "features": list(fit.feature_names),
        "n_estimators": described["n_estimators"],
        "fingerprint": digest,
    }
    joblib.dump(payload, destination)

    metadata: dict[str, Any] = {
        "filename": destination.name,
        "format_version": ARTIFACT_FORMAT_VERSION,
        "fingerprint": digest,
        "variant": variant,
        "recommended": recommended,
        "bytes": destination.stat().st_size,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "committable": destination.stat().st_size <= MAX_COMMITTED_BYTES,
        "training_window": window,
        "features": list(fit.feature_names),
        "feature_count": len(fit.feature_names),
        "hyperparameters": {
            **described["params"],
            "scale_pos_weight": fit.scale_pos_weight,
            "n_estimators": described["n_estimators"],
            "early_stopping_rounds": described["early_stopping_rounds"],
            "seed": described["seed"],
            "n_jobs": described["n_jobs"],
        },
        "metrics": {
            split: dict(metrics[split])
            for split in ("train", "validation", "test")
            if split in metrics
        },
        "baseline_comparison": _comparison(metrics, baselines),
        "libraries": _library_versions(),
        "git": git_provenance(),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }

    if prune:
        pattern = f"model-{variant}-v{ARTIFACT_FORMAT_VERSION}-*.joblib"
        for stale in directory.glob(pattern):
            if stale != destination:
                stale.unlink()
    return metadata


def _comparison(
    metrics: Mapping[str, Any], baselines: Mapping[str, Any] | None
) -> dict[str, Any]:
    """The model's test score against each baseline, in the sidecar itself.

    Carried rather than referenced. Somebody reading the model block to decide
    whether to deploy it should not have to hold two other blocks in their head
    to find out whether 0.3494 is good.
    """
    if not baselines or "test" not in metrics:
        return {}
    mine = metrics["test"]
    out: dict[str, Any] = {}
    for name, entry in baselines.items():
        theirs = entry.get("test", {})
        if not theirs:
            continue
        out[name] = {
            "pr_auc": theirs["pr_auc"],
            "pr_auc_delta": float(mine["pr_auc"] - theirs["pr_auc"]),
            "brier": theirs["brier"],
            # Lower is better for Brier, so a positive delta is an improvement.
            "brier_delta": float(theirs["brier"] - mine["brier"]),
            "beaten_on_pr_auc": bool(mine["pr_auc"] > theirs["pr_auc"]),
            "beaten_on_brier": bool(mine["brier"] < theirs["brier"]),
        }
    return out


@dataclass(frozen=True)
class LoadedModel:
    """A model read back from disk, with the metadata it must be used under."""

    estimator: Any
    features: tuple[str, ...]
    n_estimators: int
    fingerprint: str
    path: Path
    metadata: Mapping[str, Any] | None = None

    def design_matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Select the recorded features, in the recorded order, or refuse.

        Selecting *by name* is what makes the order safe rather than merely
        checked: a caller who hands over the right columns in the wrong order
        gets the right answer, and one who is missing a column gets an error
        naming it instead of a confident number from a shifted matrix.

        A bare array is refused outright. Its column order cannot be verified
        against anything, so accepting one would mean trusting the caller about
        the one thing this method exists to check.
        """
        if not isinstance(frame, pd.DataFrame):
            raise ArtifactError(
                f"expected a DataFrame, got {type(frame).__name__}. The column "
                "order of an array cannot be checked against the recorded "
                "feature list, and an order mismatch is silent: every value "
                "goes to the wrong tree and the model still returns a "
                "probability. Pass a frame with named columns."
            )

        missing = [name for name in self.features if name not in frame.columns]
        if missing:
            raise ArtifactError(
                f"{len(missing)} feature(s) missing from the frame: {missing}. "
                f"This model was trained on {len(self.features)} features; the "
                f"frame carries {len(frame.columns)} columns."
            )

        matrix = frame.loc[:, list(self.features)]
        if matrix.isna().to_numpy().any():
            nulls = matrix.columns[matrix.isna().any()].tolist()
            raise ArtifactError(
                f"the design matrix has nulls in {nulls}. The model was fitted "
                "on a population with none, so a null here is a row that "
                "should have been filtered rather than scored."
            )
        return matrix

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Probability of an anomaly in the forward window, per row."""
        return self.estimator.predict_proba(self.design_matrix(frame))[:, 1]

    def matches(self, features: Sequence[str]) -> bool:
        """Whether a feature list is this model's, order included."""
        return tuple(features) == self.features


def load_model(
    path: Path | None = None,
    *,
    variant: str | None = None,
    metrics: Path | None = None,
    verify: bool = True,
) -> LoadedModel:
    """Read an artefact back, checking it is the one the sidecar describes.

    With no arguments it reads ``metrics.json``, takes the recommended
    variant's filename, and loads that — so the sidecar is the index and there
    is no "latest" symlink to go stale.

    Args:
        path: Load this file directly, ignoring the sidecar.
        variant: Load this variant rather than the recommended one.
        metrics: An alternative ``metrics.json``.
        verify: Check the file's SHA-256 and fingerprint against the sidecar.
            Off only for a file loaded directly with no sidecar to check.

    Raises:
        ArtifactError: The sidecar has no artefact, the file is missing, its
            digest does not match, the format version is unknown, or the
            estimator lost the tree range it was scored on.
    """
    import joblib

    from machine_learning.baselines import metrics_path

    recorded: Mapping[str, Any] | None = None
    if path is None:
        sidecar = Path(metrics) if metrics is not None else metrics_path()
        if not sidecar.exists():
            raise ArtifactError(
                f"{sidecar} does not exist, so there is no record of which "
                "artefact to load. Run `python machine_learning/train.py "
                "--write`."
            )
        payload = json.loads(sidecar.read_text())
        artefacts = payload.get("model", {}).get("artifacts", {})
        if not artefacts:
            raise ArtifactError(
                f"{sidecar} records no artefacts. Run "
                "`python machine_learning/train.py --write`."
            )
        chosen = variant or payload["model"].get("recommended_variant")
        if chosen not in artefacts:
            raise ArtifactError(
                f"{sidecar} has no artefact for variant {chosen!r}; it has "
                f"{sorted(artefacts)}."
            )
        recorded = artefacts[chosen]
        path = sidecar.parent / recorded["filename"]

    path = Path(path)
    if not path.exists():
        raise ArtifactError(
            f"{path} is missing. metrics.json records it, so either it was "
            "never committed or the model needs retraining: "
            "`python machine_learning/train.py --write`."
        )

    if verify and recorded is not None:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != recorded["sha256"]:
            raise ArtifactError(
                f"{path.name} does not match the sidecar: recorded "
                f"{recorded['sha256'][:12]}, found {digest[:12]}. The file has "
                "changed since the metrics were written, so the scores beside "
                "it describe a different model."
            )

    payload = joblib.load(path)
    if payload.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ArtifactError(
            f"{path.name} is format version {payload.get('format_version')!r} "
            f"and this loader understands {ARTIFACT_FORMAT_VERSION}."
        )

    estimator = payload["estimator"]
    expected = int(payload["n_estimators"])
    effective = int(getattr(estimator, "best_iteration", expected - 1)) + 1
    if effective != expected:
        raise ArtifactError(
            f"{path.name} was scored on {expected} rounds but the estimator "
            f"reports {effective}. Early stopping keeps trees the model does "
            "not use — this booster holds them and has lost the range that "
            "excludes them, so it would score as a different model."
        )

    return LoadedModel(
        estimator=estimator,
        features=tuple(payload["features"]),
        n_estimators=expected,
        fingerprint=str(payload["fingerprint"]),
        path=path,
        metadata=recorded,
    )


def tracked_artifacts(root: Path | None = None) -> list[Path]:
    """Every ``.joblib`` git is tracking. For the size gate."""
    root = root or Path(__file__).resolve().parent.parent
    try:
        done = subprocess.run(
            ["git", "ls-files", "*.joblib"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if done.returncode != 0:
        return []
    return [root / line for line in done.stdout.split() if line]
