"""Trains the gradient-boosted classifier, against a target fixed in advance.

The model is the last thing built and the least interesting part of the
project. Everything that decides whether its score means anything was settled
earlier: the features cannot see forward, the label's boundary is pinned, the
split is chronological and purged, and the two baselines are already written
down in ``metrics.json``. What is left here is to fit a model and find out
whether it beats them.

So this module refuses to record a result unless the baselines it will be
compared against were computed on **the same warehouse snapshot**. A model
scored against a target that has since moved is not being measured, and
"fixed in advance" is only a property of a file if something enforces it.

**On ``scale_pos_weight``, and a finding worth reading before trusting a
probability.** The ticket asks for it in preference to resampling, on the
grounds that synthetic oversampling and undersampling distort the predicted
probabilities the Risk Horizon view shows a reader directly. The premise is
half right: those methods do distort probabilities. But ``scale_pos_weight``
is arithmetically the same operation — weighting the positive class by *k* is
oversampling it *k*-fold — so it distorts them the same way, for the same
reason. At the
observed ratio of 17.15 this model's mean predicted probability is 0.478
against a true test base rate of 0.136 — it tells the dashboard reader that
almost every other week is extreme.

Both are therefore trained and both are recorded. The weighted model is the one
the ticket specifies; the unweighted one is what the ticket's own stated goal
asks for, and on this data it is better on *both* axes rather than trading
ranking for calibration. The recommendation is in the README and in the run
output, and the numbers are in ``metrics.json`` so the choice is not a matter
of taking anyone's word.

Everything else is tuned on validation and nothing whatever is tuned on test:
:func:`tune` takes two frames and there is no third to pass it.

Usage::

    python machine_learning/train.py            # fit, score, print
    python machine_learning/train.py --write    # and update metrics.json

Reproducibility is a property this module is expected to have, not to claim.
The seed is fixed, the thread count is fixed at one, and
``tests/test_training.py`` runs the whole thing twice in separate processes and
requires identical metrics.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np
import pandas as pd
from sqlalchemy import Engine
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from machine_learning.artifact import (  # noqa: E402
    MAX_COMMITTED_BYTES,
    save_artifact,
)
from machine_learning.baselines import build_metrics, metrics_path  # noqa: E402
from machine_learning.evaluation import (  # noqa: E402
    SPLITS,
    Score,
    assert_splits_are_disjoint,
    evaluation_frame,
    score,
    split_frame,
)
from machine_learning.features import feature_columns  # noqa: E402
from machine_learning.labels import LABEL, positives  # noqa: E402

__all__ = [
    "EARLY_STOPPING_ROUNDS",
    "FIXED_PARAMS",
    "MAX_ROUNDS",
    "SEARCH_SPACE",
    "SEED",
    "TrainingError",
    "Fit",
    "fit_once",
    "save_models",
    "scale_pos_weight_from",
    "train_model",
    "training_matrix",
    "tune",
]

log = logging.getLogger(__name__)

#: The one seed. Passed to every estimator, and asserted by a test that runs
#: two fits in separate processes and compares the metrics byte for byte.
SEED: Final[int] = 42

#: Fixed at one thread, deliberately.
#:
#: XGBoost's histogram builder is deterministic for a given thread count, not
#: across thread counts: the per-thread gradient sums are added in whatever
#: order the threads finish, and floating-point addition is not associative. On
#: a four-core laptop and a sixteen-core runner that is two different models,
#: differing in the last digits and occasionally in a split. The dataset is
#: 45 069 rows by 27 columns and the whole search takes seconds, so a thread
#: count that does not depend on the machine costs nothing worth having.
N_JOBS: Final[int] = 1

MAX_ROUNDS: Final[int] = 2000
EARLY_STOPPING_ROUNDS: Final[int] = 50

#: Held constant across the search. Subsampling is on because the positive
#: class is 5.5% of the training rows and unsubsampled trees memorise it.
FIXED_PARAMS: Final[Mapping[str, Any]] = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "n_jobs": N_JOBS,
}

#: The grid, kept small on purpose. Twelve combinations against 5 445
#: validation rows holding 464 positives: a larger search would be choosing
#: between differences smaller than the noise in the number it is choosing on,
#: which is how a validation split gets overfitted without anyone touching test.
SEARCH_SPACE: Final[Mapping[str, tuple]] = {
    "max_depth": (3, 4, 6),
    "learning_rate": (0.03, 0.1),
    "min_child_weight": (1, 10),
}


class TrainingError(RuntimeError):
    """Raised when a model cannot honestly be recorded.

    Chiefly: the committed baselines were computed against a different
    warehouse snapshot, so there is no fixed target to compare against.
    """


@dataclass
class Fit:
    """One fitted model and everything needed to describe it later."""

    estimator: XGBClassifier
    params: dict[str, Any]
    scale_pos_weight: float
    best_iteration: int
    feature_names: tuple[str, ...]
    validation: Score | None = None
    search: list[dict[str, Any]] = field(default_factory=list)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        matrix, _ = training_matrix(frame)
        return self.estimator.predict_proba(matrix)[:, 1]

    def importances(self, top: int = 10) -> list[dict[str, Any]]:
        gains = self.estimator.feature_importances_
        order = np.argsort(gains)[::-1][:top]
        return [
            {"feature": self.feature_names[index], "gain": float(gains[index])}
            for index in order
        ]

    def describe(self) -> dict[str, Any]:
        return {
            "params": dict(sorted(self.params.items())),
            "scale_pos_weight": self.scale_pos_weight,
            "n_estimators": self.best_iteration + 1,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "max_rounds": MAX_ROUNDS,
            "seed": SEED,
            "n_jobs": N_JOBS,
            "features": list(self.feature_names),
            "feature_count": len(self.feature_names),
            "top_importances": self.importances(),
            "validation_search": self.search,
        }


def training_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """The design matrix and the target, in a fixed column order.

    Selected by :func:`~machine_learning.features.feature_columns` and never by
    dropping the keys. ``is_anomaly`` and the label sit in the same frame, and
    a model handed either of them would score beautifully and mean nothing.
    """
    columns = list(feature_columns())
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise TrainingError(f"the frame is missing {missing}.")
    matrix = frame.loc[:, columns]
    if matrix.isna().to_numpy().any():
        raise TrainingError(
            "the design matrix has nulls. The scored population is supposed to "
            "have none — see evaluation_frame() — so this means the population "
            "was built some other way."
        )
    return matrix, positives(frame[LABEL]).to_numpy()


def scale_pos_weight_from(target: np.ndarray) -> float:
    """Negatives over positives, from the training split's own class ratio."""
    positive = int(target.sum())
    if positive == 0:
        raise TrainingError("no positives in the training split.")
    return float((len(target) - positive) / positive)


def fit_once(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    params: Mapping[str, Any],
    *,
    scale_pos_weight: float,
) -> Fit:
    """Fit one estimator, stopping early on validation average precision.

    Early stopping reads validation, which is what validation is for. Test is
    not passed to this function and cannot be: it takes two frames.
    """
    x_train, y_train = training_matrix(train)
    x_validation, y_validation = training_matrix(validation)

    estimator = XGBClassifier(
        **FIXED_PARAMS,
        **params,
        n_estimators=MAX_ROUNDS,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    estimator.fit(
        x_train, y_train, eval_set=[(x_validation, y_validation)], verbose=False
    )
    return Fit(
        estimator=estimator,
        params=dict(params),
        scale_pos_weight=scale_pos_weight,
        best_iteration=int(estimator.best_iteration),
        feature_names=tuple(x_train.columns),
    )


def tune(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    scale_pos_weight: float,
) -> Fit:
    """Search the grid, keep the best validation PR-AUC. Test is never seen.

    PR-AUC rather than Brier, because ranking is what the search is for and
    calibration is decided by ``scale_pos_weight``, which is not in the grid.
    Ties break towards the earlier combination, so the choice is deterministic
    and the grid order is the tie-break rule.
    """
    names = list(SEARCH_SPACE)
    search: list[dict[str, Any]] = []
    best: Fit | None = None
    best_key: tuple[float, int] | None = None

    for index, values in enumerate(product(*(SEARCH_SPACE[name] for name in names))):
        params = dict(zip(names, values))
        candidate = fit_once(
            train, validation, params, scale_pos_weight=scale_pos_weight
        )
        result = score(validation[LABEL], candidate.predict(validation))
        candidate.validation = result
        search.append(
            {
                **params,
                "n_estimators": candidate.best_iteration + 1,
                "validation_pr_auc": result.pr_auc,
                "validation_brier": result.brier,
            }
        )
        key = (-result.pr_auc, index)
        if best_key is None or key < best_key:
            best, best_key = candidate, key

    assert best is not None
    best.search = search
    return best


def save_models(
    fits: Mapping[str, "Fit"],
    block: Mapping[str, Any],
    parts: Mapping[str, pd.DataFrame],
    *,
    baselines: Mapping[str, Any] | None = None,
    snapshot: Mapping[str, Any] | None = None,
    directory: Path | None = None,
) -> dict[str, Any]:
    """Persist **both** variants, each with the metadata that makes it usable.

    Both, because the two tickets that produced them disagree and neither is
    wrong: ML-05 specifies ``scale_pos_weight`` at the observed ratio, and
    ML-06 measures that variant losing to *doing nothing* on Brier. Saving only
    the specified one would put the model nobody should deploy on disk while
    the explanations describe a different one; saving only the recommended one
    would quietly overrule a ticket. The sidecar names which is which, and
    :func:`~machine_learning.artifact.load_model` defaults to the recommended.
    """
    _, y_train = training_matrix(parts["train"])
    recommended = block.get("recommended_variant")
    saved: dict[str, Any] = {}
    for variant, fit in fits.items():
        saved[variant] = save_artifact(
            fit,
            variant=variant,
            train=parts["train"],
            target=y_train,
            metrics=block["variants"][variant],
            baselines=baselines,
            snapshot=snapshot,
            recommended=variant == recommended,
            directory=directory,
        )
    return saved


def train_model(
    engine: Engine | None = None,
    *,
    frame: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], dict[str, Fit]]:
    """Fit both variants, score them on every split, and describe them.

    Returns the ``model`` block for ``metrics.json`` and both fitted models by
    variant name. Both, because ML-06 draws curves for each and the difference
    between them is the finding: ``variants["weighted"]`` is the one the ticket
    specifies and the one saved, ``variants["unweighted"]`` is the one its
    stated goal asks for.
    """
    population = frame if frame is not None else evaluation_frame(engine)
    population = population.sort_values(
        ["city_id", "date_key"], kind="stable"
    ).reset_index(drop=True)
    parts = split_frame(population)
    assert_splits_are_disjoint(parts)
    train, validation = parts["train"], parts["validation"]
    for name, part in parts.items():
        if part.empty:
            raise TrainingError(f"the {name} split is empty; nothing to train on.")

    _, y_train = training_matrix(train)
    observed = scale_pos_weight_from(y_train)

    variants = {
        "weighted": observed,
        "unweighted": 1.0,
    }
    block: dict[str, Any] = {
        "estimator": "xgboost.XGBClassifier",
        "xgboost_version": __import__("xgboost").__version__,
        "fitted_on": "train",
        "tuned_on": "validation",
        "resampling": "none",
        "scale_pos_weight_observed": observed,
        "specified_variant": "weighted",
        "recommended_variant": None,
        "variants": {},
    }

    fits: dict[str, Fit] = {}
    for variant, weight in variants.items():
        fit = tune(train, validation, scale_pos_weight=weight)
        fits[variant] = fit
        entry = fit.describe()
        for split_name in ("train", "validation", "test"):
            part = parts[split_name]
            predicted = fit.predict(part)
            entry[split_name] = score(part[LABEL], predicted).as_dict()
            entry[split_name]["mean_predicted"] = float(predicted.mean())
        block["variants"][variant] = entry

    # Recommended on validation, never on test — the same rule the search
    # follows. Better PR-AUC *and* better Brier, or the specified one stands.
    specified = block["variants"]["weighted"]["validation"]
    other = block["variants"]["unweighted"]["validation"]
    better = (
        other["pr_auc"] > specified["pr_auc"] and other["brier"] < specified["brier"]
    )
    block["recommended_variant"] = "unweighted" if better else "weighted"
    return block, fits


def merge_into_metrics(
    block: Mapping[str, Any],
    *,
    engine: Engine | None = None,
    frame: pd.DataFrame | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Put the model block beside the baselines, or refuse to.

    The refusal is the point. ``metrics.json`` holds baseline scores that were
    committed before this model existed, and they are only a fixed target while
    they describe the same data. If the warehouse has moved on — a city
    backfilled, a day landed — the baselines have to be re-run and re-committed
    first, and a model recorded against the stale ones would be reporting a
    comparison nobody made.
    """
    destination = Path(path) if path is not None else metrics_path()
    if not destination.exists():
        raise TrainingError(
            f"{destination} does not exist. Run "
            "`python machine_learning/baselines.py --write` and commit it "
            "before training: the target is fixed in advance or it is not "
            "fixed at all."
        )
    committed = json.loads(destination.read_text())
    fresh = build_metrics(engine, frame=frame)
    if committed.get("snapshot") != fresh["snapshot"]:
        raise TrainingError(
            "the committed baselines were computed against "
            f"{committed['snapshot']['rows']} rows to "
            f"{committed['snapshot']['last_date']}; this run saw "
            f"{fresh['snapshot']['rows']} to {fresh['snapshot']['last_date']}. "
            "Re-run baselines.py --write, commit it, and train again."
        )

    merged = dict(committed)
    merged["model"] = dict(block)
    merged["model"]["recorded_at"] = dt.datetime.now(dt.timezone.utc).isoformat(
        timespec="seconds"
    )
    return merged


def _split_window(name: str) -> str:
    split = next(item for item in SPLITS if item.name == name)
    return f"{split.start or 'start'} .. {split.end or 'end of record'}"


def _row(label: str, entry: Mapping[str, Any]) -> str:
    return (
        f"  {label:<24} PR-AUC {entry['pr_auc']:.4f}  "
        f"lift {entry['lift']:.2f}x  Brier {entry['brier']:.5f}  "
        f"mean p {entry['mean_predicted']:.4f}"
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the XGBoost classifier.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Merge the model block into metrics.json and save the artefact.",
    )
    parser.add_argument("--out", help="Write metrics somewhere other than the default.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    population = evaluation_frame()
    block, fits = train_model(frame=population)

    print(f"train      {_split_window('train')}")
    print(f"seed {SEED}   threads {N_JOBS}   resampling {block['resampling']}")
    print(f"scale_pos_weight from the observed ratio: "
          f"{block['scale_pos_weight_observed']:.3f}\n")

    committed = {}
    if metrics_path().exists():
        committed = json.loads(metrics_path().read_text()).get("baselines", {})
    for name, entry in committed.items():
        row = dict(entry["test"])
        row["mean_predicted"] = float("nan")
        print(
            f"  {name + ' (baseline)':<24} PR-AUC {row['pr_auc']:.4f}  "
            f"lift {row['lift']:.2f}x  Brier {row['brier']:.5f}"
        )
    print()
    for variant, entry in block["variants"].items():
        print(_row(f"model, {variant}", entry["test"]))

    recommended = block["recommended_variant"]
    print("\n  specified:   weighted (scale_pos_weight = observed ratio)")
    print(f"  recommended: {recommended}   — chosen on validation, never on test")
    if recommended != block["specified_variant"]:
        weighted = block["variants"]["weighted"]["test"]
        print(
            "\n  scale_pos_weight is oversampling the positive class by "
            f"{block['scale_pos_weight_observed']:.1f}x, so the weighted model "
            f"predicts a mean probability of {weighted['mean_predicted']:.3f} "
            f"against a true base rate of {weighted['base_rate']:.3f}. The Risk "
            "Horizon view shows these numbers to a reader directly."
        )

    if args.write or args.out:
        destination = Path(args.out) if args.out else metrics_path()
        merged = merge_into_metrics(block, frame=population, path=destination)
        merged["model"]["artifacts"] = save_models(
            fits,
            merged["model"],
            split_frame(population),
            baselines=merged.get("baselines"),
            snapshot=merged.get("snapshot"),
            directory=destination.parent,
        )
        destination.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {destination}")
        for variant, record in merged["model"]["artifacts"].items():
            mark = " (recommended)" if record["recommended"] else ""
            print(
                f"wrote {destination.parent / record['filename']}  "
                f"{record['bytes']:,} bytes{mark}"
            )
            if not record["committable"]:
                print(
                    f"  WARNING: over {MAX_COMMITTED_BYTES:,} bytes — too "
                    "large to commit; it belongs in a release asset."
                )
    else:
        print("\n(not written — pass --write to update metrics.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
