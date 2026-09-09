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
is arithmetically the same operation, since weighting the positive class by
*k* is oversampling it *k*-fold, so it distorts them the same way, for the same
reason. At the
observed ratio of 18.9 this model's mean predicted probability is 0.444
against a true test base rate of 0.115: it tells the dashboard reader that
almost every other week is extreme.

Both are therefore trained and both are recorded. The weighted model is the one
the ticket specifies; the unweighted one is what the ticket's own stated goal
asks for, and on this data it is better on *both* axes rather than trading
ranking for calibration. The recommendation is in the README and in the run
output, and the numbers are in ``metrics.json`` so the choice is not a matter
of taking anyone's word.

**And the recommendation was right but incomplete (ML-10).** The unweighted
model is miscalibrated too, in the other direction: it predicts 0.066 where
0.115 occurs, because it was fitted where positives are 5% of rows and scored
where they are 11.5%. :func:`calibration_report` measures what two corrections
do about that. Isotonic regression fitted on validation halves the calibration
error and is worth shipping. Prior-shift correction on top of it -- estimating
the target period's class prior by EM over the model's own posteriors, with no
labels -- is the method that ought to handle a shift of exactly this kind, and
on this data its estimator overshoots by a factor of two and a half. Both are
recorded, and so is the oracle that shows the correction is sound and only the
estimate is not.

Everything else is tuned on validation and nothing whatever is tuned on test:
:func:`tune` takes two frames and there is no third to pass it, and
:func:`fit_calibrator` takes one.

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
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from machine_learning.artifact import (  # noqa: E402
    MAX_COMMITTED_BYTES,
    save_artifact,
)
from machine_learning.baselines import build_metrics, metrics_path  # noqa: E402
from machine_learning.evaluation import (  # noqa: E402
    ALERT_BUDGET_PER_CITY_YEAR,
    CALIBRATION_BINS,
    SPLITS,
    Score,
    alerts_per_city_year,
    apply_prior_shift,
    assert_splits_are_disjoint,
    base_rate,
    best_threshold,
    decision_table,
    estimate_prior,
    evaluation_frame,
    expected_calibration_error,
    implied_cost_ratio,
    reliability_points,
    score,
    split_frame,
    threshold_for_budget,
)
from machine_learning.features import feature_columns, gold_frame  # noqa: E402
from machine_learning.labels import (  # noqa: E402
    HAZARD_DAY,
    HAZARD_EVENT,
    HORIZON_DAYS,
    LABEL,
    compose_weekly,
    person_periods,
    positives,
)

__all__ = [
    "CALIBRATION_METHOD",
    "RECENCY_HALF_LIVES",
    "EARLY_STOPPING_ROUNDS",
    "FIXED_PARAMS",
    "MAX_ROUNDS",
    "SEARCH_SPACE",
    "SEED",
    "TrainingError",
    "Fit",
    "HAZARD_FEATURES",
    "HazardFit",
    "calibration_report",
    "decision_report",
    "hazard_periods",
    "hazard_report",
    "recency_ablation",
    "recency_weights",
    "tune_hazard",
    "fit_calibrator",
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


#: Half-lives tried for the recency weight, in years, tuned on validation.
#:
#: The grid runs to infinity on purpose, the same way the climatology's
#: shrinkage grid does and for the same reason: infinity *is* the unweighted
#: fit, so the search is offered the null hypothesis as one of its options. A
#: tuned parameter that cannot choose "do nothing" is not a tuned parameter, and
#: a table whose best row is the edge of the grid is a clipped result rather
#: than a chosen one.
#:
#: One year to sixteen, over a training period of twenty-four. Below a year the
#: effective sample is a season or two of a single seasonal cycle, which is not
#: a training set; above sixteen the weight on the oldest row is more than a
#: third and the fit is barely distinguishable from the unweighted one, which
#: the infinity row already represents exactly.
RECENCY_HALF_LIVES: Final[tuple[float, ...]] = (
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
    float("inf"),
)


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
    half_life_years: float = float("inf")
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
            "recency_half_life_years": _jsonable(self.half_life_years),
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


def training_matrix(
    frame: pd.DataFrame, columns: Sequence[str] | None = None
) -> tuple[pd.DataFrame, np.ndarray]:
    """The design matrix and the target, in a fixed column order.

    Selected by :func:`~machine_learning.features.feature_columns` and never by
    dropping the keys. ``is_anomaly`` and the label sit in the same frame, and
    a model handed either of them would score beautifully and mean nothing.

    ``columns`` overrides the selection, and exists for ING-04's ablation,
    which needs the same fit over a different column list. It is a parameter
    rather than a flag because an ablation removing a block is the same
    operation as an ablation adding one, and both should go through here so
    neither can quietly reach the keys.
    """
    columns = list(columns) if columns is not None else list(feature_columns())
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise TrainingError(f"the frame is missing {missing}.")
    matrix = frame.loc[:, columns]
    if matrix.isna().to_numpy().any():
        raise TrainingError(
            "the design matrix has nulls. The scored population is supposed to "
            "have none (see evaluation_frame()), so this means the population "
            "was built some other way."
        )
    return matrix, positives(frame[LABEL]).to_numpy()


def _jsonable(value: Any) -> Any:
    """Render infinity as a string, because JSON has no word for it.

    The same treatment ``baselines.py`` gives its shrinkage grid, and for the
    same reason: the half-life grid genuinely runs to the limit, and
    ``json.dumps`` writes a bare ``Infinity`` that every strict parser rejects.
    """
    if isinstance(value, float) and np.isinf(value):
        return "inf"
    return value


def recency_weights(train: pd.DataFrame, half_life_years: float) -> np.ndarray | None:
    """Exponential decay in a row's age, normalised to mean one.

    ML-12's whole mechanism. The label's base rate rises 2.3x from the training
    period to the test one, and DBT-12 established that detrending the
    climatology removes almost none of that -- the drift is real rather than an
    artefact. This is the other half of the response: if the world the model is
    scored in is not the world it was fitted in, weight the fit towards the
    part of the record that resembles it.

    Age is measured from the **last training row**, not from today. Today moves
    with the wall clock, which would make two runs of the same commit produce
    different models, and the training split's own end is the boundary the
    weight is really about.

    **Normalised to mean one, and that is not cosmetic.** XGBoost's
    ``min_child_weight`` is a floor on the summed hessian in a leaf, which
    scales with the sample weights, so an unnormalised decay would silently
    make the same grid value a different constraint at every half-life -- the
    search would be comparing regularisation strengths while believing it was
    comparing half-lives.

    Returns:
        ``None`` at an infinite half-life, which is XGBoost's own way of
        spelling "no weights" and keeps the unweighted arm of the ablation
        bit-identical to every fit this project has made until now. A vector of
        ones would be arithmetically the same and is not the same code path.
    """
    if half_life_years <= 0:
        raise ValueError(
            f"a half-life must be positive, got {half_life_years}. Zero would "
            "put all the weight on the last day of the training split."
        )
    if np.isinf(half_life_years):
        return None
    age_days = (train["date_key"].max() - train["date_key"]).dt.days.to_numpy()
    weights = 0.5 ** (age_days / (half_life_years * 365.25))
    return weights / weights.mean()


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
    half_life_years: float = float("inf"),
    columns: Sequence[str] | None = None,
) -> Fit:
    """Fit one estimator, stopping early on validation average precision.

    Early stopping reads validation, which is what validation is for. Test is
    not passed to this function and cannot be: it takes two frames.

    ``half_life_years`` weights the training rows by recency; infinity, the
    default, weights them equally and is exactly the fit every other ticket in
    this project has been using.
    """
    x_train, y_train = training_matrix(train, columns)
    x_validation, y_validation = training_matrix(validation, columns)

    estimator = XGBClassifier(
        **FIXED_PARAMS,
        **params,
        n_estimators=MAX_ROUNDS,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    estimator.fit(
        x_train,
        y_train,
        sample_weight=recency_weights(train, half_life_years),
        eval_set=[(x_validation, y_validation)],
        verbose=False,
    )
    return Fit(
        estimator=estimator,
        params=dict(params),
        scale_pos_weight=scale_pos_weight,
        best_iteration=int(estimator.best_iteration),
        feature_names=tuple(x_train.columns),
        half_life_years=half_life_years,
    )


def tune(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    scale_pos_weight: float,
    half_life_years: float = float("inf"),
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
            train,
            validation,
            params,
            scale_pos_weight=scale_pos_weight,
            half_life_years=half_life_years,
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


# --------------------------------------------------------------------------
# Discrete-time hazard (ML-13)
# --------------------------------------------------------------------------

#: The hazard model's design matrix: the weekly model's features, plus which
#: day of the horizon the row is about.
#:
#: ``horizon_day`` as a *covariate* is what makes this one model rather than
#: seven. Seven separate fits would divide the positives seven ways and each
#: would be estimating its own copy of the same seasonal structure; one fit
#: with the period index shares everything except what genuinely differs
#: between days, which is the whole reason discrete-time hazard models are
#: written this way.
HAZARD_FEATURES: Final[tuple[str, ...]] = tuple(feature_columns()) + (HAZARD_DAY,)


@dataclass
class HazardFit:
    """A fitted hazard model, and the composition that turns it back into a week."""

    estimator: XGBClassifier
    params: dict[str, Any]
    best_iteration: int
    feature_names: tuple[str, ...]
    #: Always one. Carried because :func:`~machine_learning.artifact.save_artifact`
    #: fingerprints on it, and a hazard fitted with a class weight would be a
    #: different model that had to be distinguishable from this one.
    scale_pos_weight: float = 1.0
    search: list[dict[str, Any]] = field(default_factory=list)

    def predict(self, periods: pd.DataFrame) -> np.ndarray:
        """One hazard per person-period row."""
        return self.estimator.predict_proba(periods.loc[:, list(self.feature_names)])[
            :, 1
        ]

    def hazards(self, city_days: pd.DataFrame) -> np.ndarray:
        """Seven hazards per city-day, as a ``(rows, 7)`` array.

        **No at-risk filtering here, and that is the point.** The training
        frame keeps a city-day only until its first anomaly, because a hazard is
        conditional on having survived. At prediction time nothing has happened
        yet -- the week has not occurred -- so all seven days are asked for, and
        the conditioning is expressed by the composition rather than by dropping
        rows.
        """
        wide = np.empty((len(city_days), HORIZON_DAYS), dtype=float)
        for offset in range(1, HORIZON_DAYS + 1):
            day = city_days.assign(**{HAZARD_DAY: offset})
            wide[:, offset - 1] = self.predict(day)
        return wide

    def weekly(self, city_days: pd.DataFrame) -> np.ndarray:
        """The weekly probability the seven hazards compose to."""
        return compose_weekly(self.hazards(city_days))

    def describe(self) -> dict[str, Any]:
        gains = self.estimator.feature_importances_
        order = np.argsort(gains)[::-1][:10]
        return {
            "params": dict(sorted(self.params.items())),
            "n_estimators": self.best_iteration + 1,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "max_rounds": MAX_ROUNDS,
            "seed": SEED,
            "n_jobs": N_JOBS,
            "scale_pos_weight": self.scale_pos_weight,
            "features": list(self.feature_names),
            "feature_count": len(self.feature_names),
            "top_importances": [
                {"feature": self.feature_names[index], "gain": float(gains[index])}
                for index in order
            ],
            "validation_search": self.search,
        }


def hazard_periods(population: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Person-periods for the scored population, with their features attached.

    Reshaped from **gold** and joined onto the population, never reshaped from
    the population itself. The population has had its warm-up and its
    unlabellable rows removed, so its calendar has holes; asking "did anything
    happen on t+3" of a frame with holes in it would answer from whichever rows
    happened to survive the trim, and the answer would be quietly wrong rather
    than missing.
    """
    periods = person_periods(gold)
    merged = periods.merge(population, on=["city_id", "date_key"], how="inner")
    if len(merged) > HORIZON_DAYS * len(population):
        raise TrainingError(
            f"{len(merged)} person-periods from {len(population)} city-days is "
            f"more than {HORIZON_DAYS} each; the join changed the grain."
        )
    return merged.sort_values(
        ["city_id", "date_key", HAZARD_DAY], kind="stable"
    ).reset_index(drop=True)


def tune_hazard(train: pd.DataFrame, validation: pd.DataFrame) -> HazardFit:
    """Search the same grid, on the reshaped data, choosing on validation.

    The same grid as the weekly model, deliberately: this is meant to be an
    ablation of the *target's shape*, and re-tuning the search space at the same
    time would make any difference between the two unattributable.
    """
    names = list(SEARCH_SPACE)
    columns = list(HAZARD_FEATURES)
    x_train = train.loc[:, columns]
    y_train = train[HAZARD_EVENT].to_numpy(dtype=bool)
    x_validation = validation.loc[:, columns]
    y_validation = validation[HAZARD_EVENT].to_numpy(dtype=bool)

    search: list[dict[str, Any]] = []
    best: HazardFit | None = None
    best_key: tuple[float, int] | None = None

    for index, values in enumerate(product(*(SEARCH_SPACE[name] for name in names))):
        params = dict(zip(names, values))
        estimator = XGBClassifier(
            **FIXED_PARAMS,
            **params,
            n_estimators=MAX_ROUNDS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )
        estimator.fit(
            x_train, y_train, eval_set=[(x_validation, y_validation)], verbose=False
        )
        candidate = HazardFit(
            estimator=estimator,
            params=params,
            best_iteration=int(estimator.best_iteration),
            feature_names=tuple(columns),
        )
        # Chosen on the *composed weekly* score, not on the person-period one.
        # The product is what a reader is shown, and a hazard model that ranks
        # person-periods well but composes badly would win a search run on the
        # wrong quantity.
        weekly = score(
            validation.drop_duplicates(["city_id", "date_key"])[LABEL],
            candidate.weekly(validation.drop_duplicates(["city_id", "date_key"])),
        )
        search.append(
            {
                **params,
                "n_estimators": candidate.best_iteration + 1,
                "validation_weekly_pr_auc": weekly.pr_auc,
                "validation_weekly_brier": weekly.brier,
            }
        )
        key = (-weekly.pr_auc, index)
        if best_key is None or key < best_key:
            best, best_key = candidate, key

    assert best is not None
    best.search = search
    return best


def hazard_report(
    parts: Mapping[str, pd.DataFrame],
    periods: Mapping[str, pd.DataFrame],
    weekly_fit: "Fit",
    fitted: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit the hazard, compose it back into a week, and price the difference.

    The composition is the claim: the weekly probability falls out as
    ``1 - prod(1 - h_k)``, so the existing number is preserved rather than
    replaced. What the hazard adds is *resolution* -- seven numbers where there
    was one -- and the honest question is whether those seven differ enough from
    each other to be worth drawing.

    ``profile`` answers it. If the fitted hazards are near-uniform in *k*, then
    a seven-cell heatmap says exactly what the flat band already said, and the
    view should keep the band and say so. That is a legitimate outcome and it is
    reported as one rather than designed around.
    """
    fit = tune_hazard(periods["train"], periods["validation"])
    city_days = {name: parts[name] for name in ("validation", "test")}
    if fitted is not None:
        # Handed back so main() can persist it: predict.py needs the estimator
        # to write per-day rows, and refitting a twelve-point search at scoring
        # time is not a serving story.
        fitted["hazard"] = fit

    composed: dict[str, Any] = {}
    for name, frame in city_days.items():
        weekly = fit.weekly(frame)
        direct = weekly_fit.predict(frame)
        composed[name] = {
            "hazard": {
                **score(frame[LABEL], weekly).as_dict(),
                "mean_predicted": float(weekly.mean()),
            },
            "direct": {
                **score(frame[LABEL], direct).as_dict(),
                "mean_predicted": float(direct.mean()),
            },
            # The acceptance's "matches the direct model within tolerance". Two
            # models fitted on differently shaped data will not agree row for
            # row, so what is recorded is how far apart they are: the mean
            # absolute gap between the two weekly probabilities, and how nearly
            # they rank the same weeks.
            "mean_absolute_difference": float(np.abs(weekly - direct).mean()),
            "rank_correlation": float(
                pd.Series(weekly).corr(pd.Series(direct), method="spearman")
            ),
        }

    return {
        "target": "P(first anomaly on day t+k | none through t+k-1)",
        "composition": "weekly = 1 - prod(1 - h_k)",
        "horizon_days": HORIZON_DAYS,
        "split_by": "city-day, never person-period row",
        "rows": {name: int(len(frame)) for name, frame in periods.items()},
        "city_days": {
            name: int(frame.groupby(["city_id", "date_key"]).ngroups)
            for name, frame in periods.items()
        },
        "model": fit.describe(),
        "composed": composed,
        "profile": _hazard_profile(fit, parts["test"], periods["test"]),
    }


def _hazard_profile(
    fit: HazardFit, city_days: pd.DataFrame, periods: pd.DataFrame
) -> dict[str, Any]:
    """How many distinct days the model actually resolves. Not seven.

    The trap ML-13 names, measured, and it materialised in a sharper form than
    the ticket anticipated. The seven hazards are not *near*-uniform: within a
    city-day, days two through seven are **identical**, to the last bit. The
    median ratio of the largest to the smallest across those six is exactly
    1.000.

    ``horizon_day`` is the only column that varies across a city-day's seven
    rows, so the ensemble can only separate them by splitting on it -- and it
    splits once, between day one and the rest. Day one carries the persistence
    signal, an anomaly today making one tomorrow far likelier, and the model
    found nothing in the remaining six worth a second split.

    So the resolution earned is **two levels, not seven**: tomorrow, and the
    rest of the week. ``distinct_levels`` records that, and
    ``worth_drawing_per_day`` is false because seven cells drawn from two
    numbers is a chart claiming a resolution the model does not have -- which
    is the same objection ``risk_horizon.py`` raised against spreading the
    weekly score in the first place.
    """
    wide = fit.hazards(city_days)
    observed = periods.groupby(HAZARD_DAY)[HAZARD_EVENT].agg(["mean", "size"])
    ratios = wide.max(axis=1) / np.maximum(wide.min(axis=1), 1e-12)
    # And the same ratio with day 1 removed. The first day carries the
    # persistence signal -- an anomaly today makes one tomorrow far likelier --
    # and if it alone accounts for the spread then the honest chart is one
    # bright cell and a flat tail rather than a gradient across the week.
    tail = wide[:, 1:]
    tail_ratios = tail.max(axis=1) / np.maximum(tail.min(axis=1), 1e-12)
    return {
        "predicted_by_day": [
            {
                "horizon_day": day + 1,
                "mean_hazard": float(wide[:, day].mean()),
                "observed_hazard": float(observed["mean"].get(day + 1, float("nan"))),
                "at_risk_rows": int(observed["size"].get(day + 1, 0)),
            }
            for day in range(HORIZON_DAYS)
        ],
        "marginal_spread": float(
            wide.mean(axis=0).max() / max(wide.mean(axis=0).min(), 1e-12)
        ),
        "within_city_day_ratio": float(np.median(ratios)),
        "within_city_day_ratio_p90": float(np.quantile(ratios, 0.9)),
        "within_city_day_ratio_after_day_one": float(np.median(tail_ratios)),
        # How many genuinely different numbers a city-day's seven days hold.
        # Rounded before counting, because two hazards that differ in the
        # fifteenth decimal are the same number to any reader and to any
        # colour scale.
        "distinct_levels": int(
            np.median([len(np.unique(np.round(row, 9))) for row in wide])
        ),
        # The finding, evaluated rather than asserted. Seven cells drawn from
        # two distinct numbers claim a resolution the model does not have,
        # which is the objection risk_horizon.py raised against spreading the
        # weekly score in the first place.
        "worth_drawing_per_day": bool(
            np.median([len(np.unique(np.round(row, 9))) for row in wide])
            >= HORIZON_DAYS - 1
        ),
        # The shape BI-09 needs, stated as a share rather than as a grouping.
        # A fixed grouping like [[1], [2..7]] would be the wrong object: which
        # days share a level varies city-day by city-day, and days 2 and 3 do
        # differ for *some* of them. What is stable, and what a chart can be
        # built on, is that the tail is flat on almost every city-day.
        "share_with_flat_tail": float(
            np.mean(
                [len(np.unique(np.round(row[1:], 9))) == 1 for row in wide]
            )
        ),
        "share_with_day_one_highest": float(
            np.mean(wide[:, 0] > wide[:, 1:].max(axis=1))
        ),
        "share_with_day_one_lowest": float(
            np.mean(wide[:, 0] < wide[:, 1:].min(axis=1))
        ),
        # The finding, and the reason the two levels are worth drawing: which
        # of them is higher depends on the city-day, so a chart that fixed the
        # order would be wrong most of the time in one direction or the other.
        "by_todays_flag": _hazard_by_todays_flag(wide, city_days),
    }


def _hazard_by_todays_flag(wide: np.ndarray, city_days: pd.DataFrame) -> list[dict]:
    """The per-day profile, split on whether today itself was flagged.

    Where the two levels come from. On a city-day that is *currently*
    anomalous the hazard for tomorrow is an order of magnitude above the rest
    of the week and falls away sharply; on a quiet one it is slightly *below*
    the rest, because a week that has been ordinary so far still has six days
    left to go wrong and only one of them is tomorrow.

    Pooling the two hides both. The marginal profile shows day one at roughly
    twice day seven, which reads as a gentle decay and is not what either
    population does.
    """
    flagged = city_days["is_anomaly"].fillna(False).to_numpy(dtype=bool)
    return [
        {
            "today": label,
            "city_days": int(mask.sum()),
            "mean_hazard_by_day": [
                float(wide[mask, day].mean()) for day in range(wide.shape[1])
            ],
        }
        for label, mask in (("flagged", flagged), ("quiet", ~flagged))
        if mask.any()
    ]


# --------------------------------------------------------------------------
# Calibration under label shift (ML-10)
# --------------------------------------------------------------------------

#: Isotonic rather than Platt. Platt fits a two-parameter sigmoid, which
#: assumes the miscalibration has a sigmoid shape; the distortion here comes
#: from a class-weighting factor and a prior shift, and there is no reason it
#: should. Isotonic assumes only monotonicity, which is the one property that
#: must hold if the ranking is to be preserved, and the validation split has
#: 11 993 rows with 849 positives, which is enough for a step function not to
#: be fitting noise.
CALIBRATION_METHOD: Final[str] = "isotonic"


def fit_calibrator(fit: "Fit", validation: pd.DataFrame):
    """Fit the probability calibrator. On validation, and on nothing else.

    **It takes one frame.** There is no argument for a second, so this function
    cannot be handed the test split by a caller in a hurry, in the same way
    :func:`tune` cannot. ``tests/test_training.py`` makes the stronger check
    that a rewritten test period leaves the fitted calibrator identical, which
    is the property a signature can only suggest.

    Fitted on validation rather than on train because a calibrator fitted on
    the data the model was fitted on is calibrating against predictions the
    model has already memorised: the training-split probabilities are far too
    confident in the right direction, and the map learned from them would undo
    a distortion that only exists in sample.
    """
    truth = positives(validation[LABEL]).to_numpy()
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
        fit.predict(validation), truth
    )


def _calibration_entry(
    labels: pd.Series, predictions: np.ndarray, note: str
) -> dict[str, Any]:
    result = score(labels, predictions)
    points = reliability_points(labels, predictions, bins=CALIBRATION_BINS)
    return {
        "what": note,
        "mean_predicted": float(np.mean(predictions)),
        "expected_calibration_error": expected_calibration_error(
            labels, predictions, bins=CALIBRATION_BINS
        ),
        "brier": result.brier,
        # Carried so the claim that neither step reorders anything is checkable
        # from the file rather than only from the test that asserts it.
        "pr_auc": result.pr_auc,
        "reliability": points.to_dict("records"),
    }


def recency_ablation(
    parts: Mapping[str, pd.DataFrame],
    *,
    scale_pos_weight: float,
    baseline: "Fit",
) -> dict[str, Any]:
    """Tune the recency half-life on validation and price it against not doing it.

    ML-12. The cheapest available response to the drift: weight the training
    rows by recency so the fit is not dominated by a climate that no longer
    exists. It is deliberately the *fit* side of the same question DBT-12 asked
    of the *label*, and the two answers are meant to be read together -- how
    much of the drift is fixed by redefining the target, and how much by
    changing what the model pays attention to.

    The half-life is chosen on **validation** and on validation alone, by the
    same rule the class-weighting choice uses: better PR-AUC *and* better
    Brier, or the incumbent stands. Requiring both is what stops this trading
    away calibration to buy a hundredth of ranking, which is exactly the trade
    a recency weight is most likely to offer.

    ``mean_predicted`` is carried on every row of the ablation because it is
    where an effect would show first. The unweighted model predicts far below
    the rate that occurs, for the plain reason that it was fitted where the
    rate was half what it became; if recency weighting does anything at all,
    the first thing it should do is move that number.
    """
    train, validation = parts["train"], parts["validation"]
    search: list[dict[str, Any]] = []
    fits: dict[float, Fit] = {}

    for half_life in RECENCY_HALF_LIVES:
        candidate = (
            baseline
            if np.isinf(half_life)
            else tune(
                train,
                validation,
                scale_pos_weight=scale_pos_weight,
                half_life_years=half_life,
            )
        )
        fits[half_life] = candidate
        predicted = candidate.predict(validation)
        result = score(validation[LABEL], predicted)
        search.append(
            {
                "half_life_years": _jsonable(half_life),
                "effective_rows": _effective_rows(train, half_life),
                "oldest_row_weight": _jsonable(_oldest_weight(train, half_life)),
                "n_estimators": candidate.best_iteration + 1,
                "validation_pr_auc": result.pr_auc,
                "validation_brier": result.brier,
                "validation_mean_predicted": float(predicted.mean()),
            }
        )

    unweighted = next(
        row for row in search if row["half_life_years"] == "inf"
    )
    better = [
        row
        for row in search
        if row["half_life_years"] != "inf"
        and row["validation_pr_auc"] > unweighted["validation_pr_auc"]
        and row["validation_brier"] < unweighted["validation_brier"]
    ]
    chosen = (
        max(better, key=lambda row: row["validation_pr_auc"])["half_life_years"]
        if better
        else "inf"
    )
    chosen_life = float("inf") if chosen == "inf" else float(chosen)

    # The ablation compares the unweighted fit against the best *finite*
    # half-life, not against whatever the rule chose. When the rule chooses not
    # to weight -- which is what happens here -- comparing the choice with
    # itself prints the same row twice and hides the finding. What a reader
    # needs to see is the contrast: this is the fit, this is the best the
    # weighted alternative could manage, and this is the gap.
    contender = min(
        (row for row in search if row["half_life_years"] != "inf"),
        key=lambda row: -row["validation_pr_auc"],
    )["half_life_years"]
    arms = (
        ("unweighted", float("inf")),
        ("recency", chosen_life if chosen != "inf" else float(contender)),
    )
    ablation = []
    for label, half_life in arms:
        fit = fits[half_life]
        row: dict[str, Any] = {
            "arm": label,
            "half_life_years": _jsonable(half_life),
        }
        for split in ("validation", "test"):
            predicted = fit.predict(parts[split])
            result = score(parts[split][LABEL], predicted)
            row[split] = {
                **result.as_dict(),
                "mean_predicted": float(predicted.mean()),
            }
        ablation.append(row)

    return {
        "weighting": "exponential in the age of the row",
        "training_drift": _training_drift(parts),
        "measured_from": "the last day of the training split",
        "normalised": "to mean one, so min_child_weight means the same thing "
                      "at every half-life",
        "tuned_on": "validation",
        "rule": "better PR-AUC and better Brier, or the unweighted fit stands",
        "grid": [_jsonable(value) for value in RECENCY_HALF_LIVES],
        "search": search,
        "chosen_half_life_years": chosen,
        "helped": chosen != "inf",
        "best_finite_half_life_years": contender,
        "ablation": ablation,
        "gain_by_distance": _gain_by_distance(
            parts, fits[float("inf")], fits[float(contender)]
        ),
    }


def _gain_by_distance(
    parts: Mapping[str, pd.DataFrame], plain: "Fit", recent: "Fit"
) -> dict[str, Any]:
    """What recency weighting is worth as a function of how far away you score.

    **A diagnostic, and never a selection.** It reads the test period's labels,
    which is why it is reported after the choice has been made and can play no
    part in making it: the half-life is chosen on validation by
    :func:`recency_ablation` before this runs, and on this data that choice is
    "do not weight".

    It is here because the ablation alone would be misleading in a specific and
    expensive way. Recency weighting loses on validation and wins on test by
    about fifteen per cent, and a reader seeing only those two numbers would
    reasonably suspect noise. Splitting the test period in half shows it is not:
    the gain is close to +15% in *both* halves and negative only on validation,
    so the sign change sits at the validation/test boundary rather than
    wandering.

    What the boundary is remains open, and the table is not asked to settle it.
    Validation is both nearer the training window and drawn from a period whose
    base rate is 6.8% against test's 11.3%, and those two candidate
    explanations move together here. What can be said is narrower and still
    worth saying: **the benefit is invisible on the only period this project is
    allowed to choose on**, so the honest reading is not "recency weighting does
    not work" but "this split cannot select it".
    """
    periods: list[tuple[str, pd.DataFrame]] = [
        ("validation", parts["validation"])
    ]
    test = parts["test"]
    years = test["date_key"].dt.year
    midpoint = int(years.median())
    periods += [
        (f"test through {midpoint}", test.loc[years <= midpoint]),
        (f"test from {midpoint + 1}", test.loc[years > midpoint]),
        ("test", test),
    ]

    rows = []
    for label, part in periods:
        if part.empty or positives(part[LABEL]).nunique() < 2:
            continue
        without = score(part[LABEL], plain.predict(part))
        with_weights = score(part[LABEL], recent.predict(part))
        rows.append(
            {
                "period": label,
                "rows": int(len(part)),
                "base_rate": without.base_rate,
                "unweighted_pr_auc": without.pr_auc,
                "recency_pr_auc": with_weights.pr_auc,
                "gain": with_weights.pr_auc / without.pr_auc,
            }
        )
    on_validation = next(
        (row for row in rows if row["period"] == "validation"), None
    )
    within_test = [
        row for row in rows if row["period"].startswith("test ")
    ]
    return {
        "uses_test_labels": True,
        "used_for_selection": False,
        "periods": rows,
        # The claim the table actually supports, evaluated rather than
        # asserted: the sign changes at the boundary and does not wander inside
        # the test period. Deliberately *not* "the gain rises with distance" --
        # it does between validation and test and does not between the two
        # halves of test, and a field claiming a monotone trend would be
        # reporting the first cut that happened to show one.
        "sign_changes_at_the_validation_boundary": bool(
            on_validation is not None
            and within_test
            and on_validation["gain"] < 1.0
            and all(row["gain"] > 1.0 for row in within_test)
        ),
        "spread_within_test": (
            max(row["gain"] for row in within_test)
            - min(row["gain"] for row in within_test)
            if within_test
            else None
        ),
    }


def _training_drift(parts: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """Whether there is any drift *inside* the training window to lean on.

    The number that explains the result, and the one this ablation would be
    hard to interpret without. Recency weighting can only exploit a trend the
    training split itself contains: if the later training years look like the
    earlier ones, weighting towards them buys nothing and costs sample size,
    and the search will correctly refuse to do it.

    The split is halved rather than regressed, because a slope fitted to
    twenty-four annual rates is a number with a confidence interval wider than
    the effect and this only has to answer "is there a gradient here at all".
    """
    train = parts["train"]
    years = train["date_key"].dt.year
    midpoint = int(years.median())
    early = positives(train.loc[years <= midpoint, LABEL])
    late = positives(train.loc[years > midpoint, LABEL])
    return {
        "train_first_half": {
            "through": midpoint,
            "base_rate": float(early.mean()),
            "rows": int(len(early)),
        },
        "train_second_half": {
            "from": midpoint + 1,
            "base_rate": float(late.mean()),
            "rows": int(len(late)),
        },
        "within_train_ratio": float(late.mean() / early.mean()),
        "validation_base_rate": base_rate(parts["validation"][LABEL]),
        "test_base_rate": base_rate(parts["test"][LABEL]),
        # The whole diagnosis in one field: the drift the label undeniably has
        # is between the splits, not inside the training one, so there is no
        # gradient here for a recency weight to ride.
        "drift_is_inside_the_training_window": bool(
            late.mean() / early.mean() > 1.1
        ),
    }


def _effective_rows(train: pd.DataFrame, half_life_years: float) -> float:
    """Kish's effective sample size: what the weighted fit is really fitted on.

    ``(sum w)^2 / sum w^2``. A half-life short enough to fix the drift is also
    short enough to throw most of the record away, and the row count is how
    that shows. It is reported beside every validation score so a half-life
    that wins by a hundredth on four thousand effective rows can be recognised
    as the trade it is.
    """
    weights = recency_weights(train, half_life_years)
    if weights is None:
        return float(len(train))
    return float(weights.sum() ** 2 / np.square(weights).sum())


def _oldest_weight(train: pd.DataFrame, half_life_years: float) -> float:
    """What the first row of the record counts for, relative to the last."""
    weights = recency_weights(train, half_life_years)
    return 1.0 if weights is None else float(weights.min() / weights.max())


def calibration_report(
    fits: Mapping[str, "Fit"], parts: Mapping[str, pd.DataFrame], *, variant: str
) -> dict[str, Any]:
    """What the probabilities are worth, and what two corrections do to them.

    ``train.py`` already records that the weighted model's mean prediction is
    far above the rate it is predicting. The recommendation to prefer the
    unweighted one is right and incomplete: that model is miscalibrated too,
    just less spectacularly, and in the opposite direction. It predicts 0.066
    where 0.115 occurs, because it was fitted where positives are 5% of rows
    and scored where they are 11.5%.

    **The obvious fix is the wrong one.** Isotonic regression fitted on
    validation maps scores onto validation-period frequencies, and validation's
    base rate is 7.1% against test's 11.5%. Applied to test it arrives already
    wrong, in the same direction and for the same reason: *calibration does not
    survive label shift*. That is why this reports four rows and not two. The
    calibrated row is expected to improve on the raw one and to remain visibly
    short, and it is recorded rather than skipped so the reader can see the
    thing that does not work as well as the thing that does.

    The correction that handles the shift is prior-shift adjustment: estimate
    the class prior on the target period by EM over the model's own posteriors,
    with no labels, and re-weight. It is applied *on top of* the calibrator
    rather than instead of it, because the two fix different faults -- isotonic
    fixes the shape of the map, the prior shift fixes its level -- and neither
    subsumes the other.

    **And on this data the estimator fails, while the correction it feeds
    works.** Given the observed prior, re-weighting produces the best-calibrated
    probabilities in the table: that is the ``prior_shifted_oracle`` row, and it
    is an oracle because it is told the answer. Asked to estimate the prior
    instead, the EM converges to roughly two and a half times the truth. It is
    not a convergence failure -- the fixed point is unique and reached from
    every starting value -- and it is not an implementation fault, because the
    same routine returns validation's own prior to the digit when validation is
    both source and target. It is the known bias of this estimator under a
    weakly separating classifier: ``fixed_point_at_observed_prior`` records the
    number that produces it, the mean re-weighted posterior at the true prior,
    which sits above the true prior and so gives the iteration somewhere to
    climb.

    ``quantifier_diagnostics`` records why nothing can be done about that here.
    Two candidate posteriors could feed the EM, and on the target period they
    disagree wildly; the one that is nearly exact on test is the one that
    collapses to zero on validation, so **the choice that looks best on
    validation is the one that fails worst on test**. Selecting the other would
    be selecting on test, which is the single thing this project does not do.
    So the prior shift is recorded, in full, and not recommended.

    The prior shift is monotone and provably cannot reorder anything. Isotonic
    can and does: it collapses 17 247 distinct scores to 127, and average
    precision is tie-sensitive, so it costs about 4% of PR-AUC. That is the
    price of the calibration and it is recorded rather than absorbed -- every
    row carries its PR-AUC, and a test asserts the prior shift leaves it exactly
    alone.
    """
    validation, test = parts["validation"], parts["test"]
    calibrator = fit_calibrator(fits[variant], validation)

    raw = fits[variant].predict(test)
    calibrated = np.asarray(calibrator.predict(raw), dtype=float)

    # The prior the calibrated posteriors carry is validation's, not the
    # training split's: isotonic maps scores onto validation-period
    # frequencies, so that is the source the EM has to correct *from*. Using
    # the training base rate here would ask the EM to undo a shift that the
    # calibrator has already partly undone, and it would overshoot.
    source_prior = base_rate(validation[LABEL])
    estimated, iterations = estimate_prior(calibrated, source_prior)
    shifted = apply_prior_shift(calibrated, source_prior, estimated)

    observed = base_rate(test[LABEL])
    report: dict[str, Any] = {
        "method": CALIBRATION_METHOD,
        "fitted_on": "validation",
        "applied_to": "test",
        "calibrated_variant": variant,
        "bins": CALIBRATION_BINS,
        "bin_strategy": "quantile",
        "train_prior": base_rate(parts["train"][LABEL]),
        "source_prior": source_prior,
        "estimated_target_prior": estimated,
        "em_iterations": iterations,
        # Recorded to be read *against* the estimate, never used to produce it.
        # estimate_prior() takes no labels; the test period's answer appears in
        # this block only as a score of things that were computed without it.
        "observed_target_prior": observed,
        "prior_estimate_error": estimated - observed,
        # The number that explains the overshoot. A correct estimate is a fixed
        # point of `mean(reweight(p, prior)) == prior`; this is the left side
        # evaluated at the true prior, and it sits above it, so the iteration
        # has somewhere to climb.
        "fixed_point_at_observed_prior": float(
            apply_prior_shift(calibrated, source_prior, observed).mean()
        ),
        "quantifier_diagnostics": _quantifier_diagnostics(fits, parts),
        "decision": decision_report(calibrator, fits[variant], parts),
        "recommended": "calibrated",
        "recommendation_note": (
            "Isotonic on validation halves the calibration error and is worth "
            "shipping. The prior shift is not: its EM estimate of the target "
            "prior is badly biased here, and no choice among the candidate "
            "quantifiers can be made on validation. See "
            "quantifier_diagnostics."
        ),
        "variants": {},
    }
    entries = {
        "raw_weighted": (
            fits["weighted"].predict(test),
            "the model ML-05 specifies, uncalibrated",
        ),
        "raw_unweighted": (raw, "the recommended model, uncalibrated"),
        "calibrated": (
            calibrated,
            f"{CALIBRATION_METHOD} fitted on validation, applied to test",
        ),
        "calibrated_prior_shifted": (
            shifted,
            "and re-weighted to the prior EM estimates on the target period",
        ),
        # A ceiling, not a configuration. It is told the test period's base
        # rate, so it cannot be deployed and must never be quoted as a result;
        # it is here to separate "the correction is wrong" from "the estimate
        # is wrong", and it says the second.
        "prior_shifted_oracle": (
            apply_prior_shift(calibrated, source_prior, observed),
            "NOT SHIPPABLE: re-weighted to the observed test prior, to show "
            "what the correction is worth when the estimate is right",
        ),
    }
    for name, (predictions, note) in entries.items():
        report["variants"][name] = _calibration_entry(test[LABEL], predictions, note)
    report["variants"]["prior_shifted_oracle"]["uses_test_labels"] = True
    return report


def decision_report(
    calibrator, fit: "Fit", parts: Mapping[str, pd.DataFrame]
) -> dict[str, Any]:
    """Where the threshold comes from, and what it costs (ML-11).

    ``metrics.json`` used to record ``threshold_metric: f1``. F1 is the
    harmonic mean of precision and recall, which is a way of saying a false
    alarm and a missed heatwave are equally bad -- not a claim anyone would
    defend out loud, and one nobody had been asked to. Worse, an F1-optimal
    threshold found where positives are 7% of rows is not F1-optimal where they
    are 11.5%, so even the indefensible rule was being applied off its own
    terms.

    The threshold is now the most sensitive rule that keeps a city under
    :data:`ALERT_BUDGET_PER_CITY_YEAR` alert-days a year, chosen on
    **validation**, applied to test, and computed on the **calibrated**
    probabilities, because a budget is a statement about how often a tile
    lights up and only a calibrated probability makes the threshold that
    delivers it mean anything.

    Everything a reader needs to disagree with the choice is recorded: the
    budget, the threshold it produced, the cost ratio that threshold implies,
    and the two adjacent rules with their precision, recall and alert rate. The
    F1 threshold is recorded beside them, unchosen, so the change is a
    comparison rather than an assertion.
    """
    validation, test = parts["validation"], parts["test"]
    calibrated = {
        name: np.asarray(calibrator.predict(fit.predict(part)), dtype=float)
        for name, part in (("validation", validation), ("test", test))
    }
    chosen = threshold_for_budget(
        validation, validation[LABEL], calibrated["validation"]
    )

    report: dict[str, Any] = {
        "rule": "alert budget",
        "budget_alerts_per_city_year": ALERT_BUDGET_PER_CITY_YEAR,
        "chosen_on": "validation",
        "computed_on": "calibrated probabilities",
        "threshold": chosen,
        "implied_cost_ratio": implied_cost_ratio(chosen),
        "rejected_rule": "f1",
        "periods": {},
    }
    for name, part in (("validation", validation), ("test", test)):
        table = decision_table(part, part[LABEL], calibrated[name], chosen)
        report["periods"][name] = {
            "city_years": len(part) / 365.25,
            "anomalous_days_per_city_year": alerts_per_city_year(
                part, positives(part[LABEL]).to_numpy()
            ),
            "neighbourhood": table.to_dict("records"),
        }

    # What F1 would have picked on the same calibrated probabilities, so the
    # two rules can be read against each other. Not chosen, and recorded to be
    # compared rather than used.
    f1_threshold, f1_score = best_threshold(
        validation[LABEL], calibrated["validation"]
    )
    report["f1_alternative"] = {
        "threshold": f1_threshold,
        "validation_f1": f1_score,
        "alerts_per_city_year_on_test": alerts_per_city_year(
            test, calibrated["test"] >= f1_threshold
        ),
        "implied_cost_ratio": implied_cost_ratio(f1_threshold),
    }
    return report


def _quantifier_diagnostics(
    fits: Mapping[str, "Fit"], parts: Mapping[str, pd.DataFrame]
) -> dict[str, Any]:
    """Whether validation can pick the model that estimates the prior best.

    It cannot, and that is the reason the prior shift is not recommended. Two
    posteriors could feed the EM: the unweighted model's, which carry the
    training prior, and the weighted model's, which carry 0.5 because weighting
    the positive class by the negative-to-positive ratio *is* training at a
    balanced prior.

    Each is asked to recover a prior it was not given, twice: validation's,
    which is a shift it could be selected on, and test's, which is the one that
    matters. The errors are recorded for both. If the ranking of the two agreed
    across the periods, the better quantifier could be chosen honestly on
    validation and used on test; ``validation_picks_the_better_quantifier``
    records that it does not.
    """
    train_prior = base_rate(parts["train"][LABEL])
    candidates = {
        # The unweighted model is fitted on the split as it stands, so its
        # posteriors carry the training prior.
        "unweighted": ("unweighted", train_prior),
        # scale_pos_weight at the observed negative-to-positive ratio is
        # training at a balanced prior, so its posteriors carry 0.5.
        "weighted": ("weighted", 0.5),
    }
    rows: dict[str, Any] = {}
    for name, (variant, source) in candidates.items():
        row: dict[str, Any] = {"posteriors_from": variant, "source_prior": source}
        for period in ("validation", "test"):
            estimated, iterations = estimate_prior(
                fits[variant].predict(parts[period]), source
            )
            row[period] = {
                "estimated": estimated,
                "observed": base_rate(parts[period][LABEL]),
                "error": estimated - base_rate(parts[period][LABEL]),
                "iterations": iterations,
            }
        rows[name] = row

    best_on = {
        period: min(rows, key=lambda name: abs(rows[name][period]["error"]))
        for period in ("validation", "test")
    }
    return {
        "candidates": rows,
        "closest_on_validation": best_on["validation"],
        "closest_on_test": best_on["test"],
        "validation_picks_the_better_quantifier": (
            best_on["validation"] == best_on["test"]
        ),
    }


def _hazard_metrics(block: Mapping[str, Any]) -> dict[str, Any]:
    """The hazard's composed scores, in the per-split shape the sidecar wants.

    ``block["hazard"]["composed"]`` nests each split under the arm it belongs
    to, because its job is to sit the hazard beside the direct model. The
    artefact sidecar wants one predictor's scores per split, so the hazard arm
    is lifted out. The *composed weekly* scores, not the person-period ones: a
    reader comparing this artefact with the baselines is asking about the
    number it produces, and the number it produces is a week.
    """
    composed = block.get("hazard", {}).get("composed", {})
    return {
        split: entry["hazard"]
        for split, entry in composed.items()
        if "hazard" in entry
    }


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
        # The hazard is a variant of the *target's shape* rather than of the
        # class weighting, so it has no row in `block["variants"]`; its scores
        # live in `block["hazard"]`. Saved through the same path all the same,
        # because predict.py has to be able to load it by name and validate its
        # features against the frame it is handed.
        saved[variant] = save_artifact(
            fit,
            variant=variant,
            train=parts["train"],
            target=y_train,
            metrics=(
                block["variants"][variant]
                if variant in block["variants"]
                else _hazard_metrics(block)
            ),
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
    hazard: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], dict[str, Fit]]:
    """Fit both variants, score them on every split, and describe them.

    Args:
        hazard: The person-period frame from :func:`hazard_periods`, if the
            discrete-time hazard (ML-13) is to be fitted too. Passed in rather
            than built here because it needs the *gold* frame, which this
            function does not otherwise read, and because it costs a
            twelve-point search over 656 000 rows that most callers -- the
            leave-one-city-out folds above all -- have no use for.

    Returns the ``model`` block for ``metrics.json`` and both fitted models by
    variant name. Both, because ML-06 draws curves for each and the difference
    between them is the finding: ``variants["weighted"]`` is the one the ticket
    specifies and the one saved, ``variants["unweighted"]`` is the one its
    stated goal asks for.
    """
    hazard_fits: dict[str, Any] = {}
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

    # Recommended on validation, never on test, which is the same rule the search
    # follows. Better PR-AUC *and* better Brier, or the specified one stands.
    specified = block["variants"]["weighted"]["validation"]
    other = block["variants"]["unweighted"]["validation"]
    better = (
        other["pr_auc"] > specified["pr_auc"] and other["brier"] < specified["brier"]
    )
    block["recommended_variant"] = "unweighted" if better else "weighted"
    fits = dict(fits)
    # After the recommendation, because it is the recommended model that gets
    # calibrated, and before returning, so metrics.json cannot carry a model
    # block without the account of what its probabilities are worth.
    if hazard is not None:
        block["hazard"] = hazard_report(
            parts,
            split_frame(hazard),
            fits[block["recommended_variant"]],
            fitted=hazard_fits,
        )
    block["recency"] = recency_ablation(
        parts,
        scale_pos_weight=variants[block["recommended_variant"]],
        baseline=fits[block["recommended_variant"]],
    )
    block["calibration"] = calibration_report(
        fits, parts, variant=block["recommended_variant"]
    )
    fits.update(hazard_fits)
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
    they describe the same data. If the warehouse has moved on (a city
    backfilled, a day landed) the baselines have to be re-run and re-committed
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
    parser.add_argument(
        "--hazard",
        action="store_true",
        help=(
            "Also fit the discrete-time hazard (ML-13): seven per-day "
            "probabilities per city-day, composed back into the weekly number. "
            "Reshapes to ~656 000 training rows and searches the same grid."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    population = evaluation_frame()
    periods = None
    if args.hazard:
        # Read gold once, here, rather than inside train_model: the hazard
        # needs the *complete* calendar to know what happened on t+3, and the
        # population has had its warm-up trimmed out of it.
        periods = hazard_periods(population, gold_frame())
        print(
            f"hazard     {len(periods):,} person-periods from "
            f"{len(population):,} city-days\n"
        )
    block, fits = train_model(frame=population, hazard=periods)

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
    print(f"  recommended: {recommended}   (chosen on validation, never on test)")
    if recommended != block["specified_variant"]:
        weighted = block["variants"]["weighted"]["test"]
        print(
            "\n  scale_pos_weight is oversampling the positive class by "
            f"{block['scale_pos_weight_observed']:.1f}x, so the weighted model "
            f"predicts a mean probability of {weighted['mean_predicted']:.3f} "
            f"against a true base rate of {weighted['base_rate']:.3f}. The Risk "
            "Horizon view shows these numbers to a reader directly."
        )

    hazard = block.get("hazard")
    if hazard:
        profile = hazard["profile"]
        print("\nhazard, per-day probabilities composed back into the week")
        for split, entry in hazard["composed"].items():
            print(
                f"  {split:<11} composed PR-AUC {entry['hazard']['pr_auc']:.4f}  "
                f"direct {entry['direct']['pr_auc']:.4f}  "
                f"rank corr {entry['rank_correlation']:.3f}"
            )
        print(
            f"\n  the model resolves {profile['distinct_levels']} distinct "
            f"levels across the seven days, not {HORIZON_DAYS}: "
            f"{profile['share_with_flat_tail']:.0%} of city-days have days "
            "2-7 identical."
        )
        for entry in profile["by_todays_flag"]:
            days = "  ".join(f"{value:.4f}" for value in entry["mean_hazard_by_day"])
            print(f"  today {entry['today']:<8} ({entry['city_days']:>6}): {days}")

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
                    f"  WARNING: over {MAX_COMMITTED_BYTES:,} bytes, too "
                    "large to commit; it belongs in a release asset."
                )
    else:
        print("\n(not written; pass --write to update metrics.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
