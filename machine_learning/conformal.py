"""Distribution-free prediction sets, and what happens to them under drift.

The project's stated ethos is refusing to imply precision it does not have.
Conformal prediction is the formal version of that: given any model and a
calibration sample, it returns a *set* of labels with a coverage guarantee that
holds without assuming anything about the model or the distribution -- only
that the calibration sample and the point being predicted are exchangeable.

For a binary target the sets are small and legible. ``{quiet}`` and
``{extreme}`` are confident answers; ``{quiet, extreme}`` says the model cannot
separate the two on this week; and the empty set says neither label is
plausible at the requested confidence, which on a binary problem means the week
is unlike anything in calibration. Reporting those four states is a great deal
more honest than a probability a reader will round.

**The exchangeability assumption is violated here, and that is the point of the
ticket.** The label's base rate moves from 4.9% in the training period to 6.8%
in validation and 11.3% in test. Split conformal calibrated on validation and
applied to test would hold its guarantee on paper and break silently on the
data, because the quantile it fixed describes a period with a different
positive rate. Adaptive conformal (Gibbs & Candes, 2021) adjusts the level
online -- widening after a miss, tightening after a hit -- and maintains
long-run empirical coverage under drift. Both are computed, because the gap
between them *is* the drift, measured in the units the guarantee is stated in.

**What reads the test period, and how.** The calibration sample is the
validation split and nothing else: :func:`calibrate` takes one frame, and
``tests/test_evaluation.py`` rewrites the test period and requires the
calibration to be unchanged. The adaptive procedure then walks the test period
in time order and updates its level from each outcome *after* predicting it,
which is what "online" means and what it would do in deployment. That is
feedback rather than fitting, and the number it produces is a realised online
coverage rather than a held-out score. Saying which of the two a figure is
matters more here than anywhere else in the project, because they are both
percentages near ninety.

Usage::

    from machine_learning.conformal import calibrate, adaptive_report

    calibration = calibrate(validation[LABEL], model.predict(validation))
    report = adaptive_report(calibration, test, model.predict(test))
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from machine_learning.labels import LABEL, positives  # noqa: E402

__all__ = [
    "GAMMA",
    "TARGET_COVERAGE",
    "Calibration",
    "adaptive_report",
    "calibrate",
    "nonconformity",
    "prediction_sets",
]

#: The coverage the sets are asked to achieve, stated up front and never tuned.
#:
#: Ninety per cent, chosen before anything was measured and left alone. The
#: number a conformal method is *asked* for is the one thing in it that cannot
#: be selected after seeing how it did: a target picked to match the coverage
#: that happened to come out is not a guarantee, it is a description with a
#: Greek letter on it.
TARGET_COVERAGE: Final[float] = 0.90

#: The adaptive step size, from Gibbs & Candes.
#:
#: 0.01 against 18 100 test rows. The level moves by one percentage point per
#: miss and recovers by ``gamma * (1 - target)`` per hit, so it tracks a drift
#: that unfolds over months and ignores a single unlucky week. Larger values
#: chase noise; smaller ones cannot catch the base rate doubling that this
#: project's whole Phase 1 is about.
GAMMA: Final[float] = 0.01


def nonconformity(labels: pd.Series, probabilities) -> np.ndarray:
    """How badly the model missed, per row: ``1 - p`` of the label that occurred.

    The standard score for a probabilistic classifier. A confident right answer
    scores near zero and a confident wrong one near one, so the calibration
    quantile is "how wrong the model gets on its worst ten per cent of weeks".
    """
    truth = positives(labels).to_numpy()
    values = np.asarray(probabilities, dtype=float)
    if len(values) != len(truth):
        raise ValueError(f"{len(truth)} labels against {len(values)} predictions.")
    return np.where(truth, 1.0 - values, values)


@dataclass(frozen=True)
class Calibration:
    """Nonconformity scores from the calibration split, and the quantile rule.

    Holds the scores rather than a single threshold, because the adaptive
    procedure asks for a different quantile on every step and recomputing it
    from the sample is what makes the level meaningful rather than interpolated
    between two fixed points.
    """

    scores: np.ndarray
    fitted_on: str = "validation"

    def quantile(self, alpha: float) -> float:
        """The conformal quantile at miscoverage ``alpha``.

        ``ceil((n + 1)(1 - alpha)) / n``, not the plain empirical quantile. The
        finite-sample correction is the difference between a guarantee that
        holds for the sample size you have and one that holds asymptotically;
        with 11 987 calibration rows it moves the threshold by a hair, and
        leaving it out would be claiming an exactness the arithmetic does not
        provide.

        A level so tight that the corrected rank exceeds the sample returns
        infinity, which is the honest answer: this calibration set cannot
        certify that confidence, so every set becomes ``{quiet, extreme}``.
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha}.")
        count = len(self.scores)
        rank = int(np.ceil((count + 1) * (1.0 - alpha)))
        if rank > count:
            return float("inf")
        return float(np.sort(self.scores)[rank - 1])


def calibrate(labels: pd.Series, probabilities) -> Calibration:
    """Build the calibration from one frame. There is no second to pass it.

    The same structural argument :func:`~machine_learning.train.tune` and
    :func:`~machine_learning.train.fit_calibrator` rest on, and here it matters
    most: a conformal guarantee calibrated on the period it is then evaluated
    on is not a guarantee at all, it is a description of a sample, and it would
    report near-perfect coverage while meaning nothing.

    **Raw model probabilities, not ML-10's calibrated ones.** Isotonic is
    fitted on validation, so scores taken from it on validation are in-sample:
    they would be too small, the quantile too tight, and the coverage on test
    would fall short for a reason that had nothing to do with drift. Conformal
    needs no calibrated input -- that is rather the point of it -- so it takes
    the model's own output and leaves ML-10's correction out of the loop.
    """
    return Calibration(scores=nonconformity(labels, probabilities))


def prediction_sets(probabilities, threshold: float) -> pd.DataFrame:
    """Which labels survive at this nonconformity threshold, per row.

    A label is in the set when its own score is at or under the threshold, so
    ``extreme`` is included when ``p >= 1 - threshold`` and ``quiet`` when
    ``p <= threshold``. Both can hold at once, and neither can, and both states
    mean something: ``{quiet, extreme}`` is the model declining to separate
    them, and ``{}`` is a week unlike anything in the calibration sample.
    """
    values = np.asarray(probabilities, dtype=float)
    return pd.DataFrame(
        {
            "quiet": values <= threshold,
            "extreme": values >= 1.0 - threshold,
        }
    )


def adaptive_report(
    calibration: Calibration,
    frame: pd.DataFrame,
    probabilities,
    *,
    target: float = TARGET_COVERAGE,
    gamma: float = GAMMA,
) -> dict[str, Any]:
    """Split and adaptive conformal over the target period, with realised coverage.

    Both, because the comparison is the finding. Split conformal fixes its
    level on the calibration sample and never moves; adaptive conformal walks
    the period in time order and adjusts after each outcome. Under
    exchangeability they agree; under drift the fixed one drifts away from its
    target and the adaptive one does not, and the size of that gap is the drift
    expressed as a coverage error.

    Returns realised coverage overall, per city and per year, plus the average
    set size. Coverage without set size is not a result: a method that always
    returns ``{quiet, extreme}`` covers everything and says nothing.
    """
    truth = positives(frame[LABEL]).to_numpy()
    values = np.asarray(probabilities, dtype=float)
    order = np.argsort(frame["date_key"].to_numpy(), kind="stable")

    fixed_threshold = calibration.quantile(1.0 - target)
    fixed = prediction_sets(values, fixed_threshold)

    # The adaptive walk, in date order. alpha moves *after* the outcome is
    # revealed, which is the whole shape of the method: the set for row t is
    # formed from what happened up to t-1 and nothing later.
    alpha = 1.0 - target
    levels = np.empty(len(values), dtype=float)
    covered_adaptive = np.empty(len(values), dtype=bool)
    sizes_adaptive = np.empty(len(values), dtype=int)
    for position in order:
        levels[position] = alpha
        threshold = calibration.quantile(min(max(alpha, 1e-6), 1 - 1e-6))
        in_quiet = values[position] <= threshold
        in_extreme = values[position] >= 1.0 - threshold
        held = in_extreme if truth[position] else in_quiet
        covered_adaptive[position] = held
        sizes_adaptive[position] = int(in_quiet) + int(in_extreme)
        alpha = alpha + gamma * ((1.0 - target) - (0.0 if held else 1.0))
        alpha = min(max(alpha, 1e-6), 1 - 1e-6)

    covered_fixed = np.where(truth, fixed["extreme"], fixed["quiet"]).astype(bool)
    sizes_fixed = fixed.sum(axis=1).to_numpy()

    report: dict[str, Any] = {
        "target_coverage": target,
        "gamma": gamma,
        "calibrated_on": calibration.fitted_on,
        "calibration_rows": int(len(calibration.scores)),
        "scored_on": "test",
        "note": (
            "Adaptive coverage is realised online: the level is updated from "
            "each outcome after that row has been predicted, which is what the "
            "method does in deployment. It is not a held-out score."
        ),
        "split": _coverage_block(frame, covered_fixed, sizes_fixed, target),
        "adaptive": _coverage_block(
            frame, covered_adaptive, sizes_adaptive, target
        ),
        "fixed_threshold": fixed_threshold,
        "alpha_start": 1.0 - target,
        "alpha_end": float(alpha),
        "alpha_range": [float(levels.min()), float(levels.max())],
    }
    report["adaptive_is_closer_to_target"] = bool(
        abs(report["adaptive"]["coverage"] - target)
        <= abs(report["split"]["coverage"] - target)
    )
    return report


def _coverage_block(
    frame: pd.DataFrame, covered: np.ndarray, sizes, target: float
) -> dict[str, Any]:
    """Realised coverage overall, per city and per year, with set sizes."""
    held = pd.Series(np.asarray(covered, dtype=bool), index=frame.index)
    size = pd.Series(np.asarray(sizes, dtype=float), index=frame.index)
    years = frame["date_key"].dt.year

    def rows(grouped_by) -> list[dict[str, Any]]:
        return [
            {
                "key": key if isinstance(key, str) else int(key),
                "rows": int(len(group)),
                "coverage": float(held.loc[group.index].mean()),
                "mean_set_size": float(size.loc[group.index].mean()),
                "shortfall": float(held.loc[group.index].mean() - target),
            }
            for key, group in grouped_by
        ]

    return {
        "coverage": float(held.mean()),
        "mean_set_size": float(size.mean()),
        "shortfall": float(held.mean() - target),
        "by_city": rows(frame.groupby(frame["city_id"], sort=True)),
        "by_year": rows(frame.groupby(years, sort=True)),
        # Coverage is trivially satisfiable by never committing: the share of
        # rows where the set holds both labels says how often that happened.
        "share_uninformative": float((size >= 2).mean()),
        "share_empty": float((size == 0).mean()),
    }
