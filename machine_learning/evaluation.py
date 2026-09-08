"""The chronological split, and the two numbers everything is scored on.

Separate from the baselines and from the model because all three have to be
measured the same way. A baseline scored with one implementation of average
precision and a model scored with another are not comparable, and the
difference would live entirely in tie handling, where a rule-based baseline
puts thousands of rows on the same score and a trained model puts none. So the
metrics come from ``sklearn`` and are called from one place.

**The split is chronological, and it is purged.** Train to 2018, validate
2019-2021, test 2022 onwards, exactly as the proposal specifies. A random split
would leak the future through every rolling feature and produce a meaningless
score.

Cutting on the date alone is not quite enough. The label at *t* is an anomaly
in t+1 .. t+7, so the last seven days of the training period carry a label
built from the first days of validation. Seven days per city per boundary is a
rounding error in row count and not one in principle: it is the training set
being told what happened next door. :data:`PURGE_DAYS` drops them, and
:func:`assert_splits_are_disjoint` proves no train label window reaches into
validation.

**PR-AUC needs its no-skill line quoted beside it.** Average precision for a
random ranker is the positive rate, and the positive rate here is 5.50% in
train and 13.58% in test, so 0.20 is a good score on one and a poor one on the
other. Every :class:`Score` carries the base rate it was measured against, and
:func:`score` also returns the lift over it, because a PR-AUC with no reference
is the thing this whole ticket exists to prevent.

**There is a second axis to hold out on, and it is not time.** ML-08 asks
whether the model can score a city it has never seen, which the chronological
split says nothing about: every split contains every city. :func:`hold_out_city`
and :func:`assert_city_is_held_out` are the primitives for that fold, kept here
beside the time split because they are the same kind of object -- a partition
the rest of the code must not be free to write for itself -- and because the
city fold is composed *with* the time split rather than instead of it. A
leave-one-city-out fold is still trained to 2018 and still purged.

Usage::

    from machine_learning.evaluation import SPLITS, score, split_frame

    parts = split_frame(frame)
    result = score(parts["test"][LABEL], predictions)
    print(result.pr_auc, result.base_rate, result.lift)
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from machine_learning.labels import (  # noqa: E402
    HORIZON_DAYS,
    LABEL,
    drop_unlabelled,
    positives,
    training_frame,
)
from machine_learning.features import (  # noqa: E402
    drop_warmup,
    gold_frame,
    on_daily_calendar,
    require_grain,
    trailing_anomaly_counts,
)

__all__ = [
    "ANOMALY_THRESHOLD",
    "CALIBRATION_BINS",
    "ANOMALY_THRESHOLDS",
    "MIN_HELD_OUT_POSITIVES",
    "MIN_HELD_OUT_ROWS",
    "PERSISTENCE_COUNT",
    "PERSISTENCE_FLAG",
    "PERSISTENCE_WINDOW",
    "PURGE_DAYS",
    "SPLITS",
    "Score",
    "Split",
    "assert_splits_are_disjoint",
    "EMBARGO_DAYS",
    "add_persistence_signal",
    "assert_city_is_held_out",
    "assert_splits_are_ordered",
    "base_rate",
    "apply_prior_shift",
    "boundary_report",
    "drop_scorable_gaps",
    "estimate_prior",
    "expected_calibration_error",
    "evaluation_frame",
    "hold_out_city",
    "lift_over",
    "reflag",
    "reliability_points",
    "scorable_cities",
    "score",
    "split_frame",
    "split_summary",
]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Split:
    """One period of the chronological split, closed at both ends it has."""

    name: str
    start: dt.date | None
    end: dt.date | None

    def contains(self, dates: pd.Series) -> pd.Series:
        keep = pd.Series(True, index=dates.index)
        if self.start is not None:
            keep &= dates >= pd.Timestamp(self.start)
        if self.end is not None:
            keep &= dates <= pd.Timestamp(self.end)
        return keep


#: The proposal's split, and the one every score in this project is reported on.
#: Written as dates rather than derived from the data: a split that moves with
#: the backfill would make two runs incomparable for a reason nobody could see.
SPLITS: Final[tuple[Split, ...]] = (
    Split("train", dt.date(1995, 1, 1), dt.date(2018, 12, 31)),
    Split("validation", dt.date(2019, 1, 1), dt.date(2021, 12, 31)),
    Split("test", dt.date(2022, 1, 1), None),
)

#: "This week", in days, counting today. Seven, because the label is seven days
#: forward and the persistence rule is meant to be its mirror.
PERSISTENCE_WINDOW: Final[int] = HORIZON_DAYS

PERSISTENCE_FLAG: Final[str] = "had_anomaly_last_7d"
PERSISTENCE_COUNT: Final[str] = f"anomaly_days_trailing{PERSISTENCE_WINDOW}"

#: Days dropped from the end of every split that has a later one after it.
#:
#: Equal to the label horizon, because that is exactly how far a label reaches:
#: a row dated 2018-12-31 is labelled by days up to 2019-01-07, which is
#: validation. Purging is cheap here (seven rows per city per boundary) and
#: the alternative is a training set that has been told the first week of the
#: period it is about to be validated on.
PURGE_DAYS: Final[int] = HORIZON_DAYS


#: Days optionally dropped from the *start* of every split that has an earlier
#: one before it. Zero by default, and the default is the argued position.
#:
#: A validation row on 2019-01-01 has a 30-day rolling mean reaching back to
#: 2018-12-03, which is training data. That is not leakage, it is deployment.
#: A model predicting on 2019-01-01 in production has all of 2018 behind it,
#: and blanking it here would measure a system nobody is going to run. Leakage
#: is a *training* row reading forwards, which the backward-only features and
#: :data:`PURGE_DAYS` between them already rule out.
#:
#: What a start-of-split embargo would buy is not leakage safety but sample
#: independence: the last training rows and the first validation rows share
#: some of the same days inside their windows, so the two sets are mildly
#: correlated and the score mildly optimistic. It is offered so that claim can
#: be measured rather than argued, and the measurement is in the build log:
#: it moves test PR-AUC by less than a thousandth.
EMBARGO_DAYS: Final[int] = 0


#: The |Z| the warehouse flags an anomaly at, and the one every committed
#: number in this project is measured against. Kept here as well as in
#: ``dbt_project``'s ``anomaly_z_threshold`` because the sweep below has to be
#: able to say which of its points is the shipped one.
ANOMALY_THRESHOLD: Final[float] = 2.5

#: The thresholds ML-09 re-runs the whole evaluation at.
#:
#: 2.5 is a choice, and every conclusion in this project inherits it: the
#: label, two of the twenty-seven features, all three baselines, both model
#: variants and every per-city verdict. A finding that holds only at 2.5 is a
#: finding about 2.5. Three points either side of it are enough to see whether
#: a verdict is a property of the model or of the line, and few enough that the
#: sweep is a minute rather than an afternoon.
#:
#: 2.0 and 3.0 rather than a finer grid, because the interesting quantity is
#: whether a verdict *flips*, not where it flips. A grid fine enough to locate
#: the crossing would invite reading a threshold off it, which is exactly the
#: decision ML-11 is meant to make on cost rather than on a curve.
ANOMALY_THRESHOLDS: Final[tuple[float, ...]] = (2.0, 2.5, 3.0)


@dataclass(frozen=True)
class Score:
    """What a set of predictions is worth, with the reference it needs.

    ``pr_auc`` alone says nothing: average precision for a random ranker equals
    the positive rate, so the same 0.20 is skilful against a 5% base rate and
    worthless against 25%. ``lift`` is the ratio, and it is the number to read
    first.
    """

    rows: int
    positives: int
    base_rate: float
    pr_auc: float
    brier: float
    lift: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def base_rate(labels: pd.Series) -> float:
    """The positive rate, and the PR-AUC a random ranker would score."""
    return float(positives(labels).mean())


def score(labels: pd.Series, predictions) -> Score:
    """Average precision and Brier, with the base rate they are read against.

    Args:
        labels: The nullable-boolean target. Must contain no nulls: an
            unlabelled row has no answer to be right or wrong about, and
            averaging over it would quietly change the denominator.
        predictions: Predicted probabilities in [0, 1], aligned to ``labels``.

    Raises:
        ValueError: The label has nulls, the lengths differ, a prediction sits
            outside [0, 1], or one class is missing entirely.
    """
    labels = pd.Series(labels)
    values = np.asarray(predictions, dtype=float)
    if labels.isna().any():
        raise ValueError(
            f"{int(labels.isna().sum())} unlabelled rows reached scoring. "
            "Call drop_unlabelled() first; averaging over a row with no answer "
            "changes the denominator without changing the numerator."
        )
    if len(values) != len(labels):
        raise ValueError(f"{len(labels)} labels against {len(values)} predictions.")
    if not np.isfinite(values).all():
        raise ValueError("predictions contain NaN or inf.")
    if values.min() < 0.0 or values.max() > 1.0:
        raise ValueError(
            f"predictions must be probabilities in [0, 1]; got "
            f"[{values.min():.3f}, {values.max():.3f}]. Brier is meaningless "
            "on a score that is not one."
        )

    truth = positives(labels).to_numpy()
    if truth.all() or not truth.any():
        raise ValueError(
            "one class is missing, so average precision is undefined. This "
            "usually means a split has been sliced too thin."
        )

    rate = float(truth.mean())
    pr_auc = float(average_precision_score(truth, values))
    return Score(
        rows=len(truth),
        positives=int(truth.sum()),
        base_rate=rate,
        pr_auc=pr_auc,
        brier=float(brier_score_loss(truth, values)),
        lift=pr_auc / rate,
    )


#: Bins for a reliability curve and for the calibration error computed from it.
#:
#: Ten over 18 100 test rows is ~1 800 a bin, which is enough for the observed
#: rate in each to mean something; twenty would draw a jagged line and invite
#: reading noise as miscalibration.
CALIBRATION_BINS: Final[int] = 10


def reliability_points(
    labels: pd.Series,
    predictions,
    *,
    bins: int = CALIBRATION_BINS,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Observed rate against predicted rate, with the weight of each bin.

    **Quantile bins, not equal-width, and the choice changes the number.** This
    model's predictions pile between 0.01 and 0.30; equal-width bins put nearly
    every row in the first one and reduce a reliability curve to two points and
    eight empty boxes, and the calibration error computed from it to a single
    comparison of two means. Quantile bins spend the resolution where the rows
    are. The cost is that the figure is not the textbook ECE, so ``strategy`` is
    carried into ``metrics.json`` beside every number computed from it: a
    calibration error quoted without its binning is not comparable with anyone
    else's.

    ``weight`` is the share of rows in each bin, and it is what makes the error
    below an expectation rather than an average over bins. Equal-count bins make
    those weights nearly equal, which is another reason to prefer them: an
    unweighted mean over equal-width bins lets a bin holding nine rows count as
    much as one holding nine thousand.
    """
    truth = positives(labels).to_numpy()
    values = np.asarray(predictions, dtype=float)
    if len(values) != len(truth):
        raise ValueError(f"{len(truth)} labels against {len(values)} predictions.")
    if strategy == "quantile":
        # duplicates="drop": a predictor emitting three distinct values has
        # three usable bins, not ten, and asking for ten is an error rather
        # than a reason to fail. Persistence is exactly that predictor.
        edges = pd.qcut(values, bins, duplicates="drop", labels=False)
    elif strategy == "uniform":
        edges = pd.cut(values, bins, labels=False, include_lowest=True)
    else:
        raise ValueError(f"unknown binning strategy {strategy!r}.")

    frame = pd.DataFrame({"bin": edges, "truth": truth, "predicted": values})
    grouped = frame.groupby("bin", sort=True)
    points = grouped.agg(
        predicted=("predicted", "mean"),
        observed=("truth", "mean"),
        rows=("truth", "size"),
    ).reset_index(drop=True)
    points["weight"] = points["rows"] / len(frame)
    return points


def expected_calibration_error(
    labels: pd.Series,
    predictions,
    *,
    bins: int = CALIBRATION_BINS,
    strategy: str = "quantile",
) -> float:
    """How far the predicted probabilities sit from the rates they claim.

    The row-weighted mean of ``|observed - predicted|`` across the bins of
    :func:`reliability_points`. Zero is perfect; the base rate itself scores
    zero on a single bin and is still useless, which is why this is reported
    beside PR-AUC and never instead of it.

    Brier already notices a probability is wrong, so this is not a replacement
    for it either. Brier is a proper scoring rule and mixes calibration with
    discrimination: a model can improve it by ranking better while staying just
    as badly calibrated, which is exactly what happens here. This isolates the
    half ML-10 is about.
    """
    points = reliability_points(labels, predictions, bins=bins, strategy=strategy)
    gap = (points["observed"] - points["predicted"]).abs()
    return float((gap * points["weight"]).sum())


def apply_prior_shift(predictions, source_prior: float, target_prior: float):
    """Re-weight posteriors from one class prior to another.

    The correction underneath ML-10. A model fitted where positives are 7% of
    rows emits posteriors that carry that 7% inside them; scored on a period
    where positives are 11.5%, every probability is too low by a factor that
    depends on the probability itself, not by a constant. Under the standard
    label-shift assumption -- the class-conditional feature distribution
    p(x | y) is unchanged and only p(y) moves -- the fix is Bayes' rule applied
    twice, and it is exact:

        p'(1|x) = w1 p(1|x) / (w1 p(1|x) + w0 p(0|x))

    with ``w1 = target/source`` and ``w0 = (1-target)/(1-source)``.

    It is **monotone in p**, so it cannot change a ranking: PR-AUC is identical
    before and after, and a test asserts that. Everything it moves is the
    level, which is the only thing that was wrong.
    """
    for name, value in (("source", source_prior), ("target", target_prior)):
        if not 0.0 < value < 1.0:
            raise ValueError(
                f"the {name} prior must be strictly between 0 and 1, got {value}. "
                "A prior of 0 or 1 asserts the class cannot occur, and the "
                "re-weighting divides by it."
            )
    values = np.asarray(predictions, dtype=float)
    positive = (target_prior / source_prior) * values
    negative = ((1.0 - target_prior) / (1.0 - source_prior)) * (1.0 - values)
    total = positive + negative
    return np.divide(
        positive, total, out=np.full_like(values, target_prior), where=total > 0
    )


def estimate_prior(
    predictions,
    source_prior: float,
    *,
    max_iterations: int = 200,
    tolerance: float = 1e-10,
) -> tuple[float, int]:
    """Estimate the target period's class prior from predictions alone. No labels.

    Saerens, Latinne & Decaestecker (2002): EM over the target sample, holding
    the model fixed. Each step re-weights the posteriors to the current prior
    estimate and takes their mean as the next estimate, which is the M-step of
    a mixture whose components are the two class-conditionals.

    **This function takes no labels, and that is the guarantee rather than a
    convenience.** It cannot be fitted on the test answers because it is not
    given them; the strongest statement available about a leak is that the
    signature makes it impossible. What it does read is the test period's
    *predictions*, which is transductive and deliberate: in deployment the
    recent weeks are exactly the data you have without their labels, and
    estimating the prior on them is the operation the method exists to perform.
    Calling that leakage would be calling deployment leakage.

    Args:
        predictions: Posteriors on the target period, from a model calibrated
            to ``source_prior``. After isotonic regression that is the
            *validation* base rate and not the training one, because isotonic
            maps scores onto validation-period frequencies.
        source_prior: The prior those posteriors currently carry.

    Returns:
        The estimated prior and the iterations it took. The count is returned
        rather than logged because an estimate that ran to ``max_iterations``
        has not converged, and a reader of ``metrics.json`` should be able to
        see that without rerunning anything.
    """
    values = np.asarray(predictions, dtype=float)
    if values.size == 0:
        raise ValueError("cannot estimate a prior from no predictions.")
    prior = float(source_prior)
    for iteration in range(1, max_iterations + 1):
        updated = float(apply_prior_shift(values, source_prior, prior).mean())
        # Clipped away from the boundary: an estimate that reaches exactly 0 or
        # 1 makes the next re-weighting divide by zero, and a prior of 0 is a
        # claim no finite sample supports.
        updated = min(max(updated, 1e-12), 1.0 - 1e-12)
        if abs(updated - prior) < tolerance:
            return updated, iteration
        prior = updated
    return prior, max_iterations


def lift_over(result: Score, reference: Score) -> dict[str, float]:
    """What a predictor is worth against another predictor, not against chance.

    ``Score.lift`` divides by the base rate, which answers "is this better than
    guessing". That is the right first question and the wrong last one. The
    base rate is a *floor*: on this data persistence scores 1.69x it pooled and
    up to 2.9x in a single city, so a new method reported at 1.8x over chance
    sounds like progress and is a regression. Every method arriving from
    phases 2 to 4 will be measured against something, and if the something is
    the floor it will look better than it is.

    So the reference is passed in and named at the call site. Both figures read
    the same way round -- above one is better -- which for Brier means the
    ratio is inverted, because Brier is a loss.

    Returns:
        ``pr_auc`` and ``brier`` ratios. A reference scoring zero on either
        yields ``inf``, which is honest: nothing divided into something is not
        a comparison, and a nan here would be silently dropped by every
        aggregation downstream.
    """
    return {
        "pr_auc": (
            result.pr_auc / reference.pr_auc
            if reference.pr_auc > 0
            else float("inf")
        ),
        "brier": (
            reference.brier / result.brier if result.brier > 0 else float("inf")
        ),
    }


def reflag(gold: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Re-derive ``is_anomaly`` from the Z-score at a different threshold.

    In Python rather than by rebuilding the mart, and the reason is not speed.
    Rebuilding ``fact_weather_anomalies`` with a different
    ``anomaly_z_threshold`` writes the sweep's intermediate states into the
    warehouse every other model, every committed metric and the dashboard all
    read from. A run interrupted between two points of the sweep would leave
    the project describing a threshold nobody chose, and the failure would be
    silent because every number would still be a plausible number.

    The arithmetic is the mart's, transcribed once: ``abs(z) > threshold``,
    null-preserving, because an unscored day is not a quiet day.
    ``tests/test_baselines.py`` asserts that reflagging at
    :data:`ANOMALY_THRESHOLD` reproduces the warehouse's own column exactly, so
    the middle point of the sweep is provably the shipped pipeline rather than
    a reimplementation of it that happens to agree.

    **The threshold is not only the label's.** ``is_anomaly`` feeds
    ``anomaly_days_trailing30`` and the persistence signal as well as the
    label, so moving it moves two of the twenty-seven features and the
    strongest baseline at the same time as the target. That is the point: the
    sweep asks what the *project* looks like at 2.0, not what the model scores
    when only its answer key is changed.
    """
    if "z_temperature_2m_mean" not in gold.columns:
        raise ValueError(
            "the frame has no 'z_temperature_2m_mean' to re-flag from. This "
            "needs the gold frame, before features are built."
        )
    z = gold["z_temperature_2m_mean"]
    flagged = pd.Series(pd.NA, index=gold.index, dtype="boolean")
    flagged = flagged.mask(z.notna(), z.abs() > threshold)
    return gold.assign(is_anomaly=flagged)


def split_frame(
    frame: pd.DataFrame,
    *,
    purge: bool = True,
    embargo_days: int = EMBARGO_DAYS,
) -> dict[str, pd.DataFrame]:
    """Cut a labelled frame into train, validation and test. Strictly by time.

    The one function that does this. Not because splitting is hard, but because
    a split written inline is a split written twice, and the second one is
    where the shuffle gets in.

    The two trims are not symmetric, and the asymmetry is the whole argument:

    * ``purge`` drops :data:`PURGE_DAYS` from the **end** of a split, because
      the label reaches forward and a training row must not be labelled by the
      period it is about to be validated on. This is leakage, and it is on.
    * ``embargo_days`` drops days from the **start** of a split, because a
      rolling feature reaches backward across the boundary. This is not
      leakage, it is what deployment looks like, and it is off. See
      :data:`EMBARGO_DAYS`.

    Args:
        frame: Any frame with a ``date_key`` column.
        purge: Drop the label horizon from the end of every split that has a
            later one after it. Off only to demonstrate, in tests, what leaves
            without it.
        embargo_days: Drop this many days from the start of every split that
            has an earlier one before it. Zero by default.

    Returns:
        One frame per split name, each with a fresh index. A split with no rows
        is present and empty rather than absent, so a caller iterating the
        splits cannot silently skip one.
    """
    if embargo_days < 0:
        raise ValueError(f"embargo_days must not be negative, got {embargo_days}.")

    parts: dict[str, pd.DataFrame] = {}
    for index, split in enumerate(SPLITS):
        keep = split.contains(frame["date_key"])
        if purge and index < len(SPLITS) - 1 and split.end is not None:
            boundary = pd.Timestamp(split.end) - pd.Timedelta(days=PURGE_DAYS - 1)
            keep &= frame["date_key"] < boundary
        if embargo_days and index > 0 and split.start is not None:
            opens = pd.Timestamp(split.start) + pd.Timedelta(days=embargo_days)
            keep &= frame["date_key"] >= opens
        parts[split.name] = frame.loc[keep].reset_index(drop=True)
    return parts


def _consecutive(parts: Mapping[str, pd.DataFrame]):
    """Yield ``(earlier_name, earlier, later_name, later)`` for non-empty pairs."""
    ordered = [split.name for split in SPLITS]
    for earlier, later in zip(ordered, ordered[1:]):
        before, after = parts[earlier], parts[later]
        if before.empty or after.empty:
            continue
        yield earlier, before, later, after


def assert_splits_are_ordered(parts: Mapping[str, pd.DataFrame]) -> None:
    """Raise unless every split ends strictly before the next one starts.

    The plain statement of a chronological split, and the one worth writing
    even though :func:`assert_splits_are_disjoint` implies it. This is the
    property that fails loudly the moment somebody shuffles: a random split
    puts 2023 rows in the training set, and the maximum training date jumps
    past the minimum validation date in a way no amount of reading the
    surrounding code would reveal.
    """
    for earlier, before, later, after in _consecutive(parts):
        last, first = before["date_key"].max(), after["date_key"].min()
        if last >= first:
            raise ValueError(
                f"{earlier} reaches {last.date()} and {later} starts "
                f"{first.date()}: the split is not chronological. A shuffle is "
                "the usual cause."
            )


def assert_splits_are_disjoint(parts: Mapping[str, pd.DataFrame]) -> None:
    """Raise unless no *label window* in one split reaches into the next.

    Strictly stronger than :func:`assert_splits_are_ordered`, which it runs
    first: ordering asks that the rows not overlap, this asks that what the
    rows *know* not overlap. A split can be perfectly ordered and still hand
    the training set the first week of validation through the label.
    """
    assert_splits_are_ordered(parts)
    for earlier, before, later, after in _consecutive(parts):
        reach = before["date_key"].max() + pd.Timedelta(days=HORIZON_DAYS)
        if reach >= after["date_key"].min():
            raise ValueError(
                f"the {earlier} split ends {before['date_key'].max().date()}, "
                f"whose label reaches {reach.date()}, into {later}, which "
                f"starts {after['date_key'].min().date()}. Purge the boundary."
            )


def split_summary(parts: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Rows, positives and base rate per split. The context every score needs."""
    rows = []
    for split in SPLITS:
        part = parts[split.name]
        labelled = part[LABEL].notna().sum() if len(part) else 0
        rows.append(
            {
                "split": split.name,
                "start": part["date_key"].min().date() if len(part) else None,
                "end": part["date_key"].max().date() if len(part) else None,
                "rows": len(part),
                "positives": int(positives(part[LABEL]).sum()) if len(part) else 0,
                "base_rate": base_rate(part[LABEL]) if labelled else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def evaluation_frame(
    engine=None, *, threshold: float | None = None, **kwargs
) -> pd.DataFrame:
    """The rows every model and every baseline is scored on. One definition.

    Labelled, past the feature warm-up, and with every model input present.
    None of the three is optional: a row with no label has no answer to be
    right about, and a row the model cannot read is a row a baseline would
    score anyway, which turns "the model beats persistence" into a comparison
    across two different test sets.

    The third condition used to be free. Until the daily backfill widened, the
    only rows with a null feature outside the warm-up belonged to cities that
    were unlabelled anyway, and this function could observe the property rather
    than enforce it. It is no longer free: a city that lands a discontinuous
    record has null windows spanning each hole, far outside its warm-up, and 19
    such rows arrived with London, Reykjavík and Sydney. Dropping them here is
    the same act :func:`~machine_learning.features.drop_warmup` documents and
    declines to perform on its own, and it is done in one place so the model
    and the baselines cannot end up disagreeing about which rows exist.

    Args:
        threshold: Re-flag ``is_anomaly`` at this |Z| before building anything,
            for ML-09's sweep. ``None`` reads the warehouse's own column, which
            is what every committed number is measured on. Passing
            :data:`ANOMALY_THRESHOLD` explicitly is not the same code path and
            is not meant to be: a test asserts the two agree row for row, which
            is what makes the sweep's middle point evidence about the shipped
            pipeline rather than about a copy of it.
    """
    if threshold is None:
        whole = add_persistence_signal(training_frame(engine, **kwargs))
    else:
        # One read, re-flagged, then the ordinary pipeline. `training_frame`
        # takes the frame rather than the sweep rebuilding features and labels
        # itself, so the merge that aligns them stays in one place.
        gold = reflag(gold_frame(engine, **kwargs), threshold)
        whole = add_persistence_signal(training_frame(frame=gold))
    return drop_scorable_gaps(drop_warmup(drop_unlabelled(whole)))


def drop_scorable_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove rows carrying a null model input, and say how many left.

    Logged rather than silent. This is the one trim in the population that is
    a fact about the data rather than about the arithmetic, so a run in which
    it suddenly removes thousands of rows should be readable from the output
    of the run and not from a diff in ``metrics.json``.
    """
    if "has_missing_feature" not in frame.columns:
        raise ValueError(
            "'has_missing_feature' is missing, so nothing here knows which "
            "rows the model can read. Build the frame with build_features()."
        )
    unreadable = frame["has_missing_feature"].fillna(True).to_numpy(dtype=bool)
    if unreadable.any():
        by_city = frame.loc[unreadable, "city_id"].value_counts().to_dict()
        log.info(
            "dropping %d row(s) with a null model input, past the warm-up: %s",
            int(unreadable.sum()),
            ", ".join(f"{city} {count}" for city, count in sorted(by_city.items())),
        )
    return frame.loc[~unreadable].reset_index(drop=True)


def add_persistence_signal(frame: pd.DataFrame) -> pd.DataFrame:
    """Add "was there an anomaly in t-6 .. t", as a nullable boolean.

    **Computed before the population is trimmed, and that ordering is the whole
    point.** The window looks back seven days, so a row needs seven days of
    record behind it, which every row past a thirty-day feature warm-up has.
    Compute it on the trimmed population instead and the first six rows of each
    city lose a signal they are entitled to, because the window falls off the
    start of the *slice* rather than off the start of the record. It cost 36
    rows and a third probability cell that should not exist, which is a small
    number and exactly the shape of the mistake this project keeps finding: a
    window measured against the request instead of against the data.

    Uses the same counter as ``anomaly_days_trailing30`` rather than a second
    implementation, so the two cannot drift on what a null flag means: an
    unscored day is not a quiet day, and a window spanning a hole is unknown
    rather than empty.
    """
    prepared = require_grain(frame, ("city_id", "date_key", "is_anomaly"))
    calendar = on_daily_calendar(prepared)
    flagged, scored_days = trailing_anomaly_counts(calendar, PERSISTENCE_WINDOW)

    signal = pd.Series(pd.NA, index=calendar.index, dtype="boolean")
    # A positive is certain from one flagged day; a negative needs the whole
    # window scored. The same asymmetry the label itself is built on.
    signal = signal.mask(flagged > 0, True)
    signal = signal.mask((flagged == 0) & (scored_days == PERSISTENCE_WINDOW), False)

    resolved = calendar.assign(**{PERSISTENCE_FLAG: signal, PERSISTENCE_COUNT: flagged})
    resolved = resolved.loc[resolved["is_observed"]]
    columns = ["city_id", "date_key", PERSISTENCE_FLAG, PERSISTENCE_COUNT]

    # Recomputed, not appended. Merging onto a frame that already carries these
    # columns would silently rename both copies to `_x` and `_y` and leave
    # neither under the name every caller looks up.
    target = frame.drop(columns=[PERSISTENCE_FLAG, PERSISTENCE_COUNT], errors="ignore")
    merged = target.merge(
        resolved.loc[:, columns], on=["city_id", "date_key"], how="left"
    )
    if len(merged) != len(frame):
        raise ValueError(
            f"the persistence signal changed the grain: {len(frame)} rows in, "
            f"{len(merged)} out."
        )
    return merged



def boundary_report(parts: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """What sits either side of every boundary, and how much air is between.

    ``gap_days`` is the distance from the last row of one split to the first of
    the next; ``label_reach`` is where the last row's label window ends. The
    second must land before the next split starts, and printing both makes the
    purge visible as a number rather than as a claim in a docstring.
    """
    rows = []
    for earlier, before, later, after in _consecutive(parts):
        last, first = before["date_key"].max(), after["date_key"].min()
        rows.append(
            {
                "boundary": f"{earlier} -> {later}",
                "last": last.date(),
                "first": first.date(),
                "gap_days": int((first - last).days),
                "label_reach": (last + pd.Timedelta(days=HORIZON_DAYS)).date(),
                "reach_clears": bool(
                    last + pd.Timedelta(days=HORIZON_DAYS) < first
                ),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Leave-one-city-out (ML-08)
# --------------------------------------------------------------------------

#: The smallest held-out test split worth quoting a PR-AUC for.
#:
#: Both numbers, not one. Rows alone would admit a city with three hundred test
#: days and four positives, where average precision moves several points if a
#: single row reorders and the lift computed from it is noise wearing four
#: decimal places. Positives alone would admit a city whose test split is one
#: short summer. Two hundred rows is over half a year at the daily grain, and
#: twenty positives is the point below which the number stops being worth a
#: sentence in the README.
#:
#: A city that fails either is **named** in the report rather than dropped from
#: it, the same way an uningested city is: "not enough test rows to hold out"
#: and "held out and scored badly" are opposite findings, and a table that
#: simply lacks the row cannot tell them apart.
MIN_HELD_OUT_ROWS: Final[int] = 200
MIN_HELD_OUT_POSITIVES: Final[int] = 20


def hold_out_city(
    frame: pd.DataFrame, city_id: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition a population into (every other city, this one).

    The whole of the leave-one-city-out design is this one line, and it is a
    function so that exactly one line exists. The alternative -- a boolean mask
    written at the call site next to the code that trains -- is how a fold ends
    up filtering the training frame and forgetting the validation frame, which
    would leak the held-out city into early stopping and the grid search while
    every date-based check in this module still passed.

    Partitions rather than filters: the two frames together are the input, row
    for row, so a row cannot be quietly lost by a mask that means neither.

    Raises:
        ValueError: ``city_id`` is not in the frame. Holding out a city that
            is not there would return the whole population as the training
            fold and an empty frame as the test one, and score nothing while
            looking like it had.
    """
    if "city_id" not in frame.columns:
        raise ValueError("the frame has no 'city_id' column to hold out on.")
    mask = frame["city_id"].to_numpy() == city_id
    if not mask.any():
        raise ValueError(
            f"{city_id!r} is not in this population, so holding it out would "
            f"train on everything and score nothing. Present: "
            f"{sorted(frame['city_id'].unique())}."
        )
    others = frame.loc[~mask].reset_index(drop=True)
    held = frame.loc[mask].reset_index(drop=True)
    return others, held


def assert_city_is_held_out(
    city_id: str,
    folds: Mapping[str, pd.DataFrame],
    *,
    held_out: pd.DataFrame | None = None,
) -> None:
    """Raise unless ``city_id`` is absent from every fold it is fitted on.

    The check the whole experiment rests on. A leave-one-city-out score is only
    evidence about transfer if the model has never seen the city, and there is
    no date, no boundary and no row count that would reveal the failure: a fold
    that quietly kept Cairo in its validation set produces a perfectly ordered
    chronological split, a sensible-looking PR-AUC, and an answer to a question
    nobody asked.

    Args:
        city_id: The city being held out.
        folds: Frames the model is allowed to read -- typically ``train`` and
            ``validation``. Every one is checked.
        held_out: The frame being scored, checked to contain that city and
            nothing else. Optional, because the guard is worth having even
            where only the training side is to hand.
    """
    for name, fold in folds.items():
        if "city_id" not in fold.columns:
            raise ValueError(f"the {name} fold has no 'city_id' column.")
        rows = int((fold["city_id"].to_numpy() == city_id).sum())
        if rows:
            raise ValueError(
                f"{rows} {city_id} row(s) reached the {name} fold of the "
                f"{city_id} hold-out. The model would be scoring a city it "
                "has been trained on, which is the one thing this evaluation "
                "exists to rule out."
            )
    if held_out is not None:
        strangers = sorted(set(held_out["city_id"].unique()) - {city_id})
        if strangers:
            raise ValueError(
                f"the {city_id} hold-out is being scored on {strangers} as "
                "well, so the number would not be about that city."
            )


def scorable_cities(
    parts: Mapping[str, pd.DataFrame],
    *,
    min_rows: int = MIN_HELD_OUT_ROWS,
    min_positives: int = MIN_HELD_OUT_POSITIVES,
) -> tuple[list[str], dict[str, str]]:
    """Which cities have a test split worth holding out, and why the rest do not.

    Returns the eligible cities and, beside them, a reason for every city that
    reaches the test split and is turned away. A city that reaches it with two
    rows is a statement about the backfill and not about the model, and the
    difference has to survive into the report; a city that does not reach the
    test split at all is accounted for one level up, by
    :func:`~machine_learning.evaluate.hold_out_reasons`, which is where the
    roster is known.
    """
    test = parts["test"]
    eligible: list[str] = []
    excluded: dict[str, str] = {}
    for city_id, group in test.groupby("city_id", sort=True):
        truth = positives(group[LABEL])
        rows, positive = len(group), int(truth.sum())
        if rows < min_rows:
            excluded[str(city_id)] = (
                f"{rows} test rows, under the {min_rows} needed to quote a "
                "PR-AUC"
            )
        elif positive < min_positives:
            excluded[str(city_id)] = (
                f"{positive} positive test rows, under the {min_positives} "
                "needed to quote a PR-AUC"
            )
        elif positive == rows:
            excluded[str(city_id)] = "no negative test rows; PR-AUC undefined"
        else:
            eligible.append(str(city_id))
    return eligible, excluded


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report the chronological split and check its boundaries."
    )
    parser.add_argument(
        "--embargo-days",
        type=int,
        default=EMBARGO_DAYS,
        help="Drop this many days from the start of each later split.",
    )
    parser.add_argument(
        "--no-purge",
        action="store_true",
        help="Skip the end-of-split purge, to see what it was doing.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parts = split_frame(
        evaluation_frame(),
        purge=not args.no_purge,
        embargo_days=args.embargo_days,
    )

    print(f"purge {0 if args.no_purge else PURGE_DAYS} days   "
          f"embargo {args.embargo_days} days\n")
    summary = split_summary(parts)
    summary["base_rate"] = summary["base_rate"].map("{:.2%}".format)
    print(summary.to_string(index=False))

    print()
    print(boundary_report(parts).to_string(index=False))

    print()
    try:
        assert_splits_are_disjoint(parts)
    except ValueError as exc:
        print(f"FAIL  {exc}")
        return 1
    print("OK    every split ends strictly before the next begins,")
    print("      and no label window crosses a boundary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
