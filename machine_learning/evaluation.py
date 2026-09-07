"""The chronological split, and the two numbers everything is scored on.

Separate from the baselines and from the model because all three have to be
measured the same way. A baseline scored with one implementation of average
precision and a model scored with another are not comparable, and the
difference would live entirely in tie handling — where a rule-based baseline
puts thousands of rows on the same score and a trained model puts none. So the
metrics come from ``sklearn`` and are called from one place.

**The split is chronological, and it is purged.** Train to 2018, validate
2019–2021, test 2022 onwards, exactly as the proposal specifies. A random split
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
train and 13.58% in test — so 0.20 is a good score on one and a poor one on the
other. Every :class:`Score` carries the base rate it was measured against, and
:func:`score` also returns the lift over it, because a PR-AUC with no reference
is the thing this whole ticket exists to prevent.

Usage::

    from machine_learning.evaluation import SPLITS, score, split_frame

    parts = split_frame(frame)
    result = score(parts["test"][LABEL], predictions)
    print(result.pr_auc, result.base_rate, result.lift)
"""

from __future__ import annotations

import datetime as dt
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Mapping

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
    on_daily_calendar,
    require_grain,
    trailing_anomaly_counts,
)

__all__ = [
    "PERSISTENCE_COUNT",
    "PERSISTENCE_FLAG",
    "PERSISTENCE_WINDOW",
    "PURGE_DAYS",
    "SPLITS",
    "Score",
    "Split",
    "assert_splits_are_disjoint",
    "add_persistence_signal",
    "base_rate",
    "evaluation_frame",
    "score",
    "split_frame",
    "split_summary",
]


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
#: validation. Purging is cheap here — seven rows per city per boundary — and
#: the alternative is a training set that has been told the first week of the
#: period it is about to be validated on.
PURGE_DAYS: Final[int] = HORIZON_DAYS


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
    """The positive rate — and the PR-AUC a random ranker would score."""
    return float(positives(labels).mean())


def score(labels: pd.Series, predictions) -> Score:
    """Average precision and Brier, with the base rate they are read against.

    Args:
        labels: The nullable-boolean target. Must contain no nulls — an
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


def split_frame(
    frame: pd.DataFrame, *, purge: bool = True
) -> dict[str, pd.DataFrame]:
    """Cut a labelled frame into train, validation and test.

    Args:
        frame: Any frame with a ``date_key`` column.
        purge: Drop :data:`PURGE_DAYS` from the end of every split that has a
            later one after it, so no label window crosses a boundary. Off only
            to demonstrate, in tests, what leaves without it.

    Returns:
        One frame per split name, each with a fresh index. A split with no rows
        is present and empty rather than absent, so a caller iterating the
        splits cannot silently skip one.
    """
    parts: dict[str, pd.DataFrame] = {}
    for index, split in enumerate(SPLITS):
        keep = split.contains(frame["date_key"])
        if purge and index < len(SPLITS) - 1 and split.end is not None:
            boundary = pd.Timestamp(split.end) - pd.Timedelta(days=PURGE_DAYS - 1)
            keep &= frame["date_key"] < boundary
        parts[split.name] = frame.loc[keep].reset_index(drop=True)
    return parts


def assert_splits_are_disjoint(parts: Mapping[str, pd.DataFrame]) -> None:
    """Raise unless no label window in one split reaches into the next.

    The check the purge exists to satisfy, written as an assertion rather than
    a comment so it is run rather than believed.
    """
    ordered = [split.name for split in SPLITS]
    for earlier, later in zip(ordered, ordered[1:]):
        before, after = parts[earlier], parts[later]
        if before.empty or after.empty:
            continue
        reach = before["date_key"].max() + pd.Timedelta(days=HORIZON_DAYS)
        if reach >= after["date_key"].min():
            raise ValueError(
                f"the {earlier} split ends {before['date_key'].max().date()}, "
                f"whose label reaches {reach.date()} — into {later}, which "
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


def evaluation_frame(engine=None, **kwargs) -> pd.DataFrame:
    """The rows every model and every baseline is scored on. One definition.

    Labelled, and past the feature warm-up. Both conditions matter and neither
    is optional: a row with no label has no answer to be right about, and a row
    inside the warm-up has null features, so a baseline that ignores features
    could be scored on rows the model cannot see. Comparing the two would then
    be comparing different test sets, which is the failure this ticket exists
    to prevent wearing a different hat.

    On this snapshot the two conditions leave a population with **no missing
    feature at all** — the rows with a null feature outside the warm-up belong
    to the three cities that are unlabelled anyway — and a test asserts it, so
    a future city cannot quietly introduce a row the model must skip and the
    baseline scores.
    """
    whole = add_persistence_signal(training_frame(engine, **kwargs))
    return drop_warmup(drop_unlabelled(whole))

def add_persistence_signal(frame: pd.DataFrame) -> pd.DataFrame:
    """Add "was there an anomaly in t−6 .. t", as a nullable boolean.

    **Computed before the population is trimmed, and that ordering is the whole
    point.** The window looks back seven days, so a row needs seven days of
    record behind it — which every row past a thirty-day feature warm-up has.
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

