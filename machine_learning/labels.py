"""Builds the binary target: an anomaly in the forward seven-day window.

This is the one place in the project where looking forward is correct, and it
lives in its own module for that reason. ``features.py`` must never contain a
window that reaches past *t*; keeping the single deliberate look-ahead in a
separate file means a forward shift cannot be added there by accident, and
cannot be missed in review here.

**The window is t+1 .. t+7, inclusive at both ends.** Seven days, and day *t*
is not one of them. Both boundaries are load-bearing:

* *t is excluded* because it is a feature. ``anomaly_days_trailing30`` counts
  today's flag, and a label window that also started today would hand the
  classifier its own answer through a column that looks entirely innocent.
* *t+7 is included* because "within a week" means seven days, not six. An
  off-by-one here does not fail anything: it produces a slightly different
  positive rate and a model that is quietly answering a different question.

``tests/test_labels.py`` asserts both ends by putting a single anomaly into an
otherwise quiet series and checking that it labels **exactly** the seven rows
before it: not the day itself, not the eighth day back.

**A positive is certain; a negative has to be earned.** The label is:

============================ ==========================================
``True``                     at least one day in the window is flagged
``False``                    no day is flagged **and all seven are scored**
``<NA>``                     otherwise: the window is not fully knowable
============================ ==========================================

That third row is where the last seven days of every city series go, and they
go there by the same rule as everything else rather than by a special case: at
the end of the record the forward window runs off the edge, fewer than seven
days are scored, and the label is unknown. It is also where a hole in a series
goes, and where the three cities with no climatology baseline go, all 365
days of each, since a day that could never be flagged cannot make a window
quiet. Coercing any of those to ``False`` would put days that were never
measured into the negative class and deflate the positive rate with them.

Usage::

    from machine_learning.labels import LABEL, drop_unlabelled, training_frame

    frame = drop_unlabelled(training_frame())
    y = frame[LABEL]

Run ``python machine_learning/labels.py`` for the positive rate overall and
per city.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import pandas as pd
from sqlalchemy import Engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from machine_learning.features import (  # noqa: E402
    FeatureError,
    build_features,
    drop_warmup,
    feature_columns,
    gold_frame,
    on_daily_calendar,
    require_grain,
)

__all__ = [
    "HORIZON_DAYS",
    "LABEL",
    "LABEL_COLUMNS",
    "LABEL_REQUIRED_COLUMNS",
    "WINDOW_END",
    "WINDOW_START",
    "build_labels",
    "drop_unlabelled",
    "load_labels",
    "positive_rate_by_city",
    "positives",
    "single_feature_auc",
    "training_frame",
]

log = logging.getLogger(__name__)

#: The forecast horizon in days. The window is t+1 .. t+7 inclusive.
HORIZON_DAYS: Final[int] = 7

#: The offsets the window covers, stated as a range rather than implied by a
#: loop bound. ``WINDOW_START = 1`` is what excludes day *t* from its own
#: label; ``WINDOW_END = 7`` is what makes "within a week" seven days.
WINDOW_START: Final[int] = 1
WINDOW_END: Final[int] = HORIZON_DAYS

#: The target column.
LABEL: Final[str] = f"anomaly_within_{HORIZON_DAYS}d"

#: The count of flagged days in the window. A *lower bound* wherever the window
#: holds unscored days, and carried because the label throws away how bad the
#: week was, useful for ML-05's severity work, and the column that makes an
#: off-by-one in the window visible as a changed distribution rather than as a
#: changed rate of a few tenths of a percent.
FORWARD_COUNT: Final[str] = f"anomaly_days_forward{HORIZON_DAYS}"

#: How many of the seven days carried a flag at all. The denominator, and the
#: reason a ``False`` can be trusted: it is only ever written when this is 7.
SCORED_COUNT: Final[str] = f"label_days_scored{HORIZON_DAYS}"

LABEL_COLUMNS: Final[tuple[str, ...]] = (LABEL, FORWARD_COUNT, SCORED_COUNT)

#: What :func:`build_labels` needs. Notably *not* the weather: the label is a
#: function of the anomaly flag alone, so no feature can enter it by accident.
LABEL_REQUIRED_COLUMNS: Final[tuple[str, ...]] = ("city_id", "date_key", "is_anomaly")


def build_labels(observations: pd.DataFrame) -> pd.DataFrame:
    """Turn gold rows into the target. Pure, and never touches a database.

    Args:
        observations: One row per city-day carrying
            :data:`LABEL_REQUIRED_COLUMNS`. Order does not matter; duplicates
            are an error.

    Returns:
        One row per **observed** city-day (the same count as the input)
        with ``city_id``, ``date_key`` and :data:`LABEL_COLUMNS`, sorted by
        ``(city_id, date_key)``. Unlabellable rows are present and null, not
        removed; :func:`drop_unlabelled` is the thing that removes them.

    Raises:
        FeatureError: A required column is missing, or a city-day repeats.
    """
    frame = require_grain(observations, LABEL_REQUIRED_COLUMNS)
    calendar = on_daily_calendar(frame)

    flag = calendar["is_anomaly"]
    observed = calendar["is_observed"]
    # Split before shifting, so "flagged" and "flagged or unknown" cannot be
    # confused by a shift that turns a missing row into a False.
    hit = pd.Series(
        np.where(observed, flag.fillna(False).to_numpy(dtype=bool), False),
        index=calendar.index,
    )
    known = pd.Series(
        np.where(observed, flag.notna().to_numpy(dtype=bool), False),
        index=calendar.index,
    )
    calendar["_hit"] = hit
    calendar["_known"] = known
    grouped = calendar.groupby("city_id", sort=False)

    # The window, enumerated. Seven shifts rather than a rolling frame on a
    # reversed series: the offsets t+1 .. t+7 are the whole specification of
    # this module, and they should be readable as those offsets.
    forward_hits = pd.Series(0, index=calendar.index, dtype="int64")
    forward_known = pd.Series(0, index=calendar.index, dtype="int64")
    for offset in range(WINDOW_START, WINDOW_END + 1):
        forward_hits += grouped["_hit"].shift(-offset).fillna(False).astype("int64")
        forward_known += grouped["_known"].shift(-offset).fillna(False).astype("int64")

    complete = forward_known == HORIZON_DAYS
    label = pd.Series(pd.NA, index=calendar.index, dtype="boolean")
    # A positive is certain the moment one flagged day is in the window, even
    # if another day of it is unknown: an anomaly that happened cannot be
    # un-happened by a gap beside it. A negative needs the whole window.
    label = label.mask(forward_hits > 0, True)
    label = label.mask((forward_hits == 0) & complete, False)

    labels = calendar.assign(
        **{LABEL: label, FORWARD_COUNT: forward_hits, SCORED_COUNT: forward_known}
    )
    labels = labels.loc[labels["is_observed"]]
    columns = ["city_id", "date_key", *LABEL_COLUMNS]
    return labels.loc[:, columns].reset_index(drop=True)


def load_labels(
    engine: Engine | None = None,
    *,
    cities: Sequence[str] | None = None,
    start: dt.date | str | None = None,
    end: dt.date | str | None = None,
) -> pd.DataFrame:
    """Read gold and build the target, padding ``end`` by the horizon.

    The padding is the mirror of :func:`~machine_learning.features.load_features`
    padding ``start``. Asking for labels up to 2020-12-31 and building them
    from exactly those rows makes the last week of December unlabellable,
    except the record does not end there, the request does.
    """
    padded = end
    if end is not None:
        padded = pd.Timestamp(end) + pd.Timedelta(days=HORIZON_DAYS)
    frame = gold_frame(engine, cities=cities, start=start, end=padded)
    labels = build_labels(frame)
    if end is not None:
        labels = labels.loc[labels["date_key"] <= pd.Timestamp(end)]
    return labels.reset_index(drop=True)


def training_frame(
    engine: Engine | None = None,
    *,
    cities: Sequence[str] | None = None,
    start: dt.date | str | None = None,
    end: dt.date | str | None = None,
) -> pd.DataFrame:
    """Features and label on one row, read in a single pass over gold.

    One function rather than two calls the trainer joins itself, because the
    join has exactly one correct form, an inner join on ``(city_id,
    date_key)`` against frames of identical grain, and every incorrect form
    produces a frame that trains. A merge that silently dropped rows, or one
    that shifted the label by a day, would show up as a metric nobody can
    explain.

    A slice is filtered at the end rather than at the query: the read runs
    from the start of each city's record and :data:`HORIZON_DAYS` past the
    requested end, so every row carries the features and the label it would
    carry in a full build. Trimming first would give the first rows of the
    slice a warm-up they have already served and the last rows a label the
    warehouse can answer.
    """
    padded_end = pd.Timestamp(end) + pd.Timedelta(days=HORIZON_DAYS) if end else None
    frame = gold_frame(engine, cities=cities, start=None, end=padded_end)
    features = build_features(frame)
    labels = build_labels(frame)

    merged = features.merge(labels, on=["city_id", "date_key"], how="inner")
    if len(merged) != len(features):
        raise FeatureError(
            f"features and labels did not align: {len(features)} feature rows, "
            f"{len(labels)} label rows, {len(merged)} joined. Both are built "
            "from the same gold frame at the same grain, so a mismatch means "
            "one of them changed the grain."
        )
    if start is not None:
        merged = merged.loc[merged["date_key"] >= pd.Timestamp(start)]
    if end is not None:
        merged = merged.loc[merged["date_key"] <= pd.Timestamp(end)]
    return merged.reset_index(drop=True)


def positives(label: pd.Series) -> pd.Series:
    """``True`` where the label is positive, ``False`` for negative *and* null.

    The target is a nullable boolean, so ``label == True`` keeps NA as NA and
    counting the result is not counting positives. One helper rather than that
    comparison scattered around, because the difference between "not positive"
    and "not known" is the distinction this whole module is built on and it
    should not be re-derived at every call site.
    """
    return label.fillna(False).astype(bool)


def drop_unlabelled(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove rows whose label is unknown. An explicit act, like the warm-up.

    Never done inside :func:`build_labels`, for the same reason the warm-up is
    not dropped inside ``build_features``: the row count that leaves this
    module has to match the row count in the warehouse, or the difference is
    discovered as an unexplained gap in a metric rather than as a decision.
    """
    return frame.loc[frame[LABEL].notna()].reset_index(drop=True)


def positive_rate_by_city(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-city label counts and positive rate, plus what is unlabellable.

    Per city rather than only overall, because one pooled rate hides the two
    failures worth catching: a city contributing no labelled rows at all, and a
    city whose rate is nothing like the rest of the set.
    """
    rows = []
    for city_id, group in frame.groupby("city_id", sort=True):
        labelled = group[LABEL].notna()
        positive = int(positives(group[LABEL]).sum())
        rows.append(
            {
                "city_id": city_id,
                "rows": len(group),
                "labelled": int(labelled.sum()),
                "unlabelled": int((~labelled).sum()),
                "positives": positive,
                "positive_rate": (
                    positive / int(labelled.sum()) if labelled.any() else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def single_feature_auc(frame: pd.DataFrame, columns: Sequence[str] | None = None):
    """Rank AUC of every feature taken alone, against the label.

    The test for "no feature reconstructs the label" needs a measure, and exact
    reconstruction is the wrong one: on floating-point columns every value is
    unique, so *some* function maps each of them to the label and the check
    passes vacuously. AUC asks the question that matters, which is whether
    this column on its own can order the city-days so that every positive comes
    first, and answers 1.0 for a leaked label and around 0.5 for noise.

    Computed by ranks rather than by thresholds, so it is invariant to any
    monotone transform: a leak does not escape by being logged or negated.
    Rows where the feature is null are dropped, and the count is reported, so a
    column with two non-null rows cannot post a perfect score.

    Returns:
        A frame of ``feature``, ``auc``, ``rows``, sorted by distance from 0.5.
    """
    columns = list(columns) if columns is not None else list(feature_columns())
    target = frame[LABEL]
    rows = []
    for column in columns:
        usable = target.notna() & frame[column].notna()
        labels = target.loc[usable].astype(bool).to_numpy()
        values = frame.loc[usable, column].to_numpy(dtype=float)
        positives = int(labels.sum())
        negatives = int(len(labels) - positives)
        if positives == 0 or negatives == 0:
            rows.append({"feature": column, "auc": float("nan"), "rows": len(labels)})
            continue
        ranks = pd.Series(values).rank().to_numpy()
        auc = (ranks[labels].sum() - positives * (positives + 1) / 2) / (
            positives * negatives
        )
        rows.append({"feature": column, "auc": float(auc), "rows": len(labels)})
    report = pd.DataFrame(rows)
    return report.reindex(
        (report["auc"] - 0.5).abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and describe the forward-window anomaly label."
    )
    parser.add_argument(
        "--city", action="append", dest="cities", help="Restrict to a city_id."
    )
    parser.add_argument("--start", help="Earliest date in the result (YYYY-MM-DD).")
    parser.add_argument("--end", help="Latest date in the result (YYYY-MM-DD).")
    parser.add_argument(
        "--auc",
        action="store_true",
        help="Also score every feature alone against the label.",
    )
    parser.add_argument("--out", help="Write the labelled training frame to CSV.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    frame = training_frame(cities=args.cities, start=args.start, end=args.end)
    labelled = drop_unlabelled(frame)
    positive = int(positives(labelled[LABEL]).sum())

    print(f"window         t+{WINDOW_START} .. t+{WINDOW_END} inclusive")
    print(f"rows           {len(frame):,}")
    print(f"labelled       {len(labelled):,}")
    print(f"unlabelled     {len(frame) - len(labelled):,}")
    if labelled.empty:
        print("no labelled rows")
        return 0
    print(f"positives      {positive:,}")
    print(f"positive rate  {positive / len(labelled):.2%}")

    print("\nby city")
    report = positive_rate_by_city(frame)
    report["positive_rate"] = report["positive_rate"].map("{:.2%}".format)
    print(report.to_string(index=False))

    if args.auc:
        print("\nsingle-feature AUC against the label")
        print(single_feature_auc(drop_warmup(labelled)).head(10).to_string(index=False))

    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        labelled.to_csv(destination, index=False)
        print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
