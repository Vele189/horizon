"""The two baselines, scored and written down before any model is trained.

A PR-AUC with nothing beside it is not a result. Average precision for a random
ranker is the positive rate, so 0.20 is strong against a 5% base rate and poor
against 25%, and the base rate here moves from 5.50% in the training period to
13.58% in the test one. Reporting a model score without these numbers is the
single most common weakness in a portfolio ML project, and the fix is not to
report them afterwards: it is to fix the target in advance, in a file that is
committed, so the model is measured against a number that was written down
before anyone saw how it did.

Two baselines, both from the proposal:

* **Persistence.** Predict an anomaly next week if one occurred this week.
  The rule is a flag, but a flag has no Brier score worth having: emitting 0
  and 1 makes every mistake maximally confident. So the rule is *calibrated* on
  the training split (what fraction of weeks following an anomalous week were
  themselves anomalous) and the baseline emits that probability. The ranking
  is unchanged, so PR-AUC is the rule's own; only the calibration is fixed.
* **Climatology.** Predict the historical rate for that city and week of year.

And one reference that is not a baseline: the **train base rate**, predicted
constantly for every row. Its PR-AUC is the test base rate by construction, and
it is quoted so that the two real baselines have a floor and not just a ceiling.

**Everything is fitted on the training split only.** The temptation is to
compute "the historical rate" over the whole record, which is one line shorter
and wrong: the label's base rate roughly doubles and doubles again across this
split, so a climatology fitted on all of it would carry the test period's own
answer back into the training period. It would look like a strong baseline and
be an unfair one.

Usage::

    python machine_learning/baselines.py            # score and print
    python machine_learning/baselines.py --write    # and update metrics.json

The scores in ``machine_learning/artifacts/metrics.json`` are tied to a
snapshot of the warehouse, recorded alongside them. The backfill is not
finished; when more cities land these numbers change, and the file has to be
rebuilt and re-committed **before** a model is compared against it, or "fixed in
advance" quietly stops being true.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np
import pandas as pd
from sqlalchemy import Engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings  # noqa: E402
from machine_learning.evaluation import (  # noqa: E402
    EMBARGO_DAYS,
    PERSISTENCE_FLAG,
    PERSISTENCE_WINDOW,
    PURGE_DAYS,
    SPLITS,
    assert_splits_are_disjoint,
    evaluation_frame,
    score,
    split_frame,
    split_summary,
)
from machine_learning.labels import HORIZON_DAYS, LABEL, positives  # noqa: E402

__all__ = [
    "METRICS_SCHEMA_VERSION",
    "PERSISTENCE_WINDOW",
    "SMOOTHING_GRID",
    "Baseline",
    "BaseRateReference",
    "ClimatologyBaseline",
    "PersistenceBaseline",
    "build_metrics",
    "metrics_path",
    "write_metrics",
]

log = logging.getLogger(__name__)

#: Bumped when the shape of ``metrics.json`` changes, so a reader can tell a
#: file it does not understand from one that merely has different numbers.
#:
#: 2. ML-04 added ``split.embargo_days``, so a file records not only the trim
#: that was applied at the end of each split but the one that was deliberately
#: not applied at the start.
#: 3. ML-05 added a top-level ``model`` block, written by ``train.py`` beside
#: the baselines it is compared against. A file may legitimately lack it: the
#: baselines exist before the model does, which is the whole point of them.
#: 4. ML-06 added ``evaluation``, written by ``evaluate.py``: the test-split
#: table, the per-city breakdown, and the per-metric verdict.
#: 5. ML-07 added ``explainability``, written by ``explain.py``: the SHAP
#: ranking, the both-tails response, and the two explained predictions.
METRICS_SCHEMA_VERSION: Final[int] = 5

#: Pseudo-counts tried for the climatology's shrinkage, chosen on **validation**
#: Brier. A (city, week) cell holds around 130 training rows here, so a cell
#: that happened to see no positives would otherwise predict exactly zero, a
#: probability no amount of evidence can justify from 130 observations.
#:
#: The grid runs to infinity on purpose. A tuned parameter that lands on the
#: largest value offered is not a tuned parameter, it is a clipped one, and the
#: first version of this grid stopped at 100 and did exactly that: validation
#: Brier was still improving at the edge. The limit is the honest end of the
#: range: at infinity every week cell collapses to its city's own rate, which
#: is a real hypothesis about this data and, as it turns out, the one
#: validation prefers.
SMOOTHING_GRID: Final[tuple[float, ...]] = (
    0.0,
    1.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
    500.0,
    2000.0,
    float("inf"),
)

def _jsonable(value: Any) -> Any:
    """Render infinity as a string, because JSON has no word for it.

    ``json.dumps`` writes bare ``Infinity``, which Python reads back and every
    strict parser rejects. The pseudo-count grid genuinely runs to the limit,
    so the file has to say so in a way a JavaScript dashboard can also read.
    """
    if isinstance(value, float) and np.isinf(value):
        return "inf"
    return value


@dataclass
class Baseline:
    """A predictor with no parameters worth tuning and no excuse for being wrong.

    Subclasses fit on the training split and emit a probability per row. The
    contract is deliberately the one a scikit-learn estimator would satisfy for
    ``predict_proba``'s positive column, so ML-04's model can be swapped in
    without the scoring code noticing.
    """

    name: str = "baseline"
    fitted: bool = field(default=False, init=False, repr=False)

    def fit(self, train: pd.DataFrame) -> "Baseline":
        raise NotImplementedError

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> dict[str, Any]:
        return {}

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError(
                f"{self.name} has not been fitted. Fit on the training split "
                "before predicting, or the baseline is scoring against data it "
                "has already seen."
            )


@dataclass
class BaseRateReference(Baseline):
    """Predict the training positive rate, constantly. Not a baseline, a floor.

    Its PR-AUC on any split is that split's own base rate, to within tie
    handling, which makes it the sanity check on every other number in the
    file: a baseline that does not beat this one has no skill at all.
    """

    name: str = "base_rate"
    rate: float = float("nan")

    def fit(self, train: pd.DataFrame) -> "BaseRateReference":
        self.rate = float(positives(train[LABEL]).mean())
        self.fitted = True
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        self._require_fitted()
        return np.full(len(frame), self.rate, dtype=float)

    def params(self) -> dict[str, Any]:
        return {"rate": self.rate}


@dataclass
class PersistenceBaseline(Baseline):
    """Predict an anomaly next week if one occurred this week.

    Three cells, not two. "An anomalous week", "a quiet week fully observed",
    and "a week that could not be judged" are different states, and folding the
    third into the second would assert a week was quiet on the strength of a
    gap in the record. On the current snapshot the third cell is **empty**, and
    that is a property worth checking rather than assuming: the population
    starts thirty days into each city's record, so a seven-day window always
    closes. It held 36 rows until the signal was moved to be computed before
    the population was trimmed.

    Each cell's probability is the observed rate of positives in that cell
    **in the training split**. That is what makes the Brier score meaningful:
    the rule ranks, and the training split calibrates.
    """

    name: str = "persistence"
    after_anomaly: float = float("nan")
    after_quiet: float = float("nan")
    when_unknown: float = float("nan")
    cell_rows: dict[str, int] = field(default_factory=dict)

    def fit(self, train: pd.DataFrame) -> "PersistenceBaseline":
        signal = train[PERSISTENCE_FLAG]
        target = positives(train[LABEL])
        overall = float(target.mean())

        cells = {
            "after_anomaly": signal.eq(True).fillna(False).to_numpy(dtype=bool),
            "after_quiet": signal.eq(False).fillna(False).to_numpy(dtype=bool),
            "when_unknown": signal.isna().to_numpy(dtype=bool),
        }
        for cell, mask in cells.items():
            rows = int(mask.sum())
            self.cell_rows[cell] = rows
            # An empty cell falls back to the overall training rate rather than
            # to nan: an unfittable cell is not a licence to emit a number that
            # cannot be scored.
            setattr(
                self, cell, float(target.to_numpy()[mask].mean()) if rows else overall
            )
        self.fitted = True
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        self._require_fitted()
        signal = frame[PERSISTENCE_FLAG]
        out = np.full(len(frame), self.when_unknown, dtype=float)
        out[signal.eq(True).fillna(False).to_numpy(dtype=bool)] = self.after_anomaly
        out[signal.eq(False).fillna(False).to_numpy(dtype=bool)] = self.after_quiet
        return out

    def params(self) -> dict[str, Any]:
        return {
            "window_days": PERSISTENCE_WINDOW,
            "after_anomaly": self.after_anomaly,
            "after_quiet": self.after_quiet,
            "when_unknown": self.when_unknown,
            "train_rows_per_cell": dict(self.cell_rows),
        }


@dataclass
class ClimatologyBaseline(Baseline):
    """Predict the historical rate for that city and ISO week of year.

    Fitted on the training split, so the rate for Cairo in week 30 is what
    Cairo's week 30 did between 1995 and 2018, not what it did over the whole
    record, which would carry the test period's answer back into training.

    Each cell is shrunk towards its city's own training rate rather than left
    raw. A (city, week) cell holds roughly 130 training rows; a cell that
    happened to see no positives would otherwise predict exactly zero, which
    130 observations cannot justify and which a Brier score punishes hard the
    first time it is wrong. The pseudo-count is chosen on the **validation**
    split, never on test.

    Cities and weeks unseen in training fall back to the city rate, then to the
    overall rate, in that order, because a city's own level is a better guess
    than the pooled one.
    """

    name: str = "climatology"
    smoothing: float = 20.0
    overall: float = float("nan")
    city_rates: dict[str, float] = field(default_factory=dict)
    cell_rates: dict[tuple[str, int], float] = field(default_factory=dict)
    cell_rows: dict[tuple[str, int], int] = field(default_factory=dict)
    smoothing_search: list[dict[str, float]] = field(default_factory=list)

    @staticmethod
    def week_of_year(frame: pd.DataFrame) -> pd.Series:
        """ISO week, 1-53.

        ISO rather than ``day_of_year // 7`` because week 1 is defined by where
        the year's first Thursday falls, so it stays aligned to the weekly
        cycle across the leap-year boundary instead of drifting a day every
        four years. Week 53 exists in some years and is sparse; the shrinkage
        is what stops that being a problem.
        """
        return frame["date_key"].dt.isocalendar().week.astype(int)

    def fit(self, train: pd.DataFrame) -> "ClimatologyBaseline":
        target = positives(train[LABEL])
        weeks = self.week_of_year(train)
        self.overall = float(target.mean())
        self.city_rates = {
            str(city): float(group.mean())
            for city, group in target.groupby(train["city_id"].to_numpy())
        }

        grouped = pd.DataFrame(
            {"city_id": train["city_id"].to_numpy(), "week": weeks.to_numpy(),
             "y": target.to_numpy()}
        ).groupby(["city_id", "week"])["y"].agg(["sum", "count"])

        self.cell_rates = {}
        self.cell_rows = {}
        for (city, week), row in grouped.iterrows():
            prior = self.city_rates.get(str(city), self.overall)
            if np.isinf(self.smoothing):
                # The limit, computed rather than approached. At a large finite
                # pseudo-count the week term survives as a rounding-sized
                # perturbation that still breaks ties, and on this data it
                # breaks them the wrong way, costing more than it is worth.
                rate = prior
            else:
                rate = (row["sum"] + self.smoothing * prior) / (
                    row["count"] + self.smoothing
                )
            self.cell_rates[(str(city), int(week))] = float(rate)
            self.cell_rows[(str(city), int(week))] = int(row["count"])
        self.fitted = True
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        self._require_fitted()
        weeks = self.week_of_year(frame).to_numpy()
        cities = frame["city_id"].to_numpy()
        return np.array(
            [
                self.cell_rates.get(
                    (str(city), int(week)),
                    self.city_rates.get(str(city), self.overall),
                )
                for city, week in zip(cities, weeks)
            ],
            dtype=float,
        )

    def params(self) -> dict[str, Any]:
        counts = list(self.cell_rows.values())
        return {
            "smoothing": _jsonable(self.smoothing),
            "smoothing_grid": [_jsonable(value) for value in SMOOTHING_GRID],
            "smoothing_search": [
                {key: _jsonable(value) for key, value in entry.items()}
                for entry in self.smoothing_search
            ],
            "collapsed_to_city_rate": bool(np.isinf(self.smoothing)),
            "cells": len(self.cell_rates),
            "median_rows_per_cell": (
                float(np.median(counts)) if counts else float("nan")
            ),
            "min_rows_per_cell": int(min(counts)) if counts else 0,
            "overall_rate": self.overall,
        }

    @classmethod
    def tuned(
        cls, train: pd.DataFrame, validation: pd.DataFrame
    ) -> "ClimatologyBaseline":
        """Fit at every pseudo-count in the grid; keep the best validation Brier.

        Brier rather than PR-AUC, because shrinkage is a calibration decision:
        it moves probabilities towards a prior and barely reorders anything, so
        PR-AUC would be nearly flat across the grid and the choice would be
        noise. Test is not consulted.
        """
        search: list[dict[str, float]] = []
        candidates: list[tuple[float, ClimatologyBaseline]] = []
        for smoothing in SMOOTHING_GRID:
            candidate = cls(smoothing=smoothing).fit(train)
            result = score(validation[LABEL], candidate.predict(validation))
            search.append(
                {
                    "smoothing": smoothing,
                    "validation_brier": result.brier,
                    "validation_pr_auc": result.pr_auc,
                }
            )
            candidates.append((result.brier, candidate))

        # Ties break towards the first, which is the least shrinkage that did
        # as well; the grid is ordered, so the choice is deterministic.
        best = min(candidates, key=lambda item: item[0])[1]
        best.smoothing_search = search
        return best


def metrics_path() -> Path:
    """``machine_learning/artifacts/metrics.json``, from the configured dir."""
    return get_settings().model_artifact_dir / "metrics.json"


def _snapshot(frame: pd.DataFrame) -> dict[str, Any]:
    """What the scores were computed against.

    Without this the file says "the target was fixed in advance" and cannot
    prove it. The backfill is mid-flight, so a metrics file with no snapshot is
    a number attached to a dataset nobody can reconstruct.
    """
    per_city = frame.groupby("city_id").agg(
        rows=("date_key", "size"),
        first=("date_key", "min"),
        last=("date_key", "max"),
    )
    return {
        "rows": len(frame),
        "cities": int(frame["city_id"].nunique()),
        "first_date": str(frame["date_key"].min().date()),
        "last_date": str(frame["date_key"].max().date()),
        "rows_by_city": {
            str(city): int(row["rows"]) for city, row in per_city.iterrows()
        },
        "last_date_by_city": {
            str(city): str(row["last"].date()) for city, row in per_city.iterrows()
        },
    }


def build_metrics(
    engine: Engine | None = None, *, frame: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Fit both baselines on train and score them on validation and test.

    Returns the whole ``metrics.json`` payload, including the split summary and
    the warehouse snapshot the numbers belong to.
    """
    population = frame if frame is not None else evaluation_frame(engine)
    # Sorted before anything is measured. Brier is a mean over a float array,
    # and a mean is summation-order dependent in its last digit or two, enough
    # to make a committed file differ from one run to the next for no reason a
    # reviewer could act on. The real path arrives sorted already; this makes
    # it true whatever the caller hands over.
    population = population.sort_values(
        ["city_id", "date_key"], kind="stable"
    ).reset_index(drop=True)
    if PERSISTENCE_FLAG not in population.columns:
        raise ValueError(
            f"{PERSISTENCE_FLAG!r} is missing. It has to be computed on the "
            "whole record and not on the scored population. Build the frame "
            "with evaluation_frame(), which does it in that order, or the "
            "first rows of each city lose a signal they are entitled to."
        )
    parts = split_frame(population)
    assert_splits_are_disjoint(parts)

    train, validation = parts["train"], parts["validation"]
    for name, part in parts.items():
        if part.empty:
            raise ValueError(
                f"the {name} split is empty on this snapshot, so no baseline "
                "can be scored on it. Backfill further before fixing a target."
            )

    baselines: list[Baseline] = [
        BaseRateReference().fit(train),
        PersistenceBaseline().fit(train),
        ClimatologyBaseline.tuned(train, validation),
    ]

    scored: dict[str, Any] = {}
    for baseline in baselines:
        entry: dict[str, Any] = {
            "fitted_on": "train",
            "params": baseline.params(),
        }
        for split_name in ("train", "validation", "test"):
            result = score(
                parts[split_name][LABEL], baseline.predict(parts[split_name])
            )
            entry[split_name] = result.as_dict()
        scored[baseline.name] = entry

    summary = split_summary(parts)
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "label": {
            "window": f"t+1 .. t+{HORIZON_DAYS} inclusive",
            "horizon_days": HORIZON_DAYS,
        },
        "split": {
            "purge_days": PURGE_DAYS,
            "embargo_days": EMBARGO_DAYS,
            "periods": [
                {
                    "name": split.name,
                    "start": str(split.start) if split.start else None,
                    "end": str(split.end) if split.end else None,
                }
                for split in SPLITS
            ],
            "summary": [
                {
                    key: (str(value) if isinstance(value, dt.date) else value)
                    for key, value in row.items()
                }
                for row in summary.to_dict("records")
            ],
        },
        "snapshot": _snapshot(population),
        "baselines": scored,
    }


def write_metrics(payload: Mapping[str, Any], path: Path | None = None) -> Path:
    """Write ``metrics.json``, pretty and with a trailing newline.

    Formatted for a diff rather than for a parser: this file is committed, and
    the point of committing it is that a change to a baseline score shows up in
    a review as a changed line.
    """
    destination = Path(path) if path is not None else metrics_path()
    destination.parent.mkdir(parents=True, exist_ok=True)

    # The downstream blocks are kept only while they still describe the same
    # data. If the snapshot has moved, the model was scored against a target
    # that no longer exists, and leaving its numbers beside the new baselines
    # would invite exactly the comparison nobody made.
    payload = dict(payload)
    if destination.exists():
        previous = json.loads(destination.read_text())
        downstream = ("model", "evaluation", "explainability")
        carried = {key: previous[key] for key in downstream if key in previous}
        if carried:
            if previous.get("snapshot") == payload.get("snapshot"):
                payload.update(carried)
            else:
                log.warning(
                    "dropping %s: they describe %s rows to %s and the "
                    "baselines now describe %s rows to %s. Re-run train.py "
                    "--write, evaluate.py --write and explain.py --write.",
                    " and ".join(sorted(carried)),
                    previous["snapshot"]["rows"],
                    previous["snapshot"]["last_date"],
                    payload["snapshot"]["rows"],
                    payload["snapshot"]["last_date"],
                )

    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return destination


def _format_row(name: str, split: str, result: Mapping[str, Any]) -> str:
    return (
        f"  {name:<12} {split:<11} "
        f"PR-AUC {result['pr_auc']:.4f}  "
        f"base {result['base_rate']:.4f}  "
        f"lift {result['lift']:.2f}x  "
        f"Brier {result['brier']:.5f}"
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the persistence and climatology baselines."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Update machine_learning/artifacts/metrics.json.",
    )
    parser.add_argument("--out", help="Write somewhere other than the default path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    payload = build_metrics()

    print("splits")
    for row in payload["split"]["summary"]:
        print(
            f"  {row['split']:<11} {row['start']} .. {row['end']}  "
            f"{row['rows']:>6,} rows  {row['positives']:>5,} positive  "
            f"{row['base_rate']:.2%}"
        )
    print(f"\n  purge: {payload['split']['purge_days']} days at each boundary")

    print("\nbaselines (fitted on train)")
    for name, entry in payload["baselines"].items():
        for split_name in ("validation", "test"):
            print(_format_row(name, split_name, entry[split_name]))

    if args.write or args.out:
        written = write_metrics(payload, Path(args.out) if args.out else None)
        print(f"\nwrote {written}")
    else:
        print("\n(not written; pass --write to update metrics.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
