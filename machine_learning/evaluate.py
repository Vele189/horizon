"""Scores the model against the baselines, and refuses to report accuracy.

**Accuracy is excluded deliberately.** At a 13.58% test base rate, a model that
answers "no anomaly" every single time is 86.4% accurate. Reporting that number
beside a real result is not a rounding of the truth, it is an inversion of it —
and any reader who knows the field reads it as a signal that nobody in the
project understood the base rate. The scikit-learn function for it does not
appear anywhere in this repository, and a test enforces that — the forbidden
names are listed in ``tests/test_evaluation.py`` rather than here, so the scan
does not find its own explanation.

What is reported instead:

* **PR-AUC**, with the base rate printed beside it, because average precision
  for a random ranker *is* the base rate.
* **F1, precision and recall** at a threshold chosen on **validation** — never
  on test, and never at 0.5. A model whose mean prediction is 0.07 classifies
  nothing at 0.5 and would score F1 = 0.00 while ranking better than anything
  else in the table.
* **Brier**, which is the only one of these that notices a probability is
  wrong rather than merely badly ordered.
* **A precision–recall curve and a calibration curve**, saved as SVG with a
  rasterised PNG beside them — the same arrangement as the lineage diagram, and
  for the same reasons: SVG diffs as text, PNG is what a README renders. A test
  regenerates both and fails if they differ, so the figures cannot go stale.

Everything is tabulated against the baselines committed in ML-03, per predictor
and per city. The per-city view is not decoration: pooled numbers hide a model
that wins on one city and loses on five, and the cities in this set differ by a
factor of four in base rate.

Usage::

    python machine_learning/evaluate.py           # tables to the terminal
    python machine_learning/evaluate.py --write   # and figures + metrics.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve
from sqlalchemy import Engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from machine_learning.baselines import (  # noqa: E402
    BaseRateReference,
    ClimatologyBaseline,
    PersistenceBaseline,
    metrics_path,
)
from machine_learning.evaluation import (  # noqa: E402
    evaluation_frame,
    score,
    split_frame,
)
from machine_learning.labels import LABEL, positives  # noqa: E402
from machine_learning.train import train_model  # noqa: E402

__all__ = [
    "CALIBRATION_BINS",
    "FIGURE_DIR",
    "PALETTE",
    "best_threshold",
    "build_evaluation",
    "calibration_points",
    "classification_at",
    "per_city_table",
    "precision_recall_points",
    "render_figures",
    "summary_table",
]

log = logging.getLogger(__name__)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
FIGURE_DIR: Final[Path] = REPO_ROOT / "docs" / "images"

#: Quantile bins for the reliability curve. Ten over 8 506 test rows is ~850 a
#: bin, which is enough for the observed rate in each to mean something; twenty
#: would draw a jagged line and invite reading noise as miscalibration.
CALIBRATION_BINS: Final[int] = 10

#: Borrowed from ``render_lineage.py`` so the project's figures look like one
#: project rather than five.
PALETTE: Final[Mapping[str, str]] = {
    "base_rate": "#6b7280",
    "climatology": "#8c6d3f",
    "persistence": "#2f6f4e",
    "model_weighted": "#7a5ea8",
    "model_unweighted": "#1f4e79",
}

#: Printed wherever the table is, so the omission is a statement rather than a
#: gap somebody might think to fill in.
ACCURACY_NOTE: Final[str] = (
    "Accuracy is excluded: at this base rate, always answering \"no anomaly\" "
    "scores {:.1%} and predicts nothing."
)


def best_threshold(labels: pd.Series, predictions) -> tuple[float, float]:
    """The threshold maximising F1, and the F1 it reaches.

    Chosen on **validation**, applied to test. Never 0.5: a threshold is a
    decision about the cost of a false alarm against a missed week, and 0.5 is
    only that decision by coincidence. On a model whose mean prediction is 0.07
    it is the decision to never raise an alarm at all.

    Ties break towards the **lower** threshold — the more sensitive of two
    equally good rules — and the sweep is over the thresholds the data itself
    produces, so no grid resolution is being chosen invisibly.
    """
    truth = positives(labels).to_numpy()
    values = np.asarray(predictions, dtype=float)
    precision, recall, thresholds = precision_recall_curve(truth, values)
    # precision_recall_curve returns one more point than thresholds: the final
    # point is recall 0, precision 1, which no threshold produces.
    precision, recall = precision[:-1], recall[:-1]
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def classification_at(labels: pd.Series, predictions, threshold: float) -> dict:
    """Precision, recall, F1 and the counts behind them. No accuracy.

    The counts are carried because F1 alone hides which way a predictor is
    wrong, and a weather warning that is wrong by crying wolf is a different
    product from one that is wrong by staying quiet.
    """
    truth = positives(labels).to_numpy()
    flagged = np.asarray(predictions, dtype=float) >= threshold

    true_positive = int((flagged & truth).sum())
    false_positive = int((flagged & ~truth).sum())
    false_negative = int((~flagged & truth).sum())
    true_negative = int((~flagged & ~truth).sum())

    precision = (
        true_positive / (true_positive + false_positive) if flagged.any() else 0.0
    )
    recall = true_positive / (true_positive + false_negative) if truth.any() else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "flagged": int(flagged.sum()),
    }


def precision_recall_points(labels: pd.Series, predictions) -> pd.DataFrame:
    """The curve itself, as points, so it can be plotted or asserted."""
    truth = positives(labels).to_numpy()
    precision, recall, _ = precision_recall_curve(
        truth, np.asarray(predictions, dtype=float)
    )
    return pd.DataFrame({"recall": recall, "precision": precision})


def calibration_points(labels: pd.Series, predictions, bins: int = CALIBRATION_BINS):
    """Observed rate against predicted rate, in equal-count bins.

    Quantile bins rather than equal-width. With predictions piled between 0.02
    and 0.27, equal-width bins would put nearly every row in the first one and
    draw a reliability curve out of two points and eight empty boxes.
    """
    truth = positives(labels).to_numpy()
    values = np.asarray(predictions, dtype=float)
    observed, predicted = calibration_curve(
        truth, values, n_bins=bins, strategy="quantile"
    )
    return pd.DataFrame({"predicted": predicted, "observed": observed})


def _predictors(parts: Mapping[str, pd.DataFrame], fits) -> dict[str, Any]:
    """Every predictor in the comparison, baselines first, all fitted on train."""
    train, validation = parts["train"], parts["validation"]
    predictors: dict[str, Any] = {
        "base_rate": BaseRateReference().fit(train).predict,
        "persistence": PersistenceBaseline().fit(train).predict,
        "climatology": ClimatologyBaseline.tuned(train, validation).predict,
    }
    for variant, fit in fits.items():
        predictors[f"model_{variant}"] = fit.predict
    return predictors


def summary_table(
    parts: Mapping[str, pd.DataFrame], predictors: Mapping[str, Any]
) -> pd.DataFrame:
    """One row per predictor: the test-split table the ticket asks for."""
    rows = []
    for name, predict in predictors.items():
        validation_predictions = predict(parts["validation"])
        threshold, _ = best_threshold(
            parts["validation"][LABEL], validation_predictions
        )

        test_predictions = predict(parts["test"])
        result = score(parts["test"][LABEL], test_predictions)
        classified = classification_at(
            parts["test"][LABEL], test_predictions, threshold
        )
        rows.append(
            {
                "predictor": name,
                "pr_auc": result.pr_auc,
                "lift": result.lift,
                "brier": result.brier,
                "f1": classified["f1"],
                "precision": classified["precision"],
                "recall": classified["recall"],
                "threshold": threshold,
                "flagged": classified["flagged"],
                "mean_predicted": float(np.mean(test_predictions)),
                "base_rate": result.base_rate,
            }
        )
    return pd.DataFrame(rows)


def per_city_table(
    parts: Mapping[str, pd.DataFrame], predictors: Mapping[str, Any]
) -> pd.DataFrame:
    """PR-AUC per city, because a pooled win can be one city carrying five.

    Each city is scored against **its own** base rate. Singapore's positive
    rate is four times Delhi's, so a shared reference line would rank the
    cities by their climate rather than the predictor by its skill.
    """
    test = parts["test"]
    rows = []
    for city_id, group in test.groupby("city_id", sort=True):
        if positives(group[LABEL]).nunique() < 2:
            # One class only: average precision is undefined, and a city with
            # no positive week in the test period cannot rank anything.
            rows.append({"city_id": city_id, "rows": len(group), "base_rate": float(
                positives(group[LABEL]).mean()
            )})
            continue
        row: dict[str, Any] = {"city_id": city_id, "rows": len(group)}
        for name, predict in predictors.items():
            result = score(group[LABEL], predict(group))
            row["base_rate"] = result.base_rate
            row[f"{name}_pr_auc"] = result.pr_auc
            row[f"{name}_lift"] = result.lift
        rows.append(row)
    return pd.DataFrame(rows)


def build_evaluation(
    engine: Engine | None = None,
    *,
    frame: pd.DataFrame | None = None,
    roster: Sequence[str] | None = None,
    ingested: Sequence[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score everything on test and return the report plus its plot data.

    Args:
        engine: Warehouse connection, used only when ``frame`` is not given.
        frame: A scored population, for tests and for reuse.
        roster: Every city the registry knows about.
        ingested: Every city with observations in the warehouse. Given both,
            the report separates three states that a per-city table would
            otherwise flatten into one gap: scored, ingested but not scorable
            on the test split, and never ingested at all. The ticket names
            Moscow as the contrast to Singapore, and Moscow is in the third
            state — a table that simply lacks the row says nothing about why.
    """
    population = frame if frame is not None else evaluation_frame(engine)
    parts = split_frame(population)
    for name, part in parts.items():
        if part.empty:
            raise ValueError(f"the {name} split is empty; nothing to evaluate.")

    _, fits = train_model(frame=population)
    predictors = _predictors(parts, fits)

    summary = summary_table(parts, predictors)
    cities = per_city_table(parts, predictors)
    # Three states, not two. "Not in the test split" and "not in the warehouse
    # at all" are different facts about a city, and the ticket names Moscow —
    # which has not backfilled, so its absence from the per-city table says
    # nothing about Moscow and everything about the ingestion quota.
    present = sorted(parts["test"]["city_id"].unique())
    scorable = sorted(population["city_id"].unique())
    landed = sorted(ingested) if ingested is not None else scorable
    outside = sorted(set(landed) - set(present))
    absent = sorted(set(roster) - set(landed)) if roster else []

    base = float(summary.loc[summary["predictor"] == "base_rate", "base_rate"].iloc[0])
    report = {
        "scored_on": "test",
        "threshold_chosen_on": "validation",
        "threshold_metric": "f1",
        "calibration_bins": CALIBRATION_BINS,
        "accuracy_reported": False,
        "accuracy_note": ACCURACY_NOTE.format(1 - base),
        "cities_scored": present,
        "cities_ingested_but_not_scored": outside,
        "cities_not_ingested": absent,
        "summary": summary.to_dict("records"),
        "per_city": cities.to_dict("records"),
        "verdict": _verdict(summary),
    }
    # Predicted once and carried, rather than recomputed per curve: the model
    # is refitted inside this call and asking it twice for the same answer is
    # nine seconds and an invitation to have two of them.
    predictions = {
        name: predict(parts["test"]) for name, predict in predictors.items()
    }
    curves = {
        name: {
            "precision_recall": precision_recall_points(
                parts["test"][LABEL], values
            ),
            "calibration": calibration_points(parts["test"][LABEL], values),
        }
        for name, values in predictions.items()
    }
    plots = {
        "curves": curves,
        "predictions": predictions,
        "base_rate": base,
        "parts": parts,
    }
    return report, plots


def _verdict(summary: pd.DataFrame) -> dict[str, Any]:
    """Did each model variant beat both baselines, on each metric, yes or no.

    Written out per metric rather than summarised, because the honest answer
    here is not the same on both: the specified model beats every baseline on
    ranking and loses to *doing nothing* on calibration.
    """
    indexed = summary.set_index("predictor")
    baselines = ["persistence", "climatology", "base_rate"]
    out: dict[str, Any] = {}
    for name in indexed.index:
        if not name.startswith("model_"):
            continue
        beaten = {
            "pr_auc": {
                other: bool(indexed.loc[name, "pr_auc"] > indexed.loc[other, "pr_auc"])
                for other in baselines
            },
            "f1": {
                other: bool(indexed.loc[name, "f1"] > indexed.loc[other, "f1"])
                for other in baselines
            },
            # Lower is better, so the comparison flips.
            "brier": {
                other: bool(indexed.loc[name, "brier"] < indexed.loc[other, "brier"])
                for other in baselines
            },
        }
        out[name] = {
            "beats_every_baseline": {
                metric: all(result.values()) for metric, result in beaten.items()
            },
            "detail": beaten,
        }
    return out


def _thin(points: pd.DataFrame, limit: int = 800) -> pd.DataFrame:
    """Fewer vertices to draw, without moving the line.

    The precision–recall curve has one point per distinct prediction — 8 506 of
    them for a model — and an SVG carrying all of them is a hundred kilobytes
    of coordinates for a line that is smooth at any size a reader will view it.
    The *metric* is computed from every point; only the drawing is thinned, and
    the first and last are always kept so the endpoints are exact.
    """
    if len(points) <= limit:
        return points
    step = int(np.ceil(len(points) / limit))
    kept = points.iloc[::step]
    if not kept.index.equals(points.index[-1:]):
        kept = pd.concat([kept, points.iloc[[-1]]])
    return kept


def _style():
    import matplotlib

    matplotlib.use("Agg")
    # Fixed so two runs produce byte-identical files: matplotlib salts the
    # element ids it generates, and without this the SVG differs on every call
    # while drawing exactly the same picture.
    matplotlib.rcParams["svg.hashsalt"] = "horizon"
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#4b5563",
            "axes.labelcolor": "#111827",
            "axes.grid": True,
            "grid.color": "#e5e7eb",
            "grid.linewidth": 0.8,
            "text.color": "#111827",
            "xtick.color": "#4b5563",
            "ytick.color": "#4b5563",
            "font.size": 9,
            "legend.frameon": False,
        }
    )
    return plt


def _save(figure, stem: str, directory: Path) -> list[Path]:
    """SVG plus a rasterised PNG, the way the lineage diagram is kept.

    Both are written with their timestamp metadata suppressed, so regenerating
    an unchanged figure produces an unchanged file and a diff means the picture
    actually moved.
    """
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix, metadata, dpi in (
        (".svg", {"Date": None}, None),
        (".png", {"Software": None}, 160),
    ):
        path = directory / f"{stem}{suffix}"
        figure.savefig(path, metadata=metadata, dpi=dpi, bbox_inches="tight")
        written.append(path)
    return written


def render_figures(
    plots: Mapping[str, Any], directory: Path | None = None
) -> list[Path]:
    """Draw the precision–recall and calibration figures."""
    plt = _style()
    directory = Path(directory) if directory is not None else FIGURE_DIR
    curves, base = plots["curves"], plots["base_rate"]
    written: list[Path] = []

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for name, data in curves.items():
        if name == "base_rate":
            continue
        points = data["precision_recall"]
        # `steps-post`, not a straight line between the points.
        #
        # A predictor with two distinct scores has three points on this curve,
        # and joining them with straight segments draws operating points it
        # cannot reach: persistence cannot be run at recall 0.6 at all. The
        # diagonal that produces sits *above* the model for half the range and
        # integrates to roughly twice the average precision the same predictor
        # actually scores, so the picture and the table would disagree. A step
        # function is what average precision sums, and it is what a reader can
        # actually buy.
        drawn = _thin(points)
        axis.plot(
            drawn["recall"],
            drawn["precision"],
            color=PALETTE.get(name, "#111827"),
            linewidth=1.8,
            drawstyle="steps-post",
            label=name.replace("_", " "),
        )
        # Where a predictor has only a handful of operating points, mark them,
        # so its coarseness is visible rather than inferred from the shape.
        if len(points) <= 12:
            axis.plot(
                points["recall"],
                points["precision"],
                marker="o",
                markersize=4,
                linestyle="none",
                color=PALETTE.get(name, "#111827"),
            )
    axis.axhline(
        base, color=PALETTE["base_rate"], linestyle="--", linewidth=1.2,
        label=f"no skill ({base:.3f})",
    )
    axis.set_xlabel("recall")
    axis.set_ylabel("precision")
    axis.set_title("Precision–recall on the test split (2022–2026)", loc="left")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend(loc="upper right")
    written += _save(figure, "precision_recall", directory)
    plt.close(figure)

    figure, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4.4))
    limit = 0.0
    for name in ("model_weighted", "model_unweighted", "persistence"):
        if name not in curves:
            continue
        points = curves[name]["calibration"]
        limit = max(limit, float(points[["predicted", "observed"]].to_numpy().max()))
        left.plot(
            points["predicted"], points["observed"], marker="o", markersize=4,
            color=PALETTE.get(name, "#111827"), linewidth=1.6,
            label=name.replace("_", " "),
        )
    edge = min(1.0, limit * 1.08)
    left.plot([0, edge], [0, edge], color="#9ca3af", linestyle="--", linewidth=1.2,
              label="perfectly calibrated")
    left.axhline(base, color=PALETTE["base_rate"], linestyle=":", linewidth=1.0)
    left.set_xlabel("mean predicted probability")
    left.set_ylabel("observed rate")
    left.set_title("Reliability, test split", loc="left")
    left.legend(loc="upper left")

    test = plots["parts"]["test"]
    for name in ("model_weighted", "model_unweighted"):
        if name not in curves:
            continue
        predictions = plots["predictions"][name]
        right.hist(
            predictions, bins=40, histtype="step", linewidth=1.6,
            color=PALETTE.get(name, "#111827"), label=name.replace("_", " "),
        )
    right.axvline(base, color=PALETTE["base_rate"], linestyle="--", linewidth=1.2,
                  label=f"true rate ({base:.3f})")
    right.set_xlabel("predicted probability")
    right.set_ylabel(f"test rows (of {len(test):,})")
    right.set_yscale("log")
    right.set_title("What the dashboard would show", loc="left")
    right.legend(loc="upper right")
    written += _save(figure, "calibration", directory)
    plt.close(figure)
    return written


def city_roster(engine: Engine | None = None) -> tuple[list[str], list[str]]:
    """The registry, and which of it has actually been ingested.

    Two lists rather than one, because a city can be missing from the per-city
    table for two unrelated reasons and the difference is the whole point of
    reporting the absence. ``dim_cities`` is built from ``config/cities.yml``
    and says what the set *is*; the fact table says what has been observed of
    it, which is exactly the gap the reconciliation report exists to surface.
    """
    from ingestion.loader import engine_from_settings

    owned = engine is None
    engine = engine if engine is not None else engine_from_settings()
    try:
        with engine.connect() as connection:
            roster = [
                row[0]
                for row in connection.execute(
                    text("select city_id from gold_marts.dim_cities order by city_id")
                )
            ]
            ingested = [
                row[0]
                for row in connection.execute(
                    text(
                        "select distinct city_id from "
                        "gold_marts.fact_weather_observations order by city_id"
                    )
                )
            ]
    finally:
        if owned:
            engine.dispose()
    return roster, ingested


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the model against the baselines on the test split."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Save the figures and merge the report into metrics.json.",
    )
    parser.add_argument(
        "--figures", help="Write figures somewhere other than docs/images."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    population = evaluation_frame()
    roster, ingested = city_roster()
    report, plots = build_evaluation(
        frame=population, roster=roster, ingested=ingested
    )

    summary = pd.DataFrame(report["summary"])
    shown = summary[
        ["predictor", "pr_auc", "lift", "brier", "f1", "precision", "recall",
         "threshold", "flagged"]
    ].copy()
    for column in ("pr_auc", "brier", "f1", "precision", "recall", "threshold"):
        shown[column] = shown[column].map("{:.4f}".format)
    shown["lift"] = summary["lift"].map("{:.2f}x".format)
    print("test split 2022-01-01 .. 2026-09-01, threshold chosen on validation\n")
    print(shown.to_string(index=False))
    print(f"\n{report['accuracy_note']}")

    print("\nper city (PR-AUC against that city's own base rate)")
    cities = pd.DataFrame(report["per_city"])
    columns = ["city_id", "rows", "base_rate", "persistence_pr_auc",
               "model_unweighted_pr_auc", "model_unweighted_lift"]
    view = cities.loc[:, [c for c in columns if c in cities.columns]].copy()
    view["base_rate"] = view["base_rate"].map("{:.1%}".format)
    for column in ("persistence_pr_auc", "model_unweighted_pr_auc"):
        if column in view:
            view[column] = view[column].map("{:.4f}".format)
    if "model_unweighted_lift" in view:
        view["model_unweighted_lift"] = view["model_unweighted_lift"].map(
            "{:.2f}x".format
        )
    print(view.to_string(index=False))
    if report["cities_ingested_but_not_scored"]:
        print(
            "\ningested but not scorable on the test split: "
            f"{', '.join(report['cities_ingested_but_not_scored'])}"
        )
    if report["cities_not_ingested"]:
        print(
            "not ingested, so not scorable at all: "
            f"{', '.join(report['cities_not_ingested'])}"
        )

    print("\nverdict")
    for name, entry in report["verdict"].items():
        beats = entry["beats_every_baseline"]
        for metric, passed in beats.items():
            mark = "beats every baseline" if passed else "DOES NOT beat every baseline"
            print(f"  {name:<18} {metric:<7} {mark}")

    if args.write or args.figures:
        directory = Path(args.figures) if args.figures else FIGURE_DIR
        written = render_figures(plots, directory)
        for path in written:
            print(f"\nwrote {path}")
        destination = metrics_path()
        if destination.exists():
            payload = json.loads(destination.read_text())
            payload["evaluation"] = dict(report)
            payload["evaluation"]["recorded_at"] = dt.datetime.now(
                dt.timezone.utc
            ).isoformat(timespec="seconds")
            destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            print(f"wrote {destination}")
    else:
        print("\n(not written — pass --write for figures and metrics.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
