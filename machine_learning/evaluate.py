"""Scores the model against the baselines, and refuses to report accuracy.

**Accuracy is excluded deliberately.** At a 13.58% test base rate, a model that
answers "no anomaly" every single time is 86.4% accurate. Reporting that number
beside a real result is not a rounding of the truth, it is an inversion of it,
and any reader who knows the field reads it as a signal that nobody in the
project understood the base rate. The scikit-learn function for it does not
appear anywhere in this repository, and a test enforces that. The forbidden
names are listed in ``tests/test_evaluation.py`` rather than here, so the scan
does not find its own explanation.

What is reported instead:

* **PR-AUC**, with the base rate printed beside it, because average precision
  for a random ranker *is* the base rate.
* **F1, precision and recall** at a threshold chosen on **validation**, never
  on test, and never at 0.5. A model whose mean prediction is 0.07 classifies
  nothing at 0.5 and would score F1 = 0.00 while ranking better than anything
  else in the table.
* **Brier**, which is the only one of these that notices a probability is
  wrong rather than merely badly ordered.
* **A precision-recall curve and a calibration curve**, saved as SVG with a
  rasterised PNG beside them, the same arrangement as the lineage diagram, and
  for the same reasons: SVG diffs as text, PNG is what a README renders. A test
  regenerates both and fails if they differ, so the figures cannot go stale.

Everything is tabulated against the baselines committed in ML-03, per predictor
and per city. The per-city view is not decoration: pooled numbers hide a model
that wins on one city and loses on five, and the cities in this set differ by a
factor of four in base rate.

**Leave-one-city-out is a separate mode, and it is opt-in.** There is no city
identifier among the twenty-seven features, so the model can in principle score
a city it has never seen. ``--leave-one-city-out`` refits once per scored city
with that city removed from the training and validation splits altogether and
reports what it manages on the city's own test rows, against that city's own
persistence baseline rather than against its base rate. It costs one full
training run per city, which is why it is a flag rather than part of every run,
and it is the experiment that decides whether this project ships one model or
five.

Usage::

    python machine_learning/evaluate.py           # tables to the terminal
    python machine_learning/evaluate.py --write   # and figures + metrics.json
    python machine_learning/evaluate.py --leave-one-city-out --write
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
from sqlalchemy import Engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from machine_learning.baselines import (  # noqa: E402
    BaseRateReference,
    ClimatologyBaseline,
    PersistenceBaseline,
    metrics_path,
)
from machine_learning.evaluation import (  # noqa: E402
    MIN_HELD_OUT_POSITIVES,
    MIN_HELD_OUT_ROWS,
    assert_city_is_held_out,
    evaluation_frame,
    hold_out_city,
    scorable_cities,
    score,
    split_frame,
)
from machine_learning.features import city_roster  # noqa: E402
from machine_learning.labels import LABEL, positives  # noqa: E402
from machine_learning.train import Fit, train_model  # noqa: E402

__all__ = [
    "CALIBRATION_BINS",
    "FIGURE_DIR",
    "MIN_TRAINING_CITIES",
    "PALETTE",
    "best_threshold",
    "build_evaluation",
    "build_leave_one_city_out",
    "calibration_points",
    "classification_at",
    "hold_out_reasons",
    "leave_one_city_out_fold",
    "leave_one_city_out_table",
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

    Ties break towards the **lower** threshold, the more sensitive of two
    equally good rules, and the sweep is over the thresholds the data itself
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
            state, and a table that simply lacks the row says nothing about why.
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
    # at all" are different facts about a city, and the ticket names Moscow,
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
        # Carried so the leave-one-city-out block can put each city's held-out
        # score beside what *this* fit scored on the same rows. Refitting for
        # the in-sample column would be a second model differing in its last
        # digits from the one the rest of metrics.json describes.
        "fits": fits,
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


# --------------------------------------------------------------------------
# Leave-one-city-out (ML-08)
# --------------------------------------------------------------------------
#
# There is no city identifier in the feature set. The twenty-seven inputs are
# that city's own weather plus ``latitude`` and ``elevation_m``, so the model is
# architecturally capable of scoring a city it has never seen. Whether it can is
# a separate question, and it is the one that decides what this project is: if a
# held-out city scores near its own persistence baseline then the model is five
# per-city curve-fits sharing a file, and the honest product is five models.
#
# The yardstick is **persistence, not the base rate**, and the difference
# matters more here than anywhere else in the evaluation. Held-out cities differ
# fourfold in base rate, so a raw PR-AUC is not comparable across them and even
# lift over the base rate flatters a tropical city, where a small sigma turns a
# modest trend into a near-constant exceedance and "an anomaly followed an
# anomaly" is most of the available signal. Persistence already collects that
# signal for free. A model that only matches it has learned nothing worth
# shipping, whatever its lift over chance says.


#: Every held-out fold needs at least this many other cities to train on.
#:
#: Two, because "it generalises" is a claim about a population and one other
#: city cannot support it: a model fitted on Singapore alone and scored on Lagos
#: measures the similarity of two tropical cities, not transfer. It is a floor
#: on the experiment being meaningful, not on it running.
MIN_TRAINING_CITIES: Final[int] = 2


def hold_out_reasons(
    population: pd.DataFrame,
    parts: Mapping[str, pd.DataFrame],
    *,
    roster: Sequence[str] | None = None,
    ingested: Sequence[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Which cities can be held out, and a reason for every city that cannot.

    The reasons are half the deliverable. A leave-one-city-out table that holds
    out five of fifteen cities and simply omits the other ten reads as though
    the ten failed, when in fact Moscow has three days of record and Reykjavik
    has fourteen rows in the training split. "Not eligible" and "eligible and
    unimpressive" are opposite findings and a missing row cannot tell them
    apart, which is the same argument ``cities_not_ingested`` already makes one
    table up.
    """
    eligible, thin = scorable_cities(parts)
    reasons: dict[str, str] = dict(thin)

    in_population = set(population["city_id"].unique())
    in_test = set(parts["test"]["city_id"].unique())
    landed = set(ingested) if ingested is not None else in_population
    known = set(roster) if roster is not None else landed | in_population

    last_seen = population.groupby("city_id")["date_key"].max()
    for city in sorted(known - set(eligible) - set(reasons)):
        if city not in landed:
            reasons[city] = "not ingested"
        elif city not in in_population:
            reasons[city] = "no labelled rows past the feature warm-up"
        elif city not in in_test:
            reasons[city] = (
                "no rows in the test split; its record ends "
                f"{last_seen[city].date()}"
            )
    return eligible, reasons


def _persistence_reference(
    train_fold: pd.DataFrame, held_test: pd.DataFrame, fitted_on: str
) -> dict[str, Any]:
    """Persistence scored on the held-out city's test rows.

    Fitted where the caller says and nowhere else, and fitted twice per fold on
    purpose. The rule -- "an anomaly next week if there was one this week" --
    emits one probability per cell, so what a fit decides is the *level* of
    those three numbers and not their order, and PR-AUC depends only on the
    order. Calibrate it on the held-out city's own history or on the other
    cities' and the two normally rank that city's test rows identically, which
    is why the ticket's "that city's own persistence baseline" is unambiguous
    about the metric it asks for. Normally, not necessarily: a city whose
    training period ran anti-persistent would order its cells the other way,
    so the record keeps both fits and :func:`leave_one_city_out_fold` records
    whether they agreed.

    The yardstick for every comparison is the **own** fit, because it is the
    harder one and the one a reader would actually reach for: it is what you
    could have for that city without a model at all.
    """
    if train_fold.empty:
        return {
            "fitted_on": fitted_on,
            "available": False,
            "why": "the fold has no training rows to calibrate the rule on",
        }
    baseline = PersistenceBaseline().fit(train_fold)
    result = score(held_test[LABEL], baseline.predict(held_test))
    return {
        "fitted_on": fitted_on,
        "available": True,
        "train_rows": int(len(train_fold)),
        "pr_auc": result.pr_auc,
        "brier": result.brier,
        "lift_over_base_rate": result.lift,
    }


def leave_one_city_out_fold(
    population: pd.DataFrame,
    city_id: str,
    *,
    in_sample: Mapping[str, Any],
) -> dict[str, Any]:
    """Train on every other city, score this one, and prove it was never seen.

    Args:
        population: The whole scored population, every city.
        city_id: The city to hold out of training entirely -- out of the
            training split, out of the validation split the grid search and
            early stopping read, and out of the baseline's calibration.
        in_sample: What the all-cities model scored on this same city's test
            rows, by variant, so the two numbers sit in one record and the
            comparison cannot be assembled wrongly by a reader joining two
            tables.

    Returns:
        One record per held-out city, carrying both model variants, both
        persistence references, and the base rate they are all read against.
    """
    others, held = hold_out_city(population, city_id)
    fold = split_frame(others)
    held_parts = split_frame(held)
    held_test = held_parts["test"]

    # Before anything is fitted. The assertion is cheap and the failure it
    # guards against is invisible in every other check in this module: a fold
    # that kept the city produces a well-ordered chronological split, a
    # plausible PR-AUC, and an answer to a different question.
    assert_city_is_held_out(
        city_id,
        {"train": fold["train"], "validation": fold["validation"]},
        held_out=held_test,
    )

    training_cities = sorted(fold["train"]["city_id"].unique())
    if len(training_cities) < MIN_TRAINING_CITIES:
        raise ValueError(
            f"holding out {city_id} leaves {training_cities} to train on, "
            f"fewer than {MIN_TRAINING_CITIES}. A model fitted on one other "
            "city and scored on this one measures how alike two cities are, "
            "not whether the model transfers."
        )

    fold_block, fits = train_model(frame=others)
    truth = positives(held_test[LABEL])

    record: dict[str, Any] = {
        "city_id": city_id,
        "rows": int(len(held_test)),
        "positives": int(truth.sum()),
        "base_rate": float(truth.mean()),
        "train_rows": int(len(fold["train"])),
        "validation_rows": int(len(fold["validation"])),
        "train_cities": training_cities,
        "fold_recommended_variant": fold_block["recommended_variant"],
        "persistence": _persistence_reference(
            fold["train"], held_test, "the other cities' training split"
        ),
        "persistence_own": _persistence_reference(
            held_parts["train"], held_test, "this city's own training split"
        ),
        "variants": {},
    }

    own, fold_mate = record["persistence_own"], record["persistence"]
    if own.get("available") and fold_mate.get("available"):
        record["persistence_fits_agree_on_ranking"] = bool(
            np.isclose(own["pr_auc"], fold_mate["pr_auc"])
        )

    for variant, fit in fits.items():
        held_out = score(held_test[LABEL], fit.predict(held_test))
        inside = in_sample.get(variant, {})
        entry: dict[str, Any] = {
            "held_out": held_out.as_dict(),
            "in_sample": dict(inside),
            # ``held_out["lift"]`` is already the lift over this city's base
            # rate. It is kept because the ticket asks for the base rate beside
            # every figure, and read second: the trap it warns about is exactly
            # a reader ranking cities by this column.
            "beats_base_rate": bool(held_out.lift > 1.0),
        }
        if own.get("available"):
            entry["lift_over_persistence"] = held_out.pr_auc / own["pr_auc"]
            # Brier is a loss, so the ratio is inverted to keep every number in
            # this record reading the same way: above one is better.
            entry["brier_ratio_to_persistence"] = own["brier"] / held_out.brier
            entry["beats_own_persistence"] = {
                "pr_auc": bool(held_out.pr_auc > own["pr_auc"]),
                "brier": bool(held_out.brier < own["brier"]),
            }
        if inside.get("pr_auc"):
            entry["retained_of_in_sample"] = held_out.pr_auc / inside["pr_auc"]
        record["variants"][variant] = entry

    log.info(
        "held out %-10s %d rows, base rate %.3f, PR-AUC %.4f in sample, "
        "%.4f held out",
        city_id,
        record["rows"],
        record["base_rate"],
        in_sample.get("unweighted", {}).get("pr_auc", float("nan")),
        record["variants"]["unweighted"]["held_out"]["pr_auc"],
    )
    return record


def _statement(verdict: Mapping[str, Any]) -> str:
    """The one sentence the README is required to carry, generated not typed.

    Generated, because a verdict written by hand is a verdict that survives the
    numbers that produced it. ``tests/test_evaluation.py`` asserts the README
    contains this string, so a re-run that flips the finding fails the suite
    until the prose is changed to match.
    """
    beaten = verdict["cities_beating_own_persistence"]
    total = verdict["cities"]
    if not total:
        return (
            "Leave-one-city-out could not be run: no city has enough test rows "
            "to hold out."
        )
    lift = verdict["median_lift_over_persistence"]
    retained = verdict["median_retained_of_in_sample"]
    if verdict["transfers"]:
        return (
            f"Held out of training entirely, each of the {total} scored cities "
            f"is still ranked better by the model than by its own persistence "
            f"baseline ({beaten} of {total}, median PR-AUC {lift:.2f}x "
            f"persistence and {retained:.0%} of the same city's in-sample "
            "score), so the model transfers to a city it has never seen."
        )
    return (
        f"Held out of training entirely, the model beats its own persistence "
        f"baseline in only {beaten} of the {total} scored cities (median "
        f"PR-AUC {lift:.2f}x persistence and {retained:.0%} of the same city's "
        "in-sample score), so it does not transfer, and on this evidence the "
        "honest product is one model per city rather than one model."
    )


def build_leave_one_city_out(
    population: pd.DataFrame,
    *,
    fits: Mapping[str, Fit] | None = None,
    roster: Sequence[str] | None = None,
    ingested: Sequence[str] | None = None,
    cities: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the whole experiment: one fold per eligible city, plus the verdict.

    Args:
        population: The scored population, every city, as
            :func:`~machine_learning.evaluation.evaluation_frame` builds it.
        fits: The all-cities models, if the caller has already fitted them, so
            the in-sample column beside each held-out score comes from the same
            fit the rest of ``metrics.json`` describes rather than from a
            second one that would differ in its last digits.
        roster: Every city the registry knows about, for the exclusion reasons.
        ingested: Every city with observations, likewise.
        cities: Hold out exactly these, overriding the eligibility rule. For
            tests; a real run should not be choosing its own sample.
    """
    population = population.sort_values(
        ["city_id", "date_key"], kind="stable"
    ).reset_index(drop=True)
    parts = split_frame(population)
    eligible, reasons = hold_out_reasons(
        population, parts, roster=roster, ingested=ingested
    )
    held_out_cities = list(cities) if cities is not None else eligible

    if fits is None:
        _, fits = train_model(frame=population)

    # The in-sample column: the all-cities model scored on each city's own test
    # rows. Computed here rather than read back from ``per_city``, because the
    # comparison this block exists to make is only meaningful if both halves
    # come from the same fit and the same rows.
    test = parts["test"]
    in_sample: dict[str, dict[str, Any]] = {}
    for city in held_out_cities:
        rows = test.loc[test["city_id"] == city]
        in_sample[city] = {
            variant: score(rows[LABEL], fit.predict(rows)).as_dict()
            for variant, fit in fits.items()
        }

    records = [
        leave_one_city_out_fold(population, city, in_sample=in_sample[city])
        for city in held_out_cities
    ]

    variant = "unweighted"
    if records:
        # The project's recommended variant, chosen on validation by train.py.
        # Read from a fold rather than hard-coded, so the two cannot drift.
        variant = records[0]["fold_recommended_variant"]
    beaten = {
        record["city_id"]: bool(
            record["variants"][variant]
            .get("beats_own_persistence", {})
            .get("pr_auc", False)
        )
        for record in records
    }
    over_base = {
        record["city_id"]: bool(record["variants"][variant]["beats_base_rate"])
        for record in records
    }
    lifts = [
        record["variants"][variant]["lift_over_persistence"]
        for record in records
        if "lift_over_persistence" in record["variants"][variant]
    ]
    retained = [
        record["variants"][variant]["retained_of_in_sample"]
        for record in records
        if "retained_of_in_sample" in record["variants"][variant]
    ]
    verdict: dict[str, Any] = {
        "variant": variant,
        "cities": len(records),
        "beats_own_persistence": beaten,
        "beats_base_rate": over_base,
        "cities_beating_own_persistence": int(sum(beaten.values())),
        "median_lift_over_persistence": (
            float(np.median(lifts)) if lifts else float("nan")
        ),
        "median_retained_of_in_sample": (
            float(np.median(retained)) if retained else float("nan")
        ),
        # The bar the ticket sets, and it is the strict one: every city, not a
        # majority. A model that transfers to four cities and fails on the
        # fifth has not answered "can it score a city it has never seen", it
        # has raised the question of what is different about the fifth.
        "transfers": bool(records) and all(beaten.values()),
    }
    verdict["statement"] = _statement(verdict)

    return {
        "scored_on": "test",
        "trained_on": "every other city's train and validation splits",
        "yardstick": "that city's own persistence baseline, not its base rate",
        "min_held_out_rows": MIN_HELD_OUT_ROWS,
        "min_held_out_positives": MIN_HELD_OUT_POSITIVES,
        "min_training_cities": MIN_TRAINING_CITIES,
        "cities_held_out": [record["city_id"] for record in records],
        "cities_not_held_out": dict(sorted(reasons.items())),
        "folds": records,
        "verdict": verdict,
    }


def leave_one_city_out_table(block: Mapping[str, Any]) -> pd.DataFrame:
    """The held-out numbers as a table, with the base rate beside every one.

    The base rate is not decoration in this table, it is the trap the ticket
    names: Singapore's positives are four times Delhi's, so a column of raw
    PR-AUCs sorts the cities by climate and says nothing about the model.
    """
    variant = block["verdict"]["variant"]
    rows = []
    for record in block["folds"]:
        entry = record["variants"][variant]
        own = record["persistence_own"]
        rows.append(
            {
                "city_id": record["city_id"],
                "rows": record["rows"],
                "base_rate": record["base_rate"],
                "in_sample_pr_auc": entry["in_sample"].get("pr_auc", float("nan")),
                "held_out_pr_auc": entry["held_out"]["pr_auc"],
                "held_out_brier": entry["held_out"]["brier"],
                "persistence_pr_auc": own.get("pr_auc", float("nan")),
                "lift_over_persistence": entry.get(
                    "lift_over_persistence", float("nan")
                ),
                "retained_of_in_sample": entry.get(
                    "retained_of_in_sample", float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _thin(points: pd.DataFrame, limit: int = 800) -> pd.DataFrame:
    """Fewer vertices to draw, without moving the line.

    The precision-recall curve has one point per distinct prediction, 8 506 of
    them for a model, and an SVG carrying all of them is a hundred kilobytes
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
    """Draw the precision-recall and calibration figures."""
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
    parser.add_argument(
        "--leave-one-city-out",
        action="store_true",
        help=(
            "Also refit once per scored city, holding that city out of "
            "training entirely, and record the result. Adds one full training "
            "run per city, so it is opt-in rather than part of every run."
        ),
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

    loco = None
    if args.leave_one_city_out:
        print("\nleave-one-city-out: one refit per city, holding it out entirely")
        loco = build_leave_one_city_out(
            population, fits=plots["fits"], roster=roster, ingested=ingested
        )
        table = leave_one_city_out_table(loco)
        if table.empty:
            print("  no city has enough test rows to hold out.")
        else:
            shown = table.copy()
            shown["base_rate"] = shown["base_rate"].map("{:.1%}".format)
            for column in (
                "in_sample_pr_auc",
                "held_out_pr_auc",
                "held_out_brier",
                "persistence_pr_auc",
            ):
                shown[column] = shown[column].map("{:.4f}".format)
            for column in ("lift_over_persistence", "retained_of_in_sample"):
                shown[column] = shown[column].map("{:.2f}x".format)
            print(shown.to_string(index=False))
        for city, why in loco["cities_not_held_out"].items():
            print(f"  not held out: {city:<14} {why}")
        print(f"\n  {loco['verdict']['statement']}")

    if args.write or args.figures:
        directory = Path(args.figures) if args.figures else FIGURE_DIR
        written = render_figures(plots, directory)
        for path in written:
            print(f"\nwrote {path}")
        destination = metrics_path()
        if destination.exists():
            payload = json.loads(destination.read_text())
            stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
            payload["evaluation"] = dict(report)
            payload["evaluation"]["recorded_at"] = stamp
            if loco is not None:
                payload["leave_one_city_out"] = dict(loco)
                payload["leave_one_city_out"]["recorded_at"] = stamp
            destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            print(f"wrote {destination}")
            if loco is None and "leave_one_city_out" in payload:
                print(
                    "  the leave-one-city-out block is from an earlier run; "
                    "pass --leave-one-city-out to refresh it."
                )
    else:
        print("\n(not written; pass --write for figures and metrics.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
