"""SHAP values for the classifier: what it uses, and why it said what it said.

Computed with XGBoost's own TreeSHAP (``pred_contribs=True``) rather than the
``shap`` package. Not to avoid the dependency for its own sake, since the
values are *bit-identical* and a test asserts that against the reference
implementation, but because the package pulls a compiler toolchain into an
application that
deploys to Streamlit Community Cloud, and the numbers it would contribute are
already in the model.

**The values are log-odds, not probabilities.** A contribution of +2.49 does not
mean "adds 249% risk"; it means the model's logit moved 2.49 from a base of
-2.90, which is 5.2% to 40%. Anything that puts these numbers in front of a
reader has to say which space they are in, or a dashboard will report an
explanation that does not add up to the probability printed beside it.

The identity that makes them worth trusting is exact and is asserted rather
than cited: for every row, the contributions plus the bias equal the model's
margin. If that ever stops holding, the explanation is not an explanation of
this model.

Three artefacts:

* a **beeswarm** of the top features across the test split, showing the
  direction each pushes as well as how far;
* two **individual predictions** explained side by side, the most confident
  true positive and the most confident false positive, chosen at the same
  validation-tuned threshold the F1 in the evaluation table uses;
* a **top-ten ranking** recorded in ``metrics.json``.

Usage::

    python machine_learning/explain.py            # rankings to the terminal
    python machine_learning/explain.py --write    # and figures + metrics.json
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
from xgboost import DMatrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from machine_learning.baselines import metrics_path  # noqa: E402
from machine_learning.evaluate import (  # noqa: E402
    FIGURE_DIR,
    PALETTE,
    _save,
    _style,
    best_threshold,
)
from machine_learning.evaluation import (  # noqa: E402
    evaluation_frame,
    split_frame,
)
from machine_learning.labels import LABEL, positives  # noqa: E402
from machine_learning.train import SEED, train_model, training_matrix  # noqa: E402

__all__ = [
    "STATIC_CITY_FEATURES",
    "TOP_N",
    "Contributions",
    "build_explanation",
    "contributions",
    "explain_case",
    "global_importance",
    "pick_cases",
    "render_figures",
]

log = logging.getLogger(__name__)

#: How many features the ranking and the beeswarm carry. Ten is the ticket's
#: number and also about where the tail flattens: the eleventh feature here
#: contributes a sixth of what the second does.
TOP_N: Final[int] = 10

#: Features that are constant within a city. They are not meteorology: a model
#: leaning on them is looking up which city it is, and city base rates in this
#: set run from 5.2% to 23.7%. Named so the ranking can say that out loud
#: rather than leaving a reader to notice that latitude does not vary.
STATIC_CITY_FEATURES: Final[tuple[str, ...]] = ("latitude", "elevation_m")

#: Bins for the both-tails check, on the climatological Z. The label is
#: ``abs(z) > 2.5``, so a model that has understood its target should raise risk
#: at *both* ends and be quietest in the middle.
Z_BINS: Final[tuple[float, ...]] = (-np.inf, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, np.inf)


class Contributions:
    """TreeSHAP output for one frame, with the identity that validates it."""

    def __init__(self, values: np.ndarray, bias: np.ndarray, matrix: pd.DataFrame):
        self.values = values
        self.bias = bias
        self.matrix = matrix
        self.features = tuple(matrix.columns)

    def __len__(self) -> int:
        return len(self.matrix)

    def column(self, feature: str) -> np.ndarray:
        return self.values[:, self.features.index(feature)]

    def margins(self) -> np.ndarray:
        """What the model's logit must equal if these are its contributions."""
        return self.values.sum(axis=1) + self.bias

    def probabilities(self) -> np.ndarray:
        """The margins as probabilities, to compare against ``predict_proba``.

        The identity that makes these contributions an explanation *of this
        model* rather than of some model: pushed through the logistic, they
        have to reproduce the number the classifier actually emitted.
        """
        return 1.0 / (1.0 + np.exp(-self.margins()))


def contributions(fit, frame: pd.DataFrame) -> Contributions:
    """Exact TreeSHAP contributions in log-odds, plus the bias column.

    XGBoost returns one extra column holding the base value, the model's
    output with no features, and it is kept separate rather than folded in,
    because the base value is not an explanation of anything and a beeswarm
    that included it would rank "being a city-day at all" first.

    **``iteration_range`` is not optional here.** The model stops early on
    validation, so ``predict_proba`` scores with the first ``best_iteration+1``
    trees and ``pred_contribs`` defaults to *all* of them. Left alone, the two
    disagree: the first draft of this module explained a Delhi day at margin
    0.56 while the model it was explaining had scored the same day at 0.695,
    which is margin 0.82. The explanation would have been of a model nobody
    runs: internally consistent, since the contributions still summed to the
    fuller model's own margin, and wrong. :func:`Contributions.margins` is
    asserted against ``predict_proba`` for exactly this reason.
    """
    matrix, _ = training_matrix(frame)
    booster = fit.estimator.get_booster()
    trees = (0, fit.best_iteration + 1)
    raw = booster.predict(DMatrix(matrix), pred_contribs=True, iteration_range=trees)
    return Contributions(raw[:, :-1], raw[:, -1], matrix)


def global_importance(shap: Contributions) -> pd.DataFrame:
    """Rank features by mean absolute contribution, with the direction.

    ``mean_abs`` is the ranking: how much this feature moves the answer. It
    says nothing about *which way*, so two more columns come with it:
    ``mean_signed`` is the average push, and ``correlation`` is between the
    feature's value and its contribution. A feature can matter enormously and
    have a mean push of nearly zero, which is what a U-shaped response looks
    like, and reporting only the magnitude would hide exactly that.
    """
    rows = []
    for index, feature in enumerate(shap.features):
        column = shap.values[:, index]
        values = shap.matrix.iloc[:, index].to_numpy(dtype=float)
        # Both sides need to vary. A feature the model never split on has a
        # constant zero contribution, and a feature constant within the frame
        # has no spread of its own, so correlation is undefined either way, and
        # numpy answers with a nan and a warning rather than a refusal.
        varies = values.std() > 0 and column.std() > 0
        rows.append(
            {
                "feature": feature,
                "mean_abs": float(np.abs(column).mean()),
                "mean_signed": float(column.mean()),
                "correlation": (
                    float(np.corrcoef(values, column)[0, 1]) if varies else 0.0
                ),
                "static_per_city": feature in STATIC_CITY_FEATURES,
            }
        )
    ranked = pd.DataFrame(rows).sort_values("mean_abs", ascending=False)
    return ranked.reset_index(drop=True)


def explain_case(shap: Contributions, position: int, top: int = 8) -> pd.DataFrame:
    """The contributions behind one prediction, largest first."""
    column = shap.values[position]
    frame = pd.DataFrame(
        {
            "feature": list(shap.features),
            "shap": column,
            "value": shap.matrix.iloc[position].to_numpy(dtype=float),
        }
    )
    frame = frame.reindex(frame["shap"].abs().sort_values(ascending=False).index)
    return frame.head(top).reset_index(drop=True)


def pick_cases(
    frame: pd.DataFrame, predictions: np.ndarray, threshold: float
) -> dict[str, int]:
    """The most confident true positive and false positive, by position.

    Most confident rather than nearest the threshold, deliberately. A marginal
    case explains why a coin landed on its edge; a confident one explains what
    the model believes, and a confident *mistake* is the one worth reading.
    """
    truth = positives(frame[LABEL]).to_numpy()
    flagged = predictions >= threshold
    picks: dict[str, int] = {}
    for name, mask in (
        ("true_positive", flagged & truth),
        ("false_positive", flagged & ~truth),
    ):
        if not mask.any():
            continue
        picks[name] = int(np.argmax(np.where(mask, predictions, -np.inf)))
    return picks


def tail_response(shap: Contributions, feature: str = "z_temperature_2m_mean"):
    """Mean contribution of ``feature`` by band, for the both-tails check.

    The label is ``abs(z) > 2.5``. Nothing told the model that; it saw a binary
    column. If it has understood its target it will push risk *up* at both ends
    of the Z distribution and *down* in the middle, and this is where that
    shows, or does not.
    """
    values = shap.matrix[feature].to_numpy(dtype=float)
    bands = pd.cut(values, list(Z_BINS))
    table = pd.DataFrame({"band": bands, "shap": shap.column(feature)})
    grouped = table.groupby("band", observed=True)["shap"].agg(["size", "mean"])
    grouped = grouped.rename(columns={"size": "rows", "mean": "mean_shap"})
    return grouped.reset_index()


def build_explanation(
    engine=None, *, frame: pd.DataFrame | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rank the features, explain two predictions, and return both with plot data."""
    population = frame if frame is not None else evaluation_frame(engine)
    parts = split_frame(population)
    for name, part in parts.items():
        if part.empty:
            raise ValueError(f"the {name} split is empty; nothing to explain.")

    block, fits = train_model(frame=population)
    variant = block["recommended_variant"]
    fit = fits[variant]

    test = parts["test"]
    shap = contributions(fit, test)
    ranked = global_importance(shap)

    predictions = fit.predict(test)
    threshold, _ = best_threshold(
        parts["validation"][LABEL], fit.predict(parts["validation"])
    )
    picks = pick_cases(test, predictions, threshold)

    cases: dict[str, Any] = {}
    for name, position in picks.items():
        row = test.iloc[position]
        cases[name] = {
            "city_id": str(row["city_id"]),
            "date_key": str(row["date_key"].date()),
            "predicted": float(predictions[position]),
            "margin": float(shap.margins()[position]),
            "base_value": float(shap.bias[position]),
            "label": bool(positives(test[LABEL]).to_numpy()[position]),
            "contributions": explain_case(shap, position).to_dict("records"),
        }

    static = ranked.loc[ranked["static_per_city"], "mean_abs"].sum()
    report = {
        "explained_variant": variant,
        "space": "log-odds",
        "note": (
            "SHAP values are in log-odds. The base value is "
            f"{float(shap.bias[0]):.4f}, which is "
            f"{1 / (1 + np.exp(-float(shap.bias[0]))):.4f} as a probability."
        ),
        "implementation": "xgboost pred_contribs (TreeSHAP)",
        "rows_explained": len(shap),
        "threshold": threshold,
        "top_features": ranked.head(TOP_N).to_dict("records"),
        "static_city_share": float(static / ranked["mean_abs"].sum()),
        "tail_response": tail_response(shap).astype(
            {"band": str}
        ).to_dict("records"),
        "cases": cases,
    }
    plots = {"shap": shap, "ranked": ranked, "cases": cases, "picks": picks}
    return report, plots


def render_figures(
    plots: Mapping[str, Any], directory: Path | None = None
) -> list[Path]:
    """A beeswarm of the ranking, and the two explained predictions."""
    plt = _style()
    import matplotlib as mpl

    directory = Path(directory) if directory is not None else FIGURE_DIR
    shap: Contributions = plots["shap"]
    ranked: pd.DataFrame = plots["ranked"]
    written: list[Path] = []

    top = ranked.head(TOP_N)["feature"].tolist()[::-1]
    figure, axis = plt.subplots(figsize=(8.4, 5.4))
    # Seeded, because the vertical scatter is cosmetic and a figure that moves
    # between runs cannot be diffed or committed.
    rng = np.random.default_rng(SEED)
    colours = mpl.colormaps["coolwarm"]
    for row, feature in enumerate(top):
        column = shap.column(feature)
        values = shap.matrix[feature].to_numpy(dtype=float)
        # Shaded on the 1st-99th percentile rather than the full range, so a
        # single outlying value cannot flatten every other point to one colour.
        low, high = np.percentile(values, [1, 99])
        if high > low:
            shade = np.clip((values - low) / (high - low), 0, 1)
        else:
            shade = np.zeros_like(values)
        axis.scatter(
            column,
            row + rng.uniform(-0.22, 0.22, len(column)),
            c=shade,
            cmap=colours,
            s=3.5,
            alpha=0.5,
            linewidths=0,
            rasterized=True,
        )
    axis.axvline(0, color="#9ca3af", linewidth=1.0)
    axis.set_yticks(range(len(top)))
    axis.set_yticklabels(top)
    axis.set_xlabel("SHAP contribution (log-odds)")
    axis.set_title(
        "What moves the model, test split — colour is the feature's own value",
        loc="left",
    )
    axis.grid(axis="y", visible=False)
    bar = figure.colorbar(
        mpl.cm.ScalarMappable(cmap=colours), ax=axis, pad=0.01, fraction=0.03
    )
    bar.set_ticks([0, 1])
    bar.set_ticklabels(["low", "high"])
    written += _save(figure, "shap_summary", directory)
    plt.close(figure)

    cases = plots["cases"]
    if cases:
        figure, axes = plt.subplots(
            1, len(cases), figsize=(6.4 * len(cases), 4.6), sharex=True
        )
        # The tick labels are two lines of feature name and value, and they are
        # drawn outside the axes; without the gap the right panel's labels land
        # on top of the left panel's bars.
        figure.subplots_adjust(wspace=0.55)
        axes = np.atleast_1d(axes)
        for axis, (name, case) in zip(axes, cases.items()):
            detail = pd.DataFrame(case["contributions"]).iloc[::-1]
            colour = [
                PALETTE["model_unweighted"] if value > 0 else PALETTE["climatology"]
                for value in detail["shap"]
            ]
            axis.barh(range(len(detail)), detail["shap"], color=colour, height=0.62)
            axis.set_yticks(range(len(detail)))
            axis.set_yticklabels(
                [
                    f"{feature}\n= {value:,.2f}"
                    for feature, value in zip(detail["feature"], detail["value"])
                ],
                fontsize=7.5,
            )
            axis.axvline(0, color="#9ca3af", linewidth=1.0)
            axis.set_xlabel("SHAP contribution (log-odds)")
            axis.grid(axis="y", visible=False)
            axis.set_title(
                f"{name.replace('_', ' ')} — {case['city_id']} "
                f"{case['date_key']}\np = {case['predicted']:.3f}, "
                f"base {case['base_value']:.2f} → margin {case['margin']:.2f}",
                loc="left",
                fontsize=9,
            )
        written += _save(figure, "shap_cases", directory)
        plt.close(figure)
    return written


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain the classifier with TreeSHAP."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Save the figures and merge the ranking into metrics.json.",
    )
    parser.add_argument(
        "--figures", help="Write figures somewhere other than docs/images."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    report, plots = build_explanation()
    print(f"explaining the {report['explained_variant']} model, "
          f"{report['rows_explained']:,} test rows")
    print(f"{report['note']}\n")

    ranking = pd.DataFrame(report["top_features"]).copy()
    ranking.index = range(1, len(ranking) + 1)
    for column in ("mean_abs", "mean_signed", "correlation"):
        ranking[column] = ranking[column].map("{:+.4f}".format)
    print(ranking.to_string())
    print(
        f"\nstatic per-city features are {report['static_city_share']:.1%} of the "
        "total contribution: that is the model looking up which city it is."
    )

    print("\nboth tails? mean contribution of z_temperature_2m_mean by band")
    tails = pd.DataFrame(report["tail_response"])
    print(tails.to_string(index=False))

    for name, case in report["cases"].items():
        print(
            f"\n{name.replace('_', ' ').upper()}  {case['city_id']} "
            f"{case['date_key']}  p={case['predicted']:.3f}  "
            f"actual={'anomaly' if case['label'] else 'quiet'}"
        )
        detail = pd.DataFrame(case["contributions"]).head(5)
        for row in detail.itertuples():
            print(f"    {row.feature:<34} {row.shap:+.4f}   (value {row.value:,.2f})")

    if args.write or args.figures:
        directory = Path(args.figures) if args.figures else FIGURE_DIR
        for path in render_figures(plots, directory):
            print(f"\nwrote {path}")
        destination = metrics_path()
        if destination.exists():
            payload = json.loads(destination.read_text())
            payload["explainability"] = dict(report)
            payload["explainability"]["recorded_at"] = dt.datetime.now(
                dt.timezone.utc
            ).isoformat(timespec="seconds")
            destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            print(f"wrote {destination}")
    else:
        print("\n(not written; pass --write for figures and metrics.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
