"""View 4: Risk Horizon.

Which cities are flagged for the coming week?

This is where the model becomes visible to someone who will never read
``train.py``, which makes what the picture *claims* more important here than
anywhere else in the dashboard.

**One score per city per week, not seven daily scores.** The model's target is
"does an anomaly occur at any point in the next seven days", so
``fact_ml_predictions`` holds one row per city per forecast date with
``horizon_days = 7``. There is no per-day probability to draw, and inventing
one by spreading the week's number across seven cells would be a chart claiming
a resolution the model does not have. So each city is drawn as a single
continuous band across the seven days its score covers: the calendar tells the
reader *which* days, and the absence of any internal boundary tells them the
score does not vary within them.

**Ten of fifteen cities have no prediction, and each absence has a reason.**
Those reasons are read from the model's own ``metrics.json`` rather than
guessed from missing rows. The model records which cities it scored, which
were ingested but not scorable, and which were never ingested.

**Which anomaly it is predicting (DBT-13).** The target is `is_anomaly`, so
this view answers whether next week will be *unusual for the record* -- against
every year of that city's history rather than against the recent climate. The
warehouse also carries a detrended flag, `is_anomaly_detrended`, which asks
whether a week is unusual for the present climate; it is measured and
deliberately not shipped, because detrending removes only 5% of the label's
drift across the split and would erase the signal the Climate Matrix exists to
draw. `docs/proposal.md` §5.3 records the choice.

**The drivers are the model's, recorded at evaluation time.** The top SHAP
features come from the committed metrics file, computed on the test split when
the model was explained. They are global rather than per-cell: a per-row
attribution would need the estimator and the feature matrix at request time,
and neither is available to a dashboard that deliberately ships no warehouse
and no XGBoost.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import get_settings  # noqa: E402

from dashboard import theme  # noqa: E402
from dashboard.database import GOLD_SCHEMA, run_query  # noqa: E402
from dashboard.views._scaffold import ViewMeta, current_mode  # noqa: E402

VIEW = ViewMeta(
    title="Risk Horizon",
    icon=":material/calendar_month:",
    url_path="risk-horizon",
    question="Which cities are flagged for the coming week?",
    caption=(
        "The model's score for each city over the seven days after the last "
        "observed one, against the threshold chosen on validation. It predicts "
        "an anomaly that is *unusual for the record* — measured against every "
        "year of that city's history, not against the recent climate."
    ),
    source_table=f"{GOLD_SCHEMA}.fact_ml_predictions",
)

# Every registered city against the most recent forecast, scored or not. The
# left join is what keeps the ten unscored cities on the page: a reader has to
# be able to see that the model covers a third of the roster, and an inner join
# would present five cities as though they were the world.
_LATEST_SQL = f"""
    with latest as (
        select max(forecast_date) as forecast_date
          from {GOLD_SCHEMA}.fact_ml_predictions
    )
    select c.city_id,
           c.name,
           c.country,
           p.forecast_date,
           p.horizon_start,
           p.horizon_end,
           p.horizon_days,
           p.risk_score,
           p.prediction_label,
           p.decision_threshold,
           p.model_version,
           p.model_variant,
           p.feature_count,
           p.scored_at
      from {GOLD_SCHEMA}.dim_cities c
      left join latest l on true
      left join {GOLD_SCHEMA}.fact_ml_predictions p
             on p.city_id = c.city_id
            and p.forecast_date = l.forecast_date
     order by c.name
"""


@lru_cache(maxsize=1)
def model_report() -> Mapping[str, Any]:
    """The committed evaluation record, read as data rather than imported.

    ``metrics.json`` is loaded straight from disk instead of through
    ``machine_learning``, because importing that package would pull XGBoost and
    scikit-learn into a deployment whose only job is to read finished rows out
    of Postgres. The file is committed precisely so this is possible.
    """
    path = get_settings().model_artifact_dir / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def top_drivers(limit: int = 8) -> pd.DataFrame:
    """The model's strongest features by mean absolute SHAP value."""
    features = model_report().get("explainability", {}).get("top_features", [])
    if not features:
        return pd.DataFrame(columns=["feature", "mean_abs", "mean_signed"])
    frame = pd.DataFrame(features).head(limit)
    return frame.loc[:, [c for c in ("feature", "mean_abs", "mean_signed") if c in frame]]


def calibration() -> Mapping[str, Any]:
    """What the model's probabilities are worth, from the committed record.

    ML-10 measured it and recorded four variants; this view needs two of them,
    the model as it is scored and the same model with the calibrator applied.
    Read rather than recomputed, for the same reason the SHAP drivers are: this
    deployment ships no XGBoost and no warehouse, and could not reproduce the
    number if it wanted to.
    """
    return model_report().get("model", {}).get("calibration", {})


def reliability_points(variant: str) -> pd.DataFrame:
    """One variant's reliability curve, as points a chart can draw."""
    entry = calibration().get("variants", {}).get(variant, {})
    points = entry.get("reliability", [])
    if not points:
        return pd.DataFrame(columns=["predicted", "observed", "rows", "weight"])
    return pd.DataFrame(points)


def calibration_error(variant: str) -> float | None:
    entry = calibration().get("variants", {}).get(variant, {})
    value = entry.get("expected_calibration_error")
    return float(value) if value is not None else None


def trust_sentence() -> str:
    """How much a reader should believe the number, in a sentence.

    Chosen by the recorded error rather than written once and left: the
    threshold is :data:`theme.CALIBRATION_TRUST_CEILING`, and a retrain that
    moved the error past it would move this sentence with it.
    """
    raw = calibration_error("raw_unweighted")
    if raw is None:
        return ""
    if raw <= theme.CALIBRATION_TRUST_CEILING:
        return (
            f"The scores in the grid are within **{raw:.1%}** of the rate they "
            "claim, averaged over the test split, so they can be read as "
            "probabilities."
        )
    calibrated = calibration_error("calibrated")
    fixed = (
        f" Calibrating on the validation split brings it to **{calibrated:.1%}**, "
        "and that calibrator is not shipped with the model, so the grid is "
        "painted from the uncalibrated score."
        if calibrated is not None
        else ""
    )
    return (
        f"**Read the grid as a ranking, not as a percentage.** The scores sit "
        f"**{raw:.1%}** away from the rate they claim, averaged over the test "
        f"split, which is enough to turn one-in-twelve into one-in-eight.{fixed}"
    )


def alert_budget() -> Mapping[str, Any]:
    """The decision the project's threshold encodes, read from the record.

    A threshold shown on its own is a number a reader has to take on trust.
    ML-11 replaced "whatever maximises F1" -- which asserts that a false alarm
    and a missed heatwave cost the same -- with a budget somebody can argue
    with, and the point of putting it here is that they can argue with it
    without opening ``metrics.json``.
    """
    return model_report().get("model", {}).get("calibration", {}).get("decision", {})


def budget_sentence() -> str:
    """The budget in plain English, or nothing if it has not been recorded."""
    decision = alert_budget()
    if not decision:
        return ""
    budget = decision["budget_alerts_per_city_year"]
    ratio = decision["implied_cost_ratio"]
    return (
        f"The threshold is set by an **alert budget**: no city should light up "
        f"more than **{budget:.0f} days a year**. On a calibrated probability "
        f"that is the same as saying **{ratio:.1f} false alarms are worth one "
        f"missed extreme week** — F1, the rule this replaced, silently said "
        f"one."
    )


def absence_reasons() -> dict[str, str]:
    """Why a city has no score, in the model's own words.

    Three states, and they wait on different things. It is the same distinction the
    map draws, taken from the evaluation record rather than inferred from which
    rows happen to be missing.
    """
    evaluation = model_report().get("evaluation", {})
    reasons: dict[str, str] = {}
    for city in evaluation.get("cities_not_ingested", []):
        reasons[city] = "not ingested"
    for city in evaluation.get("cities_ingested_but_not_scored", []):
        reasons[city] = "ingested, but not enough history to score"
    return reasons


def load() -> pd.DataFrame:
    return run_query(_LATEST_SQL)


def horizon_days(frame: pd.DataFrame) -> list[dt.date]:
    """The calendar days the current forecast covers."""
    scored = frame[frame["horizon_start"].notna()]
    if scored.empty:
        return []
    start = scored.iloc[0]["horizon_start"]
    length = int(scored.iloc[0]["horizon_days"])
    return [start + dt.timedelta(days=offset) for offset in range(length)]


def vintage(frame: pd.DataFrame) -> Mapping[str, Any]:
    """Which model produced this, and when it was run."""
    scored = frame[frame["risk_score"].notna()]
    if scored.empty:
        return {}
    row = scored.iloc[0]
    return {
        "model_version": row["model_version"],
        "model_variant": row["model_variant"],
        "scored_at": row["scored_at"],
        "forecast_date": row["forecast_date"],
        "feature_count": int(row["feature_count"]),
        "threshold": float(row["decision_threshold"]),
    }


def _threshold_rule() -> str:
    """How the threshold was arrived at, named rather than left implicit."""
    decision = alert_budget()
    if not decision:
        return "F1 on validation"
    return (
        f"{decision['rejected_rule'].upper()} on validation "
        f"(the recorded rule is an alert budget of "
        f"{decision['budget_alerts_per_city_year']:.0f} a year)"
    )


def _cell_text(row: pd.Series, day: dt.date, reasons: Mapping[str, str]) -> str:
    head = f"<b>{row['name']}</b>, {row['country']}<br>{day:%a %d %b %Y}"
    if pd.isna(row["risk_score"]):
        reason = reasons.get(row["city_id"], "not scored by this model")
        return f"{head}<br><i>{reason}</i>"
    verdict = "above threshold" if row["prediction_label"] else "below threshold"
    return (
        f"{head}"
        f"<br><b>{row['risk_score']:.3f}</b> risk score, {verdict}"
        f"<br>threshold <b>{row['decision_threshold']:.4f}</b>"
        f"<br><span style='font-size:0.85em'>one score for "
        f"{int(row['horizon_days'])} days: "
        f"{row['horizon_start']:%d %b} - {row['horizon_end']:%d %b}</span>"
    )


def grid(frame: pd.DataFrame, days: Sequence[dt.date]) -> tuple[list, list, list]:
    """Rows of step indices and tooltips, one column per horizon day."""
    reasons = absence_reasons()
    steps: list[list[float | None]] = []
    texts: list[list[str]] = []
    for _, row in frame.iterrows():
        if pd.isna(row["risk_score"]):
            value: float | None = None
        else:
            value = theme.risk_step(
                float(row["risk_score"]), float(row["decision_threshold"])
            ) + 0.5
        steps.append([value] * len(days))
        texts.append([_cell_text(row, day, reasons) for day in days])
    return list(frame["name"]), steps, texts


def _discrete_scale(colours: Sequence[str]) -> list[list]:
    total = len(colours)
    scale: list[list] = []
    for index, colour in enumerate(colours):
        scale.append([index / total, colour])
        scale.append([(index + 1) / total, colour])
    return scale


def _figure(frame: pd.DataFrame, days: Sequence[dt.date], mode: theme.Mode) -> go.Figure:
    tokens = theme.chrome(mode)
    colours = theme.risk_scale(mode)
    names, steps, texts = grid(frame, days)

    figure = go.Figure(
        go.Heatmap(
            z=steps,
            x=[f"{day:%a}<br>{day:%d %b}" for day in days],
            y=names,
            text=texts,
            hovertemplate="%{text}<extra></extra>",
            hoverongaps=True,
            colorscale=_discrete_scale(colours),
            zmin=0,
            zmax=len(colours),
            showscale=False,
            # No gap *within* a row: the seven days carry one score, and an
            # internal boundary would draw seven cells where the model made one
            # statement. Rows are separated; days are not.
            xgap=0,
            ygap=3,
        )
    )
    figure.update_layout(
        height=max(360, 32 * len(names) + 110),
        margin={"r": 8, "t": 8, "l": 8, "b": 8},
        paper_bgcolor=tokens["surface"],
        plot_bgcolor=tokens["surface"],
        font={"color": tokens["ink_secondary"]},
        hoverlabel={"align": "left"},
        xaxis={"showgrid": False, "ticks": "", "side": "top", "type": "category"},
        yaxis={"showgrid": False, "ticks": "", "type": "category", "autorange": "reversed"},
        dragmode=False,
    )
    return figure


def _reliability_figure(mode: str) -> go.Figure | None:
    """Predicted against observed, with the line a perfect model would draw.

    The diagonal is the whole chart. Everything else is a curve to compare
    against it: below the line the model is over-confident, above it the model
    is saying "quiet" more often than it should, and this model is above it
    everywhere, which is the shape of a model fitted where positives are 4.9%
    of rows and scored where they are 11.3%.

    Bins hold equal *counts* rather than equal widths, so the leftmost point
    carries as many city-days as the rightmost. That is why the points are not
    evenly spaced along the x-axis, and why they can be read as equally
    trustworthy.
    """
    tokens = theme.chrome(mode)
    colours = theme.calibration_colours(mode)
    series = [
        ("raw_unweighted", "as scored", colours["raw"]),
        ("calibrated", "calibrated on validation", colours["calibrated"]),
    ]
    drawn = [(name, label, colour) for name, label, colour in series
             if not reliability_points(name).empty]
    if not drawn:
        return None

    limit = max(
        float(reliability_points(name)[["predicted", "observed"]].to_numpy().max())
        for name, _, _ in drawn
    )
    edge = min(1.0, limit * 1.1)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0, edge], y=[0, edge], mode="lines", name="perfectly calibrated",
            line={"color": tokens["axis"], "width": 1.2, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    for name, label, colour in drawn:
        points = reliability_points(name)
        figure.add_trace(
            go.Scatter(
                x=points["predicted"], y=points["observed"], mode="lines+markers",
                name=label, line={"color": colour, "width": 2},
                marker={"size": 7, "color": colour},
                customdata=points["rows"],
                hovertemplate=(
                    "predicted %{x:.3f}<br>observed %{y:.3f}"
                    "<br>%{customdata:,} city-days<extra>" + label + "</extra>"
                ),
            )
        )
    figure.update_layout(
        height=320,
        margin={"r": 8, "t": 8, "l": 8, "b": 8},
        paper_bgcolor=tokens["surface"],
        plot_bgcolor=tokens["surface"],
        font={"color": tokens["ink_secondary"]},
        legend={"orientation": "h", "y": -0.2},
        xaxis={"title": "predicted probability", "range": [0, edge], "zeroline": False},
        yaxis={"title": "observed rate", "range": [0, edge], "zeroline": False},
        dragmode=False,
    )
    return figure


DISCLAIMER: Final[str] = (
    "**This is a demonstration model, not an operational forecast.** It is a "
    "gradient-boosted tree fitted to thirty years of reanalysis and scored "
    "against a fixed test split. It is not numerical weather prediction, which is "
    "what actual forecasting uses and what this could not compete with. Do not "
    "plan anything around these numbers."
)


def render() -> None:
    st.title(VIEW.title)
    st.caption(VIEW.caption)
    st.warning(DISCLAIMER)

    frame = load()
    mode = current_mode()
    days = horizon_days(frame)
    stamp = vintage(frame)

    if not days or not stamp:
        st.info(
            "No predictions have been written yet. `machine_learning/predict.py` "
            "scores the horizon and `serving/promote.py` publishes it.",
            icon=":material/pending:",
        )
        return

    st.plotly_chart(
        _figure(frame, days, mode),
        width="stretch",
        config={"displayModeBar": False},
    )

    st.caption(
        f"One score per city for the whole window "
        f"{days[0]:%d %b} - {days[-1]:%d %b}. The band is that score, not "
        f"seven daily estimates. A row with no fill is a city the model does "
        f"not score; hover it for the reason."
    )

    key, drivers = st.columns([3, 2])

    with key:
        st.markdown(
            theme.risk_key_html(mode, stamp["threshold"]), unsafe_allow_html=True
        )
        # Beside the threshold rather than in the vintage panel. The number on
        # the key is where a reader decides what "flagged" means, and a
        # threshold shown without the decision it encodes is a number they have
        # to take on trust.
        sentence = budget_sentence()
        if sentence:
            st.caption(sentence)
        flagged = frame[frame["prediction_label"].fillna(False)]
        st.metric(
            "Flagged for this week",
            f"{len(flagged)} of {int(frame['risk_score'].notna().sum())} scored",
            help=f"{len(frame)} cities in the registry.",
        )
        if not flagged.empty:
            st.caption("Flagged: " + ", ".join(sorted(flagged["name"])))

    with drivers:
        st.markdown("**What the model leans on**")
        st.caption(
            "Mean absolute SHAP value on the test split, from the committed "
            "evaluation record. Global to the model, not per city."
        )
        table = top_drivers()
        if table.empty:
            st.caption("No explanation has been recorded for this model.")
        else:
            st.dataframe(
                table.rename(
                    columns={
                        "feature": "Feature",
                        "mean_abs": "mean |SHAP|",
                        "mean_signed": "mean signed",
                    }
                ).round(3),
                hide_index=True,
                width="stretch",
            )

    with st.expander("How much to trust the number"):
        block = calibration()
        if not block:
            st.caption(
                "No calibration has been recorded for this model. Run "
                "`machine_learning/train.py --write`."
            )
        else:
            sentence = trust_sentence()
            if sentence:
                st.markdown(sentence)

            left, right = st.columns([2, 3])
            with left:
                raw = calibration_error("raw_unweighted")
                calibrated = calibration_error("calibrated")
                if raw is not None:
                    st.metric(
                        "Calibration error, as scored",
                        f"{raw:.1%}",
                        delta=(
                            None if calibrated is None
                            else f"{calibrated - raw:+.1%} calibrated"
                        ),
                        delta_color="inverse",
                        help=(
                            "Expected calibration error: how far the predicted "
                            "probabilities sit from the rates they claim, "
                            "averaged over ten equal-count bins of the test "
                            "split. Zero is perfect. The base rate itself "
                            "scores zero and predicts nothing, which is why "
                            "this is never read without the ranking beside it."
                        ),
                    )
                observed = block.get("observed_target_prior")
                if observed is not None:
                    st.caption(
                        f"Anomalous weeks are **{observed:.1%}** of the test "
                        "split. A model fitted where they were half that will "
                        "say *quiet* more often than it should."
                    )
            with right:
                figure = _reliability_figure(mode)
                if figure is None:
                    st.caption("No reliability curve has been recorded.")
                else:
                    st.plotly_chart(
                        figure, width="stretch",
                        config={"displayModeBar": False},
                    )
            st.caption(
                "Both curves sit **above** the diagonal, which means the model "
                "under-states risk rather than over-stating it: at every level "
                "of predicted probability, more weeks turned out anomalous "
                "than it said. Prior-shift correction was measured for this and "
                "is not shipped, because its estimate of the target period's "
                "rate is badly biased here; the model card has the numbers."
            )

    with st.expander("Model vintage"):
        st.markdown(
            f"""
| | |
|---|---|
| Model | `{stamp['model_version']}` |
| Variant | {stamp['model_variant']} |
| Features | {stamp['feature_count']} |
| Decision threshold | {stamp['threshold']:.4f} |
| Chosen by | {_threshold_rule()} |
| Forecast issued for | {stamp['forecast_date']:%d %B %Y} |
| Scored at | {stamp['scored_at']:%Y-%m-%d %H:%M %Z} |
"""
        )
        st.caption(
            "The forecast date is the last day of *observed* data the score was "
            "computed from; the horizon covers the days after it. Scored at is "
            "when `predict.py` ran, which is a different question from how "
            "fresh the weather behind it is."
        )
        st.caption(
            "The scores in this view are the model's raw output and the "
            "threshold applied to them is still the F1 one, because the budget "
            "rule is defined on calibrated probabilities and the calibrator is "
            "not shipped with the model artefact. The budget and what it buys "
            "are recorded in the model card."
        )
