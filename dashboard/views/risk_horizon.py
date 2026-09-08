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
        "observed one, against the threshold chosen on validation."
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

    with st.expander("Model vintage"):
        st.markdown(
            f"""
| | |
|---|---|
| Model | `{stamp['model_version']}` |
| Variant | {stamp['model_variant']} |
| Features | {stamp['feature_count']} |
| Decision threshold | {stamp['threshold']:.4f} |
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
