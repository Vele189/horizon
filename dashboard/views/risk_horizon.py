"""View 4 — Risk Horizon."""

from __future__ import annotations

from dashboard.database import GOLD_SCHEMA
from dashboard.views._scaffold import PendingView

VIEW = PendingView(
    title="Risk Horizon",
    icon=":material/calendar_month:",
    url_path="risk-horizon",
    question="Which cities are flagged for the coming week?",
    caption=(
        "The model's score for each city over the seven days after the last "
        "observed one, against the threshold chosen on validation."
    ),
    encoding="diverging",
    ticket="BI-04",
    source_table=f"{GOLD_SCHEMA}.fact_ml_predictions",
    probe_sql=f"""
        select
            count(*)                                       as rows,
            count(distinct city_id)                        as cities,
            max(forecast_date)::text                       as forecast_date,
            sum(case when prediction_label then 1 else 0 end) as flagged
        from {GOLD_SCHEMA}.fact_ml_predictions
    """,
    probe_labels={
        "rows": "Scored rows",
        "cities": "Cities",
        "forecast_date": "Forecast date",
        "flagged": "Above threshold",
    },
)


def render() -> None:
    VIEW.render()
