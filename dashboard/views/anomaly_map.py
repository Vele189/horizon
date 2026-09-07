"""View 1 — Global Anomaly Map."""

from __future__ import annotations

from dashboard.database import GOLD_SCHEMA
from dashboard.views._scaffold import PendingView

VIEW = PendingView(
    title="Global Anomaly Map",
    icon=":material/public:",
    url_path="anomaly-map",
    question="Where is it abnormally hot or cold right now?",
    caption=(
        "Each city's most recent day, sized by how far it sat from its own "
        "climatology and coloured by which way."
    ),
    encoding="diverging",
    ticket="BI-03",
    source_table=f"{GOLD_SCHEMA}.fact_weather_anomalies",
    probe_sql=f"""
        select
            count(*)                    as rows,
            count(distinct city_id)     as cities,
            max(date_key)::text         as latest,
            sum(case when is_anomaly then 1 else 0 end) as flagged
        from {GOLD_SCHEMA}.fact_weather_anomalies
    """,
    probe_labels={
        "rows": "City-days",
        "cities": "Cities",
        "latest": "Latest day",
        "flagged": "Flagged anomalies",
    },
)


def render() -> None:
    VIEW.render()
