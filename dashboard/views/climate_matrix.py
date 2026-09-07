"""View 2 — Climate Matrix."""

from __future__ import annotations

from dashboard.database import GOLD_SCHEMA
from dashboard.views._scaffold import PendingView

VIEW = PendingView(
    title="Climate Matrix",
    icon=":material/grid_on:",
    url_path="climate-matrix",
    question="Which cities are seeing more extremes over time?",
    caption=(
        "One cell per city and year, coloured by how many days that year ran "
        "anomalously warm or cold."
    ),
    encoding="diverging",
    ticket="BI-03",
    source_table=f"{GOLD_SCHEMA}.fact_weather_anomalies",
    probe_sql=f"""
        select
            count(distinct year)                        as years,
            min(year)                                   as first_year,
            max(year)                                   as last_year,
            count(distinct city_id) * count(distinct year) as cells
        from {GOLD_SCHEMA}.fact_weather_anomalies
    """,
    probe_labels={
        "years": "Years",
        "first_year": "From",
        "last_year": "To",
        "cells": "Cells",
    },
)


def render() -> None:
    VIEW.render()
