"""View 3 — Storm Dynamics."""

from __future__ import annotations

from dashboard.database import GOLD_SCHEMA
from dashboard.views._scaffold import PendingView

VIEW = PendingView(
    title="Storm Dynamics",
    icon=":material/storm:",
    url_path="storm-dynamics",
    question="Do pressure crashes track with wind extremes?",
    caption=(
        "Every hour of the last two years, plotted as three-hour pressure "
        "change against the gust that accompanied it."
    ),
    # The one view that does not use the anomaly scale, and the one with an
    # open colour question. The proposal specifies "coloured by city" over
    # fifteen cities; no palette carries fifteen categorical hues that stay
    # distinguishable under colour-vision deficiency, and inventing hues to
    # fill the gap is what makes a scatter unreadable. BI-05 resolves it by
    # encoding something other than identity — small multiples per city, or a
    # density surface with one city highlighted on selection — rather than by
    # stretching the palette. Recorded here so the decision is made on purpose.
    encoding=(
        "This view does not use the anomaly scale. Its colour question — "
        "fifteen cities is more categories than any colour-blind-safe "
        "categorical palette carries — is settled in BI-05."
    ),
    ticket="BI-05",
    source_table=f"{GOLD_SCHEMA}.fact_weather_hourly",
    probe_sql=f"""
        select
            count(*)                                  as rows,
            count(pressure_tendency_3h)               as with_tendency,
            count(wind_gusts_10m)                     as with_gusts,
            max(observation_hour)::date::text         as latest
        from {GOLD_SCHEMA}.fact_weather_hourly
    """,
    probe_labels={
        "rows": "Hours",
        "with_tendency": "With tendency",
        "with_gusts": "With gusts",
        "latest": "Latest day",
    },
)


def render() -> None:
    VIEW.render()
