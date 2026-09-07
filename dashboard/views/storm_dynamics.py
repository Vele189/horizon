"""View 3 — Storm Dynamics.

Do pressure crashes track with wind extremes?

One point per city-day: the largest 24-hour pressure change that day against
that day's strongest gust. Two years of hourly observations for all fifteen
cities, which is the one mart in this warehouse that is complete.

**The naive answer is no, and it is wrong.** Signed pressure change against
peak gust correlates at ρ = −0.04 — a blob. The relationship is not linear, it
is **V-shaped**: a deep low passing gives a sharp fall and then a sharp rise,
and both limbs are windy. Against the *magnitude* of the swing the same data
gives ρ = +0.31. The chart keeps the signed axis the ticket asks for precisely
so the V is visible, and the caption reports both numbers rather than the
flattering one.

**And the honest answer is "in some cities".** The correlation runs from +0.43
in Reykjavík and +0.42 in Auckland to −0.08 in Singapore and +0.02 in Lagos.
Mid-latitude cities sit under a storm track and tropical ones do not, so the
question has a different answer depending on where it is asked. That variation
is the finding, which is why every city's coefficient is on screen rather than
one pooled number.

**Colour does not carry identity here.** Fifteen cities is more identities than
any colour-blind-safe palette holds — see ``theme.py`` for the search that
settles it — so one city is emphasised at a time against a grey field and the
identity of all fifteen lives in the sorted table, where position carries it.
"""

from __future__ import annotations

from typing import Final

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import theme
from dashboard.database import GOLD_SCHEMA, run_query
from dashboard.views._scaffold import ViewMeta, current_mode

VIEW = ViewMeta(
    title="Storm Dynamics",
    icon=":material/storm:",
    url_path="storm-dynamics",
    question="Do pressure crashes track with wind extremes?",
    caption=(
        "One point per city-day: the largest 24-hour pressure change that day, "
        "against the strongest gust that accompanied it."
    ),
    source_table=f"{GOLD_SCHEMA}.fact_weather_hourly",
)

ALL_CITIES: Final[str] = "All cities"

# Aggregated to city-day in the database. The hourly fact is 262 800 usable
# rows and a scatter cannot show them; grouping to 10 950 points is a 24-fold
# reduction that loses nothing the question needs, because "did this day have a
# pressure crash and a gale" is a question about a day.
#
# The signed change kept is the largest *by magnitude*, not the sharpest fall.
# Keeping only falls would show one limb of the V and hide that the rise behind
# a departing low is windy too — which is half the physical story.
_DAILY_SQL = f"""
    with daily as (
        select city_id,
               date_key,
               max(wind_gusts_10m)        as peak_gust,
               min(pressure_tendency_24h) as sharpest_fall,
               max(pressure_tendency_24h) as sharpest_rise,
               -- Counted, not inferred. A day is usually 24 usable hours and
               -- the caption quotes the total; multiplying the day count by 24
               -- would be a guess wearing a measurement's clothes on any day
               -- the archive is short.
               count(*)                   as hours
          from {GOLD_SCHEMA}.fact_weather_hourly
         where pressure_tendency_24h is not null
           and wind_gusts_10m is not null
         group by city_id, date_key
    )
    select d.city_id,
           c.name,
           d.date_key,
           d.peak_gust,
           d.sharpest_fall,
           d.sharpest_rise,
           d.hours,
           case when abs(d.sharpest_fall) >= abs(d.sharpest_rise)
                then d.sharpest_fall else d.sharpest_rise
           end as pressure_change_24h
      from daily d
      join {GOLD_SCHEMA}.dim_cities c on c.city_id = d.city_id
     order by c.name, d.date_key
"""

# Spearman rather than Pearson. Gust distributions have a long right tail and a
# handful of storms would otherwise set the coefficient on their own; rank
# correlation asks the question the caption asks — when the pressure moves
# more, does the wind rank higher — without letting four days answer it.
CORRELATION_METHOD: Final[str] = "spearman"


def load() -> pd.DataFrame:
    frame = run_query(_DAILY_SQL)
    frame["swing"] = frame["pressure_change_24h"].abs()
    return frame


def correlations(frame: pd.DataFrame) -> pd.Series:
    """Per-city rank correlation of swing magnitude against peak gust."""
    return (
        frame.groupby("name")
        .apply(
            lambda g: g["swing"].corr(g["peak_gust"], method=CORRELATION_METHOD),
            include_groups=False,
        )
        .sort_values(ascending=False)
    )


def overall(frame: pd.DataFrame) -> dict[str, float]:
    """The two numbers the caption has to reconcile.

    Computed from the frame on screen rather than written down, so the claim
    the caption makes cannot drift from the data it is made about.
    """
    return {
        "signed": frame["pressure_change_24h"].corr(
            frame["peak_gust"], method=CORRELATION_METHOD
        ),
        "magnitude": frame["swing"].corr(frame["peak_gust"], method=CORRELATION_METHOD),
    }


def _figure(frame: pd.DataFrame, selected: str, mode: theme.Mode) -> go.Figure:
    tokens = theme.chrome(mode)
    accent = theme.emphasis(mode)

    highlighted = frame[frame["name"] == selected] if selected != ALL_CITIES else None
    context = frame if highlighted is None else frame[frame["name"] != selected]

    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=context["pressure_change_24h"],
            y=context["peak_gust"],
            mode="markers",
            name="all cities",
            marker={"size": 4, "color": tokens["ink_muted"], "opacity": 0.22},
            customdata=context[["name", "date_key"]],
            hovertemplate=(
                "<b>%{customdata[0]}</b> · %{customdata[1]}"
                "<br><b>%{x:+.1f} hPa</b> over 24 h"
                "<br><b>%{y:.1f} km/h</b> peak gust<extra></extra>"
            ),
        )
    )
    if highlighted is not None and not highlighted.empty:
        figure.add_trace(
            go.Scattergl(
                x=highlighted["pressure_change_24h"],
                y=highlighted["peak_gust"],
                mode="markers",
                name=selected,
                marker={
                    "size": 6,
                    "color": accent,
                    # The surface ring keeps a highlighted point legible where
                    # it lands inside the grey cloud.
                    "line": {"width": 1, "color": tokens["surface"]},
                },
                customdata=highlighted[["name", "date_key"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b> · %{customdata[1]}"
                    "<br><b>%{x:+.1f} hPa</b> over 24 h"
                    "<br><b>%{y:.1f} km/h</b> peak gust<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        height=520,
        margin={"r": 8, "t": 8, "l": 8, "b": 8},
        paper_bgcolor=tokens["surface"],
        plot_bgcolor=tokens["surface"],
        font={"color": tokens["ink_secondary"]},
        showlegend=False,
        hovermode="closest",
        hoverlabel={"align": "left"},
        xaxis={
            "title": "24-hour pressure change (hPa)",
            "zeroline": True,
            "zerolinecolor": tokens["axis"],
            "gridcolor": tokens["gridline"],
            "linecolor": tokens["axis"],
        },
        yaxis={
            "title": "Peak gust (km/h)",
            "zeroline": False,
            "gridcolor": tokens["gridline"],
            "linecolor": tokens["axis"],
        },
    )
    return figure


def render() -> None:
    st.title(VIEW.title)
    st.caption(VIEW.caption)

    frame = load()
    mode = current_mode()
    names = sorted(frame["name"].unique())

    chooser, _ = st.columns([1, 2])
    with chooser:
        selected = st.selectbox(
            "Highlight a city",
            [ALL_CITIES, *names],
            key="storm_city",
            help=(
                "One at a time. Fifteen cities is more identities than a "
                "colour-blind-safe palette carries, so the table below is "
                "where every city's number lives."
            ),
        )

    st.plotly_chart(
        _figure(frame, selected, mode),
        width="stretch",
        config={"displayModeBar": False},
    )

    pooled = overall(frame)
    per_city = correlations(frame)

    # Every number here is computed from the frame above, so the sentence
    # cannot come apart from the picture it describes.
    st.markdown(
        f"**What the chart shows.** Pressure change and gust are all but "
        f"uncorrelated when the *sign* is kept (ρ = {pooled['signed']:+.2f}) — "
        f"the cloud is a V, not a line, because a passing low brings a sharp "
        f"fall and then a sharp rise and both are windy. Against the "
        f"**size** of the swing, ignoring direction, the same days correlate "
        f"at ρ = {pooled['magnitude']:+.2f}. So pressure crashes do track with "
        f"wind extremes, but the relationship is about how far the barometer "
        f"moved, not which way."
    )
    st.markdown(
        f"**And it depends where you ask.** {per_city.index[0]} reaches "
        f"ρ = {per_city.iloc[0]:+.2f} while {per_city.index[-1]} sits at "
        f"ρ = {per_city.iloc[-1]:+.2f}: mid-latitude cities lie under a storm "
        f"track that drives pressure and wind together, and tropical ones do "
        f"not. One pooled number would hide that."
    )

    table, note = st.columns([2, 1])
    with table:
        st.dataframe(
            per_city.round(3).rename("ρ  swing vs gust").to_frame(),
            width="stretch",
            height=320,
        )
    with note:
        st.caption(
            f"{len(frame):,} city-days, aggregated in the warehouse from "
            f"{int(frame['hours'].sum()):,} hourly observations. Spearman rank "
            f"correlation — gust distributions have a long right tail, and a "
            f"handful of storms should not set the coefficient on their own."
        )
