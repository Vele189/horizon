"""View 2 — Climate Matrix.

Which cities are seeing more extremes over time?

One cell per city-year, shaded by how many days that year ran more than 2.5σ
from that city's own seasonal normal. Years across, cities down.

**A count is a magnitude, so the ramp is sequential.** Hot days climb the warm
ramp and cold days climb the cool one — one hue each, light to dark. The
diverging blue-grey-red scale is *not* used for either, because zero-to-twenty
has a bottom and a top and no meaningful middle: painting it on two hues either
side of a neutral would invent a direction the number does not have. Only the
**net** view, hot minus cold, is genuinely signed, and that is the one that goes
back to the diverging scale.

**Zero is a value; absent is not.** A city-year with no extremes is the palest
step of the ramp. A city-year the backfill has not reached is a hole in the
grid, and its tooltip says so. The two are the same colour on most heatmaps and
they are not the same statement.

**Sorted by trend, not by name.** Alphabetical order puts Auckland above Buenos
Aires and hides the thing the chart is for. The default ranks cities by the
slope of their anomaly-day count against year, so a reader sees the pattern in
the row order before reading a single cell.
"""

from __future__ import annotations

from typing import Final, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import theme
from dashboard.database import GOLD_SCHEMA, run_query
from dashboard.views._scaffold import ViewMeta, current_mode

VIEW = ViewMeta(
    title="Climate Matrix",
    icon=":material/grid_on:",
    url_path="climate-matrix",
    question="Which cities are seeing more extremes over time?",
    caption=(
        "One cell per city and year, shaded by the number of days that ran "
        "more than 2.5σ from that city's own seasonal normal."
    ),
    source_table=f"{GOLD_SCHEMA}.fact_weather_anomalies",
)

# Every city crossed with every year, then the counts joined on. The cross join
# is what guarantees the grid is complete: aggregating the fact alone would
# return only the city-years that have rows, and the heatmap would silently
# change shape as the backfill advanced.
#
# Aggregated in SQL rather than in pandas. The fact is 60k rows and the answer
# is 480 — sending the difference over the wire to group it locally would be
# paying for the same arithmetic twice, once in bandwidth.
_MATRIX_SQL = f"""
    with span as (
        select min(year) as first_year, max(year) as last_year
          from {GOLD_SCHEMA}.fact_weather_anomalies
    ),
    years as (
        select generate_series(first_year, last_year) as year from span
    ),
    grid as (
        select c.city_id, c.name, y.year
          from {GOLD_SCHEMA}.dim_cities c
          cross join years y
    )
    select g.city_id,
           g.name,
           g.year,
           count(a.date_key) filter (
               where a.is_anomaly and a.anomaly_direction = 'hot'
           ) as hot_days,
           count(a.date_key) filter (
               where a.is_anomaly and a.anomaly_direction = 'cold'
           ) as cold_days,
           count(a.z_temperature_2m_mean) as scored_days
      from grid g
      left join {GOLD_SCHEMA}.fact_weather_anomalies a
             on a.city_id = g.city_id
            and a.year = g.year
     group by g.city_id, g.name, g.year
     order by g.name, g.year
"""

# Upper bound of each bucket but the last, which is open. Chosen on the actual
# distribution: half of all scored city-years sit at zero or one, so an even
# split would put almost everything in one step and waste four.
COUNT_BREAKS: Final[tuple[int, ...]] = (0, 2, 5, 10)
COUNT_LABELS: Final[tuple[str, ...]] = ("0", "1–2", "3–5", "6–10", "11+")

# The same boundaries mirrored, so a reader moving between the hot view and the
# net view is not also learning a new set of buckets.
NET_LABELS: Final[tuple[str, ...]] = (
    "≤ −11", "−10…−6", "−5…−3", "−2…−1", "0", "1…2", "3…5", "6…10", "11+",
)

METRICS: Final[Mapping[str, str]] = {
    "Hot": "hot",
    "Cold": "cold",
    "Net": "net",
}

# Below this a slope is arithmetic rather than evidence. Tokyo has four years
# in the marts; a line through four points would sort it against cities with
# thirty-two as though the two numbers meant the same thing.
MIN_TREND_YEARS: Final[int] = 10


def count_bucket(count: int) -> int:
    """Which of the five sequential steps a count falls in."""
    return sum(1 for boundary in COUNT_BREAKS if count > boundary)


def net_bucket(net: int) -> int:
    """Which of the nine diverging steps a signed difference falls in."""
    distance = sum(1 for boundary in COUNT_BREAKS if abs(net) > boundary)
    if distance == 0:
        return theme.NEUTRAL_INDEX
    return theme.NEUTRAL_INDEX + (distance if net > 0 else -distance)


def load() -> pd.DataFrame:
    frame = run_query(_MATRIX_SQL)
    frame["net_days"] = frame["hot_days"] - frame["cold_days"]
    frame["scored"] = frame["scored_days"] > 0
    return frame


def value_column(metric: str) -> str:
    return {"hot": "hot_days", "cold": "cold_days", "net": "net_days"}[metric]


def trends(frame: pd.DataFrame, metric: str) -> pd.Series:
    """Least-squares slope of the metric against year, per city.

    In anomaly-days per decade, because per-year reads as a column of zeroes
    to three decimal places and invites the reader to conclude nothing is
    happening.

    Cities with fewer than :data:`MIN_TREND_YEARS` scored years get NaN rather
    than a number. A slope through four points is not a weaker version of a
    slope through thirty-two; it is a different quantity, and ranking them
    together would put a city at the top of the chart on the strength of a
    coincidence.
    """
    column = value_column(metric)
    out: dict[str, float] = {}
    for name, group in frame[frame["scored"]].groupby("name"):
        if len(group) < MIN_TREND_YEARS:
            out[name] = float("nan")
            continue
        slope = np.polyfit(group["year"].to_numpy(float), group[column].to_numpy(float), 1)[0]
        out[name] = slope * 10
    for name in frame["name"].unique():
        out.setdefault(name, float("nan"))
    return pd.Series(out, name="trend")


def order_cities(frame: pd.DataFrame, metric: str, sort: str) -> list[str]:
    """Row order. Cities with nothing to say sort to the bottom, always.

    Whatever the sort, a city the backfill has not reached carries no
    information and is placed last rather than interleaved — otherwise nine
    empty rows sit between the six that answer the question.
    """
    column = value_column(metric)
    names = sorted(frame["name"].unique())
    scored = frame[frame["scored"]].groupby("name")[column].sum()
    trend = trends(frame, metric)

    def key(name: str) -> tuple:
        has_data = name in scored.index
        if sort == "Trend":
            value = trend.get(name, float("nan"))
            return (not has_data, np.isnan(value), -(value if not np.isnan(value) else 0), name)
        if sort == "Total":
            return (not has_data, False, -float(scored.get(name, 0)), name)
        return (not has_data, False, 0.0, name)

    # Reversed at the end: Plotly's y axis counts upward from the bottom, so
    # the first city in reading order has to be the last one handed over.
    return sorted(names, key=key)[::-1]


def _cell_text(row: pd.Series) -> str:
    head = f"<b>{row['name']}</b> · {int(row['year'])}"
    if not row["scored"]:
        return f"{head}<br><i>not ingested</i>"
    return (
        f"{head}"
        f"<br><b>{int(row['hot_days'])}</b> hot days"
        f"<br><b>{int(row['cold_days'])}</b> cold days"
        f"<br><b>{int(row['net_days']):+d}</b> net"
        f"<br><span style='font-size:0.85em'>{int(row['scored_days'])} days scored</span>"
    )


def _discrete_scale(colours: Sequence[str]) -> list[list]:
    """Plotly wants a continuous scale; this makes it render as blocks."""
    steps = len(colours)
    scale: list[list] = []
    for index, colour in enumerate(colours):
        scale.append([index / steps, colour])
        scale.append([(index + 1) / steps, colour])
    return scale


def matrix(frame: pd.DataFrame, metric: str, order: Sequence[str]) -> pd.DataFrame:
    """Bucket index per cell, NaN where nothing was ingested."""
    frame = frame.copy()
    column = value_column(metric)
    bucket = net_bucket if metric == "net" else count_bucket
    frame["bucket"] = [
        bucket(int(value)) + 0.5 if scored else np.nan
        for value, scored in zip(frame[column], frame["scored"])
    ]
    frame["cell_text"] = [_cell_text(row) for _, row in frame.iterrows()]
    frame["name"] = pd.Categorical(frame["name"], categories=order, ordered=True)
    return frame.sort_values(["name", "year"])


def _figure(frame: pd.DataFrame, metric: str, mode: theme.Mode) -> go.Figure:
    tokens = theme.chrome(mode)
    colours = (
        theme.diverging_scale(mode)
        if metric == "net"
        else theme.sequential_scale(metric, mode)
    )

    grid = frame.pivot(index="name", columns="year", values="bucket")
    text = frame.pivot(index="name", columns="year", values="cell_text")

    figure = go.Figure(
        go.Heatmap(
            z=grid.to_numpy(),
            x=[str(year) for year in grid.columns],
            y=list(grid.index),
            text=text.to_numpy(),
            hovertemplate="%{text}<extra></extra>",
            hoverongaps=True,
            colorscale=_discrete_scale(colours),
            zmin=0,
            zmax=len(colours),
            showscale=False,
            # The 2px separator is the surface showing through, which is also
            # what an un-ingested cell is — so a hole in the grid reads as a
            # wider gap rather than as a colour a reader has to decode.
            xgap=2,
            ygap=2,
        )
    )
    figure.update_layout(
        height=max(320, 34 * len(grid.index) + 90),
        margin={"r": 8, "t": 8, "l": 8, "b": 8},
        paper_bgcolor=tokens["surface"],
        plot_bgcolor=tokens["surface"],
        font={"color": tokens["ink_secondary"]},
        hoverlabel={"align": "left"},
        xaxis={"showgrid": False, "ticks": "", "side": "bottom", "type": "category"},
        yaxis={"showgrid": False, "ticks": "", "type": "category"},
        dragmode=False,
    )
    return figure


def render() -> None:
    st.title(VIEW.title)
    st.caption(VIEW.caption)

    controls, sorting = st.columns([2, 1])
    with controls:
        label = st.segmented_control(
            "Anomalies",
            list(METRICS),
            default="Hot",
            key="climate_matrix_metric",
            help="Net is hot days minus cold days — the only signed view, and "
            "the only one on the diverging scale.",
        )
    with sorting:
        sort = st.selectbox(
            "Order cities by",
            ["Trend", "Total", "Name"],
            key="climate_matrix_sort",
            help=f"Trend is the slope over year, for cities with at least "
            f"{MIN_TREND_YEARS} scored years.",
        )

    metric = METRICS[label or "Hot"]
    mode = current_mode()

    frame = load()
    order = order_cities(frame, metric, sort)
    figure = _figure(matrix(frame, metric, order), metric, mode)

    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    key, ranking = st.columns([3, 2])
    with key:
        st.caption(
            f"{'Net anomaly days (hot − cold)' if metric == 'net' else f'{label} anomaly days'} "
            f"per city-year. A gap is a year that has not been ingested — not a year with none."
        )
        if metric == "net":
            st.markdown(
                theme.anomaly_key_html(mode).replace("Z-score", "net days"),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                theme.sequential_key_html(metric, mode, COUNT_LABELS),
                unsafe_allow_html=True,
            )

    with ranking:
        trend = trends(frame, metric).dropna().sort_values(ascending=False)
        if trend.empty:
            st.caption(
                f"No city has {MIN_TREND_YEARS} scored years yet, so no trend "
                f"is reported."
            )
        else:
            st.caption(f"Trend, {label.lower()} days per decade")
            st.dataframe(
                trend.round(2).rename("per decade").to_frame(),
                width="stretch",
            )
