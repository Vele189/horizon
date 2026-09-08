"""View 1: Global Anomaly Map.

Where is it abnormally hot or cold, on a chosen day?

**Fifteen points, always.** The query left-joins the anomaly fact onto
``dim_cities``, so every city in the registry is on the map at its own
coordinates whether or not the warehouse has scored it. A map that plotted
only the cities with data would silently redraw the world each time the
backfill advanced, and a reader would have no way to tell "normal here" from
"nothing ingested here", which are opposite statements. Cities without a
score are drawn as open rings and say why in their tooltip.

**Two channels, one number.** Size carries magnitude and colour carries
direction, both from the same signed Z-score. That is deliberate redundancy
rather than waste: size survives colour-vision deficiency and a greyscale
print, colour survives a small marker, and the pair is what makes an event
visible at a glance rather than findable on inspection.

**The colour breaks are the warehouse's.** ``theme.anomaly_step`` bins at
0.5 / 1.5 / 2.5 / 3.5, so the two outermost steps of each arm hold exactly the
rows dbt flags with ``is_anomaly``. The map and the mart cannot disagree about
what counts as an anomaly, and a test asserts it at the boundary value where
they otherwise would.

**Which climatology this is, said out loud (DBT-13).** The warehouse carries
two, and a reader looking at a red dot is owed the question it answers. This
map shows *unusual for the record*: the day scored against a baseline built
from every year of that city's history, so a hot day in 2024 is compared with
the whole 1995-2026 record and not only with the recent part of it.

The alternative is *unusual for this era*, the detrended baseline DBT-12 built,
which walks the normal along a fitted warming trend to meet the year being
scored. It is a legitimate and different product: under it a record-hot day is
measured against a warmed baseline and reads less anomalous. It is not what
this view shows, and it is not shipped anywhere, because it does not answer the
question a reader of a map of today's extremes is asking, and because DBT-12
measured what it buys - five per cent of the label's drift across the
chronological split - against what it costs, which is that every number in the
project would mean something else. `docs/proposal.md` §5.3 records the choice.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import theme
from dashboard.database import GOLD_SCHEMA, run_query
from dashboard.views._scaffold import ViewMeta

VIEW = ViewMeta(
    title="Global Anomaly Map",
    icon=":material/public:",
    url_path="anomaly-map",
    question="Where is it abnormally hot or cold right now?",
    caption=(
        "Each city on one day, sized by how far it sat from its own seasonal "
        "normal and coloured by which way. The normal is built from every year "
        "on record, so this is *unusual for the record* rather than unusual "
        "for the present climate; the detrended alternative is measured in the "
        "warehouse and deliberately not shipped."
    ),
    source_table=f"{GOLD_SCHEMA}.fact_weather_anomalies",
)

# The scored day, every city, whether or not it was scored.
_DAY_SQL = f"""
    select c.city_id,
           c.name,
           c.country,
           c.latitude,
           c.longitude,
           a.temperature_2m_mean        as observed_c,
           a.mean_temperature_2m_mean   as baseline_c,
           a.stddev_temperature_2m_mean as baseline_sigma,
           a.anomaly_z_critical,
           a.z_temperature_2m_mean      as z,
           a.departure_c,
           a.is_anomaly,
           a.baseline_observations,
           (a.city_id is not null)      as observed
      from {GOLD_SCHEMA}.dim_cities c
      left join {GOLD_SCHEMA}.fact_weather_anomalies a
             on a.city_id = c.city_id
            and a.date_key = :day
     order by c.name
"""

# The window the date selector may move in, and the day it opens on.
_COVERAGE_SQL = f"""
    select min(date_key)                                        as first_day,
           max(date_key)                                        as last_day,
           max(date_key) filter (where z_temperature_2m_mean is not null)
                                                                as last_scored_day
      from {GOLD_SCHEMA}.fact_weather_anomalies
"""


def _coverage() -> Mapping[str, Any]:
    frame = run_query(_COVERAGE_SQL, reference=True)
    return frame.iloc[0].to_dict()


def _day(day: dt.date) -> pd.DataFrame:
    return run_query(_DAY_SQL, {"day": day})


def _tooltip(row: pd.Series) -> str:
    """One city's readout. The value leads and the label follows it.

    Both reasons a city can be unscored are named rather than collapsed into
    one, because they call for different things: an absent observation waits
    on the backfill, a missing baseline waits on enough reference years for a
    sigma to mean anything.
    """
    head = f"<b>{row['name']}</b>, {row['country']}<br>{row['date_label']}"

    if pd.isna(row["z"]):
        if not row["observed"]:
            return f"{head}<br><i>no observation for this date</i>"
        observed = (
            f"{row['observed_c']:.1f} °C observed"
            if pd.notna(row["observed_c"])
            else "observed"
        )
        return f"{head}<br>{observed}<br><i>no baseline yet, too few reference years</i>"

    return (
        f"{head}"
        f"<br><b>Z {row['z']:+.2f}</b>"
        f"{' (flagged)' if row['is_anomaly'] else ''}"
        f"<br><b>{row['observed_c']:.1f} °C</b> observed"
        f"<br><b>{row['baseline_c']:.1f} °C</b> baseline μ"
        f" (σ {row['baseline_sigma']:.1f})"
        f"<br><b>{row['departure_c']:+.1f} °C</b> departure"
        # Which baseline, on every scored point. The warehouse holds two, and a
        # Z-score with no statement of what it was measured against is the same
        # omission as a PR-AUC with no base rate beside it.
        f"<br><span style='font-size:0.85em'>"
        f"{int(row['baseline_observations'])} reference observations, "
        f"all years (not detrended)"
        f"{_baseline_note(row)}</span>"
    )


def _baseline_note(row: pd.Series) -> str:
    """Say when this city was held to a wider bar, and why.

    The observation count was already printed, and a count alone asks the
    reader to know what it implies. Since DBT-14 the threshold is a function of
    it -- a sigma estimated from fifteen observations earns a wider bar than one
    estimated from four hundred -- so a thin baseline is no longer a caveat the
    reader has to supply, it is a number the mart computed and this can quote.

    Silent on a complete baseline. At 459 observations the bar moves by four
    parts in a thousand, and a note on every point about a correction that
    changes nothing is noise that trains people to ignore the one that matters.
    """
    critical = row.get("anomaly_z_critical")
    if pd.isna(critical):
        return ""
    if float(critical) <= theme.ANOMALY_Z_THRESHOLD * 1.01:
        return ""
    return (
        f"<br><i>thin baseline: judged at |Z| &gt; {float(critical):.2f} "
        f"rather than {theme.ANOMALY_Z_THRESHOLD:.1f}, "
        f"because sigma rests on so few observations</i>"
    )


def _figure(frame: pd.DataFrame) -> go.Figure:
    ground = theme.map_chrome()
    scored = frame[frame["z"].notna()]
    unscored = frame[frame["z"].isna()]

    figure = go.Figure()

    # Drawn first so a scored city is never hidden behind an open ring.
    if not unscored.empty:
        figure.add_trace(
            go.Scattergeo(
                lon=unscored["longitude"],
                lat=unscored["latitude"],
                mode="markers",
                name="not scored",
                marker={
                    # Shape, not colour, carries this distinction, which has to
                    # survive both colour-vision deficiency and the fact that
                    # every fill on this map already means a number.
                    "symbol": "circle-open",
                    "size": theme.MARKER_MIN_PX + 2,
                    "line": {"color": ground["ring"], "width": 2},
                },
                text=unscored["tooltip"],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    if not scored.empty:
        figure.add_trace(
            go.Scattergeo(
                lon=scored["longitude"],
                lat=scored["latitude"],
                mode="markers",
                name="scored",
                marker={
                    "size": scored["diameter"],
                    "color": scored["colour"],
                    "line": {"color": ground["ring"], "width": 2},
                },
                text=scored["tooltip"],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    figure.update_geos(
        projection_type="natural earth",
        showcountries=True,
        showcoastlines=True,
        showland=True,
        showocean=True,
        showframe=False,
        landcolor=ground["land"],
        oceancolor=ground["ocean"],
        bgcolor=ground["ocean"],
        coastlinecolor=ground["coastline"],
        countrycolor=ground["coastline"],
        coastlinewidth=0.5,
        countrywidth=0.5,
    )
    figure.update_layout(
        showlegend=False,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=520,
        paper_bgcolor=ground["ocean"],
        geo_bgcolor=ground["ocean"],
        hoverlabel={"align": "left"},
        dragmode="pan",
    )
    return figure


def prepare(frame: pd.DataFrame, day: dt.date) -> pd.DataFrame:
    """Attach the encoded columns. Split out so a test can read them."""
    frame = frame.copy()
    frame["date_label"] = day.strftime("%d %B %Y")
    frame["colour"] = [
        theme.anomaly_colour(z) if pd.notna(z) else None for z in frame["z"]
    ]
    frame["diameter"] = [
        theme.marker_diameter(z) if pd.notna(z) else None for z in frame["z"]
    ]
    frame["tooltip"] = [_tooltip(row) for _, row in frame.iterrows()]
    return frame


@dataclass(frozen=True)
class Event:
    """One dated extreme from the registry, and what it should look like."""

    city_id: str
    city: str
    date: dt.date
    direction: str
    description: str

    @property
    def label(self) -> str:
        return f"{self.city}, {self.date:%d %b %Y}"


def _events() -> Sequence[Event]:
    """The DBT-11 validation events, read from the registry rather than restated.

    `config/cities.yml` is a test fixture as much as a config file, and these
    are the seven dates the Day 8 gate checks the climatology against. Offering
    them here means the map can be pointed at a documented extreme in one
    click, which is how you confirm it lights up on a real event rather than on
    a day that merely looks busy.
    """
    from cities import load_cities

    return [
        Event(
            city_id=city.id,
            city=city.name,
            date=city.validation_event.date,
            direction=city.validation_event.direction,
            description=city.validation_event.description,
        )
        for city in load_cities()
        if city.validation_event is not None
    ]


_CITY_COVERAGE_SQL = f"""
    select city_id,
           min(date_key)                    as first_day,
           max(date_key)                    as last_day,
           count(*)                         as observed_days,
           count(z_temperature_2m_mean)     as scored_days
      from {GOLD_SCHEMA}.fact_weather_anomalies
     group by city_id
"""


def verify(event: Event, frame: pd.DataFrame, coverage: pd.DataFrame) -> tuple[str, str]:
    """Did the map light up on a documented extreme? Returns (state, sentence).

    This is the Day 8 gate's question asked of the picture rather than of the
    warehouse, and it answers in the gate's own terms: an event whose city has
    not been ingested is **pending**, not passed. A verification that went
    green on absent data would be the exact failure the gate exists to prevent,
    wearing the costume of success.
    """
    row = frame.loc[frame["city_id"] == event.city_id]
    scored = not row.empty and pd.notna(row.iloc[0]["z"])

    if not scored:
        # Three different absences, and they wait on different things. Saying
        # "no data" to all three would misreport the warehouse: London has a
        # year of observations and no baseline, which is not the same as
        # Moscow, which has nothing.
        window = coverage.loc[coverage["city_id"] == event.city_id]
        if window.empty or not int(window.iloc[0]["observed_days"]):
            held = "nothing ingested for this city yet"
        elif not int(window.iloc[0]["scored_days"]):
            first, last = window.iloc[0]["first_day"], window.iloc[0]["last_day"]
            held = (
                f"{first:%d %b %Y} to {last:%d %b %Y} is ingested, but no baseline "
                f"has enough reference years to score against"
            )
        else:
            first, last = window.iloc[0]["first_day"], window.iloc[0]["last_day"]
            held = f"the marts hold {first:%d %b %Y} to {last:%d %b %Y}"
        return "pending", (
            f"**{event.city}, {event.date:%d %b %Y}: cannot be checked yet.** "
            f"No scored observation for this day ({held}). The event stays "
            f"pending rather than passing; the map will show it the moment the "
            f"backfill reaches it."
        )

    z = float(row.iloc[0]["z"])
    flagged = bool(row.iloc[0]["is_anomaly"])
    observed_direction = "hot" if z > 0 else "cold"
    agrees = observed_direction == event.direction

    if flagged and agrees:
        return "pass", (
            f"**{event.city}, {event.date:%d %b %Y}: Z {z:+.2f}, flagged.** "
            f"A documented {event.direction} extreme surfaces as one, in the "
            f"right direction. The climatology holds here."
        )
    if agrees:
        return "weak", (
            f"**{event.city}, {event.date:%d %b %Y}: Z {z:+.2f}, not flagged.** "
            f"The departure is in the documented {event.direction} direction but "
            f"does not clear {theme.ANOMALY_Z_THRESHOLD}."
        )
    return "fail", (
        f"**{event.city}, {event.date:%d %b %Y}: Z {z:+.2f}.** A documented "
        f"{event.direction} event reads {observed_direction}. Something is wrong "
        f"with the climatology, not with the map."
    )


_NO_EVENT = "—"
_APPLIED = "_anomaly_map_applied_event"
_OUT_OF_RANGE = "_anomaly_map_event_out_of_range"


def _selected_day(coverage: Mapping[str, Any]) -> tuple[dt.date, "Event | None"]:
    """The filter row: a free date picker, plus the documented events.

    One row, above the chart, scoping everything below it.
    """
    first, last = coverage["first_day"], coverage["last_day"]
    default = coverage["last_scored_day"] or last

    if "anomaly_map_day" not in st.session_state:
        st.session_state["anomaly_map_day"] = default

    picker, jump = st.columns([1, 2])

    events = {event.label: event for event in _events()}

    with jump:
        choice = st.selectbox(
            "Jump to a documented extreme",
            [_NO_EVENT, *events],
            key="anomaly_map_event",
            help=(
                "The seven dated events in config/cities.yml that the Day 8 "
                "validation gate checks the climatology against."
            ),
        )
        # Applied once, when the selection changes. Applying it on every rerun
        # would let the jump overrule the date picker for as long as an event
        # stayed selected: the reader would move the date and watch it snap
        # back, with nothing on screen explaining why.
        if choice != _NO_EVENT and st.session_state.get(_APPLIED) != choice:
            st.session_state[_APPLIED] = choice
            chosen = events[choice].date
            st.session_state["anomaly_map_day"] = min(max(chosen, first), last)

    with picker:
        day = st.date_input(
            "Date",
            min_value=first,
            max_value=last,
            key="anomaly_map_day",
            format="YYYY-MM-DD",
        )

    selected = events.get(choice)
    # Only while the map is actually showing the event's day. Leaving the
    # verdict on screen after the reader moved the date would attach it to a
    # picture it is not about.
    return day, selected if selected and selected.date == day else None


def render() -> None:
    st.title(VIEW.title)
    st.caption(VIEW.caption)

    coverage = _coverage()
    day, event = _selected_day(coverage)

    frame = prepare(_day(day), day)

    if event is not None:
        state, sentence = verify(event, frame, run_query(_CITY_COVERAGE_SQL, reference=True))
        {"pass": st.success, "weak": st.warning, "fail": st.error, "pending": st.info}[state](
            sentence, icon=None
        )
        st.caption(event.description)
    scored = int(frame["z"].notna().sum())
    flagged = int(frame["is_anomaly"].fillna(False).sum())

    st.plotly_chart(
        _figure(frame),
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False},
    )

    key, size, count = st.columns([3, 2, 2])
    with key:
        st.markdown(theme.anomaly_key_html(), unsafe_allow_html=True)
    with size:
        st.markdown(theme.size_key_html(), unsafe_allow_html=True)
    with count:
        st.metric("Flagged this day", flagged, help="|Z| above 2.5, both tails.")
        st.caption(f"{scored} of {len(frame)} cities scored.")

    with st.expander("The numbers behind the map"):
        st.dataframe(
            frame.loc[
                :,
                ["name", "country", "observed_c", "baseline_c", "z", "departure_c", "is_anomaly"],
            ].rename(
                columns={
                    "name": "City",
                    "country": "Country",
                    "observed_c": "Observed °C",
                    "baseline_c": "Baseline μ °C",
                    "z": "Z",
                    "departure_c": "Departure °C",
                    "is_anomaly": "Flagged",
                }
            ),
            hide_index=True,
            width="stretch",
        )
