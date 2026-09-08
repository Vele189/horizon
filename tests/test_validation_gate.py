"""VALIDATION GATE: DBT-11.

Seven cities in `config/cities.yml` carry a dated, documented extreme-weather
event. If those dates do not surface as anomalies, the climatology is wrong and
everything downstream (features, labels, model, dashboard) is built on sand.

This is the automated form of that check. It reads the events from the registry
rather than restating them, so a change to an event date is a change to a test,
which is what `cities.py` promises.

**A missing city skips with a reason; it does not pass.** The daily backfill is
quota-bound across days, and a gate that silently goes green on absent data is
worse than no gate at all: it is the specific failure this ticket exists to
prevent, wearing the costume of success.

**The gate runs twice, once per climatology definition (DBT-13).** DBT-12 added
a detrended baseline beside the plain one, and the worry it raises is specific:
these events are records *against the historical record* - Buenos Aires reached
41.1 °C, its highest since 1957 - and a record-hot day measured against a
baseline that has been warmed to meet it reads less anomalous. If detrending
broke the gate on real events, that would not be a bug in the gate; it would be
the strongest possible argument about which definition the product should ship.

It does not break it. Every checkable event moves by 0.01 to 0.19 sigma, in the
direction the trend predicts, and **not one changes verdict**. The two
definitions are named here as `record` and `era`, both are run, and
:func:`test_no_event_changes_verdict_between_the_definitions` names any event
that ever starts to disagree, with its Z under each.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("sqlalchemy")

from sqlalchemy import text  # noqa: E402

from cities import load_cities  # noqa: E402

GOLD = "gold_marts"
ANOMALIES = f"{GOLD}.fact_weather_anomalies"

#: A documented single-day extreme should be a large departure from its own
#: seasonal normal. 2.5 is the project's anomaly threshold; the gate also
#: records how far past it each event lands, because an event that only just
#: clears is worth knowing about.
THRESHOLD = 2.5

#: How many reference years the baseline needs before its sigma means anything.
#: Below this the Z-score is dominated by sampling noise and a pass or a fail
#: would both be uninformative.
MINIMUM_REFERENCE_YEARS = 10

#: The two questions a climatology can answer, and the columns that answer them.
#:
#: ``record`` is *unusual for the record*: the leave-one-year-out baseline over
#: the whole reference period. ``era`` is *unusual for this era*: DBT-12's
#: detrended baseline, walked along a per-city, per-day trend to meet the year
#: being scored. They are different products, not two implementations of one.
DEFINITIONS: dict[str, tuple[str, str, str]] = {
    "record": (
        "z_temperature_2m_mean",
        "is_anomaly",
        "anomaly_direction",
    ),
    "era": (
        "z_temperature_2m_mean_detrended",
        "is_anomaly_detrended",
        "anomaly_direction_detrended",
    ),
}

#: How far either side of a documented event its own episode is taken to reach.
#:
#: Three weeks. A heat wave is not a day, and the days around a documented
#: extreme belong to the same weather rather than to the comparison set it is
#: being ranked against. Moscow 2010 is the case that settles the number: its
#: run of extreme days spans 25 July to 10 August, and anything shorter would
#: leave the event competing with itself.
EPISODE_DAYS = 21

#: The one the product ships, and the answer DBT-13 exists to force.
#:
#: *Unusual for the record*, everywhere, with the detrended flag carried beside
#: it as a measurement rather than as a product. Three things decided it, and
#: the first is the one that mattered:
#:
#: 1. Detrending does not fix what it was proposed to fix. The label's base
#:    rate rises 2.29x from the training period to the test one; detrended it
#:    rises 2.22x. Five per cent of the drift goes. Changing what every number
#:    in the project means, to buy that, is not a trade worth making.
#: 2. No documented event changes verdict, so there is no evidence from the
#:    gate for preferring one. This test module is where that is established.
#: 3. The Climate Matrix exists to draw how anomaly counts move across thirty
#:    years. Detrending removes, by construction, the signal that view is for.
#:
#: Stated in ``docs/proposal.md`` §5.3 and in the model card, and asserted below
#: so the statement cannot quietly stop matching the code.
SHIPPED_DEFINITION = "record"

EVENTS = [
    (city.id, city.validation_event)
    for city in load_cities()
    if city.validation_event is not None
]


def query(engine, sql: str, **params):
    with engine.connect() as connection:
        return connection.execute(text(sql), params).fetchall()


def scored_day(engine, city_id: str, day: dt.date):
    """Everything both definitions say about one day, in one row.

    Both, in one query, deliberately. The gate's whole question is whether the
    two definitions agree about a given day, and reading them separately would
    make that a comparison of two queries that could drift in their filters.

    ``z_temperature_2m_mean`` is selected twice, once under its own name and
    once as ``z``. :data:`DEFINITIONS` has to name real columns, because the
    ranking query interpolates them into SQL, and the shorter name is what the
    two single-event investigations at the bottom of this module read.
    """
    rows = query(
        engine,
        f"""select a.z_temperature_2m_mean,
                   a.z_temperature_2m_mean as z,
                   a.z_temperature_2m_max as z_max,
                   a.z_temperature_2m_min as z_min,
                   a.anomaly_direction, a.is_anomaly, a.departure_c,
                   a.baseline_observations,
                   a.z_temperature_2m_mean_detrended,
                   a.is_anomaly_detrended,
                   a.anomaly_direction_detrended,
                   a.departure_c_detrended,
                   a.trend_slope_c_per_year, a.trend_reference_years,
                   o.temperature_2m_max, o.temperature_2m_mean
            from {ANOMALIES} a
            join {GOLD}.fact_weather_observations o
              on o.city_id = a.city_id and o.date_key = a.date_key
            where a.city_id = :c and a.date_key = :d""",
        c=city_id,
        d=day,
    )
    return rows[0] if rows else None


def reading(day, definition: str):
    """The (z, flag, direction) triple one definition gives for a scored day."""
    z_column, flag_column, direction_column = DEFINITIONS[definition]
    return (
        getattr(day, z_column),
        getattr(day, flag_column),
        getattr(day, direction_column),
    )


def reference_years(engine, city_id: str) -> int:
    return query(
        engine,
        f"select count(distinct year) from {ANOMALIES} where city_id = :c",
        c=city_id,
    )[0][0]


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------


def test_the_registry_carries_seven_events() -> None:
    """The fixture this gate runs on, asserted so it cannot quietly shrink."""
    assert len(EVENTS) == 7, [city_id for city_id, _ in EVENTS]


def checkable(engine, city_id: str, event) -> int:
    """Reference years, or a skip that says the gate is not satisfied."""
    years = reference_years(engine, city_id)
    if years == 0:
        pytest.skip(
            f"{city_id} has not backfilled; {event.date} cannot be checked. "
            "The gate is NOT satisfied for this event."
        )
    if years < MINIMUM_REFERENCE_YEARS:
        pytest.skip(
            f"{city_id} has {years} reference years, below the "
            f"{MINIMUM_REFERENCE_YEARS} a meaningful sigma needs. "
            "The gate is NOT satisfied for this event."
        )
    return years


@pytest.mark.parametrize("definition", sorted(DEFINITIONS))
@pytest.mark.parametrize(
    ("city_id", "event"), EVENTS, ids=[f"{c}-{e.date}" for c, e in EVENTS]
)
def test_the_documented_event_surfaces_as_an_anomaly(
    engine, city_id, event, definition
) -> None:
    """The gate, run once per climatology definition. Both verdicts reported.

    Parametrised over the definitions rather than checking the shipped one and
    printing the other, so each verdict is a row in the test report with its own
    name. "The gate passes" and "the gate passes on the definition we happen to
    ship" are different claims, and only the first is worth the word gate.
    """
    years = checkable(engine, city_id, event)

    day = scored_day(engine, city_id, event.date)
    assert day is not None, (
        f"{city_id} has {years} years of data but not {event.date} itself"
    )
    z, flagged, direction = reading(day, definition)
    assert z is not None, (
        f"{city_id} {event.date} has no {definition} baseline to score against"
    )

    assert flagged, (
        f"{city_id} {event.date} ({event.description[:60]}) scored "
        f"Z={z:+.2f} against the {definition} climatology, under the "
        f"{THRESHOLD} threshold. Observed mean "
        f"{day.temperature_2m_mean:.1f} °C, departure {day.departure_c:+.1f} °C."
    )
    assert direction == event.direction, (
        f"{city_id} {event.date} flagged {direction} against the {definition} "
        f"climatology, expected {event.direction} (Z={z:+.2f})"
    )


def test_no_event_changes_verdict_between_the_definitions(engine) -> None:
    """The question DBT-13 was written to answer, asked of every event at once.

    The worry was concrete and reasonable: these events are records against the
    *historical record*, and detrending measures a record-hot day against a
    baseline warmed to meet it, so a hot event must read less anomalous. It
    does. Every checkable event moves in exactly that direction, hot events
    towards zero and Sao Paulo's cold event away from it.

    None of them moves far enough to change verdict. That is what makes this a
    finding rather than a warning: there is no evidence *from the gate* for
    preferring one definition, so the choice has to be made on what the
    definitions mean and on DBT-12's measurement of what detrending buys, which
    is what :data:`SHIPPED_DEFINITION` records.

    Any event that ever does disagree is named here with its Z under each, so
    the day this stops being true, the failure says which event and by how much
    rather than that a count changed.
    """
    disagreements = []
    compared = 0
    for city_id, event in EVENTS:
        if reference_years(engine, city_id) < MINIMUM_REFERENCE_YEARS:
            continue
        day = scored_day(engine, city_id, event.date)
        if day is None:
            continue
        verdicts = {name: reading(day, name) for name in DEFINITIONS}
        if any(z is None for z, _, _ in verdicts.values()):
            continue
        compared += 1
        flags = {name: bool(flag) for name, (_, flag, _) in verdicts.items()}
        directions = {name: direction for name, (_, _, direction) in verdicts.items()}
        if len(set(flags.values())) > 1 or len(set(directions.values())) > 1:
            disagreements.append(
                f"{city_id} {event.date}: "
                + ", ".join(
                    f"{name} Z={verdicts[name][0]:+.3f} "
                    f"{'flags ' + str(directions[name]) if flags[name] else 'does not flag'}"
                    for name in sorted(DEFINITIONS)
                )
            )
    if compared == 0:
        pytest.skip("no event has enough reference years to compare definitions")
    assert not disagreements, (
        f"{len(disagreements)} of {compared} documented events now read "
        "differently under the two climatologies. That is not a bug in the "
        "gate; it is evidence about which definition the product should ship, "
        "and docs/proposal.md §5.3 and the model card have to be revisited: "
        + "; ".join(disagreements)
    )


def test_detrending_moves_a_hot_event_towards_the_ordinary(engine) -> None:
    """The direction, which is the half a verdict comparison cannot see.

    Two events could agree on the flag while the detrending did nothing at all,
    or did the opposite of what it claims. Under a warming trend a hot day late
    in the record must read *less* anomalous against a baseline walked forward
    to meet it, and a cold day must read *more*. Get the sign wrong and the
    gate still passes, the ranges still hold, and the flag silently becomes
    more sensitive to recent heat rather than less.
    """
    moved = []
    for city_id, event in EVENTS:
        if reference_years(engine, city_id) < MINIMUM_REFERENCE_YEARS:
            continue
        day = scored_day(engine, city_id, event.date)
        if day is None or day.z is None or day.z_temperature_2m_mean_detrended is None:
            continue
        if day.trend_slope_c_per_year is None or day.trend_slope_c_per_year <= 0:
            continue
        moved.append((city_id, event.direction, day.z,
                      day.z_temperature_2m_mean_detrended))
    if not moved:
        pytest.skip("no checkable event sits on a warming trend")
    for city_id, direction, z, z_detrended in moved:
        if direction == "hot":
            assert z_detrended < z, (
                f"{city_id}: detrending made a hot event on a warming trend "
                f"*more* anomalous, {z:+.3f} -> {z_detrended:+.3f}. The sign "
                "of the correction is wrong."
            )
        else:
            assert z_detrended < z, (
                f"{city_id}: detrending should push a cold event further from "
                f"zero on a warming trend, got {z:+.3f} -> {z_detrended:+.3f}"
            )


@pytest.mark.parametrize("definition", sorted(DEFINITIONS))
@pytest.mark.parametrize(
    ("city_id", "event"), EVENTS, ids=[f"{c}-{e.date}" for c, e in EVENTS]
)
def test_the_event_is_the_most_extreme_of_its_season(
    engine, city_id, event, definition
) -> None:
    """A stronger claim than the flag: the documented day should stand out.

    Flagging is necessary but not sufficient: a climatology that flagged every
    other day in the month would also flag this one. The event should rank near
    the top of its own ±15-day neighbourhood across the whole record, under
    either definition.

    **An event's own days do not count against it**, and getting that wrong is
    what this test used to do. Moscow's 2010 heat wave ran for a fortnight, and
    the twelve most extreme days in Moscow's late-July neighbourhood across
    thirty-two years are *all of them from 2010*. Ranked against them the
    documented date came sixth and the test failed, reporting that the event did
    not stand out when what it had actually found was an event that stood out so
    completely it filled every place above itself. Excluding the days within
    :data:`EPISODE_DAYS` of the event, Moscow is first of 992.

    So the comparison is against *other* episodes. That is the question the
    registry is asking - was this day extreme for this city - and the previous
    form silently punished exactly the longest and most severe events.

    **The neighbourhood is circular.** Buenos Aires' event is 11 January, whose
    ±15 days reach back to 27 December, and a plain ``abs(a - b) <= 15`` puts
    those fifteen days 350 apart and drops them. That halves the comparison set
    for the events sitting at the year boundary, and it halves it silently: the
    test still runs, still ranks, and still passes. The wrap is the same
    double-modulo the climatology window uses, for the same reason.
    """
    years = reference_years(engine, city_id)
    if years < MINIMUM_REFERENCE_YEARS:
        pytest.skip(f"{city_id} has {years} reference years; cannot rank")

    z_column = DEFINITIONS[definition][0]
    comparison = ">" if event.direction == "hot" else "<"
    rank = query(
        engine,
        f"""with target as (
              select climatology_day from {GOLD}.dim_date where date_day = :d
            ),
            neighbourhood as (
              select a.date_key, a.{z_column} as z
              from {ANOMALIES} a
              join {GOLD}.dim_date d on d.date_day = a.date_key
              cross join target
              where a.city_id = :c and a.{z_column} is not null
                and least(
                      abs(d.climatology_day - target.climatology_day),
                      366 - abs(d.climatology_day - target.climatology_day)
                    ) <= 15)
            select count(*) filter (
                     where z {comparison}
                           (select z from neighbourhood where date_key = :d)
                       and abs(date_key - :d) > {EPISODE_DAYS}) + 1,
                   count(*) filter (where abs(date_key - :d) > {EPISODE_DAYS}),
                   count(*) filter (
                     where z {comparison}
                           (select z from neighbourhood where date_key = :d)) + 1
            from neighbourhood""",
        c=city_id,
        d=event.date,
    )
    if not rank or rank[0][1] == 0:
        pytest.skip(f"{city_id} {event.date} not in the warehouse")
    position, total, position_including_own_episode = rank[0]
    # Roughly 31 days x ~32 years of comparable days, so a documented extreme
    # should sit in the top handful rather than merely in the top percent.
    # Three rather than one: London 2022 is beaten by one day from another year
    # and is still plainly the event the registry means.
    assert position <= max(3, total * 0.001), (
        f"{city_id} {event.date} ranks {position} of {total} comparable days "
        f"from other episodes under the {definition} climatology "
        f"({position_including_own_episode} counting its own)"
    )


def test_the_gate_reports_how_much_of_itself_it_could_run(engine) -> None:
    """The gate's own coverage, asserted so partial completion is visible.

    Passing six of seven and skipping the last is not a passing gate. This
    fails while any event is unverifiable, so the ticket cannot close on a
    green suite that quietly checked nothing.
    """
    checkable = []
    blocked = []
    for city_id, event in EVENTS:
        years = reference_years(engine, city_id)
        (checkable if years >= MINIMUM_REFERENCE_YEARS else blocked).append(
            (city_id, str(event.date), years)
        )
    assert not blocked, (
        f"{len(blocked)} of {len(EVENTS)} validation events cannot be checked "
        f"for want of ingested data: {blocked}. The gate is NOT passed."
    )


# ---------------------------------------------------------------------------
# The answer, stated in writing
# ---------------------------------------------------------------------------


STATEMENTS = {
    "docs/proposal.md": "§5.3",
    "docs/model-card.md": "the model card",
    "dashboard/views/anomaly_map.py": "the Anomaly Map",
    # Added by BI-08. Both views show the reader a consequence of the flag --
    # one paints the day, the other predicts the week -- so both owe them the
    # question it answers. A definition stated on one view and not the other is
    # the arrangement a reader would least expect and least notice.
    "dashboard/views/risk_horizon.py": "the Risk Horizon",
}


@pytest.mark.parametrize("path", sorted(STATEMENTS))
def test_the_shipped_definition_is_stated_where_a_reader_will_find_it(path) -> None:
    """DBT-13's actual deliverable: the answer is in writing, not in a default.

    Two climatologies exist in the warehouse and only one is shipped. A reader
    looking at a red dot on the Anomaly Map is owed the question it answers,
    and a reviewer reading the proposal is owed the choice and its reason. The
    failure this guards against is not a wrong answer; it is the code quietly
    settling a question the documents never asked, which is precisely what
    happens when a second column lands beside a first and nothing says which is
    which.
    """
    text_of = (Path(__file__).resolve().parent.parent / path).read_text(
        encoding="utf-8"
    )
    lowered = text_of.lower()
    assert "unusual for the record" in lowered, (
        f"{path} does not say which question the anomaly flag answers"
    )
    assert "detrend" in lowered, (
        f"{path} does not mention the definition that was considered and not "
        "shipped, so a reader cannot tell a choice from an oversight"
    )


def test_the_shipped_definition_is_one_the_warehouse_actually_carries() -> None:
    assert SHIPPED_DEFINITION in DEFINITIONS
    assert set(DEFINITIONS) == {"record", "era"}


# ---------------------------------------------------------------------------
# The two events the ticket names that the registry does not carry
# ---------------------------------------------------------------------------


def test_phoenix_july_2023_is_the_record_month_in_the_data(engine) -> None:
    """The event is present and correctly extreme, as a *duration*.

    The ticket asks for Phoenix's 31-day July 2023 streak to flag. It does not,
    and the investigation says why rather than the threshold being lowered to
    make it: Phoenix in July is always about 46 °C, so no single day of the
    streak departs far from its own seasonal normal. The peak was Z = +1.97.

    What was unprecedented is how long it lasted, and a single-day Z-score
    cannot express duration by construction. The data has it right, since 2023
    is rank 1 of 32 for days at or above 43.3 °C, so this is a limit of the
    detector, not a defect in the climatology, and the honest response is to
    record it rather than to tune the threshold until it passes.
    """
    if reference_years(engine, "phoenix") < MINIMUM_REFERENCE_YEARS:
        pytest.skip("phoenix has not backfilled")
    ranked = query(
        engine,
        f"""select extract(year from date_key)::int,
                   count(*) filter (where temperature_2m_max >= 43.3)
            from {GOLD}.fact_weather_observations
            where city_id = 'phoenix' and extract(month from date_key) = 7
            group by 1 order by 2 desc""",
    )
    assert ranked[0][0] == 2023, ranked[:3]
    assert ranked[0][1] > ranked[1][1], "2023 should lead outright"

    peak = query(
        engine,
        f"""select max(z_temperature_2m_mean) from {ANOMALIES}
            where city_id = 'phoenix'
              and date_key between date '2023-06-30' and date '2023-07-31'""",
    )[0][0]
    assert peak is not None
    assert peak < THRESHOLD, (
        "if this now exceeds the threshold the detector has changed; "
        "revisit the reasoning above"
    )


def test_delhi_late_may_2024_is_hot_but_not_the_record(engine) -> None:
    """The other event the ticket names and the registry does not carry.

    29 May 2024 scored Z = +2.22, hot and under the threshold. That is
    proportionate: in this grid cell 2024 was the second-warmest late May in
    thirty-two years, behind 1998. A detector that called the second-warmest
    such day a 2.5-sigma extreme would be miscalibrated.
    """
    if reference_years(engine, "delhi") < MINIMUM_REFERENCE_YEARS:
        pytest.skip("delhi has not backfilled")
    day = scored_day(engine, "delhi", dt.date(2024, 5, 29))
    assert day is not None
    assert day.z > 1.5, "it should still read as clearly warm"
    assert day.anomaly_direction in ("hot", "none")

    ranked = query(
        engine,
        f"""select extract(year from date_key)::int, max(temperature_2m_max)
            from {GOLD}.fact_weather_observations
            where city_id = 'delhi'
              and to_char(date_key, 'MM-DD') between '05-20' and '06-05'
            group by 1 order by 2 desc limit 3""",
    )
    assert 2024 in [r[0] for r in ranked], ranked
