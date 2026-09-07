"""VALIDATION GATE — DBT-11.

Seven cities in `config/cities.yml` carry a dated, documented extreme-weather
event. If those dates do not surface as anomalies, the climatology is wrong and
everything downstream — features, labels, model, dashboard — is built on sand.

This is the automated form of that check. It reads the events from the registry
rather than restating them, so a change to an event date is a change to a test,
which is what `cities.py` promises.

**A missing city skips with a reason; it does not pass.** The daily backfill is
quota-bound across days, and a gate that silently goes green on absent data is
worse than no gate at all — it is the specific failure this ticket exists to
prevent, wearing the costume of success.
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

EVENTS = [
    (city.id, city.validation_event)
    for city in load_cities()
    if city.validation_event is not None
]


def query(engine, sql: str, **params):
    with engine.connect() as connection:
        return connection.execute(text(sql), params).fetchall()


def scored_day(engine, city_id: str, day: dt.date):
    rows = query(
        engine,
        f"""select a.z_temperature_2m_mean as z,
                   a.z_temperature_2m_max as z_max,
                   a.z_temperature_2m_min as z_min,
                   a.anomaly_direction, a.is_anomaly, a.departure_c,
                   a.baseline_observations,
                   o.temperature_2m_max, o.temperature_2m_mean
            from {ANOMALIES} a
            join {GOLD}.fact_weather_observations o
              on o.city_id = a.city_id and o.date_key = a.date_key
            where a.city_id = :c and a.date_key = :d""",
        c=city_id,
        d=day,
    )
    return rows[0] if rows else None


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


@pytest.mark.parametrize(
    ("city_id", "event"), EVENTS, ids=[f"{c}-{e.date}" for c, e in EVENTS]
)
def test_the_documented_event_surfaces_as_an_anomaly(engine, city_id, event) -> None:
    """The gate. A dated, documented extreme must flag in the right direction."""
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

    day = scored_day(engine, city_id, event.date)
    assert day is not None, (
        f"{city_id} has {years} years of data but not {event.date} itself"
    )
    assert day.z is not None, (
        f"{city_id} {event.date} has no leakage-free baseline to score against"
    )

    assert day.is_anomaly, (
        f"{city_id} {event.date} ({event.description[:60]}) scored "
        f"Z={day.z:+.2f}, under the {THRESHOLD} threshold. "
        f"Observed mean {day.temperature_2m_mean:.1f} °C, "
        f"departure {day.departure_c:+.1f} °C."
    )
    assert day.anomaly_direction == event.direction, (
        f"{city_id} {event.date} flagged {day.anomaly_direction}, "
        f"expected {event.direction} (Z={day.z:+.2f})"
    )


@pytest.mark.parametrize(
    ("city_id", "event"), EVENTS, ids=[f"{c}-{e.date}" for c, e in EVENTS]
)
def test_the_event_is_the_most_extreme_of_its_season(engine, city_id, event) -> None:
    """A stronger claim than the flag: the documented day should stand out.

    Flagging is necessary but not sufficient — a climatology that flagged every
    other day in the month would also flag this one. The event should rank near
    the top of its own ±15-day neighbourhood across the whole record.
    """
    years = reference_years(engine, city_id)
    if years < MINIMUM_REFERENCE_YEARS:
        pytest.skip(f"{city_id} has {years} reference years; cannot rank")

    rank = query(
        engine,
        f"""with neighbourhood as (
              select a.date_key, a.z_temperature_2m_mean as z
              from {ANOMALIES} a
              join {GOLD}.dim_date d on d.date_day = a.date_key
              where a.city_id = :c and a.z_temperature_2m_mean is not null
                and abs(d.climatology_day
                        - (select climatology_day from {GOLD}.dim_date
                           where month_day = to_char(:d::date, 'MM-DD') limit 1)) <= 15)
            select count(*) filter (
                     where {'z >' if event.direction == 'hot' else 'z <'}
                           (select z from neighbourhood where date_key = :d)) + 1,
                   count(*)
            from neighbourhood""",
        c=city_id,
        d=event.date,
    )
    if not rank or rank[0][1] == 0:
        pytest.skip(f"{city_id} {event.date} not in the warehouse")
    position, total = rank[0]
    assert position <= max(3, total * 0.001), (
        f"{city_id} {event.date} ranks {position} of {total} comparable days"
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
# The two events the ticket names that the registry does not carry
# ---------------------------------------------------------------------------


def test_phoenix_july_2023_is_the_record_month_in_the_data(engine) -> None:
    """The event is present and correctly extreme — as a *duration*.

    The ticket asks for Phoenix's 31-day July 2023 streak to flag. It does not,
    and the investigation says why rather than the threshold being lowered to
    make it: Phoenix in July is always about 46 °C, so no single day of the
    streak departs far from its own seasonal normal. The peak was Z = +1.97.

    What was unprecedented is how long it lasted, and a single-day Z-score
    cannot express duration by construction. The data has it right — 2023 is
    rank 1 of 32 for days at or above 43.3 °C — so this is a limit of the
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

    29 May 2024 scored Z = +2.22 — hot, and under the threshold. That is
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
