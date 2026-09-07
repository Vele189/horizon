"""Per-city reconciliation: expected against actual, and every gap explained.

A gap is not automatically an error. The ERA5 archive has genuine boundaries,
the backfill is quota-bound across days, and a range nobody has asked for yet
is missing for a reason that is not a defect. But every gap has to be *named*,
because the failure this guards against is a hole nobody notices until a
thirty-year climatology is quietly computed over twenty-eight.

Every gap is placed in exactly one category:

``archive boundary``
    Outside what the source can serve — before ERA5 begins, or after the edge
    the archive currently reaches. Nothing can fill it.
``not ingested``
    Inside the servable range, and no completed unit in the manifest covers
    it. The backfill has not got there. Expected while a multi-day backfill is
    in progress; a defect once it reports complete.
``api limitation``
    A unit the manifest records as complete, whose recorded row count is short
    of the window it covers. Structurally prevented — the client asserts the
    row count against the requested range before parsing and the loader
    asserts it again — so a non-empty result here means one of those assertions
    has been weakened.
``unexplained``
    Inside the servable range, covered by a completed unit, and missing
    anyway. Nothing should ever land here.

Usage::

    python ingestion/reconcile.py --grain daily
    python ingestion/reconcile.py --grain daily --write docs/

Writing emits a markdown artefact suitable for committing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Sequence

from sqlalchemy import Engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cities import City, load_cities  # noqa: E402
from config import Settings, get_settings  # noqa: E402
from ingestion.client import ARCHIVE_START, Grain  # noqa: E402
from ingestion.loader import BRONZE_SCHEMA, TABLE_BY_GRAIN, engine_from_settings  # noqa: E402
from ingestion.planner import (  # noqa: E402
    BACKFILL_START,
    Manifest,
    _add_months,
    archive_end_date,
)

__all__ = [
    "CATEGORIES",
    "CityReconciliation",
    "Gap",
    "Report",
    "reconcile",
    "to_markdown",
]

log = logging.getLogger(__name__)

ARCHIVE_BOUNDARY: Final[str] = "archive boundary"
NOT_INGESTED: Final[str] = "not ingested"
API_LIMITATION: Final[str] = "api limitation"
UNEXPLAINED: Final[str] = "unexplained"

CATEGORIES: Final[tuple[str, ...]] = (
    ARCHIVE_BOUNDARY,
    NOT_INGESTED,
    API_LIMITATION,
    UNEXPLAINED,
)

#: Gaps at or under this are not listed individually. One missing day in a
#: thirty-year series is noise; four consecutive is a story.
MIN_REPORTED_GAP_DAYS: Final[int] = 3

_STEP: Final[dict[str, dt.timedelta]] = {
    "daily": dt.timedelta(days=1),
    "hourly": dt.timedelta(hours=1),
}


@dataclass(frozen=True)
class Gap:
    """A run of missing observations, and why it is missing."""

    city_id: str
    grain: Grain
    start: dt.date
    end: dt.date
    category: str
    note: str = ""

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    @property
    def observations(self) -> int:
        return self.days if self.grain == "daily" else self.days * 24

    def __str__(self) -> str:
        return f"{self.city_id} {self.start}..{self.end} ({self.days}d, {self.category})"


@dataclass(frozen=True)
class CityReconciliation:
    """One city's expected-against-actual, with its gaps catalogued."""

    city_id: str
    expected: int
    #: Every row held for this city, in range or not, duplicated or not.
    actual: int
    #: Distinct observation times inside the reconciled range.
    distinct: int
    #: Distinct observation times anywhere. Sits between the two above, and
    #: separates the two ways ``actual`` can exceed ``distinct``: the same
    #: observation landed twice, and an observation outside the range.
    distinct_all: int
    first: dt.date | None
    last: dt.date | None
    gaps: tuple[Gap, ...]

    @property
    def delta(self) -> int:
        """Distinct observations minus expected. Negative is a shortfall."""
        return self.distinct - self.expected

    @property
    def duplicates(self) -> int:
        """Rows that repeat an observation already held."""
        return self.actual - self.distinct_all

    @property
    def gap_count(self) -> int:
        return len(self.gaps)

    @property
    def outside_range(self) -> int:
        """Distinct observations held for this city outside the window.

        A surplus, not coverage: real observations this range did not ask for.
        Named rather than absorbed, because a row count above expectations is
        as much a discrepancy as one below — and it is a different one from a
        duplicate, which is why both are counted separately.
        """
        return max(0, self.distinct_all - self.distinct)

    @property
    def reconciled(self) -> bool:
        return self.delta == 0 and not self.unexplained

    @property
    def unexplained(self) -> tuple[Gap, ...]:
        return tuple(g for g in self.gaps if g.category == UNEXPLAINED)


@dataclass(frozen=True)
class Report:
    """The whole reconciliation, for one grain over one range."""

    grain: Grain
    start: dt.date
    end: dt.date
    generated_at: dt.datetime
    cities: tuple[CityReconciliation, ...]
    missing_cities: tuple[str, ...]
    min_gap_days: int = MIN_REPORTED_GAP_DAYS

    @property
    def gaps(self) -> tuple[Gap, ...]:
        return tuple(g for c in self.cities for g in c.gaps)

    def by_category(self) -> dict[str, list[Gap]]:
        grouped: dict[str, list[Gap]] = {c: [] for c in CATEGORIES}
        for gap in self.gaps:
            grouped[gap.category].append(gap)
        return grouped

    @property
    def unexplained(self) -> tuple[Gap, ...]:
        return tuple(g for g in self.gaps if g.category == UNEXPLAINED)

    @property
    def reconciled(self) -> bool:
        return not self.missing_cities and all(c.reconciled for c in self.cities)


# ---------------------------------------------------------------------------
# Interval arithmetic
# ---------------------------------------------------------------------------

Span = tuple[dt.date, dt.date]


def _intersect(a: Span, b: Span) -> Span | None:
    start, end = max(a[0], b[0]), min(a[1], b[1])
    return (start, end) if start <= end else None


def _subtract(span: Span, cuts: Iterable[Span]) -> list[Span]:
    """``span`` minus every span in ``cuts``. Cuts need not be sorted."""
    remaining = [span]
    for cut in cuts:
        nxt: list[Span] = []
        for piece in remaining:
            overlap = _intersect(piece, cut)
            if overlap is None:
                nxt.append(piece)
                continue
            if piece[0] < overlap[0]:
                nxt.append((piece[0], overlap[0] - dt.timedelta(days=1)))
            if overlap[1] < piece[1]:
                nxt.append((overlap[1] + dt.timedelta(days=1), piece[1]))
        remaining = nxt
    return remaining


def _categorise(
    span: Span,
    *,
    city_id: str,
    grain: Grain,
    servable: Span,
    covered: Sequence[Span],
    short_units: Sequence[Span],
) -> list[Gap]:
    """Split one missing span into the categories that explain its parts."""
    gaps: list[Gap] = []

    # 1. Anything the source cannot serve, whatever the manifest says.
    for piece in _subtract(span, [servable]):
        gaps.append(
            Gap(
                city_id, grain, *piece,
                category=ARCHIVE_BOUNDARY,
                note=f"outside the servable range {servable[0]}..{servable[1]}",
            )
        )

    inside = _intersect(span, servable)
    if inside is None:
        return gaps

    # 2. Inside, and a unit that landed short is on record for it.
    for cut in short_units:
        overlap = _intersect(inside, cut)
        if overlap:
            gaps.append(
                Gap(
                    city_id, grain, *overlap,
                    category=API_LIMITATION,
                    note="a completed unit recorded fewer rows than its window",
                )
            )
    inside_pieces = _subtract(inside, short_units)

    # 3. Inside, never fetched — the backfill has not reached it.
    for piece in inside_pieces:
        for uncovered in _subtract(piece, covered):
            gaps.append(
                Gap(
                    city_id, grain, *uncovered,
                    category=NOT_INGESTED,
                    note="no completed unit in the manifest covers this range",
                )
            )
        # 4. Inside, fetched, recorded complete, and missing anyway.
        for cut in covered:
            overlap = _intersect(piece, cut)
            if overlap:
                gaps.append(
                    Gap(
                        city_id, grain, *overlap,
                        category=UNEXPLAINED,
                        note="a completed unit covers this range but no rows landed",
                    )
                )
    return gaps


# ---------------------------------------------------------------------------
# Reading the warehouse
# ---------------------------------------------------------------------------


def _observed(engine: Engine, grain: Grain) -> dict[str, list[Span]]:
    """Contiguous runs of observations per city, as inclusive date spans.

    Found with ``lag`` rather than by pulling every timestamp: the hourly table
    alone is 280 000 rows, and only the boundaries matter.
    """
    table = f"{BRONZE_SCHEMA}.{TABLE_BY_GRAIN[grain]}"
    step = _STEP[grain]
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                f"""
                with observed as (
                    select distinct city_id, observation_time from {table}
                ),
                stepped as (
                    select city_id, observation_time,
                           lag(observation_time) over (
                               partition by city_id order by observation_time
                           ) as previous
                    from observed
                ),
                marked as (
                    select city_id, observation_time,
                           case when previous is null
                                  or observation_time - previous > :step
                                then 1 else 0 end as starts_run
                    from stepped
                ),
                grouped as (
                    select city_id, observation_time,
                           sum(starts_run) over (
                               partition by city_id order by observation_time
                               rows between unbounded preceding and current row
                           ) as run
                    from marked
                )
                select city_id, min(observation_time), max(observation_time)
                from grouped group by city_id, run order by city_id, 2
                """
            ),
            {"step": step},
        ).fetchall()

    runs: dict[str, list[Span]] = {}
    for city_id, first, last in rows:
        runs.setdefault(city_id, []).append((first.date(), last.date()))
    return runs


def _counts(engine: Engine, grain: Grain) -> dict[str, tuple[int, int]]:
    table = f"{BRONZE_SCHEMA}.{TABLE_BY_GRAIN[grain]}"
    with engine.connect() as connection:
        return {
            city_id: (rows, distinct)
            for city_id, rows, distinct in connection.execute(
                text(
                    f"select city_id, count(*), "
                    f"count(distinct observation_time) from {table} "
                    "group by city_id"
                )
            )
        }


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def reconcile(
    engine: Engine,
    *,
    grain: Grain = "daily",
    start: dt.date | None = None,
    end: dt.date | None = None,
    cities: Iterable[City] | None = None,
    manifest: Manifest | None = None,
    min_gap_days: int = MIN_REPORTED_GAP_DAYS,
    settings: Settings | None = None,
) -> Report:
    """Compare expected against actual for every city, and explain every gap."""
    resolved_settings = settings if settings is not None else get_settings()
    registry = list(cities) if cities is not None else list(load_cities())
    resolved_manifest = (
        manifest
        if manifest is not None
        else Manifest(resolved_settings.ingest_manifest_path)
    )
    resolved_end = end or archive_end_date()
    if start is not None:
        resolved_start = start
    elif grain == "daily":
        resolved_start = BACKFILL_START
    else:
        resolved_start = _add_months(
            resolved_end, -resolved_settings.ingest_hourly_months
        )

    window: Span = (resolved_start, resolved_end)
    servable: Span = (ARCHIVE_START, archive_end_date())
    per_city_step = 1 if grain == "daily" else 24
    expected = ((resolved_end - resolved_start).days + 1) * per_city_step

    observed = _observed(engine, grain)
    counts = _counts(engine, grain)

    reconciliations: list[CityReconciliation] = []
    missing: list[str] = []
    for city in registry:
        runs = observed.get(city.id) or []
        if not runs:
            missing.append(city.id)

        rows, distinct_all = counts.get(city.id, (0, 0))
        covered = resolved_manifest.coverage(city.id, grain)
        short_units = tuple(
            (entry.start, entry.end)
            for entry in resolved_manifest.entries
            if entry.city_id == city.id
            and entry.grain == grain
            and entry.rows
            < ((entry.end - entry.start).days + 1) * per_city_step
        )

        # Only the part of the observed data inside the window counts towards
        # this window's reconciliation; rows outside it are a surplus, reported
        # separately rather than treated as coverage.
        inside_runs = [r for r in (_intersect(run, window) for run in runs) if r]
        distinct_inside = sum(
            ((r[1] - r[0]).days + 1) * per_city_step for r in inside_runs
        )

        gaps: list[Gap] = []
        for missing_span in _subtract(window, inside_runs):
            gaps.extend(
                _categorise(
                    missing_span,
                    city_id=city.id,
                    grain=grain,
                    servable=servable,
                    covered=covered,
                    short_units=short_units,
                )
            )
        reportable = tuple(
            sorted(
                (g for g in gaps if g.days > min_gap_days),
                key=lambda g: (g.city_id, g.start),
            )
        )

        reconciliations.append(
            CityReconciliation(
                city_id=city.id,
                expected=expected,
                actual=rows,
                distinct=distinct_inside,
                distinct_all=distinct_all,
                first=min((r[0] for r in runs), default=None),
                last=max((r[1] for r in runs), default=None),
                gaps=reportable,
            )
        )

    return Report(
        grain=grain,
        start=resolved_start,
        end=resolved_end,
        generated_at=dt.datetime.now(dt.timezone.utc),
        cities=tuple(reconciliations),
        missing_cities=tuple(missing),
        min_gap_days=min_gap_days,
    )


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def to_markdown(report: Report, accepted: dict[str, str] | None = None) -> str:
    """Render the report as a committable artefact.

    Args:
        accepted: Category or gap description -> the reason it is accepted.
            Anything left unexplained is called out as such rather than
            quietly rendered.
    """
    accepted = accepted or {}
    grouped = report.by_category()
    lines: list[str] = []
    add = lines.append

    add(f"# Ingestion reconciliation — `{report.grain}`")
    add("")
    add(
        f"Generated {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')} "
        f"by `python ingestion/reconcile.py --grain {report.grain}`."
    )
    add("")
    add(
        f"Range **{report.start} .. {report.end}**, "
        f"{report.cities[0].expected:,} observations expected per city "
        f"across {len(report.cities)} cities."
    )
    add("")

    add("## Per city")
    add("")
    add("| city | expected | actual | distinct | delta | gaps |")
    add("|---|---:|---:|---:|---:|---:|")
    for city in sorted(report.cities, key=lambda c: c.city_id):
        delta = f"{city.delta:+,}" if city.delta else "0"
        add(
            f"| `{city.city_id}` | {city.expected:,} | {city.actual:,} | "
            f"{city.distinct:,} | {delta} | {city.gap_count} |"
        )
    add("")
    add(
        "`actual` counts every row held for the city; `distinct` counts "
        "distinct observation times inside the range, which is what `delta` "
        "compares against `expected`. The two diverge for two separate "
        "reasons, reported separately below: the same observation landed twice "
        "— legal, since bronze is append-only and silver deduplicates — or the "
        "row falls outside the range this report asked about."
    )
    add("")

    surplus = [c for c in report.cities if c.outside_range]
    if surplus:
        add("**Rows outside the reconciled range** — a surplus, not a gap:")
        add("")
        add("| city | rows outside |")
        add("|---|---:|")
        for city in sorted(surplus, key=lambda c: c.city_id):
            add(f"| `{city.city_id}` | {city.outside_range:,} |")
        add("")
        add(
            "Real observations the range did not ask about — usually a wider "
            "window ingested earlier. They do not count towards `delta` and "
            "are not a defect; downstream models filter by range."
        )
        add("")

    if report.missing_cities:
        add(
            f"**{len(report.missing_cities)} cities absent entirely:** "
            + ", ".join(f"`{c}`" for c in report.missing_cities)
        )
        add("")

    add("## Gaps by category")
    add("")
    add(f"Every gap longer than {report.min_gap_days} days, categorised.")
    add("")
    add("| category | gaps | observations | meaning |")
    add("|---|---:|---:|---|")
    meanings = {
        ARCHIVE_BOUNDARY: "Outside what ERA5 can serve. Nothing can fill it.",
        NOT_INGESTED: "Inside the servable range; the backfill has not reached it.",
        API_LIMITATION: "A completed unit recorded fewer rows than its window.",
        UNEXPLAINED: "Fetched, recorded complete, and missing anyway.",
    }
    for category in CATEGORIES:
        found = grouped[category]
        add(
            f"| {category} | {len(found)} | "
            f"{sum(g.observations for g in found):,} | {meanings[category]} |"
        )
    add("")

    for category in CATEGORIES:
        found = grouped[category]
        if not found:
            add(f"### {category}")
            add("")
            add("None.")
            if category in accepted:
                add("")
                add(accepted[category])
            add("")
            continue

        add(f"### {category} — {len(found)} gap(s)")
        add("")
        if category in accepted:
            add(accepted[category])
            add("")
        add("| city | from | to | days |")
        add("|---|---|---|---:|")
        for gap in sorted(found, key=lambda g: (g.city_id, g.start))[:200]:
            add(f"| `{gap.city_id}` | {gap.start} | {gap.end} | {gap.days:,} |")
        if len(found) > 200:
            add(f"| … | | | {len(found) - 200} more |")
        add("")

    add("## Verdict")
    add("")
    if report.unexplained:
        add(
            f"**{len(report.unexplained)} gap(s) remain unexplained.** Each is "
            "a range a completed unit claims to cover, with no rows to show "
            "for it. This should be impossible and wants investigating before "
            "anything downstream trusts these tables."
        )
    else:
        add(
            "**No gap is unexplained.** Every one is either outside what the "
            "archive can serve, or inside a range the backfill has not reached "
            "yet — and the latter shrinks to nothing as the backfill "
            "completes."
        )
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


ACCEPTED: Final[dict[str, str]] = {
    ARCHIVE_BOUNDARY: (
        "Accepted. The ERA5 archive begins 1940-01-01 and trails the present "
        "by several days; the planner already stops short of the edge, so "
        "anything here is a range that was asked for outside those bounds."
    ),
    NOT_INGESTED: (
        "Accepted while the backfill is in progress. The daily grain costs "
        "~26 000 weighted API calls against a free-tier allowance of 10 000 a "
        "day, so it completes across roughly three days. Every range here is "
        "pending, not lost — the manifest resumes rather than restarts."
    ),
    API_LIMITATION: (
        "Structurally prevented rather than merely absent: the client asserts "
        "the returned row count against the requested range before parsing, "
        "and the loader asserts it again before writing. A short response "
        "raises instead of landing."
    ),
    UNEXPLAINED: (
        "Nothing should ever land here. A gap in this category means a "
        "completed unit covers a range with no rows to show for it."
    ),
}


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconcile expected against actual rows, and explain every gap."
    )
    parser.add_argument("--grain", choices=("daily", "hourly"), default="daily")
    parser.add_argument("--start", type=dt.date.fromisoformat, default=None)
    parser.add_argument("--end", "--anchor", type=dt.date.fromisoformat, default=None)
    parser.add_argument("--min-gap-days", type=int, default=MIN_REPORTED_GAP_DAYS)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--write",
        type=Path,
        default=None,
        metavar="DIR",
        help="write the markdown artefact into this directory",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    engine = engine_from_settings(settings)
    try:
        report = reconcile(
            engine,
            grain=args.grain,
            start=args.start,
            end=args.end,
            manifest=Manifest(args.manifest or settings.ingest_manifest_path),
            min_gap_days=args.min_gap_days,
            settings=settings,
        )
    finally:
        engine.dispose()

    markdown = to_markdown(report, ACCEPTED)
    if args.write:
        args.write.mkdir(parents=True, exist_ok=True)
        path = args.write / f"ingestion-reconciliation-{args.grain}.md"
        path.write_text(markdown + "\n", encoding="utf-8")
        print(f"wrote {path}")
    else:
        print(markdown)

    return 1 if report.unexplained else 0


if __name__ == "__main__":
    raise SystemExit(_main())
