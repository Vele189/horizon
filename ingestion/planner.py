"""Decomposes the backfill into resumable city × window work units.

Three things live here: the arithmetic that turns "fifteen cities, thirty
years" into a deterministic list of requests, an append-only manifest recording
which of those have landed, and the pacing needed to stay inside Open-Meteo's
budget. ING-04 executes the queue; this module only decides what the queue is.

**Open-Meteo does not meter HTTP requests, it meters weighted API calls.** The
documented rule is that a request costs ``(variables / 10) x (days / 14)``
calls, each factor floored at 1. Measured against the live API on 2026-09-07:
five HTTP requests for London — one, two, five, ten, and thirty years of daily
data — were refused on the fifth with

    HTTP 429 ... Minutely API request limit exceeded.

Five requests is nowhere near the documented 600 calls/min. Their *weighted*
cost is 55 + 110 + 274 + 548 = 987 by the fourth, which crosses 600 exactly
where the refusal landed. The formula is therefore the one to plan against, and
:func:`api_call_weight` implements it.

Two consequences shape everything below.

*   **Chunk size is quota-neutral above fourteen days.** Weight is proportional
    to days, so ten one-year units cost the same as one ten-year unit. Below
    fourteen days the floor makes short units cost a full call each, so smaller
    is never cheaper — it only buys finer resume granularity and smaller
    payloads. That frees the default to be chosen for legibility: one calendar
    year, which measures 47 KiB daily and 732 KiB hourly.
*   **The full backfill does not fit in one day of free quota.** Thirty-one
    years of daily observations across fifteen cities is roughly 26 000
    weighted calls against an allowance of 10 000 per day. This is not a
    problem to engineer around — it is the reason the manifest exists. Run
    until the budget is spent, stop, resume tomorrow.

Usage::

    from ingestion.planner import Manifest, plan_backfill

    plan = plan_backfill()
    for unit in plan.units:
        ...

Run ``python ingestion/planner.py`` for the plan, or ``--progress`` for what
the manifest says has already landed.
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Iterable, Iterator, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cities import City, CityRegistry, load_cities  # noqa: E402
from config import Settings, get_settings  # noqa: E402
from ingestion.client import (  # noqa: E402
    ARCHIVE_START,
    DAILY_VARIABLES,
    HOURLY_VARIABLES,
    Grain,
)

__all__ = [
    "BACKFILL_START",
    "DAILY_QUOTA_CALLS",
    "HOURLY_QUOTA_CALLS",
    "HOURLY_BACKFILL_MONTHS",
    "Manifest",
    "ManifestEntry",
    "Plan",
    "WorkUnit",
    "api_call_weight",
    "archive_end_date",
    "delay_seconds_for",
    "plan_backfill",
    "windows",
]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The shape of the backfill (§5.1 of docs/proposal.md)
# ---------------------------------------------------------------------------

#: Daily observations reach back thirty years, which is what a climatological
#: normal requires. The draft's ten years could not have produced one.
BACKFILL_START: Final[dt.date] = dt.date(1995, 1, 1)

#: Hourly observations are only needed for the storm-dynamics view, so they
#: trail the present rather than reaching back.
HOURLY_BACKFILL_MONTHS: Final[int] = 24

#: The archive trails the present. Open-Meteo states its exact cut-off in the
#: 400 it returns for a range beyond it — on 2026-09-07 that was yesterday —
#: but ERA5 final data lags further, so plans stop short of the edge rather
#: than planning units that may 400 on the day they run.
ARCHIVE_LAG_DAYS: Final[int] = 5

# ---------------------------------------------------------------------------
# Open-Meteo's billing model
# ---------------------------------------------------------------------------

#: A request covering more days than this is charged proportionally.
WEIGHT_FREE_DAYS: Final[int] = 14

#: A request naming more variables than this is charged proportionally.
WEIGHT_FREE_VARIABLES: Final[int] = 10

#: Free-tier allowances, from https://open-meteo.com/en/pricing.
MINUTELY_QUOTA_CALLS: Final[float] = 600.0
HOURLY_QUOTA_CALLS: Final[float] = 5_000.0
DAILY_QUOTA_CALLS: Final[float] = 10_000.0

#: Plan against a fraction of each allowance. The budget is shared with
#: anything else on the same address, the weight formula is documented rather
#: than guaranteed, and the cost of being wrong is a 429 — far more than the
#: seconds this headroom costs.
MINUTELY_SAFETY_FACTOR: Final[float] = 0.5
HOURLY_SAFETY_FACTOR: Final[float] = 0.9

#: Bytes per row, measured across daily and hourly responses on 2026-09-07
#: (130 B/row daily at 21 variables, 85 B/row hourly at 12). Used only to
#: estimate how much a plan will pull; nothing depends on it being exact.
BYTES_PER_ROW: Final[dict[str, int]] = {"daily": 130, "hourly": 85}

_VARIABLE_COUNT: Final[dict[str, int]] = {
    "daily": len(DAILY_VARIABLES),
    "hourly": len(HOURLY_VARIABLES),
}

_ROWS_PER_DAY: Final[dict[str, int]] = {"daily": 1, "hourly": 24}

MANIFEST_FIELDS: Final[tuple[str, ...]] = (
    "city_id",
    "grain",
    "start",
    "end",
    "rows",
    "weight",
    "completed_at",
)


def api_call_weight(days: int, variables: int) -> float:
    """What Open-Meteo charges for one request, in API calls.

    ``(variables / 10) x (days / 14)``, each factor floored at 1. The floors
    are why a plan of very short windows is more expensive than the same range
    in fortnightly ones, and why there is no saving in going below a fortnight.
    """
    if days < 1 or variables < 1:
        raise ValueError(f"days and variables must be positive, got {days}, {variables}")
    return max(1.0, variables / WEIGHT_FREE_VARIABLES) * max(
        1.0, days / WEIGHT_FREE_DAYS
    )


def delay_seconds_for(weight: float, floor: float) -> float:
    """Seconds to wait after a request of this weight.

    Paces against weighted calls rather than request count — a fixed
    one-second delay between one-year daily units would spend 3 300 calls a
    minute against a 600 budget — and against **both** short allowances, not
    just the minutely one.

    Pacing on the minutely limit alone is what the first real backfill run got
    wrong. Half of 600 calls a minute is 18 000 an hour, against an hourly
    allowance of 5 000: the run cleared the minutely bar on every request and
    still collected

        HTTP 429 ... Hourly API request limit exceeded

    nineteen minutes in, having spent 5 059 calls. The hourly limit is the
    binding one at any pace worth using, and it works out four times slower —
    roughly 44 seconds between one-year daily units rather than 11.

    ``floor`` is the configured politeness minimum, applied even when a unit is
    small enough to need no pacing at all.
    """
    per_minute = 60.0 * weight / (MINUTELY_QUOTA_CALLS * MINUTELY_SAFETY_FACTOR)
    per_hour = 3600.0 * weight / (HOURLY_QUOTA_CALLS * HOURLY_SAFETY_FACTOR)
    return max(floor, per_minute, per_hour)


def archive_end_date(today: dt.date | None = None) -> dt.date:
    """The last day worth asking for."""
    today = today or dt.datetime.now(dt.timezone.utc).date()
    return today - dt.timedelta(days=ARCHIVE_LAG_DAYS)


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


def _add_months(date: dt.date, months: int) -> dt.date:
    """Shift by whole months, clamping the day to the target month's length."""
    total = date.month - 1 + months
    year = date.year + total // 12
    month = total % 12 + 1
    return dt.date(year, month, min(date.day, calendar.monthrange(year, month)[1]))


def windows(
    start: dt.date, end: dt.date, chunk_months: int
) -> Iterator[tuple[dt.date, dt.date]]:
    """Split ``[start, end]`` into consecutive inclusive windows.

    Walks forward in whole months from ``start``, so a January start with
    ``chunk_months=12`` yields exact calendar years and the window boundaries
    do not drift as the run date changes. The final window is clipped to
    ``end`` and may be shorter than the rest.
    """
    if chunk_months < 1:
        raise ValueError(f"chunk_months must be at least 1, got {chunk_months}.")
    if end < start:
        raise ValueError(f"end {end} precedes start {start}.")

    cursor = start
    while cursor <= end:
        nxt = _add_months(cursor, chunk_months)
        yield cursor, min(nxt - dt.timedelta(days=1), end)
        cursor = nxt


# ---------------------------------------------------------------------------
# Work units
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class WorkUnit:
    """One request: one city, one grain, one inclusive date window."""

    city_id: str
    grain: Grain
    start: dt.date
    end: dt.date

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(
                f"{self.city_id}: end {self.end} precedes start {self.start}."
            )

    @property
    def key(self) -> str:
        """A stable identifier, readable in a log line and in the manifest."""
        return f"{self.city_id}/{self.grain}/{self.start}/{self.end}"

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    @property
    def expected_rows(self) -> int:
        return self.days * _ROWS_PER_DAY[self.grain]

    @property
    def weight(self) -> float:
        """Weighted API calls this unit will cost."""
        return api_call_weight(self.days, _VARIABLE_COUNT[self.grain])

    @property
    def estimated_bytes(self) -> int:
        return self.expected_rows * BYTES_PER_ROW[self.grain]


@dataclass(frozen=True)
class ManifestEntry:
    """One completed unit, as recorded on disk."""

    city_id: str
    grain: str
    start: dt.date
    end: dt.date
    rows: int
    weight: float
    completed_at: dt.datetime

    def as_json(self) -> str:
        return json.dumps(
            {
                "city_id": self.city_id,
                "grain": self.grain,
                "start": self.start.isoformat(),
                "end": self.end.isoformat(),
                "rows": self.rows,
                "weight": round(self.weight, 2),
                "completed_at": self.completed_at.isoformat(),
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, line: str) -> "ManifestEntry":
        raw = json.loads(line)
        missing = [f for f in MANIFEST_FIELDS if f not in raw]
        if missing:
            raise ValueError(f"missing field(s) {missing}")
        return cls(
            city_id=str(raw["city_id"]),
            grain=str(raw["grain"]),
            start=dt.date.fromisoformat(raw["start"]),
            end=dt.date.fromisoformat(raw["end"]),
            rows=int(raw["rows"]),
            weight=float(raw["weight"]),
            completed_at=dt.datetime.fromisoformat(raw["completed_at"]),
        )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _merge(spans: Iterable[tuple[dt.date, dt.date]]) -> list[tuple[dt.date, dt.date]]:
    """Merge date spans, joining those that touch or overlap."""
    merged: list[tuple[dt.date, dt.date]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1] + dt.timedelta(days=1):
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


class Manifest:
    """Append-only JSONL record of which units have landed.

    Append-only rather than a rewritten document because the failure mode being
    designed against is the run dying mid-write. A truncated final line loses
    one unit, which is re-fetched; a truncated rewrite loses the whole history.
    Each record is flushed and fsynced before the next unit starts, so a record
    on disk means the rows really are in the warehouse.

    Completion is tested by **date coverage**, not by matching the window key.
    A unit is skipped when its whole range is already covered for that city and
    grain, however that coverage was assembled. This is what lets the chunk
    size change between sessions — the natural response to a 429 or a timeout
    is to halve it, and re-fetching everything landed so far would be a harsh
    price for that.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._entries: list[ManifestEntry] | None = None
        self._coverage: dict[tuple[str, str], list[tuple[dt.date, dt.date]]] = {}

    # -- reading ----------------------------------------------------------

    def _load(self) -> list[ManifestEntry]:
        if self._entries is not None:
            return self._entries

        entries: list[ManifestEntry] = []
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(ManifestEntry.from_json(line))
                    except (ValueError, json.JSONDecodeError) as exc:
                        # A partial last line is what a killed run leaves
                        # behind. Losing one unit to a re-fetch is the correct
                        # price; refusing to start is not.
                        log.warning(
                            "%s line %d is unreadable and will be ignored (%s); "
                            "its unit will be re-fetched",
                            self.path,
                            number,
                            exc,
                        )

        self._entries = entries
        self._coverage = {}
        for entry in entries:
            self._coverage.setdefault((entry.city_id, entry.grain), []).append(
                (entry.start, entry.end)
            )
        for key, spans in self._coverage.items():
            self._coverage[key] = _merge(spans)
        return entries

    @property
    def entries(self) -> tuple[ManifestEntry, ...]:
        return tuple(self._load())

    def __len__(self) -> int:
        return len(self._load())

    def coverage(self, city_id: str, grain: str) -> tuple[tuple[dt.date, dt.date], ...]:
        """The merged date spans already landed for one city and grain."""
        self._load()
        return tuple(self._coverage.get((city_id, grain), ()))

    def is_complete(self, unit: WorkUnit) -> bool:
        """Is every day of this unit already covered?"""
        return any(
            span_start <= unit.start and unit.end <= span_end
            for span_start, span_end in self.coverage(unit.city_id, unit.grain)
        )

    def completed_rows(self) -> int:
        return sum(entry.rows for entry in self._load())

    def completed_weight(self) -> float:
        return sum(entry.weight for entry in self._load())

    # -- writing ----------------------------------------------------------

    def record(
        self,
        unit: WorkUnit,
        *,
        rows: int,
        completed_at: dt.datetime | None = None,
    ) -> ManifestEntry:
        """Mark a unit landed. Call this only after the rows are committed.

        Recording before the warehouse write commits would turn a crash into
        silent data loss: the next run would skip a window whose rows are not
        there, and nothing downstream would report a hole.
        """
        entry = ManifestEntry(
            city_id=unit.city_id,
            grain=unit.grain,
            start=unit.start,
            end=unit.end,
            rows=rows,
            weight=unit.weight,
            completed_at=completed_at or dt.datetime.now(dt.timezone.utc),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(entry.as_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        if self._entries is not None:
            self._entries.append(entry)
            key = (entry.city_id, entry.grain)
            self._coverage[key] = _merge(
                [*self._coverage.get(key, []), (entry.start, entry.end)]
            )
        return entry

    def compact(self) -> int:
        """Rewrite the manifest with unreadable lines dropped and spans merged.

        Purely cosmetic — nothing depends on it. Useful after a long backfill
        has accumulated hundreds of one-year records. Writes to a temporary
        file in the same directory and renames over the original, so an
        interruption leaves the previous manifest intact.
        """
        entries = self._load()
        merged: list[ManifestEntry] = []
        by_key: dict[tuple[str, str], list[ManifestEntry]] = {}
        for entry in entries:
            by_key.setdefault((entry.city_id, entry.grain), []).append(entry)

        for (city_id, grain), group in sorted(by_key.items()):
            for start, end in _merge((e.start, e.end) for e in group):
                covered = [e for e in group if start <= e.start and e.end <= end]
                merged.append(
                    ManifestEntry(
                        city_id=city_id,
                        grain=grain,
                        start=start,
                        end=end,
                        rows=sum(e.rows for e in covered),
                        weight=sum(e.weight for e in covered),
                        completed_at=max(e.completed_at for e in covered),
                    )
                )

        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.path.parent, delete=False
        )
        try:
            with handle:
                for entry in merged:
                    handle.write(entry.as_json() + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(handle.name, self.path)
        except BaseException:
            Path(handle.name).unlink(missing_ok=True)
            raise

        self._entries = None
        self._load()
        return len(entries) - len(merged)


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    """What a run should do, and what it will cost."""

    units: tuple[WorkUnit, ...]
    skipped: tuple[WorkUnit, ...]
    chunk_months: int
    delay_floor_seconds: float
    generated_at: dt.datetime = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc)
    )

    def __len__(self) -> int:
        return len(self.units)

    def __iter__(self) -> Iterator[WorkUnit]:
        return iter(self.units)

    @property
    def total_units(self) -> int:
        return len(self.units) + len(self.skipped)

    @property
    def weight(self) -> float:
        """Weighted API calls the pending units will cost."""
        return sum(unit.weight for unit in self.units)

    @property
    def estimated_bytes(self) -> int:
        return sum(unit.estimated_bytes for unit in self.units)

    @property
    def expected_rows(self) -> int:
        return sum(unit.expected_rows for unit in self.units)

    @property
    def quota_days(self) -> float:
        """Days of free-tier allowance this plan needs. Frequently above one."""
        return self.weight / DAILY_QUOTA_CALLS

    def delay_for(self, unit: WorkUnit) -> float:
        return delay_seconds_for(unit.weight, self.delay_floor_seconds)

    @property
    def estimated_seconds(self) -> float:
        """Wall time if nothing is refused, dominated by the pacing delays."""
        return sum(self.delay_for(unit) for unit in self.units)

    def within_daily_quota(self) -> tuple[WorkUnit, ...]:
        """The prefix of the plan that fits in one day's allowance.

        A run that spends its whole allowance and then keeps going earns a 429
        per remaining unit. Stopping at the budget instead leaves the manifest
        in a clean state and tomorrow's session picks up exactly here.
        """
        taken: list[WorkUnit] = []
        spent = 0.0
        for unit in self.units:
            if spent + unit.weight > DAILY_QUOTA_CALLS:
                break
            taken.append(unit)
            spent += unit.weight
        return tuple(taken)


def _units_for(
    city: City, grain: Grain, start: dt.date, end: dt.date, chunk_months: int
) -> list[WorkUnit]:
    if end < start:
        return []
    return [
        WorkUnit(city_id=city.id, grain=grain, start=window_start, end=window_end)
        for window_start, window_end in windows(start, end, chunk_months)
    ]


def plan_backfill(
    *,
    grains: Sequence[Grain] = ("daily", "hourly"),
    cities: Iterable[City] | CityRegistry | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
    chunk_months: int | None = None,
    manifest: Manifest | None = None,
    settings: Settings | None = None,
) -> Plan:
    """Build the deterministic work queue, minus whatever has already landed.

    Args:
        grains: Which grains to plan. Both by default.
        cities: Defaults to every city in ``config/cities.yml``, in file order.
        start: First day for the daily grain. Defaults to
            :data:`BACKFILL_START`. Hourly always trails the end by
            :data:`HOURLY_BACKFILL_MONTHS` regardless, since a thirty-year
            hourly pull is neither needed nor affordable.
        end: Last day. Defaults to :func:`archive_end_date`.
        chunk_months: Months per unit. Defaults to ``INGEST_CHUNK_MONTHS``.
        manifest: Completed units are excluded. Defaults to the manifest at
            ``INGEST_MANIFEST_PATH``.
        settings: Override the process configuration. Intended for tests.

    Returns:
        A :class:`Plan` whose ``units`` are pending and ``skipped`` are those
        the manifest already covers.

    The order is city-major and then chronological, matching ``cities.yml``, so
    two runs of the same inputs produce byte-identical plans and a city becomes
    complete — and therefore usable by the Day 8 validation gate — as early as
    possible rather than every city finishing at once at the end.
    """
    resolved_settings = settings if settings is not None else get_settings()
    registry = cities if cities is not None else load_cities()
    resolved_chunk = (
        chunk_months
        if chunk_months is not None
        else resolved_settings.ingest_chunk_months
    )
    resolved_manifest = (
        manifest
        if manifest is not None
        else Manifest(resolved_settings.ingest_manifest_path)
    )

    unknown = [grain for grain in grains if grain not in _VARIABLE_COUNT]
    if unknown:
        raise ValueError(f"unknown grain(s) {unknown}; expected 'daily' or 'hourly'.")

    resolved_end = end if end is not None else archive_end_date()
    daily_start = start if start is not None else BACKFILL_START
    if daily_start < ARCHIVE_START:
        raise ValueError(
            f"start {daily_start} precedes the ERA5 archive ({ARCHIVE_START})."
        )
    hourly_start = max(
        daily_start, _add_months(resolved_end, -HOURLY_BACKFILL_MONTHS)
    )

    pending: list[WorkUnit] = []
    skipped: list[WorkUnit] = []
    for city in registry:
        for grain in grains:
            grain_start = daily_start if grain == "daily" else hourly_start
            for unit in _units_for(
                city, grain, grain_start, resolved_end, resolved_chunk
            ):
                if resolved_manifest.is_complete(unit):
                    skipped.append(unit)
                else:
                    pending.append(unit)

    return Plan(
        units=tuple(pending),
        skipped=tuple(skipped),
        chunk_months=resolved_chunk,
        delay_floor_seconds=resolved_settings.request_delay_seconds,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _humanise(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _mib(count: int) -> str:
    return f"{count / 1024 / 1024:.1f} MiB"


def _print_progress(manifest: Manifest, registry: CityRegistry) -> None:
    print(f"Manifest: {manifest.path}")
    if not len(manifest):
        print("  empty — nothing has landed yet")
        return
    print(
        f"  {len(manifest)} records, {manifest.completed_rows():,} rows, "
        f"{manifest.completed_weight():,.0f} weighted calls spent\n"
    )
    width = max(len(city.id) for city in registry)
    print(f"  {'city'.ljust(width)}  {'daily coverage':27}  hourly coverage")
    print("  " + "-" * (width + 50))
    for city in registry:
        cells = []
        for grain in ("daily", "hourly"):
            spans = manifest.coverage(city.id, grain)
            cells.append(
                ", ".join(f"{s}..{e}" for s, e in spans) if spans else "—"
            )
        print(f"  {city.id.ljust(width)}  {cells[0]:27}  {cells[1]}")


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan the backfill as resumable city-window work units."
    )
    parser.add_argument("--grain", choices=("daily", "hourly", "both"), default="both")
    parser.add_argument("--city", action="append", dest="cities", metavar="CITY_ID")
    parser.add_argument("--chunk-months", type=int, default=None)
    parser.add_argument("--start", type=dt.date.fromisoformat, default=None)
    parser.add_argument("--end", type=dt.date.fromisoformat, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--progress", action="store_true", help="show what has landed, then exit"
    )
    parser.add_argument(
        "--list", action="store_true", help="print every pending unit key"
    )
    parser.add_argument(
        "--compact", action="store_true", help="merge manifest records, then exit"
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level, format="%(levelname)-8s %(name)s  %(message)s"
    )
    registry = load_cities()
    manifest = Manifest(args.manifest or settings.ingest_manifest_path)

    if args.compact:
        removed = manifest.compact()
        print(f"compacted {manifest.path}: {removed} record(s) merged away")
        return 0

    selected = registry
    if args.cities:
        try:
            selected = [registry[city_id] for city_id in args.cities]
        except Exception as exc:  # CityConfigError carries the known ids
            print(f"FAILED  {exc}", file=sys.stderr)
            return 1

    if args.progress:
        _print_progress(manifest, registry)
        return 0

    grains: tuple[Grain, ...] = (
        ("daily", "hourly") if args.grain == "both" else (args.grain,)
    )
    try:
        plan = plan_backfill(
            grains=grains,
            cities=selected,
            start=args.start,
            end=args.end,
            chunk_months=args.chunk_months,
            manifest=manifest,
            settings=settings,
        )
    except ValueError as exc:
        print(f"FAILED  {exc}", file=sys.stderr)
        return 1

    cities_planned = len({unit.city_id for unit in (*plan.units, *plan.skipped)})
    print(
        f"{plan.total_units} work units across {cities_planned} cities "
        f"at {plan.chunk_months} month(s) each\n"
    )
    print(f"  already landed  {len(plan.skipped)}")
    print(f"  pending         {len(plan.units)}")
    if not plan.units:
        print("\n  nothing to do — the manifest covers every planned window.")
        return 0

    for grain in ("daily", "hourly"):
        units = [u for u in plan.units if u.grain == grain]
        if units:
            weight = sum(u.weight for u in units)
            rows = sum(u.expected_rows for u in units)
            print(
                f"  {grain:7}       {len(units):5} units  {rows:>10,} rows  "
                f"{weight:>9,.0f} calls"
            )

    within = plan.within_daily_quota()
    print(f"\n  expected rows   {plan.expected_rows:,}")
    print(f"  download        ~{_mib(plan.estimated_bytes)}")
    print(f"  weighted cost   {plan.weight:,.0f} API calls")
    print(
        f"  free-tier days  {plan.quota_days:.1f} "
        f"(10,000 calls/day; this run can do {len(within)} of "
        f"{len(plan.units)} units)"
    )
    print(f"  paced runtime   ~{_humanise(plan.estimated_seconds)} of request delay")

    if plan.quota_days > 1:
        print(
            f"\n  This plan exceeds one day of free quota. Run it, let it stop at "
            f"the budget,\n  and resume tomorrow — the manifest at {manifest.path}\n"
            "  makes that free."
        )

    if args.list:
        print()
        for unit in plan.units:
            print(
                f"  {unit.key:44} {unit.expected_rows:>6} rows  "
                f"{unit.weight:7.1f} calls  wait {plan.delay_for(unit):5.1f}s"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
