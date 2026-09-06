"""Gzipped raw payloads on disk — the "raw" in bronze, without the storage bill.

One file per work unit, at ``data/raw/{grain}/{city_id}/{start}_{end}.json.gz``.
What lands there is the response body exactly as it arrived: not re-serialised,
not reordered, not validated. Nothing this project wrote has touched it.

The point is a specific failure. A parsing bug found on day seven — a unit
misread, a timestamp off by an hour, a column mapped to the wrong variable —
costs a re-run of the transformation if the payloads are on disk, and a
2.7-day re-pull of the whole archive if they are not. So the write happens
*before* the parse, through :func:`ingestion.client.fetch_observations`'s
``on_payload`` hook, and the parser that replay feeds is the same one the live
path uses: fixing a bug fixes replay by construction.

Payloads are not stored in Postgres. A per-row jsonb column would multiply the
warehouse footprint several times over for data already parsed into the columns
beside it, and would be fatal to Neon's 0.5 GB allowance (§5.4). ``data/`` is
git-ignored.

Usage::

    from ingestion import archive

    fetch_observations(..., on_payload=archive.writer_for(unit))   # ING-04
    for response in archive.replay():                              # rebuild
        ...

Run ``python ingestion/archive.py --stats`` for what is on disk, or
``--verify`` to re-parse every archived payload without touching the network.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Iterable, Iterator, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Settings, get_settings  # noqa: E402
from ingestion.client import (  # noqa: E402
    ArchiveResponse,
    ArchiveResponseError,
    Grain,
    parse_payload,
    request_url,
)
from ingestion.planner import WorkUnit  # noqa: E402

__all__ = [
    "ArchiveFileError",
    "ArchiveStats",
    "GrainStats",
    "SUFFIX",
    "archive_path",
    "archived_units",
    "exists",
    "read_bytes",
    "read_payload",
    "replay",
    "replay_unit",
    "stats",
    "write",
    "writer_for",
]

log = logging.getLogger(__name__)

SUFFIX: Final[str] = ".json.gz"

#: gzip's default. Level 9 was measured at under 1% better on these payloads
#: for roughly three times the CPU, which is a poor trade across 525 units.
COMPRESS_LEVEL: Final[int] = 6

_GRAINS: Final[tuple[str, ...]] = ("daily", "hourly")
_WINDOW_RE: Final[re.Pattern[str]] = re.compile(
    r"^(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})$"
)


def _root(root: Path | str | None, settings: Settings | None = None) -> Path:
    if root is not None:
        return Path(root)
    return (settings or get_settings()).data_raw_dir


def archive_path(
    unit: WorkUnit, root: Path | str | None = None, *, settings: Settings | None = None
) -> Path:
    """Where one unit's payload lives.

    The path carries the whole identity of the unit — grain, city, window — so
    the directory tree is its own index. Nothing needs a sidecar file to know
    what is on disk, and a human can find one city's 1998 by looking.
    """
    return (
        _root(root, settings)
        / unit.grain
        / unit.city_id
        / f"{unit.start.isoformat()}_{unit.end.isoformat()}{SUFFIX}"
    )


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write(
    unit: WorkUnit,
    raw: bytes,
    root: Path | str | None = None,
    *,
    settings: Settings | None = None,
) -> Path:
    """Gzip ``raw`` to this unit's path, atomically. Returns the path written.

    Written to a temporary file in the destination directory, fsynced, then
    renamed over the target. A rename within one filesystem is atomic, so a
    reader never sees a half-written archive and a crash mid-write leaves the
    previous file — or no file — rather than a corrupt one.

    ``mtime=0`` and an empty ``filename`` keep the output byte-identical for
    identical input, which is what makes re-archiving a unit detectable as a
    no-op rather than as a change. Both are needed: left to itself ``GzipFile``
    stamps the header with the current time and with ``fileobj.name``, and the
    latter is the randomised temporary file this writes through.
    """
    path = archive_path(unit, root, settings=settings)
    path.parent.mkdir(parents=True, exist_ok=True)

    handle = tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    )
    try:
        with handle:
            with gzip.GzipFile(
                filename="",
                fileobj=handle,
                mode="wb",
                compresslevel=COMPRESS_LEVEL,
                mtime=0,
            ) as compressed:
                compressed.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise

    log.debug("archived %s (%d bytes raw)", path, len(raw))
    return path


def writer_for(
    unit: WorkUnit, root: Path | str | None = None, *, settings: Settings | None = None
) -> Callable[[bytes, str], None]:
    """An ``on_payload`` hook that archives this unit.

    The URL argument is ignored: it is reproduced exactly by
    :func:`~ingestion.client.request_url` from the unit alone, so storing it
    would be storing a derived value.
    """

    def archive(raw: bytes, url: str) -> None:  # noqa: ARG001 - hook signature
        write(unit, raw, root, settings=settings)

    return archive


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


class ArchiveFileError(RuntimeError):
    """An archived file exists but cannot be read back."""


def exists(
    unit: WorkUnit, root: Path | str | None = None, *, settings: Settings | None = None
) -> bool:
    return archive_path(unit, root, settings=settings).is_file()


def read_bytes(
    unit: WorkUnit, root: Path | str | None = None, *, settings: Settings | None = None
) -> bytes:
    """The payload as it arrived, decompressed."""
    path = archive_path(unit, root, settings=settings)
    try:
        with gzip.open(path, "rb") as handle:
            return handle.read()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"no archived payload at {path}") from exc
    except (OSError, EOFError) as exc:
        raise ArchiveFileError(f"{path} is not readable gzip: {exc}") from exc


def read_payload(
    unit: WorkUnit, root: Path | str | None = None, *, settings: Settings | None = None
) -> dict[str, Any]:
    raw = read_bytes(unit, root, settings=settings)
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        path = archive_path(unit, root, settings=settings)
        raise ArchiveFileError(f"{path} does not contain JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArchiveFileError(
            f"{archive_path(unit, root, settings=settings)} contains "
            f"{type(payload).__name__}, expected a JSON object."
        )
    return payload


def _unit_from_path(path: Path, root: Path) -> WorkUnit | None:
    """Recover the work unit a path encodes, or None if it encodes none."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    if len(relative.parts) != 3 or not path.name.endswith(SUFFIX):
        return None

    grain, city_id, filename = relative.parts
    if grain not in _GRAINS:
        return None
    match = _WINDOW_RE.match(filename[: -len(SUFFIX)])
    if not match:
        return None
    try:
        start = dt.date.fromisoformat(match.group(1))
        end = dt.date.fromisoformat(match.group(2))
        return WorkUnit(city_id=city_id, grain=grain, start=start, end=end)
    except ValueError:
        return None


def archived_units(
    root: Path | str | None = None,
    *,
    grain: Grain | None = None,
    city_id: str | None = None,
    settings: Settings | None = None,
) -> Iterator[WorkUnit]:
    """Every unit on disk, in deterministic order.

    Discovery walks the tree rather than consulting the manifest, so the
    archive stands on its own: a manifest deleted to force a re-run does not
    make the payloads already on disk unfindable.
    """
    base = _root(root, settings)
    if not base.is_dir():
        return

    units: list[WorkUnit] = []
    for path in base.rglob(f"*{SUFFIX}"):
        unit = _unit_from_path(path, base)
        if unit is None:
            log.warning("ignoring unrecognised archive file %s", path)
            continue
        if grain is not None and unit.grain != grain:
            continue
        if city_id is not None and unit.city_id != city_id:
            continue
        units.append(unit)

    yield from sorted(units, key=lambda u: (u.city_id, u.grain, u.start, u.end))


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def replay_unit(
    unit: WorkUnit,
    root: Path | str | None = None,
    *,
    settings: Settings | None = None,
) -> ArchiveResponse:
    """Rebuild one unit's parsed response from disk. No network.

    Runs the archived payload through the same :func:`_validate` the live path
    uses, so a fix to the parser applies to replay without being applied twice.
    The ``source_url`` is reproduced by :func:`~ingestion.client.request_url`,
    which prepares the identical string the request carried — the one thing
    here that reads ``config/cities.yml``, because the requested coordinates
    are not recoverable from a response that reports the snapped grid cell.
    """
    payload = read_payload(unit, root, settings=settings)
    url = request_url(unit.city_id, unit.start, unit.end, unit.grain, settings=settings)
    return parse_payload(
        payload,
        city_id=unit.city_id,
        grain=unit.grain,
        start=unit.start,
        end=unit.end,
        url=url,
    )


def replay(
    root: Path | str | None = None,
    *,
    units: Iterable[WorkUnit] | None = None,
    grain: Grain | None = None,
    city_id: str | None = None,
    settings: Settings | None = None,
) -> Iterator[ArchiveResponse]:
    """Rebuild bronze from the archive, in full, with no network calls.

    Yields lazily: the full archive is hundreds of megabytes decompressed and
    materialising it as a list would be a needless way to run out of memory.

    Args:
        root: Archive root. Defaults to ``DATA_RAW_DIR``. First and positional
            because "replay everything on disk" is the call that matters;
            ``units`` is keyword-only so the two can never be swapped.
        units: Which units to replay. Defaults to everything on disk.
        grain, city_id: Narrow the default discovery.
    """
    selected = (
        archived_units(root, grain=grain, city_id=city_id, settings=settings)
        if units is None
        else units
    )
    for unit in selected:
        yield replay_unit(unit, root, settings=settings)


# ---------------------------------------------------------------------------
# Size
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GrainStats:
    """One grain's share of the archive."""

    files: int
    compressed_bytes: int
    rows: int

    @property
    def bytes_per_row(self) -> float:
        return self.compressed_bytes / self.rows if self.rows else 0.0


@dataclass(frozen=True)
class ArchiveStats:
    """What is on disk, and what it cost."""

    by_grain: dict[str, GrainStats]

    @property
    def files(self) -> int:
        return sum(g.files for g in self.by_grain.values())

    @property
    def compressed_bytes(self) -> int:
        return sum(g.compressed_bytes for g in self.by_grain.values())

    @property
    def rows(self) -> int:
        return sum(g.rows for g in self.by_grain.values())

    def project(self, rows_by_grain: Mapping[str, int]) -> int:
        """Compressed size of an archive holding this many rows of each grain.

        Per grain, not blended. A daily row carries 21 variables and a hourly
        row 12, and they compress at roughly 34 and 15 bytes per row — so a
        single average rate applied to a row count with a different daily/hourly
        mix than the sample is simply the wrong number. Grains whose rate has
        not been measured contribute nothing rather than a guess.
        """
        return round(
            sum(
                self.by_grain[grain].bytes_per_row * rows
                for grain, rows in rows_by_grain.items()
                if grain in self.by_grain
            )
        )


def stats(
    root: Path | str | None = None,
    *,
    settings: Settings | None = None,
    count_rows: bool = True,
) -> ArchiveStats:
    """Measure the archive. ``count_rows=False`` skips decompressing it."""
    base = _root(root, settings)
    buckets: dict[str, list[int]] = {}

    for unit in archived_units(base, settings=settings):
        size = archive_path(unit, base, settings=settings).stat().st_size
        bucket = buckets.setdefault(unit.grain, [0, 0, 0])
        bucket[0] += 1
        bucket[1] += size
        if count_rows:
            bucket[2] += unit.expected_rows

    return ArchiveStats(
        by_grain={
            grain: GrainStats(files=n, compressed_bytes=size, rows=rows)
            for grain, (n, size, rows) in sorted(buckets.items())
        }
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _mib(count: float) -> str:
    return f"{count / 1024 / 1024:.1f} MiB"


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect, verify, and replay the raw payload archive."
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--grain", choices=_GRAINS, default=None)
    parser.add_argument("--city", dest="city_id", default=None)
    parser.add_argument("--stats", action="store_true", help="size what is on disk")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="re-parse every archived payload, no network",
    )
    parser.add_argument(
        "--list", action="store_true", help="print every archived unit key"
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level, format="%(levelname)-8s %(name)s  %(message)s"
    )
    root = args.root or settings.data_raw_dir

    if args.list:
        for unit in archived_units(
            root, grain=args.grain, city_id=args.city_id, settings=settings
        ):
            print(f"  {unit.key}")
        return 0

    if args.verify:
        checked = 0
        failed = 0
        for unit in archived_units(
            root, grain=args.grain, city_id=args.city_id, settings=settings
        ):
            checked += 1
            try:
                response = replay_unit(unit, root, settings=settings)
            except (ArchiveFileError, ArchiveResponseError, FileNotFoundError) as exc:
                failed += 1
                print(f"  FAIL  {unit.key}: {exc}")
                continue
            if len(response) != unit.expected_rows:
                failed += 1
                print(
                    f"  FAIL  {unit.key}: {len(response)} rows, expected "
                    f"{unit.expected_rows}"
                )
        print(f"\n{checked} archived unit(s) re-parsed, {failed} failed")
        return 1 if failed else 0

    measured = stats(root, settings=settings)
    print(f"Archive: {root}")
    if not measured.files:
        print("  empty — nothing archived yet")
        return 0

    print(
        f"  {measured.files} files, {_mib(measured.compressed_bytes)} on disk, "
        f"{measured.rows:,} rows"
    )
    for grain, grain_stats in measured.by_grain.items():
        print(
            f"    {grain:7} {grain_stats.files:5} files  "
            f"{_mib(grain_stats.compressed_bytes):>10}  "
            f"{grain_stats.rows:>9,} rows  "
            f"{grain_stats.bytes_per_row:5.1f} B/row"
        )

    if args.stats:
        from ingestion.planner import Manifest, plan_backfill

        # An empty manifest, so the projection covers the whole backfill rather
        # than only what is still pending.
        with tempfile.TemporaryDirectory() as tmp:
            plan = plan_backfill(
                manifest=Manifest(Path(tmp) / "none.jsonl"), settings=settings
            )
        rows_by_grain: dict[str, int] = {}
        for unit in plan.units:
            rows_by_grain[unit.grain] = (
                rows_by_grain.get(unit.grain, 0) + unit.expected_rows
            )
        unmeasured = sorted(set(rows_by_grain) - set(measured.by_grain))
        print(f"\n  Projected full archive ({plan.expected_rows:,} rows):")
        for grain, rows in sorted(rows_by_grain.items()):
            if grain in measured.by_grain:
                print(
                    f"    {grain:7} {rows:>9,} rows  "
                    f"{_mib(measured.project({grain: rows})):>10}"
                )
        print(f"    {'total':7} {'':>9}       "
              f"{_mib(measured.project(rows_by_grain)):>10}")
        if unmeasured:
            print(f"    (no sample yet for {unmeasured}; excluded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
