"""The pipeline, end to end: ingest → dbt build → predict → promote.

    python run_pipeline.py                      # daily incremental update
    python run_pipeline.py --mode backfill      # the full thirty-year run
    python run_pipeline.py --only dbt predict   # two stages, in order
    python run_pipeline.py --skip promote       # everything but Neon
    python run_pipeline.py --dry-run            # print the plan, run nothing

**It stops at the first failure, and that is the whole point.** A pipeline that
continues past a broken transform publishes yesterday's predictions as though
they were today's — and does it quietly, because every later stage still
succeeds. Promotion in particular would happily copy a stale mart to Neon and
report success. So a non-zero exit from any stage halts the run, and the runner
exits non-zero itself. There is no ``--keep-going``.

**Stages are subprocesses, not imports.** dbt has to be one regardless, and
running the others the same way means the pipeline executes exactly what a
person executes by hand — one code path, not two that can drift. It also keeps
each stage's own ``logging.basicConfig`` from fighting this file's.

The cost is that a stage cannot hand back its row counts, and the answer to
that is better than the thing it replaces: **this file counts the rows itself,
from the warehouse, before and after each stage.** An independent measurement
beats a self-report — a stage that fails to write anything and exits zero is
caught by a delta of nothing, where its own summary would have said "wrote 0
rows" and been believed.

Logging is structured, and there are two streams of it. Stdout gets a
human-readable line per event. ``logs/pipeline-<timestamp>.jsonl`` gets one
JSON object per event, with the stage, the counts, the duration and the exit
code — which is the form you want when the question is "what did the run do at
02:00 last Tuesday" rather than "what is it doing now".
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ConfigError, Settings, get_settings, mask_secret  # noqa: E402

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent

log = logging.getLogger("pipeline")

# How far back a daily run re-fetches. More than one day on purpose: the
# archive revises recent observations, and a window that only asked for
# yesterday would never pick up a correction to the day before.
DAILY_LOOKBACK_DAYS: Final[int] = 7

# Where each stage's rows land. The runner counts these before and after, so a
# stage that exits zero having written nothing is still visible.
BRONZE: Final[str] = "bronze_raw"
SILVER: Final[str] = "silver_staging"
GOLD: Final[str] = "gold_marts"


def readable(command: Sequence[str]) -> str:
    """A command as it would be typed, not as it was resolved.

    Every path inside the project is shown relative to it. Absolute paths make
    a log longer, harder to compare between runs, and — since a log is a file
    that gets pasted into issues and committed as an artefact — they publish
    the operator's home directory for no diagnostic gain.
    """
    parts = []
    for argument in command:
        try:
            parts.append(str(Path(argument).relative_to(PROJECT_ROOT)))
        except ValueError:
            parts.append(argument)
    return " ".join(parts)


def dbt_executable() -> Path:
    """The dbt that belongs to the interpreter running this pipeline.

    Falls back to the bare name only when there is no sibling binary, so an
    unusual installation still works and the common case stops depending on
    PATH ordering.
    """
    sibling = Path(sys.executable).parent / "dbt"
    return sibling if sibling.exists() else Path("dbt")


class StageFailed(RuntimeError):
    """A stage exited non-zero. The run stops here."""


@dataclass(frozen=True)
class Stage:
    """One step of the pipeline.

    Attributes:
        name: What ``--only`` and ``--skip`` call it.
        summary: One line, for the plan and the logs.
        schemas: Which schemas to count before and after. The count is the
            cheapest data-quality signal there is and the first thing anyone
            debugging this will want.
        target: ``local`` or ``serving`` — which database those counts come
            from. Only promotion reads the far side.
    """

    name: str
    summary: str
    schemas: tuple[str, ...]
    target: str = "local"

    def command(self, mode: str, settings: Settings) -> list[str]:
        raise NotImplementedError


@dataclass(frozen=True)
class Ingest(Stage):
    def command(self, mode: str, settings: Settings) -> list[str]:
        base = [sys.executable, "ingestion/backfill.py"]
        if mode == "daily":
            start = dt.date.today() - dt.timedelta(days=DAILY_LOOKBACK_DAYS)
            # Daily grain only. The hourly window is a rolling two years that
            # the backfill mode maintains; re-walking it every night would
            # spend the API budget re-fetching what is already on disk.
            return [*base, "--grain", "daily", "--start", start.isoformat()]
        return [*base, "--grain", "daily"]


@dataclass(frozen=True)
class IngestHourly(Stage):
    def command(self, mode: str, settings: Settings) -> list[str]:
        return [sys.executable, "ingestion/backfill.py", "--grain", "hourly"]


@dataclass(frozen=True)
class DbtBuild(Stage):
    def command(self, mode: str, settings: Settings) -> list[str]:
        # Through dbt_env.py, which derives dbt's host/user/password variables
        # from DATABASE_URL. Calling dbt directly would need a second set of
        # variables that could drift from the first.
        #
        # dbt is named by its full path beside the running interpreter, not as
        # a bare "dbt" off PATH. dbt_env.py execs the command, and a bare name
        # is resolved against whatever PATH happens to hold — which on this
        # machine found a stale binary in an unrelated tool's cache and failed
        # with a FileNotFoundError that said nothing about dbt. The pipeline
        # runs the dbt belonging to the environment it is itself running in.
        return [
            sys.executable, "dbt_analytics/dbt_env.py", "--",
            str(dbt_executable()), "build", "--project-dir", "dbt_analytics",
        ]


@dataclass(frozen=True)
class Predict(Stage):
    def command(self, mode: str, settings: Settings) -> list[str]:
        return [sys.executable, "machine_learning/predict.py"]


@dataclass(frozen=True)
class Promote(Stage):
    def command(self, mode: str, settings: Settings) -> list[str]:
        # --verify compares contents rather than only row counts. It costs one
        # extra scan per side and it is the difference between "the promotion
        # ran" and "the serving copy is the local one".
        return [sys.executable, "serving/promote.py", "--verify"]


STAGES: Final[tuple[Stage, ...]] = (
    Ingest(
        name="ingest",
        summary="fetch and land daily observations into bronze",
        schemas=(BRONZE,),
    ),
    IngestHourly(
        name="ingest-hourly",
        summary="fetch and land the rolling hourly window into bronze",
        schemas=(BRONZE,),
    ),
    DbtBuild(
        name="dbt",
        summary="build and test silver and gold",
        schemas=(SILVER, GOLD),
    ),
    Predict(
        name="predict",
        summary="score the horizon into fact_ml_predictions",
        schemas=(GOLD,),
    ),
    Promote(
        name="promote",
        summary="copy the gold marts to the serving database",
        schemas=(GOLD,),
        target="serving",
    ),
)

STAGE_NAMES: Final[tuple[str, ...]] = tuple(stage.name for stage in STAGES)

# The hourly window is a backfill concern. A nightly run that re-walked two
# years of hourly units would spend its API budget discovering it already has
# them.
MODES: Final[Mapping[str, tuple[str, ...]]] = {
    "daily": ("ingest", "dbt", "predict", "promote"),
    "backfill": STAGE_NAMES,
}


# ---------------------------------------------------------------------------
# Logging — OPS-02
# ---------------------------------------------------------------------------


class JsonLines(logging.Formatter):
    """One JSON object per record, for the file handler.

    Structured because the questions asked of a pipeline log are structured:
    which stage, how long, how many rows, what exit code. Grepping prose for
    those answers works until the first time it matters.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": dt.datetime.fromtimestamp(record.created, dt.timezone.utc).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        for key, value in getattr(record, "fields", {}).items():
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, sort_keys=True)


class Console(logging.Formatter):
    """A readable line, for a person watching the run."""

    def format(self, record: logging.LogRecord) -> str:
        stamp = dt.datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        fields = getattr(record, "fields", {})
        stage = fields.get("stage")
        prefix = f"{stamp}  {record.levelname:<7} "
        head = f"[{stage}] " if stage else ""
        extras = " ".join(
            f"{key}={value}"
            for key, value in fields.items()
            if key not in {"stage", "run_id"} and value is not None
        )
        return f"{prefix}{head}{record.getMessage()}" + (f"  {extras}" if extras else "")


def configure_logging(log_dir: Path, run_id: str, level: str = "INFO") -> Path:
    """Stdout for a person, a JSON-lines file for later. Returns the file path.

    ``logs/`` is git-ignored — a run's output is a record of one execution, not
    a fact about the repository. One sample lives in ``docs/`` on purpose, as
    an artefact rather than as an accident.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    destination = log_dir / f"pipeline-{run_id}.jsonl"

    root = logging.getLogger("pipeline")
    root.setLevel(level)
    root.handlers.clear()
    root.propagate = False

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(Console())
    root.addHandler(stream)

    handle = logging.FileHandler(destination, encoding="utf-8")
    handle.setFormatter(JsonLines())
    root.addHandler(handle)

    return destination


def event(level: int, message: str, **fields: Any) -> None:
    """Log once, to both streams, with the fields attached rather than formatted."""
    log.log(level, message, extra={"fields": fields})


# ---------------------------------------------------------------------------
# Row counts — the cheapest data-quality signal there is
# ---------------------------------------------------------------------------


def count_rows(schemas: Sequence[str], *, target: str, settings: Settings) -> dict[str, int]:
    """Live row counts per table for the given schemas.

    Counted rather than estimated: ``pg_class.reltuples`` is free but is a
    planner estimate that lags until the next analyze, and a pipeline log whose
    numbers are approximately right is one nobody can use to prove anything.
    The gold marts are tens of thousands of rows; the exact count is cheap.

    Returns an empty mapping when the target is not configured or not
    reachable. Counting is instrumentation and must never be the thing that
    fails a run.
    """
    from sqlalchemy import create_engine, text

    try:
        url = (
            settings.require_serving_database_url()
            if target == "serving"
            else settings.require_database_url()
        )
    except ConfigError as exc:
        event(logging.WARNING, "row counts unavailable", target=target, reason=str(exc))
        return {}

    counts: dict[str, int] = {}
    engine = None
    try:
        # Inside the guard, not before it. Building the engine can fail on a
        # malformed URL or a missing driver, and instrumentation that can kill
        # the run it is measuring is worse than no instrumentation.
        engine = create_engine(url, future=True)
        with engine.connect() as connection:
            tables = connection.execute(
                text(
                    "select table_schema, table_name from information_schema.tables "
                    "where table_schema = any(:schemas) and table_type = 'BASE TABLE' "
                    "order by 1, 2"
                ),
                {"schemas": list(schemas)},
            ).fetchall()
            for schema, table in tables:
                total = connection.execute(
                    text(f'select count(*) from "{schema}"."{table}"')
                ).scalar_one()
                counts[f"{schema}.{table}"] = int(total)
    except Exception as exc:  # noqa: BLE001 - instrumentation never fails a run
        event(logging.WARNING, "row counts unavailable", target=target,
              reason=f"{type(exc).__name__}: {exc}")
        return {}
    finally:
        if engine is not None:
            engine.dispose()
    return counts


def _delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    return {
        table: after.get(table, 0) - before.get(table, 0)
        for table in sorted(set(before) | set(after))
        if after.get(table, 0) != before.get(table, 0)
    }


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


@dataclass
class StageOutcome:
    name: str
    exit_code: int
    seconds: float
    rows_before: dict[str, int] = field(default_factory=dict)
    rows_after: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    @property
    def total_before(self) -> int:
        return sum(self.rows_before.values())

    @property
    def total_after(self) -> int:
        return sum(self.rows_after.values())


def run_stage(stage: Stage, mode: str, settings: Settings, *, dry_run: bool) -> StageOutcome:
    """Run one stage, counting rows either side of it.

    Raises:
        StageFailed: The stage exited non-zero. Nothing after it runs.
    """
    command = stage.command(mode, settings)
    printable = readable(command)

    if dry_run:
        event(logging.INFO, "would run", stage=stage.name, command=printable)
        return StageOutcome(stage.name, 0, 0.0)

    before = count_rows(stage.schemas, target=stage.target, settings=settings)
    event(
        logging.INFO, "stage started",
        stage=stage.name, command=printable, target=stage.target,
        rows_in=sum(before.values()), tables=len(before),
    )

    started = time.perf_counter()
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=os.environ.copy())
    seconds = time.perf_counter() - started

    after = count_rows(stage.schemas, target=stage.target, settings=settings)
    outcome = StageOutcome(stage.name, completed.returncode, seconds, before, after)
    changed = _delta(before, after)

    if not outcome.ok:
        event(
            logging.ERROR, "stage failed",
            stage=stage.name, exit_code=completed.returncode,
            seconds=round(seconds, 2), rows_in=outcome.total_before,
            rows_out=outcome.total_after, changed=changed or None,
        )
        raise StageFailed(
            f"{stage.name} exited {completed.returncode}. The run stops here: "
            f"continuing would publish results built on a stage that did not finish."
        )

    event(
        logging.INFO, "stage finished",
        stage=stage.name, seconds=round(seconds, 2),
        rows_in=outcome.total_before, rows_out=outcome.total_after,
        rows_delta=outcome.total_after - outcome.total_before,
        changed=changed or None,
    )
    return outcome


def plan(mode: str, only: Sequence[str] | None, skip: Sequence[str] | None) -> list[Stage]:
    """Which stages run, in pipeline order whatever order they were asked for.

    ``--only ingest dbt`` and ``--only dbt ingest`` are the same run. Letting
    the flag reorder the pipeline would make it possible to build gold from
    bronze that had not been fetched yet, and to do it by typo.
    """
    chosen = set(only) if only else set(MODES[mode])
    chosen -= set(skip or ())
    return [stage for stage in STAGES if stage.name in chosen]


def run(
    *,
    mode: str = "daily",
    only: Sequence[str] | None = None,
    skip: Sequence[str] | None = None,
    dry_run: bool = False,
    settings: Settings | None = None,
) -> list[StageOutcome]:
    """Execute the pipeline. Raises :class:`StageFailed` at the first failure."""
    resolved = settings if settings is not None else get_settings()
    stages = plan(mode, only, skip)

    event(
        logging.INFO, "run started",
        mode=mode, stages=[stage.name for stage in stages],
        database=mask_secret(resolved.database_url),
        dry_run=dry_run or None,
    )

    outcomes: list[StageOutcome] = []
    for stage in stages:
        outcomes.append(run_stage(stage, mode, resolved, dry_run=dry_run))
    return outcomes


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Stages, in order: " + ", ".join(STAGE_NAMES),
    )
    parser.add_argument(
        "--mode",
        choices=sorted(MODES),
        default="daily",
        help=(
            "daily: a %d-day incremental window, daily grain only. "
            "backfill: the full range including the hourly window. "
            "(default: daily)" % DAILY_LOOKBACK_DAYS
        ),
    )
    parser.add_argument(
        "--only", nargs="+", choices=STAGE_NAMES, metavar="STAGE",
        help="run only these stages, still in pipeline order",
    )
    parser.add_argument(
        "--skip", nargs="+", choices=STAGE_NAMES, metavar="STAGE",
        help="run everything the mode selects except these",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the commands that would run, and run none of them",
    )
    parser.add_argument("--log-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        settings = get_settings()
    except ConfigError as exc:
        print(f"FAILED  {exc}", file=sys.stderr)
        return 2

    run_id = f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    destination = configure_logging(
        args.log_dir or settings.log_dir, run_id, settings.log_level
    )

    started = time.perf_counter()
    event(logging.INFO, "pipeline", run_id=run_id, log=str(destination))

    try:
        outcomes = run(
            mode=args.mode, only=args.only, skip=args.skip,
            dry_run=args.dry_run, settings=settings,
        )
    except StageFailed as exc:
        event(
            logging.ERROR, "run failed",
            seconds=round(time.perf_counter() - started, 2), reason=str(exc),
        )
        print(f"\nFAILED  {exc}", file=sys.stderr)
        print(f"        log: {destination}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        event(logging.ERROR, "run interrupted",
              seconds=round(time.perf_counter() - started, 2))
        return 130

    event(
        logging.INFO, "run finished",
        seconds=round(time.perf_counter() - started, 2),
        stages=len(outcomes),
        rows_delta=sum(o.total_after - o.total_before for o in outcomes),
    )
    print(f"\nOK      {len(outcomes)} stage(s); log: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
