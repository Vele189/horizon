"""Tests for the pipeline entrypoint and its logging.

Two things are worth the line count.

**Halting is the feature.** A pipeline that continues past a broken transform
publishes stale predictions as though they were fresh, and does it quietly,
because every stage after the broken one still succeeds. So the tests here do
not merely check that a failure is reported — they check that nothing after it
ran, which is the part a reader of the log would otherwise have to infer.

**Row counts are measured, not reported.** The runner counts rows itself either
side of each stage rather than trusting a stage's own summary, and the
difference matters most in the case that is hardest to notice: a stage that
writes nothing and exits zero. Its self-report would say so and be believed;
an independent count says the delta was nothing.

Nothing here runs a stage. Every subprocess is scripted, because a test that
shelled out to dbt would be a test of dbt.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_pipeline  # noqa: E402
from run_pipeline import (  # noqa: E402
    MODES,
    STAGE_NAMES,
    StageFailed,
    main,
    plan,
    run,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def scripted(monkeypatch):
    """Replace every subprocess with a scripted exit code, and record the order."""
    calls: list[list[str]] = []

    def install(exit_codes: dict[str, int] | None = None):
        codes = exit_codes or {}

        def fake_run(command, **kwargs):
            calls.append(command)
            joined = " ".join(command)
            for fragment, code in codes.items():
                if fragment in joined:
                    return subprocess.CompletedProcess(command, code)
            return subprocess.CompletedProcess(command, 0)

        monkeypatch.setattr(run_pipeline.subprocess, "run", fake_run)
        # Counting is instrumentation; it must not reach a database here.
        monkeypatch.setattr(run_pipeline, "count_rows", lambda *a, **k: {})
        return calls

    return install


@pytest.fixture(autouse=True)
def quiet_logging(tmp_path):
    """Send the run's logs somewhere disposable."""
    run_pipeline.configure_logging(tmp_path / "logs", "test")
    yield
    logging.getLogger("pipeline").handlers.clear()


# --------------------------------------------------------------------------
# The order, and the stages
# --------------------------------------------------------------------------


def test_the_four_stages_run_in_pipeline_order() -> None:
    """Ingest, transform, score, publish. Each depends on the one before it."""
    daily = [stage.name for stage in plan("daily", None, None)]
    assert daily == ["ingest", "dbt", "predict", "promote"]


def test_a_flag_cannot_reorder_the_pipeline() -> None:
    """``--only dbt ingest`` is the same run as ``--only ingest dbt``.

    Letting the flag decide the order would make it possible to build gold from
    bronze that had not been fetched yet, and to do it by typo.
    """
    forwards = [s.name for s in plan("daily", ["ingest", "dbt"], None)]
    backwards = [s.name for s in plan("daily", ["dbt", "ingest"], None)]
    assert forwards == backwards == ["ingest", "dbt"]


def test_stages_are_runnable_in_isolation() -> None:
    for name in STAGE_NAMES:
        assert [s.name for s in plan("backfill", [name], None)] == [name]


def test_skip_removes_only_what_it_names() -> None:
    remaining = [s.name for s in plan("daily", None, ["promote"])]
    assert remaining == ["ingest", "dbt", "predict"]
    assert "promote" not in remaining


def test_the_two_modes_differ_by_the_hourly_window() -> None:
    """A nightly run must not re-walk two years of hourly units.

    The hourly window is a rolling two years the backfill maintains. Re-walking
    it every night would spend the API budget rediscovering what is already on
    disk, which is the kind of waste that only shows up as a quota error weeks
    later.
    """
    daily = set(MODES["daily"])
    backfill = set(MODES["backfill"])
    assert "ingest-hourly" in backfill
    assert "ingest-hourly" not in daily
    assert daily < backfill


def test_daily_asks_for_a_window_and_backfill_does_not(scripted) -> None:
    """Incremental means a date window; full means the planner's whole range."""
    from config import get_settings

    settings = get_settings()
    ingest = next(s for s in run_pipeline.STAGES if s.name == "ingest")

    daily = ingest.command("daily", settings)
    assert "--start" in daily
    assert "--grain" in daily and "daily" in daily

    full = ingest.command("backfill", settings)
    assert "--start" not in full


def test_the_daily_window_reaches_back_past_yesterday() -> None:
    """The archive revises recent observations.

    A window that asked only for yesterday would never pick up a correction to
    the day before, and the warehouse would keep a number the source no longer
    stands behind.
    """
    assert run_pipeline.DAILY_LOOKBACK_DAYS > 1


# --------------------------------------------------------------------------
# Halting
# --------------------------------------------------------------------------


def test_a_failed_stage_stops_everything_after_it(scripted) -> None:
    """The reason this file exists.

    Promotion after a failed transform copies a stale mart to Neon and reports
    success, because promotion itself worked. The only safe behaviour is to
    stop.
    """
    calls = scripted({"dbt_env.py": 1})

    with pytest.raises(StageFailed) as raised:
        run(mode="daily")

    ran = [" ".join(c) for c in calls]
    assert any("backfill.py" in c for c in ran), "ingest did not run"
    assert any("dbt_env.py" in c for c in ran), "dbt did not run"
    assert not any("predict.py" in c for c in ran), "predict ran after dbt failed"
    assert not any("promote.py" in c for c in ran), "promote ran after dbt failed"
    assert "stops here" in str(raised.value)


@pytest.mark.parametrize("failing", ["backfill.py", "dbt_env.py", "predict.py", "promote.py"])
def test_any_stage_failing_halts_the_run(scripted, failing) -> None:
    scripted({failing: 2})
    with pytest.raises(StageFailed):
        run(mode="daily")


def test_the_process_exits_non_zero_when_a_stage_fails(scripted) -> None:
    """A zero exit from a broken run is how a scheduler learns nothing."""
    scripted({"predict.py": 3})
    assert main(["--mode", "daily"]) == 1


def test_the_process_exits_zero_when_every_stage_succeeds(scripted) -> None:
    scripted({})
    assert main(["--mode", "daily"]) == 0


def test_there_is_no_way_to_ask_it_to_keep_going() -> None:
    """A --keep-going flag is the feature this pipeline must not grow.

    Asserted over the source rather than trusted, because it is exactly the
    kind of convenience that gets added at 2 a.m. to get one run through.
    """
    import ast

    tree = ast.parse((PROJECT_ROOT / "run_pipeline.py").read_text(encoding="utf-8"))
    declared = {
        argument.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for argument in node.args
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
    }
    # Asserted over the declared flags rather than over the source text: the
    # docstring says there is no --keep-going, and a substring search would
    # find the sentence promising its absence and call it the flag.
    for flag in ("--keep-going", "--continue-on-error", "--ignore-errors", "--force"):
        assert flag not in declared


def test_dry_run_runs_nothing(scripted) -> None:
    calls = scripted({})
    outcomes = run(mode="backfill", dry_run=True)
    assert calls == []
    assert len(outcomes) == len(MODES["backfill"])


# --------------------------------------------------------------------------
# Logging — OPS-02
# --------------------------------------------------------------------------


def test_logs_go_to_stdout_and_to_a_file(tmp_path, scripted, capsys) -> None:
    scripted({})
    destination = run_pipeline.configure_logging(tmp_path / "logs", "abc")
    run(mode="daily")

    printed = capsys.readouterr().out
    assert "stage started" in printed
    assert "stage finished" in printed

    written = destination.read_text(encoding="utf-8").splitlines()
    assert written, "nothing reached the log file"
    assert destination.suffix == ".jsonl"


def test_every_logged_line_is_a_parseable_object(tmp_path, scripted) -> None:
    """Structured means machine-readable, or it means nothing."""
    scripted({})
    destination = run_pipeline.configure_logging(tmp_path / "logs", "abc")
    run(mode="daily")

    records = [json.loads(line) for line in destination.read_text().splitlines()]
    assert records
    for record in records:
        assert {"ts", "level", "event"} <= set(record)
        assert record["level"] in {"INFO", "WARNING", "ERROR"}


def test_each_stage_logs_counts_either_side_and_a_duration(tmp_path, monkeypatch) -> None:
    """Row counts in and out, and the seconds between them.

    The counts are the cheapest data-quality signal there is, and the delta is
    what catches the failure a self-report cannot: a stage that writes nothing
    and exits zero.
    """
    counts = iter([{"bronze_raw.a": 10}, {"bronze_raw.a": 25}])
    monkeypatch.setattr(run_pipeline, "count_rows", lambda *a, **k: next(counts))
    monkeypatch.setattr(
        run_pipeline.subprocess, "run",
        lambda command, **k: subprocess.CompletedProcess(command, 0),
    )
    destination = run_pipeline.configure_logging(tmp_path / "logs", "abc")
    run(mode="daily", only=["ingest"])

    records = [json.loads(line) for line in destination.read_text().splitlines()]
    started = next(r for r in records if r["event"] == "stage started")
    finished = next(r for r in records if r["event"] == "stage finished")

    assert started["stage"] == "ingest" and started["rows_in"] == 10
    assert finished["rows_in"] == 10 and finished["rows_out"] == 25
    assert finished["rows_delta"] == 15
    assert finished["changed"] == {"bronze_raw.a": 15}
    assert isinstance(finished["seconds"], (int, float))


def test_a_failure_is_logged_with_its_exit_code(tmp_path, scripted) -> None:
    scripted({"dbt_env.py": 7})
    destination = run_pipeline.configure_logging(tmp_path / "logs", "abc")
    with pytest.raises(StageFailed):
        run(mode="daily")

    records = [json.loads(line) for line in destination.read_text().splitlines()]
    failure = next(r for r in records if r["event"] == "stage failed")
    assert failure["stage"] == "dbt"
    assert failure["exit_code"] == 7
    assert failure["level"] == "ERROR"


def test_pipeline_events_are_logged_rather_than_printed() -> None:
    """"Structured logging configured, not bare print statements."

    The two prints that remain are the final human summary on stdout and the
    config-error path before logging exists — neither is an event, and both are
    accompanied by a logged record.
    """
    import ast

    tree = ast.parse((PROJECT_ROOT / "run_pipeline.py").read_text(encoding="utf-8"))
    prints = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]
    assert len(prints) <= 4, f"{len(prints)} print calls — events belong in the log"


def test_the_database_url_is_masked_in_the_log(tmp_path, scripted) -> None:
    """A log is a file that gets pasted into issues."""
    scripted({})
    destination = run_pipeline.configure_logging(tmp_path / "logs", "abc")
    run(mode="daily")
    written = destination.read_text(encoding="utf-8")
    assert "***" in written or "<unset>" in written
    assert "postgresql://climate:climate@" not in written


def test_counting_never_fails_a_run(monkeypatch) -> None:
    """Instrumentation that can break the thing it measures is worse than none."""
    from config import get_settings

    def explode(*args, **kwargs):
        raise RuntimeError("no database here")

    monkeypatch.setattr(run_pipeline, "create_engine", explode, raising=False)
    import sqlalchemy

    monkeypatch.setattr(sqlalchemy, "create_engine", explode)
    assert run_pipeline.count_rows(("gold_marts",), target="local",
                                   settings=get_settings()) == {}


def test_the_log_directory_is_git_ignored() -> None:
    """A run's output records one execution; it is not a fact about the repo."""
    rules = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "logs/" in rules

    tracked = subprocess.run(
        ["git", "ls-files", "logs/"], cwd=PROJECT_ROOT, capture_output=True, text=True
    ).stdout.split()
    assert [p for p in tracked if p.endswith(".jsonl")] == []
