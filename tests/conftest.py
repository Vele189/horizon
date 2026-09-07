"""Fixtures shared by the ingestion test suites.

The database ones live here rather than in each module because they are
identical wherever they appear and because a session-scoped engine is one
connection attempt for the whole run instead of one per module. Modules that
need different settings define their own ``settings`` fixture, which shadows
the one here.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ConfigError, Settings, get_settings  # noqa: E402


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "live: makes a real request to the Open-Meteo archive. Deselect with "
        "-m 'not live' for a fully hermetic run.",
    )


@pytest.fixture(autouse=True)
def _forbid_accidental_network(request, monkeypatch):
    """Make a real HTTP call impossible outside tests marked ``live``.

    The retry tests are the ones most worth trusting and the hardest to trust:
    a 503-then-success path that quietly reached the real API would pass for
    the wrong reason, and would pass differently on a bad day. Rather than
    assert that no test calls out, this makes the call fail — so the guarantee
    is enforced rather than reviewed.

    Only :class:`requests.adapters.HTTPAdapter` is blocked. The scripted
    transport is a separate ``BaseAdapter`` and is unaffected, and psycopg2
    reaches the warehouse through a different library entirely.
    """
    if "live" in request.keywords:
        return
    import requests

    def forbidden(self, request_, *args, **kwargs):
        raise AssertionError(
            f"{request.node.nodeid} attempted a real HTTP request to "
            f"{request_.url.split('?')[0]}. Script the transport, or mark the "
            "test @pytest.mark.live."
        )

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", forbidden)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """An empty payload archive."""
    return tmp_path / "raw"


@pytest.fixture
def manifest(tmp_path: Path):
    from ingestion.planner import Manifest

    return Manifest(tmp_path / "manifest.jsonl")


@pytest.fixture
def settings(root: Path, tmp_path: Path) -> Settings:
    """Settings pointed at a scratch archive and an unreachable API host."""
    return dataclasses.replace(
        get_settings(),
        openmeteo_base_url="https://archive-api.test/v1/archive",
        data_raw_dir=root,
        ingest_manifest_path=tmp_path / "manifest.jsonl",
        max_retry_attempts=1,
    )


@pytest.fixture(scope="session")
def engine():
    """The real warehouse. Skips the suite when there isn't one."""
    from sqlalchemy import text

    from ingestion.loader import engine_from_settings

    try:
        get_settings().require_database_url()
    except ConfigError as exc:
        pytest.skip(f"no DATABASE_URL: {exc}")
    built = engine_from_settings()
    try:
        with built.connect() as connection:
            connection.execute(text("select 1"))
    except Exception as exc:  # noqa: BLE001 - any driver failure means skip
        built.dispose()
        pytest.skip(f"database unreachable: {exc}")
    yield built
    built.dispose()


@pytest.fixture
def cleanup(engine):
    """Collects batch ids and deletes their rows, however the test ended.

    The runner commits per unit by design, so a wrapping transaction would be
    testing something it does not do. Deleting by batch is the same escape
    hatch an operator gets for a bad run.
    """
    from sqlalchemy import text

    from ingestion.loader import BRONZE_SCHEMA, TABLE_BY_GRAIN

    batches: list = []
    yield batches
    with engine.begin() as connection:
        for batch_id in batches:
            for table in TABLE_BY_GRAIN.values():
                connection.execute(
                    text(
                        f"delete from {BRONZE_SCHEMA}.{table} "
                        "where batch_id = :batch"
                    ),
                    {"batch": str(batch_id)},
                )


@pytest.fixture
def instant(monkeypatch):
    """Drop the politeness delay; pacing has its own tests."""
    from ingestion.planner import Plan

    monkeypatch.setattr(Plan, "delay_for", lambda self, unit: 0.0)


@pytest.fixture
def no_network():
    """Declare that a test is *about* not touching the network.

    Redundant with the autouse guard above, and kept because a test named for
    replaying from disk should say so at its signature rather than rely on a
    fixture the reader has to go and find.
    """
    return None


@pytest.fixture
def scripted(monkeypatch):
    """Install a scripted session in place of the runner's real one."""
    from http_fixtures import session_for

    from ingestion import backfill

    def install(script: list):
        session, adapter = session_for(script)
        monkeypatch.setattr(backfill, "build_session", lambda: session)
        return adapter

    return install
