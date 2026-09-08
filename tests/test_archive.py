"""Tests for raw payload archival and replay.

Two claims carry the ticket, and each gets a test that would fail loudly if it
stopped being true: the payload reaches disk *before* anything parses it, and
bronze can be rebuilt from disk with the network unplugged. The rest is the
path layout, atomicity, and the size measurement.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

requests = pytest.importorskip("requests")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Settings, get_settings  # noqa: E402
from ingestion import archive  # noqa: E402
from ingestion.client import (  # noqa: E402
    ArchiveResponseError,
    ArchiveTransportError,
    fetch_observations,
    request_url,
)
from ingestion.planner import WorkUnit  # noqa: E402
from http_fixtures import daily_payload, hourly_payload, responds, session_for  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CITY = "london"
START = dt.date(2023, 1, 1)
END = dt.date(2023, 1, 3)
DAYS = 3

UNIT = WorkUnit(city_id=CITY, grain="daily", start=START, end=END)
HOURLY_UNIT = WorkUnit(
    city_id=CITY, grain="hourly", start=START, end=START
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "raw"


@pytest.fixture
def settings(root: Path) -> Settings:
    return dataclasses.replace(
        get_settings(),
        openmeteo_base_url="https://archive-api.test/v1/archive",
        data_raw_dir=root,
        max_retry_attempts=2,
    )


def payload_bytes(days: int = DAYS, start: dt.date = START, **overrides) -> bytes:
    return json.dumps(daily_payload(days, start, **overrides)).encode("utf-8")


def fetch(script, settings, *, unit=UNIT, **kwargs):
    session, adapter = session_for(script)
    try:
        result = fetch_observations(
            unit.city_id, unit.start, unit.end, unit.grain,
            session=session, settings=settings, **kwargs,
        )
    finally:
        session.close()
    return result, adapter


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_path_is_grain_city_window(root: Path) -> None:
    assert archive.archive_path(UNIT, root) == (
        root / "daily" / "london" / "2023-01-01_2023-01-03.json.gz"
    )


def test_the_path_encodes_the_whole_unit(root: Path) -> None:
    """The tree is its own index; nothing needs a sidecar to know what is here."""
    archive.write(UNIT, payload_bytes(), root)
    (recovered,) = archive.archived_units(root)
    assert recovered == UNIT


def test_grains_and_cities_do_not_collide(root: Path) -> None:
    archive.write(UNIT, payload_bytes(), root)
    archive.write(dataclasses.replace(UNIT, grain="hourly"), b"{}", root)
    archive.write(dataclasses.replace(UNIT, city_id="tokyo"), b"{}", root)
    assert len(list(archive.archived_units(root))) == 3


def test_the_configured_root_is_used_when_none_is_given(settings, root) -> None:
    archive.write(UNIT, payload_bytes(), settings=settings)
    assert archive.archive_path(UNIT, settings=settings).is_file()
    assert archive.archive_path(UNIT, settings=settings).is_relative_to(root)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_what_comes_back_is_exactly_what_went_in(root: Path) -> None:
    """Not re-serialised, not reordered: byte-identical."""
    raw = payload_bytes()
    archive.write(UNIT, raw, root)
    assert archive.read_bytes(UNIT, root) == raw


def test_the_file_is_gzip(root: Path) -> None:
    path = archive.write(UNIT, payload_bytes(), root)
    assert path.read_bytes()[:2] == b"\x1f\x8b"
    with gzip.open(path, "rb") as handle:
        assert json.loads(handle.read())["utc_offset_seconds"] == 0


def test_it_actually_compresses(root: Path) -> None:
    raw = payload_bytes(days=365, start=dt.date(2023, 1, 1))
    path = archive.write(UNIT, raw, root)
    assert path.stat().st_size < len(raw) / 2


def test_writing_creates_missing_directories(root: Path) -> None:
    assert not root.exists()
    archive.write(UNIT, payload_bytes(), root)
    assert archive.archive_path(UNIT, root).is_file()


def test_identical_input_gives_identical_bytes(root: Path, tmp_path: Path) -> None:
    """mtime=0 in the header, so re-archiving is detectably a no-op."""
    raw = payload_bytes()
    first = archive.write(UNIT, raw, root).read_bytes()
    second = archive.write(UNIT, raw, tmp_path / "other").read_bytes()
    assert first == second


def test_rewriting_replaces_and_leaves_no_temporary_files(root: Path) -> None:
    archive.write(UNIT, payload_bytes(days=DAYS), root)
    archive.write(UNIT, payload_bytes(days=DAYS, start=dt.date(2024, 1, 1)), root)
    directory = archive.archive_path(UNIT, root).parent
    assert [p.name for p in directory.iterdir()] == [
        archive.archive_path(UNIT, root).name
    ]
    assert b"2024-01-01" in archive.read_bytes(UNIT, root)


def test_a_failed_write_leaves_no_partial_file(root: Path, monkeypatch) -> None:
    """A crash mid-write must leave the previous file, or none, but never a stub."""
    archive.write(UNIT, payload_bytes(), root)
    good = archive.archive_path(UNIT, root).read_bytes()

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(archive.os, "replace", explode)
    with pytest.raises(OSError, match="disk full"):
        archive.write(UNIT, b'{"different": true}', root)

    directory = archive.archive_path(UNIT, root).parent
    assert [p.name for p in directory.iterdir()] == [
        archive.archive_path(UNIT, root).name
    ]
    assert archive.archive_path(UNIT, root).read_bytes() == good


# ---------------------------------------------------------------------------
# The archive write happens before the parse
# ---------------------------------------------------------------------------


def test_the_payload_is_archived_before_it_is_parsed(settings, root) -> None:
    """The whole ticket in one test.

    This response is well-formed JSON that fails validation: the units come
    back in Fahrenheit. If archival ran after parsing, the payload would be
    lost and the fix would cost a re-pull.
    """
    body = daily_payload(DAYS, START)
    body["daily_units"]["temperature_2m_max"] = "°F"

    with pytest.raises(ArchiveResponseError):
        fetch(
            [responds(json_body=body)],
            settings,
            on_payload=archive.writer_for(UNIT, root),
        )

    assert archive.exists(UNIT, root)
    assert json.loads(archive.read_bytes(UNIT, root))["daily_units"][
        "temperature_2m_max"
    ] == "°F"


def test_the_hook_sees_the_bytes_not_the_parsed_object(settings, root) -> None:
    seen: list[tuple[bytes, str]] = []
    fetch(
        [responds(json_body=daily_payload(DAYS, START))],
        settings,
        on_payload=lambda raw, url: seen.append((raw, url)),
    )
    (raw, url), = seen
    assert isinstance(raw, bytes)
    assert url.startswith(settings.openmeteo_base_url)


def test_retried_attempts_are_not_archived(settings, root) -> None:
    """Only the response that succeeded reaches disk."""
    written: list[bytes] = []
    fetch(
        [
            responds(500, text="upstream is unwell"),
            responds(json_body=daily_payload(DAYS, START)),
        ],
        settings,
        on_payload=lambda raw, url: written.append(raw),
    )
    assert len(written) == 1
    assert b"upstream is unwell" not in written[0]


def test_a_request_that_never_succeeds_archives_nothing(settings, root) -> None:
    with pytest.raises(ArchiveTransportError):
        fetch(
            [requests.exceptions.ReadTimeout("boom")] * 2,
            settings,
            on_payload=archive.writer_for(UNIT, root),
        )
    assert not archive.exists(UNIT, root)


def test_a_4xx_archives_nothing(settings, root) -> None:
    from ingestion.client import ArchiveRequestError

    with pytest.raises(ArchiveRequestError):
        fetch(
            [responds(400, json_body={"error": True, "reason": "bad range"})],
            settings,
            on_payload=archive.writer_for(UNIT, root),
        )
    assert not archive.exists(UNIT, root)


def test_a_failing_hook_is_not_swallowed(settings, root) -> None:
    """A response fetched and then dropped on the floor is worse than a crash."""

    def refuse(raw: bytes, url: str) -> None:
        raise OSError("no space left on device")

    with pytest.raises(OSError, match="no space left"):
        fetch([responds(json_body=daily_payload(DAYS, START))], settings,
              on_payload=refuse)


def test_no_hook_means_no_archive(settings, root) -> None:
    response, _ = fetch([responds(json_body=daily_payload(DAYS, START))], settings)
    assert len(response) == DAYS
    assert not root.exists()


# ---------------------------------------------------------------------------
# Replay rebuilds bronze with no network
# ---------------------------------------------------------------------------


@pytest.fixture
def no_network(monkeypatch):
    """Make any outbound HTTP call an immediate, obvious failure."""

    def forbidden(*args, **kwargs):
        raise AssertionError("replay must not touch the network")

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", forbidden)


def test_replay_reproduces_the_live_response(settings, root, no_network) -> None:
    """Identical rows, identical provenance, from disk with the socket shut."""
    live, _ = fetch(
        [responds(json_body=daily_payload(DAYS, START))],
        settings,
        on_payload=archive.writer_for(UNIT, root),
    )
    replayed = archive.replay_unit(UNIT, root, settings=settings)

    assert replayed.times == live.times
    assert dict(replayed.values) == dict(live.values)
    assert dict(replayed.units) == dict(live.units)
    assert replayed.url == live.url
    assert (replayed.latitude, replayed.longitude) == (live.latitude, live.longitude)
    assert replayed.elevation_m == live.elevation_m
    assert list(replayed.rows()) == list(live.rows())


def test_replay_rebuilds_the_whole_archive(settings, root, no_network) -> None:
    units = [
        WorkUnit(city_id=city, grain="daily", start=START, end=END)
        for city in ("london", "tokyo", "cairo")
    ]
    for unit in units:
        archive.write(unit, payload_bytes(), root)
    archive.write(
        HOURLY_UNIT, json.dumps(hourly_payload(1, START)).encode(), root
    )

    replayed = list(archive.replay(root, settings=settings))
    assert len(replayed) == 4
    assert sum(len(r) for r in replayed) == DAYS * 3 + 24
    assert {r.city_id for r in replayed} == {"london", "tokyo", "cairo"}


def test_replay_is_lazy(settings, root, no_network) -> None:
    """Hundreds of megabytes decompressed is not something to materialise."""
    archive.write(UNIT, payload_bytes(), root)
    stream = archive.replay(root, settings=settings)
    assert next(stream).city_id == CITY


def test_replay_can_be_narrowed(settings, root, no_network) -> None:
    archive.write(UNIT, payload_bytes(), root)
    archive.write(dataclasses.replace(UNIT, city_id="tokyo"), payload_bytes(), root)
    archive.write(HOURLY_UNIT, json.dumps(hourly_payload(1, START)).encode(), root)

    assert len(list(archive.replay(root, grain="daily", settings=settings))) == 2
    assert len(list(archive.replay(root, city_id="tokyo", settings=settings))) == 1


def test_replay_url_matches_what_the_request_would_have_been(settings) -> None:
    """The derived URL must be replayable against the API, not a placeholder."""
    assert request_url(
        CITY, START, END, "daily", settings=settings
    ).startswith(settings.openmeteo_base_url)


# ---------------------------------------------------------------------------
# Deriving the request URL bronze no longer stores
# ---------------------------------------------------------------------------


def test_the_url_is_derived_from_the_window_that_covers_a_row(
    settings, root, no_network
) -> None:
    """What replaces the source_url column, at 0 bytes per row instead of 712."""
    archive.write(UNIT, payload_bytes(), root)
    derived = archive.source_url_for(
        CITY, "daily", dt.datetime(2023, 1, 2, tzinfo=dt.timezone.utc),
        root, settings=settings,
    )
    assert derived == request_url(CITY, START, END, "daily", settings=settings)


def test_the_derived_url_equals_what_the_response_carried(settings, root) -> None:
    live, _ = fetch(
        [responds(json_body=daily_payload(DAYS, START))],
        settings,
        on_payload=archive.writer_for(UNIT, root),
    )
    for moment in live.times:
        assert archive.source_url_for(
            CITY, "daily", moment, root, settings=settings
        ) == live.url


def test_a_date_and_a_datetime_resolve_the_same(settings, root, no_network) -> None:
    archive.write(UNIT, payload_bytes(), root)
    as_date = archive.source_url_for(CITY, "daily", START, root, settings=settings)
    as_datetime = archive.source_url_for(
        CITY, "daily", dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc),
        root, settings=settings,
    )
    assert as_date == as_datetime


def test_the_covering_window_is_the_one_that_was_fetched(
    settings, root, no_network
) -> None:
    """Chunk size can change between runs; the archive records what really ran."""
    wide = WorkUnit(city_id=CITY, grain="daily",
                    start=dt.date(2023, 1, 1), end=dt.date(2023, 6, 30))
    archive.write(wide, payload_bytes(days=181, start=dt.date(2023, 1, 1)), root)
    found = archive.covering_unit(
        CITY, "daily", dt.date(2023, 3, 15), root, settings=settings
    )
    assert found == wide
    assert archive.source_url_for(
        CITY, "daily", dt.date(2023, 3, 15), root, settings=settings
    ) == request_url(CITY, wide.start, wide.end, "daily", settings=settings)


def test_a_row_outside_every_archived_window_says_so(
    settings, root, no_network
) -> None:
    archive.write(UNIT, payload_bytes(), root)
    assert archive.covering_unit(
        CITY, "daily", dt.date(2024, 6, 1), root, settings=settings
    ) is None
    with pytest.raises(LookupError, match="no archived daily payload"):
        archive.source_url_for(
            CITY, "daily", dt.date(2024, 6, 1), root, settings=settings
        )


def test_grains_do_not_borrow_each_others_windows(
    settings, root, no_network
) -> None:
    archive.write(UNIT, payload_bytes(), root)
    with pytest.raises(LookupError):
        archive.source_url_for(CITY, "hourly", START, root, settings=settings)


def test_deriving_needs_no_network(settings, root, no_network) -> None:
    archive.write(UNIT, payload_bytes(), root)
    assert archive.source_url_for(CITY, "daily", START, root, settings=settings)


def test_replay_applies_the_same_parser_as_the_live_path(
    settings, root, no_network
) -> None:
    """A payload the live path would reject is rejected on replay too."""
    body = daily_payload(DAYS, START)
    body["utc_offset_seconds"] = 3600
    archive.write(UNIT, json.dumps(body).encode(), root)
    with pytest.raises(ArchiveResponseError, match="utc_offset_seconds"):
        archive.replay_unit(UNIT, root, settings=settings)


# ---------------------------------------------------------------------------
# Discovery and damaged files
# ---------------------------------------------------------------------------


def test_discovery_is_deterministic(root: Path) -> None:
    for city in ("tokyo", "london", "cairo"):
        for year in (2021, 2020):
            archive.write(
                WorkUnit(city_id=city, grain="daily",
                         start=dt.date(year, 1, 1), end=dt.date(year, 12, 31)),
                b"{}", root,
            )
    keys = [u.key for u in archive.archived_units(root)]
    assert keys == sorted(keys)
    assert keys == [u.key for u in archive.archived_units(root)]


def test_unrecognised_files_are_ignored(root: Path) -> None:
    archive.write(UNIT, payload_bytes(), root)
    (root / "daily" / "london" / "notes.txt").write_text("scratch", encoding="utf-8")
    (root / "daily" / "london" / "nope.json.gz").write_bytes(b"")
    (root / "monthly").mkdir()
    (root / "monthly" / "x.json.gz").write_bytes(b"")
    assert [u.key for u in archive.archived_units(root)] == [UNIT.key]


def test_an_absent_archive_is_empty_not_an_error(root: Path) -> None:
    assert list(archive.archived_units(root)) == []
    assert archive.stats(root).files == 0


def test_a_missing_payload_names_the_path(root: Path) -> None:
    with pytest.raises(FileNotFoundError, match="2023-01-01_2023-01-03"):
        archive.read_bytes(UNIT, root)


def test_a_file_that_is_not_gzip_is_reported(root: Path) -> None:
    path = archive.archive_path(UNIT, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"this is not gzip")
    with pytest.raises(archive.ArchiveFileError, match="gzip"):
        archive.read_bytes(UNIT, root)


def test_gzip_that_is_not_json_is_reported(root: Path) -> None:
    archive.write(UNIT, b"{truncated", root)
    with pytest.raises(archive.ArchiveFileError, match="does not contain JSON"):
        archive.read_payload(UNIT, root)


def test_json_that_is_not_an_object_is_reported(root: Path) -> None:
    archive.write(UNIT, b"[1, 2, 3]", root)
    with pytest.raises(archive.ArchiveFileError, match="expected a JSON object"):
        archive.read_payload(UNIT, root)


# ---------------------------------------------------------------------------
# Size
# ---------------------------------------------------------------------------


def test_stats_count_files_bytes_and_rows(root: Path) -> None:
    archive.write(UNIT, payload_bytes(), root)
    archive.write(HOURLY_UNIT, json.dumps(hourly_payload(1, START)).encode(), root)

    measured = archive.stats(root)
    assert measured.files == 2
    assert measured.rows == DAYS + 24
    assert measured.compressed_bytes == sum(
        archive.archive_path(u, root).stat().st_size for u in (UNIT, HOURLY_UNIT)
    )
    assert measured.by_grain["daily"].rows == DAYS
    assert measured.by_grain["hourly"].rows == 24


def test_projection_is_per_grain_not_blended(root: Path) -> None:
    """A daily row carries 21 variables and an hourly row 12; one rate is wrong."""
    archive.write(UNIT, payload_bytes(days=DAYS), root)
    archive.write(HOURLY_UNIT, json.dumps(hourly_payload(1, START)).encode(), root)
    measured = archive.stats(root)

    daily_rate = measured.by_grain["daily"].bytes_per_row
    hourly_rate = measured.by_grain["hourly"].bytes_per_row
    assert measured.project({"daily": 1000}) == round(daily_rate * 1000)
    assert measured.project({"daily": 1000, "hourly": 1000}) == round(
        daily_rate * 1000 + hourly_rate * 1000
    )


def test_projection_ignores_a_grain_it_has_never_seen(root: Path) -> None:
    archive.write(UNIT, payload_bytes(), root)
    measured = archive.stats(root)
    assert measured.project({"hourly": 10_000}) == 0


def test_stats_can_skip_decompression(root: Path) -> None:
    archive.write(UNIT, payload_bytes(), root)
    assert archive.stats(root, count_rows=False).rows == 0
    assert archive.stats(root, count_rows=False).files == 1


# ---------------------------------------------------------------------------
# Nothing archived is ever committed
# ---------------------------------------------------------------------------


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )


@pytest.fixture(scope="module")
def in_git_repo() -> None:
    if git("rev-parse", "--git-dir").returncode != 0:
        pytest.skip("not a git repository")


@pytest.mark.parametrize(
    "path",
    ["data/", "data/raw/", "data/raw/daily/london/1998-01-01_1998-12-31.json.gz"],
)
def test_the_archive_is_git_ignored(in_git_repo, path: str) -> None:
    assert git("check-ignore", "-q", path).returncode == 0, f"{path} is not ignored"


def test_nothing_under_data_is_tracked(in_git_repo) -> None:
    assert git("ls-files", "data/").stdout.strip() == ""
