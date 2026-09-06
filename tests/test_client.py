"""Tests for the Open-Meteo archive client.

Everything except the final section runs offline. The transport is a scripted
``requests`` adapter mounted on a real :class:`requests.Session`, so the whole
stack the client actually uses — parameter encoding, the timeout tuple, header
handling — is exercised rather than mocked away.

``tenacity``'s sleep is replaced with a recorder. That makes the retry tests
instant and, more usefully, turns the backoff itself into something assertable:
"honours Retry-After" is a claim about a number of seconds, not about a code
path being taken.

The last section makes one real request. It skips rather than fails when the
network is unavailable, matching tests/test_schema.py's treatment of a missing
database.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import re
import sys
from email.utils import format_datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

requests = pytest.importorskip("requests")
tenacity = pytest.importorskip("tenacity")

from requests.adapters import BaseAdapter  # noqa: E402
from requests.structures import CaseInsensitiveDict  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cities import get_city  # noqa: E402
from config import Settings, get_settings  # noqa: E402
from ingestion.client import (  # noqa: E402
    ARCHIVE_START,
    DAILY_UNITS,
    DAILY_VARIABLES,
    HOURLY_UNITS,
    HOURLY_VARIABLES,
    MAX_BACKOFF_SECONDS,
    MAX_RETRY_AFTER_SECONDS,
    USER_AGENT,
    ArchiveRateLimited,
    ArchiveRequestError,
    ArchiveResponseError,
    ArchiveServerError,
    ArchiveTransportError,
    _parse_retry_after,
    _wait_archive,
    fetch_observations,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_SQL = REPO_ROOT / "ingestion" / "schema.sql"

CITY = "london"
START = dt.date(2023, 1, 1)
END = dt.date(2023, 1, 3)  # three days keeps the fixtures readable
DAYS = 3


# ---------------------------------------------------------------------------
# Scripted transport
# ---------------------------------------------------------------------------


class ScriptedAdapter(BaseAdapter):
    """Replays a fixed script of responses and exceptions, recording calls.

    Each entry is either a ``requests.Response`` factory (a callable taking the
    prepared request) or an exception instance to raise. The script must be
    consumed exactly — a test that expects three attempts and gets two fails
    loudly rather than passing on a coincidence.
    """

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.calls: list[SimpleNamespace] = []

    def send(self, request, stream=False, timeout=None, verify=True, cert=None,
             proxies=None):
        self.calls.append(SimpleNamespace(request=request, timeout=timeout))
        if not self.script:
            raise AssertionError(
                f"unscripted request #{len(self.calls)} to {request.url}"
            )
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item(request)

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass


def responds(status: int = 200, *, json_body=None, text: str | None = None,
             headers: dict | None = None):
    """A response factory for :class:`ScriptedAdapter`."""

    def build(request) -> requests.Response:
        response = requests.Response()
        response.status_code = status
        response.url = request.url
        response.request = request
        response.headers = CaseInsensitiveDict(headers or {})
        if json_body is not None:
            body = json.dumps(json_body)
            response.headers.setdefault("Content-Type", "application/json")
        else:
            body = text or ""
        response._content = body.encode("utf-8")
        response.encoding = "utf-8"
        return response

    return build


@pytest.fixture
def settings() -> Settings:
    """Process settings with retry knobs pinned, so tests do not drift."""
    return dataclasses.replace(
        get_settings(),
        openmeteo_base_url="https://archive-api.test/v1/archive",
        request_connect_timeout_seconds=7,
        request_timeout_seconds=23,
        max_retry_attempts=3,
        retry_backoff_seconds=2,
    )


@pytest.fixture
def slept(monkeypatch) -> list[float]:
    """Every backoff tenacity would have taken, in order, without taking it."""
    recorded: list[float] = []
    monkeypatch.setattr(tenacity.nap.time, "sleep", recorded.append)
    return recorded


def session_for(script: list) -> tuple[requests.Session, ScriptedAdapter]:
    adapter = ScriptedAdapter(script)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session, adapter


# ---------------------------------------------------------------------------
# Payload fixtures
# ---------------------------------------------------------------------------


def daily_payload(days: int = DAYS, start: dt.date = START, **overrides) -> dict:
    times = [(start + dt.timedelta(days=i)).isoformat() for i in range(days)]
    payload = {
        "latitude": 51.493847,
        "longitude": -0.1630249,
        "generationtime_ms": 1.5,
        "utc_offset_seconds": 0,
        "timezone": "GMT",
        "timezone_abbreviation": "GMT",
        "elevation": 16.0,
        "daily_units": {"time": "iso8601", **DAILY_UNITS},
        "daily": {
            "time": times,
            **{name: [1.0] * days for name in DAILY_VARIABLES},
        },
    }
    payload.update(overrides)
    return payload


def hourly_payload(days: int = 1, start: dt.date = START, **overrides) -> dict:
    hours = days * 24
    base = dt.datetime.combine(start, dt.time())
    times = [(base + dt.timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M")
             for i in range(hours)]
    payload = {
        "latitude": 51.493847,
        "longitude": -0.1630249,
        "generationtime_ms": 2.5,
        "utc_offset_seconds": 0,
        "timezone": "GMT",
        "timezone_abbreviation": "GMT",
        "elevation": 16.0,
        "hourly_units": {"time": "iso8601", **HOURLY_UNITS},
        "hourly": {
            "time": times,
            **{name: [2.0] * hours for name in HOURLY_VARIABLES},
        },
    }
    payload.update(overrides)
    return payload


def fetch(script, settings, *, city=CITY, start=START, end=END, grain="daily"):
    session, adapter = session_for(script)
    try:
        result = fetch_observations(
            city, start, end, grain, session=session, settings=settings
        )
    finally:
        session.close()
    return result, adapter


def query_of(adapter: ScriptedAdapter, index: int = 0) -> dict[str, str]:
    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(adapter.calls[index].request.url)
    return {k: v[0] for k, v in parse_qs(parsed.query).items()}


# ---------------------------------------------------------------------------
# The request: metric units and UTC asked for explicitly
# ---------------------------------------------------------------------------


def test_request_asks_for_utc_rather_than_accepting_local_time(settings) -> None:
    _, adapter = fetch([responds(json_body=daily_payload())], settings)
    assert query_of(adapter)["timezone"] == "UTC"


def test_request_asks_for_metric_units_rather_than_accepting_defaults(
    settings,
) -> None:
    _, adapter = fetch([responds(json_body=daily_payload())], settings)
    query = query_of(adapter)
    assert query["temperature_unit"] == "celsius"
    assert query["wind_speed_unit"] == "kmh"
    assert query["precipitation_unit"] == "mm"
    assert query["timeformat"] == "iso8601"
    assert query["cell_selection"] == "land"


def test_request_uses_the_registry_coordinates(settings) -> None:
    _, adapter = fetch([responds(json_body=daily_payload())], settings)
    query = query_of(adapter)
    city = get_city(CITY)
    assert float(query["latitude"]) == pytest.approx(city.lat)
    assert float(query["longitude"]) == pytest.approx(city.lon)
    assert query["start_date"] == START.isoformat()
    assert query["end_date"] == END.isoformat()


@pytest.mark.parametrize(
    ("grain", "expected"),
    [("daily", DAILY_VARIABLES), ("hourly", HOURLY_VARIABLES)],
)
def test_request_names_every_variable_for_the_grain(settings, grain, expected) -> None:
    payload = daily_payload() if grain == "daily" else hourly_payload()
    end = END if grain == "daily" else START
    _, adapter = fetch(
        [responds(json_body=payload)], settings, grain=grain, end=end
    )
    query = query_of(adapter)
    assert query[grain].split(",") == list(expected)
    other = "hourly" if grain == "daily" else "daily"
    assert other not in query


def test_connect_and_read_timeouts_are_both_explicit(settings) -> None:
    """requests applies no timeout at all unless given one."""
    _, adapter = fetch([responds(json_body=daily_payload())], settings)
    assert adapter.calls[0].timeout == (7.0, 23.0)


def test_the_session_identifies_the_project(settings) -> None:
    _, adapter = fetch([responds(json_body=daily_payload())], settings)
    assert adapter.calls[0].request.headers["User-Agent"] == USER_AGENT


# ---------------------------------------------------------------------------
# The response: parsed, UTC, provenance intact
# ---------------------------------------------------------------------------


def test_daily_timestamps_are_midnight_utc(settings) -> None:
    response, _ = fetch([responds(json_body=daily_payload())], settings)
    assert len(response) == DAYS
    for moment in response.times:
        assert moment.tzinfo is not None
        assert moment.utcoffset() == dt.timedelta(0)
        assert (moment.hour, moment.minute, moment.second) == (0, 0, 0)
    assert response.times[0].date() == START
    assert response.times[-1].date() == END


def test_hourly_timestamps_are_on_the_hour_in_utc(settings) -> None:
    response, _ = fetch(
        [responds(json_body=hourly_payload())], settings, end=START, grain="hourly"
    )
    assert len(response) == 24
    assert {m.minute for m in response.times} == {0}
    assert {m.second for m in response.times} == {0}
    assert [m.hour for m in response.times] == list(range(24))


def test_response_records_the_grid_cell_that_answered(settings) -> None:
    """Open-Meteo snaps to the nearest cell; bronze stores what replied."""
    response, _ = fetch([responds(json_body=daily_payload())], settings)
    city = get_city(CITY)
    assert response.latitude == 51.493847
    assert response.longitude == -0.1630249
    assert response.elevation_m == 16.0
    assert (response.latitude, response.longitude) != (city.lat, city.lon)


def test_rows_carry_the_provenance_bronze_requires(settings) -> None:
    response, _ = fetch([responds(json_body=daily_payload())], settings)
    rows = list(response.rows())
    assert len(rows) == DAYS
    first = rows[0]
    assert first["city_id"] == CITY
    assert first["source_url"] == response.url
    assert first["api_latitude"] == response.latitude
    assert first["api_longitude"] == response.longitude
    assert first["api_elevation_m"] == response.elevation_m
    assert first["observation_time"] == response.times[0]
    assert set(DAILY_VARIABLES) <= set(first)


def test_source_url_is_the_exact_request(settings) -> None:
    response, adapter = fetch([responds(json_body=daily_payload())], settings)
    assert response.url == adapter.calls[0].request.url
    assert "timezone=UTC" in response.url


def test_nulls_are_preserved_not_coerced(settings) -> None:
    """A grid point without a variable returns null; that is data, not an error."""
    payload = daily_payload()
    payload["daily"]["snowfall_sum"] = [None] * DAYS
    response, _ = fetch([responds(json_body=payload)], settings)
    assert response.values["snowfall_sum"] == (None, None, None)
    assert response.null_counts()["snowfall_sum"] == DAYS
    assert all(row["snowfall_sum"] is None for row in response.rows())


# ---------------------------------------------------------------------------
# Retry on 5xx and connection errors, capped
# ---------------------------------------------------------------------------


def test_server_error_is_retried_then_succeeds(settings, slept) -> None:
    response, adapter = fetch(
        [
            responds(502, text="<html>bad gateway</html>"),
            responds(503, text="unavailable"),
            responds(json_body=daily_payload()),
        ],
        settings,
    )
    assert len(response) == DAYS
    assert len(adapter.calls) == 3
    assert len(slept) == 2


def test_connection_error_is_retried(settings, slept) -> None:
    response, adapter = fetch(
        [
            requests.exceptions.ConnectionError("connection refused"),
            responds(json_body=daily_payload()),
        ],
        settings,
    )
    assert len(response) == DAYS
    assert len(adapter.calls) == 2


def test_read_timeout_is_retried(settings, slept) -> None:
    response, adapter = fetch(
        [
            requests.exceptions.ReadTimeout("read timed out"),
            responds(json_body=daily_payload()),
        ],
        settings,
    )
    assert len(response) == DAYS
    assert len(adapter.calls) == 2


def test_retries_are_capped_and_the_real_error_survives(settings, slept) -> None:
    """After the cap the caller sees the transport failure, not RetryError."""
    with pytest.raises(ArchiveTransportError) as excinfo:
        fetch([requests.exceptions.ReadTimeout("boom")] * 3, settings)
    assert "connect 7.0s, read 23.0s" in str(excinfo.value)
    assert len(slept) == settings.max_retry_attempts - 1


def test_the_cap_counts_attempts_not_retries(settings, slept) -> None:
    """max_retry_attempts=3 means one try and two retries, not four requests."""
    session, adapter = session_for([responds(500, text="nope")] * 3)
    try:
        with pytest.raises(ArchiveServerError) as excinfo:
            fetch_observations(
                CITY, START, END, session=session, settings=settings
            )
    finally:
        session.close()
    assert len(adapter.calls) == 3
    assert len(slept) == 2
    assert excinfo.value.status_code == 500


def test_undecodable_body_is_retried_as_transport_noise(settings, slept) -> None:
    response, adapter = fetch(
        [
            responds(200, text="{truncated"),
            responds(json_body=daily_payload()),
        ],
        settings,
    )
    assert len(response) == DAYS
    assert len(adapter.calls) == 2


def test_backoff_is_exponential_and_bounded(settings, slept) -> None:
    with pytest.raises(ArchiveServerError):
        fetch([responds(500, text="nope")] * 3, settings)
    assert len(slept) == 2
    # initial=2, exp_base=2, jitter in [0, 2): 2..4 then 4..6.
    assert 2.0 <= slept[0] < 4.0
    assert 4.0 <= slept[1] < 6.0
    assert slept[1] > slept[0]
    assert all(s <= MAX_BACKOFF_SECONDS for s in slept)


def test_backoff_is_jittered() -> None:
    """Fifteen cities failing together must not retry in lockstep."""
    wait = _wait_archive(2.0)
    state = SimpleNamespace(
        attempt_number=1,
        outcome=SimpleNamespace(exception=lambda: ArchiveServerError(
            "boom", status_code=500, url="https://archive-api.test/"
        )),
        idle_for=0.0,
        seconds_since_start=0.0,
    )
    draws = {wait(state) for _ in range(50)}
    assert len(draws) > 1, "backoff is not jittered"
    assert all(2.0 <= d < 4.0 for d in draws)


# ---------------------------------------------------------------------------
# HTTP 429: honour Retry-After rather than backing off blindly
# ---------------------------------------------------------------------------


def test_429_waits_exactly_as_long_as_retry_after_says(settings, slept) -> None:
    response, adapter = fetch(
        [
            responds(429, text="rate limited", headers={"Retry-After": "7"}),
            responds(json_body=daily_payload()),
        ],
        settings,
    )
    assert len(response) == DAYS
    assert slept == [7.0]


def test_429_understands_the_http_date_form_of_retry_after(settings, slept) -> None:
    when = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=12)
    fetch(
        [
            responds(429, text="slow down",
                     headers={"Retry-After": format_datetime(when, usegmt=True)}),
            responds(json_body=daily_payload()),
        ],
        settings,
    )
    assert len(slept) == 1
    assert 9.0 <= slept[0] <= 12.0


def test_retry_after_cannot_park_the_backfill_indefinitely(settings, slept) -> None:
    fetch(
        [
            responds(429, text="come back tomorrow",
                     headers={"Retry-After": "86400"}),
            responds(json_body=daily_payload()),
        ],
        settings,
    )
    assert slept == [MAX_RETRY_AFTER_SECONDS]


def test_429_without_retry_after_falls_back_to_backoff(settings, slept) -> None:
    fetch(
        [
            responds(429, text="rate limited"),
            responds(json_body=daily_payload()),
        ],
        settings,
    )
    assert len(slept) == 1
    assert 2.0 <= slept[0] < 4.0


def test_429_is_reported_with_its_retry_after_when_it_never_clears(
    settings, slept
) -> None:
    with pytest.raises(ArchiveRateLimited) as excinfo:
        fetch([responds(429, text="no", headers={"Retry-After": "5"})] * 3, settings)
    assert excinfo.value.retry_after == 5.0
    assert excinfo.value.status_code == 429
    assert slept == [5.0, 5.0]


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (None, None),
        ("", None),
        ("30", 30.0),
        ("0", 0.0),
        ("-5", 0.0),
        ("not a number", None),
        ("1.5", None),  # delay-seconds is an integer per RFC 9110
    ],
)
def test_parse_retry_after(header, expected) -> None:
    assert _parse_retry_after(header) == expected


def test_parse_retry_after_ignores_a_date_in_the_past() -> None:
    past = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)
    assert _parse_retry_after(format_datetime(past, usegmt=True)) == 0.0


# ---------------------------------------------------------------------------
# Non-retryable 4xx: fail on the first attempt, with the body
# ---------------------------------------------------------------------------


def test_400_raises_immediately_without_retrying(settings, slept) -> None:
    reason = "Parameter 'start_date' is out of allowed range from 1940-01-01"
    with pytest.raises(ArchiveRequestError) as excinfo:
        fetch(
            [responds(400, json_body={"error": True, "reason": reason})],
            settings,
        )
    assert excinfo.value.status_code == 400
    assert reason in str(excinfo.value)
    assert excinfo.value.body == reason
    assert slept == [], "a 4xx must not be retried"


@pytest.mark.parametrize("status", [400, 401, 403, 404, 414, 422])
def test_no_4xx_is_retried(settings, slept, status) -> None:
    with pytest.raises(ArchiveRequestError):
        fetch([responds(status, text="denied")], settings)
    assert slept == []


def test_a_non_json_4xx_body_still_reaches_the_message(settings) -> None:
    with pytest.raises(ArchiveRequestError) as excinfo:
        fetch([responds(403, text="<html>\n  Forbidden by proxy\n</html>")], settings)
    assert "Forbidden by proxy" in str(excinfo.value)


def test_a_long_body_is_truncated_in_the_message(settings) -> None:
    """An HTML error page from a proxy must not become a 5000-character log line."""
    with pytest.raises(ArchiveRequestError) as excinfo:
        fetch([responds(400, text="x" * 5000)], settings)
    body = excinfo.value.body
    assert body.count("x") == 500
    assert body.endswith("…")
    assert body in str(excinfo.value)


def test_error_true_in_a_200_body_is_still_a_request_error(settings, slept) -> None:
    with pytest.raises(ArchiveRequestError) as excinfo:
        fetch(
            [responds(200, json_body={"error": True, "reason": "Data corrupted"})],
            settings,
        )
    assert "Data corrupted" in str(excinfo.value)
    assert slept == []


# ---------------------------------------------------------------------------
# A 200 that is not what was asked for
# ---------------------------------------------------------------------------


def test_a_shifted_timezone_is_refused(settings) -> None:
    """The failure this whole client exists to prevent."""
    payload = daily_payload(utc_offset_seconds=3600, timezone="Europe/London")
    with pytest.raises(ArchiveResponseError, match="utc_offset_seconds"):
        fetch([responds(json_body=payload)], settings)


def test_a_changed_unit_is_refused(settings) -> None:
    payload = daily_payload()
    payload["daily_units"]["temperature_2m_max"] = "°F"
    with pytest.raises(ArchiveResponseError, match="temperature_2m_max"):
        fetch([responds(json_body=payload)], settings)


def test_snowfall_in_millimetres_is_refused(settings) -> None:
    """snowfall_sum is centimetres; every other depth is millimetres."""
    payload = daily_payload()
    payload["daily_units"]["snowfall_sum"] = "mm"
    with pytest.raises(ArchiveResponseError, match="snowfall_sum"):
        fetch([responds(json_body=payload)], settings)


def test_a_missing_variable_is_refused(settings) -> None:
    payload = daily_payload()
    del payload["daily"]["dew_point_2m_mean"]
    with pytest.raises(ArchiveResponseError, match="dew_point_2m_mean"):
        fetch([responds(json_body=payload)], settings)


def test_a_short_range_is_refused(settings) -> None:
    """A range quietly narrower than the one requested would leave a hole."""
    with pytest.raises(ArchiveResponseError, match="expected 3 daily timestamps"):
        fetch([responds(json_body=daily_payload(days=2))], settings)


def test_a_ragged_series_is_refused(settings) -> None:
    payload = daily_payload()
    payload["daily"]["rain_sum"] = [1.0, 2.0]
    with pytest.raises(ArchiveResponseError, match="rain_sum"):
        fetch([responds(json_body=payload)], settings)


def test_a_shifted_range_is_refused(settings) -> None:
    payload = daily_payload(start=dt.date(2023, 2, 1))
    with pytest.raises(ArchiveResponseError, match="spans"):
        fetch([responds(json_body=payload)], settings)


def test_a_missing_grain_block_is_refused(settings) -> None:
    payload = daily_payload()
    del payload["daily"]
    with pytest.raises(ArchiveResponseError, match="daily"):
        fetch([responds(json_body=payload)], settings)


def test_a_non_numeric_value_is_refused(settings) -> None:
    payload = daily_payload()
    payload["daily"]["temperature_2m_mean"] = ["warm", 1.0, 2.0]
    with pytest.raises(ArchiveResponseError, match="non-numeric"):
        fetch([responds(json_body=payload)], settings)


def test_a_bad_response_is_not_retried(settings, slept) -> None:
    """Repeating the request would produce the same wrong answer."""
    payload = daily_payload(utc_offset_seconds=3600)
    with pytest.raises(ArchiveResponseError):
        fetch([responds(json_body=payload)], settings)
    assert slept == []


# ---------------------------------------------------------------------------
# Arguments are checked before a round trip is spent
# ---------------------------------------------------------------------------


def bad_call(settings, **kwargs):
    session, adapter = session_for([])
    try:
        with pytest.raises(ValueError) as excinfo:
            fetch_observations(session=session, settings=settings, **kwargs)
    finally:
        session.close()
    assert adapter.calls == [], "argument validation must not hit the network"
    return str(excinfo.value)


def test_end_before_start_is_rejected(settings) -> None:
    message = bad_call(
        settings, city=CITY, start=dt.date(2023, 5, 1), end=dt.date(2023, 4, 1)
    )
    assert "precedes start" in message


def test_a_range_before_the_archive_is_rejected(settings) -> None:
    message = bad_call(
        settings, city=CITY, start=dt.date(1939, 12, 31), end=dt.date(1940, 6, 1)
    )
    assert str(ARCHIVE_START) in message


def test_a_future_range_is_rejected(settings) -> None:
    tomorrow = dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=1)
    message = bad_call(settings, city=CITY, start=START, end=tomorrow)
    assert "future" in message


def test_an_unknown_grain_is_rejected(settings) -> None:
    message = bad_call(settings, city=CITY, start=START, end=END, grain="monthly")
    assert "daily" in message and "hourly" in message


def test_a_datetime_is_rejected(settings) -> None:
    """Silently truncating one would misreport the range in source_url."""
    message = bad_call(
        settings, city=CITY, start=dt.datetime(2023, 1, 1, 6), end=END
    )
    assert "not datetimes" in message


def test_an_unknown_city_is_rejected(settings) -> None:
    from cities import CityConfigError

    session, adapter = session_for([])
    try:
        with pytest.raises(CityConfigError):
            fetch_observations(
                "atlantis", START, END, session=session, settings=settings
            )
    finally:
        session.close()
    assert adapter.calls == []


def test_a_city_object_is_accepted_without_the_registry_lookup(settings) -> None:
    response, _ = fetch(
        [responds(json_body=daily_payload())], settings, city=get_city(CITY)
    )
    assert response.city_id == CITY


# ---------------------------------------------------------------------------
# The variable list is the bronze schema's contract
# ---------------------------------------------------------------------------


def columns_of(table: str) -> set[str]:
    sql = SCHEMA_SQL.read_text(encoding="utf-8")
    match = re.search(
        rf"create table if not exists bronze_raw\.{table} \((.*?)\n\);",
        sql,
        re.DOTALL,
    )
    assert match, f"could not find the DDL for {table}"
    return {
        m.group(1)
        for m in re.finditer(r"^\s{4}([a-z_][a-z0-9_]*)\s+\S", match.group(1), re.M)
    }


@pytest.mark.parametrize(
    ("table", "variables"),
    [
        ("observations_daily", DAILY_VARIABLES),
        ("observations_hourly", HOURLY_VARIABLES),
    ],
)
def test_every_requested_variable_has_a_bronze_column(table, variables) -> None:
    """A variable with nowhere to land is a request paid for and thrown away."""
    missing = set(variables) - columns_of(table)
    assert not missing, f"{table} has no column for {sorted(missing)}"


@pytest.mark.parametrize(
    ("table", "variables"),
    [
        ("observations_daily", DAILY_VARIABLES),
        ("observations_hourly", HOURLY_VARIABLES),
    ],
)
def test_every_observation_column_is_requested(table, variables) -> None:
    """And a column nothing fills would sit null forever."""
    metadata = {
        "id", "city_id", "observation_time", "ingested_at", "source_url",
        "batch_id", "api_latitude", "api_longitude", "api_elevation_m",
        "constraint",
    }
    unfilled = columns_of(table) - set(variables) - metadata
    assert not unfilled, f"{table} has columns nothing requests: {sorted(unfilled)}"


# ---------------------------------------------------------------------------
# One live request, diffed against the documentation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live_settings() -> Settings:
    """One attempt only — an offline test run should skip, not back off."""
    return dataclasses.replace(get_settings(), max_retry_attempts=1)


@pytest.fixture(scope="module")
def live_year(live_settings):
    try:
        return fetch_observations(
            CITY, dt.date(2023, 1, 1), dt.date(2023, 12, 31), "daily",
            settings=live_settings,
        )
    except ArchiveTransportError as exc:
        pytest.skip(f"archive API unreachable: {exc}")


def test_live_year_returns_every_day(live_year) -> None:
    assert len(live_year) == 365
    assert live_year.times[0].date() == dt.date(2023, 1, 1)
    assert live_year.times[-1].date() == dt.date(2023, 12, 31)


def test_live_units_are_the_documented_ones(live_year) -> None:
    assert dict(live_year.units) == dict(DAILY_UNITS)


def test_live_values_are_physically_plausible(live_year) -> None:
    """A unit swap that kept its label would show up here as absurd numbers."""
    highs = [v for v in live_year.values["temperature_2m_max"] if v is not None]
    lows = [v for v in live_year.values["temperature_2m_min"] if v is not None]
    assert len(highs) == 365
    assert -20.0 < min(lows) and max(highs) < 45.0, "London is not in Fahrenheit"
    assert all(
        low <= high
        for low, high in zip(lows, highs)
        if low is not None and high is not None
    )
    pressures = [v for v in live_year.values["pressure_msl_mean"] if v is not None]
    assert 950.0 < min(pressures) and max(pressures) < 1060.0
    humidity = [v for v in live_year.values["relative_humidity_2m_mean"] if v is not None]
    assert 0 <= min(humidity) and max(humidity) <= 100


def test_live_leap_year_has_the_extra_day(live_settings) -> None:
    """Day-of-year climatology joins are where leap days go wrong."""
    try:
        response = fetch_observations(
            CITY, dt.date(2024, 2, 26), dt.date(2024, 3, 2), "daily",
            settings=live_settings,
        )
    except ArchiveTransportError as exc:
        pytest.skip(f"archive API unreachable: {exc}")
    assert len(response) == 6
    assert dt.date(2024, 2, 29) in {m.date() for m in response.times}
