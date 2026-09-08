"""A scripted HTTP transport and payload builders, shared by the ingestion tests.

The transport is a ``requests`` adapter mounted on a real
:class:`requests.Session`, not a mock of the client. Everything the client
actually relies on (parameter encoding, the timeout tuple, header handling,
status codes) runs for real; only the socket is replaced.
"""

from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace

import requests
from requests.adapters import BaseAdapter
from requests.structures import CaseInsensitiveDict

from ingestion.client import DAILY_UNITS, DAILY_VARIABLES, HOURLY_UNITS, HOURLY_VARIABLES

LONDON_GRID = (51.493847, -0.1630249, 16.0)

#: Variables the API returns as JSON integers, and the bronze schema declares
#: smallint. Fixtures must respect that: a payload full of floats would exercise
#: a COPY path real data never takes, and hide the one it does.
INTEGER_VARIABLES = frozenset(
    {
        "weather_code",
        "cloud_cover",
        "cloud_cover_mean",
        "relative_humidity_2m",
        "relative_humidity_2m_mean",
        "wind_direction_10m",
        "wind_direction_10m_dominant",
    }
)


def _series(name: str, count: int) -> list:
    """Values shaped like the API's: integers where it sends integers."""
    if name in INTEGER_VARIABLES:
        return [i % 100 for i in range(count)]
    return [float(i) for i in range(count)]


class ScriptedAdapter(BaseAdapter):
    """Replays a fixed script of responses and exceptions, recording calls.

    Each entry is either a response factory (a callable taking the prepared
    request) or an exception instance to raise. The script must be consumed
    exactly: a test that expects three attempts and gets two fails loudly
    rather than passing on a coincidence.
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


def session_for(script: list) -> tuple[requests.Session, ScriptedAdapter]:
    adapter = ScriptedAdapter(script)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session, adapter


def daily_payload(days: int, start: dt.date, **overrides) -> dict:
    times = [(start + dt.timedelta(days=i)).isoformat() for i in range(days)]
    latitude, longitude, elevation = LONDON_GRID
    payload = {
        "latitude": latitude,
        "longitude": longitude,
        "generationtime_ms": 1.5,
        "utc_offset_seconds": 0,
        "timezone": "GMT",
        "timezone_abbreviation": "GMT",
        "elevation": elevation,
        "daily_units": {"time": "iso8601", **DAILY_UNITS},
        "daily": {
            "time": times,
            **{name: _series(name, days) for name in DAILY_VARIABLES},
        },
    }
    payload.update(overrides)
    return payload


def hourly_payload(days: int, start: dt.date, **overrides) -> dict:
    hours = days * 24
    base = dt.datetime.combine(start, dt.time())
    times = [(base + dt.timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M")
             for i in range(hours)]
    latitude, longitude, elevation = LONDON_GRID
    payload = {
        "latitude": latitude,
        "longitude": longitude,
        "generationtime_ms": 2.5,
        "utc_offset_seconds": 0,
        "timezone": "GMT",
        "timezone_abbreviation": "GMT",
        "elevation": elevation,
        "hourly_units": {"time": "iso8601", **HOURLY_UNITS},
        "hourly": {
            "time": times,
            **{name: _series(name, hours) for name in HOURLY_VARIABLES},
        },
    }
    payload.update(overrides)
    return payload
