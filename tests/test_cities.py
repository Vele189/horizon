"""Tests for the city registry.

`config/cities.yml` is a test fixture as much as a config file: the Day 8
validation gate reads its seven dated events. These tests cover two things:
the invariants decision D2 fixed for the selection, and the loader's refusal to
accept the mistakes that would silently corrupt a 30-year baseline.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cities import (  # noqa: E402
    CityConfigError,
    CityRegistry,
    _load,
    load_cities,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CITIES_YML = REPO_ROOT / "config" / "cities.yml"


@pytest.fixture(scope="module")
def registry() -> CityRegistry:
    return load_cities()


@pytest.fixture
def raw_document() -> dict:
    return yaml.safe_load(CITIES_YML.read_text(encoding="utf-8"))


def write_registry(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "cities.yml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The selection itself: decision D2
# ---------------------------------------------------------------------------


def test_fifteen_cities(registry: CityRegistry) -> None:
    assert len(registry) == 15


def test_hemisphere_split_is_ten_north_five_south(registry: CityRegistry) -> None:
    assert len(registry.northern()) == 10
    assert len(registry.southern()) == 5


def test_hemisphere_is_derived_from_latitude(registry: CityRegistry) -> None:
    for city in registry:
        assert city.hemisphere == ("north" if city.lat >= 0 else "south")


def test_ids_are_unique(registry: CityRegistry) -> None:
    assert len(set(registry.ids)) == len(registry)


def test_koppen_spans_af_to_dfb(registry: CityRegistry) -> None:
    classes = {c.koppen for c in registry}
    assert "Af" in classes, "tropical rainforest endpoint missing"
    assert "Dfb" in classes, "continental endpoint missing"
    assert len({k[0] for k in classes}) >= 4, "too few main Köppen groups"


def test_latitude_and_elevation_envelope(registry: CityRegistry) -> None:
    lats = [c.lat for c in registry]
    elevations = [c.elevation_m for c in registry]
    assert max(lats) == pytest.approx(64.15, abs=0.05)   # Reykjavík
    assert min(lats) == pytest.approx(-36.85, abs=0.05)  # Auckland
    assert min(elevations) == 10      # Lagos
    assert max(elevations) == 1753    # Johannesburg


def test_timezone_traps_present(registry: CityRegistry) -> None:
    """The four cases §D2 selected the set to exercise."""
    zones = {c.id: c.timezone for c in registry}
    assert zones["phoenix"] == "America/Phoenix"          # no DST, US longitude
    assert zones["reykjavik"] == "Atlantic/Reykjavik"     # UTC+0 year-round
    assert zones["delhi"] == "Asia/Kolkata"               # UTC+05:30
    assert zones["sydney"] == "Australia/Sydney"          # southern DST
    assert zones["auckland"] == "Pacific/Auckland"        # southern DST


def test_non_temperate_cities_do_not_claim_four_seasons(
    registry: CityRegistry,
) -> None:
    assert registry["singapore"].season_model == "seasonless"
    assert registry["lagos"].season_model == "wet_dry"


# ---------------------------------------------------------------------------
# The Day 8 gate fixture
# ---------------------------------------------------------------------------

EXPECTED_EVENTS = {
    "tokyo": dt.date(2018, 7, 23),
    "portland": dt.date(2021, 6, 28),
    "london": dt.date(2022, 7, 19),
    "moscow": dt.date(2010, 7, 29),
    "sao_paulo": dt.date(2021, 7, 30),
    "sydney": dt.date(2020, 1, 4),
    "buenos_aires": dt.date(2022, 1, 11),
}


def test_seven_validation_events(registry: CityRegistry) -> None:
    assert len(registry.with_validation_events()) == 7


def test_validation_events_are_correctly_dated(registry: CityRegistry) -> None:
    actual = {
        c.id: c.validation_event.date for c in registry.with_validation_events()
    }
    assert actual == EXPECTED_EVENTS


def test_validation_events_include_a_cold_anomaly(registry: CityRegistry) -> None:
    """A gate that only tests heat cannot prove the Z-score is two-tailed."""
    directions = {
        c.validation_event.direction for c in registry.with_validation_events()
    }
    assert directions == {"hot", "cold"}


def test_validation_events_fall_within_the_ingested_window(
    registry: CityRegistry,
) -> None:
    """Daily observations start in 1995; an earlier event cannot be checked."""
    for city in registry.with_validation_events():
        assert city.validation_event.date >= dt.date(1995, 1, 1), city.id
        assert city.validation_event.date <= dt.date.today(), city.id


def test_every_event_has_a_description(registry: CityRegistry) -> None:
    for city in registry.with_validation_events():
        assert len(city.validation_event.description.strip()) > 40, city.id


# ---------------------------------------------------------------------------
# Loader rejects what would silently corrupt the baseline
# ---------------------------------------------------------------------------


def test_rejects_duplicate_ids(tmp_path: Path, raw_document: dict) -> None:
    raw_document["cities"].append(dict(raw_document["cities"][0]))
    raw_document["cities"][-1]["lat"] = 1.0  # avoid tripping the coord check
    raw_document["cities"][-1]["lon"] = 2.0
    with pytest.raises(CityConfigError, match="duplicate city id"):
        _load(write_registry(tmp_path, raw_document))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lat", 91.0),
        ("lat", -90.5),
        ("lon", 180.5),
        ("lon", -181.0),
    ],
)
def test_rejects_out_of_range_coordinates(
    tmp_path: Path, raw_document: dict, field: str, value: float
) -> None:
    raw_document["cities"][0][field] = value
    with pytest.raises(CityConfigError, match="outside"):
        _load(write_registry(tmp_path, raw_document))


def test_rejects_null_island(tmp_path: Path, raw_document: dict) -> None:
    """(0, 0) is the classic missing-value default and returns marine data."""
    raw_document["cities"][0]["lat"] = 0.0
    raw_document["cities"][0]["lon"] = 0.0
    with pytest.raises(CityConfigError, match=r"\(0, 0\)"):
        _load(write_registry(tmp_path, raw_document))


@pytest.mark.parametrize(
    "timezone",
    ["Europe/Nowhere", "PST", "UTC+5:30", "America/Phoenix "],
)
def test_rejects_unknown_timezone_strings(
    tmp_path: Path, raw_document: dict, timezone: str
) -> None:
    raw_document["cities"][0]["timezone"] = timezone
    with pytest.raises(CityConfigError, match="not a known IANA zone"):
        _load(write_registry(tmp_path, raw_document))


def test_rejects_duplicate_coordinates(tmp_path: Path, raw_document: dict) -> None:
    first, second = raw_document["cities"][0], raw_document["cities"][1]
    second["lat"], second["lon"] = first["lat"], first["lon"]
    with pytest.raises(CityConfigError, match="same ERA5 grid cell"):
        _load(write_registry(tmp_path, raw_document))


def test_rejects_unknown_keys(tmp_path: Path, raw_document: dict) -> None:
    """A typo'd key must fail rather than be silently dropped."""
    raw_document["cities"][0]["lattitude"] = 12.0
    with pytest.raises(CityConfigError, match="unknown key"):
        _load(write_registry(tmp_path, raw_document))


def test_rejects_missing_required_keys(tmp_path: Path, raw_document: dict) -> None:
    del raw_document["cities"][0]["timezone"]
    with pytest.raises(CityConfigError, match="missing required key"):
        _load(write_registry(tmp_path, raw_document))


def test_rejects_bad_koppen(tmp_path: Path, raw_document: dict) -> None:
    raw_document["cities"][0]["koppen"] = "Zx"
    with pytest.raises(CityConfigError, match="not a valid classification"):
        _load(write_registry(tmp_path, raw_document))


def test_rejects_future_event_date(tmp_path: Path, raw_document: dict) -> None:
    target = next(c for c in raw_document["cities"] if "validation_event" in c)
    target["validation_event"]["date"] = dt.date.today() + dt.timedelta(days=1)
    with pytest.raises(CityConfigError, match="in the future"):
        _load(write_registry(tmp_path, raw_document))


def test_rejects_event_before_the_archive(tmp_path: Path, raw_document: dict) -> None:
    target = next(c for c in raw_document["cities"] if "validation_event" in c)
    target["validation_event"]["date"] = dt.date(1939, 1, 1)
    with pytest.raises(CityConfigError, match="precedes the ERA5 archive"):
        _load(write_registry(tmp_path, raw_document))


def test_rejects_bad_event_direction(tmp_path: Path, raw_document: dict) -> None:
    target = next(c for c in raw_document["cities"] if "validation_event" in c)
    target["validation_event"]["direction"] = "warm"
    with pytest.raises(CityConfigError, match="direction must be one of"):
        _load(write_registry(tmp_path, raw_document))


def test_unknown_city_lookup_lists_known_ids(registry: CityRegistry) -> None:
    with pytest.raises(CityConfigError, match="unknown city id"):
        registry["atlantis"]
