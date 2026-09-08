"""Typed loader for `config/cities.yml`.

`config/cities.yml` is the single source of truth for the fifteen target
cities. Nothing downstream may hardcode a city name, coordinate, or timezone:
ingestion, dbt seeds, feature construction, and the dashboard all read through
here.

The file is also the Day 8 validation fixture: seven cities carry a dated
extreme-weather event that the climatology must surface as a |Z| > 2.5 anomaly
(§7.2 of docs/proposal.md). Validation is therefore strict by design. A typo in
a coordinate or an IANA zone must fail at load, not silently pull observations
for the wrong grid cell and poison a 30-year baseline.

Usage::

    from cities import load_cities, get_city

    for city in load_cities():
        print(city.id, city.lat, city.lon, city.timezone)

Run ``python cities.py`` for a summary of the loaded set.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import MISSING, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

from config import get_settings

Direction = Literal["hot", "cold"]
SeasonModel = Literal["four_season", "wet_dry", "seasonless"]

_VALID_DIRECTIONS: frozenset[str] = frozenset({"hot", "cold"})
_VALID_SEASON_MODELS: frozenset[str] = frozenset(
    {"four_season", "wet_dry", "seasonless"}
)
# Continent-level, deliberately coarse: this exists to group fifteen cities in
# a dashboard filter, not to encode geography. A finer scheme (UN M49
# subregions, say) would put most of these in a bucket of one.
_VALID_REGIONS: frozenset[str] = frozenset(
    {"Africa", "Asia", "Europe", "North America", "Oceania", "South America"}
)
# Köppen classes are two or three characters: a main group, a precipitation
# letter, and an optional temperature letter.
_KOPPEN_MAIN: frozenset[str] = frozenset("ABCDE")

# The archive reaches back to 1940; anything earlier cannot be validated
# against ingested data.
_EARLIEST_EVENT = dt.date(1940, 1, 1)


class CityConfigError(ValueError):
    """Raised when `config/cities.yml` is malformed.

    Carries the offending city id wherever one is known, because the failure
    that matters is "which entry is wrong", not "something is wrong".
    """


@dataclass(frozen=True)
class ValidationEvent:
    """A dated extreme-weather event the Day 8 gate must reproduce."""

    date: dt.date
    direction: Direction
    description: str
    observed_c: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.date, dt.date):
            raise CityConfigError(
                f"validation_event.date must be a date, got {self.date!r}. "
                "Write it unquoted as YYYY-MM-DD so YAML parses it as a date."
            )
        if self.date < _EARLIEST_EVENT:
            raise CityConfigError(
                f"validation_event.date {self.date} precedes the ERA5 archive "
                f"({_EARLIEST_EVENT}); it can never be validated against "
                "ingested data."
            )
        if self.date > dt.date.today():
            raise CityConfigError(
                f"validation_event.date {self.date} is in the future."
            )
        if self.direction not in _VALID_DIRECTIONS:
            raise CityConfigError(
                f"validation_event.direction must be one of "
                f"{sorted(_VALID_DIRECTIONS)}, got {self.direction!r}."
            )
        if not self.description.strip():
            raise CityConfigError("validation_event.description must not be empty.")
        if self.observed_c is not None and not -95.0 <= self.observed_c <= 60.0:
            raise CityConfigError(
                f"validation_event.observed_c {self.observed_c} is outside the "
                "range of temperatures ever recorded on Earth."
            )


@dataclass(frozen=True)
class City:
    """One target city, validated on construction."""

    id: str
    name: str
    country: str
    country_code: str
    region: str
    lat: float
    lon: float
    elevation_m: float
    timezone: str
    koppen: str
    season_model: SeasonModel
    role: str | None = None
    validation_event: ValidationEvent | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.id.replace("_", "").isalnum():
            raise CityConfigError(
                f"id {self.id!r} must be a non-empty alphanumeric slug "
                "(underscores allowed)."
            )
        if self.id != self.id.lower():
            raise CityConfigError(f"id {self.id!r} must be lowercase.")

        for label, value in (("name", self.name), ("country", self.country)):
            if not str(value).strip():
                raise CityConfigError(f"{self.id}: {label} must not be empty.")

        if len(self.country_code) != 2 or not self.country_code.isupper():
            raise CityConfigError(
                f"{self.id}: country_code {self.country_code!r} must be a "
                "two-letter uppercase ISO 3166-1 alpha-2 code."
            )

        if not -90.0 <= self.lat <= 90.0:
            raise CityConfigError(
                f"{self.id}: lat {self.lat} is outside [-90, 90]."
            )
        if not -180.0 <= self.lon <= 180.0:
            raise CityConfigError(
                f"{self.id}: lon {self.lon} is outside [-180, 180]."
            )
        # A coordinate left at the origin is the classic silent default; it
        # sits in the Gulf of Guinea and would return plausible-looking marine
        # data rather than an error.
        if self.lat == 0.0 and self.lon == 0.0:
            raise CityConfigError(
                f"{self.id}: coordinates are (0, 0), almost certainly a "
                "missing value rather than a real location."
            )

        if not -430.0 <= self.elevation_m <= 5100.0:
            raise CityConfigError(
                f"{self.id}: elevation_m {self.elevation_m} is outside the "
                "range of inhabited elevations [-430, 5100]."
            )

        try:
            ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise CityConfigError(
                f"{self.id}: timezone {self.timezone!r} is not a known IANA "
                "zone. Use a canonical name such as 'America/Phoenix', not an "
                "abbreviation or a fixed UTC offset."
            ) from exc

        koppen = self.koppen
        if not 2 <= len(koppen) <= 3 or koppen[0] not in _KOPPEN_MAIN:
            raise CityConfigError(
                f"{self.id}: koppen {koppen!r} is not a valid classification "
                f"(main group must be one of {sorted(_KOPPEN_MAIN)})."
            )

        if self.region not in _VALID_REGIONS:
            raise CityConfigError(
                f"{self.id}: region must be one of {sorted(_VALID_REGIONS)}, "
                f"got {self.region!r}."
            )

        if self.season_model not in _VALID_SEASON_MODELS:
            raise CityConfigError(
                f"{self.id}: season_model must be one of "
                f"{sorted(_VALID_SEASON_MODELS)}, got {self.season_model!r}."
            )

    @property
    def hemisphere(self) -> Literal["north", "south"]:
        """Derived, never configured; the coordinate is the only truth."""
        return "north" if self.lat >= 0 else "south"

    @property
    def tzinfo(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


@dataclass(frozen=True)
class CityRegistry:
    """The full validated set, with the cross-entry invariants enforced."""

    cities: tuple[City, ...]
    _by_id: dict[str, City] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_id", {c.id: c for c in self.cities})

    def __iter__(self):
        return iter(self.cities)

    def __len__(self) -> int:
        return len(self.cities)

    def __getitem__(self, city_id: str) -> City:
        try:
            return self._by_id[city_id]
        except KeyError:
            raise CityConfigError(
                f"unknown city id {city_id!r}; known ids: "
                f"{sorted(self._by_id)}"
            ) from None

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(c.id for c in self.cities)

    def northern(self) -> tuple[City, ...]:
        return tuple(c for c in self.cities if c.hemisphere == "north")

    def southern(self) -> tuple[City, ...]:
        return tuple(c for c in self.cities if c.hemisphere == "south")

    def with_validation_events(self) -> tuple[City, ...]:
        """The Day 8 gate fixture: cities carrying a dated event."""
        return tuple(c for c in self.cities if c.validation_event is not None)


def _build_city(raw: Any, index: int) -> City:
    if not isinstance(raw, dict):
        raise CityConfigError(
            f"cities[{index}] must be a mapping, got {type(raw).__name__}."
        )

    known = set(City.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise CityConfigError(
            f"cities[{index}] ({raw.get('id', '?')}): unknown key(s) "
            f"{sorted(unknown)}. Known keys: {sorted(known)}."
        )

    # A field with no default is required; `role` and `validation_event`
    # default to None and are optional. Note MISSING, not None: a field
    # without a default has `default is MISSING`, and testing against None
    # here would silently require nothing at all.
    required = {
        name
        for name, f in City.__dataclass_fields__.items()
        if f.default is MISSING
    }
    missing = required - set(raw)
    if missing:
        raise CityConfigError(
            f"cities[{index}] ({raw.get('id', '?')}): missing required key(s) "
            f"{sorted(missing)}."
        )

    event_raw = raw.get("validation_event")
    event = None
    if event_raw is not None:
        if not isinstance(event_raw, dict):
            raise CityConfigError(
                f"{raw['id']}: validation_event must be a mapping."
            )
        event_known = set(ValidationEvent.__dataclass_fields__)
        event_unknown = set(event_raw) - event_known
        if event_unknown:
            raise CityConfigError(
                f"{raw['id']}: validation_event has unknown key(s) "
                f"{sorted(event_unknown)}."
            )
        event_missing = {"date", "direction", "description"} - set(event_raw)
        if event_missing:
            raise CityConfigError(
                f"{raw['id']}: validation_event missing "
                f"{sorted(event_missing)}."
            )
        event = ValidationEvent(**event_raw)

    numeric = {}
    for key in ("lat", "lon", "elevation_m"):
        try:
            numeric[key] = float(raw[key])
        except (TypeError, ValueError) as exc:
            raise CityConfigError(
                f"{raw['id']}: {key} must be a number, got {raw[key]!r}."
            ) from exc

    return City(
        id=str(raw["id"]),
        name=str(raw["name"]),
        country=str(raw["country"]),
        country_code=str(raw["country_code"]),
        region=str(raw["region"]),
        timezone=str(raw["timezone"]),
        koppen=str(raw["koppen"]),
        season_model=str(raw["season_model"]),
        role=str(raw["role"]).strip() if raw.get("role") else None,
        validation_event=event,
        **numeric,
    )


def _load(path: Path) -> CityRegistry:
    if not path.exists():
        raise CityConfigError(
            f"city config not found at {path}. CITIES_CONFIG_PATH in .env "
            "points here."
        )

    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise CityConfigError(f"{path} is not valid YAML: {exc}") from exc

    if not isinstance(document, dict) or "cities" not in document:
        raise CityConfigError(f"{path} must be a mapping with a 'cities' key.")

    raw_cities = document["cities"]
    if not isinstance(raw_cities, list) or not raw_cities:
        raise CityConfigError(f"{path}: 'cities' must be a non-empty list.")

    cities = tuple(_build_city(raw, i) for i, raw in enumerate(raw_cities))

    seen: dict[str, int] = {}
    for i, city in enumerate(cities):
        if city.id in seen:
            raise CityConfigError(
                f"duplicate city id {city.id!r} at cities[{seen[city.id]}] and "
                f"cities[{i}]. Ids key the warehouse dimension and must be "
                "unique."
            )
        seen[city.id] = i

    # Two entries resolving to the same grid cell would double-count a city
    # under different names.
    coords: dict[tuple[float, float], str] = {}
    for city in cities:
        key = (round(city.lat, 3), round(city.lon, 3))
        if key in coords:
            raise CityConfigError(
                f"{city.id} and {coords[key]} share coordinates {key}; they "
                "would resolve to the same ERA5 grid cell."
            )
        coords[key] = city.id

    return CityRegistry(cities=cities)


@lru_cache(maxsize=1)
def load_cities() -> CityRegistry:
    """Load, validate, and cache the city registry."""
    return _load(get_settings().cities_config_path)


def get_city(city_id: str) -> City:
    """Look up one city by id, raising with the known ids if absent."""
    return load_cities()[city_id]


def reload_cities() -> CityRegistry:
    """Drop the cache and re-read from disk. Intended for tests."""
    load_cities.cache_clear()
    return load_cities()


if __name__ == "__main__":
    registry = load_cities()
    north, south = registry.northern(), registry.southern()
    events = registry.with_validation_events()

    print(f"{len(registry)} cities: {len(north)} north / {len(south)} south\n")
    header = f"  {'id':14} {'lat':>8} {'lon':>9} {'elev':>6}  {'koppen':6} {'timezone':28} event"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for c in registry:
        ev = c.validation_event.date.isoformat() if c.validation_event else ""
        print(
            f"  {c.id:14} {c.lat:8.4f} {c.lon:9.4f} {c.elevation_m:5.0f}m  "
            f"{c.koppen:6} {c.timezone:28} {ev}"
        )

    lats = [c.lat for c in registry]
    elevs = [c.elevation_m for c in registry]
    print(
        f"\n  latitude   {max(lats):.2f}°N ({max(registry, key=lambda c: c.lat).id})"
        f" -> {abs(min(lats)):.2f}°S ({min(registry, key=lambda c: c.lat).id})"
    )
    print(
        f"  elevation  {min(elevs):.0f} m ({min(registry, key=lambda c: c.elevation_m).id})"
        f" -> {max(elevs):.0f} m ({max(registry, key=lambda c: c.elevation_m).id})"
    )
    print(f"  koppen     {sorted({c.koppen for c in registry})}")
    print(f"  seasons    {sorted({c.season_model for c in registry})}")
    print(f"  events     {len(events)}: {', '.join(c.id for c in events)}")
