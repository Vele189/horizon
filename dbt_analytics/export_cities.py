"""Writes `seeds/cities.csv` from `config/cities.yml`.

dbt needs the city metadata in the warehouse, above all each city's IANA
timezone, so a local-time view is possible without re-deriving it, and
`cities.yml` is the single source of that. Rather than maintain a second copy
by hand, this generates the seed and a test asserts the two agree.

    python dbt_analytics/export_cities.py           # regenerate
    python dbt_analytics/export_cities.py --check   # exit 1 if stale
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cities import load_cities  # noqa: E402

SEED = Path(__file__).resolve().parent / "seeds" / "cities.csv"

#: Ordered deliberately: identity, geography, then classification. `hemisphere`
#: is derived from the latitude rather than configured, since the coordinate is
#: the only truth, but it is materialised here because a season mapping needs it
#: and re-deriving `lat >= 0` in every model invites one of them to get it
#: backwards.
FIELDS = (
    "city_id",
    "name",
    "country",
    "country_code",
    "region",
    "latitude",
    "longitude",
    "elevation_m",
    "timezone",
    "hemisphere",
    "koppen",
    "season_model",
    "role",
)


def render() -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for city in load_cities():
        writer.writerow(
            {
                "city_id": city.id,
                "name": city.name,
                "country": city.country,
                "country_code": city.country_code,
                "region": city.region,
                "latitude": city.lat,
                "longitude": city.lon,
                "elevation_m": city.elevation_m,
                "timezone": city.timezone,
                "hemisphere": city.hemisphere,
                "koppen": city.koppen,
                "season_model": city.season_model,
                "role": city.role or "",
            }
        )
    return buffer.getvalue()


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="report staleness instead of writing"
    )
    args = parser.parse_args(argv)

    rendered = render()
    if args.check:
        current = SEED.read_text(encoding="utf-8") if SEED.exists() else ""
        if current == rendered:
            print(f"{SEED} is current")
            return 0
        print(f"{SEED} is stale; run without --check", file=sys.stderr)
        return 1

    SEED.parent.mkdir(parents=True, exist_ok=True)
    SEED.write_text(rendered, encoding="utf-8")
    print(f"wrote {SEED} ({len(load_cities())} cities)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
