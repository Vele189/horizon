"""Connectivity smoke check for both warehouse targets.

Proves the claim FND-03 exists to establish: local Docker Postgres and Neon
serving are reached by *the same code path*, differing only by which
environment variable supplies the connection string. If this script connects
to both, nothing downstream needs a target-specific branch.

    python tests/check_connection.py            # both targets
    python tests/check_connection.py --target serving
    python tests/check_connection.py --target serving --cold

`--cold` reports the wall-clock time to first query, which is what you want
after Neon has scaled compute to zero (five minutes idle on the free plan).

Connection strings are never printed. Only the masked form from config.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2  # noqa: E402

from config import ConfigError, get_settings, mask_secret  # noqa: E402

TARGETS = ("local", "serving")


def _url_for(target: str) -> str:
    settings = get_settings()
    if target == "local":
        return settings.require_database_url()
    return settings.require_serving_database_url()


def check(target: str, *, cold: bool = False) -> bool:
    """Connect to one target and report what came back. Returns success."""
    label = "local (Docker Postgres)" if target == "local" else "serving (Neon)"
    print(f"\n=== {target}: {label} ===")

    try:
        url = _url_for(target)
    except ConfigError as exc:
        print(f"  SKIP  {exc}")
        return False

    print(f"  url            {mask_secret(url)}")
    if target == "serving":
        print(f"  pooled         {'yes' if '-pooler.' in url else 'NO, not the pooled endpoint'}")
        print(f"  sslmode        {'require' if 'sslmode=require' in url else 'MISSING'}")

    started = time.perf_counter()
    try:
        conn = psycopg2.connect(url)
    except Exception as exc:  # noqa: BLE001 - diagnostic surface
        print(f"  FAIL  connection refused: {type(exc).__name__}: {exc}")
        return False

    connect_ms = (time.perf_counter() - started) * 1000
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "select version(), current_database(), current_user, "
                "current_setting('TimeZone'), now()"
            )
            version, database, user, timezone, server_now = cur.fetchone()
        first_query_ms = (time.perf_counter() - started) * 1000
        ssl_in_use = getattr(conn.info, "ssl_in_use", None)
    finally:
        conn.close()

    print(f"  server         {version.split(' on ')[0]}")
    print(f"  database|user  {database} | {user}")
    print(f"  timezone       {timezone}")
    print(f"  server time    {server_now}")
    print(f"  tls in use     {ssl_in_use}")
    print(f"  connect        {connect_ms:8.1f} ms")
    print(f"  first query    {first_query_ms:8.1f} ms" + ("   <-- cold start" if cold else ""))
    print("  OK")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=TARGETS, help="default: both")
    parser.add_argument(
        "--cold",
        action="store_true",
        help="label the timing as a cold start (run after 5+ minutes idle)",
    )
    args = parser.parse_args()

    targets = (args.target,) if args.target else TARGETS
    results = {t: check(t, cold=args.cold) for t in targets}

    print("\n=== summary ===")
    for target, ok in results.items():
        print(f"  {target:8} {'OK' if ok else 'FAILED'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
