"""Apply the bronze DDL to whichever warehouse DATABASE_URL points at.

The DDL is idempotent, so this is safe to run repeatedly and safe to call from
`run_pipeline.py` before ingestion. It reads the connection through config.py
rather than the environment directly, which is what keeps the local/Neon switch
a matter of configuration.

    python ingestion/apply_schema.py            # apply
    python ingestion/apply_schema.py --check    # report state, change nothing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings  # noqa: E402

SCHEMA_SQL = Path(__file__).resolve().parent / "schema.sql"
EXPECTED_SCHEMAS = ("bronze_raw", "silver_staging", "gold_marts")
EXPECTED_TABLES = (
    ("bronze_raw", "observations_daily"),
    ("bronze_raw", "observations_hourly"),
)


def report(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "select nspname from pg_namespace where nspname = any(%s) order by 1",
            (list(EXPECTED_SCHEMAS),),
        )
        present = [r[0] for r in cur.fetchall()]
        for schema in EXPECTED_SCHEMAS:
            print(f"  schema  {schema:16} {'present' if schema in present else 'MISSING'}")

        for schema, table in EXPECTED_TABLES:
            cur.execute(
                "select count(*) from information_schema.columns "
                "where table_schema = %s and table_name = %s",
                (schema, table),
            )
            columns = cur.fetchone()[0]
            cur.execute(
                "select count(*) from pg_indexes "
                "where schemaname = %s and tablename = %s",
                (schema, table),
            )
            indexes = cur.fetchone()[0]
            cur.execute(
                "select count(*) from information_schema.columns "
                "where table_schema = %s and table_name = %s and data_type = 'jsonb'",
                (schema, table),
            )
            jsonb = cur.fetchone()[0]
            state = f"{columns} columns, {indexes} indexes" if columns else "MISSING"
            print(f"  table   {schema}.{table:22} {state}")
            if jsonb:
                print(f"          WARNING: {jsonb} jsonb column(s) present")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="report state without applying"
    )
    args = parser.parse_args()

    settings = get_settings()
    url = settings.require_database_url()
    print(f"target: {settings.environment}")

    with psycopg2.connect(url) as conn:
        conn.autocommit = False
        if not args.check:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_SQL.read_text(encoding="utf-8"))
            conn.commit()
            print(f"applied: {SCHEMA_SQL.name}")
        report(conn)
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
