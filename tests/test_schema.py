"""Tests for the bronze landing schema.

These need a reachable database — the local Docker Postgres from FND-02. They
skip rather than fail when DATABASE_URL is unset or unreachable, so CI without
a warehouse stays green.

Every write happens inside a transaction that is rolled back, so the tests
leave no rows behind.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

psycopg2 = pytest.importorskip("psycopg2")

from config import ConfigError, get_settings  # noqa: E402

SCHEMA_SQL = Path(__file__).resolve().parent.parent / "ingestion" / "schema.sql"

METADATA_COLUMNS = ("city_id", "observation_time", "ingested_at", "batch_id")
BRONZE_TABLES = ("observations_daily", "observations_hourly")


@pytest.fixture(scope="module")
def conn():
    try:
        url = get_settings().require_database_url()
    except ConfigError as exc:
        pytest.skip(f"no DATABASE_URL: {exc}")
    try:
        connection = psycopg2.connect(url, connect_timeout=5)
    except psycopg2.OperationalError as exc:
        pytest.skip(f"database unreachable: {exc}")
    yield connection
    connection.close()


@pytest.fixture
def tx(conn):
    """A cursor whose work is always rolled back."""
    conn.rollback()
    with conn.cursor() as cur:
        yield cur
    conn.rollback()


def catalog_fingerprint(cur) -> str:
    cur.execute(
        """
        select table_schema, table_name, column_name, data_type, is_nullable,
               column_default
        from information_schema.columns
        where table_schema in ('bronze_raw', 'silver_staging', 'gold_marts')
        order by 1, 2, 3
        """
    )
    columns = cur.fetchall()
    cur.execute(
        """
        select schemaname, indexname, indexdef from pg_indexes
        where schemaname in ('bronze_raw', 'silver_staging', 'gold_marts')
        order by 1, 2
        """
    )
    indexes = cur.fetchall()
    cur.execute(
        """
        select conname, pg_get_constraintdef(oid) from pg_constraint
        where connamespace = 'bronze_raw'::regnamespace order by 1
        """
    )
    constraints = cur.fetchall()
    blob = repr(columns) + repr(indexes) + repr(constraints)
    return hashlib.sha256(blob.encode()).hexdigest()


def daily_row(observation_time: dt.datetime) -> tuple:
    return ("test_city", observation_time, str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("schema", ["bronze_raw", "silver_staging", "gold_marts"])
def test_medallion_schemas_exist(tx, schema: str) -> None:
    tx.execute("select 1 from pg_namespace where nspname = %s", (schema,))
    assert tx.fetchone() is not None


@pytest.mark.parametrize("table", BRONZE_TABLES)
def test_bronze_tables_exist(tx, table: str) -> None:
    tx.execute(
        "select 1 from information_schema.tables "
        "where table_schema = 'bronze_raw' and table_name = %s",
        (table,),
    )
    assert tx.fetchone() is not None


@pytest.mark.parametrize("table", BRONZE_TABLES)
def test_every_row_carries_the_five_metadata_columns(tx, table: str) -> None:
    tx.execute(
        "select column_name, is_nullable from information_schema.columns "
        "where table_schema = 'bronze_raw' and table_name = %s",
        (table,),
    )
    nullability = dict(tx.fetchall())
    for column in METADATA_COLUMNS:
        assert column in nullability, f"{table} is missing {column}"
        assert nullability[column] == "NO", f"{table}.{column} must be NOT NULL"


def test_no_jsonb_payload_anywhere_in_bronze(tx) -> None:
    """Raw payloads go to gzip on disk, never into a column (§5.4)."""
    tx.execute(
        "select table_name, column_name, data_type "
        "from information_schema.columns "
        "where table_schema = 'bronze_raw' and data_type in ('jsonb', 'json')"
    )
    assert tx.fetchall() == []


@pytest.mark.parametrize("table", BRONZE_TABLES)
def test_dedup_index_leads_with_city_and_time(tx, table: str) -> None:
    tx.execute(
        "select indexdef from pg_indexes "
        "where schemaname = 'bronze_raw' and tablename = %s",
        (table,),
    )
    definitions = [row[0] for row in tx.fetchall()]
    assert any(
        "(city_id, observation_time" in d for d in definitions
    ), f"{table} has no (city_id, observation_time...) index: {definitions}"


@pytest.mark.parametrize("table", BRONZE_TABLES)
def test_natural_key_is_not_unique(tx, table: str) -> None:
    """Bronze is append-only; a unique key would reject a legitimate re-ingest."""
    tx.execute(
        "select indexdef from pg_indexes "
        "where schemaname = 'bronze_raw' and tablename = %s",
        (table,),
    )
    for definition in (row[0] for row in tx.fetchall()):
        if "(city_id, observation_time" in definition:
            assert "UNIQUE" not in definition.upper()


def test_daily_carries_the_gold_fact_columns(tx) -> None:
    """The columns fact_weather_observations needs must land in bronze."""
    tx.execute(
        "select column_name from information_schema.columns "
        "where table_schema = 'bronze_raw' and table_name = 'observations_daily'"
    )
    columns = {row[0] for row in tx.fetchall()}
    required = {
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
        "wind_gusts_10m_max",
        "surface_pressure_mean",
        "relative_humidity_2m_mean",
    }
    assert required <= columns, f"missing: {sorted(required - columns)}"


def test_hourly_carries_the_storm_dynamics_columns(tx) -> None:
    tx.execute(
        "select column_name from information_schema.columns "
        "where table_schema = 'bronze_raw' and table_name = 'observations_hourly'"
    )
    columns = {row[0] for row in tx.fetchall()}
    required = {"temperature_2m", "surface_pressure", "pressure_msl", "wind_speed_10m"}
    assert required <= columns, f"missing: {sorted(required - columns)}"


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


def test_daily_accepts_midnight_utc(tx) -> None:
    tx.execute(
        "insert into bronze_raw.observations_daily "
        "(city_id, observation_time, batch_id) values (%s, %s, %s)",
        daily_row(dt.datetime(2022, 7, 19, tzinfo=dt.timezone.utc)),
    )


def test_daily_rejects_non_midnight(tx) -> None:
    """A daily row at 12:00 means timezone=UTC was omitted from the request."""
    with pytest.raises(psycopg2.errors.CheckViolation):
        tx.execute(
            "insert into bronze_raw.observations_daily "
            "(city_id, observation_time, batch_id) values (%s, %s, %s)",
            daily_row(dt.datetime(2022, 7, 19, 12, 0, tzinfo=dt.timezone.utc)),
        )


def test_hourly_rejects_off_the_hour(tx) -> None:
    with pytest.raises(psycopg2.errors.CheckViolation):
        tx.execute(
            "insert into bronze_raw.observations_hourly "
            "(city_id, observation_time, batch_id) values (%s, %s, %s)",
            daily_row(dt.datetime(2022, 7, 19, 12, 30, tzinfo=dt.timezone.utc)),
        )


def test_blank_city_id_rejected(tx) -> None:
    with pytest.raises(psycopg2.errors.CheckViolation):
        tx.execute(
            "insert into bronze_raw.observations_daily "
            "(city_id, observation_time, batch_id) values (%s, %s, %s)",
            ("   ", dt.datetime(2022, 7, 19, tzinfo=dt.timezone.utc),
             str(uuid.uuid4())),
        )


def test_append_only_allows_duplicate_natural_keys(tx) -> None:
    """Re-ingesting a range must insert, not conflict; silver deduplicates."""
    when = dt.datetime(2022, 7, 19, tzinfo=dt.timezone.utc)
    for _ in range(2):
        tx.execute(
            "insert into bronze_raw.observations_daily "
            "(city_id, observation_time, batch_id) values (%s, %s, %s)",
            daily_row(when),
        )
    tx.execute(
        "select count(*) from bronze_raw.observations_daily "
        "where city_id = 'test_city' and observation_time = %s",
        (when,),
    )
    assert tx.fetchone()[0] == 2


def test_ingested_at_defaults_to_now(tx) -> None:
    tx.execute(
        "insert into bronze_raw.observations_daily "
        "(city_id, observation_time, batch_id) values (%s, %s, %s) "
        "returning ingested_at",
        daily_row(dt.datetime(2022, 7, 19, tzinfo=dt.timezone.utc)),
    )
    ingested_at = tx.fetchone()[0]
    assert abs((dt.datetime.now(dt.timezone.utc) - ingested_at).total_seconds()) < 60


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_reapplying_ddl_is_a_no_op(conn) -> None:
    """The whole point of IF NOT EXISTS: a re-run must change nothing."""
    conn.rollback()
    with conn.cursor() as cur:
        before = catalog_fingerprint(cur)
        cur.execute(SCHEMA_SQL.read_text(encoding="utf-8"))
        after = catalog_fingerprint(cur)
    conn.rollback()
    assert before == after
