"""Tests for the bronze loader.

The offline half checks the frame that goes to Postgres: column set and order,
nulls that stay null, and the absence of the dtype coercion pandas would apply
if left to itself. The database half loads for real and reads back, inside a
transaction that is always rolled back, so the tests leave no rows behind.

What these mostly assert is that nothing happened. Bronze's job is faithful
landing, so the interesting claims are negative: no unit converted, no timezone
shifted, no null filled, no duplicate removed. Each of those has a test that
would fail if a future well-meaning change added the cleverness.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pd = pytest.importorskip("pandas")
sqlalchemy = pytest.importorskip("sqlalchemy")

from sqlalchemy import event, text  # noqa: E402

from config import ConfigError, get_settings  # noqa: E402
from ingestion import archive  # noqa: E402
from ingestion.client import (  # noqa: E402
    DAILY_VARIABLES,
    HOURLY_VARIABLES,
    parse_payload,
)
from ingestion.loader import (  # noqa: E402
    BRONZE_SCHEMA,
    CHUNK_ROWS,
    PROVENANCE_COLUMNS,
    TABLE_BY_GRAIN,
    LoadError,
    _copy_insert,
    _csv_buffer,
    build_frame,
    columns_for,
    engine_from_settings,
    load_archive,
    load_response,
    load_unit,
)
from ingestion.planner import Manifest, WorkUnit  # noqa: E402
from http_fixtures import daily_payload, hourly_payload  # noqa: E402

# parse_payload takes a city id as data, so the frame tests can use a name that
# cannot collide with anything real. Replay rebuilds the request URL through the
# registry, so the archive-backed tests need a city that exists.
CITY = "test_loader"
ARCHIVE_CITY = "london"
START = dt.date(2023, 1, 1)
END = dt.date(2023, 1, 3)
DAYS = 3
URL = "https://archive-api.test/v1/archive?latitude=51.5&fake=1"

BATCH = uuid.UUID("00000000-0000-4000-8000-000000000001")
INGESTED_AT = dt.datetime(2026, 9, 7, 12, 0, tzinfo=dt.timezone.utc)


def response_for(grain: str = "daily", days: int = DAYS, **overrides):
    body = (
        daily_payload(days, START, **overrides)
        if grain == "daily"
        else hourly_payload(days, START, **overrides)
    )
    end = START + dt.timedelta(days=days - 1)
    return parse_payload(
        body, city_id=CITY, grain=grain, start=START, end=end, url=URL
    )


def frame_for(grain: str = "daily", days: int = DAYS, **overrides) -> pd.DataFrame:
    return build_frame(
        response_for(grain, days, **overrides),
        batch_id=BATCH,
        ingested_at=INGESTED_AT,
    )


# ---------------------------------------------------------------------------
# The frame that goes to Postgres
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grain", ["daily", "hourly"])
def test_columns_are_provenance_then_variables(grain: str) -> None:
    variables = DAILY_VARIABLES if grain == "daily" else HOURLY_VARIABLES
    assert columns_for(grain) == (*PROVENANCE_COLUMNS, *variables)
    assert list(frame_for(grain, days=1).columns) == list(columns_for(grain))


def test_one_row_per_timestamp() -> None:
    assert len(frame_for("daily", days=DAYS)) == DAYS
    assert len(frame_for("hourly", days=2)) == 48


def test_every_row_carries_the_run_identity() -> None:
    frame = frame_for()
    assert "source_url" not in frame.columns
    assert (frame["batch_id"] == BATCH).all()
    assert (frame["ingested_at"] == INGESTED_AT).all()
    assert (frame["city_id"] == CITY).all()
    assert frame[list(PROVENANCE_COLUMNS)].notna().all().all()


def test_nothing_is_coerced_to_float() -> None:
    """pandas widens any column holding a null to float64. dtype=object does not.

    weather_code is the one that shows it: the API sends WMO codes as integers
    and the column is smallint, so a silent widening to 51.0 would be rejected
    by COPY — or worse, accepted somewhere it should not be.
    """
    payload = daily_payload(DAYS, START)
    payload["daily"]["weather_code"] = [51, 3, 61]
    frame = build_frame(
        parse_payload(payload, city_id=CITY, grain="daily", start=START,
                      end=END, url=URL),
        batch_id=BATCH,
        ingested_at=INGESTED_AT,
    )
    assert set(frame.dtypes.astype(str)) == {"object"}
    assert frame["weather_code"].tolist() == [51, 3, 61]
    assert all(isinstance(v, int) for v in frame["weather_code"])


def test_a_null_stays_none_and_does_not_become_nan() -> None:
    """A NaN would land as a float, not as SQL NULL."""
    payload = daily_payload(DAYS, START)
    payload["daily"]["snowfall_sum"] = [None] * DAYS
    frame = build_frame(
        parse_payload(payload, city_id=CITY, grain="daily", start=START,
                      end=END, url=URL),
        batch_id=BATCH,
        ingested_at=INGESTED_AT,
    )
    assert frame["snowfall_sum"].tolist() == [None, None, None]
    assert not any(isinstance(v, float) for v in frame["snowfall_sum"])


def test_a_partly_null_column_keeps_its_other_values() -> None:
    payload = daily_payload(DAYS, START)
    payload["daily"]["rain_sum"] = [1.5, None, 3.5]
    frame = build_frame(
        parse_payload(payload, city_id=CITY, grain="daily", start=START,
                      end=END, url=URL),
        batch_id=BATCH,
        ingested_at=INGESTED_AT,
    )
    assert frame["rain_sum"].tolist() == [1.5, None, 3.5]


def test_timestamps_reach_the_frame_already_utc_aware() -> None:
    frame = frame_for()
    for moment in frame["observation_time"]:
        assert moment.utcoffset() == dt.timedelta(0)
        assert (moment.hour, moment.minute) == (0, 0)


# ---------------------------------------------------------------------------
# CSV rendering, which is the whole null-preservation mechanism
# ---------------------------------------------------------------------------


def test_none_becomes_an_unquoted_empty_field() -> None:
    """Postgres reads that as NULL; a quoted empty field would be a string."""
    assert _csv_buffer([[1, None, "x"]]).read() == '"1",,"x"\n'


def test_an_empty_string_stays_quoted_and_distinct_from_null() -> None:
    """The default QUOTE_MINIMAL renders both as nothing, silently nulling one."""
    rendered = _csv_buffer([["", None]]).read()
    assert rendered == '"",\n'
    naive = io.StringIO()
    csv.writer(naive, lineterminator="\n").writerow(["", None])
    assert naive.getvalue() == ",\n", "the mode this deliberately avoids"


def test_text_containing_a_comma_survives() -> None:
    """Text fields can carry commas; a naive join would corrupt them."""
    url = "https://x.test/v1?daily=a,b,c&tz=UTC"
    assert list(csv.reader(_csv_buffer([[url]])))[0] == [url]


def test_a_value_copy_cannot_cast_names_itself(conn) -> None:
    """Bronze does not convert types, so an upstream change stops the load.

    A float where the schema declares smallint means the API changed how it
    represents an integer. Landing 98.0 as 98 would hide that, and the payload
    is already archived — so nothing is lost by refusing, only delayed.
    """
    payload = daily_payload(DAYS, START)
    payload["daily"]["weather_code"] = [51.5, 3.5, 61.5]
    response = parse_payload(
        payload, city_id=CITY, grain="daily", start=START, end=END, url=URL
    )
    with pytest.raises(LoadError, match="representation"):
        load_response(response, conn, batch_id=BATCH, ingested_at=INGESTED_AT)


# ---------------------------------------------------------------------------
# The database
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
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
def conn(engine):
    """A connection whose work is always rolled back."""
    connection = engine.connect()
    transaction = connection.begin()
    try:
        yield connection
    finally:
        transaction.rollback()
        connection.close()


def landed(conn, grain: str = "daily", columns: str = "*") -> list:
    return conn.execute(
        text(
            f"select {columns} from {BRONZE_SCHEMA}.{TABLE_BY_GRAIN[grain]} "
            "where city_id = :city order by observation_time"
        ),
        {"city": CITY},
    ).fetchall()


def test_a_city_year_lands_with_the_payload_row_count(conn) -> None:
    """The acceptance criterion, at the size the backfill actually uses."""
    response = response_for("daily", days=365)
    written = load_response(
        response, conn, batch_id=BATCH, ingested_at=INGESTED_AT
    )
    assert written == len(response) == 365
    assert len(landed(conn)) == 365


def test_a_year_of_hourly_lands(conn) -> None:
    response = response_for("hourly", days=365)
    written = load_response(
        response, conn, batch_id=BATCH, ingested_at=INGESTED_AT
    )
    assert written == 8760
    assert len(landed(conn, "hourly")) == 8760


def test_no_insert_statement_is_ever_issued(conn) -> None:
    """Batch loading via COPY, not a row-by-row INSERT loop.

    436 000 single-row inserts is hours of round trips; COPY moves the same
    rows in seconds. Counting statements proves the mechanism rather than
    trusting a comment.
    """
    statements: list[str] = []

    @event.listens_for(conn, "before_cursor_execute")
    def record(conn_, cursor, statement, *args):  # noqa: ANN001
        statements.append(statement)

    try:
        load_response(
            response_for("daily", days=365), conn,
            batch_id=BATCH, ingested_at=INGESTED_AT, verify_columns=False,
        )
    finally:
        event.remove(conn, "before_cursor_execute", record)

    assert not [s for s in statements if s.lstrip().upper().startswith("INSERT")]
    assert len(landed(conn)) == 365


def test_values_land_exactly_as_the_payload_had_them(conn) -> None:
    """No unit conversion, no rounding, no rescaling."""
    response = response_for("daily", days=DAYS)
    load_response(response, conn, batch_id=BATCH, ingested_at=INGESTED_AT)

    variables = list(DAILY_VARIABLES)
    rows = landed(conn, columns=", ".join(variables))
    expected = list(response.rows())
    assert len(rows) == len(expected)
    for row, want in zip(rows, expected):
        for name, got in zip(variables, row):
            assert got == pytest.approx(want[name]), name


def test_timestamps_land_without_a_timezone_shift(conn) -> None:
    response = response_for("daily", days=DAYS)
    load_response(response, conn, batch_id=BATCH, ingested_at=INGESTED_AT)
    landed_times = [row[0] for row in landed(conn, columns="observation_time")]
    assert landed_times == list(response.times)
    assert all(t.utcoffset() == dt.timedelta(0) for t in landed_times)


def test_hourly_timestamps_land_on_the_hour(conn) -> None:
    response = response_for("hourly", days=1)
    load_response(response, conn, batch_id=BATCH, ingested_at=INGESTED_AT)
    times = [row[0] for row in landed(conn, "hourly", "observation_time")]
    assert [t.hour for t in times] == list(range(24))
    assert {t.minute for t in times} == {0}


def test_a_null_lands_as_sql_null_not_zero(conn) -> None:
    """Zero would mean "it did not snow"; null means "no data here"."""
    payload = daily_payload(DAYS, START)
    payload["daily"]["snowfall_sum"] = [None] * DAYS
    payload["daily"]["weather_code"] = [None, 3, None]
    response = parse_payload(
        payload, city_id=CITY, grain="daily", start=START, end=END, url=URL
    )
    load_response(response, conn, batch_id=BATCH, ingested_at=INGESTED_AT)

    rows = landed(conn, columns="snowfall_sum, weather_code")
    assert [r[0] for r in rows] == [None, None, None]
    assert [r[1] for r in rows] == [None, 3, None]

    nulls = conn.execute(
        text(
            f"select count(*) from {BRONZE_SCHEMA}.observations_daily "
            "where city_id = :city and snowfall_sum is null"
        ),
        {"city": CITY},
    ).scalar()
    assert nulls == DAYS


def test_no_row_is_dropped_when_a_whole_column_is_null(conn) -> None:
    payload = daily_payload(DAYS, START)
    for name in DAILY_VARIABLES:
        payload["daily"][name] = [None] * DAYS
    response = parse_payload(
        payload, city_id=CITY, grain="daily", start=START, end=END, url=URL
    )
    assert load_response(
        response, conn, batch_id=BATCH, ingested_at=INGESTED_AT
    ) == DAYS
    assert len(landed(conn)) == DAYS


def test_loading_the_same_unit_twice_duplicates(conn) -> None:
    """Bronze is append-only; silver deduplicates. Doing it here too is wrong."""
    response = response_for("daily", days=DAYS)
    for _ in range(2):
        load_response(response, conn, batch_id=uuid.uuid4(),
                      ingested_at=INGESTED_AT)
    assert len(landed(conn)) == DAYS * 2


def test_every_landed_row_carries_its_metadata_columns(conn) -> None:
    response = response_for("daily", days=DAYS)
    load_response(response, conn, batch_id=BATCH, ingested_at=INGESTED_AT)
    rows = landed(
        conn, columns="city_id, observation_time, ingested_at, batch_id"
    )
    for city_id, observation_time, ingested_at, batch_id in rows:
        assert city_id == CITY
        assert observation_time is not None
        assert ingested_at == INGESTED_AT
        assert uuid.UUID(str(batch_id)) == BATCH


def test_one_ingested_at_for_the_whole_run(conn) -> None:
    """Silver orders by ingested_at desc; rows from one run should tie."""
    load_response(response_for("daily", days=365), conn,
                  batch_id=BATCH, ingested_at=INGESTED_AT)
    distinct = conn.execute(
        text(
            f"select count(distinct ingested_at) from "
            f"{BRONZE_SCHEMA}.observations_daily where city_id = :city"
        ),
        {"city": CITY},
    ).scalar()
    assert distinct == 1


def test_the_grid_cell_lands_alongside_the_observations(conn) -> None:
    response = response_for("daily", days=DAYS)
    load_response(response, conn, batch_id=BATCH, ingested_at=INGESTED_AT)
    rows = landed(conn, columns="api_latitude, api_longitude, api_elevation_m")
    assert {tuple(r) for r in rows} == {
        (response.latitude, response.longitude, response.elevation_m)
    }


def test_a_batch_can_be_deleted_wholesale(conn) -> None:
    """The reason batch_id exists: undo one bad run without touching the rest."""
    keep, drop = uuid.uuid4(), uuid.uuid4()
    load_response(response_for("daily", days=DAYS), conn,
                  batch_id=keep, ingested_at=INGESTED_AT)
    load_response(response_for("daily", days=DAYS), conn,
                  batch_id=drop, ingested_at=INGESTED_AT)
    assert len(landed(conn)) == DAYS * 2

    conn.execute(
        text(f"delete from {BRONZE_SCHEMA}.observations_daily "
             "where batch_id = :batch"),
        {"batch": str(drop)},
    )
    remaining = landed(conn, columns="batch_id")
    assert len(remaining) == DAYS
    assert {uuid.UUID(str(r[0])) for r in remaining} == {keep}


def test_a_missing_table_says_what_to_run(conn) -> None:
    from ingestion.loader import _assert_columns_exist

    with pytest.raises(LoadError, match="apply_schema"):
        _assert_columns_exist(conn, "observations_nowhere", ["city_id"])


def test_a_column_with_nowhere_to_land_is_named(conn) -> None:
    from ingestion.loader import _assert_columns_exist

    with pytest.raises(LoadError, match="invented_variable"):
        _assert_columns_exist(
            conn, "observations_daily", ["city_id", "invented_variable"]
        )


# ---------------------------------------------------------------------------
# Reading from the archive, never from the network
# ---------------------------------------------------------------------------


@pytest.fixture
def no_network(monkeypatch):
    import requests

    def forbidden(*args, **kwargs):
        raise AssertionError("the loader must not touch the network")

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", forbidden)


@pytest.fixture
def stocked(tmp_path: Path):
    """An archive on disk holding two real-shaped units."""
    import json

    root = tmp_path / "raw"
    units = []
    for grain, days in (("daily", DAYS), ("hourly", 1)):
        unit = WorkUnit(
            city_id=ARCHIVE_CITY, grain=grain,
            start=START, end=START + dt.timedelta(days=days - 1),
        )
        body = (
            daily_payload(days, START) if grain == "daily"
            else hourly_payload(days, START)
        )
        archive.write(unit, json.dumps(body).encode("utf-8"), root)
        units.append(unit)
    return root, units


def test_load_unit_reads_from_disk(conn, stocked, no_network) -> None:
    root, (daily_unit, _) = stocked
    assert load_unit(
        daily_unit, conn, batch_id=BATCH, ingested_at=INGESTED_AT, root=root
    ) == DAYS
    rows = conn.execute(
        text(f"select count(*) from {BRONZE_SCHEMA}.observations_daily "
             "where batch_id = :batch"),
        {"batch": str(BATCH)},
    ).scalar()
    assert rows == DAYS


def test_load_unit_refuses_a_payload_of_the_wrong_length(
    conn, tmp_path, no_network
) -> None:
    import json

    root = tmp_path / "raw"
    unit = WorkUnit(city_id=ARCHIVE_CITY, grain="daily", start=START,
                    end=START + dt.timedelta(days=9))
    archive.write(unit, json.dumps(daily_payload(DAYS, START)).encode(), root)
    with pytest.raises(Exception) as excinfo:
        load_unit(unit, conn, batch_id=BATCH, ingested_at=INGESTED_AT, root=root)
    assert "10 daily timestamps" in str(excinfo.value)


def test_load_archive_walks_the_whole_archive(engine, stocked, no_network) -> None:
    root, units = stocked
    result = load_archive(root, engine=engine)
    try:
        assert result.units == 2
        assert result.rows == DAYS + 24
        assert result.by_table["observations_daily"] == DAYS
        assert result.by_table["observations_hourly"] == 24
        assert result.seconds > 0
    finally:
        with engine.begin() as connection:
            for table in TABLE_BY_GRAIN.values():
                connection.execute(
                    text(f"delete from {BRONZE_SCHEMA}.{table} "
                         "where batch_id = :batch"),
                    {"batch": str(result.batch_id)},
                )


def test_load_archive_records_the_manifest_after_committing(
    engine, stocked, tmp_path, no_network
) -> None:
    """A manifest entry must never claim rows the warehouse does not have."""
    root, units = stocked
    manifest = Manifest(tmp_path / "manifest.jsonl")
    result = load_archive(root, engine=engine, manifest=manifest)
    try:
        assert len(manifest) == 2
        assert all(manifest.is_complete(unit) for unit in units)
        assert manifest.completed_rows() == DAYS + 24
    finally:
        with engine.begin() as connection:
            for table in TABLE_BY_GRAIN.values():
                connection.execute(
                    text(f"delete from {BRONZE_SCHEMA}.{table} "
                         "where batch_id = :batch"),
                    {"batch": str(result.batch_id)},
                )


def test_load_archive_narrows_by_grain_and_city(engine, stocked, no_network) -> None:
    root, _ = stocked
    result = load_archive(root, grain="daily", engine=engine)
    try:
        assert result.units == 1
        assert result.rows == DAYS
    finally:
        with engine.begin() as connection:
            connection.execute(
                text(f"delete from {BRONZE_SCHEMA}.observations_daily "
                     "where batch_id = :batch"),
                {"batch": str(result.batch_id)},
            )


def test_an_empty_archive_loads_nothing(engine, tmp_path, no_network) -> None:
    result = load_archive(tmp_path / "empty", engine=engine)
    assert (result.units, result.rows) == (0, 0)


# ---------------------------------------------------------------------------
# The loader writes every column the schema declares
# ---------------------------------------------------------------------------


SCHEMA_SQL = Path(__file__).resolve().parent.parent / "ingestion" / "schema.sql"


def schema_columns(table: str) -> set[str]:
    import re

    sql = SCHEMA_SQL.read_text(encoding="utf-8")
    match = re.search(
        rf"create table if not exists bronze_raw\.{table} \((.*?)\n\);", sql, re.DOTALL
    )
    assert match, table
    return {
        m.group(1)
        for m in re.finditer(r"^\s{4}([a-z_][a-z0-9_]*)\s+\S", match.group(1), re.M)
    }


@pytest.mark.parametrize("grain", ["daily", "hourly"])
def test_the_loader_fills_every_column_but_the_surrogate_key(grain: str) -> None:
    """A column nothing writes would sit null forever."""
    unfilled = schema_columns(TABLE_BY_GRAIN[grain]) - set(columns_for(grain))
    assert unfilled == {"id", "constraint"}


@pytest.mark.parametrize("grain", ["daily", "hourly"])
def test_the_loader_invents_no_column(grain: str) -> None:
    assert set(columns_for(grain)) <= schema_columns(TABLE_BY_GRAIN[grain])


def test_the_copy_method_is_what_to_sql_uses(monkeypatch) -> None:
    captured: dict = {}

    def fake_to_sql(self, name, con, **kwargs):
        captured.update(name=name, **kwargs)
        return len(self)

    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql)
    load_response(
        response_for("daily", days=DAYS), object(),
        batch_id=BATCH, ingested_at=INGESTED_AT, verify_columns=False,
    )
    assert captured["method"] is _copy_insert
    assert captured["chunksize"] == CHUNK_ROWS
    assert captured["if_exists"] == "append"
    assert captured["schema"] == BRONZE_SCHEMA
    assert captured["index"] is False
