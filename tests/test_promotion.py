"""Tests for the gold → Neon promotion.

Two kinds, and the split is deliberate. The first kind never touches a
database: what DDL is rendered, what counts as drift, what reconciles, what a
run costs. Those are the decisions, and they should be checkable in
milliseconds by someone who has neither warehouse.

The second kind promotes for real — from the local gold marts into a scratch
database on the *same local server*. That exercises the whole path (introspect,
create, COPY, index, comment, reconcile, re-run) against real marts with real
types, without spending a single second of Neon's free compute allowance. The
one thing it cannot cover is Neon itself, and the run against Neon is recorded
in the README rather than asserted here, because a test that bills a quota is a
test nobody runs.
"""

from __future__ import annotations

import ast
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("psycopg2")

import psycopg2  # noqa: E402

from config import ConfigError, get_settings, split_database_url  # noqa: E402
from serving.promote import (  # noqa: E402
    EXCLUDED_PREFIXES,
    EXCLUDED_SUFFIXES,
    GOLD_SCHEMA,
    LOCAL_ONLY_SCHEMAS,
    SCALE_TO_ZERO_SECONDS,
    Column,
    PromotionError,
    PromotionReport,
    Storage,
    TableDefinition,
    TablePromotion,
    create_table_sql,
    describe_table,
    discover_tables,
    drifted_columns,
    fingerprint,
    plan,
    promote,
    storage,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE = REPO_ROOT / "serving" / "promote.py"

#: The predictions table is named in the ticket alongside "gold marts", so it
#: is checked by name rather than left to the discovery to find.
PREDICTIONS = "fact_ml_predictions"


def _definition(**overrides) -> TableDefinition:
    base = dict(
        name="fact_example",
        columns=(
            Column("city_id", "text", not_null=True, default=None),
            Column("value", "double precision", not_null=False, default=None),
        ),
        constraints=(),
        indexes=(),
        table_comment=None,
        column_comments=(),
        rows=0,
        bytes=0,
    )
    base.update(overrides)
    return TableDefinition(**base)


# ---------------------------------------------------------------------------
# The boundary: only gold crosses
# ---------------------------------------------------------------------------


def test_only_the_gold_schema_is_named():
    """The promotion has one source schema, and it is a constant."""
    assert GOLD_SCHEMA == "gold_marts"


def _executable_strings(source: str) -> list[str]:
    """Every string literal the module can execute, prose excluded.

    Docstrings are skipped and comments never reach the tree, so what is left
    is the strings that can become SQL, identifiers, or error messages. That is
    the set the boundary has to hold over — a schema named in a paragraph
    explaining why it stays local is the documentation working, not a leak.
    """
    tree = ast.parse(source)
    documentation: set[int] = set()
    declaration: set[int] = set()

    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            if ast.get_docstring(node, clean=False) is not None:
                documentation.add(id(node.body[0].value))
        # The exclusion constant is the one place the names belong.
        target = getattr(node, "target", None)
        if isinstance(node, ast.AnnAssign) and isinstance(target, ast.Name):
            if target.id == "LOCAL_ONLY_SCHEMAS":
                declaration.update(id(child) for child in ast.walk(node))

    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in documentation
        and id(node) not in declaration
    ]


def test_bronze_and_silver_appear_only_as_an_exclusion():
    """No string the module can execute names a local-only schema.

    Asserted over the parsed module rather than by running anything, because
    the guarantee is "there is no code path" and a behavioural test can only
    show that the paths it happened to take did not take one. Every SQL string
    in this module is a literal, so a query that reached bronze or silver would
    have to say so here.
    """
    strings = _executable_strings(MODULE.read_text(encoding="utf-8"))
    assert strings, "the module is built from string literals; expected some"

    for schema in LOCAL_ONLY_SCHEMAS:
        offenders = [text for text in strings if schema in text]
        assert not offenders, (
            f"{schema!r} appears in executable strings {offenders!r}; the "
            f"promotion must have no path that reads it"
        )


def test_the_boundary_test_can_fail():
    """A guard nobody has seen fail is a guard nobody should trust."""
    leaky = (
        "LOCAL_ONLY_SCHEMAS: Final[tuple[str, ...]] = ('bronze_raw',)\n"
        "QUERY = 'select * from bronze_raw.observations_daily'\n"
    )
    assert any("bronze_raw" in text for text in _executable_strings(leaky))

    prose_only = (
        '''"""Bronze_raw and bronze_raw stay local."""\n'''
        "LOCAL_ONLY_SCHEMAS: Final[tuple[str, ...]] = ('bronze_raw',)\n"
    )
    assert not any("bronze_raw" in text for text in _executable_strings(prose_only))


def test_local_only_schemas_are_the_ones_that_exist_locally(local_conn):
    """The exclusion names real schemas, not a stale guess at their names."""
    with local_conn.cursor() as cur:
        cur.execute(
            "select nspname from pg_namespace where nspname = any(%s)",
            (list(LOCAL_ONLY_SCHEMAS),),
        )
        found = {row[0] for row in cur.fetchall()}
    assert found == set(LOCAL_ONLY_SCHEMAS)


# ---------------------------------------------------------------------------
# Rendering the DDL
# ---------------------------------------------------------------------------


def test_create_table_renders_types_nullability_and_defaults():
    sql = create_table_sql(
        _definition(
            columns=(
                Column("city_id", "text", not_null=True, default=None),
                Column("scored_at", "timestamp with time zone", True, "now()"),
                Column("note", "text", not_null=False, default=None),
            )
        )
    )
    assert '"city_id" text not null' in sql
    assert '"scored_at" timestamp with time zone default now() not null' in sql
    assert '"note" text' in sql
    assert "not null" not in sql.split('"note" text')[1].split("\n")[0]


def test_create_table_carries_constraints_into_the_table_body():
    """Checks and keys land with the table, not in a later statement.

    A table that exists for even one statement without its primary key is a
    table that could accept a duplicate in that window. Inside one transaction
    the window is invisible, but the shape is what is being asserted.
    """
    sql = create_table_sql(
        _definition(
            constraints=(
                ("fact_example_pkey", "PRIMARY KEY (city_id)"),
                ("fact_example_positive", "CHECK ((value > (0)::double precision))"),
            )
        )
    )
    assert 'constraint "fact_example_pkey" PRIMARY KEY (city_id)' in sql
    assert 'constraint "fact_example_positive" CHECK' in sql


def test_identifiers_are_quoted():
    sql = create_table_sql(_definition(name="select"))
    assert '"gold_marts"."select"' in sql


def test_column_list_names_every_column():
    """Both ends of the COPY name their columns, so physical order cannot bite.

    A target built by an earlier run may hold the same columns in a different
    order. Naming them makes that a non-event; ``select *`` would make it a
    silent mis-load.
    """
    definition = _definition()
    assert definition.column_list == '"city_id", "value"'
    assert definition.column_names == ("city_id", "value")


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------


def test_no_drift_when_the_shapes_match():
    definition = _definition()
    target = {"city_id": "text", "value": "double precision"}
    assert drifted_columns(definition, target) == ()


def test_column_order_is_not_drift():
    definition = _definition()
    assert drifted_columns(definition, {"value": "double precision", "city_id": "text"}) == ()


def test_a_new_local_column_is_drift():
    definition = _definition()
    problems = drifted_columns(definition, {"city_id": "text"})
    assert len(problems) == 1
    assert "value" in problems[0] and "absent on Neon" in problems[0]


def test_a_retyped_column_is_drift():
    definition = _definition()
    problems = drifted_columns(
        definition, {"city_id": "text", "value": "real"}
    )
    assert len(problems) == 1
    assert "double precision locally, real on Neon" in problems[0]


def test_a_column_dropped_from_the_mart_is_drift():
    definition = _definition()
    problems = drifted_columns(
        definition,
        {"city_id": "text", "value": "double precision", "legacy": "text"},
    )
    assert len(problems) == 1
    assert "legacy" in problems[0] and "gone from the local mart" in problems[0]


# ---------------------------------------------------------------------------
# Reconciliation semantics
# ---------------------------------------------------------------------------


def test_matching_counts_reconcile_when_contents_were_not_checked():
    result = TablePromotion(table="t", source_rows=10, target_rows=10)
    assert result.reconciled
    assert not result.verified


def test_mismatched_counts_do_not_reconcile():
    result = TablePromotion(table="t", source_rows=10, target_rows=9)
    assert not result.reconciled
    assert result.delta == -1


def test_matching_counts_with_differing_contents_do_not_reconcile():
    """The failure counts alone cannot see: the right number of wrong rows."""
    result = TablePromotion(
        table="t",
        source_rows=10,
        target_rows=10,
        source_checksum=111,
        target_checksum=222,
    )
    assert not result.reconciled
    assert not result.verified


def test_matching_counts_and_contents_reconcile_and_verify():
    result = TablePromotion(
        table="t", source_rows=10, target_rows=10, source_checksum=7, target_checksum=7
    )
    assert result.reconciled and result.verified


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def test_compute_includes_the_scale_to_zero_tail():
    """A promotion is billed for its tail, so the tail is in the number.

    Neon keeps the endpoint up for five minutes after the last query. Quoting
    only the active time would understate every run by the same fixed amount,
    which is exactly the kind of error that survives review.
    """
    report = PromotionReport(neon_session_seconds=60.0)
    assert report.compute_hours == pytest.approx((60.0 + SCALE_TO_ZERO_SECONDS) / 3600)
    assert SCALE_TO_ZERO_SECONDS == 300


def test_the_tail_dominates_a_short_promotion():
    """The measured promotion is under a minute; the fixed tail is five.

    Recorded as a test because it is the reason a promotion costs what it does,
    and the reason running it twice is nearly twice the cost of running it once
    rather than a rounding difference.
    """
    report = PromotionReport(neon_session_seconds=59.0)
    assert report.compute_hours < 0.1
    assert SCALE_TO_ZERO_SECONDS / (59.0 + SCALE_TO_ZERO_SECONDS) > 0.8


def test_storage_headroom_is_the_cap_minus_the_use():
    measured = Storage(used_bytes=128 * 1024 * 1024, cap_bytes=512 * 1024 * 1024)
    assert measured.headroom_bytes == 384 * 1024 * 1024
    assert measured.used_fraction == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_dbt_leftovers_are_not_promotable():
    """dbt's interrupted-run relations and the wiring check are not marts."""
    for name in ("fact_weather_hourly__dbt_tmp", "dim_cities__dbt_backup"):
        assert name.endswith(EXCLUDED_SUFFIXES)
    assert "_wiring_check_mart".startswith(EXCLUDED_PREFIXES)
    assert not "fact_weather_hourly".startswith(EXCLUDED_PREFIXES)
    assert not "fact_weather_hourly".endswith(EXCLUDED_SUFFIXES)


# ---------------------------------------------------------------------------
# Against the real gold layer
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def local_url() -> str:
    try:
        return get_settings().require_database_url()
    except ConfigError as exc:
        pytest.skip(f"no DATABASE_URL: {exc}")


@pytest.fixture(scope="session")
def local_conn(local_url: str):
    """A read-only connection to the local warehouse."""
    try:
        conn = psycopg2.connect(local_url)
    except Exception as exc:  # noqa: BLE001 - any driver failure means skip
        pytest.skip(f"local warehouse unreachable: {exc}")
    conn.set_session(readonly=True)
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def scratch_target(local_url: str):
    """A throwaway database on the local server, standing in for Neon.

    The same server rather than a second container: the promotion's job is to
    reproduce a schema and move rows between two Postgres connections, and that
    is fully exercised by two databases. What a second *host* would add is
    latency and a Neon major version, and the Neon run is measured rather than
    asserted — see this module's docstring.
    """
    parts = split_database_url(local_url)
    name = f"horizon_promote_{uuid.uuid4().hex[:8]}"

    admin = psycopg2.connect(local_url)
    admin.autocommit = True
    try:
        with admin.cursor() as cur:
            cur.execute(f'create database "{name}"')
    except psycopg2.Error as exc:
        admin.close()
        pytest.skip(f"cannot create a scratch database: {exc}")

    target_url = (
        f"postgresql://{parts.user}:{parts.password}@{parts.host}:{parts.port}/{name}"
    )
    conn = psycopg2.connect(target_url)
    conn.autocommit = False
    yield conn

    conn.close()
    with admin.cursor() as cur:
        cur.execute(f'drop database if exists "{name}" with (force)')
    admin.close()


def test_discovery_finds_the_marts_and_the_predictions(local_conn):
    found = discover_tables(local_conn)
    assert PREDICTIONS in found, "the predictions table must cross with the marts"
    assert any(name.startswith("dim_") for name in found)
    assert any(name.startswith("fact_") for name in found)
    assert found == tuple(sorted(found)), "a stable order makes runs comparable"


def test_plan_refuses_a_table_that_is_not_in_gold(local_conn):
    with pytest.raises(PromotionError, match="not in gold_marts"):
        plan(local_conn, ["observations_daily"])


def test_plan_refuses_a_typo_rather_than_promoting_less(local_conn):
    """A misspelled ``--table`` must not quietly promote nothing."""
    with pytest.raises(PromotionError, match="fact_ml_prediction"):
        plan(local_conn, ["fact_ml_prediction"])


def test_the_predictions_table_keeps_its_semantics_in_the_rendered_ddl(local_conn):
    """The check constraints are the grain, so they have to survive the copy.

    ``fact_ml_predictions`` encodes its whole contract in constraints — the
    horizon starts the day after the forecast date, the label is the score
    against the threshold. A serving copy without them would accept a row the
    local table would reject, which is the one difference between the two
    databases that would matter.
    """
    definition = describe_table(local_conn, PREDICTIONS)
    sql = create_table_sql(definition)
    assert "PRIMARY KEY (city_id, forecast_date)" in sql
    assert "horizon_start = (forecast_date + 1)" in sql
    assert "prediction_label = (risk_score >= decision_threshold)" in sql


@pytest.fixture(scope="session")
def promoted(local_conn, scratch_target):
    """One real promotion of the whole gold layer, verified."""
    definitions = plan(local_conn)
    report = promote(local_conn, scratch_target, definitions, verify=True)
    return definitions, report


def test_a_promotion_reconciles_rows_and_contents(promoted):
    _, report = promoted
    assert report.reconciled
    assert report.verified
    assert report.rows > 0
    for result in report.tables:
        assert result.source_rows == result.target_rows
        assert result.source_checksum == result.target_checksum


def test_a_promotion_creates_every_table_it_planned(promoted):
    definitions, report = promoted
    assert [t.table for t in report.tables] == [d.name for d in definitions]
    assert all(t.created for t in report.tables)


def test_indexes_are_recreated_on_the_target(promoted, scratch_target):
    """Every index on a local mart exists on the target, by definition.

    Compared as definitions rather than as names: a name that matched while the
    columns differed would pass a weaker test and leave the dashboard scanning.
    """
    definitions, _ = promoted
    with scratch_target.cursor() as cur:
        cur.execute(
            "select indexdef from pg_indexes where schemaname = %s", (GOLD_SCHEMA,)
        )
        on_target = {row[0] for row in cur.fetchall()}

    expected = {
        indexdef for definition in definitions for _, indexdef in definition.indexes
    }
    assert expected, "the facts define indexes; the test is worthless without them"
    assert expected <= on_target


def test_constraints_are_recreated_on_the_target(promoted, scratch_target):
    definitions, _ = promoted
    with scratch_target.cursor() as cur:
        cur.execute(
            """
            select pg_get_constraintdef(con.oid)
            from pg_constraint con
            join pg_class rel on rel.oid = con.conrelid
            join pg_namespace n on n.oid = rel.relnamespace
            where n.nspname = %s and con.contype in ('p', 'u', 'c')
            """,
            (GOLD_SCHEMA,),
        )
        on_target = {row[0] for row in cur.fetchall()}

    expected = {clause for d in definitions for _, clause in d.constraints}
    assert expected <= on_target


def test_comments_are_recreated_on_the_target(promoted, scratch_target):
    definitions, _ = promoted
    expected = {
        (d.name, column, comment)
        for d in definitions
        for column, comment in d.column_comments
    }
    assert expected, "fact_ml_predictions documents its columns; expected some"

    with scratch_target.cursor() as cur:
        cur.execute(
            """
            select c.relname, a.attname, col_description(a.attrelid, a.attnum)
            from pg_attribute a
            join pg_class c on c.oid = a.attrelid
            join pg_namespace n on n.oid = c.relnamespace
            where n.nspname = %s and a.attnum > 0 and not a.attisdropped
              and col_description(a.attrelid, a.attnum) is not null
            """,
            (GOLD_SCHEMA,),
        )
        on_target = {tuple(row) for row in cur.fetchall()}
    assert expected <= on_target


def test_nothing_but_gold_reached_the_target(promoted, scratch_target):
    """The boundary, checked on the far side rather than only in the source."""
    with scratch_target.cursor() as cur:
        cur.execute(
            "select distinct table_schema from information_schema.tables "
            "where table_schema not in ('pg_catalog', 'information_schema')"
        )
        schemas = {row[0] for row in cur.fetchall()}
    assert schemas <= {GOLD_SCHEMA, "public"}
    assert not schemas & set(LOCAL_ONLY_SCHEMAS)


def test_re_running_replaces_rather_than_appends(promoted, local_conn, scratch_target):
    """The idempotency claim, run rather than asserted.

    A second promotion over the first must leave the same row counts, the same
    contents, and no second copy of any index. "Re-runnable" that only means
    "does not error" would still double every table.
    """
    definitions, first = promoted

    with scratch_target.cursor() as cur:
        cur.execute(
            "select count(*) from pg_indexes where schemaname = %s", (GOLD_SCHEMA,)
        )
        indexes_before = cur.fetchone()[0]

    second = promote(local_conn, scratch_target, definitions, verify=True)

    assert second.reconciled and second.verified
    assert second.rows == first.rows
    assert [t.target_rows for t in second.tables] == [
        t.target_rows for t in first.tables
    ]
    assert [t.source_checksum for t in second.tables] == [
        t.source_checksum for t in first.tables
    ]
    assert all(not t.created and not t.recreated for t in second.tables)
    assert all(t.indexes_created == 0 for t in second.tables)

    with scratch_target.cursor() as cur:
        cur.execute(
            "select count(*) from pg_indexes where schemaname = %s", (GOLD_SCHEMA,)
        )
        assert cur.fetchone()[0] == indexes_before


def test_drift_stops_a_promotion_instead_of_reshaping_the_target(
    promoted, local_conn, scratch_target
):
    """A mart that gained a column must not be promoted by accident.

    The target is deliberately damaged — a column dropped on the serving side —
    and the promotion must refuse. Reshaping a serving database is a thing to
    do on purpose, with ``--recreate``, not a side effect of a routine run.
    """
    definitions, _ = promoted
    victim = next(d for d in definitions if d.name == PREDICTIONS)
    droppable = victim.columns[-1].name

    with scratch_target.cursor() as cur:
        cur.execute(
            f'alter table {GOLD_SCHEMA}.{PREDICTIONS} drop column "{droppable}"'
        )
    scratch_target.commit()

    with pytest.raises(PromotionError, match="drifted"):
        promote(local_conn, scratch_target, [victim])

    with pytest.raises(PromotionError, match="--recreate"):
        promote(local_conn, scratch_target, [victim])

    repaired = promote(local_conn, scratch_target, [victim], recreate=True, verify=True)
    assert repaired.tables[0].recreated
    assert repaired.reconciled and repaired.verified


def test_a_dry_run_writes_nothing(local_conn, scratch_target):
    """The plan is readable without committing to it."""
    with scratch_target.cursor() as cur:
        cur.execute(f'drop schema if exists "{GOLD_SCHEMA}" cascade')
    scratch_target.commit()

    definitions = plan(local_conn)
    report = promote(local_conn, scratch_target, definitions, dry_run=True)

    assert report.dry_run
    assert all(t.created for t in report.tables)
    assert report.copied_bytes == 0

    with scratch_target.cursor() as cur:
        cur.execute(
            "select count(*) from information_schema.tables where table_schema = %s",
            (GOLD_SCHEMA,),
        )
        assert cur.fetchone()[0] == 0


def test_storage_is_unmeasured_on_a_target_that_is_not_neon(scratch_target):
    """Plain Postgres has no pg_cluster_size, and that is not a failure.

    Nothing in the copy is Neon-specific; only the storage report is. Probing
    the catalogue rather than attempting the ``create extension`` matters here:
    a failed statement would abort the transaction the promotion runs in.
    """
    assert storage(scratch_target) is None
    with scratch_target.cursor() as cur:
        cur.execute("select 1")
        assert cur.fetchone() == (1,)


def test_fingerprint_sees_a_changed_value(local_conn, scratch_target):
    """The content check has to be able to fail, or it proves nothing."""
    definitions = plan(local_conn, [PREDICTIONS])
    promote(local_conn, scratch_target, definitions, verify=True)
    definition = definitions[0]

    before = fingerprint(scratch_target, definition)
    with scratch_target.cursor() as cur:
        cur.execute(
            f"update {GOLD_SCHEMA}.{PREDICTIONS} "
            f"set model_variant = model_variant || '-tampered' "
            f"where ctid = (select ctid from {GOLD_SCHEMA}.{PREDICTIONS} limit 1)"
        )
    after = fingerprint(scratch_target, definition)
    scratch_target.rollback()

    assert before[0] == after[0], "one row edited, not added"
    assert before[1] != after[1], "the same count, a different fingerprint"
