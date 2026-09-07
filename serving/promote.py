"""Copies the finished gold layer from the local warehouse to Neon.

This is the boundary. Bronze and silver never cross it — not because the
promotion could not carry them, but because nothing on the other side has any
use for them and Neon's free plan gives 0.5 GB for everything. Raw landings and
thin deduplication views are working material; a dashboard reads finished
marts. So the source is one schema, ``gold_marts``, named once as a constant,
and every query here is scoped to it. There is no code path that reaches
``bronze_raw`` or ``silver_staging``, and a test asserts that.

    python serving/promote.py                  # promote everything in gold
    python serving/promote.py --dry-run        # plan, drift and sizes only
    python serving/promote.py --table fact_ml_predictions
    python serving/promote.py --recreate       # rebuild tables whose shape drifted

**The DDL is read from the local catalog, never written by hand.** A second
copy of the marts' shape — a ``serving_schema.sql`` beside the dbt models —
would be one more file to keep in step with seven models, and the day it fell
behind the promotion would build yesterday's columns and fail in a COPY with a
message about counts rather than about drift. The local warehouse is the
definition: columns, types, not-nulls, defaults, primary keys, check
constraints, indexes and comments are all introspected from ``pg_catalog`` and
replayed. Adding a column to a dbt model and re-running is the whole change.

**The whole promotion is one transaction.** Not one per table. A dashboard
joining ``fact_ml_predictions`` to ``dim_cities`` while a per-table promotion
was halfway through would read a new fact against an old dimension and show a
number that never existed. One transaction costs a longer ``ACCESS EXCLUSIVE``
lock — the promotion is minutes, and the readers are a handful — and buys two
things worth more than that: the swap is atomic, and a failure at table six
leaves Neon in exactly the state it was in before table one. That second
property *is* the idempotency: re-running after a failure is not a repair, it
is the same run again.

**Each table is extracted to a buffer, then loaded.** The obvious alternative
is to pipe ``COPY TO STDOUT`` straight into ``COPY FROM STDIN`` so the two
halves overlap. It is also a thread, a pipe, and a deadlock whenever the
target errors while the source is still writing. The overlap it buys is worth
having only if extraction is a meaningful share of the run, and it is not: the
source is Postgres in a container on the same machine and the target is across
an ocean. The measured split is printed for every table so that claim stays
checkable rather than remaining an assumption.

**Storage is checked against what the server says, not against 0.5 GB.** Neon
enforces the cap itself, as ``neon.max_cluster_size``, and reports the current
consumption through ``pg_cluster_size()`` from its own extension. Reading both
means the headroom line cannot disagree with the thing that will actually
refuse the write.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Sequence

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ConfigError, get_settings, mask_secret  # noqa: E402

__all__ = [
    "GOLD_SCHEMA",
    "LOCAL_ONLY_SCHEMAS",
    "SCALE_TO_ZERO_SECONDS",
    "Column",
    "PromotionError",
    "PromotionReport",
    "TableDefinition",
    "TablePromotion",
    "create_table_sql",
    "describe_table",
    "discover_tables",
    "drifted_columns",
    "fingerprint",
    "plan",
    "promote",
    "storage",
]

#: The only schema that crosses. Everything in this module is scoped to it.
GOLD_SCHEMA: Final[str] = "gold_marts"

#: Named so the exclusion is a stated decision rather than an absence. These
#: are never read here; the constant exists to be printed and to be asserted.
LOCAL_ONLY_SCHEMAS: Final[tuple[str, ...]] = ("bronze_raw", "silver_staging")

#: Objects in gold that are not marts. dbt leaves ``__dbt_tmp`` relations
#: behind when a run is interrupted, and the schema-wiring check (§ README)
#: builds a ``_wiring_check_mart``. Neither is a deliverable and neither should
#: be silently promoted because it happened to be sitting in the schema.
EXCLUDED_PREFIXES: Final[tuple[str, ...]] = ("_",)
EXCLUDED_SUFFIXES: Final[tuple[str, ...]] = ("__dbt_tmp", "__dbt_backup")

#: Neon scales an idle compute endpoint to zero after five minutes on the free
#: plan. Those five minutes are billed, so a promotion's compute cost is its
#: active time plus this tail — quoting only the active time would understate
#: every run by the same fixed amount.
SCALE_TO_ZERO_SECONDS: Final[int] = 300

#: How Neon reports the storage it will refuse to exceed, and what it is using.
STORAGE_CAP_SETTING: Final[str] = "neon.max_cluster_size"
CLUSTER_SIZE_FUNCTION: Final[str] = "pg_cluster_size()"


class PromotionError(RuntimeError):
    """The promotion cannot proceed, and guessing would be worse than stopping."""


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _qualified(table: str) -> str:
    return f"{_quote(GOLD_SCHEMA)}.{_quote(table)}"


# ---------------------------------------------------------------------------
# Reading the local definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Column:
    """One column, as the target must recreate it."""

    name: str
    type: str
    not_null: bool
    default: str | None

    def ddl(self) -> str:
        parts = [_quote(self.name), self.type]
        if self.default is not None:
            parts.append(f"default {self.default}")
        if self.not_null:
            parts.append("not null")
        return " ".join(parts)


@dataclass(frozen=True)
class TableDefinition:
    """Everything about one mart that the serving copy has to reproduce.

    Constraints are restricted to primary keys, uniques and checks. Foreign
    keys are deliberately not carried: there are none in gold — the facts
    select from silver rather than joining the dimensions, precisely so that a
    fact whose city is missing shows up as a broken relationship test instead
    of vanishing — and carrying them would make the promotion order-dependent
    for no gain.
    """

    name: str
    columns: tuple[Column, ...]
    constraints: tuple[tuple[str, str], ...]
    indexes: tuple[tuple[str, str], ...]
    table_comment: str | None
    column_comments: tuple[tuple[str, str], ...]
    rows: int
    bytes: int

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    @property
    def column_list(self) -> str:
        """The columns, named and quoted, for both ends of the COPY.

        Naming them on both sides rather than relying on ``select *`` is what
        makes the copy independent of physical column order, so a target table
        built by an older run still loads correctly.
        """
        return ", ".join(_quote(name) for name in self.column_names)


def discover_tables(conn) -> tuple[str, ...]:
    """Every promotable base table in ``gold_marts``, in a stable order.

    Discovery rather than a hardcoded list: a mart added to dbt should be
    promoted by re-running this, not by remembering to edit it. The filtering
    is what keeps "everything in gold" from meaning "including dbt's litter".
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            select c.relname
            from pg_class c
            join pg_namespace n on n.oid = c.relnamespace
            where n.nspname = %s and c.relkind = 'r'
            order by c.relname
            """,
            (GOLD_SCHEMA,),
        )
        found = [row[0] for row in cur.fetchall()]

    return tuple(
        name
        for name in found
        if not name.startswith(EXCLUDED_PREFIXES)
        and not name.endswith(EXCLUDED_SUFFIXES)
    )


def describe_table(conn, table: str) -> TableDefinition:
    """Introspect one mart into everything needed to rebuild and fill it."""
    qualified = f"{GOLD_SCHEMA}.{table}"

    with conn.cursor() as cur:
        cur.execute(
            """
            select a.attname,
                   format_type(a.atttypid, a.atttypmod),
                   a.attnotnull,
                   pg_get_expr(d.adbin, d.adrelid)
            from pg_attribute a
            left join pg_attrdef d
                   on d.adrelid = a.attrelid and d.adnum = a.attnum
            where a.attrelid = %s::regclass
              and a.attnum > 0
              and not a.attisdropped
            order by a.attnum
            """,
            (qualified,),
        )
        columns = tuple(
            Column(name=name, type=type_, not_null=not_null, default=default)
            for name, type_, not_null, default in cur.fetchall()
        )
        if not columns:
            raise PromotionError(f"{qualified} has no columns — is it a table?")

        # Primary keys, uniques and checks travel with the table body so the
        # target is constrained from the moment it exists, not from whenever a
        # later statement got around to it.
        cur.execute(
            """
            select conname, pg_get_constraintdef(oid)
            from pg_constraint
            where conrelid = %s::regclass and contype in ('p', 'u', 'c')
            order by contype, conname
            """,
            (qualified,),
        )
        constraints = tuple(cur.fetchall())

        # Indexes that back a constraint are created by the constraint; issuing
        # them again would be a duplicate under a different name.
        cur.execute(
            """
            select i.relname, pg_get_indexdef(x.indexrelid)
            from pg_index x
            join pg_class i on i.oid = x.indexrelid
            where x.indrelid = %s::regclass
              and not exists (
                  select 1 from pg_constraint c where c.conindid = x.indexrelid
              )
            order by i.relname
            """,
            (qualified,),
        )
        indexes = tuple(cur.fetchall())

        cur.execute("select obj_description(%s::regclass, 'pg_class')", (qualified,))
        table_comment = cur.fetchone()[0]

        cur.execute(
            """
            select a.attname, col_description(a.attrelid, a.attnum)
            from pg_attribute a
            where a.attrelid = %s::regclass
              and a.attnum > 0
              and not a.attisdropped
              and col_description(a.attrelid, a.attnum) is not null
            order by a.attnum
            """,
            (qualified,),
        )
        column_comments = tuple(cur.fetchall())

        cur.execute(f"select count(*) from {_qualified(table)}")
        rows = cur.fetchone()[0]

        cur.execute(
            "select pg_total_relation_size(%s::regclass)", (qualified,)
        )
        size = cur.fetchone()[0]

    return TableDefinition(
        name=table,
        columns=columns,
        constraints=constraints,
        indexes=indexes,
        table_comment=table_comment,
        column_comments=column_comments,
        rows=rows,
        bytes=size,
    )


def create_table_sql(definition: TableDefinition) -> str:
    """Render the ``create table`` for one mart."""
    body = [column.ddl() for column in definition.columns]
    body += [
        f"constraint {_quote(name)} {clause}"
        for name, clause in definition.constraints
    ]
    joined = ",\n    ".join(body)
    return f"create table {_qualified(definition.name)} (\n    {joined}\n)"


#: Renders every column of a row as text, hashes it to 32 bits, and sums the
#: hashes. Summing rather than concatenating makes it independent of row order,
#: so neither side needs a sort — which matters because the target would have
#: to sort 263 000 rows over a serverless connection to produce one.
#:
#: What it compares is the *text rendering* of each row, which is exactly what
#: COPY transmitted, so agreement means the bytes that left the local warehouse
#: are the bytes that landed. The two settings are pinned rather than inherited
#: because the rendering of a float and of a timestamptz depends on them, and
#: the two servers are different major versions with independent defaults.
FINGERPRINT_SETTINGS: Final[tuple[str, ...]] = (
    "set local extra_float_digits = 1",
    "set local timezone = 'UTC'",
)


def fingerprint(conn, definition: TableDefinition) -> tuple[int, int]:
    """Row count and an order-independent content hash for one table.

    Returns:
        ``(rows, checksum)``. Two tables with the same pair hold the same rows
        with the same values, whatever order they are stored in.
    """
    with conn.cursor() as cur:
        for statement in FINGERPRINT_SETTINGS:
            cur.execute(statement)
        cur.execute(
            f"select count(*), "
            f"       coalesce(sum(('x' || substr(md5(row::text), 1, 8))"
            f"                    ::bit(32)::bigint), 0) "
            f"from (select {definition.column_list} "
            f"      from {_qualified(definition.name)}) as row"
        )
        rows, checksum = cur.fetchone()
    return int(rows), int(checksum)


# ---------------------------------------------------------------------------
# Reading what the target already has
# ---------------------------------------------------------------------------


def _target_columns(conn, table: str) -> dict[str, str] | None:
    """The target's column shape, or ``None`` if the table is not there."""
    with conn.cursor() as cur:
        cur.execute(
            """
            select a.attname, format_type(a.atttypid, a.atttypmod)
            from pg_attribute a
            join pg_class c on c.oid = a.attrelid
            join pg_namespace n on n.oid = c.relnamespace
            where n.nspname = %s and c.relname = %s
              and a.attnum > 0 and not a.attisdropped
            """,
            (GOLD_SCHEMA, table),
        )
        rows = cur.fetchall()
    return {name: type_ for name, type_ in rows} if rows else None


def drifted_columns(
    definition: TableDefinition, target: dict[str, str]
) -> tuple[str, ...]:
    """Differences between the local mart and the serving copy, as sentences.

    Column *order* is not drift — the COPY names its columns on both sides, so
    a target built by an older run loads fine. Names and types are, and both
    fail a COPY with a message about the data rather than about the schema,
    which is why they are checked here first.
    """
    problems: list[str] = []
    local = {column.name: column.type for column in definition.columns}

    for name, type_ in local.items():
        if name not in target:
            problems.append(f"{name}: absent on Neon, present locally as {type_}")
        elif target[name] != type_:
            problems.append(f"{name}: {type_} locally, {target[name]} on Neon")
    for name in target:
        if name not in local:
            problems.append(f"{name}: present on Neon, gone from the local mart")

    return tuple(problems)


# ---------------------------------------------------------------------------
# Storage and compute, as the server reports them
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Storage:
    """What Neon says it is using and what it will refuse to exceed."""

    used_bytes: int
    cap_bytes: int

    @property
    def headroom_bytes(self) -> int:
        return self.cap_bytes - self.used_bytes

    @property
    def used_fraction(self) -> float:
        return self.used_bytes / self.cap_bytes if self.cap_bytes else 0.0


def storage(conn) -> Storage | None:
    """Read the serving project's size and its hard cap from the server.

    ``pg_cluster_size()`` comes from Neon's own extension and reports the
    project's synthetic storage — the number the free plan's 0.5 GB is measured
    against — rather than one database's heap. ``neon.max_cluster_size`` is the
    limit the server itself enforces. Both are read rather than assumed so the
    headroom reported here cannot disagree with the thing that refuses a write.

    Returns:
        ``None`` when the target is not a Neon endpoint. Nothing else in the
        promotion is Neon-specific — the copy is plain Postgres — so a target
        without these is reported as unmeasured rather than treated as a
        failure. Availability is checked in the catalogue rather than by
        attempting the ``create`` and catching, because a failed statement
        would abort the transaction the promotion is about to run in.
    """
    with conn.cursor() as cur:
        cur.execute("select 1 from pg_available_extensions where name = 'neon'")
        if cur.fetchone() is None:
            return None
        cur.execute("create extension if not exists neon")
        cur.execute(f"select {CLUSTER_SIZE_FUNCTION}")
        used = int(cur.fetchone()[0])
        # The setting carries its unit ("512MB"); pg_size_bytes resolves it
        # rather than this module deciding what Neon meant.
        cur.execute("select pg_size_bytes(current_setting(%s))", (STORAGE_CAP_SETTING,))
        cap = int(cur.fetchone()[0])
    return Storage(used_bytes=used, cap_bytes=cap)


# ---------------------------------------------------------------------------
# The promotion
# ---------------------------------------------------------------------------


@dataclass
class TablePromotion:
    """What happened to one mart, measured rather than assumed."""

    table: str
    source_rows: int
    target_rows: int = 0
    copied_bytes: int = 0
    created: bool = False
    recreated: bool = False
    indexes_created: int = 0
    extract_seconds: float = 0.0
    load_seconds: float = 0.0
    source_checksum: int | None = None
    target_checksum: int | None = None
    verify_seconds: float = 0.0
    #: Counted on the target after the copy, not inferred from the source.
    #: The two agree by construction, which is the reason to read the one the
    #: report claims to be describing rather than the one it is easier to reach.
    target_indexes: int = 0

    @property
    def verified(self) -> bool:
        """Whether the contents were compared, and agreed."""
        return (
            self.source_checksum is not None
            and self.source_checksum == self.target_checksum
        )

    @property
    def reconciled(self) -> bool:
        """Counts agree — and contents too, when they were checked.

        Counts alone would pass a promotion that moved the right number of
        wrong rows. When ``--verify`` was asked for, agreement means agreement
        of the values as well.
        """
        if self.source_rows != self.target_rows:
            return False
        if self.source_checksum is None:
            return True
        return self.source_checksum == self.target_checksum

    @property
    def delta(self) -> int:
        return self.target_rows - self.source_rows


@dataclass
class PromotionReport:
    """The whole run, in the form the checklist asks it to be reported."""

    tables: list[TablePromotion] = field(default_factory=list)
    storage_before: Storage | None = None
    storage_after: Storage | None = None
    neon_session_seconds: float = 0.0
    dry_run: bool = False

    @property
    def rows(self) -> int:
        return sum(t.source_rows for t in self.tables)

    @property
    def copied_bytes(self) -> int:
        return sum(t.copied_bytes for t in self.tables)

    @property
    def reconciled(self) -> bool:
        return all(t.reconciled for t in self.tables)

    @property
    def verified(self) -> bool:
        return bool(self.tables) and all(t.verified for t in self.tables)

    @property
    def compute_hours(self) -> float:
        """Endpoint-hours the promotion is billed for, tail included.

        Neon bills for the time the compute endpoint is *up*, and the endpoint
        stays up for :data:`SCALE_TO_ZERO_SECONDS` after the last query. A
        promotion therefore costs its own duration plus a fixed five minutes,
        multiplied by the endpoint's compute size in CU. The CU multiplier is
        not readable from SQL, so this returns endpoint-hours at 1 CU and the
        report states the multiplication rather than burying it.
        """
        return (self.neon_session_seconds + SCALE_TO_ZERO_SECONDS) / 3600.0


def _copy_table(source, target, definition: TableDefinition) -> tuple[int, float, float]:
    """Extract one mart to a buffer, then load it. Returns bytes and timings.

    The buffer spools to disk past 32 MB, so the largest mart — 50 MB of hourly
    facts — never sits in memory whole, and a mart ten times that size would
    still promote on a laptop.
    """
    columns = definition.column_list
    with tempfile.SpooledTemporaryFile(max_size=32 * 1024 * 1024, mode="w+b") as buffer:
        started = time.perf_counter()
        with source.cursor() as cur:
            cur.copy_expert(
                f"copy (select {columns} from {_qualified(definition.name)}) to stdout",
                buffer,
            )
        copied = buffer.tell()
        extract_seconds = time.perf_counter() - started

        buffer.seek(0)
        started = time.perf_counter()
        with target.cursor() as cur:
            cur.copy_expert(
                f"copy {_qualified(definition.name)} ({columns}) from stdin", buffer
            )
        load_seconds = time.perf_counter() - started

    return copied, extract_seconds, load_seconds


def plan(source, tables: Sequence[str] | None = None) -> tuple[TableDefinition, ...]:
    """Work out what would cross, and read enough of it to print a plan.

    Separate from :func:`promote` so the plan can be shown before the first
    write, and so the introspection happens once rather than once for the
    report and again for the run.

    Raises:
        PromotionError: A named table is not in ``gold_marts``, or the schema
            holds nothing promotable. Both are worth stopping for: the first is
            a typo that would otherwise promote less than was asked for, the
            second a dbt run that has not happened.
    """
    available = discover_tables(source)
    if tables:
        unknown = [name for name in tables if name not in available]
        if unknown:
            raise PromotionError(
                f"not in {GOLD_SCHEMA}: {', '.join(sorted(unknown))}. "
                f"Present: {', '.join(available)}."
            )
        selected = tuple(name for name in available if name in set(tables))
    else:
        selected = available

    if not selected:
        raise PromotionError(
            f"{GOLD_SCHEMA} holds no promotable tables. Run dbt against the "
            f"local warehouse first."
        )

    return tuple(describe_table(source, name) for name in selected)


def promote(
    source,
    target,
    definitions: Sequence[TableDefinition],
    *,
    recreate: bool = False,
    dry_run: bool = False,
    verify: bool = False,
    on_table=None,
) -> PromotionReport:
    """Copy the gold layer to the serving database, atomically.

    Args:
        source: A connection to the local warehouse. Read only.
        target: A connection to Neon. Left untouched when ``dry_run``.
        definitions: What to promote, from :func:`plan`.
        recreate: Drop and rebuild a target table whose shape has drifted.
            Without it, drift raises — a promotion that quietly reshapes the
            serving database is not something to do as a side effect.
        dry_run: Report the plan, the drift and the sizes; write nothing.
        verify: Compare the contents of each table as well as its row count.
            Costs one extra full scan on each side, and the target's scan is
            the expensive one — see :func:`fingerprint`.
        on_table: Called with each :class:`TablePromotion` as it completes, so
            a long run can print progress instead of going quiet.

    Raises:
        PromotionError: A shape has drifted without ``recreate``, or a row
            count did not reconcile. The second is raised *before* the commit,
            so a promotion that does not reconcile is not one that happened.
    """
    report = PromotionReport(dry_run=dry_run)

    session_started = time.perf_counter()
    report.storage_before = storage(target)

    try:
        with target.cursor() as cur:
            for definition in definitions:
                result = TablePromotion(
                    table=definition.name, source_rows=definition.rows
                )
                existing = _target_columns(target, definition.name)

                if existing is not None:
                    problems = drifted_columns(definition, existing)
                    if problems and not recreate:
                        raise PromotionError(
                            f"{GOLD_SCHEMA}.{definition.name} has drifted from the "
                            f"local mart:\n  "
                            + "\n  ".join(problems)
                            + "\nRe-run with --recreate to rebuild it on Neon."
                        )
                    if problems:
                        result.recreated = True
                else:
                    result.created = True

                if dry_run:
                    report.tables.append(result)
                    if on_table:
                        on_table(result)
                    continue

                cur.execute(f"create schema if not exists {_quote(GOLD_SCHEMA)}")

                if result.recreated:
                    cur.execute(f"drop table {_qualified(definition.name)}")
                if result.created or result.recreated:
                    cur.execute(create_table_sql(definition))
                else:
                    # Replace rather than append. The marts are a full rebuild
                    # of a fixed window every time, so an upsert would need a
                    # key per table and would still leave rows that dbt had
                    # dropped. Inside this transaction the truncate is
                    # invisible to readers until the commit.
                    cur.execute(f"truncate table {_qualified(definition.name)}")

                copied, extract, load = _copy_table(source, target, definition)
                result.copied_bytes = copied
                result.extract_seconds = extract
                result.load_seconds = load

                # After the rows, not before: a fresh table's indexes build
                # once over a full heap instead of being maintained per row.
                cur.execute(
                    "select indexname from pg_indexes "
                    "where schemaname = %s and tablename = %s",
                    (GOLD_SCHEMA, definition.name),
                )
                existing_indexes = {row[0] for row in cur.fetchall()}
                for name, indexdef in definition.indexes:
                    if name not in existing_indexes:
                        cur.execute(indexdef)
                        result.indexes_created += 1

                cur.execute(
                    "select count(*) from pg_indexes "
                    "where schemaname = %s and tablename = %s",
                    (GOLD_SCHEMA, definition.name),
                )
                result.target_indexes = cur.fetchone()[0]

                if definition.table_comment:
                    cur.execute(
                        f"comment on table {_qualified(definition.name)} is %s",
                        (definition.table_comment,),
                    )
                for column, comment in definition.column_comments:
                    cur.execute(
                        f"comment on column {_qualified(definition.name)}."
                        f"{_quote(column)} is %s",
                        (comment,),
                    )

                cur.execute(f"select count(*) from {_qualified(definition.name)}")
                result.target_rows = cur.fetchone()[0]

                if verify:
                    started = time.perf_counter()
                    source_rows, result.source_checksum = fingerprint(
                        source, definition
                    )
                    _, result.target_checksum = fingerprint(target, definition)
                    result.verify_seconds = time.perf_counter() - started
                    if source_rows != result.source_rows:
                        # The local mart changed under the promotion — a dbt
                        # run in another terminal. Everything copied is now of
                        # unknown vintage, so it does not get committed.
                        raise PromotionError(
                            f"{definition.name} changed locally during the "
                            f"promotion: {result.source_rows:,} rows when it was "
                            f"read, {source_rows:,} now. Nothing was committed."
                        )

                report.tables.append(result)
                if on_table:
                    on_table(result)

        if not dry_run:
            unreconciled = [t for t in report.tables if not t.reconciled]
            if unreconciled:
                raise PromotionError(
                    "reconciliation failed; rolling back:\n  "
                    + "\n  ".join(
                        f"{t.table}: {t.source_rows:,} local, {t.target_rows:,} Neon "
                        f"({t.delta:+,})"
                        if t.source_rows != t.target_rows
                        else f"{t.table}: {t.source_rows:,} rows on both sides but "
                        f"the contents differ ({t.source_checksum} vs "
                        f"{t.target_checksum})"
                        for t in unreconciled
                    )
                )
            target.commit()
        else:
            target.rollback()
    except Exception:
        target.rollback()
        raise

    report.storage_after = storage(target)
    report.neon_session_seconds = time.perf_counter() - session_started
    return report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _size(value: int) -> str:
    """Bytes at a scale that does not round the small dimensions to zero."""
    if value < 1024 * 1024:
        return f"{value / 1024:.0f} kB"
    return f"{value / 1024 / 1024:.1f} MB"


def _print_report(report: PromotionReport, definitions: Sequence[TableDefinition]) -> None:
    by_name = {d.name: d for d in definitions}

    print("\n=== promoted ===")
    header = f"  {'table':28}{'rows':>10}{'copied':>11}{'extract':>10}{'load':>10}  state"
    print(header)
    for result in report.tables:
        state = (
            "recreated"
            if result.recreated
            else "created"
            if result.created
            else "replaced"
        )
        if result.indexes_created:
            state += f", {result.indexes_created} index(es)"
        print(
            f"  {result.table:28}{result.source_rows:>10,}"
            f"{_size(result.copied_bytes):>11}"
            f"{result.extract_seconds:>9.1f}s{result.load_seconds:>9.1f}s  {state}"
        )
    print(
        f"  {'total':28}{report.rows:>10,}{_size(report.copied_bytes):>11}"
        f"{sum(t.extract_seconds for t in report.tables):>9.1f}s"
        f"{sum(t.load_seconds for t in report.tables):>9.1f}s"
    )

    verified = any(t.source_checksum is not None for t in report.tables)
    print("\n=== reconciliation ===")
    extra = f"{'content hash':>22}" if verified else ""
    print(f"  {'table':28}{'local':>10}{'Neon':>10}{'delta':>8}{extra}")
    for result in report.tables:
        hashes = ""
        if verified:
            mark = "=" if result.verified else "x"
            hashes = f"{result.source_checksum:>21,}{mark}"
        print(
            f"  {result.table:28}{result.source_rows:>10,}{result.target_rows:>10,}"
            f"{result.delta:>+8,}{hashes}  {'OK' if result.reconciled else 'MISMATCH'}"
        )
    verdict = "RECONCILED" if report.reconciled else "FAILED"
    if verified:
        verdict += " (rows and contents)" if report.verified else " (rows only)"
    else:
        verdict += " (rows only — pass --verify to compare contents)"
    print(f"  {'all tables':28}  {verdict}")

    print("\n=== indexes on the target ===")
    for result in report.tables:
        definition = by_name[result.table]
        expected = len(definition.indexes) + sum(
            1
            for _, clause in definition.constraints
            if clause.startswith(("PRIMARY KEY", "UNIQUE"))
        )
        if not expected:
            note = "  (none defined on the local mart)"
        elif result.target_indexes == expected:
            note = "  matches the local mart"
        else:
            note = f"  MISMATCH — {expected} on the local mart"
        print(f"  {result.table:28}{result.target_indexes:>3} index(es){note}")

    before, after = report.storage_before, report.storage_after
    if not (before and after):
        print("\n=== storage ===")
        print("  not measured — the target is not a Neon endpoint "
              "(no pg_cluster_size to read)")
    else:
        print("\n=== Neon storage ===")
        print(f"  before        {_size(before.used_bytes):>10}")
        print(f"  after         {_size(after.used_bytes):>10}")
        # Signed, and it can be negative: Neon retains history for a window and
        # ages it out, so a re-promotion of unchanged marts can end smaller
        # than it started. The number to hold onto is "after", not "change".
        change = after.used_bytes - before.used_bytes
        sign = "+" if change >= 0 else "-"
        print(f"  change        {sign + _size(abs(change)):>10}   over this run")
        print(
            f"  cap           {_size(after.cap_bytes):>10}"
            f"   ({STORAGE_CAP_SETTING}, enforced by the server)"
        )
        print(
            f"  headroom      {_size(after.headroom_bytes):>10}"
            f"   ({(1 - after.used_fraction) * 100:.1f}% free)"
        )
        verdict = "UNDER CAP" if after.headroom_bytes > 0 else "OVER CAP"
        print(f"  verdict       {verdict:>10}   using {after.used_fraction * 100:.1f}%")

    print("\n=== Neon compute ===")
    print(f"  session       {report.neon_session_seconds:>10.1f} s  endpoint held active")
    print(f"  scale-to-zero {SCALE_TO_ZERO_SECONDS:>10.1f} s  billed tail after the last query")
    print(
        f"  billed        {report.compute_hours:>10.4f} endpoint-hours"
        f"  = ({report.neon_session_seconds:.0f} + {SCALE_TO_ZERO_SECONDS}) / 3600"
    )
    print("  compute-hours = endpoint-hours x endpoint size in CU (console is authoritative)")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--table",
        action="append",
        dest="tables",
        metavar="NAME",
        help=f"promote one mart from {GOLD_SCHEMA}; repeatable. Default: all",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="drop and rebuild a Neon table whose shape has drifted",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report the plan, the drift and the sizes; write nothing",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "compare table contents, not only row counts. One extra scan per "
            "side; a mismatch rolls the promotion back"
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=get_settings().log_level, format="%(levelname)s %(name)s: %(message)s"
    )
    settings = get_settings()

    try:
        source_url = settings.require_database_url()
        target_url = settings.require_serving_database_url()
    except ConfigError as exc:
        print(f"FAIL  {exc}")
        return 2

    print("=== promotion ===")
    print(f"  source        {mask_secret(source_url)}")
    print(f"  target        {mask_secret(target_url)}")
    print(f"  crosses       {GOLD_SCHEMA}")
    print(f"  stays local   {', '.join(LOCAL_ONLY_SCHEMAS)}")
    if args.dry_run:
        print("  mode          dry run — nothing will be written")

    source = psycopg2.connect(source_url)
    target = psycopg2.connect(target_url)
    source.autocommit = False
    target.autocommit = False

    try:
        definitions = plan(source, args.tables)

        print("\n=== plan ===")
        print(f"  {'table':28}{'rows':>10}{'local size':>12}")
        for definition in definitions:
            print(
                f"  {definition.name:28}{definition.rows:>10,}"
                f"{_size(definition.bytes):>12}"
            )
        print(
            f"  {'total':28}{sum(d.rows for d in definitions):>10,}"
            f"{_size(sum(d.bytes for d in definitions)):>12}"
        )

        if not args.dry_run:
            print("\n=== copying ===", flush=True)

        def progress(result: TablePromotion) -> None:
            # Printed as each table lands rather than at the end: the run is
            # minutes long and a silent terminal is indistinguishable from a
            # hung one.
            if args.dry_run:
                return
            rate = (
                result.copied_bytes / result.load_seconds / 1024 / 1024
                if result.load_seconds
                else 0.0
            )
            checked = (
                f"  verified in {result.verify_seconds:.1f}s"
                if result.source_checksum is not None
                else ""
            )
            print(
                f"  {result.table:28}{result.target_rows:>10,} rows"
                f"{_size(result.copied_bytes):>11}"
                f"{result.load_seconds:>8.1f}s  {rate:5.1f} MB/s{checked}",
                flush=True,
            )

        report = promote(
            source,
            target,
            definitions,
            recreate=args.recreate,
            dry_run=args.dry_run,
            verify=args.verify,
            on_table=progress,
        )
    except PromotionError as exc:
        print(f"\nFAIL  {exc}")
        return 1
    finally:
        source.close()
        target.close()

    if args.dry_run:
        print("\n=== dry run ===")
        for result in report.tables:
            state = (
                "would recreate"
                if result.recreated
                else "would create"
                if result.created
                else "would replace"
            )
            print(f"  {result.table:28}{result.source_rows:>10,}  {state}")
        before = report.storage_before
        if before:
            print(f"\n  Neon storage now {_size(before.used_bytes)} of "
                  f"{_size(before.cap_bytes)}")
        return 0

    _print_report(report, definitions)
    print("\nOK  promotion committed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
