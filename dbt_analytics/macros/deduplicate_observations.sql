{#
    The silver deduplication, shared by both grains.

    Bronze is append-only by design: re-ingesting a window inserts a second
    copy rather than replacing the first, and there is no unique constraint on
    the natural key because one would reject a legitimate re-ingest. That makes
    deduplication silver's job, and this is where it happens.

    **PostgreSQL has no QUALIFY clause.** That is Snowflake, BigQuery and
    DuckDB syntax; the original project draft proposed it. The Postgres form is
    a subquery with row_number() and a `where rank = 1` filter, which is what
    this generates.

    The ordering is `ingested_at desc, id desc`, and the tiebreaker is not
    decoration. The loader stamps one ingested_at per run, deliberately, so
    that rows from one run tie rather than being ordered by how long the COPY
    took to reach them, so two copies of a window landed by the *same* run
    would tie here and row_number() would pick between them arbitrarily,
    differently on each build. `id desc` breaks the tie with the surrogate key,
    which is monotonic per insert, so the model is deterministic.

    Columns come from the relation itself rather than a hand-maintained list.
    Bronze has already lost a column once this project (source_url, dropped
    when it measured 83% of the row payload), and a list here would have
    needed the same edit or silently kept selecting a column that no longer
    exists.
#}
{% macro deduplicate_observations(source_name, table_name) %}

{%- set relation = source(source_name, table_name) -%}

{#
    dbt renders every model twice: once at parse time to build the DAG, and
    again at execute time to produce SQL. Warehouse introspection only works in
    the second pass: during parsing `adapter.get_columns_in_relation` returns
    nothing at all, and an unguarded column list is silently empty. Guarding on
    `execute` is the canonical shape, and skipping it here produced a
    compilation error naming the model's own relation rather than the source's.
#}
{%- set columns = [] -%}
{%- if execute -%}
    {%- set columns = adapter.get_columns_in_relation(relation)
            | map(attribute='name') | list -%}
    {%- if 'city_id' not in columns or 'observation_time' not in columns -%}
        {{ exceptions.raise_compiler_error(
            relation ~ " has no (city_id, observation_time) to deduplicate on; "
            ~ "found " ~ (columns | join(', ') or "no columns")
        ) }}
    {%- endif -%}
{%- endif -%}

with ranked as (

    select
        {% for column in columns %}"{{ column }}",
        {% endfor %}row_number() over (
            partition by city_id, observation_time
            order by ingested_at desc, id desc
        ) as _dedup_rank

    from {{ relation }}

)

select
    {% for column in columns %}"{{ column }}"{{ "," if not loop.last }}
    {% endfor %}
from ranked
where _dedup_rank = 1

{% endmacro %}
