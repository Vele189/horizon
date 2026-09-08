{#
    Use the configured schema verbatim rather than prefixing it with the
    target's.

    dbt's default builds `<target.schema>_<custom>`, so a model configured into
    `gold_marts` against target schema `public` would land in `public_gold_marts`,
    a schema nothing else in this project knows about, next to the empty
    `gold_marts` that ingestion/schema.sql created. The medallion schema names
    are fixed by §5.2 and shared with the Python loader, so they are used as
    written.

    The default exists to keep several developers on one warehouse out of each
    other's way. That is not the situation here: local development has a
    private database in Docker, and the only other target is the Neon serving
    database, which holds one copy of the marts by design.
#}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- if custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
