{#
    A composite uniqueness test, written here rather than pulled from dbt_utils.

    dbt's built-in `unique` takes one column, and the key that matters in this
    warehouse is (city_id, observation_time). The package that usually supplies
    this is one dependency, one lockfile and one network fetch for ten lines of
    SQL, so it is ten lines of SQL.

    Returns the offending keys, so a failure says *which* pairs are duplicated
    rather than only that some are.
#}
{% test unique_combination_of_columns(model, combination_of_columns) %}

{%- set columns = combination_of_columns | join(', ') -%}

select
    {{ columns }},
    count(*) as duplicate_rows
from {{ model }}
group by {{ columns }}
having count(*) > 1

{% endtest %}
