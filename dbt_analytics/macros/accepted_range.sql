{#
    A numeric bounds test, written here rather than pulled from dbt_utils:
    same reasoning as `unique_combination_of_columns`.

    Two properties matter and neither is the default anywhere:

    *   **Nulls pass.** Bronze preserves nulls rather than filling them, so a
        null is an absence of measurement, not an out-of-range value. A test
        that failed on them would make preserving nulls impossible.
    *   **`expression` overrides the column.** A bound is sometimes natural in
        a different unit from the stored one (wind is stored in km/h and
        bounded in m/s) and the conversion belongs in a macro rather than
        retyped as inline arithmetic per entry.

    Returns the offending values so a failure says what was out of range, not
    merely that something was.
#}
{% test accepted_range(model, column_name, min_value=none, max_value=none,
                       expression=none) %}

{%- if min_value is none and max_value is none -%}
    {{ exceptions.raise_compiler_error(
        "accepted_range on " ~ column_name ~ " sets neither bound"
    ) }}
{%- endif -%}

{%- set checked = expression if expression is not none else column_name -%}

select
    {{ column_name }} as column_value,
    {{ checked }} as checked_value
from {{ model }}
where {{ checked }} is not null
  and (
    {%- if min_value is not none %}
        {{ checked }} < {{ min_value }}
    {%- endif %}
    {%- if min_value is not none and max_value is not none %} or {% endif %}
    {%- if max_value is not none %}
        {{ checked }} > {{ max_value }}
    {%- endif %}
  )

{% endtest %}
