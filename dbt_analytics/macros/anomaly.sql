{#
    The Z-score and its flags, written once.

    Three things here are easy to get wrong in a way that still produces a
    plausible column:

    *   **Both tails.** `z > 2.5` reads naturally and silently discards every
        cold extreme. Moscow's January is the strongest case: a project that
        flags only heat would report it as one of the calmest cities in the
        set. The flag is on `abs(z)`.
    *   **A null sigma is not a zero anomaly.** Where no leakage-free baseline
        exists, the honest answer is "unknown", not "not an anomaly". Coercing
        it to `none` would quietly assert something the data cannot support,
        and would count those days in the denominator of every anomaly rate.
    *   **Division by sigma.** Guarded by nullif, though DBT-09 asserts sigma is
        never zero — the guard is for the day that assertion is relaxed rather
        than for today.
#}

{% macro z_score(observed, mean_column, stddev_column) -%}
    (({{ observed }} - {{ mean_column }}) / nullif({{ stddev_column }}, 0))
{%- endmacro %}


{% macro anomaly_direction(z_column, threshold) -%}
    case
        when {{ z_column }} is null then null
        when {{ z_column }} >  {{ threshold }} then 'hot'
        when {{ z_column }} < -{{ threshold }} then 'cold'
        else 'none'
    end
{%- endmacro %}
