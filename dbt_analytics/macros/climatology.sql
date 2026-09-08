{#
    Mean and standard deviation recovered from power sums.

    Written once because the identity is easy to get subtly wrong and the
    result still looks like a number:

        σ² = (Σx² - (Σx)²/n) / (n - 1)

    Three details that each cost silence rather than an error:

    *   **n - 1, not n.** The sample standard deviation. A population σ over a
        thirty-year sample understates the spread, which inflates every Z-score
        built on it and makes ordinary weather look extreme.
    *   **Guard n < 2.** With one observation the denominator is zero; with
        none, the mean divides by zero too. Both return null, which is the
        honest answer for "no baseline here".
    *   **`greatest(..., 0)` inside the root.** The subtraction can go very
        slightly negative through floating-point cancellation when every value
        in the window is identical, and `sqrt` of -1e-17 is an error rather
        than the zero it should be.
#}

{% macro climatology_mean(sum_column, count_column) -%}
    (case when {{ count_column }} > 0
          then ({{ sum_column }} / {{ count_column }})::double precision
     end)
{%- endmacro %}


{% macro climatology_stddev(sum_column, sum_sq_column, count_column) -%}
    (case when {{ count_column }} > 1
          then sqrt(greatest(
                   ({{ sum_sq_column }}
                    - ({{ sum_column }} ^ 2) / {{ count_column }})
                   / ({{ count_column }} - 1),
                   0
               ))::double precision
     end)
{%- endmacro %}
