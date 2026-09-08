{#
    Peaks-over-threshold return period, in years.

        P(X > z) = rate * [1 + shape * (z - u) / scale] ^ (-1 / shape)
        years    = 1 / (P * 365.25)

    Written as a macro because `fact_anomaly_return_periods` evaluates it three
    times per row -- once at the shape's point estimate and once at each end of
    its bootstrap interval -- and three copies of an exponent with a division by
    a parameter that can be zero is three chances to get one of them wrong.

    Three things this handles that a direct transcription would not:

    **shape near zero.** The formula divides by the shape, and the exponential
    limit is the right answer there rather than an error. The cut is 1e-8, well
    below any fitted value and well above where numeric division stops being
    meaningful.

    **Past the upper endpoint.** A negative shape means the tail is bounded --
    there is a hottest possible day -- and the support term goes non-positive
    beyond it. NULL, meaning "this model does not place this day", not a very
    large number. Three of the eleven cities have a bounded tail on the whole
    interval and would otherwise be asked to raise a negative number to a
    fractional power.

    **`rate` is the declustered rate.** Callers must pass the exceedance rate
    from `fact_extreme_value`, which is events per day after runs are collapsed
    to their peaks. Passing a raw exceedance fraction would divide every period
    here by about two, in the direction that makes the answer sound dramatic.
#}
{% macro return_period(level, threshold, shape, scale, rate, days_per_year=365.25) %}
    (
        case
            when {{ level }} is null
              or {{ shape }} is null
              or {{ scale }} is null
              or {{ level }} <= {{ threshold }}
                then null
            when abs({{ shape }}) < 1e-8
                then 1.0 / (
                    {{ rate }}
                    * exp(-({{ level }} - {{ threshold }}) / {{ scale }})
                    * {{ days_per_year }}
                )
            when 1.0 + {{ shape }} * ({{ level }} - {{ threshold }}) / {{ scale }} <= 0
                then null
            else 1.0 / (
                {{ rate }}
                * power(
                    1.0 + {{ shape }} * ({{ level }} - {{ threshold }}) / {{ scale }},
                    -1.0 / {{ shape }}
                )
                * {{ days_per_year }}
            )
        end
    )
{% endmacro %}
