-- Within a city, a bigger departure must be a rarer day.
--
-- The return period is a monotone function of abs(Z) given fixed parameters,
-- so this holds by construction -- which is exactly why it is worth asserting.
-- The construction involves a per-row exponent, a division by a shape that can
-- be either sign, and a `power()` whose argument can go non-positive, and a
-- sign error in any of those produces finite, plausible numbers that happen to
-- run backwards. A reader would notice that on the map only by knowing what
-- the answer should have been.
--
-- Ranks rather than a self-join on adjacent values: comparing each row against
-- the next-largest departure in the same city catches a local inversion, which
-- a min/max check would not.
with ranked as (

    select
        city_id,
        date_key,
        abs_z,
        return_period_years,
        lag(abs_z) over (partition by city_id order by abs_z)
            as previous_abs_z,
        lag(return_period_years) over (partition by city_id order by abs_z)
            as previous_return_period
    from {{ ref('fact_anomaly_return_periods') }}
    where return_period_years is not null

)

select *
from ranked
where previous_abs_z is not null
  and abs_z > previous_abs_z
  and return_period_years < previous_return_period
