{{ config(materialized='table') }}

-- ============================================================================
-- How unusual, in years
-- ============================================================================
-- **Grain: one row per city per day past that city's tail threshold.** Not
-- every scored day. Below the threshold the generalised Pareto says nothing --
-- it is a model *for* exceedances, fitted only to them -- and a row carrying a
-- return period for an ordinary Tuesday would be an extrapolation dressed as a
-- lookup. About five per cent of days survive, which is the point: this table
-- is for the days a reader stops on.
--
-- **What a row means.** "A day this far from normal, in either direction,
-- happens about once every `return_period_years` years in this city." Both
-- tails, on abs(Z), because that is what `is_anomaly` is on and what the
-- Anomaly Map colours -- a one-sided version would lose Moscow's January.
--
-- **Why the arithmetic is here and the fit is not.** `fact_extreme_value` is
-- fitted by Python, one row per city, four parameters and their intervals.
-- Turning those into a per-day answer is a join and an exponent, which is
-- exactly what SQL is for, and doing it here rather than in the fit means the
-- derivation is inspectable: the numbers a reader sees are recomputed from
-- published parameters every build, not carried forward from a run nobody can
-- repeat.
--
-- **The shape sensitivity is not a confidence interval, and is named so.**
-- `return_period_years_shape_low` and `_shape_high` are the return period
-- recomputed at the ends of the shape's bootstrap interval, holding the scale
-- at its point estimate. That is a *sensitivity*: it answers "how much does
-- this answer depend on the parameter we know least about" and not "where does
-- the true return period lie". A genuine interval would need the joint
-- distribution of shape and scale, which are strongly and negatively
-- correlated in a GPD fit, and propagating them independently would give an
-- interval wider than the truth while looking more rigorous. Naming the
-- columns after what was varied is what stops them being read as the other
-- thing.
--
-- **The band decides what may be said, and `return_period_is_reportable` is
-- where that decision lives.** Its width is not uniform, it explodes with the
-- exceedance, and the transition is sharp. Sensitivity ratio, high end over
-- low, median across the eleven cities:
--
--     |Z|    2.6    3.0    3.4    3.8     4.2      4.6
--     ratio  1.3x   1.8x   4.1x   9.6x    134x    4879x
--
-- Up to 3 sigma every city is inside a factor of three: "about a one-in-two-year
-- day" is a sentence the data supports. By 4.2 sigma the median answer spans two
-- orders of magnitude and two cities have gone past their fitted upper endpoint
-- entirely. Phoenix at 4 sigma reads 28 years, and its shape interval puts it
-- anywhere between 9 years and 152,405.
--
-- So a row is reportable when the band is inside one order of magnitude, and
-- otherwise only its floor may be quoted -- "at least a one-in-30-year day".
-- The floor is the stable end: it comes from the heaviest tail in the interval,
-- which is the reading that makes an extreme most frequent, so understating
-- rarity is the direction the uncertainty is resolved in. That is deliberate.
-- Overstating it is the failure that would make this table exciting and wrong.
-- ============================================================================

with tails as (

    select
        city_id,
        threshold,
        exceedance_rate,
        shape,
        shape_low,
        shape_high,
        scale
    from {{ source('gold', 'fact_extreme_value') }}

),

exceedances as (

    select
        anomalies.city_id,
        anomalies.date_key,
        anomalies.z_temperature_2m_mean,
        anomalies.temperature_2m_mean,
        anomalies.departure_c,
        anomalies.is_anomaly,
        anomalies.anomaly_direction,
        abs(anomalies.z_temperature_2m_mean) as abs_z,
        tails.threshold,
        tails.exceedance_rate,
        tails.shape,
        tails.shape_low,
        tails.shape_high,
        tails.scale
    from {{ ref('fact_weather_anomalies') }} as anomalies
    inner join tails
        on anomalies.city_id = tails.city_id
    where anomalies.z_temperature_2m_mean is not null
      and abs(anomalies.z_temperature_2m_mean) > tails.threshold

),

computed as (

    select
        city_id,
        date_key,
        temperature_2m_mean,
        departure_c,
        z_temperature_2m_mean,
        abs_z,
        is_anomaly,
        anomaly_direction,

        -- The fit's own terms, carried so a row is self-describing. A reader who
        -- wants to check the exponent below has everything it used on the same row.
        threshold as tail_threshold,
        exceedance_rate as tail_exceedance_rate,
        shape as tail_shape,
        scale as tail_scale,

        {{ return_period('abs_z', 'threshold', 'shape', 'scale', 'exceedance_rate') }}
            as return_period_years,

        -- Sensitivity, not an interval. Note the crossing: a *larger* shape means a
        -- heavier tail, so it makes an extreme MORE frequent and the period
        -- SHORTER. The low-shape end therefore gives the long period. The column
        -- names say which parameter was moved, not which end of a range came out,
        -- so this is a fact about the model and not a naming accident to fix.
        {{ return_period('abs_z', 'threshold', 'shape_high', 'scale', 'exceedance_rate') }}
            as return_period_years_shape_low,
        {{ return_period('abs_z', 'threshold', 'shape_low', 'scale', 'exceedance_rate') }}
            as return_period_years_shape_high

    from exceedances

),

judged as (

    select
        *,

        -- One order of magnitude, and NULL at the high end counts as too wide:
        -- a row whose upper reading is "beyond the fitted endpoint" has an
        -- unbounded band, not a missing one.
        (
            return_period_years_shape_low is not null
            and return_period_years_shape_high is not null
            and return_period_years_shape_high
                <= 10.0 * return_period_years_shape_low
        ) as return_period_is_reportable

    from computed

)

select
    *,

    -- What the Anomaly Map is allowed to say, decided once here rather than in
    -- the view, so the SQL and the tooltip cannot drift apart. A reader gets a
    -- number when the fit supports a number and a floor when it does not, and
    -- never a number whose last two digits are noise.
    (case
        when return_period_is_reportable
            then round(return_period_years::numeric, 1)
        else round(return_period_years_shape_low::numeric, 1)
     end) as return_period_years_quoted,
    (case
        when return_period_is_reportable then 'about'
        else 'at least'
     end) as return_period_qualifier

from judged
