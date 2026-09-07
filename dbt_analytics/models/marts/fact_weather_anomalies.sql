{{
    config(
        materialized = 'table',
        indexes = [
            {'columns': ['city_id', 'date_key'], 'unique': True},
            {'columns': ['anomaly_direction']},
            {'columns': ['city_id', 'is_anomaly']},
        ],
    )
}}

-- Daily temperature anomalies as Z-scores against the leakage-safe baseline.
--
-- A separate model rather than columns on fact_weather_observations, because
-- the climatology is built *from* that fact and adding its output back would
-- be a cycle.
--
-- **Both tails.** A cold anomaly at Z < -2.5 is as much an extreme event as a
-- heat anomaly, and the flag is on abs(z). Writing `z > 2.5` reads naturally
-- and silently discards half the signal; Moscow's January is the strongest
-- case, and a warm-only project would report it as one of the calmest cities
-- in the set.
--
-- **A null Z is not a quiet day.** Where the leave-one-year-out baseline has
-- no observations left — a city with a single reference year — sigma is null,
-- Z is null, and so are the flags. "Unknown" is the honest answer; coercing to
-- `none` would assert the day was ordinary on no evidence, and would put those
-- days in the denominator of every anomaly rate.
with joined as (

    select
        facts.city_id,
        facts.date_key,
        calendar.month_day,
        calendar.year,
        calendar.climatology_day,

        facts.temperature_2m_mean,
        facts.temperature_2m_max,
        facts.temperature_2m_min,

        climatology.mean_temperature_2m_mean,
        climatology.stddev_temperature_2m_mean,
        climatology.mean_temperature_2m_max,
        climatology.stddev_temperature_2m_max,
        climatology.mean_temperature_2m_min,
        climatology.stddev_temperature_2m_min,
        climatology.observations as baseline_observations,
        climatology.excludes_own_year

    from {{ ref('fact_weather_observations') }} as facts
    join {{ ref('dim_date') }} as calendar
      on calendar.date_day = facts.date_key
    left join {{ ref('fact_climatology') }} as climatology
      on climatology.city_id = facts.city_id
     and climatology.month_day = calendar.month_day
     and climatology.for_year = calendar.year

),

scored as (

    select
        *,
        {{ z_score('temperature_2m_mean', 'mean_temperature_2m_mean',
                   'stddev_temperature_2m_mean') }} as z_temperature_2m_mean,
        {{ z_score('temperature_2m_max', 'mean_temperature_2m_max',
                   'stddev_temperature_2m_max') }} as z_temperature_2m_max,
        {{ z_score('temperature_2m_min', 'mean_temperature_2m_min',
                   'stddev_temperature_2m_min') }} as z_temperature_2m_min
    from joined

)

select
    city_id,
    date_key,
    month_day,
    year,

    -- Observed, and the baseline it is scored against.
    temperature_2m_mean,
    mean_temperature_2m_mean,
    stddev_temperature_2m_mean,
    baseline_observations,
    excludes_own_year,

    -- Departure in degrees as well as in sigmas. The Z-score answers "how
    -- unusual"; this answers "how much", and a dashboard needs both — 2.6
    -- sigma is a headline in Singapore at 1.8 °C and in Moscow at 20 °C.
    (temperature_2m_mean - mean_temperature_2m_mean) as departure_c,

    z_temperature_2m_mean,
    z_temperature_2m_max,
    z_temperature_2m_min,

    -- The flag, on both tails. Null where no baseline exists.
    (case when z_temperature_2m_mean is null then null
          else abs(z_temperature_2m_mean) > {{ var('anomaly_z_threshold', 2.5) }}
     end) as is_anomaly,

    {{ anomaly_direction('z_temperature_2m_mean',
                         var('anomaly_z_threshold', 2.5)) }} as anomaly_direction

from scored
