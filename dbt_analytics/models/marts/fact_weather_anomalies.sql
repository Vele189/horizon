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
-- no observations left (a city with a single reference year) sigma is null,
-- Z is null, and so are the flags. "Unknown" is the honest answer; coercing to
-- `none` would assert the day was ordinary on no evidence, and would put those
-- days in the denominator of every anomaly rate.
--
-- **Two flags, and nothing switches over yet (DBT-12).** `is_anomaly` scores
-- against a baseline spanning the whole record, so a warming city is measured
-- against a mean that includes its own cooler decades and "anomalously hot"
-- drifts towards meaning "recent". `is_anomaly_detrended` scores against a
-- baseline with a per-city, per-day linear trend in year taken out of it,
-- fitted only on years before the one being labelled. Both are carried, in
-- full, side by side. Nothing downstream reads the second one: the two are
-- different products -- *unusual for this era* against *unusual for the
-- record* -- and DBT-13 is the ticket that answers, in writing, which question
-- this project is asking. Letting a default settle it is the failure mode this
-- shape exists to prevent.
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
        climatology.excludes_own_year,

        climatology.mean_temperature_2m_mean_detrended,
        climatology.stddev_temperature_2m_mean_detrended,
        climatology.trend_slope_c_per_year,
        climatology.trend_stderr_c_per_year,
        climatology.trend_reference_years

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
                   'stddev_temperature_2m_min') }} as z_temperature_2m_min,

        -- The observation itself is not adjusted, and does not need to be: the
        -- baseline was detrended *to this row's own year*, so the trend term
        -- for the observation is exactly zero. What moved is the mean it is
        -- measured from.
        {{ z_score('temperature_2m_mean', 'mean_temperature_2m_mean_detrended',
                   'stddev_temperature_2m_mean_detrended') }}
            as z_temperature_2m_mean_detrended
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
    -- unusual"; this answers "how much", and a dashboard needs both: 2.6
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
                         var('anomaly_z_threshold', 2.5)) }} as anomaly_direction,

    -- DBT-12. The same three columns against the detrended baseline, and the
    -- trend that produced it, so a reader can see how much was taken out and
    -- how well it was known before deciding whether to believe the difference.
    mean_temperature_2m_mean_detrended,
    stddev_temperature_2m_mean_detrended,
    trend_slope_c_per_year,
    trend_stderr_c_per_year,
    trend_reference_years,

    (temperature_2m_mean - mean_temperature_2m_mean_detrended)
        as departure_c_detrended,

    z_temperature_2m_mean_detrended,

    (case when z_temperature_2m_mean_detrended is null then null
          else abs(z_temperature_2m_mean_detrended)
               > {{ var('anomaly_z_threshold', 2.5) }}
     end) as is_anomaly_detrended,

    {{ anomaly_direction('z_temperature_2m_mean_detrended',
                         var('anomaly_z_threshold', 2.5)) }}
        as anomaly_direction_detrended

from scored
