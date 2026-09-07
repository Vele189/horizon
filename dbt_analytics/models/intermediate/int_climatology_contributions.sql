-- Each source year's contribution to each target day's window.
--
-- The pieces a leave-one-year-out baseline is assembled from: for a city, a
-- target day, and a source year, the count and the power sums of everything
-- that year contributed to that day's ±N window.
--
-- Sums rather than a recomputed aggregate because the whole point is to
-- subtract. Computing a leave-one-out mean directly would mean re-aggregating
-- the window once per excluded year — thirty-one passes over the same rows.
-- With Σn, Σx and Σx² the exclusion is arithmetic, and the identity
-- σ² = (Σx² − (Σx)²/n) / (n−1) recovers the standard deviation. A test
-- cross-checks the result against Postgres's own stddev_samp on the case
-- where nothing is excluded, so the algebra is verified rather than trusted.
with observations as (

    select
        facts.city_id,
        calendar.climatology_day,
        calendar.year,
        facts.temperature_2m_mean,
        facts.temperature_2m_max,
        facts.temperature_2m_min
    from {{ ref('fact_weather_observations') }} as facts
    join {{ ref('dim_date') }} as calendar
      on calendar.date_day = facts.date_key
    where facts.temperature_2m_mean is not null
    {% if var('climatology_start_year', none) is not none %}
      and calendar.year >= {{ var('climatology_start_year') }}
    {% endif %}
    {% if var('climatology_end_year', none) is not none %}
      and calendar.year <= {{ var('climatology_end_year') }}
    {% endif %}

)

select
    observations.city_id,
    window_days.target_day,
    observations.year as source_year,

    count(*) as observations,

    sum(observations.temperature_2m_mean::numeric)      as sum_mean,
    sum(observations.temperature_2m_mean::numeric ^ 2)  as sum_sq_mean,
    sum(observations.temperature_2m_max::numeric)       as sum_max,
    sum(observations.temperature_2m_max::numeric ^ 2)   as sum_sq_max,
    sum(observations.temperature_2m_min::numeric)       as sum_min,
    sum(observations.temperature_2m_min::numeric ^ 2)   as sum_sq_min

from observations
join {{ ref('int_climatology_window') }} as window_days
  on window_days.source_day = observations.climatology_day
group by observations.city_id, window_days.target_day, observations.year
