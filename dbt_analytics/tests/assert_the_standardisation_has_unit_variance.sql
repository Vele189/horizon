-- A Z-score should have a standard deviation of one.
--
-- This is the check that catches a sigma computed over the wrong window, the
-- wrong grouping, or the wrong sample: all of those still produce a plausible
-- column of numbers, and all of them show up here as a spread that is not one.
--
-- Only cities with a full reference period are checked. A baseline built from
-- three years (Tokyo, mid-backfill) legitimately over-disperses, and holding
-- it to the same bar would fail on a sample-size effect rather than a defect.
with spread as (

    select
        city_id,
        stddev_samp(z_temperature_2m_mean) as sd_of_z,
        avg(baseline_observations) as baseline_size
    from {{ ref('fact_weather_anomalies') }}
    where z_temperature_2m_mean is not null
    group by city_id

)

select city_id, sd_of_z, baseline_size
from spread
where baseline_size > 300
  and (sd_of_z < 0.9 or sd_of_z > 1.1)
