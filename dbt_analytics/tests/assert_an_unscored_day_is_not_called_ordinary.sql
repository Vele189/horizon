-- Where there is no leakage-free baseline, every flag must be null.
--
-- Calling those days `none` would assert they were ordinary on no evidence,
-- and would silently pad the denominator of every anomaly rate with days that
-- could not have been flagged.
select
    city_id,
    date_key,
    z_temperature_2m_mean,
    is_anomaly,
    anomaly_direction
from {{ ref('fact_weather_anomalies') }}
where z_temperature_2m_mean is null
  and (is_anomaly is not null or anomaly_direction is not null)
