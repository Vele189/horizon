-- The flag must be on abs(z), not on z.
--
-- `z > 2.5` reads naturally, passes review, and silently discards every cold
-- extreme. Phoenix would lose 125 of its 145 flagged days and Moscow's January
-- would vanish entirely, leaving a project that reports the coldest city in
-- the set as one of the calmest.
--
-- Asserted from the data rather than from the SQL: every row past the
-- threshold in either direction must be flagged, and no row inside it may be.
--
-- **The threshold is `anomaly_z_critical` and no longer the configured
-- constant (DBT-14).** This test compared against `anomaly_z_threshold`, and
-- when sigma's uncertainty started widening the bar it failed on 52 rows --
-- correctly, and for the wrong reason. Those rows are the ones the correction
-- removed: departures past 2.5 whose baseline was too thin to support the
-- claim. The property worth keeping is that the flag is on `abs(z)` and agrees
-- with the bar it was judged at; the constant was never the property, it was
-- how the bar happened to be spelled.
select
    city_id,
    date_key,
    z_temperature_2m_mean,
    anomaly_z_critical,
    is_anomaly,
    anomaly_direction
from {{ ref('fact_weather_anomalies') }}
where z_temperature_2m_mean is not null
  and (
        (abs(z_temperature_2m_mean) > anomaly_z_critical
         and (is_anomaly is not true or anomaly_direction = 'none'))
     or (abs(z_temperature_2m_mean) <= anomaly_z_critical
         and (is_anomaly is not false or anomaly_direction <> 'none'))
  )
