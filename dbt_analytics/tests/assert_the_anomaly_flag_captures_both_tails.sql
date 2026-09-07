-- The flag must be on abs(z), not on z.
--
-- `z > 2.5` reads naturally, passes review, and silently discards every cold
-- extreme. Phoenix would lose 125 of its 145 flagged days and Moscow's January
-- would vanish entirely, leaving a project that reports the coldest city in
-- the set as one of the calmest.
--
-- Asserted from the data rather than from the SQL: every row past the
-- threshold in either direction must be flagged, and no row inside it may be.
select
    city_id,
    date_key,
    z_temperature_2m_mean,
    is_anomaly,
    anomaly_direction
from {{ ref('fact_weather_anomalies') }}
where z_temperature_2m_mean is not null
  and (
        (abs(z_temperature_2m_mean) > {{ var('anomaly_z_threshold', 2.5) }}
         and (is_anomaly is not true or anomaly_direction = 'none'))
     or (abs(z_temperature_2m_mean) <= {{ var('anomaly_z_threshold', 2.5) }}
         and (is_anomaly is not false or anomaly_direction <> 'none'))
  )
