-- hot means warmer than the baseline, cold means colder.
--
-- An inverted branch would flag the right days and label every one of them
-- backwards, which every count-based test would pass, and which would put
-- Moscow's January in the heatwave column.
select
    city_id,
    date_key,
    z_temperature_2m_mean,
    departure_c,
    anomaly_direction
from {{ ref('fact_weather_anomalies') }}
where (anomaly_direction = 'hot'  and z_temperature_2m_mean <= 0)
   or (anomaly_direction = 'cold' and z_temperature_2m_mean >= 0)
   or (anomaly_direction = 'hot'  and departure_c <= 0)
   or (anomaly_direction = 'cold' and departure_c >= 0)
