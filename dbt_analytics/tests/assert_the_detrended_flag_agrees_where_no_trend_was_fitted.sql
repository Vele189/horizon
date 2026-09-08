-- The same statement one model downstream, on the flag a reader actually sees.
--
-- `fact_climatology` asserting that an unfitted trend leaves the baseline alone
-- is a claim about two columns; this is the claim about the product. Where no
-- trend was fitted the two flags must be the same flag, so any difference
-- between the detrended anomaly rate and the plain one is attributable to a
-- trend that was actually estimated, and never to a join that lost rows or a
-- null that turned into a false.
select
    city_id,
    date_key,
    trend_reference_years,
    z_temperature_2m_mean,
    z_temperature_2m_mean_detrended,
    is_anomaly,
    is_anomaly_detrended
from {{ ref('fact_weather_anomalies') }}
where trend_slope_c_per_year is null
  and (is_anomaly          is distinct from is_anomaly_detrended
    or anomaly_direction   is distinct from anomaly_direction_detrended
    or z_temperature_2m_mean is distinct from z_temperature_2m_mean_detrended)
