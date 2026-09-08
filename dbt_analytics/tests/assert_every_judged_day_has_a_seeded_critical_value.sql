-- No judged day may fall back to the normal quantile.
--
-- The seed covers one to a thousand degrees of freedom, and above it the model
-- coalesces to the nominal |Z|. That fallback exists so a widened smoothing
-- window cannot break the build, and on this data nothing uses it -- which is
-- worth asserting rather than assuming, because a row that took the fallback
-- would be judged by the old rule while every column around it claimed the new
-- one. The failure is invisible: the threshold is still a plausible number, and
-- it is the number DBT-14 exists to stop using.
select
    city_id,
    date_key,
    baseline_observations,
    baseline_degrees_of_freedom,
    anomaly_z_critical
from {{ ref('fact_weather_anomalies') }}
where is_anomaly is not null
  and (baseline_degrees_of_freedom < 1
    or baseline_degrees_of_freedom > 1000
    or anomaly_z_critical is null)
