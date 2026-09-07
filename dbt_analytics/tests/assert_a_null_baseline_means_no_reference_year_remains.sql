-- Null is permitted, but only for the one reason.
--
-- Every null sigma must be a row where excluding the labelled year emptied the
-- window. A null appearing anywhere else would mean the arithmetic failed
-- while looking like an honest absence.
select
    city_id,
    for_year,
    observations,
    reference_years,
    stddev_temperature_2m_mean
from {{ ref('fact_climatology') }}
where stddev_temperature_2m_mean is null
  and observations > 1
