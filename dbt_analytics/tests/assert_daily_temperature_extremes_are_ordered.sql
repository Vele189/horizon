-- Daily minimum cannot exceed daily maximum.
--
-- Both columns pass their range test whichever way round they are, so a swap
-- during a refactor would be invisible to the bounds alone.
select
    city_id,
    observation_time,
    temperature_2m_min,
    temperature_2m_max
from {{ ref('stg_observations_daily') }}
where temperature_2m_min is not null
  and temperature_2m_max is not null
  and temperature_2m_min > temperature_2m_max
