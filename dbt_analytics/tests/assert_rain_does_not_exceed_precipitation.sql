-- Rain is a component of total precipitation, so it cannot be larger than it.
--
-- This is the assertion that would catch snowfall_sum being added to
-- precipitation without its centimetre-to-millimetre conversion, or the two
-- columns being mapped to each other's API variables.
select
    city_id,
    observation_time,
    rain_sum,
    precipitation_sum
from {{ ref('stg_observations_daily') }}
where rain_sum is not null
  and precipitation_sum is not null
  -- numeric, not real: see assert_dew_point_does_not_exceed_temperature
  and (rain_sum::numeric - precipitation_sum::numeric) > 0.01
