-- Air cannot hold a dew point above its own temperature.
--
-- Two things make this less obvious than it looks.
--
-- The tolerance is not slack, it is the source's precision. ERA5 reports both
-- fields to 0.1 °C, so when the air is saturated and the two are genuinely
-- equal, independent rounding can put the dew point one step above. Measured
-- across 274,920 hourly rows: 16 exceed, every one by exactly one step, all in
-- Singapore. A zero-tolerance assertion would fail on correct data, which is
-- how a real check gets switched off.
--
-- And the comparison is in `numeric`, not in the columns' own `real`. In
-- float4, 23.1 - 23.0 is 0.10000038, so `> temperature + 0.1` is true for a
-- difference of exactly one step and this test failed on 14 rows that are
-- correct. A tolerance compared in floating point is not the tolerance you
-- wrote.
{% set tolerance = 0.1 %}

select 'daily' as grain, city_id, observation_time,
       temperature_2m_mean as temperature, dew_point_2m_mean as dew_point
from {{ ref('stg_observations_daily') }}
where (dew_point_2m_mean::numeric - temperature_2m_mean::numeric) > {{ tolerance }}

union all

select 'hourly' as grain, city_id, observation_time,
       temperature_2m as temperature, dew_point_2m as dew_point
from {{ ref('stg_observations_hourly') }}
where (dew_point_2m::numeric - temperature_2m::numeric) > {{ tolerance }}
