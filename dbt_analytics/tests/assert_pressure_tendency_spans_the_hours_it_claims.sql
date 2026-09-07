-- A tendency labelled 3h must be measured over exactly 3 hours.
--
-- This is the assertion that `lag(pressure, 3)` fails the moment a row is
-- missing: it would reach four hours back and report the difference as a
-- three-hour change. Recomputed here by joining on the timestamp rather than
-- by counting rows, so the two methods have to agree.
with recomputed as (

    select
        current_hour.city_id,
        current_hour.observation_hour,
        current_hour.pressure_tendency_3h as stored,
        current_hour.pressure_msl - three_hours_ago.pressure_msl as expected
    from {{ ref('fact_weather_hourly') }} as current_hour
    join {{ ref('fact_weather_hourly') }} as three_hours_ago
      on three_hours_ago.city_id = current_hour.city_id
     and three_hours_ago.observation_hour
         = current_hour.observation_hour - interval '3 hours'

)

select city_id, observation_hour, stored, expected
from recomputed
where stored is distinct from expected
  and abs(coalesce(stored, 0) - coalesce(expected, 0)) > 0.001
