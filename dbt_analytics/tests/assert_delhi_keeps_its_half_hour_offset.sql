-- Delhi is UTC+05:30, and the thirty minutes must survive every conversion.
--
-- Integer-hour arithmetic is the trap: `observation_time + interval '5 hours'`
-- looks like a timezone conversion, passes review, and is wrong by half an
-- hour for a sixth of the world's population. Every hourly reading converted
-- to Asia/Kolkata must land on :30, never on the hour.
select
    city_id,
    observation_time,
    {{ to_local_time('observation_time', "'Asia/Kolkata'") }} as local_time
from {{ ref('stg_observations_hourly') }}
where city_id = 'delhi'
  and extract(minute from {{ to_local_time('observation_time', "'Asia/Kolkata'") }}) <> 30
