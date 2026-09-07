-- The counterweight to the Phoenix test.
--
-- "Phoenix never shifts" is only evidence of correct timezone handling if some
-- other city does shift. A pipeline that dropped timezone conversion entirely
-- would satisfy Phoenix perfectly. Portland is on the same longitude and does
-- observe daylight saving, so it must take exactly two distinct offsets.
with offsets as (

    select distinct
        ({{ to_local_time('observation_time', "'America/Los_Angeles'") }}
            at time zone 'UTC') - observation_time as offset_from_utc
    from {{ ref('stg_observations_hourly') }}
    where city_id = 'portland'

)

select offset_from_utc
from offsets
where (select count(*) from offsets) <> 2
