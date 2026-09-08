-- Sydney's clocks go forward in October and back in April.
--
-- The trap is a pipeline that hardcodes the northern calendar, March forward
-- and October back, which for Sydney is not merely wrong but inverted: it would
-- add an hour exactly when an hour should be subtracted, doubling the error.
--
-- Asserted by checking that Sydney's UTC offset really does take both values
-- and takes the daylight-saving one in the southern summer. January is high
-- summer in Sydney and must be UTC+11; July is winter and must be UTC+10.
with offsets as (
    select
        extract(month from observation_time) as utc_month,
        extract(
            hour from
            ({{ to_local_time('observation_time', "'Australia/Sydney'") }}
                at time zone 'UTC') - observation_time
        ) as offset_hours
    from {{ ref('stg_observations_hourly') }}
    where city_id = 'sydney'
)
select utc_month, offset_hours, count(*) as rows
from offsets
where (utc_month = 1 and offset_hours <> 11)
   or (utc_month = 7 and offset_hours <> 10)
group by utc_month, offset_hours
