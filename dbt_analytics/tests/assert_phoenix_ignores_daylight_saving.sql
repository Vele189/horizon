-- Phoenix keeps standard time all year, on a longitude where its neighbours
-- do not.
--
-- The trap is a pipeline that infers a zone from a country or a longitude
-- rather than reading the configured IANA name. Such a pipeline gives Phoenix
-- America/Denver, and every Phoenix reading between March and November lands
-- an hour out — a shift small enough to look like weather.
--
-- Asserted as an invariant rather than by comparing two chosen dates: across
-- every US transition in the window, Phoenix's offset from UTC must never
-- change. Its DST-observing neighbour Portland has its own test below, so a
-- bug that flattened every city to a single offset fails there instead.
with offsets as (

    select distinct
        ({{ to_local_time('observation_time', "'America/Phoenix'") }}
            at time zone 'UTC') - observation_time as offset_from_utc
    from {{ ref('stg_observations_hourly') }}
    where city_id = 'phoenix'

)

select offset_from_utc
from offsets
where (select count(*) from offsets) > 1
