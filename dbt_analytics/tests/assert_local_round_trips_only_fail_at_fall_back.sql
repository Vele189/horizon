-- Local time is a one-way view of UTC, and this bounds how lossy it is.
--
-- `AT TIME ZONE` returns a naive wall-clock reading, and at a daylight-saving
-- fall-back two different instants produce the same reading, so converting
-- back cannot recover which. That is a property of clocks, not a bug, and it
-- is why UTC is what silver stores and what everything joins on.
--
-- What would be a bug is conversion breaking wholesale. Measured across
-- 274,920 hourly rows, exactly ten fail to round trip: one per DST-observing
-- city per autumn transition in the window, each in the repeated hour. This
-- allows a small multiple of that and fails on anything more, so a conversion
-- that silently stopped working shows up as thousands rather than ten.
with round_trips as (

    select
        observations.city_id,
        count(*) as ambiguous_rows
    from {{ ref('stg_observations_hourly') }} as observations
    join {{ ref('stg_cities') }} as cities
      on cities.city_id = observations.city_id
    where (
        {{ to_local_time('observations.observation_time', 'cities.timezone') }}
            at time zone cities.timezone
    ) <> observations.observation_time
    group by observations.city_id

)

select city_id, ambiguous_rows
from round_trips
-- Two transitions a year, and the window is two years, so four is the ceiling
-- for a city and anything above it means conversion, not clocks.
where ambiguous_rows > 4
