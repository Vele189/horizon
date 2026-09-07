-- The table claims 24 months; this asserts it holds 24 months.
--
-- Silver carries more than the window: two cities have an extra eight months
-- from an archival sample that took a calendar year rather than the anchored
-- period. Without the filter they would quietly widen a table documented as
-- trailing-24-months, and any per-city average over "the window" would be
-- computed over different windows for different cities.
with span as (

    select
        city_id,
        min(observation_hour) as first_hour,
        max(observation_hour) as last_hour,
        (max(observation_hour) at time zone 'UTC')::date
            - (min(observation_hour) at time zone 'UTC')::date as days
    from {{ ref('fact_weather_hourly') }}
    group by city_id

)

select city_id, first_hour, last_hour, days
from span
-- 24 months is 730 or 731 days depending on where the leap day falls.
where days not between 729 and 732
