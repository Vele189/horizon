-- One row per city per UTC day.
--
-- The unique test proves the pair does not repeat. This proves the key means
-- what it says: date_key must be the UTC date of observation_time, so a
-- timezone slipping in anywhere between silver and here would put two
-- observations on one date_key for the cities furthest from Greenwich.
select
    city_id,
    date_key,
    observation_time
from {{ ref('fact_weather_observations') }}
where date_key <> (observation_time at time zone 'UTC')::date
