-- A null tendency is correct at the start of a city's series and nowhere else.
--
-- Exactly three nulls per city for the 3-hour column and twenty-four for the
-- 24-hour one. More than that means the frame stopped matching — a gap, or a
-- partition key that lost a city.
with nulls as (

    select
        city_id,
        count(*) filter (where pressure_tendency_3h is null)  as null_3h,
        count(*) filter (where pressure_tendency_24h is null) as null_24h
    from {{ ref('fact_weather_hourly') }}
    where pressure_msl is not null
    group by city_id

)

select city_id, null_3h, null_24h
from nulls
where null_3h <> 3 or null_24h <> 24
