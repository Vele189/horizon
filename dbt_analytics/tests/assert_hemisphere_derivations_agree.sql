-- The hemisphere is derived twice, in two languages, and they must agree.
--
-- cities.py computes it from `lat >= 0` and seeds it; dim_cities recomputes it
-- from the latitude column in SQL. Neither is authoritative over the other,
-- which is the point: a disagreement means one of them has drifted, and a
-- silent drift inverts summer and winter for the five southern cities in every
-- season mapping downstream.
--
-- The equator is the boundary and `>=` is the tie-break in both. A city at
-- exactly 0.0 would expose an off-by-one here; none is, but the test would
-- catch the day one is added.
select
    dimension.city_id,
    dimension.latitude,
    dimension.hemisphere as derived_in_sql,
    seeded.hemisphere    as derived_in_python
from {{ ref('dim_cities') }} as dimension
join {{ ref('stg_cities') }} as seeded
  on seeded.city_id = dimension.city_id
where dimension.hemisphere <> seeded.hemisphere
