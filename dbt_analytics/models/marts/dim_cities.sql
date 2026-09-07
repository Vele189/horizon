-- The city dimension.
--
-- Built from config/cities.yml, never from observation data. That direction
-- matters: a dimension inferred from what happened to land would list fourteen
-- cities while a backfill is mid-flight, and would silently lose a city whose
-- ingestion failed. The registry says what the set *is*; the facts say what has
-- been observed of it, and the difference between the two is exactly what the
-- reconciliation report exists to surface.
select
    city_id,
    name,
    country,
    country_code,
    region,
    latitude,
    longitude,
    elevation_m,
    timezone,
    koppen,

    -- Derived from the coordinate, never configured. A hardcoded hemisphere is
    -- a second source of truth that can disagree with the latitude beside it,
    -- and the season mapping downstream reads this — so for the five southern
    -- cities a wrong value inverts summer and winter rather than merely
    -- mislabelling them. Asserted against the registry's own derivation in
    -- tests/, so Python and SQL cannot drift apart.
    case when latitude >= 0 then 'north' else 'south' end as hemisphere,

    season_model,
    role

from {{ ref('stg_cities') }}
