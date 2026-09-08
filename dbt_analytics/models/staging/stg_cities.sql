-- The city registry, typed and made available to SQL.
--
-- config/cities.yml is the single source; dbt_analytics/export_cities.py
-- generates the seed and a test asserts the two agree. The timezone is the
-- column that matters most here: it is what makes a local-time view possible
-- downstream without every model re-deriving it.
select
    city_id,
    name,
    country,
    country_code,
    region,
    latitude::double precision      as latitude,
    longitude::double precision     as longitude,
    elevation_m::real               as elevation_m,
    timezone,
    hemisphere,
    koppen,
    season_model,
    nullif(role, '')                as role
from {{ ref('cities') }}
