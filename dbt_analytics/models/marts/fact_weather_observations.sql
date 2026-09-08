{{
    config(
        materialized = 'table',
        indexes = [
            {'columns': ['city_id', 'date_key'], 'unique': True},
            {'columns': ['date_key']},
        ],
    )
}}

-- The primary daily fact: one row per city per UTC day.
--
-- Selected from silver rather than joined to the dimensions. An inner join to
-- dim_cities would enforce referential integrity by *dropping* any row it
-- could not match, which is the wrong failure: a fact silently missing a city
-- looks exactly like a city with no weather. The relationships tests assert
-- the same property and fail loudly instead.
--
-- Silver has already deduplicated on (city_id, observation_time) and asserted
-- units, so the grain here is inherited rather than re-derived. The unique
-- test on (city_id, date_key) is what proves that inheritance held.
select
    -- Keys -------------------------------------------------------------
    observations.city_id,

    -- The dimensional key, and the grain. Daily aggregates are computed over
    -- UTC calendar days upstream, since ING-01 requests timezone=UTC and
    -- asserts the offset is zero on every response, so this cast is a projection of a
    -- day that is already a UTC day, not a timezone conversion.
    (observations.observation_time at time zone 'UTC')::date as date_key,

    -- The exact instant, carried alongside the key. Eight bytes for the
    -- ability to join back to silver and bronze without reconstructing a
    -- midnight, and for a timestamp that says which zone it is in, which the date
    -- above does not.
    observations.observation_time,

    -- Which ERA5 grid cell answered, as distinct from the coordinate
    -- dim_cities holds, which is the one that was asked for.
    observations.api_latitude,
    observations.api_longitude,
    observations.api_elevation_m,

    -- Temperature, °C ---------------------------------------------------
    observations.temperature_2m_min,
    observations.temperature_2m_max,
    observations.temperature_2m_mean,
    observations.apparent_temperature_min,
    observations.apparent_temperature_max,
    observations.apparent_temperature_mean,
    observations.dew_point_2m_mean,

    -- Precipitation. snowfall_sum is cm; snowfall_sum_mm is the comparable one.
    observations.precipitation_sum,
    observations.rain_sum,
    observations.snowfall_sum,
    observations.snowfall_sum_mm,
    observations.precipitation_hours,

    -- Wind, km/h; direction in degrees ----------------------------------
    observations.wind_speed_10m_mean,
    observations.wind_speed_10m_max,
    observations.wind_gusts_10m_max,
    observations.wind_direction_10m_dominant,

    -- Pressure, hPa. surface is at the grid cell's elevation, msl is reduced
    -- to sea level; Johannesburg at 1753 m reads 822 against an msl of 998.
    observations.surface_pressure_mean,
    observations.pressure_msl_mean,

    -- Moisture, cloud, radiation ----------------------------------------
    observations.relative_humidity_2m_mean,
    observations.cloud_cover_mean,
    observations.shortwave_radiation_sum,

    -- WMO 4677 present-weather code.
    observations.weather_code,

    -- Lineage -----------------------------------------------------------
    -- Carried so a mart row can be traced to the run that produced it, and so
    -- a bad batch can be found here as well as in bronze.
    observations.ingested_at,
    observations.batch_id

from {{ ref('stg_observations_daily') }} as observations
