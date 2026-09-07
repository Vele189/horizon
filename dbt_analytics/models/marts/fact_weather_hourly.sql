{{
    config(
        materialized = 'table',
        indexes = [
            {'columns': ['city_id', 'observation_hour'], 'unique': True},
            {'columns': ['observation_hour']},
        ],
    )
}}

-- Hourly facts for the trailing 24 months, feeding the Storm Dynamics view and
-- nothing else.
--
-- Thirty years at this grain would be over four million rows for no analytical
-- benefit — the storm view reads a rolling window, and the climatology is
-- built from the daily grain. It would also be the single largest object the
-- pipeline produces, which matters because gold is the only layer promoted to
-- Neon's 0.5 GB allowance.
--
-- The window is derived from the data rather than pinned. Silver holds
-- whatever ingestion landed, and two cities carry an extra eight months from
-- an earlier archival sample that took a calendar year rather than the
-- anchored window; without this filter they would contribute rows outside the
-- 24 months this table claims to cover.
with bounds as (

    select
        (max(observation_time) at time zone 'UTC')::date as window_end,
        ((max(observation_time) at time zone 'UTC')::date
            - interval '{{ var("hourly_window_months", 24) }} months')::date
            as window_start
    from {{ ref('stg_observations_hourly') }}

),

windowed as (

    select observations.*
    from {{ ref('stg_observations_hourly') }} as observations
    cross join bounds
    where (observations.observation_time at time zone 'UTC')::date
              >= bounds.window_start
      and (observations.observation_time at time zone 'UTC')::date
              <= bounds.window_end

),

with_tendency as (

    select
        *,

        -- Pressure tendency: the change in sea-level pressure over a fixed
        -- span, and the primary storm-development signal. Three hours is the
        -- synoptic standard; twenty-four is what "bomb cyclone" is defined on
        -- (a fall of roughly 24 hPa in 24 hours at mid-latitudes).
        --
        -- Sea-level pressure, not surface: surface pressure carries the grid
        -- cell's elevation, so a tendency computed on it would compare
        -- Johannesburg's 822 hPa against London's 1013 the moment anything
        -- aggregated across cities.
        --
        -- The frame is RANGE over an interval, not `lag(n)`. `lag(pressure, 3)`
        -- counts *rows*, so a single missing hour makes it reach four hours
        -- back and report the result as a three-hour change — a fabricated
        -- storm signal, from data that merely had a hole. RANGE asks for the
        -- reading exactly three hours earlier and returns null when there
        -- isn't one, which is the honest answer. Silver has no gaps today;
        -- this costs nothing and stays right if it ever does.
        pressure_msl - first_value(pressure_msl) over (
            partition by city_id
            order by observation_time
            range between interval '3 hours' preceding
                      and interval '3 hours' preceding
        ) as pressure_tendency_3h,

        pressure_msl - first_value(pressure_msl) over (
            partition by city_id
            order by observation_time
            range between interval '24 hours' preceding
                      and interval '24 hours' preceding
        ) as pressure_tendency_24h

    from windowed

)

select
    -- Keys -------------------------------------------------------------
    city_id,
    observation_time as observation_hour,
    (observation_time at time zone 'UTC')::date as date_key,

    -- Which ERA5 grid cell answered.
    api_latitude,
    api_longitude,
    api_elevation_m,

    -- Storm dynamics ----------------------------------------------------
    pressure_msl,
    surface_pressure,
    pressure_tendency_3h,
    pressure_tendency_24h,
    wind_speed_10m,
    wind_gusts_10m,
    wind_direction_10m,

    -- Thermodynamics ----------------------------------------------------
    temperature_2m,
    apparent_temperature,
    dew_point_2m,
    relative_humidity_2m,

    -- Precipitation and cloud -------------------------------------------
    precipitation,
    cloud_cover,
    weather_code,

    -- Lineage -----------------------------------------------------------
    ingested_at,
    batch_id

from with_tendency
