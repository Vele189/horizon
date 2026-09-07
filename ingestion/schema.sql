-- ============================================================================
-- Bronze landing schema
-- ============================================================================
-- Idempotent by construction: every statement is IF NOT EXISTS, IF EXISTS, or
-- a COMMENT, so re-running against an existing database is a no-op. Apply with
--
--     python ingestion/apply_schema.py
--
-- Bronze is append-only. Re-ingesting a city-date range inserts new rows
-- rather than updating existing ones; silver deduplicates by taking the most
-- recent `ingested_at` per (city_id, observation_time). Nothing here may be
-- mutated in place — the whole point of an append-only landing zone is that a
-- bad transformation is recoverable without re-hitting the API.
--
-- There is deliberately NO jsonb payload column. Storing the raw response
-- per row would multiply the footprint several times over for data already
-- parsed into the columns beside it, would bloat the local database, and
-- would be fatal if bronze were ever promoted to Neon's 0.5 GB allowance.
-- Raw payloads are archived to disk as gzip under DATA_RAW_DIR (ING-03), so
-- replay remains possible without paying for it in the warehouse.
--
-- There is likewise NO source_url column, for the same reason at a smaller
-- scale. It was measured at 712 bytes per daily row and 466 per hourly one —
-- 83% and 81% of the row payload, roughly 300 MB across the full backfill —
-- and every value was one of a few hundred distinct strings repeated once per
-- row. It is also fully derivable: ingestion.client.request_url() rebuilds the
-- exact string from (city_id, grain, start, end), which is what ING-03's
-- replay already depends on. Use ingestion.archive.source_url_for() to recover
-- it for a given row.
--
-- Units are Open-Meteo's metric defaults, asserted in dbt rather than
-- converted here (§5.3). Column names match the API's variable names so a
-- row can be traced back to the request that produced it.
-- ============================================================================

create schema if not exists bronze_raw;
create schema if not exists silver_staging;
create schema if not exists gold_marts;

comment on schema bronze_raw is
  'Append-only landing zone. Parsed API columns plus ingestion metadata; no raw payloads.';
comment on schema silver_staging is
  'Deduplicated, unit-asserted, UTC-cast staging models built by dbt.';
comment on schema gold_marts is
  'Star schema, climatology, and predictions. The only layer promoted to Neon.';


-- ----------------------------------------------------------------------------
-- bronze_raw.observations_daily
-- ----------------------------------------------------------------------------
-- One row per city per UTC day, ~30 years back (1995 →). Approximately
-- 164,000 rows at 15 cities.
-- ----------------------------------------------------------------------------
create table if not exists bronze_raw.observations_daily (
    -- Surrogate key. Bronze permits duplicates by design, so the natural key
    -- (city_id, observation_time) cannot be the primary key.
    id                          bigint generated always as identity primary key,

    -- Ingestion metadata — every row carries all four.
    city_id                     text        not null,
    observation_time            timestamptz not null,
    ingested_at                 timestamptz not null default now(),
    batch_id                    uuid        not null,

    -- Which ERA5 grid cell actually answered. Open-Meteo snaps a request to
    -- the nearest cell, so these differ from the city's configured
    -- coordinates and elevation (London: 51.5074/-0.1278 at 11 m resolves to
    -- 51.4938/-0.1630 at 16 m). Recorded per row so provenance survives a
    -- change to cities.yml.
    api_latitude                double precision,
    api_longitude               double precision,
    api_elevation_m             real,

    -- Temperature, °C
    temperature_2m_max          real,
    temperature_2m_min          real,
    temperature_2m_mean         real,
    apparent_temperature_max    real,
    apparent_temperature_min    real,
    apparent_temperature_mean   real,

    -- Precipitation. snowfall_sum is cm, everything else mm; precipitation_hours is h.
    precipitation_sum           real,
    rain_sum                    real,
    snowfall_sum                real,
    precipitation_hours         real,

    -- Wind, km/h; direction in degrees
    wind_speed_10m_max          real,
    wind_speed_10m_mean         real,
    wind_gusts_10m_max          real,
    wind_direction_10m_dominant smallint,

    -- Pressure, hPa
    surface_pressure_mean       real,
    pressure_msl_mean           real,

    -- Moisture and cloud
    relative_humidity_2m_mean   smallint,   -- %
    dew_point_2m_mean           real,       -- °C
    cloud_cover_mean            smallint,   -- %

    -- Radiation, MJ/m²
    shortwave_radiation_sum     real,

    -- WMO code
    weather_code                smallint,

    -- Daily aggregates are computed over UTC calendar days, so the timestamp
    -- must land exactly on midnight UTC. A row at any other time means the
    -- request was made without timezone=UTC and the day boundaries are wrong.
    constraint observations_daily_midnight_utc
        check (observation_time = date_trunc('day', observation_time at time zone 'UTC') at time zone 'UTC'),
    constraint observations_daily_city_id_not_blank
        check (length(btrim(city_id)) > 0)
);

comment on table bronze_raw.observations_daily is
  'Append-only daily observations from the Open-Meteo archive (ERA5). Deduplicated downstream in silver.';
comment on column bronze_raw.observations_daily.city_id is
  'Slug from config/cities.yml. Not a foreign key — bronze must land even if the city config changes.';
comment on column bronze_raw.observations_daily.observation_time is
  'Midnight UTC of the aggregated day.';
comment on column bronze_raw.observations_daily.batch_id is
  'Groups every row written by one extraction run; lets a bad batch be deleted wholesale.';


-- ----------------------------------------------------------------------------
-- bronze_raw.observations_hourly
-- ----------------------------------------------------------------------------
-- One row per city per hour, trailing 24 months only — all the storm-dynamics
-- view needs (§5.1). Approximately 263,000 rows at 15 cities.
-- ----------------------------------------------------------------------------
create table if not exists bronze_raw.observations_hourly (
    id                       bigint generated always as identity primary key,

    city_id                  text        not null,
    observation_time         timestamptz not null,
    ingested_at              timestamptz not null default now(),
    batch_id                 uuid        not null,

    api_latitude             double precision,
    api_longitude            double precision,
    api_elevation_m          real,

    temperature_2m           real,       -- °C
    apparent_temperature     real,       -- °C
    relative_humidity_2m     smallint,   -- %
    dew_point_2m             real,       -- °C

    surface_pressure         real,       -- hPa
    pressure_msl             real,       -- hPa

    wind_speed_10m           real,       -- km/h
    wind_gusts_10m           real,       -- km/h
    wind_direction_10m       smallint,   -- degrees

    precipitation            real,       -- mm
    cloud_cover              smallint,   -- %
    weather_code             smallint,

    constraint observations_hourly_on_the_hour
        check (date_part('minute', observation_time at time zone 'UTC') = 0
               and date_part('second', observation_time at time zone 'UTC') = 0),
    constraint observations_hourly_city_id_not_blank
        check (length(btrim(city_id)) > 0)
);

comment on table bronze_raw.observations_hourly is
  'Append-only hourly observations, trailing 24 months. Feeds the storm-dynamics view.';
comment on column bronze_raw.observations_hourly.observation_time is
  'Top of the hour, UTC.';


-- ----------------------------------------------------------------------------
-- Migrations
-- ----------------------------------------------------------------------------
-- Idempotent like everything above: a no-op on a database created from the
-- current DDL, and the corrective step on one created before it. Postgres
-- marks a dropped column dead rather than rewriting the heap, so the space is
-- only returned by a VACUUM FULL — which is not run here, because it takes an
-- exclusive lock and that is an operator's decision, not a schema file's.
alter table bronze_raw.observations_daily  drop column if exists source_url;
alter table bronze_raw.observations_hourly drop column if exists source_url;


-- ----------------------------------------------------------------------------
-- Indexes
-- ----------------------------------------------------------------------------
-- The silver dedup is a window function partitioned by (city_id,
-- observation_time) ordered by ingested_at desc. Carrying ingested_at as a
-- third key in matching order lets that read straight from the index instead
-- of sorting each partition. The leading two columns still serve ordinary
-- city-and-date lookups.
create index if not exists observations_daily_city_time_idx
    on bronze_raw.observations_daily (city_id, observation_time, ingested_at desc);

create index if not exists observations_hourly_city_time_idx
    on bronze_raw.observations_hourly (city_id, observation_time, ingested_at desc);

-- Deleting or auditing a single extraction run.
create index if not exists observations_daily_batch_idx
    on bronze_raw.observations_daily (batch_id);

create index if not exists observations_hourly_batch_idx
    on bronze_raw.observations_hourly (batch_id);
