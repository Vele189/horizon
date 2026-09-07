-- The fact must carry every deduplicated silver observation, and no more.
--
-- Asserted against silver rather than against a row-count constant. The
-- backfill runs across days, so a hardcoded 173,520 would be red for most of
-- its life and would teach everyone to ignore it; this stays true at every
-- point in between and still catches a join that dropped rows.
select
    (select count(*) from {{ ref('stg_observations_daily') }}) as silver_rows,
    (select count(*) from {{ ref('fact_weather_observations') }}) as fact_rows
where (select count(*) from {{ ref('stg_observations_daily') }})
   <> (select count(*) from {{ ref('fact_weather_observations') }})
