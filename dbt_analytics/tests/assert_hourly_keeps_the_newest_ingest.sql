-- The daily test's counterpart. See its comment for why this is separate from
-- the uniqueness check.
with newest as (
    select city_id, observation_time, max(ingested_at) as newest_ingest
    from {{ source('bronze', 'observations_hourly') }}
    group by city_id, observation_time
)
select
    staged.city_id,
    staged.observation_time,
    staged.ingested_at,
    newest.newest_ingest
from {{ ref('stg_observations_hourly') }} as staged
join newest
  on newest.city_id = staged.city_id
 and newest.observation_time = staged.observation_time
where staged.ingested_at <> newest.newest_ingest
