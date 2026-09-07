-- Deduplication must keep the *newest* copy, not an arbitrary one.
--
-- The unique test proves one row survives per key; it says nothing about
-- which. This one does: every surviving row must carry the greatest
-- ingested_at bronze holds for its key. A row_number() ordered ascending by
-- mistake would pass the unique test and silently serve the oldest data.
with newest as (
    select city_id, observation_time, max(ingested_at) as newest_ingest
    from {{ source('bronze', 'observations_daily') }}
    group by city_id, observation_time
)
select
    staged.city_id,
    staged.observation_time,
    staged.ingested_at,
    newest.newest_ingest
from {{ ref('stg_observations_daily') }} as staged
join newest
  on newest.city_id = staged.city_id
 and newest.observation_time = staged.observation_time
where staged.ingested_at <> newest.newest_ingest
