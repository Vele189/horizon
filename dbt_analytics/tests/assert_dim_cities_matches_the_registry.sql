-- The dimension is the registry, no more and no less.
--
-- Built from cities.yml rather than from observation data on purpose: a
-- dimension inferred from what has landed would list fourteen cities during a
-- backfill and would quietly lose one whose ingestion failed. This asserts the
-- two directions of that claim: no city invented, and none dropped.
select 'missing from dimension' as problem, city_id
from {{ ref('stg_cities') }}
where city_id not in (select city_id from {{ ref('dim_cities') }})

union all

select 'not in the registry' as problem, city_id
from {{ ref('dim_cities') }}
where city_id not in (select city_id from {{ ref('stg_cities') }})
