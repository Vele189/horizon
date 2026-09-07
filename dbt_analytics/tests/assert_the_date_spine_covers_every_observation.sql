-- Every observed day must have a spine row to join to.
--
-- The reverse is not asserted: a spine day with no observations is ordinary,
-- both at the leading edge and wherever a backfill has not reached.
{% set grains = ['daily', 'hourly'] %}
{% for grain in grains %}
select distinct
    '{{ grain }}' as grain,
    (observation_time at time zone 'UTC')::date as observed_day
from {{ ref('stg_observations_' ~ grain) }}
where (observation_time at time zone 'UTC')::date
      not in (select date_day from {{ ref('dim_date') }})
{% if not loop.last %}union all{% endif %}
{% endfor %}
