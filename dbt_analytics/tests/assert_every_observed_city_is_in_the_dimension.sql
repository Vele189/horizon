-- Every city with facts must have a dimension row to join to.
--
-- The reverse does not hold and must not be asserted: a city in the dimension
-- with no observations yet is the normal state of a multi-day backfill, not an
-- error. The reconciliation report is what tracks that direction.
{% set grains = ['daily', 'hourly'] %}
{% for grain in grains %}
select distinct '{{ grain }}' as grain, city_id
from {{ ref('stg_observations_' ~ grain) }}
where city_id not in (select city_id from {{ ref('dim_cities') }})
{% if not loop.last %}union all{% endif %}
{% endfor %}
