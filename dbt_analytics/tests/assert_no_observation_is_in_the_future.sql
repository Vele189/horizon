-- A reanalysis archive cannot report weather that has not happened.
--
-- A future timestamp means a timezone was applied twice, a date was parsed
-- with the wrong century, or a test fixture leaked into a real table. All
-- three are quiet: the row looks ordinary, and only the ordering gives it away.
--
-- The margin is a day, not zero: the archive edge moves and a run started
-- before midnight UTC can legitimately land a row stamped for a day this
-- query would otherwise call tomorrow.
{% set grains = ['daily', 'hourly'] %}
{% for grain in grains %}
select
    '{{ grain }}' as grain,
    city_id,
    observation_time,
    ingested_at
from {{ ref('stg_observations_' ~ grain) }}
where observation_time > now() + interval '1 day'
   or ingested_at > now() + interval '1 day'
{% if not loop.last %}union all{% endif %}
{% endfor %}
