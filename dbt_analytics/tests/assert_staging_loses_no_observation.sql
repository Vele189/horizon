-- Deduplication must remove duplicates and nothing else.
--
-- A partition key typo, partitioning by city_id alone say, would collapse
-- an entire city to one row and still pass every uniqueness test, because the
-- result would indeed be unique. This counts distinct keys on both sides and
-- fails if silver holds fewer observations than bronze knows about.
{% set grains = ['daily', 'hourly'] %}

{% for grain in grains %}
select
    '{{ grain }}' as grain,
    bronze_keys.n as bronze_distinct,
    staged.n as staged_rows
from
    (select count(*) as n from (
        select distinct city_id, observation_time
        from {{ source('bronze', 'observations_' ~ grain) }}
    ) d) as bronze_keys,
    (select count(*) as n from {{ ref('stg_observations_' ~ grain) }}) as staged
where bronze_keys.n <> staged.n
{% if not loop.last %}union all{% endif %}
{% endfor %}
