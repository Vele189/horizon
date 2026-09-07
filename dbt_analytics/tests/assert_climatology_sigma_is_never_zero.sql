-- A zero sigma would make every Z-score infinite or undefined.
--
-- It requires ~450 observations in a fortnight-wide window to be bit-identical,
-- which for a real temperature series cannot happen — so a zero here means the
-- window collapsed, not that the weather was constant.
--
-- Null is different and is allowed: it means the leave-one-year-out exclusion
-- left nothing to compute from, which is the honest answer for a city with a
-- single reference year. Asserting non-null instead would force a silent
-- fallback to the leaky baseline, which is the failure this model exists to
-- prevent.
select
    city_id,
    month_day,
    for_year,
    observations,
    stddev_temperature_2m_mean
from {{ ref('fact_climatology') }}
where stddev_temperature_2m_mean = 0
