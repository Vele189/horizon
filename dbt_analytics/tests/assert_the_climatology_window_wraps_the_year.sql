-- 1 January's window must reach back into December.
--
-- A non-circular window would build the year's first and last weeks from half
-- the observations of every other week, precisely where the northern winter
-- extremes sit, so the sigma there would be estimated from the thinnest data
-- and the most extreme days would be scored against the least reliable
-- baseline.
--
-- Asserted by size: the windows at the year boundary must be as full as one
-- in the middle of the year.
with boundary as (
    select city_id, for_year, month_day, observations
    from {{ ref('fact_climatology') }}
    where month_day in ('01-01', '12-31')
),
midyear as (
    select city_id, for_year, observations
    from {{ ref('fact_climatology') }} where month_day = '07-01'
)
select
    boundary.city_id,
    boundary.month_day,
    boundary.for_year,
    boundary.observations as boundary_window,
    midyear.observations as midyear_window
from boundary
join midyear
  on midyear.city_id = boundary.city_id and midyear.for_year = boundary.for_year
where midyear.observations > 0
  and boundary.observations < midyear.observations * 0.8
