-- 29 February is a row of its own, and its window is not short.
--
-- Under raw day_of_year it would collide with 1 March, so it is keyed on
-- month_day instead. Its *own* sample is a quarter the size of other days —
-- eight leap years in thirty-two — but the +/-7 day window around it is drawn
-- from 22 February to 7 March in every year, leap or not, so the baseline
-- behind it is as strong as any other day's.
--
-- This asserts that: the leap day exists for every city that has one, and its
-- window holds at least eighty per cent of what the day before it holds. A
-- naive implementation that filtered the window to leap years would fail here
-- with roughly a quarter.
with leap_day as (
    select city_id, for_year, observations
    from {{ ref('fact_climatology') }} where month_day = '02-29'
),
day_before as (
    select city_id, for_year, observations
    from {{ ref('fact_climatology') }} where month_day = '02-28'
)
select
    leap_day.city_id,
    leap_day.for_year,
    leap_day.observations as leap_day_window,
    day_before.observations as prior_day_window
from leap_day
join day_before
  on day_before.city_id = leap_day.city_id
 and day_before.for_year = leap_day.for_year
where day_before.observations > 0
  and leap_day.observations < day_before.observations * 0.8
