-- month_day must mean the same day in every year, and day_of_year must not.
--
-- The first half is what makes month_day a safe climatology key. The second is
-- the reason it is needed: raw day_of_year takes two values for 1 March, so a
-- climatology grouped by it mixes 1 March with 29 February in leap years and
-- shifts everything after February by a day in the other three.
select 'month_day drifts' as problem, month_day, count(distinct day_of_year_common) as values
from {{ ref('dim_date') }}
group by month_day
having count(distinct day_of_year_common) > 1

union all

select 'day_of_year is stable after all', '03-01', count(distinct day_of_year)
from {{ ref('dim_date') }}
where month_day = '03-01'
having count(distinct day_of_year) < 2
