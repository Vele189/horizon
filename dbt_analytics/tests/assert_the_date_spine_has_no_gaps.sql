-- Every day between the first and the last, with none missing and none twice.
--
-- A spine with a hole is worse than no spine: a left join against it drops the
-- observations for the missing days without a word, and the resulting series
-- looks complete because the dimension said so.
with ordered as (

    select
        date_day,
        lag(date_day) over (order by date_day) as previous_day
    from {{ ref('dim_date') }}

)

select date_day, previous_day, date_day - previous_day as gap_days
from ordered
where previous_day is not null
  and date_day - previous_day <> 1
