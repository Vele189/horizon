-- The ±N-day smoothing window, as an explicit day-to-day mapping.
--
-- A single calendar day's normal is built from ~30 observations, one per
-- reference year, and at that sample size the standard deviation is noise: a
-- Z-score against it says more about which thirty days happened to be sampled
-- than about the weather. Widening to ±7 days gives 15 calendar days × ~30
-- years ≈ 450 observations, which is a stable estimate of a distribution that
-- genuinely does vary slowly across a fortnight.
--
-- The window is **circular**. 1 January's neighbourhood includes 25 December
-- through 8 January, and a non-circular window would silently build the
-- year's first and last week from half as much data as every other week —
-- exactly where the northern winter extremes are.
--
-- Measured on `climatology_day`, the day-of-year a date would have in a leap
-- year, because raw day_of_year gives 31 December two different numbers.
with days as (

    select generate_series(1, 366) as climatology_day

),

offsets as (

    select generate_series(
        -{{ var('climatology_smoothing_days', 7) }},
         {{ var('climatology_smoothing_days', 7) }}
    ) as day_offset

)

select
    days.climatology_day as target_day,
    -- Wrap into 1..366. The double modulo is because Postgres's % keeps the
    -- sign of the dividend, so -6 % 366 is -6 rather than 360.
    ((days.climatology_day - 1 + offsets.day_offset) % 366 + 366) % 366 + 1
        as source_day,
    offsets.day_offset
from days
cross join offsets
