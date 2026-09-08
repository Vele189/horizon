-- Where no trend could be fitted, the detrended baseline must equal the plain
-- one, exactly.
--
-- The first years of every city's record have no expanding window to fit on,
-- so `is_anomaly_detrended` is defined there as `is_anomaly` rather than as
-- null. That choice is what makes the two flags comparable across a split
-- instead of comparing a full population against a truncated one, and it is
-- only honest if "no trend" really does mean "unchanged". A coalesce written
-- the other way round, or a slope silently defaulting to something other than
-- zero, would move the baseline of the earliest years by an amount nobody
-- asked for and nobody would see: the columns would still be populated and
-- still be plausible.
--
-- Compared exactly rather than within a tolerance. Both sides come from the
-- same power sums through the same macro, and a zero slope leaves the sums
-- arithmetically identical, so any difference at all is a difference in kind.
select
    city_id,
    climatology_day,
    for_year,
    trend_reference_years,
    mean_temperature_2m_mean,
    mean_temperature_2m_mean_detrended,
    stddev_temperature_2m_mean,
    stddev_temperature_2m_mean_detrended
from {{ ref('fact_climatology') }}
where trend_slope_c_per_year is null
  and (mean_temperature_2m_mean   is distinct from mean_temperature_2m_mean_detrended
    or stddev_temperature_2m_mean is distinct from stddev_temperature_2m_mean_detrended)
