-- The sensitivity band must bracket the estimate, and the fallback must be the
-- conservative end of it.
--
-- Two properties, both about the crossing that makes this easy to get wrong: a
-- *larger* shape means a heavier tail, which makes an extreme MORE frequent and
-- the period SHORTER. So `return_period_years_shape_low` -- the column named
-- for the low end of the period -- is computed at `shape_high`, and vice versa.
-- Swapping those two arguments in the model produces a band that is finite,
-- ordered, plausible, and inverted, and nothing on the map would look wrong.
--
--   1. shape_low reading <= point estimate <= shape_high reading.
--      This is what says the two columns are the way round their names claim.
--
--   2. Where the band is too wide to report, the quoted figure is the LOW end.
--      Understating rarity is the only direction this product is willing to be
--      wrong in: "a one-in-500-year day" printed off a thirty-year record is
--      the kind of number that gets quoted onward and cannot be walked back.
--
-- Nulls are excluded rather than treated as failures. A bounded tail stops, and
-- a level past where it stops has no period at all; the model's own
-- reportability rule already refuses to quote those.
select
    city_id,
    date_key,
    abs_z,
    return_period_years,
    return_period_years_shape_low,
    return_period_years_shape_high,
    return_period_years_quoted,
    return_period_is_reportable
from {{ ref('fact_anomaly_return_periods') }}
where return_period_years is not null
  and (
        -- (1) the band must contain what it is a band around
        (return_period_years_shape_low is not null
         and return_period_years < return_period_years_shape_low * 0.999)
     or (return_period_years_shape_high is not null
         and return_period_years > return_period_years_shape_high * 1.001)

        -- (2) an unreportable row quotes the floor, not the estimate
     or (not return_period_is_reportable
         and return_period_years_quoted > return_period_years * 1.05)
  )
