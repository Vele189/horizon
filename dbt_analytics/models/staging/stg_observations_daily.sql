-- Deduplicated bronze, plus the one conversion the source's own inconsistency
-- makes necessary. Everything else is asserted rather than converted: ING-01
-- requests metric units explicitly and checks them on every response, so
-- converting here would be undoing work already verified upstream.
with deduplicated as (

    {{ deduplicate_observations('bronze', 'observations_daily') }}

)

select
    *,
    -- snowfall_sum is centimetres while precipitation_sum and rain_sum are
    -- millimetres. Published in millimetres too, so no downstream model has to
    -- remember that one column in the row is a different scale.
    {{ cm_to_mm('snowfall_sum') }} as snowfall_sum_mm

from deduplicated
