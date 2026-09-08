-- The trend for year t must be fitted on years strictly before t.
--
-- This is DBT-12's whole point and the one defect in it that would be
-- invisible. A trend fitted across the entire record uses 2026 to decide what
-- was normal in 2023; the slope it produces is perfectly plausible, the model
-- builds, every range test passes, and the label has quietly learned the
-- future. Nothing about the number itself would reveal it.
--
-- So the frame is checked by counting rather than by re-deriving the
-- regression. `trend_reference_years` is what the window function accumulated;
-- the subquery counts the years that *should* have been in the frame, straight
-- from the contributions the model was built from. If the frame reached its
-- own year, or one after it, the two disagree. Counting rather than
-- recomputing the slope is deliberate: a test that re-implements the algebra
-- it is checking passes whenever both copies are wrong in the same way, and
-- the boundary is the property at issue, not the arithmetic.
--
-- The arithmetic has its own check. `tests/test_dbt_climatology.py` compares
-- the slope against Postgres's own `regr_slope` over exactly the years before
-- each target year, and compiles the model twice to assert that dropping the
-- last year of the record leaves every earlier year's trend untouched.
with expected as (

    select
        climatology.city_id,
        climatology.climatology_day,
        climatology.for_year,
        climatology.trend_reference_years,
        climatology.trend_observations,
        count(prior.source_year)             as expected_years,
        coalesce(sum(prior.observations), 0) as expected_observations
    from {{ ref('fact_climatology') }} as climatology
    left join {{ ref('int_climatology_contributions') }} as prior
      on prior.city_id     = climatology.city_id
     and prior.target_day  = climatology.climatology_day
     and prior.source_year < climatology.for_year
    group by 1, 2, 3, 4, 5

)

select *
from expected
where trend_reference_years <> expected_years
   or trend_observations    <> expected_observations
