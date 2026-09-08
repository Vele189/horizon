-- The requirement the whole model is shaped around.
--
-- Two claims, because the obvious single assertion is wrong. "The leave-one-out
-- count is always smaller than the all-years count" fails legitimately at the
-- edges of the record: 2026 reaches only to the archive edge, so it contributes
-- nothing to windows centred after early September and there is nothing there
-- to exclude.
--
-- So: the arithmetic must hold exactly everywhere, and the exclusion must
-- actually remove something wherever the labelled year did contribute. The
-- second is what catches an exclusion that silently did nothing, the failure
-- that makes ML-05's metrics flattering and wrong.
{% if var('climatology_exclude_own_year', true) %}

select 'arithmetic does not balance' as problem,
       city_id, month_day, for_year, observations, observations_all_years
from {{ ref('fact_climatology') }}
where observations <> observations_all_years - excluded_observations

union all

select 'year contributed but was not excluded' as problem,
       city_id, month_day, for_year, observations, observations_all_years
from {{ ref('fact_climatology') }}
where excluded_observations > 0
  and observations >= observations_all_years

{% else %}

-- The leaky variant was built on purpose. Assert it really is leaky, so a
-- misread var cannot quietly produce the safe one under the wrong label.
select 'exclusion applied despite the flag' as problem,
       city_id, month_day, for_year, observations, observations_all_years
from {{ ref('fact_climatology') }}
where observations <> observations_all_years

{% endif %}
