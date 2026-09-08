{{
    config(
        materialized = 'table',
        indexes = [
            {'columns': ['city_id', 'month_day', 'for_year'], 'unique': True},
            {'columns': ['city_id', 'for_year']},
        ],
    )
}}

-- The day-of-year climatology, one baseline per city per calendar day per
-- year being labelled.
--
-- **The baseline for an observation excludes that observation's own year.**
-- This is the requirement the whole model is shaped around, and it is not a
-- refinement. A normal computed over all years includes the very day it is
-- about to label: the observation contributes to its own μ and σ, which pulls
-- the mean towards it and inflates σ by its own deviation. The Z-score that
-- results is systematically too small, extremes look less extreme, and a model
-- trained on those labels in ML-05 is scoring against a target that has
-- already seen its own answer. The metrics come out flattering and the model
-- is worse than they say.
--
-- With ~30 reference years the leakage is roughly 1/30 of the signal: small
-- enough to be invisible in a spot check and large enough to matter in a
-- ranking. It is left as `climatology_exclude_own_year`, defaulting to true,
-- so the leaky variant can be built deliberately for comparison rather than
-- reached by accident.
--
-- The all-years figures are carried alongside for exactly that comparison, and
-- so the published "30-year normal" a dashboard shows is available without a
-- second model.
with contributions as (

    select * from {{ ref('int_climatology_contributions') }}

),

totals as (

    select
        city_id,
        target_day,
        sum(observations)    as observations,
        count(*)             as reference_years,
        sum(sum_mean)        as sum_mean,
        sum(sum_sq_mean)     as sum_sq_mean,
        sum(sum_max)         as sum_max,
        sum(sum_sq_max)      as sum_sq_max,
        sum(sum_min)         as sum_min,
        sum(sum_sq_min)      as sum_sq_min
    from contributions
    group by city_id, target_day

),

-- Every (city, day, year) that could need a baseline, whether or not that
-- year contributed to the window: a year with no observations still needs the
-- normal to label against.
targets as (

    select distinct
        totals.city_id,
        totals.target_day,
        years.source_year as for_year
    from totals
    join (select distinct city_id, source_year from contributions) as years
      on years.city_id = totals.city_id

),

excluded as (

    select
        targets.city_id,
        targets.target_day,
        targets.for_year,

        totals.observations    as all_observations,
        totals.reference_years,
        totals.sum_mean,
        totals.sum_sq_mean,
        totals.sum_max,
        totals.sum_sq_max,
        totals.sum_min,
        totals.sum_sq_min,

        -- The excluded year's own contribution. Zero when that year did not
        -- reach this day's window at all.
        coalesce(own.observations, 0) as own_observations,
        coalesce(own.sum_mean, 0)     as own_sum_mean,
        coalesce(own.sum_sq_mean, 0)  as own_sum_sq_mean,
        coalesce(own.sum_max, 0)      as own_sum_max,
        coalesce(own.sum_sq_max, 0)   as own_sum_sq_max,
        coalesce(own.sum_min, 0)      as own_sum_min,
        coalesce(own.sum_sq_min, 0)   as own_sum_sq_min

    from targets
    join totals
      on totals.city_id = targets.city_id
     and totals.target_day = targets.target_day
    left join contributions as own
      on own.city_id = targets.city_id
     and own.target_day = targets.target_day
     and own.source_year = targets.for_year

),

applied as (

    select
        *,
        {% set leave_out = var('climatology_exclude_own_year', true) %}
        {% if leave_out %}
        -- Leave-one-year-out: the window minus the year being labelled.
        (all_observations - own_observations) as baseline_observations,
        (sum_mean    - own_sum_mean)          as baseline_sum_mean,
        (sum_sq_mean - own_sum_sq_mean)       as baseline_sum_sq_mean,
        (sum_max     - own_sum_max)           as baseline_sum_max,
        (sum_sq_max  - own_sum_sq_max)        as baseline_sum_sq_max,
        (sum_min     - own_sum_min)           as baseline_sum_min,
        (sum_sq_min  - own_sum_sq_min)        as baseline_sum_sq_min,
        true as excludes_own_year
        {% else %}
        -- Deliberately leaky, for comparison only.
        all_observations as baseline_observations,
        sum_mean         as baseline_sum_mean,
        sum_sq_mean      as baseline_sum_sq_mean,
        sum_max          as baseline_sum_max,
        sum_sq_max       as baseline_sum_sq_max,
        sum_min          as baseline_sum_min,
        sum_sq_min       as baseline_sum_sq_min,
        false as excludes_own_year
        {% endif %}
    from excluded

)

select
    applied.city_id,
    calendar.month_day,
    applied.target_day as climatology_day,
    applied.for_year,
    applied.excludes_own_year,

    applied.baseline_observations as observations,
    applied.reference_years,

    -- How much the exclusion actually removed. Zero is legitimate and common
    -- at the edges of the record: a year that contributed nothing to this
    -- day's window has nothing to exclude. 2026 reaches only to the archive
    -- edge, so it is absent from every window centred after early September.
    -- Carried so that "the exclusion did nothing here" is distinguishable from
    -- "the exclusion is broken", which is otherwise the same number.
    applied.own_observations as excluded_observations,

    -- The baseline that labels an observation from `for_year`.
    {{ climatology_mean('baseline_sum_mean', 'baseline_observations') }}
        as mean_temperature_2m_mean,
    {{ climatology_stddev('baseline_sum_mean', 'baseline_sum_sq_mean',
                          'baseline_observations') }}
        as stddev_temperature_2m_mean,
    {{ climatology_mean('baseline_sum_max', 'baseline_observations') }}
        as mean_temperature_2m_max,
    {{ climatology_stddev('baseline_sum_max', 'baseline_sum_sq_max',
                          'baseline_observations') }}
        as stddev_temperature_2m_max,
    {{ climatology_mean('baseline_sum_min', 'baseline_observations') }}
        as mean_temperature_2m_min,
    {{ climatology_stddev('baseline_sum_min', 'baseline_sum_sq_min',
                          'baseline_observations') }}
        as stddev_temperature_2m_min,

    -- The published normal, over every reference year including this one.
    -- Carried so a dashboard can show "the 30-year normal" without a second
    -- model, and so the cost of the leakage is measurable rather than
    -- asserted.
    applied.all_observations as observations_all_years,
    {{ climatology_mean('sum_mean', 'all_observations') }}
        as mean_temperature_2m_mean_all_years,
    {{ climatology_stddev('sum_mean', 'sum_sq_mean', 'all_observations') }}
        as stddev_temperature_2m_mean_all_years

from applied
join (select distinct climatology_day, month_day from {{ ref('dim_date') }}) as calendar
  on calendar.climatology_day = applied.target_day
