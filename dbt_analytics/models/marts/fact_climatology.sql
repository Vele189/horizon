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
--
-- **The detrended baseline, and why its window is a different shape (DBT-12).**
-- The exclusion above removes the labelled observation from its own baseline.
-- It does nothing about the other reason a late observation reads hot: the
-- record has warmed within it, so a baseline spanning 1995-2026 sits below the
-- climate of 2024 and "anomalously hot" drifts towards meaning "recent".
-- corr(year, Z) is positive in every city here, from +0.05 in Delhi to +0.38
-- in Lagos.
--
-- So a per-city, per-climatology_day linear trend in year is fitted and
-- subtracted before standardising, producing a second mean and sigma beside
-- the first rather than in place of it. Nothing downstream switches over: this
-- model publishes both, `fact_weather_anomalies` flags on both, and DBT-13
-- decides which definition the product ships.
--
-- **The trend is fitted on an expanding window ending at the year before the
-- one it labels, and that is the ticket rather than a refinement.** A trend
-- fitted across the whole record uses 2026 to decide what was normal in 2023.
-- Every feature in this project is backward-only by construction; a *label*
-- that could see forward would be a worse defect than the one being fixed, and
-- it would be invisible, because a forward-looking trend produces a perfectly
-- plausible slope. The window is expressed as `rows between unbounded
-- preceding and 1 preceding` over the years, so the boundary is a property of
-- the frame rather than of a predicate someone has to keep true.
--
-- The regression needs no new pass over the observations. Ordinary least
-- squares on (year, temperature) is a function of N, sum(Y), sum(Y^2),
-- sum(x) and sum(x*Y), and `int_climatology_contributions` already carries
-- the per-year counts and power sums; the year is constant within each of its
-- groups, so the cross terms are that group's sums multiplied by its year.
-- The same identity that makes the leave-one-year-out exclusion a subtraction
-- makes the trend a window function.
--
-- A rolling thirty-year WMO-style normal was considered and rejected: the
-- record starts in 1995, so a trailing thirty-year window does not exist
-- before 2025.
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
        sum(sum_sq_min)      as sum_sq_min,

        -- The year-weighted cross terms the trend is recovered from. The year
        -- is constant inside each contribution row, so summing it `observations`
        -- times is the same as multiplying, and no second pass over silver is
        -- needed. Kept numeric: `sum_year_sq` runs to ~4e9 per year while the
        -- quantity finally wanted from it is ~1e5, and Postgres numeric
        -- subtracts exactly where double precision would lose five digits.
        sum(source_year::numeric * observations)         as sum_year,
        sum(source_year::numeric ^ 2 * observations)     as sum_year_sq,
        sum(source_year::numeric * sum_mean)             as sum_year_mean
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
        totals.sum_year,
        totals.sum_year_sq,
        totals.sum_year_mean,

        -- The excluded year's own contribution. Zero when that year did not
        -- reach this day's window at all.
        coalesce(own.observations, 0) as own_observations,
        coalesce(own.sum_mean, 0)     as own_sum_mean,
        coalesce(own.sum_sq_mean, 0)  as own_sum_sq_mean,
        coalesce(own.sum_max, 0)      as own_sum_max,
        coalesce(own.sum_sq_max, 0)   as own_sum_sq_max,
        coalesce(own.sum_min, 0)      as own_sum_min,
        coalesce(own.sum_sq_min, 0)   as own_sum_sq_min,

        -- The trend window: every year strictly before the one being labelled.
        --
        -- `rows between unbounded preceding and 1 preceding` is the whole
        -- guarantee. Each row of `targets` is one (city, day, year) and carries
        -- that year's own contribution, so the frame accumulates exactly the
        -- years before it and cannot reach the year it labels or any year
        -- after. An empty frame -- the first year of a city's record -- sums to
        -- null and is coalesced to zero, which then fails the fit for want of
        -- years rather than by dividing by one.
        coalesce(sum(coalesce(own.observations, 0)) over prior_years, 0)
            as trend_observations,
        coalesce(sum(case when coalesce(own.observations, 0) > 0 then 1 else 0 end)
                 over prior_years, 0)
            as trend_reference_years,
        coalesce(sum(targets.for_year::numeric * coalesce(own.observations, 0))
                 over prior_years, 0)
            as trend_sum_year,
        coalesce(sum(targets.for_year::numeric ^ 2 * coalesce(own.observations, 0))
                 over prior_years, 0)
            as trend_sum_year_sq,
        coalesce(sum(coalesce(own.sum_mean, 0)) over prior_years, 0)
            as trend_sum_mean,
        coalesce(sum(coalesce(own.sum_sq_mean, 0)) over prior_years, 0)
            as trend_sum_sq_mean,
        coalesce(sum(targets.for_year::numeric * coalesce(own.sum_mean, 0))
                 over prior_years, 0)
            as trend_sum_year_mean

    from targets
    join totals
      on totals.city_id = targets.city_id
     and totals.target_day = targets.target_day
    left join contributions as own
      on own.city_id = targets.city_id
     and own.target_day = targets.target_day
     and own.source_year = targets.for_year

    window prior_years as (
        partition by targets.city_id, targets.target_day
        order by targets.for_year
        rows between unbounded preceding and 1 preceding
    )

),

-- The fitted trend, and how much of it is signal.
--
-- Ordinary least squares of temperature on year over the prior-years window.
-- `trend_stderr_c_per_year` is carried beside the slope because at this sample
-- size it is not decoration: a fit over K years has a standard error falling
-- as K^-1.5, and a warming signal of a few hundredths of a degree a year is
-- inside that error until K is large. Subtracting a slope that is mostly noise
-- adds variance to the baseline instead of removing trend from it, so the
-- number that says which of the two happened is published rather than assumed.
--
-- `climatology_trend_min_years` is the floor below which no trend is fitted at
-- all and the detrended baseline is defined to equal the plain one. That is
-- not a fallback, it is the honest content of an expanding window: in 1997
-- there is no trend to know yet, and asserting one would be the same mistake
-- as looking forward, made in the opposite direction.
--
-- Fifteen, and the number was swept rather than picked. Against the drift the
-- detrended flag exists to reduce -- the label's base rate rising 2.29x from
-- the training period to the test one -- the thresholds give:
--
--     min_years   rows with a trend   fits over 2 s.e.   drift, detrended
--             5               83.9%              48.2%              2.46x
--            10               68.1%              50.6%              2.29x
--            15               52.3%              52.6%              2.22x
--            20               36.4%              56.3%              2.21x
--
-- At five the detrended flag drifts *more* than the plain one. That is the
-- whole argument in one row: a slope fitted on five years is mostly noise, and
-- subtracting noise from a baseline adds exceedances rather than removing
-- them. Fifteen is the smallest floor at which that has stopped happening and
-- at which more than half the fitted slopes are distinguishable from zero;
-- twenty buys another hundredth and costs a third of the coverage.
trended as (

    select
        *,
        {% set min_years = var('climatology_trend_min_years', 15) %}
        (case
            when trend_reference_years >= {{ min_years }}
             and (trend_observations * trend_sum_year_sq
                  - trend_sum_year ^ 2) > 0
            then (trend_observations * trend_sum_year_mean
                  - trend_sum_year * trend_sum_mean)
                 / (trend_observations * trend_sum_year_sq
                    - trend_sum_year ^ 2)
         end) as trend_slope
    from excluded

),

fitted as (

    select
        *,
        -- Residual standard error of the slope, from the same power sums:
        -- SE(b) = sqrt( (Syy - b*Sxy) / (N - 2) / Sxx ), with Sxx, Sxy and Syy
        -- the centred sums of squares. greatest(..., 0) because the residual
        -- sum of squares can land a hair below zero through cancellation when
        -- the fit is near-exact.
        (case
            when trend_slope is not null and trend_observations > 2
            then sqrt(greatest(
                     (
                         (trend_sum_sq_mean - trend_sum_mean ^ 2 / trend_observations)
                         - trend_slope
                           * (trend_sum_year_mean
                              - trend_sum_year * trend_sum_mean / trend_observations)
                     ) / (trend_observations - 2)
                     / nullif(trend_sum_year_sq
                              - trend_sum_year ^ 2 / trend_observations, 0),
                     0
                 ))
         end) as trend_stderr
    from trended

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
        (sum_year      - for_year::numeric * own_observations)
            as baseline_sum_year,
        (sum_year_sq   - for_year::numeric ^ 2 * own_observations)
            as baseline_sum_year_sq,
        (sum_year_mean - for_year::numeric * own_sum_mean)
            as baseline_sum_year_mean,
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
        sum_year         as baseline_sum_year,
        sum_year_sq      as baseline_sum_year_sq,
        sum_year_mean    as baseline_sum_year_mean,
        false as excludes_own_year
        {% endif %}
    from fitted

),

-- The baseline with the trend taken out of it, referenced to the year being
-- labelled.
--
-- Detrending maps an observation from year Y to x' = x - b*(Y - t), where t is
-- `for_year`. Referencing to t rather than to the window's mean year is what
-- keeps the observation itself untouched: its own Y is t, so x'_t = x_t, and
-- `fact_weather_anomalies` scores the same temperature it always did against a
-- baseline that has been walked forward to meet it. Written in centred form --
-- sums of (Y - t) rather than of Y -- so the quantities stay small and the
-- subtraction is not a difference of billions.
detrended as (

    select
        *,
        (baseline_sum_year - baseline_observations * for_year::numeric)
            as baseline_sum_offset,
        (baseline_sum_year_sq
         - 2 * for_year::numeric * baseline_sum_year
         + baseline_observations * for_year::numeric ^ 2)
            as baseline_sum_offset_sq,
        (baseline_sum_year_mean - for_year::numeric * baseline_sum_mean)
            as baseline_sum_offset_mean
    from applied

),

shifted as (

    select
        *,
        -- Sum(x') and Sum(x'^2) over the baseline window.
        --
        -- The unfitted case is written as a branch rather than as a zero
        -- slope, and the difference is not cosmetic. Subtracting `0 * offset`
        -- is arithmetically nothing but it is not *nothing*: Postgres numeric
        -- carries a scale, the product widens it, and the wider operand then
        -- divides and squares to a different scale inside
        -- `climatology_stddev`, landing a few bits away in double precision.
        -- The claim this model makes is that a year with no fitted trend has a
        -- detrended baseline identical to its plain one, and a data test
        -- compares them with `is distinct from`. Branching makes the claim
        -- exactly true instead of true to twelve decimal places.
        (case when trend_slope is null then baseline_sum_mean
              else baseline_sum_mean - trend_slope * baseline_sum_offset
         end) as detrended_sum_mean,
        (case when trend_slope is null then baseline_sum_sq_mean
              else baseline_sum_sq_mean
                   - 2 * trend_slope * baseline_sum_offset_mean
                   + trend_slope ^ 2 * baseline_sum_offset_sq
         end) as detrended_sum_sq_mean
    from detrended

)

select
    shifted.city_id,
    calendar.month_day,
    shifted.target_day as climatology_day,
    shifted.for_year,
    shifted.excludes_own_year,

    shifted.baseline_observations as observations,
    shifted.reference_years,

    -- How much the exclusion actually removed. Zero is legitimate and common
    -- at the edges of the record: a year that contributed nothing to this
    -- day's window has nothing to exclude. 2026 reaches only to the archive
    -- edge, so it is absent from every window centred after early September.
    -- Carried so that "the exclusion did nothing here" is distinguishable from
    -- "the exclusion is broken", which is otherwise the same number.
    shifted.own_observations as excluded_observations,

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
    shifted.all_observations as observations_all_years,
    {{ climatology_mean('sum_mean', 'all_observations') }}
        as mean_temperature_2m_mean_all_years,
    {{ climatology_stddev('sum_mean', 'sum_sq_mean', 'all_observations') }}
        as stddev_temperature_2m_mean_all_years,

    -- DBT-12. The detrended baseline, published beside the plain one rather
    -- than in place of it.
    shifted.trend_reference_years,
    shifted.trend_observations,
    shifted.trend_slope::double precision  as trend_slope_c_per_year,
    shifted.trend_stderr::double precision as trend_stderr_c_per_year,
    {{ climatology_mean('detrended_sum_mean', 'baseline_observations') }}
        as mean_temperature_2m_mean_detrended,
    {{ climatology_stddev('detrended_sum_mean', 'detrended_sum_sq_mean',
                          'baseline_observations') }}
        as stddev_temperature_2m_mean_detrended

from shifted
join (select distinct climatology_day, month_day from {{ ref('dim_date') }}) as calendar
  on calendar.climatology_day = shifted.target_day
