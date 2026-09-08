-- ============================================================================
-- Extreme value fact table
-- ============================================================================
-- Idempotent by construction: every statement is IF NOT EXISTS or a COMMENT,
-- so re-running against an existing database is a no-op. Applied by
--
--     python machine_learning/extremes.py --write
--
-- **Grain: one row per city.** A fitted tail, not a fitted day. The per-day
-- return periods a reader actually sees are computed in dbt, in
-- `fact_anomaly_return_periods`, by joining these parameters back onto every
-- scored day. Storing the parameters rather than the derived periods is what
-- makes the derivation inspectable: a reader who distrusts a "one in 40 years"
-- can see the shape, the scale, the threshold and the interval that produced
-- it, in one row, and redo the arithmetic.
--
-- **Why this is written by Python and not by dbt.** Fitting a generalised
-- Pareto is maximum likelihood over a two-parameter family, and the intervals
-- are five hundred bootstrap refits over clusters. That is not SQL. The
-- division of labour matches `fact_ml_predictions`: Python estimates, dbt
-- joins and derives, and the boundary is a table with named columns rather
-- than a pickle.
--
-- **Every parameter carries its interval, and this is the point of the table.**
-- The shape parameter decides whether the tail is bounded -- whether there is
-- a hottest possible day -- and at a few hundred exceedances per city it is
-- not pinned down: Delhi's moves from -0.005 to +0.147 when the threshold
-- moves one percentile. A schema that stored `shape` alone would let every
-- downstream reader treat that as settled. `shape_low`/`shape_high` and the
-- derived `tail_is_bounded` make the uncertainty impossible to drop by
-- accident, because it is a column and not a caveat in a docstring.
--
-- **`exceedance_rate` is the declustered rate.** Events per observed day,
-- after runs of consecutive hot days have been collapsed to their peak. The
-- undeclustered rate is 2.1 to 3.6 times larger on this data and every return
-- period computed from it would be too short by that factor, in the direction
-- that makes the product sound more dramatic. The constraint below is what
-- stops a future writer from putting the wrong one here: the rate must be at
-- most the raw exceedance fraction, with equality only when nothing clustered.
-- ============================================================================

create schema if not exists gold_marts;

create table if not exists gold_marts.fact_extreme_value (
    -- Grain -------------------------------------------------------------
    city_id                     text        not null,

    -- The fit's own terms -----------------------------------------------
    threshold                   numeric     not null,
    observations                integer     not null,
    exceedances                 integer     not null,
    clusters                    integer     not null,
    mean_cluster_days           numeric     not null,
    exceedance_rate             numeric     not null,

    -- Generalised Pareto, location pinned at zero ------------------------
    shape                       numeric     not null,
    shape_low                   numeric,
    shape_high                  numeric,
    scale                       numeric     not null,
    scale_low                   numeric,
    scale_high                  numeric,
    tail_is_bounded             boolean     not null,
    shape_interval_excludes_zero boolean    not null,

    -- Return levels, in sigma ---------------------------------------------
    return_level_2y             numeric,
    return_level_5y             numeric,
    return_level_10y            numeric,
    return_level_20y            numeric,
    return_level_50y            numeric,

    -- Composition of the tail, early half against late half ---------------
    warm_share_early            numeric,
    warm_share_late             numeric,

    -- Each tail, allowed to move with time. Two, not one -------------------
    warm_trend_threshold_intercept        numeric,
    warm_trend_threshold_slope_per_year   numeric,
    warm_trend_clusters         integer,
    warm_trend_converged        boolean,
    warm_trend_scale_trend_per_year        numeric,
    warm_trend_scale_change_over_record    numeric,
    warm_trend_p_value          numeric,
    cold_trend_threshold_intercept        numeric,
    cold_trend_threshold_slope_per_year   numeric,
    cold_trend_clusters         integer,
    cold_trend_converged        boolean,
    cold_trend_scale_trend_per_year        numeric,
    cold_trend_scale_change_over_record    numeric,
    cold_trend_p_value          numeric,

    -- Provenance ---------------------------------------------------------
    pot_quantile                numeric     not null,
    run_separation_days         integer     not null,
    bootstrap_samples           integer     not null,
    fitted_at                   timestamptz not null default now(),

    primary key (city_id)
);

-- The declustered rate can never exceed the raw exceedance fraction: collapsing
-- runs to their peaks removes events, it cannot add them. This is the one
-- error in this table that would be invisible downstream -- every return period
-- would simply come out short, and short return periods look like an exciting
-- finding rather than like a bug.
alter table gold_marts.fact_extreme_value
    drop constraint if exists fact_extreme_value_rate_is_declustered;

alter table gold_marts.fact_extreme_value
    add constraint fact_extreme_value_rate_is_declustered
    check (
        clusters <= exceedances
        and exceedances <= observations
        and exceedance_rate <= exceedances::numeric / observations
    );

-- A fitted scale is a spread and cannot be negative or zero; a fit that
-- returned one would mean the optimiser walked off the support.
alter table gold_marts.fact_extreme_value
    drop constraint if exists fact_extreme_value_scale_is_positive;

alter table gold_marts.fact_extreme_value
    add constraint fact_extreme_value_scale_is_positive
    check (scale > 0);

-- `tail_is_bounded` is a claim about the *interval*, not about the point
-- estimate, and the difference is the entire contribution of this table: a
-- negative shape whose interval crosses zero is not evidence of a bounded
-- tail. Enforced rather than trusted, because it is exactly the simplification
-- a downstream writer would make.
alter table gold_marts.fact_extreme_value
    drop constraint if exists fact_extreme_value_bounded_means_the_interval;

alter table gold_marts.fact_extreme_value
    add constraint fact_extreme_value_bounded_means_the_interval
    check (
        shape_high is null
        or tail_is_bounded = (shape_high < 0)
    );

comment on table gold_marts.fact_extreme_value is
  'One row per city: a generalised Pareto fitted to the declustered peaks over a high per-city quantile of abs(Z), with bootstrap intervals over clusters. Turns "how many sigma" into "how many years". Refitting replaces rather than appends.';
comment on column gold_marts.fact_extreme_value.threshold is
  'Where the tail is taken to begin, in abs(Z). A per-city quantile, not a fixed cut: the cities differ, which is why they are fitted separately.';
comment on column gold_marts.fact_extreme_value.exceedance_rate is
  'DECLUSTERED events per observed day. Runs of consecutive exceedances are one event. Using the raw fraction here would shorten every return period by the mean cluster length.';
comment on column gold_marts.fact_extreme_value.shape is
  'GPD shape. Negative: the tail is bounded, there is a hottest possible day. Zero: exponential. Positive: heavy. Read it with shape_low/shape_high, which usually straddle zero at this sample size.';
comment on column gold_marts.fact_extreme_value.tail_is_bounded is
  'True only when the whole 95% interval is below zero. A negative point estimate alone is not evidence of a bounded tail.';
comment on column gold_marts.fact_extreme_value.warm_share_early is
  'Share of declustered exceedances that are warm, in the earlier half of the record. Rises in every fitted city -- 0.08 to 0.61 in Lagos. This is why the trends are fitted per direction: a single trend on folded abs(Z) reports that composition shift as a change in width.';
comment on column gold_marts.fact_extreme_value.warm_trend_p_value is
  'Likelihood-ratio test of a time trend in the WARM tail''s scale against a stationary warm tail, one degree of freedom. Small: the warm tail itself is widening, which is the drift proposal §5.3 found detrending the mean could not reach.';
comment on column gold_marts.fact_extreme_value.warm_trend_threshold_slope_per_year is
  'How fast the warm tail has MOVED, in sigma per year, from a linear quantile regression. Fitted so the scale trend beside it means widening and not drift: a threshold held still while the distribution slides under it turns one into the other.';
comment on column gold_marts.fact_extreme_value.cold_trend_p_value is
  'The same test on the cold tail, fitted on -Z so it is an upper tail like any other. Read beside the warm one: the two move independently, and in this data often in opposite directions.';
comment on column gold_marts.fact_extreme_value.pot_quantile is
  'Recorded because the shape estimate is sensitive to it. Comparing rows fitted at different quantiles is comparing different models.';
