-- ============================================================================
-- Prediction fact table
-- ============================================================================
-- Idempotent by construction: every statement is IF NOT EXISTS or a COMMENT,
-- so re-running against an existing database is a no-op. Applied by
--
--     python machine_learning/predict.py
--
-- **Grain: one row per city per forecast_date per horizon_day.** Not per model
-- version, and that is a decision rather than an omission. The Risk Horizon view asks "what
-- is the risk for this city right now" and must get exactly one answer; with
-- model_version in the key it would get one row per model ever run and would
-- have to pick between them in the presentation layer, which is where that
-- choice is least visible. A prediction here is the *current best* answer for
-- a city-day, and re-scoring replaces it. If a history of predictions is ever
-- wanted, it belongs in a separate append-only table with its own grain, not
-- in the one a dashboard reads.
--
-- **`horizon_day` is what makes a row self-describing (ML-13).** The model now
-- produces a per-day hazard as well as a weekly probability, and the two are
-- different quantities that would otherwise share a column. Rather than a
-- second table or a convention a reader has to know, every row carries the day
-- it is about:
--
--     horizon_day = 0   the whole window, horizon_start .. horizon_end
--     horizon_day = k   the single day forecast_date + k
--
-- Zero is not a sentinel to be looked up. `horizon_start` and `horizon_end` on
-- the same row already say what it covers, and the check constraints below
-- enforce that they agree with `horizon_day`: a day row spans one day, and the
-- window row spans `horizon_days`. `risk_score` therefore has one meaning
-- everywhere -- the probability of an anomaly in the span this row covers --
-- rather than two meanings distinguished by which table you read it from.
--
-- The dashboard's weekly grid reads `where horizon_day = 0`; BI-09's per-day
-- cells read 1 .. 7.
--
-- **Three different times, because they answer three different questions.**
--   forecast_date  the last day of *observed* data the score was computed from
--   horizon_start  the first day the score covers, always forecast_date + 1,
--                  because day t is a feature and cannot be in its own window
--   horizon_end    the last day it covers, forecast_date + horizon_days
--   scored_at      when the row was written, which is not when it was about
--
-- A dashboard that shows "risk for the coming week" needs the first three to
-- label the axis and the fourth to say how stale the answer is. Collapsing any
-- of them into "date" is the ambiguity this table exists to avoid.
--
-- The check constraints are the semantics, enforced. A row whose horizon does
-- not start the day after its forecast_date is not a differently-shaped row,
-- it is a bug that would otherwise be discovered from a chart.
-- ============================================================================

create schema if not exists gold_marts;

create table if not exists gold_marts.fact_ml_predictions (
    -- Grain -------------------------------------------------------------
    city_id             text        not null,
    forecast_date       date        not null,

    -- The window the score is about, carried rather than implied. A reader
    -- of one row should not need the model's documentation to know what
    -- seven days it covers.
    horizon_start       date        not null,
    horizon_end         date        not null,
    horizon_days        smallint    not null,

    -- Which day of the horizon this row is about; 0 is the whole window.
    horizon_day         smallint    not null default 0,

    -- The answer ---------------------------------------------------------
    -- Probability that |Z| > 2.5 occurs on at least one day in the horizon.
    risk_score          double precision not null,

    -- The same answer as a decision, at a threshold chosen on the validation
    -- split rather than at 0.5. The threshold is stored beside the label
    -- because a boolean with no operating point behind it cannot be audited,
    -- and because changing it later must not silently reinterpret old rows.
    prediction_label    boolean     not null,
    decision_threshold  double precision not null,

    -- Provenance ---------------------------------------------------------
    -- The artefact's own versioned filename stem, so a row can be traced to
    -- the exact file, its metrics, and its training window.
    model_version       text        not null,
    model_variant       text        not null,
    feature_count       smallint    not null,
    scored_at           timestamptz not null default now(),

    primary key (city_id, forecast_date, horizon_day),

    constraint fact_ml_predictions_risk_is_a_probability
        check (risk_score >= 0.0 and risk_score <= 1.0),
    constraint fact_ml_predictions_threshold_is_a_probability
        check (decision_threshold >= 0.0 and decision_threshold <= 1.0),
    constraint fact_ml_predictions_horizon_is_positive
        check (horizon_days > 0),
    -- The forward window starts the day *after* the forecast date. Day t is a
    -- feature; a window that included it would be scoring the model on
    -- something it was given.
    -- A window row starts the day after the forecast date and runs the whole
    -- horizon; a day row is that one day, at both ends. Written as one pair of
    -- constraints over `horizon_day` rather than as two shapes of row, so a
    -- row that claims to be day 3 and spans a week cannot be inserted.
    constraint fact_ml_predictions_horizon_day_is_in_the_window
        check (horizon_day >= 0 and horizon_day <= horizon_days),
    constraint fact_ml_predictions_horizon_starts_where_it_should
        check (
            horizon_start = forecast_date
                + case when horizon_day = 0 then 1 else horizon_day end
        ),
    constraint fact_ml_predictions_horizon_ends_where_it_should
        check (
            horizon_end = forecast_date
                + case when horizon_day = 0 then horizon_days else horizon_day end
        ),
    -- The label is the score against the threshold, and nothing else -- on a
    -- window row. A day row carries the *week's* decision beside its own
    -- hazard, deliberately: ML-11 chose a threshold for a weekly alert budget
    -- and no threshold has ever been chosen for a single day, so deriving a
    -- per-day label here would invent an operating point nobody selected. The
    -- check is therefore scoped rather than dropped, so the property still
    -- holds everywhere it means anything.
    constraint fact_ml_predictions_label_matches_the_threshold
        check (
            horizon_day > 0
            or prediction_label = (risk_score >= decision_threshold)
        )
);

-- ----------------------------------------------------------------------------
-- Migration: horizon_day (ML-13)
-- ----------------------------------------------------------------------------
-- The create above is a no-op on a database that already has the table, so the
-- column has to be added separately for one built before ML-13. Written as
-- plain idempotent DDL rather than a migration framework, which is what the
-- rest of this file is and what a single-writer derived mart needs.
--
-- Existing rows are weekly rows, so the default of 0 is correct for every one
-- of them and no backfill is needed.
alter table gold_marts.fact_ml_predictions
    add column if not exists horizon_day smallint not null default 0;

do $$
begin
    -- Widen the key only if it is still the pre-ML-13 one. Comparing the
    -- column list rather than the constraint name, because the name is
    -- unchanged by the widening and would say nothing.
    if exists (
        select 1
        from information_schema.key_column_usage
        where table_schema = 'gold_marts'
          and table_name = 'fact_ml_predictions'
          and constraint_name = 'fact_ml_predictions_pkey'
        group by constraint_name
        having count(*) = 2
    ) then
        alter table gold_marts.fact_ml_predictions
            drop constraint fact_ml_predictions_pkey;
        alter table gold_marts.fact_ml_predictions
            add primary key (city_id, forecast_date, horizon_day);
    end if;
end $$;

-- The old single-shape horizon constraints cannot hold for day rows. Dropped
-- by name and replaced with the horizon_day-aware pair; `if exists` so this is
-- a no-op on a database created after ML-13.
alter table gold_marts.fact_ml_predictions
    drop constraint if exists fact_ml_predictions_horizon_starts_the_next_day;

-- `horizon_ends_where_it_should` keeps its name and changes its meaning, which
-- is the one case `if not exists` cannot detect: on a pre-ML-13 database the
-- constraint is present, still says `horizon_end = forecast_date +
-- horizon_days`, and would reject every day row while a check on the name
-- alone reported everything in order. Dropped unconditionally and re-added.
alter table gold_marts.fact_ml_predictions
    drop constraint if exists fact_ml_predictions_horizon_ends_where_it_should;

do $$
begin
    if not exists (
        select 1 from pg_constraint
        where conname = 'fact_ml_predictions_horizon_day_is_in_the_window'
    ) then
        alter table gold_marts.fact_ml_predictions
            add constraint fact_ml_predictions_horizon_day_is_in_the_window
            check (horizon_day >= 0 and horizon_day <= horizon_days),
            add constraint fact_ml_predictions_horizon_starts_where_it_should
            check (
                horizon_start = forecast_date
                    + case when horizon_day = 0 then 1 else horizon_day end
            );
    end if;
end $$;

alter table gold_marts.fact_ml_predictions
    add constraint fact_ml_predictions_horizon_ends_where_it_should
    check (
        horizon_end = forecast_date
            + case when horizon_day = 0 then horizon_days else horizon_day end
    );

-- Same story as the horizon constraints: the name is unchanged and the meaning
-- is not, so `if not exists` would report a pre-ML-13 constraint as current
-- while it rejected every day row.
alter table gold_marts.fact_ml_predictions
    drop constraint if exists fact_ml_predictions_label_matches_the_threshold;

alter table gold_marts.fact_ml_predictions
    add constraint fact_ml_predictions_label_matches_the_threshold
    check (
        horizon_day > 0
        or prediction_label = (risk_score >= decision_threshold)
    );

-- BI-09 reads the day rows for one forecast date across every city.
create index if not exists fact_ml_predictions_horizon_day_idx
    on gold_marts.fact_ml_predictions (forecast_date desc, horizon_day);

comment on table gold_marts.fact_ml_predictions is
  'One row per city per forecast_date per horizon_day: the current risk of an extreme temperature anomaly over the span the row covers. horizon_day 0 is the whole window; 1..n are single days. Re-scoring replaces rather than appends.';
comment on column gold_marts.fact_ml_predictions.horizon_day is
  'Which day of the horizon this row is about. 0 = the whole window (horizon_start..horizon_end); k = the single day forecast_date + k, from the discrete-time hazard. horizon_start and horizon_end say the same thing and are constrained to agree.';
comment on column gold_marts.fact_ml_predictions.forecast_date is
  'Last day of observed data used. The score covers the days AFTER this one.';
comment on column gold_marts.fact_ml_predictions.horizon_start is
  'First day covered, always forecast_date + 1; day t is a feature, not part of its own window.';
comment on column gold_marts.fact_ml_predictions.risk_score is
  'P(|Z| > 2.5 on at least one day in horizon_start..horizon_end). On a horizon_day = 0 row that is the week; on a day row it is that day''s hazard, conditional on no anomaly earlier in the window. Trained on a period with a lower base rate than the present, so it reads low; see the calibration note in the README.';
comment on column gold_marts.fact_ml_predictions.model_version is
  'Versioned artefact filename stem, e.g. model-unweighted-v1-2ad772ff7b18.';
comment on column gold_marts.fact_ml_predictions.scored_at is
  'When the row was written. Freshness of the answer, not of the data; forecast_date is that.';

-- The dashboard reads the newest forecast_date per city, and an operator
-- clearing a bad run reads by model_version.
create index if not exists fact_ml_predictions_forecast_date_idx
    on gold_marts.fact_ml_predictions (forecast_date desc);
create index if not exists fact_ml_predictions_model_version_idx
    on gold_marts.fact_ml_predictions (model_version);
