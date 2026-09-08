-- ============================================================================
-- Prediction fact table
-- ============================================================================
-- Idempotent by construction: every statement is IF NOT EXISTS or a COMMENT,
-- so re-running against an existing database is a no-op. Applied by
--
--     python machine_learning/predict.py
--
-- **Grain: one row per city per forecast_date.** Not per model version, and
-- that is a decision rather than an omission. The Risk Horizon view asks "what
-- is the risk for this city right now" and must get exactly one answer; with
-- model_version in the key it would get one row per model ever run and would
-- have to pick between them in the presentation layer, which is where that
-- choice is least visible. A prediction here is the *current best* answer for
-- a city-day, and re-scoring replaces it. If a history of predictions is ever
-- wanted, it belongs in a separate append-only table with its own grain, not
-- in the one a dashboard reads.
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

    primary key (city_id, forecast_date),

    constraint fact_ml_predictions_risk_is_a_probability
        check (risk_score >= 0.0 and risk_score <= 1.0),
    constraint fact_ml_predictions_threshold_is_a_probability
        check (decision_threshold >= 0.0 and decision_threshold <= 1.0),
    constraint fact_ml_predictions_horizon_is_positive
        check (horizon_days > 0),
    -- The forward window starts the day *after* the forecast date. Day t is a
    -- feature; a window that included it would be scoring the model on
    -- something it was given.
    constraint fact_ml_predictions_horizon_starts_the_next_day
        check (horizon_start = forecast_date + 1),
    constraint fact_ml_predictions_horizon_ends_where_it_should
        check (horizon_end = forecast_date + horizon_days),
    -- The label is the score against the threshold, and nothing else.
    constraint fact_ml_predictions_label_matches_the_threshold
        check (prediction_label = (risk_score >= decision_threshold))
);

comment on table gold_marts.fact_ml_predictions is
  'One row per city per forecast_date: the current risk of an extreme temperature anomaly in the following horizon_days. Re-scoring replaces rather than appends.';
comment on column gold_marts.fact_ml_predictions.forecast_date is
  'Last day of observed data used. The score covers the days AFTER this one.';
comment on column gold_marts.fact_ml_predictions.horizon_start is
  'First day covered, always forecast_date + 1; day t is a feature, not part of its own window.';
comment on column gold_marts.fact_ml_predictions.risk_score is
  'P(|Z| > 2.5 on at least one day in horizon_start..horizon_end). Trained on a period with a lower base rate than the present, so it reads low; see the calibration note in the README.';
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
