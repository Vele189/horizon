# Model card: short-horizon wind gust nowcast

**What it does.** Given a city and an hour, it estimates the **peak wind gust**
over the next 24, 48 or 72 hours, in km/h. Three separate models, one per
horizon.

**What it is not.** This is not the seven-day anomaly classifier and does not
replace it. That model answers "will a temperature anomaly occur"; this one
answers "how hard will the wind gust". They share a repository, a warehouse and
a discipline, and nothing else — not the target, not the grain, not the split
boundaries, not the baselines. Reading a number from one against a number from
the other is a category error.

It is also not a weather forecast. It has never seen a pressure field, a
satellite image or a model run; it reads one city's own recent hours. An
operational nowcast assimilates observations across a region and solves the
equations of motion. This fits a gradient-boosted tree to a station's history
and beats persistence, which is a much smaller claim.

| | |
|---|---|
| Type | XGBoost gradient-boosted trees, regression on `reg:squarederror` |
| Target | `max(wind_gusts_10m)` over `t+1 .. t+H`, km/h |
| Horizons | 24, 48 and 72 hours, fitted separately |
| Trained | 2024-09-04 to 2025-08-31, one full annual cycle |
| Evaluated | 2026-01-01 to 2026-09-01, 15 cities |
| Source | `gold_marts.fact_weather_hourly`, 263 160 rows, no gaps and no nulls |

---

## Key figures

Every number here is asserted against
`machine_learning/artifacts/nowcast-metrics.json` by
`tests/test_nowcast_card.py`, so this table cannot drift from the models it
describes.

| figure | value |
|---|---|
| cities | 15 |
| feature_count | 37 |
| seed | 42 |
| train_rows_24h | 129 975 |
| validation_rows_24h | 43 200 |
| test_rows_24h | 87 840 |
| test_sd_observed_24h | 12.15 |
| test_rmse_24h | 8.376 |
| test_mae_24h | 6.270 |
| test_persistence_rmse_24h | 11.208 |
| test_climatology_month_rmse_24h | 11.243 |
| test_skill_vs_persistence_24h | 0.2526 |
| test_skill_vs_climatology_month_24h | 0.2550 |
| test_rmse_48h | 9.538 |
| test_persistence_rmse_48h | 12.450 |
| test_climatology_month_rmse_48h | 11.367 |
| test_skill_vs_persistence_48h | 0.2338 |
| test_skill_vs_climatology_month_48h | 0.1609 |
| test_rmse_72h | 10.113 |
| test_persistence_rmse_72h | 12.517 |
| test_climatology_month_rmse_72h | 11.392 |
| test_skill_vs_persistence_72h | 0.1920 |
| test_skill_vs_climatology_month_72h | 0.1123 |
| test_non_overlapping_rmse_24h | 8.490 |
| test_non_overlapping_rmse_72h | 10.165 |
| worst_city_skill_72h | -0.2531 |

---

## Intended use

**In scope.** Illustrating that a warehouse-backed feature pipeline can produce
a short-horizon regression with real skill over the standard naive baselines,
in a portfolio dashboard, read by someone who can also read this card.

**Out of scope, and these are not hypothetical.**

- **Any decision where wind matters.** Not for aviation, marine, construction,
  crane operation, event safety, grid planning, or anything where a person acts
  on the number. Operational gust guidance comes from numerical weather
  prediction and this is not that.
- **Cities outside the fifteen.** The model carries no city identifier, so it
  will produce a number for anywhere; the number has not been validated
  anywhere else.
- **Singapore at 48 and 72 hours,** where it is measurably worse than that
  city's own monthly mean. See Limitation 3.
- **Extremes.** Squared-error regression fits the conditional mean. The
  strongest gusts are exactly where it will under-predict, and no part of this
  card should be read as a claim about damaging wind.

---

## Data

**Source.** ERA5 reanalysis via the Open-Meteo archive API, hourly grain,
loaded and reconciled by `ingestion/` on the same manifest discipline as the
daily record.

**Coverage.** Fifteen cities, 2024-09-02 to 2026-09-02, 17 544 hours each,
263 160 rows. Zero nulls in every column the model reads and no missing hours
in any city — this is the only mart in the warehouse that is complete, which is
why the nowcast exists on it and the anomaly model does not.

**Features.** 37 columns, all trailing windows ending at hour *t*, listed in
`machine_learning/features.py` and built by `build_hourly_features`. Current
conditions, pressure tendency at 3 and 24 hours and its magnitude, rolling
gust/pressure/precipitation statistics at 6, 24 and 72 hours, and cyclical
encodings of hour and day of year.

**Wind direction is a sine and a cosine, never a bearing.** 359° and 1° are two
degrees apart and 358 units apart on a number line. Fed raw, a tree learns a
split at north that means nothing. `tests/test_nowcast.py` asserts the circular
distance directly.

---

## Evaluation

**Split.** Chronological, purged by the horizon. Training is exactly one year,
which is the constraint everything else bends around: a model that has not seen
a full annual cycle has never seen the season it is asked about. Validation is
the four months after training and test the eight after that.

An origin within *H* hours of a fold boundary has a target reaching into the
next fold, so those origins are dropped from the end of each fold. The leak
would be small, real, and would flatter precisely the rows used for early
stopping.

**Baselines, and why there are two.** The ticket asks for persistence — the
peak gust over the *previous* window of the same length — and at 24 hours it is
a genuine opponent. At 48 and 72 it is not: its RMSE (12.45 and 12.52) is
**worse than simply predicting each city's monthly mean** (11.37 and 11.39), so
a skill number quoted against it alone at those horizons would be a statement
about persistence rather than about the model. Both are reported at every
horizon, and the climatology is fitted on training rows only.

**Skill decays with horizon, as it should.**

| horizon | RMSE | vs persistence | vs climatology | observed SD |
|---|---|---|---|---|
| 24h | 8.38 | +25.3% | +25.5% | 12.15 |
| 48h | 9.54 | +23.4% | +16.1% | 12.65 |
| 72h | 10.11 | +19.2% | +11.2% | 12.96 |

**Overlapping origins inflate precision, not skill.** Hourly origins mean two
forecasts an hour apart share 71 of 72 hours of their answer, so the effective
sample size is far below 87 840. Re-scored on one origin a day the RMSE is
8.490 at 24 hours against 8.376 on all origins, and 10.165 against 10.113 at
72 — the same answer, from 3 660 independent windows instead.

### What the features are actually worth

The importance table says the model is mostly current wind: `wind_speed_10m`
alone carries 0.31 of the gain at 24 hours, and `pressure_tendency_24h` 0.017.
**That reading is wrong, and the ablation says so.** Gain is split among
correlated features close to arbitrarily, and every pressure column here is
correlated with every wind column through the weather that produced both.

Refitting without each block, on validation and on test:

| dropped | 24h val / test | 48h val / test | 72h val / test |
|---|---|---|---|
| pressure | +3.50% / +3.79% | +2.08% / +3.86% | +3.11% / +2.41% |
| recent wind | +6.14% / +8.46% | +2.32% / +4.29% | +0.82% / +2.36% |
| seasonal | +0.27% / −0.46% | −0.12% / −0.13% | +0.91% / −1.13% |

Positive is the cost of removal. Two things follow.

**The pressure block earns its place**, worth 2 to 4% of RMSE at every horizon
on both folds, which is the ticket's premise vindicated against its own
importance table. And its value *overtakes* current wind as the horizon grows:
at 24 hours recent wind is worth more than twice as much as pressure, and by 72
hours they are level (+2.41% against +2.36%). The further ahead the question,
the less the current gust says and the more the pressure field does.

**The seasonal block is kept although test says drop it.** Test prefers its
removal at all three horizons; validation prefers keeping it at two of three.
Validation is the fold decisions may be made on, so the block stays. Choosing
otherwise would be a feature-selection step performed with test labels, which
is the thing this project refuses everywhere else — and with one year of
training there is a good reason to distrust the test reading anyway: a
day-of-year feature fitted on a single annual cycle can only memorise it.

---

## Limitations

### 1. One year of training, and folds that are not seasonally comparable

Two years of hourly record cannot give both a full annual cycle in training and
seasonally matched folds. Training gets the cycle; validation is September to
December and test is January to September, so they are different weather.
Every skill figure here carries that. A random split would balance the seasons
and would also let the model see 3 p.m. to predict 4 p.m. on the same
afternoon, which is worse.

### 2. It fits the conditional mean, so it under-predicts extremes

Squared-error regression is the right loss for the average case and the wrong
one for the tail. The strongest gusts are where the model will be furthest
short, and that is the direction that matters for anything a person would act
on. There is no quantile or distributional head here and no extreme-value
treatment of the residuals; `machine_learning/extremes.py` does that work for
temperature and nothing equivalent exists for wind.

### 3. In one city the seasonal mean beats it outright

Singapore's gusts have a standard deviation of about 5 km/h — there is very
little to predict. At 24 hours the model is 8.8% better than that city's
monthly climatology; at 48 hours it is 11.8% **worse**, and at 72 hours 25.3%
worse. São Paulo joins it at 72 hours (−6.3%) and London is level (−0.0%).

This is reported rather than fixed. Fixing it by falling back to climatology
per city and horizon would be a rule chosen by looking at test results, and the
honest version of "the model does not help here" is a card that says so.

### 4. No city identifier, and no per-city fit

One model serves all fifteen cities and carries no city column, so it cannot
learn that Reykjavík is windy — it has to read that off the recent hours. This
is deliberate (the daily model's per-city lookup is Limitation 4 of its own
card) and it costs accuracy in the cities with the most distinctive climate.

### 5. Two years cannot contain a rare event

The record holds two of each season. Any gust regime that recurs on a longer
cycle than that is absent from training and from test, and the model has no way
to represent it.

---

## Ethical considerations

The out-of-scope list is the substance of this section. A gust nowcast is the
kind of output that looks operational — it is a speed, in the units a forecast
uses, on a familiar horizon — and the distance between it and something a crane
operator could use is not visible from the number. The model card, the
dashboard and this repository's README all say the same thing: this is a
demonstration of a pipeline, and the only defensible use is reading it beside
the evidence that produced it.

---

## Reproducing this

```bash
python dbt_analytics/dbt_env.py -- dbt build          # fact_weather_hourly
python machine_learning/nowcast.py --ablate pressure recent_wind seasonal --write
```

Single-threaded and seeded, so the metrics are bit-identical across machines:
XGBoost's histogram builder is deterministic for a given thread count and not
across them, because per-thread gradient sums are added in whatever order the
threads finish and floating-point addition is not associative.
