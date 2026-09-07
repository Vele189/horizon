# Model card — extreme temperature anomaly classifier

**What it does.** Given a city and a date, it estimates the probability that at
least one day in the following week is an extreme temperature anomaly — daily
mean temperature more than 2.5 standard deviations from that city's
day-of-year climatology, in either direction.

**What it is not.** This is not a weather forecast and it does not compete with
ECMWF, GFS, or any operational numerical weather prediction. It has never seen
a pressure field, a satellite image, or a model run. It reads one city's own
recorded history and nothing else, so it is closer to an actuarial estimate
than a forecast: it knows that anomalies cluster and that some weeks of the
year are riskier than others, and it does not know that a ridge is building
over the Sahara. Its purpose is to demonstrate a complete feature engineering,
training and evaluation loop against warehouse data, honestly measured.

| | |
|---|---|
| Version | `model-unweighted-v1-2ad772ff7b18` |
| Type | XGBoost gradient-boosted trees, binary classification |
| Trained | 1995-01-31 to 2018-12-24, 45 069 city-days, 6 cities |
| Evaluated | 2022-01-01 to 2026-09-01, 8 506 city-days, 5 cities |
| Snapshot | 59 090 rows to 2026-09-01 — **the backfill is incomplete** |

---

## Key figures

Every number here is asserted against `machine_learning/artifacts/metrics.json`
by `tests/test_model_card.py`, so this table cannot drift from the model it
describes. If a retrain moves a score, this card fails until it is updated.

| figure | value |
|---|---|
| model_version | model-unweighted-v1-2ad772ff7b18 |
| feature_count | 27 |
| train_rows | 45069 |
| train_positive_rate | 0.0551 |
| test_rows | 8506 |
| test_positive_rate | 0.1358 |
| test_pr_auc | 0.3494 |
| test_brier | 0.1102 |
| test_f1 | 0.3863 |
| test_precision | 0.3735 |
| test_recall | 0.4000 |
| decision_threshold | 0.1024 |
| test_mean_predicted | 0.0703 |
| baseline_persistence_pr_auc | 0.2293 |
| baseline_climatology_pr_auc | 0.1514 |
| no_skill_pr_auc | 0.1358 |
| static_city_share_of_shap | 0.1371 |
| n_estimators | 18 |
| seed | 42 |

---

## Intended use

**In scope.** Illustrative risk ranking across a small set of cities, in a
portfolio dashboard, read by someone who can also read this card. Ranking is
what it does well: it orders city-days by risk substantially better than either
baseline, and the ordering holds in every city individually.

**Out of scope, and these are not hypothetical.**

- **Operational or public-safety use.** Not for emergency planning, health
  advisories, agricultural decisions, insurance pricing, or anything where a
  person acts on the number. It is under-calibrated in the *dangerous*
  direction — see Limitation 1.
- **Any city not in the training set.** 13.7% of the model's decision weight is
  a per-city lookup, so it does not transfer.
- **Displaying `risk_score` as a probability without recalibration.** The rank
  is meaningful; the level is not.
- **Long-range or single-day forecasting.** It answers one question — "any
  extreme day in the next seven" — and cannot say which day, how severe, or how
  long an event will last.

---

## Data

**Source.** ERA5 reanalysis via the Open-Meteo archive API. Gridded
reanalysis, not station observations — the model sees whichever ~25 km grid
cell answers for a city's coordinates.

**Target.** `|Z| > 2.5` on at least one day in `t+1 … t+7`, where Z is the
daily mean temperature against a ±7-day day-of-year climatology that **excludes
the observation's own year**. Both tails: a 2.5σ cold outbreak counts.

**Features.** 27, all strictly backward-looking from day *t* inclusive. The
order is part of the contract — a caller who supplies the right columns in the
wrong order gets a confident wrong answer — so the list is printed rather than
described:

1. `temperature_2m_mean`
2. `temperature_2m_mean_lag1`
3. `temperature_2m_mean_lag3`
4. `temperature_2m_mean_lag7`
5. `temperature_2m_mean_lag14`
6. `temperature_2m_mean_roll7_mean`
7. `temperature_2m_mean_roll7_var`
8. `temperature_2m_mean_roll30_mean`
9. `temperature_2m_mean_roll30_var`
10. `pressure_msl_mean`
11. `pressure_msl_mean_lag1`
12. `pressure_msl_mean_lag3`
13. `pressure_msl_mean_lag7`
14. `pressure_msl_mean_lag14`
15. `pressure_msl_mean_roll7_mean`
16. `pressure_msl_mean_roll7_var`
17. `pressure_msl_mean_roll30_mean`
18. `pressure_msl_mean_roll30_var`
19. `temperature_2m_mean_z_trailing30`
20. `pressure_tendency_24h`
21. `pressure_tendency_72h`
22. `z_temperature_2m_mean`
23. `anomaly_days_trailing30`
24. `day_of_year_sin`
25. `day_of_year_cos`
26. `latitude`
27. `elevation_m`

`temperature_2m_mean_z_trailing30` standardises day *t* against the 30 days
**before** it, not the 30 ending on it: including *t* in its own baseline pulls
the mean towards it and shrinks the score of exactly the extreme days that
matter.

**Splits.** Chronological and purged. Train to 2018-12-24, validate to
2021-12-24, test from 2022-01-01, with seven days dropped at each boundary so
no training row's label window reaches into the period it is validated on.

---

## Evaluation

Test split, against baselines fixed and committed before the model was trained.

| | PR-AUC | lift | Brier | F1 |
|---|---:|---:|---:|---:|
| no-skill reference | 0.1358 | 1.00× | 0.1239 | 0.2391 |
| climatology baseline | 0.1514 | 1.12× | 0.1245 | 0.2116 |
| persistence baseline | 0.2293 | 1.69× | 0.1146 | 0.3807 |
| **this model** | **0.3494** | **2.57×** | **0.1102** | **0.3863** |

It beats both baselines on all three metrics, and beats persistence in every
city individually — 1.7× in Singapore to 5.9× in Delhi.

**Accuracy is not reported.** At a 13.58% base rate, always answering "no
anomaly" scores 86.4% and predicts nothing.

**F1 is reported at 0.1024**, the threshold that maximises F1 on the
*validation* split. At 0.5 this model flags nothing at all and scores F1 =
0.00. Note how little F1 separates the model from persistence (0.3863 against
0.3807) compared with PR-AUC (0.3494 against 0.2293): F1 collapses the curve to
one point, and that point is where persistence is strongest.

---

## Limitations

### 1. It under-states risk in the present climate

The most important line in this card. The positive rate is **5.51% in the
training period and 13.58% in the test period** — it roughly doubles, then
doubles again — because the climatology baseline spans the whole record and the
climate has warmed within it. The model is correctly calibrated to a world that
no longer exists.

In consequence its mean predicted probability on the test split is **0.070
where 0.136 actually occurs**, and the shortfall runs through every decile. It
is wrong in the direction that matters for a warning system: it says "quiet"
more often than it should. **`risk_score` must be recalibrated before any
reader sees it as a percentage.**

### 2. It is not a forecast

No pressure fields, no upper-air data, no NWP output, no teleconnection
indices. It knows a city's own history. Its most confident false positive is
diagnostic: Cairo on 2026-02-16, eight days into a hot spell with a Z of 2.95,
scored 0.683 — and the spell simply broke the next day. Its most confident
correct call, Delhi on 2026-03-09, had *less* evidence. **At this horizon it
can tell you a spell is running; it cannot tell you when it will end**, and
nothing in the feature set could.

### 3. Six cities, five scorable, and all of them hot

Trained on Cairo, Delhi, Lagos, Phoenix, Singapore and Tokyo. No mid-latitude
maritime climate, no continental winter, no Southern Hemisphere. The pooled
metrics are not representative of the intended fifteen-city set and will move
when the backfill completes. Ten of fifteen cities cannot be scored at all
today — six never ingested, four with no usable window.

### 4. 13.7% of the model is a city lookup

`latitude` and `elevation_m` are third and fifth by SHAP importance and are
constant within a city. The model is not learning about elevation; it is
learning *which city*, and city base rates run 5.2% to 23.7%. This is
legitimate and useful with six cities and transfers to none.

### 5. Tokyo's baseline is four years, not thirty

Only four years of Tokyo have been ingested, so its leave-one-year-out σ is
noisy and it flags 3.9% of days against ~1.6% elsewhere. It contributes to
training with an inflated anomaly rate.

### 6. The label conflates two things

A day can be flagged because it was unusual for its time of year, or because
the whole record has warmed and a fixed-period baseline now sits low.
`corr(year, Z)` is positive in every city, from +0.05 in Delhi to +0.38 in
Lagos. The model inherits that conflation intact.

### 7. The climatology feature is not strictly backward in time

Every rolling and lag feature is strictly backward — proved by rebuilding after
rewriting the future and requiring the past to come back bit-identical. The
climatological Z is the exception: its baseline excludes the observation's own
year but not the years *after* it. A 2003 row is scored against a climatology
that has seen 2020. It is a per-(city, day-of-year) constant rather than a path
from any particular future day, and the alternative — an expanding climatology
— would give the early record a baseline of two or three years. The trade is
deliberate.

### 8. A single-day Z cannot express duration

Phoenix's July 2023 heat dome peaks at Z = +1.97 and flags **zero** days. What
was unprecedented was how long it lasted, and a per-day threshold cannot say
that by construction. The detector — and therefore this model — is blind to
duration-defined events.

### 9. Seasonality is real but non-stationary

Fitted and scored inside the training period, a (city, week-of-year) baseline
is worth 2.47× no-skill. Carried across the split it is worth **less than
nothing** — 0.91×. Fitted on the test period itself it is worth 2.53× again.
The seasonal structure is still there; it is *different* structure, because the
anomaly mix flips from 347 cold / 224 hot in training to 61 / 231 in test.

### 10. Everything is fixed against one snapshot

All figures here describe 59 090 rows to 2026-09-01, across 6 cities. The
backfill is not finished. When cities land, the baselines must be re-run and
re-committed **before** the model is compared against them again, or "target
fixed in advance" stops being true.

---

## What was checked, and what it would take to break it

| property | how it is enforced |
|---|---|
| No feature sees the future | Rewrite every day after a cut, rebuild, require the past bit-identical |
| The label window is exactly t+1…t+7 | One anomaly in a quiet series must label exactly the seven rows before it |
| No feature reconstructs the label | Max single-feature rank AUC 0.643 (`anomaly_days_trailing30` — the persistence signal, not a leak) |
| The split is chronological | `max(train) < min(validation)`, plus a 7-day purge, plus a repo-wide scan for random splitters |
| No resampling | Repo-wide scan for `imblearn` and the oversamplers |
| Reproducible | Seed 42, single thread; two runs in separate processes must produce identical metrics |
| The explanation is of *this* model | SHAP contributions plus base value must reproduce `predict_proba` exactly |
| The artefact is what the metrics describe | SHA-256 checked on load; feature list and order checked before predicting |

Each of those has a companion test that deliberately breaks the property and
requires the check to catch it, because a check that passes by finding nothing
is otherwise indistinguishable from one that looks nowhere.

---

## Ethical considerations

The failure mode with consequences is Limitation 1: **the model understates
risk in the current climate**, and it understates it most for the extreme
weeks. A system that quietly says "quiet" is worse than one that is visibly
uncertain. Anything built on this must show the calibration curve beside the
number, or recalibrate first.

The second is scope. Extreme heat is a mortality risk, concentrated among
people least able to avoid it. A model trained on six grid cells over a
truncated record is a demonstration of method, and presenting it as anything
else — in a dashboard, a portfolio, or a conversation — would be a
misrepresentation regardless of how good the PR-AUC looked.

---

## Reproducing this

```bash
python machine_learning/baselines.py --write   # fix the target first
python machine_learning/train.py     --write   # fit both variants
python machine_learning/evaluate.py  --write   # score, and draw the curves
python machine_learning/explain.py   --write   # SHAP
python machine_learning/predict.py             # score the current horizon
```

Seed 42, `n_jobs=1`. The thread count is pinned because XGBoost's histogram
builder is deterministic for a given thread count and not across thread counts:
per-thread gradient sums are added in whatever order the threads finish, so a
four-core laptop and a sixteen-core runner produce two different models.
