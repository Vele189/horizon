# Model card: extreme temperature anomaly classifier

**What it does.** Given a city and a date, it estimates the probability that at
least one day in the following week is an extreme temperature anomaly, meaning daily
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
| Version | `model-unweighted-v1-e3559dcc6b5a` |
| Type | XGBoost gradient-boosted trees, binary classification |
| Trained | 1995-01-31 to 2018-12-24, 96 019 city-days, 11 cities |
| Evaluated | 2022-01-01 to 2026-09-02, 18 100 city-days, 11 cities |
| Snapshot | 126 266 rows to 2026-09-02, and **the backfill is incomplete** |

---

## Key figures

Every number here is asserted against `machine_learning/artifacts/metrics.json`
by `tests/test_model_card.py`, so this table cannot drift from the model it
describes. If a retrain moves a score, this card fails until it is updated.

| figure | value |
|---|---|
| model_version | model-unweighted-v1-e3559dcc6b5a |
| feature_count | 27 |
| train_rows | 96019 |
| train_positive_rate | 0.0487 |
| test_rows | 18100 |
| test_positive_rate | 0.1130 |
| test_pr_auc | 0.2886 |
| test_brier | 0.0946 |
| test_f1 | 0.3255 |
| test_precision | 0.3284 |
| test_recall | 0.3227 |
| decision_threshold | 0.1174 |
| alert_budget_per_city_year | 20.0 |
| alert_budget_threshold | 0.2010 |
| implied_cost_ratio | 3.98 |
| test_mean_predicted | 0.0611 |
| baseline_persistence_pr_auc | 0.1913 |
| baseline_climatology_pr_auc | 0.1250 |
| no_skill_pr_auc | 0.1130 |
| static_city_share_of_shap | 0.0956 |
| n_estimators | 126 |
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
  direction; see Limitation 1.
- **Any city not in the training set.** 13.7% of the model's decision weight is
  a per-city lookup, so it does not transfer.
- **Displaying `risk_score` as a probability without recalibration.** The rank
  is meaningful; the level is not.
- **Long-range or single-day forecasting.** It answers one question, "any
  extreme day in the next seven", and cannot say which day, how severe, or how
  long an event will last.

---

## Data

**Source.** ERA5 reanalysis via the Open-Meteo archive API. Gridded
reanalysis, not station observations: the model sees whichever ~25 km grid
cell answers for a city's coordinates.

**Target.** `|Z| > 2.5` on at least one day in `t+1 ... t+7`, where Z is the
daily mean temperature against a ±7-day day-of-year climatology that **excludes
the observation's own year**. Both tails: a 2.5σ cold outbreak counts.

**Features.** 27, all strictly backward-looking from day *t* inclusive. The
order is part of the contract, because a caller who supplies the right columns
in the wrong order gets a confident wrong answer, so the list is printed rather than
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

| | PR-AUC | vs base rate | vs persistence | Brier | F1 |
|---|---:|---:|---:|---:|---:|
| no-skill reference | 0.1130 | 1.00x | 0.59x | 0.1043 | 0.2030 |
| climatology baseline | 0.1250 | 1.11x | 0.65x | 0.1045 | 0.1597 |
| persistence baseline | 0.1913 | 1.69x | 1.00x | 0.0976 | 0.3418 |
| **this model** | **0.2886** | **2.55x** | **1.51x** | **0.0946** | 0.3255 |

Every score carries its lift over **persistence** as well as over the base
rate. The base rate is the floor a random ranker scores by construction;
persistence is the number that has to be beaten. The climatology baseline shows
why the distinction is not pedantic: 1.11x the base rate is barely
distinguishable from no out-of-sample skill at all, because shrinkage collapses
it to a per-city rate and the per-city ordering does not survive the split
(Spearman near zero between the training and test periods).

This model beats every baseline on ranking and on calibration, and beats
persistence in every city individually, from 1.36x in Singapore to 2.92x in
Phoenix. On F1 it is *behind* persistence, 0.3255 against 0.3418.

**Which of those survive a different threshold.** ML-09 re-ran the whole
evaluation at |Z| 2.0, 2.5 and 3.0 — refitting, since the threshold moves two
features as well as the label:

| \|Z\| | model PR-AUC | persistence | vs persistence | model F1 | persistence F1 |
|---:|---:|---:|---:|---:|---:|
| 2.0 | 0.4918 | 0.3370 | 1.46x | 0.4647 | 0.4183 |
| 2.5 | 0.2886 | 0.1913 | 1.51x | 0.3255 | 0.3418 |
| 3.0 | 0.1915 | 0.0964 | 1.99x | 0.2769 | 0.2524 |

PR-AUC and Brier hold at all three. **F1 does not** — the model is behind
persistence at 2.5 and ahead at 2.0 and 3.0 — so the ranking and calibration
verdicts are properties of the model and the F1 verdict is a property of where
the line sits. The advantage on ranking *grows* with the threshold, 1.46x to
1.99x, so the model is not living on the easy half of the distribution. At
|Z| > 3.0 it beats persistence in ten of eleven cities rather than all eleven.

**Accuracy is not reported.** At a 13.58% base rate, always answering "no
anomaly" scores 86.4% and predicts nothing.

### The decision the threshold encodes

`threshold_metric: f1` was what this file used to record, and F1 is the
harmonic mean of precision and recall — a way of saying that a false alarm and
a missed heatwave are equally bad. Nobody would defend that out loud, and
nobody had been asked to. Worse, an F1-optimal threshold found where positives
are 7.1% of rows is not F1-optimal where they are 11.5%, so even the
indefensible rule was being applied off its own terms.

**The rule is now an alert budget: a city may light up no more than 20 days a
year.** A cost ratio would be the more fundamental object — how many false
alarms are worth one missed extreme week — and nobody here has that number.
This project ships a dashboard, not a warning system with a loss function
behind it, and inventing a ratio to justify a threshold would be dressing an
arbitrary choice as an analysis. A budget can be defended by someone who knows
the product rather than the cost of a heatwave.

Twenty, because alerts arrive in runs rather than singly — the label is a
seven-day window, and at this threshold a run averages 2.3 days — so twenty
alert-days is roughly nine separate alert periods a year, one every six weeks.
Often enough to be worth looking at, rare enough not to become wallpaper.

The two rules are not really rivals, and that is the useful part. On a
**calibrated** probability the expected-cost-minimising cut for a cost ratio
*c* is 1/(1+*c*), so a threshold *is* a cost ratio, and this one asserts that
**3.98 false alarms are worth one missed week**. F1 at 0.5 was asserting 1.0,
silently. ML-10 is what makes that translation legal; on a raw score it would
be arithmetic with no meaning attached.

The threshold is chosen on **validation** and the neighbours either side are
recorded so the trade is visible rather than asserted:

| | threshold | precision | recall | alerts per city-year |
|---|---:|---:|---:|---:|
| looser | 0.1975 | 0.301 | 0.276 | 22.9 |
| **chosen** | **0.2010** | **0.329** | **0.237** | **17.9** |
| tighter | 0.2714 | 0.397 | 0.187 | 11.7 |

**And the budget is overspent on test, by 61%.** The same threshold delivers
32.3 alerts per city-year there, at precision 0.356 and recall 0.278, because
validation has fewer anomalous days per city-year than test does. A budget set
on one period and spent on another is not a guarantee; it is the same drift
every other ticket in this phase is about, and it is recorded as a number
rather than left to be discovered. F1 on the same calibrated probabilities
would have chosen 0.1765 and spent 39.6.

**What the dashboard currently applies is still the F1 threshold**, because
`fact_ml_predictions` holds raw model scores and the budget rule is defined on
calibrated ones. Shipping it means shipping the calibrator alongside the model
artefact, which is a serving change and not this ticket's.

---

**F1 is reported at 0.1174**, the threshold that maximises F1 on the
*validation* split. At 0.5 this model flags nothing at all and scores F1 =
0.00. Note how little F1 separates the model from persistence — 0.3255 against
0.3418, the wrong way — compared with PR-AUC, 0.2886 against 0.1913: F1
collapses the whole curve to one point, and that point is where persistence is
strongest. The sweep above is the clearest available demonstration that a
single-point metric measures the point and not the predictors: the same two
predictors change places twice as the threshold moves.

---

## Limitations

### 1. It under-states risk in the present climate

The most important line in this card. The positive rate is **4.87% in the
training period and 11.30% in the test period**, more than doubling, because
the climatology baseline spans the whole record and the climate has warmed
within it. The model is correctly calibrated to a world that no longer exists.

DBT-12 tested whether that is an artefact of the baseline and found it is not:
detrending the climatology removes only 5% of the drift. See limitation 7.

In consequence its mean predicted probability on the test split is **0.061
where 0.113 actually occurs**, and the shortfall runs through every decile. It
is wrong in the direction that matters for a warning system: it says "quiet"
more often than it should. **`risk_score` must be recalibrated before any
reader sees it as a percentage.**

**Recency weighting was measured and refused (ML-12).** Weighting the training
rows towards the recent record is the cheapest response to that drift, and on
validation it loses monotonically: every year of history discarded costs both
ranking and calibration, and the search correctly declines to weight at all.
The reason is recorded beside the result — the training split contains no drift
to lean on, running at 4.98% in 1995-2006 and 4.75% in 2007-2018, with the
whole rise sitting between the splits. On *test* a sixteen-year half-life is
15% better on PR-AUC, in both halves of the period, but choosing it on that
basis would be selecting on test. It is recorded and not shipped.

ML-10 measured what recalibration is worth. Isotonic regression fitted on
validation, never on test, halves the expected calibration error, from 0.052 to
0.030, and moves the mean prediction closer to the 0.113 that occurs. It costs
a few per cent of PR-AUC, because isotonic collapses thousands of distinct
scores into a few dozen flat runs and average precision is tie-sensitive. That trade is
worth making for a number a reader sees as a percentage and not for one they
see as a rank, which is why it is recorded rather than applied: the Risk
Horizon view shows rank bands, and BI-08 is the ticket that decides what a
calibrated probability is shown as.

Prior-shift correction on top of it — re-estimating the target period's class
prior by EM over the model's own posteriors, with no labels — is the method
that ought to handle a shift of exactly this kind, and here its estimator
returns a badly biased estimate. Given the right prior the correction is the
best row in the table at ECE 0.019, so the fault is the estimate and not
the adjustment; and no choice among the candidate quantifiers can be made on
validation, because the one that is nearly exact on test collapses to zero
there. It is recorded in full, in `model.calibration`, and not shipped.

### 2. It is not a forecast

No pressure fields, no upper-air data, no NWP output, no teleconnection
indices. It knows a city's own history. Its most confident false positive is
diagnostic: Cairo on 2026-02-16, eight days into a hot spell with a Z of 2.95,
scored 0.683, and the spell simply broke the next day. Its most confident
correct call, Delhi on 2026-03-09, had *less* evidence. **At this horizon it
can tell you a spell is running; it cannot tell you when it will end**, and
nothing in the feature set could.

### 3. Eleven cities of fifteen, and four of them still absent

Trained on Cairo, Delhi, Lagos, London, Moscow, Phoenix, Portland, Reykjavík,
São Paulo, Singapore and Tokyo — mid-latitude maritime, continental winter,
desert, tropical and Southern Hemisphere all now represented, which the earlier
six-city set was not. Sydney has eighteen scored days and Auckland, Buenos
Aires and Johannesburg three each, so four cities are still not scorable at
all and the pooled metrics will move again when they land.

The per-city spread is what the pooled figures hide: base rates run from 4.2%
in Reykjavík to 23.7% in Singapore, and the model's advantage over persistence
from 1.28x to 3.40x. Read the per-city table, not the pooled row.

### 4. 10.4% of the model is a city lookup

`latitude` and `elevation_m` are constant within a city, so to that extent the
model is not learning about elevation, it is learning *which city*, and city
base rates run 4.2% to 23.7%.

This was recorded here as a flat statement that the model "transfers to none",
and ML-08 tested it rather than leaving it asserted. It is wrong. Held out of
training entirely, each of the eleven scored cities is still ranked better by
the model than by its own persistence baseline, at a median 1.61x and a median
98% of the same city's in-sample PR-AUC. London pays the most for being unseen
at 87% and Phoenix gains the most at 119%. Whatever the static columns are
doing, the model is not depending on having met the city: the
`leave_one_city_out` block in `metrics.json` carries every fold, and each
fold's own record names the cities it trained on.

### 5. σ is estimated, and the flag now says so

A Z-score divides by a σ that was estimated from a finite window, and treating
it as known makes the standardised departure a t-statistic read against a
normal table. The error is one-directional: a city whose baseline rests on few
observations over-flags by construction. Sydney, on five reference
observations, flagged 16.7% of its eighteen scored days against 1.0-2.0%
everywhere else, with sd(Z) = 1.67 where every complete city sits within 0.02
of one.

DBT-14 judges each day against a Student-t critical value on its own degrees of
freedom, scaled by √(1 + 1/n) because the day is not in its own baseline:
`anomaly_z_critical` is 2.513 at a complete city's 459 observations, 2.96 at
fifteen, and 62.8 at two, which is the honest answer — two observations cannot
establish that anything is unusual. Sydney now flags 1 of 18 days, a count no
longer distinguishable from the complete-city rate.

It is a correction rather than a new definition: 52 days across the warehouse
lost their flag, two of them Sydney's and fifty from complete cities sitting
between 2.500 and 2.513. Tokyo used to be this entry, at forty-five
observations and sd(Z) = 1.14; its record is now complete.

### 6. How much to believe one week's answer, formally

The probability this model gives a week is a number a reader will round. ML-14
reports what can be said instead, without assuming anything about the model or
the distribution: a conformal **prediction set**, which is `{quiet}`,
`{extreme}`, `{quiet, extreme}` when the evidence does not separate them, or
empty when the week is unlike anything in the calibration sample.

Calibrated on the validation split at a target of **90% coverage**, stated
before anything was measured. And the point of the ticket is what happens next:

| year | split conformal | adaptive |
|---|---:|---:|
| 2022 | 0.906 | 0.899 |
| 2023 | 0.843 | 0.900 |
| 2024 | 0.830 | 0.901 |
| 2025 | 0.806 | 0.899 |
| 2026 | **0.731** | 0.898 |

Ordinary split conformal assumes exchangeability, which a base rate moving from
6.8% to 11.3% violates. Its guarantee holds on paper and **decays silently on
the data**, from on-target in 2022 to sixteen points short by 2026. Adaptive
conformal adjusts its level online and holds 90% in every year, paying for it
with sets that grow from 0.91 labels to 1.08 and a share of "cannot say"
answers that rises from 0% to 10.6%.

That growth *is* the drift, measured in the units the guarantee is stated in,
which is the most legible statement of it anywhere in this project.

Coverage is reported with set size everywhere, because a method that always
returns both labels covers everything and says nothing. And the adaptive figure
is a **realised online coverage**, not a held-out score: the level is updated
from each outcome after that row has been predicted, which is what the method
does in deployment. The calibration sample is validation and only validation; a
test rewrites the test period and requires the calibration to be identical.

### 7. The label conflates two things, and that is now a choice

A day can be flagged because it was unusual for its time of year, or because
the whole record has warmed and a fixed-period baseline now sits low.
`corr(year, Z)` is positive in nine of eleven complete cities, from +0.05 in
Delhi to +0.40 in Lagos. The model inherits that conflation intact, and ML-09's
threshold sweep confirms it is not an artefact of where the line sits: the base
rate more than doubles across the split at |Z| 2.0, 2.5 and 3.0 alike.

**This model is trained on *unusual for the record*.** DBT-12 built the
alternative - a detrended baseline, with a per-city, per-day trend in year
removed and the result referenced to the year being scored, the trend fitted
only on prior years - and DBT-13 ran the validation gate against both. The
detrended flag is carried in `fact_weather_anomalies` and nothing reads it.
Three findings decided that, and the first is the one that mattered:

* Detrending removes **5%** of the drift it was built for: the label's base
  rate rises 2.29x from the training period to the test one, and 2.22x
  detrended. The drift lives in the tail; the trend lives in the centre.
* **No documented extreme changes verdict** between the two. Every checkable
  event moves 0.01-0.19 sigma, in the direction the trend predicts, and none
  crosses the threshold.
* The Climate Matrix view exists to show anomaly counts moving across thirty
  years, and detrending removes that signal by construction.

`docs/proposal.md` §5.3 records the decision in full. The consequence for this
model is the one stated above: the non-stationarity in its label is real rather
than an artefact of the climatology, so it is ML-10's prior-shift correction
that has to address it, not a different flag.

### 8. The climatology feature is not strictly backward in time

Every rolling and lag feature is strictly backward, proved by rebuilding after
rewriting the future and requiring the past to come back bit-identical. The
climatological Z is the exception: its baseline excludes the observation's own
year but not the years *after* it. A 2003 row is scored against a climatology
that has seen 2020. It is a per-(city, day-of-year) constant rather than a path
from any particular future day, and the alternative, an expanding climatology,
would give the early record a baseline of two or three years. The trade is
deliberate.

### 9. A single-day Z cannot express duration

Phoenix's July 2023 heat dome peaks at Z = +1.97 and flags **zero** days. What
was unprecedented was how long it lasted, and a per-day threshold cannot say
that by construction. The detector, and therefore this model, is blind to
duration-defined events.

### 10. Seasonality is real but non-stationary

Fitted and scored inside the training period, a (city, week-of-year) baseline
is worth 2.47x no-skill. Carried across the split it is worth **less than
nothing**, at 0.91x. Fitted on the test period itself it is worth 2.53x again.
The seasonal structure is still there; it is *different* structure, because the
anomaly mix flips from 347 cold / 224 hot in training to 61 / 231 in test.

### 11. Everything is fixed against one snapshot

All figures here describe 59 090 rows to 2026-09-01, across 6 cities. The
backfill is not finished. When cities land, the baselines must be re-run and
re-committed **before** the model is compared against them again, or "target
fixed in advance" stops being true.

---

## What was checked, and what it would take to break it

| property | how it is enforced |
|---|---|
| No feature sees the future | Rewrite every day after a cut, rebuild, require the past bit-identical |
| The label window is exactly t+1...t+7 | One anomaly in a quiet series must label exactly the seven rows before it |
| No feature reconstructs the label | Max single-feature rank AUC 0.643 (`anomaly_days_trailing30`, the persistence signal, not a leak) |
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
else, in a dashboard, a portfolio, or a conversation, would be a
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
