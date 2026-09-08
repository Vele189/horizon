# Climate Volatility & Risk Engine

An end-to-end analytics platform that ingests three decades of global weather
observations, models them into a governed dimensional warehouse, and forecasts
extreme temperature anomalies across fifteen major cities.

Python, PostgreSQL, dbt, XGBoost, Streamlit.

## Live dashboard

<!-- BI-07: replace the placeholder below with the deployed URL, and set the
     same URL in the repository's About field. tests/check_deployment.py
     verifies it and records the cold start. -->

**Not yet deployed.** The app is built, verified against Neon, and cleared for
publication. The deploy itself is a console action that has not been taken.

```bash
streamlit run dashboard/app.py                    # locally, against Neon
python tests/check_deployment.py <url> --cold     # once it is published
```

## Five-minute setup

```bash
git clone https://github.com/<user>/horizon && cd horizon
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-pipeline.txt   # ingestion, dbt, model
                                           # (it pulls requirements.txt, the
                                           #  dashboard runtime, with it)

cp .env.example .env                       # then fill in the values
python config.py                           # prints resolved config, secrets masked

docker compose up -d                       # local PostgreSQL 16
python ingestion/backfill.py --grain daily  # then --grain hourly
python dbt_analytics/dbt_env.py -- dbt build
python machine_learning/train.py --write
python machine_learning/evaluate.py --thresholds --leave-one-city-out --write
python machine_learning/extremes.py --write   # fits the tails, then:
python dbt_analytics/dbt_env.py -- dbt build --select fact_anomaly_return_periods
streamlit run dashboard/app.py
```

No credentials are committed to this repository and none ever have been. Every
variable is documented in [`.env.example`](.env.example), and
[`config.py`](config.py) is the only module that reads the environment. `.env`
is git-ignored.

## Architecture

```
Open-Meteo Archive API
        |
        v
  ingestion/         planner, rate-limited client, gzipped raw archive,
        |            COPY loader, reconciliation
        v
  bronze  ->  silver  ->  gold          PostgreSQL 16 (local, Docker)
        dbt_analytics/                  dedup, unit assertions, UTC handling,
        |                               dimensions, facts, climatology, Z-scores
        v
  machine_learning/  feature matrix, forward-window label, baselines,
        |            chronological split, XGBoost, SHAP, scoring
        v
  serving/promote.py  gold marts + predictions only
        |
        v
  Neon PostgreSQL  ->  dashboard/  (Streamlit Community Cloud)
```

![dbt lineage: bronze source through silver staging and intermediate to gold marts](docs/images/lineage.png)

The diagram is generated from the dbt manifest by
[`render_lineage.py`](dbt_analytics/render_lineage.py), and a test fails if the
committed SVG is stale.

### Two Postgres environments

Switched by a single environment variable. No code branches on which one is in
use.

| | Host | Holds | Used for |
|---|---|---|---|
| **Local** | PostgreSQL 16 in Docker Compose | bronze, silver, gold | The 30-year backfill, all dbt iteration, model training |
| **Serving** | Neon free plan | gold marts and predictions only | What the public Streamlit dashboard reads |

The backfill runs locally and Neon receives finished gold marts only. Two
free-plan limits force this: 0.5 GB of storage, and 100 compute-hours per
month. Raw API responses are kept as gzipped files under a git-ignored
`data/raw/`, never as a per-row JSON column. Bronze and silver never leave the
local container.

## Data

Fifteen cities, chosen for climate diversity and hemisphere balance. The
registry is [`config/cities.yml`](config/cities.yml), which is also the fixture
the validation gate reads its dated events from.

| Grain | Coverage | Rows landed | Notes |
|---|---|---:|---|
| Daily | 1995-2026 target | 130 094 | 356 of 480 city-years; eleven cities complete |
| Hourly | Trailing 24 months | 280 728 | Complete for all fifteen cities |

Hourly is restricted to 24 months on purpose. The Storm Dynamics view is the
only consumer, and thirty years of hourly data would add roughly four million
rows for no analytical benefit.

Bronze projects to about 106 MB, inside the 250-300 MB budget in the proposal.
Landing speed is roughly 28 000 rows/s via `COPY`, which is never the
bottleneck; the API quota is.

## The model

[`machine_learning/train.py`](machine_learning/train.py) fits an XGBoost
classifier on the training split, tunes twelve hyperparameter combinations on
validation, and scores the result against baselines that were committed before
the model existed.

| test split | PR-AUC | vs base rate | **vs persistence** | Brier | F1 | mean predicted |
|---|---:|---:|---:|---:|---:|---:|
| no-skill reference | 0.1130 | 1.00x | 0.59x | 0.1043 | 0.2030 | |
| climatology baseline | 0.1250 | 1.11x | 0.65x | 0.1045 | 0.1597 | |
| persistence baseline | 0.1913 | 1.69x | 1.00x | 0.0976 | 0.3418 | |
| model, weighted *(as specified)* | 0.3268 | 2.89x | 1.71x | 0.2282 | 0.3610 | 0.483 |
| **model, unweighted** *(recommended)* | **0.2886** | **2.55x** | **1.51x** | **0.0946** | 0.3255 | 0.061 |

**Every score is reported against persistence, not only against the base
rate.** The base rate is the floor: average precision for a random ranker *is*
the positive rate, so a lift over it only says a predictor is not noise.
Persistence — "an anomaly next week if there was one this week" — is the number
that has to be beaten, and it is 1.69x higher. A method from a later phase
reported only against chance will look better than it is. The climatology
baseline is the cautionary case: 1.11x the base rate is barely distinguishable
from no out-of-sample skill at all, and only the second column shows that at a
glance.

The recommended model beats every baseline on ranking and on calibration, and
beats persistence in every city individually. On F1 it is behind persistence at
|Z| > 2.5 (0.3255 against 0.3418) and ahead at 2.0 and 3.0 — which is the
whole reason the next section exists.

### Which findings survive a different threshold

`|Z| > 2.5` is a choice, and everything inherits it: the label, two of the
twenty-seven features, all three baselines and every verdict above.
`evaluate.py --thresholds` re-runs the whole evaluation at 2.0, 2.5 and 3.0 —
refitting rather than rescoring, because moving the threshold moves the
features built from the flag as well as the target.

| \|Z\| | test base rate | model PR-AUC | persistence | vs persistence | model F1 | persistence F1 | cities beaten |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.0 | 0.2645 | 0.4918 | 0.3370 | 1.46x | 0.4647 | 0.4183 | 11/11 |
| **2.5** *(shipped)* | 0.1130 | 0.2886 | 0.1913 | 1.51x | 0.3255 | 0.3418 | 11/11 |
| 3.0 | 0.0437 | 0.1915 | 0.0964 | 1.99x | 0.2769 | 0.2524 | 10/11 |

Re-run at |Z| thresholds 2, 2.5, 3, the recommended model beats every baseline
on PR-AUC and Brier at all three, and on F1 at some but not all of them.

Ranking and calibration are properties of the model; the F1 verdict is a
property of where the line happens to sit, and reverses twice across the sweep.
That is exactly what this mode was built to expose, and it is why the claim
above is qualified rather than confident. The advantage over persistence on
ranking *grows* as the threshold rises, from 1.46x at 2.0 to 1.99x at 3.0, so
the model is not living on the easy half of the distribution. At |Z| > 3.0 it
beats persistence in ten of eleven cities rather than all eleven.

### Does it work on a city it has never seen

There is no city identifier among the twenty-seven features, so the model can
in principle score a city it was not trained on. `evaluate.py
--leave-one-city-out` refits once per scored city with that city removed from
the training and validation splits altogether, then scores it against **its own
persistence baseline** rather than its base rate, because the eleven cities
differ sixfold in base rate and a raw PR-AUC would sort them by climate.

| held out | base rate | in-sample PR-AUC | held-out PR-AUC | own persistence | lift | retained |
|---|---:|---:|---:|---:|---:|---:|
| cairo | 17.6% | 0.3645 | 0.3913 | 0.2374 | 1.65x | 1.07x |
| delhi | 4.8% | 0.2330 | 0.3266 | 0.1282 | 2.55x | 1.40x |
| lagos | 15.1% | 0.4388 | 0.4916 | 0.3464 | 1.42x | 1.12x |
| london | 10.4% | 0.4112 | 0.4285 | 0.2585 | 1.66x | 1.04x |
| moscow | 5.4% | 0.1507 | 0.1761 | 0.1257 | 1.40x | 1.17x |
| phoenix | 5.5% | 0.1011 | 0.1771 | 0.0607 | 2.92x | 1.75x |
| portland | 10.8% | 0.1858 | 0.1727 | 0.1107 | 1.56x | 0.93x |
| reykjavik | 3.8% | 0.1023 | 0.1015 | 0.0521 | 1.95x | 0.99x |
| sao_paulo | 10.9% | 0.2583 | 0.2883 | 0.1597 | 1.81x | 1.12x |
| singapore | 23.7% | 0.4187 | 0.4103 | 0.3013 | 1.36x | 0.98x |
| tokyo | 16.2% | 0.3384 | 0.3524 | 0.2188 | 1.61x | 1.04x |

Held out of training entirely, each of the 11 scored cities is still ranked
better by the model than by its own persistence baseline (11 of 11, median
PR-AUC 1.65x persistence and 107% of the same city's in-sample score), so the
model transfers to a city it has never seen.

The retention column deserves suspicion rather than celebration: a median of
1.07 means the held-out model usually scores *better* than the one that had
seen the city, and Phoenix reaches 1.75. With no city identifier and no
per-city capacity to spare there is little for the model to gain from a city's
own rows, and removing them changes which grid point the validation search
picks — so the spread is mostly the search, not transfer. What the column
establishes is the absence of a collapse, which is what the ticket asked.

Four of the fifteen registry cities are still absent — Sydney has eighteen
scored days and Auckland, Buenos Aires and Johannesburg three each — and each
is named with a reason in the `leave_one_city_out` block of `metrics.json`.

### How much to believe one week's answer

`evaluate.py --conformal` reports a distribution-free **prediction set** per
week — `{quiet}`, `{extreme}`, both when the evidence does not separate them,
or empty — calibrated on validation at a target of 90% coverage stated up
front.

Calibrated on validation for 90% coverage, split conformal delivers 83.2% on
the test period and decays year by year to 73.1% in 2026; adaptive conformal
holds 90.0% overall and within half a point of target in every year, paying for
it with sets that grow from 0.91 labels to 1.08.

Split conformal assumes the calibration and test periods are exchangeable, and
a base rate moving from 6.8% to 11.3% is the textbook violation: its guarantee
holds on paper and decays silently, from on-target in 2022 to sixteen points
short in 2026. Adaptive conformal adjusts its level online and holds. The set
size it pays with — and the share of "cannot say" answers, rising from 0% to
10.6% — is the drift measured in the units the guarantee is stated in.

Full metrics, feature importances, and the intended use of the model are in the
[model card](docs/model-card.md), which is generated from the run manifest and
cannot go stale.

### How unusual, in years

A Z-score answers *is this unusual*. A reader wants *how unusual*, and the unit
for that is time. `machine_learning/extremes.py` fits a generalised Pareto to
each city's declustered tail — a five-day heatwave is one event, and on this
data the mean cluster runs 1.9 to 2.6 days, so counting them singly would halve
every return period — and `fact_anomaly_return_periods` turns those parameters
into a per-day answer the Anomaly Map can size markers by.

**The tail is not pinned down, and the map says so rather than rounding.** Seven
of the eleven fitted cities have a shape parameter whose bootstrap interval
crosses zero: the data cannot say whether their tail is bounded. The
consequence is quantitative and sharp. Recomputing a return period at the ends
of that interval, the median band across cities is a factor of 1.8 at three
sigma, 9.6 at 3.8, and 134 at 4.2. Phoenix at four sigma reads 28 years and its
interval puts it between 9 years and 152 000.

So the map quotes a number where the band is inside one order of magnitude and
a floor — "at least a 1-in-15-year day" — where it is not, taking the floor from
the heaviest tail in the interval, so the error runs toward *understating*
rarity. Portland's 2021 heat dome reads "at least 1-in-15-years", which is
conservative and defensible; the alternative is a headline number off a
thirty-year record that gets quoted onward and cannot be walked back.

### The tails are moving, and mostly not widening

Fitting a time trend to the tail is where this ticket nearly went wrong twice.

Folding both tails into `abs(Z)` and fitting one trend called Phoenix and
Reykjavík significantly **narrowing** — cities whose warm share of exceedances
nearly doubled. The warm share rises in every one of the eleven cities, from
7.5% to 61% in Lagos and 19% to 53% in Singapore: a single trend was being
fitted to a mixture whose composition inverts across the record, and it
reported that inversion as a change in width.

Splitting by direction fixed that and left a second confound. A threshold held
still while the distribution slides under it turns a *location* drift into an
apparent change in width — on a simulated record with a pure 0.02 σ/year drift
and a rigorously constant variance, the fixed-threshold fit called the cold
tail narrowing at p = 0.002. Nothing had narrowed. Fitting the threshold as a
line in time recovers the injected drift to 0.0202 and reports no width trend
in either direction.

With both separated, the answer is smaller and more honest than either wrong
version: **every city's warm tail is moving** (up to +0.048 σ/year in Lagos)
and **its cold tail is moving toward the mean**, while only four of twenty-two
directional width trends survive at p < 0.05. The fixed-threshold version
claimed seven. Both tests are in `tests/test_extremes.py`, one asserting the
current behaviour and one asserting that the old method got it wrong, so the
pair says exactly what changed if either is reverted.

## Known limitations

**The validation gate does not pass, and it should not.**
`tests/test_validation_gate.py` checks seven dated extreme-weather events
against the warehouse, **against both climatology definitions**. Five are now
checkable; Sydney and Buenos Aires still wait on the backfill, which costs
about 26 000 weighted API calls at the daily grain against a free-tier
allowance of 10 000 a day. A missing city skips with a reason and does not
pass, and a separate test fails while any event is unverifiable, so the gate
cannot close on a green suite that checked nothing.

Of the five checkable, four flag under both definitions and Tokyo flags under
neither. Not one event changes verdict between them, which is the finding
DBT-13 exists to produce.

**The model is under-confident on the test period.** It predicts 0.061 where
0.113 occurs. This is the non-stationary base rate: positives run at 4.87% in
the training period and 11.30% in the test period, so a model fitted on the
early record is calibrated to a world that has since warmed. Ranking is sound
and observed risk rises monotonically across the deciles, but the level is not.
The Risk Horizon view therefore shows rank bands rather than raw probabilities.
Detrending the climatology does not fix it — see below — and neither does
prior-shift correction, whose EM estimator returns a target prior of 0.299
against an observed 0.115. Isotonic regression fitted on validation does help,
halving the calibration error for 4% of the ranking, and is recorded in
`model.calibration` for BI-08 to decide what to do with.

**The flag now knows how good its own baseline is.** A Z-score divides by a
sigma that was *estimated*, and treating it as known made the standardised
departure a t-statistic read against a normal table — so a city with a thin
baseline over-flagged by construction. Sydney, on five reference observations,
flagged 16.7% of its eighteen scored days against 1.0-2.0% everywhere else.
Each day is now judged against a Student-t critical value on its own degrees of
freedom: 2.513 at a complete city's 459 observations, 2.96 at fifteen, 62.8 at
two. Sydney flags 1 of 18 instead of 3, which is no longer distinguishable from
the complete-city rate.

**Three events do not flag, and the threshold was not lowered to make them.**
Phoenix in July 2023 peaks at Z = +1.97, Delhi on 29 May 2024 at Z = +2.22, and
Tokyo on 23 July 2018 at Z = +2.23 — the last of these under the detrended
climatology too, at +2.22, so it is not an artefact of which baseline is used.
All three are national records that are not 2.5-sigma days in their own grid
cell: Phoenix's was a duration a single-day Z cannot express, Delhi's was the
second-warmest such day in thirty-two years, and Kumagaya's 41.1 °C was 45 km
from the Tokyo cell, which reached +5.6 °C against a July sigma of about 2.5.
Each investigation is recorded as an executable test, including a guard that
fires if Phoenix ever does clear the threshold.

**Two climatologies exist and one is shipped.** `is_anomaly` answers *unusual
for the record*; `is_anomaly_detrended` answers *unusual for this era*, from a
baseline with a per-city warming trend removed, fitted only on prior years. The
first is what every view and the model use. Detrending removes half the
correlation between year and Z but only 5% of the label's base-rate drift
across the split — the drift lives in the tail and the trend lives in the
centre — and it would erase the signal the Climate Matrix exists to draw. The
decision, and the measurements behind it, are in
[`docs/proposal.md`](docs/proposal.md) §5.3.

**This model does not compete with ECMWF or GFS.** It demonstrates a complete
feature-engineering, training, and evaluation loop against warehouse data.

## Dashboard

Four views, all reading gold marts from Neon.

- **Global Anomaly Map.** Where it was abnormally hot or cold on a chosen day.
  Fifteen points sized by distance from each city's own seasonal normal and
  coloured by direction.
- **Climate Matrix.** One cell per city-year, shaded by how many days that year
  ran more than 2.5 sigma from that city's seasonal normal.
- **Storm Dynamics.** One point per city-day: the largest 24-hour pressure
  change against the strongest gust that accompanied it. The only view with no
  gaps, because the hourly mart is complete.
- **Risk Horizon.** Fifteen cities across seven days, showing where the model
  is flagging elevated risk.

Colours, sizes, and thresholds live in [`dashboard/theme.py`](dashboard/theme.py)
and are under test, including colour-blind separation and contrast against both
light and dark surfaces.

## Testing

```bash
pip install -r requirements-dev.txt
pytest                                    # full suite, needs a local warehouse
pytest tests/test_dashboard.py            # dashboard only, no database
python dbt_analytics/dbt_env.py -- dbt test
```

The suite covers the ingestion client and its failure policy, idempotency and
reconciliation, every dbt model, leakage in the feature matrix and the split,
model behaviour, and the dashboard's colour and layout decisions. It also
checks the deployment manifest: a test walks every import under `dashboard/`
and fails if one is missing from `requirements.txt`.

## Layout

```
ingestion/          API client, planner, raw archive, bronze loader, reconciliation
dbt_analytics/      staging, intermediate, and gold marts, plus macros and tests
machine_learning/   features, labels, baselines, split, training, SHAP, scoring,
                    extreme value fits
serving/            promotion of gold marts and predictions to Neon
dashboard/          Streamlit app, theme, database access, four views
config/             city registry
tests/              pytest suite and deployment checks
docs/               proposal, model card, build log, figures
```

## Further reading

- [Build log](docs/build-log.md). The full engineering write-up: every design
  decision, the measurements behind it, and the defects found along the way.
- [Proposal](docs/proposal.md). Scope, delivery plan, and what was ruled out.
- [Model card](docs/model-card.md). Intended use, metrics, and limitations.

## Licence

[MIT](LICENSE)
