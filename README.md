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
| no-skill reference | 0.1150 | 1.00x | 0.60x | 0.1060 | 0.2062 | |
| climatology baseline | 0.1145 | 1.00x | 0.60x | 0.1062 | 0.2062 | |
| persistence baseline | 0.1915 | 1.67x | 1.00x | 0.0992 | 0.3398 | |
| model, weighted *(as specified)* | 0.3026 | 2.63x | 1.58x | 0.2077 | 0.3461 | 0.444 |
| **model, unweighted** *(recommended)* | **0.3313** | **2.88x** | **1.73x** | **0.0934** | **0.3578** | 0.066 |

**Every score is reported against persistence, not only against the base
rate.** The base rate is the floor: average precision for a random ranker *is*
the positive rate, so a lift over it only says a predictor is not noise.
Persistence — "an anomaly next week if there was one this week" — is the number
that has to be beaten, and it is 1.67x higher. A method from a later phase
reported only against chance will look better than it is. The climatology
baseline is the cautionary case: it scores 1.00x the base rate on test, which
is to say it has no out-of-sample skill at all, and only the second column
makes that visible at a glance.

The recommended model beats every baseline on PR-AUC, on Brier and on F1, and
beats persistence in every city individually, which a test asserts separately
because a pooled win can be one city carrying the rest.

### Which findings survive a different threshold

`|Z| > 2.5` is a choice, and everything inherits it: the label, two of the
twenty-seven features, all three baselines and every verdict above.
`evaluate.py --thresholds` re-runs the whole evaluation at 2.0, 2.5 and 3.0 —
refitting rather than rescoring, because moving the threshold moves the
features built from the flag as well as the target.

| \|Z\| | test base rate | model PR-AUC | persistence | vs persistence | model F1 | persistence F1 | cities beaten |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.0 | 0.2667 | 0.4851 | 0.3398 | 1.43x | 0.4643 | 0.4212 | 11/11 |
| **2.5** *(shipped)* | 0.1150 | 0.3313 | 0.1915 | 1.73x | 0.3578 | 0.3398 | 11/11 |
| 3.0 | 0.0445 | 0.1937 | 0.0995 | 1.95x | 0.2736 | 0.2577 | 10/11 |

Re-run at |Z| thresholds 2, 2.5, 3, the recommended model beats every baseline
on PR-AUC, Brier and F1 at all three.

The advantage over persistence *grows* as the threshold rises, from 1.43x at
2.0 to 1.95x at 3.0, so the model is not living on the easy half of the
distribution. The one qualification is per-city: at |Z| > 3.0 it beats
persistence in ten of eleven cities rather than all eleven, at a base rate of
4.5% where a single city's ranking is thin.

### Does it work on a city it has never seen

There is no city identifier among the twenty-seven features, so the model can
in principle score a city it was not trained on. `evaluate.py
--leave-one-city-out` refits once per scored city with that city removed from
the training and validation splits altogether, then scores it against **its own
persistence baseline** rather than its base rate, because the eleven cities
differ fivefold in base rate and a raw PR-AUC would sort them by climate.

| held out | base rate | in-sample PR-AUC | held-out PR-AUC | own persistence | lift | retained |
|---|---:|---:|---:|---:|---:|---:|
| cairo | 17.8% | 0.3884 | 0.3894 | 0.2420 | 1.61x | 1.00x |
| delhi | 5.2% | 0.2994 | 0.2896 | 0.1178 | 2.46x | 0.97x |
| lagos | 15.7% | 0.4858 | 0.5023 | 0.3399 | 1.48x | 1.03x |
| london | 10.4% | 0.4388 | 0.3809 | 0.2585 | 1.47x | 0.87x |
| moscow | 5.4% | 0.2059 | 0.1920 | 0.1303 | 1.47x | 0.93x |
| phoenix | 5.6% | 0.1852 | 0.2210 | 0.0650 | 3.40x | 1.19x |
| portland | 10.8% | 0.1847 | 0.1824 | 0.1107 | 1.65x | 0.99x |
| reykjavik | 4.2% | 0.1220 | 0.1241 | 0.0527 | 2.36x | 1.02x |
| sao_paulo | 10.9% | 0.2719 | 0.2521 | 0.1597 | 1.58x | 0.93x |
| singapore | 23.7% | 0.4223 | 0.3863 | 0.3013 | 1.28x | 0.91x |
| tokyo | 16.6% | 0.3719 | 0.3640 | 0.2174 | 1.67x | 0.98x |

Held out of training entirely, each of the 11 scored cities is still ranked
better by the model than by its own persistence baseline (11 of 11, median
PR-AUC 1.61x persistence and 98% of the same city's in-sample score), so the
model transfers to a city it has never seen.

London loses the most by being unseen, at 87% of its in-sample score, and
Phoenix gains the most at 119%; the median is 98%, which is what a model with
no city identifier and no per-city capacity to spare should do. Four of the
fifteen registry cities are still absent — Sydney has eighteen scored days and
Auckland, Buenos Aires and Johannesburg three each — and each is named with a
reason in the `leave_one_city_out` block of `metrics.json` rather than left out
of the table.

Full metrics, feature importances, and the intended use of the model are in the
[model card](docs/model-card.md), which is generated from the run manifest and
cannot go stale.

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

**The model is under-confident on the test period.** It predicts 0.066 where
0.115 occurs. This is the non-stationary base rate: positives run at 5.02% in
the training period and 11.50% in the test period, so a model fitted on the
early record is calibrated to a world that has since warmed. Ranking is sound
and observed risk rises monotonically across the deciles, but the level is not.
The Risk Horizon view therefore shows rank bands rather than raw probabilities.
Detrending the climatology does not fix it — see below — and neither does
prior-shift correction, whose EM estimator returns a target prior of 0.299
against an observed 0.115. Isotonic regression fitted on validation does help,
halving the calibration error for 4% of the ranking, and is recorded in
`model.calibration` for BI-08 to decide what to do with.

**One city's baseline is too thin to trust.** Sydney has eighteen scored days,
and on a five-observation baseline it flags 16.7% of them against 1.0-2.0% in
every complete city, with sd(Z) = 1.67 where the others sit within 0.02 of one.
It is excluded from every per-city table for want of test rows, so it moves no
headline figure, but it is the remaining instance of the defect DBT-14 exists
to fix: the flag treats sigma as known when it is an estimate, and at small
baselines a noisy one.

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
machine_learning/   features, labels, baselines, split, training, SHAP, scoring
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
