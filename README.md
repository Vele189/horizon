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
python machine_learning/evaluate.py --leave-one-city-out --write
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
| Daily | 1994-2025 target | 173 505 | 164 of 480 city-years ingested |
| Hourly | Trailing 24 months | 263 160 | Complete for all fifteen cities |

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

| test split | PR-AUC | lift | Brier | mean predicted |
|---|---:|---:|---:|---:|
| no-skill reference | 0.1358 | 1.00x | 0.12398 | |
| climatology baseline | 0.1514 | 1.12x | 0.12454 | |
| persistence baseline | 0.2293 | 1.69x | 0.11466 | |
| model, weighted *(as specified)* | 0.3206 | 2.36x | 0.21993 | 0.470 |
| **model, unweighted** *(recommended)* | **0.3612** | **2.66x** | **0.10870** | 0.072 |

Both variants beat both baselines on ranking. The recommended one beats
persistence by 58% on PR-AUC, beats it on Brier, and beats it in every city
individually, which a test asserts separately because a pooled win can be one
city carrying the rest. On F1 at the validation-chosen threshold the two are
level to a thousandth (0.3797 against 0.3807), which says more about F1 than
about either predictor: it collapses the whole curve to one point, and that
point is where persistence is strongest.

### Does it work on a city it has never seen

There is no city identifier among the twenty-seven features, so the model can
in principle score a city it was not trained on. `evaluate.py
--leave-one-city-out` refits once per scored city with that city removed from
the training and validation splits altogether, then scores it against **its own
persistence baseline** rather than its base rate, because the five cities differ
fourfold in base rate and a raw PR-AUC would sort them by climate.

| held out | base rate | in-sample PR-AUC | held-out PR-AUC | own persistence | lift |
|---|---:|---:|---:|---:|---:|
| cairo | 17.8% | 0.3698 | 0.3785 | 0.2421 | 1.56x |
| delhi | 5.2% | 0.3289 | 0.3403 | 0.1178 | 2.89x |
| lagos | 15.6% | 0.4564 | 0.4962 | 0.3362 | 1.48x |
| phoenix | 5.7% | 0.2236 | 0.2751 | 0.0651 | 4.23x |
| singapore | 23.7% | 0.4227 | 0.3363 | 0.3013 | 1.12x |

Held out of training entirely, each of the 5 scored cities is still ranked
better by the model than by its own persistence baseline (5 of 5, median PR-AUC
1.56x persistence and 103% of the same city's in-sample score), so the model
transfers to a city it has never seen.

Singapore is the only city that loses anything by being unseen, at 80% of its
in-sample score; the other four are at or above theirs, which is what a model
with no city identifier and no per-city capacity to spare should do. Read it
against the sample it rests on: five cities, all hot, and the ten still
backfilling are named with a reason in the `leave_one_city_out` block of
`metrics.json` rather than left out of the table.

Full metrics, feature importances, and the intended use of the model are in the
[model card](docs/model-card.md), which is generated from the run manifest and
cannot go stale.

## Known limitations

**The validation gate does not pass, and it should not.**
`tests/test_validation_gate.py` checks seven dated extreme-weather events
against the warehouse. None can currently be verified because their cities have
not backfilled: the daily grain costs about 26 000 weighted API calls against a
free-tier allowance of 10 000 a day. A missing city skips with a reason and
does not pass, and a separate test fails while any event is unverifiable, so
the gate cannot close on a green suite that checked nothing.

**The model is under-confident on the test period.** It predicts 0.072 where
0.136 occurs. This is the non-stationary base rate: positives run at 5.43% in
the training period and 13.58% in the test period, so a model fitted on the
early record is calibrated to a world that has since warmed. Ranking is sound
and observed risk rises monotonically across the deciles, but the level is not.
The Risk Horizon view therefore shows rank bands rather than raw probabilities.

**Twenty-six rows from two barely-backfilled cities move the headline by 3%.**
London and Reykjavík have eighteen scored days each, and on a baseline that
short they flag 56% and 89% of those days against 1.3-2.0% in every complete
city. The twelve and fourteen rows that reach the training split are therefore
almost all positives and almost all spurious, and removing them takes test
PR-AUC from 0.3612 to 0.3722. The numbers above are the ones the pipeline
actually produces and have not been improved by choosing the training set;
DBT-14 is the ticket for a flag that knows how thin its own baseline is.

**Two events do not flag, and the threshold was not lowered to make them.**
Phoenix in July 2023 peaks at Z = +1.97 and Delhi on 29 May 2024 at Z = +2.22.
Both investigations are recorded as executable tests, including a guard that
fires if Phoenix ever does clear the threshold.

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
