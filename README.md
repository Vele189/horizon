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
| no-skill reference | 0.1358 | 1.00x | 0.12386 | |
| climatology baseline | 0.1514 | 1.12x | 0.12454 | |
| persistence baseline | 0.2293 | 1.69x | 0.11462 | |
| model, weighted *(as specified)* | 0.3303 | 2.43x | 0.22529 | 0.478 |
| **model, unweighted** *(recommended)* | **0.3494** | **2.57x** | **0.11019** | 0.070 |

Both variants beat both baselines. The recommended one beats persistence by 52%
on PR-AUC, and beats it in every city individually, which a test asserts
separately because a pooled win can be one city carrying the rest.

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

**The model is under-confident on the test period.** It predicts 0.070 where
0.136 occurs. This is the non-stationary base rate: positives run at 5.51% in
the training period and 13.58% in the test period, so a model fitted on the
early record is calibrated to a world that has since warmed. Ranking is sound
and observed risk rises monotonically across the deciles, but the level is not.
The Risk Horizon view therefore shows rank bands rather than raw probabilities.

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
