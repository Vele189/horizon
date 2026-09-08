# Climate Volatility & Risk Engine
### Project Proposal and Delivery Plan

**Version:** 2.0
**Date:** September 2026
**Type:** Portfolio / demonstration project
**Duration:** 15 working days at ~8 hrs/day (~120 hrs), plus an optional stretch week
**Warehouse:** Neon serverless Postgres (hosted) + Docker Postgres (local dev)
**Owner:** [Your name]

---

## 1. Executive Summary

This project builds an end-to-end analytics platform that ingests three decades of global weather observations, transforms them into a governed dimensional warehouse, and uses that warehouse to identify and forecast extreme temperature anomalies across fifteen major cities.

The deliverable is not a weather forecast. Operational meteorology is dominated by physics-based numerical weather prediction models that a gradient-boosted tree cannot compete with. The deliverable is a **complete, reproducible data platform** (ingestion, warehouse, transformation, feature store, model, and dashboard) assembled the way a working data team would assemble it, with the engineering decisions documented and defensible.

The intended audience is a hiring manager or technical interviewer. The project succeeds if a reviewer can clone the repository, run one command, and see the full pipeline execute; and if the accompanying documentation demonstrates that the author understands *why* each layer exists.

---

## 2. Objectives

| # | Objective | Measure of success |
|---|---|---|
| O1 | Build a fault-tolerant ingestion layer | Pipeline recovers from API rate limits, timeouts, and partial failures without manual intervention; re-running is idempotent |
| O2 | Establish a governed warehouse with medallion architecture | Three schemas with enforced contracts; every model tested; lineage documented |
| O3 | Produce a defensible statistical definition of "extreme weather" | Z-score anomaly flag computed against a genuine long-term climatological baseline |
| O4 | Train a classifier that beats a naive baseline | PR-AUC materially above both a persistence baseline and a climatology baseline, on a time-based holdout |
| O5 | Deliver an executive-legible dashboard | Publicly accessible URL; four views; loads in under five seconds |
| O6 | Make the whole thing reproducible | `make run` executes ingestion -> transform -> predict end-to-end on a clean machine |

---

## 3. Scope

### In scope
- Historical daily and hourly observations for 15 cities, 1995-present
- Batch ingestion from a single primary API, with a documented fallback
- dbt transformation across bronze / silver / gold layers
- One supervised classification model with a benchmarked baseline
- One published dashboard
- Local orchestration via a Python entrypoint, containerised with Docker Compose

### Out of scope
- Real-time or streaming ingestion
- Cloud deployment, Kubernetes, or managed orchestration (Airflow/Dagster)
- Physics-based modelling or reanalysis of raw sensor data
- Model retraining automation or drift monitoring
- Authentication, multi-tenancy, or any production hardening

Anything in the out-of-scope list is a *stretch goal*, documented in the README as "what I'd build next." Naming these deliberately is itself a signal of engineering judgment.

---

## 4. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  SOURCE                                                          │
│  Open-Meteo Historical Archive API  (ERA5 reanalysis)            │
└────────────────────────────┬────────────────────────────────────┘
                             │  Python · requests · tenacity
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  BRONZE  ·  bronze_raw schema                                    │
│  Raw payloads, append-only, ingestion metadata preserved         │
└────────────────────────────┬────────────────────────────────────┘
                             │  dbt
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  SILVER  ·  silver_staging schema                                │
│  Deduplicated · unit-normalised · UTC-cast · typed · tested      │
└────────────────────────────┬────────────────────────────────────┘
                             │  dbt
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  GOLD  ·  gold_marts schema                                      │
│  Star schema + rolling climatology + Z-score anomaly flags       │
└──────────────┬──────────────────────────────┬───────────────────┘
               │                              │
               ▼                              ▼
┌──────────────────────────┐   ┌──────────────────────────────────┐
│  ML PIPELINE             │   │  BI LAYER                        │
│  features -> train ->      │   │  Tableau (extract) or Streamlit  │
│  predict                 │   │  4 dashboard views               │
│  writes fact_predictions ├──▶│                                  │
└──────────────────────────┘   └──────────────────────────────────┘
```

**Stack:** Python 3.11 · Neon serverless Postgres (serving) · PostgreSQL 16 via Docker (local dev) · dbt-postgres · scikit-learn / XGBoost · Streamlit + Plotly · Docker Compose

---

## 5. Assumptions and Data Design

### 5.1 Source selection

Open-Meteo's historical archive endpoint is the primary source: it is free, requires no API key, is backed by ERA5 reanalysis, and reaches back to 1940. NOAA GHCN-Daily is the documented fallback if station-level rather than gridded data is preferred.

> **Correction to the original draft:** the draft proposed pulling 10 years of history while computing a *30-year* rolling baseline. This is not possible: a 30-year climatological normal requires 30 years of data. **Resolution:** ingest daily observations from 1995 to present (≈30 years), which yields the baseline; ingest hourly observations only for the trailing 24 months, which is all the storm-dynamics view needs.

### 5.2 Volume estimate

| Grain | Cities | Span | Approx. rows |
|---|---|---|---|
| Daily | 15 | 30 years | ~164,000 |
| Hourly | 15 | 2 years | ~263,000 |

Under half a million rows total. **Do not over-engineer for scale you do not have**, and say so in the README, because knowing when *not* to reach for Spark is a stronger signal than reaching for it.

### 5.4 Warehouse topology (decided)

Two environments, switched by a single `DATABASE_URL` environment variable:

| Environment | Host | Purpose |
|---|---|---|
| **Local** | PostgreSQL 16 in Docker Compose | All development: backfill, dbt iteration, model training. Fast, free, offline-capable. |
| **Serving** | Neon free plan | Holds gold marts and predictions only. This is what the public Streamlit dashboard reads. |

**Why Neon rather than Supabase.** The dashboard must be reachable from Streamlit Community Cloud, so a laptop-local database is not an option for serving. Supabase's free tier pauses projects after seven days of inactivity and requires a manual restore from the dashboard. For a portfolio link that a recruiter may open weeks after you send it, that is disqualifying. Neon instead scales compute to zero after five minutes idle and reactivates automatically within a few hundred milliseconds on the next query. Nothing to babysit, no keep-alive cron job to maintain.

**Two constraints that shape the design:**

1. **0.5 GB storage per project.** Do *not* store raw API JSON as a per-row column, because that alone will exhaust the budget. Keep raw responses as gzipped files under a git-ignored `data/raw/`, and land only parsed columns plus `ingested_at` and `source_url` in bronze. Expect ~250-300 MB across all layers.
2. **100 compute-hours per month.** Run the 30-year backfill and all dbt iteration against **local Docker**, then push only the gold marts to Neon. Hammering a serverless database with a multi-hour backfill is both slow and wasteful of the free allowance.

Having dev and serving environments separated by configuration, rather than hardcoded, is itself worth a paragraph in the README.

### 5.3 Data quality issues to handle explicitly

Reanalysis data is cleaner than the draft implied. Rather than manufacture messiness, the pipeline should handle the problems that genuinely occur and document them:

- Gaps where a requested date range exceeds archive availability
- Sub-daily nulls for variables not available at every grid point
- Timezone handling: the API returns local time by default; all timestamps must be explicitly requested and stored as UTC
- Leap-day handling in day-of-year climatology joins (a real and commonly-botched edge case)
- Unit consistency across variables requested in separate calls

If a deliberately messy ingestion exercise is wanted, a small synthetic corruption layer can be added to bronze and the cleaning logic demonstrated against it, but it should be labelled as synthetic, not passed off as source behaviour.

#### Which question the anomaly flag answers (decided, DBT-13)

An anomaly is a departure from a normal, and there is more than one defensible normal. The warehouse computes two and the product ships one, so the choice is recorded here rather than left to whichever column a query happened to select.

**Shipped: *unusual for the record*.** `is_anomaly`, from a leave-one-year-out baseline over every year of that city's history. A hot day in 2024 is measured against the whole 1995-2026 record.

**Built, measured, and not shipped: *unusual for this era*.** `is_anomaly_detrended`, from the same baseline with a per-city, per-day linear trend in year removed and the result referenced to the year being scored. The trend is fitted only on years before that one, on an expanding window, because a label that can see forward would be a worse defect than the drift it was built to fix.

They are different products, not two implementations of one. The reasons for shipping the first:

1. **Detrending does not fix what it was proposed to fix.** The seven-day label's base rate rises 2.29x from the training period to the test period. Detrended, it rises 2.22x. Five per cent of the drift goes. The drift lives in the tail and the trend lives in the centre: a slope of 0.3 °C/decade moves the baseline by about a tenth of a sigma, which halves the correlation between year and Z and barely touches the rate at which days clear 2.5 sigma.
2. **No documented event changes verdict.** The validation gate runs against both flags. Every checkable event moves by 0.01 to 0.19 sigma, in the direction the trend predicts, and not one crosses the threshold. There is no evidence from the gate for preferring either, which removes the argument that would have overridden point 1.
3. **The Climate Matrix exists to draw how anomaly counts move across thirty years.** Detrending removes, by construction, the signal that view is for. A product cannot ship a flag that erases one of its own four views.

The choice is uniform across the four views rather than split per view. A split would be legitimate - the Risk Horizon is arguably asking an operational question about now - but it would mean two flags with the same name meaning different things in different tabs, for a difference DBT-12 measured at five per cent, and the reader would have to carry which was which.

`is_anomaly_detrended` remains in `fact_weather_anomalies` in full, with its fitted slope and that slope's standard error beside it, because the measurement is worth keeping and because the decision should be re-checkable rather than re-argued.

---

## 6. Workstreams

### Workstream 1: Ingestion (Bronze)

**Goal:** A reliable, resumable extractor that lands raw observations in Postgres.

Tasks:
1. Define the 15 target cities in a version-controlled config file (`cities.yml`) with name, country, latitude, longitude, and elevation. Config, not hardcoded constants.
2. Build an API client with `tenacity` for exponential backoff, respect for HTTP 429, and a request timeout.
3. Chunk requests by city and year to stay within response-size limits.
4. Land responses in `bronze_raw.observations_daily` and `bronze_raw.observations_hourly`, preserving the raw JSON alongside parsed columns, plus `ingested_at` and `source_url`.
5. Make ingestion idempotent: re-running for the same city-date range must not duplicate rows.
6. Stand up `docker-compose.yml` with Postgres 16 and a persistent volume for local development.
7. Create the Neon project and confirm connectivity. Both targets read from `DATABASE_URL`; never hardcode a connection string.

**Acceptance criteria:** Ingestion runs clean from an empty database. Killing it mid-run and restarting produces identical row counts. Row counts match expected values per city-year.

---

### Workstream 2: Transformation (Silver & Gold)

**Goal:** Turn raw landings into a tested, documented, query-optimised star schema.

#### Silver
- **Deduplication.** PostgreSQL does **not** support the `QUALIFY` clause proposed in the original draft; that is Snowflake, BigQuery, and DuckDB syntax. The Postgres form is:

  ```sql
  select * from (
    select *,
           row_number() over (
             partition by city_id, observation_time
             order by ingested_at desc
           ) as rn
    from {{ ref('bronze_observations') }}
  ) ranked
  where rn = 1
  ```

- **Unit standardisation.** Request metric units at the API layer where possible; enforce and assert them in dbt rather than converting downstream.
- **Temporal normalisation.** Cast all timestamps to `timestamptz` in UTC. Store the city's IANA timezone on the dimension so local-time views remain possible.

#### Gold: star schema
| Model | Grain | Contents |
|---|---|---|
| `fact_weather_observations` | one row per city per day | temp min/max/mean, precipitation, wind speed/gust, surface pressure, humidity, FKs |
| `fact_weather_hourly` | one row per city per hour | wind speed, pressure, temperature, trailing 24 months only |
| `dim_cities` | one row per city | name, country, region, lat/lon, elevation, timezone, population |
| `dim_date` | one row per day | calendar attributes, meteorological season (hemisphere-aware) |
| `fact_ml_predictions` | one row per city per forecast date | risk score, label, model version, scored_at |

> **Note:** `dim_cities` replaces the draft's `dim_weather_stations`. Open-Meteo returns gridded reanalysis, not station observations, so using "station" language would misrepresent the data source, and an interviewer who knows the domain will notice.

#### Climatology and anomaly detection
- Compute, per city and per day-of-year, the mean (μ) and standard deviation (σ) across a 30-year reference window, smoothed with a ±7-day window to avoid noisy single-day normals.
- Flag anomalies where:

  Z = (T_observed - μ_dayofyear) / σ_dayofyear, with |Z| > 2.5

- **Guard against leakage:** the climatology used to label any given observation must be computed from a reference period that excludes the observation's own year. Build this in as a dbt model parameter from the start, not as a patch after the model trains suspiciously well.

**Acceptance criteria:** `dbt build` passes with zero failures. Every gold model has a primary key test, not-null tests on keys, and a description. `dbt docs generate` produces a full lineage graph.

---

### Workstream 3: Machine Learning

**Goal:** A classifier that predicts whether a city will experience an extreme temperature anomaly within a forward 7-day window and, critically, an honest evaluation of whether it works.

#### Target definition
Binary label: does |Z| > 2.5 occur on any day in the forward 7-day window from date *t*? Expected positive rate is roughly 3-6% of city-days: imbalanced but tractable.

#### Features
- Lagged temperature and pressure at t-1, t-3, t-7, t-14
- Rolling 7-day and 30-day mean and variance
- Trailing Z-score (current anomaly state)
- Pressure tendency (24h and 72h change)
- Day-of-year encoded cyclically (sin/cos), latitude, elevation
- Count of anomaly days in the trailing 30

#### Evaluation: the part that matters
1. **Split chronologically.** Train on 1995-2018, validate 2019-2021, test 2022-present. A random split leaks future information through rolling features and will produce a meaningless score.
2. **Benchmark against two baselines** before reporting any model metric:
   - *Persistence:* predict "anomaly next week" if an anomaly occurred this week.
   - *Climatology:* predict the historical base rate for that city and week of year.
3. Report **PR-AUC, F1, Brier score, and a calibration curve.** Accuracy is meaningless at a 4% base rate and reporting it would undercut the whole project.
4. Handle imbalance with `scale_pos_weight` rather than resampling, which distorts calibration.
5. Produce SHAP feature importances for the dashboard and README.

> **Be explicit in the README:** this model is not competing with ECMWF or GFS. Its purpose is to demonstrate a complete feature-engineering, training, and evaluation loop against warehouse data. Stating this plainly reads as competence; overclaiming reads as inexperience.

**Acceptance criteria:** Model beats both baselines on test-set PR-AUC. Training is reproducible from a fixed seed. Model artefact is versioned via `joblib` with a metadata sidecar recording training window, feature list, and metrics.

---

### Workstream 4: Business Intelligence

**Goal:** A publicly accessible dashboard that a non-technical reader can interpret in under a minute.

#### ⚠ Critical tooling constraint

The original plan assumes Tableau reads live from PostgreSQL. For a portfolio project this will not work as described. Tableau Public, the free tier and the only one that gives a shareable public URL, cannot open a live connection to a private database or server. It reads flat files, Google Sheets, and web data connectors only. It also cannot save workbooks to disk, which conflicts with the draft's plan to commit `climate_briefing.twbx` to the repository.

Three viable options:

| Option | Cost | Live DB connection | Shareable URL | Verdict |
|---|---|---|---|---|
| **Tableau Public** | Free | No, CSV/extract export required | Yes | Workable; add a `export_for_tableau.py` step that writes gold marts to CSV |
| **Tableau Desktop** | Paid (free 14-day trial) | Yes | No public hosting | Only if a licence is already available |
| **Streamlit + Plotly** | Free | Yes, direct to Postgres | Yes, via Streamlit Community Cloud | **Recommended.** Live data, free hosting, and demonstrates Python skills alongside the pipeline |

**Decided: Streamlit primary.** It preserves the live warehouse connection the architecture is built around: the app reads directly from Neon, so the architecture diagram is honest rather than aspirational. A static Tableau Public version built from exported CSVs is a stretch goal, worth the extra day only if a target job description names Tableau explicitly.

**Deployment note:** Streamlit Community Cloud requires a public GitHub repository. Set up `.env`, `.gitignore`, and Streamlit secrets on Day 1, before the first commit, because a credential that enters git history is difficult to remove and will be visible to everyone who reviews the repo.

#### Dashboard views

| View | Visualisation | Question it answers |
|---|---|---|
| 1. Global Anomaly Map | Dark-basemap point map, size = anomaly magnitude, colour = warm/cold divergence | Where is it abnormally hot or cold right now? |
| 2. Climate Matrix | Heatmap, years on X, cities on Y, cell = annual anomaly-day count | Which cities are seeing more extremes over time? |
| 3. Storm Dynamics | Scatter, pressure drop vs. peak wind gust, coloured by city | Do pressure crashes track with wind extremes? |
| 4. Risk Horizon | 7-day forward grid, cell colour = predicted risk score | Which cities are flagged for the coming week? |

Two notes on the draft's visual spec: "glowing neon" and "flashing" effects are not available in Tableau and read as decoration rather than analysis. Use a diverging colour scale (blue-grey-red) which is both accessible and standard in climate visualisation. Prioritise legibility over spectacle: the audience is an analyst, not a sci-fi art director.

**Acceptance criteria:** Dashboard is live at a public URL. All four views render. A colour-blind-safe palette is used. Each view has a one-line plain-English caption.

---

### Workstream 5: Orchestration & Repository

**Goal:** Turn five components into one system a stranger can run.

- `run_pipeline.py`, the sequential entrypoint: ingest -> `dbt build` -> `predict.py` -> export
- Structured logging with timestamps and row counts at each stage; fail loudly
- `Makefile` with `make setup`, `make run`, `make test`
- `.env.example` with every required variable documented; no secrets committed
- GitHub Actions workflow running `ruff` and `pytest` on push
- README with: architecture diagram, live dashboard link, model metrics table, five-minute local setup instructions, and a "what I'd do differently" section

```
climate-risk-engine/
├── README.md
├── Makefile
├── docker-compose.yml
├── .env.example
├── config/
│   └── cities.yml
├── ingestion/
│   ├── client.py              # API wrapper, retry/backoff
│   ├── extract.py             # orchestrates pulls, writes to bronze
│   └── schema.sql             # bronze DDL
├── dbt_analytics/
│   ├── dbt_project.yml
│   ├── profiles.yml.example
│   └── models/
│       ├── staging/           # dedup, unit + timezone normalisation
│       ├── intermediate/      # climatology windows, Z-scores
│       └── marts/             # star schema
├── machine_learning/
│   ├── features.py            # feature construction from gold
│   ├── train.py               # training + evaluation + baselines
│   ├── predict.py             # inference -> fact_ml_predictions
│   └── artifacts/             # versioned .joblib + metrics.json
├── dashboard/
│   ├── app.py                 # Streamlit application
│   └── export_for_tableau.py  # gold marts -> CSV for Tableau Public
├── tests/
├── docs/
│   └── images/                # architecture diagram, dashboard screenshots
└── run_pipeline.py
```

---

## 7. Schedule

**Budget:** 8 hours/day, 5 days/week, approximately 40 hours per week.

This is roughly four times the effort level the original five-week plan assumed. At this pace the work is 15 working days, not six weeks. The schedule below is deliberately compressed: an over-long schedule at a full-time pace does not produce a better project, it produces gold-plating on the parts that already work while the hard parts stay unfinished.

Days are sized to be genuinely full. If a day finishes early, move to the next one rather than polishing; the stretch list in §7.2 exists to absorb spare capacity productively.

### 7.1 Core build: 15 working days

**Week 1: Foundation and ingestion**

| Day | Focus | End-of-day state |
|---|---|---|
| 1 | Scaffolding | Repo initialised, `.gitignore` and `.env.example` committed *before* any credential exists. Docker Postgres up. Neon project created and reachable. `cities.yml` in place. Bronze DDL applied. |
| 2 | API client | `client.py` with tenacity backoff, timeout, and 429 handling. Single-city single-year pull verified against the API docs by hand. |
| 3 | Backfill | Full 30-year daily pull plus 24-month hourly pull for all 15 cities, landed in local bronze. Start this early; it will run for hours. |
| 4 | Ingestion hardening | Idempotency proven (re-run produces identical row counts). Row-count reconciliation per city-year. `pytest` covering the client's retry paths. |
| 5 | dbt silver | Project initialised. Staging models: deduplication via `row_number()` subquery, unit assertions, `timestamptz` casting. All silver tests green. |

**Week 2: Modelling and machine learning**

| Day | Focus | End-of-day state |
|---|---|---|
| 6 | Gold star schema | `fact_weather_observations`, `fact_weather_hourly`, `dim_cities`, `dim_date` built and tested. Hemisphere-aware season mapping handles Lagos and Singapore correctly. |
| 7 | Climatology | Day-of-year μ and σ per city with ±7-day smoothing. Leakage-safe reference windows (observation year excluded from its own baseline). Z-score anomaly flags generated. |
| 8 | **VALIDATION GATE** | Anomaly flags checked against the seven documented events in `cities.yml`. Leap-day join verified. Anomaly rate per city sanity-checked. **Do not proceed until this passes.** |
| 9 | Features and baselines | `features.py` producing the lag/rolling/tendency feature set. Persistence and climatology baselines scored and recorded. |
| 10 | Training | XGBoost trained on the chronological split. PR-AUC, F1, Brier, calibration curve, SHAP values recorded. Model beats both baselines, or the reason it does not is diagnosed. |

**Week 3: Serving, orchestration, delivery**

| Day | Focus | End-of-day state |
|---|---|---|
| 11 | Inference and promotion | `predict.py` writing `fact_ml_predictions`. Gold marts promoted from local to Neon. Dashboard queries verified against the hosted copy. |
| 12 | Dashboard I | Views 1 (Global Anomaly Map) and 2 (Climate Matrix) built in Streamlit against live Neon. |
| 13 | Dashboard II | Views 3 (Storm Dynamics) and 4 (Risk Horizon) built. App deployed to Streamlit Community Cloud with a working public URL. |
| 14 | Orchestration | `run_pipeline.py` end-to-end. `Makefile` with setup/run/test. GitHub Actions running ruff and pytest. Structured logging with row counts per stage. |
| 15 | Delivery | README complete: architecture diagram, metrics table, dashboard link, five-minute setup, limitations section. Clean-clone test on a fresh directory. Screenshots captured. |

### 7.2 Day 8 is the gate, not a formality

Seven cities in `cities.yml` carry a dated extreme-weather event: the 2021 Pacific Northwest heat dome, the 2022 UK 40°C record, the 2010 Moscow heat wave, and four others. If those dates do not surface as |Z| > 2.5 anomalies, the climatology logic is wrong and everything downstream is built on sand.

At a full-time pace the temptation is to blow through this checkpoint because momentum feels good. Resist it. Finding a broken baseline on Day 8 costs half a day. Finding it on Day 15, after the model has trained on bad labels and the dashboard has been built on bad predictions, costs the project.

### 7.3 Optional stretch week (Days 16-20)

Only after every box in §9 is ticked. In priority order:

1. **Dagster orchestration** replacing `run_pipeline.py`, the single addition that most changes how the project reads to a data-engineering hiring manager, since it demonstrates asset lineage and scheduling rather than a sequential script.
2. **dbt snapshots** on `dim_cities` to demonstrate slowly-changing-dimension handling.
3. **A model card** documenting intended use, training window, known failure modes, and calibration. Increasingly expected, rarely produced by portfolio projects.
4. **Great Expectations or dbt-expectations** for data-quality assertions beyond dbt's built-in tests.
5. **Tableau Public version** from exported CSVs, if a target role names Tableau.

---

## 8. Risk Register

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| Baseline window exceeds available history | High | Resolved | Ingest 30 years from the outset (§5.1) |
| `QUALIFY` syntax fails on Postgres | Medium | Resolved | Use subquery + `row_number()` (§Workstream 2) |
| Tableau Public cannot reach Postgres | High | Certain | Streamlit primary, Tableau via CSV export (§Workstream 4) |
| Target leakage inflates model metrics | High | Likely if unguarded | Chronological split; exclude observation year from its own climatology |
| Model fails to beat baselines | Medium | Possible | Acceptable outcome if honestly reported and analysed; a documented negative result with diagnosis is stronger than an unexamined positive one |
| API rate limits stall a 30-year backfill | Medium | Likely | Chunked requests, backoff, resumable ingestion, overnight first run |
| Scope creep across five workstreams | High | Likely | Out-of-scope list in §3 is binding; stretch goals go in the README, not the repo |
| Neon 0.5 GB storage exhausted | Medium | Likely if unguarded | No raw JSON columns in bronze; raw payloads gzipped to disk; only gold marts promoted to Neon (§5.4) |
| Neon 100 CU-hour allowance consumed by backfill | Medium | Likely if unguarded | All backfill and dbt iteration runs against local Docker; Neon receives finished marts only |
| Validation gate skipped under full-time momentum | High | Likely | Day 8 is a hard stop; no Day 9 work begins until the seven documented events surface as anomalies (§7.2) |
| Credential committed to a public repo | High | Possible | `.gitignore` and `.env.example` committed on Day 1 before any secret exists; Streamlit Cloud requires a public repo |

---

## 9. Definition of Done

The project is complete when all of the following are true:

- [ ] `git clone` -> `make setup` -> `make run` succeeds on a clean machine
- [ ] Every dbt model has tests and a description; `dbt build` is green
- [ ] Model metrics are recorded alongside both baselines on a chronological test split
- [ ] Dashboard is live at a public URL and linked from the README
- [ ] README contains the architecture diagram, metrics table, setup steps, and a limitations section
- [ ] Anomaly detection has been validated against at least three known historical extreme-weather events
- [ ] No credentials are present anywhere in git history
- [ ] Pipeline runs against both local Docker and Neon by changing `DATABASE_URL` alone
- [ ] Neon project sits under 0.5 GB with headroom

---

## 10. Decisions Log

All four open questions from v1.0 are now resolved. Recorded here because a reviewer reading the repo should be able to see not just what was built but why the alternatives were rejected.

| # | Decision | Chosen | Rejected alternative and why |
|---|---|---|---|
| D1 | Warehouse host | **Neon** free plan for serving, Docker Postgres for local dev, switched by `DATABASE_URL` | Supabase: free projects pause after seven days of inactivity and need manual restore, which breaks a portfolio link opened weeks later. Local-only: unreachable from Streamlit Community Cloud. |
| D2 | City list | **15 cities**, 10 northern / 5 southern, specified in `config/cities.yml` | An all-temperate list would have made Z-scores look like over-engineering. The chosen set spans Af to Dfb, 64°N to 37°S, 10 m to 1753 m elevation, and includes four deliberate timezone edge cases. |
| D3 | Dashboard tool | **Streamlit + Plotly**, deployed to Streamlit Community Cloud | Tableau Public cannot connect to a private Postgres instance, so the live-warehouse architecture would have become a CSV export in disguise. Tableau Desktop has no free public hosting. |
| D4 | Time budget | **8 hrs/day x 5 days = ~40 hrs/week**, 15 working days | The original six-week schedule assumed 10-15 hrs/week and would have left ~130 surplus hours, which in practice become gold-plating rather than additional scope. |

### Notes on D2: why these fifteen

Three properties of the selection do specific work and should be explained in the README:

- **Singapore and Moscow are the bookends.** Singapore's σ is minimal, so a 2 °C departure is a three-sigma event; Moscow's is the widest in the set. Together they demonstrate why a fixed absolute threshold would be useless and a Z-score is not.
- **Lagos and Singapore break the four-season assumption.** Neither has a temperate thermal cycle. `dim_date` must handle wet/dry and effectively-seasonless regimes rather than mapping months to four northern seasons.
- **Four cities are timezone traps.** Phoenix observes no DST despite a US longitude; Reykjavik is UTC+0 year-round; Delhi is UTC+05:30, which breaks integer-hour arithmetic; Sydney and Auckland run DST on the opposite calendar. Handling all four means the timezone logic is correct by design rather than by accident.
