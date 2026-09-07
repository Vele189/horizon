# Climate Volatility & Risk Engine

An end-to-end analytics platform that ingests three decades of global weather observations, models them into a governed dimensional warehouse, and forecasts extreme temperature anomalies across fifteen major cities.

> **Status:** in development. This README is a stub; the full write-up — architecture diagram, model metrics, live dashboard link, and five-minute setup — lands on Day 15. The complete proposal and delivery plan is in [`docs/proposal.md`](docs/proposal.md).

## Configuration

No credentials are committed to this repository, and none ever have been. All configuration is supplied through environment variables:

```bash
cp .env.example .env   # then fill in the values
python config.py       # prints the resolved config, secrets masked
```

Every variable is named and documented in [`.env.example`](.env.example), and [`config.py`](config.py) is the only module that reads the environment. `.env` is git-ignored.

## Warehouse topology

Two Postgres environments, switched by a single environment variable. No code
branches on which one is in use.

| | Host | Holds | Used for |
|---|---|---|---|
| **Local** | PostgreSQL 16 in Docker Compose | bronze → silver → gold | The 30-year backfill, all dbt iteration, model training |
| **Serving** | Neon free plan, project `horizon` (`aged-paper-67892047`, `aws-us-east-2`) | gold marts and predictions only | What the public Streamlit dashboard reads |

**The backfill runs locally and Neon receives finished gold marts only.** Two
free-plan limits force this and shape everything downstream:

- **0.5 GB storage.** Raw API responses are kept as gzipped files under a
  git-ignored `data/raw/`, never as a per-row JSON column — that alone would
  exhaust the budget. Bronze and silver never leave the local container.
- **100 compute-hours per month.** A multi-hour backfill against a serverless
  database is slow and wastes the allowance. Neon sees one bulk load per
  promotion, then read-only dashboard traffic.

Neon scales compute to zero after five minutes idle and resumes on the next
query, so there is no keep-alive job to maintain. Measured from a development
machine in South Africa against `aws-us-east-2`:

| | Connect + first query |
|---|---|
| Warm (compute active, median of 5) | 2409 ms |
| Cold (after 340 s idle) | 3619 ms |
| **Cold-start penalty** | **~1.2 s** |

Only the 1.2 s is Neon resuming compute. The 2.4 s floor underneath it is
distance: TCP handshake to that region measures 271 ms round-trip and TCP+TLS
553 ms, so a Postgres connection's handshake and SCRAM exchange spend roughly
eight round-trips crossing an ocean. **This is a local-development cost, not a
production one** — the dashboard is deployed to Streamlit Community Cloud,
which sits on the same continent as the database. Do not tune the schema in
response to latency observed from a laptop.

Reproduce with:

```bash
python tests/check_connection.py --target serving --cold   # after 5 min idle
python tests/check_connection.py                           # both targets
```

### Quota dashboards

Free-plan usage is not exposed through the API, so these are console links:

- Project overview and storage — <https://console.neon.tech/app/projects/aged-paper-67892047>
- Compute metrics — <https://console.neon.tech/app/projects/aged-paper-67892047/monitoring>
- Org usage against the free-plan allowance — <https://console.neon.tech/app/orgs/org-twilight-mode-94780402/billing>

### Known deviation

Neon provisioned the project on **PostgreSQL 18**; local development runs
**PostgreSQL 16**, as specified. The gold marts use no version-specific syntax
and both targets are verified by the same connection check, but the skew is
recorded here rather than discovered later.

## Source API

Open-Meteo's historical archive (ERA5 reanalysis), `archive-api.open-meteo.com/v1/archive`.
No API key, no credential variable. [`ingestion/client.py`](ingestion/client.py)
is the only module that talks to it.

```bash
python ingestion/client.py --city london --year 2023            # daily
python ingestion/client.py --city singapore --year 2024 --grain hourly
```

### Nothing is left to a default

The API's defaults are local time and metric units. A request that omits
`timezone=UTC` still returns 200 and still looks like weather — it just has the
day boundaries shifted, which would poison a 30-year climatological baseline
without failing any obvious downstream test. So the client states every
unit-bearing parameter (`timezone`, `temperature_unit`, `wind_speed_unit`,
`precipitation_unit`, `timeformat`, `cell_selection`) and then **verifies the
response against what it asked for**: a non-zero `utc_offset_seconds`, a changed
unit string, a missing variable, or a row count that does not match the
requested range all raise rather than land.

All 21 daily and 12 hourly variables were diffed against the documentation
against a live response on 2026-09-07. The unit each is expected to report is
asserted on every response, so the day an upstream default changes the run stops
instead of writing kilometres per hour into a column commented `km/h`. Two
details that a units assertion catches and a schema does not: `snowfall_sum` is
**centimetres** while every other depth is millimetres, and
`shortwave_radiation_sum` is **MJ/m²**, not W/m².

### Requests are snapped to a grid cell, and the client records which

ERA5 is gridded reanalysis, not station data. The coordinates in
`config/cities.yml` are not the coordinates that answer:

| | Requested | Answered |
|---|---|---|
| London | 51.5074, −0.1278 at 11 m | 51.4938, −0.1630 at 16 m |
| Singapore | 1.3521, 103.8198 at 15 m | 1.3708, 103.8024 at 46 m |

Every row therefore carries `api_latitude`, `api_longitude`, and
`api_elevation_m`, so provenance survives an edit to `cities.yml`. The request
URL is *not* stored per row — see [Measured storage](#measured-storage) — it is
derived from the archived window by `archive.source_url_for()`. `cell_selection=land` is sent explicitly — it is the
default, but it is the parameter deciding whether Lagos is Lagos or the Bight of
Benin. `elevation` is deliberately *not* sent; passing it would override
Open-Meteo's 90 m DEM downscaling and change the values returned.

### Failure policy

The backfill is hundreds of requests running for hours, so a transient 502 is
not a possibility but a certainty. Measured against London from a development
machine:

| | Rows | Response | Wall | API generation |
|---|---|---|---|---|
| One city-year, daily | 365 | 47 KB | 970 ms | 35 ms |
| One city-year, hourly | 8 784 | 732 KB | 809 ms | 230 ms |

- **Timeouts are an explicit (connect, read) pair.** `requests` applies none by
  default; a socket that opens and then goes quiet would hang the run forever
  with no traceback. `REQUEST_CONNECT_TIMEOUT_SECONDS` and
  `REQUEST_TIMEOUT_SECONDS`.
- **5xx, timeouts, and truncated bodies** retry with exponential backoff plus
  jitter, capped at `MAX_RETRY_ATTEMPTS` total attempts. The jitter is not
  decoration: without it, fifteen cities that failed against one upstream blip
  retry in lockstep and reproduce it.
- **429 honours `Retry-After`** — both the delay-seconds and HTTP-date forms —
  rather than backing off blindly, which would otherwise retry sooner than the
  server allows or wait far longer than needed. It is capped at five minutes so
  a misconfigured proxy cannot park the backfill for a day. Open-Meteo sends
  *no* header and a body reading "Minutely API request limit exceeded. Please
  try again in one minute", so the headerless case waits a full minute rather
  than falling back to exponential backoff — 2s, 6s, 10s, 16s never spans the
  window the server is asking for, and burns every attempt for nothing.
- **4xx raises on the first attempt** with the response body in the message.
  Open-Meteo's `reason` field names the offending parameter, and a bad request
  will fail identically on every retry.
- **Retries are re-raised as the real exception**, not a `RetryError` wrapper,
  so a caller can tell a rate limit from a dead connection.

## Backfill planner

[`ingestion/planner.py`](ingestion/planner.py) decomposes the backfill into
city × window work units, records what has landed, and paces the run.

```bash
python ingestion/planner.py                    # the plan and what it costs
python ingestion/planner.py --list             # every pending unit
python ingestion/planner.py --progress         # what the manifest says landed
```

### Open-Meteo meters weighted calls, not HTTP requests

This is the finding that shapes the whole workstream. A request costs
`(variables / 10) × (days / 14)` API calls, each factor floored at 1 — so one
city-year of the 21 daily variables costs **55 calls, not one**.

Measured on 2026-09-07: five requests for London — one, two, five, ten and
thirty years of daily data — were refused on the fifth with `HTTP 429 Minutely
API request limit exceeded`. Five requests is nowhere near the documented
600/min. Their *weighted* cost is 55 + 110 + 274 + 548 = 987 by the fourth,
crossing 600 exactly where the refusal landed. The formula is what the planner
budgets against, and `test_the_observed_429_is_consistent_with_the_formula`
pins that reasoning down.

Two consequences:

- **Chunk size is quota-neutral above a fortnight.** Weight is proportional to
  days, so ten one-year units cost what one ten-year unit costs. Below 14 days
  the floor makes short units cost a full call each — smaller is never cheaper,
  it only buys finer resume granularity. That frees the default to be chosen
  for legibility: **one calendar year**, measuring 47 KiB daily / 732 KiB
  hourly, at 55 / 31 weighted calls.
- **A fixed inter-request delay is not enough.** One second between city-years
  would spend 3 300 calls a minute against a 600 budget. `REQUEST_DELAY_SECONDS`
  is a *floor*; the planner derives the real delay from each unit's weight.
- **And the minutely allowance is not the binding one.** Half of 600 calls a
  minute is 18 000 an hour, against an hourly allowance of 5 000. The first
  real backfill run cleared the minutely bar on every request and still
  collected `HTTP 429 Hourly API request limit exceeded` nineteen minutes in,
  having spent 5 059 calls. Pacing now takes the larger of the two derived
  delays, which is four times slower — **~44 s between one-year daily units**,
  not 11.

### The backfill does not fit in one day of free quota

| | Units | Rows | Weighted calls |
|---|---|---|---|
| Daily, 1995 → present | 480 | 173 505 | 26 026 |
| Hourly, trailing 24 months | 45 | 263 160 | 957 |
| **Total** | **525** | **436 665** | **26 983** |

Against an allowance of 10 000 calls/day that is **2.7 days**, not the "runs for
hours" the delivery plan assumed. This is not something to engineer around — it
is the reason the manifest exists. `Plan.within_daily_quota()` returns the
prefix that fits in today's allowance; the run stops there and tomorrow's
session resumes from the manifest.

### Resumability

Completed units are appended to a JSONL manifest (`INGEST_MANIFEST_PATH`,
git-ignored) and fsynced before the next unit starts, so a record on disk means
the rows really are in the warehouse. Append-only rather than a rewritten
document because the failure being designed against is the run dying mid-write:
a truncated final line costs one re-fetched unit, a truncated rewrite costs the
whole history.

Completion is tested by **date coverage**, not by matching the window key, so
the chunk size can change between sessions. The natural response to a 429 or a
timeout is to halve it, and re-fetching everything landed so far would be a
harsh price for that:

```
5 one-year units planned, 2 landed
  → re-plan at 12 months:  3 pending, 2 skipped
  → re-plan at  6 months:  6 pending, 4 skipped   # still skipped
```

The limit is honest and tested: *growing* the chunk past a landed boundary
re-fetches the partly-covered window.

## Raw payload archive

Every response is gzipped to
`data/raw/{grain}/{city_id}/{start}_{end}.json.gz` exactly as it arrived — not
re-serialised, not reordered, not validated. `data/` is git-ignored and nothing
under it is tracked.

```bash
python ingestion/archive.py             # what is on disk
python ingestion/archive.py --verify    # re-parse everything, no network
python ingestion/archive.py --stats     # measured size, projected to the full backfill
```

### The write happens before the parse

That ordering is the whole point. A parsing bug found on day seven — a unit
misread, a timestamp off by an hour, a column mapped to the wrong variable —
costs a re-run of the transformation if the payloads are on disk, and a
[2.7-day re-pull](#the-backfill-does-not-fit-in-one-day-of-free-quota) if they
are not. So archival hangs off an `on_payload` hook that
`fetch_observations` calls between the successful response and
`parse_payload`, and a response that fails validation is still on disk
afterwards. The hook runs outside the retry loop, so failed attempts are never
archived, and an exception from it propagates — a response fetched and then
dropped on the floor is worse than a loud failure.

Replay feeds archived payloads through the *same* `parse_payload` the live
path uses, so fixing a parser bug fixes replay by construction rather than
twice. The request URL is reproduced by `request_url()`, which prepares the
identical string through `requests` rather than assembling it by hand — tested
against a live response. That same function is what lets bronze omit the column
entirely.

### Measured size

Sampled across five climates and both grains (London, Reykjavík, Singapore,
Phoenix, Sydney daily; London and Cairo hourly):

| | Rows sampled | Compressed | Ratio |
|---|---|---|---|
| Daily (21 variables) | 1 826 | 33.5 B/row | 29% of raw |
| Hourly (12 variables) | 17 568 | 14.9 B/row | 20% of raw |

Projected across the full 436 665-row backfill: **5.5 MiB daily + 3.7 MiB
hourly ≈ 9.3 MiB**. Two decimal orders below the 0.5 GB Neon allowance the
per-row `jsonb` column would have eaten into — and it never goes near the
database at all.

Files are written to a temporary name in the destination directory, fsynced,
then renamed over the target, so a crash mid-write leaves the previous file
rather than a truncated one. `mtime=0` and an empty gzip `filename` field keep
the output byte-identical for identical input, which makes re-archiving a unit
detectably a no-op.

## Bronze loader

[`ingestion/loader.py`](ingestion/loader.py) replays archived payloads into
`bronze_raw.observations_daily` and `_hourly`. It reads from disk, never from
the network.

```bash
python ingestion/loader.py --dry-run     # what would be loaded
python ingestion/loader.py               # load everything not yet in the manifest
```

### What it does not do

Bronze's job is faithful landing. Nothing here converts a unit, shifts a
timezone, fills a gap, or deduplicates a row — each of those is a decision, and
a decision belongs in dbt where it is SQL, version controlled and tested, not
buried in a Python loader where it is invisible to anyone reading the models.
Every one of those negatives has a test that fails if a future well-meaning
change adds the cleverness.

The judgement call is null handling, and it is exercised in the direction of
doing nothing: a null in the payload is a null in the warehouse. Coercing it to
zero would turn "this grid cell reports no snowfall data" into "it did not
snow" — a different claim, a wrong one, and indistinguishable from a real
measurement afterwards.

Two things make that work, and both were caught by tests rather than reasoned
about:

- The frame is built with `dtype=object`. Left to itself pandas widens any
  column containing a null to `float64` and replaces the null with `NaN`, so
  `weather_code` would arrive as `51.0` and a missing value would land as a
  float rather than SQL NULL.
- The CSV handed to `COPY` uses `QUOTE_NOTNULL`, not the default
  `QUOTE_MINIMAL`. Postgres reads an *unquoted* empty CSV field as NULL and a
  quoted one as an empty string; `QUOTE_MINIMAL` writes both as nothing at all,
  which would silently null any empty string.

Type conversion is refused rather than performed. A `smallint` column rejecting
`"98.0"` means the API changed how it represents an integer, and landing it
quietly as `98` would hide that — the payload is already archived, so nothing
is lost by stopping, only delayed.

### Throughput

`to_sql`'s `method=` hook runs `COPY FROM STDIN` instead of pandas' multi-row
`INSERT`. Measured locally: **19 394 rows in 0.7 s (~28 000 rows/s)**, which
puts the full 436 665-row backfill at roughly 15 seconds of load time. A test
asserts no `INSERT` statement is issued at all, so the mechanism is pinned
rather than described.

Each unit commits in its own transaction, and the manifest is recorded *after*
that commit — so a manifest entry always means the rows are really in the
warehouse. `batch_id` groups every row one run wrote, so a bad run is undone
with a single `delete ... where batch_id = ...`.

## Running the backfill

[`ingestion/backfill.py`](ingestion/backfill.py) is the only module that drives
the others.

```bash
python ingestion/backfill.py --grain daily --dry-run   # what it would do
python ingestion/backfill.py --grain daily             # run it
python ingestion/backfill.py --report                  # what landed
```

Per unit, in this order and no other:

1. If the payload is already on disk, **no request is made** — the archive is
   the fetch cache as well as the replay source, so a run killed between
   fetching and loading costs nothing to resume.
2. Otherwise fetch it, archiving before parsing.
3. Load it into bronze in its own transaction.
4. Record it in the manifest, *after* that transaction commits.

Step 4 last is the whole resumability story. A manifest entry means the rows
are in the warehouse; anything less would let a crash leave the next run
skipping a window whose rows are missing, with nothing downstream reporting the
hole.

Ctrl-C once finishes the unit in flight and stops cleanly — the signal sets a
flag checked between units rather than raising wherever the interpreter happens
to be, which could be mid-`COPY` or in the one window between the warehouse
commit and the manifest write. Twice restores the default handler.

### Three rate limits, and only one is worth waiting out

The 429 body names the allowance it spent, and the three want different
responses:

| Body says | Response | Why |
|---|---|---|
| `Minutely API request limit exceeded` | wait 60 s, retry | A minute passes inside a request's retry budget |
| `Hourly API request limit exceeded` | fail immediately | An hour does not. Four 60 s retries take four minutes to arrive where they started |
| `Daily API request limit exceeded` | fail immediately | Same, more so |

Measured on the same failure: **1 second to stop, against 4 minutes before**.
With a manifest, stopping and resuming is free, which is what makes failing
fast the cheaper answer.

### Measured throughput

| | |
|---|---|
| Sustainable pace | ~44 s/unit (hourly allowance, 90% of 5 000 calls/h) |
| Landing speed | ~28 000 rows/s (`COPY`) — never the bottleneck |
| Hourly ceiling | ~81 one-year daily units, ~4 500 weighted calls |
| Daily ceiling | ~180 units, ~9 900 calls |
| **Full daily backfill** | **480 units, 26 026 calls → ~2.7 days of free quota** |

The pacing dominates entirely: a one-year unit takes ~0.3 s to fetch and ~0.02 s
to load, then waits 44 s. Day 3's "several hours of wall-clock time" is
optimistic by a factor of roughly ten, and the fix is not engineering — it is
running the thing across three days, which the manifest makes free.

## Hourly grain, and what bronze actually costs

Hourly observations cover the **trailing 24 months only** — the Storm Dynamics
view is their only consumer. Thirty years at the same grain would be over four
million rows for no analytical benefit, and would threaten Neon's allowance if
bronze were ever promoted.

```bash
python ingestion/backfill.py --grain hourly --anchor 2026-09-02
python ingestion/backfill.py --report --grain hourly --anchor 2026-09-02
```

The window is **anchored, not relative to now**. `--anchor` names the date it
ends at and `INGEST_HOURLY_MONTHS` how far back it reaches. Left to the default
the anchor is the archive edge, which moves — so "the trailing 24 months"
describes a different window tomorrow, and a plan that cannot be reproduced
cannot be verified. 45 units, 263 160 rows, 957 weighted calls: under a tenth
of a day's free allowance, against the daily grain's 26 026.

### Measured storage

| | Rows | Bytes/row | Projected full table |
|---|---|---|---|
| `observations_daily` | 173 505 | 270 | **47 MB** |
| `observations_hourly` | 263 160 | 224 | **59 MB** |

Bronze projects to **~106 MB**, comfortably inside the proposal's ~250–300 MB
budget for all layers. It did not start there.

**`source_url` was 83% of the daily row payload and 81% of the hourly one** —
712 and 466 bytes of the same few hundred distinct strings, repeated once per
row, for a projected **300 MB of the original 373**. §5.2 asked for it per row;
measuring it showed the column was both the dominant cost and entirely
derivable, since `request_url()` already reconstructs the exact string from
`(city_id, grain, start, end)` — which is what ING-03's replay depends on.

It is gone from the schema. `archive.source_url_for(city_id, grain,
observation_time)` finds the archived window covering a row and rebuilds its
URL, at no bytes per row. **Deviation from §5.2, deliberate:** the trade is
268 MB of warehouse against `data/raw/` becoming load-bearing for provenance as
well as for replay — delete the archive and row-level URLs are no longer
recoverable. `schema.sql` carries an idempotent `drop column if exists` so an
existing database converges on re-apply; Postgres only returns the space on
`VACUUM FULL`, which the schema file deliberately does not run.

The other thing measurement turned up, also now in `--report`:
**`pg_total_relation_size` counts dead tuples.** The first hourly reading was
31 MB against a true 12 MB — the table had been loaded and cleared repeatedly
during development. The report names the dead share when it exceeds a tenth and
says to `VACUUM FULL`.

## Idempotency

`make run` invokes ingestion on every call, so a loader that is not idempotent
corrupts the warehouse a little further each time. The guarantee has two layers
and neither is sufficient alone.

**1. Run level — the manifest.** A unit recorded as landed is not planned
again, so a re-run makes no requests and writes no rows. This is exact, not
approximate: the row count after a re-run is the *same number*.

Measured against the three cities complete in bronze — 35 069 rows, ids
5272..135388:

```
re-run 1:  0 units planned, 0 requests, 0 rows, 0.00s
re-run 2:  0 units planned, 0 requests, 0 rows, 0.00s
re-run 3:  0 units planned, 0 requests, 0 rows, 0.00s
fingerprint (count, min id, max id, distinct observations): identical
```

A no-op re-run costs one manifest read and no network at all, which is what
makes putting ingestion in `make run` reasonable rather than merely tolerable.

**2. Row level — append-only bronze, deduplicated in silver (DBT-02).** There
is exactly one window layer 1 cannot cover: a crash between the warehouse
commit and the manifest write leaves rows that nothing has recorded, and the
next run lands them again.

That direction is deliberate. The manifest is written *last* — writing it first
would let the same crash leave a manifest claiming rows that are not there, and
a hole is silent where a duplicate is not. Duplicates are recoverable; holes
are discovered in month three by a climatology that is quietly wrong.

So bronze carries no unique constraint on `(city_id, observation_time)` — one
would reject a legitimate re-ingest — and silver takes the most recent
`ingested_at` per pair:

```sql
select * from (
  select *, row_number() over (
    partition by city_id, observation_time order by ingested_at desc
  ) as rn
  from bronze_raw.observations_daily
) ranked where rn = 1
```

`tests/test_idempotency.py` runs that query over deliberately duplicated
bronze, so "deferred to DBT-02" is a demonstrated claim and not a promise: it
collapses to exactly one row per pair, keeps the newest copy, and does not
reach across cities.

**Blast radius.** Re-running one city touches only that city — a test pins the
others' row count, id range, distinct timestamps and value sum, all unchanged.
Re-running one city-year duplicates exactly that year and nothing else.

**Re-landing is free.** A lost manifest costs no quota: the payloads are
already in `data/raw/`, so the second run serves every unit from the archive
and makes zero requests.

`--report` surfaces the symptom directly, since duplicates are legal and
therefore easy to stop noticing:

```
36,895 rows total
365 of them (1.0%) are a second copy of an observation already present.
```

## Reconciliation

Expected against actual, per city, with every gap catalogued and categorised.

```bash
python ingestion/reconcile.py --grain daily
python ingestion/reconcile.py --grain hourly --anchor 2026-09-02 --write docs/
```

Committed artefacts:
[`docs/ingestion-reconciliation-daily.md`](docs/ingestion-reconciliation-daily.md),
[`docs/ingestion-reconciliation-hourly.md`](docs/ingestion-reconciliation-hourly.md).

A gap is not automatically an error — ERA5 has genuine boundaries and the
backfill is quota-bound across days — but every one is *named*, because the
failure this guards against is a hole nobody notices until a thirty-year
climatology is quietly computed over twenty-eight. Every gap over three days
lands in exactly one category:

| category | meaning | status |
|---|---|---|
| archive boundary | Outside what ERA5 can serve. Nothing can fill it. | accepted |
| not ingested | Inside the servable range; the backfill has not reached it. | accepted while in progress |
| api limitation | A completed unit recorded fewer rows than its window. | structurally prevented |
| **unexplained** | Fetched, recorded complete, and missing anyway. | **must be zero** |

`api limitation` is empty by construction rather than by luck: the client
asserts the returned row count against the requested range *before* parsing,
and the loader asserts it again before writing, so a short response raises
instead of landing. A non-empty result there means one of those assertions has
been weakened.

**Hourly is fully reconciled** — 15/15 cities, 17 544 observations each, delta
zero, no gaps. **Daily is mid-backfill**: 16 gaps, all `not ingested`, none
unexplained.

### Two discrepancies that are not gaps

- **Duplicates.** `singapore` daily carries 365 rows more than it has distinct
  observations: one city-year landed twice, once by an out-of-band loader run
  that bypassed the manifest. Legal, and silver deduplicates it.
- **Surplus.** `cairo` and `london` hourly each hold 5 880 observations
  *before* the anchored window — the ING-03 archival samples, which took a
  calendar year rather than the trailing-24-month window. Real data the range
  did not ask about.

The report separates these from each other and from `delta`, because a row
count above expectation is as much a discrepancy as one below, and the two
have different causes.

## dbt

```bash
python dbt_analytics/dbt_env.py --write-profile      # once, generates the git-ignored profiles.yml
python dbt_analytics/dbt_env.py -- dbt build         # every run
python dbt_analytics/dbt_env.py -- dbt source freshness
```

| layer | path | materialised | schema |
|---|---|---|---|
| silver | `models/staging` | view | `silver_staging` |
| — | `models/intermediate` | ephemeral | `silver_staging` |
| gold | `models/marts` | table | `gold_marts` |

Staging models are views because they are thin projections over bronze and
materialising them would double the storage for no query benefit — nothing
reads silver directly. Marts are tables because the dashboard queries them over
a serverless connection, where the difference between reading a table and
re-running a thirty-year window function is the difference between a usable
dashboard and a slow one. Intermediate models are ephemeral, so they inline
into the marts rather than becoming objects nobody queries.

### One connection string, split on demand

dbt's postgres adapter takes host, user, password, port and dbname as separate
settings and cannot accept a URL. The obvious fix — a second set of `DBT_*`
variables in `.env` — creates two places a connection lives and one of them
goes stale, so a dbt run quietly builds against yesterday's database.

Instead `DATABASE_URL` stays the only place a connection is written, and
[`dbt_analytics/dbt_env.py`](dbt_analytics/dbt_env.py) splits it into what dbt
needs and injects them. Every field in
[`profiles.yml.example`](dbt_analytics/profiles.yml.example) is an `env_var()`
lookup — nothing is hardcoded, and `profiles.yml` itself is git-ignored. A test
parses the template for the variables it reads and asserts the bridge supplies
every one, because either file can change without the other and the failure is
a dbt run against nothing.

`dev` (local Docker) is the default; `prod` (Neon) exists only to promote gold
marts. `SERVING_DATABASE_URL` being unset omits the prod variables rather than
faking them, so `--target prod` fails loudly instead of connecting somewhere
unintended.

### The schema override that stops `gold_marts` becoming `public_gold_marts`

dbt's default `generate_schema_name` builds `<target.schema>_<custom>`. A mart
configured into `gold_marts` would land in `public_gold_marts` — beside the
empty `gold_marts` that `ingestion/schema.sql` created, with nothing to say
which is real. The default exists to keep several developers on one warehouse
apart; here local development has a private database in Docker and the only
other target holds one copy of the marts by design. So
[`macros/generate_schema_name.sql`](dbt_analytics/macros/generate_schema_name.sql)
uses the configured name verbatim.

Verified end to end rather than asserted — a throwaway model in each layer,
run and then dropped:

```
OK created sql view model  silver_staging._wiring_check       CREATE VIEW
OK created sql table model gold_marts._wiring_check_mart      SELECT 1
```

### Source freshness

Both bronze tables are declared with `ingested_at` as the freshness clock —
when *this pipeline* landed a row, not when the observation happened, which
would report every row as decades stale.

Thresholds are deliberately loose: **warn after 7 days, error after 30**. The
archive trails the present by several days, the backfill is quota-bound across
roughly three, and the analysis is a thirty-year climatology where a week of
staleness moves no number that matters. A month means the pipeline has stopped,
which does. Anything tighter would fail continuously during a normal backfill
and teach everyone to ignore the check.

## Silver: deduplication

Bronze is append-only — re-ingesting a window inserts a second copy rather than
replacing the first, and there is no unique constraint on the natural key
because one would reject a legitimate re-ingest. Deduplication is therefore
silver's job.

**PostgreSQL has no `QUALIFY` clause.** That is Snowflake, BigQuery and DuckDB
syntax. The Postgres form is a `row_number()` subquery filtered to rank 1:

```sql
select … from (
    select …, row_number() over (
        partition by city_id, observation_time
        order by ingested_at desc, id desc
    ) as _dedup_rank
    from bronze_raw.observations_daily
) ranked
where _dedup_rank = 1
```

Both grains share one macro,
[`deduplicate_observations`](dbt_analytics/macros/deduplicate_observations.sql).
Columns are read from the relation rather than a hand-maintained list — bronze
has already lost a column once (`source_url`), and a list would have needed the
same edit or kept selecting something that no longer exists.

The `id desc` tiebreaker is not decoration. The loader stamps **one
`ingested_at` per run**, so two copies of a window landed by the same run tie
on the sort key and `row_number()` would pick between them differently on each
build.

| | bronze rows | staging rows | removed |
|---|---:|---:|---:|
| daily | 61 127 | 60 396 | 731 (1.20%) |
| hourly | 280 728 | 274 920 | 5 808 (2.07%) |

Measured 2026-09-07, mid-backfill. The daily figure is one city-year landed
twice by an out-of-band loader run that bypassed the manifest; the hourly
figure is `cairo` and `london`, where an archival sample took a calendar year
and overlapped the later trailing-24-month window by 121 days each.

### Uniqueness is not enough, and the tests know it

`unique_combination_of_columns` proves *one* row survives per key. It says
nothing about *which*. Reversing the sort to `ingested_at asc` leaves it green
and silently serves the oldest copy of every observation — so
`assert_{grain}_keeps_the_newest_ingest` checks the survivor carries the
greatest `ingested_at` bronze holds for its key.

Verified by mutation rather than assumed. Three deliberate breakages, and what
caught each:

| mutation | caught by |
|---|---|
| `where _dedup_rank = 1` removed | uniqueness (both), and no-observation-lost |
| order flipped to `asc` | **only** keeps-the-newest-ingest, on 731 rows |
| partition narrowed to `city_id` | **only** no-observation-lost |

Each singular test is the sole thing standing between one real defect and a
green build.

## Silver: unit assertions

ING-01 asks the API for metric units explicitly and checks them on every
response, so silver's job here is to **assert, not convert**. Converting would
undo work already verified upstream; asserting catches the day it stops being
true.

`dbt build` is green on 56 nodes — 2 models and 54 tests.

| quantity | bound | why that bound |
|---|---|---|
| temperature, dew point | −90 … 60 °C | coldest and hottest ever recorded, rounded outwards |
| sea-level pressure | 850 … 1085 hPa | 870 (Typhoon Tip) to 1084.8 (Agata, Siberia) |
| **surface** pressure | 700 … 1085 hPa | *not* the MSL bound — see below |
| wind speed, gusts | 0 … 120 m/s | asserted through `kmh_to_ms`, stored as km/h |
| humidity, cloud cover | 0 … 100 % | |
| precipitation, snowfall, radiation | ≥ 0 | |
| precipitation hours | 0 … 24 | |

### Two bounds the specification gets wrong for this data

**Surface pressure is not sea-level pressure.** Johannesburg sits at 1753 m and
reads down to **822 hPa** — below an 850 hPa floor — while its sea-level
pressure is a perfectly ordinary 998. The 850–1085 range is a *mean sea level*
range; applying it to `surface_pressure` fails on correct data at altitude. So
`pressure_msl_*` gets 850 and `surface_pressure_*` gets 700, which still sits
far above anything a kPa or inHg mix-up would produce.

**Wind is stored in km/h, and the bound is quoted in m/s.** A 0–120 bound
applied to km/h passes today — the largest gust on record here is **119.9
km/h**, 0.1 under — and breaks on the next ordinary winter storm. The bound is
the physical one (0–120 m/s) applied through `kmh_to_ms`, so the effective
ceiling is 432 km/h.

Verified honestly: removing that conversion **is not currently caught**, because
119.9 < 120. The conversion protects against a false alarm on real data, not
against a missed error — the opposite of the usual reason for a unit test, and
worth saying rather than implying.

### The one conversion that is genuinely required

Open-Meteo reports `snowfall_sum` in **centimetres** while `precipitation_sum`
and `rain_sum` in the same row are **millimetres**. Silver publishes
`snowfall_sum_mm` via the `cm_to_mm` macro so no downstream model has to
remember that one column in the row is a different scale. Both conversion
factors live in [`macros/units.sql`](dbt_analytics/macros/units.sql) — a
`/ 3.6` typed into six schema entries is six chances to type `* 3.6`.

### Ranges pass on swapped columns

`temperature_2m_min` and `temperature_2m_max` satisfy the same bound whichever
way round they are, so three singular tests assert the physics: min ≤ max, rain
≤ total precipitation, dew point ≤ air temperature.

That last one needs a tolerance, and the tolerance needed fixing twice. ERA5
reports to 0.1 °C, so a saturated hour where the two are genuinely equal can
round the dew point one step above — 16 hourly rows, all Singapore, all by
exactly one step. And **a tolerance compared in `real` is not the tolerance you
wrote**: in float4, `23.1 − 23.0` is `0.10000038`, so `> temperature + 0.1` was
true and the test failed on 14 correct rows. Casting both sides to `numeric`
makes the difference exactly `0.1` and the comparison mean what it says.

Verified by mutation, as with the dedup tests: tightening the temperature
ceiling to 30 °C or the humidity ceiling to 50% is caught immediately, so
`accepted_range` is evaluating rather than passing vacuously.

## Silver: UTC and the timezone traps

Everything in silver is `timestamptz`. A `timestamp without time zone` is a
wall-clock reading with no instant attached — it sorts and compares against
other naive values without complaint, and means something different for each of
fifteen cities. A test checks the catalogue rather than a column list, so a
model added tomorrow is covered without anyone remembering.

The city's IANA zone is carried on `stg_cities`, seeded from `cities.yml` by
[`export_cities.py`](dbt_analytics/export_cities.py) — a test asserts the two
agree, because two copies of the same list is one copy and one liability.

### Local time is one-way, and that is not a shortcut

`AT TIME ZONE` returns a *naive* wall-clock reading, and at a daylight-saving
fall-back two different instants produce the same reading — so converting back
cannot recover which. Measured across 274 920 hourly rows: **exactly 10 fail to
round-trip**, one per DST-observing city per autumn transition, each in the
repeated hour.

That is clocks, not a bug. UTC is what silver stores and what everything joins
on; `to_local_time()` exists for display. A test bounds the loss at four rows
per city, so conversion breaking wholesale shows up as thousands rather than
ten.

### The four traps, verified against landed rows

**Phoenix** keeps standard time all year on a US longitude. Across the 2025
spring-forward, with Portland on the same longitude for contrast:

```
UTC 08:00   Phoenix 01:00   Portland 00:00
UTC 09:00   Phoenix 02:00   Portland 01:00
UTC 10:00   Phoenix 03:00   Portland 03:00   ← Portland skips 02:00; Phoenix walks through it
```

Phoenix takes **one** UTC offset across two years; Portland takes **two**. Both
are asserted — "Phoenix never shifts" proves nothing on its own, since a
pipeline that skipped conversion entirely would satisfy it perfectly.

**Delhi** is UTC+05:30, and `observation_time + interval '5 hours'` looks like
a conversion, passes review, and is wrong by half an hour for 1.4 billion
people. **17 544 of 17 544** Delhi rows land on `:30`.

**Sydney** runs DST on the southern calendar, so a hardcoded northern one is
not merely wrong but *inverted* — adding an hour exactly where one should be
subtracted, a two-hour error. January (high summer) is **+11**, July is **+10**,
and the October transition moves the offset mid-file:

```
UTC 10-04 15:00   Sydney 10-05 01:00   offset 10:00
UTC 10-04 16:00   Sydney 10-05 03:00   offset 11:00
```

**Reykjavik** is UTC+0 year round — the city where a broken conversion looks
correct, which is why it is asserted to be exactly zero rather than left to
pass by accident.

All 15 configured zones are checked against `pg_timezone_names`: Python's
`zoneinfo` and Postgres's tz database are different databases, and cities.yml
validates against the first.

## Gold: `dim_cities`

One row per city, built from `config/cities.yml` — **never from observation
data**. A dimension inferred from what happened to land would list fourteen
cities during a backfill and would quietly lose one whose ingestion failed. The
registry says what the set *is*; the facts say what has been observed of it, and
the gap between them is what the reconciliation report exists to surface.

| | |
|---|---|
| rows | 15, one per city |
| columns | `city_id`, `name`, `country`, `country_code`, `region`, `latitude`, `longitude`, `elevation_m`, `timezone`, `koppen`, `hemisphere`, `season_model`, `role` |
| materialised | table, in `gold_marts` |

### These are grid cells, not weather stations

The original draft called this `dim_weather_stations`. Open-Meteo serves ERA5
reanalysis — a physical model reconciled with observations onto a regular grid,
not readings from an instrument at a named place. There is no station, no
instrument history, no siting metadata and no station identifier to join on,
and the station framing would misdescribe the source to anyone who knows the
domain.

So the coordinates here are what was *asked for*; the `api_latitude`,
`api_longitude` and `api_elevation_m` on every fact row are what *replied*.
London's 51.5074/−0.1278 at 11 m resolves to 51.4938/−0.1630 at 16 m.

### `hemisphere` is derived twice, on purpose

`cities.py` computes it from `lat >= 0` and seeds it; `dim_cities` recomputes it
in SQL from the latitude column. Neither is authoritative — a test asserts they
agree, so a drift between Python and SQL is caught rather than absorbed.

That matters more than it looks. The season mapping reads this column, so for
the five southern cities a wrong value doesn't mislabel summer and winter, it
**inverts** them. Verified by planting `hemisphere = 'north'` on Sydney: the
cross-check fails, and passes again once the seed is restored.

### `region` was added to `cities.yml`

The column didn't exist. Rather than derive it from `country_code` in SQL —
which would put the mapping in a second place and make adding a city a two-file
edit — it is now a validated field on the registry, constrained to six
continent-level values. Deliberately coarse: it groups fifteen cities in a
dashboard filter rather than encoding geography, and a finer scheme would put
most of them in a bucket of one.

### One dbt operational note

**A `--full-refresh` seed drops its table with `CASCADE`**, taking dependent
views with it — `stg_cities` vanished and every model reading it errored until
the next `dbt build`. Changing a seed's column set requires `--full-refresh`,
so the two go together: `dbt build --full-refresh`, not `dbt seed
--full-refresh` alone.

## Gold: `dim_date` and hemisphere-aware seasons

The date spine runs from the backfill's start to a year past the archive edge —
generated, not derived from the facts. A spine built from what has landed would
have a hole wherever ingestion does, and a join against it would *hide* the
hole rather than reveal it.

### `dim_date` has no `season` column, deliberately

A season is not a property of a date. **15 December is summer in Sydney and
winter in London**, and the same spine row has to serve both. So `dim_date`
carries `season_northern` and `season_southern` side by side, and
`dim_city_season` (15 cities × 12 months = 180 rows) resolves the right one per
city.

| regime | cities | December |
|---|---|---|
| `four_season` north | 8 | winter |
| `four_season` south | 5 | **summer** |
| `wet_dry` | lagos | dry |
| `seasonless` | singapore | year_round |

**Tropical cities are not forced into four seasons.** Lagos — tropical monsoon,
no thermal season worth the name — gets wet and dry. Singapore, within 1.4° of
the equator with neither a thermal cycle nor a dry season, gets `year_round`,
which says there is no season rather than inventing one. Calling a Lagos
December "winter" would describe nothing *and* would pull it into a
northern-winter cohort in every seasonal aggregate.

Mutation-checked: making the southern branch identical to the northern (a
global month lookup) fails immediately, and so does reaching for the
four-season macro instead of the regime-aware `season_for()`.

### The leap-year trap, and the key that avoids it

`day_of_year` is 1–366, so **1 March is day 60 in a common year and 61 in a
leap year**:

```
2024-02-28   doy 59   common 59   md 02-28
2024-02-29   doy 60   common 59   md 02-29   ← leap day
2024-03-01   doy 61   common 60   md 03-01
2025-03-01   doy 60   common 60   md 03-01   ← same day, different doy
```

Grouping a thirty-year climatology by `day_of_year` therefore mixes 1 March
with 29 February and shifts every day after February by one in three years out
of four — a systematic error that reads as a seasonal signal.

So `month_day` (`MM-DD`) is the climatology join key: stable in every year, 366
distinct values, and 29 February simply has a quarter of the sample size —
which is true and worth knowing rather than hidden. `day_of_year_common` is
there for anything needing a contiguous numeric axis; 29 February shares day 59
with 28 February so nothing after it shifts.

A test asserts raw `day_of_year` takes **both** 60 and 61 for 1 March — the
trap stated as a fact rather than described in a comment.

## Licence

[MIT](LICENSE)
