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

## Gold: `fact_weather_observations`

One row per city per UTC day, with foreign keys to `dim_cities` and `dim_date`.
Materialised as a table with a **unique** index on `(city_id, date_key)` and a
second on `date_key` alone.

Measures: temperature min/max/mean and apparent equivalents, dew point,
precipitation / rain / snowfall (both cm and mm) / precipitation hours, wind
speed mean and max, gusts, direction, surface and sea-level pressure, humidity,
cloud cover, radiation, and the WMO code — plus the grid cell that answered and
the ingestion lineage.

### It selects from silver and does not join the dimensions

An inner join to `dim_cities` would enforce referential integrity by
**dropping** any row it could not match, and a fact silently missing a city is
indistinguishable from a city with no weather. The `relationships` tests assert
the same property and fail loudly instead.

Verified by planting an orphan `atlantis` row: it trips both the foreign-key
test *and* the row-count-against-silver test, and both go green when it's
removed.

### The index earns its place

```
Bitmap Heap Scan on fact_weather_observations
  ->  Bitmap Index Scan on (city_id, date_key)
        Index Cond: city_id = 'london' AND date_key between …
```

That's the shape the dashboard issues — one city, one date range — over a
serverless connection where a sequential scan of 173 520 rows is the difference
between a usable dashboard and a slow one.

### Row count is asserted against silver, not a constant

The proposal's ~164 000 assumed a round thirty years; the configured range runs
1995-01-01 to the archive edge, so the completed figure is **15 × 11 568 =
173 520**. The backfill runs across days, so a hardcoded number would be red
for most of its life and would teach everyone to ignore it. The test asserts
the fact carries exactly as many rows as silver, plus a per-city check that any
*completed* city has no gap between its first and last day.

Currently **60 396 rows across 9 cities** — the remaining six are still
backfilling.

## Gold: `fact_weather_hourly`

One row per city per hour for the trailing 24 months — **263 160 rows**, 15
cities, 2024-09-02 → 2026-09-02. Feeds the Storm Dynamics view and nothing
else.

### The window is derived, not pinned

Silver holds 274 920 hourly rows; this table holds 263 160. The difference is
`cairo` and `london`, which carry an extra eight months from an ING-03
archival sample that took a calendar year rather than the anchored window.
Unfiltered they would sit in a table documented as trailing-24-months, and a
per-city average "over the window" would cover a different window per city.

### Pressure tendency uses a `RANGE` frame, not `lag(n)`

`lag(pressure, 3)` counts **rows**, not hours. One missing hour makes it reach
four hours back and report the result as a three-hour change — a fabricated
storm signal from data that merely had a hole. Demonstrated on a five-row
fixture with 03:00 removed:

```
ts       p        lag(3)   range 3h
04:00    1004.0      4.0        3.0   ← lag spans 4 hours and calls it 3
```

`RANGE BETWEEN INTERVAL '3 hours' PRECEDING AND INTERVAL '3 hours' PRECEDING`
asks for the reading exactly three hours earlier and returns null when there
isn't one. Silver has no gaps today; the correct form costs nothing and stays
correct if it ever does.

Computed on **sea-level** pressure, not surface — surface pressure carries the
grid cell's elevation, so a tendency on it would compare Johannesburg's 822 hPa
against London's 1013 the moment anything aggregated across cities.

### It finds real storms

A tendency can be arithmetically right and still meaningless, so it was checked
against the weather. The six deepest 24-hour falls in the table are **all
Reykjavík** — the North Atlantic storm track — reaching **−41 hPa/24h** against
the ≈−24 hPa that defines explosive cyclogenesis, with sea-level pressure down
to 949.6 hPa. The correlation between 24-hour tendency and gust strength is
−0.109: weak, but correctly signed.

Nulls are exactly 3 and 24 per city — the start of each series and nowhere
else.

### Gold against the Neon budget

| object | rows | total | bytes/row |
|---|---:|---:|---:|
| `fact_weather_hourly` | 263 160 | **52.0 MB** | 198 |
| `fact_weather_observations` | 60 396 | 14.2 MB | 235 |
| `dim_date` | 11 938 | 1.4 MB | 121 |
| `dim_city_season`, `dim_cities` | 195 | 0.1 MB | |
| **total** | | **67.8 MB** | |

That is **13.6% of Neon's 500 MB free plan** today, and **~94 MB (19%)** once
the daily backfill completes. Gold is the only layer promoted to Neon, so
unlike the bronze measurement this budget is a real constraint rather than a
yardstick.

The hourly fact is the largest object in the warehouse, which is exactly why it
is capped at 24 months: thirty years at this grain would be over four million
rows and, at 198 bytes each, would not fit in the allowance at all. A test
asserts that arithmetic rather than restating the claim.

## Gold: leakage-safe climatology

Mean and standard deviation of daily temperature, per city per calendar day,
smoothed over ±7 days, **computed excluding the year being labelled**.

### What the leakage costs, measured

A normal computed over all years includes the very day it is about to label:
the observation contributes to its own μ and inflates its own σ by its own
deviation. Every Z-score comes out too small.

| | leakage-safe | leaky |
|---|---:|---:|
| shift in μ from the exclusion | 0.037 °C | — |
| shift in σ | −0.07% | — |
| **days labelled \|Z\| > 2.5** | **976** | **740** |

The per-day shift is invisible in a spot check. It changes **a quarter of the
extreme-day labels**, because the shift is small everywhere and the events live
in the tail where small shifts decide membership. A model trained on the leaky
labels is scoring against a target that has already seen its own answer, and
its metrics come out flattering.

Left as `climatology_exclude_own_year` (default true), so the leaky variant is
built deliberately for comparison rather than reached by accident — and a test
asserts that setting it false really does produce the leaky one, so a misread
var cannot quietly ship the wrong thing under the right label.

### How it is computed

Leave-one-out over 31 reference years would mean re-aggregating each window
once per excluded year. Instead each (city, day, source year) contributes
`n`, `Σx`, `Σx²`, and the exclusion is a subtraction, with
σ² = (Σx² − (Σx)²/n)/(n−1) recovering the deviation.

That identity is easy to get subtly wrong and the result still looks like a
number, so it is **cross-checked against Postgres's own `stddev_samp`** on the
no-exclusion case: μ agrees exactly, σ to 2×10⁻¹⁵.

### ±7 days, and the circle

A single day's normal rests on ~30 observations, one per year, and at that
sample size σ is noise. The window gives 15 calendar days × ~30 years ≈ **455
observations**. It is circular — 1 January draws on 25 December through
8 January — because a non-circular window would build the year's first and last
weeks from half the data, exactly where the northern winter extremes sit.

Measured on `climatology_day`, the day-of-year a date *would* have in a leap
year, because raw `day_of_year` gives 31 December two different numbers.

### Leap day

Keyed on `month_day`, so 29 February is its own row rather than colliding with
1 March. Its **own** sample is a quarter the size — eight leap years in
thirty-two — but its **window** is full, drawn from 22 February to 7 March in
every year. A test asserts the leap-day window is within 20% of 28 February's;
an implementation that filtered the window to leap years would show a quarter.

### σ is never zero, and null means something

No σ is zero — 455 observations across a fortnight cannot be identical.

**1 098 rows have a null σ**: 3 cities × 366 days, for `london`, `reykjavik`
and `sydney` — each of which currently holds a *single* year, from the ING-03
archival samples. Leave-one-year-out removes their only year and leaves
nothing. Null is the honest answer; falling back to the all-years value would
silently reintroduce the exact leakage the model removes. A test asserts nulls
appear **only** where the reference period is one year.

### The σ sanity check, and a correction to it

| city | within-window σ | σ of all days pooled |
|---|---:|---:|
| lagos | **0.631** | 1.329 |
| singapore | 0.704 | **0.873** |
| delhi | 2.148 | 6.977 |
| cairo | 2.269 | — |
| phoenix | 3.221 | 9.163 |

**Singapore is the least variable city — on the pooled σ.** On the
*within-window* σ, Lagos is slightly lower. Both measurements are correct and
they answer different questions: Lagos has a 3.4 °C seasonal swing against
Singapore's 1.6 °C, but is marginally steadier *around* that curve.

The climatology needs the second quantity, because a Z-score should measure
departure from the seasonal normal, not from the annual mean. Both orderings
are asserted, each against the σ it is actually about.

Moscow has not finished backfilling, so the "largest σ" half of the check
**skips rather than passes** — a check that silently passes on absent data is
worse than one that says it is waiting. Phoenix leads so far at 3.221 °C,
which is what a desert with a large seasonal swing should look like.

## Gold: Z-score anomaly flags

`(observed − μ) / σ` against the leakage-safe baseline, flagged on **`abs(z)`**
past 2.5, with a `hot` / `cold` / `none` direction.

### Both tails, or half the signal

`z > 2.5` reads naturally and silently discards every cold extreme. Phoenix
would lose **125 of its 145** flagged days. A separate test asserts the
*direction* matches the sign, because an inverted branch flags exactly the
right days and labels every one backwards — which every count-based test
passes, and which puts Moscow's January in the heatwave column.

| | days | rate |
|---|---:|---:|
| scored city-days | 59 301 | |
| anomalies | 976 | **1.65%** |
| hot | 524 | 0.88% |
| cold | 452 | 0.76% |

1.24% is the normal-distribution expectation; real residuals have fatter tails.
The band the ticket asks for is 0.5–3%.

### Unknown is not "ordinary"

1 095 city-days have a null Z — the three cities holding a single reference
year, where leave-one-year-out leaves nothing. Their flags are **null, not
`none`**. Calling them ordinary would assert it on no evidence *and* pad the
denominator of every anomaly rate with days that could never have been flagged.

### Three outliers, each investigated

| city | rate | hot / cold | why |
|---|---:|---|---|
| tokyo | 3.90% | 30 / 27 | baseline of **45** not 455 — only 4 years backfilled, so σ is noisy. sd(Z) = 1.14 where every complete city is 1.00 |
| cairo | 2.01% | **208 / 25** | most right-skewed residuals in the set, **+0.59** |
| phoenix | 1.25% | **20 / 125** | the only left-skewed city, **−0.39** — desert heat has a radiative ceiling, cold outbreaks are sharp |

Each is asserted as an *explanation* rather than tolerated as an exception: the
high-rate test requires that any city above 3% has a small baseline and
over-disperses, and the skew test requires a skewed city to lean the direction
its skew predicts.

`sd(Z) ≈ 1.00` for every full-record city, which is the check that catches a σ
computed over the wrong window or grouping — all of those still produce a
plausible column of numbers.

### A warming trend runs through every city

`corr(year, Z)` is positive everywhere: **+0.05** (Delhi) to **+0.38** (Lagos),
with Cairo's hot anomalies averaging year 2016.5 against 2007.0 for its cold
ones.

That is real signal, not artefact — the baseline spans the whole reference
period, so a trending series produces hot anomalies late and cold early. But it
means the flag currently conflates *"unusual for this day of year"* with
*"warmer than the thirty-year mean because the climate has warmed"*. **ML-05
will need to make that distinction deliberately**, so it is recorded here
rather than discovered there.

### Moscow

The ticket's key check. Moscow has **not backfilled yet** — it is 8th in city
order and the daily grain is quota-bound across days. I re-pointed the backfill
driver to fetch Moscow first.

Its test **skips with a reason rather than passing**. A check that silently
passes on absent data is worse than one that says it is waiting, and this is
the specific check the ticket names — a false green here would be the worst
kind.

## VALIDATION GATE — not passed

`tests/test_validation_gate.py` is the automated form of DBT-11. It reads the
seven dated events from `config/cities.yml` rather than restating them, so a
change to an event date is a change to a test.

**It is currently red, and it should be.** None of the seven events can be
checked: their cities have not backfilled. The daily grain costs ~26 000
weighted API calls against a free-tier allowance of 10 000 a day, and 164 of
480 units have landed.

| city | event | status |
|---|---|---|
| portland | 2021-06-28, PNW heat dome | 0 years ingested |
| moscow | 2010-07-29, Russian heat wave | 0 years |
| sao_paulo | 2021-07-30, Antarctic cold wave | 0 years |
| buenos_aires | 2022-01-11, Southern Cone heat | 0 years |
| london | 2022-07-19, first UK 40 °C | 1 year |
| sydney | 2020-01-04, Black Summer | 1 year |
| tokyo | 2018-07-23, Japan heat wave | 4 years |

A missing city **skips with a reason and does not pass**, and a separate test
fails while *any* event is unverifiable — so the ticket cannot close on a green
suite that quietly checked nothing. That is the specific failure this gate
exists to prevent, wearing the costume of success. The backfill driver has been
re-pointed to fetch these seven cities first.

### The two events the ticket names that the registry does not carry

The checklist asks for Phoenix (July 2023) and Delhi (29 May 2024). Neither is
a configured validation event, and both are fully backfilled — so both were
checked anyway. **Neither flags**, and the investigation says why rather than
the threshold being lowered until they do.

**Phoenix, July 2023.** Peak Z = **+1.97**, zero flagged days in the 31-day
streak. Ruled out in turn:

- *the warming trend* — the July window mean moved −0.08 °C over the record
- *the measure* — Z(max) is +1.77, **lower** than Z(mean)
- *inflated σ* — 2.94 at that date against a 3.22 annual mean, so lower
- *the data* — 2023 is **rank 1 of 32** for July days ≥ 43.3 °C, 23 against a
  next-best 17

Phoenix in July is always about 46 °C, so no single day of the streak departs
far from its own seasonal normal. What was unprecedented is how long it lasted,
and **a single-day Z-score cannot express duration by construction**. A rolling
31-day mean-Z was tried and does not fix it either — March 2026 scores higher
than July 2023 on that measure. This is a limit of the detector, not a defect
in the climatology.

**Delhi, 29 May 2024.** Z = **+2.22**, under the threshold — and proportionate:
in this grid cell 2024 was the *second*-warmest late May in thirty-two years,
behind 1998. A detector that called the second-warmest such day a 2.5σ extreme
would be miscalibrated.

Both findings are recorded as executable tests, including a guard that fires if
Phoenix ever *does* clear the threshold, so the reasoning gets revisited rather
than silently invalidated.

## Lineage

![dbt lineage: bronze source through silver staging and intermediate to gold marts](docs/images/lineage.png)

```bash
python dbt_analytics/dbt_env.py -- dbt docs generate
python dbt_analytics/render_lineage.py     # regenerates docs/images/lineage.svg
```

`dbt build` is green end to end: **190 nodes, PASS=190, WARN=0, ERROR=0**, and
no deprecation warnings. Every model, source and seed carries a description,
and **every column in every layer is documented** — not just gold.

Shared column descriptions live in `models/_docs.md` as dbt doc blocks.
`city_id` appears in nine models, and nine copies is nine chances to drift — a
test asserts every one resolves to the same text, which is how the five
different descriptions it had accumulated were found.

### The diagram is generated, not screenshotted

[`render_lineage.py`](dbt_analytics/render_lineage.py) lays the DAG out by
layer from `target/manifest.json`. dbt's own docs site draws a force-directed
graph, which is fine to explore and poor to read at a glance: the layers are
the whole point of a medallion architecture and a force layout does not show
them. SVG stays crisp at any size, diffs as text, and needs no browser; the PNG
beside it is rasterised for the README. A test regenerates the SVG and fails if
it differs, so the picture cannot go stale.

### Drawing it found a real defect

`int_climatology_contributions` selected from `fact_weather_observations` and
`dim_date` — both marts. The lineage ran **staging → marts → intermediate →
marts**, with arrows pointing backwards into the intermediate column.
Everything built and all 190 nodes passed; the graph was simply not a layering
anyone could follow.

It now reads `stg_observations_daily` and derives the calendar columns it needs
from a `climatology_day_of()` macro shared with `dim_date`, so the two
derivations cannot drift — and a drift would have been quiet, since both would
still produce a number between 1 and 366. Two tests now enforce it: no model
may depend on a later layer, and the intermediate layer may read only staging.

That is the argument for this ticket in one example. The DAG was correct,
tested, and unreadable, and only drawing it made the difference visible.

## Feature matrix

`machine_learning/features.py` turns the gold layer into **27 model inputs**,
one row per city-day. 60 396 rows across the 9 cities backfilled so far —
exactly the row count of `fact_weather_observations`, because nothing is
dropped.

### The as-of rule, and where the label has to start

A row dated *t* uses observations on days **≤ t**, day *t* included. The daily
aggregate is complete at the end of the day, and discarding it would throw
away the most informative value available.

That puts an obligation on ML-02: **the label runs t+1 .. t+7, never
t .. t+6.** `anomaly_days_trailing30` counts today's flag, so a label window
that also starts today hands the model its own answer. The convention is
written into the module docstring and asserted by the test that keeps
`is_anomaly` out of `feature_columns()`.

### Proving a window cannot see forward

Reading the formulas is not proof. The central test rewrites every day *after*
a cut point — temperature +40 °C, pressure −60 hPa, every anomaly flag
inverted — rebuilds, and asserts every row at or before the cut is
bit-identical. Any window that reaches forward by a day moves those rows,
whatever it is called and however it is written. Four cut points, so a 30-day
peek is not invisible at a cut near the start.

A test that cannot fail proves nothing, so one of them builds a centred window
on the same data and asserts the check catches it. Which is also the honest
measure of what leakage looks like from the outside:

| 7-day mean temperature | corr with T+3 |
|---|---:|
| trailing, t−6 .. t | 0.9419 |
| centred, t−3 .. t+3 | **0.9714** |

Not a red flag. A modest, entirely plausible improvement — which is exactly
why this is caught structurally rather than noticed in a metric.

### Windows are calendar windows, not row windows

`rolling(7)` counts *rows*. Over a series with a hole that is eight calendar
days, reported as seven — a fabricated number from data that merely had a gap,
and the same trap `fact_weather_hourly` avoids with a `RANGE` frame. Each city
is reindexed onto a contiguous daily calendar before anything is shifted, so a
row offset *is* a day offset and a window spanning a hole is null.

All nine cities are contiguous today, so the reindex changes nothing. It costs
one pass and stays right if that stops being true.

### The trailing Z excludes the day it scores

`temperature_2m_mean_z_trailing30` standardises today against the **30 days
before it**, not the 30 days ending on it. With *t* inside its own window it
pulls the mean 1/30 of the way towards itself and inflates σ by its own
deviation — the same self-labelling `fact_climatology` excludes a whole year to
avoid, one window smaller. It is not a rounding difference:

| baseline | mean \|Z\| | sd | days \|Z\| > 2.5 | > 3.0 |
|---|---:|---:|---:|---:|
| t−30 .. t−1 — used | 1.051 | 1.304 | **2 882** | **1 131** |
| t−29 .. t — self-included | 0.981 | 1.190 | 1 439 | 365 |

Self-inclusion **halves the extremes**, and every one it removes is a day the
classifier most needs to see.

`sd(Z) = 1.30` rather than 1.00 is by design and not a defect: a 30-day local
baseline does not remove the seasonal cycle, so a day in a fast-warming month
sits well above the month behind it. That is what this feature is *for* —
"unusual against recent conditions". `z_temperature_2m_mean`, carried
alongside, is the seasonally corrected companion.

### Pressure tendency is a daily proxy, and says so

The 24h and 72h tendencies are day-mean-to-day-mean changes in sea-level
pressure, not the instantaneous tendency `fact_weather_hourly` carries. Against
that sharper measure at 12Z, over the 3 650 city-days where both exist:

- correlation **0.958**
- sd 1.71 hPa daily against 1.98 hPa hourly — the daily mean smooths the peak

The hourly fact covers **6.0%** of the matrix: 24 months against thirty years.
A feature that is null for 94% of rows is not a feature.

### Nulls are flagged, not dropped

Every rolling statistic requires its full window (`min_periods == window`). A
30-day mean over 11 days is a different statistic, and letting it into the same
column makes a feature's meaning depend on how far into the series its row
sits.

- **270 warm-up rows** — 9 cities × 30 days, 0.45% of the matrix — present and
  flagged `is_warmup`. Removing them is `drop_warmup()`, which the caller has
  to say out loud; a feature module that quietly shortens the record hands the
  trainer a row count that does not match the warehouse's.
- **1 005 rows outside the warm-up** carry a null feature. All of them are
  `z_temperature_2m_mean`, and all of them are London, Reykjavík and Sydney —
  the three cities holding a single reference year, where leave-one-year-out
  leaves nothing to score against. A test asserts that count against the
  warehouse and fails on *any* other unexplained null.

`is_warmup` and `has_missing_feature` are separate columns because they answer
different questions. A hole in a series produces nulls far outside the warm-up;
a fully-scored row inside it is still unusable.

### Unknown is not a quiet month

`anomaly_days_trailing30` counts flagged days; `anomaly_days_scored30` counts
how many of the 30 carried a flag at all. Folding a null flag into "not an
anomaly" would report a quiet month that was never measured — and 1 008 rows
have `scored30 = 0`, so a single coerced column would have shown three cities
with a perfect anomaly-free record they never earned.

The denominator is deliberately **not** a model input. It is a fact about how
far the backfill has got, and a classifier allowed to learn from it learns
which cities are half-ingested.

### It finds a real event

21.6% of rows have at least one anomaly day behind them. The highest count in
the set is Delhi's **23 of 30**, in the window ending 2002-08-02 — every one
hot, mean Z **+3.23**. That is the July 2002 monsoon failure, and it is the
kind of month the forward-window label exists to predict.

### Day-of-year, and the leap-year phase shift

The cyclical encoding runs off a 365-day axis with 29 February folded onto 28,
because raw `day_of_year` numbers every day after February one higher in a leap
year — a one-day phase shift in the sin/cos pair, in three years out of four,
which a model reads as a real difference between leap and common years.

That is `dim_date.day_of_year_common` recomputed in Python, so `build_features`
stays a pure function of the rows it is handed and can be tested on forty
synthetic days without a database. Two derivations are only safe while
something asserts they agree, so a test checks it against every date in
`dim_date` — the same arrangement `dim_cities.hemisphere` has.

### What is deliberately still leaky

`z_temperature_2m_mean` and `is_anomaly` come from `fact_weather_anomalies`,
whose baseline excludes the observation's own year but not the years *after*
it. A 2003 row is scored against a climatology that has seen 2020.

It is a per-(city, day-of-year) constant rather than a path from the future to
any particular day, and the alternative — an expanding climatology using only
prior years — would give the early record a baseline of two or three years and
a σ far too noisy to score against. The trade is deliberate, and it is recorded
here rather than found later; `temperature_2m_mean_z_trailing30` is the
strictly-backward companion for exactly this reason.

## Target label

`machine_learning/labels.py` builds the binary target: **does an anomaly occur
on any day in t+1 .. t+7?** It is a separate module from `features.py` for one
reason — this is the only place in the project where looking forward is
correct, and a forward shift cannot be added to the feature builder by accident
if the single deliberate one lives somewhere else.

### The window is t+1 .. t+7, and both ends are load-bearing

**t is excluded** because it is a feature. `anomaly_days_trailing30` counts
today's flag, so a label window that also started today would hand the
classifier its own answer through a column that looks entirely innocent.

**t+7 is included** because "within a week" is seven days. An off-by-one at
either end fails nothing on its own — it produces a slightly different positive
rate and a model quietly answering a different question.

So the boundary is asserted rather than described. One anomaly is dropped into
an otherwise quiet series, and the test requires it to label **exactly** the
seven rows before it:

```
day       23 24 25 26 27 28 29 [30] 31
flag       .  .  .  .  .  .  .   X   .
label      1  1  1  1  1  1  1   0   0
           └──────── t+1..t+7 ───┘
```

Day 30 is negative on its own anomaly, and day 22 is negative too. Both ends
are also covered by a parametrised sweep over offsets 0 through 9.

### A positive is certain; a negative has to be earned

| label | when |
|---|---|
| `True` | at least one day in the window is flagged |
| `False` | no day is flagged **and all seven are scored** |
| `<NA>` | otherwise — the window cannot be closed |

The third row is where the last seven days of every series go, and they go
there by the same rule as everything else rather than by a special case: at the
end of the record the window runs off the edge, fewer than seven days are
scored, and a negative cannot be earned. The three cities with no climatology
baseline land there too — all 365 days of each — because a day that could never
be flagged cannot make a week quiet.

It also means a city ending in an anomalous week loses fewer than seven rows,
which is not an exception but the same rule read the other way:

| city | last day | tail rows lost | why |
|---|---|---:|---|
| cairo | quiet | 7 | nothing to see, window cannot close |
| lagos | anomalous 31 Aug | 2 | earlier rows are positive on a window that never closes |
| singapore | anomalous 2 Sep | 1 | the anomaly is the final day |

**1 126 of 60 396 rows are unlabelled** — 1 095 for the three unscored cities,
31 in the tails. `drop_unlabelled()` is a separate call, like `drop_warmup()`.

### The positive rate is 6.93%, and the ticket expected 3–6%

59 270 labelled city-days, **4 110 positive**. The gap from the expected band is
accounted for rather than shrugged at.

Under independence a daily rate *p* gives a weekly rate of 1−(1−p)⁷. Anomalies
clump, so the observed rate is always below that, and the ratio between them is
what the window construction actually controls:

| | daily | independent 7-day | observed | ratio |
|---|---:|---:|---:|---:|
| pooled | 1.64% | 10.95% | **6.93%** | 0.633 |

974 anomaly days fall in 587 runs — mean run 1.66 days, longest 12 — which is
the clustering that ratio measures.

Now feed the same arithmetic the number the 3–6% expectation was drawn from. A
normal distribution puts 1.24% of days past 2.5σ; carried through the window
and the clustering, that is **5.3%** — inside the band. The entire excess is
that real residuals have fatter tails than a normal, which
`fact_weather_anomalies` already measured at 1.65% a day against that
theoretical 1.24%. The label is not wide; the tails are fat.

That decomposition is the test, not a paragraph: it asserts the Gaussian rate
lands in 3–6% and the observed one in 3–10%.

### Rates run 4.6% to 15.3%, and the spread is one number

| city | daily | independent | observed | ratio |
|---|---:|---:|---:|---:|
| tokyo | 3.92% | 24.42% | **15.27%** | 0.625 |
| cairo | 2.02% | 13.28% | 8.71% | 0.656 |
| lagos | 1.61% | 10.73% | 7.63% | 0.712 |
| singapore | 1.67% | 11.11% | 7.54% | 0.678 |
| phoenix | 1.25% | 8.46% | 5.16% | 0.610 |
| delhi | 1.38% | 9.29% | 4.58% | 0.493 |

Every city sits between **0.49 and 0.71** of its own independence bound. The
threefold spread in positive rate is the spread in daily anomaly rates and
nothing else — the window behaves the same everywhere. Tokyo leads because only
four of its years have backfilled, so its σ is noisy and it flags 3.9% of days,
which is the `fact_weather_anomalies` finding arriving intact rather than a new
problem. A test asserts the ratio band per city, so a city that ever clusters
differently fails rather than blending into an average.

Six cities carry labels and five of them are hot climates. No mid-latitude city
has backfilled, so the pooled rate is not yet representative of the fifteen-city
set and will move when it is.

### The base rate is not stationary, and ML-03 needs to know now

The proposal splits chronologically — train to 2018, validate to 2021, test
after. The positive rate is not the same in those three periods:

| period | rows | positives | rate |
|---|---:|---:|---:|
| train 1995–2018 | 45 284 | 2 491 | **5.50%** |
| validate 2019–2021 | 5 480 | 464 | **8.47%** |
| test 2022–2026 | 8 506 | 1 155 | **13.58%** |

It roughly doubles, then doubles again. This is the warming trend expressed
through a climatology whose baseline spans the whole record — the positive
corr(year, Z) already found in every city, landing directly on the target.

A model trained at one base rate and scored at another is miscalibrated before
it starts, and the proposal asks for a Brier score and a calibration curve. So
this is recorded as a test that **fails if the shift disappears**, rather than
as a note: the reasoning gets revisited rather than silently invalidated.

### No feature can reconstruct the label

Exact reconstruction is the wrong measure. On floating-point columns every
value is unique, so "some function maps this column to the label" is true of
all of them and the check passes vacuously. Rank AUC asks the question that
matters — could this column alone order the city-days with every positive
first — and answers 1.0 for a leaked label, 0.5 for noise, and is invariant to
any monotone transform, so a leak cannot escape by being logged or negated.

| feature | AUC |
|---|---:|
| `anomaly_days_trailing30` | **0.643** |
| `z_temperature_2m_mean` | 0.557 |
| `elevation_m` | 0.454 |
| `temperature_2m_mean_roll7_var` | 0.527 |
| everything else | within 0.012 of 0.5 |

Nothing is close to reconstruction. The strongest is the persistence signal the
proposal names as baseline one — a city that has been anomalous lately is more
likely to be anomalous next week — and it is the thing the model has to beat,
not a leak.

`elevation_m` at 0.454 is worth naming: it is constant per city, so it is not
measuring elevation but *which city*, and city rates run 4.6% to 15.3%. A model
given static geography will learn base rates from it. That is legitimate and
useful, and it is also why per-city evaluation is going to matter more than a
pooled score.

Two tests keep this honest. One asserts the maximum stays under 0.90 — the
reconstruction bound — and under 0.75, a regression guard with deliberate
headroom over the measured 0.643. The other proves the check can fail, by
scoring a copied label (1.000) and a count taken over the label's own window
(the shape a stray `shift(-1)` would produce) against noise.

The construction makes this structural rather than lucky: the label is a
function of the anomaly flag alone. Temperature and pressure are not inputs to
it, and a test rewrites both — +60 °C, pressure negated — and requires the label
frame to come back identical.

## Baselines, fixed in advance

`machine_learning/baselines.py` scores two baselines and writes them to
`machine_learning/artifacts/metrics.json`, **committed before any model is
trained**. A PR-AUC with nothing beside it is not a result: average precision
for a random ranker is the positive rate, and the positive rate here is 5.51%
in the training period and 13.58% in the test one, so the same 0.20 is strong on
one and poor on the other.

| | rows | positives | base rate | PR-AUC | lift | Brier |
|---|---:|---:|---:|---:|---:|---:|
| no-skill reference | 8 506 | 1 155 | 13.58% | 0.1358 | 1.00× | 0.12386 |
| **persistence** | 8 506 | 1 155 | 13.58% | **0.2293** | **1.69×** | **0.11462** |
| **climatology** | 8 506 | 1 155 | 13.58% | 0.1514 | 1.12× | 0.12454 |

Test split, both fitted on train only. **The model has to beat 0.2293.**

### The split is purged, not just cut

Train to 2018, validate 2019–2021, test from 2022 — the proposal's split. But
cutting on the date alone is not enough: the label at *t* is an anomaly in
t+1 .. t+7, so a row dated 2018-12-31 is labelled by days that belong to
validation. Seven rows per city per boundary is a rounding error in row count
and not one in principle — it is the training set being told what happened next.

`PURGE_DAYS` drops them, so train ends 2018-12-24. The test that matters runs
the split with the purge **off** and requires the disjointness check to raise,
so the purge is a fact rather than an intention.

### Persistence: a rule, calibrated

The rule is the proposal's — an anomaly next week if one occurred this week —
but a rule emits a flag, and a flag has no Brier score worth having: 0 and 1
make every mistake maximally confident. So the rule is calibrated on the
training split and the baseline emits the resulting probability. The ranking is
unchanged, so PR-AUC is the rule's own; only the calibration is fixed.

| training cell | rows | P(anomaly next week) |
|---|---:|---:|
| after an anomalous week | 2 486 | **19.99%** |
| after a quiet week | 42 583 | 4.66% |
| week could not be judged | 0 | — |

A 4.3× separation, and it holds up out of sample in every city — 1.15× in
Phoenix to 2.27× in Delhi.

That empty third cell is not decoration. It held **36 rows** until the signal
was moved to be computed before the population is trimmed rather than after.
The seven-day window was falling off the start of the *slice* instead of the
start of the record — the same mistake as measuring a warm-up against the
request rather than the data, for the third time in this workstream. It is now
computed once, on the whole record, and `build_metrics` refuses a population
that arrives without it.

### Climatology: the week-of-year signal does not survive the split

This is the interesting one. Fitted and scored **inside** the training period,
the (city, week-of-year) climatology is worth 2.47× no-skill, so the seasonal
structure is real. Carried across the split it is worth **less than nothing**:

| | lift |
|---|---:|
| train, in-sample | 2.47× |
| validation | 0.93× |
| test | **0.91×** |
| test, fitted on test (the ceiling) | 2.53× |

The last row is the point. The test period has just as much (city, week)
structure as the training period — **it is different structure**. The anomaly
mix flips:

| | cold | hot |
|---|---:|---:|
| train 1995–2018 | 347 | 224 |
| test 2022–2026 | 61 | **231** |

Hot extremes fall in different weeks than cold ones, so a climatology fitted on
a cold-dominated era points at the wrong weeks for a hot-dominated one. It is
not stale in a way more data fixes: refitting on train *and* validation still
only reaches 1.18×.

So validation shrinks the week term away entirely. The pseudo-count is chosen on
validation Brier over a grid that **runs to infinity**, and infinity is what it
picks:

| pseudo-count | 0 | 20 | 100 | 500 | 2 000 | ∞ |
|---|---:|---:|---:|---:|---:|---:|
| validation Brier | 0.08116 | 0.08084 | 0.08014 | 0.07942 | 0.07921 | **0.07913** |
| validation PR-AUC | 0.0792 | 0.0780 | 0.0782 | 0.0816 | 0.0870 | **0.0994** |

The first version of this grid stopped at 100 and reported a "tuned" value that
was simply its own edge — validation Brier was still falling there. A parameter
chosen at the boundary of its grid is clipped, not tuned, and a test now
requires the grid to reach its limit.

At the limit every week cell collapses to its city's own rate, and the surviving
baseline is a per-city base rate at 1.12×. The limit is *computed* rather than
approached, because at a large finite pseudo-count the week term survives as a
rounding-sized perturbation that still breaks ties — and it breaks them the
wrong way, costing 0.0275 of PR-AUC against the exact collapse.

This is recorded as a test that fails if the raw week climatology ever ranks
above random out of sample, so the finding gets revisited rather than quietly
invalidated.

### What ML-04 should take from this

- **Beat 0.2293.** That is persistence on test, and it is not a weak opponent.
- Seasonal features are informative and **non-stationary**. `day_of_year_sin` /
  `cos` carry real signal — the in-sample ceiling is 2.53× — but the mapping
  from season to anomaly risk has changed within the record. A model that fits
  it hard on 1995–2018 will be fitting a regime that has gone.
- The base rate moves from 5.51% to 13.58% across the split, so **every
  probability trained on the early record is systematically low**. The proposal
  asks for a calibration curve; this is what it will show.

### The target is fixed against a snapshot, and the file says which

`metrics.json` records the row count, city list and last date it was computed
on — 59 090 rows across 6 cities to 2026-09-01. The backfill is not finished,
and when more cities land these numbers change.

So the test that compares the committed file against a fresh run **skips with a
reason** when the snapshot has moved, naming the command to rebuild it. A test
that silently passed on a rebuilt file would defeat the point of committing one;
a test that failed on every new city would be noise. The file is not wrong when
it goes stale, it is stale — and it has to be rebuilt and re-committed *before*
a model is compared against it, or "fixed in advance" quietly stops being true.

Two more guards on the file itself: the payload must be strict JSON, since the
infinite pseudo-count would otherwise be written as a bare `Infinity` that
Python reads back happily and no other parser accepts; and two runs over the
same rows in different orders must agree to the last digit, which they did not
until `build_metrics` sorted its input — a Brier score is a mean over a float
array, and a mean is summation-order dependent in its final ULP.

## Chronological split harness

`machine_learning/evaluation.py` cuts the labelled frame strictly by time. Run
`python machine_learning/evaluation.py` for the report:

| split | start | end | rows | positives | base rate |
|---|---|---|---:|---:|---:|
| train | 1995-01-31 | 2018-12-24 | 45 069 | 2 483 | **5.51%** |
| validation | 2019-01-01 | 2021-12-24 | 5 445 | 464 | **8.52%** |
| test | 2022-01-01 | 2026-09-01 | 8 506 | 1 155 | **13.58%** |

One function, `split_frame()`, and nothing splits inline. Not because splitting
is hard — because a split written inline is a split written twice, and the
second one is where the shuffle gets in.

### Three checks, weakest to strongest

**Ordered.** Every split ends strictly before the next begins:
`max(train) < min(validation)`, the assertion the ticket names. This is the one
a shuffle breaks, and the error message says so — a random split puts 2023 rows
in training, and nothing else in the pipeline would notice.

**Disjoint.** No *label window* crosses a boundary either. Strictly stronger,
and it runs the ordering check first. A split can be perfectly ordered and
still hand the training set the first week of validation through the label,
which is why train ends 2018-12-24 and not 2018-12-31:

| boundary | last | first | gap | label reaches | clears |
|---|---|---|---:|---|---|
| train → validation | 2018-12-24 | 2019-01-01 | 8 days | 2018-12-31 | yes |
| validation → test | 2021-12-24 | 2022-01-01 | 8 days | 2021-12-31 | yes |

**Confirmed end to end.** Every observation from 2019-01-01 onwards is replaced
with nonsense — temperature +40 °C, pressure −60 hPa, every anomaly flag
inverted — features and labels are rebuilt from scratch, the split is re-cut,
and the training split must come back **bit-identical**. That assertion does
not inspect how any window is written, so a window that reaches forward fails
it however cleverly it is expressed.

The same test with the purge switched off fails, and it is asserted to fail —
on exactly the last seven rows. That is what makes the first two worth running.

### Backward reach across a boundary is deployment, not leakage

The asymmetry is the part worth getting right, and it is easy to get wrong in
the safe-looking direction.

A validation row on 2019-01-05 has a 30-day rolling mean built from December
2018 — training data. **That is correct.** A model predicting that day in
production has all of 2018 behind it, and blanking it here would measure a
system nobody is going to run. Leakage is a *training* row reading forwards,
which the backward-only features and the purge between them already rule out.

So there is a test that asserts the backward reach **exists**: rewrite 2018 and
require the opening of validation to move. It is there so nobody later "fixes"
the harness into reporting a worse number for a better-sounding reason. The
same test pins the other end — past the longest window the rows are identical
again, so the reach is bounded and known.

### The embargo is off, and that was measured rather than assumed

There is a real concern hiding under the correct one. The last training rows
and the first validation rows share some of the same days inside their windows,
so the two sets are mildly correlated and the score mildly optimistic. That is
the standard argument for an *embargo* at the start of a split — sample
independence, not leakage.

`split_frame(embargo_days=...)` implements it, `EMBARGO_DAYS` is 0, and the
reason is a number rather than a preference:

| embargo | test rows | base rate | persistence PR-AUC | lift |
|---:|---:|---:|---:|---:|
| 0 days | 8 506 | 13.58% | 0.2293 | 1.69× |
| 7 days | 8 471 | 13.63% | 0.2297 | 1.68× |
| 30 days | 8 356 | 13.73% | 0.2305 | 1.68× |
| 60 days | 8 206 | 13.81% | 0.2347 | 1.70× |

Holding back a month moves test PR-AUC by **0.0012, upward** — the opposite
direction from the optimism an embargo removes. The correlation is not there at
this window length, so paying for it in realism would buy nothing. A test
fails if that stops being true, and `metrics.json` records `embargo_days: 0`
so a file says not only what trim was applied but what was deliberately not.

### `train_test_split` is absent, and something checks

A test scans **every `.py` file in the repository** — not just the ML package,
because the failure it guards against is a quick train/test split appearing in
a dashboard script where nobody would think to look. It forbids
`sklearn.model_selection` wholesale rather than function by function:
everything in it shuffles except `TimeSeriesSplit`, and a project that needs
that one should reach for it deliberately and delete the line.

A second test plants the forbidden import in a temporary file and requires the
scan to catch it, because a check that passes by finding nothing is otherwise
indistinguishable from a check that looks nowhere.

## The model

`machine_learning/train.py` fits an XGBoost classifier on the training split,
tunes it on validation, and scores it against the baselines that were committed
before it existed.

| test split | PR-AUC | lift | Brier | mean predicted |
|---|---:|---:|---:|---:|
| no-skill reference | 0.1358 | 1.00× | 0.12386 | — |
| climatology baseline | 0.1514 | 1.12× | 0.12454 | — |
| persistence baseline | 0.2293 | 1.69× | 0.11462 | — |
| **model, weighted** *(as specified)* | 0.3303 | 2.43× | 0.22529 | 0.478 |
| **model, unweighted** *(recommended)* | **0.3494** | **2.57×** | **0.11019** | 0.070 |

**Both variants beat both baselines.** The recommended one beats persistence by
52% on PR-AUC, and it beats it in every city individually — a test asserts that,
because a pooled win can be one city carrying five:

| city | base rate | model | persistence |
|---|---:|---:|---:|
| phoenix | 5.7% | 0.2244 (3.97×) | 0.0651 (1.15×) |
| delhi | 5.2% | 0.3055 (5.90×) | 0.1178 (2.27×) |
| lagos | 15.6% | 0.4420 (2.84×) | 0.3362 (2.16×) |
| singapore | 23.7% | 0.4008 (1.69×) | 0.3013 (1.27×) |
| cairo | 17.8% | 0.3609 (2.03×) | 0.2421 (1.36×) |

### `scale_pos_weight` is oversampling, and the ticket asked for it to avoid oversampling

The ticket specifies `scale_pos_weight` **rather than** resampling, on the
grounds that SMOTE and undersampling distort the predicted probabilities the
Risk Horizon view shows a reader directly. The premise is half right — those
methods do distort probabilities — but the conclusion does not follow.
Weighting the positive class by *k* is arithmetically the same operation as
oversampling it *k*-fold. It distorts the probabilities the same way, for the
same reason.

At the observed ratio of **17.15**, here is what the dashboard would be
showing. Test-set deciles, predicted against observed:

| decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| weighted, predicted | .391 | .419 | .430 | .443 | .455 | .467 | .484 | .505 | .552 | .632 |
| weighted, observed | .097 | .040 | .057 | .101 | .094 | .077 | .105 | .167 | .204 | .417 |
| unweighted, predicted | .026 | .030 | .033 | .036 | .040 | .045 | .052 | .068 | .103 | .270 |
| unweighted, observed | .061 | .046 | .071 | .076 | .061 | .102 | .120 | .169 | .220 | .431 |

The weighted model's *quietest* decile is told to a reader as a 39% chance of
an extreme week; it happens 10% of the time. Its mean prediction is **0.478
against a base rate of 0.136**.

So both are trained and both are recorded. The weighted model is the one the
ticket specifies and the one saved to `model.joblib`; the unweighted one is
what the ticket's own stated goal asks for, and on this data it does not even
trade ranking for calibration — **it is better on both axes**. The
recommendation is chosen on validation by the same rule as everything else, and
a test pins the inflation so it cannot quietly stop being true.

**No resampling anywhere.** A scan over every `.py` file in the repository
forbids `imblearn`, the oversamplers and `sklearn.utils.resample`, with a
companion test that plants one in a temp file and requires the scan to catch it
— the same arrangement as the split scan.

### The calibration failure ML-02 predicted

Look at the unweighted row again. It is *under*-confident: it predicts 0.070
where 0.136 happens, roughly half, and the shortfall runs through every decile.

That is not a defect in the model. It is the non-stationary base rate this
project recorded two tickets ago as a test: the positive rate is 5.51% in the
training period and 13.58% in the test one, so a model fitted on the early
record is correctly calibrated to a world that has since warmed. The ranking is
sound — observed risk rises monotonically across the deciles — and the level is
not.

It was written down before the model existed, so it arrives as a confirmation
rather than a surprise. **The Risk Horizon view should not print these
probabilities raw.** A recalibration fitted on validation would fix the level
without touching the ranking; that is a dashboard decision and it is not in
this ticket, but it should not be discovered from a screenshot.

### Tuned on validation, and there is no third argument

`tune()` takes a training frame and a validation frame. Test cannot be passed
to it. Twelve combinations — depth, learning rate, minimum child weight — with
the number of rounds chosen by early stopping on validation average precision.

The grid is deliberately small. Its top four candidates sit within 0.005 PR-AUC
of each other on 5 445 validation rows holding 464 positives, which is already
inside the noise; a larger search would be choosing between differences smaller
than the number it is choosing on, and that is how a validation split gets
overfitted without anyone touching test. The full search is recorded in
`metrics.json`.

The structural argument is backed by a behavioural one: a test rewrites every
label in the test split, retunes, and requires identical parameters, identical
round count, and identical predictions.

### What the model actually leans on

| feature | gain |
|---|---:|
| `z_temperature_2m_mean` | 0.143 |
| `latitude` | 0.077 |
| `anomaly_days_trailing30` | 0.075 |
| `elevation_m` | 0.054 |
| `temperature_2m_mean_roll30_mean` | 0.044 |

The top two substantive features are the climatological anomaly state and the
persistence count — the model is beating the persistence baseline partly by
using it, which is the expected shape.

**`latitude` and `elevation_m` together are 13.1% of the gain**, and they are
constants per city. The model is not learning about latitude; it is learning
*which city*, and city base rates run 5.2% to 23.7%. That is legitimate and
useful with six cities in the set, and it will not transfer to a city the model
has not seen. Worth knowing before anyone points this at a sixteenth city.

### Reproducible, and checked in a fresh process

The seed is fixed at 42 and the thread count at **one**. That second one is not
caution: XGBoost's histogram builder is deterministic for a given thread count,
not across thread counts — the per-thread gradient sums are added in whatever
order the threads finish, and floating-point addition is not associative. A
four-core laptop and a sixteen-core runner produce two different models. The
whole search takes nine seconds on 45 069 rows by 27 columns, so a thread count
that does not depend on the machine costs nothing worth having.

The test runs the entire training path **twice in separate interpreters** and
compares the metrics byte for byte, because an in-process repeat cannot see the
failure worth catching. A companion test changes the seed and requires the
model to change, so the seed is known to be doing something.

### The target has to be the one that was fixed in advance

`train.py` refuses to record a result unless the committed baselines were
computed on the same warehouse snapshot. If a city has backfilled since, the
baselines must be re-run and re-committed first — a model scored against a
target that has moved is not being measured. Two tests exercise the refusal.

The reverse holds too: re-running `baselines.py --write` keeps the recorded
model block only while the snapshot still matches, and drops it with a warning
when it does not, rather than leaving stale model scores sitting beside fresh
baselines inviting a comparison nobody made.

`model.joblib` is gitignored; `metrics.json` is not. That is the right way
round — a binary nobody can diff is not evidence, and the numbers are.

## Licence

[MIT](LICENSE)
