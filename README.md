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
`api_elevation_m` alongside the exact `source_url`, so provenance survives an
edit to `cities.yml`. `cell_selection=land` is sent explicitly — it is the
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
twice. `source_url` is reproduced by `request_url()`, which prepares the
identical string through `requests` rather than assembling it by hand — tested
against a live response.

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

### Measured storage, and the column that dominates it

| | Rows | Bytes/row | Projected full table |
|---|---|---|---|
| `observations_daily` | 173 505 | 1 020 | **177 MB** |
| `observations_hourly` | 263 160 | 746 | **196 MB** |

Two things that measurement turned up, both now reported by
`--report` rather than left to be rediscovered:

- **`pg_total_relation_size` counts dead tuples.** The first hourly reading was
  31 MB against a true 12 MB — the table had been loaded and cleared several
  times during development. The report names the dead share when it exceeds a
  tenth and says to `VACUUM FULL`.
- **`source_url` is 83% of the daily row payload and 81% of the hourly one** —
  712 and 466 bytes, the same handful of distinct strings repeated once per
  row. Of bronze's projected ~373 MB, roughly **300 MB is that one column**.
  Without it bronze would be about 62 MB.

Bronze never leaves the local container, so this threatens nothing today — but
it is worth stating plainly, because the proposal budgets ~250–300 MB *across
all layers* and bronze alone exceeds that. The column is also fully redundant:
`request_url()` reconstructs it exactly from `(city_id, grain, start, end)`,
which is what replay already relies on. Whether to keep the denormalised copy
for one-glance replayability or derive it is a schema decision, recorded here
rather than made unilaterally.

## Licence

[MIT](LICENSE)
