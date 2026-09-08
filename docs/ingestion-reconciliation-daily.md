# Ingestion reconciliation: `daily`

Generated 2026-09-07 01:26 UTC by `python ingestion/reconcile.py --grain daily`.

Range **1995-01-01 .. 2026-09-02**, 11,568 observations expected per city across 15 cities.

## Per city

| city | expected | actual | distinct | delta | gaps |
|---|---:|---:|---:|---:|---:|
| `auckland` | 11,568 | 0 | 0 | -11,568 | 1 |
| `buenos_aires` | 11,568 | 0 | 0 | -11,568 | 1 |
| `cairo` | 11,568 | 365 | 365 | -11,203 | 1 |
| `delhi` | 11,568 | 11,568 | 11,568 | 0 | 0 |
| `johannesburg` | 11,568 | 0 | 0 | -11,568 | 1 |
| `lagos` | 11,568 | 11,568 | 11,568 | 0 | 0 |
| `london` | 11,568 | 365 | 365 | -11,203 | 2 |
| `moscow` | 11,568 | 0 | 0 | -11,568 | 1 |
| `phoenix` | 11,568 | 366 | 366 | -11,202 | 2 |
| `portland` | 11,568 | 0 | 0 | -11,568 | 1 |
| `reykjavik` | 11,568 | 365 | 365 | -11,203 | 2 |
| `sao_paulo` | 11,568 | 0 | 0 | -11,568 | 1 |
| `singapore` | 11,568 | 11,933 | 11,568 | 0 | 0 |
| `sydney` | 11,568 | 365 | 365 | -11,203 | 2 |
| `tokyo` | 11,568 | 0 | 0 | -11,568 | 1 |

`actual` counts every row held for the city; `distinct` counts distinct observation times inside the range, which is what `delta` compares against `expected`. The two diverge for two separate reasons, reported separately below: the same observation landed twice (legal, since bronze is append-only and silver deduplicates), or the row falls outside the range this report asked about.

**7 cities absent entirely:** `tokyo`, `portland`, `moscow`, `sao_paulo`, `johannesburg`, `buenos_aires`, `auckland`

## Gaps by category

Every gap longer than 3 days, categorised.

| category | gaps | observations | meaning |
|---|---:|---:|---|
| archive boundary | 0 | 0 | Outside what ERA5 can serve. Nothing can fill it. |
| not ingested | 16 | 136,990 | Inside the servable range; the backfill has not reached it. |
| api limitation | 0 | 0 | A completed unit recorded fewer rows than its window. |
| unexplained | 0 | 0 | Fetched, recorded complete, and missing anyway. |

### archive boundary

None.

Accepted. The ERA5 archive begins 1940-01-01 and trails the present by several days; the planner already stops short of the edge, so anything here is a range that was asked for outside those bounds.

### not ingested: 16 gap(s)

Accepted while the backfill is in progress. The daily grain costs ~26 000 weighted API calls against a free-tier allowance of 10 000 a day, so it completes across roughly three days. Every range here is pending, not lost; the manifest resumes rather than restarts.

| city | from | to | days |
|---|---|---|---:|
| `auckland` | 1995-01-01 | 2026-09-02 | 11,568 |
| `buenos_aires` | 1995-01-01 | 2026-09-02 | 11,568 |
| `cairo` | 1996-01-01 | 2026-09-02 | 11,203 |
| `johannesburg` | 1995-01-01 | 2026-09-02 | 11,568 |
| `london` | 1995-01-01 | 1997-12-31 | 1,096 |
| `london` | 1999-01-01 | 2026-09-02 | 10,107 |
| `moscow` | 1995-01-01 | 2026-09-02 | 11,568 |
| `phoenix` | 1995-01-01 | 2015-12-31 | 7,670 |
| `phoenix` | 2017-01-01 | 2026-09-02 | 3,532 |
| `portland` | 1995-01-01 | 2026-09-02 | 11,568 |
| `reykjavik` | 1995-01-01 | 2002-12-31 | 2,922 |
| `reykjavik` | 2004-01-01 | 2026-09-02 | 8,281 |
| `sao_paulo` | 1995-01-01 | 2026-09-02 | 11,568 |
| `sydney` | 1995-01-01 | 2020-12-31 | 9,497 |
| `sydney` | 2022-01-01 | 2026-09-02 | 1,706 |
| `tokyo` | 1995-01-01 | 2026-09-02 | 11,568 |

### api limitation

None.

Structurally prevented rather than merely absent: the client asserts the returned row count against the requested range before parsing, and the loader asserts it again before writing. A short response raises instead of landing.

### unexplained

None.

Nothing should ever land here. A gap in this category means a completed unit covers a range with no rows to show for it.

## Verdict

**No gap is unexplained.** Every one is either outside what the archive can serve, or inside a range the backfill has not reached yet, and the latter shrinks to nothing as the backfill completes.

