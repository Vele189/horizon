# Ingestion reconciliation: `hourly`

Generated 2026-09-07 01:26 UTC by `python ingestion/reconcile.py --grain hourly`.

Range **2024-09-02 .. 2026-09-02**, 17,544 observations expected per city across 15 cities.

## Per city

| city | expected | actual | distinct | delta | gaps |
|---|---:|---:|---:|---:|---:|
| `auckland` | 17,544 | 17,544 | 17,544 | 0 | 0 |
| `buenos_aires` | 17,544 | 17,544 | 17,544 | 0 | 0 |
| `cairo` | 17,544 | 26,328 | 17,544 | 0 | 0 |
| `delhi` | 17,544 | 17,544 | 17,544 | 0 | 0 |
| `johannesburg` | 17,544 | 17,544 | 17,544 | 0 | 0 |
| `lagos` | 17,544 | 17,544 | 17,544 | 0 | 0 |
| `london` | 17,544 | 26,328 | 17,544 | 0 | 0 |
| `moscow` | 17,544 | 17,544 | 17,544 | 0 | 0 |
| `phoenix` | 17,544 | 17,544 | 17,544 | 0 | 0 |
| `portland` | 17,544 | 17,544 | 17,544 | 0 | 0 |
| `reykjavik` | 17,544 | 17,544 | 17,544 | 0 | 0 |
| `sao_paulo` | 17,544 | 17,544 | 17,544 | 0 | 0 |
| `singapore` | 17,544 | 17,544 | 17,544 | 0 | 0 |
| `sydney` | 17,544 | 17,544 | 17,544 | 0 | 0 |
| `tokyo` | 17,544 | 17,544 | 17,544 | 0 | 0 |

`actual` counts every row held for the city; `distinct` counts distinct observation times inside the range, which is what `delta` compares against `expected`. The two diverge for two separate reasons, reported separately below: the same observation landed twice (legal, since bronze is append-only and silver deduplicates), or the row falls outside the range this report asked about.

**Rows outside the reconciled range**, a surplus, not a gap:

| city | rows outside |
|---|---:|
| `cairo` | 5,880 |
| `london` | 5,880 |

Real observations the range did not ask about, usually a wider window ingested earlier. They do not count towards `delta` and are not a defect; downstream models filter by range.

## Gaps by category

Every gap longer than 3 days, categorised.

| category | gaps | observations | meaning |
|---|---:|---:|---|
| archive boundary | 0 | 0 | Outside what ERA5 can serve. Nothing can fill it. |
| not ingested | 0 | 0 | Inside the servable range; the backfill has not reached it. |
| api limitation | 0 | 0 | A completed unit recorded fewer rows than its window. |
| unexplained | 0 | 0 | Fetched, recorded complete, and missing anyway. |

### archive boundary

None.

Accepted. The ERA5 archive begins 1940-01-01 and trails the present by several days; the planner already stops short of the edge, so anything here is a range that was asked for outside those bounds.

### not ingested

None.

Accepted while the backfill is in progress. The daily grain costs ~26 000 weighted API calls against a free-tier allowance of 10 000 a day, so it completes across roughly three days. Every range here is pending, not lost; the manifest resumes rather than restarts.

### api limitation

None.

Structurally prevented rather than merely absent: the client asserts the returned row count against the requested range before parsing, and the loader asserts it again before writing. A short response raises instead of landing.

### unexplained

None.

Nothing should ever land here. A gap in this category means a completed unit covers a range with no rows to show for it.

## Verdict

**No gap is unexplained.** Every one is either outside what the archive can serve, or inside a range the backfill has not reached yet, and the latter shrinks to nothing as the backfill completes.

