"""Writes `seeds/t_critical.csv`: Student-t quantiles by degrees of freedom.

**Why a table and not a formula.** The anomaly flag compares a departure with a
sigma that has been *estimated*, and at small baselines that estimate is noisy,
so the honest exceedance test is a t-interval rather than a normal one. Postgres
has no inverse-t, and the usual closed forms are worst exactly where this
matters: the Cornish-Fisher expansion is 7% low at 14 degrees of freedom and 10%
low at 4, which is the whole population this correction exists for. A table
generated from `scipy.stats.t` is exact, diffs as text, and can be checked
against any reference; an approximation is a page of magic constants nobody can
check by reading.

So the seed is a table of mathematical constants, in the same arrangement
`export_cities.py` uses for the city registry: generated, committed, and
asserted to match its source by a test.

**It is generated for one threshold.** The critical value depends on the tail
probability, and the tail probability comes from `anomaly_z_threshold`. The
threshold is written into the file as a column so it is self-describing, and a
dbt test refuses to build if the configured threshold and the seeded one have
drifted apart. ML-09's sweep re-flags in Python, against `scipy` directly, so it
is unaffected.

    python dbt_analytics/export_t_critical.py           # regenerate
    python dbt_analytics/export_t_critical.py --check   # exit 1 if stale
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

from scipy.stats import norm, t as student_t

SEED = Path(__file__).resolve().parent / "seeds" / "t_critical.csv"

FIELDS = ("degrees_of_freedom", "z_threshold", "critical_value")

#: The |Z| the project flags at, and the one this table is generated for. Kept
#: as a literal rather than imported from `machine_learning`, because dbt's
#: environment is not the model's and this script must run with neither pandas
#: nor xgboost installed.
Z_THRESHOLD = 2.5

#: Rows generated, one per degree of freedom.
#:
#: A thousand covers every baseline this project can produce: the window is
#: +/-7 days over ~31 years, so a complete city has about 460 observations, and
#: doubling the smoothing window would still fit. Above it the mart falls back
#: to the normal quantile, which by then differs from the t by under a tenth of
#: a percent, and a dbt test asserts no row actually takes that path.
MAX_DEGREES_OF_FREEDOM = 1000

#: Decimal places written. Twelve is far beyond what the comparison needs and
#: keeps the file exactly reproducible across platforms, which a full float
#: repr would not be.
PLACES = 12


def rows(threshold: float = Z_THRESHOLD) -> list[dict[str, object]]:
    """One row per degree of freedom, at the two-tailed probability of ``|Z| > threshold``.

    The flag is on both tails, so the reference probability is
    ``2 * (1 - Phi(z))`` and the critical value is the ``1 - alpha/2`` quantile
    of the t. At large degrees of freedom it converges on the threshold itself,
    which is the property that makes this a correction rather than a change of
    definition.
    """
    tail = 2.0 * norm.sf(threshold)
    return [
        {
            "degrees_of_freedom": df,
            "z_threshold": threshold,
            "critical_value": round(float(student_t.isf(tail / 2.0, df)), PLACES),
        }
        for df in range(1, MAX_DEGREES_OF_FREEDOM + 1)
    ]


def render(threshold: float = Z_THRESHOLD) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows(threshold))
    return buffer.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="Exit 1 if the seed is stale."
    )
    args = parser.parse_args(argv)

    rendered = render()
    if args.check:
        current = SEED.read_text(encoding="utf-8") if SEED.exists() else ""
        if current != rendered:
            print(f"{SEED} is stale; run `python {Path(__file__).name}`.")
            return 1
        print(f"{SEED} is current.")
        return 0

    SEED.parent.mkdir(parents=True, exist_ok=True)
    SEED.write_text(rendered, encoding="utf-8")
    print(f"wrote {SEED} ({MAX_DEGREES_OF_FREEDOM} rows at |Z| > {Z_THRESHOLD})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
