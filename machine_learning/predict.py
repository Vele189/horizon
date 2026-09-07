"""Scores the current horizon and writes it to ``fact_ml_predictions``.

The table the Risk Horizon view reads, so its grain and its freshness
semantics are the deliverable as much as the numbers are.

**One row per city per forecast date.** ``forecast_date`` is the last day of
*observed* data the score was computed from, and the score covers the days
**after** it — day *t* is a feature, so a window including it would be scoring
the model on something it was handed. The window is carried in the row rather
than implied: ``horizon_start``, ``horizon_end``, ``horizon_days``. A reader of
one row should not need this docstring to know which seven days it is about.

**Re-scoring replaces.** The primary key is ``(city_id, forecast_date)`` and
the write is an upsert, so running twice for the same day leaves one row, not
two. ``model_version`` is deliberately *not* in the key: a dashboard asking
"what is the risk for this city right now" must get exactly one answer, and a
table keyed by model returns one row per model ever run and pushes the choice
into the presentation layer, where it is least visible.

**Nothing here computes a feature.** The matrix comes from ``features.py`` and
the model's own recorded feature list decides the columns and their order. The
two are checked against each other before a single row is written: if
``features.py`` has gained, lost or renamed a column since the model was
trained, this refuses to run rather than scoring a matrix the model has never
seen.

Usage::

    python machine_learning/predict.py                 # score and write
    python machine_learning/predict.py --dry-run       # score and print only
    python machine_learning/predict.py --dates 30      # a longer strip

The table is created on first run; the DDL is idempotent.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Any, Final, Sequence

import pandas as pd
from sqlalchemy import Engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.loader import engine_from_settings  # noqa: E402
from machine_learning.artifact import load_model  # noqa: E402
from machine_learning.baselines import metrics_path  # noqa: E402
from machine_learning.features import (  # noqa: E402
    WARMUP_DAYS,
    city_roster,
    feature_columns,
    load_features,
)
from machine_learning.labels import HORIZON_DAYS  # noqa: E402

__all__ = [
    "DEFAULT_DATES",
    "PREDICTIONS_TABLE",
    "PredictionError",
    "apply_schema",
    "coverage",
    "decision_threshold",
    "score_horizon",
    "write_predictions",
]

log = logging.getLogger(__name__)

PREDICTIONS_TABLE: Final[str] = "gold_marts.fact_ml_predictions"
SCHEMA_SQL: Final[Path] = Path(__file__).resolve().parent / "predictions.sql"

#: How many recent forecast dates to score per city. One would be enough for a
#: "current risk" tile; seven gives the dashboard a short strip so a reader can
#: see risk building or fading rather than only where it stands today.
DEFAULT_DATES: Final[int] = 7


class PredictionError(RuntimeError):
    """Raised when scoring would produce a row nobody should trust.

    The feature schema not matching the model's is the main one, and it is
    fatal rather than a warning: a matrix with a renamed column still has the
    right shape, and the model will return a probability for every row of it.
    """


def apply_schema(engine: Engine) -> None:
    """Create the table if it is not there. Idempotent, safe to repeat."""
    statements = SCHEMA_SQL.read_text()
    with engine.begin() as connection:
        connection.execute(text(statements))


def decision_threshold(metrics: Path | None = None) -> tuple[float, str]:
    """The operating point the evaluation reported, and which model it is for.

    Read from ``metrics.json`` rather than defaulted to 0.5. The label written
    beside every risk score has to mean something a reader can look up, and the
    only threshold with measured precision and recall behind it is the one
    chosen on validation in ML-06. At 0.5 this model flags nothing at all.

    Raises:
        PredictionError: No metrics file, or no evaluation recorded in it.
    """
    path = Path(metrics) if metrics is not None else metrics_path()
    if not path.exists():
        raise PredictionError(
            f"{path} does not exist, so there is no evaluated operating point "
            "to label predictions at. Run the pipeline: baselines.py --write, "
            "train.py --write, evaluate.py --write."
        )
    payload = json.loads(path.read_text())
    evaluation = payload.get("evaluation")
    if not evaluation:
        raise PredictionError(
            f"{path} records no evaluation, so the decision threshold is "
            "unknown. Run `python machine_learning/evaluate.py --write`."
        )
    variant = payload.get("model", {}).get("recommended_variant", "unweighted")
    wanted = f"model_{variant}"
    for row in evaluation["summary"]:
        if row["predictor"] == wanted:
            return float(row["threshold"]), variant
    raise PredictionError(
        f"{path} has no evaluation row for {wanted!r}; it has "
        f"{[row['predictor'] for row in evaluation['summary']]}."
    )


def score_horizon(
    engine: Engine | None = None,
    *,
    dates: int = DEFAULT_DATES,
    model=None,
    threshold: float | None = None,
    frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score the most recent ``dates`` forecast dates for every scorable city.

    Returns the rows to write and a coverage record saying which cities were
    scored and, for the rest, why not — a city missing from the output is a
    fact about the backfill or about its climatology, and a table that simply
    lacks the row says neither.

    Raises:
        PredictionError: The feature schema does not match the model's.
    """
    owned = engine is None and frame is None
    if engine is None and frame is None:
        engine = engine_from_settings()
    try:
        loaded = model if model is not None else load_model()
        expected = list(loaded.features)
        current = list(feature_columns())
        if expected != current:
            raise PredictionError(
                "the feature schema has moved since the model was trained.\n"
                f"  model expects {len(expected)}: {expected}\n"
                f"  features.py provides {len(current)}: {current}\n"
                "Retrain before scoring: a matrix with a renamed or reordered "
                "column still has the right shape, and the model will return a "
                "probability for every row of it."
            )

        roster: list[str] = []
        ingested: list[str] = []
        if engine is not None:
            roster, ingested = city_roster(engine)
        if frame is None:
            latest = _latest_observation(engine)
            # Padded by the warm-up so the earliest scored date has its full
            # history, then trimmed — the same trap as everywhere else in this
            # workstream: a window measured against the request, not the data.
            start = latest - pd.Timedelta(days=dates + WARMUP_DAYS + 5)
            frame = load_features(engine, start=start)
    finally:
        if owned and engine is not None:
            engine.dispose()

    if frame.empty:
        return pd.DataFrame(), coverage(frame, frame, roster, ingested)

    scorable = frame.loc[~frame["is_warmup"] & ~frame["has_missing_feature"]]
    recent = (
        scorable.sort_values(["city_id", "date_key"])
        .groupby("city_id", sort=True)
        .tail(dates)
        .reset_index(drop=True)
    )
    if recent.empty:
        return pd.DataFrame(), coverage(frame, recent, roster, ingested)

    if threshold is None:
        threshold, _ = decision_threshold()
    risk = loaded.predict(recent)

    forecast_date = recent["date_key"].dt.date
    predictions = pd.DataFrame(
        {
            "city_id": recent["city_id"].astype(str),
            "forecast_date": forecast_date,
            "horizon_start": forecast_date + dt.timedelta(days=1),
            "horizon_end": forecast_date + dt.timedelta(days=HORIZON_DAYS),
            "horizon_days": HORIZON_DAYS,
            "risk_score": risk.astype(float),
            "prediction_label": risk >= threshold,
            "decision_threshold": float(threshold),
            "model_version": loaded.path.stem,
            "model_variant": _variant_of(loaded),
            "feature_count": len(loaded.features),
        }
    )
    return predictions, coverage(frame, recent, roster, ingested)


def _variant_of(loaded) -> str:
    if loaded.metadata and loaded.metadata.get("variant"):
        return str(loaded.metadata["variant"])
    # model-<variant>-v<n>-<fingerprint>
    parts = loaded.path.stem.split("-")
    return parts[1] if len(parts) > 2 else "unknown"


def _latest_observation(engine: Engine) -> pd.Timestamp:
    with engine.connect() as connection:
        latest = connection.execute(
            text("select max(date_key) from gold_marts.fact_weather_observations")
        ).scalar()
    if latest is None:
        raise PredictionError("fact_weather_observations is empty; nothing to score.")
    return pd.Timestamp(latest)


def coverage(
    frame: pd.DataFrame,
    scored: pd.DataFrame,
    roster: Sequence[str] = (),
    ingested: Sequence[str] = (),
) -> dict[str, Any]:
    """Every city in the registry, and for each unscored one, which reason.

    Named reasons rather than one gap. "Not in the output" is not an answer: a
    city can be absent because it has never been ingested, because it has no
    climatology baseline and therefore a null feature, because it is still
    inside its warm-up, or because its record stops before the scored window —
    and those are four different things to do something about. The ticket asks
    for output verified across all fifteen cities, and five of them can be
    scored today; the report is where the other ten say why not.
    """
    present = set(frame["city_id"].unique()) if len(frame) else set()
    scored_cities = sorted(scored["city_id"].unique()) if len(scored) else []
    known = set(roster) | present
    landed = set(ingested) if ingested else present

    reasons: dict[str, str] = {}
    for city_id in sorted(known - set(scored_cities)):
        if city_id not in landed:
            reasons[city_id] = "never ingested"
            continue
        rows = frame.loc[frame["city_id"] == city_id] if len(frame) else frame
        if not len(rows):
            reasons[city_id] = "no observations inside the scored window"
            continue
        recent = rows.sort_values("date_key").tail(1)
        if bool(recent["is_warmup"].iloc[0]):
            reasons[city_id] = "inside the feature warm-up"
        elif bool(recent["has_missing_feature"].iloc[0]):
            reasons[city_id] = "a feature is null — no climatology baseline"
        else:
            reasons[city_id] = "no rows in the scored window"

    return {
        "roster": sorted(known),
        "scored": scored_cities,
        "not_scored": reasons,
    }


def write_predictions(engine: Engine, predictions: pd.DataFrame) -> int:
    """Upsert on ``(city_id, forecast_date)``. Re-running replaces.

    One statement per row inside one transaction rather than a bulk COPY: this
    is at most a few hundred rows, the upsert is what makes the run repeatable,
    and a COPY cannot express it without a staging table that would have to be
    cleaned up on failure.
    """
    if predictions.empty:
        return 0
    statement = text(
        f"""
        insert into {PREDICTIONS_TABLE} (
            city_id, forecast_date, horizon_start, horizon_end, horizon_days,
            risk_score, prediction_label, decision_threshold,
            model_version, model_variant, feature_count, scored_at
        ) values (
            :city_id, :forecast_date, :horizon_start, :horizon_end, :horizon_days,
            :risk_score, :prediction_label, :decision_threshold,
            :model_version, :model_variant, :feature_count, now()
        )
        on conflict (city_id, forecast_date) do update set
            horizon_start      = excluded.horizon_start,
            horizon_end        = excluded.horizon_end,
            horizon_days       = excluded.horizon_days,
            risk_score         = excluded.risk_score,
            prediction_label   = excluded.prediction_label,
            decision_threshold = excluded.decision_threshold,
            model_version      = excluded.model_version,
            model_variant      = excluded.model_variant,
            feature_count      = excluded.feature_count,
            scored_at          = now()
        """
    )
    rows = predictions.to_dict("records")
    for row in rows:
        row["risk_score"] = float(row["risk_score"])
        row["prediction_label"] = bool(row["prediction_label"])
        row["horizon_days"] = int(row["horizon_days"])
        row["feature_count"] = int(row["feature_count"])
    with engine.begin() as connection:
        connection.execute(statement, rows)
    return len(rows)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the current horizon into fact_ml_predictions."
    )
    parser.add_argument(
        "--dates",
        type=int,
        default=DEFAULT_DATES,
        help=f"Forecast dates to score per city (default {DEFAULT_DATES}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Score and print, write nothing."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    engine = engine_from_settings()
    try:
        loaded = load_model()
        threshold, variant = decision_threshold()
        print(f"model    {loaded.path.name}")
        print(f"variant  {variant}   threshold {threshold:.4f}   "
              f"features {len(loaded.features)}")

        predictions, report = score_horizon(
            engine, dates=args.dates, model=loaded, threshold=threshold
        )
        if predictions.empty:
            print("\nnothing scorable.")
            return 1

        print(f"\n{len(predictions):,} rows, "
              f"{predictions['city_id'].nunique()} cities, "
              f"{predictions['forecast_date'].min()} .. "
              f"{predictions['forecast_date'].max()}")

        latest = predictions.sort_values("forecast_date").groupby("city_id").tail(1)
        view = latest.loc[
            :, ["city_id", "forecast_date", "horizon_start", "horizon_end",
                "risk_score", "prediction_label"]
        ].sort_values("risk_score", ascending=False)
        view["risk_score"] = view["risk_score"].map("{:.4f}".format)
        print("\nlatest forecast per city")
        print(view.to_string(index=False))

        print(
            f"\ncoverage: {len(report['scored'])} of {len(report['roster'])} "
            "registry cities scored"
        )
        for city_id, reason in report["not_scored"].items():
            print(f"  {city_id:<14} {reason}")

        if args.dry_run:
            print("\n(dry run — nothing written)")
            return 0

        apply_schema(engine)
        written = write_predictions(engine, predictions)
        print(f"\nwrote {written:,} rows to {PREDICTIONS_TABLE}")
        with engine.connect() as connection:
            total = connection.execute(
                text(f"select count(*) from {PREDICTIONS_TABLE}")
            ).scalar()
        print(f"{PREDICTIONS_TABLE} now holds {total:,} rows")
    finally:
        engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
