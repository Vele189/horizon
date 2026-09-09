"""A one-to-three-day wind gust nowcast, and the yardstick it is measured on.

`fact_weather_hourly` is the one complete mart in this warehouse: 263 160 rows,
fifteen cities, two years, fully reconciled with no gaps and no nulls. Two
years is far too short for the thirty-year anomaly model and exactly right for
a short-horizon nowcast, and the Storm Dynamics view already found the signal
this consumes -- the V-shaped relationship between pressure swing magnitude and
peak gust, rho = +0.31 pooled and +0.43 in Reykjavik.

**This is a separate model, not a replacement.** It has its own card
(`docs/nowcast-card.md`), its own artefacts, its own split boundaries and its
own target. The seven-day model answers "will a temperature anomaly occur";
this answers "how hard will the wind gust". Sharing a repository is the only
thing they share, and running them together would produce one number that meant
two things.

**The target.** `max(wind_gusts_10m)` over the next 24, 48 or 72 hours, from
each city-hour. A regression, because the quantity a reader wants is a speed
rather than a probability, and because two years does not contain enough
threshold exceedances per city to fit a classifier that is not mostly noise.

**Two yardsticks, because the ticket's one is weak where it matters.**
Persistence -- the peak gust over the *previous* window of the same length -- is
the required baseline, and at 24 hours it is a real opponent. At 48 and 72 it
is not: its RMSE is *worse* than simply predicting each city's mean, so beating
it there would be a claim about persistence and not about the model. A per-city
per-month climatology, fitted on training rows only, is reported beside it and
is the harder of the two at every horizon.

Usage::

    python machine_learning/nowcast.py             # fit and print
    python machine_learning/nowcast.py --write     # and write the metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.loader import engine_from_settings  # noqa: E402
from machine_learning.features import (  # noqa: E402
    HOURLY_REQUIRED,
    NOWCAST_HORIZONS,
    build_hourly_features,
    hourly_feature_columns,
)

__all__ = [
    "NOWCAST_HORIZONS",
    "SPLIT_BOUNDARIES",
    "Split",
    "baseline_frame",
    "ABLATION_GROUPS",
    "ablate",
    "build_nowcast",
    "climatology",
    "evaluate_horizon",
    "load_hourly",
    "purged_split",
    "skill",
]

log = logging.getLogger(__name__)

ARTIFACTS: Final[Path] = Path(__file__).resolve().parent / "artifacts"
METRICS_PATH: Final[Path] = ARTIFACTS / "nowcast-metrics.json"

SEED: Final[int] = 42

#: One thread, for the same reason `train.py` fixes it: XGBoost's histogram
#: builder is deterministic for a given thread count and not across them,
#: because per-thread gradient sums are added in whatever order the threads
#: finish and floating-point addition is not associative. `test_training.py`
#: requires bit-identical metrics across processes and this model is held to
#: the same rule.
N_JOBS: Final[int] = 1

MAX_ROUNDS: Final[int] = 1500
EARLY_STOPPING_ROUNDS: Final[int] = 50

FIXED_PARAMS: Final[Mapping[str, Any]] = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "n_jobs": N_JOBS,
}

#: Where the chronological folds break.
#:
#: **Training is exactly one year, and that is the whole reason for these
#: dates.** The hourly record runs 2024-09-02 to 2026-09-02. A model that has
#: seen eight months has never seen the season it is asked about; a full annual
#: cycle in training is the minimum for `day_of_year_sin` to mean anything, and
#: two years is not enough to have that *and* seasonally matched folds.
#:
#: So validation is the four months after training and test is the eight after
#: that, and they cover different seasons from each other. That is a real
#: limitation of a two-year record rather than a choice between better options,
#: and it is stated in the card rather than hidden behind a random split -- a
#: random split would be seasonally balanced, and would also let the model see
#: 3 p.m. to predict 4 p.m. on the same afternoon.
SPLIT_BOUNDARIES: Final[tuple[str, str]] = ("2025-09-02", "2026-01-01")

#: Origins kept for the non-overlapping check, by hour of day.
#:
#: Hourly origins give 24 overlapping targets a day: at a 72-hour horizon two
#: origins an hour apart share 71 hours of their answer. That is fine for
#: fitting -- more views of the same weather is still signal -- and overstates
#: the *precision* of a test metric, because the effective sample size is far
#: below the row count. Re-scoring on one origin a day is the check, and it is
#: reported beside the headline rather than instead of it.
NON_OVERLAPPING_HOUR: Final[int] = 0

HOURLY_SQL: Final[str] = f"""
    select {", ".join(HOURLY_REQUIRED)}
    from gold_marts.fact_weather_hourly
    order by city_id, observation_hour
"""


def load_hourly(engine: Engine) -> pd.DataFrame:
    """Every hourly observation. Small enough to hold: 263 160 rows."""
    frame = pd.read_sql(text(HOURLY_SQL), engine)
    frame["observation_hour"] = pd.to_datetime(frame["observation_hour"], utc=True)
    return frame


@dataclass(frozen=True)
class Split:
    """One horizon's chronological folds, already purged."""

    horizon_hours: int
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }


def purged_split(frame: pd.DataFrame, horizon_hours: int) -> Split:
    """Chronological folds with a gap the width of the horizon.

    An origin in the last ``horizon_hours`` of a fold has a target that reaches
    into the next one. Left in, the training rows nearest the boundary are
    answered by hours the validation fold is scored on, and the leak is small,
    real, and entirely invisible in any metric -- it makes the model look
    better on exactly the rows used to stop it early.

    The purge is dropped from the *end* of each fold rather than the start of
    the next, because the offending row is the one whose answer crosses over,
    and that row is the earlier one.
    """
    first, second = (pd.Timestamp(bound, tz="UTC") for bound in SPLIT_BOUNDARIES)
    gap = pd.Timedelta(hours=horizon_hours)
    hours = frame["observation_hour"]
    return Split(
        horizon_hours=horizon_hours,
        train=frame.loc[hours < first - gap],
        validation=frame.loc[(hours >= first) & (hours < second - gap)],
        test=frame.loc[hours >= second],
    )


def climatology(train: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Mean peak gust per city, and per city-month, from training rows only.

    Fitted on training rows and *only* training rows. A climatology computed
    over the whole record would be a baseline that had seen the test period,
    which is the failure this project checks for everywhere else; a weak
    baseline built that way flatters the model twice, by being wrong and by
    being unfairly informed.

    Two of them because they answer differently. The per-city mean says how
    windy a place is. The per-city-month mean adds the season, and it is the
    stronger opponent -- strong enough to beat the model outright in one city.
    """
    return (
        train.groupby("city_id")["peak_gust_ahead"].mean(),
        train.groupby(["city_id", "month"])["peak_gust_ahead"].mean(),
    )


def baseline_frame(part: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Attach every baseline's prediction to a scored fold."""
    by_city, by_city_month = climatology(train)
    keys = pd.MultiIndex.from_arrays([part["city_id"], part["month"]])
    return part.assign(
        baseline_persistence=part["persistence_gust"],
        baseline_climatology=part["city_id"].map(by_city).to_numpy(),
        # A city-month absent from training has no seasonal baseline; falling
        # back to that city's overall mean is honest, and falling back to the
        # global mean would quietly compare cities against each other.
        baseline_climatology_month=(
            pd.Series(keys.map(by_city_month), index=part.index)
            .astype(float)
            .fillna(part["city_id"].map(by_city))
            .to_numpy()
        ),
    )


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def skill(model_error: float, baseline_error: float) -> float:
    """Fractional error reduction: ``1 - model / baseline``.

    Positive means better than the baseline, zero means level, negative means
    worse. Reported rather than a bare RMSE because an RMSE of 8 km/h is good
    in Reykjavik and mediocre in Singapore, and a table of raw errors sorted by
    city is a table sorted by how windy the city is.
    """
    if baseline_error <= 0:
        return float("nan")
    return 1.0 - model_error / baseline_error


def _scores(part: pd.DataFrame, predicted: np.ndarray) -> dict[str, Any]:
    """Every metric for one fold or one city, model and baselines together."""
    actual = part["peak_gust_ahead"].to_numpy(dtype=float)
    model_rmse, model_mae = _rmse(actual, predicted), _mae(actual, predicted)
    scores: dict[str, Any] = {
        "rows": int(len(part)),
        "mean_observed": float(actual.mean()),
        "sd_observed": float(actual.std(ddof=1)) if len(part) > 1 else float("nan"),
        "rmse": model_rmse,
        "mae": model_mae,
    }
    for name in ("persistence", "climatology", "climatology_month"):
        reference = part[f"baseline_{name}"].to_numpy(dtype=float)
        scores[f"{name}_rmse"] = _rmse(actual, reference)
        scores[f"{name}_mae"] = _mae(actual, reference)
        scores[f"skill_vs_{name}"] = skill(model_rmse, scores[f"{name}_rmse"])
    return scores


def evaluate_horizon(
    frame: pd.DataFrame, horizon_hours: int
) -> tuple[dict[str, Any], XGBRegressor]:
    """Fit one horizon and score it against both baselines, pooled and per city."""
    split = purged_split(frame, horizon_hours)
    inputs = list(hourly_feature_columns())

    estimator = XGBRegressor(
        **FIXED_PARAMS,
        n_estimators=MAX_ROUNDS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    estimator.fit(
        split.train[inputs],
        split.train["peak_gust_ahead"],
        eval_set=[(split.validation[inputs], split.validation["peak_gust_ahead"])],
        verbose=False,
    )

    report: dict[str, Any] = {
        "horizon_hours": horizon_hours,
        "rows": split.sizes(),
        "window": {
            name: {
                "first": str(part["observation_hour"].min()),
                "last": str(part["observation_hour"].max()),
            }
            for name, part in (
                ("train", split.train),
                ("validation", split.validation),
                ("test", split.test),
            )
        },
        "best_iteration": int(estimator.best_iteration),
        "feature_count": len(inputs),
        "top_importances": _importances(estimator, inputs),
    }

    for name, part in (
        ("train", split.train), ("validation", split.validation), ("test", split.test)
    ):
        scored = baseline_frame(part, split.train)
        report[name] = _scores(scored, estimator.predict(part[inputs]))

    test = baseline_frame(split.test, split.train)
    predicted = estimator.predict(split.test[inputs])
    report["test_by_city"] = {
        str(city): _scores(group, predicted[mask])
        for city, group, mask in _by_city(test, split.test["city_id"])
    }

    # The same test rows, one origin a day. Not a different model and not a
    # different split: the same predictions, re-aggregated over origins whose
    # target windows do not overlap.
    standalone = test["observation_hour"].dt.hour == NON_OVERLAPPING_HOUR
    report["test_non_overlapping"] = _scores(
        test.loc[standalone], predicted[standalone.to_numpy()]
    )
    return report, estimator


def _by_city(frame: pd.DataFrame, cities: pd.Series):
    """City, its rows, and the boolean mask selecting them from the predictions."""
    for city in sorted(cities.unique()):
        mask = (cities == city).to_numpy()
        yield city, frame.loc[mask], mask


def _importances(estimator: XGBRegressor, inputs: Sequence[str]) -> list[dict[str, Any]]:
    """The ten features carrying the most gain, named."""
    gains = estimator.feature_importances_
    order = np.argsort(gains)[::-1][:10]
    return [
        {"feature": inputs[index], "gain": float(gains[index])} for index in order
    ]


#: Feature groups the ablation removes, one at a time.
#:
#: The ticket's premise is that the Storm Dynamics view found a signal --
#: pressure swing magnitude against peak gust -- that nothing consumed yet.
#: This measures whether consuming it helped, because the importance table says
#: it barely did and an importance table is not evidence. XGBoost's gain is
#: split among correlated features essentially arbitrarily, and every pressure
#: feature here is correlated with every wind feature through the weather that
#: produced both; the only honest way to ask "does this block carry anything"
#: is to remove it and refit.
ABLATION_GROUPS: Final[Mapping[str, tuple[str, ...]]] = {
    "pressure": (
        "pressure_msl",
        "pressure_tendency_3h",
        "pressure_tendency_24h",
        "abs_pressure_tendency_3h",
        "abs_pressure_tendency_24h",
        "pressure_min6",
        "pressure_min24",
        "pressure_min72",
        "pressure_range6",
        "pressure_range24",
        "pressure_range72",
    ),
    "recent_wind": (
        "wind_gusts_10m",
        "wind_speed_10m",
        "gust_max6",
        "gust_mean6",
        "speed_mean6",
    ),
    "seasonal": (
        "hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos",
    ),
}


def ablate(
    frame: pd.DataFrame, horizon_hours: int, group: str
) -> dict[str, Any]:
    """Refit without one feature group and report what it cost.

    Refitted rather than zeroed or permuted. Zeroing feeds the model a value it
    never saw in training; permuting breaks the correlation structure the trees
    were built on and charges the group for its neighbours' splits too. Removing
    the columns and fitting again asks the question actually being asked: what
    is this model worth without them.

    The split, the seed and the thread count are identical to the full fit, so
    the difference is the group and nothing else.
    """
    if group not in ABLATION_GROUPS:
        raise ValueError(f"unknown group {group!r}; have {sorted(ABLATION_GROUPS)}.")
    split = purged_split(frame, horizon_hours)
    kept = [
        column
        for column in hourly_feature_columns()
        if column not in ABLATION_GROUPS[group]
    ]

    estimator = XGBRegressor(
        **FIXED_PARAMS,
        n_estimators=MAX_ROUNDS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    estimator.fit(
        split.train[kept],
        split.train["peak_gust_ahead"],
        eval_set=[(split.validation[kept], split.validation["peak_gust_ahead"])],
        verbose=False,
    )
    return {
        "group": group,
        "dropped": list(ABLATION_GROUPS[group]),
        "feature_count": len(kept),
        # Validation as well as test, and validation is the one a decision may
        # be made on. An ablation read off test is a feature-selection step
        # performed with test labels, which is the thing this project refuses
        # everywhere else; without the validation column beside it, a negative
        # cost here would be an invitation to commit exactly that.
        "validation": _scores(
            baseline_frame(split.validation, split.train),
            estimator.predict(split.validation[kept]),
        ),
        "test": _scores(
            baseline_frame(split.test, split.train),
            estimator.predict(split.test[kept]),
        ),
    }


def build_nowcast(
    hourly: pd.DataFrame,
    horizons: Sequence[int] = NOWCAST_HORIZONS,
    *,
    ablations: Sequence[str] = (),
) -> dict[str, Any]:
    """Every horizon, fitted and scored. The whole record this writes."""
    report: dict[str, Any] = {
        "target": "max(wind_gusts_10m) over t+1 .. t+horizon_hours, km/h",
        "grain": "one row per city per hour",
        "cities": sorted(hourly["city_id"].unique()),
        "split_boundaries": list(SPLIT_BOUNDARIES),
        "non_overlapping_origin_hour": NON_OVERLAPPING_HOUR,
        "seed": SEED,
        "n_jobs": N_JOBS,
        "horizons": {},
    }
    for horizon in horizons:
        log.info("horizon %dh: building features", horizon)
        frame = build_hourly_features(hourly, horizon)
        scored, _ = evaluate_horizon(frame, horizon)
        if ablations:
            scored["ablations"] = {}
            for group in ablations:
                log.info("horizon %dh: ablating %s", horizon, group)
                scored["ablations"][group] = ablate(frame, horizon, group)
        report["horizons"][str(horizon)] = scored
    return report


def _horizon_table(report: Mapping[str, Any]) -> str:
    """The headline: one row per horizon, model against both baselines."""
    rows = []
    for horizon, scored in report["horizons"].items():
        test = scored["test"]
        rows.append(
            {
                "horizon": f"{horizon}h",
                "rows": f"{test['rows']:,}",
                "rmse": f"{test['rmse']:.2f}",
                "persist": f"{test['persistence_rmse']:.2f}",
                "clim_mo": f"{test['climatology_month_rmse']:.2f}",
                "vs_persist": f"{test['skill_vs_persistence']:+.1%}",
                "vs_clim_mo": f"{test['skill_vs_climatology_month']:+.1%}",
                "sd": f"{test['sd_observed']:.2f}",
            }
        )
    return pd.DataFrame(rows).to_string(index=False)


def _city_table(scored: Mapping[str, Any]) -> str:
    """One horizon, per city, sorted by the harder of the two skills."""
    rows = [
        {
            "city": city,
            "rmse": city_scores["rmse"],
            "persist": city_scores["persistence_rmse"],
            "clim_mo": city_scores["climatology_month_rmse"],
            "vs_persist": city_scores["skill_vs_persistence"],
            "vs_clim_mo": city_scores["skill_vs_climatology_month"],
        }
        for city, city_scores in scored["test_by_city"].items()
    ]
    frame = pd.DataFrame(rows).sort_values("vs_clim_mo")
    for column in ("rmse", "persist", "clim_mo"):
        frame[column] = frame[column].map("{:.2f}".format)
    for column in ("vs_persist", "vs_clim_mo"):
        frame[column] = frame[column].map("{:+.1%}".format)
    return frame.to_string(index=False)


def _ablation_table(report: Mapping[str, Any]) -> str:
    """What each feature group is worth, in RMSE, at each horizon."""
    rows = []
    for horizon, scored in report["horizons"].items():
        for group, result in scored.get("ablations", {}).items():
            row = {"horizon": f"{horizon}h", "dropped": group,
                   "features": result["feature_count"]}
            for fold in ("validation", "test"):
                full = scored[fold]["rmse"]
                without = result[fold]["rmse"]
                row[f"{fold[:3]}_full"] = f"{full:.3f}"
                row[f"{fold[:3]}_without"] = f"{without:.3f}"
                row[f"{fold[:3]}_cost"] = f"{(without - full) / full:+.2%}"
            rows.append(row)
    return pd.DataFrame(rows).to_string(index=False) if rows else "  (none)"


def write_metrics(report: Mapping[str, Any], path: Path = METRICS_PATH) -> Path:
    """Write the record the card is checked against.

    Separate from `metrics.json` rather than a key inside it. The two models
    have different targets, different splits and different baselines, and a
    single file would invite a reader -- or a test -- to compare a PR-AUC
    against an RMSE because they were adjacent.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the short-horizon gust nowcast and score it."
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(NOWCAST_HORIZONS),
        help=f"Horizons in hours (default {' '.join(map(str, NOWCAST_HORIZONS))}).",
    )
    parser.add_argument(
        "--ablate",
        nargs="*",
        choices=sorted(ABLATION_GROUPS),
        default=[],
        help="Refit without each named feature group and report the cost.",
    )
    parser.add_argument(
        "--write", action="store_true", help="Write the metrics. Default prints only."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    engine = engine_from_settings()
    try:
        hourly = load_hourly(engine)
    finally:
        engine.dispose()

    print(
        f"{len(hourly):,} hourly observations, "
        f"{hourly['city_id'].nunique()} cities, "
        f"{hourly['observation_hour'].min():%Y-%m-%d} .. "
        f"{hourly['observation_hour'].max():%Y-%m-%d}"
    )
    report = build_nowcast(hourly, args.horizons, ablations=args.ablate)

    print(f"\npeak gust over the next N hours, km/h\n{_horizon_table(report)}")
    for horizon, scored in report["horizons"].items():
        standalone = scored["test_non_overlapping"]
        print(
            f"\n{horizon}h, per city (test)\n{_city_table(scored)}"
            f"\n  one origin a day ({standalone['rows']:,} rows): "
            f"RMSE {standalone['rmse']:.2f} against {scored['test']['rmse']:.2f} "
            f"on all origins"
        )

    if args.ablate:
        print(f"\nablations (test RMSE)\n{_ablation_table(report)}")

    if not args.write:
        print("\n(no --write: nothing written)")
        return 0
    print(f"\nwrote {write_metrics(report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
