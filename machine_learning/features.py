"""Builds the model-ready feature matrix from the gold layer.

Every column here is a statement about what was knowable at the end of day
*t*. A feature that reaches forward by a single day is leakage, and leakage of
this kind does not announce itself: it raises PR-AUC, survives review, and is
only ever found by someone asking why the model is so good. The evaluation
cannot catch it either: a chronological split protects against training on
the future, not against a *feature* that already contains it.

So the guarantee is structural rather than reviewed:

* **The as-of convention.** A row dated *t* uses observations on days ``<= t``
  and nothing else. Day *t* itself is included, because the daily aggregate is
  complete at the end of the day, and discarding it would throw away the most
  informative value available. **The label must therefore start at t+1**:
  ML-02 builds "anomaly on any day in t+1 .. t+7", never "t .. t+6", or
  :data:`anomaly_days_trailing30` hands it the answer.
* **Windows are calendar windows, not row windows.** Each city is reindexed
  onto a contiguous daily calendar before anything is shifted, so ``shift(3)``
  is three *days* and not three *rows*. This is the same trap
  ``fact_weather_hourly`` avoids with a ``RANGE`` frame: on a series with a
  hole, ``lag(3)`` silently reaches four days back and reports the result as a
  three-day change. Silver has no gaps today; the reindex costs nothing and
  stays right if one appears.
* **Partial windows are null, not "close enough".** Every rolling statistic
  requires its full window (``min_periods == window``). A 30-day mean over 11
  days is a different statistic, and mixing the two into one column produces a
  feature whose meaning changes with row position.
* **Nothing is dropped silently.** :func:`build_features` returns one row per
  observed city-day, warm-up rows included and flagged. Discarding them is
  :func:`drop_warmup`, which the caller has to say out loud.

Two things the strictness does *not* cover, recorded here rather than
discovered later:

1. ``z_temperature_2m_mean`` and ``is_anomaly`` come from
   ``fact_weather_anomalies``, whose baseline is a ±7-day day-of-year window
   over every reference year *except the observation's own*. That removes the
   leakage that matters (a day contributing to the baseline that labels it),
   but the surviving years include years after *t*. A 2003 row is scored
   against a climatology that has seen 2020. It is a per-(city, day-of-year)
   constant rather than a path from the future to any particular day, and the
   alternative, an expanding climatology that uses only prior years, would
   give the early record a baseline of two or three years and a σ too noisy to
   score against. The trade is deliberate; :data:`temperature_2m_mean_z_trailing30`
   is the strictly-backward companion.
2. Pressure tendency is computed from **daily mean** sea-level pressure, so
   ``pressure_tendency_24h`` is a day-mean-to-day-mean change and not the
   instantaneous 24-hour tendency ``fact_weather_hourly`` carries. The hourly
   fact holds the sharper measure and only 24 months of it; the training window
   is thirty years. A feature that is null for 93% of rows is not a feature.

Usage::

    from machine_learning.features import feature_columns, load_features

    frame = load_features()                       # every city, whole record
    matrix = frame[list(feature_columns())]

Run ``python machine_learning/features.py`` for a summary of the built matrix,
``--out features.csv`` to write it, or ``--city delhi --start 2020-01-01`` for
a slice. The slice is padded backwards by the warm-up so a windowed request
is not silently degraded.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.loader import engine_from_settings  # noqa: E402

__all__ = [
    "ANOMALY_COUNT_WINDOW",
    "GOLD_SCHEMA",
    "LAGS",
    "PASSTHROUGH_COLUMNS",
    "PRESSURE",
    "REQUIRED_COLUMNS",
    "ROLLING_WINDOWS",
    "TEMPERATURE",
    "TENDENCY_DAYS",
    "TRAILING_Z_WINDOW",
    "WARMUP_DAYS",
    "FeatureError",
    "build_features",
    "city_roster",
    "day_of_year_common",
    "drop_warmup",
    "feature_columns",
    "gold_frame",
    "load_features",
    "missing_report",
    "on_daily_calendar",
    "require_grain",
    "trailing_anomaly_counts",
]

log = logging.getLogger(__name__)

GOLD_SCHEMA: Final[str] = "gold_marts"

#: The two series everything is lagged and rolled over. Daily mean temperature
#: is the variable the label is defined on; mean sea-level pressure is the
#: storm-development signal. *Sea-level*, not surface: surface pressure carries
#: the grid cell's elevation, so a lag over it would be comparing Johannesburg
#: at 822 hPa against London at 1013 the moment anything pooled across cities.
#: Min and max temperature are deliberately not lagged: they are highly
#: collinear with the mean and would triple the matrix for very little.
TEMPERATURE: Final[str] = "temperature_2m_mean"
PRESSURE: Final[str] = "pressure_msl_mean"

LAGS: Final[tuple[int, ...]] = (1, 3, 7, 14)
ROLLING_WINDOWS: Final[tuple[int, ...]] = (7, 30)

#: Pressure tendency spans, in days. 24h and 72h at the daily grain are the
#: one-day and three-day changes.
TENDENCY_DAYS: Final[tuple[int, ...]] = (1, 3)

#: The trailing standardisation window, in days, and it **excludes day t**.
#:
#: Scoring an observation against a baseline it contributed to is the same
#: mistake ``fact_climatology`` exists to avoid, one window smaller: with day
#: *t* inside a 30-day mean it pulls that mean 1/30 of the way towards itself
#: and inflates σ by its own deviation, so the Z of an extreme day comes out
#: systematically too small. The window is t-30 .. t-1, the recent past that
#: *t* is unusual with respect to.
TRAILING_Z_WINDOW: Final[int] = 30

#: The anomaly-count window, in days, and it **includes day t**: today's flag
#: is known at the end of today, and the label starts at t+1.
ANOMALY_COUNT_WINDOW: Final[int] = 30

#: Days of prior record a row needs before every feature is defined. The
#: binding constraint is the trailing Z-score, which looks back 30 days without
#: counting *t*; the 30-day rollings need 29, and the longest lag needs 14.
WARMUP_DAYS: Final[int] = max(
    max(LAGS),
    max(ROLLING_WINDOWS) - 1,
    ANOMALY_COUNT_WINDOW - 1,
    TRAILING_Z_WINDOW,
    max(TENDENCY_DAYS),
)

#: What :func:`build_features` needs to be handed. Everything else it derives.
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "city_id",
    "date_key",
    TEMPERATURE,
    PRESSURE,
    "z_temperature_2m_mean",
    "is_anomaly",
    "latitude",
    "elevation_m",
)

#: Columns carried through the matrix that are **not** model inputs.
#:
#: ``is_anomaly`` is the label source, and feeding a classifier the flag whose
#: forward window it is predicting is the shortest path to a meaningless score.
#: ``anomaly_days_scored30`` is the denominator of the anomaly count: a
#: coverage fact about the warehouse, not about the weather, and a model
#: allowed to learn from it learns which cities are half-backfilled.
#: ``is_warmup``, ``history_days`` and ``has_missing_feature`` are the null
#: bookkeeping. Select inputs with :func:`feature_columns`, never by dropping
#: the keys.
PASSTHROUGH_COLUMNS: Final[tuple[str, ...]] = (
    "city_id",
    "date_key",
    "is_anomaly",
    "anomaly_days_scored30",
    "history_days",
    "is_warmup",
    "has_missing_feature",
)


class FeatureError(RuntimeError):
    """Raised when the input frame cannot produce an honest feature matrix.

    Duplicated city-days and missing columns are both silent corruptions: the
    first makes a "7-day" window span five days, the second makes a feature
    quietly absent from the matrix a trainer then reports metrics for.
    """


def feature_columns(*, teleconnections: bool = False) -> tuple[str, ...]:
    """The model input columns, in a stable order.

    Built from the same constants the features are, so the list and the matrix
    cannot drift apart. A test asserts the built frame carries exactly these
    plus :data:`PASSTHROUGH_COLUMNS`.

    ``teleconnections`` is **off by default, and that is the ticket's own
    framing.** ING-04 asks for an ablation *against the current feature set*,
    which requires the current feature set to still exist: switching the four
    NOAA indices on here would change the shipped model's inputs before any
    evidence said they help, and would silently invalidate the committed
    artefact whose feature list the model card asserts. They go on when a
    caller asks for them, the ablation measures what they are worth, and the
    default changes only if the answer justifies it.
    """
    columns: list[str] = []
    for series in (TEMPERATURE, PRESSURE):
        columns.append(series)
        columns.extend(f"{series}_lag{lag}" for lag in LAGS)
        for window in ROLLING_WINDOWS:
            columns.append(f"{series}_roll{window}_mean")
            columns.append(f"{series}_roll{window}_var")
    columns.append(f"{TEMPERATURE}_z_trailing{TRAILING_Z_WINDOW}")
    columns.extend(f"pressure_tendency_{days * 24}h" for days in TENDENCY_DAYS)
    columns.append("z_temperature_2m_mean")
    columns.append(f"anomaly_days_trailing{ANOMALY_COUNT_WINDOW}")
    columns.extend(("day_of_year_sin", "day_of_year_cos", "latitude", "elevation_m"))
    if teleconnections:
        columns.extend(TELECONNECTION_FEATURES)
    return tuple(columns)


def day_of_year_common(dates: pd.Series) -> pd.Series:
    """Day of year on a 365-day axis, with 29 February folded onto 28.

    The second derivation of ``dim_date.day_of_year_common``, and asserted
    equal to it against every date in the warehouse. It is the same arrangement
    ``dim_cities.hemisphere`` has, for the same reason: the alternative is a
    join that exists only to fetch an integer that the date already determines.

    Raw ``day_of_year`` cannot serve as a cyclical axis. It is 1..366, so in a
    leap year every day after February is numbered one higher than the same
    calendar day in a common year: a one-day phase shift in the sin/cos pair,
    in three years out of four, which reads to a model as a genuine seasonal
    difference between leap and common years.
    """
    day_of_year = dates.dt.dayofyear
    is_leap = dates.dt.is_leap_year
    return day_of_year.where(~(is_leap & (day_of_year > 59)), day_of_year - 1)


def gold_frame(
    engine: Engine | None = None,
    *,
    cities: Sequence[str] | None = None,
    start: dt.date | str | None = None,
    end: dt.date | str | None = None,
) -> pd.DataFrame:
    """Read one row per city-day from the gold layer.

    Both joins are LEFT joins. An inner join would enforce referential
    integrity by *dropping* rows it could not match, which is the wrong
    failure: a city-day missing its anomaly row would look exactly like a
    city-day that was never observed, and the feature matrix would be short
    without saying so. A null Z arrives as a null Z.

    Args:
        engine: Warehouse connection. Defaults to ``DATABASE_URL``.
        cities: Restrict to these ``city_id`` values. Default: all of them.
        start: Earliest ``date_key`` to read, inclusive. Note this is the raw
            bound; :func:`load_features` is the one that pads it by the
            warm-up.
        end: Latest ``date_key`` to read, inclusive.

    Returns:
        A frame sorted by ``(city_id, date_key)`` with
        :data:`REQUIRED_COLUMNS`, ``date_key`` as ``datetime64``.
    """
    owned = engine is None
    engine = engine if engine is not None else engine_from_settings()
    where: list[str] = []
    params: dict[str, object] = {}
    if cities is not None:
        where.append("observations.city_id = any(:cities)")
        params["cities"] = list(cities)
    if start is not None:
        where.append("observations.date_key >= :start")
        params["start"] = pd.Timestamp(start).date()
    if end is not None:
        where.append("observations.date_key <= :end")
        params["end"] = pd.Timestamp(end).date()
    predicate = f"where {' and '.join(where)}" if where else ""

    sql = f"""
        select
            observations.city_id,
            observations.date_key,
            observations.{TEMPERATURE},
            observations.{PRESSURE},
            anomalies.z_temperature_2m_mean,
            anomalies.is_anomaly,
            -- Not a feature and not narrowed into the matrix: ML-09's threshold
            -- sweep needs it to re-derive the flag, because since DBT-14 the
            -- exceedance threshold depends on how many observations the sigma
            -- was estimated from.
            anomalies.baseline_observations,
            cities.latitude,
            cities.elevation_m
        from {GOLD_SCHEMA}.fact_weather_observations as observations
        left join {GOLD_SCHEMA}.fact_weather_anomalies as anomalies
               on anomalies.city_id = observations.city_id
              and anomalies.date_key = observations.date_key
        left join {GOLD_SCHEMA}.dim_cities as cities
               on cities.city_id = observations.city_id
        {predicate}
        order by observations.city_id, observations.date_key
    """
    try:
        with engine.connect() as connection:
            frame = pd.read_sql_query(text(sql), connection, params=params)
    finally:
        if owned:
            engine.dispose()

    frame["date_key"] = pd.to_datetime(frame["date_key"])
    return frame


def require_grain(
    observations: pd.DataFrame, columns: Sequence[str]
) -> pd.DataFrame:
    """Check the grain, narrow to ``columns``, and sort. Shared with ML-02.

    Both the feature matrix and the label depend on a row offset meaning a day
    offset, so both need the same three guarantees before they compute
    anything: the columns are there, no city-day repeats, and the rows are in
    order. One implementation, because two would be free to drift and the
    drift would be silent: a duplicated city-day makes a "7-day" window span
    six days and still produces a plausible column of numbers.

    Raises:
        FeatureError: A required column is missing, or a city-day repeats.
    """
    missing = [name for name in columns if name not in observations.columns]
    if missing:
        raise FeatureError(
            f"gold frame is missing {missing}. Expected {list(columns)}; "
            f"got {list(observations.columns)}."
        )

    frame = observations.loc[:, list(columns)].copy()
    frame["date_key"] = pd.to_datetime(frame["date_key"])
    if "is_anomaly" in frame.columns:
        # Nullable boolean, always. Left alone, this column arrives as `bool`
        # from a fully-scored city and as `object` from one with nulls, so the
        # dtypes would depend on which cities had backfilled, and a frame
        # built for one city would not concatenate cleanly with a frame built
        # for the next. `boolean` also keeps "unscored" distinguishable from
        # False, which is the distinction the anomaly counts and the label are
        # both built on.
        frame["is_anomaly"] = frame["is_anomaly"].astype("boolean")

    duplicated = frame.duplicated(subset=["city_id", "date_key"])
    if duplicated.any():
        offenders = frame.loc[duplicated, ["city_id", "date_key"]].head(5)
        raise FeatureError(
            f"{int(duplicated.sum())} duplicated city-days, e.g. "
            f"{offenders.to_dict('records')}. The grain is one row per city "
            "per day; a repeat makes every window span fewer days than it "
            "claims."
        )

    frame = frame.sort_values(["city_id", "date_key"], kind="stable")
    frame["is_observed"] = True
    return frame


def city_roster(engine: Engine | None = None) -> tuple[list[str], list[str]]:
    """The registry, and which of it has actually been ingested.

    Two lists rather than one, because a city can be absent from a downstream
    table for two unrelated reasons and the difference is the whole point of
    reporting the absence. ``dim_cities`` is built from ``config/cities.yml``
    and says what the set *is*; the fact table says what has been observed of
    it, and the gap between them is what the reconciliation report exists to
    surface.

    Here rather than in the evaluation or inference modules because both need
    it and neither should own it: a second copy would be free to drift, and a
    drift would be silent, since both would still return a plausible list of
    city names.
    """
    owned = engine is None
    engine = engine if engine is not None else engine_from_settings()
    try:
        with engine.connect() as connection:
            roster = [
                row[0]
                for row in connection.execute(
                    text(
                        f"select city_id from {GOLD_SCHEMA}.dim_cities "
                        "order by city_id"
                    )
                )
            ]
            ingested = [
                row[0]
                for row in connection.execute(
                    text(
                        "select distinct city_id from "
                        f"{GOLD_SCHEMA}.fact_weather_observations order by city_id"
                    )
                )
            ]
    finally:
        if owned:
            engine.dispose()
    return roster, ingested


def build_features(observations: pd.DataFrame) -> pd.DataFrame:
    """Turn gold rows into the feature matrix. Pure, and never touches a database.

    The database stays out so the leakage guarantee is testable: the same
    function that builds thirty years for nine cities builds two hundred
    synthetic days, and ``tests/test_features.py`` proves on those that
    rewriting the future changes no row in the past.

    Args:
        observations: One row per city-day, carrying :data:`REQUIRED_COLUMNS`.
            Order does not matter; duplicates are an error.

    Returns:
        One row per **observed** city-day (the same count as the input) with
        :func:`feature_columns` and :data:`PASSTHROUGH_COLUMNS`, sorted by
        ``(city_id, date_key)`` with a fresh index. Rows inside a city's
        warm-up are present and flagged, not removed.

    Raises:
        FeatureError: A required column is missing, or a city-day repeats.
    """
    frame = require_grain(observations, REQUIRED_COLUMNS)
    calendar = on_daily_calendar(frame)
    grouped = calendar.groupby("city_id", sort=False)

    # Accumulated and assigned in one go rather than written back column by
    # column, so nothing is ever derived from a frame a previous feature has
    # already altered.
    columns: dict[str, pd.Series] = {}

    for series in (TEMPERATURE, PRESSURE):
        for lag in LAGS:
            columns[f"{series}_lag{lag}"] = grouped[series].shift(lag)
        for window in ROLLING_WINDOWS:
            columns[f"{series}_roll{window}_mean"] = _rolling(
                grouped[series], window, "mean"
            )
            # Sample variance (ddof=1): the window is a sample of the local
            # climate, not the whole of it, and at n=7 the two differ by 17%.
            columns[f"{series}_roll{window}_var"] = _rolling(
                grouped[series], window, "var"
            )

    # Trailing standardisation over the days *before* t. Shifting the series
    # one day before rolling is what excludes t; see TRAILING_Z_WINDOW.
    prior = grouped[TEMPERATURE].shift(1).groupby(calendar["city_id"], sort=False)
    prior_mean = _rolling(prior, TRAILING_Z_WINDOW, "mean")
    # A zero σ divides to ±inf and reads as an infinitely extreme day. Over 30
    # days of float temperature it should not happen, and a feature that is
    # silently infinite when it does is worse than one that is null.
    prior_std = _rolling(prior, TRAILING_Z_WINDOW, "std").where(lambda std: std > 0)
    columns[f"{TEMPERATURE}_z_trailing{TRAILING_Z_WINDOW}"] = (
        calendar[TEMPERATURE] - prior_mean
    ) / prior_std

    for days in TENDENCY_DAYS:
        columns[f"pressure_tendency_{days * 24}h"] = (
            calendar[PRESSURE] - grouped[PRESSURE].shift(days)
        )

    columns.update(_anomaly_counts(calendar))

    # Static and calendar features. No window, so nothing to look forward into.
    angle = 2 * np.pi * day_of_year_common(calendar["date_key"]) / 365.0
    columns["day_of_year_sin"] = np.sin(angle)
    columns["day_of_year_cos"] = np.cos(angle)

    features = calendar.assign(**columns)
    features = features.loc[features["is_observed"]].copy()
    _add_null_bookkeeping(features)

    ordered = list(PASSTHROUGH_COLUMNS) + list(feature_columns())
    return features.loc[:, ordered].reset_index(drop=True)


def on_daily_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    """Reindex each city onto every day between its first and last observation.

    This is what makes a row offset a day offset. Inserted days carry nulls and
    ``is_observed = False``: they occupy their position in every window, so a
    window spanning a hole is null rather than quietly reaching further back,
    and they are dropped before the matrix is returned because a day with no
    observation has no features and no label either.

    Shared with ML-02, which needs the same discipline pointing the other way:
    a forward window that reaches over a hole would call a day it never saw
    quiet.
    """
    pieces: list[pd.DataFrame] = []
    for city_id, group in frame.groupby("city_id", sort=True):
        indexed = group.set_index("date_key").drop(columns=["city_id"])
        span = pd.date_range(indexed.index.min(), indexed.index.max(), freq="D")
        reindexed = indexed.reindex(span)
        reindexed.index.name = "date_key"
        reindexed["city_id"] = city_id
        reindexed["is_observed"] = reindexed["is_observed"].eq(True)
        # Static per city, so the reindexed days inherit rather than blank:
        # a city's latitude did not stop existing on a day it was not observed,
        # and leaving them null would only make the diagnostics noisier.
        for column in ("latitude", "elevation_m"):
            if column in reindexed.columns:
                reindexed[column] = reindexed[column].ffill().bfill()
        pieces.append(reindexed.reset_index())
    return pd.concat(pieces, ignore_index=True)


def _rolling(grouped, window: int, statistic: str) -> pd.Series:
    """A grouped rolling statistic over a full window, aligned to the input.

    ``min_periods=window`` is the whole point: pandas defaults it to the window
    size for ``rolling(int)``, but stating it makes the intent explicit and
    survives someone reaching for ``min_periods=1`` to "fix" the nulls at the
    start of each series. Those nulls are the honest answer.
    """
    return grouped.transform(
        lambda series: getattr(series.rolling(window, min_periods=window), statistic)()
    )


def trailing_anomaly_counts(
    calendar: pd.DataFrame, window: int
) -> tuple[pd.Series, pd.Series]:
    """Days flagged, and days scorable, over the trailing ``window``, inclusive of t.

    A null ``is_anomaly`` is not a quiet day. It is a day with no baseline to
    score against, and 1 095 of them exist for the three cities holding a
    single reference year. Counting a null as "not an anomaly" would assert the
    day was ordinary on no evidence, and would put it in the denominator of
    every rate computed downstream.

    So two series rather than one coerced one. The first counts days flagged
    ``true``, and is a *lower bound* wherever the window holds unscored days;
    the second is how many of the window carried a flag at all. Equal to the
    window, the count is exact; below it, the caller can see the count is
    partial rather than read a quiet month off a coverage gap.

    Both require every calendar day of the window to be present in the record:
    a window spanning a hole is null, the same rule the rollings follow.

    Args:
        calendar: A frame from :func:`on_daily_calendar`, so a row offset is a
            day offset.
        window: Trailing days, counting *t* as one of them.

    Returns:
        ``(flagged, scored)``, both aligned to ``calendar``.
    """
    observed = calendar["is_observed"].to_numpy(dtype=bool)
    flag = calendar["is_anomaly"]
    known = flag.notna().to_numpy(dtype=bool)
    hit = known & flag.fillna(False).to_numpy(dtype=bool)

    counted = []
    for values in (hit, known):
        # NaN on unobserved days, so min_periods=window rejects any window
        # that spans one instead of treating the hole as a quiet day.
        series = pd.Series(
            np.where(observed, values.astype(float), np.nan), index=calendar.index
        )
        counted.append(
            _rolling(series.groupby(calendar["city_id"], sort=False), window, "sum")
        )
    return counted[0], counted[1]


def _anomaly_counts(calendar: pd.DataFrame) -> dict[str, pd.Series]:
    """The trailing counts under the names the feature matrix gives them."""
    window = ANOMALY_COUNT_WINDOW
    flagged, scored = trailing_anomaly_counts(calendar, window)
    return {
        f"anomaly_days_trailing{window}": flagged,
        f"anomaly_days_scored{window}": scored,
    }


def _add_null_bookkeeping(features: pd.DataFrame) -> None:
    """Say which rows are incomplete, and why, in place.

    Two different questions, so two columns. ``is_warmup`` is positional: the
    row is inside the first :data:`WARMUP_DAYS` days of its city's record, so
    some window has not filled yet and the nulls are expected and permanent.
    ``has_missing_feature`` is empirical: *some* model input is null on this
    row, for whatever reason: a warm-up, a hole in the series, or a city with
    no climatology baseline. Neither implies the other. A city with a gap has
    null features far outside its warm-up, and a fully-scored row inside the
    warm-up is still unusable.

    ``history_days`` counts calendar days since the city's first record rather
    than rows, so a series with a hole in it is not credited with days it does
    not have. It is a **lower bound**: measured from the earliest row this
    build was handed, which for a slice is the start of the slice's padded
    window and not the start of the record. ``is_warmup`` is exact anyway:
    :func:`load_features` pads by :data:`WARMUP_DAYS`, so a row with enough
    real history always has enough history in the frame.
    """
    first_seen = features.groupby("city_id")["date_key"].transform("min")
    features["history_days"] = (features["date_key"] - first_seen).dt.days
    features["is_warmup"] = features["history_days"] < WARMUP_DAYS
    features["has_missing_feature"] = (
        features.loc[:, list(feature_columns())].isna().any(axis=1)
    )


def drop_warmup(features: pd.DataFrame) -> pd.DataFrame:
    """Remove the rows inside each city's warm-up. An explicit act, deliberately.

    :func:`build_features` never drops them. A feature module that quietly
    shortens the record hands the trainer a matrix whose row count does not
    match the warehouse's, and the difference is discovered, if it is
    discovered at all, as an unexplained gap in a metric.

    Note this does **not** guarantee a null-free matrix: a hole in a city's
    series produces null windows well outside the warm-up, and a city with no
    climatology baseline has a null Z on every row of its record. Filter on
    ``has_missing_feature`` for that, and see :func:`missing_report` for what
    it would cost.
    """
    return features.loc[~features["is_warmup"]].reset_index(drop=True)


def missing_report(features: pd.DataFrame) -> pd.DataFrame:
    """Null counts per feature, split into warm-up and everything else.

    The split is the point. Nulls inside the warm-up are arithmetic (a 30-day
    window on day 3) and need no explanation. Nulls outside it are a fact
    about the data, and every one should have a name: a hole in the series, or
    a city whose leave-one-year-out baseline left nothing behind.
    """
    warmup = features["is_warmup"]
    rows = []
    for column in feature_columns():
        null = features[column].isna()
        rows.append(
            {
                "feature": column,
                "null_warmup": int((null & warmup).sum()),
                "null_after_warmup": int((null & ~warmup).sum()),
                "null_total": int(null.sum()),
            }
        )
    return pd.DataFrame(rows)


def load_features(
    engine: Engine | None = None,
    *,
    cities: Sequence[str] | None = None,
    start: dt.date | str | None = None,
    end: dt.date | str | None = None,
    teleconnections: bool = False,
) -> pd.DataFrame:
    """Read gold and build the matrix, padding ``start`` by the warm-up.

    The padding is the reason this is not two calls at the call site. Asking
    for 2020 onwards and building features from exactly those rows gives the
    first 30 days of 2020 the warm-up nulls of a series that begins in 2020,
    except the series does not begin in 2020, the *request* does. The extra
    :data:`WARMUP_DAYS` of history are read, used, and trimmed off, so a slice
    and a full build agree on every row they share.

    Args:
        engine: Warehouse connection. Defaults to ``DATABASE_URL``.
        cities: Restrict to these ``city_id`` values.
        start: Earliest ``date_key`` in the **result**, inclusive.
        end: Latest ``date_key`` in the result, inclusive.
        teleconnections: Attach the four NOAA indices, joined on the date each
            became readable. Off by default; see :func:`feature_columns`.

    Note:
        Every feature and ``is_warmup`` match a full build exactly.
        ``history_days`` does not: it counts from the padded window rather than
        from the start of the record, so on a slice it is a lower bound.
    """
    padded = start
    if start is not None:
        padded = pd.Timestamp(start) - pd.Timedelta(days=WARMUP_DAYS)
    frame = gold_frame(engine, cities=cities, start=padded, end=end)
    features = build_features(frame)
    if teleconnections:
        from ingestion.teleconnections import read_vintages

        owned = engine is None
        connection = engine or engine_from_settings()
        try:
            features = attach_teleconnections(features, read_vintages(connection))
        finally:
            if owned:
                connection.dispose()
    if start is not None:
        keep = features["date_key"] >= pd.Timestamp(start)
        features = features.loc[keep].reset_index(drop=True)
    return features


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the ML feature matrix from the gold layer."
    )
    parser.add_argument(
        "--city",
        action="append",
        dest="cities",
        help="Restrict to a city_id. Repeatable.",
    )
    parser.add_argument("--start", help="Earliest date in the result (YYYY-MM-DD).")
    parser.add_argument("--end", help="Latest date in the result (YYYY-MM-DD).")
    parser.add_argument(
        "--drop-warmup",
        action="store_true",
        help="Drop rows inside each city's warm-up before writing.",
    )
    parser.add_argument("--out", help="Write the matrix to this CSV path.")
    return parser.parse_args(argv)


# ===========================================================================
# Teleconnection indices (ING-04/ML-17)
# ===========================================================================

#: The large-scale indices, in a stable order.
#:
#: Every other feature is one city's own history. These four are the first that
#: are not: ENSO, the North Atlantic and Arctic Oscillations, and the Indian
#: Ocean Dipole.
TELECONNECTION_FEATURES: Final[tuple[str, ...]] = ("oni", "nao", "ao", "dmi")


def teleconnection_steps(vintages: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per index, the value that was current at each moment it changed.

    **Why this is a walk and not a `merge_asof`.** The obvious join takes, for
    a date *D*, the index row with the greatest `publication_date` at or before
    it. That is right until the first revision and wrong afterwards: a restated
    value for January 2024 published in 2029 has a *late* publication date and
    an *old* nominal period, so the naive join would answer "what is the ENSO
    state today" with a correction to a five-year-old month.

    What is wanted is the value of the most recent *period* known at *D*, using
    the most recent *version* of that period known at *D*. So the vintages are
    walked in publication order, holding a dict of period to current value, and
    at each publication a step is emitted carrying the value of the newest
    period in the dict. A revision updates its own period and only changes the
    answer if that period is still the newest one.

    Returns:
        One frame per index, columns ``publication_date`` and ``value``, sorted
        and deduplicated to the last step on each date.
    """
    steps: dict[str, pd.DataFrame] = {}
    for index_id, rows in vintages.groupby("index_id", sort=True):
        ordered = rows.sort_values(["publication_date", "vintage_at"], kind="stable")
        current: dict[Any, float] = {}
        emitted: list[tuple[Any, float]] = []
        for row in ordered.itertuples():
            current[row.nominal_period] = float(row.value)
            newest = max(current)
            emitted.append((row.publication_date, current[newest]))
        frame = pd.DataFrame(emitted, columns=["publication_date", "value"])
        # Nanosecond resolution on both sides. `date_key` arrives from the
        # warehouse as datetime64[us] or [s] depending on the driver, and
        # `merge_asof` refuses to join two datetime columns of different
        # resolution rather than coercing them.
        frame["publication_date"] = pd.to_datetime(
            frame["publication_date"]
        ).astype("datetime64[ns]")
        # Several periods can publish on one date; the last one written wins,
        # which is the same answer a reader on that date would get.
        steps[str(index_id)] = (
            frame.drop_duplicates("publication_date", keep="last")
            .sort_values("publication_date")
            .reset_index(drop=True)
        )
    return steps


def attach_teleconnections(
    features: pd.DataFrame, vintages: pd.DataFrame
) -> pd.DataFrame:
    """Join each index as it stood on the row's own date.

    The join key is `publication_date`, never `nominal_period`. Joining on the
    label is the leak this whole feature exists to avoid: the ONI labelled
    January cannot be computed until February has ended, so a nominal join
    reads six weeks into the future while looking entirely correct.

    A row earlier than an index's first publication gets a null rather than a
    back-filled first value. The 1995 rows genuinely had no DMI vintage
    available under this pipeline's rule, and filling them would assert
    knowledge that did not exist.
    """
    if vintages.empty:
        raise FeatureError("no teleconnection vintages; run ingestion/teleconnections.py")
    out = features.copy()
    dates = pd.to_datetime(out["date_key"])
    steps = teleconnection_steps(vintages)

    missing = [name for name in TELECONNECTION_FEATURES if name not in steps]
    if missing:
        raise FeatureError(f"teleconnection vintages are missing {missing}.")

    ordering = dates.argsort(kind="stable")
    sorted_dates = dates.iloc[ordering].astype("datetime64[ns]")
    for name in TELECONNECTION_FEATURES:
        joined = pd.merge_asof(
            pd.DataFrame({"date_key": sorted_dates.to_numpy()}),
            steps[name].rename(columns={"publication_date": "date_key"}),
            on="date_key",
            direction="backward",
        )
        values = pd.Series(index=out.index, dtype="float64")
        values.iloc[ordering.to_numpy()] = joined["value"].to_numpy()
        out[name] = values
    return out


# ===========================================================================
# The hourly matrix (ML-16)
# ===========================================================================
#
# A separate builder rather than a mode of `build_features`, because it is a
# different product on a different grain answering a different question. The
# daily matrix predicts a temperature anomaly seven days out from thirty years
# of climatology; this one predicts the peak wind gust one to three days out
# from two years of hourly observations. They share a repository and almost
# nothing else -- not the target, not the horizon, not the baseline, and not
# the split boundaries.

#: Nowcast horizons, in hours.
#:
#: One to three days. Below 24 hours the answer is largely persistence and
#: there is nothing for a model to add; beyond 72 the hourly record's two years
#: stop containing enough distinct weather systems for the longer patterns to
#: be estimated rather than memorised.
NOWCAST_HORIZONS: Final[tuple[int, ...]] = (24, 48, 72)

#: Rolling windows over the hourly series, in hours.
#:
#: Six hours is the current system, twenty-four the diurnal cycle, seventy-two
#: the synoptic one. Chosen to bracket the timescales rather than to be dense:
#: overlapping rolling maxima are highly collinear and adding more of them
#: buys correlation, not information.
NOWCAST_WINDOWS: Final[tuple[int, ...]] = (6, 24, 72)

#: The hourly columns the builder needs.
HOURLY_REQUIRED: Final[tuple[str, ...]] = (
    "city_id",
    "observation_hour",
    "pressure_msl",
    "pressure_tendency_3h",
    "pressure_tendency_24h",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
    "temperature_2m",
    "dew_point_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
)

#: Carried through the hourly matrix but never a model input.
HOURLY_PASSTHROUGH: Final[tuple[str, ...]] = (
    "city_id",
    "observation_hour",
    "peak_gust_ahead",
    "persistence_gust",
    "horizon_hours",
    "month",
)


def hourly_feature_columns() -> tuple[str, ...]:
    """The nowcast's model inputs, in a stable order.

    Derived from the same constants the builder uses, so the list and the
    matrix cannot disagree -- the daily side's rule, applied here for the same
    reason.
    """
    rolled = tuple(
        f"{stem}{window}"
        for window in NOWCAST_WINDOWS
        for stem in (
            "gust_max", "gust_mean", "speed_mean",
            "pressure_min", "pressure_range", "precipitation_sum",
        )
    )
    return (
        "wind_gusts_10m",
        "wind_speed_10m",
        "pressure_msl",
        "pressure_tendency_3h",
        "pressure_tendency_24h",
        "abs_pressure_tendency_3h",
        "abs_pressure_tendency_24h",
        "temperature_2m",
        "dew_point_2m",
        "dew_point_depression",
        "relative_humidity_2m",
        "precipitation",
        "cloud_cover",
        "wind_direction_sin",
        "wind_direction_cos",
        *rolled,
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
    )


def build_hourly_features(
    hourly: pd.DataFrame, horizon_hours: int
) -> pd.DataFrame:
    """One row per city-hour: what is known at *t*, and the peak gust after it.

    The target is ``max(wind_gusts_10m)`` over ``t+1 .. t+horizon_hours``.
    Hour *t* is a feature and is deliberately outside its own window, the same
    boundary the daily label draws at day *t*.

    Three choices that would each be a silent defect if made the other way.

    **Wind direction is encoded as a sine and a cosine, never as degrees.**
    359 and 1 are two degrees apart and 358 units apart on a number line, so a
    tree fed raw bearings learns a split at north that has no meaning. This is
    the kind of error that costs a little accuracy, breaks no test, and is
    invisible in a feature importance table.

    **Every rolling window ends at *t* inclusive and looks backward.** Pandas'
    default rolling is trailing, which is what is wanted; the target is built
    by reversing the series, which is where a sign error would put future gusts
    into the features. `peak_gust_ahead` is asserted disjoint from the window
    the features are drawn from by a test.

    **`persistence_gust` is carried, not computed downstream.** The baseline is
    the peak gust over the *previous* ``horizon_hours``, and it has to be the
    same span as the target or the comparison is between different questions.
    Deriving it here, from the same constant, is what keeps them equal.

    Returns:
        The matrix, with warm-up rows dropped: the first 72 hours of each city
        have no 72-hour window and the last ``horizon_hours`` have no target.
    """
    missing = [column for column in HOURLY_REQUIRED if column not in hourly.columns]
    if missing:
        raise FeatureError(f"hourly frame is missing {missing}.")
    if horizon_hours <= 0:
        raise FeatureError(f"horizon must be positive, got {horizon_hours}.")

    parts = [
        _hourly_city(city.sort_values("observation_hour"), horizon_hours)
        for _, city in hourly.groupby("city_id", sort=True)
    ]
    frame = pd.concat(parts, ignore_index=True)
    return frame.dropna(subset=["peak_gust_ahead", "persistence_gust", *hourly_feature_columns()])


def _hourly_city(city: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    """One city's hourly matrix. Grouped work stays out of `build_hourly_features`."""
    hours = city["observation_hour"]
    gusts, pressure = city["wind_gusts_10m"], city["pressure_msl"]

    out = pd.DataFrame(index=city.index)
    out["city_id"] = city["city_id"].to_numpy()
    out["observation_hour"] = hours.to_numpy()
    for column in (
        "wind_gusts_10m", "wind_speed_10m", "pressure_msl",
        "pressure_tendency_3h", "pressure_tendency_24h", "temperature_2m",
        "dew_point_2m", "relative_humidity_2m", "precipitation", "cloud_cover",
    ):
        out[column] = city[column].to_numpy()

    # Magnitude, not direction. The Storm Dynamics view found the relationship
    # between pressure swing and peak gust is V-shaped: a fast rise and a fast
    # fall both mean wind, and a signed tendency asks a tree to rediscover that
    # by splitting twice.
    out["abs_pressure_tendency_3h"] = city["pressure_tendency_3h"].abs().to_numpy()
    out["abs_pressure_tendency_24h"] = city["pressure_tendency_24h"].abs().to_numpy()

    # Dew point depression is the moisture signal in the form that means
    # something across cities; the dew point alone is mostly temperature.
    out["dew_point_depression"] = (
        city["temperature_2m"] - city["dew_point_2m"]
    ).to_numpy()

    radians = np.deg2rad(city["wind_direction_10m"].astype(float))
    out["wind_direction_sin"] = np.sin(radians).to_numpy()
    out["wind_direction_cos"] = np.cos(radians).to_numpy()

    for window in NOWCAST_WINDOWS:
        trailing = {
            "gust_max": gusts.rolling(window, min_periods=window).max(),
            "gust_mean": gusts.rolling(window, min_periods=window).mean(),
            "speed_mean": city["wind_speed_10m"].rolling(window, min_periods=window).mean(),
            "pressure_min": pressure.rolling(window, min_periods=window).min(),
            "pressure_range": (
                pressure.rolling(window, min_periods=window).max()
                - pressure.rolling(window, min_periods=window).min()
            ),
            "precipitation_sum": city["precipitation"].rolling(
                window, min_periods=window
            ).sum(),
        }
        for stem, values in trailing.items():
            out[f"{stem}{window}"] = values.to_numpy()

    hour = hours.dt.hour
    day_of_year = hours.dt.dayofyear
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24).to_numpy()
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24).to_numpy()
    out["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25).to_numpy()
    out["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25).to_numpy()

    # The target: reverse, take a trailing max, reverse back, then shift so the
    # window starts at t+1. Written this way rather than with a forward-looking
    # rolling because pandas has no forward max and a hand-rolled loop over
    # 263 160 rows would be the slow, wrong-by-one version of this line.
    reversed_max = gusts[::-1].rolling(horizon_hours, min_periods=horizon_hours).max()[::-1]
    out["peak_gust_ahead"] = reversed_max.shift(-1).to_numpy()
    out["persistence_gust"] = (
        gusts.rolling(horizon_hours, min_periods=horizon_hours).max().to_numpy()
    )
    out["horizon_hours"] = horizon_hours
    out["month"] = hours.dt.month.to_numpy()
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    features = load_features(cities=args.cities, start=args.start, end=args.end)
    if args.drop_warmup:
        features = drop_warmup(features)

    inputs = feature_columns()
    print(f"rows           {len(features):,}")
    print(f"cities         {features['city_id'].nunique()}")
    if len(features):
        print(
            f"dates          {features['date_key'].min().date()} .. "
            f"{features['date_key'].max().date()}"
        )
    print(f"features       {len(inputs)}")
    print(f"warm-up rows   {int(features['is_warmup'].sum()):,}")
    print(f"rows with a null feature   {int(features['has_missing_feature'].sum()):,}")

    report = missing_report(features)
    incomplete = report.loc[report["null_total"] > 0]
    if len(incomplete):
        print("\nnulls by feature")
        print(incomplete.to_string(index=False))

    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(destination, index=False)
        print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
