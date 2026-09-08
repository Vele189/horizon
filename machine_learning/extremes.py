"""Peaks over threshold: how unusual a day is, in years rather than in sigmas.

A Z-score answers *is this unusual*. A reader wants *how unusual*, and the
natural unit for that is time: "a Z of 3.5 is a one-in-twelve-year day in
Reykjavik and a one-in-forty-year day in Singapore" is a sentence somebody can
act on, where "3.5 sigma" is a sentence they have to convert.

**A Gaussian Z-score is the wrong instrument for a tail.** It is standardised
by a mean and a variance, which are properties of the middle of a
distribution, and the whole content of an extreme is that it is not in the
middle. Extreme value theory gives the right one: above a high enough
threshold, exceedances of almost any distribution converge to a generalised
Pareto, whose shape parameter says whether the tail is bounded, exponential or
heavy. That is an asymptotic result about tails specifically, rather than a
normal approximation being asked to work somewhere it was never good.

Three things this module refuses to do without.

**Declustering.** A five-day heatwave is one event. Counting it as five
exceedances triples the apparent frequency of extremes and every return period
computed from it is wrong by that factor -- and wrong in the direction that
makes the product look more dramatic, which is the direction to be most
suspicious of. Runs declustering keeps the peak of each cluster and discards
the rest; on this data the mean cluster runs 2.1 to 3.6 days, so it is not a
technicality.

**Confidence intervals on the shape.** At ~30 years a city has a few hundred
declustered exceedances and the shape parameter is unstable: Delhi's moves from
-0.005 to +0.147 when the threshold moves from the 95th to the 98th
percentile. A point estimate of the shape implies a definite answer about
whether the tail is bounded, and this data does not contain one. The intervals
are bootstrapped over *clusters* rather than days, because the clusters are the
independent units and resampling days would treat one heatwave as several.

**A non-stationary fit.** A stationary tail under a warming trend is the same
mistake DBT-12 fixes upstream: it would attribute a rising frequency of
extremes to chance and quote return periods from a climate that is no longer
current. The scale is allowed to vary with time, and a likelihood-ratio test
against the stationary fit says whether the data supports it.

Usage::

    python machine_learning/extremes.py            # fit and print
    python machine_learning/extremes.py --write    # and write the mart
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, genpareto
from sqlalchemy import Engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.loader import engine_from_settings  # noqa: E402

__all__ = [
    "BOOTSTRAP_SAMPLES",
    "MIN_CLUSTERS",
    "POT_QUANTILE",
    "RETURN_PERIODS",
    "RUN_SEPARATION_DAYS",
    "CityTail",
    "build_extremes",
    "fit_city",
    "decluster",
    "fit_nonstationary",
    "moving_threshold",
    "fit_stationary",
    "return_level",
    "return_period",
]

log = logging.getLogger(__name__)

DAYS_PER_YEAR: Final[float] = 365.25

#: Where the tail is taken to begin, as a per-city quantile of that city's own
#: Z-scores.
#:
#: A quantile rather than an absolute Z, because the point of fitting per city
#: is that the cities differ, and a fixed cut would hand Cairo four times the
#: sample it hands Phoenix. The 95th leaves 160 to 271 declustered clusters per
#: city here, which is enough for a shape parameter to mean something; the 98th
#: leaves 77 to 141, and the shapes visibly wander between the two, which is
#: the instability the intervals exist to report rather than a reason to pick
#: whichever looks tidier.
POT_QUANTILE: Final[float] = 0.95

#: Quiet days needed between exceedances before they count as separate events.
#:
#: Three. A synoptic system persists for days, so consecutive hot days are one
#: event observed repeatedly; the mean cluster on this data is 2.1 to 3.6 days,
#: which is what a weather system looks like and not what independent draws
#: look like.
RUN_SEPARATION_DAYS: Final[int] = 3

#: Below this many clusters, no fit is reported at all.
#:
#: Fifty. A generalised Pareto has two free parameters and its shape is the
#: unstable one; fitted to a handful of exceedances it will return numbers, and
#: they will be noise wearing three decimal places. Sydney has eighteen scored
#: days in total and is excluded by this rather than by a special case.
MIN_CLUSTERS: Final[int] = 50

#: Bootstrap resamples for the parameter intervals.
BOOTSTRAP_SAMPLES: Final[int] = 500

#: Return periods reported, in years.
RETURN_PERIODS: Final[tuple[int, ...]] = (2, 5, 10, 20, 50)

SEED: Final[int] = 42


def decluster(
    frame: pd.DataFrame,
    threshold: float,
    *,
    separation: int = RUN_SEPARATION_DAYS,
    column: str = "z_temperature_2m_mean",
) -> pd.DataFrame:
    """One row per *event*: the peak of each run of exceedances.

    Runs declustering. Exceedances separated by fewer than ``separation`` quiet
    days belong to the same weather, so only the largest is kept. Without this
    a five-day heatwave contributes five points to a fit that assumes
    independence, the exceedance rate is overstated by the mean cluster length,
    and every return period comes out too short -- in the direction that makes
    the product sound more dramatic.

    Returns:
        The peak rows, with ``cluster`` and ``cluster_days``, in date order.
    """
    if separation < 1:
        raise ValueError(f"separation must be at least one day, got {separation}.")
    ordered = frame.sort_values("date_key", kind="stable")
    above = ordered.loc[ordered[column] > threshold]
    if above.empty:
        return above.assign(cluster=pd.Series(dtype="int64"), cluster_days=0)

    gaps = above["date_key"].diff().dt.days
    # The first exceedance always opens a cluster; `separation` quiet days
    # between two of them closes the previous one.
    starts = gaps.isna() | (gaps > separation)
    above = above.assign(cluster=starts.cumsum())
    sizes = above.groupby("cluster")["date_key"].transform("size")
    peaks = above.loc[above.groupby("cluster")[column].idxmax()]
    return peaks.assign(cluster_days=sizes.loc[peaks.index]).sort_values(
        "date_key", kind="stable"
    )


def fit_stationary(excesses: np.ndarray) -> tuple[float, float, float]:
    """Generalised Pareto over the excesses, location pinned at zero.

    ``floc=0`` because an excess is by construction the amount *above* the
    threshold: a fitted location would be a second threshold, estimated from
    the same data, and the two would trade off against each other.

    Returns:
        shape, scale, and the log-likelihood.
    """
    values = np.asarray(excesses, dtype=float)
    shape, _, scale = genpareto.fit(values, floc=0.0)
    return float(shape), float(scale), float(
        np.sum(genpareto.logpdf(values, shape, loc=0.0, scale=scale))
    )


def _negative_log_likelihood(
    parameters: np.ndarray, excesses: np.ndarray, times: np.ndarray
) -> float:
    """GPD with a log-linear time trend in the scale, constant shape.

    ``sigma(t) = exp(b0 + b1 * t)``. The log link keeps the scale positive for
    every value the optimiser tries, which a linear one does not, and makes
    ``b1`` a proportional change per unit time rather than an absolute one --
    the interpretable quantity when the thing being scaled is a spread.
    """
    intercept, slope, shape = parameters
    scale = np.exp(intercept + slope * times)
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        return np.inf
    scaled = excesses / scale
    if abs(shape) < 1e-8:
        return float(np.sum(np.log(scale) + scaled))
    support = 1.0 + shape * scaled
    if np.any(support <= 0):
        return np.inf
    return float(np.sum(np.log(scale) + (1.0 + 1.0 / shape) * np.log(support)))


def fit_nonstationary(
    excesses: np.ndarray, times: np.ndarray
) -> dict[str, Any]:
    """The same tail, allowed to change scale with time, and whether it does.

    A stationary tail under a warming trend quotes return periods from a
    climate that is no longer current, which is DBT-12's mistake one layer up.
    So the scale is given a time trend and the two fits are compared by a
    likelihood-ratio test: twice the difference in log-likelihood against a
    chi-squared on one degree of freedom, which is the one extra parameter.

    ``times`` is in years from the first exceedance, so ``scale_trend_per_year``
    is a proportional change per year and ``scale_change_over_record`` is what
    it amounts to across the whole record.

    **Feed this one direction of the tail at a time.** See
    `fit_directional_trend`: run on folded ``abs(Z)`` it fits a mixture whose
    composition inverts over the record, and reports the inversion as a change
    in width.
    """
    values = np.asarray(excesses, dtype=float)
    clock = np.asarray(times, dtype=float)
    shape, scale, stationary_loglik = fit_stationary(values)

    start = np.array([np.log(scale), 0.0, shape])
    fitted = minimize(
        _negative_log_likelihood,
        start,
        args=(values, clock),
        method="Nelder-Mead",
        options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-8},
    )
    if not fitted.success or not np.isfinite(fitted.fun):
        return {"converged": False}

    intercept, slope, trend_shape = (float(value) for value in fitted.x)
    loglik = -float(fitted.fun)
    statistic = 2.0 * (loglik - stationary_loglik)
    span = float(clock.max() - clock.min())
    return {
        "converged": True,
        "log_scale_intercept": intercept,
        "scale_trend_per_year": slope,
        "shape": trend_shape,
        "log_likelihood": loglik,
        "stationary_log_likelihood": stationary_loglik,
        "likelihood_ratio": statistic,
        # One degree of freedom: the trend is the single added parameter.
        "p_value": float(chi2.sf(max(statistic, 0.0), df=1)),
        "scale_change_over_record": float(np.exp(slope * span)),
        "record_years": span,
    }


def return_level(
    period_years: float,
    *,
    threshold: float,
    shape: float,
    scale: float,
    exceedance_rate: float,
    observations_per_year: float = DAYS_PER_YEAR,
) -> float:
    """The Z exceeded once per ``period_years`` on average.

    ``u + (scale / shape) * ((m * n * rate) ** shape - 1)``, with the
    exponential limit taken directly when the shape is near zero rather than
    left to divide by it.

    ``exceedance_rate`` is the *declustered* rate: events per observation, not
    exceeding days per observation. Passing the undeclustered rate is the error
    this whole module is arranged to prevent, and it would shorten every return
    period by the mean cluster length.
    """
    expected = period_years * observations_per_year * exceedance_rate
    if expected <= 1.0:
        # A period shorter than the mean spacing between events: the level is
        # below the threshold and the GPD says nothing about it.
        return float("nan")
    if abs(shape) < 1e-8:
        return float(threshold + scale * np.log(expected))
    return float(threshold + (scale / shape) * (expected**shape - 1.0))


def return_period(
    level: float,
    *,
    threshold: float,
    shape: float,
    scale: float,
    exceedance_rate: float,
    observations_per_year: float = DAYS_PER_YEAR,
) -> float:
    """How often a Z of ``level`` is exceeded, in years. The inverse of above.

    Returns ``inf`` beyond the upper endpoint of a bounded tail: a negative
    shape means the distribution stops, and a level past where it stops has no
    return period rather than a very long one. Saying "one in 40 000 years"
    about a value the fitted model calls impossible would be worse than saying
    nothing.
    """
    if level <= threshold:
        return float("nan")
    excess = level - threshold
    if abs(shape) < 1e-8:
        survival = np.exp(-excess / scale)
    else:
        support = 1.0 + shape * excess / scale
        if support <= 0:
            return float("inf")
        survival = support ** (-1.0 / shape)
    probability = exceedance_rate * survival
    if probability <= 0:
        return float("inf")
    return float(1.0 / (probability * observations_per_year))


def _pinball_loss(
    parameters: np.ndarray, times: np.ndarray, values: np.ndarray, quantile: float
) -> float:
    """Asymmetric absolute loss, minimised by the conditional quantile."""
    residuals = values - (parameters[0] + parameters[1] * times)
    return float(
        np.sum(np.where(residuals >= 0, quantile * residuals,
                        (quantile - 1.0) * residuals))
    )


def moving_threshold(
    times: np.ndarray, values: np.ndarray, quantile: float = POT_QUANTILE
) -> tuple[float, float]:
    """A tail threshold that moves with the record: linear quantile regression.

    **Why a fixed threshold cannot answer the trend question.** Ask whether a
    tail is *widening* while holding the threshold still, and a distribution
    that merely *slides* will answer yes. As the whole distribution shifts up,
    its lower reaches cross a fixed bar and the excesses above that bar are
    drawn from an ever-less-truncated region, so the fitted scale grows with no
    change in the tail's width at all. On the cold side the same drift empties
    the region above the bar and the scale appears to shrink.

    That is not hypothetical. On a simulated record with a pure location drift
    of 0.02 sigma a year and a rigorously constant variance, the fixed-threshold
    fit called the cold tail significantly narrowing, p = 0.002. Nothing had
    narrowed. With the threshold allowed to move, the same record gives a
    threshold slope of 0.0202 a year -- recovering the drift that was put in --
    and no width trend in either direction, p = 0.69 and p = 0.17.

    So the threshold is fitted as a line in time by minimising the pinball
    loss, which is the loss whose minimiser is the conditional quantile, and
    the GPD is fitted to excesses above *that*. The slope it returns is a
    quantity in its own right and the one the caller should read as "the
    climate moved": it separates cleanly from the scale trend, which is then
    free to mean what its name says.

    Returns:
        Intercept and slope, in the units of ``values`` per unit of ``times``.
    """
    start = np.array([float(np.quantile(values, quantile)), 0.0])
    fitted = minimize(
        _pinball_loss,
        start,
        args=(times, values, quantile),
        method="Nelder-Mead",
        options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-8},
    )
    if not fitted.success:
        # A flat threshold at the marginal quantile: the same answer the fixed
        # version would give, and the caller sees `threshold_converged` false
        # rather than a slope invented by a failed optimiser.
        return float(start[0]), 0.0
    return float(fitted.x[0]), float(fitted.x[1])


def fit_directional_trend(
    frame: pd.DataFrame,
    direction: str,
    *,
    quantile: float = POT_QUANTILE,
    separation: int = RUN_SEPARATION_DAYS,
    minimum_clusters: int = MIN_CLUSTERS,
) -> dict[str, Any]:
    """Is the *warm* tail widening? Is the *cold* one? Asked separately.

    **Why this is not one fit on ``abs(Z)``.** The Z-scores are referenced to a
    leave-one-year-out baseline computed over the whole 1995-2026 record, so
    under a warming climate the early years sit below that baseline and the
    late years above it. The exceedances therefore do not just get larger or
    smaller over the record, they *change sign*: the warm share of declustered
    exceedances rises in all eleven fitted cities, from 7.5% to 61% in Lagos
    and 19% to 53% in Singapore, comparing before and after 2011.

    A single log-linear scale trend fitted to the folded ``abs(Z)`` exceedances
    is fitted to a mixture whose composition inverts across the record, and it
    reports that inversion as a change in width. That is not a subtle bias. The
    folded fit called Phoenix and Reykjavik significantly *narrowing*
    (p = 0.017, p = 0.006) -- cities whose warm exceedance share nearly doubled
    -- because their cold tail, which dominates early, is retreating faster
    than their warm tail is growing, and one parameter cannot say both.

    So each direction gets its own threshold, its own declustering and its own
    fit. ``warm`` is the upper tail of signed Z, ``cold`` the lower.

    **And the threshold moves.** See `moving_threshold`: a bar held still while
    the distribution slides under it turns a location drift into an apparent
    change in width, which is the same conflation one level down. The
    per-direction split stops the two *tails* being averaged together; the
    moving threshold stops each tail's *drift* being read as its *width*.

    Returns:
        The `fit_nonstationary` payload, plus ``threshold_intercept``,
        ``threshold_slope_per_year`` -- the drift, which is the quantity to
        read as "the climate moved" -- and ``clusters``, so the trend can be
        judged against the sample it came from. ``{"converged": False}`` with
        those still present when there is too little tail to fit.
    """
    if direction not in {"warm", "cold"}:
        raise ValueError(f"direction must be 'warm' or 'cold', got {direction!r}.")

    signed = frame["z_temperature_2m_mean"]
    # The cold tail is reflected so it is an upper tail like any other: the GPD
    # is a model for exceedances above a threshold, and negating is what makes
    # "five sigma below normal" one of those.
    oriented = frame.assign(z=signed if direction == "warm" else -signed)
    ordered = oriented.sort_values("date_key", kind="stable")

    years = (
        ordered["date_key"] - ordered["date_key"].min()
    ).dt.days.to_numpy(dtype=float) / DAYS_PER_YEAR
    values = ordered["z"].to_numpy(dtype=float)
    intercept, slope = moving_threshold(years, values, quantile)
    ordered = ordered.assign(tail_threshold=intercept + slope * years)

    # `decluster` takes a scalar bar, so the exceedance test is done here
    # against the moving one and the runs are collapsed on the excess.
    above = ordered.loc[ordered["z"] > ordered["tail_threshold"]].copy()
    above["excess"] = above["z"] - above["tail_threshold"]
    peaks = decluster(above, 0.0, separation=separation, column="excess")

    summary = {
        "threshold_intercept": intercept,
        "threshold_slope_per_year": slope,
        "clusters": len(peaks),
    }
    if len(peaks) < minimum_clusters:
        return {"converged": False, **summary}

    since = (
        peaks["date_key"] - peaks["date_key"].min()
    ).dt.days.to_numpy(dtype=float) / DAYS_PER_YEAR
    return {
        **fit_nonstationary(peaks["excess"].to_numpy(dtype=float), since),
        **summary,
    }


@dataclass
class CityTail:
    """One city's fitted tail, with everything needed to read it sceptically."""

    city_id: str
    threshold: float
    observations: int
    exceedances: int
    clusters: int
    mean_cluster_days: float
    exceedance_rate: float
    shape: float
    scale: float
    shape_interval: tuple[float, float]
    scale_interval: tuple[float, float]
    return_levels: dict[int, float]
    warm_share_early: float
    warm_share_late: float
    warm_trend: dict[str, Any] = field(default_factory=dict)
    cold_trend: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "city_id": self.city_id,
            "threshold": self.threshold,
            "observations": self.observations,
            "exceedances": self.exceedances,
            "clusters": self.clusters,
            "mean_cluster_days": self.mean_cluster_days,
            "exceedance_rate": self.exceedance_rate,
            "shape": self.shape,
            "shape_low": self.shape_interval[0],
            "shape_high": self.shape_interval[1],
            "scale": self.scale,
            "scale_low": self.scale_interval[0],
            "scale_high": self.scale_interval[1],
            # A bounded tail is a claim about the world -- that there is a
            # hottest possible day -- and the interval is what says whether the
            # data supports it. Recorded as its own field because a reader
            # comparing two negative point estimates would otherwise have to
            # do this arithmetic themselves.
            "tail_is_bounded": bool(self.shape_interval[1] < 0),
            "shape_interval_excludes_zero": bool(
                self.shape_interval[0] > 0 or self.shape_interval[1] < 0
            ),
            # The evidence for fitting the trends by direction, kept beside
            # them so a reader meeting a warm and a cold trend that disagree
            # can see why one folded number would have been neither.
            "warm_share_early": self.warm_share_early,
            "warm_share_late": self.warm_share_late,
            **{
                f"return_level_{period}y": value
                for period, value in self.return_levels.items()
            },
            **{
                f"warm_trend_{key}": value
                for key, value in self.warm_trend.items()
            },
            **{
                f"cold_trend_{key}": value
                for key, value in self.cold_trend.items()
            },
        }


#: Read the shipped Z, not the detrended one.
#:
#: Proposal §5.3 ships *unusual for the record* and gives the reason: detrending
#: the centre removes five per cent of the drift, because "the drift lives in
#: the tail and the trend lives in the centre". This module fits the tail
#: directly, so it is the one place in the project that can go after that drift
#: where it actually lives -- `fit_nonstationary` puts the time trend in the
#: scale of the exceedances rather than in the mean they are measured from. The
#: detrended column would subtract a centre-trend first and leave the tail
#: trend to be found twice.
OBSERVATIONS_SQL: Final[str] = """
    select
        city_id,
        date_key,
        z_temperature_2m_mean
    from gold_marts.fact_weather_anomalies
    where z_temperature_2m_mean is not null
    order by city_id, date_key
"""

EXTREMES_TABLE: Final[str] = "gold_marts.fact_extreme_value"
SCHEMA_SQL: Final[Path] = Path(__file__).resolve().parent / "extremes.sql"


def load_observations(engine: Engine) -> pd.DataFrame:
    """Every scored day, both tails folded into one."""
    frame = pd.read_sql(text(OBSERVATIONS_SQL), engine)
    frame["date_key"] = pd.to_datetime(frame["date_key"])
    return frame


def _bootstrap_intervals(
    peaks: pd.DataFrame,
    *,
    excess_column: str,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = SEED,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Percentile intervals for shape and scale, resampling *clusters*.

    The unit of independence is the cluster, not the day. Resampling days would
    put five days of one heatwave into a replicate as five draws, understate
    the variance, and return intervals narrower than the data earns -- the
    same error declustering exists to prevent, reintroduced one level down.

    Since declustering leaves one row per cluster, resampling rows here *is*
    resampling clusters; the assertion below is what keeps that true if the
    declustering ever changes.
    """
    if peaks["cluster"].nunique() != len(peaks):
        raise ValueError(
            "bootstrap expects one row per cluster; decluster first."
        )
    excesses = peaks[excess_column].to_numpy(dtype=float)
    generator = np.random.default_rng(seed)
    shapes: list[float] = []
    scales: list[float] = []
    for _ in range(samples):
        draw = generator.choice(excesses, size=len(excesses), replace=True)
        if np.allclose(draw, draw[0]):
            continue
        try:
            shape, scale, _ = fit_stationary(draw)
        except (ValueError, RuntimeError):
            continue
        if np.isfinite(shape) and np.isfinite(scale):
            shapes.append(shape)
            scales.append(scale)
    if len(shapes) < samples // 2:
        # Too many replicates failed to fit for a percentile of them to mean
        # anything. An interval nobody can defend is worse than no interval.
        return (float("nan"), float("nan")), (float("nan"), float("nan"))
    return (
        (float(np.percentile(shapes, 2.5)), float(np.percentile(shapes, 97.5))),
        (float(np.percentile(scales, 2.5)), float(np.percentile(scales, 97.5))),
    )


def fit_city(
    frame: pd.DataFrame,
    city_id: str,
    *,
    quantile: float = POT_QUANTILE,
    separation: int = RUN_SEPARATION_DAYS,
    minimum_clusters: int = MIN_CLUSTERS,
    samples: int = BOOTSTRAP_SAMPLES,
) -> CityTail | None:
    """One city's tail, or ``None`` when there is not enough of it to fit.

    Both tails are folded together on ``abs(z)``, matching `is_anomaly`, which
    is on ``abs(z)`` for the reason the dbt test spells out: a one-sided flag
    loses Moscow's January entirely. A return period here is therefore *a day
    this far from normal in either direction*, which is the quantity the
    Anomaly Map already colours.
    """
    magnitudes = frame.assign(z=frame["z_temperature_2m_mean"].abs())
    threshold = float(magnitudes["z"].quantile(quantile))
    peaks = decluster(magnitudes, threshold, separation=separation, column="z")
    if len(peaks) < minimum_clusters:
        log.info(
            "%s: %d clusters, below the %d needed to fit; skipped.",
            city_id, len(peaks), minimum_clusters,
        )
        return None

    peaks = peaks.assign(excess=peaks["z"] - threshold)
    shape, scale, _ = fit_stationary(peaks["excess"].to_numpy())
    shape_interval, scale_interval = _bootstrap_intervals(
        peaks, excess_column="excess", samples=samples
    )

    observations = len(magnitudes)
    exceedances = int((magnitudes["z"] > threshold).sum())
    # The declustered rate: events per observed day. Using `exceedances` here
    # instead of `len(peaks)` would shorten every return period below by the
    # mean cluster length, which is the headline error this module is built to
    # avoid, so it is written once and passed everywhere.
    rate = len(peaks) / observations

    # The composition shift that forces the trends to be fitted per direction.
    # Split at the midpoint of the record rather than a fixed year, so a city
    # with a shorter history is halved rather than compared to somebody else's
    # calendar.
    midpoint = peaks["date_key"].min() + (
        peaks["date_key"].max() - peaks["date_key"].min()
    ) / 2
    warm = peaks["z_temperature_2m_mean"] > 0
    early, late = peaks["date_key"] <= midpoint, peaks["date_key"] > midpoint

    return CityTail(
        city_id=city_id,
        threshold=threshold,
        observations=observations,
        exceedances=exceedances,
        clusters=len(peaks),
        mean_cluster_days=float(peaks["cluster_days"].mean()),
        exceedance_rate=rate,
        shape=shape,
        scale=scale,
        shape_interval=shape_interval,
        scale_interval=scale_interval,
        return_levels={
            period: return_level(
                period,
                threshold=threshold,
                shape=shape,
                scale=scale,
                exceedance_rate=rate,
            )
            for period in RETURN_PERIODS
        },
        warm_share_early=float(warm[early].mean()) if early.any() else float("nan"),
        warm_share_late=float(warm[late].mean()) if late.any() else float("nan"),
        warm_trend=fit_directional_trend(
            frame, "warm", quantile=quantile, separation=separation,
            minimum_clusters=minimum_clusters,
        ),
        cold_trend=fit_directional_trend(
            frame, "cold", quantile=quantile, separation=separation,
            minimum_clusters=minimum_clusters,
        ),
    )


def build_extremes(
    frame: pd.DataFrame,
    *,
    quantile: float = POT_QUANTILE,
    separation: int = RUN_SEPARATION_DAYS,
    minimum_clusters: int = MIN_CLUSTERS,
    samples: int = BOOTSTRAP_SAMPLES,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Fit every city that has enough tail to fit.

    Returns:
        The parameter table, and why each skipped city was skipped -- the same
        shape `predict.coverage` returns, because "which cities are missing and
        why" is a question every stage of this project has to answer and it
        should not be answered differently in each one.
    """
    fits: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    for city_id, city in frame.groupby("city_id", sort=True):
        tail = fit_city(
            city,
            str(city_id),
            quantile=quantile,
            separation=separation,
            minimum_clusters=minimum_clusters,
            samples=samples,
        )
        if tail is None:
            magnitudes = city["z_temperature_2m_mean"].abs()
            above = int((magnitudes > magnitudes.quantile(quantile)).sum())
            skipped[str(city_id)] = (
                f"{len(city):,} scored days, {above} exceedances, fewer than "
                f"{minimum_clusters} declustered clusters"
            )
            continue
        fits.append(tail.as_dict())
    return pd.DataFrame(fits), skipped


#: The columns the mart keeps, in order.
#:
#: `CityTail.as_dict` also carries the intermediate quantities the fit produced
#: -- the log-scale intercept, both log-likelihoods, the trend's own shape --
#: which belong in the printed report and not in a table other things join to.
#: Listing the kept columns here rather than writing whatever the dataclass
#: happens to hold means adding a diagnostic to the report cannot silently
#: change the schema.
PERSISTED_COLUMNS: Final[tuple[str, ...]] = (
    "city_id",
    "threshold",
    "observations",
    "exceedances",
    "clusters",
    "mean_cluster_days",
    "exceedance_rate",
    "shape",
    "shape_low",
    "shape_high",
    "scale",
    "scale_low",
    "scale_high",
    "tail_is_bounded",
    "shape_interval_excludes_zero",
    "return_level_2y",
    "return_level_5y",
    "return_level_10y",
    "return_level_20y",
    "return_level_50y",
    "warm_share_early",
    "warm_share_late",
    "warm_trend_threshold_intercept",
    "warm_trend_threshold_slope_per_year",
    "warm_trend_clusters",
    "warm_trend_converged",
    "warm_trend_scale_trend_per_year",
    "warm_trend_scale_change_over_record",
    "warm_trend_p_value",
    "cold_trend_threshold_intercept",
    "cold_trend_threshold_slope_per_year",
    "cold_trend_clusters",
    "cold_trend_converged",
    "cold_trend_scale_trend_per_year",
    "cold_trend_scale_change_over_record",
    "cold_trend_p_value",
    "pot_quantile",
    "run_separation_days",
    "bootstrap_samples",
)

INTEGER_COLUMNS: Final[frozenset[str]] = frozenset(
    {"observations", "exceedances", "clusters", "run_separation_days",
     "bootstrap_samples", "warm_trend_clusters", "cold_trend_clusters"}
)
BOOLEAN_COLUMNS: Final[frozenset[str]] = frozenset(
    {"tail_is_bounded", "shape_interval_excludes_zero",
     "warm_trend_converged", "cold_trend_converged"}
)


def apply_schema(engine: Engine) -> None:
    """Apply the idempotent DDL."""
    with engine.begin() as connection:
        connection.execute(text(SCHEMA_SQL.read_text()))


def write_extremes(engine: Engine, fits: pd.DataFrame) -> int:
    """Upsert one row per city. Refitting replaces."""
    if fits.empty:
        return 0
    prepared = fits.reindex(columns=list(PERSISTED_COLUMNS))
    assignments = ",\n            ".join(
        f"{column} = excluded.{column}"
        for column in PERSISTED_COLUMNS
        if column != "city_id"
    )
    statement = text(
        f"""
        insert into {EXTREMES_TABLE} (
            {", ".join(PERSISTED_COLUMNS)}, fitted_at
        ) values (
            {", ".join(f":{column}" for column in PERSISTED_COLUMNS)}, now()
        )
        on conflict (city_id) do update set
            {assignments},
            fitted_at = now()
        """
    )
    rows: list[dict[str, Any]] = []
    for record in prepared.to_dict("records"):
        row: dict[str, Any] = {}
        for column, value in record.items():
            # NaN is how a non-converging fit and a missing bootstrap arrive
            # here, and Postgres numeric has no NaN worth storing: a null says
            # "not estimated", which is what happened.
            if value is None or (
                isinstance(value, float) and not np.isfinite(value)
            ):
                row[column] = None
            elif column in INTEGER_COLUMNS:
                row[column] = int(value)
            elif column in BOOLEAN_COLUMNS:
                row[column] = bool(value)
            elif column == "city_id":
                row[column] = str(value)
            else:
                row[column] = float(value)
        rows.append(row)
    with engine.begin() as connection:
        connection.execute(statement, rows)
    return len(rows)


def _fit_report(fits: pd.DataFrame) -> str:
    """The table a reader should look at before believing any return period."""
    view = fits.loc[
        :,
        ["city_id", "threshold", "clusters", "mean_cluster_days", "shape",
         "shape_low", "shape_high", "scale", "return_level_10y",
         "return_level_50y"],
    ].copy()
    for column in view.columns:
        if column in {"city_id", "clusters"}:
            continue
        view[column] = view[column].map(
            lambda value: "  --  " if pd.isna(value) else f"{value:.3f}"
        )
    return view.to_string(index=False)


def _trend_report(fits: pd.DataFrame) -> str:
    """Whether each city's warm and cold tails are widening, side by side.

    Reported separately from the parameter table because it answers a different
    question and invites a different mistake. A small p-value here does not
    mean the return levels above are wrong; it means they are an average over
    a record in which the tail was not constant, and one end of that record is
    wider than the average. Whether that matters is a decision about the
    product, and it is made in the model card rather than here.

    The two warm-share columns are printed alongside because they are why
    there are two trends and not one: see `fit_directional_trend`.

    ``drift`` is the tail threshold's own slope, in sigma a year -- how far the
    tail has *moved*. ``x`` is what the fitted scale does over the record once
    that movement is taken out -- how much it has *widened*. They are separate
    columns because they are separate claims, and a single non-stationary fit
    against a fixed bar would have reported their sum as the second one.
    """
    view = fits.loc[
        :,
        ["city_id", "warm_share_early", "warm_share_late",
         "warm_trend_threshold_slope_per_year",
         "warm_trend_scale_change_over_record", "warm_trend_p_value",
         "cold_trend_threshold_slope_per_year",
         "cold_trend_scale_change_over_record", "cold_trend_p_value"],
    ].copy()
    view.columns = ["city_id", "warm_early", "warm_late",
                    "warm_drift", "warm_x", "warm_p",
                    "cold_drift", "cold_x", "cold_p"]
    for column in view.columns[1:]:
        view[column] = view[column].map(
            lambda value: "  --  " if pd.isna(value) else f"{value:.3f}"
        )
    return view.to_string(index=False)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a generalised Pareto tail per city and report it."
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=POT_QUANTILE,
        help=f"Per-city quantile where the tail begins (default {POT_QUANTILE}).",
    )
    parser.add_argument(
        "--separation",
        type=int,
        default=RUN_SEPARATION_DAYS,
        help=(
            "Quiet days between exceedances before they are separate events "
            f"(default {RUN_SEPARATION_DAYS})."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=BOOTSTRAP_SAMPLES,
        help=f"Bootstrap resamples over clusters (default {BOOTSTRAP_SAMPLES}).",
    )
    parser.add_argument(
        "--write", action="store_true", help="Write the mart. Default prints only."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    engine = engine_from_settings()
    try:
        observations = load_observations(engine)
        print(
            f"{len(observations):,} scored days across "
            f"{observations['city_id'].nunique()} cities"
        )
        fits, skipped = build_extremes(
            observations,
            quantile=args.quantile,
            separation=args.separation,
            samples=args.samples,
        )
        if fits.empty:
            print("\nno city has enough tail to fit.")
            return 1

        fits = fits.assign(
            pot_quantile=args.quantile,
            run_separation_days=args.separation,
            bootstrap_samples=args.samples,
        )
        print(
            f"\ntail fitted at the {args.quantile:.0%} quantile, "
            f"run separation {args.separation} days, "
            f"{args.samples} bootstrap resamples over clusters"
        )
        print(f"\n{_fit_report(fits)}")
        bounded = int(fits["tail_is_bounded"].sum())
        decided = int(fits["shape_interval_excludes_zero"].sum())
        print(
            f"\n{decided} of {len(fits)} cities have a shape interval that "
            f"excludes zero; {bounded} are bounded above."
        )
        print(
            "\ntrend per direction: drift = threshold slope in sigma/yr, "
            "x = scale multiplier over the record once drift is removed"
        )
        print(_trend_report(fits))

        for city_id, reason in sorted(skipped.items()):
            print(f"  skipped {city_id:<14} {reason}")

        if not args.write:
            print("\n(no --write: nothing written)")
            return 0

        apply_schema(engine)
        written = write_extremes(engine, fits)
        print(f"\nwrote {written} rows to {EXTREMES_TABLE}")
    finally:
        engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
