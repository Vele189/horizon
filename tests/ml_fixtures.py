"""Synthetic gold frames for the machine-learning suites.

Shared rather than copied because two suites assert things about the *same*
pipeline: the split tests need exactly the frame the baseline tests use, or
"the training split did not move" is a claim about a different dataset than the
one the baselines were fitted on.

The series is deliberately noisy and long enough to cross both split
boundaries. A ramp would not do: on a monotone series a window that reaches one
day the wrong way still produces plausible numbers, and an equality check
against a smooth function can pass on the wrong window by coincidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from machine_learning.evaluation import add_persistence_signal  # noqa: E402
from machine_learning.features import build_features  # noqa: E402
from machine_learning.labels import LABEL, build_labels  # noqa: E402


def spanning(
    days: int = 11000,
    *,
    city_id: str = "alpha",
    start: str = "1995-01-01",
    seed: int = 4,
) -> pd.DataFrame:
    """A gold-shaped frame long enough to cross both split boundaries."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    season = 12 * np.sin(2 * np.pi * np.arange(days) / 365.25)
    return pd.DataFrame(
        {
            "city_id": city_id,
            "date_key": dates,
            "temperature_2m_mean": 15 + season + rng.normal(scale=3.0, size=days),
            "pressure_msl_mean": 1013 + rng.normal(scale=8.0, size=days),
            "z_temperature_2m_mean": rng.normal(size=days),
            "is_anomaly": pd.array(rng.random(days) < 0.02, dtype="boolean"),
            "latitude": 30.0,
            "elevation_m": 20.0,
        }
    )


def scored_population(gold: pd.DataFrame) -> pd.DataFrame:
    """Gold rows through the whole pipeline, in the order the real one uses.

    Features and label from the same frame, persistence signal added **before**
    the trim, then unlabellable and warm-up rows dropped. The ordering is load
    bearing: reversed, the first rows of the population lose a signal they are
    entitled to, because the window falls off the start of the slice rather
    than the start of the record.
    """
    features = build_features(gold)
    labels = build_labels(gold)
    merged = add_persistence_signal(
        features.merge(labels, on=["city_id", "date_key"], how="inner")
    )
    return merged.loc[merged[LABEL].notna() & ~merged["is_warmup"]].reset_index(
        drop=True
    )


def labelled_span(**kwargs) -> pd.DataFrame:
    """:func:`spanning` all the way through :func:`scored_population`."""
    return scored_population(spanning(**kwargs))


def rewrite_from(
    gold: pd.DataFrame,
    when: str | pd.Timestamp,
    until: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """The same gold frame with a period replaced by nonsense.

    Every column a feature or a label could possibly read, moved far enough
    that no rounding tolerance could hide it. What survives unchanged after a
    rebuild is what genuinely did not depend on that period.

    Args:
        gold: The frame to rewrite.
        when: First day to rewrite, inclusive.
        until: Last day to rewrite, inclusive. Open-ended by default. Bounded
            when the point is to rewrite an *earlier* period and watch how far
            forward it reaches.
    """
    changed = gold.copy()
    later = changed["date_key"] >= pd.Timestamp(when)
    if until is not None:
        later &= changed["date_key"] <= pd.Timestamp(until)
    changed.loc[later, "temperature_2m_mean"] += 40.0
    changed.loc[later, "pressure_msl_mean"] -= 60.0
    changed.loc[later, "z_temperature_2m_mean"] *= -5.0
    changed.loc[later, "is_anomaly"] = ~changed.loc[later, "is_anomaly"]
    return changed


def repository_sources(exclude: "set[Path] | None" = None):
    """Every ``.py`` file in the repository, for the absence scans.

    Two tickets ask for something to be *absent*, a random splitter (ML-04)
    and a resampler (ML-05), and absence is only checked if something walks
    the tree. Not just the machine-learning package: the failure worth catching
    is a quick shuffled split or an oversampler in a dashboard script, where
    nobody would think to look.

    The forbidden names themselves are deliberately not written here. A helper
    every scan imports must not contain the strings those scans search for, or
    the first thing each one finds is this docstring.
    """
    root = Path(__file__).resolve().parent.parent
    skipped = exclude or set()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if relative.parts[0] in {".venv", ".git", "__pycache__"}:
            continue
        if path in skipped:
            continue
        yield relative, path.read_text(encoding="utf-8", errors="replace")
