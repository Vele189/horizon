"""Tests for the chronological split harness.

The second of the project's two leakage traps, and the one that most often
survives into a published repository unnoticed — because a random split does
not look wrong. It produces a training set, a test set, and a number, and the
number is simply too good for a reason nothing in the code points at.

So the harness is checked in three ways, weakest to strongest:

1. **Ordered.** Every split ends strictly before the next begins. This is what
   a shuffle breaks, and it is the assertion the ticket names.
2. **Disjoint.** No *label window* crosses a boundary either. Strictly stronger:
   a split can be perfectly ordered and still hand the training set the first
   week of validation through the label.
3. **Confirmed end to end.** Rewrite every observation from the validation
   boundary onwards, rebuild features and labels from scratch, split again, and
   require the training split to come back bit-identical. That is the whole
   claim in one assertion, and it does not care how any individual window is
   written.

The converse is tested too, and deliberately: a validation row's rolling window
*does* reach back into the training period, and must. That is not leakage, it
is deployment — a model predicting on 2019-01-01 in production has all of 2018
behind it. A test asserts the backward reach exists so that nobody later
"fixes" it.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from ml_fixtures import (  # noqa: E402
    labelled_span,
    rewrite_from,
    scored_population,
    spanning,
)

from machine_learning.evaluation import (  # noqa: E402
    EMBARGO_DAYS,
    PURGE_DAYS,
    SPLITS,
    assert_splits_are_disjoint,
    assert_splits_are_ordered,
    boundary_report,
    evaluation_frame,
    split_frame,
    split_summary,
)
from machine_learning.features import WARMUP_DAYS, feature_columns  # noqa: E402
from machine_learning.labels import HORIZON_DAYS, LABEL  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

VALIDATION_START = SPLITS[1].start
TEST_START = SPLITS[2].start


# --------------------------------------------------------------------------
# The periods, and the function
# --------------------------------------------------------------------------


def test_the_periods_are_the_ones_the_proposal_specifies() -> None:
    assert [split.name for split in SPLITS] == ["train", "validation", "test"]
    assert SPLITS[0].start == dt.date(1995, 1, 1)
    assert SPLITS[0].end == dt.date(2018, 12, 31)
    assert SPLITS[1].start == dt.date(2019, 1, 1)
    assert SPLITS[1].end == dt.date(2021, 12, 31)
    assert SPLITS[2].start == dt.date(2022, 1, 1)
    assert SPLITS[2].end is None, "the test split runs to the end of the record"


def test_every_split_is_present_even_when_empty() -> None:
    """A caller iterating the splits must not silently skip a missing one."""
    parts = split_frame(labelled_span(days=400))  # 1995–1996 only
    assert set(parts) == {"train", "validation", "test"}
    assert parts["validation"].empty
    assert parts["test"].empty


def test_the_split_is_a_function_of_the_dates_and_nothing_else() -> None:
    """Same rows in any order, same split. No hidden dependence on arrival."""
    frame = labelled_span()
    ordered = split_frame(frame)
    shuffled = split_frame(frame.sample(frac=1.0, random_state=6))
    for name in ordered:
        pd.testing.assert_frame_equal(
            ordered[name].sort_values("date_key").reset_index(drop=True),
            shuffled[name].sort_values("date_key").reset_index(drop=True),
        )


# --------------------------------------------------------------------------
# Ordered, and what breaks it
# --------------------------------------------------------------------------


def test_the_maximum_train_date_is_strictly_less_than_the_minimum_validation() -> None:
    """The ticket's assertion, stated plainly."""
    parts = split_frame(labelled_span())
    assert parts["train"]["date_key"].max() < parts["validation"]["date_key"].min()
    assert parts["validation"]["date_key"].max() < parts["test"]["date_key"].min()
    assert_splits_are_ordered(parts)


def test_a_shuffled_split_is_caught_the_moment_it_is_checked() -> None:
    """What a random split looks like to this harness.

    ``train_test_split`` on this frame would put 2023 rows in training and 1997
    rows in test, and every downstream number would still compute. The ordering
    assertion is what turns that into an error message.
    """
    frame = labelled_span()
    shuffled = frame.sample(frac=1.0, random_state=11).reset_index(drop=True)
    cut = int(len(shuffled) * 0.7)
    random_parts = {
        "train": shuffled.iloc[:cut].reset_index(drop=True),
        "validation": shuffled.iloc[cut : cut + 1000].reset_index(drop=True),
        "test": shuffled.iloc[cut + 1000 :].reset_index(drop=True),
    }
    with pytest.raises(ValueError, match="not chronological"):
        assert_splits_are_ordered(random_parts)


def test_disjointness_is_strictly_stronger_than_ordering() -> None:
    """A split can be perfectly ordered and still leak through the label."""
    frame = labelled_span()
    unpurged = split_frame(frame, purge=False)

    assert_splits_are_ordered(unpurged)  # the rows do not overlap
    with pytest.raises(ValueError, match="Purge the boundary"):
        assert_splits_are_disjoint(unpurged)  # what they know does


# --------------------------------------------------------------------------
# The purge
# --------------------------------------------------------------------------


def test_the_purge_costs_exactly_the_horizon_at_each_boundary() -> None:
    frame = labelled_span()
    purged = split_frame(frame, purge=True)
    unpurged = split_frame(frame, purge=False)

    assert len(unpurged["train"]) - len(purged["train"]) == PURGE_DAYS
    assert len(unpurged["validation"]) - len(purged["validation"]) == PURGE_DAYS
    # The last split has nothing after it to leak into, so nothing is dropped.
    assert len(unpurged["test"]) == len(purged["test"])
    assert_splits_are_disjoint(purged)


def test_the_boundary_report_shows_the_purge_as_a_number() -> None:
    report = boundary_report(split_frame(labelled_span())).set_index("boundary")
    assert list(report.index) == ["train → validation", "validation → test"]
    assert (report["gap_days"] > HORIZON_DAYS).all()
    assert report["reach_clears"].all()


# --------------------------------------------------------------------------
# Confirmed end to end: nothing in train depends on the future
# --------------------------------------------------------------------------


def test_rewriting_the_validation_era_leaves_the_training_split_untouched() -> None:
    """The whole claim, in one assertion.

    Every observation from 2019-01-01 onwards is replaced with nonsense, the
    features and the label are rebuilt from scratch, and the training split
    must come back identical — every rolling window, every lag, every label.
    This does not inspect how any window is written, so a window that reaches
    forward fails it however cleverly it is expressed.
    """
    gold = spanning()
    baseline = split_frame(scored_population(gold))["train"]
    rebuilt = split_frame(scored_population(rewrite_from(gold, VALIDATION_START)))[
        "train"
    ]

    assert len(baseline) > 8000, "the training split should be most of the record"
    pd.testing.assert_frame_equal(baseline, rebuilt)


def test_rewriting_the_test_era_leaves_train_and_validation_untouched() -> None:
    gold = spanning()
    baseline = split_frame(scored_population(gold))
    rebuilt = split_frame(scored_population(rewrite_from(gold, TEST_START)))

    for name in ("train", "validation"):
        assert not baseline[name].empty
        pd.testing.assert_frame_equal(baseline[name], rebuilt[name])


def test_without_the_purge_the_future_does_reach_the_training_split() -> None:
    """The same comparison, with the purge off. It fails, and it should.

    Which is what makes the two tests above worth running: the boundary rows
    the purge removes are exactly the ones whose label is written by the period
    they are about to be validated on.
    """
    gold = spanning()
    baseline = split_frame(scored_population(gold), purge=False)["train"]
    rebuilt = split_frame(
        scored_population(rewrite_from(gold, VALIDATION_START)), purge=False
    )["train"]

    assert len(baseline) == len(rebuilt)
    assert not baseline[LABEL].equals(rebuilt[LABEL]), (
        "the unpurged training split did not move when validation was "
        "rewritten, so the purge is not removing what it claims to"
    )
    # And it is precisely the last PURGE_DAYS rows that moved.
    moved = baseline.index[~baseline[LABEL].eq(rebuilt[LABEL]).fillna(False)]
    assert moved.min() >= len(baseline) - PURGE_DAYS


def test_a_validation_feature_does_reach_back_into_training() -> None:
    """Backward reach across a boundary is deployment, not leakage.

    A 30-day mean on 2019-01-05 is built from December 2018 — and must be. A
    model predicting that day in production has all of 2018 behind it, and
    blanking it here would measure a system nobody is going to run. This test
    asserts the reach exists so it is not later "fixed" into a harness that
    reports a worse number for a better-sounding reason.
    """
    gold = spanning()
    # Rewrite 2018 only — entirely inside training — and leave validation's own
    # observations alone, so anything that moves there moved by reaching back.
    disturbed = rewrite_from(gold, "2018-01-01", until="2018-12-31")
    baseline = split_frame(scored_population(gold))["validation"]
    rebuilt = split_frame(scored_population(disturbed))["validation"]

    columns = list(feature_columns())
    opening = baseline.head(WARMUP_DAYS)[columns]
    assert not opening.equals(rebuilt.head(WARMUP_DAYS)[columns]), (
        "a validation row's rolling window no longer reaches back across the "
        "boundary — which would mean the harness is measuring a model with no "
        "history at prediction time, and that is not the model being deployed"
    )

    # And the reach is bounded by the longest window: past it the features have
    # cleared the boundary entirely, so those rows are identical again.
    tail = baseline.iloc[WARMUP_DAYS + 1 :]
    rewritten_tail = rebuilt.iloc[WARMUP_DAYS + 1 :]
    pd.testing.assert_frame_equal(tail, rewritten_tail)


# --------------------------------------------------------------------------
# The embargo, which is off, and the measurement that says why
# --------------------------------------------------------------------------


def test_the_embargo_is_off_by_default() -> None:
    assert EMBARGO_DAYS == 0
    frame = labelled_span()
    assert split_frame(frame)["validation"].equals(
        split_frame(frame, embargo_days=0)["validation"]
    )


def test_the_embargo_removes_the_start_of_every_later_split() -> None:
    frame = labelled_span()
    plain = split_frame(frame)
    held = split_frame(frame, embargo_days=WARMUP_DAYS)

    for name in ("validation", "test"):
        assert len(held[name]) < len(plain[name])
        assert held[name]["date_key"].min() >= plain[name]["date_key"].min() + (
            pd.Timedelta(days=WARMUP_DAYS)
        )
    # Training has nothing before it, so it is untouched.
    pd.testing.assert_frame_equal(plain["train"], held["train"])
    assert_splits_are_disjoint(held)


def test_a_negative_embargo_is_refused() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        split_frame(labelled_span(days=400), embargo_days=-1)


# --------------------------------------------------------------------------
# Positive rate per split
# --------------------------------------------------------------------------


def test_every_split_reports_a_positive_rate() -> None:
    summary = split_summary(split_frame(labelled_span()))
    assert list(summary["split"]) == ["train", "validation", "test"]
    assert (summary["rows"] > 0).all()
    assert summary["base_rate"].between(0.0, 1.0).all()
    assert (summary["positives"] <= summary["rows"]).all()


# --------------------------------------------------------------------------
# No random splitter anywhere
# --------------------------------------------------------------------------

#: Names that mean the split was not chronological. ``sklearn.model_selection``
#: is forbidden wholesale rather than function by function: everything in it
#: shuffles except ``TimeSeriesSplit``, and a project that needs that one should
#: reach for it deliberately and delete this line, not slip it past a list.
FORBIDDEN = (
    "train_test_split",
    "sklearn.model_selection",
    "ShuffleSplit",
    "KFold",
    "cross_val_score",
    "cross_validate",
)


def test_no_random_splitter_appears_anywhere_in_the_codebase() -> None:
    """The ticket asks for its absence, so absence is what is checked.

    Scans every ``.py`` file in the repository, not just the machine-learning
    package: the failure this guards against is somebody adding a quick
    train/test split in a dashboard script or a notebook-shaped helper, where
    nobody would think to look for it.

    This file is skipped, because it has to name what it forbids.
    """
    offenders = []
    for path in sorted(REPO_ROOT.rglob("*.py")):
        relative = path.relative_to(REPO_ROOT)
        if relative.parts[0] in {".venv", ".git"} or path == Path(__file__):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for name in FORBIDDEN:
            if name in text:
                offenders.append(f"{relative}: {name}")

    assert not offenders, (
        "a random splitter reached the codebase — a shuffled split leaks the "
        f"future through every rolling feature: {offenders}"
    )


def test_the_scan_would_notice(tmp_path) -> None:
    """Proof the scan can fail, since it passes by finding nothing."""
    planted = tmp_path / "sneaky.py"
    planted.write_text("from sklearn.model_selection import train_test_split\n")
    text = planted.read_text()
    assert any(name in text for name in FORBIDDEN)


# --------------------------------------------------------------------------
# Against the warehouse
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def population(engine):
    from sqlalchemy import text

    with engine.connect() as connection:
        rows = connection.execute(
            text("select count(*) from gold_marts.fact_weather_anomalies")
        ).scalar()
    if not rows:
        pytest.skip("fact_weather_anomalies not built")
    frame = evaluation_frame(engine)
    parts = split_frame(frame)
    if any(part.empty for part in parts.values()):
        pytest.skip("not enough of the record backfilled to fill every split")
    return frame


def test_the_real_split_is_ordered_and_disjoint(population) -> None:
    parts = split_frame(population)
    assert_splits_are_disjoint(parts)
    assert parts["train"]["date_key"].max() == pd.Timestamp("2018-12-24")
    assert parts["validation"]["date_key"].min() == pd.Timestamp("2019-01-01")


def test_the_real_split_reports_a_rising_positive_rate(population) -> None:
    """Reported per split, and the report is where the trend is visible."""
    summary = split_summary(split_frame(population)).set_index("split")
    for name in ("train", "validation", "test"):
        assert summary.loc[name, "rows"] > 1000
        assert 0.01 < summary.loc[name, "base_rate"] < 0.30
    assert (
        summary.loc["train", "base_rate"]
        < summary.loc["validation", "base_rate"]
        < summary.loc["test", "base_rate"]
    )


def test_the_embargo_would_not_change_the_verdict(population) -> None:
    """The measurement behind leaving it off.

    Holding back a month at the start of each later split — which is what
    removing every shared feature window would take — moves test PR-AUC by
    about a thousandth, and *upward*. The sample-correlation optimism an
    embargo exists to remove is not present at this window length, so paying
    for it in realism would buy nothing.
    """
    from machine_learning.baselines import PersistenceBaseline
    from machine_learning.evaluation import score

    scores = {}
    for embargo in (0, WARMUP_DAYS):
        parts = split_frame(population, embargo_days=embargo)
        fitted = PersistenceBaseline().fit(parts["train"])
        scores[embargo] = score(
            parts["test"][LABEL], fitted.predict(parts["test"])
        ).pr_auc

    assert abs(scores[WARMUP_DAYS] - scores[0]) < 0.01, (
        f"the embargo now moves test PR-AUC from {scores[0]:.4f} to "
        f"{scores[WARMUP_DAYS]:.4f}; overlapping feature windows have started "
        "to matter and the argument for leaving it off needs revisiting"
    )
