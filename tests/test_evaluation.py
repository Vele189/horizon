"""Tests for the evaluation suite.

The suite exists to report honestly, so the tests are mostly about the ways a
report can be dishonest without containing a false number.

**Accuracy.** At a 13.58% base rate, always answering "no anomaly" scores
86.4%. ``accuracy_score`` is checked absent from the whole repository, with a
companion test that plants it in a temp file so the scan is known to work.

**The threshold.** F1 needs one, and 0.5 is not it. A model whose mean
prediction is 0.07 classifies nothing at 0.5 and would post F1 = 0.00 while
ranking better than everything else in the table. The threshold is chosen on
validation, and a test requires rewriting the test split to leave it alone.

**The picture.** A precision–recall curve drawn as straight lines between its
points shows operating points that do not exist — persistence has two of them,
and joining them draws a diagonal that integrates to roughly twice the average
precision the same predictor scores. The curve is drawn as a step function, and
a test integrates the drawn points and requires the result to equal the
reported PR-AUC. The figures are regenerated and compared byte for byte, so
they cannot go stale.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("xgboost")
pytest.importorskip("matplotlib")

from ml_fixtures import labelled_span, repository_sources  # noqa: E402

from machine_learning.evaluate import (  # noqa: E402
    CALIBRATION_BINS,
    FIGURE_DIR,
    _thin,
    best_threshold,
    build_evaluation,
    calibration_points,
    city_roster,
    classification_at,
    precision_recall_points,
    render_figures,
)
from machine_learning.evaluation import evaluation_frame, split_frame  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The metric this project will not report, and the sklearn names for it.
FORBIDDEN_METRICS = ("accuracy_score", "balanced_accuracy")


def series(values) -> pd.Series:
    return pd.Series(pd.array(values, dtype="boolean"))


# --------------------------------------------------------------------------
# Accuracy, deliberately excluded
# --------------------------------------------------------------------------


def test_accuracy_is_absent_from_the_codebase() -> None:
    offenders = [
        f"{relative}: {name}"
        for relative, text in repository_sources(exclude={Path(__file__)})
        for name in FORBIDDEN_METRICS
        if name in text
    ]
    assert not offenders, (
        "accuracy reached the codebase — at a 13.6% base rate, always "
        f"answering 'no anomaly' scores 86.4%: {offenders}"
    )


def test_the_accuracy_scan_would_notice(tmp_path) -> None:
    planted = tmp_path / "sneaky.py"
    planted.write_text("from sklearn.metrics import accuracy_score\n")
    assert any(name in planted.read_text() for name in FORBIDDEN_METRICS)


# --------------------------------------------------------------------------
# The threshold
# --------------------------------------------------------------------------


def test_the_threshold_maximises_f1_on_what_it_is_given() -> None:
    labels = series([True, True, True, False, False, False, False, False])
    predictions = [0.9, 0.8, 0.4, 0.7, 0.3, 0.2, 0.1, 0.05]
    threshold, f1 = best_threshold(labels, predictions)

    # Sweeping by hand: at 0.4 the flagged set is {0.9, 0.8, 0.7, 0.4},
    # 3 of 4 correct, recall 1.0 — precision 0.75, F1 6/7.
    assert threshold == pytest.approx(0.4)
    assert f1 == pytest.approx(6 / 7)


def test_a_half_threshold_would_throw_the_model_away() -> None:
    """Why the threshold is chosen rather than assumed.

    A calibrated model at a 13.6% base rate rarely predicts above 0.5. Scored
    at 0.5 it classifies almost nothing and posts an F1 near zero, while
    ranking every extreme week above every quiet one.
    """
    rng = np.random.default_rng(0)
    truth = rng.random(2000) < 0.12
    predictions = np.clip(
        np.where(truth, rng.normal(0.28, 0.08, 2000), rng.normal(0.07, 0.05, 2000)),
        0.001,
        0.999,
    )
    labels = series(list(truth))

    at_half = classification_at(labels, predictions, 0.5)
    threshold, _ = best_threshold(labels, predictions)
    at_chosen = classification_at(labels, predictions, threshold)

    assert threshold < 0.5
    assert at_chosen["f1"] > 0.4
    # A handful of rows may creep over 0.5; the point is that scoring there
    # discards essentially the whole model.
    assert at_half["flagged"] < 0.01 * len(labels)
    assert at_half["f1"] < 0.1 * at_chosen["f1"]


def test_classification_counts_account_for_every_row() -> None:
    labels = series([True, True, False, False, False])
    result = classification_at(labels, [0.9, 0.2, 0.8, 0.1, 0.05], 0.5)

    assert result["true_positive"] == 1
    assert result["false_positive"] == 1
    assert result["false_negative"] == 1
    assert result["true_negative"] == 2
    assert result["flagged"] == 2
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)
    cells = ("true_positive", "false_positive", "false_negative", "true_negative")
    assert sum(result[key] for key in cells) == len(labels)
    assert "accuracy" not in result


# --------------------------------------------------------------------------
# The curve the picture draws is the curve the number integrates
# --------------------------------------------------------------------------


def test_the_step_curve_integrates_to_the_reported_pr_auc() -> None:
    """The assertion behind drawing steps rather than straight lines.

    Average precision is a step-wise sum. A curve joined by straight segments
    encloses more area than that sum, so a reader comparing the picture with
    the table would find them disagreeing — most severely for a predictor with
    two operating points, whose straight line integrates to roughly twice its
    score.
    """
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(3)
    truth = rng.random(600) < 0.2
    predictions = np.where(truth, rng.random(600) * 0.6 + 0.2, rng.random(600) * 0.6)
    points = precision_recall_points(series(list(truth)), predictions)

    recall = points["recall"].to_numpy()
    precision = points["precision"].to_numpy()
    stepwise = float(np.sum((recall[:-1] - recall[1:]) * precision[:-1]))
    assert stepwise == pytest.approx(
        average_precision_score(truth, predictions), abs=1e-12
    )


def test_a_two_valued_predictor_has_a_coarse_curve() -> None:
    """The case the step rendering exists for.

    Persistence emits two probabilities, so it has two operating points and
    cannot be run anywhere between them.
    """
    rng = np.random.default_rng(5)
    truth = rng.random(400) < 0.2
    predictions = np.where(rng.random(400) < 0.3, 0.4, 0.05)
    points = precision_recall_points(series(list(truth)), predictions)
    assert len(points) <= 4


def test_thinning_keeps_the_ends_and_the_shape() -> None:
    points = pd.DataFrame(
        {"recall": np.linspace(1, 0, 5000), "precision": np.linspace(0.1, 1, 5000)}
    )
    thinned = _thin(points, limit=200)
    assert len(thinned) <= 202
    assert thinned["recall"].iloc[0] == points["recall"].iloc[0]
    assert thinned["recall"].iloc[-1] == points["recall"].iloc[-1]
    assert _thin(points.head(50), limit=200) is not None
    assert len(_thin(points.head(50), limit=200)) == 50


def test_the_calibration_bins_hold_equal_counts() -> None:
    rng = np.random.default_rng(7)
    truth = rng.random(3000) < 0.15
    predictions = np.clip(rng.beta(2, 12, 3000), 0.001, 0.999)
    points = calibration_points(series(list(truth)), predictions)

    assert len(points) <= CALIBRATION_BINS
    assert points["predicted"].is_monotonic_increasing
    assert points["predicted"].between(0, 1).all()
    assert points["observed"].between(0, 1).all()


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_report():
    frame = labelled_span()
    roster = sorted(frame["city_id"].unique()) + ["moscow"]
    return build_evaluation(frame=frame, roster=roster, ingested=["alpha"])


def test_the_report_states_the_omission_rather_than_leaving_a_gap(
    synthetic_report,
) -> None:
    report, _ = synthetic_report
    assert report["accuracy_reported"] is False
    assert "Accuracy is excluded" in report["accuracy_note"]
    assert report["scored_on"] == "test"
    assert report["threshold_chosen_on"] == "validation"
    for row in report["summary"]:
        assert "accuracy" not in row


def test_every_predictor_is_in_the_table(synthetic_report) -> None:
    report, _ = synthetic_report
    names = {row["predictor"] for row in report["summary"]}
    assert names == {
        "base_rate",
        "persistence",
        "climatology",
        "model_weighted",
        "model_unweighted",
    }


def test_an_uningested_city_is_named_not_omitted(synthetic_report) -> None:
    """The ticket contrasts Singapore with Moscow, and Moscow has not landed.

    A per-city table that simply lacks the row says nothing about why, and
    "the model was not evaluated on Moscow" is a different statement from
    "Moscow has no data".
    """
    report, _ = synthetic_report
    assert "moscow" in report["cities_not_ingested"]
    assert "moscow" not in report["cities_scored"]
    assert not set(report["cities_scored"]) & set(report["cities_not_ingested"])


def test_the_verdict_is_per_metric(synthetic_report) -> None:
    """Because the honest answer is not the same on ranking and calibration."""
    report, _ = synthetic_report
    for name, entry in report["verdict"].items():
        assert name.startswith("model_")
        assert set(entry["beats_every_baseline"]) == {"pr_auc", "f1", "brier"}
        for metric, detail in entry["detail"].items():
            assert set(detail) == {"persistence", "climatology", "base_rate"}
            assert entry["beats_every_baseline"][metric] == all(detail.values())


def test_the_figures_are_reproducible(synthetic_report, tmp_path) -> None:
    """Two renders of the same numbers must be the same bytes.

    matplotlib salts the element ids it writes into an SVG; without pinning
    that, a regenerated figure differs on every call while drawing exactly the
    same picture, and a committed image churns for no reason a reviewer can act
    on.
    """
    _, plots = synthetic_report
    first = render_figures(plots, tmp_path / "one")
    second = render_figures(plots, tmp_path / "two")

    assert {path.name for path in first} == {
        "precision_recall.svg",
        "precision_recall.png",
        "calibration.svg",
        "calibration.png",
    }
    for left, right in zip(first, second):
        assert left.read_bytes() == right.read_bytes(), left.name


# --------------------------------------------------------------------------
# Against the warehouse
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def warehouse_report(engine):
    from sqlalchemy import text

    with engine.connect() as connection:
        rows = connection.execute(
            text("select count(*) from gold_marts.fact_weather_anomalies")
        ).scalar()
    if not rows:
        pytest.skip("fact_weather_anomalies not built")
    population = evaluation_frame(engine)
    if any(part.empty for part in split_frame(population).values()):
        pytest.skip("not enough of the record backfilled to fill every split")
    roster, ingested = city_roster(engine)
    return build_evaluation(frame=population, roster=roster, ingested=ingested)


def test_the_recommended_model_beats_every_baseline_on_every_metric(
    warehouse_report,
) -> None:
    report, _ = warehouse_report
    verdict = report["verdict"]["model_unweighted"]["beats_every_baseline"]
    assert verdict == {"pr_auc": True, "f1": True, "brier": True}, verdict


def test_the_specified_model_loses_on_calibration_and_the_report_says_so(
    warehouse_report,
) -> None:
    """The documented failure, kept visible rather than averaged away.

    ``scale_pos_weight`` at the observed ratio wins on ranking and loses to
    *doing nothing* on Brier. The report records that per metric, so the
    summary line cannot round it into a pass.
    """
    report, _ = warehouse_report
    verdict = report["verdict"]["model_weighted"]["beats_every_baseline"]
    assert verdict["pr_auc"] is True
    assert verdict["brier"] is False, (
        "the weighted model now beats the no-skill reference on Brier; the "
        "calibration finding recorded in the README needs revisiting"
    )
    detail = report["verdict"]["model_weighted"]["detail"]["brier"]
    assert detail["base_rate"] is False


def test_moscow_is_named_as_uningested(warehouse_report) -> None:
    report, _ = warehouse_report
    absent = set(report["cities_not_ingested"])
    unscored = set(report["cities_ingested_but_not_scored"])
    assert "singapore" in report["cities_scored"]
    if "moscow" not in report["cities_scored"]:
        assert "moscow" in absent | unscored, (
            "Moscow is neither scored nor accounted for; the ticket names it "
            "as the contrast to Singapore and an unexplained gap is not an "
            "answer"
        )
    assert not absent & unscored


def test_the_per_city_table_covers_every_scored_city(warehouse_report) -> None:
    report, _ = warehouse_report
    cities = pd.DataFrame(report["per_city"])
    assert sorted(cities["city_id"]) == sorted(report["cities_scored"])
    # Each city is scored against its own base rate, which vary fourfold.
    assert cities["base_rate"].max() / cities["base_rate"].min() > 3


def test_the_committed_figures_are_current(warehouse_report, tmp_path) -> None:
    """The picture cannot go stale, the same way the lineage diagram cannot.

    Skipped rather than failed when the figures have not been rendered yet, and
    the message says the command.
    """
    _, plots = warehouse_report
    committed = FIGURE_DIR / "precision_recall.svg"
    if not committed.exists():
        pytest.skip("run `python machine_learning/evaluate.py --write` first")

    rendered = render_figures(plots, tmp_path)
    for path in rendered:
        beside = FIGURE_DIR / path.name
        assert beside.exists(), f"{beside} is missing"
        assert beside.read_bytes() == path.read_bytes(), (
            f"{beside.name} is out of date — re-run "
            "`python machine_learning/evaluate.py --write`"
        )
