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

**The picture.** A precision-recall curve drawn as straight lines between its
points shows operating points that do not exist. Persistence has two of them,
and joining them draws a diagonal that integrates to roughly twice the average
precision the same predictor scores. The curve is drawn as a step function, and
a test integrates the drawn points and requires the result to equal the
reported PR-AUC. The figures are regenerated and compared byte for byte, so
they cannot go stale.

**The city fold.** ML-08's whole claim rests on the held-out city being absent
from the fold that trained the model, and nothing else in the suite would
notice if it were not: a fold that kept the city still splits chronologically,
still purges its boundaries, and still posts a plausible PR-AUC. So the
partition is asserted directly, on every city, and the guard that asserts it is
itself checked against a planted row. The README's verdict sentence is
generated from the numbers and compared against the committed prose, so a
re-run that flips the finding fails here until the prose is changed.
"""

from __future__ import annotations

import json
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

from machine_learning.baselines import metrics_path  # noqa: E402
from machine_learning.conformal import (  # noqa: E402
    Calibration,
    adaptive_report,
    calibrate,
    prediction_sets,
)
from machine_learning.evaluate import (  # noqa: E402
    CALIBRATION_BINS,
    FIGURE_DIR,
    _thin,
    best_threshold,
    build_evaluation,
    build_leave_one_city_out,
    calibration_points,
    city_roster,
    classification_at,
    hold_out_reasons,
    leave_one_city_out_table,
    precision_recall_points,
    render_figures,
)
from machine_learning.evaluation import (  # noqa: E402
    MIN_HELD_OUT_POSITIVES,
    MIN_HELD_OUT_ROWS,
    assert_city_is_held_out,
    evaluation_frame,
    hold_out_city,
    scorable_cities,
    split_frame,
)
from machine_learning.labels import LABEL  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The metric this project will not report, and the sklearn names for it.
FORBIDDEN_METRICS = ("accuracy_score", "balanced_accuracy")


def series(values) -> pd.Series:
    return pd.Series(pd.array(values, dtype="boolean"))


def flattened(text: str) -> str:
    """Text with its line breaks forgotten.

    The README wraps at seventy-nine columns and the generated verdict is one
    long sentence, so a substring check against the file as written would fail
    on the wrap alone and say nothing about whether the finding is stated.
    """
    return " ".join(text.split())


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
        "accuracy reached the codebase: at a 13.6% base rate, always "
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
    # 3 of 4 correct, recall 1.0, precision 0.75, F1 6/7.
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
    the table would find them disagreeing, most severely for a predictor with
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
# Leave one city out
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def three_cities():
    """Three synthetic cities through the whole pipeline.

    Three rather than two, because :data:`MIN_TRAINING_CITIES` refuses a fold
    with one city left to learn from, and a fixture that could not exercise a
    passing fold would only ever test the refusal.
    """
    from ml_fixtures import scored_population, spanning

    gold = pd.concat(
        [
            spanning(city_id=name, seed=seed)
            for seed, name in enumerate(("alpha", "beta", "gamma"), start=4)
        ],
        ignore_index=True,
    )
    return scored_population(gold)


def test_the_held_out_city_never_appears_in_its_own_training_fold(
    three_cities,
) -> None:
    """The assertion the whole experiment rests on, made on every city.

    A leave-one-city-out score means one thing and one thing only: the model
    had never seen this city. Every other check in the suite is blind to the
    failure. A fold that quietly kept Cairo in its validation set splits
    chronologically, purges its boundaries, posts a believable PR-AUC, and
    answers a question nobody asked.
    """
    for city in sorted(three_cities["city_id"].unique()):
        others, held = hold_out_city(three_cities, city)
        fold = split_frame(others)

        for name, part in fold.items():
            assert (part["city_id"] == city).sum() == 0, (
                f"{city} reached the {name} fold of its own hold-out"
            )
        assert set(held["city_id"]) == {city}
        # And through the guard the pipeline itself calls, on the same frames.
        assert_city_is_held_out(
            city,
            {"train": fold["train"], "validation": fold["validation"]},
            held_out=split_frame(held)["test"],
        )


def test_holding_a_city_out_partitions_rather_than_filters(three_cities) -> None:
    """Nothing is lost between the two sides, so nothing can go missing quietly."""
    others, held = hold_out_city(three_cities, "beta")
    assert len(others) + len(held) == len(three_cities)
    assert set(others["city_id"]) | set(held["city_id"]) == set(
        three_cities["city_id"]
    )
    assert not set(others["city_id"]) & set(held["city_id"])


def test_the_hold_out_guard_would_notice_a_leaked_row(three_cities) -> None:
    """The companion to the assertion above: prove the guard can fail.

    An assertion that has never been seen to fail is a comment.
    """
    others, held = hold_out_city(three_cities, "beta")
    leaked = pd.concat([others, held.head(3)], ignore_index=True)

    with pytest.raises(ValueError, match="3 beta row"):
        assert_city_is_held_out("beta", {"train": leaked})

    with pytest.raises(ValueError, match="scored on"):
        assert_city_is_held_out(
            "beta", {"train": others}, held_out=three_cities
        )


def test_holding_out_a_city_that_is_not_there_is_an_error(three_cities) -> None:
    """Otherwise the fold trains on everything and scores nothing, silently."""
    with pytest.raises(ValueError, match="not in this population"):
        hold_out_city(three_cities, "atlantis")


def test_a_city_with_too_few_test_rows_is_named_rather_than_dropped(
    three_cities,
) -> None:
    """"Not eligible" and "eligible and unimpressive" are opposite findings.

    A table that simply lacks the row cannot tell them apart, which is the same
    argument the report already makes about an uningested city one table up.
    """
    parts = split_frame(three_cities)
    trimmed = dict(parts)
    keep = parts["test"]["city_id"] != "gamma"
    thin = parts["test"].loc[~keep].head(MIN_HELD_OUT_ROWS - 1)
    trimmed["test"] = pd.concat(
        [parts["test"].loc[keep], thin], ignore_index=True
    )

    eligible, excluded = scorable_cities(trimmed)
    assert eligible == ["alpha", "beta"]
    assert "gamma" in excluded
    assert str(MIN_HELD_OUT_ROWS) in excluded["gamma"]

    _, reasons = hold_out_reasons(
        three_cities, trimmed, roster=["alpha", "beta", "gamma", "moscow"],
        ingested=["alpha", "beta", "gamma"],
    )
    assert reasons["moscow"] == "not ingested"
    assert "gamma" in reasons


def test_the_eligibility_floor_needs_both_rows_and_positives() -> None:
    """Rows alone would admit a city with four positives in a year of test."""
    assert MIN_HELD_OUT_ROWS > 0 and MIN_HELD_OUT_POSITIVES > 0
    frame = pd.DataFrame(
        {
            "city_id": ["delta"] * MIN_HELD_OUT_ROWS,
            LABEL: pd.array(
                [True] * (MIN_HELD_OUT_POSITIVES - 1)
                + [False] * (MIN_HELD_OUT_ROWS - MIN_HELD_OUT_POSITIVES + 1),
                dtype="boolean",
            ),
        }
    )
    eligible, excluded = scorable_cities({"test": frame})
    assert eligible == []
    assert "positive test rows" in excluded["delta"]


@pytest.fixture(scope="module")
def synthetic_hold_out(three_cities):
    return build_leave_one_city_out(
        three_cities,
        roster=["alpha", "beta", "gamma", "moscow"],
        ingested=["alpha", "beta", "gamma"],
    )


def test_every_fold_records_the_cities_it_trained_on(synthetic_hold_out) -> None:
    """The claim in the file, not only in the code that wrote it.

    ``metrics.json`` is what a reader sees, so the fold's own record has to
    carry the evidence rather than leaving it in an assertion that ran once.
    """
    assert synthetic_hold_out["cities_held_out"] == ["alpha", "beta", "gamma"]
    for record in synthetic_hold_out["folds"]:
        assert record["city_id"] not in record["train_cities"]
        assert len(record["train_cities"]) >= 2
        assert record["rows"] >= MIN_HELD_OUT_ROWS
        assert record["positives"] >= MIN_HELD_OUT_POSITIVES


def test_the_yardstick_is_persistence_and_the_base_rate_is_beside_it(
    synthetic_hold_out,
) -> None:
    """The trap the ticket names, asserted as arithmetic.

    Held-out cities differ in base rate, so a raw PR-AUC is not comparable
    across them. Every fold therefore carries the lift over that city's own
    persistence baseline *and* the base rate the PR-AUC was measured against.
    """
    variant = synthetic_hold_out["verdict"]["variant"]
    assert "persistence" in synthetic_hold_out["yardstick"]

    for record in synthetic_hold_out["folds"]:
        entry = record["variants"][variant]
        own = record["persistence_own"]
        assert own["fitted_on"] == "this city's own training split"
        assert record["persistence"]["fitted_on"] == (
            "the other cities' training split"
        )
        assert 0.0 < record["base_rate"] < 1.0
        assert record["base_rate"] == pytest.approx(entry["held_out"]["base_rate"])
        assert entry["lift_over_persistence"] == pytest.approx(
            entry["held_out"]["pr_auc"] / own["pr_auc"]
        )
        # Brier is a loss; the ratio is inverted so every figure in the record
        # reads the same way round.
        assert entry["brier_ratio_to_persistence"] == pytest.approx(
            own["brier"] / entry["held_out"]["brier"]
        )
        assert entry["retained_of_in_sample"] == pytest.approx(
            entry["held_out"]["pr_auc"] / entry["in_sample"]["pr_auc"]
        )


def test_the_verdict_is_strict_and_says_which_way_it_fell(
    synthetic_hold_out,
) -> None:
    """Every city, not a majority, and a sentence either way.

    A model that transfers to two cities and fails on the third has not
    answered "can it score a city it has never seen"; it has raised a question
    about the third.
    """
    verdict = synthetic_hold_out["verdict"]
    beaten = verdict["beats_own_persistence"]
    assert set(beaten) == set(synthetic_hold_out["cities_held_out"])
    assert verdict["cities_beating_own_persistence"] == sum(beaten.values())
    assert verdict["transfers"] is all(beaten.values())
    assert verdict["statement"].endswith(".")
    if verdict["transfers"]:
        assert "transfers to a city it has never seen" in verdict["statement"]
    else:
        assert "does not transfer" in verdict["statement"]


def test_the_table_carries_the_base_rate_beside_every_score(
    synthetic_hold_out,
) -> None:
    table = leave_one_city_out_table(synthetic_hold_out)
    assert list(table["city_id"]) == synthetic_hold_out["cities_held_out"]
    for column in ("base_rate", "held_out_pr_auc", "persistence_pr_auc",
                   "lift_over_persistence", "in_sample_pr_auc"):
        assert column in table.columns
        assert table[column].notna().all()


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


def test_the_recommended_model_beats_every_baseline_on_ranking_and_calibration(
    warehouse_report,
) -> None:
    """Two of the three, and the third is the one ML-09 was built to expose.

    PR-AUC and Brier are properties of the model, and ML-09's sweep says they
    hold at |Z| 2.0, 2.5 and 3.0 alike. F1 is a property of the model *and* a
    threshold, and it does not: the recommended model is behind persistence at
    2.5 and ahead at 2.0 and 3.0.

    This test has now asserted three different F1 verdicts across three
    snapshots -- a win, a tie to a thousandth, and a loss -- while PR-AUC and
    Brier never moved. That is not instability in the model, it is a
    single-point metric measuring the point, and it is why the claim in the
    README is qualified and why the sweep exists at all. So the two durable
    legs are asserted as booleans and the third as a margin, which is the shape
    of what is actually known.
    """
    report, _ = warehouse_report
    verdict = report["verdict"]["model_unweighted"]["beats_every_baseline"]
    assert verdict["pr_auc"] is True
    assert verdict["brier"] is True

    summary = pd.DataFrame(report["summary"]).set_index("predictor")
    model, persistence = summary.loc["model_unweighted"], summary.loc["persistence"]
    assert model["pr_auc"] > 1.4 * persistence["pr_auc"]
    assert model["lift_over_persistence"] == pytest.approx(
        model["pr_auc"] / persistence["pr_auc"]
    )
    assert persistence["lift_over_persistence"] == pytest.approx(1.0)
    assert abs(model["f1"] - persistence["f1"]) < 0.05, (
        "the model and persistence have separated on F1 by more than the "
        "threshold sweep can account for; the README's qualification needs "
        "revisiting"
    )


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
        "calibration finding recorded in the build log needs revisiting"
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
            f"{beside.name} is out of date; re-run "
            "`python machine_learning/evaluate.py --write`"
        )


def test_the_readme_states_the_leave_one_city_out_verdict() -> None:
    """The sentence in the README is the sentence the numbers produced.

    Generated rather than typed, and compared rather than trusted. A verdict
    written by hand outlives the run that justified it, and this ticket is the
    one whose answer is most likely to change: it is the cheapest experiment in
    the roadmap and the one every later ticket is gated on.
    """
    path = metrics_path()
    if not path.exists():
        pytest.skip("no metrics.json; run baselines.py --write first")
    block = json.loads(path.read_text()).get("leave_one_city_out")
    if block is None:
        pytest.skip(
            "run `python machine_learning/evaluate.py --leave-one-city-out "
            "--write` first"
        )

    statement = block["verdict"]["statement"]
    readme = flattened((REPO_ROOT / "README.md").read_text())
    assert flattened(statement) in readme, (
        "the README's leave-one-city-out verdict no longer matches "
        f"metrics.json, which now says: {statement}"
    )


def test_the_recorded_hold_out_never_trained_on_the_city_it_scored() -> None:
    """The committed file has to carry the evidence, not just the conclusion."""
    path = metrics_path()
    if not path.exists():
        pytest.skip("no metrics.json; run baselines.py --write first")
    block = json.loads(path.read_text()).get("leave_one_city_out")
    if block is None:
        pytest.skip("no leave-one-city-out block recorded")

    for record in block["folds"]:
        assert record["city_id"] not in record["train_cities"], record["city_id"]
        assert len(record["train_cities"]) >= block["min_training_cities"]
        assert record["rows"] >= block["min_held_out_rows"]
        assert record["positives"] >= block["min_held_out_positives"]
    named = set(block["cities_held_out"]) | set(block["cities_not_held_out"])
    assert not set(block["cities_held_out"]) & set(block["cities_not_held_out"])
    assert named, "the block accounts for no city at all"


# --------------------------------------------------------------------------
# Adaptive conformal prediction (ML-14)
# --------------------------------------------------------------------------


def test_the_calibration_never_intersects_the_test_period(three_cities) -> None:
    """The acceptance's third bullet, behaviourally rather than structurally.

    ``calibrate`` takes one frame, so there is no test split to hand it. That
    is the same argument ``tune`` and ``fit_calibrator`` rest on, and here the
    failure it prevents is the most attractive in the project: a conformal
    predictor calibrated on the period it is then evaluated on reports coverage
    near the target by construction, and the number looks exactly like a
    guarantee that held.

    So the test period is rewritten and the calibration required to be
    identical, quantile for quantile.
    """
    parts = split_frame(three_cities)
    from machine_learning.train import tune

    fit = tune(parts["train"], parts["validation"], scale_pos_weight=1.0)
    baseline = calibrate(
        parts["validation"][LABEL], fit.predict(parts["validation"])
    )

    wrecked = three_cities.copy()
    later = wrecked["date_key"] >= pd.Timestamp("2022-01-01")
    wrecked.loc[later, LABEL] = ~wrecked.loc[later, LABEL]
    rebuilt_parts = split_frame(wrecked)
    rebuilt = calibrate(
        rebuilt_parts["validation"][LABEL], fit.predict(rebuilt_parts["validation"])
    )

    np.testing.assert_array_equal(baseline.scores, rebuilt.scores)
    for alpha in (0.05, 0.10, 0.25):
        assert baseline.quantile(alpha) == rebuilt.quantile(alpha)


def test_the_conformal_quantile_carries_the_finite_sample_correction() -> None:
    """``ceil((n + 1)(1 - alpha)) / n``, not the plain empirical quantile.

    The correction is the difference between a guarantee that holds at the
    sample size you have and one that holds asymptotically. It also decides
    what happens when the level is tighter than the sample can certify, and
    infinity is the honest answer there: every set becomes both labels, which
    is a useless prediction and an accurate one.
    """
    scores = np.linspace(0.0, 1.0, 100)
    calibration = Calibration(scores=scores)

    # rank = ceil(101 * 0.9) = 91, so the 91st smallest of 100.
    assert calibration.quantile(0.10) == pytest.approx(scores[90])
    # Tighter than 100 rows can certify: ceil(101 * 0.999) = 101 > 100.
    assert calibration.quantile(0.001) == float("inf")
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="strictly between"):
            calibration.quantile(bad)


def test_a_prediction_set_can_hold_both_labels_or_neither() -> None:
    """Four states, and the two unusual ones are the point of the method.

    A probability rounds to a decision. A set says which decisions the evidence
    supports: both labels when the model cannot separate them, and neither when
    the week is unlike anything in calibration. Collapsing those to a number
    is what conformal prediction exists to avoid.
    """
    sets = prediction_sets([0.02, 0.5, 0.98], threshold=0.2)
    assert list(sets["quiet"]) == [True, False, False]
    assert list(sets["extreme"]) == [False, False, True]
    assert sets.sum(axis=1).tolist() == [1, 0, 1]

    # A loose threshold admits both; the set is honest and uninformative.
    wide = prediction_sets([0.5], threshold=0.6)
    assert bool(wide["quiet"].iloc[0]) and bool(wide["extreme"].iloc[0])


def test_the_adaptive_level_moves_only_after_the_outcome(three_cities) -> None:
    """Online means online: row t's set is formed before row t is revealed.

    If the level were updated before scoring, the procedure would be reading
    each outcome to predict it and the coverage would be meaningless. The check
    is that the first row is scored at exactly the starting level.
    """
    parts = split_frame(three_cities)
    from machine_learning.train import tune

    fit = tune(parts["train"], parts["validation"], scale_pos_weight=1.0)
    calibration = calibrate(
        parts["validation"][LABEL], fit.predict(parts["validation"])
    )
    report = adaptive_report(
        calibration, parts["test"], fit.predict(parts["test"]), target=0.9
    )

    assert report["alpha_start"] == pytest.approx(0.1)
    assert report["alpha_range"][0] <= report["alpha_start"]
    assert report["alpha_range"][1] >= report["alpha_start"]
    assert report["calibrated_on"] == "validation"
    assert "not a held-out score" in report["note"]


def test_coverage_is_reported_with_the_set_size_that_bought_it(
    three_cities,
) -> None:
    """Coverage alone is trivially satisfiable by never committing.

    A method that always returns both labels covers everything and says
    nothing, so every coverage figure in the block is accompanied by the mean
    set size and the share of rows where the set held both labels.
    """
    parts = split_frame(three_cities)
    from machine_learning.train import tune

    fit = tune(parts["train"], parts["validation"], scale_pos_weight=1.0)
    calibration = calibrate(
        parts["validation"][LABEL], fit.predict(parts["validation"])
    )
    report = adaptive_report(calibration, parts["test"], fit.predict(parts["test"]))

    for arm in ("split", "adaptive"):
        entry = report[arm]
        assert 0.0 <= entry["coverage"] <= 1.0
        assert entry["shortfall"] == pytest.approx(
            entry["coverage"] - report["target_coverage"]
        )
        assert 0.0 <= entry["mean_set_size"] <= 2.0
        assert entry["by_city"] and entry["by_year"]
        for row in entry["by_city"] + entry["by_year"]:
            assert {"rows", "coverage", "mean_set_size", "shortfall"} <= set(row)
        assert 0.0 <= entry["share_uninformative"] <= 1.0


def test_the_recorded_conformal_block_reports_both_arms() -> None:
    """And the committed sentence is the one the numbers produced."""
    path = metrics_path()
    if not path.exists():
        pytest.skip("no metrics.json; run baselines.py --write first")
    block = json.loads(path.read_text()).get("conformal")
    if block is None:
        pytest.skip("run `evaluate.py --conformal --write` first")

    assert block["calibrated_on"] == "validation"
    assert block["scored_on"] == "test"
    assert 0.0 < block["target_coverage"] < 1.0
    for arm in ("split", "adaptive"):
        assert 0.0 <= block[arm]["coverage"] <= 1.0
    assert block["adaptive_is_closer_to_target"] is (
        abs(block["adaptive"]["coverage"] - block["target_coverage"])
        <= abs(block["split"]["coverage"] - block["target_coverage"])
    )
    assert block["statement"].endswith(".")
