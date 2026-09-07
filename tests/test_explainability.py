"""Tests for the SHAP explanations.

An explanation is only worth reading if it is an explanation of the model that
actually ran. The identity that guarantees that is exact, so it is asserted
rather than cited: pushed through the logistic, the contributions plus the base
value must reproduce ``predict_proba`` to floating-point.

That is not a formality. The first draft of ``explain.py`` used XGBoost's
default tree range, which is *every* tree, while the classifier stops early on
validation and scores with the first eighteen. The contributions summed
perfectly — to the wrong model's margin — and explained a Delhi day at 0.64
that the model had scored at 0.695. Anchoring the identity to ``predict_proba``
rather than to the booster's own margin is what catches that.

The rest are plausibility checks, and the strongest one is the both-tails
response. The label is ``abs(z) > 2.5``; nothing told the model that, it saw a
binary column. If it has understood its target it will push risk up at *both*
ends of the Z distribution and be quietest in the middle.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("xgboost")
pytest.importorskip("matplotlib")

from ml_fixtures import labelled_span  # noqa: E402

from machine_learning.evaluate import FIGURE_DIR, best_threshold  # noqa: E402
from machine_learning.evaluation import evaluation_frame, split_frame  # noqa: E402
from machine_learning.explain import (  # noqa: E402
    STATIC_CITY_FEATURES,
    TOP_N,
    build_explanation,
    contributions,
    explain_case,
    global_importance,
    pick_cases,
    render_figures,
)
from machine_learning.labels import LABEL, positives  # noqa: E402
from machine_learning.train import train_model  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def synthetic():
    frame = labelled_span()
    parts = split_frame(frame)
    block, fits = train_model(frame=frame)
    fit = fits[block["recommended_variant"]]
    return frame, parts, fit


# --------------------------------------------------------------------------
# The identity that makes it an explanation of *this* model
# --------------------------------------------------------------------------


def test_the_contributions_reproduce_the_scored_probability(synthetic) -> None:
    """Contributions + base value, through the logistic, is ``predict_proba``.

    Anchored to the classifier's own output rather than to the booster's
    margin. The booster will happily report a margin for every tree it holds,
    including the ones early stopping discarded, and contributions that sum to
    *that* are self-consistent and describe a model nobody runs.
    """
    _, parts, fit = synthetic
    test = parts["test"]
    shap = contributions(fit, test)

    np.testing.assert_allclose(shap.probabilities(), fit.predict(test), atol=1e-6)
    assert shap.values.shape == (len(test), len(shap.features))
    assert shap.bias.shape == (len(test),)


def test_using_every_tree_would_break_the_identity(synthetic) -> None:
    """The bug the identity catches, reproduced deliberately.

    Explaining with the full forest instead of the scored range still produces
    contributions that sum to *a* margin. It is just not this model's.
    """
    from xgboost import DMatrix

    _, parts, fit = synthetic
    test = parts["test"]
    matrix = test.loc[:, list(contributions(fit, test).features)]
    every_tree = fit.estimator.get_booster().predict(
        DMatrix(matrix), pred_contribs=True
    )
    probabilities = 1.0 / (1.0 + np.exp(-every_tree.sum(axis=1)))

    if fit.best_iteration + 1 >= fit.estimator.n_estimators:
        pytest.skip("early stopping did not discard any trees on this fit")
    assert not np.allclose(probabilities, fit.predict(test), atol=1e-6)


def test_the_contributions_match_the_reference_implementation(synthetic) -> None:
    """The same numbers the ``shap`` package produces, so the shortcut is one.

    ``shap`` is a development dependency only: it pulls a compiler toolchain
    into an application that deploys to Streamlit Community Cloud, and the
    values it would contribute are already inside XGBoost.
    """
    shap_package = pytest.importorskip("shap")

    _, parts, fit = synthetic
    sample = parts["test"].head(400)
    mine = contributions(fit, sample)
    matrix = sample.loc[:, list(mine.features)]

    explainer = shap_package.TreeExplainer(fit.estimator)
    reference = explainer.shap_values(matrix)
    np.testing.assert_allclose(mine.values, reference, atol=1e-6)


def test_the_base_value_is_kept_out_of_the_ranking(synthetic) -> None:
    """Otherwise "being a city-day at all" ranks first."""
    _, parts, fit = synthetic
    shap = contributions(fit, parts["test"])
    ranked = global_importance(shap)
    assert len(ranked) == len(shap.features)
    assert "bias" not in set(ranked["feature"])
    assert "base_value" not in set(ranked["feature"])
    assert np.allclose(shap.bias, shap.bias[0])


# --------------------------------------------------------------------------
# The ranking
# --------------------------------------------------------------------------


def test_the_ranking_is_mean_absolute_contribution(synthetic) -> None:
    _, parts, fit = synthetic
    shap = contributions(fit, parts["test"])
    ranked = global_importance(shap).set_index("feature")

    for feature in list(ranked.index)[:5]:
        expected = float(np.abs(shap.column(feature)).mean())
        assert ranked.loc[feature, "mean_abs"] == pytest.approx(expected)
        assert ranked.loc[feature, "mean_signed"] == pytest.approx(
            float(shap.column(feature).mean())
        )
    assert ranked["mean_abs"].is_monotonic_decreasing


def test_the_ranking_names_the_static_city_features(synthetic) -> None:
    """A model leaning on these is looking up which city it is."""
    _, parts, fit = synthetic
    ranked = global_importance(contributions(fit, parts["test"]))
    flagged = set(ranked.loc[ranked["static_per_city"], "feature"])
    assert flagged == set(STATIC_CITY_FEATURES)


def test_a_case_explanation_is_a_subset_of_the_full_decomposition(
    synthetic,
) -> None:
    _, parts, fit = synthetic
    shap = contributions(fit, parts["test"])
    detail = explain_case(shap, 3, top=6)

    assert len(detail) == 6
    assert detail["shap"].abs().is_monotonic_decreasing
    assert set(detail["feature"]) <= set(shap.features)
    # The top six are a slice of a decomposition that is complete.
    assert shap.values[3].sum() + shap.bias[3] == pytest.approx(shap.margins()[3])


def test_case_selection_finds_a_real_hit_and_a_real_miss(synthetic) -> None:
    _, parts, fit = synthetic
    test = parts["test"]
    predictions = fit.predict(test)
    threshold, _ = best_threshold(
        parts["validation"][LABEL], fit.predict(parts["validation"])
    )
    picks = pick_cases(test, predictions, threshold)
    truth = positives(test[LABEL]).to_numpy()

    if "true_positive" in picks:
        position = picks["true_positive"]
        assert truth[position]
        assert predictions[position] >= threshold
    if "false_positive" in picks:
        position = picks["false_positive"]
        assert not truth[position]
        assert predictions[position] >= threshold
    assert picks, "no row cleared the threshold at all"


# --------------------------------------------------------------------------
# Against the warehouse: is the ranking meteorologically plausible?
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def explained(engine):
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
    return build_explanation(frame=population)


def test_the_model_raises_risk_on_both_tails(explained) -> None:
    """The strongest plausibility check available, and it is not a soft one.

    The target is ``abs(z) > 2.5``. The model was handed a binary column and
    never told what produced it. If it has learned the shape of its own label
    it will push risk up at both extremes of Z and be quietest near zero — and
    a model that had only memorised the test period, which runs 231 hot
    anomalies to 61 cold, would show the hot tail and not the cold one.
    """
    report, _ = explained
    bands = pd.DataFrame(report["tail_response"]).set_index("band")
    cold = bands.loc["(-inf, -2.5]", "mean_shap"]
    hot = bands.loc["(2.5, inf]", "mean_shap"]
    middle = bands.loc["(-0.5, 0.5]", "mean_shap"]

    assert middle < 0, "the quietest band should push risk down"
    assert cold > 1.0, f"the cold tail should raise risk; it contributes {cold:.3f}"
    assert hot > 1.0, f"the hot tail should raise risk; it contributes {hot:.3f}"
    assert cold > middle + 1.5 and hot > middle + 1.5
    # Near-symmetric: neither tail is more than twice the other.
    assert 0.5 < cold / hot < 2.0, (
        f"the tails are lopsided — cold {cold:.3f} against hot {hot:.3f}. The "
        "model may be fitting the test period's hot-dominated mix rather than "
        "the label's definition."
    )


def test_the_top_feature_is_weather_and_not_a_city_lookup(explained) -> None:
    report, _ = explained
    ranking = pd.DataFrame(report["top_features"])
    assert ranking.iloc[0]["feature"] not in STATIC_CITY_FEATURES, (
        "the model's strongest signal is a per-city constant, which means it "
        "is mostly predicting base rates rather than weather"
    )
    assert len(ranking) == TOP_N


def test_the_city_lookup_is_bounded_and_reported(explained) -> None:
    """Latitude and elevation do carry weight, and the report says how much."""
    report, _ = explained
    share = report["static_city_share"]
    assert 0.0 < share < 0.25, (
        f"per-city constants are {share:.1%} of the total contribution. Above "
        "a quarter the model is a base-rate table with weather attached, and "
        "nothing it learned would transfer to a city it has not seen."
    )


def test_the_seasonal_anomaly_outranks_the_raw_temperature(explained) -> None:
    """A seasonal target needs a seasonally adjusted input.

    The label is "unusual for this day of year". Raw daily temperature is
    mostly a statement about the season, so a model ranking it above the
    Z-score would be reaching the right answer through the wrong quantity.
    """
    report, _ = explained
    ranked = {row["feature"]: index for index, row in enumerate(report["top_features"])}
    assert "z_temperature_2m_mean" in ranked
    if "temperature_2m_mean" in ranked:
        assert ranked["z_temperature_2m_mean"] < ranked["temperature_2m_mean"]


def test_the_persistence_signal_is_near_the_top(explained) -> None:
    """It is the baseline's whole content, so the model should be using it."""
    report, _ = explained
    order = [row["feature"] for row in report["top_features"]]
    assert "anomaly_days_trailing30" in order[:3]
    persistence = "anomaly_days_trailing30"
    entry = next(
        row for row in report["top_features"] if row["feature"] == persistence
    )
    assert entry["correlation"] > 0.5, (
        "more anomalous days behind should mean more risk ahead; a negative "
        "or flat relationship would invert the persistence baseline"
    )


def test_a_confident_mistake_is_a_spell_that_broke(explained) -> None:
    """What the false positive is *for*.

    The model's most confident error is not a hallucination: it is a city
    eight days into a hot spell, with the same evidence as the most confident
    correct call. The lesson is a limit of the target — at this horizon the
    model can say a spell is running, not when it will end — and it is only
    visible because the case was chosen for confidence rather than for
    marginality.
    """
    report, _ = explained
    cases = report["cases"]
    assert {"true_positive", "false_positive"} <= set(cases)

    hit, miss = cases["true_positive"], cases["false_positive"]
    assert hit["label"] is True and miss["label"] is False
    for case in (hit, miss):
        leading = case["contributions"][0]
        assert leading["feature"] == "z_temperature_2m_mean"
        assert leading["shap"] > 1.0
        assert leading["value"] > 2.0, (
            "the confident cases should be days that are already extreme"
        )
    # Same evidence, opposite outcomes — that is the point of showing both.
    assert abs(hit["predicted"] - miss["predicted"]) < 0.1


def test_the_recorded_ranking_matches_a_fresh_run(explained) -> None:
    report, _ = explained
    path = REPO_ROOT / "machine_learning/artifacts/metrics.json"
    committed = json.loads(path.read_text())
    if "explainability" not in committed:
        pytest.skip("no explanation recorded yet; run explain.py --write")

    recorded = committed["explainability"]
    if recorded["rows_explained"] != report["rows_explained"]:
        pytest.skip(
            "metrics.json was written against a different snapshot; re-run "
            "the pipeline and commit before comparing."
        )
    assert [row["feature"] for row in recorded["top_features"]] == [
        row["feature"] for row in report["top_features"]
    ]
    for left, right in zip(recorded["top_features"], report["top_features"]):
        assert left["mean_abs"] == pytest.approx(right["mean_abs"], rel=1e-9)


def test_the_committed_figures_are_current(explained, tmp_path) -> None:
    _, plots = explained
    if not (FIGURE_DIR / "shap_summary.svg").exists():
        pytest.skip("run `python machine_learning/explain.py --write` first")

    for path in render_figures(plots, tmp_path):
        beside = FIGURE_DIR / path.name
        assert beside.exists(), f"{beside} is missing"
        assert beside.read_bytes() == path.read_bytes(), (
            f"{beside.name} is out of date — re-run "
            "`python machine_learning/explain.py --write`"
        )


def test_the_figures_are_reproducible(explained, tmp_path) -> None:
    """The beeswarm jitters its points; the jitter is seeded."""
    _, plots = explained
    first = render_figures(plots, tmp_path / "one")
    second = render_figures(plots, tmp_path / "two")
    for left, right in zip(first, second):
        assert left.read_bytes() == right.read_bytes(), left.name


def test_the_report_says_which_space_the_numbers_are_in(explained) -> None:
    """A contribution of +2.29 is log-odds, not 229% of anything."""
    report, _ = explained
    assert report["space"] == "log-odds"
    assert "log-odds" in report["note"]
    assert report["explained_variant"] == "unweighted"
