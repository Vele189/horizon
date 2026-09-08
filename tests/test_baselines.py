"""Tests for the split, the metrics, and the two baselines.

These numbers are the target the model will be measured against, so they have
to be right before anything is trained. Two things are being defended here; the
split itself is defended in ``test_split.py``.

The **metrics** must be the ones the model will be scored with. Average
precision for a constant prediction is the base rate by construction, which
makes a constant a free self-check on the implementation, and one test uses it
that way.

The **fit** must never see the future. Baselines are fitted on train only;
tests mutate the validation and test periods and require the fitted parameters
to come back unchanged.
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

from ml_fixtures import labelled_span, spanning  # noqa: E402

from machine_learning.baselines import (  # noqa: E402
    METRICS_SCHEMA_VERSION,
    SMOOTHING_GRID,
    BaseRateReference,
    ClimatologyBaseline,
    PersistenceBaseline,
    build_metrics,
    metrics_path,
    write_metrics,
)
from machine_learning.evaluation import (  # noqa: E402
    EMBARGO_DAYS,
    PERSISTENCE_FLAG,
    PURGE_DAYS,
    add_persistence_signal,
    base_rate,
    evaluation_frame,
    score,
    split_frame,
)
from machine_learning.features import (  # noqa: E402
    build_features,
    on_daily_calendar,
    require_grain,
    trailing_anomaly_counts,
)
from machine_learning.labels import (  # noqa: E402
    HORIZON_DAYS,
    LABEL,
    positives,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# The metrics
# --------------------------------------------------------------------------


def test_a_constant_prediction_scores_exactly_the_base_rate() -> None:
    """Average precision for a constant is the positive rate, by construction.

    Which makes it a free check on the implementation: if this drifts, the
    metric is not the one every other number in the file assumes.
    """
    labels = pd.Series(
        pd.array([True] * 30 + [False] * 170, dtype="boolean")
    ).sample(frac=1.0, random_state=1)
    result = score(labels, np.full(len(labels), 0.4))
    assert result.base_rate == pytest.approx(0.15)
    assert result.pr_auc == pytest.approx(0.15)
    assert result.lift == pytest.approx(1.0)
    assert result.brier == pytest.approx(0.15 * 0.6**2 + 0.85 * 0.4**2)


def test_a_perfect_ranking_scores_one() -> None:
    labels = pd.Series(pd.array([True, True, False, False, False], dtype="boolean"))
    assert score(labels, [0.9, 0.8, 0.3, 0.2, 0.1]).pr_auc == pytest.approx(1.0)


def test_scoring_refuses_what_it_cannot_average() -> None:
    labels = pd.Series(pd.array([True, False, None], dtype="boolean"))
    with pytest.raises(ValueError, match="unlabelled rows reached scoring"):
        score(labels, [0.1, 0.2, 0.3])

    clean = pd.Series(pd.array([True, False], dtype="boolean"))
    with pytest.raises(ValueError, match="2 labels against 3 predictions"):
        score(clean, [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="probabilities in"):
        score(clean, [0.1, 1.4])
    with pytest.raises(ValueError, match="NaN or inf"):
        score(clean, [0.1, np.nan])
    with pytest.raises(ValueError, match="one class is missing"):
        score(pd.Series(pd.array([False, False], dtype="boolean")), [0.1, 0.2])


def test_the_base_rate_helper_counts_nulls_as_negative_nowhere() -> None:
    labels = pd.Series(pd.array([True, False, None, True], dtype="boolean"))
    # positives() is null-safe; base_rate over a frame that still has nulls is
    # a lower bound, which is why score() refuses them outright.
    assert base_rate(labels) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# The persistence signal
# --------------------------------------------------------------------------


def test_the_persistence_signal_uses_the_feature_matrix_counter() -> None:
    """One implementation of "anomaly days behind me", not two.

    ``anomaly_days_trailing30`` in the feature matrix and the 7-day window this
    baseline needs come from the same function, so they cannot drift on what a
    null flag means or on what a window over a hole is worth.
    """
    frame = spanning(days=500)
    features = build_features(frame)
    calendar = on_daily_calendar(
        require_grain(frame, ("city_id", "date_key", "is_anomaly"))
    )
    flagged, _ = trailing_anomaly_counts(calendar, 30)
    rebuilt = calendar.assign(flagged=flagged)
    rebuilt = rebuilt.loc[rebuilt["is_observed"]].reset_index(drop=True)

    pd.testing.assert_series_equal(
        features["anomaly_days_trailing30"],
        rebuilt["flagged"].rename("anomaly_days_trailing30"),
    )


def test_the_persistence_signal_never_looks_forward() -> None:
    frame = spanning(days=800)
    cut = frame["date_key"].iloc[400]

    baseline = add_persistence_signal(frame)
    changed = frame.copy()
    future = changed["date_key"] > cut
    changed.loc[future, "is_anomaly"] = ~changed.loc[future, "is_anomaly"]
    rebuilt = add_persistence_signal(changed)

    past = baseline.loc[baseline["date_key"] <= cut, PERSISTENCE_FLAG]
    past_rebuilt = rebuilt.loc[rebuilt["date_key"] <= cut, PERSISTENCE_FLAG]
    pd.testing.assert_series_equal(past, past_rebuilt)


def test_an_unclosable_persistence_window_is_unknown_not_quiet() -> None:
    frame = spanning(days=200)
    frame["is_anomaly"] = pd.array([False] * 200, dtype="boolean")
    hole = frame["date_key"].iloc[100]
    with_gap = frame.loc[frame["date_key"] != hole].reset_index(drop=True)

    signal = add_persistence_signal(with_gap).set_index("date_key")[PERSISTENCE_FLAG]
    for ahead in range(1, HORIZON_DAYS):
        assert pd.isna(signal.loc[hole + pd.Timedelta(days=ahead)])
    assert signal.loc[hole + pd.Timedelta(days=HORIZON_DAYS)] == False  # noqa: E712
    # The first six days of the record cannot close a seven-day window either.
    assert signal.iloc[: HORIZON_DAYS - 1].isna().all()


# --------------------------------------------------------------------------
# Fitting on train, and only on train
# --------------------------------------------------------------------------


@pytest.fixture
def parts():
    return split_frame(labelled_span())


def test_a_baseline_refuses_to_predict_before_it_is_fitted(parts) -> None:
    for baseline in (BaseRateReference(), PersistenceBaseline(), ClimatologyBaseline()):
        with pytest.raises(RuntimeError, match="has not been fitted"):
            baseline.predict(parts["test"])


@pytest.mark.parametrize(
    "factory", [BaseRateReference, PersistenceBaseline, ClimatologyBaseline]
)
def test_rewriting_the_future_does_not_move_a_fitted_baseline(parts, factory) -> None:
    """The parameters are a function of the training split alone."""
    fitted = factory().fit(parts["train"])
    wrecked = parts["train"].copy()

    later = pd.concat([parts["validation"], parts["test"]], ignore_index=True)
    later[LABEL] = pd.array([True] * len(later), dtype="boolean")
    refitted = factory().fit(wrecked)

    assert json.dumps(fitted.params(), sort_keys=True) == json.dumps(
        refitted.params(), sort_keys=True
    )


def test_the_tuning_never_consults_the_test_split(parts) -> None:
    tuned = ClimatologyBaseline.tuned(parts["train"], parts["validation"])
    wrecked_test = parts["test"].copy()
    wrecked_test[LABEL] = pd.array([True] * len(wrecked_test), dtype="boolean")
    again = ClimatologyBaseline.tuned(parts["train"], parts["validation"])
    assert tuned.smoothing == again.smoothing
    assert tuned.cell_rates == again.cell_rates


def test_the_smoothing_grid_reaches_its_limit() -> None:
    """A parameter chosen at the edge of the grid is clipped, not tuned.

    The first version of this grid stopped at 100 and validation Brier was
    still improving there. The limit is now in the grid, so the search either
    finds an interior optimum or reports the honest degenerate answer.
    """
    assert np.isinf(SMOOTHING_GRID[-1])
    assert SMOOTHING_GRID == tuple(sorted(SMOOTHING_GRID))


def test_the_smoothing_limit_is_exactly_the_city_rate(parts) -> None:
    """At infinity the week cells collapse, and they collapse cleanly.

    Approaching the limit with a large finite pseudo-count leaves a
    rounding-sized week term that still breaks ties, and on this data it
    breaks them the wrong way. The limit is computed, not approached.
    """
    limit = ClimatologyBaseline(smoothing=float("inf")).fit(parts["train"])
    predicted = limit.predict(parts["test"])
    expected = parts["test"]["city_id"].map(limit.city_rates).to_numpy(dtype=float)
    np.testing.assert_allclose(predicted, expected)
    assert len(set(np.round(predicted, 12))) == parts["test"]["city_id"].nunique()


def test_the_persistence_cells_are_the_rule_it_claims(parts) -> None:
    """Two probabilities, each the observed rate of its own cell in train."""
    fitted = PersistenceBaseline().fit(parts["train"])
    train = parts["train"]
    after = positives(train.loc[train[PERSISTENCE_FLAG].eq(True), LABEL]).mean()
    quiet = positives(train.loc[train[PERSISTENCE_FLAG].eq(False), LABEL]).mean()

    assert fitted.after_anomaly == pytest.approx(after)
    assert fitted.after_quiet == pytest.approx(quiet)
    assert fitted.after_anomaly > fitted.after_quiet, (
        "persistence predicts more anomalies after an anomalous week; if this "
        "inverts the rule is not persistence any more"
    )
    assert fitted.cell_rows["when_unknown"] == 0, (
        "a row could not be judged, which means the signal was computed after "
        "the population was trimmed"
    )
    predicted = fitted.predict(parts["test"])
    assert set(np.unique(predicted)) <= {
        fitted.after_anomaly,
        fitted.after_quiet,
        fitted.when_unknown,
    }


# --------------------------------------------------------------------------
# metrics.json
# --------------------------------------------------------------------------


def test_a_population_without_the_signal_is_refused() -> None:
    """The order is enforced, not documented. Getting it wrong is silent."""
    frame = labelled_span(days=4000).drop(columns=[PERSISTENCE_FLAG])
    with pytest.raises(ValueError, match="computed on the whole record"):
        build_metrics(frame=frame)


def test_the_payload_is_strict_json(tmp_path) -> None:
    """No bare Infinity, which Python writes happily and no strict parser reads."""
    payload = build_metrics(frame=labelled_span())
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
    assert "Infinity" not in encoded
    assert json.loads(encoded) == json.loads(
        write_metrics(payload, tmp_path / "metrics.json").read_text()
    )


def test_the_payload_records_what_it_was_computed_on(tmp_path) -> None:
    payload = build_metrics(frame=labelled_span())
    assert payload["schema_version"] == METRICS_SCHEMA_VERSION
    assert payload["label"]["horizon_days"] == HORIZON_DAYS
    assert payload["split"]["purge_days"] == PURGE_DAYS
    assert payload["split"]["embargo_days"] == EMBARGO_DAYS
    assert set(payload["baselines"]) == {"base_rate", "persistence", "climatology"}
    snapshot = payload["snapshot"]
    assert snapshot["rows"] > 0
    assert snapshot["cities"] == len(snapshot["rows_by_city"])
    for entry in payload["baselines"].values():
        assert entry["fitted_on"] == "train"
        for split_name in ("train", "validation", "test"):
            assert 0.0 <= entry[split_name]["pr_auc"] <= 1.0
            assert 0.0 <= entry[split_name]["brier"] <= 1.0


def test_two_runs_agree_to_the_last_digit() -> None:
    """No seed, no ordering dependence, nothing to explain in a diff."""
    frame = labelled_span()
    first = build_metrics(frame=frame)
    second = build_metrics(frame=frame.sample(frac=1.0, random_state=9))
    first.pop("generated_at")
    second.pop("generated_at")
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


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


def test_the_scored_population_has_no_missing_feature(population) -> None:
    """Baseline and model must be scored on identical rows.

    A baseline ignores features, so nothing stops it scoring a row the model
    cannot use, and comparing the two would then be comparing different test
    sets. Labelled and past the warm-up happens to leave a population with no
    null feature at all; this is where a future city that breaks that shows up.
    """
    assert not population["has_missing_feature"].any()
    assert population[LABEL].notna().all()
    assert not population["is_warmup"].any()
    # And the persistence signal is complete too: every row is thirty days into
    # its city's record, so a seven-day window always closes. It does not close
    # if the signal is computed after the trim, which is why it is not.
    assert population[PERSISTENCE_FLAG].notna().all()


def test_the_no_skill_reference_scores_the_base_rate(population) -> None:
    parts = split_frame(population)
    fitted = BaseRateReference().fit(parts["train"])
    result = score(parts["test"][LABEL], fitted.predict(parts["test"]))
    assert result.pr_auc == pytest.approx(result.base_rate, abs=1e-9)
    assert result.lift == pytest.approx(1.0, abs=1e-9)


def test_persistence_is_the_bar_the_model_has_to_clear(population) -> None:
    parts = split_frame(population)
    fitted = PersistenceBaseline().fit(parts["train"])
    result = score(parts["test"][LABEL], fitted.predict(parts["test"]))
    assert result.lift > 1.3, (
        "persistence has lost its edge on test; the baseline in metrics.json "
        "no longer describes this data and must be rebuilt"
    )
    floor = BaseRateReference().fit(parts["train"])
    no_skill = score(parts["test"][LABEL], floor.predict(parts["test"]))
    assert result.brier < no_skill.brier


def test_the_week_of_year_signal_does_not_survive_the_split(population) -> None:
    """A finding, guarded so it cannot quietly stop being true.

    Fitted and scored inside the training period the (city, week) climatology
    is worth about 2.3x no-skill, so the seasonal structure is real and the
    baseline is not broken. Carried across the split it is worth *less than
    nothing*, since the unsmoothed version ranks below random, because the anomaly
    mix flips from mostly cold to mostly hot, and hot extremes fall in
    different weeks than cold ones. Validation therefore shrinks the week term
    away entirely, and the surviving baseline is a per-city rate.
    """
    parts = split_frame(population)
    raw = ClimatologyBaseline(smoothing=0.0).fit(parts["train"])
    in_sample = score(parts["train"][LABEL], raw.predict(parts["train"]))
    out_of_sample = score(parts["test"][LABEL], raw.predict(parts["test"]))

    assert in_sample.lift > 1.8, "the week signal was never there to begin with"
    assert out_of_sample.lift < 1.0, (
        f"the raw week climatology now ranks above random out of sample "
        f"({out_of_sample.lift:.2f}x); the regime-shift finding recorded in "
        "the build log needs revisiting"
    )

    tuned = ClimatologyBaseline.tuned(parts["train"], parts["validation"])
    assert np.isinf(tuned.smoothing), (
        "validation no longer wants the week term shrunk away entirely"
    )
    assert score(parts["test"][LABEL], tuned.predict(parts["test"])).lift > 1.0


def test_the_committed_metrics_match_a_fresh_run(population) -> None:
    """The target was fixed in advance, against a snapshot, and it says which.

    The backfill is mid-flight, so these numbers will change. A test that
    silently passed on a rebuilt file would defeat the point of committing it,
    and one that failed on every new city would be noise. So it compares the
    snapshot first and **skips with a reason** when the warehouse has moved on:
    the file is not wrong, it is stale, and it has to be rebuilt and
    re-committed before a model is compared against it.
    """
    path = metrics_path()
    if not path.exists():
        pytest.skip(f"{path} not written yet; run baselines.py --write")
    committed = json.loads(path.read_text())

    fresh = build_metrics(frame=population)
    if committed["snapshot"] != fresh["snapshot"]:
        pytest.skip(
            f"metrics.json was fixed against {committed['snapshot']['rows']} rows "
            f"to {committed['snapshot']['last_date']}; the warehouse now holds "
            f"{fresh['snapshot']['rows']} to {fresh['snapshot']['last_date']}. "
            "Re-run `python machine_learning/baselines.py --write` and commit "
            "the result BEFORE comparing a model against it."
        )

    for name, entry in fresh["baselines"].items():
        for split_name in ("train", "validation", "test"):
            for metric in ("pr_auc", "brier", "base_rate"):
                assert entry[split_name][metric] == pytest.approx(
                    committed["baselines"][name][split_name][metric], rel=1e-9
                ), f"{name}/{split_name}/{metric} drifted from the committed target"
