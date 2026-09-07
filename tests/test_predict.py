"""Tests for the inference script and the table it writes.

The Risk Horizon view reads ``fact_ml_predictions``, so the things worth
defending are the ones that would make a chart quietly wrong rather than
visibly broken.

**The grain.** One row per city per forecast date, enforced by a primary key
and by an upsert, so a second run of the same day replaces rather than
appends. A dashboard reading a table that had grown a duplicate would either
double-count or pick one arbitrarily, and neither shows up as an error.

**The window.** ``horizon_start`` is always the day *after* ``forecast_date``,
because day *t* is a feature. That is a check constraint rather than a
convention: a row whose horizon starts on its own forecast date is not a
differently-shaped row, it is a bug that would otherwise be found in a chart.

**The schema.** If ``features.py`` has gained, lost or renamed a column since
the model was trained, scoring must fail rather than proceed — a matrix with a
renamed column still has the right shape and the model will return a
probability for every row of it.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("xgboost")
pytest.importorskip("sqlalchemy")

from machine_learning.artifact import LoadedModel  # noqa: E402
from machine_learning.features import feature_columns  # noqa: E402
from machine_learning.labels import HORIZON_DAYS  # noqa: E402
from machine_learning.predict import (  # noqa: E402
    DEFAULT_DATES,
    PREDICTIONS_TABLE,
    PredictionError,
    apply_schema,
    coverage,
    decision_threshold,
    score_horizon,
    write_predictions,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SENTINEL = "__test_city__"


def sentinel_rows(count: int = 3, *, risk: float = 0.42) -> pd.DataFrame:
    base = dt.date(1990, 1, 1)
    dates = [base + dt.timedelta(days=index) for index in range(count)]
    return pd.DataFrame(
        {
            "city_id": SENTINEL,
            "forecast_date": dates,
            "horizon_start": [day + dt.timedelta(days=1) for day in dates],
            "horizon_end": [day + dt.timedelta(days=HORIZON_DAYS) for day in dates],
            "horizon_days": HORIZON_DAYS,
            "risk_score": risk,
            "prediction_label": risk >= 0.1,
            "decision_threshold": 0.1,
            "model_version": "model-unweighted-v1-testtesttest",
            "model_variant": "unweighted",
            "feature_count": len(feature_columns()),
        }
    )


# --------------------------------------------------------------------------
# The schema check, before anything is written
# --------------------------------------------------------------------------


def test_a_moved_feature_schema_is_fatal() -> None:
    """Not a warning. A renamed column keeps the shape and loses the meaning."""
    stale = LoadedModel(
        estimator=None,
        features=("temperature_2m_mean", "a_feature_that_no_longer_exists"),
        n_estimators=1,
        fingerprint="deadbeefcafe",
        path=Path("model-unweighted-v1-deadbeefcafe.joblib"),
    )
    frame = pd.DataFrame({"city_id": ["x"], "date_key": [pd.Timestamp("2020-01-01")]})
    with pytest.raises(PredictionError, match="feature schema has moved"):
        score_horizon(frame=frame, model=stale)


def test_the_schema_check_names_both_lists() -> None:
    stale = LoadedModel(
        estimator=None,
        features=tuple(feature_columns())[:-1],
        n_estimators=1,
        fingerprint="deadbeefcafe",
        path=Path("m.joblib"),
    )
    frame = pd.DataFrame({"city_id": ["x"], "date_key": [pd.Timestamp("2020-01-01")]})
    with pytest.raises(PredictionError) as raised:
        score_horizon(frame=frame, model=stale)
    message = str(raised.value)
    assert "model expects" in message and "features.py provides" in message
    assert "Retrain before scoring" in message


def test_no_feature_logic_is_duplicated_here() -> None:
    """The ticket asks that features.py be reused, so nothing is recomputed.

    A window computed here would be a second implementation free to drift from
    the one the model was trained on, and a drift would be silent — both would
    still produce a plausible column of numbers.
    """
    source = (REPO_ROOT / "machine_learning" / "predict.py").read_text()
    for forbidden in (".rolling(", ".shift(", "day_of_year", "np.sin", "np.cos"):
        assert forbidden not in source, (
            f"predict.py computes {forbidden!r} itself; feature construction "
            "belongs in features.py"
        )


# --------------------------------------------------------------------------
# The threshold comes from the evaluation
# --------------------------------------------------------------------------


def test_the_threshold_is_the_evaluated_operating_point() -> None:
    threshold, variant = decision_threshold()
    payload = json.loads(
        (REPO_ROOT / "machine_learning/artifacts/metrics.json").read_text()
    )
    row = next(
        item
        for item in payload["evaluation"]["summary"]
        if item["predictor"] == f"model_{variant}"
    )
    assert threshold == pytest.approx(row["threshold"])
    assert 0.0 < threshold < 0.5, (
        "a calibrated model at this base rate rarely predicts above 0.5; a "
        "threshold there would label nothing"
    )


def test_a_missing_evaluation_is_refused_not_defaulted(tmp_path) -> None:
    absent = tmp_path / "nothing.json"
    with pytest.raises(PredictionError, match="does not exist"):
        decision_threshold(absent)

    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps({"baselines": {}}))
    with pytest.raises(PredictionError, match="records no evaluation"):
        decision_threshold(bare)

    wrong = tmp_path / "wrong.json"
    wrong.write_text(
        json.dumps(
            {
                "model": {"recommended_variant": "unweighted"},
                "evaluation": {
                    "summary": [{"predictor": "persistence", "threshold": 0.2}]
                },
            }
        )
    )
    with pytest.raises(PredictionError, match="no evaluation row"):
        decision_threshold(wrong)


# --------------------------------------------------------------------------
# Coverage names every city
# --------------------------------------------------------------------------


def test_coverage_accounts_for_every_registry_city() -> None:
    frame = pd.DataFrame(
        {
            "city_id": ["alpha", "beta"],
            "date_key": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "is_warmup": [False, True],
            "has_missing_feature": [False, False],
        }
    )
    scored = frame.head(1)
    report = coverage(
        frame, scored, roster=["alpha", "beta", "gamma"], ingested=["alpha", "beta"]
    )

    assert report["roster"] == ["alpha", "beta", "gamma"]
    assert report["scored"] == ["alpha"]
    assert report["not_scored"]["beta"] == "inside the feature warm-up"
    assert report["not_scored"]["gamma"] == "never ingested"
    assert set(report["scored"]) | set(report["not_scored"]) == set(report["roster"])


def test_coverage_distinguishes_a_null_feature_from_a_warm_up() -> None:
    frame = pd.DataFrame(
        {
            "city_id": ["alpha"],
            "date_key": pd.to_datetime(["2026-01-01"]),
            "is_warmup": [False],
            "has_missing_feature": [True],
        }
    )
    report = coverage(frame, frame.head(0), roster=["alpha"], ingested=["alpha"])
    assert "no climatology baseline" in report["not_scored"]["alpha"]


# --------------------------------------------------------------------------
# Against the warehouse
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prepared(engine):
    apply_schema(engine)
    return engine


@pytest.fixture
def sentinel(prepared):
    """Rows under a city_id no backfill will ever produce, removed after."""
    from sqlalchemy import text

    yield prepared
    with prepared.begin() as connection:
        connection.execute(
            text(f"delete from {PREDICTIONS_TABLE} where city_id = :city"),
            {"city": SENTINEL},
        )


def test_the_table_holds_one_row_per_city_and_forecast_date(prepared) -> None:
    from sqlalchemy import text

    with prepared.connect() as connection:
        rows = connection.execute(
            text(f"select count(*) from {PREDICTIONS_TABLE}")
        ).scalar()
        distinct = connection.execute(
            text(
                "select count(*) from (select distinct city_id, forecast_date "
                f"from {PREDICTIONS_TABLE}) as grain"
            )
        ).scalar()
    if not rows:
        pytest.skip("no predictions written yet; run predict.py")
    assert rows == distinct


def test_re_running_replaces_rather_than_appends(sentinel) -> None:
    from sqlalchemy import text

    engine = sentinel
    first = write_predictions(engine, sentinel_rows(risk=0.42))
    assert first == 3

    with engine.connect() as connection:
        before = connection.execute(
            text(
                f"select count(*), max(scored_at) from {PREDICTIONS_TABLE} "
                "where city_id = :city"
            ),
            {"city": SENTINEL},
        ).one()

    write_predictions(engine, sentinel_rows(risk=0.77))
    with engine.connect() as connection:
        after = connection.execute(
            text(
                "select count(*), max(scored_at), max(risk_score) from "
                f"{PREDICTIONS_TABLE} where city_id = :city"
            ),
            {"city": SENTINEL},
        ).one()

    assert after[0] == before[0] == 3, "a second run appended instead of replacing"
    assert after[2] == pytest.approx(0.77), "the row was not updated"
    assert after[1] >= before[1], "scored_at should advance on a rewrite"


def test_the_horizon_constraints_are_enforced_by_the_database(sentinel) -> None:
    """The semantics are constraints, not conventions.

    Each of these would otherwise be discovered from a chart with the wrong
    dates on its axis.
    """
    from sqlalchemy.exc import IntegrityError

    engine = sentinel
    broken = sentinel_rows(1)
    broken["horizon_start"] = broken["forecast_date"]  # not the day after
    with pytest.raises(IntegrityError):
        write_predictions(engine, broken)

    broken = sentinel_rows(1)
    broken["risk_score"] = 1.4
    with pytest.raises(IntegrityError):
        write_predictions(engine, broken)

    broken = sentinel_rows(1)
    broken["prediction_label"] = False  # risk 0.42 is over the 0.1 threshold
    with pytest.raises(IntegrityError):
        write_predictions(engine, broken)

    broken = sentinel_rows(1)
    broken["horizon_end"] = broken["forecast_date"]
    with pytest.raises(IntegrityError):
        write_predictions(engine, broken)


def test_the_written_rows_say_which_model_made_them(prepared) -> None:
    from sqlalchemy import text

    with prepared.connect() as connection:
        rows = connection.execute(
            text(
                "select distinct model_version, model_variant, feature_count "
                f"from {PREDICTIONS_TABLE} where city_id <> :sentinel"
            ),
            {"sentinel": SENTINEL},
        ).fetchall()
    if not rows:
        pytest.skip("no predictions written yet; run predict.py")

    payload = json.loads(
        (REPO_ROOT / "machine_learning/artifacts/metrics.json").read_text()
    )
    artefacts = payload.get("model", {}).get("artifacts", {})
    known = {Path(entry["filename"]).stem for entry in artefacts.values()}
    for version, variant, count in rows:
        assert version in known, f"{version} is not a committed artefact"
        assert variant in artefacts
        assert count == len(feature_columns())


def test_a_real_scoring_run_covers_the_horizon_it_claims(engine) -> None:
    from sqlalchemy import text

    with engine.connect() as connection:
        observations = connection.execute(
            text("select count(*) from gold_marts.fact_weather_observations")
        ).scalar()
    if not observations:
        pytest.skip("fact_weather_observations not built")

    predictions, report = score_horizon(engine, dates=DEFAULT_DATES)
    if predictions.empty:
        pytest.skip("nothing scorable on this snapshot")

    assert len(report["roster"]) == 15, "the registry should hold fifteen cities"
    assert set(report["scored"]) | set(report["not_scored"]) == set(report["roster"])

    # Every scored city gets the same strip of forecast dates, each covering
    # exactly the label's horizon, starting the day after.
    per_city = predictions.groupby("city_id")["forecast_date"].nunique()
    assert (per_city <= DEFAULT_DATES).all()
    assert (predictions["horizon_days"] == HORIZON_DAYS).all()
    spans = pd.to_datetime(predictions["horizon_end"]) - pd.to_datetime(
        predictions["horizon_start"]
    )
    assert (spans == pd.Timedelta(days=HORIZON_DAYS - 1)).all()
    starts = pd.to_datetime(predictions["horizon_start"]) - pd.to_datetime(
        predictions["forecast_date"]
    )
    assert (starts == pd.Timedelta(days=1)).all()

    assert predictions["risk_score"].between(0, 1).all()
    assert (
        predictions["prediction_label"]
        == (predictions["risk_score"] >= predictions["decision_threshold"])
    ).all()
    assert predictions["model_version"].nunique() == 1


def test_the_unscored_cities_have_named_reasons(engine) -> None:
    """Ten of fifteen cannot be scored today, and the report says why for each.

    The ticket asks for output verified across all fifteen. Five can be scored;
    an empty row for the rest would be a fabrication and a missing row would be
    silent, so each one carries a reason instead.
    """
    _, report = score_horizon(engine, dates=1)
    if not report["roster"]:
        pytest.skip("dim_cities not built")
    for city_id, reason in report["not_scored"].items():
        assert reason, city_id
        assert reason in {
            "never ingested",
            "inside the feature warm-up",
            "a feature is null — no climatology baseline",
            "no observations inside the scored window",
            "no rows in the scored window",
        }, f"{city_id}: {reason}"
