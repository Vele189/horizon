"""Tests that keep the model card honest.

A model card is the one document most likely to be written once and then
quietly outlived by the model it describes. Every retrain moves a score, and
nothing in a markdown file objects.

So the card carries a **Key figures** table, and every row of it is looked up
by name in ``metrics.json`` and compared. A retrain that changes a number makes
this suite fail until the card is updated, which is the only arrangement under
which a reader can trust it.

The rest is structural: the sections a model card is expected to have, the
feature list matching what the model actually records, and the two claims the
proposal specifically asks be made in writing: that this does not compete with
operational forecasting, and that accuracy is not being reported.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
CARD = REPO_ROOT / "docs" / "model-card.md"
METRICS = REPO_ROOT / "machine_learning" / "artifacts" / "metrics.json"

#: Where each key figure comes from. A callable rather than a dotted path,
#: because several of them are a lookup inside a list and the point is that the
#: card is checked against the *source*, not against a second transcription.
FIGURES = {
    "model_version": lambda d: Path(
        d["model"]["artifacts"][d["model"]["recommended_variant"]]["filename"]
    ).stem,
    "feature_count": lambda d: _artifact(d)["feature_count"],
    "train_rows": lambda d: _artifact(d)["training_window"]["rows"],
    "train_positive_rate": lambda d: _artifact(d)["training_window"]["positive_rate"],
    "test_rows": lambda d: _recommended(d)["test"]["rows"],
    "test_positive_rate": lambda d: _recommended(d)["test"]["base_rate"],
    "test_pr_auc": lambda d: _recommended(d)["test"]["pr_auc"],
    "test_brier": lambda d: _recommended(d)["test"]["brier"],
    "test_mean_predicted": lambda d: _recommended(d)["test"]["mean_predicted"],
    "test_f1": lambda d: _evaluation_row(d)["f1"],
    "test_precision": lambda d: _evaluation_row(d)["precision"],
    "test_recall": lambda d: _evaluation_row(d)["recall"],
    "decision_threshold": lambda d: _evaluation_row(d)["threshold"],
    "alert_budget_per_city_year": lambda d: _decision(d)["budget_alerts_per_city_year"],
    "alert_budget_threshold": lambda d: _decision(d)["threshold"],
    "implied_cost_ratio": lambda d: _decision(d)["implied_cost_ratio"],
    "baseline_persistence_pr_auc": lambda d: d["baselines"]["persistence"]["test"][
        "pr_auc"
    ],
    "baseline_climatology_pr_auc": lambda d: d["baselines"]["climatology"]["test"][
        "pr_auc"
    ],
    "no_skill_pr_auc": lambda d: d["baselines"]["base_rate"]["test"]["pr_auc"],
    "static_city_share_of_shap": lambda d: d["explainability"]["static_city_share"],
    "n_estimators": lambda d: _artifact(d)["hyperparameters"]["n_estimators"],
    "seed": lambda d: _artifact(d)["hyperparameters"]["seed"],
}

#: The sections a model card is expected to carry. Absence of any of them is
#: the difference between a model card and a README section.
REQUIRED_SECTIONS = (
    "Intended use",
    "Data",
    "Evaluation",
    "Limitations",
    "Ethical considerations",
    "Reproducing this",
)


def _recommended(payload):
    return payload["model"]["variants"][payload["model"]["recommended_variant"]]


def _artifact(payload):
    return payload["model"]["artifacts"][payload["model"]["recommended_variant"]]


def _decision(payload):
    return payload["model"]["calibration"]["decision"]


def _evaluation_row(payload):
    wanted = f"model_{payload['model']['recommended_variant']}"
    return next(
        row for row in payload["evaluation"]["summary"] if row["predictor"] == wanted
    )


@pytest.fixture(scope="module")
def card() -> str:
    if not CARD.exists():
        pytest.fail(f"{CARD.relative_to(REPO_ROOT)} is missing")
    return CARD.read_text()


@pytest.fixture(scope="module")
def metrics():
    if not METRICS.exists():
        pytest.skip("metrics.json not written yet")
    payload = json.loads(METRICS.read_text())
    for key in ("model", "evaluation", "explainability", "baselines"):
        if key not in payload:
            pytest.skip(f"metrics.json has no {key!r} block yet")
    if "artifacts" not in payload["model"]:
        pytest.skip("no artefacts recorded yet; run train.py --write")
    return payload


def stated(card: str) -> dict[str, str]:
    """The Key figures table, parsed by name rather than by position.

    Scoped to that one section: the card has several two-column tables and a
    header row reading ``| figure | value |`` is not a figure.
    """
    section = card.split("## Key figures", 1)
    assert len(section) == 2, "no Key figures section"
    body = section[1].split("\n---", 1)[0]
    rows = re.findall(r"^\|\s*([a-z0-9_]+)\s*\|\s*([^|]+?)\s*\|\s*$", body, re.M)
    return {name: value for name, value in rows if name != "figure"}


# --------------------------------------------------------------------------
# The numbers
# --------------------------------------------------------------------------


def test_the_card_states_every_key_figure(card) -> None:
    missing = sorted(set(FIGURES) - set(stated(card)))
    assert not missing, f"the Key figures table is missing {missing}"


def test_no_figure_is_stated_that_is_not_checked(card) -> None:
    """A row nothing verifies is exactly the row that goes stale."""
    extra = sorted(set(stated(card)) - set(FIGURES))
    assert not extra, (
        f"the Key figures table states {extra}, which nothing checks against "
        "metrics.json. Either wire it up in FIGURES or take it out."
    )


@pytest.mark.parametrize("figure", sorted(FIGURES))
def test_each_key_figure_matches_the_metrics(card, metrics, figure) -> None:
    """The card cannot outlive the model it describes.

    Compared at the precision the card states, so writing 0.3494 is checked to
    four places rather than against the full float, because the card is for reading.
    """
    claimed = stated(card)[figure]
    actual = FIGURES[figure](metrics)

    if isinstance(actual, str):
        assert claimed == actual
        return
    if isinstance(actual, int):
        assert int(claimed.replace(" ", "").replace(",", "")) == actual
        return

    places = len(claimed.split(".")[1]) if "." in claimed else 0
    assert float(claimed) == pytest.approx(round(float(actual), places), abs=1e-9), (
        f"{figure}: the card says {claimed}, metrics.json says {actual}"
    )


def test_the_card_names_the_artefact_that_is_actually_committed(card, metrics) -> None:
    """A retrain changes the fingerprint, and the filename carries it."""
    artefact = _artifact(metrics)
    stem = Path(artefact["filename"]).stem
    assert stem in card
    assert (METRICS.parent / artefact["filename"]).exists(), (
        "the card names an artefact that is not on disk"
    )


def test_the_card_lists_every_feature_by_name_and_in_order(card, metrics) -> None:
    """The order is the part a caller can get wrong silently, so the card
    prints the list rather than describing it."""
    from machine_learning.features import feature_columns

    artefact = _artifact(metrics)
    assert artefact["features"] == list(feature_columns())
    assert str(len(feature_columns())) in stated(card)["feature_count"]

    listed = re.findall(r"^\s*\d+\.\s+`([a-z0-9_]+)`", card, re.M)
    assert listed == list(feature_columns()), (
        "the card's numbered feature list does not match feature_columns(), "
        "in content or in order"
    )


# --------------------------------------------------------------------------
# The shape of a model card
# --------------------------------------------------------------------------


def test_the_card_has_the_sections_a_model_card_has(card) -> None:
    for section in REQUIRED_SECTIONS:
        assert re.search(rf"^#+\s.*{re.escape(section)}", card, re.M | re.I), (
            f"no {section!r} section"
        )


def test_the_card_says_what_the_model_is_not(card) -> None:
    """The proposal asks for this in writing: stating it plainly reads as
    competence, and overclaiming reads as inexperience."""
    lowered = card.lower()
    assert "not a weather forecast" in lowered
    assert "ecmwf" in lowered and "gfs" in lowered
    assert "out of scope" in lowered


def test_the_card_does_not_report_accuracy(card) -> None:
    """It may explain why accuracy is excluded; it may not quote one as a score."""
    assert "86.4%" in card, "the card should say what always-quiet would score"
    assert not re.search(r"\|\s*accuracy\s*\|", card, re.I), (
        "accuracy appears as a reported figure"
    )
    # The sklearn function name is deliberately not written here. The scan in
    # test_evaluation.py already forbids it in every .py file, and naming it
    # in a third test file is how that scan keeps finding its own explanation.


def test_the_card_leads_its_limitations_with_the_calibration_one(card) -> None:
    """It is the failure with consequences, so it is not third on the list."""
    limitations = card.split("## Limitations", 1)
    assert len(limitations) == 2, "no Limitations section"
    body = limitations[1]
    first = body.split("### ")[1]
    assert "under-state" in first.lower() or "understate" in first.lower()
    assert "recalibrat" in first.lower()


def test_every_limitation_the_workstream_found_is_recorded(card) -> None:
    """The nine tickets before this one each surfaced something; none of it
    should have to be rediscovered from the code."""
    lowered = card.lower()
    for finding in (
        "not a forecast",           # no synoptic input
        "city lookup",              # latitude/elevation are constants
        # A city whose baseline rests on too few observations over-flags by
        # construction. Matched on the finding rather than on the city: this
        # read "four years" while Tokyo was the example, and Tokyo's record has
        # since completed. The defect moved to Sydney; it did not go away.
        "over-flag",
        "duration",                 # phoenix 2023
        "non-stationary",           # the week-of-year regime shift
        "snapshot",                 # the incomplete backfill
        "own year",                 # the climatology's residual leak
    ):
        assert finding in lowered, f"the card does not record: {finding}"


def test_the_readme_points_at_the_card() -> None:
    readme = (REPO_ROOT / "README.md").read_text()
    assert "docs/model-card.md" in readme
