"""The nowcast card is checked against the run it describes.

Same contract as `test_model_card.py`: every figure in the Key figures table is
resolved from `nowcast-metrics.json`, and nothing may appear there that is not
resolved. A card is a document people quote from, and one that quietly falls a
retrain behind is worse than none, because nobody thinks to distrust it.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
CARD = REPO_ROOT / "docs" / "nowcast-card.md"
METRICS = REPO_ROOT / "machine_learning" / "artifacts" / "nowcast-metrics.json"


def _horizon(payload, hours: int):
    return payload["horizons"][str(hours)]


def _test(payload, hours: int, key: str):
    return _horizon(payload, hours)["test"][key]


#: Where each figure comes from. Callables rather than dotted paths, because
#: several are a lookup inside a horizon and the point is that the card is
#: checked against the *source* and not against a second transcription.
FIGURES = {
    "cities": lambda d: len(d["cities"]),
    "feature_count": lambda d: _horizon(d, 24)["feature_count"],
    "seed": lambda d: d["seed"],
    "train_rows_24h": lambda d: _horizon(d, 24)["rows"]["train"],
    "validation_rows_24h": lambda d: _horizon(d, 24)["rows"]["validation"],
    "test_rows_24h": lambda d: _horizon(d, 24)["rows"]["test"],
    "test_sd_observed_24h": lambda d: _test(d, 24, "sd_observed"),
    "test_rmse_24h": lambda d: _test(d, 24, "rmse"),
    "test_mae_24h": lambda d: _test(d, 24, "mae"),
    "test_persistence_rmse_24h": lambda d: _test(d, 24, "persistence_rmse"),
    "test_climatology_month_rmse_24h": lambda d: _test(d, 24, "climatology_month_rmse"),
    "test_skill_vs_persistence_24h": lambda d: _test(d, 24, "skill_vs_persistence"),
    "test_skill_vs_climatology_month_24h": lambda d: _test(
        d, 24, "skill_vs_climatology_month"
    ),
    "test_rmse_48h": lambda d: _test(d, 48, "rmse"),
    "test_persistence_rmse_48h": lambda d: _test(d, 48, "persistence_rmse"),
    "test_climatology_month_rmse_48h": lambda d: _test(d, 48, "climatology_month_rmse"),
    "test_skill_vs_persistence_48h": lambda d: _test(d, 48, "skill_vs_persistence"),
    "test_skill_vs_climatology_month_48h": lambda d: _test(
        d, 48, "skill_vs_climatology_month"
    ),
    "test_rmse_72h": lambda d: _test(d, 72, "rmse"),
    "test_persistence_rmse_72h": lambda d: _test(d, 72, "persistence_rmse"),
    "test_climatology_month_rmse_72h": lambda d: _test(d, 72, "climatology_month_rmse"),
    "test_skill_vs_persistence_72h": lambda d: _test(d, 72, "skill_vs_persistence"),
    "test_skill_vs_climatology_month_72h": lambda d: _test(
        d, 72, "skill_vs_climatology_month"
    ),
    "test_non_overlapping_rmse_24h": lambda d: _horizon(d, 24)[
        "test_non_overlapping"
    ]["rmse"],
    "test_non_overlapping_rmse_72h": lambda d: _horizon(d, 72)[
        "test_non_overlapping"
    ]["rmse"],
    "worst_city_skill_72h": lambda d: min(
        city["skill_vs_climatology_month"]
        for city in _horizon(d, 72)["test_by_city"].values()
    ),
}

REQUIRED_SECTIONS = (
    "Intended use",
    "Data",
    "Evaluation",
    "Limitations",
    "Ethical considerations",
    "Reproducing this",
)


def flatten(text: str) -> str:
    """Lower-cased, accent-stripped, single-spaced.

    Prose assertions must not depend on where markdown happened to wrap a line
    or on whether a city is spelled with its diacritic. Both were real failures
    here: "does not replace it" straddled a line break, and the card says
    "Sao Paulo" with an accent while the warehouse key is `sao_paulo`.
    """
    unaccented = "".join(
        character
        for character in unicodedata.normalize("NFD", text)
        if not unicodedata.combining(character)
    )
    return " ".join(unaccented.lower().split())


@pytest.fixture(scope="module")
def card() -> str:
    if not CARD.exists():
        pytest.fail(f"{CARD.relative_to(REPO_ROOT)} is missing")
    return CARD.read_text()


@pytest.fixture(scope="module")
def metrics():
    if not METRICS.exists():
        pytest.skip("nowcast-metrics.json not written yet")
    payload = json.loads(METRICS.read_text())
    if "horizons" not in payload:
        pytest.skip("nowcast-metrics.json has no horizons block yet")
    return payload


def stated(card: str) -> dict[str, str]:
    """The Key figures table, parsed by name rather than by position."""
    section = card.split("## Key figures", 1)
    assert len(section) == 2, "no Key figures section"
    body = section[1].split("\n---", 1)[0]
    rows = re.findall(r"^\|\s*([a-z0-9_]+)\s*\|\s*([^|]+?)\s*\|\s*$", body, re.M)
    return {name: value for name, value in rows if name != "figure"}


def test_the_card_states_every_key_figure(card) -> None:
    missing = sorted(set(FIGURES) - set(stated(card)))
    assert not missing, f"the Key figures table is missing {missing}"


def test_no_figure_is_stated_that_is_not_checked(card) -> None:
    """A row nothing verifies is exactly the row that goes stale."""
    extra = sorted(set(stated(card)) - set(FIGURES))
    assert not extra, (
        f"the Key figures table states {extra}, which nothing checks against "
        "nowcast-metrics.json. Either wire it up in FIGURES or take it out."
    )


@pytest.mark.parametrize("figure", sorted(FIGURES))
def test_each_key_figure_matches_the_metrics(card, metrics, figure) -> None:
    """Compared at the precision the card states, because the card is for reading."""
    claimed = stated(card)[figure]
    actual = FIGURES[figure](metrics)

    if isinstance(actual, int):
        assert int(claimed.replace(" ", "").replace(",", "")) == actual
        return
    places = len(claimed.split(".")[1]) if "." in claimed else 0
    assert float(claimed) == pytest.approx(round(float(actual), places), abs=1e-9), (
        f"{figure}: the card says {claimed}, nowcast-metrics.json says {actual}"
    )


def test_the_card_carries_every_required_section(card) -> None:
    missing = [name for name in REQUIRED_SECTIONS if f"## {name}" not in card]
    assert not missing, missing


def test_the_card_says_it_is_not_the_other_model(card) -> None:
    """Two model cards in one repository is two chances to read the wrong one.

    The distinction is the first thing a reader needs and the easiest thing to
    leave implicit, so it is asserted rather than assumed.
    """
    flat = flatten(card)
    assert "does not replace it" in flat
    assert "not a weather forecast" in flat


def test_the_card_reports_where_the_model_loses(card, metrics) -> None:
    """A card that only lists wins is advertising.

    The nowcast is beaten outright by a monthly mean in at least one city at 72
    hours, and the card must say which and by how much rather than quoting the
    pooled figure and stopping.
    """
    losing = {
        city
        for city, scores in _horizon(metrics, 72)["test_by_city"].items()
        if scores["skill_vs_climatology_month"] < 0
    }
    assert losing, "no city loses at 72h; this test's premise needs revisiting"
    flat = flatten(card)
    for city in losing:
        assert city.replace("_", " ") in flat, (
            f"{city} is beaten by climatology at 72h and the card does not say so"
        )


def test_the_two_cards_do_not_claim_each_others_numbers(card) -> None:
    """The seven-day model's headline figures must not appear here.

    Both cards are markdown tables of metrics in the same repository, and the
    failure mode is a copy-paste that survives review because every number in
    it looks plausible.
    """
    other = (REPO_ROOT / "docs" / "model-card.md").read_text()
    for token in ("pr_auc", "brier", "pr-auc"):
        assert token in flatten(other)
        assert token not in flatten(card), (
            f"{token!r} belongs to the anomaly classifier, not to the nowcast"
        )
