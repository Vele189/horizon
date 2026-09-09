"""Tests for the NOAA teleconnection indices, and for when they may be read.

The ticket calls this the ticket most likely to introduce silent leakage, and
it is right. An index value carries a nominal label -- "January 1998" -- and
that label is not when the value existed. Join on it and the model reads the
future while every existing test stays green: the column is correctly named,
correctly typed, correctly joined, non-null, and answers a question about days
that had not happened.

There is no metric that catches this. Every score simply improves. So the rule
is asserted directly, at the boundary, on both sides of it.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("sqlalchemy")

from ingestion.teleconnections import (  # noqa: E402
    SEASONS,
    Index,
    TeleconnectionError,
    changed_rows,
    covered_span,
    latest_vintage,
    load_registry,
    parse_cpc_monthly,
    parse_cpc_seasonal,
    parse_psl_monthly,
    publication_date,
)
from machine_learning.features import (  # noqa: E402
    TELECONNECTION_FEATURES,
    attach_teleconnections,
    feature_columns,
    teleconnection_steps,
)

REGISTRY = {index.id: index for index in load_registry()}


def vintage_rows(rows: list[dict]) -> pd.DataFrame:
    """A vintage table shaped like the warehouse's, from literals."""
    frame = pd.DataFrame(rows)
    spans = [
        covered_span(REGISTRY[row["index_id"]], row["nominal_period"])
        for _, row in frame.iterrows()
    ]
    frame["covers_start"] = [start for start, _ in spans]
    frame["covers_end"] = [end for _, end in spans]
    if "publication_date" not in frame:
        frame["publication_date"] = [
            publication_date(REGISTRY[row["index_id"]], row["nominal_period"])
            for _, row in frame.iterrows()
        ]
    if "vintage_at" not in frame:
        frame["vintage_at"] = pd.Timestamp("2026-09-09", tz="UTC")
    return frame


# ---------------------------------------------------------------------------
# The boundary, which is the trap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index_id", sorted(REGISTRY))
def test_a_value_is_invisible_the_day_before_it_publishes(index_id):
    """Not readable on publication_date - 1, readable on publication_date.

    The whole ticket in one assertion, checked at the boundary rather than in
    the middle where any off-by-one would pass.
    """
    index = REGISTRY[index_id]
    nominal = dt.date(2010, 6, 1)
    published = publication_date(index, nominal)
    rows = vintage_rows([
        {"index_id": index_id, "nominal_period": dt.date(2010, 1, 1), "value": -1.0},
        {"index_id": index_id, "nominal_period": nominal, "value": 42.0},
    ])
    probe = pd.DataFrame({
        "date_key": pd.to_datetime([
            published - dt.timedelta(days=1), published
        ])
    })
    joined = attach_teleconnections(probe, _all_indices(rows))

    assert joined[index_id].iloc[0] != 42.0, (
        f"{index_id} was readable a day before it published"
    )
    assert joined[index_id].iloc[1] == 42.0


def _all_indices(rows: pd.DataFrame) -> pd.DataFrame:
    """Pad a single-index frame so `attach_teleconnections` has all four."""
    parts = [rows]
    present = set(rows["index_id"])
    for name in TELECONNECTION_FEATURES:
        if name not in present:
            parts.append(
                vintage_rows([
                    {"index_id": name,
                     "nominal_period": dt.date(1990, 1, 1),
                     "value": 0.0}
                ])
            )
    return pd.concat(parts, ignore_index=True)


def test_a_centred_index_covers_a_month_past_its_own_label():
    """The ONI labelled January covers December through February.

    This single month is the difference between a correct join and a leak, and
    it is invisible in the column name. The ONI for a month cannot exist until
    the month *after* it has finished.
    """
    oni = REGISTRY["oni"]
    start, end = covered_span(oni, dt.date(1998, 1, 1))

    assert start == dt.date(1997, 12, 1)
    assert end == dt.date(1998, 2, 28)
    assert publication_date(oni, dt.date(1998, 1, 1)) > dt.date(1998, 2, 28)


def test_a_monthly_index_covers_only_its_own_month():
    nao = REGISTRY["nao"]
    assert covered_span(nao, dt.date(1998, 1, 1)) == (
        dt.date(1998, 1, 1), dt.date(1998, 1, 31)
    )


@pytest.mark.parametrize("index_id", sorted(REGISTRY))
def test_publication_always_follows_the_last_month_covered(index_id):
    """The invariant, over a span of periods including leap years and Decembers.

    Also enforced as a check constraint on the table, so a row violating it
    cannot be written; asserted here so the arithmetic is wrong loudly rather
    than at insert time.
    """
    index = REGISTRY[index_id]
    for year in (1996, 1999, 2000, 2024):
        for month in range(1, 13):
            nominal = dt.date(year, month, 1)
            _, covered_to = covered_span(index, nominal)
            assert publication_date(index, nominal) > covered_to


def test_joining_on_the_nominal_period_would_have_leaked():
    """The counterfactual, pinned so the correct join stays distinguishable.

    A join on the label gives the December 1997 ONI from 1 December 1997. The
    correct join gives it from 15 February 1998, two and a half months later,
    because the value labelled December summarises November through January and
    then waits for the lag.

    On 15 December 1997 the newest ONI that existed was the one labelled
    *October*, published that very day: it covers September through November,
    and the lag runs from the end of November. The November and December
    values were still inside their own coverage windows and did not exist.
    Half the fixture is in the future on the date being scored, which is the
    size of the leak a nominal join would take.
    """
    oni = REGISTRY["oni"]
    rows = vintage_rows([
        {"index_id": "oni", "nominal_period": dt.date(1997, 9, 1), "value": 1.82},
        {"index_id": "oni", "nominal_period": dt.date(1997, 10, 1), "value": 2.24},
        {"index_id": "oni", "nominal_period": dt.date(1997, 11, 1), "value": 2.34},
        {"index_id": "oni", "nominal_period": dt.date(1997, 12, 1), "value": 2.37},
    ])
    inside_its_label = pd.DataFrame({"date_key": pd.to_datetime(["1997-12-15"])})
    joined = attach_teleconnections(inside_its_label, _all_indices(rows))

    assert joined["oni"].iloc[0] == 2.24, "a value from its own label leaked"
    assert publication_date(oni, dt.date(1997, 10, 1)) == dt.date(1997, 12, 15)
    assert publication_date(oni, dt.date(1997, 11, 1)) == dt.date(1998, 1, 15)
    assert publication_date(oni, dt.date(1997, 12, 1)) == dt.date(1998, 2, 15)


# ---------------------------------------------------------------------------
# Vintages and revisions
# ---------------------------------------------------------------------------


def test_a_revision_does_not_overwrite_what_an_earlier_date_could_see():
    """A 2029 restatement of 2024 must not reach a row scored in 2024.

    NOAA restates ENSO history when the base period shifts, every five years.
    A model scoring a day in 2024 has to read the number that stood then.
    """
    rows = vintage_rows([
        {"index_id": "oni", "nominal_period": dt.date(2024, 1, 1), "value": 1.0,
         "publication_date": dt.date(2024, 3, 15),
         "vintage_at": pd.Timestamp("2024-03-15", tz="UTC")},
        {"index_id": "oni", "nominal_period": dt.date(2024, 1, 1), "value": 9.0,
         "publication_date": dt.date(2029, 1, 10),
         "vintage_at": pd.Timestamp("2029-01-10", tz="UTC")},
    ])
    probe = pd.DataFrame({"date_key": pd.to_datetime(["2024-06-01", "2029-06-01"])})
    joined = attach_teleconnections(probe, _all_indices(rows))

    assert joined["oni"].iloc[0] == 1.0, "a future revision reached a past row"
    assert joined["oni"].iloc[1] == 9.0


def test_a_late_revision_of_an_old_period_is_not_the_current_value():
    """The bug a `merge_asof` on publication date alone would have.

    A restatement of an old month has a *late* publication date and an *old*
    nominal period. Taking the most recently published row would answer "what
    is the ENSO state now" with a correction to a five-year-old month. The
    walk in `teleconnection_steps` holds the newest *period*, not the newest
    publication.
    """
    rows = vintage_rows([
        {"index_id": "oni", "nominal_period": dt.date(2024, 1, 1), "value": 1.0,
         "publication_date": dt.date(2024, 3, 15),
         "vintage_at": pd.Timestamp("2024-03-15", tz="UTC")},
        {"index_id": "oni", "nominal_period": dt.date(2028, 6, 1), "value": 2.0,
         "publication_date": dt.date(2028, 8, 15),
         "vintage_at": pd.Timestamp("2028-08-15", tz="UTC")},
        # The restatement, published last and about the oldest period.
        {"index_id": "oni", "nominal_period": dt.date(2024, 1, 1), "value": 9.0,
         "publication_date": dt.date(2029, 1, 10),
         "vintage_at": pd.Timestamp("2029-01-10", tz="UTC")},
    ])
    probe = pd.DataFrame({"date_key": pd.to_datetime(["2029-06-01"])})
    joined = attach_teleconnections(probe, _all_indices(rows))

    assert joined["oni"].iloc[0] == 2.0, (
        "a restatement of an old month became the current index value"
    )


def test_an_unchanged_rerun_writes_nothing():
    """Append-on-change. A daily job over a 76-year series must not grow it."""
    existing = vintage_rows([
        {"index_id": "nao", "nominal_period": dt.date(2020, 1, 1), "value": 0.5},
        {"index_id": "nao", "nominal_period": dt.date(2020, 2, 1), "value": -0.25},
    ])
    fresh = existing.drop(columns=["vintage_at", "publication_date"])
    fresh = vintage_rows(fresh.to_dict("records"))

    assert changed_rows(fresh, existing).empty


def test_a_changed_value_is_written_and_a_rounding_wobble_is_not():
    """The tolerance exists so representation noise is not filed as history."""
    existing = vintage_rows([
        {"index_id": "nao", "nominal_period": dt.date(2020, 1, 1), "value": 0.5},
        {"index_id": "nao", "nominal_period": dt.date(2020, 2, 1), "value": -0.25},
    ])
    fresh = vintage_rows([
        {"index_id": "nao", "nominal_period": dt.date(2020, 1, 1), "value": 0.5 + 1e-9},
        {"index_id": "nao", "nominal_period": dt.date(2020, 2, 1), "value": -0.30},
    ])
    changed = changed_rows(fresh, existing)

    assert list(changed["nominal_period"]) == [dt.date(2020, 2, 1)]


def test_latest_vintage_takes_the_newest_row_per_period():
    rows = vintage_rows([
        {"index_id": "nao", "nominal_period": dt.date(2020, 1, 1), "value": 1.0,
         "vintage_at": pd.Timestamp("2024-01-01", tz="UTC")},
        {"index_id": "nao", "nominal_period": dt.date(2020, 1, 1), "value": 2.0,
         "vintage_at": pd.Timestamp("2025-01-01", tz="UTC")},
    ])
    assert latest_vintage(rows)["value"].tolist() == [2.0]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_the_seasonal_parser_dates_a_season_by_its_centre():
    body = " SEAS  YR   TOTAL   ANOM\n  DJF 1998  26.5  2.22\n  NDJ 1997  26.9  2.37\n"
    parsed = parse_cpc_seasonal(REGISTRY["oni"], body).set_index("nominal_period")

    assert parsed.loc[dt.date(1998, 1, 1), "value"] == 2.22
    assert parsed.loc[dt.date(1997, 12, 1), "value"] == 2.37


def test_the_seasonal_parser_ignores_the_header_and_junk():
    body = " SEAS  YR   TOTAL   ANOM\nnot a row\n  JJA 2020  27.0  -0.4\n"
    assert len(parse_cpc_seasonal(REGISTRY["oni"], body)) == 1


def test_the_psl_parser_drops_the_missing_sentinel():
    """-9999 for months that have not happened.

    Carried through, the DMI would read as a catastrophic negative dipole for
    every remaining month of the current year -- a huge, confident, entirely
    fictional feature value.
    """
    body = (
        " 1870 2026\n"
        "2026     0.123     0.529     0.285     0.279     0.146 -9999.000 "
        "-9999.000 -9999.000 -9999.000 -9999.000 -9999.000 -9999.000\n"
    )
    parsed = parse_psl_monthly(REGISTRY["dmi"], body)

    assert len(parsed) == 5
    assert parsed["value"].min() > -1.0
    assert parsed["nominal_period"].max() == dt.date(2026, 5, 1)


def test_the_monthly_parser_reads_year_month_value():
    body = " 1950    1    0.9200\n 1950    2    0.4000\n"
    parsed = parse_cpc_monthly(REGISTRY["nao"], body)

    assert list(parsed["value"]) == [0.92, 0.40]
    assert list(parsed["nominal_period"]) == [dt.date(1950, 1, 1), dt.date(1950, 2, 1)]


def test_an_empty_parse_is_an_error_and_not_an_empty_frame():
    """A silently empty feed would be reported by the ablation as "no help"."""
    with pytest.raises(TeleconnectionError, match="parsed no values"):
        parse_cpc_monthly(REGISTRY["nao"], "nothing parseable here\n")


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_every_registry_entry_is_usable():
    assert {index.id for index in load_registry()} == set(TELECONNECTION_FEATURES)
    for index in load_registry():
        assert index.publication_lag_days > 0
        assert index.description


def test_a_negative_lag_is_refused():
    """A negative lag is a value readable before it exists."""
    with pytest.raises(TeleconnectionError, match="read the future"):
        Index(
            id="bad", name="", driver="", url="", format="cpc_monthly",
            covers_months=1, centred=False, publication_lag_days=-1,
            description="",
        )


def test_a_centred_window_must_have_a_middle():
    with pytest.raises(TeleconnectionError, match="odd number of months"):
        Index(
            id="bad", name="", driver="", url="", format="cpc_seasonal",
            covers_months=2, centred=True, publication_lag_days=15,
            description="",
        )


def test_the_seasons_are_the_twelve_overlapping_triples():
    assert len(SEASONS) == 12
    assert SEASONS[0] == "DJF" and SEASONS[-1] == "NDJ"


# ---------------------------------------------------------------------------
# The feature list
# ---------------------------------------------------------------------------


def test_the_indices_are_off_by_default():
    """The shipped model's inputs must not change before the ablation says so.

    Switching them on here would invalidate the committed artefact whose
    feature list the model card asserts, and would do it silently.
    """
    assert not set(TELECONNECTION_FEATURES) & set(feature_columns())
    assert set(TELECONNECTION_FEATURES) <= set(feature_columns(teleconnections=True))
    assert len(feature_columns(teleconnections=True)) == len(feature_columns()) + 4


def test_a_row_before_any_publication_gets_a_null_not_a_backfill():
    """Filling would assert knowledge that did not exist."""
    rows = vintage_rows([
        {"index_id": "nao", "nominal_period": dt.date(2020, 1, 1), "value": 1.0},
    ])
    probe = pd.DataFrame({"date_key": pd.to_datetime(["1990-01-01"])})
    joined = attach_teleconnections(probe, _all_indices(rows))

    assert pd.isna(joined["nao"].iloc[0])


def test_the_join_does_not_depend_on_row_order():
    """Rows arrive grouped by city, so the date column is not monotonic.

    An as-of join written against a sorted frame and handed an unsorted one
    fails quietly, misaligning values by city rather than raising.
    """
    rows = vintage_rows([
        {"index_id": "nao", "nominal_period": dt.date(2020, 1, 1), "value": 1.0},
        {"index_id": "nao", "nominal_period": dt.date(2020, 6, 1), "value": 2.0},
    ])
    dates = pd.to_datetime(["2021-01-01", "2020-03-01", "2020-09-01"])
    ordered = attach_teleconnections(
        pd.DataFrame({"date_key": dates.sort_values()}), _all_indices(rows)
    )
    shuffled = attach_teleconnections(
        pd.DataFrame({"date_key": dates}), _all_indices(rows)
    )

    lookup = dict(zip(ordered["date_key"], ordered["nao"]))
    for date, value in zip(shuffled["date_key"], shuffled["nao"]):
        assert value == lookup[date]


def test_the_steps_are_one_per_publication_date():
    rows = vintage_rows([
        {"index_id": "nao", "nominal_period": dt.date(2020, 1, 1), "value": 1.0},
        {"index_id": "nao", "nominal_period": dt.date(2020, 2, 1), "value": 2.0},
    ])
    steps = teleconnection_steps(rows)["nao"]

    assert steps["publication_date"].is_monotonic_increasing
    assert not steps["publication_date"].duplicated().any()


# ---------------------------------------------------------------------------
# The ablation, and the decision it is allowed to make
# ---------------------------------------------------------------------------

METRICS = Path(__file__).resolve().parent.parent / "machine_learning" / "artifacts" / "metrics.json"


@pytest.fixture(scope="module")
def recorded():
    """The committed ablation block, or skip."""
    import json

    if not METRICS.exists():
        pytest.skip("metrics.json not written yet")
    payload = json.loads(METRICS.read_text())
    if "teleconnections" not in payload:
        pytest.skip("no teleconnections block; run evaluate.py --teleconnections")
    return payload["teleconnections"]


def test_the_ablation_holds_the_hyperparameters_fixed(recorded):
    """Both arms must differ in columns and in nothing else.

    Re-tuning each arm would compare a tuned model with indices against a tuned
    model without, which is a fair comparison of two pipelines and useless for
    the question asked: what are four columns worth.
    """
    assert recorded["hyperparameters_are_fixed"] is True
    # `n_estimators` is decided by early stopping per arm and must not be
    # carried over, or the arm with more columns inherits a round count chosen
    # for the arm with fewer.
    assert "n_estimators" not in recorded["hyperparameters"]
    assert recorded["arms"]["with"]["feature_count"] == (
        recorded["arms"]["without"]["feature_count"] + len(recorded["block"])
    )


def test_both_arms_were_scored_on_the_same_rows(recorded):
    """One population, two column lists.

    Two reads could diverge by a row, and a difference in denominator would
    read as a difference in score.
    """
    for fold in ("validation", "test"):
        assert (
            recorded["arms"]["with"][fold]["rows"]
            == recorded["arms"]["without"][fold]["rows"]
            == recorded["rows"][fold]
        )
        assert (
            recorded["arms"]["with"][fold]["base_rate"]
            == recorded["arms"]["without"][fold]["base_rate"]
        )


def test_the_verdict_is_decided_on_validation(recorded):
    """Not on test, however much test likes them.

    This is the fifth time in this phase that something scores better on test
    than on validation, and the rule has not changed: shipping on a test
    reading is feature selection with test labels.
    """
    verdict = recorded["verdict"]
    assert verdict["decided_on"] == "validation"
    assert verdict["ships"] == (verdict["validation_delta_pr_auc"] > 0)


def test_the_shipped_feature_set_matches_the_verdict(recorded):
    """If the indices do not ship, they must be off by default, and vice versa.

    The one assertion that ties the measurement to the code. A verdict of "they
    do not ship" beside a default feature list that includes them would be a
    documented decision the pipeline ignores.
    """
    on_by_default = set(TELECONNECTION_FEATURES) <= set(feature_columns())
    assert on_by_default == recorded["verdict"]["ships"], (
        f"the ablation says ships={recorded['verdict']['ships']} but "
        f"feature_columns() {'includes' if on_by_default else 'excludes'} them"
    )


def test_the_per_city_deltas_agree_with_the_arms(recorded):
    """The delta column is arithmetic, and arithmetic is worth checking once."""
    for fold in ("validation", "test"):
        for city in recorded["by_city"][fold].values():
            assert city["delta_pr_auc"] == pytest.approx(
                city["with"]["pr_auc"] - city["without"]["pr_auc"]
            )


def test_the_ablation_reports_every_scorable_city(recorded):
    """Per city, because the claim was about specific cities.

    The indices were proposed on the grounds that they would help the tropical
    cities where the model is weakest. A pooled PR-AUC over eleven cities with
    different base rates cannot answer that, so the block carries both.
    """
    for fold in ("validation", "test"):
        assert len(recorded["by_city"][fold]) >= 8


def test_shipped_params_drops_the_round_count(tmp_path):
    """`n_estimators` comes from early stopping, not from the committed model."""
    import json

    from machine_learning.evaluate import shipped_params

    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({
        "model": {
            "recommended_variant": "unweighted",
            "artifacts": {"unweighted": {"hyperparameters": {
                "learning_rate": 0.1, "max_depth": 6, "min_child_weight": 10,
                "scale_pos_weight": 1.0, "n_estimators": 126, "seed": 42,
            }}},
        }
    }))
    params = shipped_params(path)

    assert "n_estimators" not in params and "seed" not in params
    assert params["max_depth"] == 6


def test_merge_block_leaves_the_rest_of_the_file_alone(tmp_path):
    """A partial refresh of metrics.json would be worse than a stale one.

    The `evaluation` block is written by evaluate.py and the `model` block by
    train.py. Refreshing one against a widened warehouse while the other stays
    put leaves a file whose two halves describe different city sets, and
    nothing about it looks stale.
    """
    import json

    from machine_learning.evaluate import merge_block

    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"model": {"keep": 1}, "evaluation": {"keep": 2}}))
    merge_block("teleconnections", {"verdict": "no"}, path)
    payload = json.loads(path.read_text())

    assert payload["model"] == {"keep": 1}
    assert payload["evaluation"] == {"keep": 2}
    assert payload["teleconnections"]["verdict"] == "no"
    assert "recorded_at" in payload["teleconnections"]
