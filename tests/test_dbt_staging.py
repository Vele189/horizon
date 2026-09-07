"""Tests for the silver staging models.

The dbt tests already assert the data; these assert the things dbt cannot,
plus one integration run so a broken model fails `pytest` and not only
`dbt build`.

Worth stating plainly, because it is the reason two of the dbt tests exist:
uniqueness proves *one* row survives per key and says nothing about *which*.
Reversing the sort order to `ingested_at asc` leaves every uniqueness test
green and silently serves the oldest copy of every observation. Verified by
mutation — with the order flipped, `unique_combination_of_columns`,
`unique_id` and `assert_staging_loses_no_observation` all pass, and only
`assert_daily_keeps_the_newest_ingest` fails, on 731 rows.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dbt_analytics.dbt_env import dbt_environment  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DBT_DIR = REPO_ROOT / "dbt_analytics"
MACRO = DBT_DIR / "macros" / "deduplicate_observations.sql"
STAGING = DBT_DIR / "models" / "staging"
GRAINS = ("daily", "hourly")


@pytest.fixture(scope="module")
def macro() -> str:
    return MACRO.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def staging_yml() -> dict:
    return yaml.safe_load((STAGING / "_staging.yml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The deduplication itself
# ---------------------------------------------------------------------------


def test_it_does_not_use_qualify(macro: str) -> None:
    """QUALIFY is Snowflake, BigQuery and DuckDB. Postgres has no such clause."""
    assert "qualify" not in macro.lower().replace("qualify clause", "")


def test_it_uses_a_row_number_subquery_filtered_to_rank_one(macro: str) -> None:
    assert "row_number() over" in macro
    assert re.search(r"where\s+_dedup_rank\s*=\s*1", macro)


def test_it_partitions_by_the_natural_key(macro: str) -> None:
    assert re.search(
        r"partition by\s+city_id,\s*observation_time", macro
    ), "partitioning by anything else would collapse rows that are not duplicates"


def test_it_orders_newest_ingest_first(macro: str) -> None:
    assert re.search(r"order by\s+ingested_at desc", macro)


def test_it_breaks_ties_deterministically(macro: str) -> None:
    """The loader stamps one ingested_at per run, so ties are reachable."""
    assert re.search(r"order by\s+ingested_at desc,\s*id desc", macro)


def test_the_rank_column_does_not_leak(macro: str) -> None:
    """A stray _dedup_rank in silver would propagate into every mart."""
    final_select = macro.rsplit("from ranked", 1)[0].rsplit("select", 1)[1]
    assert "_dedup_rank" not in final_select


def test_columns_are_taken_from_the_relation_not_a_list(macro: str) -> None:
    """Bronze has already lost a column once; a hand-maintained list would rot."""
    assert "adapter.get_columns_in_relation" in macro


def test_introspection_is_guarded_by_execute(macro: str) -> None:
    """Unguarded, the column list is silently empty during dbt's parse pass."""
    assert "{%- if execute -%}" in macro or "{% if execute %}" in macro


# ---------------------------------------------------------------------------
# Both grains, and their tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grain", GRAINS)
def test_a_model_exists_for_each_grain(grain: str) -> None:
    model = STAGING / f"stg_observations_{grain}.sql"
    assert model.is_file()
    body = model.read_text(encoding="utf-8")
    assert "deduplicate_observations" in body
    assert f"observations_{grain}" in body


@pytest.mark.parametrize("grain", GRAINS)
def test_each_model_is_uniqueness_tested_on_the_natural_key(
    staging_yml, grain: str
) -> None:
    model = next(
        m for m in staging_yml["models"] if m["name"] == f"stg_observations_{grain}"
    )
    combos = [
        t["unique_combination_of_columns"]["combination_of_columns"]
        for t in model["tests"]
        if "unique_combination_of_columns" in t
    ]
    assert ["city_id", "observation_time"] in combos


@pytest.mark.parametrize("grain", GRAINS)
def test_each_model_records_its_duplicate_count(staging_yml, grain: str) -> None:
    """The ticket asks for before and after in the description, not in a commit."""
    model = next(
        m for m in staging_yml["models"] if m["name"] == f"stg_observations_{grain}"
    )
    description = model["description"]
    assert "Duplicates removed" in description
    numbers = re.findall(r"[\d,]{4,}", description)
    assert len(numbers) >= 2, "before and after should both be stated"


@pytest.mark.parametrize("grain", GRAINS)
def test_each_grain_asserts_the_newest_copy_survived(grain: str) -> None:
    test_file = DBT_DIR / "tests" / f"assert_{grain}_keeps_the_newest_ingest.sql"
    assert test_file.is_file()
    body = test_file.read_text(encoding="utf-8")
    assert "max(ingested_at)" in body
    assert f"stg_observations_{grain}" in body


def test_a_narrowed_partition_would_be_caught() -> None:
    """Partitioning by city_id alone yields a unique result that is also wrong."""
    body = (DBT_DIR / "tests" / "assert_staging_loses_no_observation.sql").read_text(
        encoding="utf-8"
    )
    assert "distinct city_id, observation_time" in body
    for grain in GRAINS:
        assert grain in body


# ---------------------------------------------------------------------------
# It actually builds
# ---------------------------------------------------------------------------


def test_the_staging_layer_builds_and_all_its_tests_pass(engine) -> None:
    """The integration check: a broken model fails pytest, not only dbt build."""
    pytest.importorskip("dbt.cli.main")
    if not (DBT_DIR / "profiles.yml").exists():
        pytest.skip("profiles.yml not generated; run dbt_env.py --write-profile")

    result = subprocess.run(
        [
            sys.executable, "-m", "dbt.cli.main", "build",
            "--project-dir", str(DBT_DIR),
            "--select", "path:models/staging",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, **dbt_environment()},
        check=False,
    )
    summary = re.search(r"Done\. PASS=(\d+) WARN=(\d+) ERROR=(\d+)", result.stdout)
    assert summary, result.stdout[-3000:]
    passed, warned, errored = (int(g) for g in summary.groups())
    assert errored == 0, result.stdout[-3000:]
    assert warned == 0, result.stdout[-3000:]
    assert passed >= 12, f"expected both models and their tests, got {passed}"
