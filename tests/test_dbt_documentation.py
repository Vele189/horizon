"""Tests for documentation coverage and lineage shape.

An undocumented model is a model a reviewer cannot assess, so coverage is
asserted rather than reviewed. It is the kind of thing that is complete on the
day it is written and 80% complete a month later.

The lineage tests exist because the DAG had a real defect that only became
visible when it was drawn: `int_climatology_contributions` selected from
`fact_weather_observations` and `dim_date`, making the lineage run
staging -> marts -> intermediate -> marts. It built fine and every test passed;
it was simply not a layering anyone could follow.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
DBT_DIR = REPO_ROOT / "dbt_analytics"
MANIFEST = DBT_DIR / "target" / "manifest.json"
CATALOG = DBT_DIR / "target" / "catalog.json"
IMAGES = REPO_ROOT / "docs" / "images"


@pytest.fixture(scope="module")
def manifest() -> dict:
    if not MANIFEST.exists():
        pytest.skip("run `dbt docs generate` first")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def catalog() -> dict:
    if not CATALOG.exists():
        pytest.skip("run `dbt docs generate` first")
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def buildable(manifest: dict) -> dict[str, dict]:
    return {
        key: node
        for key, node in manifest["nodes"].items()
        if node["resource_type"] in ("model", "seed")
    }


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def test_every_model_has_a_description(manifest) -> None:
    missing = [
        node["name"]
        for node in buildable(manifest).values()
        if not node.get("description", "").strip()
    ]
    assert not missing, missing


def test_every_source_table_has_a_description(manifest) -> None:
    missing = [
        f"{node['source_name']}.{node['name']}"
        for node in manifest["sources"].values()
        if not node.get("description", "").strip()
    ]
    assert not missing, missing


def test_every_gold_column_is_documented(manifest, catalog) -> None:
    """The checklist's requirement, per column rather than per model."""
    missing: list[str] = []
    for key, node in buildable(manifest).items():
        if node.get("schema") != "gold_marts":
            continue
        documented = {
            c for c, v in node.get("columns", {}).items()
            if v.get("description", "").strip()
        }
        actual = set(catalog["nodes"].get(key, {}).get("columns", {}))
        missing += [f"{node['name']}.{c}" for c in sorted(actual - documented)]
    assert not missing, missing


def test_every_column_anywhere_is_documented(manifest, catalog) -> None:
    """Stronger than the checklist asks, and cheap once doc blocks exist."""
    missing: list[str] = []
    for key, node in buildable(manifest).items():
        documented = {
            c for c, v in node.get("columns", {}).items()
            if v.get("description", "").strip()
        }
        actual = set(catalog["nodes"].get(key, {}).get("columns", {}))
        missing += [f"{node['name']}.{c}" for c in sorted(actual - documented)]
    assert not missing, missing


def test_shared_descriptions_come_from_doc_blocks(manifest) -> None:
    """`city_id` appears in nine models; nine copies is nine chances to drift."""
    docs = (DBT_DIR / "models" / "_docs.md").read_text(encoding="utf-8")
    assert "{% docs col_city_id %}" in docs

    descriptions = [
        node["columns"]["city_id"]["description"]
        for node in buildable(manifest).values()
        if "city_id" in node.get("columns", {})
    ]
    assert len(descriptions) >= 8
    assert len(set(descriptions)) == 1, "city_id is described differently somewhere"


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------


LAYER_ORDER = {"source": 0, "seed": 0, "staging": 1, "intermediate": 2, "marts": 3}


def layer_of(node: dict) -> str:
    if node["resource_type"] in ("source", "seed"):
        return node["resource_type"]
    return node["fqn"][1]


def test_the_dag_only_flows_forward(manifest) -> None:
    """No model may depend on a later layer.

    This is not pedantry: it was violated. `int_climatology_contributions`
    selected from `fact_weather_observations` and `dim_date`, so the lineage
    ran staging -> marts -> intermediate -> marts. Everything built and every test
    passed; the graph was simply unreadable, which is the whole artefact this
    ticket is about.
    """
    nodes = {
        **{k: v for k, v in manifest["nodes"].items()
           if v["resource_type"] in ("model", "seed")},
        **manifest["sources"],
    }
    violations = []
    for child, parents in manifest["parent_map"].items():
        if child not in nodes:
            continue
        child_layer = LAYER_ORDER[layer_of(nodes[child])]
        for parent in parents:
            if parent not in nodes:
                continue
            parent_layer = LAYER_ORDER[layer_of(nodes[parent])]
            if parent_layer > child_layer:
                violations.append(
                    f"{nodes[parent]['name']} ({layer_of(nodes[parent])}) "
                    f"-> {nodes[child]['name']} ({layer_of(nodes[child])})"
                )
    assert not violations, violations


def test_the_intermediate_layer_reads_only_staging(manifest) -> None:
    nodes = {
        **{k: v for k, v in manifest["nodes"].items()
           if v["resource_type"] in ("model", "seed")},
        **manifest["sources"],
    }
    for child, parents in manifest["parent_map"].items():
        if child not in nodes or layer_of(nodes[child]) != "intermediate":
            continue
        for parent in parents:
            if parent in nodes:
                assert layer_of(nodes[parent]) in ("staging", "intermediate"), (
                    f"{nodes[child]['name']} reads {nodes[parent]['name']}"
                )


def test_every_model_is_reachable_from_a_source_or_seed(manifest) -> None:
    """An orphan model is one nothing feeds and nothing checks."""
    nodes = {
        k: v for k, v in manifest["nodes"].items()
        if v["resource_type"] == "model"
    }
    # Generated from generate_series rather than selected from anything:
    # a date spine derived from the facts would have a hole wherever
    # ingestion does, and a join against it would hide the hole.
    roots = {"int_climatology_window", "dim_date"}
    for key, node in nodes.items():
        if node["name"] in roots:
            continue
        assert manifest["parent_map"].get(key), f"{node['name']} has no parents"


def test_every_layer_is_present(manifest) -> None:
    layers = {
        layer_of(node)
        for node in list(buildable(manifest).values()) + list(manifest["sources"].values())
    }
    assert {"source", "seed", "staging", "intermediate", "marts"} <= layers


# ---------------------------------------------------------------------------
# The artefact
# ---------------------------------------------------------------------------


def test_the_lineage_image_exists_and_is_current(manifest) -> None:
    """Regenerated from the manifest, so it cannot drift from the project."""
    svg = IMAGES / "lineage.svg"
    png = IMAGES / "lineage.png"
    assert svg.is_file() and png.is_file()

    rendered = svg.read_text(encoding="utf-8")
    for node in buildable(manifest).values():
        assert node["name"] in rendered, f"{node['name']} missing from the diagram"
    for node in manifest["sources"].values():
        assert f"{node['source_name']}.{node['name']}" in rendered


def test_the_lineage_image_regenerates_identically() -> None:
    """The check that catches a diagram edited by hand or left stale."""
    svg = IMAGES / "lineage.svg"
    before = svg.read_text(encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "dbt_analytics/render_lineage.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert svg.read_text(encoding="utf-8") == before, (
        "docs/images/lineage.svg is stale; re-run render_lineage.py"
    )
