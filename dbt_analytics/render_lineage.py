"""Renders the dbt DAG as a layered SVG for the README.

dbt's own docs site draws a force-directed graph, which is fine to explore and
poor to read at a glance: the layers are the whole point of a medallion
architecture and a force layout does not show them. This lays the same graph
out left to right by layer, so source -> staging -> intermediate -> marts is the
shape of the picture rather than something to trace.

Generated from `target/manifest.json`, so it cannot drift from the project;
run `dbt docs generate` first. SVG rather than a screenshot because it stays
crisp at any size, diffs as text, and needs no browser to produce.

    python dbt_analytics/render_lineage.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
MANIFEST = PROJECT / "target" / "manifest.json"
OUTPUT = PROJECT.parent / "docs" / "images" / "lineage.svg"

LAYERS = ("source", "seed", "staging", "intermediate", "marts")
COLOUR = {
    "source": ("#8c6d3f", "#fdf6e8"),
    "seed": ("#6b7280", "#f3f4f6"),
    "staging": ("#2f6f4e", "#e8f5ee"),
    "intermediate": ("#7a5ea8", "#f1ecf9"),
    "marts": ("#1f4e79", "#e7f0f8"),
}
LABEL = {
    "source": "bronze (source)",
    "seed": "seed",
    "staging": "silver (staging)",
    "intermediate": "intermediate",
    "marts": "gold (marts)",
}

BOX_W, BOX_H, GAP_Y, GAP_X, PAD = 218, 40, 18, 108, 32
HEADER = 64


def layer_of(node: dict) -> str:
    if node["resource_type"] == "source":
        return "source"
    if node["resource_type"] == "seed":
        return "seed"
    return node["fqn"][1]


def build() -> str:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    nodes: dict[str, dict] = {}
    for key, node in {**manifest["nodes"], **manifest["sources"]}.items():
        if node["resource_type"] not in ("model", "seed", "source"):
            continue
        nodes[key] = {
            "name": node["name"]
            if node["resource_type"] != "source"
            else f"{node['source_name']}.{node['name']}",
            "layer": layer_of(node),
            "materialized": node.get("config", {}).get("materialized", ""),
        }

    edges = [
        (parent, child)
        for child, node in manifest["parent_map"].items()
        if child in nodes
        for parent in node
        if parent in nodes
    ]

    # Within a column, order by depth so intra-layer edges flow downward
    # rather than crossing: marts depend on marts, and an arbitrary order
    # turns that into a tangle.
    depth: dict[str, int] = {}

    def depth_of(key: str, seen: frozenset[str] = frozenset()) -> int:
        if key in depth:
            return depth[key]
        if key in seen:
            return 0
        parents = [p for p in manifest["parent_map"].get(key, []) if p in nodes]
        value = 0 if not parents else 1 + max(
            depth_of(p, seen | {key}) for p in parents
        )
        depth[key] = value
        return value

    columns = {layer: [] for layer in LAYERS}
    for key, node in sorted(
        nodes.items(), key=lambda kv: (depth_of(kv[0]), kv[1]["name"])
    ):
        columns[node["layer"]].append(key)
    columns = {layer: keys for layer, keys in columns.items() if keys}

    position: dict[str, tuple[float, float]] = {}
    tallest = max(len(keys) for keys in columns.values())
    height = HEADER + tallest * (BOX_H + GAP_Y) + PAD
    for index, (layer, keys) in enumerate(columns.items()):
        x = PAD + index * (BOX_W + GAP_X)
        block = len(keys) * (BOX_H + GAP_Y) - GAP_Y
        top = HEADER + (height - HEADER - PAD - block) / 2
        for row, key in enumerate(keys):
            position[key] = (x, top + row * (BOX_H + GAP_Y))
    width = PAD * 2 + len(columns) * BOX_W + (len(columns) - 1) * GAP_X

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height:.0f}" viewBox="0 0 {width} {height:.0f}" '
        'font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif">',
        '<style>text{dominant-baseline:middle}</style>',
        f'<rect width="{width}" height="{height:.0f}" fill="#ffffff"/>',
        '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M0,0 L10,5 L0,10 z" fill="#9aa4b2"/></marker></defs>',
    ]

    for index, layer in enumerate(columns):
        x = PAD + index * (BOX_W + GAP_X)
        stroke, _ = COLOUR[layer]
        out.append(
            f'<text x="{x + BOX_W / 2:.0f}" y="30" text-anchor="middle" '
            f'font-size="13" font-weight="600" fill="{stroke}">'
            f"{LABEL[layer]}</text>"
        )

    for parent, child in edges:
        if parent not in position or child not in position:
            continue
        px, py = position[parent]
        cx, cy = position[child]
        if px == cx:
            # Same column: marts depend on marts. Bow the edge out to the left
            # so it reads as a connection rather than an arrowhead stub on the
            # box edge.
            y1, y2 = py + BOX_H / 2, cy + BOX_H / 2
            bow = px - 26 - 10 * abs(y2 - y1) / (BOX_H + GAP_Y)
            out.append(
                f'<path d="M{px:.0f},{y1:.0f} C{bow:.0f},{y1:.0f} '
                f'{bow:.0f},{y2:.0f} {px - 8:.0f},{y2:.0f}" fill="none" '
                'stroke="#9aa4b2" stroke-width="1.1" marker-end="url(#a)" '
                'opacity="0.65" stroke-dasharray="3 2"/>'
            )
            continue
        x1, y1 = px + BOX_W, py + BOX_H / 2
        x2, y2 = cx, cy + BOX_H / 2
        mid = (x1 + x2) / 2
        out.append(
            f'<path d="M{x1:.0f},{y1:.0f} C{mid:.0f},{y1:.0f} {mid:.0f},{y2:.0f} '
            f'{x2 - 8:.0f},{y2:.0f}" fill="none" stroke="#9aa4b2" '
            'stroke-width="1.3" marker-end="url(#a)" opacity="0.8"/>'
        )

    for key, node in nodes.items():
        x, y = position[key]
        stroke, fill = COLOUR[node["layer"]]
        out.append(
            f'<rect x="{x:.0f}" y="{y:.0f}" width="{BOX_W}" height="{BOX_H}" rx="7" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>'
        )
        out.append(
            f'<text x="{x + 12:.0f}" y="{y + BOX_H / 2 - 6:.0f}" font-size="12.5" '
            f'font-weight="600" fill="#1a1a1a">{node["name"]}</text>'
        )
        detail = node["materialized"] or "source"
        out.append(
            f'<text x="{x + 12:.0f}" y="{y + BOX_H / 2 + 10:.0f}" font-size="10.5" '
            f'fill="#6b7280">{detail}</text>'
        )

    out.append("</svg>")
    return "\n".join(out)


def main() -> int:
    if not MANIFEST.exists():
        print("run `dbt docs generate` first", file=sys.stderr)
        return 1
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
