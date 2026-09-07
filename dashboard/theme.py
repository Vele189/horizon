"""Every colour the dashboard uses, and the reasoning that fixed each one.

This module is the **only** place a hex value is written. A palette scattered
across four view modules drifts within a day: the map's "hot" stops matching
the matrix's "hot", and the two views stop being about the same thing. Views
import :data:`DIVERGING`, :data:`SURFACE` and the ink tokens from here and
never spell a colour themselves — a test walks the AST of every other module
under ``dashboard/`` and fails on any hex literal it finds.

The encoding
------------

Anomalies are **signed**: a city is colder than its climatology or warmer than
it, and zero is a meaningful middle rather than an axis end. That makes the
scale *diverging* — two hues away from a neutral midpoint — and rules out a
sequential ramp (which would claim cold and hot are the same thing seen weakly
and strongly) and a rainbow (which invents ordering where the hue wheel has
none).

Blue for cold, red for warm, grey between them. The direction is not a
preference: it is the convention every published temperature-anomaly figure
uses, and inverting it to be interesting would cost a reader more than it could
possibly buy.

How the steps were chosen
-------------------------

The poles are ColorBrewer's ``RdBu`` extremes — the diverging scheme climate
publication has standardised on, and the one ColorBrewer marks colourblind-safe
— taken as an OKLCH hue each (252.4° cold, 22.4° warm). Everything between them
was then *generated* rather than eyeballed:

* **Lightness** runs on an even ladder outward from the midpoint, so a step of
  colour means a step of anomaly. Both arms sit at the same lightness at the
  same distance from zero (0.426 against 0.427 at the poles), so +3σ and −3σ
  carry equal visual weight and neither sign looks more urgent than the other.
* **Chroma** rises 0.65 → 0.80 → 0.95 → 1.00 of the pole's, clipped to the
  largest value that stays inside sRGB at that lightness and hue.
* **The midpoint is grey**, never a hue. A hue at zero would read as a third
  category and give "normal" a temperature of its own.

The ladders were then searched — not adjusted by hand — for the combination
maximising the worst-case separation under simulated colour-vision deficiency,
subject to the steps staying evenly spaced. ``tests/test_dashboard.py`` recomputes
every one of those measurements from these constants and fails if a value drifts,
so the numbers quoted in the README cannot go stale.

What was measured, and what it means
------------------------------------

Under protanopia and deuteranopia (Machado 2009, severity 1.0), distances are
Euclidean in OKLab ×100:

===============================  ======  ======
                                  light    dark
===============================  ======  ======
Cold vs warm at equal magnitude    12.3    12.8
Any step vs the neutral midpoint   15.2    10.5
Adjacent steps                     10.8     9.9
Lightness-step evenness            1.23    1.40
===============================  ======  ======

The first row is the one that matters: it is whether a reader with the most
common form of colour blindness can tell *which way* an anomaly went. At 12.3
it is comfortably above the 8 that counts as separated, so the sign survives —
which is the whole claim a red-blue climate figure makes.

The one number below its floor
------------------------------

The palest step of each arm sits at 1.81:1 against the light surface, under the
2:1 floor that says a mark must be visible against empty background. A search
over lightness ladders showed the floor and even spacing cannot both be had:
clearing 2:1 costs a doubled first step, which would exaggerate small anomalies
and compress large ones — a worse chart than a pale swatch. Even spacing wins,
and the mitigation is structural rather than chromatic: heatmap cells tile the
plot with a surface gap between them, map points carry a surface ring, and
legend swatches carry a hairline border. Nothing relies on that step being
legible against bare paper. In dark mode the same step measures 2.30:1 and the
question does not arise.

There is no accent hue
----------------------

Chrome — links, focus rings, the selected nav item — wants an accent, and every
chromatic candidate tested collapsed into the ramp under simulation: a teal at
mid-lightness lands 1.0–6.0 from a blue step under deuteranopia, well inside the
distance that means "the same colour". Nine steps of blue and red leave no room
for a tenth hue that stays distinct from all of them. So the accent is ink, and
the palette keeps its promise that a colour on this page means a number.
"""

from __future__ import annotations

from typing import Final, Literal, Mapping, Sequence

__all__ = [
    "COLD_HUE_DEGREES",
    "DIVERGING",
    "Mode",
    "NEUTRAL",
    "SURFACE",
    "WARM_HUE_DEGREES",
    "chrome",
    "diverging_scale",
    "resolve_mode",
]

Mode = Literal["light", "dark"]

# The OKLCH hues the arms were generated on, kept because the tests regenerate
# the ramps from them rather than trusting the hex values below.
COLD_HUE_DEGREES: Final[float] = 252.4
WARM_HUE_DEGREES: Final[float] = 22.4

# The nine steps, cold pole first. Odd by design: an even ramp has no step for
# "no anomaly" and forces zero onto one side of the boundary or the other.
DIVERGING: Final[Mapping[Mode, tuple[str, ...]]] = {
    "light": (
        "#004f92",  # coldest
        "#3574b7",
        "#669ad5",
        "#97c2f3",
        "#f0efec",  # neutral: no meaningful departure
        "#ffa09c",
        "#e36d6c",
        "#c3383f",
        "#97001d",  # warmest
    ),
    "dark": (
        "#90c4ff",
        "#609fe6",
        "#497cb5",
        "#335a86",
        "#2e2e2c",
        "#8f3738",
        "#c14e4f",
        "#f56667",
        "#ffa29e",
    ),
}

# Index of the neutral step. Views that need "the cold arm" or "the warm arm"
# slice on this rather than hardcoding 4.
NEUTRAL_INDEX: Final[int] = 4

NEUTRAL: Final[Mapping[Mode, str]] = {
    mode: steps[NEUTRAL_INDEX] for mode, steps in DIVERGING.items()
}

# The surface the ramp was validated against. Contrast and lightness are
# meaningless without it, which is why it lives beside the ramp and why
# .streamlit/config.toml is checked against it by a test rather than
# maintained in parallel.
SURFACE: Final[Mapping[Mode, str]] = {"light": "#fcfcfb", "dark": "#1a1a19"}

# Chrome. Deliberately achromatic — see the module docstring.
_CHROME: Final[Mapping[Mode, Mapping[str, str]]] = {
    "light": {
        "surface": "#fcfcfb",
        "plane": "#f9f9f7",
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "ink_muted": "#898781",
        "gridline": "#e1e0d9",
        "axis": "#c3c2b7",
    },
    "dark": {
        "surface": "#1a1a19",
        "plane": "#0d0d0d",
        "ink": "#ffffff",
        "ink_secondary": "#c3c2b7",
        "ink_muted": "#898781",
        "gridline": "#2c2c2a",
        "axis": "#383835",
    },
}


def resolve_mode(reported: str | None) -> Mode:
    """Map Streamlit's reported theme onto a mode this module has colours for.

    ``st.context.theme.type`` is ``None`` on the first run of a session and can
    briefly report the previous value while a theme change propagates. Neither
    is worth a branch in a view: the ramp is validated in both modes, so the
    cost of guessing light and being wrong for one rerun is a slightly
    mismatched frame, not an unreadable one.
    """
    return "dark" if reported == "dark" else "light"


def chrome(mode: Mode) -> Mapping[str, str]:
    """The non-data colours — surfaces, ink, gridlines — for one mode."""
    return _CHROME[mode]


def diverging_scale(mode: Mode) -> tuple[str, ...]:
    """The nine anomaly steps, cold pole first."""
    return DIVERGING[mode]


def plotly_colorscale(mode: Mode) -> list[tuple[float, str]]:
    """The ramp as Plotly wants it: positions in [0, 1] paired with colours.

    Built from the same tuple the swatches are, so a view that plots and a
    legend that explains cannot disagree.
    """
    steps = DIVERGING[mode]
    last = len(steps) - 1
    return [(index / last, colour) for index, colour in enumerate(steps)]


def diverging_legend_html(
    mode: Mode,
    *,
    cold_label: str = "colder than normal",
    warm_label: str = "warmer than normal",
) -> str:
    """A horizontal key for the anomaly scale.

    Swatches carry a hairline border in the muted ink. That border is not
    decoration — it is what keeps the palest step of each arm legible against
    the page, which it is not on colour alone (see the module docstring).
    """
    tokens = chrome(mode)
    swatches = "".join(
        f'<span style="flex:1;height:14px;background:{colour};'
        f'border:1px solid {tokens["axis"]};"></span>'
        for colour in DIVERGING[mode]
    )
    return (
        f'<div style="margin:0.25rem 0 0.75rem 0;">'
        f'<div style="display:flex;gap:2px;">{swatches}</div>'
        f'<div style="display:flex;justify-content:space-between;'
        f'font-size:0.75rem;color:{tokens["ink_muted"]};margin-top:0.25rem;">'
        f"<span>{cold_label}</span><span>no departure</span>"
        f"<span>{warm_label}</span></div></div>"
    )


def _assert_shape(steps: Sequence[str]) -> None:
    """Cheap structural guard, run at import so a bad edit fails immediately."""
    if len(steps) % 2 == 0:
        raise ValueError("a diverging ramp needs an odd step count for its midpoint")
    if len(steps) != len(set(steps)):
        raise ValueError("a repeated step means two anomaly magnitudes share a colour")


for _mode, _steps in DIVERGING.items():
    _assert_shape(_steps)
    if len(_steps) != len(DIVERGING["light"]):
        raise ValueError("light and dark ramps must have the same number of steps")
