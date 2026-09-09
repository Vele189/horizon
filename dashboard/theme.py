"""Every colour the dashboard uses, and the reasoning that fixed each one.

This module is the **only** place a hex value is written. A palette scattered
across four view modules drifts within a day: the map's "hot" stops matching
the matrix's "hot", and the two views stop being about the same thing. Views
import :data:`DIVERGING`, :data:`SURFACE` and the ink tokens from here and
never spell a colour themselves. A test walks the AST of every other module
under ``dashboard/`` and fails on any hex literal it finds.

The encoding
------------

Anomalies are **signed**: a city is colder than its climatology or warmer than
it, and zero is a meaningful middle rather than an axis end. That makes the
scale *diverging* (two hues away from a neutral midpoint) and rules out a
sequential ramp (which would claim cold and hot are the same thing seen weakly
and strongly) and a rainbow (which invents ordering where the hue wheel has
none).

Blue for cold, red for warm, grey between them. The direction is not a
preference: it is the convention every published temperature-anomaly figure
uses, and inverting it to be interesting would cost a reader more than it could
possibly buy.

How the steps were chosen
-------------------------

The poles are ColorBrewer's ``RdBu`` extremes: the diverging scheme climate
publication has standardised on, and the one ColorBrewer marks colourblind-safe.
Each is taken as an OKLCH hue (252.4° cold, 22.4° warm). Everything between them
was then *generated* rather than eyeballed:

* **Lightness** runs on an even ladder outward from the midpoint, so a step of
  colour means a step of anomaly. Both arms sit at the same lightness at the
  same distance from zero (0.426 against 0.427 at the poles), so +3σ and -3σ
  carry equal visual weight and neither sign looks more urgent than the other.
* **Chroma** rises 0.65 -> 0.80 -> 0.95 -> 1.00 of the pole's, clipped to the
  largest value that stays inside sRGB at that lightness and hue.
* **The midpoint is grey**, never a hue. A hue at zero would read as a third
  category and give "normal" a temperature of its own.

The ladders were then searched, not adjusted by hand, for the combination
maximising the worst-case separation under simulated colour-vision deficiency,
subject to the steps staying evenly spaced. ``tests/test_dashboard.py`` recomputes
every one of those measurements from these constants and fails if a value drifts,
so the numbers quoted in ``docs/build-log.md`` cannot go stale.

What was measured, and what it means
------------------------------------

Under protanopia and deuteranopia (Machado 2009, severity 1.0), distances are
Euclidean in OKLab x100:

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
it is comfortably above the 8 that counts as separated, so the sign survives,
which is the whole claim a red-blue climate figure makes.

The one number below its floor
------------------------------

The palest step of each arm sits at 1.81:1 against the light surface, under the
2:1 floor that says a mark must be visible against empty background. A search
over lightness ladders showed the floor and even spacing cannot both be had:
clearing 2:1 costs a doubled first step, which would exaggerate small anomalies
and compress large ones, a worse chart than a pale swatch. Even spacing wins,
and the mitigation is structural rather than chromatic: heatmap cells tile the
plot with a surface gap between them, map points carry a surface ring, and
legend swatches carry a hairline border. Nothing relies on that step being
legible against bare paper. In dark mode the same step measures 2.30:1 and the
question does not arise.

Counting is not signing
-----------------------

A count (how many days in a year ran past the threshold) is a magnitude, and
it gets a **sequential** ramp instead: one hue, light to dark, on the same two
pole hues. Zero to twenty has a bottom and a top and no meaningful middle, so
painting it on two hues either side of a neutral would invent a direction the
number does not have. Only a *difference* between two counts is signed, and
only that goes back to the diverging scale.

The two counting ramps are checked the way the arms are: one hue, monotone
lightness, no adjacent pair closer than 0.06, and the end nearest the surface
still clearing 2:1. They are also checked against each other, because a
screenshot of the hot view and one of the cold view are the same picture
otherwise. They separate by 8.7 under protanopia.

A fourth ramp, and the collision it could not avoid
---------------------------------------------------

Risk Horizon needed a fifth colour job: a probability, unsigned, that is
neither hot nor cold. The model's target is an anomaly in *either* tail, so
painting it on the warm ramp would claim a direction the number does not carry.

By this point the hue circle is full. Under protanopia and deuteranopia the
usable hue space collapses toward a blue-yellow axis, and the anomaly red, the
anomaly blue and the emphasis green already occupy it. The risk ramp is violet
(hue 320°) and it is measured as **0.6** from the cold counting ramp under
protanopia: for a red-blind reader, violet minus its red *is* blue.

That collision is stated rather than designed away, because the alternative was
worse and the numbers say so. An achromatic ramp clears the cold ramp at 9.1
but sits **3.7** from the muted ink, and the muted ink is on the *same page*,
marking the ten cities the model does not score. A reader who cannot separate
"no prediction" from "low risk" in a single picture is worse off than one who
could confuse two ramps that never appear together and each carry their own
labelled legend.

The residual same-page risk is then removed entirely: unscored cities are drawn
with no fill at all rather than in grey, so the ramp only ever has to separate
from the surface.

One accent, for emphasis rather than identity
---------------------------------------------

Storm Dynamics needed to colour a scatter *by city*, and fifteen cities is more
identities than any palette carries. A search settles it rather than an
opinion: enumerating every triple of hues on a 15° grid, no three are
simultaneously separable under protanopia and deuteranopia at all pairs,
distinct from the muted ink the unselected points wear, **and** clear of the
blue and red the anomaly scale owns. The only triples that pass put a hue 18°
from the anomaly blue, which would make one colour mean two things across the
dashboard.

So that view does not encode identity in colour at all. One city is emphasised
at a time against a grey field, and the identity of all fifteen lives in a
sorted table, where position carries it. That needs exactly **one** accent, and
one is easy: green at hue 140°, 112° clear of both anomaly hues, and separated
from the context ink by 21.5 in light mode and 20.1 in dark, well past the 15
at which two colours stop being confusable.

Green appears nowhere else in the dashboard and encodes no measurement. It
means "the thing you selected", which is a property of the interface rather
than of the weather.

There is no accent hue
----------------------

Chrome (links, focus rings, the selected nav item) wants an accent, and every
chromatic candidate tested collapsed into the ramp under simulation: a teal at
mid-lightness lands 1.0-6.0 from a blue step under deuteranopia, well inside the
distance that means "the same colour". Nine steps of blue and red leave no room
for a tenth hue that stays distinct from all of them. So the accent is ink, and
the palette keeps its promise that a colour on this page means a number.
"""

from __future__ import annotations

import math
from html import escape
from typing import Final, Literal, Mapping, Sequence

__all__ = [
    "ANOMALY_BREAKS",
    "CALIBRATION",
    "CALIBRATION_TRUST_CEILING",
    "EMPHASIS",
    "RISK",
    "SEQUENTIAL",
    "SMALL_SCREEN_MAX_WIDTH_PX",
    "ANOMALY_Z_THRESHOLD",
    "COLD_HUE_DEGREES",
    "DIVERGING",
    "MARKER_MAX_PX",
    "MARKER_MIN_PX",
    "RARITY_YEARS_CAP",
    "RARITY_YEARS_FLOOR",
    "MARKER_Z_CAP",
    "Mode",
    "NEUTRAL",
    "SURFACE",
    "WARM_HUE_DEGREES",
    "anomaly_colour",
    "anomaly_key_html",
    "anomaly_step",
    "calibration_colours",
    "chrome",
    "daily_equivalent",
    "diverging_scale",
    "emphasis",
    "risk_scale",
    "map_chrome",
    "sequential_key_html",
    "sequential_scale",
    "marker_diameter",
    "resolve_mode",
    "rarity_diameter",
    "rarity_key_html",
    "size_key_html",
    "small_screen_notice_html",
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

# Chrome. Deliberately achromatic; see the module docstring.
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
    """The non-data colours (surfaces, ink, gridlines) for one mode."""
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
    decoration: it is what keeps the palest step of each arm legible against
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


# ---------------------------------------------------------------------------
# The map's surface, and what a number turns into on it
# ---------------------------------------------------------------------------
#
# The basemap is dark in both page themes, so the map uses the **dark** ramp
# whichever theme the page is in. The ramp is chosen by the surface it is
# painted on, not by the theme of the page around it. A light-mode ramp on a
# dark basemap would be validated against a background that is not there.
#
# Three of the four basemap tokens are chrome tokens already defined above.
# Only the land tone is new, and it is deliberately close to the ocean:
# measured at 6.7 apart, the coastline reads without the basemap competing
# with fifteen coloured points for attention.

MAP_MODE: Final[Mode] = "dark"

_MAP: Final[Mapping[str, str]] = {
    "ocean": _CHROME["dark"]["plane"],
    "land": "#1c1c1a",
    "coastline": _CHROME["dark"]["axis"],
    # Every filled marker carries a 2px ring in this. On a chart the ring is
    # drawn in the *surface* colour, but a map has no single surface: a point
    # may sit on ocean, on land, or across a coastline. So the ring is muted
    # ink instead, which clears both grounds by a wide margin (39.8 against
    # land, 46.4 against ocean).
    #
    # It is load-bearing rather than decorative. The neutral step is a dark
    # grey and so is the land: the fill alone separates by only 7.5, under the
    # 8 that counts as distinct. The ring is what makes a city reporting no
    # departure visible at all, which is why every marker has one and why it
    # never varies: a constant outline cannot be mistaken for an encoding.
    "ring": _CHROME["dark"]["ink_muted"],
}


def map_chrome() -> Mapping[str, str]:
    """Basemap tones: ocean, land, coastline, and the marker ring."""
    return _MAP


# The project's anomaly flag, from dbt's `anomaly_z_threshold` var. Repeated
# here because the *palette* has to know it: the breaks are placed so that
# "flagged as an anomaly" and "painted in one of the two outermost steps" are
# the same statement, and a test asserts they cannot come apart.
ANOMALY_Z_THRESHOLD: Final[float] = 2.5

# Where one step of colour becomes the next, in units of |Z|. Four breaks per
# arm, so with the neutral in the middle the nine steps are used exactly.
ANOMALY_BREAKS: Final[tuple[float, ...]] = (0.5, 1.5, ANOMALY_Z_THRESHOLD, 3.5)


def anomaly_step(z: float) -> int:
    """Index into the ramp for a signed Z-score.

    Binned rather than continuous. Nine steps that a reader can count against
    a key beat a smooth gradient they can only guess at, and the boundaries
    are the ones the warehouse already reasons in: the outermost two steps
    are exactly the rows `is_anomaly` is true for.
    """
    magnitude = abs(z)
    # Strictly greater, because dbt's flag is `abs(z) > threshold` and a Z of
    # exactly 2.5 is therefore *not* an anomaly. Binning at `>=` would paint
    # that one value in a flagged colour, and the map would disagree with the
    # warehouse on the single row where the question is live.
    distance = sum(1 for boundary in ANOMALY_BREAKS if magnitude > boundary)
    if distance == 0:
        return NEUTRAL_INDEX
    return NEUTRAL_INDEX + (distance if z > 0 else -distance)


def anomaly_colour(z: float, mode: Mode = MAP_MODE) -> str:
    """The colour a signed Z-score is painted."""
    return DIVERGING[mode][anomaly_step(z)]


# Marker sizing. Area is proportional to |Z| above a floor, so a diameter goes
# as its square root. Encoding magnitude by *radius* would quadruple the
# apparent size of a doubled anomaly.
MARKER_MIN_PX: Final[float] = 8.0
MARKER_MAX_PX: Final[float] = 40.0

# Where the scale tops out. The largest |Z| in thirty years of the marts is
# 5.09, so the cap costs nothing today and stops one freak day from shrinking
# every other city to a dot if a larger one ever lands.
MARKER_Z_CAP: Final[float] = 5.0


def marker_diameter(z: float) -> float:
    """Pixel diameter for a signed Z-score.

    The floor is not a compromise, it is the encoding: a city half a sigma from
    its own normal *is* nothing happening, and the map's job is to show where
    something is. Near-normal cities settling into equal small dots is the
    correct reading, and the exact number is in the tooltip either way.
    """
    magnitude = min(abs(z), MARKER_Z_CAP)
    span = MARKER_MAX_PX - MARKER_MIN_PX
    return MARKER_MIN_PX + span * (magnitude / MARKER_Z_CAP) ** 0.5


#: Where the rarity encoding tops out, in years.
#:
#: Fifty. The records behind the fit are about thirty years long, so a
#: fifty-year return period is already an extrapolation of most of the record
#: again; past it the fitted answers separate by hundreds of years on the
#: strength of a shape parameter whose bootstrap interval spans two orders of
#: magnitude at that reach. Capping the *channel* rather than the number means
#: the map stops distinguishing what it cannot distinguish, while the tooltip
#: still reports what the fit said.
RARITY_YEARS_CAP: Final[float] = 50.0

#: Where it starts. Below the tail threshold there is no fitted answer at all,
#: and the smallest exceedance sits near a tenth of a year.
RARITY_YEARS_FLOOR: Final[float] = 0.1


def rarity_diameter(years: float) -> float:
    """Pixel diameter for a return period, in years.

    Logarithmic, because return periods are: the interesting distances are
    one year to ten and ten to a hundred, and those are the same distance to a
    reader. On a linear channel every ordinary exceedance would collapse into
    the same dot while a single fifty-year day took the whole range -- which is
    the encoding failure the *sigma* channel avoids by capping, arrived at from
    the other direction.

    Shares `marker_diameter`'s floor and ceiling on purpose. The two encodings
    are alternatives for the same map, and a reader toggling between them is
    comparing shapes; if the pixel range moved as well, every city would appear
    to change size for a reason that had nothing to do with the data.
    """
    if not math.isfinite(years) or years <= 0:
        return MARKER_MIN_PX
    bounded = min(max(years, RARITY_YEARS_FLOOR), RARITY_YEARS_CAP)
    low, high = math.log(RARITY_YEARS_FLOOR), math.log(RARITY_YEARS_CAP)
    fraction = (math.log(bounded) - low) / (high - low)
    return MARKER_MIN_PX + (MARKER_MAX_PX - MARKER_MIN_PX) * fraction


def rarity_key_html(
    mode: Mode = MAP_MODE, *, samples: Sequence[float] = (0.5, 5.0, 50.0)
) -> str:
    """Reference circles for the rarity channel, labelled in years.

    The same key `size_key_html` draws for sigma, in the other unit, because a
    toggle that changes what size means and leaves the key alone is worse than
    no toggle: the reader has a legend that is now wrong and no reason to
    doubt it.
    """
    tokens = chrome(mode)
    ring = _MAP["ring"]
    largest = rarity_diameter(max(samples))
    circles = "".join(
        f'<div style="display:flex;flex-direction:column;align-items:center;'
        f'justify-content:flex-end;min-width:{largest + 8:.0f}px;">'
        f'<span style="width:{rarity_diameter(value):.0f}px;'
        f'height:{rarity_diameter(value):.0f}px;border-radius:50%;'
        f'border:2px solid {ring};background:{NEUTRAL[mode]};"></span>'
        f'<span style="font-size:0.7rem;color:{tokens["ink_muted"]};'
        f'margin-top:0.25rem;">1 in {value:g} yr</span></div>'
        for value in samples
    )
    return (
        f'<div style="display:flex;align-items:flex-end;gap:0.75rem;">'
        f"{circles}</div>"
    )


def anomaly_key_html(mode: Mode = MAP_MODE) -> str:
    """The colour key, labelled in the units it encodes.

    A diverging ramp with no numbers on it asks the reader to believe that
    darker means more. Labelling the breaks turns it into something they can
    read a value off.
    """
    tokens = chrome(mode)
    swatches = "".join(
        f'<span style="flex:1;height:14px;background:{colour};'
        f'border:1px solid {tokens["axis"]};"></span>'
        for colour in DIVERGING[mode]
    )
    labels = "".join(
        f'<span style="flex:1;text-align:right;transform:translateX(50%);">{value}</span>'
        for value in (
            *(f"-{break_}" for break_ in reversed(ANOMALY_BREAKS)),
            *(f"+{break_}" for break_ in ANOMALY_BREAKS),
            "",
        )
    )
    return (
        f'<div style="margin:0.25rem 0 0.5rem 0;">'
        f'<div style="display:flex;gap:2px;">{swatches}</div>'
        f'<div style="display:flex;gap:2px;font-size:0.7rem;'
        f'color:{tokens["ink_muted"]};margin-top:0.2rem;">{labels}</div>'
        f'<div style="display:flex;justify-content:space-between;'
        f'font-size:0.75rem;color:{tokens["ink_muted"]};margin-top:0.15rem;">'
        f"<span>colder than normal</span><span>Z-score</span>"
        f"<span>warmer than normal</span></div></div>"
    )


def size_key_html(mode: Mode = MAP_MODE, *, samples: Sequence[float] = (1.0, 2.5, 4.0)) -> str:
    """Reference circles, so size can be read rather than estimated.

    Plotly draws no key for a size channel. Without one, area is decoration:
    the reader can see that a point is bigger and not what bigger means.
    """
    tokens = chrome(mode)
    ring = _MAP["ring"]
    largest = marker_diameter(max(samples))
    circles = "".join(
        f'<div style="display:flex;flex-direction:column;align-items:center;'
        f'justify-content:flex-end;min-width:{largest + 8:.0f}px;">'
        f'<span style="width:{marker_diameter(value):.0f}px;'
        f'height:{marker_diameter(value):.0f}px;border-radius:50%;'
        f'border:2px solid {ring};background:{NEUTRAL[mode]};"></span>'
        f'<span style="font-size:0.7rem;color:{tokens["ink_muted"]};'
        f'margin-top:0.25rem;">|Z| {value:g}</span></div>'
        for value in samples
    )
    return (
        f'<div style="display:flex;gap:0.75rem;align-items:flex-end;'
        f'margin:0.25rem 0 0.5rem 0;">{circles}</div>'
    )


# ---------------------------------------------------------------------------
# Sequential ramps: for counting, not for signing
# ---------------------------------------------------------------------------
#
# A count of anomaly days is a **magnitude**. Zero to twenty has a bottom and a
# top and no meaningful middle, which makes it the wrong shape for the
# diverging ramp above: painting an unsigned count on two hues either side of a
# neutral invents a direction the number does not have, and puts the least
# interesting value (the middle of the range) in the most visually neutral
# place.
#
# So counting gets one hue, light to dark, on the same two pole hues the
# diverging scale uses. Hot days climb the warm ramp, cold days climb the cool
# one, and only their *difference*, which really is signed, goes back to the
# diverging scale.
#
# Five steps, generated on an even lightness ladder like the diverging arms and
# checked the same way: one hue throughout, lightness monotone, no two adjacent
# steps closer than 0.06, and the end nearest the surface still clearing 2:1
# against it. The two ramps also have to be tellable apart from each other,
# because a screenshot of the hot view and one of the cold view are the same
# picture otherwise. They separate by 8.7 under protanopia.

SEQUENTIAL: Final[Mapping[Mode, Mapping[str, tuple[str, ...]]]] = {
    "light": {
        "hot": ("#e59f9c", "#d67f7c", "#c75e5d", "#b63a3f", "#a30020"),
        "cold": ("#9ab7d9", "#799fca", "#5886ba", "#376eaa", "#0b569a"),
    },
    "dark": {
        "hot": ("#783c3b", "#9e4c4b", "#c55d5c", "#ef6e6d", "#ff9591"),
        "cold": ("#385270", "#486b94", "#5785b9", "#67a0e0", "#80bcff"),
    },
}


def sequential_scale(direction: str, mode: Mode) -> tuple[str, ...]:
    """The five counting steps for one direction, lowest count first."""
    return SEQUENTIAL[mode][direction]


def sequential_key_html(direction: str, mode: Mode, bounds: Sequence[str]) -> str:
    """A key for a counting ramp, labelled with the bucket each step holds."""
    tokens = chrome(mode)
    steps = SEQUENTIAL[mode][direction]
    cells = "".join(
        f'<div style="flex:1;text-align:center;">'
        f'<div style="height:14px;background:{colour};'
        f'border:1px solid {tokens["axis"]};"></div>'
        f'<div style="font-size:0.7rem;color:{tokens["ink_muted"]};'
        f'margin-top:0.2rem;">{label}</div></div>'
        for colour, label in zip(steps, bounds)
    )
    return (
        f'<div style="display:flex;gap:2px;margin:0.25rem 0 0.5rem 0;">{cells}</div>'
    )


# The emphasis accent. Deliberately not on the ramp's hues, and deliberately at
# a lightness well away from the context ink. Hue alone does not separate two
# colours that sit at the same lightness once a simulation flattens the chroma.
EMPHASIS: Final[Mapping[Mode, str]] = {"light": "#145700", "dark": "#6bd852"}


def emphasis(mode: Mode) -> str:
    """The single colour meaning "the city you picked"."""
    return EMPHASIS[mode]


# The risk ramp. Violet, five steps, same ladder as the counting ramps.
RISK: Final[Mapping[Mode, tuple[str, ...]]] = {
    "light": ("#cea2d8", "#bc83c9", "#a963b9", "#9642a8", "#821698"),
    "dark": ("#663f6f", "#865092", "#a762b7", "#ca74de", "#ea8cff"),
}

# Where one risk step becomes the next, as multiples of the model's own
# decision threshold rather than as absolute probabilities. The threshold is
# chosen on validation and can move when the model is retrained; breaks pinned
# to 0.05 and 0.10 would quietly stop lining up with it, and the step boundary
# that matters, the one where the model starts saying yes, would drift off
# the legend.
RISK_BREAKS: Final[tuple[float, ...]] = (0.25, 0.5, 1.0, 2.0)


def risk_scale(mode: Mode) -> tuple[str, ...]:
    """The five risk steps, lowest first."""
    return RISK[mode]


def risk_step(score: float, threshold: float) -> int:
    """Which of the five steps a risk score falls in.

    Greater-than-or-equal at each break, because the warehouse's own check
    constraint is ``prediction_label = (risk_score >= decision_threshold)``.
    The anomaly flag two views away uses a strict ``>``; the two conventions
    differ and this one follows its own table, so that "painted in one of the
    top two steps" and "the model said yes" are the same statement.
    """
    if threshold <= 0:
        raise ValueError("a decision threshold of zero has no scale")
    return sum(1 for multiple in RISK_BREAKS if score >= multiple * threshold)


# The reliability curve's three lines. Two are data and one is the truth they
# are being measured against, so the third must not compete with them: `ideal`
# is drawn as a dashed rule in the chrome's own border colour rather than as a
# fourth series, because a reader should see two curves against a reference and
# not three curves.
#
# `calibrated` takes the top of the risk ramp, so the line that says "this is
# the number the grid above is painted from" is the same violet as the grid's
# strongest step. `raw` is deliberately achromatic: it is the thing being
# improved on, and giving it a hue of its own would invite reading it as a
# third category rather than as a before.
CALIBRATION: Final[Mapping[Mode, Mapping[str, str]]] = {
    "light": {"raw": "#8a8a86", "calibrated": RISK["light"][-1]},
    "dark": {"raw": "#9a9a95", "calibrated": RISK["dark"][-1]},
}


def calibration_colours(mode: Mode) -> Mapping[str, str]:
    """The reliability curve's palette: what was measured, and what it became."""
    return CALIBRATION[mode]


#: Expected calibration error above which the view stops calling a probability
#: trustworthy in plain English.
#:
#: Two and a half points. A reader looking at "8%" is entitled to be wrong by
#: about a point without having been misled, and this model's raw error is
#: five, which is enough to turn one-in-twelve into one-in-eight. It is a
#: presentation threshold and nothing computes against it; it exists so the
#: sentence under the curve is chosen by a number rather than by whoever last
#: edited the copy.
CALIBRATION_TRUST_CEILING: Final[float] = 0.025


def daily_equivalent(threshold: float, horizon_days: int) -> float:
    """The per-day hazard that would compose to a weekly threshold.

    BI-09 paints per-day cells from ML-13's hazards, and those live on a
    different scale from the weekly score: a hazard of 0.007 against a weekly
    decision threshold of 0.117 is not sixteen times too small, it is a
    different quantity. Binning it on the weekly boundaries would paint every
    day cell in the lowest step and say "no risk anywhere" about a week the
    model flagged.

    The conversion is the same identity ML-13 composes with, run backwards:
    seven days each at ``h`` compose to ``1 - (1 - h)^7``, so the ``h`` that
    composes to the threshold is ``1 - (1 - threshold)^(1/7)``. A day above it
    is a day contributing more than its even share of a week that would just
    clear the bar, which is the comparison a reader of a per-day cell is
    actually making.
    """
    if not 0.0 <= threshold < 1.0:
        raise ValueError(f"a weekly threshold outside [0, 1): {threshold}")
    if horizon_days < 1:
        raise ValueError(f"a horizon of {horizon_days} days has no daily share")
    return 1.0 - (1.0 - threshold) ** (1.0 / horizon_days)


def risk_key_html(mode: Mode, threshold: float) -> str:
    """The risk key, labelled in probabilities and marked at the threshold."""
    tokens = chrome(mode)
    steps = RISK[mode]
    bounds = ["0"] + [f"{multiple * threshold:.3f}" for multiple in RISK_BREAKS]
    cells = "".join(
        f'<div style="flex:1;text-align:center;">'
        f'<div style="height:14px;background:{colour};'
        f'border:1px solid {tokens["axis"]};"></div>'
        f'<div style="font-size:0.7rem;color:{tokens["ink_muted"]};'
        f'margin-top:0.2rem;">≥ {label}</div></div>'
        for colour, label in zip(steps, bounds)
    )
    return (
        f'<div style="display:flex;gap:2px;margin:0.25rem 0 0.35rem 0;">{cells}</div>'
        f'<div style="font-size:0.75rem;color:{tokens["ink_muted"]};">'
        f"The model says yes at {threshold:.4f}, the third break, so the top "
        f"two steps are exactly the flagged cities.</div>"
    )


# ---------------------------------------------------------------------------
# The small-screen wall
# ---------------------------------------------------------------------------
#
# Four views built on a wide layout, a fifteen-city map, a nine-step key and a
# matrix that is a grid of dates by cities. None of that survives a 390px
# viewport: the map loses its legend, the matrix wraps into a column of
# unlabelled cells, and the reader is left scrolling a picture of a dashboard
# rather than reading one. Shipping that is worse than not shipping it, because
# a chart that is unreadable still looks like it is saying something.
#
# So below :data:`SMALL_SCREEN_MAX_WIDTH_PX` the app is replaced by a full
# viewport panel that says where to open it. Phones only, and deliberately so:
# a narrowed desktop window and a tablet in landscape both clear the breakpoint,
# and locking those out would turn a readability problem into a lockout for
# readers who could have read it fine.
#
# The switch is a CSS media query rather than a Python branch. Streamlit does
# not know the viewport until the browser tells it, which is one round trip
# after the first paint, so a server-side check would render the dashboard on a
# phone and then replace it. The query costs nothing and is right on the first
# frame.
#
# Colours come from ``prefers-color-scheme`` for the same reason:
# ``st.context.theme.type`` is ``None`` on the first run of a session (see
# :func:`resolve_mode`), and a full-screen panel painted in the wrong mode and
# corrected a beat later is a much louder mistake than a mismatched legend.
# The panel covers the viewport, so it only has to agree with the browser.

# Phones, and nothing wider. See the note above on why this is not the 960px a
# comfortable reading of the wide layout would ask for.
SMALL_SCREEN_MAX_WIDTH_PX: Final[int] = 768

# Lucide's ``laptop-minimal``, inlined. A web font for one glyph is a network
# request the panel would have to survive without, since the reader seeing this
# is on a phone and quite possibly on mobile data.
_LAPTOP_SVG: Final[str] = (
    '<svg class="horizon-wall-icon" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="1.25" stroke-linecap="round" '
    'stroke-linejoin="round" aria-hidden="true" focusable="false">'
    '<rect x="3" y="4" width="18" height="12" rx="2" ry="2"></rect>'
    '<line x1="2" y1="20" x2="22" y2="20"></line>'
    "</svg>"
)


def _wall_tokens(mode: Mode) -> str:
    """The panel's custom properties for one mode, as CSS declarations.

    Every value is a token defined above rather than a colour invented for this
    panel. The wash is the outermost cold step, which is what makes the page
    read as belonging to the same palette as the map; the headline runs the
    ramp's own two poles, cold to warm, which is the whole argument of this
    module said once in a heading.
    """
    tokens = chrome(mode)
    steps = DIVERGING[mode]
    declarations = {
        "--horizon-wall-wash": steps[NEUTRAL_INDEX - 1],
        "--horizon-wall-surface": SURFACE[mode],
        "--horizon-wall-ink": tokens["ink"],
        "--horizon-wall-ink-secondary": tokens["ink_secondary"],
        "--horizon-wall-cold": steps[0],
        "--horizon-wall-warm": steps[-1],
    }
    return "".join(f"{name}:{value};" for name, value in declarations.items())


def small_screen_notice_html(title: str) -> str:
    """The panel, and the media query that decides when it is the whole page.

    ``title`` is the product's name, passed in rather than spelled here: this
    module owns colours, and what the dashboard is called is not one.

    Hiding the app is done with ``visibility`` rather than ``display``, and on
    the one element that contains all of Streamlit's chrome. ``visibility``
    inherits and can be turned back on further down the tree, which is what
    lets the panel live inside the app it is covering; ``display:none`` on an
    ancestor cannot be undone by a descendant. One selector therefore takes the
    sidebar, the header, the toolbar and the page with it, and no list of
    Streamlit's internal test ids has to be kept current for the wall to hold.
    """
    return f"""<style>
:root {{ {_wall_tokens("light")} }}
@media (prefers-color-scheme: dark) {{ :root {{ {_wall_tokens("dark")} }} }}

/* The panel itself is off above the breakpoint. This is the rule that has to
   hold, so it does not depend on knowing Streamlit's markup. */
.horizon-wall {{ display: none; }}

/* And the row Streamlit made for it goes too: an element container left in
   the flow is an empty flex item, and the vertical block's gap would push
   every page down by it. */
[data-testid="stElementContainer"]:has(.horizon-wall) {{ display: none; }}

@media (max-width: {SMALL_SCREEN_MAX_WIDTH_PX - 1}px) {{
  html, body {{ overflow: hidden; }}

  /* Everything Streamlit draws, in one selector. */
  [data-testid="stApp"] {{ visibility: hidden; }}

  /* Back in the flow so the panel inside it renders, and taking up none of
     it, because the panel is positioned against the viewport instead. */
  [data-testid="stElementContainer"]:has(.horizon-wall) {{
    display: block;
    height: 0; min-height: 0; margin: 0; padding: 0;
  }}

  .horizon-wall {{
    visibility: visible;
    display: flex;
    flex-direction: column;
    position: fixed;
    inset: 0;
    z-index: 2147483647;
    box-sizing: border-box;
    padding: 1.75rem 1.25rem 2.5rem;
    text-align: center;
    background: linear-gradient(
      180deg,
      var(--horizon-wall-wash) 0%,
      var(--horizon-wall-surface) 55%
    );
    color: var(--horizon-wall-ink);
  }}

  .horizon-wall-brand {{
    flex: 0 0 auto;
    margin: 0;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--horizon-wall-ink);
  }}

  .horizon-wall-body {{
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    margin: 0 auto;
    max-width: 23rem;
  }}

  .horizon-wall-icon {{
    width: 68px;
    height: 68px;
    color: var(--horizon-wall-ink);
  }}

  .horizon-wall-headline {{
    margin: 0;
    padding: 0;
    font-size: 1.5rem;
    font-weight: 700;
    line-height: 1.3;
    letter-spacing: -0.02em;
    color: var(--horizon-wall-ink);
  }}

  .horizon-wall-accent {{
    background-image: linear-gradient(
      100deg,
      var(--horizon-wall-cold) 0%,
      var(--horizon-wall-warm) 100%
    );
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
  }}

  .horizon-wall-copy {{
    margin: 0;
    font-size: 1rem;
    line-height: 1.5;
    letter-spacing: -0.01em;
    color: var(--horizon-wall-ink-secondary);
  }}
}}
</style>
<div class="horizon-wall" role="alert">
  <p class="horizon-wall-brand">{escape(title)}</p>
  <div class="horizon-wall-body">
    {_LAPTOP_SVG}
    <h1 class="horizon-wall-headline">We&rsquo;re
      <span class="horizon-wall-accent">better</span> on a bigger screen</h1>
    <p class="horizon-wall-copy">Fifteen cities, a week of forecasts and a
      thirty-year baseline need the room. Open this dashboard on a laptop or
      desktop to read it.</p>
  </div>
</div>"""
