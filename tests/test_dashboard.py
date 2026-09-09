"""Tests for the dashboard shell, its connection layer, and its palette.

Three things here are worth more than the line count suggests.

**The palette's claims are recomputed, not quoted.** ``dashboard/theme.py``
states in its docstring that a reader with the commonest form of colour
blindness can tell a cold anomaly from a warm one, and gives numbers. Those
numbers are the whole justification for the scale, and a docstring cannot go
out of date quietly if a test parses it and recomputes every cell. The colour
maths (sRGB to OKLab, and the Machado 2009 colour-vision-deficiency
simulation) is implemented here rather than imported, so a palette edit that
also edited the checker would still have to survive an independent measurement.

**The cold start is exercised, not described.** Neon suspending its compute is
the one failure this app is guaranteed to meet, and "handled gracefully" is not
a claim a comment can make. The retry is driven with an engine that fails the
way a suspended compute fails, and separately with one that fails the way a
broken query fails, because a retry that cannot tell those apart turns a clear
error into a slow one.

**"Never hardcoded" is enforced.** The connection string is asserted absent by
parsing the modules, in the same shape as the bronze-exclusion test in
``test_promotion.py``: a claim about paths not taken cannot be made by a test
that only observes the paths it took.

None of it needs a database. The suite that does, the one proving the app can
actually read Neon, is ``tests/check_connection.py``.
"""

from __future__ import annotations

import ast
import datetime as dt
import math
import re
import sys
import tomllib
from pathlib import Path
from typing import Iterator, Sequence

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("streamlit")
pytest.importorskip("pandas")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy.exc import (  # noqa: E402
    DBAPIError,
    InterfaceError,
    OperationalError,
    ProgrammingError,
)

from sqlalchemy import text as sa_text  # noqa: E402

from dashboard import theme  # noqa: E402
from dashboard.database import GOLD_SCHEMA as GOLD  # noqa: E402


def _registry_events():
    """The seven dated extremes, collected at import so they can parametrize."""
    from dashboard.views.anomaly_map import _events

    return list(_events())

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
STREAMLIT_CONFIG = PROJECT_ROOT / ".streamlit" / "config.toml"

# The separation below which two colours are the same colour to a reader.
# OKLab Euclidean distance x100; 8 is "distinct", 6 is the floor that is only
# acceptable when something other than colour also carries the distinction.
SEPARATION_TARGET = 8.0

# The two forms of colour-vision deficiency that affect a red-blue scale.
# Tritanopia is measured too but does not gate: it distorts blue-yellow, and a
# blue-red ramp is close to the axis it leaves alone.
GATING_DEFICIENCIES = ("protan", "deutan")


# --------------------------------------------------------------------------
# Colour maths. Implemented here on purpose; see the module docstring.
# --------------------------------------------------------------------------

_LIN_TO_LMS = (
    (0.4122214708, 0.5363325363, 0.0514459929),
    (0.2119034982, 0.6806995451, 0.1073969566),
    (0.0883024619, 0.2817188376, 0.6299787005),
)
_LMS_TO_OKLAB = (
    (0.2104542553, 0.7936177850, -0.0040720468),
    (1.9779984951, -2.4285922050, 0.4505937099),
    (0.0259040371, 0.7827717662, -0.8086757660),
)
# Machado, Oliveira & Fernandes (2009), severity 1.0. The thresholds above are
# calibrated against this simulation, so the model is part of the standard
# rather than an implementation detail.
_MACHADO = {
    "protan": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deutan": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
    "tritan": (
        (1.255528, -0.076749, -0.178779),
        (-0.078411, 0.930809, 0.147602),
        (0.004733, 0.691367, 0.303900),
    ),
}


def _matmul(matrix, vector):
    return tuple(sum(row[i] * vector[i] for i in range(3)) for row in matrix)


def _srgb(hex_colour: str) -> tuple[float, float, float]:
    raw = hex_colour.lstrip("#")
    return tuple(int(raw[i : i + 2], 16) / 255 for i in (0, 2, 4))


def _linear(hex_colour: str) -> tuple[float, float, float]:
    return tuple(
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        for c in _srgb(hex_colour)
    )


def _oklab_from_linear(linear) -> tuple[float, float, float]:
    lms = _matmul(_LIN_TO_LMS, linear)
    return _matmul(_LMS_TO_OKLAB, tuple(math.copysign(abs(c) ** (1 / 3), c) for c in lms))


def lightness(hex_colour: str) -> float:
    return _oklab_from_linear(_linear(hex_colour))[0]


def chroma(hex_colour: str) -> float:
    _, a, b = _oklab_from_linear(_linear(hex_colour))
    return math.hypot(a, b)


def hue(hex_colour: str) -> float:
    _, a, b = _oklab_from_linear(_linear(hex_colour))
    return math.degrees(math.atan2(b, a)) % 360


def separation(first: str, second: str, deficiency: str | None = None) -> float:
    """Perceptual distance, optionally as a colour-blind reader would see it."""

    def convert(colour: str):
        linear = _linear(colour)
        if deficiency:
            linear = tuple(
                min(1.0, max(0.0, value))
                for value in _matmul(_MACHADO[deficiency], linear)
            )
        return _oklab_from_linear(linear)

    return 100 * math.dist(convert(first), convert(second))


def contrast(first: str, second: str) -> float:
    def luminance(colour: str) -> float:
        r, g, b = _linear(colour)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    high, low = sorted((luminance(first), luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def arms(mode: theme.Mode) -> tuple[Sequence[str], str, Sequence[str]]:
    """The ramp split into cold (midpoint outward), midpoint, warm (outward)."""
    steps = theme.DIVERGING[mode]
    middle = theme.NEUTRAL_INDEX
    return steps[:middle][::-1], steps[middle], steps[middle + 1 :]


MODES = tuple(theme.DIVERGING)


# --------------------------------------------------------------------------
# The palette encodes a signed quantity, and the encoding has to survive.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_midpoint_is_grey_not_a_hue(mode: theme.Mode) -> None:
    """Zero anomaly must not look like a third kind of weather.

    A hue at the midpoint reads as its own category, since "normal" acquires
    a temperature, and it also breaks the two-hue promise a diverging scale
    makes to a colour-blind reader, who is relying on there being exactly two
    directions to tell apart.
    """
    assert chroma(theme.NEUTRAL[mode]) < 0.01


@pytest.mark.parametrize("mode", MODES)
def test_the_two_arms_are_two_hues(mode: theme.Mode) -> None:
    """Each arm holds one hue, and the arms are far apart on the wheel."""
    cold, _, warm = arms(mode)

    def spread(colours: Sequence[str]) -> float:
        hues = [hue(colour) for colour in colours]
        return max(hues) - min(hues)

    assert spread(cold) < 10, "the cold arm drifts in hue"
    assert spread(warm) < 10, "the warm arm drifts in hue"

    apart = abs(hue(cold[-1]) - hue(warm[-1])) % 360
    assert 120 < min(apart, 360 - apart) < 240, "the poles are not opposed"


@pytest.mark.parametrize("mode", MODES)
def test_equal_magnitudes_carry_equal_weight(mode: theme.Mode) -> None:
    """+3σ and -3σ must look equally loud.

    If one arm is systematically darker, the scale editorialises: warming looks
    more urgent than cooling, or the reverse, for a reason that is in the
    palette rather than in the data.
    """
    cold, _, warm = arms(mode)
    for distance, (cold_step, warm_step) in enumerate(zip(cold, warm), start=1):
        assert abs(lightness(cold_step) - lightness(warm_step)) < 0.01, (
            f"step {distance} is not balanced between the arms"
        )


@pytest.mark.parametrize("mode", MODES)
def test_a_step_of_colour_is_a_step_of_anomaly(mode: theme.Mode) -> None:
    """Lightness rises monotonically outward, in near-even increments.

    Uneven steps make the scale lie about magnitude: a big jump at the
    midpoint exaggerates small departures, and the compressed end understates
    the large ones that are the entire point of an extremes dashboard.
    """
    steps = theme.DIVERGING[mode]
    values = [lightness(step) for step in steps]
    gaps = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]

    middle = theme.NEUTRAL_INDEX
    assert values[middle] == pytest.approx(max(values) if mode == "light" else min(values)), (
        "the midpoint is not the extreme of the lightness ladder"
    )
    for index in range(middle):
        rising = values[index] < values[index + 1]
        assert rising == (mode == "light"), "the cold arm is not monotone"
    for index in range(middle, len(values) - 1):
        falling = values[index] > values[index + 1]
        assert falling == (mode == "light"), "the warm arm is not monotone"

    assert min(gaps) >= 0.06, f"steps too close to separate: {min(gaps):.3f}"
    assert max(gaps) / min(gaps) <= 1.6, "the ladder is uneven enough to distort magnitude"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("deficiency", GATING_DEFICIENCIES)
def test_the_sign_of_an_anomaly_survives_colour_blindness(
    mode: theme.Mode, deficiency: str
) -> None:
    """The claim the whole scale rests on.

    A red-blue climate figure says one thing above everything else: which way
    it went. If a protanopic reader cannot separate the warm arm from the cold
    arm at the same magnitude, the figure is not merely less pretty for them:
    it is telling them the opposite of the truth half the time.
    """
    cold, _, warm = arms(mode)
    worst = min(
        separation(cold_step, warm_step, deficiency)
        for cold_step, warm_step in zip(cold, warm)
    )
    assert worst >= SEPARATION_TARGET, (
        f"cold and warm collapse to {worst:.1f} under {deficiency}"
    )


@pytest.mark.parametrize("mode", MODES)
def test_anomaly_is_distinguishable_from_no_anomaly(mode: theme.Mode) -> None:
    """Every coloured step separates from the neutral, under every deficiency."""
    cold, neutral, warm = arms(mode)
    worst = min(
        separation(step, neutral, deficiency)
        for step in (*cold, *warm)
        for deficiency in (None, *_MACHADO)
    )
    assert worst >= SEPARATION_TARGET


@pytest.mark.parametrize("mode", MODES)
def test_adjacent_steps_stay_apart(mode: theme.Mode) -> None:
    """Neighbouring steps are readable as different, simulated or not."""
    steps = theme.DIVERGING[mode]
    worst = min(
        separation(steps[index], steps[index + 1], deficiency)
        for index in range(len(steps) - 1)
        for deficiency in (None, *_MACHADO)
    )
    assert worst >= SEPARATION_TARGET


@pytest.mark.parametrize("mode", MODES)
def test_the_poles_are_visible_against_their_own_surface(mode: theme.Mode) -> None:
    """The ends of the scale clear 3:1 on the surface they were built for.

    The surface is part of the palette for exactly this reason: a contrast
    figure measured against the wrong background is not a weaker check, it is a
    meaningless one.
    """
    steps = theme.DIVERGING[mode]
    surface = theme.SURFACE[mode]
    assert contrast(steps[0], surface) >= 3.0
    assert contrast(steps[-1], surface) >= 3.0


def test_the_documented_measurements_cannot_go_stale() -> None:
    """Recompute every number theme.py quotes, from the constants it ships.

    The docstring is the argument for the palette. If it can drift from the
    hex values beside it, the argument stops being evidence and becomes
    decoration, so the table is parsed and each cell recomputed.
    """
    table = re.findall(
        r"^(\S[^\n]*?)\s{2,}(\d+\.\d+)\s+(\d+\.\d+)\s*$",
        theme.__doc__ or "",
        flags=re.MULTILINE,
    )  # the table is column-aligned, so this one keeps its line structure
    documented = {row[0].strip(): (float(row[1]), float(row[2])) for row in table}

    def measure(mode: theme.Mode) -> dict[str, float]:
        cold, neutral, warm = arms(mode)
        steps = theme.DIVERGING[mode]
        values = [lightness(step) for step in steps]
        gaps = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]
        return {
            "Cold vs warm at equal magnitude": min(
                separation(c, w, d)
                for c, w in zip(cold, warm)
                for d in GATING_DEFICIENCIES
            ),
            "Any step vs the neutral midpoint": min(
                separation(s, neutral, d)
                for s in (*cold, *warm)
                for d in (None, *_MACHADO)
            ),
            "Adjacent steps": min(
                separation(steps[i], steps[i + 1], d)
                for i in range(len(steps) - 1)
                for d in (None, *_MACHADO)
            ),
            "Lightness-step evenness": max(gaps) / min(gaps),
        }

    expected_rows = set(measure("light"))
    assert expected_rows <= set(documented), (
        f"theme.py's table lost rows: {sorted(expected_rows - set(documented))}"
    )

    for index, mode in enumerate(("light", "dark")):
        for label, value in measure(mode).items():
            assert round(value, 2 if value < 5 else 1) == documented[label][index], (
                f"{label} ({mode}) measures {value:.2f}, docstring says "
                f"{documented[label][index]}"
            )


def test_the_documented_shortfall_is_the_real_one() -> None:
    """theme.py admits one number below its floor. Check it is still that one.

    An admitted weakness that quietly got better is a docstring telling a
    reader to distrust something that is now fine; one that quietly got worse
    is the opposite and much more serious. Either way the prose has to move
    with the palette.
    """
    quoted = re.findall(r"(\d\.\d\d):1", _theme_prose())
    assert len(quoted) == 2, "expected the light shortfall and its dark counterpart"

    def worst_inner(mode: theme.Mode) -> float:
        """The palest step of either arm; the docstring quotes the weaker one."""
        cold, _, warm = arms(mode)
        surface = theme.SURFACE[mode]
        return min(contrast(cold[0], surface), contrast(warm[0], surface))

    measured_light = worst_inner("light")
    measured_dark = worst_inner("dark")

    assert round(measured_light, 2) == float(quoted[0])
    assert round(measured_dark, 2) == float(quoted[1])
    assert measured_light < 2.0 <= measured_dark, (
        "the shortfall theme.py explains no longer matches the palette"
    )


# --------------------------------------------------------------------------
# Defined once. Two ways that can stop being true.
# --------------------------------------------------------------------------


def _python_modules(directory: Path) -> Iterator[Path]:
    return (path for path in sorted(directory.rglob("*.py")) if "__pycache__" not in path.parts)


def _string_constants(tree: ast.AST) -> Iterator[str]:
    """Every string literal that is not a docstring.

    Docstrings are excluded because they *describe* the colours and the
    connection string, and a check that flagged the explanation along with the
    thing it explains would be turned off within a week.
    """
    docstrings = {
        ast.get_docstring(node, clean=False)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value not in docstrings:
                yield node.value


HEX_COLOUR = re.compile(r"#[0-9a-fA-F]{6}\b")


def test_no_module_but_theme_spells_a_colour() -> None:
    """The palette is defined once, and that is checked rather than intended.

    A hex value in a view is how two views stop agreeing on what red means.
    """
    offenders: list[str] = []
    for path in _python_modules(DASHBOARD_DIR):
        if path.name == "theme.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for value in _string_constants(tree):
            for found in HEX_COLOUR.findall(value):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}: {found}")
    assert not offenders, "colours outside theme.py: " + ", ".join(offenders)


def test_the_streamlit_theme_matches_the_palette() -> None:
    """config.toml repeats theme.py's tokens, because TOML cannot import.

    That is a duplicate, and duplicates rot. This is the check that stops it:
    Streamlit paints the surface the ramp was validated against, or the test
    fails and the contrast figures elsewhere in this file become claims about
    a background nobody is looking at.
    """
    config = tomllib.loads(STREAMLIT_CONFIG.read_text(encoding="utf-8"))["theme"]

    for mode in MODES:
        section = config[mode]
        tokens = theme.chrome(mode)
        assert section["backgroundColor"] == theme.SURFACE[mode]
        assert section["backgroundColor"] == tokens["surface"]
        assert section["secondaryBackgroundColor"] == tokens["plane"]
        assert section["textColor"] == tokens["ink"]
        assert section["primaryColor"] == tokens["ink_secondary"]
        assert section["borderColor"] == tokens["gridline"]


def test_the_chrome_accent_is_not_a_data_colour() -> None:
    """Streamlit's accent must never be mistakable for a step of the ramp.

    theme.py argues there is no chromatic accent that can be: nine steps of
    blue and red leave no room for a tenth hue. The consequence is that the
    accent has to be achromatic, and this is what holds it there.
    """
    for mode in MODES:
        accent = theme.chrome(mode)["ink_secondary"]
        assert chroma(accent) < 0.02, "the accent has acquired a hue"


# --------------------------------------------------------------------------
# The connection layer: where the credential comes from.
# --------------------------------------------------------------------------


@pytest.fixture
def database(monkeypatch):
    """dashboard.database with its caches and secrets under test control."""
    from dashboard import database as module

    monkeypatch.setattr(module, "_secret", lambda key: None)
    return module


def _settings(monkeypatch, module, **overrides):
    import dataclasses

    from config import get_settings

    resolved = dataclasses.replace(get_settings(), **overrides)
    monkeypatch.setattr(module, "get_settings", lambda: resolved)
    return resolved


NEON = "postgresql://u:p@ep-x-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require"
LOCAL = "postgresql://climate:pw@localhost:5432/climate"


def test_streamlit_secrets_win(database, monkeypatch) -> None:
    """Deployed, the secret store is the answer and .env is not consulted."""
    monkeypatch.setattr(database, "_secret", lambda key: NEON if key == "DATABASE_URL" else None)
    _settings(monkeypatch, database, serving_database_url=LOCAL, database_url=LOCAL)

    source = database.resolve_database_url()
    assert source.url == NEON
    assert "Streamlit secrets" in source.origin


def test_serving_url_is_preferred_over_the_local_warehouse(database, monkeypatch) -> None:
    """The dashboard is a serving-tier reader, on a laptop as much as in the cloud.

    DATABASE_URL locally is the Docker container that holds bronze and silver.
    Falling back to it would point a public dashboard's code at layers that
    were deliberately never promoted.
    """
    _settings(
        monkeypatch,
        database,
        environment="local",
        serving_database_url=NEON,
        database_url=LOCAL,
    )
    assert database.resolve_database_url().url == NEON


def test_database_url_is_used_only_when_the_environment_says_serving(
    database, monkeypatch
) -> None:
    """ENVIRONMENT is the project's existing switch; it is honoured, not re-invented."""
    _settings(
        monkeypatch,
        database,
        environment="local",
        serving_database_url=None,
        database_url=LOCAL,
    )
    with pytest.raises(database.DashboardConfigError):
        database.resolve_database_url()

    _settings(
        monkeypatch,
        database,
        environment="serving",
        serving_database_url=None,
        database_url=NEON,
    )
    assert database.resolve_database_url().url == NEON


def test_a_missing_secrets_file_is_not_a_missing_secret(monkeypatch) -> None:
    """st.secrets raises when no secrets.toml exists anywhere.

    That is the ordinary state of a development machine, and it must not stop
    the .env fallback from being reached. Otherwise the app cannot be run
    locally at all, which is the one thing this ticket has to deliver.
    """
    from dashboard import database as module
    from streamlit.errors import StreamlitSecretNotFoundError
    from streamlit.runtime.secrets import Secrets

    def absent(self, key, default=None):
        raise StreamlitSecretNotFoundError("no secrets found")

    # Patched on the class: the Secrets instance refuses attribute assignment,
    # which is itself a small confirmation that st.secrets is not a dict.
    monkeypatch.setattr(Secrets, "get", absent)
    assert module._secret("DATABASE_URL") is None

    _settings(monkeypatch, module, serving_database_url=NEON)
    assert module.resolve_database_url().url == NEON


def test_the_error_names_both_ways_to_fix_it(database, monkeypatch) -> None:
    """A visitor cannot fix this; whoever deployed it can, from either side."""
    _settings(monkeypatch, database, serving_database_url=None, database_url=None)
    with pytest.raises(database.DashboardConfigError) as raised:
        database.resolve_database_url()
    message = str(raised.value)
    assert "Streamlit secrets" in message
    assert "SERVING_DATABASE_URL" in message
    assert ".env" in message


CONNECTION_STRING = re.compile(r"\bpostgres(?:ql)?://[^\s\"']*[^\s\"'/]", re.IGNORECASE)


def test_no_connection_string_is_written_into_the_dashboard() -> None:
    """"Never hardcoded" as a property of the source, not of a code review.

    Asserted over the parsed modules rather than by running them, because a
    behavioural test can only show that the paths it happened to take did not
    contain a credential.
    """
    offenders: list[str] = []
    for path in _python_modules(DASHBOARD_DIR):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for value in _string_constants(tree):
            if CONNECTION_STRING.search(value):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}: {value[:40]}")
    assert not offenders, "connection strings in source: " + ", ".join(offenders)


def test_the_guard_fails_on_a_module_that_does_hardcode_one(tmp_path: Path) -> None:
    """A guard nobody has seen fail is a guard nobody should trust."""
    leaky = tmp_path / "leaky.py"
    leaky.write_text('URL = "postgresql://u:p@host/db"\n', encoding="utf-8")
    tree = ast.parse(leaky.read_text(encoding="utf-8"))
    assert any(CONNECTION_STRING.search(value) for value in _string_constants(tree))


def test_the_committed_example_holds_no_real_credential() -> None:
    """The example is committed, so it is the file most likely to leak one."""
    example = (PROJECT_ROOT / ".streamlit" / "secrets.toml.example").read_text("utf-8")
    parsed = tomllib.loads(example)
    url = parsed["DATABASE_URL"]
    assert "USER:PASSWORD" in url, "the example has been filled in with real values"
    assert "sslmode=require" in url


def test_the_real_secrets_file_is_ignored() -> None:
    """Committed .gitignore, not a habit."""
    rules = (PROJECT_ROOT / ".gitignore").read_text("utf-8")
    assert ".streamlit/secrets.toml" in rules


# --------------------------------------------------------------------------
# The connection layer: what happens when Neon is asleep.
# --------------------------------------------------------------------------


def _operational(message: str = "server closed the connection unexpectedly"):
    """What psycopg2 raises through SQLAlchemy when the socket is dead."""
    return OperationalError("select 1", {}, Exception(message))


class FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeEngine:
    """An engine that fails the way a suspended compute fails.

    ``connect`` is where a Neon suspension surfaces: ``pool_pre_ping`` probes
    the connection at checkout and raises there rather than inside the query.
    """

    def __init__(self, failures: int, error=None):
        self.failures = failures
        self.error = error or _operational()
        self.connects = 0
        self.disposals = 0

    def connect(self):
        self.connects += 1
        if self.connects <= self.failures:
            raise self.error
        return FakeConnection()

    def dispose(self):
        self.disposals += 1


@pytest.fixture
def cold_start(monkeypatch):
    """Install a fake engine and remove the retry backoff from the clock."""
    from dashboard import database as module

    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        module.pd,
        "read_sql_query",
        lambda sql, connection, params=None: pd.DataFrame({"ok": [1]}),
    )

    def install(engine):
        monkeypatch.setattr(module, "get_engine", lambda: engine)
        return engine

    return install


def test_a_suspended_compute_is_waited_out_not_reported(cold_start) -> None:
    """The behaviour the ticket asks for, driven rather than described."""
    from dashboard import database as module

    engine = cold_start(FakeEngine(failures=1))
    frame = module._execute("select 1", None)

    assert list(frame["ok"]) == [1]
    assert engine.connects == 2, "the query was not retried"
    assert engine.disposals == 1, "the stale pool was not dropped before retrying"


def test_the_whole_pool_is_dropped_not_the_one_dead_connection(cold_start) -> None:
    """Suspension kills every pooled socket at once.

    Healing them one checkout at a time pays the same failure again on the next
    query, which is how a handled cold start still looks broken to a visitor
    clicking between views.
    """
    from dashboard import database as module

    engine = cold_start(FakeEngine(failures=2))
    module._execute("select 1", None)
    assert engine.disposals == 2


def test_giving_up_says_so_in_the_dashboard_s_own_language(cold_start) -> None:
    """After the retries are spent it is an error, and a typed one."""
    from dashboard import database as module

    engine = cold_start(FakeEngine(failures=99))
    with pytest.raises(module.WarehouseUnreachable) as raised:
        module._execute("select 1", None)

    assert engine.connects == module.COLD_START_ATTEMPTS
    assert engine.disposals == module.COLD_START_ATTEMPTS
    assert isinstance(raised.value.__cause__, OperationalError)


def test_a_broken_query_is_not_retried(cold_start) -> None:
    """The distinction that keeps the retry honest.

    A missing column fails identically on every attempt. Retrying it spends
    three round trips and the backoff to produce the same error later, and
    hides a code defect behind what looks like a slow database.
    """
    from dashboard import database as module

    broken = ProgrammingError("select nope", {}, Exception('column "nope" does not exist'))
    engine = cold_start(FakeEngine(failures=99, error=broken))

    with pytest.raises(ProgrammingError):
        module._execute("select nope", None)

    assert engine.connects == 1
    assert engine.disposals == 0


@pytest.mark.parametrize(
    "error, transient",
    [
        (_operational(), True),
        (InterfaceError("select 1", {}, Exception("connection already closed")), True),
        (
            DBAPIError("select 1", {}, Exception("reset"), connection_invalidated=True),
            True,
        ),
        (DBAPIError("select 1", {}, Exception("constraint")), False),
        (ProgrammingError("select 1", {}, Exception("syntax error")), False),
    ],
)
def test_transient_means_the_database_not_the_query(error, transient) -> None:
    from dashboard import database as module

    assert module._is_transient(error) is transient


def test_a_mid_query_invalidation_is_also_a_cold_start(cold_start, monkeypatch) -> None:
    """Suspension can land after checkout, not only at it.

    SQLAlchemy marks that case with ``connection_invalidated``, and it has to
    be retried for the same reason the checkout failure is.
    """
    from dashboard import database as module

    engine = cold_start(FakeEngine(failures=0))
    calls = {"n": 0}

    def flaky(sql, connection, params=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise DBAPIError("select 1", {}, Exception("reset"), connection_invalidated=True)
        return pd.DataFrame({"ok": [1]})

    monkeypatch.setattr(module.pd, "read_sql_query", flaky)
    assert list(module._execute("select 1", None)["ok"]) == [1]
    assert calls["n"] == 2
    assert engine.disposals == 1, "an invalidated connection left the pool intact"
    assert engine.connects == 2


def test_the_retry_budget_is_bounded(cold_start) -> None:
    """A public page must fail in seconds, not hang.

    Three attempts with a linear backoff is a few seconds against a cold start
    that measures 1.2 s: enough headroom for a slow resume, short enough that
    a genuinely dead database still renders its error while someone is
    watching.
    """
    from dashboard import database as module

    budget = sum(
        module.COLD_START_BACKOFF_SECONDS * attempt
        for attempt in range(1, module.COLD_START_ATTEMPTS)
    )
    assert module.COLD_START_ATTEMPTS >= 2
    assert budget <= 10


# --------------------------------------------------------------------------
# Caching.
# --------------------------------------------------------------------------


def test_results_are_cached_and_the_ttl_is_sized_for_the_compute_budget() -> None:
    """Neon's meter counts wake-ups, not queries.

    Compute stays awake five minutes after each query, so a cache miss costs a
    five-minute minimum of the monthly allowance however fast the query runs.
    A short TTL is therefore not a small cost; it is the dominant one.
    """
    from dashboard import database as module

    assert module.CACHE_TTL_SECONDS >= 3600, "short enough to burn the compute budget"
    assert module.CACHE_TTL_REFERENCE_SECONDS >= module.CACHE_TTL_SECONDS

    wake_ups_per_day = 24 * 3600 / module.CACHE_TTL_SECONDS
    monthly_hours = wake_ups_per_day * 30 * (5 / 60)
    assert monthly_hours <= 20, (
        f"scheduled refreshes alone would spend {monthly_hours:.0f} of the 100 "
        "free compute-hours"
    )


def test_the_cold_path_declares_a_loading_state() -> None:
    """A cache miss is the slow path, and it is the one that gets the spinner.

    Streamlit shows a cached function's spinner only when it misses, which is
    exactly the request that may be waiting on a resume. A spinner on every
    call would be noise; none at all would be a frozen page.
    """
    from dashboard import database as module

    for cached in (module._query_marts, module._query_reference, module.warehouse_status):
        assert getattr(cached, "clear", None), "not a cached function"
    assert "resumes from idle" in module.SPINNER_MESSAGE


def test_parameter_order_does_not_split_the_cache(monkeypatch) -> None:
    """Two calls that mean the same thing must not wake the database twice."""
    from dashboard import database as module

    seen: list[tuple] = []
    monkeypatch.setattr(module, "_query_marts", lambda sql, key: seen.append(key))

    module.run_query("select :a, :b", {"a": 1, "b": 2})
    module.run_query("select :a, :b", {"b": 2, "a": 1})
    assert seen[0] == seen[1] == (("a", 1), ("b", 2))


def test_refreshing_drops_data_and_keeps_the_connection(monkeypatch) -> None:
    """After a promotion the results are stale; the socket is not."""
    from dashboard import database as module

    cleared: list[str] = []
    for name in ("_query_marts", "_query_reference", "warehouse_status"):
        target = getattr(module, name)
        monkeypatch.setattr(target, "clear", lambda name=name: cleared.append(name))
    monkeypatch.setattr(
        module.get_engine, "clear", lambda: cleared.append("engine")
    )

    module.clear_caches()
    assert set(cleared) == {"_query_marts", "_query_reference", "warehouse_status"}


# --------------------------------------------------------------------------
# The shell.
# --------------------------------------------------------------------------


def test_all_four_views_are_navigable() -> None:
    """Four views, in the order the proposal lists them."""
    from dashboard import views

    assert [module.VIEW.title for module in views.ORDER] == [
        "Global Anomaly Map",
        "Climate Matrix",
        "Storm Dynamics",
        "Risk Horizon",
    ]


def test_each_view_carries_the_caption_the_criteria_require() -> None:
    """"Each view has a one-line plain-English caption", §Workstream 4."""
    from dashboard import views

    for module in views.ORDER:
        view = module.VIEW
        assert view.caption and view.caption[0].isupper()
        assert "\n" not in view.caption.strip()
        assert view.question.endswith("?")


def test_view_urls_are_stable_and_distinct() -> None:
    """A view's path is a link someone may send; it should look deliberate."""
    from dashboard import views

    paths = [module.VIEW.url_path for module in views.ORDER]
    assert len(set(paths)) == len(paths)
    for path in paths:
        assert re.fullmatch(r"[a-z][a-z0-9-]*", path), path


def test_every_view_reads_only_promoted_gold() -> None:
    """Bronze and silver never left the local container, so nothing may ask for them."""
    from dashboard import views

    for module in views.ORDER:
        for statement in _sql_literals(Path(module.__file__)):
            assert "bronze_raw" not in statement and "silver_staging" not in statement, (
                f"{module.VIEW.title} reaches for a layer that was never promoted"
            )
        assert module.VIEW.source_table.startswith("gold_marts.")


def _sql_literals(path: Path) -> Iterator[str]:
    """Every string in a module that looks like a query."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "select " in node.value.lower():
                yield node.value
        elif isinstance(node, ast.JoinedStr):
            rendered = "".join(
                part.value for part in node.values if isinstance(part, ast.Constant)
            )
            if "select " in rendered.lower():
                yield rendered


def test_the_shell_holds_no_sql_and_no_colour() -> None:
    """app.py arranges. Anything else in it is something a view will need later."""
    tree = ast.parse((DASHBOARD_DIR / "app.py").read_text(encoding="utf-8"))
    for value in _string_constants(tree):
        lowered = value.lower()
        assert "select " not in lowered, f"SQL in the shell: {value[:40]}"
        assert not HEX_COLOUR.search(value), f"a colour in the shell: {value[:40]}"


# --------------------------------------------------------------------------
# The shell, rendered. What a visitor sees when something is wrong.
# --------------------------------------------------------------------------

APP = str(PROJECT_ROOT / "dashboard" / "app.py")


def _stub_frame(sql: str) -> pd.DataFrame:
    """A plausible answer for whichever query was asked, without a database."""
    if "min(date_key)" in sql and "last_scored_day" in sql:
        return pd.DataFrame(
            [{
                "first_day": dt.date(1995, 1, 1),
                "last_day": dt.date(2026, 9, 2),
                "last_scored_day": dt.date(2026, 9, 2),
            }]
        )
    if "observed_days" in sql:
        return pd.DataFrame(
            [{"city_id": "delhi", "first_day": dt.date(1995, 1, 1),
              "last_day": dt.date(2026, 9, 2), "observed_days": 11568, "scored_days": 11568}]
        )
    if "fact_weather_hourly" in sql:
        rows = []
        for day in range(40):
            swing = (day % 20) - 10
            for city_id, name in (("reykjavik", "Reykjavík"), ("singapore", "Singapore")):
                rows.append({
                    "city_id": city_id, "name": name,
                    "date_key": f"2025-01-{day % 28 + 1:02d}",
                    "peak_gust": 30.0 + 2.0 * abs(swing),
                    "sharpest_fall": float(-abs(swing)),
                    "sharpest_rise": float(abs(swing)),
                    "hours": 24,
                    "pressure_change_24h": float(swing),
                })
        return pd.DataFrame(rows)
    if "cross join years" in sql:
        rows = []
        for year in range(1995, 2027):
            rows.append({"city_id": "cairo", "name": "Cairo", "year": year,
                         "hot_days": max(0, (year - 1995) // 3), "cold_days": 1,
                         "scored_days": 365})
            rows.append({"city_id": "moscow", "name": "Moscow", "year": year,
                         "hot_days": 0, "cold_days": 0, "scored_days": 0})
        return pd.DataFrame(rows)
    if "dim_cities" in sql and "longitude" in sql:
        return pd.DataFrame(
            [
                {"city_id": "delhi", "name": "Delhi", "country": "India",
                 "latitude": 28.6, "longitude": 77.2, "observed_c": 35.0,
                 "baseline_c": 28.7, "baseline_sigma": 1.3, "z": 4.79,
                 "departure_c": 6.3, "is_anomaly": True,
                 "baseline_observations": 465.0, "observed": True},
                {"city_id": "moscow", "name": "Moscow", "country": "Russia",
                 "latitude": 55.8, "longitude": 37.6, "observed_c": None,
                 "baseline_c": None, "baseline_sigma": None, "z": None,
                 "departure_c": None, "is_anomaly": None,
                 "baseline_observations": None, "observed": False},
            ]
        )
    if "fact_ml_predictions" in sql:
        return pd.DataFrame([{
            "city_id": "singapore", "name": "Singapore", "country": "Singapore",
            "forecast_date": dt.date(2026, 9, 2),
            "horizon_start": dt.date(2026, 9, 3), "horizon_end": dt.date(2026, 9, 9),
            "horizon_days": 7, "risk_score": 0.247, "prediction_label": True,
            "decision_threshold": 0.1024,
            "model_version": "model-unweighted-v1-2ad772ff7b18",
            "model_variant": "unweighted", "feature_count": 27,
            "scored_at": pd.Timestamp("2026-09-07T09:59:25Z"),
        }])
    raise AssertionError(f"a view asked something the offline stub cannot answer: {sql[:80]}")


@pytest.fixture
def offline(monkeypatch):
    """Run the app with the warehouse replaced, so nothing reaches a network.

    Every seam has to be closed, and there are more than one: the shell calls
    ``warehouse_status`` for its sidebar, the stubs call ``run_query`` through
    ``_scaffold``, and the map calls it in its own module. Each bound the name
    at import, so each is patched where it is used rather than where it is
    defined: a single patch on ``dashboard.database`` would leave the map
    talking to Neon in a test that claims to be offline.
    """
    from dashboard import database as module
    from dashboard import views
    from dashboard.views import _scaffold

    def install(*, status=None, query=None, source=None):
        monkeypatch.setattr(module, "warehouse_status", status or (lambda: {"cities": 15}))
        answer = query or (lambda sql, *a, **k: _stub_frame(sql))
        for bound in (_scaffold, *views.ORDER):
            if hasattr(bound, "run_query"):
                monkeypatch.setattr(bound, "run_query", answer)
        if source is not None:
            monkeypatch.setattr(module, "resolve_database_url", source)

    return install


def test_every_view_renders_and_none_of_them_raises(offline) -> None:
    """All four views, with the database stubbed out.

    Rendered one at a time rather than by clicking through the shell, because
    ``AppTest`` addresses pages by script path and these are callables. The
    shell's own arrangement of them is covered structurally above and end to
    end by the default page below; what is being checked here is that each
    view's body survives a render.

    Driven through a one-line script rather than ``AppTest.from_function``,
    which re-executes the function's source into an empty namespace and so
    cannot see the module-level ``VIEW`` each view is built from.
    """
    from streamlit.testing.v1 import AppTest

    from dashboard import views

    offline()

    for module in views.ORDER:
        script = (
            "from importlib import import_module\n"
            f"import_module({module.__name__!r}).render()\n"
        )
        app = AppTest.from_string(script, default_timeout=30).run()
        assert not app.exception, f"{module.VIEW.title} raised: {app.exception}"
        assert [element.value for element in app.title] == [module.VIEW.title]
        # The caption is the acceptance criteria's, and it has to reach the
        # page rather than merely exist on the dataclass. Asserted here rather
        # than as a proxy like "some metric rendered", which a chart view has
        # no reason to satisfy.
        assert module.VIEW.caption in [element.value for element in app.caption]


def test_the_shell_renders_its_default_view(offline) -> None:
    """Shell and view together: navigation, sidebar, and a page under it."""
    from streamlit.testing.v1 import AppTest

    from dashboard import views

    first = views.ORDER[0].VIEW
    offline()

    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    assert first.title in [element.value for element in app.title]
    assert "Refresh data" in [button.label for button in app.button]


def test_a_missing_secret_is_a_sentence_not_a_traceback(offline) -> None:
    """The state a freshly deployed app is in before its secret is pasted in.

    A visitor sees an explanation; whoever deployed it sees the fix. Neither
    sees a stack trace, which on a public URL is both useless and a disclosure.
    """
    from streamlit.testing.v1 import AppTest

    from dashboard.database import DashboardConfigError

    def unconfigured():
        raise DashboardConfigError("No serving database is configured. Set DATABASE_URL.")

    offline(source=unconfigured)
    app = AppTest.from_file(APP, default_timeout=30).run()

    assert not app.exception
    assert app.error, "no error panel was rendered"
    assert "no database to read" in app.error[0].value


def test_a_sleeping_warehouse_offers_a_retry_rather_than_a_stack_trace(offline) -> None:
    """The failure this ticket exists for, as a visitor experiences it.

    Neon suspending is recoverable by waiting, so the page has to say that and
    give a way to act on it. An unhandled ``OperationalError`` says none of it
    and looks like the project is broken.
    """
    from streamlit.testing.v1 import AppTest

    from dashboard.database import WarehouseUnreachable

    def asleep(*args, **kwargs):
        raise WarehouseUnreachable("The serving warehouse did not answer after 3 attempts.")

    offline(status=asleep, query=asleep)
    app = AppTest.from_file(APP, default_timeout=30).run()

    assert not app.exception, "the cold start reached the visitor as a traceback"
    assert "did not answer" in app.error[0].value
    assert "Try again" in [button.label for button in app.button]

    # And the sidebar degrades instead of taking the navigation down with it.
    assert any("Not reachable" in warning.value for warning in app.warning)


def test_an_unpromoted_database_is_explained_rather_than_thrown(offline) -> None:
    """Deployed before the first promotion: the database answers, and refuses.

    Not a connection failure, so the cold-start path does not cover it, and not
    something a visitor can wait out, so it gets a message and no retry button
    rather than one that cannot help.
    """
    from streamlit.testing.v1 import AppTest

    def missing(*args, **kwargs):
        raise ProgrammingError(
            "select 1", {}, Exception('relation "gold_marts.dim_cities" does not exist')
        )

    offline(status=missing, query=missing)
    app = AppTest.from_file(APP, default_timeout=30).run()

    assert not app.exception, "a missing mart reached the visitor as a traceback"
    assert "did not" in app.error[0].value
    assert "promote.py" in app.markdown[-1].value
    assert "Try again" not in [button.label for button in app.button]
    assert any("No marts yet" in warning.value for warning in app.warning)


# --------------------------------------------------------------------------
# The deployment manifest. What Streamlit Community Cloud installs.
# --------------------------------------------------------------------------

DEPLOYMENT_REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
PIPELINE_REQUIREMENTS = PROJECT_ROOT / "requirements-pipeline.txt"

# Installed by the deployment but imported by nothing under dashboard/, so the
# import scan below cannot see them and would report them as dead weight.
IMPLICIT_DEPLOYMENT_PINS = {
    # SQLAlchemy loads it from a postgresql:// URL by name, not by import.
    "psycopg2-binary",
    # pandas requires it; pinned so Cloud resolves what the marts were built on.
    "numpy",
    # pandas reaches into scipy.stats for a rank correlation, from inside its
    # own call. No import under dashboard/ names it; the test below is what
    # holds it here.
    "scipy",
    # config.py's only dependency, reached through the local fallback.
    "python-dotenv",
}


def _pinned(path: Path) -> set[str]:
    """Distribution names pinned in a requirements file, ignoring -r includes."""
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        names.add(re.split(r"[=<>!~\[]", line, maxsplit=1)[0].strip().lower())
    return names


FIRST_PARTY = {"dashboard", "config", "cities", "ingestion", "machine_learning", "serving"}


def _imports_of(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def _third_party_imports(directory: Path) -> set[str]:
    """Third-party modules a directory needs, following its first-party imports.

    Transitive through our own modules on purpose. The map reads the city
    registry through ``cities.py``, whose own dependency is PyYAML, and a scan
    that stopped at the directory boundary would call PyYAML unused and then
    call it missing, in two different tests, for the same reason. What the
    deployment needs is what the *reachable* code imports.
    """
    seen: set[Path] = set()
    queue = list(_python_modules(directory))
    found: set[str] = set()

    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        for name in _imports_of(path):
            found.add(name)
            if name in FIRST_PARTY:
                module = PROJECT_ROOT / f"{name}.py"
                package = PROJECT_ROOT / name
                if module.exists():
                    queue.append(module)
                elif package.is_dir():
                    queue.extend(_python_modules(package))

    return found - set(sys.stdlib_module_names) - FIRST_PARTY


def test_the_deployment_manifest_covers_what_the_dashboard_imports() -> None:
    """A dependency left behind fails on Cloud, not here, unless this runs.

    BI-03 adds Plotly to a view. Without this, the first anyone hears of it is
    a ``ModuleNotFoundError`` on a public URL after a deploy that reported
    success, because the build installs fine and only the page breaks.
    """
    from importlib.metadata import packages_distributions

    mapping = packages_distributions()
    pinned = _pinned(DEPLOYMENT_REQUIREMENTS)

    missing = []
    for module in sorted(_third_party_imports(DASHBOARD_DIR)):
        distributions = mapping.get(module)
        if not distributions:  # not installed here; nothing to resolve it to
            continue
        if not any(name.lower() in pinned for name in distributions):
            missing.append(f"{module} (provided by {'/'.join(distributions)})")

    assert not missing, (
        "imported by the dashboard but not pinned in requirements.txt: "
        + ", ".join(missing)
    )


RANK_CORRELATIONS = {"spearman", "kendall"}


def test_the_backend_of_a_lazy_import_is_pinned_too() -> None:
    """The scan above reads import statements. This one reads what pandas does.

    ``Series.corr(method="spearman")`` implements nothing itself: it reaches
    into ``scipy.stats`` from inside pandas, at call time, so no walk over
    dashboard/ will ever see the name. The storm scatter ranks rather than
    fits, which is the whole point of the V-shape, so every load of that page
    runs it, and a manifest without scipy installs clean, deploys green, and
    dies on the visitor's screen with a redacted traceback.
    """
    askers = sorted(
        str(path.relative_to(PROJECT_ROOT))
        for path in _python_modules(DASHBOARD_DIR)
        if RANK_CORRELATIONS
        & {
            node.value
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
    )
    if not askers:  # the day nothing ranks, the pin can go with it
        return

    assert "scipy" in _pinned(DEPLOYMENT_REQUIREMENTS), (
        "a rank correlation in " + ", ".join(askers) + " with no scipy pinned"
    )
    # And that the pin resolves to something that answers, rather than to a
    # line in a file: this call is the one the page makes.
    assert pd.Series([1.0, 2.0, 3.0]).corr(
        pd.Series([2.0, 1.0, 4.0]), method="spearman"
    ) == pytest.approx(0.5)


def test_the_warehouse_toolchain_stays_out_of_the_deployment() -> None:
    """The other half of the split, and the half that quietly re-merges.

    The dashboard reads finished marts: it does not ingest, run dbt, or score.
    Adding dbt-core back to satisfy an import in a *pipeline* module would put
    a warehouse toolchain on the critical path of every deploy, for nothing.
    """
    deployment = _pinned(DEPLOYMENT_REQUIREMENTS)
    pipeline = _pinned(PIPELINE_REQUIREMENTS)
    excluded = {"dbt-core", "dbt-postgres", "matplotlib", "requests", "tenacity"}

    assert not (excluded & deployment), (
        "back in the deployment: " + ", ".join(sorted(excluded & deployment))
    )
    # And still installed somewhere, so the split moved them rather than
    # dropping them and quietly breaking `dbt build`.
    assert excluded <= pipeline, (
        "dropped entirely: " + ", ".join(sorted(excluded - pipeline))
    )


def test_every_deployment_pin_earns_its_place() -> None:
    """Nothing in the manifest that no page needs, directly or by name."""
    from importlib.metadata import packages_distributions

    mapping = packages_distributions()
    needed = {
        name.lower()
        for module in _third_party_imports(DASHBOARD_DIR)
        for name in mapping.get(module, ())
    }
    unexplained = _pinned(DEPLOYMENT_REQUIREMENTS) - needed - {
        pin.lower() for pin in IMPLICIT_DEPLOYMENT_PINS
    }
    assert not unexplained, (
        "pinned for the deployment but imported by nothing: " + ", ".join(sorted(unexplained))
    )


# --------------------------------------------------------------------------
# BI-03: the Global Anomaly Map. What a number turns into.
# --------------------------------------------------------------------------


def test_the_colour_break_is_the_warehouse_s_flag() -> None:
    """The map and the mart must agree on what counts as an anomaly.

    dbt flags ``abs(z) > 2.5``, strictly. Binning the palette at ``>=`` would
    paint a Z of exactly 2.5 in a flagged colour, and the picture would
    contradict the table on the one value where the question is live. Checked
    across the range rather than at the boundary alone, because an off-by-one
    here is invisible everywhere except at a single point.
    """
    outermost = {0, 1, len(theme.DIVERGING["dark"]) - 2, len(theme.DIVERGING["dark"]) - 1}
    for hundredths in range(-600, 601):
        z = hundredths / 100
        painted_as_anomaly = theme.anomaly_step(z) in outermost
        flagged_by_dbt = abs(z) > theme.ANOMALY_Z_THRESHOLD
        assert painted_as_anomaly is flagged_by_dbt, f"disagreement at Z {z}"


def test_the_step_boundary_sits_exactly_on_the_threshold() -> None:
    """Named separately, because it is the value the last test would lose."""
    assert theme.anomaly_step(2.50) == theme.anomaly_step(2.49)
    assert theme.anomaly_step(2.51) != theme.anomaly_step(2.50)
    assert theme.ANOMALY_Z_THRESHOLD in theme.ANOMALY_BREAKS


def test_colour_carries_direction_and_only_direction() -> None:
    """Equal and opposite Z-scores take mirrored steps, never the same one."""
    middle = theme.NEUTRAL_INDEX
    for z in (0.9, 2.0, 3.0, 4.5):
        assert theme.anomaly_step(z) + theme.anomaly_step(-z) == 2 * middle
        assert theme.anomaly_step(z) > middle > theme.anomaly_step(-z)
    assert theme.anomaly_step(0.0) == middle


def test_size_encodes_magnitude_by_area_not_radius() -> None:
    """A doubled anomaly must not look four times the size.

    Area proportional to |Z| means diameter goes as its square root. Encoding
    magnitude on the radius is the classic bubble-chart lie, and it exaggerates
    exactly the values a reader is most likely to quote.
    """
    floor, cap = theme.MARKER_MIN_PX, theme.MARKER_Z_CAP

    def area_above_floor(z: float) -> float:
        return theme.marker_diameter(z) ** 2 - floor**2

    # Compared as ratios rather than absolutes: the floor offsets both.
    doubled = area_above_floor(2.0) - area_above_floor(1.0)
    assert theme.marker_diameter(2.0) < 2 * theme.marker_diameter(1.0)
    assert doubled > 0

    assert theme.marker_diameter(0.0) == pytest.approx(floor)
    assert theme.marker_diameter(cap) == pytest.approx(theme.MARKER_MAX_PX)
    assert theme.marker_diameter(cap * 3) == pytest.approx(theme.MARKER_MAX_PX)

    magnitudes = [i / 20 for i in range(0, 120)]
    sizes = [theme.marker_diameter(z) for z in magnitudes]
    assert sizes == sorted(sizes), "size is not monotone in |Z|"
    assert all(theme.marker_diameter(z) == theme.marker_diameter(-z) for z in magnitudes)


def test_the_smallest_marker_still_clears_the_mark_minimum() -> None:
    """8px is the floor below which a dot stops being a mark."""
    assert theme.MARKER_MIN_PX >= 8


def test_the_map_reads_the_dark_ramp_whatever_the_page_theme_is() -> None:
    """The ramp follows the surface it is painted on, not the page around it.

    The basemap is dark in both themes, so a light-mode ramp would be one
    validated against a background that is not on screen.
    """
    assert theme.MAP_MODE == "dark"
    assert theme.anomaly_colour(3.0) in theme.DIVERGING["dark"]
    ground = theme.map_chrome()
    assert ground["ocean"] == theme.chrome("dark")["plane"]
    assert ground["ring"] == theme.chrome("dark")["ink_muted"]


def test_the_marker_ring_is_what_makes_a_quiet_city_visible() -> None:
    """The neutral fill does not separate from the land; the ring does.

    This is the measurement the ring exists for, and it is asserted rather
    than commented because the day someone "tidies" the ring away is the day
    every unremarkable city vanishes into the basemap.
    """
    ground = theme.map_chrome()
    neutral = theme.NEUTRAL["dark"]

    fill_vs_land = min(separation(neutral, ground["land"], d) for d in (None, *_MACHADO))
    ring_vs_land = min(separation(ground["ring"], ground["land"], d) for d in (None, *_MACHADO))

    assert fill_vs_land < SEPARATION_TARGET, "the comment about why the ring exists is stale"
    assert ring_vs_land > 20, "the ring no longer carries the mark"


# --------------------------------------------------------------------------
# BI-03: every city, every time.
# --------------------------------------------------------------------------


@pytest.fixture
def day_frame():
    """Two cities: one scored, one with nothing ingested."""
    from dashboard.views import anomaly_map

    frame = pd.DataFrame(
        [
            {"city_id": "delhi", "name": "Delhi", "country": "India",
             "latitude": 28.6, "longitude": 77.2, "observed_c": 35.0,
             "baseline_c": 28.7, "baseline_sigma": 1.32, "z": 4.79,
             "departure_c": 6.34, "is_anomaly": True,
             "baseline_observations": 465.0, "observed": True},
            {"city_id": "moscow", "name": "Moscow", "country": "Russia",
             "latitude": 55.8, "longitude": 37.6, "observed_c": None,
             "baseline_c": None, "baseline_sigma": None, "z": None,
             "departure_c": None, "is_anomaly": None,
             "baseline_observations": None, "observed": False},
            {"city_id": "london", "name": "London", "country": "United Kingdom",
             "latitude": 51.5, "longitude": -0.13, "observed_c": 18.2,
             "baseline_c": None, "baseline_sigma": None, "z": None,
             "departure_c": None, "is_anomaly": None,
             "baseline_observations": None, "observed": True},
        ]
    )
    return anomaly_map.prepare(frame, dt.date(2009, 8, 9))


def test_the_query_keeps_every_city_whether_or_not_it_was_scored() -> None:
    """A left join from the dimension, not an inner join on the fact.

    An inner join would redraw the world every time the backfill advanced, and
    a reader would have no way to tell "normal here" from "nothing ingested
    here", which are opposite statements about the same blank space.
    """
    from dashboard.views import anomaly_map

    sql = " ".join(anomaly_map._DAY_SQL.lower().split())
    assert "from gold_marts.dim_cities c" in sql
    assert "left join gold_marts.fact_weather_anomalies" in sql
    # The date filter belongs in the join, not in a where clause; moving it
    # would turn the left join back into an inner one for every other city.
    assert "and a.date_key = :day" in sql
    assert "where" not in sql.split("order by")[0]


def test_an_unscored_city_is_kept_and_marked_rather_than_dropped(day_frame) -> None:
    from dashboard.views import anomaly_map

    assert len(day_frame) == 3
    unscored = day_frame[day_frame["z"].isna()]
    assert set(unscored["name"]) == {"Moscow", "London"}
    assert unscored["colour"].isna().all(), "an unscored city was given an anomaly colour"
    assert unscored["diameter"].isna().all(), "an unscored city was given a magnitude"

    figure = anomaly_map._figure(day_frame)
    symbols = {trace.marker.symbol for trace in figure.data}
    assert "circle-open" in symbols, "absence is not carried by shape"


def test_the_two_absences_are_told_apart(day_frame) -> None:
    """Waiting on the backfill and waiting on a baseline are different states."""
    moscow = day_frame.loc[day_frame["city_id"] == "moscow", "tooltip"].iloc[0]
    london = day_frame.loc[day_frame["city_id"] == "london", "tooltip"].iloc[0]

    assert "no observation for this date" in moscow
    assert "no baseline yet" in london
    assert "18.2 °C observed" in london, "an observation we have was not shown"


def test_the_tooltip_carries_everything_the_criteria_ask_for(day_frame) -> None:
    """City, date, observed temperature, baseline mu, and the Z-score."""
    tooltip = day_frame.loc[day_frame["city_id"] == "delhi", "tooltip"].iloc[0]

    assert "Delhi" in tooltip and "India" in tooltip
    assert "09 August 2009" in tooltip
    assert "35.0 °C" in tooltip and "observed" in tooltip
    assert "28.7 °C" in tooltip and "baseline μ" in tooltip
    assert "Z +4.79" in tooltip
    assert "+6.3 °C" in tooltip and "departure" in tooltip
    assert "465 reference observations" in tooltip


def test_the_figure_paints_from_the_palette(day_frame) -> None:
    from dashboard.views import anomaly_map

    figure = anomaly_map._figure(day_frame)
    ground = theme.map_chrome()

    scored = [t for t in figure.data if t.marker.symbol != "circle-open"][0]
    assert set(scored.marker.color) <= set(theme.DIVERGING["dark"])
    assert scored.marker.line.color == ground["ring"]
    assert figure.layout.geo.landcolor == ground["land"]
    assert figure.layout.geo.oceancolor == ground["ocean"]
    assert figure.layout.showlegend is False


# --------------------------------------------------------------------------
# BI-03: the DBT-11 verification, asked of the picture.
# --------------------------------------------------------------------------


def _event(city_id="tokyo", city="Tokyo", direction="hot", date=dt.date(2018, 7, 23)):
    from dashboard.views.anomaly_map import Event

    return Event(city_id=city_id, city=city, date=date, direction=direction, description="...")


def _coverage_row(city_id, observed_days, scored_days):
    return pd.DataFrame(
        [{"city_id": city_id, "first_day": dt.date(1995, 1, 1),
          "last_day": dt.date(1998, 12, 31), "observed_days": observed_days,
          "scored_days": scored_days}]
    )


@pytest.mark.parametrize(
    "z, flagged, direction, expected",
    [
        (4.79, True, "hot", "pass"),
        (-4.79, True, "cold", "pass"),
        (1.10, False, "hot", "weak"),
        (-3.20, True, "hot", "fail"),
        (3.20, True, "cold", "fail"),
    ],
)
def test_the_verdict_reads_the_event_not_just_the_number(
    z, flagged, direction, expected
) -> None:
    """A hot event reading cold is a failure, not a pass with a large number.

    Magnitude alone would call a 3.2-sigma cold snap a successful verification
    of a documented heat wave, which is the one answer that would let a broken
    climatology through.
    """
    from dashboard.views.anomaly_map import verify

    frame = pd.DataFrame([{"city_id": "tokyo", "z": z, "is_anomaly": flagged}])
    state, sentence = verify(_event(direction=direction), frame, _coverage_row("tokyo", 1461, 1461))

    assert state == expected
    assert f"{z:+.2f}" in sentence


def test_an_uningested_event_is_pending_and_never_passes() -> None:
    """The gate's own rule, and the reason it exists.

    "A missing city skips with a reason; it does not pass". A verification
    that went green on absent data is the specific failure DBT-11 was written
    to prevent, wearing the costume of success.
    """
    from dashboard.views.anomaly_map import verify

    empty = pd.DataFrame(
        [{"city_id": "moscow", "z": None, "is_anomaly": None}]
    )
    state, sentence = verify(
        _event(city_id="moscow", city="Moscow"), empty, _coverage_row("moscow", 0, 0)
    )
    assert state == "pending"
    assert "cannot be checked yet" in sentence
    assert "nothing ingested for this city yet" in sentence


def test_pending_names_which_absence_it_is() -> None:
    """Ingested-but-unscored is not the same claim as never-ingested."""
    from dashboard.views.anomaly_map import verify

    frame = pd.DataFrame([{"city_id": "london", "z": None, "is_anomaly": None}])
    _, sentence = verify(
        _event(city_id="london", city="London"), frame, _coverage_row("london", 365, 0)
    )
    assert "is ingested, but no baseline" in sentence


def test_the_events_come_from_the_registry_rather_than_being_restated() -> None:
    """cities.yml is the fixture the gate reads; the map must read the same one."""
    from cities import load_cities

    from dashboard.views.anomaly_map import _events

    registry = {
        city.id: city.validation_event.date
        for city in load_cities()
        if city.validation_event is not None
    }
    offered = {event.city_id: event.date for event in _events()}

    assert offered == registry
    assert len(offered) == 7


# --------------------------------------------------------------------------
# BI-03: against a real warehouse. The local one: a test that bills a quota
# is a test nobody runs, and Neon holds a copy of exactly these marts.
# --------------------------------------------------------------------------


def _map_day(engine, day: dt.date) -> pd.DataFrame:
    """Run the map's own query, so the test exercises what ships."""
    from dashboard.views.anomaly_map import _DAY_SQL, prepare

    with engine.connect() as connection:
        frame = pd.read_sql_query(sa_text(_DAY_SQL), connection, params={"day": day})
    return prepare(frame, day)


@pytest.mark.parametrize(
    "event", _registry_events(), ids=lambda e: f"{e.city_id}-{e.date}"
)
def test_a_documented_extreme_lights_the_map_up(engine, event) -> None:
    """DBT-11, asked of the picture rather than of the warehouse.

    Every one of the seven currently **skips**: the daily backfill is
    quota-bound and none of these seven cities has a scored observation on its
    event date yet. That is reported rather than passed, for the reason the
    gate itself gives: a check that goes green on absent data is worse than no
    check. Each skip names what is missing, and each becomes a real assertion
    the moment the backfill reaches it, with no edit here.
    """
    from dashboard.views.anomaly_map import _CITY_COVERAGE_SQL, verify

    frame = _map_day(engine, event.date)
    assert len(frame) == 15, "the map lost a city"

    with engine.connect() as connection:
        coverage = pd.read_sql_query(sa_text(_CITY_COVERAGE_SQL), connection)

    state, sentence = verify(event, frame, coverage)
    if state == "pending":
        pytest.skip(sentence.replace("**", ""))

    assert state == "pass", sentence

    row = frame.loc[frame["city_id"] == event.city_id].iloc[0]
    # "Visibly lights up" is two channels, and both are checked: the colour is
    # one of the flagged steps, and the marker is near the top of the size
    # scale. A verdict that passed while the point stayed small and grey would
    # be true about the data and false about the map.
    outermost = {0, 1, len(theme.DIVERGING["dark"]) - 2, len(theme.DIVERGING["dark"]) - 1}
    assert theme.anomaly_step(row["z"]) in outermost
    assert row["diameter"] > theme.marker_diameter(theme.ANOMALY_Z_THRESHOLD)


def test_the_map_lights_up_on_the_strongest_anomaly_the_marts_hold(engine) -> None:
    """The mechanism, proven on data that exists.

    Not a DBT-11 event, since none of those is reachable yet, so this makes no
    claim about the climatology. It makes the claim BI-03 is responsible for:
    that a large Z-score in the mart becomes a large, pole-coloured, flagged
    marker on the map, end to end through the shipped query and encoding.
    """
    with engine.connect() as connection:
        row = connection.execute(
            sa_text(
                f"""select city_id, date_key, z_temperature_2m_mean as z
                      from {GOLD} .fact_weather_anomalies
                     where z_temperature_2m_mean is not null
                     order by abs(z_temperature_2m_mean) desc
                     limit 1""".replace(" .", ".")
            )
        ).fetchone()

    if row is None:
        pytest.skip("no scored anomalies in the warehouse yet")

    frame = _map_day(engine, row.date_key)
    assert len(frame) == 15

    hit = frame.loc[frame["city_id"] == row.city_id].iloc[0]
    outermost = {0, 1, len(theme.DIVERGING["dark"]) - 2, len(theme.DIVERGING["dark"]) - 1}

    assert bool(hit["is_anomaly"]) is True
    assert theme.anomaly_step(hit["z"]) in outermost, "the strongest day is not pole-coloured"
    assert hit["diameter"] == pytest.approx(theme.MARKER_MAX_PX, abs=1.5), (
        "the strongest day in thirty years does not reach the top of the size scale"
    )
    assert f"Z {hit['z']:+.2f}" in hit["tooltip"]


def test_the_map_plots_every_registered_city_at_its_registered_coordinates(engine) -> None:
    """Fifteen cities, at the coordinates cities.yml gives, on any date.

    Checked against the registry rather than against dim_cities, so a mistake
    that entered the warehouse would still be caught here.
    """
    from cities import load_cities

    registry = {city.id: (city.lat, city.lon) for city in load_cities()}
    frame = _map_day(engine, dt.date(2009, 8, 9))

    assert len(frame) == len(registry) == 15
    assert set(frame["city_id"]) == set(registry)
    for _, row in frame.iterrows():
        latitude, longitude = registry[row["city_id"]]
        assert row["latitude"] == pytest.approx(latitude, abs=1e-4)
        assert row["longitude"] == pytest.approx(longitude, abs=1e-4)


# --------------------------------------------------------------------------
# BI-04: the Climate Matrix. Counting is not signing.
# --------------------------------------------------------------------------

SEQUENTIAL_DIRECTIONS = ("hot", "cold")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("direction", SEQUENTIAL_DIRECTIONS)
def test_a_counting_ramp_is_one_hue_light_to_dark(mode, direction) -> None:
    """A magnitude gets a sequential ramp, checked the way the arms were.

    One hue throughout, lightness monotone, and no two adjacent steps closer
    than the 0.06 below which a step stops being a step.
    """
    steps = theme.sequential_scale(direction, mode)
    hues = [hue(step) for step in steps]
    values = [lightness(step) for step in steps]
    gaps = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]

    assert max(hues) - min(hues) < 10, "the ramp drifts in hue"
    assert values == sorted(values, reverse=(mode == "light")), "not monotone"
    assert min(gaps) >= 0.06


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("direction", SEQUENTIAL_DIRECTIONS)
def test_the_end_nearest_the_surface_still_reads_as_a_cell(mode, direction) -> None:
    """The pale end of a heatmap ramp must clear its own background.

    Unlike the diverging midpoint, which is meant to recede, the low end of a
    counting ramp is a real value that a reader has to be able to see.
    """
    steps = theme.sequential_scale(direction, mode)
    nearest = steps[0] if mode == "light" else steps[0]
    assert contrast(nearest, theme.SURFACE[mode]) >= 2.0


@pytest.mark.parametrize("mode", MODES)
def test_the_hot_and_cold_ramps_are_tellable_apart(mode) -> None:
    """A screenshot of the hot view and one of the cold view are otherwise the
    same picture. The toggle only separates them if the colours do."""
    hot = theme.sequential_scale("hot", mode)
    cold = theme.sequential_scale("cold", mode)
    worst = min(
        separation(a, b, deficiency)
        for a, b in zip(hot, cold)
        for deficiency in GATING_DEFICIENCIES
    )
    assert worst >= SEPARATION_TARGET, f"the two ramps collapse to {worst:.1f}"


def _theme_prose() -> str:
    """theme.py's docstring as one line.

    Searched with the wrapping flattened: a claim that moves across a line
    break when the paragraph is re-wrapped is the same claim, and a test that
    stopped finding it would report a missing figure rather than a changed one.
    """
    return " ".join((theme.__doc__ or "").split())


def test_the_documented_ramp_separation_cannot_go_stale() -> None:
    """theme.py quotes the number; recompute it."""
    quoted = re.search(r"separate by (\d+\.\d+) under protanopia", _theme_prose())
    assert quoted, "theme.py no longer states the figure"
    measured = min(
        separation(a, b, "protan")
        for mode in MODES
        for a, b in zip(theme.sequential_scale("hot", mode), theme.sequential_scale("cold", mode))
    )
    assert round(measured, 1) == float(quoted.group(1))


def test_a_count_never_reaches_for_the_diverging_ramp() -> None:
    """The encoding decision this view turns on.

    Zero-to-twenty has a bottom and a top and no meaningful middle. Painting it
    on two hues either side of a neutral would invent a direction the number
    does not have and park the least interesting value in the most neutral
    place. Only the net view is signed, and only it uses the diverging scale.
    """
    from dashboard.views import climate_matrix as matrix

    frame = _matrix_frame()
    for metric, expected in (
        ("hot", theme.sequential_scale("hot", "light")),
        ("cold", theme.sequential_scale("cold", "light")),
        ("net", theme.diverging_scale("light")),
    ):
        order = matrix.order_cities(frame, metric, "Name")
        figure = matrix._figure(matrix.matrix(frame, metric, order), metric, "light")
        painted = {colour for _, colour in figure.data[0].colorscale}
        assert painted == set(expected), f"{metric} is painted on the wrong ramp"


def _matrix_frame() -> pd.DataFrame:
    """Three cities: one with a long record, one short, one never ingested."""
    rows = []
    for year in range(1995, 2027):
        rows.append({"city_id": "cairo", "name": "Cairo", "year": year,
                     "hot_days": max(0, (year - 1995) // 3), "cold_days": 1,
                     "scored_days": 365})
        rows.append({"city_id": "tokyo", "name": "Tokyo", "year": year,
                     "hot_days": 5, "cold_days": 2,
                     "scored_days": 365 if year < 1999 else 0})
        rows.append({"city_id": "moscow", "name": "Moscow", "year": year,
                     "hot_days": 0, "cold_days": 0, "scored_days": 0})
    frame = pd.DataFrame(rows)
    frame["net_days"] = frame["hot_days"] - frame["cold_days"]
    frame["scored"] = frame["scored_days"] > 0
    return frame


def test_bucket_boundaries_are_the_ones_the_key_prints() -> None:
    """A key that says 3-5 and a bucket that holds 3-6 is a chart that lies."""
    from dashboard.views.climate_matrix import COUNT_LABELS, count_bucket

    assert [count_bucket(n) for n in (0, 1, 2, 3, 5, 6, 10, 11, 40)] == [
        0, 1, 1, 2, 2, 3, 3, 4, 4
    ]
    assert len(COUNT_LABELS) == len(theme.sequential_scale("hot", "light"))


def test_the_net_buckets_mirror_the_counting_ones() -> None:
    """A reader moving between views should not also learn new boundaries."""
    from dashboard.views.climate_matrix import NET_LABELS, net_bucket

    assert net_bucket(0) == theme.NEUTRAL_INDEX
    assert len(NET_LABELS) == len(theme.diverging_scale("light"))
    for value in (1, 2, 3, 5, 6, 10, 11, 30):
        assert net_bucket(value) + net_bucket(-value) == 2 * theme.NEUTRAL_INDEX
        assert net_bucket(value) > theme.NEUTRAL_INDEX


def test_zero_extremes_is_a_colour_and_absent_is_a_hole() -> None:
    """The distinction most heatmaps lose.

    A year with no extremes and a year nobody has ingested are the same shade
    of pale on most grids, and they are opposite statements.
    """
    from dashboard.views import climate_matrix as matrix

    frame = _matrix_frame()
    grid = matrix.matrix(frame, "hot", matrix.order_cities(frame, "hot", "Name"))

    quiet = grid[(grid["city_id"] == "cairo") & (grid["year"] == 1995)].iloc[0]
    absent = grid[(grid["city_id"] == "moscow") & (grid["year"] == 1995)].iloc[0]

    assert quiet["hot_days"] == 0
    assert quiet["bucket"] == pytest.approx(0.5), "a quiet year lost its colour"
    assert pd.isna(absent["bucket"]), "an un-ingested year was painted"
    assert "not ingested" in absent["cell_text"]
    assert "0</b> hot days" in quiet["cell_text"]


def test_the_grid_is_complete_whatever_the_fact_holds() -> None:
    """Every city crossed with every year, from the dimension outward.

    Aggregating the fact alone returns only the city-years that have rows, and
    the heatmap would silently change shape as the backfill advanced.
    """
    from dashboard.views.climate_matrix import _MATRIX_SQL

    sql = " ".join(_MATRIX_SQL.lower().split())
    assert "cross join years" in sql
    assert "from gold_marts.dim_cities c" in sql
    assert "left join gold_marts.fact_weather_anomalies" in sql
    # Grouped in the database. The fact is 60k rows and the answer is 480;
    # grouping locally would pay for the same arithmetic twice, once in
    # bandwidth and once against the read budget.
    assert "group by" in sql and "filter (" in sql


def test_a_trend_needs_enough_years_to_be_one() -> None:
    """Four points and thirty-two points are different quantities.

    Ranking them together would seat a city at the top of the chart on the
    strength of a coincidence.
    """
    from dashboard.views import climate_matrix as matrix

    trend = matrix.trends(_matrix_frame(), "hot")

    assert trend["Cairo"] > 0, "a rising record did not read as rising"
    assert np.isnan(trend["Tokyo"]), "four scored years produced a trend"
    assert np.isnan(trend["Moscow"]), "a city with no data produced a trend"


def test_the_trend_is_reported_per_decade() -> None:
    """Per year reads as a column of zeroes and invites the wrong conclusion."""
    from dashboard.views import climate_matrix as matrix

    frame = _matrix_frame()
    rising = frame[(frame["name"] == "Cairo")]
    slope_per_year = np.polyfit(
        rising["year"].to_numpy(float), rising["hot_days"].to_numpy(float), 1
    )[0]
    assert matrix.trends(frame, "hot")["Cairo"] == pytest.approx(slope_per_year * 10)


@pytest.mark.parametrize("sort", ["Trend", "Total", "Name"])
def test_cities_with_nothing_to_say_sort_to_the_bottom(sort) -> None:
    """Nine empty rows interleaved with six full ones is an unreadable chart."""
    from dashboard.views import climate_matrix as matrix

    frame = _matrix_frame()
    # Plotly counts its y axis upward, so reading order is the reverse.
    reading_order = matrix.order_cities(frame, "hot", sort)[::-1]
    assert reading_order[-1] == "Moscow", f"{sort} interleaved an empty city"


def test_the_default_order_is_the_trend_not_the_alphabet() -> None:
    """"Sortable by trend so the pattern is legible rather than alphabetical"."""
    from dashboard.views import climate_matrix as matrix

    frame = _matrix_frame()
    by_trend = matrix.order_cities(frame, "hot", "Trend")[::-1]
    by_name = matrix.order_cities(frame, "hot", "Name")[::-1]

    assert by_trend[0] == "Cairo", "the steepest riser is not on top"
    assert by_name == ["Cairo", "Tokyo", "Moscow"]


def test_the_matrix_reads_from_the_cache_not_the_warehouse() -> None:
    """The three-second budget is a cache-hit budget, and this is what makes it one."""
    import inspect

    from dashboard.views import climate_matrix as matrix

    source = inspect.getsource(matrix.load)
    assert "run_query" in source, "the matrix bypasses the cached query layer"


def test_the_matrix_query_returns_a_bounded_grid(engine) -> None:
    """Fifteen cities by the ingested span, and nothing larger.

    A heatmap that grew a row per city-day would still render and would still
    be under three seconds on a cache hit, and would be sending sixty
    thousand rows over the wire to fill four hundred cells.
    """
    from dashboard.views.climate_matrix import _MATRIX_SQL

    with engine.connect() as connection:
        frame = pd.read_sql_query(sa_text(_MATRIX_SQL), connection)

    cities, years = frame["city_id"].nunique(), frame["year"].nunique()
    assert cities == 15
    assert len(frame) == cities * years
    assert years >= 30, f"only {years} years; the 30-year range is not rendered"
    assert set(frame.columns) >= {"hot_days", "cold_days", "scored_days"}


def test_every_registered_city_has_a_row_even_with_nothing_ingested(engine) -> None:
    from cities import load_cities

    from dashboard.views.climate_matrix import _MATRIX_SQL

    with engine.connect() as connection:
        frame = pd.read_sql_query(sa_text(_MATRIX_SQL), connection)

    assert set(frame["city_id"]) == {city.id for city in load_cities()}
    never = frame.groupby("city_id")["scored_days"].sum()
    assert (never == 0).any(), "the fixture for the absent case has gone stale"
    assert (never > 0).any()


# --------------------------------------------------------------------------
# BI-05: Storm Dynamics. Where colour stops being able to carry identity.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_the_emphasis_accent_separates_from_the_context_it_sits_in(mode) -> None:
    """A highlighted point has to read as picked out of the grey field.

    Hue alone will not do it: a colour at the context ink's own lightness
    collapses toward it once a CVD simulation flattens the chroma, which is why
    the accent is moved in lightness as well as hue.
    """
    accent = theme.emphasis(mode)
    context = theme.chrome(mode)["ink_muted"]
    worst = min(separation(accent, context, d) for d in (None, *_MACHADO))

    assert worst >= 15, f"the accent collapses into the context at {worst:.1f}"
    assert contrast(accent, theme.SURFACE[mode]) >= 3.0


@pytest.mark.parametrize("mode", MODES)
def test_the_accent_does_not_borrow_a_hue_that_already_means_something(mode) -> None:
    """Green means "the city you picked" and nothing else in this dashboard.

    A colour that means "cold" on two views and "Reykjavík" on a third is one
    colour doing two jobs, and the reader has no way to know which.
    """
    accent_hue = hue(theme.emphasis(mode))
    for reserved in (hue(theme.DIVERGING[mode][0]), hue(theme.DIVERGING[mode][-1])):
        gap = abs(accent_hue - reserved) % 360
        assert min(gap, 360 - gap) >= 60, "the accent sits on an anomaly hue"


def test_the_documented_accent_separations_cannot_go_stale() -> None:
    """theme.py quotes both figures; recompute them."""
    prose = _theme_prose()
    quoted = re.search(
        r"context ink by (\d+\.\d+) in light mode and (\d+\.\d+) in dark", prose
    )
    assert quoted, "theme.py no longer states the separations"
    for index, mode in enumerate(("light", "dark")):
        measured = min(
            separation(theme.emphasis(mode), theme.chrome(mode)["ink_muted"], d)
            for d in (None, *_MACHADO)
        )
        assert round(measured, 1) == float(quoted.group(index + 1))


def test_colour_is_not_asked_to_carry_fifteen_identities() -> None:
    """The finding that shaped this view, asserted rather than trusted.

    Every trace is either the grey context or the single accent. A future edit
    that started colouring by city would seat fifteen hues in a scatter, which
    no palette supports and which the module docstring says was ruled out by
    search rather than by taste.
    """
    from dashboard.views import storm_dynamics as storm

    frame = _storm_frame()
    for selected in (storm.ALL_CITIES, "Reykjavík"):
        figure = storm._figure(frame, selected, "light")
        painted = {trace.marker.color for trace in figure.data}
        assert painted <= {theme.chrome("light")["ink_muted"], theme.emphasis("light")}
        assert len(figure.data) <= 2, "a trace per city has crept in"


def _storm_frame() -> pd.DataFrame:
    """Two cities whose relationships differ, like the real ones do."""
    rows = []
    for day in range(120):
        swing = (day % 30) - 15
        rows.append({"city_id": "reykjavik", "name": "Reykjavík",
                     "date_key": f"2025-01-{day % 28 + 1:02d}",
                     "pressure_change_24h": float(swing),
                     "peak_gust": 30.0 + 3.0 * abs(swing), "hours": 24})
        rows.append({"city_id": "singapore", "name": "Singapore",
                     "date_key": f"2025-01-{day % 28 + 1:02d}",
                     "pressure_change_24h": float(swing) / 5,
                     "peak_gust": 25.0 + (day % 7), "hours": 24})
    frame = pd.DataFrame(rows)
    frame["swing"] = frame["pressure_change_24h"].abs()
    return frame


def test_the_v_shape_is_why_the_signed_axis_is_kept() -> None:
    """Signed reads as nothing; magnitude reads as the relationship.

    This is the whole reason the caption quotes two numbers. A view that
    reported only the signed coefficient would tell a reader there is no
    relationship between pressure and wind, which is false.
    """
    from dashboard.views import storm_dynamics as storm

    windy = _storm_frame()
    windy = windy[windy["name"] == "Reykjavík"]
    pooled = storm.overall(windy)

    assert abs(pooled["signed"]) < 0.15, "the fixture is not V-shaped"
    assert pooled["magnitude"] > 0.9, "magnitude does not recover the relationship"


def test_the_correlation_is_ranked_not_least_squares() -> None:
    """Gust distributions have a long right tail.

    Pearson would let four storms set a city's coefficient; Spearman asks the
    question the caption asks, which is whether the wind ranks higher when the
    barometer moves more.
    """
    from dashboard.views import storm_dynamics as storm

    assert storm.CORRELATION_METHOD == "spearman"

    frame = _storm_frame()
    outlier = frame.copy()
    outlier.loc[outlier.index[0], "peak_gust"] = 10_000.0

    before = storm.correlations(frame)["Reykjavík"]
    after = storm.correlations(outlier)["Reykjavík"]
    assert abs(before - after) < 0.05, "one freak day moved the coefficient"


def test_every_city_gets_its_own_coefficient() -> None:
    """One pooled number would hide that the answer depends on the city."""
    from dashboard.views import storm_dynamics as storm

    per_city = storm.correlations(_storm_frame())
    assert set(per_city.index) == {"Reykjavík", "Singapore"}
    assert per_city.index[0] == "Reykjavík", "not sorted strongest first"
    assert per_city["Reykjavík"] > per_city["Singapore"]


def test_the_axes_are_the_ones_the_ticket_names() -> None:
    from dashboard.views import storm_dynamics as storm

    figure = storm._figure(_storm_frame(), storm.ALL_CITIES, "light")
    assert "24-hour pressure change" in figure.layout.xaxis.title.text
    assert "hPa" in figure.layout.xaxis.title.text
    assert "Peak gust" in figure.layout.yaxis.title.text
    assert "km/h" in figure.layout.yaxis.title.text


def test_the_hourly_fact_is_aggregated_in_the_warehouse() -> None:
    """A quarter of a million points cannot be drawn, and should not be sent.

    Grouping to city-day is a 24-fold reduction that loses nothing the question
    needs: "did this day have a pressure crash and a gale" is a question about
    a day.
    """
    from dashboard.views.storm_dynamics import _DAILY_SQL

    sql = " ".join(_DAILY_SQL.lower().split())
    assert "fact_weather_hourly" in sql
    assert "group by city_id, date_key" in sql
    assert "max(wind_gusts_10m)" in sql
    assert "pressure_tendency_24h" in sql
    # The count is measured rather than inferred from the day count.
    assert "count(*)" in sql and "as hours" in sql


def test_the_kept_swing_is_the_larger_of_the_two_limbs() -> None:
    """Keeping only falls would show one limb of the V and hide the other.

    The rise behind a departing low is windy too, and that is half the physical
    story the chart is meant to tell.
    """
    from dashboard.views.storm_dynamics import _DAILY_SQL

    sql = " ".join(_DAILY_SQL.lower().split())
    assert "abs(d.sharpest_fall) >= abs(d.sharpest_rise)" in sql
    assert "min(pressure_tendency_24h)" in sql and "max(pressure_tendency_24h)" in sql


def test_storm_dynamics_reads_the_complete_mart(engine) -> None:
    """The one mart that has all fifteen cities, so this view has no gaps.

    Asserted rather than assumed: if the hourly backfill ever became partial,
    this view would quietly start answering a question about a subset.
    """
    from cities import load_cities

    from dashboard.views.storm_dynamics import _DAILY_SQL

    with engine.connect() as connection:
        frame = pd.read_sql_query(sa_text(_DAILY_SQL), connection)

    assert set(frame["city_id"]) == {city.id for city in load_cities()}
    assert frame["peak_gust"].notna().all()
    assert frame["pressure_change_24h"].notna().all()
    # One row per city-day, which is what makes the point count bounded.
    assert not frame.duplicated(["city_id", "date_key"]).any()


def test_the_reported_relationship_is_the_one_the_data_has(engine) -> None:
    """The caption's claim, recomputed against the warehouse.

    The caption is generated from the frame on screen, so it cannot go stale on
    its own, but the *shape* of the claim can: if the signed correlation ever
    became strong, the sentence about a V would be wrong while still being
    arithmetically correct.
    """
    from dashboard.views import storm_dynamics as storm
    from dashboard.views.storm_dynamics import _DAILY_SQL

    with engine.connect() as connection:
        frame = pd.read_sql_query(sa_text(_DAILY_SQL), connection)
    frame["swing"] = frame["pressure_change_24h"].abs()

    pooled = storm.overall(frame)
    assert abs(pooled["signed"]) < 0.15, (
        "the signed relationship is no longer negligible, so the caption's "
        "explanation of the V no longer describes the data"
    )
    assert pooled["magnitude"] > pooled["signed"] + 0.2, (
        "magnitude no longer recovers what the sign hides"
    )

    per_city = storm.correlations(frame)
    assert len(per_city) == 15
    assert per_city.iloc[0] - per_city.iloc[-1] > 0.2, (
        "the spread across cities has collapsed, so the second caption claims a "
        "variation the data no longer shows"
    )


# --------------------------------------------------------------------------
# BI-06: Risk Horizon. Where the model becomes a picture.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_the_risk_ramp_is_a_ramp(mode) -> None:
    """A probability is a magnitude, so it gets the same checks as a count."""
    steps = theme.risk_scale(mode)
    hues = [hue(step) for step in steps]
    values = [lightness(step) for step in steps]
    gaps = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]

    assert max(hues) - min(hues) < 10
    assert values == sorted(values, reverse=(mode == "light"))
    assert min(gaps) >= 0.06
    assert contrast(steps[0], theme.SURFACE[mode]) >= 2.0
    assert contrast(steps[-1], theme.SURFACE[mode]) >= 3.0


def test_the_step_boundary_is_the_models_own_comparison() -> None:
    """``prediction_label = (risk_score >= decision_threshold)``, greater or *equal*.

    The anomaly flag two views away is a strict ``>`` and this one is not. The
    two conventions differ, each follows its own table, and getting it wrong
    here would paint a city as flagged on the exact value where the warehouse
    says it is not, or the reverse.
    """
    threshold = 0.1024
    outermost = {3, 4}
    for thousandths in range(0, 1001):
        score = thousandths / 1000
        painted_as_flagged = theme.risk_step(score, threshold) in outermost
        model_says_yes = score >= threshold
        assert painted_as_flagged is model_says_yes, f"disagreement at {score}"

    assert theme.risk_step(threshold, threshold) == 3
    assert theme.risk_step(threshold - 1e-9, threshold) == 2


def test_the_risk_breaks_follow_the_threshold_rather_than_fixed_probabilities() -> None:
    """The threshold is chosen on validation and moves when the model is retrained.

    Breaks pinned to absolute probabilities would quietly stop lining up with
    it, and the one boundary that matters would drift off the legend.
    """
    assert 1.0 in theme.RISK_BREAKS
    for threshold in (0.02, 0.1024, 0.4):
        assert theme.risk_step(threshold, threshold) == theme.RISK_BREAKS.index(1.0) + 1
        assert theme.risk_step(threshold * 0.99, threshold) < 3

    with pytest.raises(ValueError):
        theme.risk_step(0.5, 0.0)


def test_the_ramp_collision_is_the_one_theme_py_admits() -> None:
    """Violet against blue under protanopia, and why it was chosen anyway.

    theme.py argues the alternative is worse and gives three numbers. All three
    are recomputed, because an argument from measurements that no longer hold
    is just an assertion.
    """
    prose = _theme_prose()
    quoted = re.search(
        r"measured as \*\*(\d+\.\d+)\*\* from the cold counting ramp", prose
    )
    assert quoted, "theme.py no longer states the collision"

    worst = min(
        separation(a, b, deficiency)
        for mode in MODES
        for a, b in zip(theme.risk_scale(mode), theme.sequential_scale("cold", mode))
        for deficiency in GATING_DEFICIENCIES
    )
    assert round(worst, 1) == float(quoted.group(1))
    assert worst < SEPARATION_TARGET, "the admitted collision is gone; fix the prose"


def test_the_rejected_alternative_is_still_the_worse_one() -> None:
    """An achromatic ramp clears the cold ramp and collides with the muted ink.

    That trade was the whole argument: a reader who cannot separate "no
    prediction" from "low risk" *in one picture* is worse off than one who
    could confuse two ramps that never share a page.
    """
    achromatic = {
        "light": ("#d2d0cb", "#b3b1ab", "#94928c", "#75736e", "#575551"),
        "dark": ("#3a3a37", "#565450", "#73716b", "#918f88", "#b0aea6"),
    }
    for mode in MODES:
        context = theme.chrome(mode)["ink_muted"]
        grey_vs_context = min(
            separation(step, context, d) for step in achromatic[mode] for d in (None, *_MACHADO)
        )
        violet_vs_context = min(
            separation(step, context, d)
            for step in theme.risk_scale(mode)
            for d in (None, *_MACHADO)
        )
        assert grey_vs_context < violet_vs_context, (
            "the achromatic ramp is no longer the worse same-page choice"
        )


# --------------------------------------------------------------------------
# BI-06: the grid, and the claim it is allowed to make.
# --------------------------------------------------------------------------


def _risk_frame(scored=("Singapore", "Lagos"), absent=("Moscow",)) -> pd.DataFrame:
    rows = []
    for index, name in enumerate(scored):
        rows.append({
            "city_id": name.lower(), "name": name, "country": "X",
            "forecast_date": dt.date(2026, 9, 2),
            "horizon_start": dt.date(2026, 9, 3), "horizon_end": dt.date(2026, 9, 9),
            "horizon_days": 7, "risk_score": 0.25 - 0.15 * index,
            "prediction_label": True, "decision_threshold": 0.1024,
            "model_version": "model-unweighted-v1-2ad772ff7b18",
            "model_variant": "unweighted", "feature_count": 27,
            "scored_at": pd.Timestamp("2026-09-07T09:59:25Z"),
        })
    for name in absent:
        rows.append({
            "city_id": name.lower(), "name": name, "country": "Y",
            "forecast_date": None, "horizon_start": None, "horizon_end": None,
            "horizon_days": None, "risk_score": None, "prediction_label": None,
            "decision_threshold": None, "model_version": None,
            "model_variant": None, "feature_count": None, "scored_at": None,
        })
    return pd.DataFrame(rows)


def test_a_week_is_drawn_as_a_band_not_as_seven_estimates() -> None:
    """The honesty this view turns on.

    The model's target is "an anomaly at any point in the next seven days", so
    there is one probability per city per week and no per-day resolution
    underneath it. Every cell in a row therefore carries the same value;
    varying them would be a chart claiming precision the model does not have.
    """
    from dashboard.views import risk_horizon as risk

    frame = _risk_frame()
    days = risk.horizon_days(frame)
    assert len(days) == 7

    _, steps, _ = risk.grid(frame, days)
    for row in steps:
        assert len(row) == 7
        assert len(set(row)) == 1, "a row varies across the week the model scored as one"


def test_no_internal_boundary_splits_the_week() -> None:
    """A row is one continuous bar; only rows are separated.

    A gap between the seven cells would draw seven statements where the model
    made one, and no caption undoes what the grid lines say.
    """
    from dashboard.views import risk_horizon as risk

    frame = _risk_frame()
    figure = risk._figure(frame, risk.horizon_days(frame), "light")
    heatmap = figure.data[0]

    assert heatmap.xgap == 0, "the week has been split into seven cells"
    assert heatmap.ygap > 0, "the cities have run together"


def test_every_registered_city_keeps_its_row() -> None:
    """The model scores five of fifteen; showing five would present them as the world."""
    from dashboard.views.risk_horizon import _LATEST_SQL

    sql = " ".join(_LATEST_SQL.lower().split())
    assert "from gold_marts.dim_cities c" in sql
    assert "left join gold_marts.fact_ml_predictions p" in sql
    assert "max(forecast_date)" in sql


def test_an_unscored_city_is_empty_rather_than_grey() -> None:
    """Absence carries no fill at all, which is what lets the ramp be violet.

    A grey fill for "no prediction" would sit 3.7 from the muted ink and land
    in the same picture as the ramp: the same-page collision theme.py rejected
    the achromatic ramp to avoid.
    """
    from dashboard.views import risk_horizon as risk

    frame = _risk_frame()
    _, steps, _ = risk.grid(frame, risk.horizon_days(frame))
    assert steps[-1] == [None] * 7


def test_the_reason_a_city_is_missing_comes_from_the_model_not_a_guess() -> None:
    """Three states, taken from the evaluation record rather than inferred.

    "No row in the predictions table" is one observation with three different
    causes, and only the model knows which.

    Which city is in which state is not asserted, because it moves: Moscow was
    uningested when this was written and now has three days of record, so it has
    crossed from the first reason to the second without anything in the view
    changing. What has to hold is that the two lists partition the unscored
    cities and that every name in them comes back with the record's own reason
    rather than a guess assembled from missing rows.
    """
    from dashboard.views import risk_horizon as risk

    evaluation = risk.model_report()["evaluation"]
    absent = set(evaluation["cities_not_ingested"])
    unscored = set(evaluation["cities_ingested_but_not_scored"])
    reasons = risk.absence_reasons()

    assert reasons, "the committed evaluation record has no absence account"
    assert not absent & unscored, "a city cannot be both uningested and ingested"
    assert not (absent | unscored) & set(evaluation["cities_scored"])
    assert set(reasons) == absent | unscored
    assert {reasons[city] for city in absent} <= {"not ingested"}
    assert {reasons[city] for city in unscored} <= {
        "ingested, but not enough history to score"
    }


def test_the_tooltip_says_the_score_covers_the_whole_window() -> None:
    """With no per-day model, the cell still says it is one score for a week.

    BI-09 gives a covered city seven cells with seven numbers. A city the
    hazard does not cover keeps the band, and its tooltip has to keep saying
    so -- a flat row that looked like seven answers would be the overclaim the
    band existed to avoid.
    """
    from dashboard.views import risk_horizon as risk

    frame = _risk_frame()
    days = risk.horizon_days(frame)
    _, _, texts = risk.grid(frame, days)

    scored = texts[0][3]
    assert "for the week" in scored
    assert "threshold" in scored
    assert "one score for 7 days" in scored
    assert "03 Sep - 09 Sep" in scored
    assert "No per-day model covers this city" in scored


def test_a_covered_city_gets_seven_numbers_and_says_where_they_came_from() -> None:
    """The per-day cells, and the caveat that has to travel with them.

    The day and the week come from two different models and do not compose to
    each other -- ML-13 measured Lagos's days composing to 0.48 against a band
    of 0.40. A reader multiplying seven cells together is entitled to know that
    before they wonder why it does not add up, so every covered cell says it.
    """
    from dashboard.views import risk_horizon as risk

    frame = _risk_frame()
    days = risk.horizon_days(frame)
    # Spanning the ramp's breaks, which sit at 0.25x to 2x the *daily*
    # equivalent of the weekly threshold -- 0.0154 for this fixture. Values all
    # above 2x would every one land in the top step, which is what the real
    # Lagos and London do and is a true statement about them, but it would make
    # this test pass whatever grid() did with the ordering.
    hazards = pd.DataFrame(
        {
            "city_id": [frame.iloc[0]["city_id"]] * len(days),
            "horizon_day": range(1, len(days) + 1),
            "risk_score": [0.200, 0.030, 0.015, 0.008, 0.004, 0.004, 0.004],
        }
    )
    _, steps, texts = risk.grid(frame, days, hazards)

    covered = texts[0]
    assert "on this day" in covered[0]
    assert "do not compose to each other" in covered[0]
    assert "0.200" in covered[0] and "0.004" in covered[-1]

    # Seven cells, ordered as the hazards are, and the first differs from the
    # tail: the shape ML-13 found.
    assert len(steps[0]) == len(days)
    assert steps[0] == sorted(steps[0], reverse=True)
    assert steps[0][0] > steps[0][-1]
    # An uncovered city keeps one value across the row.
    assert len(set(steps[1])) == 1


def test_the_vintage_is_on_the_page() -> None:
    """"so the reader knows the vintage": the model, and when it ran."""
    from dashboard.views import risk_horizon as risk

    stamp = risk.vintage(_risk_frame())
    assert stamp["model_version"].startswith("model-")
    assert stamp["model_variant"] in {"weighted", "unweighted"}
    assert stamp["scored_at"] is not None
    assert stamp["feature_count"] == 27
    assert 0 < stamp["threshold"] < 1


def test_the_drivers_are_read_as_data_not_imported_as_a_model() -> None:
    """SHAP from the committed record, so the deployment ships no XGBoost.

    A per-cell attribution would need the estimator and the feature matrix at
    request time, and the dashboard has neither by design.
    """
    from dashboard.views import risk_horizon as risk

    drivers = risk.top_drivers()
    assert not drivers.empty
    assert list(drivers.columns)[:1] == ["feature"]
    assert drivers["mean_abs"].is_monotonic_decreasing
    assert drivers.iloc[0]["feature"] == "z_temperature_2m_mean"


def test_the_dashboard_imports_no_machine_learning_runtime() -> None:
    """The constraint that keeps the deployment small, asserted over the source.

    Importing `machine_learning` anywhere under `dashboard/` would pull XGBoost
    and scikit-learn into an app whose only job is reading finished rows.
    """
    forbidden = {"xgboost", "sklearn", "scikit_learn", "joblib", "shap", "machine_learning"}
    offenders = []
    for path in _python_modules(DASHBOARD_DIR):
        for name in _imports_of(path):
            if name in forbidden:
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}: {name}")
    assert not offenders, "ML runtime reached the dashboard: " + ", ".join(offenders)


def test_the_caption_refuses_to_be_mistaken_for_a_forecast() -> None:
    """The checklist asks for it, and it is the most important sentence here.

    This is the view a non-technical reader will screenshot.
    """
    from dashboard.views.risk_horizon import DISCLAIMER

    lowered = DISCLAIMER.lower()
    assert "demonstration model" in lowered
    assert "not an operational forecast" in lowered
    assert "numerical weather prediction" in lowered


def test_the_risk_grid_is_fifteen_by_seven_against_the_warehouse(engine) -> None:
    """The checklist's shape, on the real table."""
    from cities import load_cities

    from dashboard.views import risk_horizon as risk
    from dashboard.views.risk_horizon import _LATEST_SQL

    with engine.connect() as connection:
        frame = pd.read_sql_query(sa_text(_LATEST_SQL), connection)

    assert set(frame["city_id"]) == {city.id for city in load_cities()}

    days = risk.horizon_days(frame)
    if not days:
        pytest.skip("no predictions in the warehouse yet")

    names, steps, texts = risk.grid(frame, days)
    assert len(names) == 15
    assert len(days) == 7
    assert all(len(row) == 7 for row in steps)

    scored = frame[frame["risk_score"].notna()]
    assert not scored.empty
    # Painted-as-flagged and the stored label must be the same rows.
    for _, row in scored.iterrows():
        painted = theme.risk_step(
            float(row["risk_score"]), float(row["decision_threshold"])
        ) in {3, 4}
        assert painted is bool(row["prediction_label"]), row["city_id"]


def test_the_warehouse_agrees_with_the_committed_evaluation_record(engine) -> None:
    """The cities the model says it scored are the cities that have rows.

    Two records of the same fact, written by different steps on different days.
    If they disagree, the absence reasons on screen are about a different run
    than the numbers beside them.
    """
    from dashboard.views import risk_horizon as risk
    from dashboard.views.risk_horizon import _LATEST_SQL

    report = risk.model_report()
    claimed = set(report.get("evaluation", {}).get("cities_scored", []))
    if not claimed:
        pytest.skip("no evaluation record committed")

    with engine.connect() as connection:
        frame = pd.read_sql_query(sa_text(_LATEST_SQL), connection)

    have_rows = set(frame.loc[frame["risk_score"].notna(), "city_id"])
    assert have_rows == claimed, (
        f"predictions and metrics.json disagree: {have_rows ^ claimed}"
    )


# --------------------------------------------------------------------------
# BI-07: what the deployment needs, asserted before it is deployed.
# --------------------------------------------------------------------------


def _tracked_files() -> set[str]:
    import subprocess

    listed = subprocess.run(
        ["git", "ls-files"], cwd=PROJECT_ROOT, capture_output=True, text=True
    )
    return set(listed.stdout.split())


def test_every_file_the_dashboard_reads_at_runtime_is_committed() -> None:
    """The classic deployment failure: works here, missing there.

    Community Cloud clones the repository and runs it. A file that exists on
    this machine but is git-ignored is invisible to that clone, and the failure
    lands as a traceback on a public URL rather than here. The paths are
    resolved from ``config.py`` rather than typed out, so moving one moves the
    check with it.
    """
    from config import get_settings

    settings = get_settings()
    tracked = _tracked_files()

    required = {
        "the city registry, read for the map's validation events": settings.cities_config_path,
        "the evaluation record, read for the risk view's SHAP drivers": (
            settings.model_artifact_dir / "metrics.json"
        ),
        "the Streamlit theme": PROJECT_ROOT / ".streamlit" / "config.toml",
    }

    missing = [
        f"{path.relative_to(PROJECT_ROOT)} ({why})"
        for why, path in required.items()
        if str(path.relative_to(PROJECT_ROOT)) not in tracked
    ]
    assert not missing, "read at runtime but not committed: " + "; ".join(missing)


def test_no_secret_bearing_file_is_committed() -> None:
    """The re-scan the deploy ticket exists to force, as a standing check.

    Making the repository public is the moment anything in history becomes
    visible to everyone. This is cheap enough to run on every commit.
    """
    tracked = _tracked_files()
    forbidden = {".env", ".streamlit/secrets.toml", "secrets.yml", "secrets.yaml"}
    assert not (forbidden & tracked), f"a secret file is tracked: {forbidden & tracked}"
    assert not [p for p in tracked if p.endswith((".pem", ".key"))]
    # The examples must be committed, since they are the documentation for
    # what is missing, and must never hold a real value.
    assert ".env.example" in tracked
    assert ".streamlit/secrets.toml.example" in tracked


def test_the_committed_examples_still_hold_placeholders() -> None:
    """A filled-in example is the easiest way for a credential to reach a public repo."""
    example = (PROJECT_ROOT / ".env.example").read_text("utf-8")
    for line in example.splitlines():
        if line.startswith(("DATABASE_URL=", "SERVING_DATABASE_URL=", "POSTGRES_PASSWORD=")):
            value = line.split("=", 1)[1].strip()
            assert not value, f"{line.split('=')[0]} has a value in .env.example"


def test_the_app_starts_with_no_environment_at_all() -> None:
    """Community Cloud has no .env, only the secret it injects.

    config.py must still build, and the dashboard must still refuse politely
    rather than raise, when nothing is configured. This is the state a
    freshly deployed app is in for the seconds before its secret is pasted in.
    """
    import dataclasses

    from config import get_settings

    from dashboard import database as module

    bare = dataclasses.replace(
        get_settings(), environment="local", database_url=None, serving_database_url=None
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "get_settings", lambda: bare)
        patch.setattr(module, "_secret", lambda key: None)
        with pytest.raises(module.DashboardConfigError):
            module.resolve_database_url()


def test_the_deployment_checker_reads_the_navigation_rather_than_a_list() -> None:
    """A fifth view must not need a second edit to be checked after deploy.

    Loaded by path rather than imported as ``tests.check_deployment``: the
    tests directory is deliberately not a package, and making it one changes how
    pytest imports every module in it and breaks the flat ``from ml_fixtures
    import ...`` the rest of the suite uses.
    """
    import importlib.util

    from dashboard import views

    spec = importlib.util.spec_from_file_location(
        "check_deployment", PROJECT_ROOT / "tests" / "check_deployment.py"
    )
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)

    assert checker._view_paths() == [(m.VIEW.title, m.VIEW.url_path) for m in views.ORDER]
    assert len(checker._view_paths()) == 4


def test_the_entrypoint_streamlit_cloud_is_pointed_at_exists() -> None:
    """The main file path in the deploy form, checked against the repository."""
    tracked = _tracked_files()
    assert "dashboard/app.py" in tracked
    source = (PROJECT_ROOT / "dashboard" / "app.py").read_text("utf-8")
    assert "st.set_page_config" in source, "the entrypoint sets no page config"
    assert 'if __name__ == "__main__"' in source or "main()" in source


def test_the_view_states_the_decision_its_threshold_encodes() -> None:
    """ML-11's threshold is a claim, and the view has to make it out loud.

    A threshold printed alone is a number a reader takes on trust. The budget
    that produced it, and the cost ratio that budget implies, are the two
    figures that let someone disagree with the project rather than accept it,
    and they come from the committed record rather than from prose that could
    drift away from the model.
    """
    import json

    from dashboard.views import risk_horizon as risk

    decision = risk.alert_budget()
    if not decision:
        pytest.skip("no decision recorded; run train.py --write first")

    sentence = risk.budget_sentence()
    assert f"{decision['budget_alerts_per_city_year']:.0f} days a year" in sentence
    assert f"{decision['implied_cost_ratio']:.1f} false alarms" in sentence
    assert "F1" in sentence, "the rule that was replaced should be named"

    # And it comes from the file, not from a constant that could drift.
    path = risk.get_settings().model_artifact_dir / "metrics.json"
    recorded = json.loads(path.read_text())["model"]["calibration"]["decision"]
    assert decision["threshold"] == recorded["threshold"]


def test_the_view_says_which_threshold_it_is_actually_applying() -> None:
    """The gap between the recorded rule and the shipped one, stated.

    The budget rule is defined on calibrated probabilities and the predictions
    table holds raw scores, so the dashboard still applies the F1 threshold.
    Saying so is the difference between a documented limitation and a view that
    quietly contradicts the model card.
    """
    from dashboard.views import risk_horizon as risk

    if not risk.alert_budget():
        pytest.skip("no decision recorded; run train.py --write first")
    rule = risk._threshold_rule()
    assert "F1" in rule.upper()
    assert "budget" in rule


# ---------------------------------------------------------------------------
# Show the reader how much to trust the number (BI-08)
# ---------------------------------------------------------------------------


def test_the_reliability_curve_is_drawn_from_the_committed_record() -> None:
    """The curve is the model's own, not one the dashboard recomputed.

    This deployment ships no XGBoost and no warehouse, so it could not
    reproduce a reliability curve if it wanted to. Reading it from
    `metrics.json` is the same arrangement the SHAP drivers use, and it means
    the curve cannot disagree with the model card: both are quoting one file.
    """
    from dashboard.views import risk_horizon as risk

    if not risk.calibration():
        pytest.skip("no calibration recorded; run train.py --write first")

    for variant in ("raw_unweighted", "calibrated"):
        points = risk.reliability_points(variant)
        assert not points.empty, variant
        assert set(points.columns) >= {"predicted", "observed", "rows", "weight"}
        assert points["predicted"].is_monotonic_increasing
        assert points["predicted"].between(0, 1).all()
        assert points["observed"].between(0, 1).all()
        assert risk.calibration_error(variant) is not None


def test_the_curve_is_a_figure_and_carries_the_diagonal() -> None:
    """Three traces: two curves and the truth they are measured against.

    The diagonal is the whole chart -- a reliability curve without it is two
    lines with nothing to be right or wrong about. It is drawn in the chrome's
    axis colour rather than as a third series, so a reader sees two curves
    against a reference and not three curves.
    """
    from dashboard.views import risk_horizon as risk

    if not risk.calibration():
        pytest.skip("no calibration recorded; run train.py --write first")

    figure = risk._reliability_figure("dark")
    assert figure is not None
    names = [trace.name for trace in figure.data]
    assert "perfectly calibrated" in names
    assert len(figure.data) == 3

    reference = next(t for t in figure.data if t.name == "perfectly calibrated")
    assert list(reference.x) == list(reference.y), "the diagonal is not diagonal"
    assert reference.line.dash == "dash"
    assert reference.line.color == theme.chrome("dark")["axis"]

    curves = [t for t in figure.data if t.name != "perfectly calibrated"]
    palette = set(theme.calibration_colours("dark").values())
    assert {t.line.color for t in curves} == palette


def test_the_curve_shows_the_model_understating_risk() -> None:
    """The finding the panel exists to show, asserted rather than captioned.

    Every bin sits above the diagonal: at each level of predicted probability
    more weeks turned out anomalous than the model said. That is the shape of a
    model fitted where positives are ~4.9% of rows and scored where they are
    ~11.3%, and the caption says so. If it ever stopped being true the caption
    would be wrong and nothing else would notice.
    """
    from dashboard.views import risk_horizon as risk

    if not risk.calibration():
        pytest.skip("no calibration recorded; run train.py --write first")

    points = risk.reliability_points("raw_unweighted")
    above = (points["observed"] >= points["predicted"]).mean()
    assert above > 0.7, (
        "the model no longer under-states risk across most of the range; the "
        "caption under the reliability curve says it does"
    )


def test_the_trust_sentence_is_chosen_by_the_number_not_the_copy() -> None:
    """A verdict in prose has to move when the number it describes moves."""
    from dashboard.views import risk_horizon as risk

    if not risk.calibration():
        pytest.skip("no calibration recorded; run train.py --write first")

    raw = risk.calibration_error("raw_unweighted")
    sentence = risk.trust_sentence()
    assert sentence
    assert f"{raw:.1%}" in sentence
    if raw > theme.CALIBRATION_TRUST_CEILING:
        assert "ranking, not as a percentage" in sentence
    else:
        assert "can be read as" in sentence


def test_the_view_names_the_climatology_it_predicts_against() -> None:
    """DBT-13 decided which question the flag answers; this view has to say it.

    The warehouse carries two climatologies and the model is trained on one of
    them. A reader looking at a violet band is owed the question it answers,
    and "unusual for the record" and "unusual for this era" are different
    products rather than two phrasings of one.
    """
    from dashboard.views import risk_horizon as risk

    assert "unusual for the record" in risk.VIEW.caption.lower()
    assert "detrended" in risk.__doc__.lower()
    assert "unusual for the record" in risk.__doc__.lower()


def test_the_budget_sits_with_the_threshold_and_not_only_in_the_vintage() -> None:
    """Where a reader decides what "flagged" means is where the rule belongs.

    The risk key is the number that says which cities are lit. A threshold
    shown there without the decision it encodes is a number to take on trust,
    and a decision explained three panels away is one nobody reads.
    """
    source = (
        Path(risk_horizon_module().__file__).read_text(encoding="utf-8")
    )
    key_block = source.split("with key:", 1)[1].split("with drivers:", 1)[0]
    assert "budget_sentence()" in key_block


def risk_horizon_module():
    from dashboard.views import risk_horizon

    return risk_horizon


# ---------------------------------------------------------------------------
# The rarity encoding (ML-15/DBT-15)
# ---------------------------------------------------------------------------


@pytest.fixture
def rarity_frame():
    """Four cities, covering every state the rarity encoding has to draw.

    Delhi is far out and has a fitted tail. Cairo is a modest exceedance, also
    fitted. Portland is fitted but ordinary today, so it is below its tail
    threshold and has no period. Sydney is scored and has *no fitted tail* --
    eighteen days of record cannot support one -- which is the state most
    easily confused with "nothing happening here".
    """
    return pd.DataFrame(
        [
            {"city_id": "delhi", "name": "Delhi", "country": "India",
             "latitude": 28.6, "longitude": 77.2, "observed_c": 35.0,
             "baseline_c": 28.7, "baseline_sigma": 1.32, "z": 4.79,
             "departure_c": 6.34, "is_anomaly": True,
             "baseline_observations": 465.0, "observed": True,
             "return_years": 22.0, "return_qualifier": "at least",
             "return_is_reportable": False, "tail_fitted": True},
            {"city_id": "cairo", "name": "Cairo", "country": "Egypt",
             "latitude": 30.0, "longitude": 31.2, "observed_c": 33.0,
             "baseline_c": 29.1, "baseline_sigma": 1.9, "z": 2.04,
             "departure_c": 3.9, "is_anomaly": False,
             "baseline_observations": 465.0, "observed": True,
             "return_years": 0.1, "return_qualifier": "about",
             "return_is_reportable": True, "tail_fitted": True},
            {"city_id": "portland", "name": "Portland", "country": "United States",
             "latitude": 45.5, "longitude": -122.7, "observed_c": 19.0,
             "baseline_c": 18.3, "baseline_sigma": 3.2, "z": 0.22,
             "departure_c": 0.7, "is_anomaly": False,
             "baseline_observations": 465.0, "observed": True,
             "return_years": None, "return_qualifier": None,
             "return_is_reportable": None, "tail_fitted": True},
            {"city_id": "sydney", "name": "Sydney", "country": "Australia",
             "latitude": -33.9, "longitude": 151.2, "observed_c": 26.0,
             "baseline_c": 22.0, "baseline_sigma": 1.4, "z": 2.86,
             "departure_c": 4.0, "is_anomaly": True,
             "baseline_observations": 17.0, "observed": True,
             "return_years": None, "return_qualifier": None,
             "return_is_reportable": None, "tail_fitted": False},
        ]
    )


def test_the_two_encodings_size_the_same_day_differently(rarity_frame) -> None:
    """Sigma and rarity disagree on purpose, and that is the toggle's content.

    Two sigma is the same arithmetic everywhere and a very different rarity in
    a steady climate than in a volatile one. If the two channels produced the
    same picture the control would be decoration.
    """
    from dashboard.views import anomaly_map

    day = dt.date(2021, 6, 28)
    departure = anomaly_map.prepare(rarity_frame, day, encoding="departure")
    rarity = anomaly_map.prepare(rarity_frame, day, encoding="rarity")

    assert not departure["diameter"].equals(rarity["diameter"])
    # Colour is the same map in both: only size changes, so a reader toggling
    # is re-reading one picture rather than being shown a second one.
    assert departure["colour"].equals(rarity["colour"])


def test_an_unfitted_city_gets_no_size_on_the_rarity_encoding(rarity_frame) -> None:
    """Sydney leaves the filled trace for the open ring, rather than shrinking.

    Drawing it at the floor would say "nothing rare happened here" using the
    same mark that means "nobody could fit this city", and a reader has no way
    to tell those apart from a dot. The ring already means "no number here" on
    this map.
    """
    from dashboard.views import anomaly_map

    frame = anomaly_map.prepare(rarity_frame, dt.date(2021, 6, 28), encoding="rarity")
    sydney = frame[frame["city_id"] == "sydney"].iloc[0]

    assert not sydney["encoded"]
    assert pd.isna(sydney["diameter"])

    figure = anomaly_map._figure(frame)
    rings = next(trace for trace in figure.data if trace.name == "not scored")
    assert "Sydney" in " ".join(rings.text)


def test_a_fitted_city_below_its_threshold_is_drawn_smallest(rarity_frame) -> None:
    """Portland is fitted and ordinary, so it gets the floor and keeps its size.

    The opposite state to Sydney's, and the reason the two are distinguished:
    "we fitted this city and today is unremarkable" is a real answer, and the
    smallest circle is the right way to draw it.
    """
    from dashboard import theme
    from dashboard.views import anomaly_map

    frame = anomaly_map.prepare(rarity_frame, dt.date(2021, 6, 28), encoding="rarity")
    portland = frame[frame["city_id"] == "portland"].iloc[0]

    assert portland["encoded"]
    assert portland["diameter"] == pytest.approx(theme.MARKER_MIN_PX)


def test_the_rarity_channel_is_logarithmic_and_capped() -> None:
    """Return periods span orders of magnitude; the channel has forty pixels.

    A linear channel would collapse every ordinary exceedance into one dot
    while a single fifty-year day took the whole range. The cap exists for the
    other end: past fifty years the fitted answers separate by hundreds of
    years on a shape parameter whose interval spans two orders of magnitude,
    and the map must stop distinguishing what it cannot distinguish.
    """
    from dashboard import theme

    steps = [theme.rarity_diameter(years) for years in (0.1, 1.0, 10.0, 50.0)]
    assert steps == sorted(steps)
    # Equal ratios in years must be equal distances in pixels.
    assert steps[1] - steps[0] == pytest.approx(steps[2] - steps[1], abs=0.01)
    assert theme.rarity_diameter(500.0) == theme.rarity_diameter(theme.RARITY_YEARS_CAP)
    assert theme.rarity_diameter(float("nan")) == theme.MARKER_MIN_PX


def test_the_two_size_keys_share_a_pixel_range() -> None:
    """A toggle that moved the scale would make every city appear to change.

    The reader switching encodings is comparing shapes between two pictures. If
    the pixel range moved as well, every marker would resize for a reason that
    had nothing to do with the data.
    """
    from dashboard import theme

    assert theme.rarity_diameter(theme.RARITY_YEARS_FLOOR) == pytest.approx(
        theme.marker_diameter(0.0)
    )
    assert theme.rarity_diameter(theme.RARITY_YEARS_CAP) == pytest.approx(
        theme.marker_diameter(theme.MARKER_Z_CAP)
    )


def test_a_sub_year_period_is_phrased_as_a_frequency() -> None:
    """"A 1-in-0.1-year day" is correct and unreadable.

    Below a year the reader's question reverses: these are days a city sees
    several times a season, and the quantity they hold is how often, not how
    long between. Most rows in the mart are below a year, so this is the common
    case rather than an edge.
    """
    from dashboard.views.anomaly_map import _rarity_phrase

    assert "10 times a year" in _rarity_phrase(0.1, True)
    assert "most weeks" in _rarity_phrase(0.05, True)
    assert "1-in-" not in _rarity_phrase(0.1, True)


def test_an_unreportable_period_is_phrased_as_a_floor() -> None:
    """"At least", not "about", where the shape sensitivity spans a decade.

    Phoenix at four sigma reads 28 years and its shape interval puts it between
    9 and 152,000. The map quotes the floor and says so; printing the point
    estimate there would be a number that gets quoted onward and cannot be
    walked back.
    """
    from dashboard.views.anomaly_map import _rarity_phrase

    assert _rarity_phrase(15.1, False).startswith("At least")
    assert _rarity_phrase(4.2, True).startswith("About")


def test_the_tooltip_says_which_of_the_four_states_a_city_is_in(rarity_frame) -> None:
    """Fitted-and-rare, fitted-and-ordinary, fitted-but-below, and unfitted.

    Collapsing any pair of these would be a lie of a different kind. The one
    that matters most is the last: silence, rather than a claim the map cannot
    support.
    """
    from dashboard.views import anomaly_map

    frame = anomaly_map.prepare(rarity_frame, dt.date(2021, 6, 28), encoding="rarity")
    tips = dict(zip(frame["city_id"], frame["tooltip"]))

    assert "At least a 1-in-22-year day" in tips["delhi"]
    assert "sees about 10 times a year" in tips["cairo"]
    # Fitted but below its threshold, and unfitted entirely: both silent, and
    # neither claiming the day was ordinary on the strength of a model that was
    # not consulted.
    assert "year day" not in tips["portland"]
    assert "year day" not in tips["sydney"]


# --------------------------------------------------------------------------
# The small-screen wall
# --------------------------------------------------------------------------


WALL = theme.small_screen_notice_html("Climate Volatility & Risk Engine")


def test_the_wall_borrows_its_colours_rather_than_choosing_them() -> None:
    """The panel is part of the palette, not a page beside it.

    ``test_no_module_but_theme_spells_a_colour`` cannot see this one: the
    markup is generated *inside* theme.py, which that check skips. So the same
    promise is made here directly, against the rendered string.
    """
    palette = {
        colour
        for mode in MODES
        for colour in (*theme.DIVERGING[mode], theme.SURFACE[mode], *theme.chrome(mode).values())
    }
    assert set(HEX_COLOUR.findall(WALL)) <= palette


def test_the_wall_is_for_phones_and_nothing_wider() -> None:
    """One breakpoint, on width alone.

    Continuum gates on height too, which catches a phone held sideways at the
    cost of catching a short laptop window. This dashboard is read on laptops
    in short windows, so height does not gate and the check says so rather
    than leaving it to whoever next edits the CSS.
    """
    assert f"(max-width: {theme.SMALL_SCREEN_MAX_WIDTH_PX - 1}px)" in WALL
    assert "max-height" not in WALL
    assert "min-width" not in WALL


def test_the_wall_hides_the_app_in_a_way_a_descendant_can_undo() -> None:
    """``visibility``, not ``display``, and on the one Streamlit-owned element.

    The panel is rendered inside the app it covers. ``display:none`` on an
    ancestor cannot be reversed further down, so the panel would go with it;
    ``visibility`` inherits and can be turned back on. Getting this backwards
    produces a blank white page on a phone, which is the failure this test
    exists to catch.
    """
    assert '[data-testid="stApp"] { visibility: hidden; }' in WALL
    assert "visibility: visible;" in WALL
    assert '[data-testid="stApp"] { display: none' not in WALL


def test_the_wall_costs_a_laptop_nothing() -> None:
    """Above the breakpoint the panel is gone, and so is the row it sat in.

    Streamlit renders it into the vertical block that lays out every page, and
    that block is a flex column with a gap. An element container left in the
    flow with nothing in it is still a flex item, so hiding only the panel
    would push every view down by one gap on every screen. Both go.
    """
    container = '[data-testid="stElementContainer"]:has(.horizon-wall)'
    css = WALL.split("@media (max-width")[0]

    assert ".horizon-wall { display: none; }" in css
    assert f"{container} {{ display: none; }}" in css


def test_the_wall_paints_both_themes_without_asking_the_server() -> None:
    """Both modes ship in the one stylesheet.

    ``st.context.theme.type`` is ``None`` on the first run of a session, so a
    server-side choice would paint a full-screen panel in the wrong mode and
    correct it a beat later. The browser already knows.
    """
    assert "prefers-color-scheme: dark" in WALL
    for mode in MODES:
        assert theme.SURFACE[mode] in WALL


@pytest.mark.parametrize("mode", MODES)
def test_the_wall_s_own_text_is_readable_on_it(mode: theme.Mode) -> None:
    """The headline runs the ramp's poles, and the poles have to clear the page.

    The wash fades into the surface above the headline, so the surface is the
    background the text is actually read against. 4.5:1 is the floor for the
    body copy; the headline is large, and takes the 3:1 that large text gets.
    """
    surface = theme.SURFACE[mode]
    steps = theme.DIVERGING[mode]

    assert contrast(theme.chrome(mode)["ink_secondary"], surface) >= 4.5
    assert contrast(steps[0], surface) >= 3.0
    assert contrast(steps[-1], surface) >= 3.0


def test_the_wall_does_not_take_the_product_name_on_trust() -> None:
    """The title is interpolated into markup, so it is escaped on the way in."""
    assert "&amp;" in WALL
    assert "<script>" not in theme.small_screen_notice_html("<script>alert(1)</script>")


def test_the_wall_goes_up_before_the_database_is_asked_for() -> None:
    """A phone gets the panel, not a connection error it cannot act on.

    Checked on the source rather than by running Streamlit, because ordering
    inside ``main()`` is the whole claim and a rendered app would not show it.
    """
    source = (DASHBOARD_DIR / "app.py").read_text(encoding="utf-8")
    assert source.index("_small_screen_wall()") < source.index("source = resolve_database_url()")
