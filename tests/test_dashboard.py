"""Tests for the dashboard shell, its connection layer, and its palette.

Three things here are worth more than the line count suggests.

**The palette's claims are recomputed, not quoted.** ``dashboard/theme.py``
states in its docstring that a reader with the commonest form of colour
blindness can tell a cold anomaly from a warm one, and gives numbers. Those
numbers are the whole justification for the scale, and a docstring cannot go
out of date quietly if a test parses it and recomputes every cell. The colour
maths — sRGB to OKLab, and the Machado 2009 colour-vision-deficiency
simulation — is implemented here rather than imported, so a palette edit that
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

None of it needs a database. The suite that does — the one proving the app can
actually read Neon — is ``tests/check_connection.py``.
"""

from __future__ import annotations

import ast
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

import pandas as pd  # noqa: E402
from sqlalchemy.exc import (  # noqa: E402
    DBAPIError,
    InterfaceError,
    OperationalError,
    ProgrammingError,
)

from dashboard import theme  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
STREAMLIT_CONFIG = PROJECT_ROOT / ".streamlit" / "config.toml"

# The separation below which two colours are the same colour to a reader.
# OKLab Euclidean distance ×100; 8 is "distinct", 6 is the floor that is only
# acceptable when something other than colour also carries the distinction.
SEPARATION_TARGET = 8.0

# The two forms of colour-vision deficiency that affect a red-blue scale.
# Tritanopia is measured too but does not gate: it distorts blue-yellow, and a
# blue-red ramp is close to the axis it leaves alone.
GATING_DEFICIENCIES = ("protan", "deutan")


# --------------------------------------------------------------------------
# Colour maths. Implemented here on purpose — see the module docstring.
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

    A hue at the midpoint reads as its own category — "normal" acquires a
    temperature — and it also breaks the two-hue promise a diverging scale
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
    """+3σ and −3σ must look equally loud.

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
    arm at the same magnitude, the figure is not merely less pretty for them —
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
    decoration — so the table is parsed and each cell recomputed.
    """
    table = re.findall(
        r"^(\S[^\n]*?)\s{2,}(\d+\.\d+)\s+(\d+\.\d+)\s*$",
        theme.__doc__ or "",
        flags=re.MULTILINE,
    )
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
    quoted = re.findall(r"(\d\.\d\d):1", theme.__doc__ or "")
    assert len(quoted) == 2, "expected the light shortfall and its dark counterpart"

    def worst_inner(mode: theme.Mode) -> float:
        """The palest step of either arm — the docstring quotes the weaker one."""
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
    the .env fallback from being reached — otherwise the app cannot be run
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
    that measures 1.2 s — enough headroom for a slow resume, short enough that
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
    A short TTL is therefore not a small cost — it is the dominant one.
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
    """"Each view has a one-line plain-English caption" — §Workstream 4."""
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
        sql = module.VIEW.probe_sql
        assert "bronze_raw" not in sql and "silver_staging" not in sql
        assert module.VIEW.source_table.startswith("gold_marts.")


def test_the_diverging_views_share_one_key() -> None:
    """Three views encode a signed anomaly, and they must explain it identically.

    Storm Dynamics is the exception and says so in its own words rather than
    silently omitting a legend.
    """
    from dashboard import views

    diverging = [m for m in views.ORDER if m.VIEW.encoding == "diverging"]
    assert len(diverging) == 3
    other = [m for m in views.ORDER if m.VIEW.encoding != "diverging"]
    assert [m.VIEW.title for m in other] == ["Storm Dynamics"]
    assert "BI-03" in other[0].VIEW.encoding


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


@pytest.fixture
def offline(monkeypatch):
    """Run the app with the warehouse replaced, so nothing reaches a network.

    Both seams have to be closed: the shell calls ``warehouse_status`` for its
    sidebar, and a view calls ``run_query`` for its probe. ``_scaffold`` bound
    ``run_query`` at import, so it is patched where it is used rather than
    where it is defined.
    """
    from dashboard import database as module
    from dashboard.views import _scaffold

    def install(*, status=None, query=None, source=None):
        monkeypatch.setattr(module, "warehouse_status", status or (lambda: {"cities": 15}))
        monkeypatch.setattr(_scaffold, "run_query", query or (lambda sql, *a, **k: pd.DataFrame()))
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

    fields = {field for module in views.ORDER for field in module.VIEW.probe_labels}
    offline(query=lambda sql, *a, **k: pd.DataFrame([{field: 1 for field in fields}]))

    for module in views.ORDER:
        script = (
            "from importlib import import_module\n"
            f"import_module({module.__name__!r}).render()\n"
        )
        app = AppTest.from_string(script, default_timeout=30).run()
        assert not app.exception, f"{module.VIEW.title} raised: {app.exception}"
        assert [element.value for element in app.title] == [module.VIEW.title]
        assert len(app.metric) == len(module.VIEW.probe_labels)


def test_the_shell_renders_its_default_view(offline) -> None:
    """Shell and view together: navigation, sidebar, and a page under it."""
    from streamlit.testing.v1 import AppTest

    from dashboard import views

    first = views.ORDER[0].VIEW
    offline(query=lambda sql, *a, **k: pd.DataFrame([{f: 1 for f in first.probe_labels}]))

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
    something a visitor can wait out — so it gets a message and no retry button
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


def _third_party_imports(directory: Path) -> set[str]:
    """Top-level modules imported under a directory, minus stdlib and our own."""
    first_party = {"dashboard", "config", "ingestion", "machine_learning", "serving"}
    found: set[str] = set()
    for path in _python_modules(directory):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found - set(sys.stdlib_module_names) - first_party


def test_the_deployment_manifest_covers_what_the_dashboard_imports() -> None:
    """A dependency left behind fails on Cloud, not here — unless this runs.

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
