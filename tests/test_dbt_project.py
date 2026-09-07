"""Tests for the dbt project's configuration.

Three things here are easy to get wrong and quiet when wrong, so each has a
test rather than a comment:

*   **Schema routing.** dbt's default ``generate_schema_name`` builds
    ``<target.schema>_<custom>``, so a mart configured into ``gold_marts``
    would land in ``public_gold_marts`` — beside the empty ``gold_marts`` that
    ``ingestion/schema.sql`` created, with nothing to say which is real.
*   **Credentials in the profile.** A literal host or password in
    ``profiles.yml.example`` would be committed. Every field must be an
    ``env_var`` lookup.
*   **Drift between the template and the bridge.** The profile names variables;
    ``dbt_env.py`` supplies them. Either can change without the other, and the
    failure is a dbt run against nothing.
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

from config import ConfigError, split_database_url  # noqa: E402
from dbt_analytics.dbt_env import dbt_environment  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DBT_DIR = REPO_ROOT / "dbt_analytics"
PROJECT_YML = DBT_DIR / "dbt_project.yml"
PROFILE_TEMPLATE = DBT_DIR / "profiles.yml.example"
SOURCES_YML = DBT_DIR / "models" / "staging" / "_sources.yml"
SCHEMA_MACRO = DBT_DIR / "macros" / "generate_schema_name.sql"

#: The schemas ingestion/schema.sql creates. dbt must build into these exact
#: names, not decorated variants of them.
BRONZE, SILVER, GOLD = "bronze_raw", "silver_staging", "gold_marts"


@pytest.fixture(scope="module")
def project() -> dict:
    return yaml.safe_load(PROJECT_YML.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def profile_template() -> str:
    return PROFILE_TEMPLATE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def sources() -> dict:
    return yaml.safe_load(SOURCES_YML.read_text(encoding="utf-8"))


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )


# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------


def test_the_project_names_a_profile(project) -> None:
    assert project["name"] == "horizon"
    assert project["profile"] == "horizon"
    assert project["config-version"] == 2


@pytest.mark.parametrize("layer", ["staging", "intermediate", "marts"])
def test_every_layer_is_configured(project, layer: str) -> None:
    assert layer in project["models"]["horizon"]
    assert (DBT_DIR / "models" / layer).is_dir()


def test_model_paths_point_at_models(project) -> None:
    assert project["model-paths"] == ["models"]


# ---------------------------------------------------------------------------
# Materialisation and schema per layer
# ---------------------------------------------------------------------------


def test_staging_is_views(project) -> None:
    """Thin projections over bronze; materialising them doubles storage."""
    staging = project["models"]["horizon"]["staging"]
    assert staging["+materialized"] == "view"
    assert staging["+schema"] == SILVER


def test_marts_are_tables(project) -> None:
    """The dashboard queries these directly over a serverless connection."""
    marts = project["models"]["horizon"]["marts"]
    assert marts["+materialized"] == "table"
    assert marts["+schema"] == GOLD


def test_intermediate_is_not_materialised(project) -> None:
    intermediate = project["models"]["horizon"]["intermediate"]
    assert intermediate["+materialized"] == "ephemeral"


def test_every_layer_targets_a_schema_the_ddl_creates(project) -> None:
    """A schema dbt invents is one nothing else in the project knows about."""
    ddl = (REPO_ROOT / "ingestion" / "schema.sql").read_text(encoding="utf-8")
    created = set(re.findall(r"create schema if not exists (\w+)", ddl))
    assert {SILVER, GOLD} <= created

    configured = {
        layer["+schema"]
        for layer in project["models"]["horizon"].values()
        if "+schema" in layer
    }
    assert configured <= created, f"dbt would create {configured - created}"


def test_the_schema_macro_uses_the_custom_name_verbatim() -> None:
    """Without this override, gold_marts becomes public_gold_marts."""
    macro = SCHEMA_MACRO.read_text(encoding="utf-8")
    body = macro.split("{% macro")[1]

    assert "generate_schema_name" in macro
    # The custom name is emitted alone, with nothing concatenated onto it.
    assert re.search(r"{{-?\s*custom_schema_name\s*\|\s*trim\s*-?}}", body)
    assert "~" not in body, "a ~ here is dbt's default concatenation"
    # target.schema survives only as the fallback when no custom name is set.
    fallback, custom = body.split("{%- else -%}")
    assert "target.schema" in fallback
    assert "target.schema" not in custom


# ---------------------------------------------------------------------------
# The profile commits no credentials
# ---------------------------------------------------------------------------


def test_the_real_profile_is_git_ignored() -> None:
    assert git("check-ignore", "-q", "dbt_analytics/profiles.yml").returncode == 0
    assert git("ls-files", "dbt_analytics/profiles.yml").stdout.strip() == ""


def test_the_template_is_committed() -> None:
    assert PROFILE_TEMPLATE.is_file()
    tracked = git("ls-files", "dbt_analytics/profiles.yml.example").stdout.strip()
    assert tracked.endswith("profiles.yml.example")


@pytest.mark.parametrize(
    "field", ["host", "port", "user", "password", "dbname"]
)
def test_every_connection_field_comes_from_the_environment(
    profile_template, field: str
) -> None:
    """The ticket's requirement, checked per field rather than by eyeball."""
    for line in profile_template.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{field}:"):
            assert "env_var(" in stripped, f"{field} is not read from the environment"


def test_no_literal_credential_survives_in_the_template(profile_template) -> None:
    settings_values = {
        v for v in (os.environ.get("POSTGRES_PASSWORD"),) if v
    }
    for secret in settings_values:
        assert secret not in profile_template
    # A bare host or password would show up as a value that is not a lookup.
    for line in profile_template.splitlines():
        stripped = line.strip()
        for field in ("host:", "password:", "user:", "dbname:"):
            if stripped.startswith(field):
                assert "env_var(" in stripped, stripped


def test_both_targets_exist(profile_template) -> None:
    parsed = yaml.safe_load(profile_template)
    outputs = parsed["horizon"]["outputs"]
    assert set(outputs) == {"dev", "prod"}
    assert outputs["dev"]["type"] == outputs["prod"]["type"] == "postgres"


def test_prod_requires_tls() -> None:
    """Neon's pooled endpoint refuses a plaintext connection."""
    parsed = yaml.safe_load(PROFILE_TEMPLATE.read_text(encoding="utf-8"))
    assert "'require'" in parsed["horizon"]["outputs"]["prod"]["sslmode"]


def test_the_default_target_is_local(profile_template) -> None:
    """A multi-hour build against a serverless database is slow and wasteful."""
    parsed = yaml.safe_load(profile_template)
    assert "'dev'" in parsed["horizon"]["target"]


# ---------------------------------------------------------------------------
# The template and the bridge agree
# ---------------------------------------------------------------------------


def test_the_bridge_supplies_every_variable_the_profile_needs(
    profile_template,
) -> None:
    """Either file can change without the other; the failure is silent."""
    referenced = set(re.findall(r"env_var\(\s*'([A-Z_]+)'\s*\)", profile_template))
    supplied = set(dbt_environment())
    assert referenced <= supplied, f"profile reads {referenced - supplied}, unset"


def test_variables_with_a_default_are_still_supplied(profile_template) -> None:
    with_defaults = set(
        re.findall(r"env_var\(\s*'([A-Z_]+)'\s*,\s*'[^']*'\s*\)", profile_template)
    )
    assert with_defaults
    assert with_defaults <= set(dbt_environment())


def test_the_bridge_omits_serving_rather_than_faking_it(monkeypatch) -> None:
    """A placeholder would let --target prod connect somewhere unintended."""
    import dataclasses

    from config import get_settings

    without = dataclasses.replace(get_settings(), serving_database_url=None)
    env = dbt_environment(without)
    assert not any(k.startswith("DBT_SERVING_") for k in env)
    assert env["DBT_HOST"]


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def test_both_bronze_tables_are_declared(sources) -> None:
    source = sources["sources"][0]
    assert source["schema"] == BRONZE
    assert {t["name"] for t in source["tables"]} == {
        "observations_daily",
        "observations_hourly",
    }


def test_declared_tables_exist_in_the_ddl(sources) -> None:
    ddl = (REPO_ROOT / "ingestion" / "schema.sql").read_text(encoding="utf-8")
    for table in sources["sources"][0]["tables"]:
        assert f"bronze_raw.{table['name']}" in ddl


def test_freshness_is_measured_on_the_ingestion_clock(sources) -> None:
    """The observation clock would report every row as decades stale."""
    source = sources["sources"][0]
    assert source["loaded_at_field"] == "ingested_at"


def test_freshness_thresholds_are_set_and_ordered(sources) -> None:
    freshness = sources["sources"][0]["freshness"]
    warn, error = freshness["warn_after"], freshness["error_after"]
    assert warn["period"] == error["period"] == "day"
    assert warn["count"] < error["count"]


def test_declared_columns_exist_in_bronze(sources) -> None:
    """A documented column that was renamed is documentation that lies."""
    ddl = (REPO_ROOT / "ingestion" / "schema.sql").read_text(encoding="utf-8")
    for table in sources["sources"][0]["tables"]:
        for column in table.get("columns", []):
            assert re.search(rf"^\s+{column['name']}\s+\S", ddl, re.M), (
                f"{table['name']}.{column['name']} is not in the DDL"
            )


# ---------------------------------------------------------------------------
# Splitting the URL, which is why dbt_env exists
# ---------------------------------------------------------------------------


def test_a_local_url_splits() -> None:
    parts = split_database_url("postgresql://climate:pw@localhost:5434/climate")
    assert (parts.host, parts.port, parts.user, parts.dbname) == (
        "localhost", 5434, "climate", "climate",
    )
    assert parts.password == "pw"
    assert parts.sslmode is None


def test_a_neon_url_keeps_its_sslmode() -> None:
    parts = split_database_url(
        "postgresql://u:p@ep-x-pooler.aws.neon.tech/neondb"
        "?sslmode=require&channel_binding=require"
    )
    assert parts.host.endswith("neon.tech")
    assert parts.port == 5432, "no port in the URL means the Postgres default"
    assert parts.sslmode == "require"


def test_percent_encoded_credentials_are_decoded() -> None:
    """A password with a @ or / in it is legal, and is encoded in the URL."""
    parts = split_database_url("postgresql://us%40er:p%2Fss@host/db")
    assert parts.user == "us@er"
    assert parts.password == "p/ss"


@pytest.mark.parametrize(
    ("url", "missing"),
    [
        ("postgresql://user@/db", "host"),
        ("postgresql://host/db", "user"),
        ("postgresql://user@host", "database name"),
        ("mysql://user:p@host/db", "postgresql://"),
    ],
)
def test_an_unusable_url_says_what_is_missing(url: str, missing: str) -> None:
    """dbt's own failure for these does not mention the URL at all."""
    with pytest.raises(ConfigError, match=re.escape(missing)):
        split_database_url(url)


def test_the_env_mapping_covers_every_field() -> None:
    parts = split_database_url("postgresql://u:p@h:1234/d?sslmode=require")
    env = parts.as_env("DBT")
    assert env == {
        "DBT_HOST": "h",
        "DBT_PORT": "1234",
        "DBT_USER": "u",
        "DBT_PASSWORD": "p",
        "DBT_DBNAME": "d",
        "DBT_SSLMODE": "require",
    }


# ---------------------------------------------------------------------------
# dbt itself
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dbt_env_vars() -> dict[str, str]:
    return {**os.environ, **dbt_environment()}


def run_dbt(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "dbt.cli.main", *args, "--project-dir", str(DBT_DIR)],
        cwd=REPO_ROOT, capture_output=True, text=True, env=env, check=False,
    )


def test_dbt_debug_passes_against_local(engine, dbt_env_vars) -> None:
    """The acceptance criterion, run rather than described."""
    pytest.importorskip("dbt.cli.main")
    if not (DBT_DIR / "profiles.yml").exists():
        pytest.skip("profiles.yml not generated; run dbt_env.py --write-profile")
    result = run_dbt(["debug"], dbt_env_vars)
    assert "All checks passed" in result.stdout, result.stdout[-2000:]


def test_the_project_parses(dbt_env_vars) -> None:
    pytest.importorskip("dbt.cli.main")
    if not (DBT_DIR / "profiles.yml").exists():
        pytest.skip("profiles.yml not generated")
    result = run_dbt(["parse"], dbt_env_vars)
    assert result.returncode == 0, result.stdout[-2000:]
