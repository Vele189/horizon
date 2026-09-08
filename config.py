"""Single source of truth for environment configuration.

This module is the **only** place in the project that reads from the process
environment. Nothing else may call ``os.environ`` or ``os.getenv``. Importing
:func:`get_settings` here instead means every variable is named once, typed
once, validated once, and documented once (in ``.env.example``).

Usage::

    from config import get_settings

    settings = get_settings()
    engine = create_engine(settings.database_url)

Run ``python config.py`` to print the resolved configuration with every secret
masked. That is the fastest way to confirm a ``.env`` is wired up correctly
without echoing a connection string into a terminal history.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs, unquote, urlsplit

from dotenv import load_dotenv

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
ENV_FILE: Final[Path] = PROJECT_ROOT / ".env"

# Values loaded from .env never override variables already present in the
# environment. Streamlit Community Cloud, GitHub Actions, and Docker all inject
# configuration that way, and a stale local .env must not win over it.
load_dotenv(ENV_FILE, override=False)

_VALID_ENVIRONMENTS: Final[frozenset[str]] = frozenset({"local", "serving"})
_VALID_LOG_LEVELS: Final[frozenset[str]] = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

# Field names whose values must never be printed, logged, or committed.
_SECRET_FIELDS: Final[frozenset[str]] = frozenset(
    {"database_url", "serving_database_url", "postgres_password"}
)


class ConfigError(RuntimeError):
    """Raised when the environment is missing or malformed.

    Failing here is deliberate: a pipeline that silently falls back to a
    default database is far worse than one that refuses to start.
    """


def _get(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _get_str(name: str, default: str) -> str:
    return _get(name, default) or default


def _get_int(name: str, default: int) -> int:
    raw = _get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(
            f"{name} must be an integer, got {raw!r}. Check your .env."
        ) from exc


def _get_positive_int(name: str, default: int) -> int:
    """Coerce like :func:`_get_int`, but reject values that cannot work.

    A timeout of zero never fires, and zero retry attempts means the request is
    never made at all. Both are configuration mistakes that would otherwise
    surface as a mysterious hang or a silent no-op deep inside ingestion.
    """
    value = _get_int(name, default)
    if value < 1:
        raise ConfigError(f"{name} must be a positive integer, got {value}.")
    return value


def _get_positive_float(name: str, default: float) -> float:
    """A non-negative float. Zero is allowed; it means "no delay"."""
    raw = _get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(
            f"{name} must be a number, got {raw!r}. Check your .env."
        ) from exc
    if value < 0:
        raise ConfigError(f"{name} must not be negative, got {value}.")
    return value


def _get_path(name: str, default: str) -> Path:
    raw = _get_str(name, default)
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def mask_secret(value: str | None) -> str:
    """Render a secret safely for logs and terminals.

    Connection strings keep their shape (driver, host, database) so a
    misconfiguration is still diagnosable, but the password is replaced.
    """
    if not value:
        return "<unset>"
    masked = re.sub(r"://([^:/@]+):[^@]*@", r"://\1:***@", value)
    if masked != value:
        return masked
    return f"{value[:2]}***" if len(value) > 6 else "***"


@dataclass(frozen=True)
class DatabaseParts:
    """A connection string taken apart, for tools that will not take it whole.

    dbt's postgres adapter wants host, user, password, port and dbname as
    separate settings; it has no way to accept a URL. Rather than add a second
    set of environment variables that could drift from ``DATABASE_URL``, the
    URL stays the single source and this splits it on demand. One place the
    connection is written, one place it is read, one place it is taken apart.
    """

    host: str
    port: int
    user: str
    password: str | None
    dbname: str
    sslmode: str | None = None

    def as_env(self, prefix: str) -> dict[str, str]:
        """Shell-style variables, for injecting into a subprocess."""
        out = {
            f"{prefix}_HOST": self.host,
            f"{prefix}_PORT": str(self.port),
            f"{prefix}_USER": self.user,
            f"{prefix}_PASSWORD": self.password or "",
            f"{prefix}_DBNAME": self.dbname,
            f"{prefix}_SSLMODE": self.sslmode or "prefer",
        }
        return out


def split_database_url(url: str, *, name: str = "DATABASE_URL") -> DatabaseParts:
    """Split a Postgres URL into the fields dbt needs.

    Raises:
        ConfigError: The URL is missing a host, a user, or a database name,
            each of which dbt would otherwise fail on with a message that does
            not mention the URL at all.
    """
    parsed = urlsplit(url)
    if parsed.scheme not in {"postgres", "postgresql"}:
        raise ConfigError(
            f"{name} must be a postgresql:// URL, got scheme "
            f"{parsed.scheme or '<none>'!r}."
        )
    if not parsed.hostname:
        raise ConfigError(f"{name} has no host.")
    if not parsed.username:
        raise ConfigError(f"{name} has no user.")

    dbname = unquote(parsed.path).lstrip("/")
    if not dbname:
        raise ConfigError(f"{name} has no database name.")

    query = parse_qs(parsed.query)
    return DatabaseParts(
        host=parsed.hostname,
        port=parsed.port or 5432,
        user=unquote(parsed.username),
        password=unquote(parsed.password) if parsed.password else None,
        dbname=dbname,
        sslmode=query.get("sslmode", [None])[0],
    )


@dataclass(frozen=True)
class Settings:
    """Resolved, validated configuration for one process."""

    environment: str
    database_url: str | None
    serving_database_url: str | None
    postgres_user: str
    postgres_password: str | None
    postgres_db: str
    postgres_port: int
    openmeteo_base_url: str
    request_connect_timeout_seconds: int
    request_timeout_seconds: int
    max_retry_attempts: int
    retry_backoff_seconds: int
    request_delay_seconds: float
    ingest_chunk_months: int
    ingest_hourly_months: int
    data_raw_dir: Path
    ingest_manifest_path: Path
    model_artifact_dir: Path
    cities_config_path: Path
    log_dir: Path
    dbt_profiles_dir: Path
    dbt_target: str
    log_level: str

    def require_database_url(self) -> str:
        """Return ``DATABASE_URL`` or explain precisely how to set it."""
        if not self.database_url:
            raise ConfigError(
                "DATABASE_URL is not set. Copy .env.example to .env and fill "
                "in the connection string for your target warehouse "
                "(see §5.4 of docs/proposal.md)."
            )
        return self.database_url

    def require_serving_database_url(self) -> str:
        """Return the Neon promotion target, or explain how to set it."""
        if not self.serving_database_url:
            raise ConfigError(
                "SERVING_DATABASE_URL is not set. It is required only when "
                "promoting gold marts to Neon; set it in .env before running "
                "the promotion step."
            )
        return self.serving_database_url

    def database_parts(self) -> DatabaseParts:
        """``DATABASE_URL`` split into the fields dbt's profile reads."""
        return split_database_url(self.require_database_url())

    def serving_database_parts(self) -> DatabaseParts:
        """``SERVING_DATABASE_URL`` split the same way."""
        return split_database_url(
            self.require_serving_database_url(), name="SERVING_DATABASE_URL"
        )

    def redacted(self) -> dict[str, str]:
        """Every setting as strings, with secrets masked. Safe to log."""
        out: dict[str, str] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in _SECRET_FIELDS:
                out[field.name] = mask_secret(value)
            else:
                out[field.name] = "<unset>" if value is None else str(value)
        return out


def _build_settings() -> Settings:
    environment = _get_str("ENVIRONMENT", "local").lower()
    if environment not in _VALID_ENVIRONMENTS:
        raise ConfigError(
            f"ENVIRONMENT must be one of {sorted(_VALID_ENVIRONMENTS)}, "
            f"got {environment!r}."
        )

    log_level = _get_str("LOG_LEVEL", "INFO").upper()
    if log_level not in _VALID_LOG_LEVELS:
        raise ConfigError(
            f"LOG_LEVEL must be one of {sorted(_VALID_LOG_LEVELS)}, "
            f"got {log_level!r}."
        )

    dbt_target = _get_str("DBT_TARGET", "dev").lower()
    if dbt_target not in {"dev", "prod"}:
        raise ConfigError(f"DBT_TARGET must be 'dev' or 'prod', got {dbt_target!r}.")

    return Settings(
        environment=environment,
        database_url=_get("DATABASE_URL"),
        serving_database_url=_get("SERVING_DATABASE_URL"),
        postgres_user=_get_str("POSTGRES_USER", "climate"),
        postgres_password=_get("POSTGRES_PASSWORD"),
        postgres_db=_get_str("POSTGRES_DB", "climate"),
        postgres_port=_get_int("POSTGRES_PORT", 5432),
        openmeteo_base_url=_get_str(
            "OPENMETEO_BASE_URL", "https://archive-api.open-meteo.com/v1/archive"
        ),
        request_connect_timeout_seconds=_get_positive_int(
            "REQUEST_CONNECT_TIMEOUT_SECONDS", 10
        ),
        request_timeout_seconds=_get_positive_int("REQUEST_TIMEOUT_SECONDS", 30),
        max_retry_attempts=_get_positive_int("MAX_RETRY_ATTEMPTS", 5),
        retry_backoff_seconds=_get_positive_int("RETRY_BACKOFF_SECONDS", 2),
        request_delay_seconds=_get_positive_float("REQUEST_DELAY_SECONDS", 1.0),
        ingest_chunk_months=_get_positive_int("INGEST_CHUNK_MONTHS", 12),
        ingest_hourly_months=_get_positive_int("INGEST_HOURLY_MONTHS", 24),
        data_raw_dir=_get_path("DATA_RAW_DIR", "data/raw"),
        ingest_manifest_path=_get_path(
            "INGEST_MANIFEST_PATH", "data/manifest.jsonl"
        ),
        model_artifact_dir=_get_path("MODEL_ARTIFACT_DIR", "machine_learning/artifacts"),
        cities_config_path=_get_path("CITIES_CONFIG_PATH", "config/cities.yml"),
        log_dir=_get_path("LOG_DIR", "logs"),
        dbt_profiles_dir=_get_path("DBT_PROFILES_DIR", "dbt_analytics"),
        dbt_target=dbt_target,
        log_level=log_level,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, building them on first call."""
    return _build_settings()


def reload_settings() -> Settings:
    """Rebuild settings after the environment changes. Intended for tests."""
    get_settings.cache_clear()
    load_dotenv(ENV_FILE, override=False)
    return get_settings()


if __name__ == "__main__":
    resolved = get_settings()
    width = max(len(name) for name in resolved.redacted())
    print(f"Loaded from: {ENV_FILE if ENV_FILE.exists() else '<no .env, using defaults>'}\n")
    for key, value in resolved.redacted().items():
        print(f"  {key.ljust(width)}  {value}")
