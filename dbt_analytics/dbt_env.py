"""Bridges DATABASE_URL to the environment variables dbt's profile reads.

dbt's postgres adapter takes host, user, password, port and dbname as separate
settings and has no way to accept a URL. The alternative to this file would be
a second set of variables in ``.env`` that could drift from ``DATABASE_URL``:
one of them updated, the other not, and a dbt run quietly building against
yesterday's database. So the URL stays the single source and this splits it.

Two ways to use it, both of which leave no credential on disk::

    eval "$(python dbt_analytics/dbt_env.py)"   # export into this shell
    python dbt_analytics/dbt_env.py -- dbt run  # inject and exec

``--write-profile`` generates the git-ignored ``profiles.yml`` from the
committed template. The generated file contains ``env_var()`` lookups, not
values, so it is a copy rather than a secret.

Run with no arguments to see the exports (passwords are printed, because that
is the point of ``eval``; do not paste the output anywhere).
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ConfigError, Settings, get_settings, mask_secret  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent
TEMPLATE = PROJECT_DIR / "profiles.yml.example"
PROFILE = PROJECT_DIR / "profiles.yml"

#: dev reads these, prod reads the DBT_SERVING_* set. Both are derived, never
#: configured: nothing in .env names a dbt variable.
DEV_PREFIX = "DBT"
SERVING_PREFIX = "DBT_SERVING"


def dbt_environment(settings: Settings | None = None) -> dict[str, str]:
    """Every variable the profile needs, derived from the configured URLs.

    The serving half is omitted rather than faked when ``SERVING_DATABASE_URL``
    is unset: local development does not need it, and a placeholder would let
    ``dbt run --target prod`` connect somewhere unintended instead of failing.
    """
    resolved = settings if settings is not None else get_settings()
    env = resolved.database_parts().as_env(DEV_PREFIX)
    env["DBT_TARGET"] = resolved.dbt_target
    env["DBT_PROFILES_DIR"] = str(resolved.dbt_profiles_dir)
    try:
        env.update(resolved.serving_database_parts().as_env(SERVING_PREFIX))
    except ConfigError:
        pass
    return env


def _exports(env: dict[str, str]) -> str:
    return "\n".join(f"export {key}={shlex.quote(value)}" for key, value in env.items())


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--write-profile",
        action="store_true",
        help="generate the git-ignored profiles.yml from profiles.yml.example",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="print the resolved variables with secrets masked",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="after --, a command to run with the variables injected",
    )
    args = parser.parse_args(argv)

    try:
        env = dbt_environment()
    except ConfigError as exc:
        print(f"FAILED  {exc}", file=sys.stderr)
        return 1

    if args.write_profile:
        if not TEMPLATE.exists():
            print(f"FAILED  {TEMPLATE} is missing", file=sys.stderr)
            return 1
        PROFILE.write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"wrote {PROFILE} (git-ignored; values stay as env_var lookups)")
        return 0

    if args.show:
        secret = ("PASSWORD",)
        width = max(len(k) for k in env)
        for key, value in sorted(env.items()):
            shown = mask_secret(value) if key.endswith(secret) else value
            print(f"  {key.ljust(width)}  {shown}")
        return 0

    command = [a for a in args.command if a != "--"]
    if command:
        os.execvpe(command[0], command, {**os.environ, **env})

    print(_exports(env))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
