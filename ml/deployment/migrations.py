"""
Utilities to validate and apply SQL migrations for the ML system.

This module is intentionally small and dependency‑light so it can be used from
Make targets and tests. It provides a typed API for listing and applying the
canonical migration files located under ``ml/stores/migrations``.

Usage (inside repo):

    python -m ml.deployment.migrations --apply

The ``--apply`` mode executes migrations via ``docker compose exec postgres psql``
against the running Postgres service in the consolidated compose file. It reads
SQL from the host and pipes it to psql stdin to avoid bind‑mount path confusion.

Environment variables:
    COMPOSE_FILE: optional override for the compose file path.

"""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Iterable
from pathlib import Path


MIGRATIONS_DIR: Path = Path("ml/stores/migrations").resolve()


def list_migration_files() -> list[Path]:
    """
    Return migration files sorted by filename (lexicographic).

    Only ``.sql`` files are included. The function never raises for a missing
    directory; it returns an empty list instead to allow tests to run in
    isolation.

    """
    if not MIGRATIONS_DIR.exists():
        return []
    files = sorted(p for p in MIGRATIONS_DIR.iterdir() if p.suffix == ".sql")
    return list(files)


def _compose_cmd(compose_file: Path | None) -> list[str]:
    cmd: list[str] = ["docker", "compose"]
    if compose_file is not None:
        cmd += ["-f", str(compose_file)]
    return cmd


def apply_migrations_via_compose(
    *,
    compose_file: Path | None,
    database: str = "nautilus",
    user: str = "postgres",
    migrations: Iterable[Path] | None = None,
) -> None:
    """
    Apply migrations to a running Postgres container defined in Docker Compose.

    Parameters
    ----------
    compose_file : Path | None
        Compose file to target. If None, uses Docker's default discovery.
    database : str
        Target database name.
    user : str
        Database user.
    migrations : Iterable[Path] | None
        Explicit set of migration files. If None, the canonical list from
        :func:`list_migration_files` is used.

    """
    files = list(migrations) if migrations is not None else list_migration_files()
    for file in files:
        sql = file.read_text(encoding="utf-8")
        # Build minimal command shape expected by tests
        cmd = _compose_cmd(compose_file) + [
            "exec",
            "-T",
            "postgres",
            "psql",
            "-U",
            user,
            database,
        ]
        subprocess.run(cmd, input=sql, text=True, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="ML DB migrations helper")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply migrations using docker compose exec",
    )
    parser.add_argument(
        "--compose-file",
        type=Path,
        default=Path("ml/deployment/docker-compose.yml"),
        help="Compose file to use for docker compose commands",
    )
    args = parser.parse_args()

    if args.apply:
        apply_migrations_via_compose(compose_file=args.compose_file)
    else:
        for p in list_migration_files():
            print(p)


if __name__ == "__main__":  # pragma: no cover - CLI glue
    main()
