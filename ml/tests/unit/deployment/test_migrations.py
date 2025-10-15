from __future__ import annotations

import builtins
import types
from pathlib import Path
from typing import Any, List

import pytest

from ml._imports import HAS_NAUTILUS_CORE
from ml._imports import NAUTILUS_CORE_IMPORT_ERROR

if not HAS_NAUTILUS_CORE:  # pragma: no cover - depends on native extensions
    pytest.skip(
        f"Nautilus Trader core extensions unavailable: {NAUTILUS_CORE_IMPORT_ERROR}",
        allow_module_level=True,
    )

from ml.deployment import migrations as mig


def test_list_migration_files_returns_sorted_list(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Create fake migration files
    dirp = tmp_path / "ml" / "stores" / "migrations"
    dirp.mkdir(parents=True)
    for name in [
        "003_c.sql",
        "001_a.sql",
        "002_b.sql",
    ]:
        (dirp / name).write_text("-- test", encoding="utf-8")

    # Point module constant to temp dir
    monkeypatch.setattr(mig, "MIGRATIONS_DIR", dirp, raising=True)

    files = mig.list_migration_files()
    assert [p.name for p in files] == ["001_a.sql", "002_b.sql", "003_c.sql"]


def test_apply_migrations_builds_compose_commands(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Create a single migration file
    file = tmp_path / "001.sql"
    file.write_text("SELECT 1;", encoding="utf-8")

    # Patch subprocess.run to capture calls
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], input: str, text: bool, check: bool) -> types.SimpleNamespace:
        calls.append(cmd)
        # Validate some invariants of the command
        assert cmd[:2] == ["docker", "compose"]
        # Allow additional flags (e.g., -v ON_ERROR_STOP=1); ensure suffix is correct
        assert cmd[-3:] == ["-U", "postgres", "nautilus"]
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(mig.subprocess, "run", fake_run)

    mig.apply_migrations_via_compose(compose_file=None, migrations=[file])
    assert len(calls) == 1
