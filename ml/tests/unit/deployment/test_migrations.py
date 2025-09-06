from __future__ import annotations

from pathlib import Path
from typing import List

import builtins
import types

from ml.deployment import migrations as mig


def test_list_migration_files_returns_sorted_list(monkeypatch: object, tmp_path: Path) -> None:
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
    monkeypatch.setattr(mig, "MIGRATIONS_DIR", dirp, raising=True)  # type: ignore[arg-type]

    files = mig.list_migration_files()
    assert [p.name for p in files] == ["001_a.sql", "002_b.sql", "003_c.sql"]


def test_apply_migrations_builds_compose_commands(monkeypatch: object, tmp_path: Path) -> None:
    # Create a single migration file
    file = tmp_path / "001.sql"
    file.write_text("SELECT 1;", encoding="utf-8")

    # Patch subprocess.run to capture calls
    calls: List[list[str]] = []

    def fake_run(cmd: list[str], input: str, text: bool, check: bool) -> types.SimpleNamespace:  # type: ignore[no-redef]
        calls.append(cmd)
        # Validate some invariants of the command
        assert cmd[:2] == ["docker", "compose"]
        assert cmd[-5:] == ["-U", "postgres", "nautilus"]
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(mig.subprocess, "run", fake_run)  # type: ignore[arg-type]

    mig.apply_migrations_via_compose(compose_file=None, migrations=[file])
    assert len(calls) == 1

