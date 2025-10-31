from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from ml.scripts import validate_wave
from ml.common.subprocess_utils import SubprocessExecutionError


def test_validate_wave_runs_all_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded: list[tuple[str, ...]] = []

    def _fake_run_command(cmd: list[str]) -> None:
        recorded.append(tuple(cmd))

    monkeypatch.setattr(validate_wave, "run_command", _fake_run_command)

    doc_path = tmp_path / "doc.md"
    doc_path.write_text("fresh", encoding="utf-8")

    monkeypatch.setattr(validate_wave, "DEFAULT_DOC_PATHS", (doc_path,))
    monkeypatch.setattr(validate_wave, "DEFAULT_PYTEST_TARGETS", ("tests/unit/dummy.py",))

    exit_code = validate_wave.main(
        [
            "--manifest-dir",
            str(tmp_path),
            "--manifest-limit",
            "1",
            "--max-doc-age-hours",
            "1000",
        ],
    )

    assert exit_code == 0
    assert ("poetry", "run", "mypy", "ml", "--strict") in recorded
    assert ("poetry", "run", "ruff", "check", "ml") in recorded
    assert any(cmd[0:3] == ("poetry", "run", "pytest") for cmd in recorded)


def test_validate_wave_handles_command_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _raise(_: list[str]) -> None:
        raise SubprocessExecutionError(command=("poetry", "run"), returncode=1, stdout=None, stderr=None)

    monkeypatch.setattr(validate_wave, "run_command", _raise)
    monkeypatch.setattr(validate_wave, "DEFAULT_DOC_PATHS", (tmp_path / "doc.md",))
    (tmp_path / "doc.md").write_text("fresh", encoding="utf-8")

    exit_code = validate_wave.main(["--manifest-dir", str(tmp_path), "--manifest-limit", "1", "--max-doc-age-hours", "1000"])
    assert exit_code == 1


def test_validate_wave_detects_stale_docs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    doc_path = tmp_path / "doc.md"
    doc_path.write_text("stale", encoding="utf-8")
    old_ts = time.time() - 3600.0 * 200.0
    os.utime(doc_path, (old_ts, old_ts))

    monkeypatch.setattr(validate_wave, "DEFAULT_DOC_PATHS", (doc_path,))

    def _noop(cmd: list[str]) -> None:
        return None

    monkeypatch.setattr(validate_wave, "run_command", _noop)

    exit_code = validate_wave.main(
        [
            "--manifest-dir",
            str(tmp_path),
            "--manifest-limit",
            "1",
            "--max-doc-age-hours",
            "1",
        ],
    )
    assert exit_code == 2
