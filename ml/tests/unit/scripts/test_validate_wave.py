from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from ml.scripts import validate_wave
from ml.common.subprocess_utils import SubprocessExecutionError


def _write_manifest(tmp_path: Path) -> None:
    metadata_path = tmp_path / "dataset_metadata.json"
    metadata_payload = {
        "column_info": {
            "group_id_col": "instrument_id",
            "time_idx_col": "time_index",
            "target_col": "y",
            "static_categoricals": ["instrument_id", "exchange"],
            "static_reals": ["tick_size"],
            "time_varying_known_reals": ["time_index", "hour_sin", "hour_cos"],
            "time_varying_unknown_reals": ["return_1"],
        },
    }
    metadata_path.write_text(json.dumps(metadata_payload), encoding="utf-8")

    manifest_payload = {
        "cohort_run": {"plan_id": "plan-1", "completed_at": "2026-01-01T00:00:00Z"},
        "dataset": {"paths": {"metadata": str(metadata_path)}},
    }
    manifest_path = tmp_path / "plan-1_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")


def test_validate_wave_runs_all_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded: list[tuple[str, ...]] = []

    def _fake_run_command(cmd: list[str]) -> None:
        recorded.append(tuple(cmd))

    monkeypatch.setattr(validate_wave, "run_command", _fake_run_command)

    doc_path = tmp_path / "doc.md"
    doc_path.write_text("fresh", encoding="utf-8")
    _write_manifest(tmp_path)

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
    _write_manifest(tmp_path)

    exit_code = validate_wave.main(["--manifest-dir", str(tmp_path), "--manifest-limit", "1", "--max-doc-age-hours", "1000"])
    assert exit_code == 1


def test_validate_wave_detects_stale_docs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    doc_path = tmp_path / "doc.md"
    doc_path.write_text("stale", encoding="utf-8")
    old_ts = time.time() - 3600.0 * 200.0
    os.utime(doc_path, (old_ts, old_ts))

    monkeypatch.setattr(validate_wave, "DEFAULT_DOC_PATHS", (doc_path,))
    _write_manifest(tmp_path)

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
