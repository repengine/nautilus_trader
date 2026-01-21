from __future__ import annotations

import json

import pytest

from ml.cli import coverage_restore
from ml.deployment.entrypoint_pipeline import CoverageStatus


def test_coverage_restore_cli_emits_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    summary: CoverageStatus = {
        "last_run": "2024-01-01T00:00:00Z",
        "last_success": "2024-01-01T00:00:00Z",
        "buckets_total": 5,
        "buckets_restore_catalog": 3,
        "buckets_reingest_source": 2,
        "buckets_healthy": 0,
        "last_error": None,
    }

    class _Runner:
        def run_coverage_restoration_once(self, *, dry_run: bool = False) -> CoverageStatus:
            assert dry_run is False
            return summary

    monkeypatch.setattr(coverage_restore, "PipelineRunner", lambda: _Runner())

    exit_code = coverage_restore.main(["--json"])
    assert exit_code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["buckets_total"] == 5
    assert parsed["buckets_restore_catalog"] == 3


def test_coverage_restore_cli_supports_dry_run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary: CoverageStatus = {
        "last_run": "2024-01-01T00:00:00Z",
        "last_success": "2024-01-01T00:00:00Z",
        "buckets_total": 1,
        "buckets_restore_catalog": 1,
        "buckets_reingest_source": 0,
        "buckets_healthy": 0,
        "last_error": None,
    }

    class _Runner:
        def run_coverage_restoration_once(self, *, dry_run: bool = False) -> CoverageStatus:
            assert dry_run is True
            return summary

    monkeypatch.setattr(coverage_restore, "PipelineRunner", lambda: _Runner())

    exit_code = coverage_restore.main(["--dry-run", "--json"])
    assert exit_code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["buckets_total"] == 1
