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
        def run_coverage_restoration_once(self) -> CoverageStatus:
            return summary

    monkeypatch.setattr(coverage_restore, "PipelineRunner", lambda: _Runner())

    exit_code = coverage_restore.main(["--json"])
    assert exit_code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["buckets_total"] == 5
    assert parsed["buckets_restore_catalog"] == 3
