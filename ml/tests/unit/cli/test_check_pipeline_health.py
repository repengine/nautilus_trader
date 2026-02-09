from __future__ import annotations

import json

import pytest

import ml.cli.check_pipeline_health as cli
from ml.monitoring.health import ComponentHealth
from ml.monitoring.health import HealthStatus


def _disable_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    def _noop_configure_logging() -> None:
        return None

    def _noop_bind_log_context(**_: object) -> None:
        return None

    monkeypatch.setattr(cli, "configure_logging", _noop_configure_logging)
    monkeypatch.setattr(cli, "bind_log_context", _noop_bind_log_context)


def test_main_outputs_json_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _disable_logging(monkeypatch)

    component_health = {
        "pipeline": ComponentHealth(
            name="Pipeline",
            status=HealthStatus.WARNING,
            message="warn",
            metrics={"health_score": 99},
            issues=["stale"],
        ),
    }

    def _fake_run(connection_string: str) -> tuple[dict[str, ComponentHealth], HealthStatus, int]:
        assert connection_string == "postgresql://example"
        return component_health, HealthStatus.WARNING, 1

    monkeypatch.setattr(cli, "HAS_PSYCOPG2", True)
    monkeypatch.setattr(cli, "run_pipeline_health_checks", _fake_run)

    rc = cli.main(["--connection-string", "postgresql://example", "--json"])
    out = capsys.readouterr().out
    decoded = json.loads(out)

    assert rc == 1
    assert decoded["overall_status"] == "warning"
    assert decoded["components"]["pipeline"]["issues"] == ["stale"]


def test_main_critical_only_without_critical_issues_returns_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _disable_logging(monkeypatch)

    component_health = {
        "pipeline": ComponentHealth(
            name="Pipeline",
            status=HealthStatus.HEALTHY,
            message="ok",
            metrics={},
            issues=[],
        ),
    }

    def _fake_run(_connection_string: str) -> tuple[dict[str, ComponentHealth], HealthStatus, int]:
        return component_health, HealthStatus.HEALTHY, 0

    monkeypatch.setattr(cli, "HAS_PSYCOPG2", True)
    monkeypatch.setattr(cli, "run_pipeline_health_checks", _fake_run)

    rc = cli.main(["--critical-only"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "No critical issues found" in out


def test_main_returns_two_when_psycopg2_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _disable_logging(monkeypatch)
    monkeypatch.setattr(cli, "HAS_PSYCOPG2", False)

    rc = cli.main([])
    err = capsys.readouterr().err

    assert rc == 2
    assert "psycopg2 is required" in err
