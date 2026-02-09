from __future__ import annotations

import json
from datetime import datetime
from typing import Self

import pytest

import ml.monitoring.health as health_module
from ml.monitoring.health import ComponentHealth
from ml.monitoring.health import HealthStatus
from ml.monitoring.health import PipelineHealthChecker


def test_run_pipeline_health_checks_delegates_to_checker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    expected = {
        "pipeline": ComponentHealth(
            name="Pipeline",
            status=HealthStatus.WARNING,
            message="warn",
            metrics={},
        ),
    }

    class _StubChecker:
        def __init__(self, connection_string: str) -> None:
            captured["connection_string"] = connection_string

        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            _exc_type: object,
            _exc_val: object,
            _exc_tb: object,
        ) -> None:
            return None

        def check_all_components(self) -> dict[str, ComponentHealth]:
            return expected

        def get_overall_status(
            self,
            component_health: dict[str, ComponentHealth],
        ) -> tuple[HealthStatus, int]:
            captured["component_health"] = component_health
            return (HealthStatus.WARNING, 1)

    monkeypatch.setattr(health_module, "PipelineHealthChecker", _StubChecker)

    component_health, overall_status, exit_code = health_module.run_pipeline_health_checks(
        "postgresql://localhost/ml",
    )

    assert captured["connection_string"] == "postgresql://localhost/ml"
    assert captured["component_health"] == expected
    assert component_health == expected
    assert overall_status == HealthStatus.WARNING
    assert exit_code == 1


def test_check_all_components_maps_failures_to_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = PipelineHealthChecker("postgresql://localhost/ml")

    def _ok(name: str) -> ComponentHealth:
        return ComponentHealth(
            name=name,
            status=HealthStatus.HEALTHY,
            message="ok",
            metrics={},
        )

    def _boom() -> ComponentHealth:
        raise RuntimeError("boom")

    monkeypatch.setattr(checker, "check_pipeline_health", _boom)
    monkeypatch.setattr(checker, "check_data_collection", lambda: _ok("Data Collection"))
    monkeypatch.setattr(checker, "check_feature_computation", lambda: _ok("Feature Computation"))
    monkeypatch.setattr(checker, "check_data_freshness", lambda: _ok("Data Freshness"))
    monkeypatch.setattr(checker, "check_errors", lambda: _ok("Error Monitoring"))
    monkeypatch.setattr(checker, "check_model_performance", lambda: _ok("Model Performance"))

    results = checker.check_all_components()

    assert results["pipeline"].status == HealthStatus.UNKNOWN
    assert results["pipeline"].issues == ["boom"]
    assert "boom" in results["pipeline"].message
    assert results["data_collection"].status == HealthStatus.HEALTHY


def test_format_json_output_serializes_component_payload() -> None:
    component_health = {
        "pipeline": ComponentHealth(
            name="Pipeline",
            status=HealthStatus.HEALTHY,
            message="ready",
            metrics={"score": 100},
            last_update=datetime(2025, 1, 1, 0, 0, 0),
            issues=[],
        ),
    }

    payload = health_module.format_json_output(
        component_health,
        HealthStatus.HEALTHY,
        0,
    )
    decoded = json.loads(payload)

    assert decoded["overall_status"] == "healthy"
    assert decoded["exit_code"] == 0
    assert decoded["components"]["pipeline"]["name"] == "Pipeline"
    assert decoded["components"]["pipeline"]["status"] == "healthy"
    assert decoded["components"]["pipeline"]["last_update"] == "2025-01-01T00:00:00"
