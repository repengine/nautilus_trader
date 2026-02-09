from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest

import ml.monitoring.health as health_module
from ml.monitoring.health import ComponentHealth
from ml.monitoring.health import HealthStatus
from ml.monitoring.health import PipelineHealthChecker


class _FakePsycopgError(Exception):
    pass


class _FakeCursor:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        error: Exception | None = None,
    ) -> None:
        self._rows = rows
        self._error = error
        self.last_query: str | None = None

    def execute(self, query: str) -> None:
        self.last_query = query
        if self._error is not None:
            raise self._error

    def fetchall(self) -> list[dict[str, Any]]:
        return self._rows


class _FakeCursorContext:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def __enter__(self) -> _FakeCursor:
        return self._cursor

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any,
    ) -> None:
        return None


class _FakeConnection:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor
        self.closed = False
        self.cursor_factory: object | None = None

    def cursor(self, *, cursor_factory: object | None = None) -> _FakeCursorContext:
        self.cursor_factory = cursor_factory
        return _FakeCursorContext(self._cursor)

    def close(self) -> None:
        self.closed = True


def test_component_health_defaults_issues_list() -> None:
    component = ComponentHealth(
        name="Pipeline",
        status=HealthStatus.HEALTHY,
        message="ok",
        metrics={},
    )

    assert component.issues == []


def test_tabulate_fallback_formats_rows() -> None:
    rendered = health_module._tabulate_fallback(
        data=[["Pipeline", "healthy", "No"], ["Errors", "warning", "Yes"]],
        headers=["Component", "Status", "Has Issues"],
    )

    assert "Component" in rendered
    assert "Pipeline" in rendered
    assert "Errors" in rendered


def test_connect_and_disconnect_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_connection = _FakeConnection(_FakeCursor([]))
    fake_psycopg = SimpleNamespace(
        Error=_FakePsycopgError,
        connect=lambda _conn: fake_connection,
    )
    monkeypatch.setattr(health_module, "psycopg2", fake_psycopg, raising=False)

    checker = PipelineHealthChecker("postgresql://localhost/ml")
    checker.connect()
    checker.disconnect()

    assert checker._conn is None
    assert fake_connection.closed is True


def test_connect_raises_connection_error_on_driver_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_connect(_conn: str) -> Any:
        raise _FakePsycopgError("boom")

    fake_psycopg = SimpleNamespace(Error=_FakePsycopgError, connect=_raise_connect)
    monkeypatch.setattr(health_module, "psycopg2", fake_psycopg, raising=False)

    checker = PipelineHealthChecker("postgresql://localhost/ml")

    with pytest.raises(ConnectionError, match="Failed to connect to database"):
        checker.connect()


def test_execute_query_requires_connection() -> None:
    checker = PipelineHealthChecker("postgresql://localhost/ml")

    with pytest.raises(RuntimeError, match="Not connected to database"):
        checker._execute_query("SELECT 1")


def test_execute_query_returns_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor = _FakeCursor([{"value": 1}])
    checker = PipelineHealthChecker("postgresql://localhost/ml")
    checker._conn = _FakeConnection(cursor)

    marker_cursor = object()
    monkeypatch.setattr(health_module, "RealDictCursor", marker_cursor, raising=False)
    monkeypatch.setattr(
        health_module,
        "psycopg2",
        SimpleNamespace(Error=_FakePsycopgError),
        raising=False,
    )

    rows = checker._execute_query("SELECT 1")

    assert rows == [{"value": 1}]
    assert checker._conn.cursor_factory is marker_cursor
    assert cursor.last_query == "SELECT 1"


def test_execute_query_wraps_database_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor = _FakeCursor([], error=_FakePsycopgError("bad query"))
    checker = PipelineHealthChecker("postgresql://localhost/ml")
    checker._conn = _FakeConnection(cursor)
    monkeypatch.setattr(
        health_module,
        "psycopg2",
        SimpleNamespace(Error=_FakePsycopgError),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="Query execution failed"):
        checker._execute_query("SELECT broken")


@pytest.mark.parametrize(
    ("rows", "expected_status"),
    [
        (
            [
                {
                    "staleness_seconds": 90000.0,
                    "health_score": 70.0,
                    "instruments_processed": 4,
                    "total_features": 10,
                    "last_update_time": datetime(2026, 1, 1, 0, 0, 0),
                },
            ],
            HealthStatus.CRITICAL,
        ),
        (
            [
                {
                    "staleness_seconds": 7200.0,
                    "health_score": 80.0,
                    "instruments_processed": 2,
                    "total_features": 10,
                    "last_update_time": datetime(2026, 1, 1, 0, 0, 0),
                },
            ],
            HealthStatus.WARNING,
        ),
        (
            [
                {
                    "staleness_seconds": 10.0,
                    "health_score": 95.0,
                    "instruments_processed": 0,
                    "total_features": 10,
                    "last_update_time": datetime(2026, 1, 1, 0, 0, 0),
                },
            ],
            HealthStatus.WARNING,
        ),
        (
            [
                {
                    "staleness_seconds": 10.0,
                    "health_score": 99.0,
                    "instruments_processed": 2,
                    "total_features": 10,
                    "last_update_time": datetime(2026, 1, 1, 0, 0, 0),
                },
            ],
            HealthStatus.HEALTHY,
        ),
    ],
)
def test_check_pipeline_health_status_branches(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict[str, Any]],
    expected_status: HealthStatus,
) -> None:
    checker = PipelineHealthChecker("postgresql://localhost/ml")
    monkeypatch.setattr(checker, "_execute_query", lambda _q: rows)

    component = checker.check_pipeline_health()

    assert component.status == expected_status
    assert component.metrics["days_with_data"] == 1


def test_check_pipeline_health_handles_no_results(monkeypatch: pytest.MonkeyPatch) -> None:
    checker = PipelineHealthChecker("postgresql://localhost/ml")
    monkeypatch.setattr(checker, "_execute_query", lambda _q: [])

    component = checker.check_pipeline_health()

    assert component.status == HealthStatus.UNKNOWN
    assert component.message == "No pipeline data available"


def test_check_data_collection_status_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    checker = PipelineHealthChecker("postgresql://localhost/ml")

    monkeypatch.setattr(checker, "_execute_query", lambda _q: [{"total_records": None}])
    none_case = checker.check_data_collection()
    assert none_case.status == HealthStatus.CRITICAL

    monkeypatch.setattr(
        checker,
        "_execute_query",
        lambda _q: [
            {
                "instruments": 4,
                "total_records": 1200,
                "avg_rate": 20.5,
                "gaps_count": 2,
            },
        ],
    )
    warning_case = checker.check_data_collection()
    assert warning_case.status == HealthStatus.WARNING
    assert "collection gaps" in warning_case.issues[0]

    monkeypatch.setattr(
        checker,
        "_execute_query",
        lambda _q: [
            {
                "instruments": 5,
                "total_records": 1800,
                "avg_rate": 30.0,
                "gaps_count": 0,
            },
        ],
    )
    healthy_case = checker.check_data_collection()
    assert healthy_case.status == HealthStatus.HEALTHY
    assert healthy_case.metrics["records_last_hour"] == 1800


def test_check_feature_computation_handles_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    checker = PipelineHealthChecker("postgresql://localhost/ml")

    def _raise_query(_query: str) -> list[dict[str, Any]]:
        raise RuntimeError("db down")

    monkeypatch.setattr(checker, "_execute_query", _raise_query)

    component = checker.check_feature_computation()

    assert component.status == HealthStatus.WARNING
    assert component.message == "No feature computations today"


@pytest.mark.parametrize(
    ("avg_latency_ms", "max_latency_ms", "expected"),
    [
        (220.0, 240.0, HealthStatus.WARNING),
        (100.0, 700.0, HealthStatus.CRITICAL),
        (50.0, 120.0, HealthStatus.HEALTHY),
    ],
)
def test_check_feature_computation_latency_branches(
    monkeypatch: pytest.MonkeyPatch,
    avg_latency_ms: float,
    max_latency_ms: float,
    expected: HealthStatus,
) -> None:
    checker = PipelineHealthChecker("postgresql://localhost/ml")
    monkeypatch.setattr(
        checker,
        "_execute_query",
        lambda _q: [
            {
                "instruments": 3,
                "avg_latency_ms": avg_latency_ms,
                "max_latency_ms": max_latency_ms,
            },
        ],
    )

    component = checker.check_feature_computation()

    assert component.status == expected
    assert component.metrics["instruments"] == 3


def test_check_data_freshness_status_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> _FixedDateTime:
            return cls(2026, 1, 1, 12, 0, 0, tzinfo=tz)

    monkeypatch.setattr(health_module, "datetime", _FixedDateTime)
    checker = PipelineHealthChecker("postgresql://localhost/ml")
    now_ns = int(_FixedDateTime.now().timestamp() * 1e9)

    monkeypatch.setattr(checker, "_execute_query", lambda _q: [])
    none_case = checker.check_data_freshness()
    assert none_case.status == HealthStatus.WARNING

    monkeypatch.setattr(
        checker,
        "_execute_query",
        lambda _q: [
            {"instrument_id": "AAPL", "last_update_ns": now_ns - int(90000 * 1e9)},
            {"instrument_id": "QQQ", "last_update_ns": 0},
        ],
    )
    critical_case = checker.check_data_freshness()
    assert critical_case.status == HealthStatus.CRITICAL
    assert critical_case.metrics["critical_count"] == 1
    assert critical_case.metrics["no_data_count"] == 1

    monkeypatch.setattr(
        checker,
        "_execute_query",
        lambda _q: [
            {"instrument_id": "SPY", "last_update_ns": now_ns - int(5400 * 1e9)},
            {"instrument_id": "QQQ", "last_update_ns": now_ns - int(300 * 1e9)},
        ],
    )
    warning_case = checker.check_data_freshness()
    assert warning_case.status == HealthStatus.WARNING
    assert warning_case.metrics["warning_count"] == 1
    assert warning_case.metrics["fresh_count"] == 1

    monkeypatch.setattr(
        checker,
        "_execute_query",
        lambda _q: [
            {"instrument_id": "MSFT", "last_update_ns": now_ns - int(300 * 1e9)},
            {"instrument_id": "NVDA", "last_update_ns": now_ns - int(100 * 1e9)},
        ],
    )
    healthy_case = checker.check_data_freshness()
    assert healthy_case.status == HealthStatus.HEALTHY
    assert healthy_case.metrics["fresh_count"] == 2


def test_check_errors_status_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    checker = PipelineHealthChecker("postgresql://localhost/ml")

    monkeypatch.setattr(checker, "_execute_query", lambda _q: [])
    empty_case = checker.check_errors()
    assert empty_case.status == HealthStatus.HEALTHY

    monkeypatch.setattr(
        checker,
        "_execute_query",
        lambda _q: [
            {"error_type": "timeout", "total_errors": 40, "affected_components": 2},
            {"error_type": "network", "total_errors": 20, "affected_components": 1},
        ],
    )
    warning_case = checker.check_errors()
    assert warning_case.status == HealthStatus.WARNING
    assert warning_case.metrics["total_errors"] == 60

    monkeypatch.setattr(
        checker,
        "_execute_query",
        lambda _q: [
            {"error_type": "inference", "total_errors": 140, "affected_components": 2},
            {"error_type": "feature", "total_errors": 10, "affected_components": 1},
        ],
    )
    critical_case = checker.check_errors()
    assert critical_case.status == HealthStatus.CRITICAL
    assert any("inference errors" in issue for issue in critical_case.issues)


def test_check_model_performance_status_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    checker = PipelineHealthChecker("postgresql://localhost/ml")

    monkeypatch.setattr(checker, "_execute_query", lambda _q: [{"model_count": 0}])
    no_models = checker.check_model_performance()
    assert no_models.status == HealthStatus.WARNING

    monkeypatch.setattr(
        checker,
        "_execute_query",
        lambda _q: [
            {
                "model_count": 2,
                "overall_confidence": 0.2,
                "avg_p99_latency": 10.0,
                "max_p99_latency": 20.0,
                "unhealthy_count": 0,
            },
        ],
    )
    low_conf = checker.check_model_performance()
    assert low_conf.status == HealthStatus.WARNING

    monkeypatch.setattr(
        checker,
        "_execute_query",
        lambda _q: [
            {
                "model_count": 2,
                "overall_confidence": 0.9,
                "avg_p99_latency": 10.0,
                "max_p99_latency": 1200.0,
                "unhealthy_count": 0,
            },
        ],
    )
    high_latency = checker.check_model_performance()
    assert high_latency.status == HealthStatus.CRITICAL

    monkeypatch.setattr(
        checker,
        "_execute_query",
        lambda _q: [
            {
                "model_count": 2,
                "overall_confidence": 0.9,
                "avg_p99_latency": 10.0,
                "max_p99_latency": 20.0,
                "unhealthy_count": 1,
            },
        ],
    )
    unhealthy_case = checker.check_model_performance()
    assert unhealthy_case.status == HealthStatus.WARNING

    monkeypatch.setattr(
        checker,
        "_execute_query",
        lambda _q: [
            {
                "model_count": 2,
                "overall_confidence": 0.9,
                "avg_p99_latency": 10.0,
                "max_p99_latency": 20.0,
                "unhealthy_count": 0,
            },
        ],
    )
    healthy_case = checker.check_model_performance()
    assert healthy_case.status == HealthStatus.HEALTHY


def test_get_overall_status_priority_order() -> None:
    checker = PipelineHealthChecker("postgresql://localhost/ml")

    critical_map = {
        "pipeline": ComponentHealth("Pipeline", HealthStatus.CRITICAL, "bad", {}),
    }
    warning_map = {
        "pipeline": ComponentHealth("Pipeline", HealthStatus.WARNING, "warn", {}),
    }
    healthy_map = {
        "pipeline": ComponentHealth("Pipeline", HealthStatus.HEALTHY, "ok", {}),
    }

    assert checker.get_overall_status(critical_map) == (HealthStatus.CRITICAL, 2)
    assert checker.get_overall_status(warning_map) == (HealthStatus.WARNING, 1)
    assert checker.get_overall_status(healthy_map) == (HealthStatus.HEALTHY, 0)


def test_format_human_output_includes_summary_and_metrics() -> None:
    component_health = {
        "pipeline": ComponentHealth(
            name="Pipeline Overall",
            status=HealthStatus.WARNING,
            message="Needs attention",
            metrics={"latency_ms": 12.345, "records": 100},
            issues=["stale data"],
            last_update=datetime(2026, 1, 1, 0, 0, 0),
        ),
    }

    rendered = health_module.format_human_output(component_health, HealthStatus.WARNING)

    assert "ML PIPELINE HEALTH CHECK REPORT" in rendered
    assert "SUMMARY" in rendered
    assert "Pipeline Overall" in rendered
    assert "latency_ms: 12.35" in rendered
    assert "stale data" in rendered
