"""
Unit tests for HealthMonitor in base actor helpers.
"""

import pytest

from ml.actors.base import HealthMonitor, HealthStatus
from ml.config.base import HealthMonitorConfig


@pytest.mark.unit
def test_health_monitor_marks_unhealthy_when_model_not_loaded() -> None:
    monitor = HealthMonitor()

    monitor.set_model_loaded(False)

    assert monitor.status is HealthStatus.UNHEALTHY


@pytest.mark.unit
def test_health_monitor_degrades_on_failures_or_latency() -> None:
    config = HealthMonitorConfig(
        critical_consecutive_failures=5,
        degraded_success_rate_threshold=0.9,
        degraded_consecutive_failures=2,
        degraded_latency_violations=1,
    )
    monitor = HealthMonitor(config)

    monitor.set_model_loaded(True)
    monitor.set_indicators_initialized(True)
    monitor.update_prediction_success()
    monitor.update_prediction_failure()
    monitor.update_latency_violation()

    assert monitor.status is HealthStatus.DEGRADED


@pytest.mark.unit
def test_health_monitor_marks_unhealthy_after_critical_failures() -> None:
    config = HealthMonitorConfig(
        critical_consecutive_failures=1,
        degraded_success_rate_threshold=0.1,
        degraded_consecutive_failures=1,
        degraded_latency_violations=5,
    )
    monitor = HealthMonitor(config)

    monitor.set_model_loaded(True)
    monitor.update_prediction_failure()
    monitor.update_prediction_failure()

    assert monitor.status is HealthStatus.UNHEALTHY


@pytest.mark.unit
def test_health_monitor_to_dict_reports_state() -> None:
    monitor = HealthMonitor()

    monitor.set_model_loaded(True)
    monitor.set_indicators_initialized(True)
    monitor.update_prediction_success()

    payload = monitor.to_dict()

    assert payload["status"] == monitor.status.value
    assert payload["total_predictions"] == 1
    assert payload["failed_predictions"] == 0
