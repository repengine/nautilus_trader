#!/usr/bin/env python3

"""
Test canary deployment functionality.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ml.registry.canary import CanaryConfig
from ml.registry.canary import CanaryDeployment


class TestCanaryDeployment:
    """Test canary deployment functionality."""

    def test_canary_config_defaults(self) -> None:
        """Test CanaryConfig has sensible defaults."""
        config = CanaryConfig()

        assert config.traffic_percentage == 5.0
        assert config.success_metric == "accuracy"
        assert config.baseline_threshold == 0.95
        assert config.monitoring_duration_hours == 24
        assert config.auto_promote is True
        assert config.auto_rollback is True
        assert config.min_samples == 100
        assert config.error_rate_threshold == 0.05

    def test_canary_deployment_initialization(self) -> None:
        """Test CanaryDeployment initializes correctly."""
        config = CanaryConfig(traffic_percentage=10.0)
        deployment = CanaryDeployment(
            deployment_id="canary_001",
            model_id="model_v2",
            config=config,
            baseline_performance=0.90,
        )

        assert deployment.deployment_id == "canary_001"
        assert deployment.model_id == "model_v2"
        assert deployment.config.traffic_percentage == 10.0
        assert deployment.baseline_performance == 0.90
        assert deployment.status == "active"
        assert deployment.metrics["sample_count"] == 0

    def test_record_metric_success(self) -> None:
        """Test recording successful metrics."""
        deployment = CanaryDeployment(
            deployment_id="canary_001",
            model_id="model_v2",
            config=CanaryConfig(),
        )

        deployment.record_metric(
            metric_value=0.92,
            latency_ms=15.0,
            error_occurred=False,
        )

        assert deployment.metrics["sample_count"] == 1
        assert deployment.metrics["success_count"] == 1
        assert deployment.metrics["error_count"] == 0
        assert deployment.metrics["metric_sum"] == 0.92
        assert deployment.metrics["metric_values"] == [0.92]
        assert deployment.metrics["latency_values"] == [15.0]

    def test_record_metric_error(self) -> None:
        """Test recording error metrics."""
        deployment = CanaryDeployment(
            deployment_id="canary_001",
            model_id="model_v2",
            config=CanaryConfig(),
        )

        deployment.record_metric(
            metric_value=0.0,
            latency_ms=100.0,
            error_occurred=True,
        )

        assert deployment.metrics["sample_count"] == 1
        assert deployment.metrics["success_count"] == 0
        assert deployment.metrics["error_count"] == 1
        assert deployment.metrics["metric_sum"] == 0.0
        assert deployment.metrics["metric_values"] == []
        assert deployment.metrics["latency_values"] == [100.0]

    def test_should_rollback_high_error_rate(self) -> None:
        """Test rollback triggered by high error rate."""
        deployment = CanaryDeployment(
            deployment_id="canary_001",
            model_id="model_v2",
            config=CanaryConfig(error_rate_threshold=0.05),
        )

        # Record mostly errors
        for _ in range(30):
            deployment.record_metric(0.0, error_occurred=True)
        for _ in range(5):
            deployment.record_metric(0.9, error_occurred=False)

        should_rollback, reason = deployment.should_rollback()

        assert should_rollback is True
        assert reason == "high_error_rate"

    def test_should_rollback_performance_degradation(self) -> None:
        """Test rollback triggered by performance degradation."""
        deployment = CanaryDeployment(
            deployment_id="canary_001",
            model_id="model_v2",
            config=CanaryConfig(baseline_threshold=0.95),
            baseline_performance=0.90,
        )

        # Record poor performance
        for _ in range(50):
            deployment.record_metric(0.80, error_occurred=False)

        should_rollback, reason = deployment.should_rollback()

        assert should_rollback is True
        assert reason == "performance_degradation"

    def test_should_rollback_insufficient_samples(self) -> None:
        """Test rollback not triggered with insufficient samples."""
        deployment = CanaryDeployment(
            deployment_id="canary_001",
            model_id="model_v2",
            config=CanaryConfig(min_samples=100),
        )

        # Record only a few samples
        for _ in range(5):
            deployment.record_metric(0.50, error_occurred=False)

        should_rollback, reason = deployment.should_rollback()

        assert should_rollback is False
        assert reason == "insufficient_samples"

    def test_should_promote_success(self) -> None:
        """Test successful promotion after monitoring period."""
        config = CanaryConfig(
            monitoring_duration_hours=1,
            min_samples=10,
            baseline_threshold=0.95,
        )

        # Create deployment with initial time
        deployment = CanaryDeployment(
            deployment_id="canary_001",
            model_id="model_v2",
            config=config,
            baseline_performance=0.90,
        )

        # Override created_at to a known value
        deployment.created_at = 1000000.0

        # Record good performance
        for _ in range(20):
            deployment.record_metric(0.92, error_occurred=False)

        # Mock time for the promotion check
        with patch("ml.registry.canary.time.time") as mock_time:
            # Fast forward time to after monitoring period
            mock_time.return_value = 1000000.0 + 3700  # > 1 hour

            should_promote, reason = deployment.should_promote()

            assert should_promote is True
            assert reason == "monitoring_period_complete"

    def test_should_promote_insufficient_samples(self) -> None:
        """Test promotion blocked by insufficient samples."""
        config = CanaryConfig(min_samples=100)

        deployment = CanaryDeployment(
            deployment_id="canary_001",
            model_id="model_v2",
            config=config,
        )

        # Record only a few samples
        for _ in range(5):
            deployment.record_metric(0.95, error_occurred=False)

        should_promote, reason = deployment.should_promote()

        assert should_promote is False
        assert reason == "insufficient_samples"

    def test_should_promote_high_error_rate(self) -> None:
        """Test promotion blocked by high error rate."""
        config = CanaryConfig(
            monitoring_duration_hours=1,
            min_samples=10,
            error_rate_threshold=0.05,
        )

        deployment = CanaryDeployment(
            deployment_id="canary_001",
            model_id="model_v2",
            config=config,
        )

        # Override created_at to a known value
        deployment.created_at = 1000000.0

        # Record mix of success and errors
        for _ in range(50):
            deployment.record_metric(0.92, error_occurred=False)
        for _ in range(10):
            deployment.record_metric(0.0, error_occurred=True)

        with patch("ml.registry.canary.time.time") as mock_time:
            # Fast forward time
            mock_time.return_value = 1000000.0 + 3700

            should_promote, reason = deployment.should_promote()

            assert should_promote is False
            assert reason == "high_error_rate"

    def test_get_status_summary(self) -> None:
        """Test status summary generation."""
        deployment = CanaryDeployment(
            deployment_id="canary_001",
            model_id="model_v2",
            config=CanaryConfig(traffic_percentage=15.0),
            baseline_performance=0.90,
        )

        # Record some metrics
        deployment.record_metric(0.91, latency_ms=10.0, error_occurred=False)
        deployment.record_metric(0.92, latency_ms=12.0, error_occurred=False)
        deployment.record_metric(0.0, latency_ms=100.0, error_occurred=True)

        summary = deployment.get_status_summary()

        assert summary["deployment_id"] == "canary_001"
        assert summary["model_id"] == "model_v2"
        assert summary["status"] == "active"
        assert summary["sample_count"] == 3
        assert summary["success_count"] == 2
        assert summary["error_count"] == 1
        assert summary["error_rate"] == pytest.approx(0.333, rel=0.01)
        assert summary["current_performance"] == 0.915
        assert summary["baseline_performance"] == 0.90
        assert summary["relative_performance"] > 1.0
        assert summary["average_latency_ms"] == pytest.approx(40.67, rel=0.01)
        assert summary["traffic_percentage"] == 15.0
        assert "should_promote" in summary
        assert "promote_reason" in summary
        assert "should_rollback" in summary
        assert "rollback_reason" in summary

    def test_inactive_deployment_no_decisions(self) -> None:
        """Test that inactive deployments don't trigger decisions."""
        deployment = CanaryDeployment(
            deployment_id="canary_001",
            model_id="model_v2",
            config=CanaryConfig(),
        )

        deployment.status = "promoted"

        should_promote, promote_reason = deployment.should_promote()
        should_rollback, rollback_reason = deployment.should_rollback()

        assert should_promote is False
        assert promote_reason == "not_active"
        assert should_rollback is False
        assert rollback_reason == "not_active"
