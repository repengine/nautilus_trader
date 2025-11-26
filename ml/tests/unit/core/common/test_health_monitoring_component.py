"""
Unit tests for HealthMonitoringComponent.

This module tests the health monitoring component extracted from
MLIntegrationManager (Phase 3.6.4). Tests cover:

- Happy path: health checks, aggregation, protocol compliance
- Error conditions: unhealthy components, strict mode violations
- Edge cases: None components, missing methods

"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import PropertyMock

import pytest

from ml.common.protocols import MLComponentProtocol
from ml.core.common.health_monitoring import HealthMonitoringComponent


# =============================================================================
# Fixtures
# =============================================================================


class MockMLComponent:
    """Mock component that implements MLComponentProtocol."""

    def __init__(
        self,
        *,
        healthy: bool = True,
        health_status: dict[str, Any] | None = None,
        performance_metrics: dict[str, float] | None = None,
        config_issues: list[str] | None = None,
    ) -> None:
        self._healthy = healthy
        self._health_status = health_status or {"status": "ok"}
        self._performance_metrics = performance_metrics or {}
        self._config_issues = config_issues or []

    def get_health_status(self) -> dict[str, Any]:
        """Return health status."""
        if not self._healthy:
            raise RuntimeError("Component unhealthy")
        return self._health_status

    def get_performance_metrics(self) -> dict[str, float]:
        """Return performance metrics."""
        return self._performance_metrics

    def validate_configuration(self) -> list[str]:
        """Return configuration issues."""
        return self._config_issues


class MockStore:
    """Mock store with get_statistics method."""

    def __init__(self, *, healthy: bool = True) -> None:
        self._healthy = healthy

    def get_statistics(
        self,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> dict[str, Any]:
        """Return statistics."""
        if not self._healthy:
            raise RuntimeError("Store unhealthy")
        return {"count": 100}

    def flush(self) -> None:
        """Flush pending writes."""


class MockRegistry:
    """Mock registry with list methods."""

    def __init__(
        self,
        *,
        healthy: bool = True,
        items: list[Any] | None = None,
    ) -> None:
        self._healthy = healthy
        self._items = items or [{"id": "test"}]

    def list_features(self) -> list[Any]:
        """List features."""
        if not self._healthy:
            raise RuntimeError("Registry unhealthy")
        return self._items

    def list_models(self) -> list[Any]:
        """List models."""
        if not self._healthy:
            raise RuntimeError("Registry unhealthy")
        return self._items

    def list_strategies(self) -> list[Any]:
        """List strategies."""
        if not self._healthy:
            raise RuntimeError("Registry unhealthy")
        return self._items

    def list_datasets(self) -> list[Any]:
        """List datasets."""
        if not self._healthy:
            raise RuntimeError("Registry unhealthy")
        return self._items


class MockDataStore:
    """Mock DataStore with registry attribute."""

    def __init__(self, *, has_registry: bool = True) -> None:
        if has_registry:
            self.registry = MagicMock()


class MockPartitionManager:
    """Mock partition manager with get_partition_stats method."""

    def __init__(
        self,
        *,
        healthy: bool = True,
        stats: list[dict[str, Any]] | None = None,
    ) -> None:
        self._healthy = healthy
        # Use None check to allow empty list
        self._stats = stats if stats is not None else [{"partition": "2024-01"}]

    def get_partition_stats(self) -> list[dict[str, Any]]:
        """Return partition stats."""
        if not self._healthy:
            raise RuntimeError("Partition manager unhealthy")
        return self._stats


@pytest.fixture
def mock_healthy_stores() -> dict[str, MockStore]:
    """Provide healthy mock stores."""
    return {
        "feature_store": MockStore(healthy=True),
        "model_store": MockStore(healthy=True),
        "strategy_store": MockStore(healthy=True),
    }


@pytest.fixture
def mock_healthy_registries() -> dict[str, MockRegistry]:
    """Provide healthy mock registries."""
    return {
        "feature_registry": MockRegistry(healthy=True),
        "model_registry": MockRegistry(healthy=True),
        "strategy_registry": MockRegistry(healthy=True),
        "data_registry": MockRegistry(healthy=True),
    }


@pytest.fixture
def mock_ml_components() -> dict[str, MockMLComponent]:
    """Provide mock components implementing MLComponentProtocol."""
    return {
        "feature_store": MockMLComponent(healthy=True),
        "model_store": MockMLComponent(healthy=True),
        "strategy_store": MockMLComponent(healthy=True),
        "data_store": MockMLComponent(healthy=True),
        "feature_registry": MockMLComponent(healthy=True),
        "model_registry": MockMLComponent(healthy=True),
        "strategy_registry": MockMLComponent(healthy=True),
        "data_registry": MockMLComponent(healthy=True),
    }


@pytest.fixture
def healthy_monitoring_component(
    mock_healthy_stores: dict[str, MockStore],
    mock_healthy_registries: dict[str, MockRegistry],
) -> HealthMonitoringComponent:
    """Provide a fully healthy HealthMonitoringComponent."""
    data_store = MockDataStore(has_registry=True)
    partition_manager = MockPartitionManager(healthy=True)

    return HealthMonitoringComponent(
        feature_store=mock_healthy_stores["feature_store"],
        model_store=mock_healthy_stores["model_store"],
        strategy_store=mock_healthy_stores["strategy_store"],
        data_store=data_store,
        feature_registry=mock_healthy_registries["feature_registry"],
        model_registry=mock_healthy_registries["model_registry"],
        strategy_registry=mock_healthy_registries["strategy_registry"],
        data_registry=mock_healthy_registries["data_registry"],
        partition_manager=partition_manager,
        is_postgres_running=lambda: True,
    )


# =============================================================================
# Happy Path Tests
# =============================================================================


class TestHappyPath:
    """Tests for successful operation paths."""

    def test_check_health_returns_all_components(
        self,
        healthy_monitoring_component: HealthMonitoringComponent,
    ) -> None:
        """Verify health check covers all components.

        Input: Manager with initialized stores/registries.
        Expected Behavior: Dict with health status for each component.
        """
        health = healthy_monitoring_component.check_health()

        expected_keys = {
            "postgres",
            "feature_store",
            "model_store",
            "strategy_store",
            "feature_registry",
            "model_registry",
            "strategy_registry",
            "data_registry",
            "data_store",
            "partitions",
        }
        assert set(health.keys()) == expected_keys
        assert all(isinstance(v, bool) for v in health.values())

    def test_check_health_all_healthy_returns_all_true(
        self,
        healthy_monitoring_component: HealthMonitoringComponent,
    ) -> None:
        """Verify all components report healthy when properly configured.

        Input: All components healthy.
        Expected Behavior: All values True.
        """
        health = healthy_monitoring_component.check_health()

        assert health["postgres"] is True
        assert health["feature_store"] is True
        assert health["model_store"] is True
        assert health["strategy_store"] is True
        assert health["feature_registry"] is True
        assert health["model_registry"] is True
        assert health["strategy_registry"] is True
        assert health["data_registry"] is True
        assert health["data_store"] is True
        assert health["partitions"] is True

    def test_ensure_healthy_passes_when_all_healthy(
        self,
        healthy_monitoring_component: HealthMonitoringComponent,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify no exception when all components healthy.

        Input: All components return healthy status.
        Expected Behavior: Logs success, no exception.
        """
        # Should not raise
        healthy_monitoring_component.ensure_healthy()

        # Check log message
        assert "All ML components are healthy!" in caplog.text

    def test_aggregate_health_groups_by_domain(
        self,
        mock_ml_components: dict[str, MockMLComponent],
    ) -> None:
        """Verify domain-level health aggregation.

        Input: Components implementing MLComponentProtocol.
        Expected Behavior: Domain health reflects component health.
        """
        component = HealthMonitoringComponent(
            feature_store=mock_ml_components["feature_store"],
            model_store=mock_ml_components["model_store"],
            strategy_store=mock_ml_components["strategy_store"],
            data_store=mock_ml_components["data_store"],
            feature_registry=mock_ml_components["feature_registry"],
            model_registry=mock_ml_components["model_registry"],
            strategy_registry=mock_ml_components["strategy_registry"],
            data_registry=mock_ml_components["data_registry"],
        )

        summary = component.aggregate_health()

        # Check structure
        assert "components" in summary
        assert "domains" in summary
        assert "system" in summary

        # Check domains
        domains = summary["domains"]
        assert "data" in domains
        assert "features" in domains
        assert "model" in domains
        assert "strategy" in domains

        # All domains should be healthy
        for domain_info in domains.values():
            assert domain_info["healthy"] is True

        # System should be healthy
        assert summary["system"]["healthy"] is True
        assert summary["system"]["unhealthy"] == []

    def test_validate_protocol_compliance_passes_valid_components(
        self,
        mock_ml_components: dict[str, MockMLComponent],
    ) -> None:
        """Verify protocol validation with compliant components.

        Input: Components implementing MLComponentProtocol.
        Expected Behavior: No violations, no exception.
        """
        component = HealthMonitoringComponent(
            feature_store=mock_ml_components["feature_store"],
            model_store=mock_ml_components["model_store"],
            strategy_store=mock_ml_components["strategy_store"],
            data_store=mock_ml_components["data_store"],
            feature_registry=mock_ml_components["feature_registry"],
            model_registry=mock_ml_components["model_registry"],
            strategy_registry=mock_ml_components["strategy_registry"],
            data_registry=mock_ml_components["data_registry"],
        )

        # Should not raise
        component.validate_protocol_compliance(strict=True)


# =============================================================================
# Error Condition Tests
# =============================================================================


class TestErrorConditions:
    """Tests for error conditions."""

    def test_ensure_healthy_raises_when_component_unhealthy(
        self,
        mock_healthy_stores: dict[str, MockStore],
        mock_healthy_registries: dict[str, MockRegistry],
    ) -> None:
        """Verify exception raised for unhealthy components.

        Input: One or more unhealthy components.
        Expected Behavior: RuntimeError with unhealthy component names.
        """
        # Make feature_store unhealthy
        unhealthy_store = MockStore(healthy=False)

        component = HealthMonitoringComponent(
            feature_store=unhealthy_store,
            model_store=mock_healthy_stores["model_store"],
            strategy_store=mock_healthy_stores["strategy_store"],
            data_store=MockDataStore(has_registry=True),
            feature_registry=mock_healthy_registries["feature_registry"],
            model_registry=mock_healthy_registries["model_registry"],
            strategy_registry=mock_healthy_registries["strategy_registry"],
            data_registry=mock_healthy_registries["data_registry"],
            partition_manager=MockPartitionManager(healthy=True),
            is_postgres_running=lambda: True,
        )

        with pytest.raises(RuntimeError, match="Unhealthy components"):
            component.ensure_healthy()

    def test_validate_protocol_compliance_raises_in_strict_mode(
        self,
    ) -> None:
        """Verify strict mode raises on violations.

        Input: Component not implementing protocol, strict=True.
        Expected Behavior: RuntimeError with violations.
        """
        # Use plain object (not implementing protocol)
        component = HealthMonitoringComponent(
            feature_store=object(),
            model_store=None,
        )

        with pytest.raises(RuntimeError, match="Protocol compliance issues"):
            component.validate_protocol_compliance(strict=True)

    def test_validate_protocol_compliance_logs_warning_in_non_strict_mode(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify non-strict mode logs warning instead of raising.

        Input: Component not implementing protocol, strict=False.
        Expected Behavior: Warning logged, no exception.
        """
        component = HealthMonitoringComponent(
            feature_store=object(),
            model_store=None,
        )

        # Should not raise
        component.validate_protocol_compliance(strict=False)

        assert "Protocol compliance issues" in caplog.text

    def test_check_store_health_returns_false_on_exception(
        self,
    ) -> None:
        """Verify graceful handling of store health check failure.

        Input: Store raises exception on get_statistics.
        Expected Behavior: Returns False.
        """
        unhealthy_store = MockStore(healthy=False)
        component = HealthMonitoringComponent()

        result = component.check_store_health(unhealthy_store)

        assert result is False

    def test_check_store_health_returns_false_for_none(
        self,
    ) -> None:
        """Verify None store returns False.

        Input: None store.
        Expected Behavior: Returns False.
        """
        component = HealthMonitoringComponent()

        result = component.check_store_health(None)

        assert result is False

    def test_check_registry_health_returns_false_on_exception(
        self,
    ) -> None:
        """Verify graceful handling of registry health check failure.

        Input: Registry raises exception on list method.
        Expected Behavior: Returns False.
        """
        unhealthy_registry = MockRegistry(healthy=False)
        component = HealthMonitoringComponent()

        result = component.check_registry_health(unhealthy_registry, "list_features")

        assert result is False


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_aggregate_health_handles_none_components(
        self,
    ) -> None:
        """Verify None component handling.

        Input: Some components are None.
        Expected Behavior: None components marked as unhealthy.
        """
        component = HealthMonitoringComponent(
            feature_store=MockMLComponent(healthy=True),
            model_store=None,  # None component
            strategy_store=MockMLComponent(healthy=True),
            data_store=None,  # None component
            feature_registry=MockMLComponent(healthy=True),
            model_registry=MockMLComponent(healthy=True),
            strategy_registry=MockMLComponent(healthy=True),
            data_registry=MockMLComponent(healthy=True),
        )

        summary = component.aggregate_health()

        # None components should be marked unhealthy
        assert summary["components"]["model_store"]["healthy"] is False
        assert summary["components"]["data_store"]["healthy"] is False

        # Non-None components should be healthy
        assert summary["components"]["feature_store"]["healthy"] is True

        # System should be unhealthy due to None components
        assert summary["system"]["healthy"] is False
        assert "model_store" in summary["system"]["unhealthy"]
        assert "data_store" in summary["system"]["unhealthy"]

    def test_check_registry_health_handles_missing_method(
        self,
    ) -> None:
        """Verify handling of registry without expected method.

        Input: Registry without list_* method.
        Expected Behavior: Returns False.
        """
        # Create mock without list_features
        mock_registry = MagicMock(spec=[])
        del mock_registry.list_features  # Ensure method doesn't exist

        component = HealthMonitoringComponent()

        result = component.check_registry_health(mock_registry, "list_features")

        assert result is False

    def test_check_registry_health_handles_none_registry(
        self,
    ) -> None:
        """Verify None registry returns False.

        Input: None registry.
        Expected Behavior: Returns False.
        """
        component = HealthMonitoringComponent()

        result = component.check_registry_health(None, "list_features")

        assert result is False

    def test_check_data_store_health_returns_false_without_registry(
        self,
    ) -> None:
        """Verify DataStore without registry is unhealthy.

        Input: DataStore without registry attribute.
        Expected Behavior: Returns False.
        """
        data_store = MockDataStore(has_registry=False)
        component = HealthMonitoringComponent(data_store=data_store)

        result = component.check_data_store_health()

        assert result is False

    def test_check_data_store_health_returns_false_when_none(
        self,
    ) -> None:
        """Verify None DataStore returns False.

        Input: None DataStore.
        Expected Behavior: Returns False.
        """
        component = HealthMonitoringComponent(data_store=None)

        result = component.check_data_store_health()

        assert result is False

    def test_check_partition_health_returns_false_when_none(
        self,
    ) -> None:
        """Verify None partition manager returns False.

        Input: None partition manager.
        Expected Behavior: Returns False.
        """
        component = HealthMonitoringComponent(partition_manager=None)

        result = component.check_partition_health()

        assert result is False

    def test_check_partition_health_returns_false_when_empty_stats(
        self,
    ) -> None:
        """Verify partition manager with empty stats returns False.

        Input: Partition manager returning empty stats.
        Expected Behavior: Returns False.
        """
        partition_manager = MockPartitionManager(healthy=True, stats=[])
        component = HealthMonitoringComponent(partition_manager=partition_manager)

        result = component.check_partition_health()

        assert result is False

    def test_check_partition_health_returns_false_on_exception(
        self,
    ) -> None:
        """Verify graceful handling of partition health check failure.

        Input: Partition manager raises exception.
        Expected Behavior: Returns False.
        """
        partition_manager = MockPartitionManager(healthy=False)
        component = HealthMonitoringComponent(partition_manager=partition_manager)

        result = component.check_partition_health()

        assert result is False

    def test_check_health_postgres_check_uses_callback(
        self,
    ) -> None:
        """Verify PostgreSQL check uses injected callback.

        Input: Custom is_postgres_running callback.
        Expected Behavior: Callback is invoked.
        """
        callback_invoked = []

        def custom_callback() -> bool:
            callback_invoked.append(True)
            return True

        component = HealthMonitoringComponent(is_postgres_running=custom_callback)

        health = component.check_health()

        assert len(callback_invoked) == 1
        assert health["postgres"] is True

    def test_check_health_postgres_check_returns_false_by_default(
        self,
    ) -> None:
        """Verify default PostgreSQL check returns False.

        Input: Default is_postgres_running.
        Expected Behavior: Returns False.
        """
        component = HealthMonitoringComponent()

        health = component.check_health()

        assert health["postgres"] is False

    def test_validate_protocol_compliance_env_var_strict_true(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify strict mode from environment variable.

        Input: ML_STRICT_PROTOCOL_VALIDATION=1.
        Expected Behavior: Raises on violations.
        """
        monkeypatch.setenv("ML_STRICT_PROTOCOL_VALIDATION", "1")

        component = HealthMonitoringComponent(
            feature_store=object(),  # Not implementing protocol
        )

        with pytest.raises(RuntimeError, match="Protocol compliance issues"):
            component.validate_protocol_compliance(strict=None)

    def test_validate_protocol_compliance_env_var_strict_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify non-strict mode from environment variable.

        Input: ML_STRICT_PROTOCOL_VALIDATION not set.
        Expected Behavior: Logs warning, no exception.
        """
        monkeypatch.delenv("ML_STRICT_PROTOCOL_VALIDATION", raising=False)

        component = HealthMonitoringComponent(
            feature_store=object(),  # Not implementing protocol
        )

        # Should not raise
        component.validate_protocol_compliance(strict=None)

        assert "Protocol compliance issues" in caplog.text

    def test_aggregate_health_component_health_status_error(
        self,
    ) -> None:
        """Verify handling of get_health_status exception.

        Input: Component that raises on get_health_status.
        Expected Behavior: Component marked unhealthy.
        """
        unhealthy_component = MockMLComponent(healthy=False)
        component = HealthMonitoringComponent(
            feature_store=unhealthy_component,
        )

        summary = component.aggregate_health()

        assert summary["components"]["feature_store"]["healthy"] is False

    def test_check_store_health_uses_is_healthy_fallback(
        self,
    ) -> None:
        """Verify is_healthy() is used when get_statistics unavailable.

        Input: Store with is_healthy but no get_statistics.
        Expected Behavior: Uses is_healthy method.
        """
        mock_store = MagicMock()
        del mock_store.get_statistics  # Remove get_statistics
        mock_store.is_healthy.return_value = True

        component = HealthMonitoringComponent()

        result = component.check_store_health(mock_store)

        assert result is True
        mock_store.is_healthy.assert_called_once()

    def test_check_registry_health_list_datasets_only_checks_callable(
        self,
    ) -> None:
        """Verify list_datasets only checks method existence.

        Input: Registry with list_datasets method.
        Expected Behavior: Returns True without calling method.
        """
        mock_registry = MagicMock()
        mock_registry.list_datasets = MagicMock()

        component = HealthMonitoringComponent()

        result = component.check_registry_health(mock_registry, "list_datasets")

        assert result is True
        # list_datasets should not be called (just checked for existence)
        mock_registry.list_datasets.assert_not_called()

    def test_validate_protocol_compliance_config_issues_reported(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify configuration issues are reported in violations.

        Input: Component with configuration issues.
        Expected Behavior: Config issues in violation message.
        """
        component_with_issues = MockMLComponent(
            healthy=True,
            config_issues=["missing_required_field"],
        )
        component = HealthMonitoringComponent(
            feature_store=component_with_issues,
        )

        component.validate_protocol_compliance(strict=False)

        assert "config:missing_required_field" in caplog.text


# =============================================================================
# Domain Health Invariant Tests
# =============================================================================


class TestDomainHealthInvariants:
    """Tests for health aggregation invariants."""

    def test_system_healthy_implies_all_domains_healthy(
        self,
        mock_ml_components: dict[str, MockMLComponent],
    ) -> None:
        """Verify system health invariant.

        When system.healthy is True, all domains must also be healthy.
        """
        component = HealthMonitoringComponent(
            feature_store=mock_ml_components["feature_store"],
            model_store=mock_ml_components["model_store"],
            strategy_store=mock_ml_components["strategy_store"],
            data_store=mock_ml_components["data_store"],
            feature_registry=mock_ml_components["feature_registry"],
            model_registry=mock_ml_components["model_registry"],
            strategy_registry=mock_ml_components["strategy_registry"],
            data_registry=mock_ml_components["data_registry"],
        )

        summary = component.aggregate_health()

        if summary["system"]["healthy"]:
            for domain_info in summary["domains"].values():
                assert domain_info["healthy"] is True

    def test_unhealthy_list_matches_component_status(
        self,
    ) -> None:
        """Verify unhealthy list completeness.

        The unhealthy list should contain exactly the unhealthy components.
        """
        component = HealthMonitoringComponent(
            feature_store=MockMLComponent(healthy=True),
            model_store=None,  # Unhealthy (None)
            strategy_store=MockMLComponent(healthy=True),
            data_store=None,  # Unhealthy (None)
            feature_registry=MockMLComponent(healthy=True),
            model_registry=MockMLComponent(healthy=True),
            strategy_registry=MockMLComponent(healthy=True),
            data_registry=MockMLComponent(healthy=True),
        )

        summary = component.aggregate_health()

        unhealthy_set = set(summary["system"]["unhealthy"])
        for name, info in summary["components"].items():
            if not info["healthy"]:
                assert name in unhealthy_set
            else:
                assert name not in unhealthy_set

    def test_domain_unhealthy_when_any_component_unhealthy(
        self,
    ) -> None:
        """Verify domain becomes unhealthy when any component is unhealthy.

        If any component in a domain is unhealthy, the domain should be unhealthy.
        """
        component = HealthMonitoringComponent(
            feature_store=MockMLComponent(healthy=True),
            feature_registry=None,  # Unhealthy - features domain affected
            model_store=MockMLComponent(healthy=True),
            model_registry=MockMLComponent(healthy=True),
            strategy_store=MockMLComponent(healthy=True),
            strategy_registry=MockMLComponent(healthy=True),
            data_store=MockMLComponent(healthy=True),
            data_registry=MockMLComponent(healthy=True),
        )

        summary = component.aggregate_health()

        # Features domain should be unhealthy
        assert summary["domains"]["features"]["healthy"] is False

        # Other domains should still be healthy
        assert summary["domains"]["model"]["healthy"] is True
        assert summary["domains"]["strategy"]["healthy"] is True
        assert summary["domains"]["data"]["healthy"] is True
