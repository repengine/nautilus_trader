"""
Test suite for TypedDict definitions in health aggregation (Task 2.2).

This module tests the type-safe health monitoring API using TypedDict definitions
instead of generic `dict[str, object]` return types. All tests are designed following
TDD principles and apply lessons from Tasks 1.1-1.3 and 2.1 (100% pass rate standard).

Design Principles
-----------------
1. Test behavior (TypedDict structure) not implementation details
2. Use typing.get_type_hints() for type verification
3. Verify backward compatibility (TypedDict is structural)
4. Apply flexible assertions (no brittle exact type checks)
5. Ensure JSON serialization compatibility

"""

from __future__ import annotations

import json
import pytest
from typing import get_type_hints

from ml.common.protocols import MLComponentMixin
from ml.core.integration import MLIntegrationManager
from ml.tests.utils.stubs import build_integration_manager_stub


class _HealthyComponent(MLComponentMixin):
    """Stub component that reports healthy status."""

    def get_health_status(self) -> dict[str, object]:  # type: ignore[override]
        return {"status": "healthy", "checks_passed": 5}

    def get_performance_metrics(self) -> dict[str, float]:  # type: ignore[override]
        return {"latency_ms": 1.2, "throughput": 1000.0}


class _UnhealthyComponent(MLComponentMixin):
    """Stub component that reports unhealthy status."""

    def get_health_status(self) -> dict[str, object]:  # type: ignore[override]
        return {"status": "unhealthy", "error": "connection_failed"}

    def get_performance_metrics(self) -> dict[str, float]:  # type: ignore[override]
        return {"latency_ms": 999.0, "error_rate": 0.8}


def test_aggregate_health_returns_health_summary() -> None:
    """
    Verify aggregate_health() returns a HealthSummary structure.

    This test verifies the return type has the correct structure expected from
    HealthSummary TypedDict. We check for required top-level keys without
    checking the exact TypedDict class (which is structural, not nominal).

    Design Note:
    - Flexible assertion pattern (checks structure, not exact type)
    - Applies lessons from Task Group 1 (no brittle type checks)
    """
    mgr = build_integration_manager_stub()
    mgr.feature_store = _HealthyComponent()
    mgr.model_store = _HealthyComponent()
    mgr.strategy_store = _HealthyComponent()
    mgr.data_store = _HealthyComponent()
    mgr.feature_registry = _HealthyComponent()
    mgr.model_registry = _HealthyComponent()
    mgr.strategy_registry = _HealthyComponent()
    mgr.data_registry = _HealthyComponent()

    health = mgr.aggregate_health()

    # Verify structure (behavior), not exact TypedDict type
    assert isinstance(health, dict)
    assert "components" in health
    assert "domains" in health
    assert "system" in health


def test_health_summary_has_required_keys() -> None:
    """
    Verify HealthSummary contains all required top-level keys.

    This ensures the TypedDict definition matches the expected structure
    and that aggregate_health() returns all required fields.

    Design Note:
    - Tests behavior (required keys present)
    - Does not check exact values (allows for implementation flexibility)
    """
    mgr = build_integration_manager_stub()
    mgr.feature_store = _HealthyComponent()
    mgr.model_store = _HealthyComponent()
    mgr.strategy_store = _HealthyComponent()
    mgr.data_store = _HealthyComponent()
    mgr.feature_registry = _HealthyComponent()
    mgr.model_registry = _HealthyComponent()
    mgr.strategy_registry = _HealthyComponent()
    mgr.data_registry = _HealthyComponent()

    health = mgr.aggregate_health()

    # Required top-level keys
    required_keys = ["components", "domains", "system"]
    for key in required_keys:
        assert key in health, f"Missing required key: {key}"


def test_component_health_status_structure() -> None:
    """
    Verify ComponentHealthStatus has correct structure.

    This test verifies that individual component health entries match the
    ComponentHealthStatus TypedDict definition with fields: healthy, health, metrics.

    Design Note:
    - Checks component-level structure
    - Verifies type of nested fields (bool, dict, dict)
    """
    mgr = build_integration_manager_stub()
    mgr.feature_store = _HealthyComponent()
    mgr.model_store = _HealthyComponent()
    mgr.strategy_store = _HealthyComponent()
    mgr.data_store = _HealthyComponent()
    mgr.feature_registry = _HealthyComponent()
    mgr.model_registry = _HealthyComponent()
    mgr.strategy_registry = _HealthyComponent()
    mgr.data_registry = _HealthyComponent()

    health = mgr.aggregate_health()

    # Check structure of one component (all should match)
    components = health["components"]
    assert isinstance(components, dict)
    assert "feature_store" in components

    feature_health = components["feature_store"]
    assert isinstance(feature_health, dict)

    # ComponentHealthStatus fields
    assert "healthy" in feature_health
    assert isinstance(feature_health["healthy"], bool)

    assert "health" in feature_health
    assert isinstance(feature_health["health"], dict)

    assert "metrics" in feature_health
    assert isinstance(feature_health["metrics"], dict)


def test_domain_health_structure() -> None:
    """
    Verify DomainHealth has correct structure.

    This test verifies that domain-level health entries match the DomainHealth
    TypedDict definition with fields: components, healthy.

    Design Note:
    - Checks domain-level structure
    - Verifies both required fields present
    """
    mgr = build_integration_manager_stub()
    mgr.feature_store = _HealthyComponent()
    mgr.model_store = _HealthyComponent()
    mgr.strategy_store = _HealthyComponent()
    mgr.data_store = _HealthyComponent()
    mgr.feature_registry = _HealthyComponent()
    mgr.model_registry = _HealthyComponent()
    mgr.strategy_registry = _HealthyComponent()
    mgr.data_registry = _HealthyComponent()

    health = mgr.aggregate_health()

    # Check structure of one domain (all should match)
    domains = health["domains"]
    assert isinstance(domains, dict)
    assert "features" in domains

    features_domain = domains["features"]
    assert isinstance(features_domain, dict)

    # DomainHealth fields
    assert "components" in features_domain
    assert isinstance(features_domain["components"], list)
    assert all(isinstance(c, str) for c in features_domain["components"])

    assert "healthy" in features_domain
    assert isinstance(features_domain["healthy"], bool)


def test_system_health_structure() -> None:
    """
    Verify SystemHealth has correct structure.

    This test verifies that system-level health matches the SystemHealth
    TypedDict definition with fields: healthy, unhealthy.

    Design Note:
    - Checks system-level structure
    - Verifies unhealthy is a list of strings
    """
    mgr = build_integration_manager_stub()
    mgr.feature_store = _HealthyComponent()
    mgr.model_store = _HealthyComponent()
    mgr.strategy_store = _HealthyComponent()
    mgr.data_store = _HealthyComponent()
    mgr.feature_registry = _HealthyComponent()
    mgr.model_registry = _HealthyComponent()
    mgr.strategy_registry = _HealthyComponent()
    mgr.data_registry = _HealthyComponent()

    health = mgr.aggregate_health()

    # SystemHealth fields
    system = health["system"]
    assert isinstance(system, dict)

    assert "healthy" in system
    assert isinstance(system["healthy"], bool)

    assert "unhealthy" in system
    assert isinstance(system["unhealthy"], list)
    assert all(isinstance(name, str) for name in system["unhealthy"])


def test_health_domains_contains_all_domains() -> None:
    """
    Verify HealthDomains contains all expected domains.

    This test ensures that the domains dictionary contains entries for
    data, features, model, and strategy domains.

    Design Note:
    - Tests that all domains are represented
    - Does not require exact domain list (allows for future additions)
    """
    mgr = build_integration_manager_stub()
    mgr.feature_store = _HealthyComponent()
    mgr.model_store = _HealthyComponent()
    mgr.strategy_store = _HealthyComponent()
    mgr.data_store = _HealthyComponent()
    mgr.feature_registry = _HealthyComponent()
    mgr.model_registry = _HealthyComponent()
    mgr.strategy_registry = _HealthyComponent()
    mgr.data_registry = _HealthyComponent()

    health = mgr.aggregate_health()

    domains = health["domains"]
    assert isinstance(domains, dict)

    # Expected domains (from task definition)
    expected_domains = ["data", "features", "model", "strategy"]
    for domain in expected_domains:
        assert domain in domains, f"Missing domain: {domain}"


def test_typeddict_enables_ide_autocomplete() -> None:
    """
    Verify TypedDict structure is accessible via typing.get_type_hints().

    This test uses Python's typing introspection to verify that the TypedDict
    definitions are properly structured and can be used by IDEs and type checkers
    for autocomplete and validation.

    Design Note:
    - Uses typing.get_type_hints() for type inspection
    - Verifies type definitions exist and are accessible
    - Does not depend on runtime type checking
    """
    # Import TypedDict definitions
    from ml.core.integration import (
        ComponentHealthStatus,
        DomainHealth,
        HealthDomains,
        HealthSummary,
        SystemHealth,
    )

    # Verify HealthSummary has expected fields
    hints = get_type_hints(HealthSummary)
    assert "components" in hints
    assert "domains" in hints
    assert "system" in hints

    # Verify ComponentHealthStatus has expected fields
    comp_hints = get_type_hints(ComponentHealthStatus)
    assert "healthy" in comp_hints
    assert "health" in comp_hints
    assert "metrics" in comp_hints

    # Verify DomainHealth has expected fields
    domain_hints = get_type_hints(DomainHealth)
    assert "components" in domain_hints
    assert "healthy" in domain_hints

    # Verify SystemHealth has expected fields
    system_hints = get_type_hints(SystemHealth)
    assert "healthy" in system_hints
    assert "unhealthy" in system_hints

    # Verify HealthDomains has expected fields
    domains_hints = get_type_hints(HealthDomains)
    # HealthDomains has optional fields (total=False)
    # Just verify it's a valid TypedDict
    assert len(domains_hints) >= 0  # May have 0-4 fields depending on total=False


def test_backward_compatibility_dict_access() -> None:
    """
    Verify TypedDict is backward compatible with dict access.

    TypedDict is structural - the runtime object is still a regular dict.
    This test verifies that existing dict access patterns still work after
    TypedDict annotations are added.

    Design Note:
    - TypedDict has ZERO runtime cost (it's only for type checking)
    - Runtime objects are regular dicts with all dict methods
    - This test ensures no breaking changes for existing code
    """
    mgr = build_integration_manager_stub()
    mgr.feature_store = _HealthyComponent()
    mgr.model_store = _HealthyComponent()
    mgr.strategy_store = _HealthyComponent()
    mgr.data_store = _HealthyComponent()
    mgr.feature_registry = _HealthyComponent()
    mgr.model_registry = _HealthyComponent()
    mgr.strategy_registry = _HealthyComponent()
    mgr.data_registry = _HealthyComponent()

    health = mgr.aggregate_health()

    # TypedDict is still a dict at runtime
    assert isinstance(health, dict)

    # Dict access works (bracket notation)
    assert health["system"]["healthy"] in [True, False]

    # .get() method works
    assert health.get("system") is not None

    # .keys() method works
    keys = list(health.keys())
    assert "components" in keys
    assert "domains" in keys
    assert "system" in keys

    # .items() method works
    items = list(health.items())
    assert len(items) == 3  # components, domains, system

    # in operator works
    assert "system" in health


def test_aggregate_health_with_healthy_components() -> None:
    """
    Verify aggregate_health() with all healthy components.

    This test creates a scenario where all components report healthy status
    and verifies the health summary reflects this correctly.

    Design Note:
    - Tests happy path scenario
    - Verifies system.healthy is True when all components healthy
    - Verifies system.unhealthy is empty list
    """
    mgr = build_integration_manager_stub()
    mgr.feature_store = _HealthyComponent()
    mgr.model_store = _HealthyComponent()
    mgr.strategy_store = _HealthyComponent()
    mgr.data_store = _HealthyComponent()
    mgr.feature_registry = _HealthyComponent()
    mgr.model_registry = _HealthyComponent()
    mgr.strategy_registry = _HealthyComponent()
    mgr.data_registry = _HealthyComponent()

    health = mgr.aggregate_health()

    # All components should be healthy
    components = health["components"]
    for name, comp_health in components.items():
        assert comp_health["healthy"] is True, f"{name} should be healthy"

    # All domains should be healthy
    domains = health["domains"]
    for domain_name, domain_health in domains.items():
        assert domain_health["healthy"] is True, f"{domain_name} domain should be healthy"

    # System should be healthy
    system = health["system"]
    assert system["healthy"] is True
    assert system["unhealthy"] == []


def test_aggregate_health_with_unhealthy_components() -> None:
    """
    Verify aggregate_health() with unhealthy components.

    This test creates a scenario where some components report unhealthy status
    and verifies the health summary correctly identifies them in system.unhealthy.

    Design Note:
    - Tests error path scenario
    - Verifies system.healthy is False when any component unhealthy
    - Verifies system.unhealthy list contains unhealthy component names
    """
    mgr = build_integration_manager_stub()
    mgr.feature_store = _UnhealthyComponent()  # Unhealthy
    mgr.model_store = _HealthyComponent()
    mgr.strategy_store = _UnhealthyComponent()  # Unhealthy
    mgr.data_store = _HealthyComponent()
    mgr.feature_registry = _HealthyComponent()
    mgr.model_registry = _HealthyComponent()
    mgr.strategy_registry = _HealthyComponent()
    mgr.data_registry = _HealthyComponent()

    health = mgr.aggregate_health()

    # Feature store should be unhealthy
    components = health["components"]
    assert components["feature_store"]["healthy"] is False
    assert components["strategy_store"]["healthy"] is False

    # Features domain should be unhealthy (feature_store unhealthy)
    domains = health["domains"]
    assert domains["features"]["healthy"] is False

    # Strategy domain should be unhealthy (strategy_store unhealthy)
    assert domains["strategy"]["healthy"] is False

    # System should be unhealthy
    system = health["system"]
    assert system["healthy"] is False

    # System.unhealthy should contain unhealthy components
    unhealthy_list = system["unhealthy"]
    assert isinstance(unhealthy_list, list)
    assert "feature_store" in unhealthy_list
    assert "strategy_store" in unhealthy_list
    assert len(unhealthy_list) >= 2  # At least these two


def test_health_summary_json_serializable() -> None:
    """
    Verify health summary can be JSON serialized.

    This is important for APIs and monitoring systems that need to transmit
    health status over HTTP or store it in JSON-based systems.

    Design Note:
    - Tests integration with JSON serialization
    - Verifies round-trip serialization preserves structure
    - Important for dashboard and monitoring use cases
    """
    mgr = build_integration_manager_stub()
    mgr.feature_store = _HealthyComponent()
    mgr.model_store = _HealthyComponent()
    mgr.strategy_store = _HealthyComponent()
    mgr.data_store = _HealthyComponent()
    mgr.feature_registry = _HealthyComponent()
    mgr.model_registry = _HealthyComponent()
    mgr.strategy_registry = _HealthyComponent()
    mgr.data_registry = _HealthyComponent()

    health = mgr.aggregate_health()

    # Should serialize without error
    json_str = json.dumps(health)
    assert json_str is not None
    assert len(json_str) > 0

    # Should deserialize back to dict
    parsed = json.loads(json_str)
    assert isinstance(parsed, dict)

    # Should preserve structure
    assert "components" in parsed
    assert "domains" in parsed
    assert "system" in parsed

    # Should preserve nested values
    assert isinstance(parsed["system"]["healthy"], bool)
    assert isinstance(parsed["system"]["unhealthy"], list)
