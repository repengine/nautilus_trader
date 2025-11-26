"""
Unit tests for FeatureRegistryAccessor component.

Tests registry access properties with defensive programming validation.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from ml.features.common.feature_registry_accessor import FeatureRegistryAccessor


# ==================== Happy Path Tests ====================


def test_feature_registry_with_stores_injected(mock_stores_with_registries: Mock) -> None:
    """
    Verify feature_registry property returns FeatureRegistry when stores container has
    feature_registry attribute.

    Given a mock stores container with feature_registry attribute
    When FeatureRegistryAccessor is initialized with this stores container
    Then property returns the registry instance (not None)
    And property returns the exact registry that was injected
    """
    accessor = FeatureRegistryAccessor(stores=mock_stores_with_registries)

    # Property returns non-None
    assert accessor.feature_registry is not None

    # Property returns the exact registry that was injected
    assert accessor.feature_registry is mock_stores_with_registries.feature_registry

    # Property returns a Mock instance (type check)
    assert isinstance(accessor.feature_registry, Mock)


def test_model_registry_with_stores_injected(mock_stores_with_registries: Mock) -> None:
    """
    Verify model_registry property returns ModelRegistry when stores container has
    model_registry attribute.

    Given a mock stores container with model_registry attribute
    When FeatureRegistryAccessor is initialized with this stores container
    Then property returns the registry instance (not None)
    And property returns the exact registry that was injected
    """
    accessor = FeatureRegistryAccessor(stores=mock_stores_with_registries)

    # Property returns non-None
    assert accessor.model_registry is not None

    # Property returns the exact registry that was injected
    assert accessor.model_registry is mock_stores_with_registries.model_registry

    # Property returns a Mock instance (type check)
    assert isinstance(accessor.model_registry, Mock)


def test_strategy_registry_with_stores_injected(mock_stores_with_registries: Mock) -> None:
    """
    Verify strategy_registry property returns StrategyRegistry when stores container has
    strategy_registry attribute.

    Given a mock stores container with strategy_registry attribute
    When FeatureRegistryAccessor is initialized with this stores container
    Then property returns the registry instance (not None)
    And property returns the exact registry that was injected
    """
    accessor = FeatureRegistryAccessor(stores=mock_stores_with_registries)

    # Property returns non-None
    assert accessor.strategy_registry is not None

    # Property returns the exact registry that was injected
    assert accessor.strategy_registry is mock_stores_with_registries.strategy_registry

    # Property returns a Mock instance (type check)
    assert isinstance(accessor.strategy_registry, Mock)


def test_data_registry_with_stores_injected(mock_stores_with_registries: Mock) -> None:
    """
    Verify data_registry property returns DataRegistry when stores container has
    data_registry attribute.

    Given a mock stores container with data_registry attribute
    When FeatureRegistryAccessor is initialized with this stores container
    Then property returns the registry instance (not None)
    And property returns the exact registry that was injected
    """
    accessor = FeatureRegistryAccessor(stores=mock_stores_with_registries)

    # Property returns non-None
    assert accessor.data_registry is not None

    # Property returns the exact registry that was injected
    assert accessor.data_registry is mock_stores_with_registries.data_registry

    # Property returns a Mock instance (type check)
    assert isinstance(accessor.data_registry, Mock)


# ==================== Error Conditions - stores=None ====================


def test_feature_registry_without_stores_returns_none() -> None:
    """
    Verify feature_registry property returns None when stores=None (graceful degradation).

    Given FeatureRegistryAccessor initialized with stores=None
    When accessing feature_registry property
    Then property returns None without raising exception
    """
    accessor = FeatureRegistryAccessor(stores=None)

    # Property returns None
    assert accessor.feature_registry is None

    # No exception raised (this test passes if we get here)


def test_model_registry_without_stores_returns_none() -> None:
    """
    Verify model_registry property returns None when stores=None (graceful degradation).

    Given FeatureRegistryAccessor initialized with stores=None
    When accessing model_registry property
    Then property returns None without raising exception
    """
    accessor = FeatureRegistryAccessor(stores=None)

    # Property returns None
    assert accessor.model_registry is None

    # No exception raised


def test_strategy_registry_without_stores_returns_none() -> None:
    """
    Verify strategy_registry property returns None when stores=None (graceful degradation).

    Given FeatureRegistryAccessor initialized with stores=None
    When accessing strategy_registry property
    Then property returns None without raising exception
    """
    accessor = FeatureRegistryAccessor(stores=None)

    # Property returns None
    assert accessor.strategy_registry is None

    # No exception raised


def test_data_registry_without_stores_returns_none() -> None:
    """
    Verify data_registry property returns None when stores=None (graceful degradation).

    Given FeatureRegistryAccessor initialized with stores=None
    When accessing data_registry property
    Then property returns None without raising exception
    """
    accessor = FeatureRegistryAccessor(stores=None)

    # Property returns None
    assert accessor.data_registry is None

    # No exception raised


# ==================== Edge Cases - Missing Attributes ====================


def test_feature_registry_missing_attr_returns_none(
    mock_stores_missing_feature_registry: Mock,
) -> None:
    """
    Verify feature_registry property returns None when stores exists but lacks
    feature_registry attribute.

    Given a mock stores container WITHOUT feature_registry attribute
    When FeatureRegistryAccessor is initialized with this partial stores container
    Then property returns None without raising AttributeError
    """
    accessor = FeatureRegistryAccessor(stores=mock_stores_missing_feature_registry)

    # Confirm test setup: stores doesn't have feature_registry
    assert not hasattr(mock_stores_missing_feature_registry, "feature_registry")

    # Property returns None
    assert accessor.feature_registry is None

    # No exception raised


def test_model_registry_missing_attr_returns_none(
    mock_stores_missing_model_registry: Mock,
) -> None:
    """
    Verify model_registry property returns None when stores exists but lacks
    model_registry attribute.

    Given a mock stores container WITHOUT model_registry attribute
    When FeatureRegistryAccessor is initialized with this partial stores container
    Then property returns None without raising AttributeError
    """
    accessor = FeatureRegistryAccessor(stores=mock_stores_missing_model_registry)

    # Confirm test setup: stores doesn't have model_registry
    assert not hasattr(mock_stores_missing_model_registry, "model_registry")

    # Property returns None
    assert accessor.model_registry is None

    # No exception raised


def test_strategy_registry_missing_attr_returns_none(
    mock_stores_missing_strategy_registry: Mock,
) -> None:
    """
    Verify strategy_registry property returns None when stores exists but lacks
    strategy_registry attribute.

    Given a mock stores container WITHOUT strategy_registry attribute
    When FeatureRegistryAccessor is initialized with this partial stores container
    Then property returns None without raising AttributeError
    """
    accessor = FeatureRegistryAccessor(stores=mock_stores_missing_strategy_registry)

    # Confirm test setup: stores doesn't have strategy_registry
    assert not hasattr(mock_stores_missing_strategy_registry, "strategy_registry")

    # Property returns None
    assert accessor.strategy_registry is None

    # No exception raised


def test_data_registry_missing_attr_returns_none(
    mock_stores_missing_data_registry: Mock,
) -> None:
    """
    Verify data_registry property returns None when stores exists but lacks
    data_registry attribute.

    Given a mock stores container WITHOUT data_registry attribute
    When FeatureRegistryAccessor is initialized with this partial stores container
    Then property returns None without raising AttributeError
    """
    accessor = FeatureRegistryAccessor(stores=mock_stores_missing_data_registry)

    # Confirm test setup: stores doesn't have data_registry
    assert not hasattr(mock_stores_missing_data_registry, "data_registry")

    # Property returns None
    assert accessor.data_registry is None

    # No exception raised
