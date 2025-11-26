"""
Integration tests for FeatureRegistryAccessor component.

Tests complete workflows with all registries or partial registry availability.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from ml.features.common.feature_registry_accessor import FeatureRegistryAccessor


def test_registry_accessor_all_registries_available(
    mock_stores_with_registries: Mock,
) -> None:
    """
    Verify all 4 registry properties return correct instances when stores container
    is fully populated.

    Given a mock stores container with ALL 4 registries
    When FeatureRegistryAccessor is injected with this container
    Then all 4 properties return the correct registry instances
    And no property returns None
    """
    accessor = FeatureRegistryAccessor(stores=mock_stores_with_registries)

    # All 4 properties return non-None
    assert accessor.feature_registry is not None
    assert accessor.model_registry is not None
    assert accessor.strategy_registry is not None
    assert accessor.data_registry is not None

    # Each property returns the exact registry that was injected
    assert accessor.feature_registry is mock_stores_with_registries.feature_registry
    assert accessor.model_registry is mock_stores_with_registries.model_registry
    assert accessor.strategy_registry is mock_stores_with_registries.strategy_registry
    assert accessor.data_registry is mock_stores_with_registries.data_registry

    # Type safety: each property returns a Mock instance
    assert isinstance(accessor.feature_registry, Mock)
    assert isinstance(accessor.model_registry, Mock)
    assert isinstance(accessor.strategy_registry, Mock)
    assert isinstance(accessor.data_registry, Mock)


def test_registry_accessor_partial_stores(mock_stores_partial: Mock) -> None:
    """
    Verify accessor handles partial stores container (only some registries available)
    gracefully.

    Given a stores container with only 2 registries (feature + model)
    When FeatureRegistryAccessor is injected with this partial container
    Then available registries return instances
    And unavailable registries return None
    And no exceptions are raised when accessing missing registries
    """
    accessor = FeatureRegistryAccessor(stores=mock_stores_partial)

    # Available registries return non-None
    assert accessor.feature_registry is not None
    assert accessor.model_registry is not None

    # Available registries return the exact instances injected
    assert accessor.feature_registry is mock_stores_partial.feature_registry
    assert accessor.model_registry is mock_stores_partial.model_registry

    # Unavailable registries return None (graceful degradation)
    assert accessor.strategy_registry is None
    assert accessor.data_registry is None

    # No exceptions raised when accessing missing registries
    # (this test passes if we get here without AttributeError)
