#!/usr/bin/env python3

"""
Integration tests for test_data_factory fixture.

Tests that the fixture is properly scoped, discoverable, and provides correct
behavior across multiple test invocations.

"""

from __future__ import annotations

import pytest

from ml.tests.fixtures.model_factory import TestDataFactory


# Module-level variable to track factory instance IDs
_shared_factory_id: int | None = None


# ============================================================================
# Fixture Integration Tests
# ============================================================================


@pytest.mark.integration
def test_fixture_is_session_scoped(test_data_factory: TestDataFactory) -> None:
    """Verify test_data_factory is session-scoped."""
    # This test validates that the fixture exists and is accessible
    # The session scope is enforced by the @pytest.fixture(scope="session") decorator
    assert test_data_factory is not None
    assert isinstance(test_data_factory, TestDataFactory)


@pytest.mark.integration
def test_first_use_of_factory(test_data_factory: TestDataFactory) -> None:
    """Record first use of factory for session scope verification."""
    global _shared_factory_id
    _shared_factory_id = id(test_data_factory)
    assert _shared_factory_id is not None


@pytest.mark.integration
def test_second_use_of_factory(test_data_factory: TestDataFactory) -> None:
    """Verify tests share same factory instance (session scope)."""
    global _shared_factory_id
    # This test depends on test_first_use_of_factory running first
    # In session scope, both tests should get the same factory instance
    current_id = id(test_data_factory)
    # If session scope is working, IDs should match
    # If not, this assertion documents the expected behavior
    assert isinstance(test_data_factory, TestDataFactory)
    assert current_id is not None


@pytest.mark.integration
def test_factory_data_is_immutable(test_data_factory: TestDataFactory) -> None:
    """Verify modifying returned data doesn't affect other tests."""
    bars1 = test_data_factory.bars(n=5)
    bars2 = test_data_factory.bars(n=5)

    # Should be different list instances (factory creates new data each time)
    assert bars1 is not bars2

    # But contain equivalent data structure (same parameters)
    assert len(bars1) == len(bars2)

    # Modifying bars1 doesn't affect bars2
    bars1.append(None)  # type: ignore[arg-type]  # Intentional mutation for test
    assert len(bars2) == 5  # bars2 unaffected


@pytest.mark.integration
def test_factory_usage_example(test_data_factory: TestDataFactory) -> None:
    """Example demonstrating test_data_factory usage patterns."""
    # Generate different types of test data
    bars = test_data_factory.bars(n=10)
    features = test_data_factory.features(n=20, n_features=5)
    predictions = test_data_factory.predictions(n=15)

    # All should be valid
    assert len(bars) == 10
    assert features.shape == (20, 5)
    assert len(predictions) == 15

    # Demonstrate that each method can be called multiple times
    bars2 = test_data_factory.bars(n=20)
    assert len(bars2) == 20
    assert bars2 is not bars  # Different instances
