"""Integration tests for RuntimeAttacher runtime attachment workflows (Phase 2.2.5).

All tests marked @pytest.mark.skip for structural phase.
Full implementation in Phase 2.2.8 (facade integration).
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def integration_manager() -> Mock:
    """Provides mock MLIntegrationManager for testing."""
    manager = Mock()
    manager.attach_stores.return_value = None
    manager.attach_registries.return_value = None
    return manager


@pytest.fixture
def data_store() -> Mock:
    """Provides mock DataStore for testing."""
    return Mock()


@pytest.fixture
def feature_store() -> Mock:
    """Provides mock FeatureStore for testing."""
    return Mock()


@pytest.fixture
def model_store() -> Mock:
    """Provides mock ModelStore for testing."""
    return Mock()


@pytest.fixture
def strategy_store() -> Mock:
    """Provides mock StrategyStore for testing."""
    return Mock()


@pytest.fixture
def data_registry() -> Mock:
    """Provides mock DataRegistry for testing."""
    return Mock()


@pytest.fixture
def feature_registry() -> Mock:
    """Provides mock FeatureRegistry for testing."""
    return Mock()


@pytest.fixture
def model_registry() -> Mock:
    """Provides mock ModelRegistry for testing."""
    return Mock()


@pytest.fixture
def strategy_registry() -> Mock:
    """Provides mock StrategyRegistry for testing."""
    return Mock()


@pytest.fixture
def runtime_attacher(integration_manager: Mock):
    """Provides RuntimeAttacher instance for testing."""
    from ml.orchestration.components.runtime_attacher import RuntimeAttacher

    return RuntimeAttacher(
        integration_manager=integration_manager,
        validators=None,
    )


# ============================================================================
# INTEGRATION TESTS: RUNTIME ATTACHMENT (4 tests)
# ============================================================================


@pytest.mark.skip(reason="Structural phase - requires full implementation in Phase 2.2.8")
@pytest.mark.integration
def test_attach_runtime_attaches_stores_to_integration_manager(
    runtime_attacher: Mock,
    integration_manager: Mock,
    data_store: Mock,
    feature_store: Mock,
    model_store: Mock,
    strategy_store: Mock,
    data_registry: Mock,
    feature_registry: Mock,
    model_registry: Mock,
    strategy_registry: Mock,
) -> None:
    """Verify _attach_runtime() attaches all 4 stores to integration manager.

    Phase 2.2.8 (Full Implementation):
    - RuntimeAttacher._attach_runtime() invoked
    - integration_manager.attach_stores() called with all 4 stores
    - Stores accessible via integration_manager after attachment

    Expected Behavior:
    - Calls integration_manager.attach_stores(
        data_store=data_store,
        feature_store=feature_store,
        model_store=model_store,
        strategy_store=strategy_store,
      )
    - All 4 stores attached successfully
    """
    runtime_attacher._attach_runtime(
        data_store=data_store,
        feature_store=feature_store,
        model_store=model_store,
        strategy_store=strategy_store,
        data_registry=data_registry,
        feature_registry=feature_registry,
        model_registry=model_registry,
        strategy_registry=strategy_registry,
    )

    # Verify stores attached
    integration_manager.attach_stores.assert_called_once()
    args = integration_manager.attach_stores.call_args[1]  # kwargs
    assert args["data_store"] is data_store
    assert args["feature_store"] is feature_store
    assert args["model_store"] is model_store
    assert args["strategy_store"] is strategy_store


@pytest.mark.skip(reason="Structural phase - requires full implementation in Phase 2.2.8")
@pytest.mark.integration
def test_attach_runtime_attaches_registries_to_integration_manager(
    runtime_attacher: Mock,
    integration_manager: Mock,
    data_store: Mock,
    feature_store: Mock,
    model_store: Mock,
    strategy_store: Mock,
    data_registry: Mock,
    feature_registry: Mock,
    model_registry: Mock,
    strategy_registry: Mock,
) -> None:
    """Verify _attach_runtime() attaches all 4 registries to integration manager.

    Phase 2.2.8 (Full Implementation):
    - RuntimeAttacher._attach_runtime() invoked
    - integration_manager.attach_registries() called with all 4 registries
    - Registries accessible via integration_manager after attachment

    Expected Behavior:
    - Calls integration_manager.attach_registries(
        data_registry=data_registry,
        feature_registry=feature_registry,
        model_registry=model_registry,
        strategy_registry=strategy_registry,
      )
    - All 4 registries attached successfully
    """
    runtime_attacher._attach_runtime(
        data_store=data_store,
        feature_store=feature_store,
        model_store=model_store,
        strategy_store=strategy_store,
        data_registry=data_registry,
        feature_registry=feature_registry,
        model_registry=model_registry,
        strategy_registry=strategy_registry,
    )

    # Verify registries attached
    integration_manager.attach_registries.assert_called_once()
    args = integration_manager.attach_registries.call_args[1]  # kwargs
    assert args["data_registry"] is data_registry
    assert args["feature_registry"] is feature_registry
    assert args["model_registry"] is model_registry
    assert args["strategy_registry"] is strategy_registry


@pytest.mark.skip(reason="Structural phase - requires full implementation in Phase 2.2.8")
@pytest.mark.integration
def test_attach_runtime_coordinates_with_integration_manager(
    runtime_attacher: Mock,
    integration_manager: Mock,
    data_store: Mock,
    feature_store: Mock,
    model_store: Mock,
    strategy_store: Mock,
    data_registry: Mock,
    feature_registry: Mock,
    model_registry: Mock,
    strategy_registry: Mock,
) -> None:
    """Verify _attach_runtime() coordinates full attachment workflow.

    Phase 2.2.8 (Full Implementation):
    - attach_stores() called first
    - attach_registries() called second
    - Both complete successfully
    - Integration manager ready for pipeline execution

    Expected Behavior:
    - Call order: attach_stores → attach_registries
    - Both methods called exactly once
    - No exceptions raised
    """
    runtime_attacher._attach_runtime(
        data_store=data_store,
        feature_store=feature_store,
        model_store=model_store,
        strategy_store=strategy_store,
        data_registry=data_registry,
        feature_registry=feature_registry,
        model_registry=model_registry,
        strategy_registry=strategy_registry,
    )

    # Verify call order
    assert integration_manager.attach_stores.call_count == 1
    assert integration_manager.attach_registries.call_count == 1

    # Verify order: stores before registries
    assert integration_manager.mock_calls[0][0] == "attach_stores"
    assert integration_manager.mock_calls[1][0] == "attach_registries"


@pytest.mark.skip(reason="Structural phase - requires full implementation in Phase 2.2.8")
@pytest.mark.integration
def test_attach_runtime_handles_none_integration_manager(
    data_store: Mock,
    feature_store: Mock,
    model_store: Mock,
    strategy_store: Mock,
    data_registry: Mock,
    feature_registry: Mock,
    model_registry: Mock,
    strategy_registry: Mock,
) -> None:
    """Verify _attach_runtime() handles None integration manager gracefully.

    Phase 2.2.8 (Full Implementation):
    - _attach_runtime() detects None integration manager
    - Logs warning: 'No integration manager available, skipping runtime attachment'
    - Returns None (no attachment performed)
    - No exceptions raised

    Expected Behavior:
    - No integration manager calls (manager is None)
    - Warning logged
    - Returns None gracefully
    """
    from ml.orchestration.components.runtime_attacher import RuntimeAttacher

    attacher_no_manager = RuntimeAttacher(integration_manager=None)

    result = attacher_no_manager._attach_runtime(
        data_store=data_store,
        feature_store=feature_store,
        model_store=model_store,
        strategy_store=strategy_store,
        data_registry=data_registry,
        feature_registry=feature_registry,
        model_registry=model_registry,
        strategy_registry=strategy_registry,
    )

    assert result is None  # No exception, graceful handling
