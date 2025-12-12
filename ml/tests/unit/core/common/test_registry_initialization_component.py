"""
Unit tests for RegistryInitializationComponent.

This module tests the registry initialization component extracted from MLIntegrationManager
(Phase 3.6.3). Tests cover:

- Happy path: PostgreSQL registries, JSON registries, DataStore creation, DataRegistry injection
- Error conditions: DataRegistry injection failure, create_data_store before init
- Edge cases: Registry path creation, uninitialized registries

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from ml.core.common.registry_initialization import RegistryInitializationComponent
from ml.tests.utils.db import build_postgres_url


TEST_DB_CONNECTION = build_postgres_url()


if TYPE_CHECKING:
    pass


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tmp_registry_path(tmp_path: Path) -> Path:
    """Provide a temporary registry path for tests."""
    return tmp_path / "ml_registry"


@pytest.fixture
def postgres_component(tmp_registry_path: Path) -> RegistryInitializationComponent:
    """Provide a RegistryInitializationComponent configured for PostgreSQL."""
    return RegistryInitializationComponent(
        db_connection=TEST_DB_CONNECTION,
        registry_path=tmp_registry_path,
    )


@pytest.fixture
def json_fallback_component(tmp_registry_path: Path) -> RegistryInitializationComponent:
    """Provide a RegistryInitializationComponent in JSON fallback mode."""
    return RegistryInitializationComponent(
        db_connection=None,
        json_fallback=True,
        registry_path=tmp_registry_path,
    )


@pytest.fixture
def file_fallback_component(tmp_registry_path: Path) -> RegistryInitializationComponent:
    """Provide a RegistryInitializationComponent in file fallback mode."""
    return RegistryInitializationComponent(
        db_connection=None,
        file_fallback=True,
        registry_path=tmp_registry_path,
    )


@pytest.fixture
def mock_feature_registry() -> MagicMock:
    """Provide a mock FeatureRegistry."""
    mock = MagicMock()
    mock.list_all.return_value = []
    return mock


@pytest.fixture
def mock_model_registry() -> MagicMock:
    """Provide a mock ModelRegistry."""
    mock = MagicMock()
    mock.list_all.return_value = []
    return mock


@pytest.fixture
def mock_strategy_registry() -> MagicMock:
    """Provide a mock StrategyRegistry."""
    mock = MagicMock()
    mock.list_all.return_value = []
    return mock


# =============================================================================
# Happy Path Tests
# =============================================================================


class TestHappyPath:
    """Tests for successful operation paths."""

    def test_init_registries_creates_postgres_registries_when_connected(
        self,
        postgres_component: RegistryInitializationComponent,
        tmp_registry_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify PostgreSQL-backed registry creation.

        Input: Valid connection, no fallback.
        Expected Behavior: FeatureRegistry, ModelRegistry, StrategyRegistry, DataRegistry
        initialized with POSTGRES backend.
        """
        from ml.registry.persistence import BackendType

        # Mock registry constructors to avoid real DB connections
        mock_feature_registry = MagicMock()
        mock_model_registry = MagicMock()
        mock_strategy_registry = MagicMock()
        mock_data_registry = MagicMock()

        with patch("ml.registry.FeatureRegistry", return_value=mock_feature_registry), \
             patch("ml.registry.ModelRegistry", return_value=mock_model_registry), \
             patch("ml.registry.StrategyRegistry", return_value=mock_strategy_registry), \
             patch("ml.registry.DataRegistry", return_value=mock_data_registry):

            postgres_component.init_registries()

        # Verify all 4 registries are set
        assert postgres_component.feature_registry is mock_feature_registry
        assert postgres_component.model_registry is mock_model_registry
        assert postgres_component.strategy_registry is mock_strategy_registry
        assert postgres_component.data_registry is mock_data_registry

        # Verify persistence config uses POSTGRES backend
        assert postgres_component.persistence_config is not None
        assert postgres_component.persistence_config.backend == BackendType.POSTGRES

        # Verify registry path was created
        assert tmp_registry_path.exists()

    def test_init_registries_creates_json_registries_when_fallback(
        self,
        json_fallback_component: RegistryInitializationComponent,
        tmp_registry_path: Path,
    ) -> None:
        """Verify JSON-backed registry creation in fallback mode.

        Input: json_fallback=True.
        Expected Behavior: Registries with JSON persistence backend.
        """
        from ml.registry.persistence import BackendType

        # Mock registry constructors
        mock_feature_registry = MagicMock()
        mock_model_registry = MagicMock()
        mock_strategy_registry = MagicMock()
        mock_data_registry = MagicMock()

        with patch("ml.registry.FeatureRegistry", return_value=mock_feature_registry), \
             patch("ml.registry.ModelRegistry", return_value=mock_model_registry), \
             patch("ml.registry.StrategyRegistry", return_value=mock_strategy_registry), \
             patch("ml.registry.DataRegistry", return_value=mock_data_registry):

            json_fallback_component.init_registries()

        # Verify persistence config uses JSON backend
        assert json_fallback_component.persistence_config is not None
        assert json_fallback_component.persistence_config.backend == BackendType.JSON

    def test_init_registries_creates_data_store_with_registry(
        self,
        postgres_component: RegistryInitializationComponent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify DataStore wiring with DataRegistry.

        Input: Normal initialization.
        Expected Behavior: DataStore can be created with registry reference.
        """
        # Mock registry constructors
        mock_data_registry = MagicMock()
        mock_data_store = MagicMock()

        with patch("ml.registry.FeatureRegistry", return_value=MagicMock()), \
             patch("ml.registry.ModelRegistry", return_value=MagicMock()), \
             patch("ml.registry.StrategyRegistry", return_value=MagicMock()), \
             patch("ml.registry.DataRegistry", return_value=mock_data_registry), \
             patch("ml.core.integration.create_data_store", return_value=mock_data_store), \
             patch("ml.stores.providers.SqlMarketDataReader", return_value=MagicMock()):

            postgres_component.init_registries()
            data_store = postgres_component.create_data_store()

        assert data_store is mock_data_store

    def test_init_registries_injects_data_registry_into_stores(
        self,
        postgres_component: RegistryInitializationComponent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify FeatureStore/ModelStore get DataRegistry reference.

        Input: Stores with set_data_registry method.
        Expected Behavior: set_data_registry called on stores.
        """
        # Mock registry constructors
        mock_data_registry = MagicMock()

        # Create mock stores with set_data_registry method
        mock_feature_store = MagicMock()
        mock_feature_store.set_data_registry = MagicMock()
        mock_model_store = MagicMock()
        mock_model_store.set_data_registry = MagicMock()

        with patch("ml.registry.FeatureRegistry", return_value=MagicMock()), \
             patch("ml.registry.ModelRegistry", return_value=MagicMock()), \
             patch("ml.registry.StrategyRegistry", return_value=MagicMock()), \
             patch("ml.registry.DataRegistry", return_value=mock_data_registry):

            postgres_component.init_registries()
            postgres_component.inject_data_registry_into_stores(
                mock_feature_store,
                mock_model_store,
            )

        # Verify set_data_registry was called
        mock_feature_store.set_data_registry.assert_called_once_with(mock_data_registry)
        mock_model_store.set_data_registry.assert_called_once_with(mock_data_registry)

    def test_get_persistence_backend_returns_postgres(
        self,
        postgres_component: RegistryInitializationComponent,
    ) -> None:
        """Verify get_persistence_backend returns POSTGRES when connected.

        Input: PostgreSQL mode component with initialized registries.
        Expected Behavior: Returns 'POSTGRES'.
        """
        with patch("ml.registry.FeatureRegistry", return_value=MagicMock()), \
             patch("ml.registry.ModelRegistry", return_value=MagicMock()), \
             patch("ml.registry.StrategyRegistry", return_value=MagicMock()), \
             patch("ml.registry.DataRegistry", return_value=MagicMock()):

            postgres_component.init_registries()

        assert postgres_component.get_persistence_backend() == "POSTGRES"

    def test_get_persistence_backend_returns_json_in_fallback(
        self,
        json_fallback_component: RegistryInitializationComponent,
    ) -> None:
        """Verify get_persistence_backend returns JSON in fallback mode.

        Input: JSON fallback mode component with initialized registries.
        Expected Behavior: Returns 'JSON'.
        """
        with patch("ml.registry.FeatureRegistry", return_value=MagicMock()), \
             patch("ml.registry.ModelRegistry", return_value=MagicMock()), \
             patch("ml.registry.StrategyRegistry", return_value=MagicMock()), \
             patch("ml.registry.DataRegistry", return_value=MagicMock()):

            json_fallback_component.init_registries()

        assert json_fallback_component.get_persistence_backend() == "JSON"


# =============================================================================
# Error Condition Tests
# =============================================================================


class TestErrorConditions:
    """Tests for error handling paths."""

    def test_init_registries_handles_data_registry_injection_failure(
        self,
        postgres_component: RegistryInitializationComponent,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify graceful handling of injection failure.

        Input: Store without set_data_registry method.
        Expected Behavior: Logs debug, continues without error.
        """
        # Mock registry constructors
        mock_data_registry = MagicMock()

        # Create mock stores WITHOUT set_data_registry method
        mock_feature_store = MagicMock(spec=[])  # Empty spec = no methods
        mock_model_store = MagicMock(spec=[])

        with patch("ml.registry.FeatureRegistry", return_value=MagicMock()), \
             patch("ml.registry.ModelRegistry", return_value=MagicMock()), \
             patch("ml.registry.StrategyRegistry", return_value=MagicMock()), \
             patch("ml.registry.DataRegistry", return_value=mock_data_registry):

            postgres_component.init_registries()

            # Should not raise even though stores don't have set_data_registry
            postgres_component.inject_data_registry_into_stores(
                mock_feature_store,
                mock_model_store,
            )

        # Verify no exception was raised (test passes if we reach here)

    def test_create_data_store_raises_before_init_registries(
        self,
        postgres_component: RegistryInitializationComponent,
    ) -> None:
        """Verify error when creating DataStore before init_registries.

        Input: Component with uninitialized registries.
        Expected Behavior: RuntimeError raised.
        """
        # Don't call init_registries

        with pytest.raises(RuntimeError, match="Cannot create DataStore before"):
            postgres_component.create_data_store()

    def test_create_data_store_raises_in_json_fallback_mode(
        self,
        json_fallback_component: RegistryInitializationComponent,
    ) -> None:
        """Verify error when creating DataStore in JSON fallback mode.

        Input: JSON fallback mode with initialized registries.
        Expected Behavior: RuntimeError raised.
        """
        with patch("ml.registry.FeatureRegistry", return_value=MagicMock()), \
             patch("ml.registry.ModelRegistry", return_value=MagicMock()), \
             patch("ml.registry.StrategyRegistry", return_value=MagicMock()), \
             patch("ml.registry.DataRegistry", return_value=MagicMock()):

            json_fallback_component.init_registries()

        with pytest.raises(RuntimeError, match="DataStore creation not supported in fallback"):
            json_fallback_component.create_data_store()

    def test_create_data_store_raises_in_file_fallback_mode(
        self,
        file_fallback_component: RegistryInitializationComponent,
    ) -> None:
        """Verify error when creating DataStore in file fallback mode.

        Input: File fallback mode with initialized registries.
        Expected Behavior: RuntimeError raised.
        """
        with patch("ml.registry.FeatureRegistry", return_value=MagicMock()), \
             patch("ml.registry.ModelRegistry", return_value=MagicMock()), \
             patch("ml.registry.StrategyRegistry", return_value=MagicMock()), \
             patch("ml.registry.DataRegistry", return_value=MagicMock()):

            file_fallback_component.init_registries()

        with pytest.raises(RuntimeError, match="DataStore creation not supported in fallback"):
            file_fallback_component.create_data_store()

    def test_inject_handles_exception_gracefully(
        self,
        postgres_component: RegistryInitializationComponent,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify injection handles exceptions gracefully.

        Input: Store.set_data_registry raises exception.
        Expected Behavior: Logs debug, no exception propagates.
        """
        mock_data_registry = MagicMock()

        # Create mock store that raises on set_data_registry
        mock_feature_store = MagicMock()
        mock_feature_store.set_data_registry.side_effect = RuntimeError("Injection failed")

        with patch("ml.registry.FeatureRegistry", return_value=MagicMock()), \
             patch("ml.registry.ModelRegistry", return_value=MagicMock()), \
             patch("ml.registry.StrategyRegistry", return_value=MagicMock()), \
             patch("ml.registry.DataRegistry", return_value=mock_data_registry):

            postgres_component.init_registries()

            # Should not raise
            postgres_component.inject_data_registry_into_stores(
                mock_feature_store,
                None,
            )


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_registry_path_created_if_not_exists(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify registry path is created if it doesn't exist.

        Input: Non-existent registry path.
        Expected Behavior: Directory created during init_registries.
        """
        registry_path = tmp_path / "new_registry_path"
        assert not registry_path.exists()

        component = RegistryInitializationComponent(
            db_connection=TEST_DB_CONNECTION,
            registry_path=registry_path,
        )

        with patch("ml.registry.FeatureRegistry", return_value=MagicMock()), \
             patch("ml.registry.ModelRegistry", return_value=MagicMock()), \
             patch("ml.registry.StrategyRegistry", return_value=MagicMock()), \
             patch("ml.registry.DataRegistry", return_value=MagicMock()):

            component.init_registries()

        assert registry_path.exists()

    def test_get_persistence_backend_uninitialized(
        self,
        postgres_component: RegistryInitializationComponent,
    ) -> None:
        """Verify get_persistence_backend returns UNINITIALIZED when not init.

        Input: Component with uninitialized registries.
        Expected Behavior: Returns 'UNINITIALIZED'.
        """
        # Don't call init_registries
        assert postgres_component.get_persistence_backend() == "UNINITIALIZED"

    def test_get_registry_statistics_uninitialized(
        self,
        postgres_component: RegistryInitializationComponent,
    ) -> None:
        """Verify get_registry_statistics handles uninitialized registries.

        Input: Component with uninitialized registries.
        Expected Behavior: Returns not_initialized status for all registries.
        """
        stats = postgres_component.get_registry_statistics()

        assert stats["feature_registry"]["status"] == "not_initialized"
        assert stats["model_registry"]["status"] == "not_initialized"
        assert stats["strategy_registry"]["status"] == "not_initialized"
        assert stats["data_registry"]["status"] == "not_initialized"
        assert stats["persistence_backend"]["backend"] == "UNINITIALIZED"

    def test_get_registry_statistics_initialized(
        self,
        postgres_component: RegistryInitializationComponent,
    ) -> None:
        """Verify get_registry_statistics returns stats for initialized registries.

        Input: Component with initialized registries.
        Expected Behavior: Returns initialized status with counts.
        """
        # Create mock registries that return items
        mock_feature_registry = MagicMock()
        mock_feature_registry.list_all.return_value = [1, 2, 3]
        mock_model_registry = MagicMock()
        mock_model_registry.list_all.return_value = []

        with patch("ml.registry.FeatureRegistry", return_value=mock_feature_registry), \
             patch("ml.registry.ModelRegistry", return_value=mock_model_registry), \
             patch("ml.registry.StrategyRegistry", return_value=MagicMock()), \
             patch("ml.registry.DataRegistry", return_value=MagicMock()):

            postgres_component.init_registries()

        stats = postgres_component.get_registry_statistics()

        assert stats["feature_registry"]["status"] == "initialized"
        assert stats["feature_registry"]["count"] == 3
        assert stats["model_registry"]["count"] == 0

    def test_inject_data_registry_before_init_registries(
        self,
        postgres_component: RegistryInitializationComponent,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify inject logs debug when called before init_registries.

        Input: Component with uninitialized data_registry.
        Expected Behavior: Logs debug, returns without error.
        """
        # Don't call init_registries, so data_registry is None
        mock_feature_store = MagicMock()
        mock_model_store = MagicMock()

        # Should not raise
        postgres_component.inject_data_registry_into_stores(
            mock_feature_store,
            mock_model_store,
        )

        # set_data_registry should NOT have been called
        mock_feature_store.set_data_registry.assert_not_called()
        mock_model_store.set_data_registry.assert_not_called()

    def test_inject_handles_none_stores(
        self,
        postgres_component: RegistryInitializationComponent,
    ) -> None:
        """Verify inject handles None stores gracefully.

        Input: None feature_store and model_store.
        Expected Behavior: No exception raised.
        """
        with patch("ml.registry.FeatureRegistry", return_value=MagicMock()), \
             patch("ml.registry.ModelRegistry", return_value=MagicMock()), \
             patch("ml.registry.StrategyRegistry", return_value=MagicMock()), \
             patch("ml.registry.DataRegistry", return_value=MagicMock()):

            postgres_component.init_registries()

        # Should not raise with None stores
        postgres_component.inject_data_registry_into_stores(None, None)

    def test_default_registry_path(self) -> None:
        """Verify default registry path is ./ml_registry.

        Input: No registry_path provided.
        Expected Behavior: Uses ./ml_registry.
        """
        component = RegistryInitializationComponent(db_connection="postgresql://...")

        assert component.registry_path == Path("./ml_registry")

    def test_registries_default_to_none(self) -> None:
        """Verify registries default to None before initialization."""
        component = RegistryInitializationComponent(db_connection=None)

        assert component.feature_registry is None
        assert component.model_registry is None
        assert component.strategy_registry is None
        assert component.data_registry is None
        assert component.persistence_config is None

    def test_json_fallback_flag_default_false(self) -> None:
        """Verify json_fallback defaults to False."""
        component = RegistryInitializationComponent(db_connection=None)
        assert component.json_fallback is False

    def test_file_fallback_flag_default_false(self) -> None:
        """Verify file_fallback defaults to False."""
        component = RegistryInitializationComponent(db_connection=None)
        assert component.file_fallback is False


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestProtocolCompliance:
    """Tests for protocol compliance of initialized registries."""

    def test_component_creates_valid_persistence_config(
        self,
        postgres_component: RegistryInitializationComponent,
    ) -> None:
        """Verify persistence config has required attributes.

        Input: PostgreSQL mode component.
        Expected Behavior: persistence_config has backend and connection_string.
        """
        with patch("ml.registry.FeatureRegistry", return_value=MagicMock()), \
             patch("ml.registry.ModelRegistry", return_value=MagicMock()), \
             patch("ml.registry.StrategyRegistry", return_value=MagicMock()), \
             patch("ml.registry.DataRegistry", return_value=MagicMock()):

            postgres_component.init_registries()

        config = postgres_component.persistence_config
        assert hasattr(config, "backend")
        assert hasattr(config, "connection_string")
        assert config.connection_string == TEST_DB_CONNECTION


# =============================================================================
# Metric Emission Tests
# =============================================================================


class TestMetricEmission:
    """Tests for metric emission on fallback activation."""

    def test_init_registries_emits_metric_on_json_fallback(
        self,
        json_fallback_component: RegistryInitializationComponent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify metric emitted on JSON fallback activation.

        Input: JSON fallback mode.
        Expected Behavior: ml_registry_fallback_activations_total metric incremented.
        """
        # Track metric calls
        metric_calls: list[tuple[str, str]] = []

        # Mock the counter
        mock_counter = MagicMock()

        def mock_labels(component: str, level: str) -> MagicMock:
            metric_calls.append((component, level))
            return MagicMock()

        mock_counter.labels = mock_labels

        monkeypatch.setattr(
            "ml.core.common.registry_initialization._FALLBACK_COUNTER",
            mock_counter,
        )

        with patch("ml.registry.FeatureRegistry", return_value=MagicMock()), \
             patch("ml.registry.ModelRegistry", return_value=MagicMock()), \
             patch("ml.registry.StrategyRegistry", return_value=MagicMock()), \
             patch("ml.registry.DataRegistry", return_value=MagicMock()):

            json_fallback_component.init_registries()

        # Verify metric was called with correct labels
        assert len(metric_calls) > 0
        assert ("registry_initialization", "json") in metric_calls

    def test_init_registries_emits_metric_on_file_fallback(
        self,
        file_fallback_component: RegistryInitializationComponent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify metric emitted on file fallback activation.

        Input: File fallback mode.
        Expected Behavior: ml_registry_fallback_activations_total metric incremented with level=file.
        """
        # Track metric calls
        metric_calls: list[tuple[str, str]] = []

        # Mock the counter
        mock_counter = MagicMock()

        def mock_labels(component: str, level: str) -> MagicMock:
            metric_calls.append((component, level))
            return MagicMock()

        mock_counter.labels = mock_labels

        monkeypatch.setattr(
            "ml.core.common.registry_initialization._FALLBACK_COUNTER",
            mock_counter,
        )

        with patch("ml.registry.FeatureRegistry", return_value=MagicMock()), \
             patch("ml.registry.ModelRegistry", return_value=MagicMock()), \
             patch("ml.registry.StrategyRegistry", return_value=MagicMock()), \
             patch("ml.registry.DataRegistry", return_value=MagicMock()):

            file_fallback_component.init_registries()

        # Verify metric was called with correct labels
        assert len(metric_calls) > 0
        assert ("registry_initialization", "file") in metric_calls

    def test_init_registries_no_metric_on_postgres_mode(
        self,
        postgres_component: RegistryInitializationComponent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify no fallback metric emitted in PostgreSQL mode.

        Input: PostgreSQL mode (no fallback).
        Expected Behavior: No fallback metric incremented.
        """
        # Track metric calls
        metric_calls: list[tuple[str, str]] = []

        # Mock the counter
        mock_counter = MagicMock()

        def mock_labels(component: str, level: str) -> MagicMock:
            metric_calls.append((component, level))
            return MagicMock()

        mock_counter.labels = mock_labels

        monkeypatch.setattr(
            "ml.core.common.registry_initialization._FALLBACK_COUNTER",
            mock_counter,
        )

        with patch("ml.registry.FeatureRegistry", return_value=MagicMock()), \
             patch("ml.registry.ModelRegistry", return_value=MagicMock()), \
             patch("ml.registry.StrategyRegistry", return_value=MagicMock()), \
             patch("ml.registry.DataRegistry", return_value=MagicMock()):

            postgres_component.init_registries()

        # No fallback metric should be emitted in PostgreSQL mode
        assert len(metric_calls) == 0
