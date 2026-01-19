"""
Unit tests for StoreInitializationComponent.

This module tests the store initialization component extracted from MLIntegrationManager
(Phase 3.6.2). Tests cover:

- Happy path: PostgreSQL stores, file stores, dummy stores
- Error conditions: mkdir failures, missing db_connection
- Edge cases: None data_store handling, flush resilience

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from ml.core.common.store_initialization import StoreInitializationComponent
from ml.tests.utils.db import build_postgres_url


TEST_DB_CONNECTION = build_postgres_url()


if TYPE_CHECKING:
    pass


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tmp_file_store_path(tmp_path: Path) -> Path:
    """Provide a temporary file store path for tests."""
    return tmp_path / "file_store"


@pytest.fixture
def default_component(tmp_file_store_path: Path) -> StoreInitializationComponent:
    """Provide a default StoreInitializationComponent for unit tests."""
    return StoreInitializationComponent(
        db_connection=TEST_DB_CONNECTION,
        file_store_path=tmp_file_store_path,
    )


@pytest.fixture
def dummy_mode_component(tmp_file_store_path: Path) -> StoreInitializationComponent:
    """Provide a StoreInitializationComponent in dummy mode."""
    component = StoreInitializationComponent(
        db_connection=None,
        file_store_path=tmp_file_store_path,
    )
    component.json_fallback = True
    return component


@pytest.fixture
def file_fallback_component(tmp_file_store_path: Path) -> StoreInitializationComponent:
    """Provide a StoreInitializationComponent in file fallback mode."""
    component = StoreInitializationComponent(
        db_connection=None,
        file_store_path=tmp_file_store_path,
    )
    component.file_fallback = True
    return component


# =============================================================================
# Happy Path Tests
# =============================================================================


class TestHappyPath:
    """Tests for successful operation paths."""

    def test_init_stores_creates_postgres_stores_when_connected(
        self,
        default_component: StoreInitializationComponent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify PostgreSQL-backed store creation.

        Input: Valid connection, no fallback flags.
        Expected Behavior: FeatureStore, ModelStore, StrategyStore, EarningsStore initialized.
        """
        from ml.stores.feature_store import FeatureStore
        from ml.stores.model_store import ModelStore
        from ml.stores.strategy_store import StrategyStore
        from ml.features.earnings.store import EarningsStore

        # Mock the store constructors to avoid real DB connections
        mock_feature_store = MagicMock(spec=FeatureStore)
        mock_model_store = MagicMock(spec=ModelStore)
        mock_strategy_store = MagicMock(spec=StrategyStore)
        mock_earnings_store = MagicMock(spec=EarningsStore)

        # Patch at the module level where they are imported
        with patch("ml.stores.feature_store.FeatureStore", return_value=mock_feature_store), \
             patch("ml.stores.model_store.ModelStore", return_value=mock_model_store), \
             patch("ml.stores.strategy_store.StrategyStore", return_value=mock_strategy_store), \
             patch("ml.features.earnings.store.EarningsStore", return_value=mock_earnings_store):

            default_component.init_stores()

        assert default_component.feature_store is mock_feature_store
        assert default_component.model_store is mock_model_store
        assert default_component.strategy_store is mock_strategy_store
        assert default_component.earnings_store is mock_earnings_store
        # DataStore is None in PostgreSQL mode (set in registries init)
        assert default_component.data_store is None

    def test_init_stores_creates_file_stores_when_file_fallback(
        self,
        file_fallback_component: StoreInitializationComponent,
        tmp_file_store_path: Path,
    ) -> None:
        """Verify file-backed store creation.

        Input: file_fallback=True.
        Expected Behavior: FileFeatureStore, FileModelStore, FileStrategyStore, FileDataStore initialized.
        """
        from ml.stores.file_backed import FileDataStore
        from ml.stores.file_backed import FileEarningsStore
        from ml.stores.file_backed import FileFeatureStore
        from ml.stores.file_backed import FileModelStore
        from ml.stores.file_backed import FileStrategyStore

        file_fallback_component.init_stores()

        assert isinstance(file_fallback_component.feature_store, FileFeatureStore)
        assert isinstance(file_fallback_component.model_store, FileModelStore)
        assert isinstance(file_fallback_component.strategy_store, FileStrategyStore)
        assert isinstance(file_fallback_component.data_store, FileDataStore)
        assert isinstance(file_fallback_component.earnings_store, FileEarningsStore)

    def test_init_stores_creates_dummy_stores_when_json_fallback(
        self,
        dummy_mode_component: StoreInitializationComponent,
    ) -> None:
        """Verify dummy store creation.

        Input: json_fallback=True.
        Expected Behavior: DummyStore instances for all stores and DummyEarningsStore.
        """
        from ml.stores.base import DummyStore
        from ml.features.earnings.store import DummyEarningsStore

        dummy_mode_component.init_stores()

        assert isinstance(dummy_mode_component.feature_store, DummyStore)
        assert isinstance(dummy_mode_component.model_store, DummyStore)
        assert isinstance(dummy_mode_component.strategy_store, DummyStore)
        assert isinstance(dummy_mode_component.data_store, DummyStore)
        assert isinstance(dummy_mode_component.earnings_store, DummyEarningsStore)

    def test_enable_file_fallback_creates_directory_and_returns_true(
        self,
        tmp_file_store_path: Path,
    ) -> None:
        """Verify file fallback setup.

        Input: Valid file path.
        Expected Behavior: Directory created, file_fallback=True.
        """
        component = StoreInitializationComponent(
            db_connection=None,
            file_store_path=tmp_file_store_path,
        )

        result = component.enable_file_fallback()

        assert result is True
        assert component.file_fallback is True
        assert tmp_file_store_path.exists()

    def test_init_dummy_components_creates_all_eight_components(
        self,
        tmp_file_store_path: Path,
    ) -> None:
        """Verify dummy mode creates all components.

        Input: Dummy mode requested.
        Expected Behavior: 4 DummyStore + DummyEarningsStore + 4 DummyRegistry created.
        """
        from ml.registry.base import DummyRegistry
        from ml.stores.base import DummyStore
        from ml.features.earnings.store import DummyEarningsStore

        component = StoreInitializationComponent(
            db_connection=None,
            file_store_path=tmp_file_store_path,
        )

        component.init_dummy_components()

        # Verify 4 stores are DummyStore
        assert isinstance(component.feature_store, DummyStore)
        assert isinstance(component.model_store, DummyStore)
        assert isinstance(component.strategy_store, DummyStore)
        assert isinstance(component.data_store, DummyStore)
        assert isinstance(component.earnings_store, DummyEarningsStore)

        # Verify 4 registries are DummyRegistry
        assert isinstance(component.feature_registry, DummyRegistry)
        assert isinstance(component.model_registry, DummyRegistry)
        assert isinstance(component.strategy_registry, DummyRegistry)
        assert isinstance(component.data_registry, DummyRegistry)

    def test_flush_all_flushes_all_stores(
        self,
        file_fallback_component: StoreInitializationComponent,
    ) -> None:
        """Verify flush_all calls flush on all stores.

        Input: Initialized stores.
        Expected Behavior: All store flush methods called.
        """
        file_fallback_component.init_stores()

        # Create mocks to track flush calls
        flush_calls: list[str] = []

        original_feature_flush = file_fallback_component.feature_store.flush
        original_model_flush = file_fallback_component.model_store.flush
        original_strategy_flush = file_fallback_component.strategy_store.flush
        original_data_flush = file_fallback_component.data_store.flush

        def track_feature_flush() -> None:
            flush_calls.append("feature_store")
            original_feature_flush()

        def track_model_flush() -> None:
            flush_calls.append("model_store")
            original_model_flush()

        def track_strategy_flush() -> None:
            flush_calls.append("strategy_store")
            original_strategy_flush()

        def track_data_flush() -> None:
            flush_calls.append("data_store")
            original_data_flush()

        file_fallback_component.feature_store.flush = track_feature_flush
        file_fallback_component.model_store.flush = track_model_flush
        file_fallback_component.strategy_store.flush = track_strategy_flush
        file_fallback_component.data_store.flush = track_data_flush

        file_fallback_component.flush_all()

        assert "feature_store" in flush_calls
        assert "model_store" in flush_calls
        assert "strategy_store" in flush_calls
        assert "data_store" in flush_calls


# =============================================================================
# Error Condition Tests
# =============================================================================


class TestErrorConditions:
    """Tests for error handling paths."""

    def test_enable_file_fallback_returns_false_when_mkdir_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify graceful handling of directory creation failure.

        Input: Invalid path (permissions, etc.).
        Expected Behavior: Returns False, file_fallback remains False.
        """
        component = StoreInitializationComponent(
            db_connection=None,
            file_store_path=Path("/nonexistent/readonly/path/that/should/fail"),
        )

        # Mock mkdir to raise
        def raise_permission_error(*args: object, **kwargs: object) -> None:
            raise PermissionError("Permission denied")

        monkeypatch.setattr(Path, "mkdir", raise_permission_error)

        result = component.enable_file_fallback()

        assert result is False
        assert component.file_fallback is False

    def test_init_postgres_stores_raises_when_db_connection_none(
        self,
        tmp_file_store_path: Path,
    ) -> None:
        """Verify error when PostgreSQL mode without connection string.

        Input: db_connection=None, no fallback flags.
        Expected Behavior: ValueError raised.
        """
        component = StoreInitializationComponent(
            db_connection=None,
            file_store_path=tmp_file_store_path,
        )
        # Neither fallback is set, so it will try PostgreSQL path

        with pytest.raises(ValueError, match="db_connection required"):
            component.init_stores()

    def test_flush_all_handles_flush_exceptions(
        self,
        file_fallback_component: StoreInitializationComponent,
    ) -> None:
        """Verify flush resilience when stores raise exceptions.

        Input: Store flush() raises exception.
        Expected Behavior: Other stores still flushed, no exception propagates.
        """
        file_fallback_component.init_stores()

        # Make feature_store.flush raise
        def raise_error() -> None:
            raise RuntimeError("Flush failed")

        file_fallback_component.feature_store.flush = raise_error

        # Track other flush calls
        flush_calls: list[str] = []
        original_model_flush = file_fallback_component.model_store.flush
        original_strategy_flush = file_fallback_component.strategy_store.flush

        def track_model_flush() -> None:
            flush_calls.append("model_store")
            original_model_flush()

        def track_strategy_flush() -> None:
            flush_calls.append("strategy_store")
            original_strategy_flush()

        file_fallback_component.model_store.flush = track_model_flush
        file_fallback_component.strategy_store.flush = track_strategy_flush

        # Should not raise
        file_fallback_component.flush_all()

        # Other stores should still be flushed
        assert "model_store" in flush_calls
        assert "strategy_store" in flush_calls


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_init_stores_handles_none_data_store_gracefully(
        self,
        default_component: StoreInitializationComponent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify DataStore deferred initialization.

        Input: Normal PostgreSQL initialization path.
        Expected Behavior: data_store is None after init_stores, set in registries.
        """
        # Mock the store constructors at the actual import locations
        with patch("ml.stores.feature_store.FeatureStore", return_value=MagicMock()), \
             patch("ml.stores.model_store.ModelStore", return_value=MagicMock()), \
             patch("ml.stores.strategy_store.StrategyStore", return_value=MagicMock()):

            default_component.init_stores()

        # DataStore is None in PostgreSQL mode immediately after init_stores
        assert default_component.data_store is None

    def test_set_data_store_sets_data_store(
        self,
        default_component: StoreInitializationComponent,
    ) -> None:
        """Verify set_data_store properly sets the data_store attribute.

        Input: A mock data store.
        Expected Behavior: data_store attribute is set.
        """
        mock_data_store = MagicMock()

        default_component.set_data_store(mock_data_store)

        assert default_component.data_store is mock_data_store

    def test_get_store_statistics_returns_all_stores(
        self,
        file_fallback_component: StoreInitializationComponent,
    ) -> None:
        """Verify get_store_statistics returns stats for all stores.

        Input: Initialized file stores.
        Expected Behavior: Stats dict contains all four stores.
        """
        file_fallback_component.init_stores()

        stats = file_fallback_component.get_store_statistics()

        assert "feature_store" in stats
        assert "model_store" in stats
        assert "strategy_store" in stats
        assert "data_store" in stats

    def test_get_store_statistics_handles_uninitialized_stores(
        self,
        tmp_file_store_path: Path,
    ) -> None:
        """Verify get_store_statistics handles None stores.

        Input: Component with uninitialized stores.
        Expected Behavior: Stats show not_initialized status.
        """
        component = StoreInitializationComponent(
            db_connection=None,
            file_store_path=tmp_file_store_path,
        )
        # Don't call init_stores

        stats = component.get_store_statistics()

        # All stores should report not_initialized
        for name in ["feature_store", "model_store", "strategy_store", "data_store"]:
            assert name in stats
            assert stats[name].get("status") == "not_initialized"

    def test_get_store_statistics_handles_statistics_error(
        self,
        file_fallback_component: StoreInitializationComponent,
    ) -> None:
        """Verify get_store_statistics handles exceptions gracefully.

        Input: Store get_statistics raises.
        Expected Behavior: Returns error info, doesn't raise.
        """
        file_fallback_component.init_stores()

        # Make feature_store.get_statistics raise
        def raise_error(*args: object, **kwargs: object) -> dict:
            raise RuntimeError("Statistics failed")

        file_fallback_component.feature_store.get_statistics = raise_error

        stats = file_fallback_component.get_store_statistics()

        assert stats["feature_store"].get("error") == "Failed to get statistics"
        # Other stores should still have stats
        assert "error" not in stats.get("model_store", {})

    def test_default_file_store_path_uses_home_directory(self) -> None:
        """Verify default file_store_path uses home directory.

        Input: No file_store_path provided.
        Expected Behavior: Uses ~/.nautilus/ml/file_store.
        """
        component = StoreInitializationComponent(db_connection=None)

        # Should use default path (home directory based)
        expected_suffix = Path(".nautilus") / "ml" / "file_store"
        assert str(component.file_store_path).endswith(str(expected_suffix))

    def test_file_stores_create_directory_structure(
        self,
        file_fallback_component: StoreInitializationComponent,
        tmp_file_store_path: Path,
    ) -> None:
        """Verify file stores create proper directory structure.

        Input: file_fallback=True.
        Expected Behavior: features, models, strategies, datastore directories created.
        """
        file_fallback_component.init_stores()

        # Verify directory structure
        assert (tmp_file_store_path / "features").exists()
        assert (tmp_file_store_path / "models").exists()
        assert (tmp_file_store_path / "strategies").exists()
        assert (tmp_file_store_path / "datastore").exists()
        assert (tmp_file_store_path / "earnings").exists()

    def test_json_fallback_flag_default_false(self) -> None:
        """Verify json_fallback defaults to False."""
        component = StoreInitializationComponent(db_connection=None)
        assert component.json_fallback is False

    def test_file_fallback_flag_default_false(self) -> None:
        """Verify file_fallback defaults to False."""
        component = StoreInitializationComponent(db_connection=None)
        assert component.file_fallback is False

    def test_stores_default_to_none(self) -> None:
        """Verify stores default to None before initialization."""
        component = StoreInitializationComponent(db_connection=None)

        assert component.feature_store is None
        assert component.model_store is None
        assert component.strategy_store is None
        assert component.data_store is None


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestProtocolCompliance:
    """Tests for protocol compliance of initialized stores."""

    def test_file_stores_have_flush_method(
        self,
        file_fallback_component: StoreInitializationComponent,
    ) -> None:
        """Verify file stores have flush() method.

        Input: File-backed stores.
        Expected Behavior: All stores have flush() method.
        """
        file_fallback_component.init_stores()

        stores = [
            ("feature_store", file_fallback_component.feature_store),
            ("model_store", file_fallback_component.model_store),
            ("strategy_store", file_fallback_component.strategy_store),
            ("data_store", file_fallback_component.data_store),
        ]

        for name, store in stores:
            assert hasattr(store, "flush"), f"{name} should have flush method"
            assert callable(getattr(store, "flush")), f"{name}.flush should be callable"

    def test_file_feature_store_has_get_statistics(
        self,
        file_fallback_component: StoreInitializationComponent,
    ) -> None:
        """Verify FileFeatureStore has get_statistics method.

        Input: File-backed feature store.
        Expected Behavior: Feature store has get_statistics() method.
        """
        file_fallback_component.init_stores()

        assert hasattr(file_fallback_component.feature_store, "get_statistics")
        assert callable(getattr(file_fallback_component.feature_store, "get_statistics"))

        # Also verify FileDataStore has it
        assert hasattr(file_fallback_component.data_store, "get_statistics")

    def test_dummy_stores_implement_store_protocol(
        self,
        dummy_mode_component: StoreInitializationComponent,
    ) -> None:
        """Verify dummy stores implement StoreProtocol.

        Input: Dummy stores.
        Expected Behavior: All stores have flush() and get_statistics() methods.
        """
        dummy_mode_component.init_stores()

        stores = [
            dummy_mode_component.feature_store,
            dummy_mode_component.model_store,
            dummy_mode_component.strategy_store,
            dummy_mode_component.data_store,
        ]

        for store in stores:
            assert hasattr(store, "flush")
            assert callable(getattr(store, "flush"))
            assert hasattr(store, "get_statistics")
            assert callable(getattr(store, "get_statistics"))


# =============================================================================
# Metric Emission Tests
# =============================================================================


class TestMetricEmission:
    """Tests for metric emission on fallback activation."""

    def test_enable_file_fallback_emits_metric(
        self,
        tmp_file_store_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify metric emitted on file fallback activation.

        Input: File fallback enabled.
        Expected Behavior: ml_fallback_activations_total metric incremented.
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
            "ml.core.common.store_initialization._FALLBACK_COUNTER",
            mock_counter,
        )

        component = StoreInitializationComponent(
            db_connection=None,
            file_store_path=tmp_file_store_path,
        )

        component.enable_file_fallback()

        # Verify metric was called with correct labels
        assert len(metric_calls) > 0
        assert ("store_initialization", "file") in metric_calls
