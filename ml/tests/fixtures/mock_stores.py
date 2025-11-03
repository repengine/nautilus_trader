"""Centralized mock store fixture factory."""

from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

from ml.stores.data_store import DataStore
from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategyStore


@pytest.fixture
def mock_store_factory():
    """Factory for creating mock stores with consistent specs.

    This factory consolidates duplicate mock store fixture definitions across
    the test suite, reducing code duplication and improving maintainability.

    Usage Examples
    --------------
    Basic usage:
        def test_with_feature_store(mock_store_factory):
            store = mock_store_factory("feature")
            store.write_data.return_value = None
            # ... test logic

    With custom attributes:
        def test_with_custom_store(mock_store_factory):
            store = mock_store_factory("model", engine=None, table_name="predictions")
            # ... test logic

    With custom return values:
        def test_with_return_values(mock_store_factory):
            store = mock_store_factory("strategy")
            store.get_signals.return_value = [{"signal": "BUY"}]
            # ... test logic

    Parameters
    ----------
    store_type : str
        Type of store to create. One of: "feature", "model", "strategy", "data"
    use_spec : bool, default=True
        Whether to use the actual store class as spec for the mock.
        Set to False for tests that need fully dynamic mocks.
    **kwargs : Any
        Additional attributes to set on the mock. Can be used to:
        - Set attributes: engine=None, table_name="foo"
        - Configure return values: get_data=MagicMock(return_value=[])

    Returns
    -------
    MagicMock
        Configured mock store with appropriate spec and attributes

    Raises
    ------
    ValueError
        If store_type is not one of the supported types
    """
    def _factory(store_type: str, use_spec: bool = True, **kwargs: Any) -> MagicMock:
        """Create a mock store of the specified type.

        Args:
            store_type: One of "feature", "model", "strategy", "data"
            use_spec: Whether to use actual store class as spec
            **kwargs: Additional attributes to set on the mock

        Returns:
            MagicMock with appropriate spec
        """
        specs = {
            "feature": FeatureStore,
            "model": ModelStore,
            "strategy": StrategyStore,
            "data": DataStore,
        }

        if store_type not in specs:
            raise ValueError(
                f"Invalid store_type: {store_type}. "
                f"Must be one of: {list(specs.keys())}"
            )

        # Create mock with or without spec based on use_spec parameter
        if use_spec:
            mock = MagicMock(spec=specs[store_type])
        else:
            mock = MagicMock()

        # Set any additional attributes or configure return values
        for attr, value in kwargs.items():
            setattr(mock, attr, value)

        return mock

    return _factory


# Convenience fixtures for backward compatibility
# These allow gradual migration and don't require changing all tests at once

@pytest.fixture
def mock_feature_store(mock_store_factory) -> MagicMock:
    """Mock FeatureStore for unit tests.

    DEPRECATED: Use mock_store_factory("feature") directly.
    This fixture exists for backward compatibility during migration.
    """
    return mock_store_factory("feature")


@pytest.fixture
def mock_model_store(mock_store_factory) -> MagicMock:
    """Mock ModelStore for unit tests.

    DEPRECATED: Use mock_store_factory("model") directly.
    This fixture exists for backward compatibility during migration.
    """
    return mock_store_factory("model")


@pytest.fixture
def mock_strategy_store(mock_store_factory) -> MagicMock:
    """Mock StrategyStore for unit tests.

    DEPRECATED: Use mock_store_factory("strategy") directly.
    This fixture exists for backward compatibility during migration.
    """
    return mock_store_factory("strategy")


@pytest.fixture
def mock_data_store(mock_store_factory) -> MagicMock:
    """Mock DataStore for unit tests.

    DEPRECATED: Use mock_store_factory("data") directly.
    This fixture exists for backward compatibility during migration.
    """
    return mock_store_factory("data")


@pytest.fixture
def mock_registry_factory():
    """Factory for creating mock registries with consistent specs.

    This factory consolidates duplicate mock registry fixture definitions across
    the test suite, reducing code duplication and improving maintainability.

    Usage Examples
    --------------
    Basic usage:
        def test_with_model_registry(mock_registry_factory):
            registry = mock_registry_factory("model")
            registry.get_model.return_value = mock_model_info
            # ... test logic

    With pre-configured data:
        def test_with_feature_registry(mock_registry_factory):
            registry = mock_registry_factory("feature", with_manifest=True)
            # ... test logic

    With custom methods:
        def test_with_data_registry(mock_registry_factory):
            registry = mock_registry_factory("data")
            registry.get_manifest.return_value = mock_manifest
            # ... test logic

    Protocol-based mocking:
        def test_with_protocol(mock_registry_factory):
            registry = mock_registry_factory("protocol")
            # Generic RegistryProtocol mock
            # ... test logic

    Parameters
    ----------
    registry_type : str
        Type of registry to create. One of: "model", "feature", "data",
        "strategy", "protocol"
    use_spec : bool, default=True
        Whether to use the actual registry class as spec for the mock.
        Set to False for tests that need fully dynamic mocks.
    with_manifest : bool, default=False
        Whether to pre-configure the mock with a sample manifest.
        Only applies to model, feature, and data registries.
    **kwargs : Any
        Additional attributes to set on the mock. Can be used to:
        - Set attributes: registry_path=None, backend="json"
        - Configure return values: get_model=MagicMock(return_value=...)
        - Set side effects: list_models=MagicMock(side_effect=...)

    Returns
    -------
    MagicMock
        Configured mock registry with appropriate spec and attributes

    Raises
    ------
    ValueError
        If registry_type is not one of the supported types
    """
    def _factory(
        registry_type: str,
        use_spec: bool = True,
        with_manifest: bool = False,
        **kwargs: Any
    ) -> MagicMock:
        """Create a mock registry of the specified type.

        Args:
            registry_type: One of "model", "feature", "data", "strategy", "protocol"
            use_spec: Whether to use actual registry class as spec
            with_manifest: Whether to pre-configure with sample manifest
            **kwargs: Additional attributes to set on the mock

        Returns:
            MagicMock with appropriate spec and configuration
        """
        from ml.registry.model_registry import ModelRegistry
        from ml.registry.feature_registry import FeatureRegistry
        from ml.registry.data_registry import DataRegistry
        from ml.registry.strategy_registry import StrategyRegistry
        from ml.registry.protocols import RegistryProtocol

        specs = {
            "model": ModelRegistry,
            "feature": FeatureRegistry,
            "data": DataRegistry,
            "strategy": StrategyRegistry,
            "protocol": RegistryProtocol,
        }

        if registry_type not in specs:
            raise ValueError(
                f"Invalid registry_type: {registry_type}. "
                f"Must be one of: {list(specs.keys())}"
            )

        # Create mock with or without spec
        if use_spec:
            mock = MagicMock(spec=specs[registry_type])
        else:
            mock = MagicMock()

        # Pre-configure with manifest if requested
        if with_manifest and registry_type != "protocol" and registry_type != "strategy":
            _configure_manifest(mock, registry_type)

        # Set any additional attributes or configure return values
        for attr, value in kwargs.items():
            setattr(mock, attr, value)

        return mock

    def _configure_manifest(mock: MagicMock, registry_type: str) -> None:
        """Pre-configure mock with sample manifest data.

        Args:
            mock: The MagicMock registry to configure
            registry_type: Type of registry ("model", "feature", or "data")
        """
        import time
        from pathlib import Path
        from ml.registry.base import DataRequirements, ModelRole, ModelManifest
        from ml.registry.feature_registry import FeatureManifest, FeatureRole
        from ml.registry.dataclasses import DatasetManifest, DatasetType, StorageKind

        if registry_type == "model":
            mock_model_info = MagicMock()
            mock_model_info.manifest = ModelManifest(
                model_id="test_model_v1",
                role=ModelRole.INFERENCE,
                data_requirements=DataRequirements.L1_ONLY,
                architecture="xgboost",
                feature_schema={"feature_1": "float", "feature_2": "float", "feature_3": "float"},
                feature_schema_hash="abc123",
                version="1.0.0",
                created_at=time.time(),
                last_modified=time.time(),
                performance_metrics={"accuracy": 0.95, "precision": 0.92, "recall": 0.93},
                training_config={"n_estimators": 100, "max_depth": 5},
            )
            mock_model_info.path = Path("/tmp/test_model.onnx")
            mock_model_info.metadata = {"test": True}
            mock.get_model = MagicMock(return_value=mock_model_info)
            mock.list_models = MagicMock(return_value=["test_model_v1"])
            mock.register_model = MagicMock(return_value="test_model_v1")
            mock.load_model = MagicMock(return_value=mock_model_info)

        elif registry_type == "feature":
            mock_feature_info = MagicMock()
            mock_feature_info.manifest = FeatureManifest(
                feature_set_id="test_features_v1",
                name="Test Features",
                version="1.0.0",
                role=FeatureRole.INFERENCE_SUPPORT,
                data_requirements=DataRequirements.L1_ONLY,
                feature_names=["sma_20", "rsi_14", "volume_ratio"],
                feature_dtypes=["float32", "float32", "float32"],
                schema_hash="def456",
                pipeline_signature="pipeline_sig_123",
                pipeline_version="1.0.0",
                created_at=time.time(),
                last_modified=time.time(),
            )
            mock.get_feature_set = MagicMock(return_value=mock_feature_info)
            mock.list_feature_sets = MagicMock(return_value=["test_features_v1"])
            mock.register_feature_set = MagicMock(return_value="test_features_v1")

        elif registry_type == "data":
            manifest = DatasetManifest(
                dataset_id="test_dataset",
                dataset_type=DatasetType.FEATURES,
                storage_kind=StorageKind.POSTGRES,
                location="ml.test_dataset",
                partitioning={},
                retention_days=90,
                version="1.0.0",
                schema={"instrument_id": "str", "ts_event": "int64", "ts_init": "int64"},
                schema_hash="test_hash",
                primary_keys=["instrument_id", "ts_event"],
                ts_field="ts_event",
                seq_field=None,
                constraints={},
                lineage=[],
                pipeline_signature="test",
            )
            mock.get_manifest = MagicMock(return_value=manifest)
            mock.list_datasets = MagicMock(return_value=["test_dataset"])
            mock.register_dataset = MagicMock()
            mock.update_manifest = MagicMock()

    return _factory


# Convenience fixtures for backward compatibility with existing tests
# These allow gradual migration and don't require changing all tests at once

@pytest.fixture
def mock_model_registry(mock_registry_factory) -> MagicMock:
    """Mock ModelRegistry for unit tests.

    DEPRECATED: Use mock_registry_factory("model", with_manifest=True) directly.
    This fixture exists for backward compatibility during migration.

    Returns:
        MagicMock: Pre-configured mock ModelRegistry with sample manifest
    """
    return mock_registry_factory("model", with_manifest=True)


@pytest.fixture
def mock_feature_registry(mock_registry_factory) -> MagicMock:
    """Mock FeatureRegistry for unit tests.

    DEPRECATED: Use mock_registry_factory("feature", with_manifest=True) directly.
    This fixture exists for backward compatibility during migration.

    Returns:
        MagicMock: Pre-configured mock FeatureRegistry with sample manifest
    """
    return mock_registry_factory("feature", with_manifest=True)


@pytest.fixture
def mock_data_registry(mock_registry_factory) -> MagicMock:
    """Mock DataRegistry for unit tests.

    DEPRECATED: Use mock_registry_factory("data", with_manifest=True) directly.
    This fixture exists for backward compatibility during migration.

    Returns:
        MagicMock: Pre-configured mock DataRegistry with sample manifest
    """
    return mock_registry_factory("data", with_manifest=True)


@pytest.fixture
def mock_strategy_registry(mock_registry_factory) -> MagicMock:
    """Mock StrategyRegistry for unit tests.

    DEPRECATED: Use mock_registry_factory("strategy") directly.
    This fixture exists for backward compatibility during migration.

    Returns:
        MagicMock: Mock StrategyRegistry
    """
    return mock_registry_factory("strategy")


__all__ = [
    "mock_data_registry",
    "mock_data_store",
    "mock_feature_registry",
    "mock_feature_store",
    "mock_model_registry",
    "mock_model_store",
    "mock_registry_factory",
    "mock_store_factory",
    "mock_strategy_registry",
    "mock_strategy_store",
]
