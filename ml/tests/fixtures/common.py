#!/usr/bin/env python3

"""
Common test fixtures for ML module.

This module provides reusable fixtures for ML tests to reduce duplication and improve
maintainability.

"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from ml.config.actors import MLSignalActorConfig
from ml.config.base import MLActorConfig
from ml.config.base import MLFeatureConfig
from ml.config.registry import ModelRegistryConfig
from ml.registry.model_registry import ModelManifest
from ml.registry.feature_registry import FeatureManifest
from ml.tests.fixtures.model_factory import TestModelFactory
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import ComponentId
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Venue


if TYPE_CHECKING:
    from ml.registry.base import BaseRegistry


# ============================================================================
# Core Nautilus Type Fixtures
# ============================================================================


@pytest.fixture
def default_venue() -> Venue:
    """
    Standard test venue.
    """
    return Venue("SIM")


@pytest.fixture
def default_instrument_id(default_venue: Venue) -> InstrumentId:
    """
    Standard instrument ID for testing.

    Returns 'EUR/USD.SIM' which is used in most test cases.

    """
    return InstrumentId.from_str("EUR/USD.SIM")


@pytest.fixture
def default_bar_type(default_instrument_id: InstrumentId) -> BarType:
    """
    Standard bar type for testing.

    Returns 'EUR/USD.SIM-1-MINUTE-MID-INTERNAL' which is the most common test pattern.

    """
    return BarType.from_str("EUR/USD.SIM-1-MINUTE-MID-INTERNAL")


@pytest.fixture
def test_component_id() -> ComponentId:
    """
    Standard component ID for testing.
    """
    return ComponentId("TEST-001")


@pytest.fixture
def alternative_instrument_id() -> InstrumentId:
    """
    Alternative instrument for multi-instrument tests.
    """
    return InstrumentId.from_str("BTC/USD.SIM")


@pytest.fixture
def alternative_bar_type() -> BarType:
    """
    Alternative bar type for multi-timeframe tests.
    """
    return BarType.from_str("BTC/USD.SIM-5-MINUTE-MID-INTERNAL")


# ============================================================================
# ML Configuration Fixtures
# ============================================================================


@pytest.fixture
def base_feature_config() -> MLFeatureConfig:
    """
    Base feature configuration with common defaults.
    """
    return MLFeatureConfig(
        lookback_window=20,
        indicators={
            "sma": {"period": 20},
            "rsi": {"period": 14},
        },
        normalize_features=True,
        fill_missing_with=0.0,
    )


@pytest.fixture
def dummy_onnx_model() -> Path:
    """
    Create a minimal ONNX model for testing.

    Returns path to a valid ONNX model file that will be cleaned up after test.

    """
    model_path = TestModelFactory.create_onnx_model(
        n_features=10,
        n_outputs=1,
    )
    yield model_path
    # Cleanup
    if model_path.exists():
        model_path.unlink()
    meta_path = model_path.with_suffix(".onnx.meta")
    if meta_path.exists():
        meta_path.unlink()


@pytest.fixture
def dummy_xgboost_model() -> Path:
    """
    Create a minimal XGBoost model for testing.

    Returns path to a valid XGBoost JSON model file.

    """
    model_path = TestModelFactory.create_minimal_xgboost_model(
        n_features=10,
        model_type="classification",
    )
    yield model_path
    # Cleanup
    if model_path.exists():
        model_path.unlink()
    meta_path = model_path.with_suffix(".json.meta")
    if meta_path.exists():
        meta_path.unlink()


@pytest.fixture
def base_ml_config(
    default_bar_type: BarType,
    default_instrument_id: InstrumentId,
    dummy_onnx_model: Path,
) -> MLActorConfig:
    """
    Base ML actor configuration with all required fields.

    This fixture provides a complete, valid configuration that can be used directly or
    modified for specific test needs.

    """
    return MLActorConfig(
        model_id="test_model",
        model_path=str(dummy_onnx_model),
        bar_type=default_bar_type,
        instrument_id=default_instrument_id,
        batch_size=1,
        warm_up_period=10,
        prediction_threshold=0.5,
        use_dummy_stores=True,  # Use dummy stores for testing
    )


@pytest.fixture
def base_signal_config(
    default_bar_type: BarType,
    default_instrument_id: InstrumentId,
    dummy_onnx_model: Path,
    base_feature_config: MLFeatureConfig,
) -> MLSignalActorConfig:
    """
    Base ML signal actor configuration.
    """
    return MLSignalActorConfig(
        model_id="test_signal_model",
        model_path=str(dummy_onnx_model),
        bar_type=default_bar_type,
        instrument_id=default_instrument_id,
        feature_config=base_feature_config,
        batch_size=1,
        warm_up_period=10,
        prediction_threshold=0.5,
        use_dummy_stores=True,  # Use dummy stores for testing
        signal_strategy="threshold",
    )


# ============================================================================
# Message bus publisher (in-memory) for unit tests
# ============================================================================


class InMemoryPublisher:
    """
    Simple in-memory message publisher for unit tests.

    Stores published (topic, payload) tuples in-memory for later assertions.

    """

    def __init__(self) -> None:
        self.messages: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, payload: dict[str, Any]) -> None:  # noqa: D401 - simple interface
        self.messages.append((topic, payload))

    def clear(self) -> None:
        self.messages.clear()


@pytest.fixture
def in_memory_publisher() -> InMemoryPublisher:
    """
    Fixture providing a fresh in-memory publisher per test.
    """
    return InMemoryPublisher()


@pytest.fixture
def model_registry_config() -> ModelRegistryConfig:
    """
    Standard model registry configuration for testing.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        return ModelRegistryConfig(
            backend="json",
            backend_config={
                "storage_path": tmpdir,
            },
        )


# ============================================================================
# Mock Fixtures
# ============================================================================


@pytest.fixture
def mock_model_registry() -> MagicMock:
    """
    Fully configured mock model registry.

    Provides a registry mock with common model metadata pre-configured.

    """
    mock_registry = MagicMock()

    # Create mock model info
    mock_model_info = MagicMock()
    from ml.registry.base import ModelRole, DataRequirements
    import time

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
        performance_metrics={
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.93,
        },
        training_config={
            "n_estimators": 100,
            "max_depth": 5,
        },
    )
    mock_model_info.path = Path("/tmp/test_model.onnx")
    mock_model_info.metadata = {"test": True}

    # Configure registry methods
    mock_registry.get_model = MagicMock(return_value=mock_model_info)
    mock_registry.list_models = MagicMock(return_value=["test_model_v1"])
    mock_registry.register_model = MagicMock(return_value="test_model_v1")
    mock_registry.load_model = MagicMock(return_value=mock_model_info)

    return mock_registry


@pytest.fixture
def mock_feature_registry() -> MagicMock:
    """
    Fully configured mock feature registry.
    """
    mock_registry = MagicMock()

    # Create mock feature manifest
    from ml.registry.feature_registry import FeatureRole
    from ml.registry.base import DataRequirements
    import time

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

    # Configure registry methods
    mock_registry.get_feature_set = MagicMock(return_value=mock_feature_info)
    mock_registry.list_feature_sets = MagicMock(return_value=["test_features_v1"])
    mock_registry.register_feature_set = MagicMock(return_value="test_features_v1")

    return mock_registry


@pytest.fixture
def mock_data_store() -> MagicMock:
    """
    Fully configured mock data store.

    Provides a DataStore mock with common operations pre-configured.

    """
    mock_store = MagicMock()

    # Configure common methods
    mock_store.write = MagicMock(return_value=True)
    mock_store.read = MagicMock(return_value={"data": []})
    mock_store.query = MagicMock(return_value=[])
    mock_store.get_latest = MagicMock(return_value=None)
    mock_store.exists = MagicMock(return_value=False)

    # Configure typed read methods
    mock_store.read_features = MagicMock(return_value={})
    mock_store.read_predictions = MagicMock(return_value=[])
    mock_store.read_signals = MagicMock(return_value=[])

    mock_store.get_features_at_or_before = MagicMock(return_value=None)
    mock_store.get_latest_prediction_at_or_before = MagicMock(return_value=None)
    mock_store.get_latest_signal_at_or_before = MagicMock(return_value=None)

    return mock_store


@pytest.fixture
def mock_stores_bundle(
    mock_feature_store: MagicMock,
    mock_model_store: MagicMock,
    mock_strategy_store: MagicMock,
    mock_data_store: MagicMock,
) -> dict[str, MagicMock]:
    """
    Bundle of all store mocks for convenience.

    Returns a dictionary with all configured store mocks.

    """
    return {
        "feature_store": mock_feature_store,
        "model_store": mock_model_store,
        "strategy_store": mock_strategy_store,
        "data_store": mock_data_store,
    }


# ============================================================================
# Data Generation Fixtures
# ============================================================================


@pytest.fixture
def sample_features() -> dict[str, float]:
    """
    Sample feature dictionary for testing.
    """
    return {
        "sma_20": 1.0900,
        "rsi": 55.5,
        "volume_ratio": 1.2,
        "price_change": 0.002,
        "volatility": 0.015,
    }


@pytest.fixture
def sample_feature_array() -> np.ndarray:
    """
    Sample feature array for model inference.
    """
    rng = np.random.default_rng(42)
    return rng.standard_normal((1, 10)).astype(np.float32)


@pytest.fixture
def sample_predictions() -> np.ndarray:
    """
    Sample model predictions.
    """
    return np.array([0.65, -0.3, 0.8], dtype=np.float32)


@pytest.fixture
def test_timestamps() -> tuple[int, int]:
    """
    Standard test timestamps (ts_event, ts_init) in nanoseconds.
    """
    import time

    current_ns = int(time.time() * 1e9)
    return (current_ns, current_ns + 1000)


# ============================================================================
# Registry Manifest Fixtures
# ============================================================================


@pytest.fixture
def sample_model_manifest() -> ModelManifest:
    """
    Sample model manifest for registry testing.
    """
    from ml.registry.base import ModelRole, DataRequirements
    import time

    return ModelManifest(
        model_id="test_model",
        role=ModelRole.INFERENCE,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="xgboost",
        feature_schema={"feature_1": "float", "feature_2": "float"},
        feature_schema_hash="abc123",
        version="1.0.0",
        created_at=time.time(),
        last_modified=time.time(),
        performance_metrics={
            "accuracy": 0.95,
            "sharpe_ratio": 1.5,
        },
        training_config={
            "n_estimators": 100,
            "max_depth": 5,
        },
    )


@pytest.fixture
def sample_feature_manifest() -> FeatureManifest:
    """
    Sample feature manifest for registry testing.
    """
    from ml.registry.feature_registry import FeatureRole
    from ml.registry.base import DataRequirements
    import time

    return FeatureManifest(
        feature_set_id="test_features",
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
