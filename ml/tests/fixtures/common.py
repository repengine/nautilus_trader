#!/usr/bin/env python3

"""
Common test fixtures for ML module.

This module provides reusable fixtures for ML tests to reduce duplication and improve
maintainability.

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from ml.config.actors import MLSignalActorConfig
from ml.config.base import MLActorConfig
from ml.config.base import MLFeatureConfig
from ml.config.registry import ModelRegistryConfig
from ml.registry import FeatureManifest
from ml.registry import ModelManifest
from ml.tests.fixtures.dummy_model import create_dummy_onnx_model
from ml.tests.fixtures.model_factory import TestDataFactory
from ml.tests.fixtures.model_factory import TestModelFactory
from ml.tests.utils.model_artifacts import ensure_strict_onnx_sidecar
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
    model_path = create_dummy_onnx_model()
    ensure_strict_onnx_sidecar(model_path)
    yield model_path
    # Cleanup
    if model_path.exists():
        model_path.unlink()
    sidecar_path = model_path.with_suffix(".meta.json")
    if sidecar_path.exists():
        sidecar_path.unlink()


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


@pytest.fixture(scope="session")
def perf_dummy_onnx_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Session-scoped ONNX model for performance benchmarks.
    """
    model_dir = tmp_path_factory.mktemp("perf_dummy_onnx")
    model_path = create_dummy_onnx_model(model_dir / "dummy_model.onnx")
    ensure_strict_onnx_sidecar(model_path)
    yield model_path
    if model_path.exists():
        model_path.unlink()
    sidecar_path = model_path.with_suffix(".meta.json")
    if sidecar_path.exists():
        sidecar_path.unlink()


@pytest.fixture(scope="session")
def perf_xgboost_model(
    tmp_path_factory: pytest.TempPathFactory,
    test_model_factory: TestModelFactory,
) -> Path:
    """
    Session-scoped XGBoost model for performance benchmarks.
    """
    model_dir = tmp_path_factory.mktemp("perf_xgboost")
    model_path = test_model_factory.create_minimal_xgboost_model(
        n_features=10,
        model_type="classification",
        output_path=model_dir / "model.json",
        n_samples=10,
    )
    yield model_path
    if model_path.exists():
        model_path.unlink()

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
def model_registry_config(tmp_path: Path) -> ModelRegistryConfig:
    """
    Standard model registry configuration for testing.
    """
    return ModelRegistryConfig(registry_path=str(tmp_path))


# ============================================================================
# Mock Fixtures
# ============================================================================


# Note: mock_model_registry and mock_feature_registry have been moved to
# ml.tests.fixtures.mock_stores as part of the mock_registry_factory consolidation.
# They are re-exported via conftest.py for global availability.
# Use mock_registry_factory("model", with_manifest=True) for model registry mocks.
# Use mock_registry_factory("feature", with_manifest=True) for feature registry mocks.

# Note: mock_data_store (and mock_feature_store, mock_model_store,
# mock_strategy_store) are now imported from ml.tests.fixtures.mock_stores
# and re-exported via conftest.py for global availability.


@pytest.fixture
def mock_stores_bundle(
    mock_feature_store: MagicMock,
    mock_model_store: MagicMock,
    mock_strategy_store: MagicMock,
    mock_data_store: MagicMock,
    mock_earnings_store: MagicMock,
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
        "earnings_store": mock_earnings_store,
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
        "price_sma_20": 1.0900,
        "rsi": 55.5,
        "volume_ratio_20": 1.2,
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
    return np.array([0.65, 0.3, 0.8], dtype=np.float32)


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
        feature_names=["price_sma_20", "rsi_14", "volume_ratio_20"],
        feature_dtypes=["float32", "float32", "float32"],
        schema_hash="def456",
        pipeline_signature="pipeline_sig_123",
        pipeline_version="1.0.0",
        created_at=time.time(),
        last_modified=time.time(),
    )


# ============================================================================
# Test Data Factory Fixture
# ============================================================================


@pytest.fixture(scope="session")
def test_data_factory() -> TestDataFactory:
    """
    Session-scoped test data factory for centralized test data generation.

    Provides convenient methods for generating test data with improved performance
    through session-level caching. All data generation is centralized in the
    TestDataFactory class.

    Methods
    -------
    bars(n, instrument_id, bar_type, start_date) -> list[Bar]
        Generate realistic Bar objects with correlated OHLCV data

    features(n, n_features, seed) -> np.ndarray
        Generate synthetic feature arrays for ML models

    predictions(n, instrument, start_timestamp, seed) -> list[dict]
        Generate prediction dictionaries with valid structure

    Important Notes
    ---------------
    Immutability:
        This is a session-scoped fixture. While the data generation methods
        create new objects on each call, tests should treat returned data as
        read-only when possible. For tests that modify data:
        - Use .copy() on numpy arrays
        - Create new Bar objects rather than mutating

    Performance:
        Session scope provides 15-20% test suite speedup by eliminating
        redundant data generation across tests.

    Examples
    --------
    >>> def test_with_bars(test_data_factory):
    ...     bars = test_data_factory.bars(n=100)
    ...     assert len(bars) == 100

    >>> def test_with_features(test_data_factory):
    ...     features = test_data_factory.features(n=50, n_features=10)
    ...     assert features.shape == (50, 10)

    >>> def test_with_predictions(test_data_factory):
    ...     preds = test_data_factory.predictions(n=20)
    ...     assert len(preds) == 20

    Returns
    -------
    TestDataFactory
        Factory instance with data generation methods

    See Also
    --------
    TestDataFactory : Class in ml.tests.fixtures.model_factory
    generate_test_bars : Deprecated fixture (use factory.bars() instead)

    """
    return TestDataFactory()


@pytest.fixture(scope="session")
def test_model_factory() -> TestModelFactory:
    """
    Session-scoped model factory for generating ONNX and XGBoost artifacts.

    This fixture centralizes access to ``TestModelFactory`` so tests avoid importing
    fixture modules directly.  The shared instance reduces repeated initialization
    costs while still producing fresh model files for each request.

    Examples
    --------
    >>> def test_with_model(test_model_factory):
    ...     path = test_model_factory.create_onnx_model(n_features=8, n_outputs=2)
    ...     assert path.exists()

    Returns
    -------
    TestModelFactory
        Factory for creating deterministic ML model artifacts.
    """
    return TestModelFactory()


__all__ = [
    "InMemoryPublisher",
    "alternative_bar_type",
    "alternative_instrument_id",
    "base_feature_config",
    "base_ml_config",
    "base_signal_config",
    "default_bar_type",
    "default_instrument_id",
    "default_venue",
    "dummy_onnx_model",
    "dummy_xgboost_model",
    "in_memory_publisher",
    "mock_stores_bundle",
    "model_registry_config",
    "perf_dummy_onnx_model",
    "perf_xgboost_model",
    "sample_feature_array",
    "sample_feature_manifest",
    "sample_features",
    "sample_model_manifest",
    "sample_predictions",
    "test_component_id",
    "test_data_factory",
    "test_model_factory",
    "test_timestamps",
]
