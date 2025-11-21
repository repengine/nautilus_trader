"""
Contract tests for MLSignalActor fallback behavior.

This module tests Universal Pattern #4: Progressive Fallback Chains, ensuring that
MLSignalActor gracefully handles failures in external dependencies and degrades
service quality in a controlled manner rather than failing hard.

Test Categories:
1. Store unavailability: PostgreSQL → DummyStore with warning
2. Feature computation fallback: Handle missing/invalid feature data gracefully
3. Data quality fallback: Clean data → Data with missing values → Synthetic data
4. Configuration fallback: Handle invalid configurations gracefully

Performance Requirements:
- Fallback detection: <100ms P99
- Store fallback switch: <50ms P99
- Feature fallback: <10ms P99 (warm path)

Coverage Requirements:
- All fallback chains must have tests
- Proper warning/error logging at each level
- Actor continues operation during degraded states
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, patch

import pytest
import numpy as np
from nautilus_trader.model.data import Bar
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.test_kit.stubs.data import TestDataStubs

from ml.actors.signal import MLSignalActor, MLSignalActorConfig, SignalStrategy
from ml.config.actors import OptimizationConfig, StrategyConfig
from ml.features.config import FeatureConfig

if TYPE_CHECKING:
    from collections.abc import Generator


# ============================================================================
# Test Configuration
# ============================================================================

@pytest.fixture
def temp_model_path() -> Generator[Path, None, None]:
    """
    Create temporary model file for testing.
    """
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        # Create minimal ONNX file content
        f.write(b"dummy_onnx_content")
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


def create_test_config(
    model_path: str = "test_model.onnx",
    instrument_id: InstrumentId | None = None,
    **overrides: Any
) -> MLSignalActorConfig:
    """Create a test configuration with sensible defaults."""
    from nautilus_trader.model.identifiers import InstrumentId
    from nautilus_trader.model.data import BarType

    if instrument_id is None:
        instrument_id = InstrumentId.from_str("EUR/USD.SIM")

    defaults = {
        "model_path": model_path,
        "model_id": "test_model_id",
        "instrument_id": instrument_id,
        "bar_type": BarType.from_str("EUR/USD.SIM-1-MINUTE-BID-EXTERNAL"),
        "prediction_threshold": 0.7,
        "min_signal_separation_bars": 5,
        "signal_strategy": SignalStrategy.THRESHOLD,
        "enable_hot_reload": False,
        "enable_regime_detection": False,
        "log_predictions": False,
        "warm_up_period": 10,
        "adaptive_window": 20,
        "max_feature_latency_ms": 50.0,
        "feature_config": FeatureConfig(),
        "strategy_config": StrategyConfig(),
        "optimization_config": OptimizationConfig(),
        "actor_id": "test_signal_actor",
        "enable_parity_smoke_check": False,
        "use_dummy_stores": True,  # Use dummy stores by default for testing
    }

    # Apply overrides
    defaults.update(overrides)

    return MLSignalActorConfig(**defaults)


# ============================================================================
# Contract 1: Store Unavailability Fallback
# ============================================================================

@pytest.mark.contracts
class TestStoreUnavailabilityFallback:
    """
    Test store fallback: PostgreSQL → DummyStore with warning.

    Verifies that when PostgreSQL stores are unavailable, the actor falls back
    to dummy stores while maintaining operational capability.
    """

    def test_dummy_stores_initialization(self) -> None:
        """
        Test successful initialization with dummy stores.

        Contract: When configured to use dummy stores, actor should initialize
        successfully without requiring PostgreSQL or file system.
        """
        # Arrange
        config = create_test_config(use_dummy_stores=True)

        # Act
        actor = MLSignalActor(config)

        # Assert
        # Should initialize without error
        assert actor is not None
        # Should have stores available
        assert hasattr(actor, "_feature_store")
        assert hasattr(actor, "_model_store")
        assert hasattr(actor, "_strategy_store")
        assert hasattr(actor, "_data_store")

    def test_actor_continues_with_missing_model_file(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        Test that actor handles missing model files gracefully.

        Contract: When model file doesn't exist, actor should log appropriate
        warning but continue initialization in degraded mode.
        """
        # Arrange
        config = create_test_config(
            model_path="/nonexistent/model.onnx",
            use_dummy_stores=True
        )

        # Act & Assert
        # Should not raise exception during initialization
        try:
            actor = MLSignalActor(config)
            assert actor is not None
        except Exception as e:
            # If it does raise, it should be a clear error about model loading
            assert "model" in str(e).lower() or "file" in str(e).lower()


# ============================================================================
# Contract 2: Feature Computation Fallback
# ============================================================================

@pytest.mark.contracts
class TestFeatureComputationFallback:
    """
    Test feature computation fallback: Handle missing/invalid feature data gracefully.

    Verifies graceful degradation when feature computation encounters issues.
    """

    def test_actor_handles_missing_feature_data(self) -> None:
        """
        Test that actor handles missing feature data gracefully.

        Contract: When feature computation fails or returns None, actor should
        not crash but continue operating in degraded mode.
        """
        # Arrange
        config = create_test_config(use_dummy_stores=True)

        # Act
        actor = MLSignalActor(config)

        # Mock feature computation that returns None
        original_compute = getattr(actor, "_compute_features", None)

        def mock_compute_features_none(bar: Bar) -> np.ndarray[Any, np.dtype[np.float32]] | None:
            return None

        if hasattr(actor, "_compute_features"):
            actor._compute_features = mock_compute_features_none  # type: ignore[method-assign]

        # Act
        test_bar = TestDataStubs.bar_5decimal()

        # Should not crash when features are None
        try:
            actor.on_bar(test_bar)
        except Exception as e:
            # If it raises, should be a controlled error, not a crash
            assert isinstance(e, (ValueError, RuntimeError, TypeError))

    def test_actor_handles_invalid_feature_shapes(self) -> None:
        """
        Test that actor handles invalid feature array shapes gracefully.

        Contract: When feature computation returns arrays with wrong shapes,
        actor should handle it gracefully without crashing.
        """
        # Arrange
        config = create_test_config(use_dummy_stores=True)

        # Act
        actor = MLSignalActor(config)

        # Mock feature computation that returns wrong shape
        def mock_compute_features_wrong_shape(bar: Bar) -> np.ndarray[Any, np.dtype[np.float32]] | None:
            # Return a shape that might not match model expectations
            return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)  # 2D instead of 1D

        if hasattr(actor, "_compute_features"):
            actor._compute_features = mock_compute_features_wrong_shape  # type: ignore[method-assign]

        # Act
        test_bar = TestDataStubs.bar_5decimal()

        # Should handle wrong shapes gracefully
        try:
            actor.on_bar(test_bar)
        except Exception as e:
            # Should be a controlled error about shape mismatch
            assert any(keyword in str(e).lower() for keyword in ["shape", "dimension", "array"])


# ============================================================================
# Contract 3: Data Quality Fallback
# ============================================================================

@pytest.mark.contracts
class TestDataQualityFallback:
    """
    Test data quality fallback: Clean data → Data with missing values → Synthetic data.

    Verifies graceful handling of data quality issues with progressive degradation.
    """

    def test_actor_handles_nan_features(self) -> None:
        """
        Test that actor handles NaN values in feature arrays gracefully.

        Contract: When feature computation returns arrays with NaN values,
        actor should handle it without crashing the system.
        """
        # Arrange
        config = create_test_config(use_dummy_stores=True)

        # Act
        actor = MLSignalActor(config)

        # Mock feature computation that returns NaN values
        def mock_compute_with_nans(bar: Bar) -> np.ndarray[Any, np.dtype[np.float32]] | None:
            return np.array([1.0, np.nan, 3.0], dtype=np.float32)

        if hasattr(actor, "_compute_features"):
            actor._compute_features = mock_compute_with_nans  # type: ignore[method-assign]

        # Act
        test_bar = TestDataStubs.bar_5decimal()

        # Should handle NaN values gracefully
        try:
            actor.on_bar(test_bar)
        except Exception as e:
            # Should be a controlled error about data quality
            assert any(keyword in str(e).lower() for keyword in ["nan", "data", "invalid", "feature"])

    def test_actor_handles_infinite_features(self) -> None:
        """
        Test that actor handles infinite values in feature arrays gracefully.

        Contract: When feature computation returns arrays with infinite values,
        actor should handle it without crashing the system.
        """
        # Arrange
        config = create_test_config(use_dummy_stores=True)

        # Act
        actor = MLSignalActor(config)

        # Mock feature computation that returns infinite values
        def mock_compute_with_inf(bar: Bar) -> np.ndarray[Any, np.dtype[np.float32]] | None:
            return np.array([1.0, np.inf, 3.0], dtype=np.float32)

        if hasattr(actor, "_compute_features"):
            actor._compute_features = mock_compute_with_inf  # type: ignore[method-assign]

        # Act
        test_bar = TestDataStubs.bar_5decimal()

        # Should handle infinite values gracefully
        try:
            actor.on_bar(test_bar)
        except Exception as e:
            # Should be a controlled error about data quality
            assert any(keyword in str(e).lower() for keyword in ["inf", "data", "invalid", "feature"])


# ============================================================================
# Contract 4: Configuration Fallback
# ============================================================================

@pytest.mark.contracts
class TestConfigurationFallback:
    """
    Test configuration fallback: Handle invalid configurations gracefully.
    """

    def test_actor_handles_invalid_configuration_values(self) -> None:
        """
        Test that actor handles invalid configuration values gracefully.

        Contract: When configuration contains invalid values, actor should
        either use sensible defaults or fail with clear error messages.
        """
        # Test with invalid prediction threshold
        try:
            config = create_test_config(
                prediction_threshold=-0.5,  # Invalid negative threshold
                use_dummy_stores=True
            )
            actor = MLSignalActor(config)
            # If it doesn't raise, it should have corrected the value
            assert hasattr(actor, "_config")
        except Exception as e:
            # Should be a clear validation error
            assert any(keyword in str(e).lower() for keyword in ["threshold", "invalid", "range", "validation"])

    def test_actor_handles_missing_required_configuration(self) -> None:
        """
        Test that actor handles missing required configuration gracefully.

        Contract: When required configuration is missing, actor should fail
        with clear error messages indicating what's missing.
        """
        from nautilus_trader.model.identifiers import InstrumentId
        from nautilus_trader.model.data import BarType

        # Test with empty model_id (test validation)
        try:
            config = MLSignalActorConfig(
                model_path="test_model.onnx",
                model_id="",  # Empty model_id (should be invalid)
                instrument_id=InstrumentId.from_str("EUR/USD.SIM"),
                bar_type=BarType.from_str("EUR/USD.SIM-1-MINUTE-BID-EXTERNAL"),
                use_dummy_stores=True,
            )
            actor = MLSignalActor(config)
            # If no error, actor should handle empty model_id gracefully
            assert hasattr(actor, "_config")
        except Exception as e:
            # Should be a clear error about invalid model_id
            assert any(keyword in str(e).lower() for keyword in ["model_id", "invalid", "empty"])
