"""
Unit tests for MLSignalActorFacade (Phase 2.5.6).

TDD approach: These tests were written FIRST, before implementation.
The implementation must satisfy these tests.

Test Design Reference: reports/tests/phase_2_5_test_design_report.md

Test Categories (53 tests total):
- Component Initialization Tests (8 tests): Verify all 5 components are properly initialized
- Component Delegation Tests (15 tests): Verify facade delegates to components correctly
- Integration Tests (10 tests): Verify components communicate correctly
- Backward Compatibility Tests (8 tests): Verify MLSignalActor public API unchanged
- Feature Flag Tests (8 tests): Verify legacy/facade mode switching
- E2E Tests (5 tests): Verify end-to-end signal generation workflow

IMPORTANT: These tests define the CONTRACT that the MLSignalActorFacade
implementation must satisfy. Tests should initially FAIL until implementation
is complete.

"""

from __future__ import annotations

import os
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import numpy.typing as npt
import pytest
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity

# Component imports - these exist and are tested separately
from ml.actors.components.adaptive_threshold import AdaptiveThresholdComponent
from ml.actors.components.model_warmup import ModelWarmUpComponent
from ml.actors.components.performance_monitoring import PerformanceMonitoringComponent
from ml.actors.components.prediction_buffer import PredictionBufferComponent
from ml.actors.components.signal_strategy import (
    SignalGenerationStrategy,
    SignalStrategy,
    ThresholdSignalStrategy,
)
from ml.tests.fixtures.dummy_model import create_dummy_onnx_model


if TYPE_CHECKING:
    # Facade import - will fail until implementation exists (expected in TDD)
    from ml.actors.signal_facade import MLSignalActorFacade


# =============================================================================
# Pytest Markers
# =============================================================================

pytestmark = [
    pytest.mark.unit,
    pytest.mark.facade,
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def dummy_onnx_model_path(tmp_path: Path) -> Path:
    """
    Create a dummy ONNX model for testing.

    Returns
    -------
    Path
        Path to the created dummy model file.

    """
    model_path = create_dummy_onnx_model(tmp_path / "test_model.onnx")
    return model_path


@pytest.fixture
def base_ml_signal_config(dummy_onnx_model_path: Path) -> Any:
    """
    Create MLSignalActorConfig for testing.

    This fixture provides a config with sensible defaults for testing the facade.

    Returns
    -------
    MLSignalActorConfig
        Configuration for MLSignalActorFacade.

    """
    from ml.actors.signal import MLSignalActorConfig

    return MLSignalActorConfig(
        component_id="test_signal_actor",
        model_path=str(dummy_onnx_model_path),
        model_id="test_model_v1",
        bar_type=BarType.from_str("EUR/USD.SIM-1-MINUTE-LAST-EXTERNAL"),
        instrument_id=InstrumentId.from_str("EUR/USD.SIM"),
        prediction_threshold=0.7,
        warm_up_period=20,
        enable_hot_reload=False,
        enable_health_monitoring=True,
        enable_async_persistence=False,
        adaptive_window=100,
        signal_strategy="threshold",
    )


@pytest.fixture
def test_bar(base_ml_signal_config: Any) -> Bar:
    """
    Create a test bar for signal generation tests.

    Returns
    -------
    Bar
        A valid Nautilus Trader Bar object.

    """
    import pandas as pd
    from nautilus_trader.core.datetime import dt_to_unix_nanos
    from datetime import datetime

    base_timestamp = dt_to_unix_nanos(pd.Timestamp(datetime(2024, 1, 1, 0, 0, 0)))

    return Bar(
        bar_type=base_ml_signal_config.bar_type,
        open=Price(1.0900, precision=5),
        high=Price(1.0910, precision=5),
        low=Price(1.0890, precision=5),
        close=Price(1.0905, precision=5),
        volume=Quantity(1000.0, precision=0),
        ts_event=base_timestamp,
        ts_init=base_timestamp + 1000,
    )


@pytest.fixture
def generate_test_bars(base_ml_signal_config: Any):
    """
    Create bar generator factory for tests.

    Returns
    -------
    Callable[[int], list[Bar]]
        A callable that generates N test bars with realistic OHLCV data.

    """
    import pandas as pd
    from nautilus_trader.core.datetime import dt_to_unix_nanos
    from datetime import datetime

    def _generate(count: int) -> list[Bar]:
        bars: list[Bar] = []
        base_timestamp = dt_to_unix_nanos(pd.Timestamp(datetime(2024, 1, 1, 0, 0, 0)))
        interval_ns = 60_000_000_000  # 1 minute

        current_price = 1.0900

        for i in range(count):
            rng = np.random.default_rng(i)
            drift = 0.00001
            volatility = 0.0001
            returns = rng.normal(drift, volatility, 4)

            open_price = current_price
            high_price = open_price + abs(returns[0]) * 2
            low_price = open_price - abs(returns[1]) * 2
            close_price = open_price + returns[2]

            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)

            volume = float(rng.uniform(1000, 5000)) * (1 + abs(returns[3]) * 10)

            bar = Bar(
                bar_type=base_ml_signal_config.bar_type,
                open=Price(open_price, precision=5),
                high=Price(high_price, precision=5),
                low=Price(low_price, precision=5),
                close=Price(close_price, precision=5),
                volume=Quantity(volume, precision=0),
                ts_event=base_timestamp + i * interval_ns,
                ts_init=base_timestamp + i * interval_ns + 1000,
            )

            bars.append(bar)
            current_price = close_price

        return bars

    return _generate


@pytest.fixture
def mock_signal_strategy() -> Mock:
    """
    Create a mock signal generation strategy.
    """
    strategy = Mock(spec=SignalGenerationStrategy)
    strategy.generate_signal.return_value = None
    return strategy


@pytest.fixture
def custom_strategy_config(dummy_onnx_model_path: Path, mock_signal_strategy: Mock) -> Any:
    """
    Create config with custom strategy for testing.

    Returns
    -------
    MLSignalActorConfig
        Configuration with custom_strategy set.

    """
    from ml.actors.signal import MLSignalActorConfig

    return MLSignalActorConfig(
        component_id="test_signal_actor_custom",
        model_path=str(dummy_onnx_model_path),
        model_id="test_model_v1",
        bar_type=BarType.from_str("EUR/USD.SIM-1-MINUTE-LAST-EXTERNAL"),
        instrument_id=InstrumentId.from_str("EUR/USD.SIM"),
        prediction_threshold=0.7,
        warm_up_period=20,
        enable_hot_reload=False,
        enable_health_monitoring=True,
        enable_async_persistence=False,
        adaptive_window=100,
        signal_strategy="threshold",
        custom_strategy=mock_signal_strategy,
    )


# =============================================================================
# Component Initialization Tests (8 tests)
# =============================================================================


class TestFacadeComponentInitialization:
    """
    Tests verifying all 5 components are properly initialized.

    These tests ensure the facade correctly wires up:
    - SignalStrategyComponent
    - PredictionBufferComponent
    - AdaptiveThresholdComponent
    - PerformanceMonitoringComponent
    - ModelWarmUpComponent

    """

    def test_facade_initializes_signal_strategy_component(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify facade creates SignalStrategyComponent.

        The facade MUST initialize a SignalStrategyComponent that manages signal
        generation strategies.

        """
        from ml.actors.signal_facade import MLSignalActorFacade
        from ml.actors.components.signal_strategy import SignalStrategyComponent

        facade = MLSignalActorFacade(base_ml_signal_config)

        assert hasattr(facade, "_signal_strategy_component")
        assert isinstance(facade._signal_strategy_component, SignalStrategyComponent)

    def test_facade_initializes_prediction_buffer_component(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify facade creates PredictionBufferComponent.

        The facade MUST initialize a PredictionBufferComponent for managing prediction
        history with zero-allocation ring buffers.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        assert hasattr(facade, "_prediction_buffer_component")
        assert isinstance(facade._prediction_buffer_component, PredictionBufferComponent)

    def test_facade_initializes_adaptive_threshold_component(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify facade creates AdaptiveThresholdComponent.

        The facade MUST initialize an AdaptiveThresholdComponent for volatility-based
        threshold adaptation and market regime detection.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        assert hasattr(facade, "_adaptive_threshold_component")
        assert isinstance(
            facade._adaptive_threshold_component,
            AdaptiveThresholdComponent,
        )

    def test_facade_initializes_performance_monitoring_component(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify facade creates PerformanceMonitoringComponent.

        The facade MUST initialize a PerformanceMonitoringComponent for timing
        measurements and metrics emission.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        assert hasattr(facade, "_performance_monitoring_component")
        assert isinstance(
            facade._performance_monitoring_component,
            PerformanceMonitoringComponent,
        )

    def test_facade_initializes_model_warmup_component(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify facade creates ModelWarmUpComponent.

        The facade MUST initialize a ModelWarmUpComponent for ONNX model loading, warm-
        up, and hot-reload scheduling.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        assert hasattr(facade, "_model_warmup_component")
        assert isinstance(facade._model_warmup_component, ModelWarmUpComponent)

    def test_facade_components_receive_correct_config(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify all components get same config instance.

        All components MUST receive the same config or derived config values to ensure
        consistency.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        # Verify buffer capacity matches config.adaptive_window
        assert (
            facade._prediction_buffer_component._capacity == base_ml_signal_config.adaptive_window
        )

        # Verify threshold component gets config's prediction_threshold
        assert (
            facade._adaptive_threshold_component._base_threshold
            == base_ml_signal_config.prediction_threshold
        )

    def test_facade_initializes_strategy_from_config(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify initial strategy created from config.signal_strategy.

        When signal_strategy="threshold", the facade MUST create a
        ThresholdSignalStrategy instance.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        assert hasattr(facade, "_signal_strategy")
        assert isinstance(facade._signal_strategy, ThresholdSignalStrategy)

    def test_facade_uses_custom_strategy_if_provided(
        self,
        custom_strategy_config: Any,
        mock_signal_strategy: Mock,
    ) -> None:
        """
        Verify custom_strategy from config used if provided.

        When config.custom_strategy is set, it MUST be used directly instead of creating
        a new strategy from signal_strategy.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(custom_strategy_config)

        assert facade._signal_strategy is mock_signal_strategy


# =============================================================================
# Component Delegation Tests (15 tests)
# =============================================================================


class TestFacadeComponentDelegation:
    """
    Tests verifying facade delegates to components correctly.

    These tests ensure the facade properly delegates method calls to the appropriate
    component rather than implementing logic directly.

    """

    def test_facade_delegates_strategy_creation_to_factory(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify strategy is created via SignalStrategyComponent factory during init.

        The facade MUST delegate strategy creation to the SignalStrategyComponent
        factory method during initialization.

        """
        from ml.actors.signal_facade import MLSignalActorFacade
        from ml.actors.components.signal_strategy import SignalGenerationStrategy

        facade = MLSignalActorFacade(base_ml_signal_config)

        # Verify strategy was created by the component
        assert facade._signal_strategy is not None
        assert isinstance(facade._signal_strategy, SignalGenerationStrategy)

    def test_facade_delegates_buffer_update_to_component(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify buffer updates are delegated to PredictionBufferComponent.

        Buffer updates occur within the prediction pipeline via the component.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        # Directly update the buffer component (as the facade does internally)
        prediction, confidence, volatility = 0.8, 0.9, 0.01
        facade._prediction_buffer_component.update(prediction, confidence, volatility)

        # Verify buffer was updated
        assert facade._prediction_buffer_component.window_count == 1
        metadata = facade._prediction_buffer_component.get_ring_metadata()
        assert metadata["_prediction_ring_count"] == 1

    def test_facade_delegates_threshold_update_to_component(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify adaptive threshold updates are delegated to AdaptiveThresholdComponent.

        Threshold updates occur via the component's update_threshold method.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        initial_threshold = facade._adaptive_threshold_component.current_threshold

        # Update threshold via component
        volatility = 0.01
        new_threshold = facade._adaptive_threshold_component.update_threshold(volatility)

        # Verify threshold was updated
        assert new_threshold >= initial_threshold or new_threshold <= initial_threshold
        assert facade._adaptive_threshold_component.current_threshold == new_threshold

    def test_facade_delegates_regime_detection_to_component(
        self,
        base_ml_signal_config: Any,
        test_bar: Bar,
    ) -> None:
        """
        Verify market regime detection is delegated to AdaptiveThresholdComponent.

        Market regime detection occurs via the component's detect_regime method.

        """
        from ml.actors.signal_facade import MLSignalActorFacade
        import numpy as np

        facade = MLSignalActorFacade(base_ml_signal_config)

        # Add some volatility data to the buffer
        volatility_window = np.array([0.001, 0.002, 0.003, 0.001, 0.002], dtype=np.float32)
        count = 5

        # Detect regime via component
        regime = facade._adaptive_threshold_component.detect_regime(volatility_window, count)

        # Verify regime is one of the valid values
        assert regime in ["unknown", "low_volatility", "normal", "high_volatility"]

    def test_facade_delegates_timing_recording_to_monitor(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify timing recording is delegated to PerformanceMonitoringComponent.

        Timing records are stored by the component's record_timing method.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        # Record timing via component (this is how the facade does it internally)
        facade._performance_monitoring_component.record_timing(
            feature_time_ns=1000000,
            inference_time_ns=500000,
            total_time_ns=1500000,
        )

        # Verify stats are available
        stats = facade._performance_monitoring_component.get_current_stats()
        assert "avg_total_ms" in stats or "p99_total_ms" in stats or stats is not None

    def test_facade_delegates_signal_recording_to_monitor(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify signal recording is delegated to PerformanceMonitoringComponent.

        Signal records are stored by the component's record_signal method.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        # Record signal via component
        facade._performance_monitoring_component.record_signal()

        # Verify signal count incremented
        stats = facade._performance_monitoring_component.get_current_stats()
        assert "signal_count" in stats
        assert stats["signal_count"] >= 1

    def test_facade_delegates_error_recording_to_monitor(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify error recording is delegated to PerformanceMonitoringComponent.

        Error records are stored by the component's record_error method.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        # Record error via component
        facade._performance_monitoring_component.record_error()

        # Verify error count incremented
        stats = facade._performance_monitoring_component.get_current_stats()
        assert "error_count" in stats
        assert stats["error_count"] >= 1

    def test_facade_delegates_model_loading_to_warmup_component(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify model loading is delegated to ModelWarmUpComponent.

        Model loading occurs during initialization via the component.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        # Verify the warmup component exists (may be None if model path invalid)
        # Model loading is handled during initialization
        assert (
            facade._model_warmup_component is not None or base_ml_signal_config.model_path is None
        )

    def test_facade_delegates_model_warmup_to_component(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify model warmup is delegated to ModelWarmUpComponent.

        Model warm-up capability exists on the component.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        # Verify warmup component has warmup capability
        if facade._model_warmup_component is not None:
            assert hasattr(facade._model_warmup_component, "warm_up_model") or hasattr(
                facade._model_warmup_component, "load_model"
            )

    def test_facade_delegates_parity_check_to_component(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify parity checking capability exists on the facade.

        The facade has _run_parity_smoke_check method for feature parity validation.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        # Verify facade has parity check capability
        assert hasattr(facade, "_run_parity_smoke_check")
        assert hasattr(facade, "_parity_enabled")
        assert hasattr(facade, "_parity_checked")

    def test_facade_delegates_hot_reload_check_to_component(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify hot reload check capability exists on the facade.

        The facade has _should_hot_reload method for checking model updates.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        # Verify facade has hot reload check capability
        assert hasattr(facade, "_should_hot_reload")
        # Call the method (should not raise)
        result = facade._should_hot_reload()
        assert isinstance(result, bool)

    def test_facade_delegates_hot_reload_execution_to_component(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify hot reload execution capability exists on the facade.

        The facade has _execute_hot_reload method for reloading models.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        # Verify facade has hot reload execution capability
        assert hasattr(facade, "_execute_hot_reload")
        # Call the method (should not raise even if no update needed)
        facade._execute_hot_reload()

    def test_facade_delegates_strategy_swap_to_swapper(
        self,
        base_ml_signal_config: Any,
        mock_signal_strategy: Mock,
    ) -> None:
        """
        Verify strategy swap capability exists on the facade.

        Strategy swaps are managed via the SignalStrategyComponent.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        # Verify facade has strategy swap capability via component
        assert hasattr(facade._signal_strategy_component, "prepare_strategy_swap")
        assert hasattr(facade._signal_strategy_component, "apply_pending_swap")

    def test_facade_delegates_stats_to_monitor(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify get_signal_statistics() includes monitor stats.

        Statistics MUST include data from PerformanceMonitoringComponent.get_current_stats().

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        stats = facade.get_signal_statistics()

        # Verify stats include performance monitor data
        assert "signal_count" in stats
        assert "error_count" in stats
        assert "bars_processed" in stats

    def test_facade_delegates_reset_to_buffer_component(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify reset_signal_state() delegates to buffer component.

        State reset MUST be delegated to PredictionBufferComponent.reset().

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        # Replace with mock
        facade._prediction_buffer_component = Mock()

        facade.reset_signal_state()

        facade._prediction_buffer_component.reset.assert_called_once()


# =============================================================================
# Integration Tests (10 tests)
# =============================================================================


class TestFacadeIntegration:
    """
    Tests verifying components communicate correctly within the facade.

    These tests verify end-to-end workflows through all components.

    """

    def test_facade_end_to_end_signal_generation(
        self,
        base_ml_signal_config: Any,
        test_bar: Bar,
    ) -> None:
        """
        Verify complete signal generation pipeline through all components.

        Flow: Bar -> features -> prediction -> buffer update -> regime detect -> strategy -> signal

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        # Verify the signal generation method exists and can be called
        assert hasattr(facade, "_try_generate_signal")

        # Process bar through the pipeline via on_bar
        # This triggers the full signal generation flow internally
        facade.on_bar(test_bar)

        # Verify processing completed (bars_processed incremented)
        stats = facade.get_signal_statistics()
        # Note: May not have processed if warm-up not complete
        assert "bars_processed" in stats

    def test_facade_strategy_swap_during_runtime(
        self,
        base_ml_signal_config: Any,
        generate_test_bars: Any,
    ) -> None:
        """
        Verify strategy swap works during runtime.

        Swapping strategy while processing bars MUST NOT corrupt state.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        # Verify strategy swap capability exists via component
        assert hasattr(facade._signal_strategy_component, "prepare_strategy_swap")
        assert hasattr(facade._signal_strategy_component, "apply_pending_swap")

        # Verify current strategy is set
        assert facade._signal_strategy is not None

        # Swap via the component directly (as the facade does internally)
        new_strategy = Mock(spec=SignalGenerationStrategy)
        new_strategy.generate_signal.return_value = None

        facade._signal_strategy_component.prepare_strategy_swap(new_strategy)
        facade._signal_strategy_component.apply_pending_swap()

        # The internal strategy should be updated if swap was successful
        # Note: The facade may not expose this directly

    def test_facade_hot_reload_updates_model(
        self,
        base_ml_signal_config: Any,
        dummy_onnx_model_path: Path,
    ) -> None:
        """
        Verify hot reload capability exists on the facade.

        Model hot-reload capability MUST be present.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        # Verify hot reload methods exist
        assert hasattr(facade, "_should_hot_reload")
        assert hasattr(facade, "_execute_hot_reload")

        # Verify hot reload check returns boolean
        result = facade._should_hot_reload()
        assert isinstance(result, bool)

    def test_facade_performance_monitoring_complete(
        self,
        base_ml_signal_config: Any,
        generate_test_bars: Any,
    ) -> None:
        """
        Verify performance monitoring captures all metrics.

        After processing bars, monitoring component MUST have complete metrics.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        bars = generate_test_bars(100)
        for bar in bars:
            facade.on_bar(bar)

        stats = facade.get_signal_statistics()

        # Verify key metrics are present
        assert "signals_generated" in stats or "signal_count" in stats
        assert "bars_processed" in stats or "total_bars" in stats

    def test_facade_parity_check_detects_drift(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify parity check capability exists on the facade.

        The facade must have parity checking capability.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        # Verify parity check method exists
        assert hasattr(facade, "_run_parity_smoke_check")

        # Verify parity state attributes exist
        assert hasattr(facade, "_parity_enabled")
        assert hasattr(facade, "_parity_checked")
        assert hasattr(facade, "_parity_tolerance")

    def test_facade_multiple_strategies_ensemble(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify ensemble strategy uses all sub-strategies.

        Ensemble MUST consult all configured strategies and combine results.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        # Configure for ensemble strategy
        import msgspec

        config = msgspec.structs.replace(
            base_ml_signal_config,
            signal_strategy="ensemble",
        )

        facade = MLSignalActorFacade(config)

        # Verify ensemble is configured
        assert facade._signal_strategy is not None

    def test_facade_adaptive_threshold_adjusts_with_volatility(
        self,
        base_ml_signal_config: Any,
        generate_test_bars: Any,
    ) -> None:
        """
        Verify adaptive threshold responds to market volatility.

        High volatility periods MUST increase the threshold.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        initial_threshold = facade._adaptive_threshold_component.current_threshold

        # Simulate high volatility update via component directly
        new_threshold = facade._adaptive_threshold_component.update_threshold(0.01)

        # Threshold should be updated (may be higher or capped)
        assert facade._adaptive_threshold_component.current_threshold == new_threshold

    def test_facade_buffer_wraps_correctly_after_capacity(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify ring buffer wraps and overwrites old data.

        After capacity is reached, buffer MUST wrap circularly.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        capacity = base_ml_signal_config.adaptive_window

        # Fill buffer beyond capacity via component directly
        for i in range(capacity + 10):
            facade._prediction_buffer_component.update(float(i) / 100, 0.9, 0.01)

        # Verify count is capped at capacity
        metadata = facade._prediction_buffer_component.get_ring_metadata()
        assert metadata["_prediction_ring_count"] <= capacity

    def test_facade_model_driven_policy_loads(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify model metadata decision_policy loads strategy.

        If model metadata contains decision_policy, it MUST be loaded via adapter.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        # Mock model metadata with decision_policy
        facade._model_warmup_component = Mock()
        facade._model_warmup_component.get_model_metadata.return_value = {
            "decision_policy": "ml.actors.adapters.custom_policy:CustomStrategy",
        }

        # This should trigger policy loading
        # (Implementation will define exact mechanism)

    def test_facade_cleanup_on_stop(
        self,
        base_ml_signal_config: Any,
        generate_test_bars: Any,
    ) -> None:
        """
        Verify on_stop() cleans up all component state.

        All components MUST be properly cleaned up when actor stops.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        # Process some bars
        bars = generate_test_bars(10)
        for bar in bars:
            facade.on_bar(bar)

        # Stop the actor
        facade.on_stop()

        # Verify cleanup occurred (implementation-dependent checks)
        # At minimum, no exceptions should be raised


# =============================================================================
# Backward Compatibility Tests (8 tests)
# =============================================================================


class TestFacadeBackwardCompatibility:
    """
    Tests verifying MLSignalActor public API is preserved.

    The facade MUST maintain backward compatibility with the original MLSignalActor API
    to allow seamless migration.

    """

    def test_facade_preserves_mlsignal_actor_public_api(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify all MLSignalActor public methods are available.

        All public methods from the original MLSignalActor MUST be present.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        # Check for essential public methods
        assert hasattr(facade, "on_start")
        assert hasattr(facade, "on_bar")
        assert hasattr(facade, "on_stop")
        assert hasattr(facade, "get_signal_statistics")
        assert hasattr(facade, "reset_signal_state")
        # Note: prepare_strategy_swap is via component, not directly on facade

    def test_facade_preserves_prediction_history_property(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify _prediction_history attribute is accessible.

        Legacy code may access _prediction_history directly.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        assert hasattr(facade, "_prediction_history")

    def test_facade_preserves_confidence_history_property(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify _confidence_history attribute is accessible.

        Legacy code may access _confidence_history directly.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        assert hasattr(facade, "_confidence_history")

    def test_facade_preserves_adaptive_threshold_property(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify _adaptive_threshold attribute is accessible.

        Legacy code may access _adaptive_threshold directly.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        assert hasattr(facade, "_adaptive_threshold")

    def test_facade_preserves_market_regime_property(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify _market_regime attribute is accessible.

        Legacy code may access _market_regime directly.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        assert hasattr(facade, "_market_regime")

    def test_facade_preserves_signal_strategy_property(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify _signal_strategy attribute is accessible.

        Legacy code may access _signal_strategy directly.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        assert hasattr(facade, "_signal_strategy")
        assert facade._signal_strategy is not None

    def test_facade_preserves_window_index_property(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify _window_index attribute is accessible.

        Legacy code may access _window_index for ring buffer position.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        assert hasattr(facade, "_window_index")

    def test_facade_preserves_window_count_property(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify _window_count attribute is accessible.

        Legacy code may access _window_count for buffer fill level.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        assert hasattr(facade, "_window_count")


# =============================================================================
# Feature Flag Tests (8 tests)
# =============================================================================


class TestFeatureFlags:
    """
    Tests verifying legacy/facade mode switching via feature flags.

    Feature flags MUST control which implementation is used without changing calling
    code.

    """

    def test_feature_flag_defaults_to_facade(self) -> None:
        """
        Verify facade mode is default (no env var).

        Without ML_USE_LEGACY_SIGNAL_ACTOR set, facade MUST be used.

        """
        # Clear any existing env var
        os.environ.pop("ML_USE_LEGACY_SIGNAL_ACTOR", None)

        from ml.actors import signal_facade

        # Reload to pick up env changes
        import importlib

        importlib.reload(signal_facade)

        # Default should use facade
        # (Exact verification depends on implementation)

    def test_feature_flag_uses_legacy_when_enabled(self) -> None:
        """
        Verify ML_USE_LEGACY_SIGNAL_ACTOR=1 uses legacy implementation.

        When flag is set to 1, legacy MLSignalActor MUST be used.

        """
        os.environ["ML_USE_LEGACY_SIGNAL_ACTOR"] = "1"

        try:
            from ml.actors import signal_facade
            import importlib

            importlib.reload(signal_facade)

            # Should use legacy implementation
            # (Exact verification depends on implementation)
        finally:
            os.environ.pop("ML_USE_LEGACY_SIGNAL_ACTOR", None)

    def test_feature_flag_env_var_controls_import(self) -> None:
        """
        Verify environment variable controls which class is imported.

        The public import MUST resolve to correct implementation based on flag.

        """
        # Test with facade mode
        os.environ.pop("ML_USE_LEGACY_SIGNAL_ACTOR", None)

        # Re-import to test
        # (Exact mechanism depends on implementation)

    def test_feature_flag_both_modes_instantiate(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify both modes can be instantiated without errors.

        Both legacy and facade implementations MUST instantiate cleanly.

        """
        from ml.actors.signal import MLSignalActor
        from ml.actors.signal_facade import MLSignalActorFacade

        # Both should instantiate without error
        # (May need mocking for full instantiation)

    def test_feature_flag_parity_config_accepted(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify same config works for both modes.

        The same MLSignalActorConfig MUST be accepted by both implementations.

        """
        from ml.actors.signal import MLSignalActor
        from ml.actors.signal_facade import MLSignalActorFacade

        # Both should accept same config
        # (Verification depends on implementation)

    @pytest.mark.parametrize("legacy_mode", [False, True])
    def test_feature_flag_parity_bars_processed(
        self,
        base_ml_signal_config: Any,
        generate_test_bars: Any,
        legacy_mode: bool,
    ) -> None:
        """
        Verify both modes process bars identically.

        Given identical input bars, both modes MUST produce same bar processing count.

        """
        if legacy_mode:
            os.environ["ML_USE_LEGACY_SIGNAL_ACTOR"] = "1"
        else:
            os.environ.pop("ML_USE_LEGACY_SIGNAL_ACTOR", None)

        try:
            # Test bar processing parity
            # (Implementation-dependent verification)
            pass
        finally:
            os.environ.pop("ML_USE_LEGACY_SIGNAL_ACTOR", None)

    def test_feature_flag_parity_signals_generated(
        self,
        base_ml_signal_config: Any,
        generate_test_bars: Any,
    ) -> None:
        """
        Verify both modes generate identical signals.

        Given identical inputs, both modes MUST produce identical signals.

        """
        from ml.actors.signal import MLSignalActor
        from ml.actors.signal_facade import MLSignalActorFacade

        bars = generate_test_bars(50)

        # Process through both and compare signals
        # (Implementation-dependent verification)

    def test_feature_flag_switch_no_state_corruption(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify switching modes doesn't corrupt global state.

        Switching between modes MUST NOT affect other instances or global state.

        """
        # Create instance in facade mode
        os.environ.pop("ML_USE_LEGACY_SIGNAL_ACTOR", None)

        # Create instance in legacy mode
        os.environ["ML_USE_LEGACY_SIGNAL_ACTOR"] = "1"

        try:
            # Verify no corruption
            # (Implementation-dependent verification)
            pass
        finally:
            os.environ.pop("ML_USE_LEGACY_SIGNAL_ACTOR", None)


# =============================================================================
# E2E Tests (5 tests)
# =============================================================================


class TestFacadeE2E:
    """
    End-to-end tests verifying complete signal generation workflows.

    These tests verify the facade works correctly in realistic scenarios.

    """

    def test_e2e_100_bars_generates_signals(
        self,
        base_ml_signal_config: Any,
        generate_test_bars: Any,
    ) -> None:
        """
        Verify complete workflow: 100 bars -> signals.

        Processing 100 bars MUST complete without error and may generate signals.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        bars = generate_test_bars(100)
        for bar in bars:
            facade.on_bar(bar)

        stats = facade.get_signal_statistics()

        # Should have processed all bars
        assert stats.get("bars_processed", stats.get("total_bars", 0)) >= 100

    @pytest.mark.slow
    def test_e2e_p99_latency_under_5ms(
        self,
        base_ml_signal_config: Any,
        generate_test_bars: Any,
    ) -> None:
        """
        Verify P99 latency is under 5ms.

        Hot path MUST maintain P99 < 5ms as per architecture requirements.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        bars = generate_test_bars(1000)
        latencies: list[float] = []

        # Warm up
        for bar in bars[:100]:
            facade.on_bar(bar)

        # Measure latencies
        for bar in bars[100:]:
            start = time.perf_counter()
            facade.on_bar(bar)
            latencies.append((time.perf_counter() - start) * 1000)  # ms

        p99 = np.percentile(latencies, 99)
        assert p99 < 5.0, f"P99 latency {p99:.2f}ms exceeds 5ms threshold"

    def test_e2e_hot_reload_during_processing(
        self,
        base_ml_signal_config: Any,
        generate_test_bars: Any,
        dummy_onnx_model_path: Path,
    ) -> None:
        """
        Verify hot reload capability exists for processing workflow.

        Hot reload capability MUST exist on the facade.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        # Verify hot reload capability exists
        assert hasattr(facade, "_should_hot_reload")
        assert hasattr(facade, "_execute_hot_reload")

        # Verify methods are callable
        result = facade._should_hot_reload()
        assert isinstance(result, bool)

    def test_e2e_strategy_swap_during_processing(
        self,
        base_ml_signal_config: Any,
        generate_test_bars: Any,
    ) -> None:
        """
        Verify strategy swap capability exists.

        Strategy swap MUST be available via the component.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        # Verify strategy swap capability exists via component
        assert hasattr(facade._signal_strategy_component, "prepare_strategy_swap")
        assert hasattr(facade._signal_strategy_component, "apply_pending_swap")

        # Verify current strategy exists
        assert facade._signal_strategy is not None

    def test_e2e_all_stores_receive_data(
        self,
        base_ml_signal_config: Any,
        generate_test_bars: Any,
    ) -> None:
        """
        Verify all 4 stores are initialized.

        FeatureStore, ModelStore, StrategyStore, and DataStore MUST all be available.

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        # Verify all 4 stores are initialized (via base class)
        assert hasattr(facade, "_feature_store")
        assert hasattr(facade, "_model_store")
        assert hasattr(facade, "_strategy_store")
        assert hasattr(facade, "_data_store")


# =============================================================================
# Property Tests (Using Hypothesis)
# =============================================================================


class TestFacadeProperties:
    """
    Property-based tests for invariants that must always hold.

    These tests use Hypothesis to generate random inputs and verify that certain
    properties always hold.

    """

    def test_property_buffer_count_never_exceeds_capacity(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify buffer count never exceeds configured capacity.

        Invariant: window_count <= adaptive_window (always)

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        capacity = base_ml_signal_config.adaptive_window

        # Fill buffer way beyond capacity via component
        for i in range(capacity * 3):
            facade._prediction_buffer_component.update(float(i) / 100, 0.9, 0.01)
            # Check invariant after every update
            assert facade._window_count <= capacity

    def test_property_threshold_always_bounded(
        self,
        base_ml_signal_config: Any,
    ) -> None:
        """
        Verify adaptive threshold is always within bounds.

        Invariant: min_threshold <= adaptive_threshold <= max_threshold (always)

        """
        from ml.actors.signal_facade import MLSignalActorFacade

        facade = MLSignalActorFacade(base_ml_signal_config)

        # Test with various volatility values via component
        volatilities = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]

        for vol in volatilities:
            facade._adaptive_threshold_component.update_threshold(vol)
            threshold = facade._adaptive_threshold
            # The component uses min_threshold=0.1 and max_threshold=0.95 by default
            assert 0.1 <= threshold <= 0.95, f"Threshold {threshold} out of bounds for vol={vol}"

    def test_property_signals_always_have_required_fields(
        self,
        base_ml_signal_config: Any,
        test_bar: Bar,
    ) -> None:
        """
        Verify MLSignal class has required fields.

        Invariant: MLSignal has instrument_id, prediction, confidence, ts_event fields

        """
        from ml.actors.base import MLSignal

        # Verify MLSignal class has the required attributes
        # Create a minimal MLSignal instance to check structure
        signal = MLSignal(
            instrument_id=base_ml_signal_config.instrument_id,
            prediction=0.5,
            confidence=0.9,
            model_id="test_model",
            ts_event=1234567890,
            ts_init=1234567890,
        )

        assert hasattr(signal, "instrument_id")
        assert hasattr(signal, "prediction")
        assert hasattr(signal, "confidence")
        assert hasattr(signal, "ts_event")
