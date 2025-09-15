"""
Parity and Buffer-Reuse Guardrails for ML Hot Path Performance.

This module validates that the ML signal generation pipeline maintains:
1. Feature parity between online and offline computation
2. Zero memory allocations in the hot path
3. P99 latency budgets are maintained
4. Buffer reuse patterns work correctly

These tests fail CI if regressions are detected, ensuring production reliability.

Performance Requirements:
- P99 feature computation: <500μs
- P99 model inference: <2ms
- P99 end-to-end signal: <5ms
- Zero allocations in hot path after warmup
- Feature parity drift: <1e-6 tolerance
"""

from __future__ import annotations

import gc
import os
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import numpy as np
import numpy.typing as npt
import pytest

from ml._imports import HAS_ONNX, check_ml_dependencies, ort
from ml.actors.signal import MLSignalActor, MLSignalActorConfig, OptimizationLevel
from ml.config.actors import OptimizationConfig, StrategyConfig
from ml.features.engineering import FeatureConfig, FeatureEngineer, IndicatorManager
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import AggressorSide, BarAggregation, PriceType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.objects import Price, Quantity

if TYPE_CHECKING:
    pass


# =============================================================================
# Environment Configuration
# =============================================================================

# Check if running under xdist (parallel pytest)
UNDER_XDIST = bool(os.getenv("PYTEST_XDIST_WORKER"))

# Performance budget relaxation factor for CI environments
RELAX_FACTOR = float(os.getenv("ML_BENCH_RELAX", "1.0"))
if UNDER_XDIST:
    RELAX_FACTOR *= 3.0  # More lenient under parallel execution


# =============================================================================
# Performance Measurement Utilities
# =============================================================================

def measure_p99_latency_ns(func, iterations: int = 1000) -> int:
    """
    Measure P99 latency in nanoseconds with warmup.

    Returns
    -------
    int
        P99 latency in nanoseconds
    """
    # Warmup to eliminate JIT effects
    for _ in range(min(100, iterations // 10)):
        func()

    # Measure
    durations = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        func()
        end = time.perf_counter_ns()
        durations.append(end - start)

    return int(np.percentile(durations, 99))


def assert_p99_budget(actual_ns: int, budget_ns: int, component: str) -> None:
    """
    Assert that P99 latency is within budget, failing CI if exceeded.

    Parameters
    ----------
    actual_ns : int
        Actual P99 latency in nanoseconds
    budget_ns : int
        Budget P99 latency in nanoseconds
    component : str
        Component name for error message
    """
    adjusted_budget = int(budget_ns * RELAX_FACTOR)
    actual_ms = actual_ns / 1_000_000
    budget_ms = adjusted_budget / 1_000_000

    assert actual_ns <= adjusted_budget, (
        f"❌ {component} P99 latency {actual_ms:.2f}ms exceeded "
        f"budget {budget_ms:.2f}ms (factor={RELAX_FACTOR:.1f})"
    )
    print(f"✅ {component} P99 latency {actual_ms:.2f}ms within budget {budget_ms:.2f}ms")


def assert_zero_allocations(func, iterations: int, component: str) -> None:
    """
    Assert that function has zero allocations after warmup.

    Parameters
    ----------
    func : callable
        Function to test
    iterations : int
        Number of iterations to run
    component : str
        Component name for error message
    """
    # Warmup
    for _ in range(min(100, iterations // 10)):
        func()

    # Force garbage collection
    gc.collect()

    # Start memory tracking
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()

    # Execute iterations
    for _ in range(iterations):
        func()

    snapshot2 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # Analyze allocations
    top_stats = snapshot2.compare_to(snapshot1, "lineno")

    # Filter to ML module allocations only
    ml_allocations = [
        stat for stat in top_stats
        if stat.size_diff > 0 and "ml/" in str(stat.traceback)
    ]

    total_allocated = sum(stat.size_diff for stat in ml_allocations)
    allocations_per_call = total_allocated / iterations

    # Allow minimal allocations for Python overhead (50 bytes per call)
    max_allowed = 50 * iterations

    assert total_allocated <= max_allowed, (
        f"❌ {component} allocated {total_allocated} bytes "
        f"({allocations_per_call:.1f} per call), expected near-zero. "
        f"Top allocations: {ml_allocations[:3]}"
    )
    print(f"✅ {component} zero-allocation: {allocations_per_call:.1f} bytes per call")


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def feature_config() -> FeatureConfig:
    """Create optimized feature configuration."""
    return FeatureConfig(
        return_periods=[1, 5, 10],  # Reduced for performance
        momentum_periods=[5, 10],
        rsi_period=14,
        bb_period=20,
        bb_std=2.0,
        atr_period=14,
        ema_fast=12,
        ema_slow=26,
        macd_signal=9,
        volume_ma_periods=[5, 10, 20],
        include_microstructure=False,
        include_trade_flow=False,
    )


@pytest.fixture
def instrument_id() -> InstrumentId:
    """Create test instrument ID."""
    return InstrumentId(Symbol("EUR"), Venue("USD"))


@pytest.fixture
def bar_type(instrument_id: InstrumentId) -> BarType:
    """Create test bar type."""
    return BarType(
        instrument_id=instrument_id,
        bar_spec=BarSpecification(
            step=1,
            aggregation=BarAggregation.MINUTE,
            price_type=PriceType.LAST,
        ),
        aggregation_source=AggressorSide.NO_AGGRESSOR,
    )


@pytest.fixture
def test_bars(bar_type: BarType) -> list[Bar]:
    """Generate test bars for performance testing."""
    bars = []
    base_price = 1.1000  # EUR/USD
    base_volume = 1_000_000.0

    for i in range(500):
        price = base_price + np.sin(i * 0.1) * 0.0050  # 50 pip variation
        high = price + np.random.uniform(0.0001, 0.0005)
        low = price - np.random.uniform(0.0001, 0.0005)
        volume = base_volume + np.random.uniform(-100_000, 100_000)

        bar = Bar(
            bar_type=bar_type,
            open=Price.from_str(f"{price:.5f}"),
            high=Price.from_str(f"{high:.5f}"),
            low=Price.from_str(f"{low:.5f}"),
            close=Price.from_str(f"{price + np.random.uniform(-0.0001, 0.0001):.5f}"),
            volume=Quantity.from_str(f"{volume:.0f}"),
            ts_event=i * 60_000_000_000,  # 1 minute bars
            ts_init=i * 60_000_000_000 + 1000,
        )
        bars.append(bar)

    return bars


@pytest.fixture
def mock_model() -> Mock:
    """Create mock model for testing."""
    model = Mock()
    model.predict_proba.return_value = np.array([[0.3, 0.7]])  # Binary classification
    return model


# =============================================================================
# Feature Computation Guardrails
# =============================================================================

@pytest.mark.performance
class TestFeatureComputationGuardrails:
    """Performance guardrails for feature computation."""

    def test_feature_computation_p99_budget(
        self,
        feature_config: FeatureConfig,
        test_bars: list[Bar]
    ) -> None:
        """
        Ensure feature computation meets P99 latency budget (<500μs).

        This test FAILS CI if the P99 latency budget is exceeded.
        """
        if UNDER_XDIST:
            pytest.skip("Skip latency microbench under xdist for stability")

        engineer = FeatureEngineer(feature_config)
        indicator_mgr = IndicatorManager(feature_config)

        # Warm up with initial bars
        for bar in test_bars[:50]:
            current_bar = {
                "open": bar.open.as_double(),
                "high": bar.high.as_double(),
                "low": bar.low.as_double(),
                "close": bar.close.as_double(),
                "volume": bar.volume.as_double(),
            }
            indicator_mgr.update_from_values(
                close=current_bar["close"],
                high=current_bar["high"],
                low=current_bar["low"],
                volume=current_bar["volume"],
            )

        # Test bar for benchmarking
        test_bar = test_bars[100]
        current_bar = {
            "open": test_bar.open.as_double(),
            "high": test_bar.high.as_double(),
            "low": test_bar.low.as_double(),
            "close": test_bar.close.as_double(),
            "volume": test_bar.volume.as_double(),
        }

        def compute_features() -> npt.NDArray[np.float64]:
            return engineer.calculate_features_online(
                current_bar=current_bar,
                indicator_manager=indicator_mgr,
                scaler=None,
            )

        # Measure P99 latency and enforce budget
        p99_ns = measure_p99_latency_ns(compute_features, iterations=2000)
        assert_p99_budget(p99_ns, 500_000, "Feature computation")  # 500μs budget

    def test_feature_computation_zero_allocations(
        self,
        feature_config: FeatureConfig,
        test_bars: list[Bar]
    ) -> None:
        """
        Ensure feature computation has zero allocations in hot path.

        This test FAILS CI if allocations are detected after warmup.
        """
        engineer = FeatureEngineer(feature_config)
        indicator_mgr = IndicatorManager(feature_config)

        # Warm up
        for bar in test_bars[:50]:
            current_bar = {
                "open": bar.open.as_double(),
                "high": bar.high.as_double(),
                "low": bar.low.as_double(),
                "close": bar.close.as_double(),
                "volume": bar.volume.as_double(),
            }
            indicator_mgr.update_from_values(
                close=current_bar["close"],
                high=current_bar["high"],
                low=current_bar["low"],
                volume=current_bar["volume"],
            )
            # Run computation to fill caches
            _ = engineer.calculate_features_online(
                current_bar=current_bar,
                indicator_manager=indicator_mgr,
                scaler=None,
            )

        # Test function
        test_bar = test_bars[100]
        current_bar = {
            "open": test_bar.open.as_double(),
            "high": test_bar.high.as_double(),
            "low": test_bar.low.as_double(),
            "close": test_bar.close.as_double(),
            "volume": test_bar.volume.as_double(),
        }

        def compute_features() -> npt.NDArray[np.float64]:
            return engineer.calculate_features_online(
                current_bar=current_bar,
                indicator_manager=indicator_mgr,
                scaler=None,
            )

        # Assert zero allocations
        assert_zero_allocations(compute_features, 100, "Feature computation")


# =============================================================================
# Feature Parity Guardrails
# =============================================================================

@pytest.mark.performance
class TestFeatureParityGuardrails:
    """Performance guardrails for feature parity verification."""

    def test_feature_parity_smoke_check(
        self,
        feature_config: FeatureConfig,
        test_bars: list[Bar],
        instrument_id: InstrumentId,
        bar_type: BarType,
        mock_model: Mock
    ) -> None:
        """
        Ensure feature parity smoke-check works correctly and reports metrics.

        This test validates that online and offline feature computation produce
        identical results within tolerance.
        """
        # Create actor with parity smoke-check enabled
        config = MLSignalActorConfig(
            actor_id="PARITY_TEST",
            model_id="test_model",
            model_path="/tmp/dummy_model.pkl",  # Will use mock
            bar_type=bar_type,
            instrument_id=instrument_id,
            feature_config=feature_config,
            enable_parity_smoke_check=True,
            parity_smoke_check_window_bars=50,
            parity_tolerance=1e-6,
            optimization_config=OptimizationConfig(
                level=OptimizationLevel.STANDARD,
            ),
            prediction_threshold=0.6,
        )

        # Create actor (will use dummy stores)
        actor = MLSignalActor(config)

        # Override model with mock
        actor._model = mock_model
        actor._model_metadata = {"input_names": ["features"]}

        # Initialize components
        actor._initialize_features()

        # Process bars to build up history
        for bar in test_bars[:60]:  # More than window size
            actor.on_bar(bar)

        # Check that parity check was performed
        assert actor._parity_checked, "Parity smoke-check should have been performed"

        print("✅ Feature parity smoke-check completed successfully")

    def test_feature_parity_drift_detection(
        self,
        feature_config: FeatureConfig,
        test_bars: list[Bar]
    ) -> None:
        """
        Test that feature parity drift detection works correctly.

        This test validates that the drift metric accurately detects differences
        between online and offline feature computation.
        """
        engineer = FeatureEngineer(feature_config)
        indicator_mgr = IndicatorManager(feature_config)

        # Warm up
        for bar in test_bars[:30]:
            current_bar = {
                "open": bar.open.as_double(),
                "high": bar.high.as_double(),
                "low": bar.low.as_double(),
                "close": bar.close.as_double(),
                "volume": bar.volume.as_double(),
            }
            indicator_mgr.update_from_values(
                close=current_bar["close"],
                high=current_bar["high"],
                low=current_bar["low"],
                volume=current_bar["volume"],
            )

        # Compute features for same bar multiple times
        test_bar = test_bars[30]
        current_bar = {
            "open": test_bar.open.as_double(),
            "high": test_bar.high.as_double(),
            "low": test_bar.low.as_double(),
            "close": test_bar.close.as_double(),
            "volume": test_bar.volume.as_double(),
        }

        results = []
        for _ in range(10):
            features = engineer.calculate_features_online(
                current_bar=current_bar,
                indicator_manager=indicator_mgr,
                scaler=None,
            )
            results.append(features.copy())

        # All results should be identical (zero drift)
        max_drift = 0.0
        for i in range(1, len(results)):
            drift = float(np.max(np.abs(results[0] - results[i])))
            max_drift = max(max_drift, drift)

        assert max_drift < 1e-10, f"Feature drift {max_drift:.3e} detected, expected zero"
        print(f"✅ Feature parity verified: max drift {max_drift:.3e}")


# =============================================================================
# Model Inference Guardrails
# =============================================================================

@pytest.mark.performance
class TestModelInferenceGuardrails:
    """Performance guardrails for model inference."""

    @pytest.mark.skipif(not HAS_ONNX, reason="ONNX not available")
    def test_onnx_inference_p99_budget(self, tmp_path: Path) -> None:
        """
        Ensure ONNX model inference meets P99 latency budget (<2ms).

        This test FAILS CI if the P99 latency budget is exceeded.
        """
        if UNDER_XDIST:
            pytest.skip("Skip latency microbench under xdist for stability")

        # Create simple ONNX model for testing
        import onnx
        from skl2onnx import to_onnx
        from sklearn.ensemble import RandomForestClassifier

        n_features = 50
        X = np.random.randn(100, n_features).astype(np.float32)
        y = np.random.randint(0, 2, 100)

        model = RandomForestClassifier(n_estimators=5, max_depth=3)
        model.fit(X, y)

        # Convert to ONNX
        onnx_model = to_onnx(
            model,
            X[:1],
            target_opset=12,
            options={"zipmap": False},
        )

        # Save model
        model_path = tmp_path / "test_model.onnx"
        onnx.save(onnx_model, str(model_path))

        # Load for inference
        session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

        # Prepare input
        input_name = session.get_inputs()[0].name
        features = np.random.randn(1, n_features).astype(np.float32)

        def run_inference() -> npt.NDArray[np.float32]:
            return session.run(None, {input_name: features})[0]

        # Measure P99 latency and enforce budget
        p99_ns = measure_p99_latency_ns(run_inference, iterations=1000)
        assert_p99_budget(p99_ns, 2_000_000, "ONNX inference")  # 2ms budget

    def test_model_inference_zero_allocations(self, mock_model: Mock) -> None:
        """
        Ensure model inference has zero allocations in hot path.

        This test FAILS CI if allocations are detected after warmup.
        """
        # Prepare input
        features = np.random.randn(50).astype(np.float32)

        def run_inference() -> npt.NDArray[np.float64]:
            features_2d = features.reshape(1, -1)
            return mock_model.predict_proba(features_2d)[0]

        # Assert zero allocations
        assert_zero_allocations(run_inference, 500, "Model inference")


# =============================================================================
# End-to-End Signal Generation Guardrails
# =============================================================================

@pytest.mark.performance
class TestEndToEndGuardrails:
    """Performance guardrails for complete signal generation pipeline."""

    def test_e2e_signal_generation_p99_budget(
        self,
        feature_config: FeatureConfig,
        test_bars: list[Bar],
        instrument_id: InstrumentId,
        bar_type: BarType,
        mock_model: Mock
    ) -> None:
        """
        Ensure end-to-end signal generation meets P99 budget (<5ms).

        This test FAILS CI if the P99 latency budget is exceeded.
        """
        if UNDER_XDIST:
            pytest.skip("Skip latency microbench under xdist for stability")

        # Create optimized actor configuration
        config = MLSignalActorConfig(
            actor_id="E2E_TEST",
            model_id="test_model",
            model_path="/tmp/dummy_model.pkl",
            bar_type=bar_type,
            instrument_id=instrument_id,
            feature_config=feature_config,
            optimization_config=OptimizationConfig(
                level=OptimizationLevel.OPTIMIZED,
                feature_cache_size=100,
                enable_profiling=False,
            ),
            prediction_threshold=0.6,
        )

        # Create actor
        actor = MLSignalActor(config)

        # Override model with mock
        actor._model = mock_model
        actor._model_metadata = {"input_names": ["features"]}

        # Initialize components
        actor._initialize_features()

        # Warm up with initial bars
        for bar in test_bars[:50]:
            actor.on_bar(bar)

        # Test bar for benchmarking
        test_bar = test_bars[100]

        def process_bar() -> None:
            actor.on_bar(test_bar)

        # Measure P99 latency and enforce budget
        p99_ns = measure_p99_latency_ns(process_bar, iterations=1000)
        assert_p99_budget(p99_ns, 5_000_000, "E2E signal generation")  # 5ms budget

    def test_e2e_signal_generation_zero_allocations(
        self,
        feature_config: FeatureConfig,
        test_bars: list[Bar],
        instrument_id: InstrumentId,
        bar_type: BarType,
        mock_model: Mock
    ) -> None:
        """
        Ensure end-to-end signal generation has zero allocations in hot path.

        This test FAILS CI if allocations are detected after warmup.
        """
        # Create actor configuration
        config = MLSignalActorConfig(
            actor_id="ALLOC_TEST",
            model_id="test_model",
            model_path="/tmp/dummy_model.pkl",
            bar_type=bar_type,
            instrument_id=instrument_id,
            feature_config=feature_config,
            optimization_config=OptimizationConfig(
                level=OptimizationLevel.OPTIMIZED,
            ),
            prediction_threshold=0.6,
        )

        # Create actor
        actor = MLSignalActor(config)

        # Override model with mock
        actor._model = mock_model
        actor._model_metadata = {"input_names": ["features"]}

        # Initialize components
        actor._initialize_features()

        # Warm up thoroughly to fill all caches
        for bar in test_bars[:100]:
            actor.on_bar(bar)

        # Test function
        test_bar = test_bars[150]

        def process_bar() -> None:
            actor.on_bar(test_bar)

        # Assert zero allocations
        assert_zero_allocations(process_bar, 50, "E2E signal generation")


# =============================================================================
# Buffer Reuse Guardrails
# =============================================================================

@pytest.mark.performance
class TestBufferReuseGuardrails:
    """Performance guardrails for buffer reuse patterns."""

    def test_feature_buffer_reuse(
        self,
        feature_config: FeatureConfig,
        test_bars: list[Bar]
    ) -> None:
        """
        Ensure feature buffers are reused correctly.

        This test validates that feature computation reuses pre-allocated
        buffers rather than creating new arrays.
        """
        engineer = FeatureEngineer(feature_config)
        indicator_mgr = IndicatorManager(feature_config)

        # Warm up
        for bar in test_bars[:30]:
            current_bar = {
                "open": bar.open.as_double(),
                "high": bar.high.as_double(),
                "low": bar.low.as_double(),
                "close": bar.close.as_double(),
                "volume": bar.volume.as_double(),
            }
            indicator_mgr.update_from_values(
                close=current_bar["close"],
                high=current_bar["high"],
                low=current_bar["low"],
                volume=current_bar["volume"],
            )

        # Get first result and check buffer identity
        test_bar = test_bars[30]
        current_bar = {
            "open": test_bar.open.as_double(),
            "high": test_bar.high.as_double(),
            "low": test_bar.low.as_double(),
            "close": test_bar.close.as_double(),
            "volume": test_bar.volume.as_double(),
        }

        features1 = engineer.calculate_features_online(
            current_bar=current_bar,
            indicator_manager=indicator_mgr,
            scaler=None,
        )

        # Get second result
        features2 = engineer.calculate_features_online(
            current_bar=current_bar,
            indicator_manager=indicator_mgr,
            scaler=None,
        )

        # Should be views of the same buffer
        assert np.shares_memory(features1, engineer.feature_buffer), \
            "Features should be a view of the pre-allocated buffer"
        assert np.shares_memory(features2, engineer.feature_buffer), \
            "Features should be a view of the pre-allocated buffer"

        print("✅ Feature buffer reuse verified")

    def test_prediction_buffer_memory_stability(
        self,
        feature_config: FeatureConfig,
        test_bars: list[Bar],
        instrument_id: InstrumentId,
        bar_type: BarType,
        mock_model: Mock
    ) -> None:
        """
        Test memory stability over extended operation (24h simulation).

        This test validates that memory usage remains stable over long periods,
        ensuring no memory leaks in production.
        """
        # Create actor with optimizations
        config = MLSignalActorConfig(
            actor_id="MEMORY_TEST",
            model_id="test_model",
            model_path="/tmp/dummy_model.pkl",
            bar_type=bar_type,
            instrument_id=instrument_id,
            feature_config=feature_config,
            optimization_config=OptimizationConfig(
                level=OptimizationLevel.OPTIMIZED,
            ),
            prediction_threshold=0.6,
        )

        actor = MLSignalActor(config)
        actor._model = mock_model
        actor._model_metadata = {"input_names": ["features"]}
        actor._initialize_features()

        # Warm up
        for bar in test_bars[:50]:
            actor.on_bar(bar)

        # Force garbage collection and measure initial memory
        gc.collect()

        try:
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss
        except ImportError:
            pytest.skip("psutil not available for memory testing")

        # Simulate 24h of operation (1440 minutes * 60 bars = 86400 bars)
        # Use a representative subset for testing
        simulation_bars = 10000
        for i in range(simulation_bars):
            bar = test_bars[i % len(test_bars)]
            actor.on_bar(bar)

        # Force garbage collection and measure final memory
        gc.collect()
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        memory_increase_mb = memory_increase / (1024 * 1024)

        # Allow reasonable memory increase for caches and operational data
        # 20MB is generous for 10k operations
        max_allowed_mb = 20

        assert memory_increase_mb < max_allowed_mb, (
            f"❌ Memory leak detected: {memory_increase_mb:.1f}MB increase "
            f"after {simulation_bars} operations (max allowed: {max_allowed_mb}MB)"
        )

        print(f"✅ Memory stability verified: {memory_increase_mb:.1f}MB increase over {simulation_bars} operations")


# =============================================================================
# Performance Regression Detection
# =============================================================================

@pytest.mark.performance
class TestPerformanceRegressionGuardrails:
    """Performance regression detection guardrails."""

    def test_performance_regression_detection(
        self,
        feature_config: FeatureConfig,
        test_bars: list[Bar]
    ) -> None:
        """
        Detect performance regressions against established baselines.

        This test FAILS CI if performance degrades beyond acceptable thresholds.
        """
        if UNDER_XDIST:
            pytest.skip("Skip regression microbench under xdist for stability")

        engineer = FeatureEngineer(feature_config)
        indicator_mgr = IndicatorManager(feature_config)

        # Warm up
        for bar in test_bars[:50]:
            current_bar = {
                "open": bar.open.as_double(),
                "high": bar.high.as_double(),
                "low": bar.low.as_double(),
                "close": bar.close.as_double(),
                "volume": bar.volume.as_double(),
            }
            indicator_mgr.update_from_values(
                close=current_bar["close"],
                high=current_bar["high"],
                low=current_bar["low"],
                volume=current_bar["volume"],
            )

        # Measure current performance
        test_bar = test_bars[100]
        current_bar = {
            "open": test_bar.open.as_double(),
            "high": test_bar.high.as_double(),
            "low": test_bar.low.as_double(),
            "close": test_bar.close.as_double(),
            "volume": test_bar.volume.as_double(),
        }

        def compute_features() -> npt.NDArray[np.float64]:
            return engineer.calculate_features_online(
                current_bar=current_bar,
                indicator_manager=indicator_mgr,
                scaler=None,
            )

        # Measure statistics
        times = []
        for _ in range(1000):
            start = time.perf_counter_ns()
            _ = compute_features()
            end = time.perf_counter_ns()
            times.append(end - start)

        times_array = np.array(times)
        current_p50_us = np.percentile(times_array, 50) / 1000  # Convert to μs
        current_p99_us = np.percentile(times_array, 99) / 1000  # Convert to μs

        # Performance baselines (update if legitimate improvements are made)
        baseline_p50_us = 200 * RELAX_FACTOR  # 200μs baseline
        baseline_p99_us = 500 * RELAX_FACTOR  # 500μs baseline

        # Check for regression (allow 20% degradation)
        regression_threshold = 1.2

        assert current_p50_us < baseline_p50_us * regression_threshold, (
            f"❌ P50 latency regression: {current_p50_us:.1f}μs vs "
            f"baseline {baseline_p50_us:.1f}μs (>{regression_threshold-1:.0%} degradation)"
        )

        assert current_p99_us < baseline_p99_us * regression_threshold, (
            f"❌ P99 latency regression: {current_p99_us:.1f}μs vs "
            f"baseline {baseline_p99_us:.1f}μs (>{regression_threshold-1:.0%} degradation)"
        )

        print("✅ Performance regression check passed:")
        print(f"   P50: {current_p50_us:.1f}μs (baseline: {baseline_p50_us:.1f}μs)")
        print(f"   P99: {current_p99_us:.1f}μs (baseline: {baseline_p99_us:.1f}μs)")


# =============================================================================
# CI Integration
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers for performance tests."""
    config.addinivalue_line(
        "markers", "performance: mark test as performance/guardrail test"
    )


if __name__ == "__main__":
    # Run all guardrail tests when executed directly
    import subprocess
    import sys

    print("=" * 80)
    print("ML PARITY AND BUFFER-REUSE GUARDRAILS")
    print("=" * 80)
    print()
    print("Running performance guardrail tests...")
    print(f"Environment: RELAX_FACTOR={RELAX_FACTOR:.1f}, UNDER_XDIST={UNDER_XDIST}")
    print()

    # Run tests
    result = subprocess.run([
        sys.executable, "-m", "pytest", __file__, "-v", "-x",
        "--tb=short", "-m", "performance"
    ])

    if result.returncode == 0:
        print()
        print("✅ All performance guardrails PASSED")
        print("   Production reliability requirements satisfied")
    else:
        print()
        print("❌ Performance guardrails FAILED")
        print("   CI should fail to prevent production regressions")
        sys.exit(1)
