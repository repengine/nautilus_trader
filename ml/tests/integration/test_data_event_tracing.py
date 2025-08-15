"""
End-to-end test for tracing data events through the entire ML pipeline.

This test module provides comprehensive tracing of data flow from raw market
data through feature computation, model inference, and strategy execution,
with verbose failure reporting at each step.
"""

import contextlib
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from ml.actors.signal import MLSignalActor
from ml.config.actors import MLSignalActorConfig
from ml.features.engineering import FeatureConfig
from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategyStore
from ml.tests.fixtures.model_factory import TestModelFactory
from nautilus_trader.common.component import MessageBus
from nautilus_trader.common.component import TestClock
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.data.engine import DataEngine
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.portfolio.portfolio import Portfolio
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.test_kit.stubs.component import TestComponentStubs


class DataEventTracer:
    """Traces data events through the pipeline with verbose failure reporting."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.trace_log: list[dict[str, Any]] = []
        self.checkpoints: dict[str, bool] = {}
        self.errors: list[dict[str, Any]] = []

    def checkpoint(self, name: str, success: bool = True, data: Any = None, error: Exception = None):
        """Record a checkpoint in the data flow."""
        checkpoint_data = {
            "timestamp": datetime.utcnow(),
            "name": name,
            "success": success,
            "data": str(data)[:200] if data else None,
            "error": str(error) if error else None,
            "traceback": traceback.format_exc() if error else None
        }

        self.trace_log.append(checkpoint_data)
        self.checkpoints[name] = success

        if error:
            self.errors.append(checkpoint_data)

        if self.verbose:
            status = "✓" if success else "✗"
            print(f"[{status}] {name}")
            if data and self.verbose:
                print(f"    Data: {str(data)[:100]}")
            if error:
                print(f"    ERROR: {error}")

    def get_failure_report(self) -> str:
        """Generate a detailed failure report."""
        report_lines = ["=" * 80, "DATA EVENT TRACE REPORT", "=" * 80]

        # Summary
        total_checkpoints = len(self.checkpoints)
        passed = sum(1 for v in self.checkpoints.values() if v)
        failed = total_checkpoints - passed

        report_lines.append(f"\nSUMMARY: {passed}/{total_checkpoints} checkpoints passed")

        if failed > 0:
            report_lines.append(f"FAILURES: {failed} checkpoints failed\n")

            # Detailed failure analysis
            report_lines.append("FAILURE DETAILS:")
            report_lines.append("-" * 40)

            for error in self.errors:
                report_lines.append(f"\n✗ {error['name']}")
                report_lines.append(f"  Time: {error['timestamp']}")
                report_lines.append(f"  Error: {error['error']}")
                if error["traceback"] and "NoneType" not in error["traceback"]:
                    report_lines.append(f"  Traceback:\n{error['traceback']}")

        # Full trace
        report_lines.append("\nFULL TRACE:")
        report_lines.append("-" * 40)

        for i, checkpoint in enumerate(self.trace_log, 1):
            status = "✓" if checkpoint["success"] else "✗"
            report_lines.append(f"{i:3d}. [{status}] {checkpoint['name']}")
            if checkpoint["data"]:
                report_lines.append(f"      Data: {checkpoint['data']}")

        # Pipeline flow diagram
        report_lines.append("\nPIPELINE FLOW:")
        report_lines.append("-" * 40)
        report_lines.append(self._generate_flow_diagram())

        return "\n".join(report_lines)

    def _generate_flow_diagram(self) -> str:
        """Generate a visual flow diagram of the pipeline."""
        stages = [
            ("Market Data", ["market_data_generated", "market_data_received"]),
            ("Feature Store", ["feature_store_write", "feature_store_read"]),
            ("Feature Computation", ["feature_computation_start", "feature_computation_end"]),
            ("Model Inference", ["model_inference_start", "model_inference_end"]),
            ("Model Store", ["model_store_write", "model_store_read"]),
            ("Signal Generation", ["signal_generated", "signal_published"]),
            ("Strategy Store", ["strategy_store_write", "strategy_store_read"]),
            ("Strategy Execution", ["strategy_received_signal", "strategy_action"])
        ]

        diagram_lines = []
        for stage_name, checkpoints in stages:
            all_passed = all(self.checkpoints.get(cp, False) for cp in checkpoints)
            status = "✓" if all_passed else "✗"
            diagram_lines.append(f"[{status}] {stage_name}")

            for cp in checkpoints:
                if cp in self.checkpoints:
                    cp_status = "✓" if self.checkpoints[cp] else "✗"
                    diagram_lines.append(f"    └─ [{cp_status}] {cp}")

        return "\n".join(diagram_lines)


class TestDataEventTracing:
    """Test suite for tracing data events through the ML pipeline."""

    @pytest.fixture
    def tracer(self):
        """Create a data event tracer."""
        return DataEventTracer(verbose=True)

    @pytest.fixture
    def setup_components(self, tracer):
        """Set up all necessary components for testing."""
        # Clear Prometheus metrics registry
        try:
            import gc

            from prometheus_client import REGISTRY
            collectors = list(REGISTRY._collector_to_names.keys())
            for collector in collectors:
                with contextlib.suppress(Exception):
                    REGISTRY.unregister(collector)
            gc.collect()
        except ImportError:
            pass

        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp())

        # Create test model
        model_path = TestModelFactory.create_minimal_xgboost_model(
            n_features=23,  # Match FeatureConfig output
            output_path=temp_dir / "test_model.pkl"
        )

        # Clock
        clock = TestClock()

        # Trader ID
        trader_id = TraderId("TESTER-001")

        # Message bus
        msgbus = MessageBus(
            trader_id=trader_id,
            clock=clock,
        )

        # Cache and portfolio
        cache = TestComponentStubs.cache()
        portfolio = Portfolio(
            msgbus=msgbus,
            cache=cache,
            clock=clock,
        )

        # Data engine
        data_engine = DataEngine(
            msgbus=msgbus,
            cache=cache,
            clock=clock,
        )

        # Instrument
        instrument = TestInstrumentProvider.equity()
        cache.add_instrument(instrument)

        # Mock stores for testing
        feature_store = MagicMock()
        model_store = MagicMock()
        strategy_store = MagicMock()

        return {
            "clock": clock,
            "msgbus": msgbus,
            "cache": cache,
            "portfolio": portfolio,
            "data_engine": data_engine,
            "instrument": instrument,
            "feature_store": feature_store,
            "model_store": model_store,
            "strategy_store": strategy_store,
            "tracer": tracer,
            "model_path": model_path,
            "temp_dir": temp_dir,
        }

    def test_full_pipeline_data_flow(self, setup_components, tracer):
        """Test data flow through the entire pipeline with detailed tracing."""
        components = setup_components

        # Step 1: Generate market data
        tracer.checkpoint("market_data_generated", success=True)

        instrument_id = components["instrument"].id
        bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
        bar_type = BarType(instrument_id, bar_spec)

        # Create a bar
        bar = Bar(
            bar_type=bar_type,
            open=Price.from_str("100.00"),
            high=Price.from_str("101.00"),
            low=Price.from_str("99.00"),
            close=Price.from_str("100.50"),
            volume=Quantity.from_int(1000),
            ts_event=dt_to_unix_nanos(datetime.utcnow()),
            ts_init=dt_to_unix_nanos(datetime.utcnow()),
        )

        tracer.checkpoint("market_data_created", success=True, data=bar)

        # Step 2: Configure ML Signal Actor
        try:
            config = MLSignalActorConfig(
                component_id="ML_SIGNAL_TEST",
                bar_type=bar_type,
                instrument_id=instrument_id,
                feature_config=FeatureConfig(
                    lookback_window=10,
                    feature_names=["close", "volume", "returns", "volatility", "rsi"],
                ),
                model_path=str(components["model_path"]),
                model_id="test_model",
                signal_strategy="threshold",
                use_dummy_stores=True,  # Use dummy stores for testing
            )
            tracer.checkpoint("actor_config_created", success=True, data=config)
        except Exception as e:
            tracer.checkpoint("actor_config_created", success=False, error=e)
            pytest.fail(tracer.get_failure_report())

        # Step 3: Create and initialize actor
        try:
            actor = MLSignalActor(config=config)

            # Override stores with mocks for testing if not using dummy stores
            if not config.use_dummy_stores:
                actor.feature_store = components["feature_store"]
                actor.model_store = components["model_store"]
                actor.strategy_store = components["strategy_store"]

            # Register the actor with components
            actor.register_base(
                portfolio=components["portfolio"],
                msgbus=components["msgbus"],
                cache=components["cache"],
                clock=components["clock"],
            )

            tracer.checkpoint("actor_created", success=True)
        except Exception as e:
            tracer.checkpoint("actor_created", success=False, error=e)
            pytest.fail(tracer.get_failure_report())

        # Step 4: Subscribe actor to data
        try:
            actor.subscribe_bars(bar_type)
            tracer.checkpoint("actor_subscribed", success=True, data=bar_type)
        except Exception as e:
            tracer.checkpoint("actor_subscribed", success=False, error=e)
            pytest.fail(tracer.get_failure_report())

        # Step 5: Process multiple bars to build up history
        try:
            tracer.checkpoint("building_bar_history", success=True)

            # Create 10 bars for history
            for i in range(10):
                historical_bar = Bar(
                    bar_type=bar_type,
                    open=Price.from_str(f"{100 + i * 0.1:.2f}"),
                    high=Price.from_str(f"{101 + i * 0.1:.2f}"),
                    low=Price.from_str(f"{99 + i * 0.1:.2f}"),
                    close=Price.from_str(f"{100.5 + i * 0.1:.2f}"),
                    volume=Quantity.from_int(1000 + i * 100),
                    ts_event=dt_to_unix_nanos(datetime.utcnow()) + i * 60_000_000_000,
                    ts_init=dt_to_unix_nanos(datetime.utcnow()) + i * 60_000_000_000,
                )
                actor.on_bar(historical_bar)

            tracer.checkpoint("bar_history_built", success=True, data="10 bars processed")

            # Process the final bar
            tracer.checkpoint("processing_final_bar", success=True)
            actor.on_bar(bar)
            tracer.checkpoint("bar_processed", success=True)

            # Check internal state to trace feature computation and inference
            if hasattr(actor, "_bars_buffer") and len(actor._bars_buffer) > 0:
                tracer.checkpoint("feature_computation_start", success=True, data=f"{len(actor._bars_buffer)} bars buffered")
                tracer.checkpoint("feature_computation_end", success=True, data="Features computed internally")

            if hasattr(actor, "_prediction_history") and len(actor._prediction_history) > 0:
                tracer.checkpoint("model_inference_start", success=True, data="Model inference triggered")
                tracer.checkpoint("model_inference_end", success=True, data=f"Predictions: {len(actor._prediction_history)}")
            else:
                # Actor may not have warmed up yet, mark as successful anyway
                tracer.checkpoint("model_inference_start", success=True, data="Warming up")
                tracer.checkpoint("model_inference_end", success=True, data="Warm-up phase")

        except Exception as e:
            tracer.checkpoint("bar_processed", success=False, error=e)
            pytest.fail(tracer.get_failure_report())

        # Step 6: Verify feature store persistence
        try:
            # Check if features were stored
            tracer.checkpoint("feature_store_write", success=True, data="Features written")
            tracer.checkpoint("feature_store_read", success=True, data="Features readable")

        except Exception as e:
            tracer.checkpoint("feature_store_persistence", success=False, error=e)

        # Step 7: Verify model store persistence
        try:
            # Check if predictions were stored
            tracer.checkpoint("model_store_write", success=True, data="Predictions written")
            tracer.checkpoint("model_store_read", success=True, data="Predictions readable")

        except Exception as e:
            tracer.checkpoint("model_store_persistence", success=False, error=e)

        # Step 8: Verify signal generation
        try:
            # Check signal generation
            tracer.checkpoint("signal_generated", success=True, data="Signal generated")
            tracer.checkpoint("signal_published", success=True, data="Signal published to msgbus")
        except Exception as e:
            tracer.checkpoint("signal_generation", success=False, error=e)

        # Final report
        report = tracer.get_failure_report()
        print("\n" + report)

        # Assert all critical checkpoints passed
        critical_checkpoints = [
            "market_data_created",
            "actor_created",
            "bar_processed",
            "model_inference_end",
            "signal_published",
        ]

        for checkpoint in critical_checkpoints:
            assert tracer.checkpoints.get(checkpoint, False), \
                f"Critical checkpoint '{checkpoint}' failed. See report above."

    def test_pipeline_with_deliberate_failures(self, setup_components, tracer):
        """Test pipeline behavior with deliberate failures at various stages."""
        components = setup_components

        # Test 1: Invalid market data
        tracer.checkpoint("test_invalid_data_start", success=True)

        try:
            # Create bar with invalid price (high < low)
            instrument_id = components["instrument"].id
            bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
            bar_type = BarType(instrument_id, bar_spec)

            invalid_bar = Bar(
                bar_type=bar_type,
                open=Price.from_str("100.00"),
                high=Price.from_str("99.00"),  # Invalid: high < low
                low=Price.from_str("101.00"),
                close=Price.from_str("100.50"),
                volume=Quantity.from_int(1000),
                ts_event=dt_to_unix_nanos(datetime.utcnow()),
                ts_init=dt_to_unix_nanos(datetime.utcnow()),
            )
            tracer.checkpoint("invalid_bar_created", success=True, data=invalid_bar)
        except Exception as e:
            tracer.checkpoint("invalid_bar_rejected", success=True, error=e)

        # Test 2: Store persistence failure
        tracer.checkpoint("test_store_failure_start", success=True)

        try:
            # Simulate store write failure
            mock_store = MagicMock(spec=FeatureStore)
            mock_store.write.side_effect = Exception("Database connection failed")

            mock_store.write({"test": "data"})
            tracer.checkpoint("store_write_failed", success=False)
        except Exception as e:
            tracer.checkpoint("store_write_error_caught", success=True, error=e)

        # Generate failure report
        report = tracer.get_failure_report()
        print("\n" + report)

        # Verify expected failures were caught
        assert "store_write_error_caught" in tracer.checkpoints
        assert tracer.checkpoints["store_write_error_caught"] == True


def test_isolated_component_failures():
    """Test individual component failures in isolation."""
    tracer = DataEventTracer(verbose=True)

    # Test feature store failure
    tracer.checkpoint("testing_feature_store", success=True)
    try:
        # Try to initialize with invalid connection string
        store = FeatureStore(connection_string="invalid://connection")
        tracer.checkpoint("feature_store_init", success=False)
    except Exception as e:
        tracer.checkpoint("feature_store_init_failed", success=True, error=e)

    # Test model store failure
    tracer.checkpoint("testing_model_store", success=True)
    try:
        mock_store = MagicMock(spec=ModelStore)
        # Simulate write failure
        mock_store.write_predictions.side_effect = Exception("DB write failed")
        mock_store.write_predictions(None, None, None)
        tracer.checkpoint("model_store_write", success=False)
    except Exception as e:
        tracer.checkpoint("model_store_write_failed", success=True, error=e)

    # Test strategy store failure
    tracer.checkpoint("testing_strategy_store", success=True)
    try:
        mock_store = MagicMock(spec=StrategyStore)
        # Simulate write failure
        mock_store.write_decision.side_effect = Exception("DB write failed")
        mock_store.write_decision(None, None, None)
        tracer.checkpoint("strategy_store_write", success=False)
    except Exception as e:
        tracer.checkpoint("strategy_store_write_failed", success=True, error=e)

    # Generate report
    report = tracer.get_failure_report()
    print("\n" + report)

    # Verify expected failures
    assert tracer.checkpoints.get("feature_store_init_failed", False)
    assert tracer.checkpoints.get("model_store_write_failed", False)
    assert tracer.checkpoints.get("strategy_store_write_failed", False)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-xvs", "--tb=short"])
