"""
Demonstration of pipeline failure tracing in action.

This script shows how the tracing system helps debug real issues
in the ML pipeline by providing detailed failure information.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

from ml.actors.signal import MLSignalActor
from ml.actors.signal import MLSignalActorConfig
from ml.features.engineering import FeatureConfig
from ml.tests.fixtures.model_factory import TestModelFactory
from ml.tests.integration.test_data_event_tracing import DataEventTracer
from nautilus_trader.common.component import MessageBus
from nautilus_trader.common.component import TestClock
from nautilus_trader.core.datetime import dt_to_unix_nanos
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


def simulate_feature_computation_failure() -> None:
    """Simulate a failure during feature computation."""
    tracer = DataEventTracer(verbose=True)

    print("=" * 80)
    print("SIMULATING FEATURE COMPUTATION FAILURE")
    print("=" * 80)

    try:
        # Setup components
        temp_dir = Path(tempfile.mkdtemp())
        model_path = TestModelFactory.create_minimal_xgboost_model(
            n_features=23,
            output_path=temp_dir / "test_model.pkl"
        )

        clock = TestClock()
        trader_id = TraderId("TESTER-001")
        msgbus = MessageBus(trader_id=trader_id, clock=clock)
        cache = TestComponentStubs.cache()
        portfolio = Portfolio(msgbus=msgbus, cache=cache, clock=clock)

        instrument = TestInstrumentProvider.equity()
        cache.add_instrument(instrument)

        # Create config with invalid feature configuration
        tracer.checkpoint("creating_config", success=True)

        config = MLSignalActorConfig(
            component_id="ML_SIGNAL_TEST",
            bar_type=BarType(
                instrument.id,
                BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
            ),
            instrument_id=instrument.id,
            feature_config=FeatureConfig(
                lookback_window=5,  # Too small for some indicators
                feature_names=["invalid_feature", "another_bad_feature"],
            ),
            model_path=str(model_path),
            model_id="test_model",
            signal_strategy="threshold",
            use_dummy_stores=True,
        )

        tracer.checkpoint("config_created", success=True, data="Config with invalid features")

        # Create and register actor
        actor = MLSignalActor(config)
        actor.register_base(
            portfolio=portfolio,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
        )

        tracer.checkpoint("actor_initialized", success=True)

        # Try to process bars - this should fail during feature computation
        bar_type = config.bar_type
        actor.subscribe_bars(bar_type)

        tracer.checkpoint("processing_bars", success=True)

        # Process multiple bars
        for i in range(10):
            bar = Bar(
                bar_type=bar_type,
                open=Price.from_str(f"{100 + i * 0.1:.2f}"),
                high=Price.from_str(f"{101 + i * 0.1:.2f}"),
                low=Price.from_str(f"{99 + i * 0.1:.2f}"),
                close=Price.from_str(f"{100.5 + i * 0.1:.2f}"),
                volume=Quantity.from_int(1000 + i * 100),
                ts_event=dt_to_unix_nanos(cast(Any, datetime.utcnow())) + i * 60_000_000_000,
                ts_init=dt_to_unix_nanos(cast(Any, datetime.utcnow())) + i * 60_000_000_000,
            )
            actor.on_bar(bar)

        tracer.checkpoint("bars_processed", success=True, data="10 bars processed without error")

    except Exception as e:
        tracer.checkpoint("pipeline_failed", success=False, error=e)

    # Generate report
    print("\n" + tracer.get_failure_report())


def simulate_model_inference_failure() -> None:
    """Simulate a failure during model inference."""
    tracer = DataEventTracer(verbose=True)

    print("\n" * 2)
    print("=" * 80)
    print("SIMULATING MODEL INFERENCE FAILURE")
    print("=" * 80)

    try:
        # Setup components
        temp_dir = Path(tempfile.mkdtemp())

        # Create a corrupt model file
        corrupt_model_path = temp_dir / "corrupt_model.pkl"
        corrupt_model_path.write_text("This is not a valid model file")

        tracer.checkpoint("created_corrupt_model", success=True, data=str(corrupt_model_path))

        clock = TestClock()
        trader_id = TraderId("TESTER-001")
        msgbus = MessageBus(trader_id=trader_id, clock=clock)
        cache = TestComponentStubs.cache()
        portfolio = Portfolio(msgbus=msgbus, cache=cache, clock=clock)

        instrument = TestInstrumentProvider.equity()
        cache.add_instrument(instrument)

        # Create config pointing to corrupt model
        config = MLSignalActorConfig(
            component_id="ML_SIGNAL_TEST",
            bar_type=BarType(
                instrument.id,
                BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
            ),
            instrument_id=instrument.id,
            feature_config=FeatureConfig(lookback_window=10),
            model_path=str(corrupt_model_path),
            model_id="corrupt_model",
            signal_strategy="threshold",
            use_dummy_stores=True,
        )

        tracer.checkpoint("config_with_corrupt_model", success=True)

        # This should fail when trying to load the model
        actor = MLSignalActor(config)
        tracer.checkpoint("actor_created_unexpectedly", success=False,
                         error=Exception("Should have failed loading corrupt model"))

    except Exception as e:
        tracer.checkpoint("model_loading_failed", success=True,
                         error=e, data="Failed as expected when loading corrupt model")

    # Generate report
    print("\n" + tracer.get_failure_report())


def simulate_store_persistence_failure() -> None:
    """Simulate a failure when persisting to stores."""
    tracer = DataEventTracer(verbose=True)

    print("\n" * 2)
    print("=" * 80)
    print("SIMULATING STORE PERSISTENCE FAILURE")
    print("=" * 80)

    # Mock stores that fail on write
    feature_store = MagicMock()
    feature_store.write_features.side_effect = ConnectionError("Database connection lost")

    model_store = MagicMock()
    model_store.write_predictions.side_effect = PermissionError("Insufficient permissions")

    strategy_store = MagicMock()
    strategy_store.write_decision.side_effect = ValueError("Invalid data format")

    # Try to write to each store
    tracer.checkpoint("attempting_feature_store_write", success=True)
    try:
        feature_store.write_features({"test": "data"})
        tracer.checkpoint("feature_store_write", success=True)
    except ConnectionError as e:
        tracer.checkpoint("feature_store_write_failed", success=False, error=e)

    tracer.checkpoint("attempting_model_store_write", success=True)
    try:
        model_store.write_predictions([0.5, 0.8, 0.3])
        tracer.checkpoint("model_store_write", success=True)
    except PermissionError as e:
        tracer.checkpoint("model_store_write_failed", success=False, error=e)

    tracer.checkpoint("attempting_strategy_store_write", success=True)
    try:
        strategy_store.write_decision("BUY", 0.95)
        tracer.checkpoint("strategy_store_write", success=True)
    except ValueError as e:
        tracer.checkpoint("strategy_store_write_failed", success=False, error=e)

    # Generate report
    print("\n" + tracer.get_failure_report())


if __name__ == "__main__":
    # Run all simulations
    simulate_feature_computation_failure()
    simulate_model_inference_failure()
    simulate_store_persistence_failure()

    print("\n" * 2)
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("""
The tracing system helps identify:
1. WHERE the failure occurred (exact checkpoint)
2. WHAT failed (error message and type)
3. WHEN it failed (timestamp)
4. WHY it failed (traceback and context)
5. HOW the pipeline was affected (flow diagram)

This makes debugging ML pipeline issues much faster and more efficient!
""")
