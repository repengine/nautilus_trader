"""
Example demonstrating minimal distributed tracing integration.

This example shows how to use the optional distributed tracing features
in a cold-path ML pipeline. The tracing system is designed to be:

1. OFF by default with zero overhead
2. Optional dependency (graceful fallback)
3. Cold-path only (never used in hot-path code)
4. W3C compliant for interoperability

Run with tracing disabled (default):
    python ml/tests/examples/distributed_tracing_example.py

Run with tracing enabled:
    ML_TRACING_ENABLED=true python ml/tests/examples/distributed_tracing_example.py

Run with tracing and OTLP export:
    ML_TRACING_ENABLED=true ML_TRACING_ENDPOINT=http://localhost:4317 \
    python ml/tests/examples/distributed_tracing_example.py

"""

import time
from collections.abc import Callable
from typing import Any, cast

import pandas as pd

from ml.common import (
    emit_dataset_event_and_watermark,
    extract_and_link_from_event,
    get_correlation_and_trace_context,
)
from ml.config.events import EventStatus, Source, Stage
from ml.observability.tracing import (
    is_tracing_enabled,
    trace_cold_path,
    trace_cold_path_decorator,
    trace_inference,
)


# Example 1: Cold-path boundary tracing with context manager
def example_feature_computation():
    """
    Example of tracing feature computation boundaries.
    """
    print("=== Feature Computation Example ===")
    print(f"Tracing enabled: {is_tracing_enabled()}")

    instrument_id = "EUR/USD.SIM"

    # Simulate data loading with tracing
    with trace_cold_path("data_loading", instrument_id=instrument_id) as span:
        if span:
            span.set_attribute("data_source", "historical")
            span.set_attribute("lookback_days", 30)

        print("Loading historical data...")
        time.sleep(0.1)  # Simulate data loading
        data = pd.DataFrame(
            {
                "timestamp": range(1000),
                "price": [100 + i * 0.01 for i in range(1000)],
                "volume": [1000 + i for i in range(1000)],
            },
        )

    # Feature computation with automatic correlation_id propagation
    correlation_id = "feature_run_12345"
    with trace_cold_path("feature_computation", correlation_id=correlation_id) as span:
        if span:
            span.set_attribute("feature_count", 5)
            span.set_attribute("window_size", 20)

        print("Computing features...")
        time.sleep(0.05)  # Simulate feature computation

        # Generate features
        features = pd.DataFrame(
            {
                "price_sma_20": data["price"].rolling(20).mean(),
                "volatility": data["price"].rolling(20).std(),
                "volume_ma": data["volume"].rolling(20).mean(),
            },
        )

    print(f"Computed {len(features)} feature rows")
    return features, correlation_id


# Example 2: Decorator-based tracing
@trace_cold_path_decorator("model_training")
def example_model_training(features: pd.DataFrame) -> dict[str, Any]:
    """
    Example of tracing model training with decorator.
    """
    print("\n=== Model Training Example ===")

    print("Training model...")
    time.sleep(0.2)  # Simulate model training

    # Simulate model metrics
    model_metrics = {
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.88,
        "training_samples": len(features),
    }

    print(f"Model trained with metrics: {model_metrics}")
    return model_metrics


# Example 3: Actor inference tracing
class ExampleMLActor:
    """
    Example ML actor with tracing.
    """

    def __init__(self):
        self.model_loaded = True
        self.prediction_count = 0

    @trace_inference("signal_generation")
    def on_bar(self, bar: Any) -> str:
        """
        Example bar handler with inference tracing.
        """
        self.prediction_count += 1

        # Simulate feature computation
        features = [bar.close, bar.volume, bar.close / bar.open]

        # Simulate model prediction
        time.sleep(0.001)  # Simulate inference
        prediction = sum(features) % 2  # Mock prediction

        signal = "BUY" if prediction > 0.5 else "SELL"
        return signal

    @trace_cold_path_decorator("model_evaluation", correlation_id_param="eval_id")
    def evaluate_model(self, test_data: pd.DataFrame, eval_id: str) -> dict[str, float]:
        """
        Example model evaluation with correlation tracking.
        """
        print(f"\nEvaluating model with eval_id: {eval_id}")
        time.sleep(0.1)  # Simulate evaluation

        return {
            "test_accuracy": 0.87,
            "test_samples": len(test_data),
            "inference_latency_ms": 1.2,
        }


# Example 4: Event-driven tracing with correlation
def example_event_driven_workflow():
    """
    Example of event-driven workflow with trace context propagation.
    """
    print("\n=== Event-Driven Workflow Example ===")

    # Simulate initial data processing
    metadata = get_correlation_and_trace_context(
        run_id="pipeline_run_456",
        dataset_id="features",
        instrument_id="EUR/USD.SIM",
        ts_min=1000000000,
        ts_max=2000000000,
        count=500,
    )

    print(f"Generated metadata: {list(metadata.keys())}")

    # Simulate registry for event emission
    class MockRegistry:
        def emit_event(self, **kwargs):
            print(f"Event emitted: {kwargs['stage'].value} - {kwargs['status'].value}")

        def update_watermark(self, **kwargs):
            print(f"Watermark updated: {kwargs['dataset_id']}")

    registry = MockRegistry()

    # Emit event with trace context
    emit_dataset_event_and_watermark(
        registry=registry,
        dataset_id="features",
        instrument_id="EUR/USD.SIM",
        stage=Stage.FEATURE_COMPUTED,
        source=Source.HISTORICAL,
        run_id="pipeline_run_456",
        ts_min=1000000000,
        ts_max=2000000000,
        count=500,
        status=EventStatus.SUCCESS,
        metadata=metadata,
    )

    # Simulate event consumption with trace context linking
    print("\n--- Event Consumer Side ---")
    event_metadata = metadata.copy()
    print(f"Received event metadata: {list(event_metadata.keys())}")

    # Link trace context from event
    extract_and_link_from_event(event_metadata)

    # Process event in new traced operation
    with trace_cold_path("event_processing") as span:
        if span:
            span.set_attribute("event_type", "feature_computed")
            span.set_attribute("processing_node", "consumer_1")

        print("Processing received event...")
        time.sleep(0.02)

    print("Event processing complete")


# Example 5: Performance demonstration
def example_performance_when_disabled():
    """
    Demonstrate zero overhead when tracing disabled.
    """
    print("\n=== Performance Example ===")

    def _fast_operation_impl(x: int) -> int:
        return x * 2

    fast_operation = cast(
        Callable[[int], int],
        trace_cold_path_decorator("performance_test")(_fast_operation_impl),
    )

    # Measure performance with tracing decorators
    start_time = time.perf_counter()
    for i in range(1000):
        result = fast_operation(i)
    elapsed = time.perf_counter() - start_time

    print(f"Completed 1000 decorated function calls in {elapsed:.4f} seconds")
    print(f"Average per call: {elapsed * 1000:.4f} ms")

    # Test context manager overhead
    start_time = time.perf_counter()
    for i in range(1000):
        with trace_cold_path("perf_test"):
            result = i * 2
    elapsed = time.perf_counter() - start_time

    print(f"Completed 1000 context manager calls in {elapsed:.4f} seconds")


def main():
    """
    Run all tracing examples.
    """
    print("Distributed Tracing Example")
    print("=" * 50)
    print(f"Tracing status: {'ENABLED' if is_tracing_enabled() else 'DISABLED (default)'}")

    if is_tracing_enabled():
        print("Note: When enabled, spans will be created and optionally exported to OTLP endpoint")
    else:
        print("Note: When disabled, all tracing operations are no-ops with zero overhead")

    print()

    try:
        # Run examples
        features, correlation_id = example_feature_computation()
        model_metrics = example_model_training(features)

        # Actor example
        actor = ExampleMLActor()

        # Simulate some bar data
        class MockBar:
            def __init__(self, close: float, volume: int, open_price: float):
                self.close = close
                self.volume = volume
                self.open = open_price
                self.instrument_id = "EUR/USD.SIM"

        bars = [
            MockBar(1.1234, 1000, 1.1230),
            MockBar(1.1235, 1100, 1.1234),
            MockBar(1.1233, 900, 1.1235),
        ]

        print("\n=== Actor Inference Example ===")
        for i, bar in enumerate(bars):
            signal = actor.on_bar(bar)
            print(f"Bar {i+1}: {signal}")

        # Model evaluation with correlation
        test_data = features.iloc[:100]
        eval_metrics = actor.evaluate_model(test_data, eval_id=correlation_id)
        print(f"Evaluation metrics: {eval_metrics}")

        # Event-driven workflow
        example_event_driven_workflow()

        # Performance demonstration
        example_performance_when_disabled()

        print("\n" + "=" * 50)
        print("Example completed successfully!")
        print(f"Final prediction count: {actor.prediction_count}")

    except Exception as e:
        print(f"Error running example: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
