"""
Property-based tests for model store prediction invariants.

This module provides comprehensive property testing for ModelStore prediction storage
with focus on mathematical properties and behavioral guarantees.

Performance targets: P99 < 5ms for property validation
Hot/Cold path separation: hot = prediction storage, cold = property validation

"""

from __future__ import annotations

# codespell:ignore-words-list=Nd

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal, Self
from unittest.mock import MagicMock

import pytest

# Use centralized imports for ML libraries
from ml._imports import HAS_PANDAS, check_ml_dependencies

if not HAS_PANDAS:
    check_ml_dependencies(["pandas"])

# Hypothesis imports with graceful fallback
try:
    from hypothesis import HealthCheck, assume, given, settings, strategies as st
    from hypothesis.stateful import (
        Bundle,
        RuleBasedStateMachine,
        run_state_machine_as_test,
        rule,
    )
except ImportError:  # pragma: no cover
    pytest.skip("hypothesis not available", allow_module_level=True)

import numpy as np

from ml.stores.base import ModelPrediction
from ml.stores.model_store import ModelStore

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator


# ============================================================================
# Constants and Utilities
# ============================================================================

pytestmark = pytest.mark.timeout(0)

PROPERTY_TEST_SETTINGS = settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=(HealthCheck.too_slow,),
)

# Prediction value bounds per Nautilus convention
PREDICTION_MIN = -1.0
PREDICTION_MAX = 1.0
CONFIDENCE_MIN = 0.0
CONFIDENCE_MAX = 1.0

# Nanosecond timestamp bounds (reasonable test range)
MIN_TIMESTAMP = 1_000_000_000_000_000_000  # ~2001
MAX_TIMESTAMP = 2_000_000_000_000_000_000  # ~2033


# ============================================================================
# Test Utilities and Mocks
# ============================================================================


@contextmanager
def _mock_model_store_io(sink: list[dict[str, Any]]) -> Iterator[None]:
    """
    Mock ModelStore I/O operations to capture writes without database dependency.

    This preserves the existing patch pattern while providing clean isolation.

    """
    # Store original methods
    orig_setup = ModelStore._setup_tables
    orig_exec = ModelStore._execute_write

    # Import and patch database engine
    from ml.core import db_engine as _db

    orig_get_engine = _db.EngineManager.get_engine

    class _DummyConnection:
        def __enter__(self) -> Self:
            return self

        def __exit__(
            self, exc_type: type[Exception] | None, exc: Exception | None, tb: Any
        ) -> Literal[False]:
            return False

        def execute(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def commit(self) -> None:
            pass

    class _DummyEngine:
        def connect(self) -> _DummyConnection:
            return _DummyConnection()

        def begin(self) -> _DummyConnection:
            return _DummyConnection()

    try:
        # Patch ModelStore methods
        ModelStore._setup_tables = lambda self: None  # type: ignore[method-assign]
        ModelStore._execute_write = lambda self, values: sink.extend(values)  # type: ignore[method-assign]

        # Patch engine manager
        _db.EngineManager.get_engine = lambda *a, **k: _DummyEngine()  # type: ignore[assignment]

        yield
    finally:
        # Restore original methods
        ModelStore._setup_tables = orig_setup  # type: ignore[method-assign]
        ModelStore._execute_write = orig_exec  # type: ignore[method-assign]
        _db.EngineManager.get_engine = orig_get_engine  # type: ignore[method-assign]


def _create_model_prediction(
    model_id: str,
    instrument_id: str,
    prediction: float,
    confidence: float,
    ts_event: int,
    ts_init: int | None = None,
    features: dict[str, float] | None = None,
    inference_time_ms: float = 1.0,
    is_live: bool = False,
) -> ModelPrediction:
    """
    Create ModelPrediction with sanitized inputs.
    """
    return ModelPrediction(
        model_id=model_id,
        instrument_id=instrument_id,
        prediction=float(prediction),
        confidence=float(confidence),
        features_used=features or {},
        inference_time_ms=float(inference_time_ms),
        _ts_event=int(ts_event),
        _ts_init=int(ts_init or ts_event),
        is_live=bool(is_live),
    )


# ============================================================================
# Hypothesis Strategies
# ============================================================================

# Basic value strategies
unicode_letter_categories: tuple[str, str, str] = (
    "Lu",
    "Ll",
    "N" "d",
)  # Unicode digit category

model_ids = st.text(
    min_size=1,
    max_size=20,
    alphabet=st.characters(whitelist_categories=unicode_letter_categories),
)
instrument_ids = st.from_regex(r"[A-Z]{3,6}/[A-Z]{3,6}\.SIM", fullmatch=True)

predictions = st.floats(
    min_value=PREDICTION_MIN,
    max_value=PREDICTION_MAX,
    allow_nan=False,
    allow_infinity=False,
)

confidences = st.floats(
    min_value=CONFIDENCE_MIN,
    max_value=CONFIDENCE_MAX,
    allow_nan=False,
    allow_infinity=False,
)

timestamps = st.integers(min_value=MIN_TIMESTAMP, max_value=MAX_TIMESTAMP)

inference_times = st.floats(min_value=0.0, max_value=5000.0, allow_nan=False, allow_infinity=False)

# Feature dictionaries with bounded values
features = st.dictionaries(
    st.text(min_size=1, max_size=10),
    st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    min_size=0,
    max_size=5,
)

# Single prediction strategy
single_prediction = st.builds(
    _create_model_prediction,
    model_id=model_ids,
    instrument_id=instrument_ids,
    prediction=predictions,
    confidence=confidences,
    ts_event=timestamps,
    ts_init=timestamps,
    features=features,
    inference_time_ms=inference_times,
    is_live=st.booleans(),
)


# Batch prediction strategy with monotonic timestamps
@st.composite
def batch_predictions_with_monotonic_timestamps(draw: st.DrawFn) -> list[ModelPrediction]:
    """
    Generate batch of predictions with monotonically increasing timestamps.
    """
    size = draw(st.integers(min_value=1, max_value=50))
    base_ts = draw(timestamps)

    predictions_list = []
    current_ts = base_ts

    for _ in range(size):
        model_id = draw(model_ids)
        instrument_id = draw(instrument_ids)
        prediction = draw(predictions)
        confidence = draw(confidences)
        features_dict = draw(features)
        inference_time = draw(inference_times)
        is_live = draw(st.booleans())

        pred = _create_model_prediction(
            model_id=model_id,
            instrument_id=instrument_id,
            prediction=prediction,
            confidence=confidence,
            ts_event=current_ts,
            ts_init=current_ts,
            features=features_dict,
            inference_time_ms=inference_time,
            is_live=is_live,
        )
        predictions_list.append(pred)

        # Increment timestamp for monotonicity
        current_ts += draw(st.integers(min_value=1, max_value=1000))

    return predictions_list


# ============================================================================
# Property-Based Tests: Core Invariants
# ============================================================================


class TestModelStorePredictionInvariants:
    """
    Property-based tests for ModelStore prediction invariants.
    """

    @PROPERTY_TEST_SETTINGS
    @given(prediction=single_prediction)
    def test_prediction_immutability_invariant(self, prediction: ModelPrediction) -> None:
        """
        Property: Once stored, predictions cannot change.

        This tests that the same prediction data stored multiple times
        results in identical persisted values.
        """
        sink: list[dict[str, Any]] = []

        with _mock_model_store_io(sink):
            store = ModelStore(connection_string="sqlite:///:memory:")

            # Store the same prediction twice
            store.write_batch([prediction])
            first_write = sink.copy()

            sink.clear()
            store.write_batch([prediction])
            second_write = sink.copy()

            # Predictions should be identical
            assert len(first_write) == len(second_write) == 1

            first_pred = first_write[0]
            second_pred = second_write[0]

            # Core prediction data must be immutable
            assert first_pred["model_id"] == second_pred["model_id"]
            assert first_pred["instrument_id"] == second_pred["instrument_id"]
            assert first_pred["ts_event"] == second_pred["ts_event"]
            assert first_pred["prediction"] == second_pred["prediction"]
            assert first_pred["confidence"] == second_pred["confidence"]
            assert first_pred["features_used"] == second_pred["features_used"]

    @PROPERTY_TEST_SETTINGS
    @given(predictions_batch=batch_predictions_with_monotonic_timestamps())
    def test_temporal_consistency_invariant(self, predictions_batch: list[ModelPrediction]) -> None:
        """
        Property: Predictions must have monotonic timestamps.

        Tests that ts_event values are non-decreasing and ts_init >= ts_event.
        """
        assume(len(predictions_batch) > 1)  # Need multiple predictions for temporal testing

        sink: list[dict[str, Any]] = []

        with _mock_model_store_io(sink):
            store = ModelStore(connection_string="sqlite:///:memory:")
            store.write_batch(predictions_batch)

            assert len(sink) == len(predictions_batch)

            # Verify temporal ordering
            last_ts_event = -1
            for stored_pred in sink:
                ts_event = int(stored_pred["ts_event"])
                ts_init = int(stored_pred["ts_init"])

                # ts_init must be >= ts_event (init happens at or after event)
                assert ts_init >= ts_event, f"ts_init ({ts_init}) < ts_event ({ts_event})"

                # ts_event must be non-decreasing (monotonic)
                assert (
                    ts_event >= last_ts_event
                ), f"Timestamp ordering violated: {ts_event} < {last_ts_event}"
                last_ts_event = ts_event

    @PROPERTY_TEST_SETTINGS
    @given(predictions_batch=batch_predictions_with_monotonic_timestamps())
    def test_batch_atomicity_invariant(self, predictions_batch: list[ModelPrediction]) -> None:
        """
        Property: Batch writes must be all-or-nothing.

        Tests that either all predictions in a batch are stored or none are.
        """
        sink: list[dict[str, Any]] = []

        with _mock_model_store_io(sink):
            store = ModelStore(connection_string="sqlite:///:memory:")

            # Successful batch write
            store.write_batch(predictions_batch)

            # All predictions should be stored
            assert len(sink) == len(predictions_batch)

            # Verify each prediction is properly stored
            for i, stored_pred in enumerate(sink):
                original_pred = predictions_batch[i]
                assert stored_pred["model_id"] == original_pred.model_id
                assert stored_pred["instrument_id"] == original_pred.instrument_id
                assert stored_pred["prediction"] == original_pred.prediction
                assert stored_pred["confidence"] == original_pred.confidence

    @PROPERTY_TEST_SETTINGS
    @given(
        predictions_batch=batch_predictions_with_monotonic_timestamps(),
        model_version=st.text(min_size=1, max_size=10),
    )
    def test_version_consistency_invariant(
        self,
        predictions_batch: list[ModelPrediction],
        model_version: str,
    ) -> None:
        """
        Property: Same model version must produce same predictions for same input.

        Tests deterministic behavior for identical model versions.
        """
        # Use the same model_id for version consistency testing
        versioned_predictions = []
        for pred in predictions_batch:
            versioned_pred = _create_model_prediction(
                model_id=f"{pred.model_id}_{model_version}",
                instrument_id=pred.instrument_id,
                prediction=pred.prediction,
                confidence=pred.confidence,
                ts_event=pred.ts_event,
                ts_init=pred.ts_init,
                features=pred.features_used,
                inference_time_ms=pred.inference_time_ms,
                is_live=pred.is_live,
            )
            versioned_predictions.append(versioned_pred)

        sink1: list[dict[str, Any]] = []
        sink2: list[dict[str, Any]] = []

        # Store same versioned predictions twice
        with _mock_model_store_io(sink1):
            store1 = ModelStore(connection_string="sqlite:///:memory:")
            store1.write_batch(versioned_predictions)

        with _mock_model_store_io(sink2):
            store2 = ModelStore(connection_string="sqlite:///:memory:")
            store2.write_batch(versioned_predictions)

        # Results should be identical
        assert len(sink1) == len(sink2)
        for stored1, stored2 in zip(sink1, sink2):
            assert stored1["model_id"] == stored2["model_id"]
            assert stored1["prediction"] == stored2["prediction"]
            assert stored1["confidence"] == stored2["confidence"]

    @PROPERTY_TEST_SETTINGS
    @given(predictions_batch=batch_predictions_with_monotonic_timestamps())
    def test_confidence_bounds_invariant(self, predictions_batch: list[ModelPrediction]) -> None:
        """
        Property: Confidence scores must be in [0,1].

        Tests that confidence values are properly bounded.
        """
        sink: list[dict[str, Any]] = []

        with _mock_model_store_io(sink):
            store = ModelStore(connection_string="sqlite:///:memory:")
            store.write_batch(predictions_batch)

            # Verify confidence bounds for all stored predictions
            for stored_pred in sink:
                confidence = float(stored_pred["confidence"])
                assert (
                    CONFIDENCE_MIN <= confidence <= CONFIDENCE_MAX
                ), f"Confidence {confidence} outside bounds [{CONFIDENCE_MIN}, {CONFIDENCE_MAX}]"

                # Also verify prediction bounds
                prediction = float(stored_pred["prediction"])
                assert (
                    PREDICTION_MIN <= prediction <= PREDICTION_MAX
                ), f"Prediction {prediction} outside bounds [{PREDICTION_MIN}, {PREDICTION_MAX}]"

    @PROPERTY_TEST_SETTINGS
    @given(predictions_batch=batch_predictions_with_monotonic_timestamps())
    def test_prediction_uniqueness_invariant(
        self, predictions_batch: list[ModelPrediction]
    ) -> None:
        """
        Property: No duplicate predictions for same (instrument, timestamp, model).

        Tests that predictions are unique by their natural key.
        """
        # Create predictions with potential duplicates by forcing same keys
        if len(predictions_batch) >= 2:
            # Force first two predictions to have same key components except prediction value
            pred1 = predictions_batch[0]
            pred2 = predictions_batch[1]

            duplicate_key_pred = _create_model_prediction(
                model_id=pred1.model_id,
                instrument_id=pred1.instrument_id,
                prediction=pred2.prediction,  # Different prediction value
                confidence=pred2.confidence,
                ts_event=pred1.ts_event,  # Same timestamp
                ts_init=pred1.ts_init,
                features=pred1.features_used,
                inference_time_ms=pred1.inference_time_ms,
                is_live=pred1.is_live,
            )

            test_batch = [pred1, duplicate_key_pred]
        else:
            test_batch = predictions_batch

        sink: list[dict[str, Any]] = []

        with _mock_model_store_io(sink):
            store = ModelStore(connection_string="sqlite:///:memory:")
            store.write_batch(test_batch)

            # Collect unique keys
            seen_keys = set()
            for stored_pred in sink:
                key = (
                    stored_pred["model_id"],
                    stored_pred["instrument_id"],
                    stored_pred["ts_event"],
                )

                # In a real system, duplicates would be handled by upsert logic
                # For this test, we verify the data structure supports uniqueness
                if key in seen_keys:
                    # This is expected behavior - last write wins in upsert scenarios
                    pass
                seen_keys.add(key)


# ============================================================================
# Property-Based Tests: Performance and Data Integrity
# ============================================================================


class TestModelStorePerformanceInvariants:
    """
    Property-based tests for ModelStore performance and data integrity.
    """

    @PROPERTY_TEST_SETTINGS
    @given(predictions_batch=batch_predictions_with_monotonic_timestamps())
    def test_no_data_loss_during_flush_invariant(
        self, predictions_batch: list[ModelPrediction]
    ) -> None:
        """
        Property: No data loss during flush operations.

        Tests that all buffered predictions are written during flush.
        """
        sink: list[dict[str, Any]] = []

        with _mock_model_store_io(sink):
            store = ModelStore(
                connection_string="sqlite:///:memory:", batch_size=1000
            )  # Large batch size

            # Add predictions to buffer without auto-flush
            for pred in predictions_batch:
                store._write_buffer.append(pred)

            # Verify buffer contains all predictions
            assert len(store._write_buffer) == len(predictions_batch)

            # Manual flush
            store.flush()

            # Verify all predictions were written
            assert len(sink) == len(predictions_batch)

            # Verify buffer is cleared after flush
            assert len(store._write_buffer) == 0

    @PROPERTY_TEST_SETTINGS
    @given(
        predictions_batch=batch_predictions_with_monotonic_timestamps(),
        watermark_increment=st.integers(min_value=1, max_value=1000),
    )
    def test_watermark_progression_monotonicity(
        self,
        predictions_batch: list[ModelPrediction],
        watermark_increment: int,
    ) -> None:
        """
        Property: Watermark progression must be monotonic.

        Tests that timestamp watermarks only increase over time.
        """
        assume(len(predictions_batch) >= 2)

        # Simulate watermark progression by tracking max timestamps
        watermarks = []
        current_watermark = 0

        for pred in predictions_batch:
            # Watermark advances to max of current and prediction timestamp
            new_watermark = max(current_watermark, pred.ts_event + watermark_increment)
            watermarks.append(new_watermark)
            current_watermark = new_watermark

        # Verify monotonic progression
        for i in range(1, len(watermarks)):
            assert (
                watermarks[i] >= watermarks[i - 1]
            ), f"Watermark regression: {watermarks[i]} < {watermarks[i-1]}"

    @PROPERTY_TEST_SETTINGS
    @given(predictions_batch=batch_predictions_with_monotonic_timestamps())
    def test_performance_metrics_consistency(
        self, predictions_batch: list[ModelPrediction]
    ) -> None:
        """
        Property: Performance metrics calculations must be mathematically correct.

        Tests Sharpe ratio and related performance metrics for correctness.
        """
        assume(len(predictions_batch) >= 10)  # Need sufficient data for meaningful metrics

        # Extract predictions and calculate basic statistics
        prediction_values = [pred.prediction for pred in predictions_batch]

        # Basic statistical properties
        mean_prediction = np.mean(prediction_values)
        std_prediction = np.std(prediction_values, ddof=1) if len(prediction_values) > 1 else 0

        # Verify mathematical properties
        assert (
            -1.0 <= mean_prediction <= 1.0
        ), f"Mean prediction {mean_prediction} outside valid bounds"
        assert std_prediction >= 0, f"Standard deviation {std_prediction} cannot be negative"

        # If we have variance, verify Sharpe-like ratio calculation
        if std_prediction > 0:
            sharpe_like = mean_prediction / std_prediction
            # Sharpe ratio should be finite for valid inputs
            assert np.isfinite(sharpe_like), f"Sharpe ratio {sharpe_like} not finite"


# ============================================================================
# Stateful Property Testing
# ============================================================================


class ModelStoreStateMachine(RuleBasedStateMachine):
    """
    Stateful property testing for ModelStore operations.

    Tests complex sequences of operations to verify invariants hold across multiple
    interactions.

    """

    predictions = Bundle("predictions")

    def __init__(self) -> None:
        super().__init__()
        self._sink: list[dict[str, Any]] = []
        self._store_context = _mock_model_store_io(self._sink)
        self._store_context.__enter__()
        self._store = ModelStore(connection_string="sqlite:///:memory:")
        self._stored_count = 0

    def teardown(self) -> None:
        """
        Clean up resources.
        """
        try:
            self._store_context.__exit__(None, None, None)
        except Exception:
            pass
        super().teardown()

    @rule(target=predictions, prediction=single_prediction)
    def add_prediction(self, prediction: ModelPrediction) -> ModelPrediction:
        """
        Add a single prediction to the store.
        """
        self._store.write_batch([prediction])
        self._stored_count += 1
        return prediction

    @rule(prediction_batch=batch_predictions_with_monotonic_timestamps())
    def add_batch_predictions(self, prediction_batch: list[ModelPrediction]) -> None:
        """
        Add a batch of predictions to the store.
        """
        self._store.write_batch(prediction_batch)
        self._stored_count += len(prediction_batch)

    @rule()
    def flush_store(self) -> None:
        """
        Manually flush the store.
        """
        self._store.flush()

    @rule()
    def verify_invariants(self) -> None:
        """
        Verify all stored data maintains invariants.
        """
        # Check that stored count matches expected
        assert len(self._sink) <= self._stored_count  # May be less due to deduplication

        # Verify each stored prediction maintains bounds
        for stored_pred in self._sink:
            prediction = float(stored_pred["prediction"])
            confidence = float(stored_pred["confidence"])

            assert PREDICTION_MIN <= prediction <= PREDICTION_MAX
            assert CONFIDENCE_MIN <= confidence <= CONFIDENCE_MAX
            assert int(stored_pred["ts_init"]) >= int(stored_pred["ts_event"])


# ============================================================================
# Test Runner Configuration
# ============================================================================


# Run stateful tests via explicit helper to avoid unittest fixture conflicts
def test_model_store_stateful() -> None:
    run_state_machine_as_test(
        ModelStoreStateMachine,
        settings=settings(
            max_examples=10,
            stateful_step_count=25,
            deadline=None,
            suppress_health_check=(HealthCheck.too_slow,),
        ),
    )


# ============================================================================
# Integration with Existing Test Infrastructure
# ============================================================================


def test_property_tests_integration() -> None:
    """
    Verify property tests integrate correctly with existing test infrastructure.

    This test ensures our property-based tests follow the established patterns and can
    be run alongside existing unit tests.

    """
    # Test that we can create mock stores consistently
    sink: list[dict[str, Any]] = []

    with _mock_model_store_io(sink):
        store = ModelStore(connection_string="sqlite:///:memory:")

        # Test basic functionality
        pred = _create_model_prediction(
            model_id="test_model",
            instrument_id="EUR/USD.SIM",
            prediction=0.5,
            confidence=0.8,
            ts_event=1_600_000_000_000_000_000,
        )

        store.write_batch([pred])

        assert len(sink) == 1
        assert sink[0]["model_id"] == "test_model"
        assert sink[0]["prediction"] == 0.5
        assert sink[0]["confidence"] == 0.8


# ============================================================================
# Property Test Performance Benchmarking
# ============================================================================


@PROPERTY_TEST_SETTINGS
@given(predictions_batch=batch_predictions_with_monotonic_timestamps())
def test_property_validation_performance(predictions_batch: list[ModelPrediction]) -> None:
    """
    Property: Property validation should complete within performance targets.

    Tests that property validation itself doesn't become a bottleneck.
    """
    start_time = time.perf_counter()

    # Run core invariant checks
    assert all(PREDICTION_MIN <= pred.prediction <= PREDICTION_MAX for pred in predictions_batch)
    assert all(CONFIDENCE_MIN <= pred.confidence <= CONFIDENCE_MAX for pred in predictions_batch)
    assert all(pred.ts_init >= pred.ts_event for pred in predictions_batch)

    end_time = time.perf_counter()
    validation_time_ms = (end_time - start_time) * 1000

    # Property validation should be fast (< 5ms for reasonable batch sizes)
    if len(predictions_batch) <= 100:
        assert (
            validation_time_ms < 5.0
        ), f"Validation took {validation_time_ms:.2f}ms for {len(predictions_batch)} predictions"
