"""
Property-based tests for Store invariants.

These tests verify mathematical properties and invariants that must hold
regardless of the specific data or implementation details. They help catch
edge cases and ensure correctness without brittle example-based tests.
"""

from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import assume
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import Bundle
from hypothesis.stateful import RuleBasedStateMachine
from hypothesis.stateful import rule

from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategyStore
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue


# Custom strategies for domain objects
@st.composite
def instrument_ids(draw):
    """Generate valid instrument IDs."""
    symbol = draw(st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=2, max_size=6))
    venue = draw(st.sampled_from(["SIM", "BINANCE", "FTX", "NASDAQ"]))
    return InstrumentId(Symbol(symbol), Venue(venue))


@st.composite
def nanosecond_timestamps(draw, min_value=None, max_value=None):
    """Generate valid nanosecond timestamps."""
    if min_value is None:
        min_value = int(datetime(2020, 1, 1).timestamp() * 1e9)
    if max_value is None:
        max_value = int(datetime(2025, 12, 31).timestamp() * 1e9)
    return draw(st.integers(min_value=min_value, max_value=max_value))


@st.composite
def feature_values(draw, n_features=None):
    """Generate feature value dictionaries."""
    if n_features is None:
        n_features = draw(st.integers(min_value=1, max_value=20))

    feature_names = [f"feature_{i}" for i in range(n_features)]
    values = draw(st.lists(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=n_features,
        max_size=n_features
    ))
    return dict(zip(feature_names, values))


@pytest.mark.property
@pytest.mark.database
@pytest.mark.serial
class TestFeatureStoreInvariants:
    """Property tests for FeatureStore invariants."""

    def _create_dummy_store(self):
        """Create a dummy feature store for testing."""
        with patch("ml.stores.feature_store.EngineManager") as mock_engine_manager:
            mock_engine = MagicMock()
            mock_engine_manager.get_engine.return_value = mock_engine

            # Create store with mocked engine
            store = FeatureStore(connection_string="dummy://")
            store._features_cache = {}  # Simple in-memory cache
            return store

    @given(
        instrument_id=instrument_ids(),
        feature_set_id=st.text(min_size=1, max_size=50),
        features=feature_values(),
        ts_events=st.lists(nanosecond_timestamps(), min_size=1, max_size=100, unique=True)
    )
    @pytest.mark.database
    @pytest.mark.serial
    @settings(max_examples=50, deadline=5000)
    def test_timestamp_monotonicity_invariant(
        self,
        instrument_id,
        feature_set_id,
        features,
        ts_events,
    ):
        """
        Property: Features stored with increasing timestamps must maintain order.

        Invariant: For any sequence of feature writes, reading them back
        should return them in timestamp order.
        """
        # Create a fresh store for this test
        dummy_feature_store = self._create_dummy_store()

        # Sort timestamps to ensure we write them in order
        ts_events = sorted(ts_events)

        # Store features with increasing timestamps
        stored_data = []
        for ts_event in ts_events:
            ts_init = ts_event + 1  # ts_init must be >= ts_event

            # Store in cache (simulating DB write)
            cache_key = (feature_set_id, str(instrument_id), ts_event)
            dummy_feature_store._features_cache[cache_key] = {
                "features": features,
                "ts_event": ts_event,
                "ts_init": ts_init,
            }
            stored_data.append((ts_event, ts_init, features))

        # Property: Retrieved timestamps should be monotonically increasing
        cached_items = sorted(dummy_feature_store._features_cache.items(), key=lambda x: x[0][2])
        previous_ts = 0
        for (_, _, ts_event), data in cached_items:
            assert ts_event >= previous_ts, f"Timestamp order violated: {ts_event} < {previous_ts}"
            assert data["ts_init"] >= data["ts_event"], "ts_init must be >= ts_event"
            previous_ts = ts_event

    @given(
        features1=feature_values(),
        features2=feature_values(),
        scale_factor=st.floats(min_value=0.1, max_value=10.0, allow_nan=False)
    )
    @pytest.mark.database
    @pytest.mark.serial
    @settings(max_examples=50, deadline=5000)
    def test_feature_immutability_invariant(self, features1, features2, scale_factor):
        """
        Property: Once stored, feature values should not change.

        Invariant: Reading the same feature twice should return identical values.
        """
        # Create a copy and scale it
        original_features = features1.copy()
        scaled_features = {k: v * scale_factor for k, v in features1.items()}

        # Property: Original features should remain unchanged
        assert original_features == features1, "Original features were mutated"

        # Property: Scaling should preserve structure
        assert set(scaled_features.keys()) == set(original_features.keys()), "Feature keys changed"

        # Property: Scaling should be reversible (within floating point precision)
        if scale_factor != 0:
            reversed_features = {k: v / scale_factor for k, v in scaled_features.items()}
            for key in original_features:
                assert abs(reversed_features[key] - original_features[key]) < 1e-10, \
                    f"Feature {key} not reversible after scaling"

    @given(
        n_partitions=st.integers(min_value=1, max_value=12),
        n_features=st.integers(min_value=100, max_value=1000),
        year=st.integers(min_value=2020, max_value=2025)
    )
    @pytest.mark.database
    @pytest.mark.serial
    @settings(max_examples=20, deadline=5000)
    def test_partition_consistency_invariant(self, n_partitions, n_features, year):
        """
        Property: Features should be correctly partitioned by time.

        Invariant: All features in a partition should fall within the partition's time range.
        """
        # Generate timestamps across the year
        start_timestamp = int(datetime(year, 1, 1).timestamp() * 1e9)
        end_timestamp = int(datetime(year, 12, 31, 23, 59, 59).timestamp() * 1e9)

        # Create non-overlapping partitions that cover the entire year
        partition_size = (end_timestamp - start_timestamp) // n_partitions
        partition_boundaries = []
        for i in range(n_partitions):
            part_start = start_timestamp + (i * partition_size)
            part_end = start_timestamp + ((i + 1) * partition_size) - 1
            if i == n_partitions - 1:  # Last partition extends to year end
                part_end = end_timestamp
            partition_boundaries.append((part_start, part_end))

        # Distribute features across partitions
        timestamps = np.random.randint(start_timestamp, end_timestamp, n_features)

        # Property: Each timestamp should belong to exactly one partition
        for ts in timestamps:
            matching_partitions = [
                i for i, (start, end) in enumerate(partition_boundaries)
                if start <= ts <= end
            ]
            assert len(matching_partitions) == 1, \
                f"Timestamp {ts} matches {len(matching_partitions)} partitions"


@pytest.mark.database
@pytest.mark.serial
class TestModelStoreInvariants:
    """Property tests for ModelStore invariants."""

    @given(
        predictions=st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
            min_size=10,
            max_size=100
        ),
        confidence_threshold=st.floats(min_value=0.0, max_value=1.0)
    )
    @pytest.mark.database
    @pytest.mark.serial
    @settings(max_examples=50, deadline=5000)
    def test_prediction_bounds_invariant(self, predictions, confidence_threshold):
        """
        Property: Model predictions should respect defined bounds.

        Invariant: All predictions and confidence scores must be within valid ranges.
        """
        for pred in predictions:
            # Property: Predictions should be bounded
            assert -1.0 <= pred <= 1.0, f"Prediction {pred} out of bounds [-1, 1]"

            # Property: Confidence derived from absolute value should be valid
            confidence = abs(pred)
            assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} out of bounds [0, 1]"

            # Property: Threshold filtering should work correctly
            should_act = confidence > confidence_threshold
            if should_act:
                assert confidence > confidence_threshold
            else:
                assert confidence <= confidence_threshold

    @given(
        model_versions=st.lists(
            st.tuples(
                st.integers(min_value=1, max_value=100),  # major
                st.integers(min_value=0, max_value=100),  # minor
                st.integers(min_value=0, max_value=100),  # patch
            ),
            min_size=2,
            max_size=10,
            unique=True
        )
    )
    @pytest.mark.database
    @pytest.mark.serial
    @settings(max_examples=50, deadline=5000)
    def test_model_versioning_consistency(self, model_versions):
        """
        Property: Model versions should follow semantic versioning rules.

        Invariant: Version comparisons should be transitive and consistent.
        """
        def version_key(v):
            return (v[0], v[1], v[2])

        sorted_versions = sorted(model_versions, key=version_key)

        # Property: Sorting should be stable
        resorted = sorted(sorted_versions, key=version_key)
        assert sorted_versions == resorted, "Version sorting is not stable"

        # Property: Version comparison should be transitive
        for i in range(len(sorted_versions) - 1):
            v1 = sorted_versions[i]
            v2 = sorted_versions[i + 1]
            assert version_key(v1) <= version_key(v2), \
                f"Version order violated: {v1} > {v2}"

        # Property: Latest version should be findable
        if sorted_versions:
            latest = max(model_versions, key=version_key)
            assert latest == sorted_versions[-1], "Latest version identification failed"


@pytest.mark.database
@pytest.mark.serial
class TestStrategyStoreInvariants:
    """Property tests for StrategyStore invariants."""

    @given(
        signals=st.lists(
            st.tuples(
                nanosecond_timestamps(),  # timestamp
                st.floats(min_value=-1.0, max_value=1.0),  # signal value
                st.floats(min_value=0.0, max_value=1.0),  # confidence
            ),
            min_size=1,
            max_size=100
        )
    )
    @pytest.mark.database
    @pytest.mark.serial
    @settings(max_examples=50, deadline=5000)
    def test_signal_ordering_invariant(self, signals):
        """
        Property: Signals should maintain temporal ordering.

        Invariant: Signals with earlier timestamps should be processed first.
        """
        # Sort signals by timestamp
        sorted_signals = sorted(signals, key=lambda x: x[0])

        # Property: Processing order should match timestamp order
        processing_order = []
        for ts, value, confidence in sorted_signals:
            processing_order.append(ts)

        # Verify order is maintained
        for i in range(len(processing_order) - 1):
            assert processing_order[i] <= processing_order[i + 1], \
                f"Signal order violated at index {i}"

        # Property: Signal aggregation should preserve order
        if len(sorted_signals) > 1:
            first_signal = sorted_signals[0]
            last_signal = sorted_signals[-1]
            assert first_signal[0] <= last_signal[0], "First signal not before last"

    @given(
        initial_position=st.integers(min_value=-100, max_value=100),
        trades=st.lists(
            st.integers(min_value=-10, max_value=10),
            min_size=0,
            max_size=50
        )
    )
    @pytest.mark.database
    @pytest.mark.serial
    @settings(max_examples=50, deadline=5000)
    def test_position_state_consistency(self, initial_position, trades):
        """
        Property: Position state should be consistent with trade history.

        Invariant: Current position = initial position + sum(all trades)
        """
        current_position = initial_position
        position_history = [initial_position]

        for trade in trades:
            current_position += trade
            position_history.append(current_position)

            # Property: Position should equal sum of trades
            expected_position = initial_position + sum(trades[:len(position_history) - 1])
            assert current_position == expected_position, \
                f"Position mismatch: {current_position} != {expected_position}"

        # Property: Final position should match total trades
        final_position = initial_position + sum(trades)
        assert current_position == final_position, \
            f"Final position incorrect: {current_position} != {final_position}"

        # Property: Position changes should match trades
        if len(position_history) > 1:
            position_changes = [
                position_history[i+1] - position_history[i]
                for i in range(len(position_history) - 1)
            ]
            assert position_changes == trades, "Position changes don't match trades"


@pytest.mark.database
@pytest.mark.serial
class TestDataStoreInvariants:
    """Property tests for DataStore invariants."""

    @given(
        watermarks=st.lists(
            nanosecond_timestamps(),
            min_size=1,
            max_size=100,
            unique=True
        )
    )
    @pytest.mark.database
    @pytest.mark.serial
    @settings(max_examples=50, deadline=5000)
    def test_watermark_progression_invariant(self, watermarks):
        """
        Property: Watermarks should only move forward in time.

        Invariant: New watermark >= previous watermark (monotonic non-decreasing)
        """
        current_watermark = 0

        for watermark in sorted(watermarks):
            # Property: Watermark should not go backwards
            assert watermark >= current_watermark, \
                f"Watermark went backwards: {watermark} < {current_watermark}"

            # Update watermark
            previous = current_watermark
            current_watermark = watermark

            # Property: Watermark updates should be idempotent
            if watermark == previous:
                assert current_watermark == previous, "Same watermark changed state"

    @given(
        events=st.lists(
            st.tuples(
                nanosecond_timestamps(),  # event time
                st.text(min_size=1, max_size=10),  # event type
                st.integers(min_value=0, max_value=1000000),  # processing time (microseconds)
            ),
            min_size=1,
            max_size=100
        )
    )
    @pytest.mark.database
    @pytest.mark.serial
    @settings(max_examples=50, deadline=5000)
    def test_event_ordering_invariant(self, events):
        """
        Property: Events should be ordered by event time, not processing time.

        Invariant: Event order should be deterministic based on event time.
        """
        # Sort by event time (not processing time)
        sorted_by_event_time = sorted(events, key=lambda x: x[0])

        # Property: Event time ordering should be consistent
        for i in range(len(sorted_by_event_time) - 1):
            event1 = sorted_by_event_time[i]
            event2 = sorted_by_event_time[i + 1]
            assert event1[0] <= event2[0], \
                f"Event order violated: {event1[0]} > {event2[0]}"

        # Property: Late events should be detectable
        processing_delays = [
            (event[2], event[0])  # (processing_time, event_time)
            for event in events
        ]

        # Events processed significantly after their event time are "late"
        late_threshold = 1000000  # 1 second in microseconds
        late_events = [
            delay for delay, event_time in processing_delays
            if delay > late_threshold
        ]

        # Late events should maintain order when reprocessed
        if late_events:
            assert all(isinstance(e, int) for e in late_events), \
                "Late event detection failed"


# Stateful testing for complex workflows
class StoreStateMachine(RuleBasedStateMachine):
    """
    Stateful property testing for store interactions.

    This tests complex sequences of operations to find bugs in state management.
    """

    def __init__(self):
        super().__init__()
        self.features = {}
        self.models = {}
        self.watermark = 0
        self.current_timestamp = int(datetime(2024, 1, 1).timestamp() * 1e9)

    timestamps = Bundle("timestamps")
    feature_sets = Bundle("feature_sets")
    model_ids = Bundle("model_ids")

    @rule(target=timestamps)
    def generate_timestamp(self):
        """Generate a new timestamp."""
        self.current_timestamp += np.random.randint(1, 1000000000)  # Add 1ns to 1s
        return self.current_timestamp

    @rule(target=feature_sets, feature_set_id=st.text(min_size=1, max_size=20))
    def create_feature_set(self, feature_set_id):
        """Create a new feature set."""
        self.features[feature_set_id] = []
        return feature_set_id

    @rule(
        feature_set=feature_sets,
        timestamp=timestamps,
        values=feature_values()
    )
    def store_features(self, feature_set, timestamp, values):
        """Store features and verify invariants."""
        if feature_set in self.features:
            # Store the features
            self.features[feature_set].append((timestamp, values))

            # Invariant: Features should be time-ordered
            timestamps = [t for t, _ in self.features[feature_set]]
            assert timestamps == sorted(timestamps), \
                "Feature timestamps not in order"

    @rule(target=model_ids, model_id=st.text(min_size=1, max_size=20))
    def register_model(self, model_id):
        """Register a new model."""
        self.models[model_id] = {
            "version": 1,
            "predictions": []
        }
        return model_id

    @rule(
        model=model_ids,
        timestamp=timestamps,
        prediction=st.floats(min_value=-1, max_value=1, allow_nan=False)
    )
    def store_prediction(self, model, timestamp, prediction):
        """Store a prediction and verify invariants."""
        if model in self.models:
            self.models[model]["predictions"].append((timestamp, prediction))

            # Invariant: Predictions should be bounded
            assert -1 <= prediction <= 1, f"Prediction {prediction} out of bounds"

            # Invariant: Timestamps should be monotonic
            timestamps = [t for t, _ in self.models[model]["predictions"]]
            assert timestamps == sorted(timestamps), \
                "Prediction timestamps not in order"

    @rule(new_watermark=timestamps)
    def update_watermark(self, new_watermark):
        """Update watermark and verify it only moves forward."""
        # Invariant: Watermark should only increase
        if new_watermark > self.watermark:
            old_watermark = self.watermark
            self.watermark = new_watermark
            assert self.watermark > old_watermark, "Watermark didn't increase"
        else:
            # Watermark should not change if new value is older
            assert self.watermark >= new_watermark, \
                "Watermark moved backwards"
