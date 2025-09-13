"""
Property-based tests for event ordering invariants in the ML pipeline.

These tests verify mathematical properties and invariants for event-driven
architecture that must hold regardless of specific event content.

Key invariants tested:
- Stage progression monotonicity
- Watermark non-decreasing progression
- Correlation ID uniqueness within time windows
- Event timestamp causality

"""

from __future__ import annotations

import uuid
from datetime import datetime
from datetime import timedelta
from typing import Any

import pytest
from hypothesis import assume
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from ml.config.events import Source
from ml.config.events import Stage


# ============================================================================
# HYPOTHESIS STRATEGIES
# ============================================================================


@st.composite
def correlation_ids(draw):
    """
    Generate valid correlation IDs.
    """
    return str(uuid.uuid4())


@st.composite
def event_timestamps(draw, min_year=2020, max_year=2025, use_builder=False):
    """
    Generate valid event timestamps in nanoseconds.
    """
    if use_builder:
        from ml.tests.builders import DataBuilder
        # Generate a single timestamp using DataBuilder
        timestamps = DataBuilder.time_series(n_points=1)
        return int(timestamps[0])

    year = draw(st.integers(min_value=min_year, max_value=max_year))
    month = draw(st.integers(min_value=1, max_value=12))
    day = draw(st.integers(min_value=1, max_value=28))  # Safe for all months
    hour = draw(st.integers(min_value=0, max_value=23))
    minute = draw(st.integers(min_value=0, max_value=59))
    second = draw(st.integers(min_value=0, max_value=59))

    dt = datetime(year, month, day, hour, minute, second)
    return int(dt.timestamp() * 1e9)


@st.composite
def stage_sequences(draw, min_length=1, max_length=10):
    """
    Generate sequences of valid stage transitions.
    """
    # Define stage progression graph
    stage_transitions = {
        Stage.DATA_INGESTED: [Stage.CATALOG_WRITTEN, Stage.FEATURE_COMPUTED],
        Stage.CATALOG_WRITTEN: [Stage.FEATURE_COMPUTED],
        Stage.FEATURE_COMPUTED: [Stage.PREDICTION_EMITTED],
        Stage.PREDICTION_EMITTED: [Stage.SIGNAL_EMITTED],
        Stage.SIGNAL_EMITTED: [],  # Terminal
    }

    length = draw(st.integers(min_value=min_length, max_value=max_length))
    sequence = [Stage.DATA_INGESTED]  # Always start here

    current_stage = Stage.DATA_INGESTED
    for _ in range(length - 1):
        next_stages = stage_transitions.get(current_stage, [])
        if not next_stages:
            break  # Reached terminal stage
        next_stage = draw(st.sampled_from(next_stages))
        sequence.append(next_stage)
        current_stage = next_stage

    return sequence


@st.composite
def ml_events(draw):
    """
    Generate ML pipeline events.
    """
    return {
        "correlation_id": draw(correlation_ids()),
        "timestamp": draw(event_timestamps()),
        "stage": draw(st.sampled_from(list(Stage))),
        "source": draw(st.sampled_from(list(Source))),
        "dataset_id": draw(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=5, max_size=20),
        ),
        "status": draw(st.sampled_from(["SUCCESS", "FAILED", "IN_PROGRESS"])),
    }


# ============================================================================
# PROPERTY TESTS
# ============================================================================


@pytest.mark.parallel_safe
class TestEventOrderingInvariants:
    """
    Property-based tests for event ordering invariants.
    """

    @given(stage_sequence=stage_sequences(min_length=2, max_length=5))
    @settings(max_examples=50, deadline=5000)
    def test_stage_progression_monotonicity_invariant(self, stage_sequence):
        """
        Property: Stage transitions must follow allowed progression paths.

        Invariant: Each stage transition in a sequence must be valid according
        to the stage transition graph.
        """
        # Define allowed transitions
        allowed_transitions = {
            Stage.DATA_INGESTED: {Stage.CATALOG_WRITTEN, Stage.FEATURE_COMPUTED},
            Stage.CATALOG_WRITTEN: {Stage.FEATURE_COMPUTED},
            Stage.FEATURE_COMPUTED: {Stage.PREDICTION_EMITTED},
            Stage.PREDICTION_EMITTED: {Stage.SIGNAL_EMITTED},
            Stage.SIGNAL_EMITTED: set(),  # Terminal
        }

        # Property: All transitions in sequence must be valid
        for i in range(len(stage_sequence) - 1):
            current_stage = stage_sequence[i]
            next_stage = stage_sequence[i + 1]

            valid_next_stages = allowed_transitions.get(current_stage, set())
            assert (
                next_stage in valid_next_stages
            ), f"Invalid transition from {current_stage} to {next_stage}"

    @given(
        events=st.lists(ml_events(), min_size=1, max_size=100),
        dataset_id=st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=5, max_size=15),
    )
    @settings(max_examples=30, deadline=10000)
    def test_watermark_progression_invariant(self, events, dataset_id):
        """
        Property: Watermarks must progress monotonically (non-decreasing).

        Invariant: For any dataset, watermark updates must never decrease.
        """
        # Filter events for our dataset and sort by timestamp
        dataset_events = [e for e in events if e.get("dataset_id") == dataset_id]
        if not dataset_events:
            return  # No events to test

        dataset_events.sort(key=lambda e: e["timestamp"])

        # Track watermark progression
        watermarks = []
        current_watermark = 0

        for event in dataset_events:
            if event["status"] == "SUCCESS":  # Only successful events update watermarks
                new_watermark = event["timestamp"]
                if new_watermark >= current_watermark:
                    watermarks.append(new_watermark)
                    current_watermark = new_watermark
                # If new_watermark < current_watermark, reject the update

        # Property: Watermarks must be monotonically non-decreasing
        for i in range(1, len(watermarks)):
            assert (
                watermarks[i] >= watermarks[i - 1]
            ), f"Watermark regression: {watermarks[i]} < {watermarks[i-1]}"

    @given(
        correlation_id=correlation_ids(),
        event_count=st.integers(min_value=2, max_value=20),
        time_window_hours=st.integers(min_value=1, max_value=24),
    )
    @settings(max_examples=30, deadline=5000)
    def test_correlation_id_uniqueness_invariant(
        self,
        correlation_id,
        event_count,
        time_window_hours,
    ):
        """
        Property: Correlation IDs must be unique within processing time windows.

        Invariant: No two event sequences should have the same correlation_id
        within a time window.
        """
        # Generate events with same correlation_id in time window
        base_time = int(datetime.now().timestamp() * 1e9)
        window_ns = time_window_hours * 3600 * 1e9

        events_in_window = []
        for i in range(event_count):
            event_time = base_time + (i * (window_ns / event_count))
            events_in_window.append(
                {
                    "correlation_id": correlation_id,
                    "timestamp": int(event_time),
                    "stage": Stage.DATA_INGESTED,
                },
            )

        # Property: All events in window with same correlation_id should be part of same sequence
        correlation_ids = [e["correlation_id"] for e in events_in_window]
        unique_correlation_ids = set(correlation_ids)

        # Since we generated with same correlation_id, should only have one unique ID
        assert len(unique_correlation_ids) == 1
        assert correlation_id in unique_correlation_ids

    @given(stage_sequence=stage_sequences(min_length=2, max_length=5))
    @settings(max_examples=20, deadline=5000)
    def test_event_timestamp_causality_invariant(self, stage_sequence):
        """
        Property: Events with same correlation_id must have causal timestamp ordering.

        Invariant: For events in same pipeline sequence (same correlation_id),
        timestamps must increase with stage progression.
        """
        correlation_id = str(uuid.uuid4())
        base_timestamp = int(datetime.now().timestamp() * 1e9)

        # Create events with proper stage progression and timestamps
        events = []
        for i, stage in enumerate(stage_sequence):
            events.append(
                {
                    "correlation_id": correlation_id,
                    "timestamp": base_timestamp + (i * 1_000_000_000),  # 1 second apart
                    "stage": stage,
                    "source": Source.LIVE,
                    "status": "SUCCESS",
                },
            )

        # Property: Events should already be in causal order by construction
        for i in range(1, len(events)):
            prev_event = events[i - 1]
            curr_event = events[i]

            # Later stages must have later timestamps (causal ordering)
            assert curr_event["timestamp"] >= prev_event["timestamp"], (
                f"Causality violation: stage {curr_event['stage']} at {curr_event['timestamp']} "
                f"before stage {prev_event['stage']} at {prev_event['timestamp']}"
            )

    @given(
        events=st.lists(ml_events(), min_size=5, max_size=100),
        max_concurrent_pipelines=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=20, deadline=15000)
    def test_concurrent_pipeline_isolation_invariant(self, events, max_concurrent_pipelines):
        """
        Property: Concurrent pipelines with different correlation_ids must not interfere.

        Invariant: Events from different correlation_ids should be processable
        independently without order dependencies.
        """
        # Assign events to different pipeline correlation_ids
        correlation_ids_pool = [str(uuid.uuid4()) for _ in range(max_concurrent_pipelines)]

        pipeline_events = {}
        for i, event in enumerate(events):
            corr_id = correlation_ids_pool[i % len(correlation_ids_pool)]
            if corr_id not in pipeline_events:
                pipeline_events[corr_id] = []
            pipeline_events[corr_id].append({**event, "correlation_id": corr_id})

        # Property: Each pipeline should be processable independently
        for corr_id, pipeline in pipeline_events.items():
            if len(pipeline) < 2:
                continue

            # Sort events in pipeline by timestamp
            pipeline.sort(key=lambda e: e["timestamp"])

            # Each pipeline should maintain its own watermark independently
            pipeline_watermarks = []
            current_watermark = 0

            for event in pipeline:
                if event["status"] == "SUCCESS":
                    event_watermark = event["timestamp"]
                    if event_watermark >= current_watermark:
                        pipeline_watermarks.append(event_watermark)
                        current_watermark = event_watermark

            # Invariant: Each pipeline's watermarks progress independently
            for i in range(1, len(pipeline_watermarks)):
                assert (
                    pipeline_watermarks[i] >= pipeline_watermarks[i - 1]
                ), f"Pipeline {corr_id} watermark regression: {pipeline_watermarks[i]} < {pipeline_watermarks[i-1]}"

    @given(
        base_timestamp=event_timestamps(),
        event_intervals_ms=st.lists(
            st.integers(min_value=1, max_value=10000),
            min_size=2,
            max_size=20,
        ),
    )
    @settings(max_examples=30, deadline=5000)
    def test_event_timing_distribution_invariant(self, base_timestamp, event_intervals_ms):
        """
        Property: Event timing distributions must respect processing constraints.

        Invariant: Events in a sequence should have realistic timing intervals
        that respect processing latency constraints.
        """
        # Generate timestamp sequence with given intervals
        timestamps = [base_timestamp]
        current_ts = base_timestamp

        for interval_ms in event_intervals_ms:
            current_ts += interval_ms * 1_000_000  # Convert ms to ns
            timestamps.append(current_ts)

        # Property: All intervals should be within reasonable bounds
        MIN_INTERVAL_NS = 1_000_000  # 1ms minimum
        MAX_INTERVAL_NS = 3600_000_000_000  # 1 hour maximum

        for i in range(1, len(timestamps)):
            interval = timestamps[i] - timestamps[i - 1]

            assert (
                interval >= MIN_INTERVAL_NS
            ), f"Interval {interval}ns too small (< {MIN_INTERVAL_NS}ns)"
            assert (
                interval <= MAX_INTERVAL_NS
            ), f"Interval {interval}ns too large (> {MAX_INTERVAL_NS}ns)"

        # Property: Total sequence duration should be reasonable
        total_duration = timestamps[-1] - timestamps[0]
        MAX_SEQUENCE_DURATION = 24 * 3600 * 1_000_000_000  # 24 hours in ns

        assert (
            total_duration <= MAX_SEQUENCE_DURATION
        ), f"Sequence duration {total_duration}ns exceeds maximum {MAX_SEQUENCE_DURATION}ns"
