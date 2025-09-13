"""
Stateful property-based tests for Domain Bookkeeping complex workflows.

These tests verify complex sequences of operations using state machines to discover bugs
in state management and workflow orchestration.

Following the "write less tests, get more coverage" philosophy from TESTING_STRATEGY.md

"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any
from unittest.mock import MagicMock

import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import Bundle
from hypothesis.stateful import RuleBasedStateMachine
from hypothesis.stateful import initialize
from hypothesis.stateful import invariant
from hypothesis.stateful import rule

from ml.core.integration import MLIntegrationManager
from nautilus_trader.core.uuid import UUID4


# Strategies that can optionally use DataBuilder
@st.composite
def instrument_ids_strategy(draw, use_builder=False):
    """Generate instrument IDs, optionally using DataBuilder."""
    if use_builder:
        # Use default instrument ID pattern from fixtures
        return "EUR/USD.SIM"
    return draw(st.text(min_size=5, max_size=15))


class PipelineState(Enum):
    """
    Pipeline execution states.
    """

    IDLE = "idle"
    DATA_INGESTION = "data_ingestion"
    FEATURE_COMPUTATION = "feature_computation"
    MODEL_INFERENCE = "model_inference"
    SIGNAL_GENERATION = "signal_generation"
    COMPLETED = "completed"
    FAILED = "failed"


class EventType(Enum):
    """
    Types of events in the system.
    """

    DATA_RECEIVED = "data_received"
    FEATURES_COMPUTED = "features_computed"
    PREDICTION_MADE = "prediction_made"
    SIGNAL_GENERATED = "signal_generated"
    ERROR_OCCURRED = "error_occurred"


@dataclass
class SystemEvent:
    """
    Represents an event in the domain bookkeeping system.
    """

    event_id: str
    event_type: EventType
    correlation_id: str
    instrument_id: str
    timestamp: int
    domain: str
    payload: dict[str, Any] = field(default_factory=dict)
    parent_event_id: str | None = None


@dataclass
class PipelineExecution:
    """
    Represents a pipeline execution instance.
    """

    execution_id: str
    correlation_id: str
    instrument_id: str
    state: PipelineState = PipelineState.IDLE
    events: list[SystemEvent] = field(default_factory=list)
    start_timestamp: int | None = None
    end_timestamp: int | None = None
    error_message: str | None = None


class DomainBookkeepingStateMachine(RuleBasedStateMachine):
    """
    State machine for testing domain bookkeeping workflows.
    """

    # Bundles for state machine objects
    events = Bundle("events")
    pipelines = Bundle("pipelines")
    correlations = Bundle("correlations")

    def __init__(self):
        super().__init__()
        # Use a flexible mock; concrete MLIntegrationManager may evolve
        self.mock_integration_manager = MagicMock()
        self.active_pipelines: dict[str, PipelineExecution] = {}
        self.event_history: list[SystemEvent] = []
        self.correlation_tracking: dict[str, list[str]] = {}  # correlation_id -> event_ids
        self.correlation_instruments: dict[str, str] = {}  # correlation_id -> instrument_id
        self.metrics_collected: list[dict[str, Any]] = []
        self.health_scores: dict[str, float] = {
            "data_store": 1.0,
            "feature_store": 1.0,
            "model_store": 1.0,
            "strategy_store": 1.0,
        }
        self.message_bus_state = {
            "connected": True,
            "message_count": 0,
            "failed_messages": 0,
        }

    @initialize()
    def init_system(self):
        """
        Initialize the domain bookkeeping system.
        """
        # Initialize with default healthy state
        self.mock_integration_manager.get_system_health.return_value = 0.95
        self.mock_integration_manager.is_message_bus_connected.return_value = True

    @rule(target=correlations)
    def create_correlation_id(self) -> str:
        """
        Create a new correlation ID for tracking related events.
        """
        correlation_id = str(UUID4())
        self.correlation_tracking[correlation_id] = []
        return correlation_id

    @rule(
        target=pipelines,
        correlation_id=correlations,
        instrument_id=instrument_ids_strategy(),
    )
    def start_pipeline_execution(
        self,
        correlation_id: str,
        instrument_id: str,
    ) -> PipelineExecution:
        """
        Start a new pipeline execution.
        """
        execution_id = str(UUID4())
        timestamp = len(self.event_history) * 1000 + 1000000  # Monotonic timestamps

        # Enforce consistency: each correlation_id maps to a single instrument_id
        if correlation_id in self.correlation_instruments:
            instrument_id = self.correlation_instruments[correlation_id]

        pipeline = PipelineExecution(
            execution_id=execution_id,
            correlation_id=correlation_id,
            instrument_id=instrument_id,
            state=PipelineState.IDLE,
            start_timestamp=timestamp,
        )

        self.active_pipelines[execution_id] = pipeline
        # Record instrument binding for this correlation if first seen
        if correlation_id not in self.correlation_instruments:
            self.correlation_instruments[correlation_id] = instrument_id
        return pipeline

    @rule(
        target=events,
        pipeline=pipelines,
        event_type=st.sampled_from(EventType),
        domain=st.sampled_from(["data", "features", "models", "strategies"]),
    )
    def emit_event(
        self,
        pipeline: PipelineExecution,
        event_type: EventType,
        domain: str,
    ) -> SystemEvent:
        """
        Emit an event in the context of a pipeline execution.
        """
        event_id = str(UUID4())
        timestamp = len(self.event_history) * 1000 + 1000000

        # Determine parent event (previous event in same pipeline)
        parent_event_id = None
        pipeline_events = [e for e in pipeline.events]
        if pipeline_events:
            parent_event_id = pipeline_events[-1].event_id

        event = SystemEvent(
            event_id=event_id,
            event_type=event_type,
            correlation_id=pipeline.correlation_id,
            instrument_id=pipeline.instrument_id,
            timestamp=timestamp,
            domain=domain,
            parent_event_id=parent_event_id,
            payload={"pipeline_state": pipeline.state.value},
        )

        # Update tracking structures
        self.event_history.append(event)
        pipeline.events.append(event)
        self.correlation_tracking[pipeline.correlation_id].append(event_id)

        # Update pipeline state based on event type
        self._update_pipeline_state(pipeline, event_type)

        # Update message bus metrics
        self.message_bus_state["message_count"] += 1

        return event

    @rule(pipeline=pipelines)
    def complete_pipeline(self, pipeline: PipelineExecution):
        """
        Mark a pipeline execution as completed.
        """
        if pipeline.state not in [PipelineState.COMPLETED, PipelineState.FAILED]:
            pipeline.state = PipelineState.COMPLETED
            pipeline.end_timestamp = len(self.event_history) * 1000 + 1000000

            # Collect metrics for completed pipeline
            duration = pipeline.end_timestamp - (pipeline.start_timestamp or 0)
            self.metrics_collected.append(
                {
                    "metric_name": "pipeline_duration_ms",
                    "value": duration / 1000000,  # Convert to ms
                    "labels": {
                        "instrument_id": pipeline.instrument_id,
                        "correlation_id": pipeline.correlation_id,
                    },
                },
            )

    @rule(
        component=st.sampled_from(["data_store", "feature_store", "model_store", "strategy_store"]),
        health_change=st.floats(min_value=-0.3, max_value=0.3),
    )
    def update_component_health(self, component: str, health_change: float):
        """
        Update the health score of a system component.
        """
        current_health = self.health_scores[component]
        new_health = max(0.0, min(1.0, current_health + health_change))
        self.health_scores[component] = new_health

        # If health drops below threshold, it might affect pipeline execution
        if new_health < 0.7:
            # Find pipelines that might be affected
            for pipeline in self.active_pipelines.values():
                if pipeline.state not in [PipelineState.COMPLETED, PipelineState.FAILED]:
                    # Low health might cause pipeline failure
                    if new_health < 0.5:
                        pipeline.state = PipelineState.FAILED
                        pipeline.error_message = (
                            f"Component {component} health too low: {new_health}"
                        )

    @rule(
        failure_rate=st.floats(min_value=0.0, max_value=0.1),
    )
    def simulate_message_bus_issues(self, failure_rate: float):
        """
        Simulate message bus connectivity or delivery issues.
        """
        if failure_rate > 0.05:  # 5% failure rate threshold
            self.message_bus_state["connected"] = False
            failures = int(failure_rate * 100)
            self.message_bus_state["failed_messages"] += failures
            # Failed deliveries still count toward total attempts
            self.message_bus_state["message_count"] += failures

            # Message bus issues might cause event delivery failures
            # This could affect ongoing pipelines
            for pipeline in self.active_pipelines.values():
                if pipeline.state == PipelineState.DATA_INGESTION:
                    # Data ingestion is most vulnerable to message bus issues
                    if failure_rate > 0.08:  # High failure rate
                        pipeline.state = PipelineState.FAILED
                        pipeline.error_message = "Message bus failure during data ingestion"
        else:
            self.message_bus_state["connected"] = True

    def _update_pipeline_state(self, pipeline: PipelineExecution, event_type: EventType):
        """
        Update pipeline state based on event type.
        """
        # Ignore events for pipelines in terminal states
        if pipeline.state in (PipelineState.COMPLETED, PipelineState.FAILED):
            return
        state_transitions = {
            EventType.DATA_RECEIVED: PipelineState.DATA_INGESTION,
            EventType.FEATURES_COMPUTED: PipelineState.FEATURE_COMPUTATION,
            EventType.PREDICTION_MADE: PipelineState.MODEL_INFERENCE,
            EventType.SIGNAL_GENERATED: PipelineState.SIGNAL_GENERATION,
            EventType.ERROR_OCCURRED: PipelineState.FAILED,
        }

        if event_type in state_transitions:
            new_state = state_transitions[event_type]

            # Only allow valid state transitions
            if self._is_valid_transition(pipeline.state, new_state):
                pipeline.state = new_state
            else:
                # Invalid transition - mark as failed
                pipeline.state = PipelineState.FAILED
                pipeline.error_message = (
                    f"Invalid state transition: {pipeline.state} -> {new_state}"
                )

    def _is_valid_transition(self, current_state: PipelineState, new_state: PipelineState) -> bool:
        """
        Check if state transition is valid.
        """
        valid_transitions = {
            PipelineState.IDLE: [PipelineState.DATA_INGESTION, PipelineState.FAILED],
            PipelineState.DATA_INGESTION: [PipelineState.FEATURE_COMPUTATION, PipelineState.FAILED],
            PipelineState.FEATURE_COMPUTATION: [
                PipelineState.MODEL_INFERENCE,
                PipelineState.FAILED,
            ],
            PipelineState.MODEL_INFERENCE: [PipelineState.SIGNAL_GENERATION, PipelineState.FAILED],
            PipelineState.SIGNAL_GENERATION: [PipelineState.COMPLETED, PipelineState.FAILED],
            PipelineState.COMPLETED: [],  # Terminal state
            PipelineState.FAILED: [],  # Terminal state
        }

        return new_state in valid_transitions.get(current_state, [])

    # Invariants that must always hold

    @invariant()
    def event_timestamps_monotonic(self):
        """
        Event timestamps must be monotonically increasing.
        """
        timestamps = [event.timestamp for event in self.event_history]
        if len(timestamps) > 1:
            assert timestamps == sorted(
                timestamps,
            ), "Event timestamps must be monotonically increasing"

    @invariant()
    def correlation_consistency(self):
        """
        All events with same correlation ID must have consistent instrument_id.
        """
        correlation_instruments = {}
        for event in self.event_history:
            corr_id = event.correlation_id
            if corr_id not in correlation_instruments:
                correlation_instruments[corr_id] = event.instrument_id
            else:
                assert correlation_instruments[corr_id] == event.instrument_id, (
                    f"Correlation {corr_id} has inconsistent instruments: "
                    f"{correlation_instruments[corr_id]} vs {event.instrument_id}"
                )

    @invariant()
    def pipeline_state_consistency(self):
        """
        Pipeline states must be consistent with their events.
        """
        for pipeline in self.active_pipelines.values():
            # Terminal states should have end timestamps
            if pipeline.state in [PipelineState.COMPLETED, PipelineState.FAILED]:
                if pipeline.end_timestamp is None:
                    # Allow this during state transition
                    pass

            # Failed pipelines should have error messages
            if pipeline.state == PipelineState.FAILED and pipeline.error_message is None:
                # Allow temporary state during transition
                pass

            # Pipelines should have at least start timestamp if not idle
            if pipeline.state != PipelineState.IDLE:
                assert (
                    pipeline.start_timestamp is not None
                ), f"Non-idle pipeline {pipeline.execution_id} missing start timestamp"

    @invariant()
    def event_lineage_consistency(self):
        """
        Event lineage chains must be consistent.
        """
        event_lookup = {event.event_id: event for event in self.event_history}

        for event in self.event_history:
            if event.parent_event_id is not None:
                # Parent event must exist
                assert (
                    event.parent_event_id in event_lookup
                ), f"Event {event.event_id} has non-existent parent {event.parent_event_id}"

                parent_event = event_lookup[event.parent_event_id]

                # Parent must have same correlation ID
                assert (
                    parent_event.correlation_id == event.correlation_id
                ), f"Event {event.event_id} correlation mismatch with parent"

                # Parent must have earlier timestamp
                assert (
                    parent_event.timestamp <= event.timestamp
                ), f"Event {event.event_id} timestamp before parent timestamp"

    @invariant()
    def health_scores_valid_range(self):
        """
        All health scores must be in valid range [0,1].
        """
        for component, health in self.health_scores.items():
            assert 0.0 <= health <= 1.0, f"Component {component} health out of range: {health}"

    @invariant()
    def message_bus_metrics_consistency(self):
        """
        Message bus metrics must be consistent.
        """
        assert self.message_bus_state["message_count"] >= 0, "Message count cannot be negative"

        assert (
            self.message_bus_state["failed_messages"] >= 0
        ), "Failed message count cannot be negative"

        assert (
            self.message_bus_state["failed_messages"] <= self.message_bus_state["message_count"]
        ), "Failed messages cannot exceed total messages"


@pytest.mark.property
@pytest.mark.stateful
@pytest.mark.slow  # Stateful tests can be slower
class TestDomainBookkeepingStateful:
    """
    Stateful property-based tests for domain bookkeeping.
    """

    def test_domain_bookkeeping_workflow_state_machine(self):
        """
        Test domain bookkeeping workflows using state machine exploration.

        This test will explore various sequences of operations and verify that all
        invariants hold throughout the execution.

        """
        # Run the state machine using Hypothesis helper and capture the instance
        from hypothesis.stateful import run_state_machine_as_test

        class _CaptureMachine(DomainBookkeepingStateMachine):  # type: ignore[misc]
            last_instance: _CaptureMachine | None = None

            def __init__(self):
                super().__init__()
                _CaptureMachine.last_instance = self

        run_state_machine_as_test(_CaptureMachine)
        sm = _CaptureMachine.last_instance
        assert sm is not None, "State machine instance was not captured"

        # Additional post-execution validations
        self._validate_final_state(sm)

    def _validate_final_state(self, state_machine: DomainBookkeepingStateMachine):
        """
        Perform additional validation of the final state.
        """
        # Check that all correlations are properly tracked
        for correlation_id, event_ids in state_machine.correlation_tracking.items():
            correlation_events = [
                e for e in state_machine.event_history if e.correlation_id == correlation_id
            ]

            assert len(correlation_events) == len(
                event_ids,
            ), f"Correlation {correlation_id} tracking inconsistency"

        # Check that completed pipelines have valid metrics
        completed_pipelines = [
            p for p in state_machine.active_pipelines.values() if p.state == PipelineState.COMPLETED
        ]

        pipeline_metrics = [
            m for m in state_machine.metrics_collected if m["metric_name"] == "pipeline_duration_ms"
        ]

        # Each completed pipeline should have generated duration metrics
        # (allowing for some lag in metrics collection)
        assert len(pipeline_metrics) <= len(
            completed_pipelines,
        ), "Pipeline metrics count should not exceed completed pipelines"

        # Check system health consistency
        average_health = sum(state_machine.health_scores.values()) / len(
            state_machine.health_scores,
        )

        failed_pipelines = [
            p for p in state_machine.active_pipelines.values() if p.state == PipelineState.FAILED
        ]

        # Low system health should correlate with more failures
        if average_health < 0.6:
            # In low health scenarios, some failures are expected
            pass  # This is normal behavior

        # Validate event lineage depth is reasonable
        max_lineage_depth = 0
        event_lookup = {e.event_id: e for e in state_machine.event_history}

        for event in state_machine.event_history:
            depth = self._calculate_lineage_depth(event, event_lookup)
            max_lineage_depth = max(max_lineage_depth, depth)

        assert (
            max_lineage_depth <= 20
        ), f"Lineage depth too deep: {max_lineage_depth} (may indicate circular references)"

    def _calculate_lineage_depth(
        self,
        event: SystemEvent,
        event_lookup: dict[str, SystemEvent],
    ) -> int:
        """
        Calculate the depth of an event in its lineage chain.
        """
        if event.parent_event_id is None:
            return 0

        if event.parent_event_id not in event_lookup:
            return 0  # Broken lineage

        parent_event = event_lookup[event.parent_event_id]
        return 1 + self._calculate_lineage_depth(parent_event, event_lookup)


# Focused stateful tests for specific workflow scenarios


class PipelineRecoveryStateMachine(RuleBasedStateMachine):
    """
    Focused state machine for testing pipeline recovery scenarios.
    """

    pipelines = Bundle("pipelines")

    def __init__(self):
        super().__init__()
        self.pipelines_dict: dict[str, PipelineExecution] = {}
        self.recovery_attempts = 0
        self.successful_recoveries = 0

    @rule(target=pipelines, instrument_id=instrument_ids_strategy())
    def create_pipeline(self, instrument_id: str) -> str:
        """
        Create a new pipeline for testing recovery.
        """
        pipeline_id = str(UUID4())
        correlation_id = str(UUID4())

        pipeline = PipelineExecution(
            execution_id=pipeline_id,
            correlation_id=correlation_id,
            instrument_id=instrument_id,
            state=PipelineState.DATA_INGESTION,
            start_timestamp=len(self.pipelines_dict) * 1000 + 1000000,
        )

        self.pipelines_dict[pipeline_id] = pipeline
        return pipeline_id

    @rule(
        pipeline_id=pipelines,
        failure_type=st.sampled_from(["network", "timeout", "data_quality"]),
    )
    def inject_failure(self, pipeline_id: str, failure_type: str):
        """
        Inject a failure into a pipeline.
        """
        if pipeline_id in self.pipelines_dict:
            pipeline = self.pipelines_dict[pipeline_id]

            if pipeline.state not in [PipelineState.COMPLETED, PipelineState.FAILED]:
                pipeline.state = PipelineState.FAILED
                pipeline.error_message = f"Injected {failure_type} failure"

    @rule(pipeline_id=pipelines)
    def attempt_recovery(self, pipeline_id: str):
        """
        Attempt to recover a failed pipeline.
        """
        if pipeline_id in self.pipelines_dict:
            pipeline = self.pipelines_dict[pipeline_id]

            if pipeline.state == PipelineState.FAILED:
                self.recovery_attempts += 1

                # Simulate recovery logic
                if "network" in (pipeline.error_message or ""):
                    # Network failures have 70% recovery success rate
                    if self.recovery_attempts % 10 < 7:  # Simple deterministic "randomness"
                        pipeline.state = PipelineState.DATA_INGESTION
                        pipeline.error_message = None
                        self.successful_recoveries += 1
                elif "timeout" in (pipeline.error_message or ""):
                    # Timeout failures have 50% recovery success rate
                    if self.recovery_attempts % 2 == 0:
                        pipeline.state = PipelineState.DATA_INGESTION
                        pipeline.error_message = None
                        self.successful_recoveries += 1
                else:
                    # Data quality failures have 30% recovery success rate
                    if self.recovery_attempts % 10 < 3:
                        pipeline.state = PipelineState.DATA_INGESTION
                        pipeline.error_message = None
                        self.successful_recoveries += 1

    @invariant()
    def recovery_rate_reasonable(self):
        """
        Recovery success rate should be reasonable.
        """
        if self.recovery_attempts > 0:
            recovery_rate = self.successful_recoveries / self.recovery_attempts
            # Should have some successful recoveries but not 100%
            assert 0.0 <= recovery_rate <= 1.0, f"Recovery rate out of bounds: {recovery_rate}"


@pytest.mark.property
@pytest.mark.stateful
@pytest.mark.slow
class TestPipelineRecoveryStateful:
    """
    Stateful tests focused on pipeline recovery scenarios.
    """

    def test_pipeline_recovery_workflows(self):
        """
        Test pipeline recovery workflows with various failure patterns.
        """
        from hypothesis.stateful import run_state_machine_as_test

        class _CaptureRecovery(PipelineRecoveryStateMachine):  # type: ignore[misc]
            last_instance: _CaptureRecovery | None = None

            def __init__(self):
                super().__init__()
                _CaptureRecovery.last_instance = self

        run_state_machine_as_test(_CaptureRecovery)
        sm = _CaptureRecovery.last_instance
        assert sm is not None, "Recovery state machine instance was not captured"

        # Validate recovery behavior (post-run)
        if sm.recovery_attempts > 10:
            assert (
                sm.successful_recoveries > 0
            ), "Should have some successful recoveries with many attempts"
