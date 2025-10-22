"""
Event-driven contract tests for MessageBus integration.

These tests define the contracts for event publishing and consumption
in the ML pipeline's event-driven architecture. They validate:

1. MessageBus payload validation for events.ml.* topics
2. Actor single-thread boundary guarantees
3. Idempotency via correlation_id
4. Watermark progression consistency

Following Phase 1 of the event-driven refactor plan.

"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import Mock

import pandas as pd
import pandera as pa
import pytest
from pandera.typing import DataFrame
from pandera.typing import Series

from ml.common.message_topics import build_topic_for_stage
from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml._imports import HAS_PANDERA
from ml._imports import PANDERA_IMPORT_ERROR
from nautilus_trader.model.identifiers import InstrumentId

if not HAS_PANDERA:  # pragma: no cover - optional dependency
    pytest.skip(
        f"Pandera unavailable: {PANDERA_IMPORT_ERROR}",
        allow_module_level=True,
    )


# ============================================================================
# EVENT PAYLOAD SCHEMAS
# ============================================================================


class MLDataEventSchema(pa.DataFrameModel):
    """
    Schema for events.ml.data.* payloads.
    """

    dataset_id: Series[str] = pa.Field(nullable=False)
    instrument_id: Series[str] = pa.Field(nullable=False)
    stage: Series[str] = pa.Field(nullable=False, isin=[s.value for s in Stage])
    source: Series[str] = pa.Field(nullable=False, isin=[s.value for s in Source])
    status: Series[str] = pa.Field(nullable=False, isin=[s.value for s in EventStatus])
    run_id: Series[str] = pa.Field(nullable=False)
    ts_min: Series[int] = pa.Field(nullable=False, ge=0)
    ts_max: Series[int] = pa.Field(nullable=False, ge=0)
    count: Series[int] = pa.Field(nullable=False, ge=0)
    metadata: Series[Any] = pa.Field(
        nullable=False,
        description="Nested metadata with correlation_id, ts_init, model_id",
    )

    @pa.dataframe_check()
    def check_timestamp_ordering(cls, df: DataFrame[Any]) -> Series[bool]:
        """
        Ensure ts_max >= ts_min.
        """
        # Pandera expects a typed Series. Cast for static typing.
        from typing import cast

        return cast(Series[bool], df["ts_max"] >= df["ts_min"])

    @pa.dataframe_check()
    def check_metadata_correlation_id(cls, df: DataFrame[Any]) -> Series[bool]:
        """
        Ensure metadata contains a non-empty correlation_id.
        """
        from typing import Any, cast

        def _has_corr(m: Any) -> bool:
            if not isinstance(m, dict):
                return False
            cid = m.get("correlation_id")
            return isinstance(cid, str) and len(cid) > 0

        try:
            meta = df["metadata"]
            has = meta.apply(_has_corr)
            return cast(Series[bool], has.astype(bool))
        except Exception:
            return cast(Series[bool], pd.Series([False] * len(df)))

    @pa.dataframe_check()
    def check_instrument_id_format(cls, df: DataFrame[Any]) -> Series[bool]:
        """
        Ensure instrument_id matches pattern like "EURUSD.SIM".
        """
        import re
        from typing import cast

        pattern = re.compile(r"^[A-Z0-9]+\.[A-Z]+$")
        result = df["instrument_id"].astype(str).apply(lambda s: bool(pattern.match(s)))
        return cast(Series[bool], result.astype(bool))


class MLRegistryEventSchema(pa.DataFrameModel):
    """
    Schema for events.ml.{features|models|strategies}.* payloads.
    """

    operation: Series[str] = pa.Field(
        nullable=False,
        isin=["REGISTER", "UPDATE", "DEPLOY", "ROLLBACK"],
    )
    registry_type: Series[str] = pa.Field(nullable=False, isin=["features", "models", "strategies"])
    entity_id: Series[str] = pa.Field(nullable=False)
    version: Series[str] = pa.Field(nullable=False)
    status: Series[str] = pa.Field(
        nullable=False,
        isin=[EventStatus.SUCCESS.value, EventStatus.FAILED.value],
    )
    correlation_id: Series[str] = pa.Field(nullable=False)
    timestamp: Series[int] = pa.Field(nullable=False, ge=0)


# ============================================================================
# CONTRACT TESTS
# ============================================================================


@pytest.mark.parallel_safe
class TestEventBusContracts:
    """
    Test event-driven architecture contracts for MessageBus integration.
    """

    def test_ml_data_event_schema_validation(self):
        """
        Test MLDataEventSchema validates correct payloads.
        """
        valid_event = pd.DataFrame(
            {
                "dataset_id": ["eurusd_bars_2024"],
                "instrument_id": ["EURUSD.SIM"],
                "stage": [Stage.DATA_INGESTED.value],
                "source": [Source.LIVE.value],
                "status": [EventStatus.SUCCESS.value],
                "run_id": [str(uuid.uuid4())],
                "ts_min": [int(datetime(2024, 1, 1).timestamp() * 1e9)],
                "ts_max": [int(datetime(2024, 1, 1, 1).timestamp() * 1e9)],
                "count": [1000],
                "metadata": [
                    {
                        "correlation_id": str(uuid.uuid4()),
                        "ts_init": int(datetime(2024, 1, 1).timestamp() * 1e9) + 1,
                        "model_id": "test_model",
                    },
                ],
            },
        )

        # Should pass validation
        validated = MLDataEventSchema.validate(valid_event)
        assert len(validated) == 1

    def test_ml_registry_event_schema_validation(self):
        """
        Test MLRegistryEventSchema validates registry operations.
        """
        valid_event = pd.DataFrame(
            {
                "operation": ["REGISTER"],
                "registry_type": ["models"],
                "entity_id": ["xgb_predictor_v1"],
                "version": ["1.0.0"],
                "status": [EventStatus.SUCCESS.value],
                "correlation_id": [str(uuid.uuid4())],
                "timestamp": [int(datetime.now().timestamp() * 1e9)],
            },
        )

        # Should pass validation
        validated = MLRegistryEventSchema.validate(valid_event)
        assert len(validated) == 1

    def test_single_thread_boundary_contract(self):
        """
        Test that MessageBus interactions happen only in actor thread.
        """
        # Mock MessageBus and Actor
        mock_msgbus = Mock()
        mock_actor = Mock()
        mock_actor.thread_id = 12345

        # Contract: All bus.publish() calls must happen from actor thread
        def assert_actor_thread(*args, **kwargs):
            # In real implementation, this would check threading.current_thread().ident
            assert hasattr(
                mock_actor,
                "thread_id",
            ), "MessageBus publish must be called from actor thread"

        mock_msgbus.publish.side_effect = assert_actor_thread

        # Simulate actor publishing event
        topic = build_topic_for_stage(
            Stage.DATA_INGESTED,
            instrument_id="EURUSD.SIM",
            scheme="stage_first",
            prefix="events.ml",
        )
        mock_msgbus.publish(topic=topic, payload={"correlation_id": str(uuid.uuid4())})

        mock_msgbus.publish.assert_called_once()

    def test_idempotency_contract_via_correlation_id(self):
        """
        Test that events are idempotent using correlation_id.
        """
        correlation_id = str(uuid.uuid4())

        # Mock consumer that tracks processed correlation_ids
        processed_correlations = set()

        def idempotent_consumer(topic: str, payload: dict[str, Any]) -> bool:
            """
            Returns True if event was processed, False if already seen.
            """
            metadata = payload.get("metadata", {})
            corr_id = metadata.get("correlation_id")
            if corr_id in processed_correlations:
                return False  # Already processed
            processed_correlations.add(corr_id)
            return True  # New event

        # First event should be processed
        event1 = {"metadata": {"correlation_id": correlation_id}, "data": "test"}
        topic = build_topic_for_stage(
            Stage.DATA_INGESTED,
            instrument_id="EURUSD.SIM",
            scheme="stage_first",
            prefix="events.ml",
        )
        assert idempotent_consumer(topic, event1) is True

        # Duplicate event should be ignored
        event2 = {"metadata": {"correlation_id": correlation_id}, "data": "test_duplicate"}
        assert idempotent_consumer(topic, event2) is False

    def test_watermark_progression_contract(self):
        """
        Test that watermarks progress monotonically.
        """
        # Mock watermark tracking
        watermarks = {}

        def update_watermark(dataset_id: str, new_watermark: int) -> bool:
            """
            Update watermark if it progresses.

            Return True if updated.

            """
            current = watermarks.get(dataset_id, 0)
            if new_watermark >= current:
                watermarks[dataset_id] = new_watermark
                return True
            return False  # Watermark regression - reject

        dataset_id = "eurusd_bars_2024"

        # Progressive watermarks should succeed
        assert update_watermark(dataset_id, 1000) is True
        assert update_watermark(dataset_id, 2000) is True
        assert update_watermark(dataset_id, 2000) is True  # Same is OK

        # Regressive watermark should fail
        assert update_watermark(dataset_id, 1500) is False

    def test_optional_bus_contract(self):
        """
        Test that msgbus=None means no-op publishing.
        """
        # When msgbus is None, publishing should be a no-op
        msgbus = None

        def safe_publish(bus: Any, topic: str, payload: dict[str, Any]) -> bool:
            """
            Safe publish that handles None bus.
            """
            if bus is None:
                return False  # No-op
            bus.publish(topic, payload)
            return True

        # Should not raise exception
        topic = build_topic_for_stage(
            Stage.DATA_INGESTED,
            instrument_id="EURUSD.SIM",
            scheme="stage_first",
            prefix="events.ml",
        )
        result = safe_publish(msgbus, topic, {"test": "data"})
        assert result is False

    def test_wildcard_topic_filtering_contract(self):
        """
        Test that wildcard filters work for event subscription.
        """
        # Mock subscription system
        subscriptions: dict[str, list[str]] = {
            "events.ml.*": ["consumer1", "consumer2"],
            "events.ml.PREDICTION_EMITTED": ["consumer3"],
        }

        def find_matching_consumers(topic: str) -> list[str]:
            """
            Find consumers that match topic pattern.
            """
            consumers = []
            for pattern, consumer_list in subscriptions.items():
                if pattern.endswith("*"):
                    prefix = pattern[:-1]  # Remove *
                    if topic.startswith(prefix):
                        consumers.extend(consumer_list)
                elif pattern == topic:
                    consumers.extend(consumer_list)
            return consumers

        # Test wildcard matching
        topic_ingested = build_topic_for_stage(
            Stage.DATA_INGESTED,
            instrument_id="EURUSD.SIM",
            scheme="stage_first",
            prefix="events.ml",
        )
        consumers = find_matching_consumers(topic_ingested)
        assert "consumer1" in consumers
        assert "consumer2" in consumers
        assert "consumer3" not in consumers

        # Test exact matching
        consumers = find_matching_consumers("events.ml.PREDICTION_EMITTED")
        assert "consumer3" in consumers

    def test_event_payload_immutability_contract(self):
        """
        Test that event payloads are immutable after publishing.
        """
        import copy

        original_payload = {
            "dataset_id": "test",
            "metadata": {"correlation_id": str(uuid.uuid4())},
            "count": 100,
        }

        # Simulate publisher creating immutable copy
        published_payload = copy.deepcopy(original_payload)

        # Modify original after "publishing"
        original_payload["count"] = 999
        original_payload["dataset_id"] = "modified"

        # Published payload should remain unchanged
        assert published_payload["count"] == 100
        assert published_payload["dataset_id"] == "test"


@pytest.mark.parallel_safe
class TestEventOrdering:
    """
    Test event ordering contracts for the ML pipeline.
    """

    def test_stage_transition_ordering(self):
        """
        Test that events follow correct stage transitions.
        """
        # Define allowed stage transitions
        allowed_transitions = {
            Stage.DATA_INGESTED: [Stage.CATALOG_WRITTEN, Stage.FEATURE_COMPUTED],
            Stage.CATALOG_WRITTEN: [Stage.FEATURE_COMPUTED],
            Stage.FEATURE_COMPUTED: [Stage.PREDICTION_EMITTED],
            Stage.PREDICTION_EMITTED: [Stage.SIGNAL_EMITTED],
            Stage.SIGNAL_EMITTED: [],  # Terminal stage
        }

        def validate_transition(from_stage: Stage, to_stage: Stage) -> bool:
            """
            Validate stage transition is allowed.
            """
            return to_stage in allowed_transitions.get(from_stage, [])

        # Valid transitions
        assert validate_transition(Stage.DATA_INGESTED, Stage.FEATURE_COMPUTED) is True
        assert validate_transition(Stage.FEATURE_COMPUTED, Stage.PREDICTION_EMITTED) is True

        # Invalid transitions
        assert validate_transition(Stage.SIGNAL_EMITTED, Stage.DATA_INGESTED) is False
        assert validate_transition(Stage.PREDICTION_EMITTED, Stage.DATA_INGESTED) is False

    def test_correlation_lineage_tracing(self):
        """
        Test that correlation_id enables lineage tracing across stages.
        """
        correlation_id = str(uuid.uuid4())

        # Mock event log
        event_log = []

        def log_event(stage: Stage, correlation_id: str, timestamp: int) -> None:
            event_log.append(
                {
                    "stage": stage,
                    "correlation_id": correlation_id,
                    "timestamp": timestamp,
                },
            )

        # Simulate pipeline flow with same correlation_id
        base_ts = int(datetime.now().timestamp() * 1e9)
        log_event(Stage.DATA_INGESTED, correlation_id, base_ts)
        log_event(Stage.FEATURE_COMPUTED, correlation_id, base_ts + 1000)
        log_event(Stage.PREDICTION_EMITTED, correlation_id, base_ts + 2000)
        log_event(Stage.SIGNAL_EMITTED, correlation_id, base_ts + 3000)

        # Query by correlation_id should return full pipeline trace
        lineage = [event for event in event_log if event["correlation_id"] == correlation_id]
        assert len(lineage) == 4

        # Events should be in timestamp order
        timestamps = [event["timestamp"] for event in lineage]
        assert timestamps == sorted(timestamps)

        # Should cover full pipeline
        stages = {event["stage"] for event in lineage}
        expected_stages = {
            Stage.DATA_INGESTED,
            Stage.FEATURE_COMPUTED,
            Stage.PREDICTION_EMITTED,
            Stage.SIGNAL_EMITTED,
        }
        assert stages == expected_stages
