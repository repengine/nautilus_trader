"""Integration tests for registry event emission (Phase 2.2.4).

This module contains integration tests for RegistrySynchronizer's
message bus event emission workflows.

All tests marked @pytest.mark.skip for structural phase.
Full implementation in Phase 2.2.8.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from ml.common.message_topics import build_topic_for_stage
from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.orchestration.registry_synchronizer import RegistrySynchronizer


@pytest.mark.integration
def test_feature_refresh_event_emitted_to_message_bus() -> None:
    """Emit feature refresh event to message bus.

    Phase 2.2.4: Test skipped (structural phase).
    Phase 2.2.8: Will verify feature refresh event is published to message bus.

    Expected Behavior (Phase 2.2.8):
    - RegistrySynchronizer._emit_feature_refresh_event() invoked
    - Event published to message bus topic "ml.features.refresh"
    - Event payload contains dataset_id and features
    - Timestamp included in event
    """
    # Setup
    data_registry = Mock()
    feature_registry = Mock()
    model_registry = Mock()
    message_bus = Mock()
    data_registry.get_manifest.side_effect = Exception("missing")

    synchronizer = RegistrySynchronizer(
        data_registry=data_registry,
        feature_registry=feature_registry,
        model_registry=model_registry,
        message_bus=message_bus,
    )

    features = ["sma_20", "ema_50", "rsi_14"]

    # Execute
    synchronizer._emit_feature_refresh_event("spy_2024_ohlcv", features)

    message_bus.publish.assert_called_once()
    topic, payload = message_bus.publish.call_args.args
    assert topic == build_topic_for_stage(Stage.FEATURE_COMPUTED, "spy_2024_ohlcv")
    assert payload["dataset_id"] == "spy_2024_ohlcv"
    assert payload["features"] == features
    assert payload["stage"] == Stage.FEATURE_COMPUTED.value
    assert payload["source"] == Source.HISTORICAL.value
    assert payload["status"] == EventStatus.SUCCESS.value
    assert isinstance(payload["timestamp_ns"], int)


@pytest.mark.integration
def test_dataset_registered_event_emitted() -> None:
    """Emit dataset registered event after registration.

    Phase 2.2.4: Test skipped (structural phase).
    Phase 2.2.8: Will verify dataset registered event is emitted.

    Expected Behavior (Phase 2.2.8):
    - After _ensure_dataset_registered(), emit event
    - Event published to message bus topic "ml.datasets.registered"
    - Event payload contains dataset_id and metadata
    """
    # Setup
    data_registry = Mock()
    feature_registry = Mock()
    model_registry = Mock()
    message_bus = Mock()
    data_registry.get_manifest.side_effect = Exception("missing")

    synchronizer = RegistrySynchronizer(
        data_registry=data_registry,
        feature_registry=feature_registry,
        model_registry=model_registry,
        message_bus=message_bus,
    )

    metadata = {"symbols": ["SPY"], "row_count": 98280}

    # Execute
    synchronizer._ensure_dataset_registered("spy_2024_ohlcv", metadata)

    assert data_registry.register_dataset.called


@pytest.mark.integration
def test_event_payload_contains_metadata() -> None:
    """Verify event payload structure.

    Phase 2.2.4: Test skipped (structural phase).
    Phase 2.2.8: Will verify event payload has correct structure.

    Expected Behavior (Phase 2.2.8):
    - Event payload is valid JSON
    - Contains required fields: event_type, dataset_id, features, timestamp
    - Timestamp is ISO 8601 format
    """
    # Setup
    data_registry = Mock()
    feature_registry = Mock()
    model_registry = Mock()
    message_bus = Mock()

    synchronizer = RegistrySynchronizer(
        data_registry=data_registry,
        feature_registry=feature_registry,
        model_registry=model_registry,
        message_bus=message_bus,
    )

    features = ["sma_20", "ema_50", "rsi_14"]

    # Execute
    synchronizer._emit_feature_refresh_event("spy_2024_ohlcv", features)

    payload = message_bus.publish.call_args.args[1]
    assert payload["event_type"] == "feature_refresh"
    assert payload["dataset_id"] == "spy_2024_ohlcv"
    assert payload["features"] == features
    assert isinstance(payload["timestamp_ns"], int)
