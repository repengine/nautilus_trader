"""
Tests for actor bus bridge mutual exclusion disabling store-level publishers.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ml.actors.ml_domain_events import init_actor_bus_bridge
from ml.common.message_bus import NoopPublisher


from typing import Any


@pytest.mark.serial
def test_actor_bus_bridge_disables_store_publishers(monkeypatch: Any) -> None:
    # Enable actor bus and global bus
    monkeypatch.setenv("ML_BUS_FROM_ACTOR", "1")
    monkeypatch.setenv("ML_BUS_ENABLE", "1")
    monkeypatch.setenv("ML_BUS_BACKEND", "noop")

    # Dummy stores with publishers enabled
    class _DummyStore:
        def __init__(self) -> None:
            self.publisher = object()
            self._enable_publishing = True

    actor = SimpleNamespace(
        _feature_store=_DummyStore(),
        _model_store=_DummyStore(),
        _strategy_store=_DummyStore(),
        _data_store=_DummyStore(),
    )

    # Force NoopPublisher construction from config
    with patch("ml.actors.ml_domain_events.publisher_from_config", return_value=NoopPublisher()):
        bridge, _scheme, _prefix = init_actor_bus_bridge(actor)

    try:
        assert bridge is not None
        # Mutual exclusion: store publishers disabled and flags turned off
        for st in (
            actor._feature_store,
            actor._model_store,
            actor._strategy_store,
            actor._data_store,
        ):
            assert getattr(st, "publisher") is None
            assert getattr(st, "_enable_publishing") is False
    finally:
        # Stop background thread if started
        if bridge is not None:
            bridge.stop(drain=True)
