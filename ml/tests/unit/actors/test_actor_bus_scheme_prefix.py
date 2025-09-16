"""
Actor bus bridge returns scheme/prefix honoring environment.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ml.actors.ml_domain_events import init_actor_bus_bridge
from ml.common.message_bus import NoopPublisher


from typing import Any


@pytest.mark.serial
def test_actor_bus_scheme_and_prefix(monkeypatch: Any) -> None:
    # Enable actor bus and message bus with non-default scheme/prefix
    monkeypatch.setenv("ML_BUS_FROM_ACTOR", "1")
    monkeypatch.setenv("ML_BUS_ENABLE", "1")
    monkeypatch.setenv("ML_BUS_SCHEME", "stage_first")
    monkeypatch.setenv("ML_BUS_TOPIC_PREFIX", "events.ml.alt")

    class _S:
        def __init__(self) -> None:
            self.publisher = None
            self._enable_publishing = False

    actor = type("_A", (), {
        "_feature_store": _S(),
        "_model_store": _S(),
        "_strategy_store": _S(),
        "_data_store": _S(),
    })()

    with patch("ml.actors.ml_domain_events.publisher_from_config", return_value=NoopPublisher()):
        bridge, scheme, prefix = init_actor_bus_bridge(actor)
    try:
        assert bridge is not None
        assert scheme == "stage_first"
        assert prefix == "events.ml.alt"
    finally:
        if bridge is not None:
            bridge.stop(drain=True)
