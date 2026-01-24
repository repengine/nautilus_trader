from __future__ import annotations

import re
from typing import Iterable

import pytest

from ml.common.message_topics import (
    build_stage_topic,
    build_topic,
    build_topic_for_stage,
    map_stage_to_topic_segments,
)
from ml.config.events import Stage


def assert_allowed_topic_chars(topic: str) -> None:
    # Ensure no wildcards/illegal separators in topic
    assert all(ch not in topic for ch in {"/", "*", "#", "+", "$"}), topic


@pytest.mark.parametrize(
    "stage,expected",
    [
        (Stage.DATA_INGESTED, ("data", "created")),
        (Stage.CATALOG_WRITTEN, ("data", "updated")),
        (Stage.FEATURE_COMPUTED, ("features", "updated")),
        (Stage.PREDICTION_EMITTED, ("models", "created")),
        (Stage.SIGNAL_EMITTED, ("strategies", "created")),
        (Stage.ORDER_EVENT_EMITTED, ("orders", "created")),
    ],
)
def test_map_stage_to_segments(stage: Stage, expected: tuple[str, str]) -> None:
    assert map_stage_to_topic_segments(stage) == expected


@pytest.mark.parametrize(
    "raw_id",
    [
        "EURUSD/SIM",
        "BTCUSDT:BINANCE",
        "ETH-USD.COINBASE",
        "FOO*BAR?BAZ",
        "GBTC US Equity",
        "---weird///id###with$$bad++chars---",
        "",
    ],
)
def test_build_topic_normalization(raw_id: str) -> None:
    domain, op = ("features", "updated")
    topic = build_topic(domain, op, raw_id)
    assert topic.startswith("ml.features.updated."), topic
    assert_allowed_topic_chars(topic)
    # Only allow characters [A-Za-z0-9_.-] after the third segment
    suffix = topic.split(".", 3)[-1]
    assert re.match(r"^[A-Za-z0-9_.-]+$", suffix), suffix


@pytest.mark.parametrize("scheme", ["domain_op", "stage_first"])  # both schemes
def test_build_topic_for_stage_schemes(scheme: str) -> None:
    instrument = "EURUSD/SIM"
    topic = build_topic_for_stage(
        Stage.FEATURE_COMPUTED,
        instrument,
        scheme=scheme,
        prefix="events.ml",
    )
    if scheme == "domain_op":
        assert topic.startswith("ml.features.updated."), topic
    else:
        assert topic.startswith("events.ml.FEATURE_COMPUTED."), topic
    assert_allowed_topic_chars(topic)
