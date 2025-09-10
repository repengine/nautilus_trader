from __future__ import annotations

import random
from typing import List

import pytest
from hypothesis import given, strategies as st

from ml.consumers.aggregator import AggregatingConsumer
from ml.consumers.protocols import Envelope


def _make_envelope(idx: int, ts: int, inst: str = "EURUSD.SIM") -> Envelope:
    return {
        "id": f"e{idx}",
        "parent_id": None,
        "instrument_id": inst,
        "ts_event": ts,
        "stage": "FEATURE_COMPUTED",
        "correlation_id": "c1",
        "payload": {"x": idx},
    }


@given(
    timestamps=st.lists(
        st.integers(min_value=1, max_value=10_000), min_size=5, max_size=50, unique=True
    ),
)
def test_aggregator_monotonic_order_under_random_arrival(timestamps: List[int]) -> None:
    """
    Flushed order is non-decreasing even when arrivals are shuffled.
    """
    envs = [_make_envelope(i, ts) for i, ts in enumerate(timestamps)]
    random.shuffle(envs)
    agg = AggregatingConsumer()
    for e in envs:
        agg.handle("events.ml.FEATURE_COMPUTED", e)
    flushed = agg.advance_watermark("EURUSD.SIM", watermark_ns=max(timestamps))
    ts_out = [e["ts_event"] for e in flushed]
    assert ts_out == sorted(timestamps)


def test_aggregator_watermark_gating() -> None:
    envs = [_make_envelope(i, ts) for i, ts in enumerate([10, 20, 30, 40])]
    agg = AggregatingConsumer()
    for e in envs:
        agg.handle("events.ml.FEATURE_COMPUTED", e)
    # Gate at 25: only 10, 20 should flush
    flushed_a = agg.advance_watermark("EURUSD.SIM", watermark_ns=25)
    assert [e["ts_event"] for e in flushed_a] == [10, 20]
    # Gate beyond all: remaining should flush
    flushed_b = agg.advance_watermark("EURUSD.SIM", watermark_ns=100)
    assert [e["ts_event"] for e in flushed_b] == [30, 40]


def test_aggregator_idempotent_replay() -> None:
    a = AggregatingConsumer()
    e1 = _make_envelope(1, 100)
    a.handle("events.ml.FEATURE_COMPUTED", e1)
    a.handle("events.ml.FEATURE_COMPUTED", e1)  # duplicate
    flushed = a.advance_watermark("EURUSD.SIM", watermark_ns=200)
    assert len(flushed) == 1
    assert flushed[0]["id"] == "e1"
