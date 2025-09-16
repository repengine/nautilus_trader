#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ml.stores.mixins import sanitize_and_dedup


@dataclass
class _Row:
    strategy_id: str
    instrument_id: str
    ts_event: int
    ts_init: int


def _row_to_dict(r: _Row) -> dict[str, Any]:
    return {
        "strategy_id": r.strategy_id,
        "instrument_id": r.instrument_id,
        "ts_event": r.ts_event,
        "ts_init": r.ts_init,
    }


def _ts_any_units() -> st.SearchStrategy[int]:
    # seconds, milliseconds, microseconds, nanoseconds ranges
    return st.one_of(
        st.integers(min_value=1_000_000_000, max_value=10_000_000_000),  # seconds
        st.integers(min_value=1_000_000_000_000, max_value=10_000_000_000_000),  # ms
        st.integers(min_value=1_000_000_000_000_000, max_value=10_000_000_000_000_000),  # us
        st.integers(
            min_value=1_000_000_000_000_000_000,
            max_value=10_000_000_000_000_000_000,
        ),  # ns
    )


@given(
    rows=st.lists(
        st.builds(
            _Row,
            strategy_id=st.sampled_from(["S1", "S2"]),
            instrument_id=st.sampled_from(["EUR/USD", "SPY", "BTCUSD"]),
            ts_event=_ts_any_units(),
            ts_init=_ts_any_units(),
        ),
        min_size=1,
        max_size=50,
    ).map(
        lambda lst: lst + lst[: len(lst) // 3],
    ),  # inject duplicates
)
def test_sanitize_and_dedup_properties(rows: list[_Row]) -> None:
    # Convert to dicts and shuffle via Hypothesis
    values = [_row_to_dict(r) for r in rows]

    out = sanitize_and_dedup(
        values,
        ts_event_field="ts_event",
        ts_init_field="ts_init",
        context="property_test",
        key_fields=("strategy_id", "instrument_id", "ts_event"),
    )

    # Invariant 1: dedup by (strategy_id, instrument_id, ts_event)
    keys = set()
    for v in out:
        k = (v["strategy_id"], v["instrument_id"], int(v["ts_event"]))
        assert k not in keys
        keys.add(k)

    # Invariant 2: timestamps are normalized to nanoseconds
    for v in out:
        assert int(v["ts_event"]) >= 1_000_000_000_000_000_000
        assert int(v["ts_init"]) >= 1_000_000_000_000_000_000
