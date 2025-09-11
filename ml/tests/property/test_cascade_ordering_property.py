from __future__ import annotations

from typing import Any

from hypothesis import given, strategies as st

from ml.common.cascade import emit_cascade


@given(
    base_ts=st.integers(min_value=0, max_value=2**32),
    delays=st.lists(st.integers(min_value=0, max_value=10_000), min_size=1, max_size=50),
)
def test_cascade_with_non_negative_delays_is_monotonic(base_ts: int, delays: list[int]) -> None:
    """
    Property: Applying emit_cascade repeatedly with non-negative delays yields a non-decreasing
    sequence of ts_event values.
    """
    ev: dict[str, Any] = {
        "domain": "data",
        "event_type": "INGESTED",
        "correlation_id": "CID-TEST",
        "instrument_id": "EURUSD.SIM",
        "ts_event": base_ts,
        "event_id": "E0",
        "payload": {},
    }
    ts_vals: list[int] = [ev["ts_event"]]
    current = ev
    for i, d in enumerate(delays):
        current = emit_cascade(current, target_domain=f"stage_{i}", delay_ns=d)
        ts_vals.append(current["ts_event"])

    assert all(ts_vals[i] <= ts_vals[i + 1] for i in range(len(ts_vals) - 1))
