from __future__ import annotations

import pytest

try:  # optional dependency
    from hypothesis import given
    from hypothesis import strategies as st
except Exception:  # pragma: no cover - hypothesis optional
    pytest.skip("hypothesis not available", allow_module_level=True)

from typing import Any

from ml.consumers.idempotent import IdempotentConsumer
from ml.config.events import Source


def _payload(dataset_id: str, instrument_id: str, source: str, ts_max: int, cid: str) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "instrument_id": instrument_id,
        "source": source,
        "ts_max": ts_max,
        "metadata": {"correlation_id": cid},
    }


@given(
    dataset_id=st.sampled_from(["features", "models", "signals"]),
    instrument_id=st.from_regex(r"[A-Z]{3,6}/[A-Z]{3,6}\.SIM", fullmatch=True),
    source=st.sampled_from([s.value for s in Source]),
    ts_values=st.lists(st.integers(min_value=0, max_value=10_000), min_size=3, max_size=20),
)
def test_idempotent_consumer_property(dataset_id: str, instrument_id: str, source: str, ts_values: list[int]) -> None:
    c = IdempotentConsumer()
    # Build a sequence of events with controlled duplicates and regressions
    seen: set[str] = set()
    watermark = -1
    for i, ts in enumerate(ts_values):
        # Introduce duplicates every 4th event
        cid = f"cid-{i if i % 4 else max(0, i - 1)}"
        p = _payload(dataset_id, instrument_id, source, ts, cid)
        accepted = c.process(p)

        if cid in seen or ts < watermark:
            assert accepted is False
        else:
            assert accepted is True
            seen.add(cid)
            watermark = max(watermark, ts)

