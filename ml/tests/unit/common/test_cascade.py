from __future__ import annotations

from typing import cast
from ml.common.cascade import emit_cascade, EventDict


class TestCascade:
    def test_emit_cascade_preserves_correlation_and_adds_delay(self) -> None:
        source_any: dict[str, object] = {
            "event_id": "e1",
            "domain": "data",
            "event_type": "ingested",
            "correlation_id": "abc123",
            "instrument_id": "EURUSD.SIM",
            "ts_event": 1000,
            "payload": {"x": 1},
        }
        source = cast(EventDict, source_any)
        out = emit_cascade(source, "features", delay_ns=25)
        assert out["domain"] == "features"
        assert out["correlation_id"] == source["correlation_id"]
        assert out["ts_event"] == 1025
        assert out["source_event_id"] == source_any["event_id"]
