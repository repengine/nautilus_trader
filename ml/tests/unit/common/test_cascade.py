from __future__ import annotations

from ml.common.cascade import emit_cascade


class TestCascade:
    def test_emit_cascade_preserves_correlation_and_adds_delay(self) -> None:
        source = {
            "event_id": "e1",
            "domain": "data",
            "event_type": "ingested",
            "correlation_id": "abc123",
            "instrument_id": "EURUSD.SIM",
            "ts_event": 1000,
            "payload": {"x": 1},
        }
        out = emit_cascade(source, "features", delay_ns=25)
        assert out["domain"] == "features"
        assert out["correlation_id"] == source["correlation_id"]
        assert out["ts_event"] == 1025
        assert out["source_event_id"] == source["event_id"]
