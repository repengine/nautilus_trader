from __future__ import annotations

from typing import Any

from ml.common.message_bus import MessagePublisherProtocol
from ml.stores.mixins import publish_batch_and_rows, sanitize_and_dedup
from ml.config.events import Stage


def test_sanitize_and_dedup_normalizes_and_deduplicates() -> None:
    values = [
        {
            "model_id": "m",
            "instrument_id": "SPY",
            "ts_event": 1_700_000_000_000,  # ms
            "ts_init": 1_700_000_000,  # seconds
        },
        {
            "model_id": "m",
            "instrument_id": "SPY",
            "ts_event": 1_700_000_000_000,  # duplicate key triple
            "ts_init": 1_700_000_000,
        },
    ]

    out = sanitize_and_dedup(
        values,
        ts_event_field="ts_event",
        ts_init_field="ts_init",
        context="unit-test",
        key_fields=("model_id", "instrument_id", "ts_event"),
    )

    # Deduplicated
    assert len(out) == 1
    # Normalized to ns
    v = out[0]
    assert v["ts_event"] == 1_700_000_000_000 * 1_000_000  # ms -> ns
    assert v["ts_init"] == 1_700_000_000 * 1_000_000_000  # s -> ns


class CapturePublisher(MessagePublisherProtocol):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        self.calls.append((topic, payload))
        return True


def test_publish_batch_and_rows_stage_first_scheme() -> None:
    cap = CapturePublisher()
    rows = [
        {"instrument_id": "EUR/USD", "ts_event": 123},
        {"instrument_id": "EUR/USD", "ts_event": 456},
    ]

    publish_batch_and_rows(
        enable_publishing=True,
        publisher=cap,
        publish_mode="both",
        topic_scheme="stage_first",
        topic_prefix="events.ml",
        stage=Stage.PREDICTION_EMITTED,
        dataset_id="predictions",
        instrument_key="instrument_id",
        ts_field="ts_event",
        rows=rows,
        run_id_batch="ridb",
        run_id_row="ridr",
        source="model",
        logger=__import__("logging").getLogger(__name__),
    )

    # Expect 1 batch + 2 rows = 3 calls
    assert len(cap.calls) == 3
    topics = [t for t, _ in cap.calls]
    assert all(t.startswith("events.ml.PREDICTION_EMITTED.EUR.USD") for t in topics)
