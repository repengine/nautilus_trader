from __future__ import annotations

import time
from typing import Any, Literal, cast

import pytest

from ml.common.message_bus import MessagePublisherProtocol
from ml.stores.base import ModelPrediction
from ml.stores.model_store import ModelStore


class CapturePublisher(MessagePublisherProtocol):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        self.calls.append((topic, payload))
        return True


@pytest.mark.parametrize("mode,expected_extra", [
    ("batch", 1),
    ("row", 2),
    ("both", 3),
])
def test_model_store_publishing_modes(mode: str, expected_extra: int, monkeypatch: pytest.MonkeyPatch) -> None:
    cap = CapturePublisher()
    store = ModelStore(
        connection_string=None,
        enable_publishing=True,
        publisher=cap,
        publish_mode=cast(Literal["batch", "row", "both"], mode),
    )

    # Monkeypatch the upsert to avoid DB usage and only exercise publishing logic
    from ml.stores.mixins import publish_batch_and_rows

    def _stub_execute_upsert_and_publish(**kwargs: Any) -> None:
        publish_batch_and_rows(
            enable_publishing=bool(getattr(store, "_enable_publishing", False)),
            publisher=getattr(store, "publisher", None),
            publish_mode=getattr(store, "_publish_mode", "batch"),
            topic_scheme=getattr(store, "_topic_scheme", "domain_op"),
            topic_prefix=getattr(store, "_topic_prefix", "events.ml"),
            stage=kwargs["stage"],
            dataset_id=kwargs["dataset_id"],
            instrument_key=kwargs["instrument_key"],
            ts_field=kwargs["ts_field"],
            rows=kwargs["values"],
            run_id_batch=kwargs["run_id_batch"],
            run_id_row=kwargs["run_id_row"],
            source=kwargs["source"],
            logger=kwargs["logger"],
        )

    monkeypatch.setattr(store, "_execute_upsert_and_publish", _stub_execute_upsert_and_publish)

    now = time.time_ns()
    preds = [
        ModelPrediction(
            model_id="m",
            instrument_id="SPY",
            prediction=0.1,
            confidence=0.9,
            features_used={},
            inference_time_ms=1.0,
            _ts_event=now,
            _ts_init=now,
        ),
        ModelPrediction(
            model_id="m",
            instrument_id="SPY",
            prediction=0.2,
            confidence=0.7,
            features_used={},
            inference_time_ms=1.0,
            _ts_event=now + 1,
            _ts_init=now + 1,
        ),
    ]

    store.write_batch(preds)

    # Validate number of publish calls aligns with mode: batch=1, row=2, both=3
    assert len(cap.calls) == expected_extra
    # Basic payload fields exist
    for _, payload in cap.calls:
        assert payload.get("dataset_id") == "predictions"
        assert "stage" in payload
