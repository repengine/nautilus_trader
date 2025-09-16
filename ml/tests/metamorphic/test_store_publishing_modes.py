from __future__ import annotations

from typing import Any

import pytest

from ml.common.message_topics import build_topic_for_stage
from ml.config.events import Source
from ml.config.events import Stage
from ml.stores.mixins import publish_batch_and_rows


class CapturePublisher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        self.calls.append((topic, payload))
        return True


def _rows() -> list[dict[str, Any]]:
    return [
        {"instrument_id": "EUR/USD", "ts_event": 100},
        {"instrument_id": "EUR/USD", "ts_event": 110},
        {"instrument_id": "EUR/USD", "ts_event": 105},
    ]


@pytest.mark.metamorphic
@pytest.mark.parallel_safe
def test_publish_modes_batch_row_both_preserve_counts_and_ranges() -> None:
    pub = CapturePublisher()
    rows = _rows()
    stage = Stage.FEATURE_COMPUTED
    dataset_id = "features"
    source = Source.HISTORICAL.value

    # Batch mode
    publish_batch_and_rows(
        enable_publishing=True,
        publisher=pub,
        publish_mode="batch",
        topic_scheme="stage_first",
        topic_prefix="events.ml",
        stage=stage,
        dataset_id=dataset_id,
        instrument_key="instrument_id",
        ts_field="ts_event",
        rows=rows,
        run_id_batch="run_batch",
        run_id_row="run_row",
        source=source,
        logger=None,
    )
    assert len(pub.calls) == 1
    topic, payload = pub.calls[-1]
    assert topic.startswith("events.ml.FEATURE_COMPUTED.")
    assert payload["count"] == len(rows)
    assert payload["ts_min"] == 100 and payload["ts_max"] == 110

    # Row mode
    pub.calls.clear()
    publish_batch_and_rows(
        enable_publishing=True,
        publisher=pub,
        publish_mode="row",
        topic_scheme="domain_op",
        topic_prefix="events.ml",
        stage=stage,
        dataset_id=dataset_id,
        instrument_key="instrument_id",
        ts_field="ts_event",
        rows=rows,
        run_id_batch="run_batch",
        run_id_row="run_row",
        source=source,
        logger=None,
    )
    assert len(pub.calls) == len(rows)
    for t, p in pub.calls:
        # domain_op topics are 'ml.{domain}.{operation}.{instrument}'
        assert t.startswith("ml.") and p["count"] == 1 and p["ts_min"] == p["ts_max"]

    # Both mode
    pub.calls.clear()
    publish_batch_and_rows(
        enable_publishing=True,
        publisher=pub,
        publish_mode="both",
        topic_scheme="stage_first",
        topic_prefix="events.ml",
        stage=stage,
        dataset_id=dataset_id,
        instrument_key="instrument_id",
        ts_field="ts_event",
        rows=rows,
        run_id_batch="run_batch",
        run_id_row="run_row",
        source=source,
        logger=None,
    )
    assert len(pub.calls) == len(rows) + 1


@pytest.mark.metamorphic
def test_publish_respects_scheme_and_prefix() -> None:
    pub = CapturePublisher()
    rows = _rows()
    # Stage-first
    publish_batch_and_rows(
        enable_publishing=True,
        publisher=pub,
        publish_mode="batch",
        topic_scheme="stage_first",
        topic_prefix="custom.prefix",
        stage=Stage.PREDICTION_EMITTED,
        dataset_id="models",
        instrument_key="instrument_id",
        ts_field="ts_event",
        rows=rows,
        run_id_batch="run",
        run_id_row="row",
        source=Source.LIVE.value,
        logger=None,
    )
    topic, _ = pub.calls[-1]
    assert topic.startswith("custom.prefix.PREDICTION_EMITTED.")

    # Domain-op
    pub.calls.clear()
    publish_batch_and_rows(
        enable_publishing=True,
        publisher=pub,
        publish_mode="batch",
        topic_scheme="domain_op",
        topic_prefix="events.ml",
        stage=Stage.SIGNAL_EMITTED,
        dataset_id="signals",
        instrument_key="instrument_id",
        ts_field="ts_event",
        rows=rows,
        run_id_batch="run",
        run_id_row="row",
        source=Source.LIVE.value,
        logger=None,
    )
    t, _ = pub.calls[-1]
    expected_prefix = build_topic_for_stage(
        Stage.SIGNAL_EMITTED,
        "EUR/USD",
        scheme="domain_op",
        prefix="events.ml",
    ).rsplit(".", 1)[0]
    assert t.startswith(expected_prefix)
