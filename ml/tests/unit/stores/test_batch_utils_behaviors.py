from __future__ import annotations

from typing import Any

import pytest

from ml.config.events import Stage
from ml.stores.mixins import publish_batch_and_rows, sanitize_and_dedup


def test_sanitize_and_dedup_mixed_units_and_duplicates() -> None:
    # Two keys, with duplicates and mixed timestamp units
    values = [
        {"model_id": "m1", "instrument_id": "EUR/USD", "ts_event": 1_700_000_000, "ts_init": 1_700_000_000},  # seconds
        {"model_id": "m1", "instrument_id": "EUR/USD", "ts_event": 1_700_000_000, "ts_init": 1_700_000_000},  # dup
        {"model_id": "m2", "instrument_id": "EUR/USD", "ts_event": 1_700_000_000_000, "ts_init": 1_700_000_000_000},  # ms
        {"model_id": "m2", "instrument_id": "EUR/USD", "ts_event": 1_700_000_000_000, "ts_init": 1_700_000_000_000 + 1},  # dup with different init
    ]

    out = sanitize_and_dedup(
        values,
        ts_event_field="ts_event",
        ts_init_field="ts_init",
        context="unit-test",
        key_fields=("model_id", "instrument_id", "ts_event"),
    )

    # Duplicates collapsed on composite key; expect last occurrence retained per key
    assert len(out) == 2
    # All timestamps normalized to ns
    for row in out:
        assert row["ts_event"] >= 10**18
        assert row["ts_init"] >= 10**18


class _CapturePublisher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:  # noqa: FBT001, FBT002
        self.calls.append((topic, payload))
        return True


@pytest.mark.parametrize(
    "publish_mode, expected_calls",
    [
        ("batch", 1),
        ("row", 2),
        ("both", 3),
    ],
)
def test_publish_batch_and_rows_modes_domain_op(publish_mode: str, expected_calls: int) -> None:
    cap = _CapturePublisher()
    rows = [
        {"instrument_id": "EURUSD/SIM", "ts_event": 123},
        {"instrument_id": "EURUSD/SIM", "ts_event": 456},
    ]

    publish_batch_and_rows(
        enable_publishing=True,
        publisher=cap,
        publish_mode=publish_mode,
        topic_scheme="domain_op",
        topic_prefix="events.ml",
        stage=Stage.FEATURE_COMPUTED,
        dataset_id="features_basic",
        instrument_key="instrument_id",
        ts_field="ts_event",
        rows=rows,
        run_id_batch="rid-b",
        run_id_row="rid-r",
        source="unit",
        logger=__import__("logging").getLogger(__name__),
    )

    assert len(cap.calls) == expected_calls
    # Topics follow domain_op scheme: ml.features.updated.<instrument>
    topics = [t for t, _ in cap.calls]
    assert all(t.startswith("ml.features.updated.EURUSD.SIM") for t in topics)


def test_publish_batch_and_rows_stage_first_prefix() -> None:
    cap = _CapturePublisher()
    rows = [
        {"instrument_id": "AAPL/XNAS", "ts_event": 999},
        {"instrument_id": "AAPL/XNAS", "ts_event": 1000},
    ]

    publish_batch_and_rows(
        enable_publishing=True,
        publisher=cap,
        publish_mode="both",
        topic_scheme="stage_first",
        topic_prefix="events.ml",
        stage=Stage.FEATURE_COMPUTED,
        dataset_id="features_basic",
        instrument_key="instrument_id",
        ts_field="ts_event",
        rows=rows,
        run_id_batch="rid-b",
        run_id_row="rid-r",
        source="unit",
        logger=__import__("logging").getLogger(__name__),
    )

    # 1 batch + 2 rows
    assert len(cap.calls) == 3
    topics = [t for t, _ in cap.calls]
    assert all(t.startswith("events.ml.FEATURE_COMPUTED.AAPL.XNAS") for t in topics)
