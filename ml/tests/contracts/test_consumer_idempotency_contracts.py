from __future__ import annotations

import uuid
from typing import Any

import pytest

from ml.common.message_topics import build_topic_for_stage
from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.consumers.idempotent import IdempotentConsumer


def _make_payload(
    *,
    dataset_id: str,
    instrument_id: str,
    source: Source,
    ts_max: int,
    correlation_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "dataset_id": dataset_id,
        "instrument_id": instrument_id,
        "source": source.value,
        "stage": Stage.FEATURE_COMPUTED.value,
        "status": EventStatus.SUCCESS.value,
        "run_id": "test_run",
        "ts_min": ts_max,
        "ts_max": ts_max,
        "count": 1,
        "metadata": {"correlation_id": correlation_id or str(uuid.uuid4())},
    }
    if extra:
        payload.update(extra)
    return payload


@pytest.mark.contracts
def test_idempotent_consumer_drops_duplicates_and_enforces_watermark() -> None:
    c = IdempotentConsumer()

    cid_a = "A"
    p1 = _make_payload(
        dataset_id="features",
        instrument_id="EURUSD.SIM",
        source=Source.HISTORICAL,
        ts_max=100,
        correlation_id=cid_a,
    )
    p2 = _make_payload(
        dataset_id="features",
        instrument_id="EURUSD.SIM",
        source=Source.HISTORICAL,
        ts_max=90,  # lower than watermark
        correlation_id="B",
    )
    p3 = _make_payload(
        dataset_id="features",
        instrument_id="EURUSD.SIM",
        source=Source.HISTORICAL,
        ts_max=120,
        correlation_id=cid_a,  # duplicate correlation
    )
    p4 = _make_payload(
        dataset_id="features",
        instrument_id="EURUSD.SIM",
        source=Source.HISTORICAL,
        ts_max=130,
        correlation_id="C",
    )

    assert c.process(p1) is True
    assert c.process(p2) is False  # watermark regression
    assert c.process(p3) is False  # duplicate correlation
    assert c.process(p4) is True  # new correlation, higher watermark


@pytest.mark.contracts
def test_idempotent_consumer_keys_are_isolated() -> None:
    c = IdempotentConsumer()
    # Different instrument_id should maintain separate watermarks
    p_eur = _make_payload(
        dataset_id="features",
        instrument_id="EURUSD.SIM",
        source=Source.LIVE,
        ts_max=100,
        correlation_id="X",
    )
    p_btc_lower = _make_payload(
        dataset_id="features",
        instrument_id="BTCUSD.SIM",
        source=Source.LIVE,
        ts_max=50,
        correlation_id="Y",
    )

    assert c.process(p_eur) is True
    # Lower absolute ts_max is OK for a different key
    assert c.process(p_btc_lower) is True


@pytest.mark.contracts
def test_topics_built_for_stage_produce_valid_subjects() -> None:
    # Not asserting bus side-effects here; only that stage/scheme map consistently
    topic1 = build_topic_for_stage(
        Stage.DATA_INGESTED,
        instrument_id="EUR/USD",
        scheme="domain_op",
        prefix="events.ml",
    )
    topic2 = build_topic_for_stage(
        Stage.SIGNAL_EMITTED,
        instrument_id="SPY.NYSE",
        scheme="stage_first",
        prefix="events.ml",
    )

    assert topic1.startswith("ml.data.")  # domain_op -> ml.{domain}.operation
    assert topic2.startswith("events.ml.SIGNAL_EMITTED.")
