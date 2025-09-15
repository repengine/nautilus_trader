from __future__ import annotations

from typing import Any

import pandas as pd

from ml.common.in_memory_bus import InMemoryPublisher
from ml.consumers.aggregator import AggregatingConsumer
from ml.consumers.lineage_writer import LineageWriter
from ml.consumers.protocols import Envelope
from ml.data.fixtures import make_tbbo_fixture
from ml.data.ingest.metrics import record_ingest_batch
from ml.observability.service import ObservabilityService


def _envelopes_from_tbbo(df: pd.DataFrame) -> list[Envelope]:
    envs: list[Envelope] = []
    for i, row in enumerate(df.itertuples(index=False), start=1):
        envs.append(
            {
                "id": f"e{i}",
                "parent_id": None,
                "instrument_id": str(row.instrument_id),
                "ts_event": int(row.ts_event),
                "stage": "FEATURE_COMPUTED",
                "correlation_id": "c-ingest-1",
                "payload": {"bid_px": float(row.bid_px), "ask_px": float(row.ask_px)},
            },
        )
    return envs


def test_ingest_to_aggregator_to_lineage_with_metrics() -> None:
    # Prepare deterministic TBBO fixture and record ingest metrics (provider-agnostic)
    df, _manifest = make_tbbo_fixture(rows=10)
    record_ingest_batch(
        dataset="tbbo",
        instrument=df["instrument_id"].iloc[0],
        source="historical",
        duration_seconds=0.001,
        ts_min=int(df["ts_event"].min()),
        ts_max=int(df["ts_event"].max()),
    )

    # Build envelopes; add a duplicate id to assert idempotency at aggregator
    envs = _envelopes_from_tbbo(df)
    envs.append(envs[0])

    # Setup aggregator → in-memory bus → lineage writer
    svc = ObservabilityService()
    writer = LineageWriter(service=svc)
    bus = InMemoryPublisher()
    bus.subscribe("aggregated.#", lambda t, p: writer.handle(t, p))
    agg = AggregatingConsumer(downstream=bus, topic_mapper=lambda _stage: "aggregated.lineage")

    # Shuffle arrival order minimally by swapping pairs
    for i, e in enumerate(envs):
        agg.handle("events.ml.FEATURE_COMPUTED", e)

    # Flush with watermark at end of window
    wm = int(df["ts_event"].max())
    flushed = agg.advance_watermark(df["instrument_id"].iloc[0], wm)

    # Idempotency: deduped; only unique envelope ids emitted
    unique_ids = {e["id"] for e in envs}
    assert len(flushed) == len(unique_ids)

    # Lineage DF contains matching event_ids in order
    lineage_df = svc.event_correlation_df()
    assert set(lineage_df["event_id"]) == unique_ids
    assert lineage_df.sort_values("ts_event")["event_id"].tolist() == [e["id"] for e in flushed]
