"""
Shared batch utilities for ML stores.

This module provides helpers to:
- Sanitize timestamp fields and de-duplicate batch rows by composite keys
- Publish batch and per-row events via the store's message bus
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ml.common.message_topics import build_topic_for_stage
from ml.config.events import EventStatus
from ml.config.events import Stage


def sanitize_and_dedup(
    values: list[dict[str, Any]],
    *,
    ts_event_field: str,
    ts_init_field: str,
    context: str,
    key_fields: tuple[str, str, str],
) -> list[dict[str, Any]]:
    """
    Sanitize ts fields and de-duplicate rows within a batch.

    Parameters
    ----------
    values : list[dict[str, Any]]
        Incoming batch of row dicts (mutated in place for timestamp normalization).
    ts_event_field : str
        Field holding the event timestamp (ns).
    ts_init_field : str
        Field holding the init timestamp (ns).
    context : str
        Context label for the sanitizer (used in logs/metrics).
    key_fields : tuple[str, str, str]
        Composite key fields used to deduplicate (e.g., ("model_id","instrument_id","ts_event")).

    Returns
    -------
    list[dict[str, Any]]
        De-duplicated list of row dicts.
    """
    if not values:
        return values

    # Normalize timestamps
    from ml.common.timestamps import sanitize_timestamp_ns

    for v in values:
        if ts_event_field in v:
            v[ts_event_field] = sanitize_timestamp_ns(int(v[ts_event_field]), context=context)
        if ts_init_field in v:
            v[ts_init_field] = sanitize_timestamp_ns(int(v[ts_init_field]), context=context)

    # De-duplicate within the same batch
    k1, k2, k3 = key_fields
    dedup: dict[tuple[str, str, int], dict[str, Any]] = {}
    for v in values:
        key = (str(v[k1]), str(v[k2]), int(v[k3]))
        dedup[key] = v
    return list(dedup.values())


def publish_batch_and_rows(
    *,
    enable_publishing: bool,
    publisher: Any | None,
    publish_mode: str,
    topic_scheme: str,
    topic_prefix: str,
    stage: Stage,
    dataset_id: str,
    instrument_key: str,
    ts_field: str,
    rows: Iterable[dict[str, Any]],
    run_id_batch: str,
    run_id_row: str,
    source: str,
    logger: Any,
) -> None:
    """
    Publish batch summary and per-row events (best-effort).

    This should be called off the hot path. All publish attempts are wrapped in try/except.
    """
    rows_list = list(rows)
    if not (enable_publishing and publisher and rows_list):
        return

    # Batch summary
    if publish_mode in ("batch", "both"):
        try:
            instrument_id = str(rows_list[0].get(instrument_key, "UNKNOWN"))
            topic = build_topic_for_stage(stage, instrument_id, scheme=topic_scheme, prefix=topic_prefix)
            ts_vals = [int(r.get(ts_field, 0)) for r in rows_list]
            payload: dict[str, Any] = {
                "dataset_id": dataset_id,
                "instrument_id": instrument_id,
                "stage": stage.value,
                "source": source,
                "run_id": run_id_batch,
                "ts_min": min(ts_vals) if ts_vals else 0,
                "ts_max": max(ts_vals) if ts_vals else 0,
                "count": len(rows_list),
                "status": EventStatus.SUCCESS.value,
            }
            publisher.publish(topic, payload)
        except Exception:
            logger.debug("Batch publish failed", exc_info=True)

    # Per-row events
    if publish_mode in ("row", "both"):
        try:
            for r in rows_list:
                instrument_id = str(r.get(instrument_key, "UNKNOWN"))
                topic = build_topic_for_stage(stage, instrument_id, scheme=topic_scheme, prefix=topic_prefix)
                ts_e = int(r.get(ts_field, 0))
                row_payload: dict[str, Any] = {
                    "dataset_id": dataset_id,
                    "instrument_id": instrument_id,
                    "stage": stage.value,
                    "source": source,
                    "run_id": run_id_row,
                    "ts_min": ts_e,
                    "ts_max": ts_e,
                    "count": 1,
                    "status": EventStatus.SUCCESS.value,
                }
                publisher.publish(topic, row_payload)
        except Exception:
            logger.debug("Per-row publish failed", exc_info=True)

