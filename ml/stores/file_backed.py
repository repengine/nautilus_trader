"""
File-backed store implementations for cold-path fallback.

These stores satisfy the core store protocols using structured JSONL persistence so that
systems can continue producing durable artifacts when PostgreSQL is unavailable. They
preserve correlation IDs and timestamps, integrate with the standard metrics bootstrap,
and expose the minimal APIs required by the ML integration surface.
"""

from __future__ import annotations

import json
import logging
import time
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterable
from collections.abc import MutableMapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from threading import RLock
from typing import Any

import pandas as pd

from ml.common.correlation import make_correlation_id
from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.stores.base import FeatureData
from ml.stores.base import ModelPrediction
from ml.stores.base import StrategySignal
from ml.stores.protocols import FeatureStoreProtocol
from ml.stores.protocols import ModelStoreProtocol
from ml.stores.protocols import StrategyStoreProtocol


_LOGGER = logging.getLogger(__name__)

_DEFAULT_HISTORY_LIMIT = 1_000

__all__ = [
    "FileDataStore",
    "FileFeatureStore",
    "FileModelStore",
    "FileStrategyStore",
]


@dataclass(slots=True)
class _JsonLineStore:
    """Utility to append and reload JSONL datasets with locking and metrics."""

    path: Path
    history_limit: int = _DEFAULT_HISTORY_LIMIT
    _lock: RLock = field(init=False, repr=False)
    _records: list[dict[str, Any]] = field(init=False, repr=False)
    _dirty: list[dict[str, Any]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._records: list[dict[str, Any]] = []
        self._dirty: list[dict[str, Any]] = []
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as infile:
                for line in infile:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        _LOGGER.debug("Skipping malformed JSONL line in %s: %s", self.path, exc)
        if len(self._records) > self.history_limit:
            self._records = self._records[-self.history_limit :]

    def append(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._records.append(record)
            self._dirty.append(record)
            if len(self._records) > self.history_limit:
                self._records = self._records[-self.history_limit :]

    def bulk_append(self, records: Iterable[dict[str, Any]]) -> None:
        for rec in records:
            self.append(rec)

    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._records)

    def flush(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            with self.path.open("a", encoding="utf-8") as outfile:
                for record in self._dirty:
                    outfile.write(json.dumps(record, ensure_ascii=False))
                    outfile.write("\n")
            self._dirty.clear()


class FileFeatureStore(FeatureStoreProtocol):
    """
    Feature store implementation that persists records to JSONL.

    Parameters
    ----------
    base_path:
        Directory used to store feature records (``features.jsonl``).
    """

    def __init__(self, *, base_path: Path, history_limit: int = _DEFAULT_HISTORY_LIMIT) -> None:
        self._paths = base_path
        self._paths.mkdir(parents=True, exist_ok=True)
        self._store = _JsonLineStore(self._paths / "features.jsonl", history_limit)
        self._by_key: MutableMapping[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in self._store.records():
            self._index_record(record)
        self._feature_write_counter = get_counter(
            "ml_file_feature_writes_total",
            "Total feature records persisted via file-backed store",
            ["mode"],
        )
        self._latency_hist = get_histogram(
            "ml_file_feature_latency_seconds",
            "Feature store flush latency",
            ["operation"],
        )

    def _index_record(self, record: dict[str, Any]) -> None:
        key = (record["feature_set_id"], record["instrument_id"])
        bucket = self._by_key[key]
        ts_event = record["ts_event"]
        idx = bisect_left([item["ts_event"] for item in bucket], ts_event)
        bucket.insert(idx, record)
        if len(bucket) > self._store.history_limit:
            del bucket[0]

    def write_features(
        self,
        feature_set_id: str | None = None,
        instrument_id: str | None = None,
        features: dict[str, float] | None = None,
        ts_event: int | None = None,
        ts_init: int | None = None,
        data: Any | None = None,
    ) -> None:
        feature_set = feature_set_id or "default"
        instrument = instrument_id or "UNKNOWN"
        payload = FeatureData(
            feature_set_id=feature_set,
            instrument_id=instrument,
            values=features or {},
            _ts_event=ts_event,
            _ts_init=ts_init,
        )
        record = {
            "feature_set_id": feature_set,
            "instrument_id": instrument,
            "values": payload.feature_values,
            "ts_event": payload.ts_event,
            "ts_init": payload.ts_init,
        }
        self._store.append(record)
        self._index_record(record)
        self._feature_write_counter.labels(mode="write_features").inc()

    def write_batch(self, data: Sequence[FeatureData]) -> None:
        for item in data:
            self.write_features(
                feature_set_id=item.feature_set_id,
                instrument_id=item.instrument_id,
                features=item.feature_values,
                ts_event=item.ts_event,
                ts_init=item.ts_init,
            )

    def get_latest_at_or_before(self, instrument_id: str, ts_event: int) -> dict[str, float] | None:
        matches: list[dict[str, Any]] = []
        for (feature_set, inst), bucket in self._by_key.items():
            if inst != instrument_id:
                continue
            idx = bisect_left([rec["ts_event"] for rec in bucket], ts_event)
            if idx < len(bucket) and bucket[idx]["ts_event"] == ts_event:
                matches.append(bucket[idx])
            elif idx > 0:
                matches.append(bucket[idx - 1])
        if not matches:
            return None
        latest = max(matches, key=lambda rec: rec["ts_event"])
        return {str(k): float(v) for k, v in latest["values"].items()}

    def flush(self) -> None:
        start = time.perf_counter()
        self._store.flush()
        self._latency_hist.labels(operation="flush").observe(time.perf_counter() - start)

    def compute_realtime(
        self,
        bar: Any,
        store: bool = True,
        indicator_manager: Any | None = None,
    ) -> dict[str, float] | None:
        _LOGGER.debug("FileFeatureStore.compute_realtime noop", extra={"store": store})
        return None

    # Compatibility helpers used in tests
    def get_statistics(self, start_ns: int | None = None, end_ns: int | None = None) -> dict[str, Any]:
        records = self._store.records()
        return {
            "count": len(records),
            "ts_min": min((rec["ts_event"] for rec in records), default=None),
            "ts_max": max((rec["ts_event"] for rec in records), default=None),
            "start_ns": start_ns,
            "end_ns": end_ns,
        }


class FileModelStore(ModelStoreProtocol):
    """Model prediction store backed by JSONL persistence."""

    def __init__(self, *, base_path: Path, history_limit: int = _DEFAULT_HISTORY_LIMIT) -> None:
        self._paths = base_path
        self._paths.mkdir(parents=True, exist_ok=True)
        self._store = _JsonLineStore(self._paths / "predictions.jsonl", history_limit)
        self._by_model: MutableMapping[str, list[dict[str, Any]]] = defaultdict(list)
        for record in self._store.records():
            self._index_record(record)
        self._write_counter = get_counter(
            "ml_file_model_writes_total",
            "Total model predictions written to file store",
            ["mode"],
        )

    def _index_record(self, record: dict[str, Any]) -> None:
        model_id = record["model_id"]
        bucket = self._by_model[model_id]
        ts_event = record["ts_event"]
        idx = bisect_left([item["ts_event"] for item in bucket], ts_event)
        bucket.insert(idx, record)
        if len(bucket) > self._store.history_limit:
            del bucket[0]

    def write_prediction(
        self,
        model_id: str,
        instrument_id: str,
        prediction: float,
        confidence: float,
        features: dict[str, float],
        inference_time_ms: float,
        ts_event: int,
        is_live: bool = False,
    ) -> None:
        payload = ModelPrediction(
            model_id=model_id,
            instrument_id=instrument_id,
            prediction=prediction,
            confidence=confidence,
            features_used=features,
            inference_time_ms=inference_time_ms,
            _ts_event=ts_event,
            _ts_init=ts_event,
            is_live=is_live,
        )
        record = {
            "model_id": payload.model_id,
            "instrument_id": payload.instrument_id,
            "prediction": float(payload.prediction),
            "confidence": float(payload.confidence),
            "features_used": payload.features_used,
            "inference_time_ms": float(payload.inference_time_ms),
            "ts_event": payload.ts_event,
            "ts_init": payload.ts_init,
            "is_live": bool(payload.is_live),
        }
        self._store.append(record)
        self._index_record(record)
        self._write_counter.labels(mode="write_prediction").inc()

    def write_batch(self, data: Sequence[ModelPrediction], emit_events: bool = True) -> None:
        for item in data:
            self.write_prediction(
                model_id=item.model_id,
                instrument_id=item.instrument_id,
                prediction=item.prediction,
                confidence=item.confidence,
                features=item.features_used,
                inference_time_ms=item.inference_time_ms,
                ts_event=item.ts_event,
                is_live=item.is_live,
            )
        if emit_events:
            self._write_counter.labels(mode="batch_emit").inc()

    def read_predictions(
        self,
        model_id: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> pd.DataFrame:
        bucket = [
            rec
            for rec in self._by_model.get(model_id, [])
            if rec["instrument_id"] == instrument_id and start_ns <= rec["ts_event"] <= end_ns
        ]
        return pd.DataFrame(bucket)

    def get_model_performance(
        self,
        model_id: str,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> dict[str, Any]:
        records = self._by_model.get(model_id, [])
        filtered = [
            rec
            for rec in records
            if (start_ns is None or rec["ts_event"] >= start_ns)
            and (end_ns is None or rec["ts_event"] <= end_ns)
        ]
        return {
            "count": len(filtered),
            "avg_confidence": float(pd.Series([rec["confidence"] for rec in filtered]).mean())
            if filtered
            else 0.0,
        }

    def flush(self) -> None:
        self._store.flush()

    # Non-protocol helpers for compatibility
    def read_latest_predictions(
        self,
        model_id: str,
        instrument_id: str | None = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        bucket = self._by_model.get(model_id, [])
        if instrument_id:
            bucket = [rec for rec in bucket if rec["instrument_id"] == instrument_id]
        return pd.DataFrame(bucket[-limit:])


class FileStrategyStore(StrategyStoreProtocol):
    """Strategy signal persistence using JSONL with per-strategy indexes."""

    def __init__(self, *, base_path: Path, history_limit: int = _DEFAULT_HISTORY_LIMIT) -> None:
        self._paths = base_path
        self._paths.mkdir(parents=True, exist_ok=True)
        self._store = _JsonLineStore(self._paths / "signals.jsonl", history_limit)
        self._by_strategy: MutableMapping[str, list[dict[str, Any]]] = defaultdict(list)
        for record in self._store.records():
            self._index_record(record)
        self._write_counter = get_counter(
            "ml_file_strategy_signals_total",
            "Total strategy signals persisted via file store",
            ["mode"],
        )

    def _index_record(self, record: dict[str, Any]) -> None:
        strategy_id = record["strategy_id"]
        bucket = self._by_strategy[strategy_id]
        ts_event = record["ts_event"]
        idx = bisect_left([item["ts_event"] for item in bucket], ts_event)
        bucket.insert(idx, record)
        if len(bucket) > self._store.history_limit:
            del bucket[0]

    def write_signal(
        self,
        strategy_id: str,
        instrument_id: str,
        signal_type: str,
        strength: float,
        model_predictions: dict[str, float],
        risk_metrics: dict[str, float],
        execution_params: dict[str, Any],
        ts_event: int,
        is_live: bool = False,
    ) -> None:
        record = {
            "strategy_id": strategy_id,
            "instrument_id": instrument_id,
            "signal_type": signal_type,
            "strength": float(strength),
            "model_predictions": dict(model_predictions),
            "risk_metrics": dict(risk_metrics),
            "execution_params": dict(execution_params),
            "ts_event": int(ts_event),
            "ts_init": int(ts_event),
            "is_live": bool(is_live),
        }
        self._store.append(record)
        self._index_record(record)
        self._write_counter.labels(mode="write_signal").inc()

    def write_batch(self, data: Sequence[StrategySignal]) -> None:
        for item in data:
            self.write_signal(
                strategy_id=item.strategy_id,
                instrument_id=item.instrument_id,
                signal_type=item.signal_type,
                strength=item.strength,
                model_predictions=dict(getattr(item, "model_predictions", {})),
                risk_metrics=dict(getattr(item, "risk_metrics", {})),
                execution_params=dict(getattr(item, "execution_params", {})),
                ts_event=item.ts_event,
                is_live=bool(getattr(item, "is_live", False)),
            )
        self._write_counter.labels(mode="batch").inc()

    def read_signals(
        self,
        strategy_id: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> pd.DataFrame:
        bucket = [
            rec
            for rec in self._by_strategy.get(strategy_id, [])
            if rec["instrument_id"] == instrument_id and start_ns <= rec["ts_event"] <= end_ns
        ]
        return pd.DataFrame(bucket)

    def get_strategy_performance(
        self,
        strategy_id: str,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> dict[str, Any]:
        bucket = [
            rec
            for rec in self._by_strategy.get(strategy_id, [])
            if (start_ns is None or rec["ts_event"] >= start_ns)
            and (end_ns is None or rec["ts_event"] <= end_ns)
        ]
        return {
            "count": len(bucket),
            "avg_strength": float(pd.Series([rec["strength"] for rec in bucket]).mean())
            if bucket
            else 0.0,
        }

    def get_signal_distribution(
        self,
        strategy_id: str | None = None,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> dict[str, int]:
        bucket: Iterable[dict[str, Any]]
        if strategy_id is None:
            bucket = (rec for records in self._by_strategy.values() for rec in records)
        else:
            bucket = self._by_strategy.get(strategy_id, [])
        counts: MutableMapping[str, int] = defaultdict(int)
        for rec in bucket:
            if start_ns is not None and rec["ts_event"] < start_ns:
                continue
            if end_ns is not None and rec["ts_event"] > end_ns:
                continue
            counts[rec["signal_type"]] += 1
        return dict(counts)

    def flush(self) -> None:
        self._store.flush()


class FileDataStore:
    """Simplified data-store facade that records dataset events to JSONL."""

    def __init__(self, *, base_path: Path, history_limit: int = _DEFAULT_HISTORY_LIMIT) -> None:
        self._paths = base_path
        self._paths.mkdir(parents=True, exist_ok=True)
        self._store = _JsonLineStore(self._paths / "events.jsonl", history_limit)
        self._ingestion_dir = self._paths / "ingestion"
        self._ingestion_dir.mkdir(parents=True, exist_ok=True)
        self._event_counter = get_counter(
            "ml_file_datastore_events_total",
            "Total dataset events emitted via file-backed datastore",
            ["stage", "status"],
        )

    def write_ingestion(
        self,
        *,
        dataset_id: str,
        records: list[dict[str, Any]] | pd.DataFrame,
        source: str,
        run_id: str,
        instrument_id: str | None = None,
    ) -> int:
        if isinstance(records, pd.DataFrame):
            frame = records.copy()
        else:
            frame = pd.DataFrame.from_records(records)
        if frame.empty:
            return 0
        instrument = instrument_id or "UNKNOWN"
        ts_min = int(frame.get("ts_event", pd.Series(dtype="int64")).min() or 0)
        ts_max = int(frame.get("ts_event", pd.Series(dtype="int64")).max() or ts_min)
        target_dir = self._ingestion_dir / dataset_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{instrument}_{ts_min}_{ts_max}.jsonl"
        frame.to_json(target_path, orient="records", lines=True, force_ascii=False)
        try:
            source_enum = Source[source.upper()]
        except (KeyError, AttributeError):
            try:
                source_enum = Source(source)
            except Exception:
                source_enum = Source.HISTORICAL

        self.emit_event(
            dataset_id=dataset_id,
            instrument_id=instrument,
            stage=Stage.DATA_INGESTED,
            source=source_enum,
            run_id=run_id,
            ts_min=ts_min,
            ts_max=ts_max,
            count=int(frame.shape[0]),
            status=EventStatus.SUCCESS,
            metadata={"artifact_path": str(target_path)},
        )
        return int(frame.shape[0])

    def emit_event(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        stage: Stage,
        source: Source,
        run_id: str,
        ts_min: int,
        ts_max: int,
        count: int,
        status: EventStatus,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        corr = make_correlation_id(
            run_id=run_id,
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            ts_min=ts_min,
            ts_max=ts_max,
            count=count,
        )
        record = {
            "dataset_id": dataset_id,
            "instrument_id": instrument_id,
            "stage": stage.value,
            "source": source.value,
            "run_id": run_id,
            "ts_min": ts_min,
            "ts_max": ts_max,
            "count": count,
            "status": status.value,
            "error": error,
            "metadata": {"correlation_id": corr, **(metadata or {})},
        }
        self._store.append(record)
        self._event_counter.labels(stage=stage.value, status=status.value).inc()

    def flush(self) -> None:
        self._store.flush()

    # Compatibility hooks --------------------------------------------------
    def get_statistics(self) -> dict[str, Any]:
        records = self._store.records()
        return {
            "count": len(records),
            "stages": list({rec["stage"] for rec in records}),
        }

    def validate_configuration(self) -> list[str]:
        return []

    def get_health_status(self) -> dict[str, Any]:
        return {"healthy": True, "records": len(self._store.records())}

    def get_performance_metrics(self) -> dict[str, float]:
        return {}

    def is_healthy(self) -> bool:
        return True
