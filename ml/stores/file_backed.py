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
from collections.abc import Mapping
from collections.abc import MutableMapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from ml._imports import HAS_POLARS
from ml._imports import pl
from ml.common.correlation import make_correlation_id
from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from ml.common.timestamps import sanitize_timestamp_ns
from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.stores.base import FeatureData
from ml.stores.base import ModelPrediction
from ml.stores.base import StrategySignal
from ml.stores.earnings_store import DummyEarningsStore
from ml.stores.protocols import EarningsStoreProtocol
from ml.stores.protocols import FeatureStoreProtocol
from ml.stores.protocols import ModelStoreProtocol
from ml.stores.protocols import StrategyStoreProtocol


_LOGGER = logging.getLogger(__name__)

_DEFAULT_HISTORY_LIMIT = 1_000
_FILE_STORAGE_LABEL = "file"

__all__ = [
    "FileDataStore",
    "FileEarningsStore",
    "FileFeatureStore",
    "FileModelStore",
    "FileStrategyStore",
]


if TYPE_CHECKING:  # pragma: no cover - typing import only
    from ml.stores.validation_types import DataEvent
else:
    DataEvent = Any  # type: ignore[misc,assignment]

pl = cast(Any, pl)
_PL = pl


def _make_data_event(**kwargs: Any) -> DataEvent:
    """Create a `DataEvent` instance without causing import cycles."""
    from ml.stores.validation_types import DataEvent as _DataEvent  # Local import to avoid cyclic dependency

    return _DataEvent(**kwargs)


def _ensure_polars() -> None:
    """Ensure polars is available before performing parquet operations."""
    if not HAS_POLARS:  # pragma: no cover - defensive guard
        raise RuntimeError(
            "File-backed earnings storage requires the 'polars' package. "
            "Install polars or disable file fallback via ML_FILE_STORE_PATH.",
        )


@lru_cache(maxsize=1)
def _actuals_schema() -> dict[str, Any]:
    _ensure_polars()
    return {
        "ticker": _PL.Utf8,
        "period_end": _PL.Utf8,
        "filing_date": _PL.Utf8,
        "ts_event": _PL.Int64,
        "ts_init": _PL.Int64,
        "eps_basic": _PL.Float64,
        "eps_diluted": _PL.Float64,
        "revenue": _PL.Float64,
        "net_income": _PL.Float64,
        "operating_income": _PL.Float64,
        "shares_outstanding": _PL.Int64,
        "filing_type": _PL.Utf8,
        "fiscal_year": _PL.Int32,
        "fiscal_quarter": _PL.Int32,
        "data_source": _PL.Utf8,
    }


@lru_cache(maxsize=1)
def _estimates_schema() -> dict[str, Any]:
    _ensure_polars()
    return {
        "ticker": _PL.Utf8,
        "estimate_date": _PL.Utf8,
        "period_end": _PL.Utf8,
        "ts_event": _PL.Int64,
        "ts_init": _PL.Int64,
        "eps_consensus": _PL.Float64,
        "revenue_consensus": _PL.Float64,
        "num_analysts": _PL.Int32,
        "data_source": _PL.Utf8,
    }


def _empty_frame(schema: Mapping[str, Any]) -> Any:
    """Return an empty Polars frame respecting the provided schema."""
    _ensure_polars()
    columns = {
        name: _PL.Series(name=name, values=[], dtype=dtype)
        for name, dtype in schema.items()
    }
    return _PL.DataFrame(columns)


def _frame_from_record(record: Mapping[str, Any], schema: Mapping[str, Any]) -> Any:
    """Construct a single-row Polars frame from ``record`` enforcing ``schema``."""
    _ensure_polars()
    frame = _PL.DataFrame({name: [record.get(name)] for name in schema.keys()})
    return frame.with_columns(
        _PL.col(name).cast(dtype, strict=False) for name, dtype in schema.items()
    ).select(list(schema.keys()))


def _align_frame(frame: Any, schema: Mapping[str, Any]) -> Any:
    """Align ``frame`` to the desired ``schema`` ordering and types."""
    _ensure_polars()
    aligned = frame
    for name, dtype in schema.items():
        if name not in aligned.columns:
            aligned = aligned.with_columns(_PL.lit(None, dtype=dtype).alias(name))
    return aligned.select(list(schema.keys())).with_columns(
        _PL.col(name).cast(dtype, strict=False) for name, dtype in schema.items()
    )

@dataclass(slots=True)
class _JsonLineStore:
    """
    Utility to append and reload JSONL datasets with locking and metrics.
    """

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
    def get_statistics(
        self, start_ns: int | None = None, end_ns: int | None = None
    ) -> dict[str, Any]:
        records = self._store.records()
        return {
            "count": len(records),
            "ts_min": min((rec["ts_event"] for rec in records), default=None),
            "ts_max": max((rec["ts_event"] for rec in records), default=None),
            "start_ns": start_ns,
            "end_ns": end_ns,
        }


class FileModelStore(ModelStoreProtocol):
    """
    Model prediction store backed by JSONL persistence.
    """

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
            "avg_confidence": (
                float(pd.Series([rec["confidence"] for rec in filtered]).mean())
                if filtered
                else 0.0
            ),
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
    """
    Strategy signal persistence using JSONL with per-strategy indexes.
    """

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

    def write_signals(self, data: Sequence[StrategySignal]) -> None:
        """
        Compat helper mirroring the DB-backed store signature for legacy callers.
        """
        self.write_batch(data)

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
            "avg_strength": (
                float(pd.Series([rec["strength"] for rec in bucket]).mean()) if bucket else 0.0
            ),
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


class FileEarningsStore(EarningsStoreProtocol):
    """Parquet-backed earnings store used for file-system fallbacks."""

    def __init__(self, *, base_path: Path, history_limit: int = 10_000) -> None:
        _ensure_polars()
        self._base_path = base_path
        self._history_limit = history_limit
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._actuals_path = self._base_path / "actuals.parquet"
        self._estimates_path = self._base_path / "estimates.parquet"
        self._lock = RLock()
        self._actuals_schema = _actuals_schema()
        self._estimates_schema = _estimates_schema()
        self._actuals = self._load_frame(self._actuals_path, self._actuals_schema)
        self._estimates = self._load_frame(self._estimates_path, self._estimates_schema)
        self._actuals_dirty = False
        self._estimates_dirty = False
        _LOGGER.info("FileEarningsStore initialized at %s", self._base_path)

    def _load_frame(self, path: Path, schema: Mapping[str, Any]) -> Any:
        if path.exists():
            try:
                frame = _PL.read_parquet(path)
            except Exception as exc:  # pragma: no cover - guarded IO path
                _LOGGER.warning("Failed to load parquet '%s': %s", path, exc)
                frame = _empty_frame(schema)
        else:
            frame = _empty_frame(schema)
        return _align_frame(frame, schema)

    def _trim_history(self, frame: Any, key_columns: Sequence[str]) -> Any:
        if self._history_limit <= 0 or frame.height <= self._history_limit:
            return frame
        sort_columns = list(key_columns) + ["ts_event"]
        descending = [False] * len(key_columns) + [False]
        return frame.sort(sort_columns, descending=descending).tail(self._history_limit)

    def write_actuals(
        self,
        ticker: str,
        period_end: str,
        filing_date: str,
        eps_diluted: float | None,
        revenue: float | None,
        ts_event: int,
        ts_init: int,
        eps_basic: float | None = None,
        net_income: float | None = None,
        operating_income: float | None = None,
        shares_outstanding: int | None = None,
        filing_type: str | None = None,
        fiscal_year: int | None = None,
        fiscal_quarter: int | None = None,
    ) -> None:
        record = {
            "ticker": str(ticker),
            "period_end": str(period_end),
            "filing_date": str(filing_date),
            "ts_event": sanitize_timestamp_ns(int(ts_event), context="file_earnings_store.write_actuals:ts_event"),
            "ts_init": sanitize_timestamp_ns(int(ts_init), context="file_earnings_store.write_actuals:ts_init"),
            "eps_basic": float(eps_basic) if eps_basic is not None else None,
            "eps_diluted": float(eps_diluted) if eps_diluted is not None else None,
            "revenue": float(revenue) if revenue is not None else None,
            "net_income": float(net_income) if net_income is not None else None,
            "operating_income": float(operating_income) if operating_income is not None else None,
            "shares_outstanding": int(shares_outstanding) if shares_outstanding is not None else None,
            "filing_type": str(filing_type) if filing_type is not None else None,
            "fiscal_year": int(fiscal_year) if fiscal_year is not None else None,
            "fiscal_quarter": int(fiscal_quarter) if fiscal_quarter is not None else None,
            "data_source": _FILE_STORAGE_LABEL,
        }

        with self._lock:
            base = self._actuals.filter(
                ~(
                    (_PL.col("ticker") == record["ticker"]) & (_PL.col("period_end") == record["period_end"])
                ),
            )
            appended = _PL.concat([base, _frame_from_record(record, self._actuals_schema)], how="vertical")
            trimmed = self._trim_history(appended, ("ticker", "period_end"))
            self._actuals = _align_frame(
                trimmed.sort(["ticker", "period_end", "ts_event"], descending=[False, True, True]),
                self._actuals_schema,
            )
            self._actuals_dirty = True

    def write_estimates(
        self,
        ticker: str,
        estimate_date: str,
        period_end: str,
        eps_consensus: float | None,
        ts_event: int,
        ts_init: int,
        revenue_consensus: float | None = None,
        num_analysts: int | None = None,
    ) -> None:
        record = {
            "ticker": str(ticker),
            "estimate_date": str(estimate_date),
            "period_end": str(period_end),
            "ts_event": sanitize_timestamp_ns(int(ts_event), context="file_earnings_store.write_estimates:ts_event"),
            "ts_init": sanitize_timestamp_ns(int(ts_init), context="file_earnings_store.write_estimates:ts_init"),
            "eps_consensus": float(eps_consensus) if eps_consensus is not None else None,
            "revenue_consensus": float(revenue_consensus) if revenue_consensus is not None else None,
            "num_analysts": int(num_analysts) if num_analysts is not None else None,
            "data_source": _FILE_STORAGE_LABEL,
        }

        with self._lock:
            base = self._estimates.filter(
                ~(
                    (_PL.col("ticker") == record["ticker"]) &
                    (_PL.col("period_end") == record["period_end"]) &
                    (_PL.col("estimate_date") == record["estimate_date"])
                ),
            )
            appended = _PL.concat([base, _frame_from_record(record, self._estimates_schema)], how="vertical")
            trimmed = self._trim_history(appended, ("ticker", "period_end", "estimate_date"))
            self._estimates = _align_frame(
                trimmed.sort(
                    ["ticker", "period_end", "estimate_date", "ts_event"],
                    descending=[False, True, True, True],
                ),
                self._estimates_schema,
            )
            self._estimates_dirty = True

    def get_actuals(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            frame = self._actuals.filter(_PL.col("ticker") == str(ticker))
            if start_date is not None:
                frame = frame.filter(_PL.col("period_end") >= str(start_date))
            if end_date is not None:
                frame = frame.filter(_PL.col("period_end") <= str(end_date))
            if as_of_ts is not None:
                frame = frame.filter(_PL.col("ts_event") < int(as_of_ts))

            frame = frame.sort(["period_end", "ts_event"], descending=[True, True])
            return cast(list[dict[str, Any]], frame.to_dicts())

    def get_estimates(
        self,
        ticker: str,
        period_end: str,
        as_of_ts: int | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            frame = self._estimates.filter(
                (_PL.col("ticker") == str(ticker)) & (_PL.col("period_end") == str(period_end)),
            )
            if as_of_ts is not None:
                frame = frame.filter(_PL.col("ts_event") < int(as_of_ts))

            frame = frame.sort(["estimate_date", "ts_event"], descending=[True, True])
            rows = cast(list[dict[str, Any]], frame.to_dicts())
            if not rows:
                return None
            return rows[0]

    def flush(self) -> None:
        with self._lock:
            if self._actuals_dirty:
                _align_frame(self._actuals, self._actuals_schema).write_parquet(self._actuals_path)
                self._actuals_dirty = False
            if self._estimates_dirty:
                _align_frame(self._estimates, self._estimates_schema).write_parquet(self._estimates_path)
                self._estimates_dirty = False


class FileDataStore:
    """
    Simplified data-store facade that records dataset events to JSONL.
    """

    def __init__(
        self,
        *,
        base_path: Path,
        history_limit: int = _DEFAULT_HISTORY_LIMIT,
        earnings_store: EarningsStoreProtocol | None = None,
    ) -> None:
        self._paths = base_path
        self._paths.mkdir(parents=True, exist_ok=True)
        self._store = _JsonLineStore(self._paths / "events.jsonl", history_limit)
        self._ingestion_dir = self._paths / "ingestion"
        self._ingestion_dir.mkdir(parents=True, exist_ok=True)
        self._features_by_instrument: MutableMapping[str, list[dict[str, Any]]] = defaultdict(list)
        self._event_counter = get_counter(
            "ml_file_datastore_events_total",
            "Total dataset events emitted via file-backed datastore",
            ["stage", "status"],
        )
        self._earnings_store: EarningsStoreProtocol = earnings_store or DummyEarningsStore()

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

    def write_earnings_actual(
        self,
        *,
        ticker: str,
        period_end: str,
        filing_date: str,
        eps_diluted: float | None,
        revenue: float | None,
        ts_event: int,
        ts_init: int,
        eps_basic: float | None = None,
        net_income: float | None = None,
        operating_income: float | None = None,
        shares_outstanding: int | None = None,
        filing_type: str | None = None,
        fiscal_year: int | None = None,
        fiscal_quarter: int | None = None,
        source: str = Source.HISTORICAL.value,
        run_id: str | None = None,
    ) -> DataEvent:
        dataset_id = "ml.earnings_actuals"
        run_id_local = run_id or f"file_earnings_actual_{time.time_ns()}"
        ts_event_ns = sanitize_timestamp_ns(
            int(ts_event),
            context="file_data_store.write_earnings_actual:ts_event",
        )
        ts_init_ns = sanitize_timestamp_ns(
            int(ts_init),
            context="file_data_store.write_earnings_actual:ts_init",
        )
        self._earnings_store.write_actuals(
            ticker=ticker,
            period_end=period_end,
            filing_date=filing_date,
            eps_diluted=eps_diluted,
            revenue=revenue,
            ts_event=ts_event_ns,
            ts_init=ts_init_ns,
            eps_basic=eps_basic,
            net_income=net_income,
            operating_income=operating_income,
            shares_outstanding=shares_outstanding,
            filing_type=filing_type,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
        )

        source_enum: Source
        if isinstance(source, Source):
            source_enum = source
            source_value = source.value
        else:
            try:
                source_enum = Source(str(source))
            except Exception:
                source_enum = Source.HISTORICAL
            source_value = source_enum.value

        event = _make_data_event(
            event_id=f"{run_id_local}_{dataset_id}_{time.time_ns()}",
            dataset_id=dataset_id,
            instrument_id=ticker,
            operation="write_earnings_actual",
            source=source_value,
            run_id=run_id_local,
            ts_min=ts_event_ns,
            ts_max=ts_event_ns,
            record_count=1,
            status=EventStatus.SUCCESS.value,
            metadata={"storage": _FILE_STORAGE_LABEL},
        )

        self.emit_event(
            dataset_id=dataset_id,
            instrument_id=ticker,
            stage=Stage.DATA_INGESTED,
            source=source_enum,
            run_id=run_id_local,
            ts_min=ts_event_ns,
            ts_max=ts_event_ns,
            count=1,
            status=EventStatus.SUCCESS,
        )
        return event

    def write_earnings_estimate(
        self,
        *,
        ticker: str,
        estimate_date: str,
        period_end: str,
        eps_consensus: float | None,
        ts_event: int,
        ts_init: int,
        revenue_consensus: float | None = None,
        num_analysts: int | None = None,
        source: str = Source.HISTORICAL.value,
        run_id: str | None = None,
    ) -> DataEvent:
        dataset_id = "ml.earnings_estimates"
        run_id_local = run_id or f"file_earnings_estimate_{time.time_ns()}"
        ts_event_ns = sanitize_timestamp_ns(
            int(ts_event),
            context="file_data_store.write_earnings_estimate:ts_event",
        )
        ts_init_ns = sanitize_timestamp_ns(
            int(ts_init),
            context="file_data_store.write_earnings_estimate:ts_init",
        )
        self._earnings_store.write_estimates(
            ticker=ticker,
            estimate_date=estimate_date,
            period_end=period_end,
            eps_consensus=eps_consensus,
            ts_event=ts_event_ns,
            ts_init=ts_init_ns,
            revenue_consensus=revenue_consensus,
            num_analysts=num_analysts,
        )

        if isinstance(source, Source):
            source_enum = source
            source_value = source.value
        else:
            try:
                source_enum = Source(str(source))
            except Exception:
                source_enum = Source.HISTORICAL
            source_value = source_enum.value

        event = _make_data_event(
            event_id=f"{run_id_local}_{dataset_id}_{time.time_ns()}",
            dataset_id=dataset_id,
            instrument_id=ticker,
            operation="write_earnings_estimate",
            source=source_value,
            run_id=run_id_local,
            ts_min=ts_event_ns,
            ts_max=ts_event_ns,
            record_count=1,
            status=EventStatus.SUCCESS.value,
            metadata={"storage": _FILE_STORAGE_LABEL},
        )

        self.emit_event(
            dataset_id=dataset_id,
            instrument_id=ticker,
            stage=Stage.DATA_INGESTED,
            source=source_enum,
            run_id=run_id_local,
            ts_min=ts_event_ns,
            ts_max=ts_event_ns,
            count=1,
            status=EventStatus.SUCCESS,
        )
        return event

    def get_earnings_actuals_at_or_before(
        self,
        *,
        ticker: str,
        ts_event: int,
        limit: int = 5,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        records = self._earnings_store.get_actuals(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            as_of_ts=int(ts_event),
        )
        if limit > 0:
            return records[:limit]
        return records

    def get_earnings_estimate_at_or_before(
        self,
        *,
        ticker: str,
        period_end: str,
        ts_event: int,
    ) -> dict[str, Any] | None:
        return self._earnings_store.get_estimates(
            ticker=ticker,
            period_end=period_end,
            as_of_ts=int(ts_event),
        )

    def flush(self) -> None:
        self._store.flush()
        try:
            self._earnings_store.flush()
        except Exception:  # pragma: no cover - defensive logging path
            _LOGGER.debug("Earnings store flush failed", exc_info=True)

    def write_features(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        features: Mapping[str, float],
        ts_event: int,
        ts_init: int,
    ) -> None:
        """
        Persist feature snapshots in-memory so actors can read them during fallback.
        """
        record = {
            "dataset_id": dataset_id,
            "instrument_id": instrument_id,
            "values": dict(features),
            "ts_event": int(ts_event),
            "ts_init": int(ts_init),
        }
        bucket = self._features_by_instrument[instrument_id]
        idx = bisect_left([item["ts_event"] for item in bucket], record["ts_event"])
        bucket.insert(idx, record)
        if len(bucket) > self._store.history_limit:
            del bucket[0]

    def get_features_at_or_before(
        self, instrument_id: str, ts_event: int
    ) -> dict[str, float] | None:
        """
        Return the most recent feature mapping for an instrument at or before
        ``ts_event``.
        """
        bucket = self._features_by_instrument.get(instrument_id)
        if not bucket:
            return None
        idx = bisect_left([item["ts_event"] for item in bucket], int(ts_event))
        candidate: dict[str, Any]
        if idx < len(bucket) and bucket[idx]["ts_event"] == int(ts_event):
            candidate = bucket[idx]
        elif idx > 0:
            candidate = bucket[idx - 1]
        else:
            return None
        return {str(key): float(value) for key, value in candidate["values"].items()}

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
