"""
ModelStore service layer.

This module extracts typed, testable services from the ModelStore god class while
preserving the public API and behavior. Services are dependency-injected with a small
protocol that the facade already satisfies via existing mixins.

Phase 1 scope: internal refactor only — no public API changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, SupportsFloat, cast

from sqlalchemy import text as _text

from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.stores.base import ModelPrediction
from ml.stores.mixins import sanitize_and_dedup
from ml.stores.protocols import LoggerLike
from ml.stores.protocols import ModelClearDepsStrict
from ml.stores.protocols import ModelEventDepsStrict
from ml.stores.protocols import ModelReadDepsStrict
from ml.stores.protocols import ModelWriteDepsStrict
from ml.stores.services.common_stats import resolve_table_name as _resolve_table_name


@dataclass(slots=True)
class ModelWriteService:
    """Pure persistence for model predictions."""

    deps: ModelWriteDepsStrict
    logger: LoggerLike

    def write_batch(self, data: list[ModelPrediction], publish_bus: bool = True) -> None:
        if not data:
            return
        values: list[dict[str, Any]] = []
        for item in data:
            values.append(
                {
                    "model_id": item.model_id,
                    "instrument_id": item.instrument_id,
                    "ts_event": item.ts_event,
                    "ts_init": item.ts_init,
                    "prediction": item.prediction,
                    "confidence": item.confidence,
                    "features_used": item.features_used if item.features_used else None,
                    "inference_time_ms": item.inference_time_ms,
                    "is_live": getattr(item, "is_live", False),
                }
            )
        values = sanitize_and_dedup(
            values,
            ts_event_field="ts_event",
            ts_init_field="ts_init",
            context="ModelStore.write_batch",
            key_fields=("model_id", "instrument_id", "ts_event"),
        )
        # Preserve historical patch point used by tests: if the deps object
        # (typically the ModelStore) exposes a _execute_write method, call it
        # so monkeypatches can intercept writes without touching internals.
        # Look up optional hook (used by tests) and call if present
        exec_write_obj = getattr(self.deps, "_execute_write", None)
        if callable(exec_write_obj):
            exec_write_fn = exec_write_obj
            exec_write_fn(values)
        else:
            self.execute_write(values, publish_bus=publish_bus)

    def execute_write(self, values: list[dict[str, object]], publish_bus: bool = True) -> None:
        if not values:
            return
        self.deps._execute_upsert_and_publish(
            values=values,
            ts_event_field="ts_event",
            ts_init_field="ts_init",
            context="ModelStore._execute_write",
            key_fields=("model_id", "instrument_id", "ts_event"),
            table=self.deps.model_predictions_table,
            conflict_cols=["model_id", "instrument_id", "ts_event"],
            update_cols=[
                "prediction",
                "confidence",
                "features_used",
                "inference_time_ms",
            ],
            dataset_id="predictions",
            stage=Stage.PREDICTION_EMITTED,
            instrument_key="instrument_id",
            ts_field="ts_event",
            run_id_batch="model_store_write",
            run_id_row="model_store_row",
            source="inference",
            logger=self.logger,
            publish_bus=publish_bus,
        )


@dataclass(slots=True)
class ModelQueryService:
    """Read/query operations for model predictions."""

    deps: ModelReadDepsStrict

    def read_predictions(
        self,
        *,
        model_id: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> object:
        table_name = _resolve_table_name(
            self.deps,
            attr="model_predictions_table",
            base="ml_model_predictions",
            allowed=None,
        )
        sql = _text(
            f"SELECT ts_event, prediction, confidence, features_used, inference_time_ms\n"  # nosec B608: table name resolved from store metadata
            f"FROM {table_name}\n"
            "WHERE model_id = :model_id\n"
            "  AND instrument_id = :instrument_id\n"
            "  AND ts_event >= :start_ns\n"
            "  AND ts_event < :end_ns\n"
            "ORDER BY ts_event"
        )
        params: dict[str, object] = {
            "model_id": model_id,
            "instrument_id": instrument_id,
            "start_ns": int(start_ns),
            "end_ns": int(end_ns),
        }
        return self.deps._execute_read(
            sql,
            params,
            columns=[
                "ts_event",
                "prediction",
                "confidence",
                "features_used",
                "inference_time_ms",
            ],
        )

    def read_latest_predictions(
        self,
        *,
        model_id: str,
        instrument_id: str | None = None,
        limit: int = 100,
    ) -> object:
        table_name = _resolve_table_name(
            self.deps,
            attr="model_predictions_table",
            base="ml_model_predictions",
            allowed=None,
        )
        where_parts: list[str] = ["model_id = :model_id"]
        params: dict[str, object] = {"model_id": model_id, "limit": int(limit)}
        if instrument_id is not None:
            where_parts.append("instrument_id = :instrument_id")
            params["instrument_id"] = instrument_id
        sql = _text(
            f"SELECT model_id,\n"  # nosec B608: table name resolved from store metadata
            "       instrument_id,\n"
            "       prediction,\n"
            "       confidence,\n"
            "       inference_time_ms,\n"
            "       is_live,\n"
            "       ts_event,\n"
            "       ts_init\n"
            f"FROM {table_name}\n"
            f"WHERE {' AND '.join(where_parts)}\n"
            "ORDER BY ts_event DESC\n"
            "LIMIT :limit"
        )
        return self.deps._execute_read(
            sql,
            params,
            columns=[
                "model_id",
                "instrument_id",
                "prediction",
                "confidence",
                "inference_time_ms",
                "is_live",
                "ts_event",
                "ts_init",
            ],
        )

    def read_range(
        self,
        *,
        start_ns: int,
        end_ns: int,
        instrument_id: str | None = None,
    ) -> object:
        table_name = _resolve_table_name(
            self.deps,
            attr="model_predictions_table",
            base="ml_model_predictions",
            allowed=None,
        )
        if instrument_id is None:
            sql = _text(
                f"SELECT model_id, instrument_id, ts_event, prediction, confidence, inference_time_ms\n"  # nosec B608: table name resolved from store metadata
                f"FROM {table_name}\n"
                "WHERE ts_event >= :start_ns AND ts_event < :end_ns\n"
                "ORDER BY ts_event"
            )
            params: dict[str, object] = {"start_ns": int(start_ns), "end_ns": int(end_ns)}
        else:
            sql = _text(
                f"SELECT model_id, instrument_id, ts_event, prediction, confidence, inference_time_ms\n"  # nosec B608: table name resolved from store metadata
                f"FROM {table_name}\n"
                "WHERE ts_event >= :start_ns AND ts_event < :end_ns\n"
                "  AND instrument_id = :instrument_id\n"
                "ORDER BY ts_event"
            )
            params = {
                "start_ns": int(start_ns),
                "end_ns": int(end_ns),
                "instrument_id": instrument_id,
            }
        return self.deps._execute_read(
            sql,
            params,
            columns=[
                "model_id",
                "instrument_id",
                "ts_event",
                "prediction",
                "confidence",
                "inference_time_ms",
            ],
        )

    def get_latest_by_instrument(self, *, instrument_id: str, limit: int = 1) -> object:
        table_name = _resolve_table_name(
            self.deps,
            attr="model_predictions_table",
            base="ml_model_predictions",
            allowed=None,
        )
        sql = _text(
            f"SELECT model_id, ts_event, prediction, confidence, inference_time_ms\n"  # nosec B608: table name resolved from store metadata
            f"FROM {table_name}\n"
            "WHERE instrument_id = :instrument_id\n"
            "ORDER BY ts_event DESC\n"
            "LIMIT :limit"
        )
        params: dict[str, object] = {"instrument_id": instrument_id, "limit": int(limit)}
        return self.deps._execute_read(
            sql,
            params,
            columns=[
                "model_id",
                "ts_event",
                "prediction",
                "confidence",
                "inference_time_ms",
            ],
        )

    def get_predictions(
        self,
        *,
        model_id: str,
        start_ns: int,
        end_ns: int,
        instrument_id: str | None = None,
    ) -> object:
        if instrument_id is None:
            table_name = _resolve_table_name(
                self.deps,
                attr="model_predictions_table",
                base="ml_model_predictions",
                allowed=None,
            )
            sql = _text(
                f"SELECT model_id, instrument_id, ts_event, prediction, confidence, inference_time_ms\n"  # nosec B608: table name resolved from store metadata
                f"FROM {table_name}\n"
                "WHERE model_id = :model_id\n"
                "  AND ts_event >= :start_ns AND ts_event < :end_ns\n"
                "ORDER BY instrument_id, ts_event"
            )
            params2: dict[str, object] = {
                "model_id": model_id,
                "start_ns": int(start_ns),
                "end_ns": int(end_ns),
            }
            return self.deps._execute_read(
                sql,
                params2,
                columns=[
                    "model_id",
                    "instrument_id",
                    "ts_event",
                    "prediction",
                    "confidence",
                    "inference_time_ms",
                ],
            )

        return self.read_predictions(
            model_id=model_id,
            instrument_id=instrument_id,
            start_ns=start_ns,
            end_ns=end_ns,
        )


@dataclass(slots=True)
class ModelStatsService:
    """Aggregate/statistical read operations for model predictions."""

    deps: ModelReadDepsStrict

    def get_statistics(self, *, start_ns: int | None = None, end_ns: int | None = None) -> dict[str, object]:
        from ml.stores.services.common_stats import build_time_conditions as _conds
        conditions, params = _conds(start_ns, end_ns, field="ts_event")

        table_name = _resolve_table_name(
            self.deps,
            attr="model_predictions_table",
            base="ml_model_predictions",
            allowed=None,
        )
        from ml.stores.services.common_stats import select_min_max_ts as _minmax
        minmax = _minmax(field="ts_event", min_alias="min_ts", max_alias="max_ts")
        base_sql = (
            "SELECT COUNT(*) as total_predictions, "  # nosec B608: table name resolved from store metadata
            "COUNT(DISTINCT model_id) as unique_models, "
            "COUNT(DISTINCT instrument_id) as unique_instruments, "
            "AVG(inference_time_ms) as avg_inference_ms, "
            "MAX(inference_time_ms) as max_inference_ms, "
            f"{minmax} "
            f"FROM {table_name} "
        )
        if conditions:
            base_sql += "WHERE " + " AND ".join(conditions)

        row = self.deps._fetch_one(_text(base_sql), params)
        if row:
            return {
                "total_predictions": row[0] or 0,
                "unique_models": row[1] or 0,
                "unique_instruments": row[2] or 0,
                "avg_inference_ms": float(cast(SupportsFloat, row[3])) if row[3] is not None else 0.0,
                "max_inference_ms": float(cast(SupportsFloat, row[4])) if row[4] is not None else 0.0,
                "min_timestamp_ns": row[5] or 0,
                "max_timestamp_ns": row[6] or 0,
            }
        return {
            "total_predictions": 0,
            "unique_models": 0,
            "unique_instruments": 0,
            "avg_inference_ms": 0.0,
            "max_inference_ms": 0.0,
            "min_timestamp_ns": 0,
            "max_timestamp_ns": 0,
        }


    def get_model_performance(
        self,
        *,
        model_id: str,
        start_ns: int | None = None,
        end_ns: int | None = None,
        hours_back: int | None = None,
    ) -> dict[str, object]:
        if hours_back is not None:
            import time as _time

            end_ns = _time.time_ns()
            start_ns = end_ns - int(hours_back * 3_600_000_000_000)

        from ml.stores.services.common_stats import build_time_conditions as _conds
        conditions, params = _conds(start_ns, end_ns, field="ts_event")
        conditions.insert(0, "model_id = :model_id")
        params2: dict[str, object] = dict(params)
        params2["model_id"] = model_id

        table_name = _resolve_table_name(
            self.deps,
            attr="model_predictions_table",
            base="ml_model_predictions",
            allowed=None,
        )
        from ml.stores.services.common_stats import select_latency_summary as _latency
        from ml.stores.services.common_stats import select_numeric_stats as _num
        latency = _latency(column="inference_time_ms", include_avg=True, percentiles=(0.50, 0.95, 0.99))
        conf_stats = _num(column="confidence", prefix="confidence", include_avg=True, include_stddev=True, include_min_max=False)
        sql = (
            "SELECT COUNT(*) as prediction_count, "  # nosec B608: table name resolved from store metadata
            f"{conf_stats}, "
            f"{latency} "
            f"FROM {table_name} WHERE " + " AND ".join(conditions)
        )

        row = self.deps._fetch_one(_text(sql), params2)
        if row:
            return {
                "prediction_count": row[0] or 0,
                "avg_confidence": float(cast(SupportsFloat, row[1])) if row[1] is not None else 0.0,
                "std_confidence": float(cast(SupportsFloat, row[2])) if row[2] is not None else 0.0,
                "avg_latency_ms": float(cast(SupportsFloat, row[3])) if row[3] is not None else 0.0,
                "p50_latency_ms": float(cast(SupportsFloat, row[4])) if row[4] is not None else 0.0,
                "p95_latency_ms": float(cast(SupportsFloat, row[5])) if row[5] is not None else 0.0,
                "p99_latency_ms": float(cast(SupportsFloat, row[6])) if row[6] is not None else 0.0,
            }
        return {
            "prediction_count": 0,
            "avg_confidence": 0.0,
            "std_confidence": 0.0,
            "avg_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
        }


@dataclass(slots=True)
class ModelClearService:
    """Deletion/cleanup operations for model predictions."""

    deps: ModelClearDepsStrict

    def clear(self, *, model_id: str | None = None, instrument_id: str | None = None) -> None:
        with self.deps.engine.begin() as conn:
            delete_stmt = self.deps.model_predictions_table.delete()
            if model_id is not None:
                delete_stmt = delete_stmt.where(self.deps.model_predictions_table.c.model_id == model_id)
            if instrument_id is not None:
                delete_stmt = delete_stmt.where(
                    self.deps.model_predictions_table.c.instrument_id == instrument_id,
                )
            conn.execute(delete_stmt)


@dataclass(slots=True)
class ModelEventService:
    """Event emission service for model predictions (registry/metrics)."""

    deps: ModelEventDepsStrict
    logger: LoggerLike

    def emit_prediction_events(self, predictions: list[ModelPrediction]) -> None:
        try:
            registry = self.deps._get_data_registry()
            if registry is None:
                return

            from collections import defaultdict

            grouped: dict[tuple[str, str], list[ModelPrediction]] = defaultdict(list)
            for pred in predictions:
                grouped[(pred.model_id, pred.instrument_id)].append(pred)

            for (model_id, instrument_id), group in grouped.items():
                if not group:
                    continue

                import time as _time
                import uuid as _uuid

                run_id = f"prediction_{model_id}_{_uuid.uuid4().hex[:8]}_{int(_time.time())}"
                ts_vals = [p.ts_event for p in group]
                ts_min, ts_max = min(ts_vals), max(ts_vals)
                src_enum = Source.LIVE if getattr(group[0], "is_live", False) else Source.HISTORICAL

                # Robust import to tolerate tests that stub event_emitter
                try:
                    from ml.common import event_emitter as _ee

                    _emit_wm = getattr(_ee, "emit_dataset_event_and_watermark", None)
                    _emit = getattr(_ee, "emit_dataset_event", None)
                except Exception:
                    _emit_wm = None
                    _emit = None

                # Ensure dataset is registered before emitting events
                try:
                    registry.get_manifest("predictions")
                except (ValueError, AttributeError):
                    # Dataset not registered, create a complete manifest compatible with
                    # both JSON and PostgreSQL backends.
                    import hashlib

                    from ml.registry.dataclasses import DatasetManifest
                    from ml.registry.dataclasses import DatasetType
                    from ml.registry.dataclasses import StorageKind

                    schema = {
                        "model_id": "string",
                        "instrument_id": "string",
                        "ts_event": "int64",
                        "ts_init": "int64",
                        "prediction": "float64",
                        "confidence": "float64",
                        "features_used": "json",
                        "inference_time_ms": "float64",
                        "is_live": "bool",
                    }
                    schema_hash = hashlib.sha256(str(schema).encode()).hexdigest()

                    manifest = DatasetManifest(
                        dataset_id="predictions",
                        dataset_type=DatasetType.PREDICTIONS,
                        storage_kind=StorageKind.POSTGRES,
                        location="ml_model_predictions",
                        partitioning={"by": "ts_event", "interval": "monthly"},
                        retention_days=365,
                        schema=schema,
                        ts_field="ts_event",
                        seq_field=None,
                        primary_keys=["model_id", "instrument_id", "ts_event"],
                        schema_hash=schema_hash,
                        constraints={
                            "nullability": {
                                "model_id": False,
                                "instrument_id": False,
                                "ts_event": False,
                                "ts_init": False,
                            }
                        },
                        lineage=[],
                        pipeline_signature="model_services_auto",
                        version="1.0.0",
                        metadata={
                            # Hints used by Postgres manifest reader
                            "ts_field": "ts_event",
                            "primary_keys": ["model_id", "instrument_id", "ts_event"],
                            "auto_registered": True,
                        },
                    )
                    try:
                        registry.register_dataset(manifest)
                        self.logger.info("Registered dataset 'predictions' in registry")
                    except Exception:
                        # Best-effort registration; continue with emission path.
                        self.logger.debug(
                            "Auto-registration of predictions dataset failed; continuing",
                            exc_info=True,
                            extra={
                                "component": "model_services",
                                "dataset_id": "predictions",
                                "model_id": model_id,
                                "instrument_id": instrument_id,
                            },
                        )

                # Emit event + watermark; if watermark path fails (e.g., FK not yet visible
                # under concurrent transactions), fall back to event-only emission.
                if callable(_emit_wm):
                    try:
                        _emit_wm(
                            registry,
                            dataset_id="predictions",
                            instrument_id=instrument_id,
                            stage=Stage.PREDICTION_EMITTED,
                            source=src_enum,
                            run_id=run_id,
                            ts_min=ts_min,
                            ts_max=ts_max,
                            count=len(group),
                            status=EventStatus.SUCCESS,
                            dataset_type="predictions",
                            component=model_id,
                            metadata={"model_id": model_id},
                        )
                    except Exception:
                        try:
                            registry.emit_event(
                                dataset_id="predictions",
                                instrument_id=instrument_id,
                                stage=Stage.PREDICTION_EMITTED,
                                source=src_enum,
                                run_id=run_id,
                                ts_min=ts_min,
                                ts_max=ts_max,
                                count=len(group),
                                status=EventStatus.SUCCESS,
                                metadata={"model_id": model_id},
                            )
                        except Exception:
                            # Event path must not impact hot writes
                            self.logger.debug(
                                "Registry emit_event fallback failed; dropping prediction event",
                                exc_info=True,
                                extra={
                                    "component": "model_services",
                                    "dataset_id": "predictions",
                                    "model_id": model_id,
                                    "instrument_id": instrument_id,
                                },
                            )
                else:
                    # Fallback: direct registry calls
                    try:
                        registry.emit_event(
                            dataset_id="predictions",
                            instrument_id=instrument_id,
                            stage=Stage.PREDICTION_EMITTED,
                            source=src_enum,
                            run_id=run_id,
                            ts_min=ts_min,
                            ts_max=ts_max,
                            count=len(group),
                            status=EventStatus.SUCCESS,
                            metadata={"model_id": model_id},
                        )
                        try:
                            registry.update_watermark(
                                dataset_id="predictions",
                                instrument_id=instrument_id,
                                source=src_enum,
                                last_success_ns=ts_max,
                                count=len(group),
                                completeness_pct=100.0,
                            )
                        except Exception:
                            # Watermark update is best-effort
                            self.logger.debug(
                                "Registry watermark update failed; continuing",
                                exc_info=True,
                                extra={
                                    "component": "model_services",
                                    "dataset_id": "predictions",
                                    "model_id": model_id,
                                    "instrument_id": instrument_id,
                                },
                            )
                    except Exception:
                        self.logger.debug(
                            "Registry emit_event failed; prediction event dropped",
                            exc_info=True,
                            extra={
                                "component": "model_services",
                                "dataset_id": "predictions",
                                "model_id": model_id,
                                "instrument_id": instrument_id,
                            },
                        )
        except Exception:
            self.logger.warning("Failed to emit prediction events", exc_info=True)
