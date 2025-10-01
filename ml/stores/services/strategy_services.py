"""
StrategyStore service layer.

This module extracts typed, testable services from the StrategyStore god class while
preserving the public API and behavior. Services are dependency-injected with a small
protocol that the facade already satisfies via existing mixins.

Phase 1 scope: internal refactor only — no public API changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, SupportsFloat, SupportsInt, cast

from sqlalchemy import text as _text

from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.stores.base import StrategySignal
from ml.stores.protocols import LoggerLike
from ml.stores.protocols import StrategyClearDepsStrict
from ml.stores.protocols import StrategyEventDepsStrict
from ml.stores.protocols import StrategyReadDepsStrict
from ml.stores.protocols import StrategyWriteDepsStrict
from ml.stores.services.common_stats import build_time_conditions as _time_conditions
from ml.stores.services.common_stats import resolve_table_name as _resolve_table_name


# Strict deps imported from ml.stores.protocols


@dataclass(slots=True)
class StrategySignalWriteService:
    """Pure persistence for strategy signals."""

    deps: StrategyWriteDepsStrict
    logger: LoggerLike

    def write_batch(self, data: list[StrategySignal], publish_bus: bool = True) -> None:
        if not data:
            return

        values: list[dict[str, Any]] = []
        for item in data:
            values.append(
                {
                    "strategy_id": item.strategy_id,
                    "instrument_id": item.instrument_id,
                    "ts_event": item.ts_event,
                    "ts_init": item.ts_init,
                    "signal_type": item.signal_type,
                    "strength": item.strength,
                    "model_predictions": item.model_predictions if item.model_predictions else None,
                    "risk_metrics": item.risk_metrics if item.risk_metrics else None,
                    "execution_params": item.execution_params if item.execution_params else None,
                    "is_live": getattr(item, "is_live", False),
                }
            )

        # Preserve historical patch point used by tests: if the deps object
        # (typically the StrategyStore) exposes a _execute_write method, call it
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
        # Delegate to shared upsert + publish helper
        self.deps._execute_upsert_and_publish(
            values=values,
            ts_event_field="ts_event",
            ts_init_field="ts_init",
            context="StrategyStore._execute_write",
            key_fields=("strategy_id", "instrument_id", "ts_event"),
            table=self.deps.strategy_signals_table,
            conflict_cols=["strategy_id", "instrument_id", "ts_event"],
            update_cols=[
                "signal_type",
                "strength",
                "model_predictions",
                "risk_metrics",
                "execution_params",
                "is_live",
            ],
            dataset_id="signals",
            stage=Stage.SIGNAL_EMITTED,
            instrument_key="instrument_id",
            ts_field="ts_event",
            run_id_batch="strategy_store_write",
            run_id_row="strategy_store_row",
            source="strategy",
            logger=self.logger,
            publish_bus=publish_bus,
        )


@dataclass(slots=True)
class StrategySignalQueryService:
    """Read/query operations for strategy signals."""

    deps: StrategyReadDepsStrict

    def read_signals(
        self,
        *,
        strategy_id: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> object:
        table_name = _resolve_table_name(
            self.deps,
            attr="strategy_signals_table",
            base="ml_strategy_signals",
            allowed={"ml_strategy_signals"},
        )
        sql = _text(  # nosec B608: table name validated via allowlist
            f"""
            SELECT ts_event, signal_type, strength, model_predictions, risk_metrics, execution_params
            FROM {table_name}
            WHERE strategy_id = :strategy_id
              AND instrument_id = :instrument_id
              AND ts_event >= :start_ns
              AND ts_event < :end_ns
            ORDER BY ts_event
            """
        )
        params: dict[str, object] = {
            "strategy_id": strategy_id,
            "instrument_id": instrument_id,
            "start_ns": int(start_ns),
            "end_ns": int(end_ns),
        }
        return self.deps._execute_read(
            sql,
            params,
            columns=[
                "ts_event",
                "signal_type",
                "strength",
                "model_predictions",
                "risk_metrics",
                "execution_params",
            ],
        )

    def read_range(
        self,
        *,
        start_ns: int,
        end_ns: int,
        instrument_id: str | None = None,
    ) -> object:
        params: dict[str, object] = {"start_ns": int(start_ns), "end_ns": int(end_ns)}
        where_parts = ["ts_event >= :start_ns", "ts_event < :end_ns"]
        if instrument_id is not None:
            where_parts.append("instrument_id = :instrument_id")
            params["instrument_id"] = instrument_id
        table_name = _resolve_table_name(
            self.deps,
            attr="strategy_signals_table",
            base="ml_strategy_signals",
            allowed={"ml_strategy_signals"},
        )
        sql = _text(  # nosec B608: table name validated via allowlist
            f"""
            SELECT strategy_id, instrument_id, ts_event, signal_type, strength,
                   model_predictions, risk_metrics
            FROM {table_name}
            WHERE {' AND '.join(where_parts)}
            ORDER BY ts_event
            """
        )
        return self.deps._execute_read(
            sql,
            params,
            columns=[
                "strategy_id",
                "instrument_id",
                "ts_event",
                "signal_type",
                "strength",
                "model_predictions",
                "risk_metrics",
            ],
        )

    def get_latest(self, *, instrument_id: str, limit: int = 1) -> object:
        table_name = _resolve_table_name(
            self.deps,
            attr="strategy_signals_table",
            base="ml_strategy_signals",
            allowed={"ml_strategy_signals"},
        )
        sql = _text(  # nosec B608: table name validated via allowlist
            f"""
            SELECT strategy_id, ts_event, signal_type, strength, risk_metrics
            FROM {table_name}
            WHERE instrument_id = :instrument_id
            ORDER BY ts_event DESC
            LIMIT :limit
            """
        )
        params: dict[str, object] = {"instrument_id": instrument_id, "limit": int(limit)}
        return self.deps._execute_read(
            sql,
            params,
            columns=[
                "strategy_id",
                "ts_event",
                "signal_type",
                "strength",
                "risk_metrics",
            ],
        )

    def read_active_signals(
        self,
        *,
        hours_back: int = 1,
        limit: int = 100,
        strategy_id: str | None = None,
        instrument_id: str | None = None,
    ) -> object:
        import time as _time

        now_ns: int = int(_time.time() * 1e9)
        start_ns: int = int(now_ns - hours_back * 3600 * 1e9)

        where_parts: list[str] = ["ts_event >= :start_ns"]
        params: dict[str, Any] = {"start_ns": start_ns, "limit": int(limit)}
        if strategy_id is not None:
            where_parts.append("strategy_id = :strategy_id")
            params["strategy_id"] = strategy_id
        if instrument_id is not None:
            where_parts.append("instrument_id = :instrument_id")
            params["instrument_id"] = instrument_id

        table_name = _resolve_table_name(
            self.deps,
            attr="strategy_signals_table",
            base="ml_strategy_signals",
            allowed={"ml_strategy_signals"},
        )
        sql = _text(  # nosec B608: table name validated via allowlist
            f"""
            SELECT strategy_id,
                   instrument_id,
                   signal_type,
                   strength,
                   model_predictions,
                   risk_metrics,
                   execution_params,
                   ts_event,
                   ts_init
            FROM {table_name}
            WHERE {' AND '.join(where_parts)}
            ORDER BY ts_event DESC
            LIMIT :limit
            """
        )
        return self.deps._execute_read(
            sql,
            params,
            columns=[
                "strategy_id",
                "instrument_id",
                "signal_type",
                "strength",
                "model_predictions",
                "risk_metrics",
                "execution_params",
                "ts_event",
                "ts_init",
            ],
        )

    def get_signals(
        self,
        *,
        strategy_id: str,
        start_ns: int,
        end_ns: int,
        instrument_id: str | None = None,
    ) -> Any:
        if instrument_id is not None:
            return self.read_signals(
                strategy_id=strategy_id,
                instrument_id=instrument_id,
                start_ns=start_ns,
                end_ns=end_ns,
            )

        table_name = _resolve_table_name(
            self.deps,
            attr="strategy_signals_table",
            base="ml_strategy_signals",
            allowed={"ml_strategy_signals"},
        )
        sql = _text(  # nosec B608: table name validated via allowlist
            f"""
            SELECT strategy_id, instrument_id, ts_event, signal_type, strength,
                   model_predictions, risk_metrics
            FROM {table_name}
            WHERE strategy_id = :strategy_id
              AND ts_event >= :start_ns
              AND ts_event < :end_ns
            ORDER BY instrument_id, ts_event
            """
        )
        params: dict[str, object] = {
            "strategy_id": strategy_id,
            "start_ns": int(start_ns),
            "end_ns": int(end_ns),
        }
        return self.deps._execute_read(
            sql,
            params,
            columns=[
                "strategy_id",
                "instrument_id",
                "ts_event",
                "signal_type",
                "strength",
                "model_predictions",
                "risk_metrics",
            ],
        )


@dataclass(slots=True)
class StrategySignalStatsService:
    """Aggregate/statistical read operations for strategy signals."""

    deps: StrategyReadDepsStrict

    def get_statistics(self, *, start_ns: int | None = None, end_ns: int | None = None) -> dict[str, object]:
        table_name = _resolve_table_name(
            self.deps,
            attr="strategy_signals_table",
            base="ml_strategy_signals",
            allowed={"ml_strategy_signals"},
        )
        from ml.stores.services.common_stats import select_min_max_ts as _minmax

        minmax = _minmax(field="ts_event", min_alias="min_ts", max_alias="max_ts")
        conditions, params = _time_conditions(start_ns, end_ns, field="ts_event")
        base_sql = (
            "SELECT\n"
            "                COUNT(*) as total_signals,\n"
            "                COUNT(DISTINCT strategy_id) as unique_strategies,\n"
            "                COUNT(DISTINCT instrument_id) as unique_instruments,\n"
            "                SUM(CASE WHEN signal_type = 'BUY' THEN 1 ELSE 0 END) as buy_signals,\n"
            "                SUM(CASE WHEN signal_type = 'SELL' THEN 1 ELSE 0 END) as sell_signals,\n"
            "                SUM(CASE WHEN signal_type = 'HOLD' THEN 1 ELSE 0 END) as hold_signals,\n"
            "                AVG(strength) as avg_strength,\n"
            f"                {minmax}\n"
            f"            FROM {table_name}"
        )
        if conditions:
            base_sql += " WHERE " + " AND ".join(conditions)
        row = self.deps._fetch_one(_text(base_sql), params)
        if row:
            return {
                "total_signals": row[0] or 0,
                "unique_strategies": row[1] or 0,
                "unique_instruments": row[2] or 0,
                "buy_signals": row[3] or 0,
                "sell_signals": row[4] or 0,
                "hold_signals": row[5] or 0,
                "avg_strength": float(cast(SupportsFloat, row[6])) if row[6] is not None else 0.0,
                "min_timestamp_ns": row[7] or 0,
                "max_timestamp_ns": row[8] or 0,
            }
        return {
            "total_signals": 0,
            "unique_strategies": 0,
            "unique_instruments": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "hold_signals": 0,
            "avg_strength": 0.0,
            "min_timestamp_ns": 0,
            "max_timestamp_ns": 0,
        }

    def get_signal_distribution(
        self,
        *,
        strategy_id: str | None = None,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> dict[str, int]:
        table_name = _resolve_table_name(
            self.deps,
            attr="strategy_signals_table",
            base="ml_strategy_signals",
            allowed={"ml_strategy_signals"},
        )
        conditions, params = _time_conditions(start_ns, end_ns, field="ts_event")
        params2: dict[str, object] = dict(params)
        if strategy_id is not None:
            conditions.append("strategy_id = :strategy_id")
            params2["strategy_id"] = strategy_id
        elif not conditions:
            latest_row = self.deps._fetch_one(
                _text(  # nosec B608: table name validated via allowlist
                    f"SELECT strategy_id FROM {table_name} ORDER BY ts_event DESC LIMIT 1"
                ),
                {},
            )
            if latest_row and latest_row[0] is not None:
                conditions.append("strategy_id = :strategy_id")
                params2["strategy_id"] = str(latest_row[0])
        query = f"SELECT signal_type, COUNT(*) as count FROM {table_name}"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " GROUP BY signal_type"
        rows = self.deps._fetch_all(_text(query), params2)
        out: dict[str, int] = {}
        for signal_type, count in rows:
            out[str(signal_type)] = int(cast(SupportsInt, count))
        return out

    def get_strategy_performance(
        self,
        *,
        strategy_id: str,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> dict[str, object]:
        table_name = _resolve_table_name(
            self.deps,
            attr="strategy_signals_table",
            base="ml_strategy_signals",
            allowed={"ml_strategy_signals"},
        )
        from ml.stores.services.common_stats import select_signal_counts as _sel_counts

        conditions, params = _time_conditions(start_ns, end_ns, field="ts_event")
        params2: dict[str, object] = dict(params)
        conditions.insert(0, "strategy_id = :strategy_id")
        params2["strategy_id"] = strategy_id
        counts = _sel_counts(include_avg_strength=True)
        from ml.stores.services.common_stats import select_numeric_stats as _num
        strength_stats = _num(column="strength", prefix="strength", include_avg=False, include_stddev=True, include_min_max=True)
        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        row = self.deps._fetch_one(
            _text(
                f"SELECT\n"
                f"                {counts},\n"
                f"                {strength_stats}\n"
                f"            FROM {table_name}\n"
                f"            WHERE {where_clause}"
            ),
            params2,
        )
        if row:
            return {
                "signal_count": row[0] or 0,
                "buy_count": row[1] or 0,
                "sell_count": row[2] or 0,
                "hold_count": row[3] or 0,
                "avg_strength": float(cast(SupportsFloat, row[4])) if row[4] is not None else 0.0,
                "std_strength": float(cast(SupportsFloat, row[5])) if row[5] is not None else 0.0,
                "min_strength": float(cast(SupportsFloat, row[6])) if row[6] is not None else 0.0,
                "max_strength": float(cast(SupportsFloat, row[7])) if row[7] is not None else 0.0,
            }
        return {
            "signal_count": 0,
            "buy_count": 0,
            "sell_count": 0,
            "hold_count": 0,
            "avg_strength": 0.0,
            "std_strength": 0.0,
            "min_strength": 0.0,
            "max_strength": 0.0,
        }

    def update_performance_metrics(
        self,
        *,
        strategy_id: str,
        period_start: int,
        period_end: int,
        # write via engine
        engine: Any,
        performance_table: Any,
    ) -> None:
        from sqlalchemy.dialects.postgresql import insert as _insert

        with engine.begin() as conn:
            table_name = _resolve_table_name(
                self.deps,
                attr="strategy_signals_table",
                base="ml_strategy_signals",
                allowed={"ml_strategy_signals"},
            )
            from ml.stores.services.common_stats import select_signal_counts as _sel_counts
            counts = _sel_counts(include_avg_strength=True)
            query = _text(  # nosec B608: table name validated via allowlist
                f"""
                SELECT
                    {counts},
                    AVG((risk_metrics->>'risk_score')::float) as avg_risk_score
                FROM {table_name}
                WHERE strategy_id = :strategy_id
                AND ts_event >= :period_start
                AND ts_event < :period_end
                """
            )

            res = conn.execute(
                query,
                {
                    "strategy_id": strategy_id,
                    "period_start": period_start,
                    "period_end": period_end,
                },
            ).fetchone()

            if res and res[0] > 0:
                stmt = _insert(performance_table)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["strategy_id", "period_start"],
                    set_={
                        "period_end": period_end,
                        "signal_count": res[0],
                        "buy_count": res[1],
                        "sell_count": res[2],
                        "hold_count": res[3],
                        "avg_strength": res[4],
                        "avg_risk_score": res[5],
                    },
                )

                conn.execute(
                    stmt,
                    {
                        "strategy_id": strategy_id,
                        "period_start": period_start,
                        "period_end": period_end,
                        "signal_count": res[0],
                        "buy_count": res[1],
                        "sell_count": res[2],
                        "hold_count": res[3],
                        "avg_strength": res[4],
                        "avg_risk_score": res[5],
                    },
                )


@dataclass(slots=True)
class StrategySignalEventService:
    """Event emission service for strategy signals (registry/metrics)."""

    deps: StrategyEventDepsStrict
    logger: LoggerLike

    def emit_signal_events(self, signals: list[StrategySignal]) -> None:
        try:
            registry = self.deps._get_data_registry()
            if registry is None:
                return

            # Ensure dataset 'signals' is registered to satisfy watermark FK constraints
            try:
                registry.get_manifest("signals")
            except Exception:
                # Best-effort: attempt to register the dataset manifest; on failure
                # log a warning and emit a metric, then continue with event emission.
                try:
                    import hashlib as _hashlib

                    from ml.common.metrics_manager import MetricsManager as _MM
                    from ml.registry.dataclasses import DatasetManifest
                    from ml.registry.dataclasses import DatasetType
                    from ml.registry.dataclasses import StorageKind

                    schema = {
                        "instrument_id": "str",
                        "strategy_id": "str",
                        "ts_event": "int64",
                        "ts_init": "int64",
                        "signal_type": "str",
                        "strength": "float64",
                        "model_predictions": "json",
                        "risk_metrics": "json",
                        "execution_params": "json",
                    }
                    schema_hash = _hashlib.sha256(str(schema).encode()).hexdigest()
                    manifest = DatasetManifest(
                        dataset_id="signals",
                        dataset_type=DatasetType.SIGNALS,
                        storage_kind=StorageKind.POSTGRES,
                        location="ml_strategy_signals",
                        partitioning={"by": "ts_event", "interval": "monthly"},
                        retention_days=365,
                        schema=schema,
                        ts_field="ts_event",
                        seq_field=None,
                        primary_keys=["strategy_id", "instrument_id", "ts_event"],
                        schema_hash=schema_hash,
                        constraints={
                            "nullability": {
                                "strategy_id": False,
                                "instrument_id": False,
                                "ts_event": False,
                                "ts_init": False,
                            },
                        },
                        lineage=[],
                        pipeline_signature="strategy_services_auto",
                        version="1.0.0",
                        metadata={
                            "ts_field": "ts_event",
                            "primary_keys": ["strategy_id", "instrument_id", "ts_event"],
                            "auto_registered": True,
                        },
                    )
                    registry.register_dataset(manifest)
                except Exception as exc:
                    # Non-blocking: structured log + metric, then continue.
                    try:
                        self.logger.warning(
                            "Auto-registration of 'signals' dataset failed: %s",
                            exc,
                            extra={
                                "component": "strategy_services",
                                "operation": "register_dataset",
                                "dataset_id": "signals",
                            },
                            exc_info=True,
                        )
                        _mm = _MM.default()
                        _mm.inc(
                            "ml_pipeline_errors_total",
                            "ML pipeline errors",
                            labels={
                                "component": "strategy_services",
                                "op": "register_dataset",
                                "error_type": "exception",
                            },
                            labelnames=("component", "op", "error_type"),
                        )
                    except Exception:
                        # Metrics/logging must never affect control flow
                        try:
                            self.logger.debug(
                                "Auto-registration metrics/logging failed (ignored)",
                                exc_info=True,
                            )
                        except Exception:
                            ...

            from collections import defaultdict

            grouped: dict[tuple[str, str], list[StrategySignal]] = defaultdict(list)
            for sig in signals:
                grouped[(sig.strategy_id, sig.instrument_id)].append(sig)

            for (strategy_id, instrument_id), group in grouped.items():
                if not group:
                    continue

                import time as _time
                import uuid as _uuid

                run_id = f"signal_{strategy_id}_{_uuid.uuid4().hex[:8]}_{int(_time.time())}"
                ts_vals = [s.ts_event for s in group]
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

                if callable(_emit_wm):
                    try:
                        _emit_wm(
                            registry,
                            dataset_id="signals",
                            instrument_id=instrument_id,
                            stage=Stage.SIGNAL_EMITTED,
                            source=src_enum,
                            run_id=run_id,
                            ts_min=ts_min,
                            ts_max=ts_max,
                            count=len(group),
                            status=EventStatus.SUCCESS,
                            dataset_type="signals",
                            component=strategy_id,
                        )
                    except Exception:
                        # Fallback: event-only path if watermark update encounters FK constraints
                        try:
                            registry.emit_event(
                                dataset_id="signals",
                                instrument_id=instrument_id,
                                stage=Stage.SIGNAL_EMITTED,
                                source=src_enum,
                                run_id=run_id,
                                ts_min=ts_min,
                                ts_max=ts_max,
                                count=len(group),
                                status=EventStatus.SUCCESS,
                                metadata={"component": strategy_id},
                            )
                        except Exception:
                            try:
                                self.logger.debug(
                                    "Registry emit_event fallback failed (ignored)",
                                    exc_info=True,
                                )
                            except Exception:
                                ...
                else:
                    # Fallback: direct registry calls (non-blocking with logging)
                    try:
                        registry.emit_event(
                            dataset_id="signals",
                            instrument_id=instrument_id,
                            stage=Stage.SIGNAL_EMITTED,
                            source=src_enum,
                            run_id=run_id,
                            ts_min=ts_min,
                            ts_max=ts_max,
                            count=len(group),
                            status=EventStatus.SUCCESS,
                            metadata={"component": strategy_id},
                        )
                        try:
                            registry.update_watermark(
                                dataset_id="signals",
                                instrument_id=instrument_id,
                                source=src_enum,
                                last_success_ns=ts_max,
                                count=len(group),
                                completeness_pct=100.0,
                            )
                        except Exception:
                            # Watermark update is best-effort
                            try:
                                self.logger.debug(
                                    "Registry watermark update failed (ignored)",
                                    exc_info=True,
                                )
                            except Exception:
                                ...
                    except Exception as exc:
                        self.logger.warning("Registry emit/update failed (non-blocking): %s", exc)

        except Exception:
            # Non-blocking by design
            self.logger.warning("Failed to emit signal events", exc_info=True)


@dataclass(slots=True)
class StrategySignalClearService:
    """Deletion/cleanup operations for strategy signals."""

    deps: StrategyClearDepsStrict

    def clear(self, *, strategy_id: str | None = None, instrument_id: str | None = None) -> None:
        with self.deps.engine.begin() as conn:
            delete_stmt = self.deps.strategy_signals_table.delete()
            if strategy_id is not None:
                delete_stmt = delete_stmt.where(
                    self.deps.strategy_signals_table.c.strategy_id == strategy_id,
                )
            if instrument_id is not None:
                delete_stmt = delete_stmt.where(
                    self.deps.strategy_signals_table.c.instrument_id == instrument_id,
                )
            conn.execute(delete_stmt)
