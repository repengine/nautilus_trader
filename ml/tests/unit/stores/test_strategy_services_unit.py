from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from ml.config.events import Stage
from ml.stores.base import StrategyOrderEvent
from ml.stores.base import StrategySignal
from ml.stores.services.strategy_services import (
    StrategyOrderEventWriteService,
    StrategySignalQueryService,
    StrategySignalWriteService,
)
from ml.stores.protocols import LoggerLike, StrategyReadDepsStrict, StrategyWriteDepsStrict, TableLike


class _TableStub(TableLike):
    def __init__(self) -> None:
        self.c = object()

    def delete(self) -> None:  # pragma: no cover - unused in tests
        return None


class _FakeLogger(LoggerLike):  # pragma: no cover - trivial
    def __init__(self) -> None:
        self.messages: list[tuple[str, object]] = []

    def debug(self, msg: object, *args: object, **kwargs: Any) -> None:
        self.messages.append(("debug", msg))

    def info(self, msg: object, *args: object, **kwargs: Any) -> None:
        self.messages.append(("info", msg))

    def warning(self, msg: object, *args: object, **kwargs: Any) -> None:
        self.messages.append(("warning", msg))

    def error(self, msg: object, *args: object, **kwargs: Any) -> None:
        self.messages.append(("error", msg))


class _FakeWriteDeps(StrategyWriteDepsStrict):
    def __init__(self) -> None:
        self.strategy_signals_table: TableLike = _TableStub()
        self.strategy_order_events_table: TableLike = _TableStub()
        self.last_call: dict[str, Any] | None = None

    def _execute_upsert_and_publish(
        self,
        *,
        values: list[dict[str, object]],
        ts_event_field: str,
        ts_init_field: str,
        context: str,
        key_fields: tuple[str, str, str],
        table: TableLike,
        conflict_cols: Sequence[str],
        update_cols: Sequence[str],
        dataset_id: str,
        stage: object,
        instrument_key: str,
        ts_field: str,
        run_id_batch: str,
        run_id_row: str,
        source: str,
        logger: LoggerLike,
        publish_bus: bool = True,
    ) -> None:
        self.last_call = {
            "values": values,
            "ts_event_field": ts_event_field,
            "ts_init_field": ts_init_field,
            "context": context,
            "key_fields": key_fields,
            "table": table,
            "conflict_cols": list(conflict_cols),
            "update_cols": list(update_cols),
            "dataset_id": dataset_id,
            "stage": stage,
            "instrument_key": instrument_key,
            "ts_field": ts_field,
            "run_id_batch": run_id_batch,
            "run_id_row": run_id_row,
            "source": source,
            "publish_bus": publish_bus,
        }
        logger.info("upsert", extra={"count": len(values)})


class _FakeReadDeps(StrategyReadDepsStrict):
    def __init__(self) -> None:
        self.safe_table_calls: list[tuple[str, set[str]]] = []
        self.last_execute: dict[str, Any] | None = None

    def _safe_table(self, base: str, allowed: set[str]) -> str:  # noqa: D401
        self.safe_table_calls.append((base, allowed))
        return base

    def _execute_read(
        self,
        sql: Any,
        params: Mapping[str, object],
        *,
        columns: Sequence[str],
    ) -> Any:  # noqa: D401
        self.last_execute = {
            "sql": str(sql),
            "params": dict(params),
            "columns": list(columns),
        }

        # Return a sentinel object that mimics a DataFrame in tests as needed
        class _DF:  # minimal duck-typed stand-in
            pass

        return _DF()

    def _fetch_one(
        self,
        sql: Any,
        params: Mapping[str, object],
    ) -> tuple[Any, ...] | None:  # pragma: no cover
        return None

    def _fetch_all(
        self,
        sql: Any,
        params: Mapping[str, object],
    ) -> list[tuple[Any, ...]]:  # pragma: no cover
        return []


def test_strategy_write_service_calls_upsert_and_publish() -> None:
    deps = _FakeWriteDeps()
    svc = StrategySignalWriteService(deps, logger=_FakeLogger())
    sig = StrategySignal(
        strategy_id="s",
        instrument_id="i",
        signal_type="BUY",
        strength=1.0,
        model_predictions={"m": 0.9},
        risk_metrics={"r": 0.1},
        execution_params={"e": 1},
        _ts_event=1,
        _ts_init=2,
    )
    svc.write_batch([sig])

    assert deps.last_call is not None
    assert deps.last_call["dataset_id"] == "signals"
    assert deps.last_call["stage"] == Stage.SIGNAL_EMITTED
    assert deps.last_call["key_fields"] == ("strategy_id", "instrument_id", "ts_event")


def test_strategy_order_event_write_service_calls_upsert_and_publish() -> None:
    deps = _FakeWriteDeps()
    svc = StrategyOrderEventWriteService(deps, logger=_FakeLogger())
    evt = StrategyOrderEvent(
        event_id="evt-1",
        strategy_id="s",
        instrument_id="i",
        client_order_id="c1",
        venue_order_id=None,
        event_type="OrderSubmitted",
        payload={"type": "OrderSubmitted"},
        _ts_event=10,
        _ts_init=11,
        is_live=True,
    )
    svc.write_batch([evt])

    assert deps.last_call is not None
    assert deps.last_call["dataset_id"] == "order_events"
    assert deps.last_call["stage"] == Stage.ORDER_EVENT_EMITTED
    assert deps.last_call["key_fields"] == ("event_id", "strategy_id", "ts_event")
    assert deps.last_call["publish_bus"] is True


def test_strategy_query_service_uses_safe_table_allowlist() -> None:
    deps = _FakeReadDeps()
    svc = StrategySignalQueryService(deps)
    _ = svc.read_signals(
        strategy_id="s",
        instrument_id="i",
        start_ns=0,
        end_ns=10,
    )
    assert deps.safe_table_calls, "_safe_table should be called"
    base, allowed = deps.safe_table_calls[0]
    assert base == "ml_strategy_signals"
    assert "ml_strategy_signals" in allowed
