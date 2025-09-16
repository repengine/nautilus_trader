from __future__ import annotations

from typing import Any

import pytest

from ml.config.events import Stage
from ml.stores.base import StrategySignal
from ml.stores.services.strategy_services import (
    StrategySignalQueryService,
    StrategySignalWriteService,
)


class _FakeWriteDeps:
    def __init__(self) -> None:
        self.strategy_signals_table = object()
        self.last_call: dict[str, Any] | None = None

    def _execute_upsert_and_publish(self, **kwargs: Any) -> None:
        self.last_call = kwargs


class _FakeReadDeps:
    def __init__(self) -> None:
        self.safe_table_calls: list[tuple[str, set[str]]] = []
        self.last_execute: dict[str, Any] | None = None

    def _safe_table(self, base: str, allowed: set[str]) -> str:  # noqa: D401
        self.safe_table_calls.append((base, allowed))
        return base

    def _execute_read(
        self,
        sql: Any,
        params: dict[str, Any],
        *,
        columns: list[str],
    ) -> Any:  # noqa: D401
        self.last_execute = {"sql": str(sql), "params": params, "columns": columns}

        # Return a sentinel object that mimics a DataFrame in tests as needed
        class _DF:  # minimal duck-typed stand-in
            pass

        return _DF()

    def _fetch_one(
        self,
        sql: Any,
        params: dict[str, Any],
    ) -> tuple[Any, ...] | None:  # pragma: no cover
        return None

    def _fetch_all(
        self,
        sql: Any,
        params: dict[str, Any],
    ) -> list[tuple[Any, ...]]:  # pragma: no cover
        return []


def test_strategy_write_service_calls_upsert_and_publish() -> None:
    deps = _FakeWriteDeps()
    svc = StrategySignalWriteService(deps, logger=object())
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
