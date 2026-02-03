from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Iterable, Iterator

import pytest

try:  # optional dependency
    from hypothesis import given
    from hypothesis import strategies as st
except Exception:  # pragma: no cover - hypothesis optional
    pytest.skip("hypothesis not available", allow_module_level=True)

from ml.stores.base import ModelPrediction, StrategySignal
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategyStore


@contextmanager
def _capture_model_store_writes(
    sink: list[dict[str, Any]],
    patch_engine_manager,
) -> Iterator[None]:
    orig_setup = ModelStore._setup_tables
    orig_exec = ModelStore._execute_write
    try:
        setattr(ModelStore, "_setup_tables", lambda self: None)
        setattr(ModelStore, "_execute_write", lambda self, values: sink.extend(values))
        with patch_engine_manager():
            yield
    finally:
        setattr(ModelStore, "_setup_tables", orig_setup)
        setattr(ModelStore, "_execute_write", orig_exec)


@contextmanager
def _capture_strategy_store_writes(
    sink: list[dict[str, Any]],
    patch_engine_manager,
) -> Iterator[None]:
    orig_setup = StrategyStore._setup_tables
    orig_exec = StrategyStore._execute_write
    try:
        setattr(StrategyStore, "_setup_tables", lambda self: None)
        setattr(StrategyStore, "_execute_write", lambda self, values: sink.extend(values))
        with patch_engine_manager():
            yield
    finally:
        setattr(StrategyStore, "_setup_tables", orig_setup)
        setattr(StrategyStore, "_execute_write", orig_exec)


def _ts_range(values: Iterable[dict[str, Any]]) -> tuple[int, int]:
    ts_list = [int(v["ts_event"]) for v in values]
    return (min(ts_list), max(ts_list))


@given(
    n=st.integers(min_value=1, max_value=20),
    base_ts=st.integers(min_value=0, max_value=1_000_000),
    shift=st.integers(min_value=1, max_value=1_000_000),
)
def test_model_store_time_shift_metamorphic(
    n: int,
    base_ts: int,
    shift: int,
    patch_engine_manager,
) -> None:
    sink: list[dict[str, Any]] = []
    with _capture_model_store_writes(sink, patch_engine_manager):
        store = ModelStore(connection_string="sqlite:///:memory:")

        data = [
            ModelPrediction(
                model_id=f"m{i%3}",
                instrument_id="EUR/USD.SIM",
                prediction=math.tanh(i / 10.0),
                confidence=max(0.0, min(1.0, 0.5 + (i % 5) * 0.1)),
                features_used={},
                inference_time_ms=1.0,
                _ts_event=int(base_ts + i),
                _ts_init=int(base_ts + i),
            )
            for i in range(n)
        ]
        store.write_batch(data)
        orig = list(sink)
        sink.clear()

        shifted = [
            ModelPrediction(
                model_id=d.model_id,
                instrument_id=d.instrument_id,
                prediction=d.prediction,
                confidence=d.confidence,
                features_used=d.features_used,
                inference_time_ms=d.inference_time_ms,
                _ts_event=d.ts_event + shift,
                _ts_init=d.ts_init + shift,
            )
            for d in data
        ]
        store.write_batch(shifted)
        sh = list(sink)

    assert len(orig) == len(sh)
    o_min, o_max = _ts_range(orig)
    s_min, s_max = _ts_range(sh)
    assert (s_min - o_min) == shift
    assert (s_max - o_max) == shift

    # Non-timestamp fields unchanged
    def sig(vals: list[dict[str, Any]]) -> set[tuple[str, str, float, float]]:
        return {
            (v["model_id"], v["instrument_id"], float(v["prediction"]), float(v["confidence"]))
            for v in vals
        }

    assert sig(orig) == sig(sh)


@given(
    n=st.integers(min_value=2, max_value=20),
    base_ts=st.integers(min_value=0, max_value=1_000_000),
)
def test_model_store_permutation_invariance(
    n: int,
    base_ts: int,
    patch_engine_manager,
) -> None:
    sink: list[dict[str, Any]] = []
    with _capture_model_store_writes(sink, patch_engine_manager):
        store = ModelStore(connection_string="sqlite:///:memory:")

        data = [
            ModelPrediction(
                model_id=f"m{i%2}",
                instrument_id="EUR/USD.SIM",
                prediction=math.tanh(i / 5.0),
                confidence=0.5,
                features_used={},
                inference_time_ms=1.0,
                _ts_event=int(base_ts + i),
                _ts_init=int(base_ts + i),
            )
            for i in range(n)
        ]
        store.write_batch(data)
        orig = list(sink)
        sink.clear()

        # Reverse order permutation
        store.write_batch(list(reversed(data)))
        perm = list(sink)

    # Sets of keys must match regardless of order
    def keys(vals: list[dict[str, Any]]) -> set[tuple[str, str, int]]:
        return {(v["model_id"], v["instrument_id"], int(v["ts_event"])) for v in vals}

    assert keys(orig) == keys(perm)


@given(
    n=st.integers(min_value=1, max_value=20),
    base_ts=st.integers(min_value=0, max_value=1_000_000),
    shift=st.integers(min_value=1, max_value=1_000_000),
)
def test_strategy_store_time_shift_metamorphic(
    n: int,
    base_ts: int,
    shift: int,
    patch_engine_manager,
) -> None:
    sink: list[dict[str, Any]] = []
    with _capture_strategy_store_writes(sink, patch_engine_manager):
        store = StrategyStore(connection_string="sqlite:///:memory:")

        data = [
            StrategySignal(
                strategy_id=f"s{i%3}",
                instrument_id="EUR/USD.SIM",
                signal_type=["BUY", "SELL", "HOLD"][i % 3],
                strength=((i % 2) * 2 - 1) * 0.5 if (i % 3) != 2 else 0.0,
                model_predictions={},
                risk_metrics={},
                execution_params={},
                decision_metadata={"version": "v1"},
                _ts_event=int(base_ts + i),
                _ts_init=int(base_ts + i),
            )
            for i in range(n)
        ]
        store.write_batch(data)
        orig = list(sink)
        sink.clear()

        shifted = [
            StrategySignal(
                strategy_id=d.strategy_id,
                instrument_id=d.instrument_id,
                signal_type=d.signal_type,
                strength=d.strength,
                model_predictions=d.model_predictions,
                risk_metrics=d.risk_metrics,
                execution_params=d.execution_params,
                decision_metadata=d.decision_metadata,
                _ts_event=d.ts_event + shift,
                _ts_init=d.ts_init + shift,
            )
            for d in data
        ]
        store.write_batch(shifted)
        sh = list(sink)

    assert len(orig) == len(sh)
    o_min, o_max = _ts_range(orig)
    s_min, s_max = _ts_range(sh)
    assert (s_min - o_min) == shift
    assert (s_max - o_max) == shift


@given(
    n=st.integers(min_value=2, max_value=20),
    base_ts=st.integers(min_value=0, max_value=1_000_000),
)
def test_strategy_store_permutation_invariance(
    n: int,
    base_ts: int,
    patch_engine_manager,
) -> None:
    sink: list[dict[str, Any]] = []
    with _capture_strategy_store_writes(sink, patch_engine_manager):
        store = StrategyStore(connection_string="sqlite:///:memory:")

        data = [
            StrategySignal(
                strategy_id=f"s{i%2}",
                instrument_id="EUR/USD.SIM",
                signal_type=["BUY", "SELL", "HOLD"][i % 3],
                strength=(1.0 if (i % 3) == 0 else -1.0) if (i % 3) != 2 else 0.0,
                model_predictions={},
                risk_metrics={},
                execution_params={},
                decision_metadata={"version": "v1"},
                _ts_event=int(base_ts + i),
                _ts_init=int(base_ts + i),
            )
            for i in range(n)
        ]
        store.write_batch(data)
        orig = list(sink)
        sink.clear()

        store.write_batch(list(reversed(data)))
        perm = list(sink)

    def keys(vals: list[dict[str, Any]]) -> set[tuple[str, str, int]]:
        return {(v["strategy_id"], v["instrument_id"], int(v["ts_event"])) for v in vals}

    assert keys(orig) == keys(perm)


@given(
    n=st.integers(min_value=2, max_value=30),
    base_ts=st.integers(min_value=0, max_value=1_000_000),
)
def test_model_store_duplicate_batch_unique_key_invariance(
    n: int,
    base_ts: int,
    patch_engine_manager,
) -> None:
    """
    Duplicating rows in the batch does not change the set of unique keys.

    This validates the upsert contract at a property level without requiring DB IO.

    """
    sink: list[dict[str, Any]] = []
    with _capture_model_store_writes(sink, patch_engine_manager):
        store = ModelStore(connection_string="sqlite:///:memory:")

        base: list[ModelPrediction] = [
            ModelPrediction(
                model_id=f"m{i%3}",
                instrument_id="EUR/USD.SIM",
                prediction=0.1 * (i % 5) - 0.2,
                confidence=0.5,
                features_used={},
                inference_time_ms=1.0,
                _ts_event=int(base_ts + i),
                _ts_init=int(base_ts + i),
            )
            for i in range(n)
        ]
        dup_batch = base + base  # duplicate rows
        store.write_batch(dup_batch)
        out = list(sink)

    def keys(vals: list[dict[str, Any]]) -> set[tuple[str, str, int]]:
        return {(v["model_id"], v["instrument_id"], int(v["ts_event"])) for v in vals}

    in_keys = {(d.model_id, d.instrument_id, d.ts_event) for d in base}
    assert keys(out) == in_keys


@given(
    n=st.integers(min_value=2, max_value=30),
    base_ts=st.integers(min_value=0, max_value=1_000_000),
)
def test_strategy_store_duplicate_batch_unique_key_invariance(
    n: int,
    base_ts: int,
    patch_engine_manager,
) -> None:
    """
    Duplicating rows in the batch does not change the set of unique keys.
    """
    sink: list[dict[str, Any]] = []
    with _capture_strategy_store_writes(sink, patch_engine_manager):
        store = StrategyStore(connection_string="sqlite:///:memory:")

        base: list[StrategySignal] = [
            StrategySignal(
                strategy_id=f"s{i%3}",
                instrument_id="EUR/USD.SIM",
                signal_type=["BUY", "SELL", "HOLD"][i % 3],
                strength=(0.5 if (i % 3) == 0 else -0.5) if (i % 3) != 2 else 0.0,
                model_predictions={},
                risk_metrics={},
                execution_params={},
                decision_metadata={"version": "v1"},
                _ts_event=int(base_ts + i),
                _ts_init=int(base_ts + i),
            )
            for i in range(n)
        ]
        dup_batch = base + base
        store.write_batch(dup_batch)
        out = list(sink)

    def keys(vals: list[dict[str, Any]]) -> set[tuple[str, str, int]]:
        return {(v["strategy_id"], v["instrument_id"], int(v["ts_event"])) for v in vals}

    in_keys = {(d.strategy_id, d.instrument_id, d.ts_event) for d in base}
    assert keys(out) == in_keys
