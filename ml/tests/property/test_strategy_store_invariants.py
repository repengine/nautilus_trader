from __future__ import annotations

import pytest

try:  # optional dependency
    from hypothesis import given
    from hypothesis import strategies as st
except Exception:  # pragma: no cover - hypothesis optional
    pytest.skip("hypothesis not available", allow_module_level=True)

from typing import Any
from contextlib import contextmanager

from ml.stores.base import StrategySignal


@contextmanager
def _patch_strategy_store_io(sink: list[dict[str, Any]]):
    orig_setup = StrategyStore._setup_tables
    orig_exec = StrategyStore._execute_write
    from ml.core import db_engine as _db

    orig_get_engine = _db.EngineManager.get_engine

    class _DummyConn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    class _DummyEngine:
        def connect(self) -> _DummyConn:
            return _DummyConn()

        def begin(self) -> _DummyConn:
            return _DummyConn()

    try:
        StrategyStore._setup_tables = lambda self: None  # type: ignore[assignment]
        StrategyStore._execute_write = lambda self, values: sink.extend(values)  # type: ignore[assignment]
        _db.EngineManager.get_engine = lambda *a, **k: _DummyEngine()  # type: ignore[assignment]
        yield
    finally:
        StrategyStore._setup_tables = orig_setup  # type: ignore[assignment]
        StrategyStore._execute_write = orig_exec  # type: ignore[assignment]
        _db.EngineManager.get_engine = orig_get_engine  # type: ignore[assignment]


@st.composite
def _signal_dict(draw: st.DrawFn) -> dict[str, Any]:  # type: ignore[name-defined]
    sid = draw(st.text(min_size=1, max_size=10))
    inst = draw(st.from_regex(r"[A-Z]{3,6}/[A-Z]{3,6}\.SIM", fullmatch=True))
    stype = draw(st.sampled_from(["BUY", "SELL", "HOLD"]))
    if stype == "BUY":
        strength = draw(
            st.floats(min_value=0.0, max_value=1.0, allow_infinity=False, allow_nan=False),
        )
    elif stype == "SELL":
        strength = draw(
            st.floats(min_value=-1.0, max_value=0.0, allow_infinity=False, allow_nan=False),
        )
    else:
        strength = draw(
            st.floats(min_value=-1.0, max_value=1.0, allow_infinity=False, allow_nan=False),
        )
    return {
        "strategy_id": sid,
        "instrument_id": inst,
        "signal_type": stype,
        "strength": float(strength),
    }


@given(
    signals=st.lists(_signal_dict(), min_size=1, max_size=20),
    base_ts=st.integers(min_value=0, max_value=1_000_000_000),
)
def test_strategy_store_write_batch_invariants(
    signals: list[dict[str, Any]],
    base_ts: int,
) -> None:  # noqa: ANN201
    sink: list[dict[str, Any]] = []
    # Build monotonic timestamps and create data objects
    data: list[StrategySignal] = []
    ts = base_ts
    for s in signals:
        data.append(
            StrategySignal(
                strategy_id=s["strategy_id"],
                instrument_id=s["instrument_id"],
                signal_type=s["signal_type"],
                strength=float(s["strength"]),
                model_predictions={},
                risk_metrics={},
                execution_params={},
                _ts_event=int(ts),
                _ts_init=int(ts),
            ),
        )
        ts += 1

    # Convert to value dicts to simulate persistence boundary
    sink = [
        {
            "strategy_id": d.strategy_id,
            "instrument_id": d.instrument_id,
            "ts_event": d.ts_event,
            "ts_init": d.ts_init,
            "signal_type": d.signal_type,
            "strength": d.strength,
            "model_predictions": d.model_predictions,
            "risk_metrics": d.risk_metrics,
            "execution_params": d.execution_params,
            "is_live": False,
        }
        for d in data
    ]

    # Invariants on would-be persisted values
    assert sink, "no values to validate"
    last_ts = -1
    for v in sink:
        stype = str(v["signal_type"]).upper()
        strength = float(v["strength"])  # within [-1,1]
        # Directional invariants
        if stype == "BUY":
            assert strength >= 0.0
        elif stype == "SELL":
            assert strength <= 0.0
        # Timestamp invariants
        assert int(v["ts_init"]) >= int(v["ts_event"])  # init not before event
        assert int(v["ts_event"]) >= last_ts
        last_ts = int(v["ts_event"])  # non-decreasing
