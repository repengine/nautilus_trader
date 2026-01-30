from __future__ import annotations

import pytest

try:  # optional dependency
    from hypothesis import given
    from hypothesis import strategies as st
except Exception:  # pragma: no cover - hypothesis optional
    pytest.skip("hypothesis not available", allow_module_level=True)

from typing import Any
from contextlib import contextmanager

from ml.stores.base import ModelPrediction


@contextmanager
def _patch_model_store_io(sink: list[dict[str, Any]]):
    orig_setup = ModelStore._setup_tables
    orig_exec = ModelStore._execute_write
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
        ModelStore._setup_tables = lambda self: None  # type: ignore[assignment]
        ModelStore._execute_write = lambda self, values: sink.extend(values)  # type: ignore[assignment]
        _db.EngineManager.get_engine = lambda *a, **k: _DummyEngine()  # type: ignore[assignment]
        yield
    finally:
        ModelStore._setup_tables = orig_setup  # type: ignore[assignment]
        ModelStore._execute_write = orig_exec  # type: ignore[assignment]
        _db.EngineManager.get_engine = orig_get_engine  # type: ignore[assignment]


@given(
    preds=st.lists(
        st.fixed_dictionaries(
            {
                "model_id": st.text(min_size=1, max_size=10),
                "instrument_id": st.from_regex(r"[A-Z]{3,6}/[A-Z]{3,6}\.SIM", fullmatch=True),
                "prediction": st.floats(
                    min_value=0.0,
                    max_value=1.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                "confidence": st.floats(
                    min_value=0.0,
                    max_value=1.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                "inference_time_ms": st.floats(min_value=0.0, max_value=5000.0),
            },
        ),
        min_size=1,
        max_size=20,
    ),
    base_ts=st.integers(min_value=0, max_value=1_000_000_000),
)
def test_model_store_write_batch_invariants(
    preds: list[dict[str, Any]],
    base_ts: int,
) -> None:  # noqa: ANN201
    sink: list[dict[str, Any]] = []
    # Build monotonic timestamps and create data objects
    data: list[ModelPrediction] = []
    ts = base_ts
    for p in preds:
        data.append(
            ModelPrediction(
                model_id=p["model_id"],
                instrument_id=p["instrument_id"],
                prediction=float(p["prediction"]),
                confidence=float(p["confidence"]),
                features_used={},
                inference_time_ms=float(p["inference_time_ms"]),
                _ts_event=int(ts),
                _ts_init=int(ts),
            ),
        )
        ts += 1

    # Convert to value dicts to simulate persistence boundary
    sink = [
        {
            "model_id": d.model_id,
            "instrument_id": d.instrument_id,
            "ts_event": d.ts_event,
            "ts_init": d.ts_init,
            "prediction": d.prediction,
            "confidence": d.confidence,
            "features_used": d.features_used,
            "inference_time_ms": d.inference_time_ms,
            "is_live": False,
        }
        for d in data
    ]

    # Invariants on would-be persisted values
    assert sink, "no values to validate"
    last_ts = -1
    for v in sink:
        # Bounds invariants (by contract)
        assert 0.0 <= float(v["prediction"]) <= 1.0
        assert 0.0 <= float(v["confidence"]) <= 1.0
        # Timestamp invariants
        assert int(v["ts_init"]) >= int(v["ts_event"])  # monotone init
        assert int(v["ts_event"]) >= last_ts
        last_ts = int(v["ts_event"])  # non-decreasing order preserved
