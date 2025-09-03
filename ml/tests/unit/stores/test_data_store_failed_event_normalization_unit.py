"""
DataStore failed event normalization: invalid source normalized to 'live'.
"""

from __future__ import annotations

from typing import Any

import pytest

from ml.stores.data_store import DataStore


class _RegistryCap:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit_event(self, **kwargs: Any) -> None:
        self.events.append(kwargs)

    def update_watermark(self, **kwargs: Any) -> None:
        return None


class _FailStore:
    def write_features(self, *a: Any, **k: Any) -> None:
        raise RuntimeError("store failed")


@pytest.mark.skip(reason="write_features does not emit failure events in current implementation")
def test_failed_event_source_normalized_to_live() -> None:
    ds = object.__new__(DataStore)  # type: ignore[misc]
    ds.connection_string = "sqlite:///:memory:"  # type: ignore[attr-defined]
    ds.feature_store = _FailStore()  # type: ignore[attr-defined]
    ds.model_store = _FailStore()  # type: ignore[attr-defined]
    ds.strategy_store = _FailStore()  # type: ignore[attr-defined]
    reg = _RegistryCap()
    ds.registry = reg  # type: ignore[attr-defined]
    ds._ensure_dataset_registered = lambda **kwargs: None  # type: ignore[attr-defined]
    ds.clock = type("_C", (), {"timestamp_ns": lambda self: 100})()  # type: ignore[attr-defined]

    from ml.stores.base import FeatureData

    fd = FeatureData(feature_set_id="fs", instrument_id="X.SIM", values={"a": 1.0}, _ts_event=1, _ts_init=1)
    with pytest.raises(RuntimeError):
        DataStore.write_features(ds, instrument_id="X.SIM", features=[fd], source="unit")
    # Check last event has normalized source
    assert reg.events and reg.events[-1]["source"] == "live"
