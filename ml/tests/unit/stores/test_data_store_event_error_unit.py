"""
DataStore event emission error path: ensure write does not raise if registry fails.
"""

from __future__ import annotations

from typing import Any

from ml.stores.data_store import DataStore


class _FlakyRegistry:
    def __init__(self, fail_on: str) -> None:
        self.fail_on = fail_on

    def emit_event(self, **kwargs: Any) -> None:  # noqa: D401
        """Raise on demand"""
        if self.fail_on == "emit":
            raise RuntimeError("emit fail")

    def update_watermark(self, **kwargs: Any) -> None:  # noqa: D401
        """Raise on demand"""
        if self.fail_on == "wm":
            raise RuntimeError("wm fail")


class _NoOpStore:
    def write_features(self, *a: Any, **k: Any) -> None:  # noqa: D401
        return None


def test_write_features_tolerates_registry_emit_error() -> None:
    ds = object.__new__(DataStore)  # type: ignore[misc]
    ds.connection_string = "sqlite:///:memory:"  # type: ignore[attr-defined]
    ds.feature_store = _NoOpStore()  # type: ignore[attr-defined]
    ds.model_store = _NoOpStore()  # type: ignore[attr-defined]
    ds.strategy_store = _NoOpStore()  # type: ignore[attr-defined]
    ds.registry = _FlakyRegistry("emit")  # type: ignore[attr-defined]
    ds._ensure_dataset_registered = lambda **kwargs: None  # type: ignore[attr-defined]
    ds.clock = type("_C", (), {"timestamp_ns": lambda self: 100})()  # type: ignore[attr-defined]

    from ml.stores.base import FeatureData

    fd = FeatureData(feature_set_id="fs", instrument_id="X.SIM", values={"a": 1.0}, _ts_event=1, _ts_init=1)
    # Should not raise even if registry emit_event fails internally
    event = DataStore.write_features(ds, instrument_id="X.SIM", features=[fd])
    assert event.record_count == 1


def test_write_features_tolerates_watermark_error() -> None:
    ds = object.__new__(DataStore)  # type: ignore[misc]
    ds.connection_string = "sqlite:///:memory:"  # type: ignore[attr-defined]
    ds.feature_store = _NoOpStore()  # type: ignore[attr-defined]
    ds.model_store = _NoOpStore()  # type: ignore[attr-defined]
    ds.strategy_store = _NoOpStore()  # type: ignore[attr-defined]
    ds.registry = _FlakyRegistry("wm")  # type: ignore[attr-defined]
    ds._ensure_dataset_registered = lambda **kwargs: None  # type: ignore[attr-defined]
    ds.clock = type("_C", (), {"timestamp_ns": lambda self: 100})()  # type: ignore[attr-defined]

    from ml.stores.base import FeatureData

    fd = FeatureData(feature_set_id="fs", instrument_id="X.SIM", values={"a": 1.0}, _ts_event=1, _ts_init=1)
    event = DataStore.write_features(ds, instrument_id="X.SIM", features=[fd])
    assert event.status == "success"
