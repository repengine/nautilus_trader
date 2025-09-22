from __future__ import annotations

from typing import Any

from ml.common.event_emitter import (
    _ensure_correlation_id,
    emit_dataset_event,
    emit_dataset_event_and_watermark,
)
from typing import cast

from ml.config.events import EventStatus, Source, Stage
from ml.registry.protocols import RegistryProtocol
from pytest import MonkeyPatch


class _DummyRegistry:
    def __init__(self) -> None:
        self.last_emit: dict[str, Any] | None = None
        self.last_watermark: dict[str, Any] | None = None

    def emit_event(
        self,
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
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.last_emit = {
            "dataset_id": dataset_id,
            "instrument_id": instrument_id,
            "stage": stage,
            "source": source,
            "run_id": run_id,
            "ts_min": ts_min,
            "ts_max": ts_max,
            "count": count,
            "status": status,
            "error": error,
            "metadata": metadata,
        }

    def update_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: Source,
        last_success_ns: int,
        count: int,
        completeness_pct: float,
    ) -> None:
        self.last_watermark = {
            "dataset_id": dataset_id,
            "instrument_id": instrument_id,
            "source": source,
            "last_success_ns": last_success_ns,
            "count": count,
            "completeness_pct": completeness_pct,
        }


class _LegacyRegistry(_DummyRegistry):
    # Legacy emit_event without metadata kwarg; emulate TypeError on unexpected kwargs
    def emit_event(
        self,
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
        metadata: dict[str, object] | None = None,
    ) -> None:
        if metadata is not None:
            raise TypeError("unexpected keyword 'metadata'")
        super().emit_event(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            stage=stage,
            source=source,
            run_id=run_id,
            ts_min=ts_min,
            ts_max=ts_max,
            count=count,
            status=status,
            error=error,
            metadata=None,
        )


def test_emit_dataset_event_and_watermark_attaches_correlation_and_updates_watermark() -> None:
    reg = _DummyRegistry()
    emit_dataset_event_and_watermark(
        cast(RegistryProtocol, reg),
        dataset_id="features",
        instrument_id="EUR/USD",
        stage=Stage.FEATURE_COMPUTED,
        source=Source.LIVE,
        run_id="rid",
        ts_min=1,
        ts_max=2,
        count=3,
        status=EventStatus.SUCCESS,
    )
    assert reg.last_emit is not None
    assert reg.last_watermark is not None
    assert "metadata" in reg.last_emit
    metadata = cast(dict[str, object] | None, reg.last_emit["metadata"])
    assert metadata is not None and "correlation_id" in metadata
    assert reg.last_watermark["last_success_ns"] == 2


def test_emit_dataset_event_legacy_registry_without_metadata_kwarg() -> None:
    reg = _LegacyRegistry()
    emit_dataset_event(
        cast(RegistryProtocol, reg),
        dataset_id="predictions",
        instrument_id="EUR/USD",
        stage=Stage.MODEL_INFERRED,
        source=Source.BATCH,
        run_id="rid2",
        ts_min=5,
        ts_max=6,
        count=1,
        status=EventStatus.SUCCESS,
        error=None,
    )
    assert reg.last_emit is not None
    # legacy emit should still record and not include metadata
    assert "metadata" not in reg.last_emit


def test__ensure_correlation_id_preserves_existing_and_injects_trace_context(
    monkeypatch: MonkeyPatch,
) -> None:
    del monkeypatch
    # install tracing stub that injects a trace_context
    import sys
    from types import ModuleType

    mod = ModuleType("ml.observability.tracing")

    def inject_trace_context(meta: dict[str, object]) -> dict[str, object]:
        m = dict(meta)
        m["trace_context"] = {"traceparent": "00-abc-01"}
        return m

    setattr(mod, "inject_trace_context", inject_trace_context)
    sys.modules["ml.observability.tracing"] = mod

    meta = _ensure_correlation_id(
        metadata={"correlation_id": "keep_me"},
        run_id="rid",
        dataset_id="features",
        instrument_id="X",
        ts_min=1,
        ts_max=2,
        count=1,
    )
    # correlation_id preserved
    assert meta["correlation_id"] == "keep_me"
    # trace_context injected by stub
    assert "trace_context" in meta
