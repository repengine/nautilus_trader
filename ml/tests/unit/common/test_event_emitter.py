from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, cast

from ml.common.event_emitter import (
    _ensure_correlation_id,
    emit_dataset_event,
    emit_dataset_event_and_watermark,
)

from ml.config.events import EventStatus, Source, Stage
from ml.registry.protocols import RegistryProtocol
from pytest import LogCaptureFixture, MonkeyPatch


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


class _CounterStub:
    def __init__(self) -> None:
        self.labels_called: dict[str, str] | None = None
        self.inc_called: bool = False

    def labels(self, **labels: str) -> _CounterStub:
        self.labels_called = labels
        return self

    def inc(self) -> None:
        self.inc_called = True


def _install_tracing_stub(monkeypatch: MonkeyPatch) -> None:
    """
    Install a minimal tracing module that injects a deterministic trace-context.
    """

    mod = ModuleType("ml.observability.tracing")

    def inject_trace_context(meta: dict[str, object]) -> dict[str, object]:
        patched = dict(meta)
        patched["trace_context"] = {"traceparent": "00-abc-01"}
        return patched

    monkeypatch.setattr(mod, "inject_trace_context", inject_trace_context, raising=False)
    monkeypatch.setitem(sys.modules, "ml.observability.tracing", mod)


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


def test_emit_dataset_event_and_watermark_logs_metadata_fallback(
    caplog: LogCaptureFixture,
    monkeypatch: MonkeyPatch,
) -> None:
    caplog.set_level("WARNING", logger="ml.common.event_emitter")
    counter_stub = _CounterStub()
    monkeypatch.setattr(
        "ml.common.event_emitter._METADATA_FALLBACK_COUNTER",
        counter_stub,
        raising=True,
    )
    reg = _LegacyRegistry()

    emit_dataset_event_and_watermark(
        cast(RegistryProtocol, reg),
        dataset_id="features",
        instrument_id="EUR/USD",
        stage=Stage.FEATURE_COMPUTED,
        source=Source.LIVE,
        run_id="rid-metadata",
        ts_min=1,
        ts_max=2,
        count=3,
        status=EventStatus.SUCCESS,
        dataset_type="features",
        component="writer",
    )

    assert reg.last_emit is not None
    assert reg.last_emit.get("metadata") is None
    assert any(
        "Registry rejected dataset event metadata" in record.message
        for record in caplog.records
    )
    assert counter_stub.labels_called == {
        "dataset_type": "features",
        "component": "writer",
        "stage": Stage.FEATURE_COMPUTED.value,
        "source": Source.LIVE.value,
    }
    assert counter_stub.inc_called


def test_emit_dataset_event_logs_metadata_fallback(
    caplog: LogCaptureFixture,
    monkeypatch: MonkeyPatch,
) -> None:
    caplog.set_level("WARNING", logger="ml.common.event_emitter")
    counter_stub = _CounterStub()
    monkeypatch.setattr(
        "ml.common.event_emitter._METADATA_FALLBACK_COUNTER",
        counter_stub,
        raising=True,
    )
    reg = _LegacyRegistry()

    emit_dataset_event(
        cast(RegistryProtocol, reg),
        dataset_id="predictions",
        instrument_id="EUR/USD",
        stage=Stage.MODEL_INFERRED,
        source=Source.BATCH,
        run_id="rid3",
        ts_min=5,
        ts_max=6,
        count=1,
        status=EventStatus.SUCCESS,
        error=None,
        dataset_type="predictions",
        component="writer",
    )

    assert reg.last_emit is not None
    assert "metadata" not in reg.last_emit
    assert any(
        "Registry rejected dataset event metadata" in record.message
        for record in caplog.records
    )
    assert counter_stub.labels_called == {
        "dataset_type": "predictions",
        "component": "writer",
        "stage": Stage.MODEL_INFERRED.value,
        "source": Source.BATCH.value,
    }
    assert counter_stub.inc_called


def test__ensure_correlation_id_preserves_existing_and_injects_trace_context(
    monkeypatch: MonkeyPatch,
) -> None:
    _install_tracing_stub(monkeypatch)

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
