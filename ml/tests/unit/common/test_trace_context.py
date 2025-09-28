from __future__ import annotations

import importlib
import sys
from types import ModuleType

from pytest import MonkeyPatch

from ml.common.correlation import make_correlation_id
from ml.common.trace_context import (
    extract_and_link_from_event,
    get_correlation_and_trace_context,
)


def _install_tracing_stub(monkeypatch: MonkeyPatch, record: dict[str, object]) -> None:
    """
    Install a stub tracing module and register cleanup with pytest's monkeypatch.
    """

    mod = ModuleType("ml.observability.tracing")

    def extract_and_link_trace_context(event_metadata: dict[str, object]) -> None:
        record["extract_called_with"] = event_metadata

    def inject_trace_context(metadata: dict[str, object]) -> dict[str, object]:
        metadata = dict(metadata)
        metadata["trace_context"] = {"traceparent": "00-abc-01"}
        record["inject_called_with"] = metadata
        return metadata

    monkeypatch.setattr(
        mod,
        "extract_and_link_trace_context",
        extract_and_link_trace_context,
        raising=False,
    )
    monkeypatch.setattr(mod, "inject_trace_context", inject_trace_context, raising=False)
    monkeypatch.setitem(sys.modules, "ml.observability.tracing", mod)


def test_extract_and_link_from_event_calls_stub(monkeypatch: MonkeyPatch) -> None:
    record: dict[str, object] = {}
    _install_tracing_stub(monkeypatch, record)
    extract_and_link_from_event({"a": 1})
    assert "extract_called_with" in record


def test_get_correlation_and_trace_context_happy_path(monkeypatch: MonkeyPatch) -> None:
    record: dict[str, object] = {}
    _install_tracing_stub(monkeypatch, record)
    meta = get_correlation_and_trace_context(
        run_id="rid",
        dataset_id="features",
        instrument_id="EUR/USD",
        ts_min=1,
        ts_max=2,
        count=3,
    )
    # Correlation id matches deterministic function
    expected = make_correlation_id(
        run_id="rid",
        dataset_id="features",
        instrument_id="EUR/USD",
        ts_min=1,
        ts_max=2,
        count=3,
    )
    assert meta["correlation_id"] == expected
    # Stub inject added a trace_context
    assert "trace_context" in meta


def test_get_correlation_and_trace_context_without_tracing_module(monkeypatch: MonkeyPatch) -> None:
    # Remove tracing module to exercise ImportError branch
    sys.modules.pop("ml.observability.tracing", None)
    meta = get_correlation_and_trace_context(
        run_id="rid",
        dataset_id="features",
        instrument_id="EUR/USD",
        ts_min=10,
        ts_max=20,
        count=5,
    )
    assert "correlation_id" in meta
    # No trace_context injected
    assert "trace_context" not in meta
