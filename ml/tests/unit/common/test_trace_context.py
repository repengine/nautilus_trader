#!/usr/bin/env python3
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import builtins
import sys

from ml.common.trace_context import (
    extract_and_link_from_event,
    get_correlation_and_trace_context,
)


def test_get_correlation_and_trace_context_injects_and_builds_correlation(monkeypatch):
    # Fake tracing.inject_trace_context to tag metadata
    called: dict[str, Any] = {}

    def fake_inject(md: dict[str, Any]) -> dict[str, Any]:
        md = dict(md)
        md["trace_injected"] = True
        return md

    mod = SimpleNamespace(inject_trace_context=fake_inject)
    monkeypatch.setitem(sys.modules, "ml.observability.tracing", mod)

    md = get_correlation_and_trace_context(
        run_id="runA",
        dataset_id="features",
        instrument_id="EURUSD.SIM",
        ts_min=1,
        ts_max=2,
        count=3,
    )

    assert "correlation_id" in md
    assert md.get("trace_injected") is True


def test_extract_and_link_from_event_graceful_when_tracing_missing(monkeypatch):
    # Ensure module import raises ImportError
    if "ml.observability.tracing" in sys.modules:
        del sys.modules["ml.observability.tracing"]

    # Should not raise
    extract_and_link_from_event({"trace_context": {}})
