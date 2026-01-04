#!/usr/bin/env python3
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import sys

from pytest import MonkeyPatch
from ml.common.events_util import build_bus_payload
from ml.config.events import EventStatus, Source, Stage


def test_build_bus_payload_enum_and_trace_injection(monkeypatch: MonkeyPatch) -> None:
    # Inject fake tracing.inject_trace_context to tag metadata
    def fake_inject(md: dict[str, Any]) -> dict[str, Any]:
        md = dict(md)
        md["trace_tag"] = "ok"
        return md

    mod = SimpleNamespace(inject_trace_context=fake_inject)
    monkeypatch.setitem(sys.modules, "ml.observability.tracing", mod)

    payload: dict[str, Any] = build_bus_payload(
        dataset_id="features",
        instrument_id="EURUSD.SIM",
        stage=Stage.FEATURE_COMPUTED,
        source=Source.LIVE,
        run_id="run-1",
        ts_min=1,
        ts_max=2,
        count=3,
        status=EventStatus.SUCCESS,
        metadata={"a": 1},
        inject_trace_context=True,
    )

    assert payload["stage"] == Stage.FEATURE_COMPUTED.value
    assert payload["source"] == Source.LIVE.value
    assert payload["status"] == EventStatus.SUCCESS.value
    assert payload["metadata"]["a"] == 1
    assert payload["metadata"]["trace_tag"] == "ok"


def test_build_bus_payload_without_tracing_module(monkeypatch: MonkeyPatch) -> None:
    # Remove tracing module to exercise ImportError branch
    monkeypatch.delitem(sys.modules, "ml.observability.tracing", raising=False)

    payload: dict[str, Any] = build_bus_payload(
        dataset_id="predictions",
        instrument_id="BTCUSD.COINBASE",
        stage="PREDICTIONS",
        source="historical",
        run_id="r",
        ts_min=10,
        ts_max=20,
        count=5,
        status="SUCCESS",
        metadata={},
        inject_trace_context=True,
    )

    assert payload["source"] == "historical"
    assert isinstance(payload["metadata"], dict)
