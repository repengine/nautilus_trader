from __future__ import annotations

from ml.common.events_util import build_bus_payload
from ml.common.events_util import validate_bus_payload
from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage


def test_validate_bus_payload_accepts_valid_payload() -> None:
    payload = build_bus_payload(
        dataset_id="signals",
        instrument_id="EURUSD.SIM",
        stage=Stage.SIGNAL_EMITTED,
        source=Source.LIVE,
        run_id="run-1",
        ts_min=1,
        ts_max=2,
        count=1,
        status=EventStatus.SUCCESS,
        metadata={"correlation_id": "cid-1"},
        inject_trace_context=False,
    )

    ok, errors = validate_bus_payload(payload)
    assert ok is True
    assert errors == []


def test_validate_bus_payload_rejects_missing_fields() -> None:
    payload = {"dataset_id": "signals"}
    ok, errors = validate_bus_payload(payload)
    assert ok is False
    assert any("missing" in error for error in errors)
