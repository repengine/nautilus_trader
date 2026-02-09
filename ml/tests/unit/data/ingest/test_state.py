from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.data.ingest.resume import IngestState
from ml.data.ingest.state import load_state
from ml.data.ingest.state import save_state


def test_load_state_returns_empty_when_missing(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"

    state = load_state(path)

    assert state.last_ts_ns_by_instrument == {}


def test_load_state_handles_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("not-json", encoding="utf-8")

    state = load_state(path)

    assert state.last_ts_ns_by_instrument == {}


def test_load_state_coerces_mapping_values(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    payload = {
        "last_ts_ns_by_instrument": {
            "SPY": "123",
            "QQQ": 456.0,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    state = load_state(path)

    assert state.last_ts_ns_by_instrument == {"SPY": 123, "QQQ": 456}


def test_load_state_ignores_non_mapping_payload(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"last_ts_ns_by_instrument": ["bad"]}), encoding="utf-8")

    state = load_state(path)

    assert state.last_ts_ns_by_instrument == {}


def test_save_state_persists_json(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = IngestState(last_ts_ns_by_instrument={"SPY": 101, "QQQ": 202})

    save_state(path, state)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == {"last_ts_ns_by_instrument": {"QQQ": 202, "SPY": 101}}
