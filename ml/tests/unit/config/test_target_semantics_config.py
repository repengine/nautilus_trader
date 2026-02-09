"""Unit tests for canonical target semantics config parsing."""

from __future__ import annotations

import json

import pytest

from ml.config.targets import EXECUTION_UNRESOLVED_CONTEXT_FAIL
from ml.config.targets import EXECUTION_UNRESOLVED_CONTEXT_ZERO_RETURN
from ml.config.targets import HORIZON_RESOLUTION_WALL_CLOCK
from ml.config.targets import TARGET_SEMANTICS_CONTRACT_ID
from ml.config.targets import TARGET_SEMANTICS_CONTRACT_MAJOR
from ml.config.targets import TARGET_SEMANTICS_EPOCH_VERSION
from ml.config.targets import TARGET_SEMANTICS_REQUIRED_CAPABILITIES
from ml.config.targets import TargetSemanticsConfig


pytestmark = pytest.mark.unit


def test_target_semantics_from_dict_parses_contract_payload() -> None:
    payload = {
        "version": TARGET_SEMANTICS_EPOCH_VERSION,
        "contract": {
            "id": TARGET_SEMANTICS_CONTRACT_ID,
            "major": TARGET_SEMANTICS_CONTRACT_MAJOR,
            "capabilities": list(TARGET_SEMANTICS_REQUIRED_CAPABILITIES),
        },
        "horizons": [{"minutes": 5, "label": "5m"}],
        "binary": {"enabled": True, "threshold_bps": 12.5, "return_basis": "raw"},
        "multiclass": {"enabled": False},
        "regression": {"enabled": False},
    }

    parsed = TargetSemanticsConfig.from_dict(payload)

    assert parsed.version == TARGET_SEMANTICS_EPOCH_VERSION
    assert parsed.contract_id == TARGET_SEMANTICS_CONTRACT_ID
    assert parsed.contract_major == TARGET_SEMANTICS_CONTRACT_MAJOR
    assert parsed.capabilities == TARGET_SEMANTICS_REQUIRED_CAPABILITIES
    assert parsed.horizon_labels == ("5m",)
    assert parsed.execution_entry_price_column == "close"
    assert parsed.execution_exit_price_column == "close"
    assert parsed.execution_latency_bars == 0
    assert parsed.unresolved_execution_context_mode == EXECUTION_UNRESOLVED_CONTEXT_ZERO_RETURN


def test_target_semantics_from_json_parses_contract_payload() -> None:
    payload = {
        "contract": {
            "id": TARGET_SEMANTICS_CONTRACT_ID,
            "major": TARGET_SEMANTICS_CONTRACT_MAJOR,
            "capabilities": list(TARGET_SEMANTICS_REQUIRED_CAPABILITIES),
        },
        "horizons": [{"minutes": 15, "label": "15m"}],
    }

    parsed = TargetSemanticsConfig.from_json(json.dumps(payload))

    assert parsed.contract_metadata() == {
        "id": TARGET_SEMANTICS_CONTRACT_ID,
        "major": TARGET_SEMANTICS_CONTRACT_MAJOR,
        "capabilities": list(TARGET_SEMANTICS_REQUIRED_CAPABILITIES),
    }


def test_target_semantics_from_dict_parses_wall_clock_horizon_mode() -> None:
    payload = {
        "contract": {
            "id": TARGET_SEMANTICS_CONTRACT_ID,
            "major": TARGET_SEMANTICS_CONTRACT_MAJOR,
            "capabilities": list(TARGET_SEMANTICS_REQUIRED_CAPABILITIES),
        },
        "horizons": [{"minutes": 15, "label": "15m"}],
        "horizon_resolution_mode": HORIZON_RESOLUTION_WALL_CLOCK,
        "wall_clock_timestamp_column": "ts_event",
    }

    parsed = TargetSemanticsConfig.from_dict(payload)

    assert parsed.horizon_resolution_mode == HORIZON_RESOLUTION_WALL_CLOCK
    assert parsed.wall_clock_timestamp_column == "ts_event"


def test_target_semantics_from_dict_parses_execution_payload() -> None:
    payload = {
        "contract": {
            "id": TARGET_SEMANTICS_CONTRACT_ID,
            "major": TARGET_SEMANTICS_CONTRACT_MAJOR,
            "capabilities": list(TARGET_SEMANTICS_REQUIRED_CAPABILITIES),
        },
        "horizons": [{"minutes": 15, "label": "15m"}],
        "execution": {
            "entry_price_column": "entry_px",
            "exit_price_column": "exit_px",
            "latency_bars": 2,
            "latency_unit": "bars",
            "unresolved_context_mode": EXECUTION_UNRESOLVED_CONTEXT_FAIL,
        },
    }

    parsed = TargetSemanticsConfig.from_dict(payload)

    assert parsed.execution_entry_price_column == "entry_px"
    assert parsed.execution_exit_price_column == "exit_px"
    assert parsed.execution_latency_bars == 2
    assert parsed.unresolved_execution_context_mode == EXECUTION_UNRESOLVED_CONTEXT_FAIL
    assert parsed.execution_metadata()["latency_unit"] == "bars"


def test_target_semantics_from_dict_when_contract_payload_not_mapping_raises_value_error() -> None:
    with pytest.raises(ValueError, match="contract payload must be a mapping"):
        TargetSemanticsConfig.from_dict({"contract": "invalid"})


def test_target_semantics_from_dict_when_execution_payload_not_mapping_raises_value_error() -> None:
    with pytest.raises(ValueError, match="execution payload must be a mapping"):
        TargetSemanticsConfig.from_dict({"execution": "invalid"})


def test_target_semantics_from_dict_when_capabilities_not_sequence_raises_value_error() -> None:
    with pytest.raises(ValueError, match="capabilities must be a list/tuple"):
        TargetSemanticsConfig.from_dict(
            {
                "contract": {
                    "id": TARGET_SEMANTICS_CONTRACT_ID,
                    "major": TARGET_SEMANTICS_CONTRACT_MAJOR,
                    "capabilities": "labels_declared",
                },
            },
        )


def test_target_semantics_from_dict_when_contract_id_mismatch_raises_value_error() -> None:
    with pytest.raises(ValueError, match="contract_id is fixed"):
        TargetSemanticsConfig.from_dict(
            {
                "contract": {
                    "id": "invalid_contract",
                    "major": TARGET_SEMANTICS_CONTRACT_MAJOR,
                    "capabilities": list(TARGET_SEMANTICS_REQUIRED_CAPABILITIES),
                },
            },
        )


def test_target_semantics_from_json_when_decoded_payload_not_mapping_raises_value_error() -> None:
    with pytest.raises(ValueError, match="decode to an object"):
        TargetSemanticsConfig.from_json('["invalid"]')


def test_target_semantics_from_dict_when_horizon_mode_invalid_raises_value_error() -> None:
    with pytest.raises(ValueError, match="horizon_resolution_mode must be one of"):
        TargetSemanticsConfig.from_dict(
            {
                "horizon_resolution_mode": "invalid_mode",
            },
        )


def test_target_semantics_from_dict_when_wall_clock_timestamp_column_missing_raises_value_error() -> None:
    with pytest.raises(ValueError, match="wall_clock_timestamp_column must be a non-empty string"):
        TargetSemanticsConfig.from_dict(
            {
                "horizon_resolution_mode": HORIZON_RESOLUTION_WALL_CLOCK,
                "wall_clock_timestamp_column": "",
            },
        )


def test_target_semantics_from_dict_when_execution_latency_negative_raises_value_error() -> None:
    with pytest.raises(ValueError, match="execution_latency_bars must be >= 0"):
        TargetSemanticsConfig.from_dict(
            {
                "execution": {
                    "entry_price_column": "close",
                    "exit_price_column": "close",
                    "latency_bars": -1,
                    "latency_unit": "bars",
                    "unresolved_context_mode": EXECUTION_UNRESOLVED_CONTEXT_ZERO_RETURN,
                    "unresolved_context_return": 0.0,
                },
            },
        )


def test_target_semantics_from_dict_when_unresolved_execution_mode_invalid_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unresolved_execution_context_mode must be one of"):
        TargetSemanticsConfig.from_dict(
            {
                "execution": {
                    "entry_price_column": "close",
                    "exit_price_column": "close",
                    "latency_bars": 0,
                    "latency_unit": "bars",
                    "unresolved_context_mode": "invalid",
                },
            },
        )
