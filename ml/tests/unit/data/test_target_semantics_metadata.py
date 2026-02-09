"""Unit tests for dataset target semantics metadata requirements."""

from __future__ import annotations

import pytest

from ml.config.targets import HORIZON_RESOLUTION_BAR_INDEX
from ml.config.targets import HORIZON_RESOLUTION_WALL_CLOCK
from ml.config.targets import TARGET_SEMANTICS_CONTRACT_ID
from ml.config.targets import TARGET_SEMANTICS_CONTRACT_MAJOR
from ml.data import DatasetMetadata
from ml.data import require_target_semantics_contract
from ml.data import require_target_column_in_semantics
from ml.data import require_target_semantics_horizon_mode
from ml.data import require_target_semantics_metadata
from ml.data.metadata import require_target_semantics_execution_contract
from ml.data.vintage import VintagePolicy
from ml.tests.utils.targets import build_default_target_semantics
from ml.training.datasets.target_generator import build_target_semantics_metadata


def _build_metadata(payload: dict[str, object] | None) -> DatasetMetadata:
    return DatasetMetadata(
        dataset_id="test_dataset",
        vintage_policy=VintagePolicy.REAL_TIME,
        vintage_cutoff=None,
        build_ts="2024-01-01T00:00:00Z",
        ts_event_start=None,
        ts_event_end=None,
        overall_window=None,
        train_window=None,
        validation_window=None,
        test_window=None,
        macro_observation_counts={},
        capability_flags={},
        market_bindings=None,
        target_semantics=payload,
    )


@pytest.mark.unit
def test_require_target_semantics_metadata_when_missing_raises_value_error() -> None:
    """Verify target semantics metadata is required."""
    metadata = _build_metadata(None)

    with pytest.raises(ValueError, match="target_semantics"):
        require_target_semantics_metadata(metadata, context="unit")


@pytest.mark.unit
def test_require_target_semantics_metadata_when_present_returns_payload() -> None:
    """Verify target semantics metadata is returned when present."""
    payload = build_target_semantics_metadata(build_default_target_semantics())
    metadata = _build_metadata(payload)

    result = require_target_semantics_metadata(metadata, context="unit")
    contract = result["contract"]
    assert isinstance(contract, dict)
    assert contract["id"] == TARGET_SEMANTICS_CONTRACT_ID
    assert contract["major"] == TARGET_SEMANTICS_CONTRACT_MAJOR
    assert "horizons" in result
    assert result["horizon_resolution_mode"] == HORIZON_RESOLUTION_BAR_INDEX
    alignment = result["horizon_alignment"]
    assert isinstance(alignment, dict)
    assert alignment["mode"] == HORIZON_RESOLUTION_BAR_INDEX


@pytest.mark.unit
def test_require_target_semantics_contract_validates_capabilities() -> None:
    """Verify canonical contract and capability checks succeed for default payloads."""
    payload = build_target_semantics_metadata(build_default_target_semantics())
    metadata = _build_metadata(payload)

    contract = require_target_semantics_contract(
        metadata,
        required_capabilities=("labels_declared",),
        context="unit",
    )
    assert contract["id"] == TARGET_SEMANTICS_CONTRACT_ID


@pytest.mark.unit
def test_require_target_semantics_contract_when_version_missing_raises_value_error() -> None:
    """Verify contract validation requires canonical epoch version."""
    payload = build_target_semantics_metadata(build_default_target_semantics())
    payload.pop("version", None)
    metadata = _build_metadata(payload)

    with pytest.raises(ValueError, match="version mismatch"):
        require_target_semantics_contract(
            metadata,
            required_capabilities=("labels_declared",),
            context="unit",
        )


@pytest.mark.unit
def test_require_target_semantics_contract_when_contract_id_mismatch_raises_value_error() -> None:
    """Verify contract validation rejects mismatched contract identifiers."""
    payload = build_target_semantics_metadata(build_default_target_semantics())
    contract = payload["contract"]
    assert isinstance(contract, dict)
    contract["id"] = "unexpected_contract"
    metadata = _build_metadata(payload)

    with pytest.raises(ValueError, match=r"contract\.id mismatch"):
        require_target_semantics_contract(
            metadata,
            required_capabilities=("labels_declared",),
            context="unit",
        )


@pytest.mark.unit
def test_require_target_semantics_contract_when_contract_major_mismatch_raises_value_error() -> None:
    """Verify contract validation rejects mismatched contract major versions."""
    payload = build_target_semantics_metadata(build_default_target_semantics())
    contract = payload["contract"]
    assert isinstance(contract, dict)
    contract["major"] = TARGET_SEMANTICS_CONTRACT_MAJOR + 1
    metadata = _build_metadata(payload)

    with pytest.raises(ValueError, match=r"contract\.major mismatch"):
        require_target_semantics_contract(
            metadata,
            required_capabilities=("labels_declared",),
            context="unit",
        )


@pytest.mark.unit
def test_require_target_column_in_semantics_when_missing_raises_value_error() -> None:
    """Verify target column must be declared in target semantics labels."""
    payload = build_target_semantics_metadata(build_default_target_semantics())
    metadata = _build_metadata(payload)

    with pytest.raises(ValueError, match="target_col"):
        require_target_column_in_semantics(metadata, "y", context="unit")


@pytest.mark.unit
def test_require_target_column_in_semantics_allows_primary_label() -> None:
    """Verify target column in labels is accepted."""
    payload = build_target_semantics_metadata(build_default_target_semantics())
    metadata = _build_metadata(payload)

    require_target_column_in_semantics(metadata, "target_bin_15m", context="unit")


@pytest.mark.unit
def test_require_target_column_in_semantics_allows_legacy_alias() -> None:
    """Verify legacy aliases are accepted when enabled."""
    payload = build_target_semantics_metadata(build_default_target_semantics(legacy_aliases=True))
    metadata = _build_metadata(payload)

    require_target_column_in_semantics(metadata, "y", context="unit")


@pytest.mark.unit
def test_require_target_semantics_metadata_when_horizon_mode_missing_raises_value_error() -> None:
    """Verify horizon mode is required in target semantics metadata."""
    payload = build_target_semantics_metadata(build_default_target_semantics())
    payload.pop("horizon_resolution_mode", None)
    metadata = _build_metadata(payload)

    with pytest.raises(ValueError, match="horizon_resolution_mode"):
        require_target_semantics_metadata(metadata, context="unit")


@pytest.mark.unit
def test_require_target_semantics_metadata_when_execution_missing_raises_value_error() -> None:
    """Verify execution contract is required in target semantics metadata."""
    payload = build_target_semantics_metadata(build_default_target_semantics())
    payload.pop("execution", None)
    metadata = _build_metadata(payload)

    with pytest.raises(ValueError, match="missing required keys"):
        require_target_semantics_metadata(metadata, context="unit")


@pytest.mark.unit
def test_require_target_semantics_metadata_when_wall_clock_alignment_missing_timestamp_column_raises_value_error() -> None:
    """Verify wall-clock semantics require explicit timestamp alignment field."""
    payload = build_target_semantics_metadata(
        build_default_target_semantics(
            horizon_resolution_mode=HORIZON_RESOLUTION_WALL_CLOCK,
        ),
    )
    alignment = payload["horizon_alignment"]
    assert isinstance(alignment, dict)
    alignment.pop("timestamp_column", None)
    metadata = _build_metadata(payload)

    with pytest.raises(ValueError, match="timestamp_column"):
        require_target_semantics_metadata(metadata, context="unit")


@pytest.mark.unit
def test_require_target_semantics_horizon_mode_when_expected_mismatch_raises_value_error() -> None:
    """Verify expected horizon mode mismatch fails fast."""
    payload = build_target_semantics_metadata(build_default_target_semantics())
    metadata = _build_metadata(payload)

    with pytest.raises(ValueError, match="horizon_resolution_mode mismatch"):
        require_target_semantics_horizon_mode(
            metadata,
            expected_mode=HORIZON_RESOLUTION_WALL_CLOCK,
            context="unit",
        )


@pytest.mark.unit
def test_require_target_semantics_execution_contract_when_expected_matches_returns_payload() -> None:
    """Verify execution contract helper returns normalized payload when aligned."""
    payload = build_target_semantics_metadata(build_default_target_semantics())
    metadata = _build_metadata(payload)

    execution = require_target_semantics_execution_contract(
        metadata,
        expected_execution={
            "entry_price_column": "close",
            "exit_price_column": "close",
            "latency_bars": 0,
            "latency_unit": "bars",
            "unresolved_context_mode": "zero_return",
            "unresolved_context_return": 0.0,
        },
        context="unit",
    )
    assert execution["entry_price_column"] == "close"
    assert execution["latency_bars"] == 0


@pytest.mark.unit
def test_require_target_semantics_execution_contract_when_expected_mismatch_raises_value_error() -> None:
    """Verify execution contract mismatches fail fast."""
    payload = build_target_semantics_metadata(build_default_target_semantics())
    metadata = _build_metadata(payload)

    with pytest.raises(ValueError, match="execution\\.latency_bars mismatch"):
        require_target_semantics_execution_contract(
            metadata,
            expected_execution={
                "entry_price_column": "close",
                "exit_price_column": "close",
                "latency_bars": 1,
                "latency_unit": "bars",
                "unresolved_context_mode": "zero_return",
                "unresolved_context_return": 0.0,
            },
            context="unit",
        )
