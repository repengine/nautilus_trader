"""Unit tests for dataset target semantics metadata requirements."""

from __future__ import annotations

import pytest

from ml.data import DatasetMetadata
from ml.data import require_target_column_in_semantics
from ml.data import require_target_semantics_metadata
from ml.data.vintage import VintagePolicy
from ml.tests.utils.targets import build_default_target_semantics
from ml.training.datasets.target_generator import build_target_semantics_metadata


@pytest.mark.unit
def test_require_target_semantics_metadata_when_missing_raises_value_error() -> None:
    """Verify target semantics metadata is required."""
    metadata = DatasetMetadata(
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
        target_semantics=None,
    )

    with pytest.raises(ValueError, match="target_semantics"):
        require_target_semantics_metadata(metadata, context="unit")


@pytest.mark.unit
def test_require_target_semantics_metadata_when_present_returns_payload() -> None:
    """Verify target semantics metadata is returned when present."""
    payload = build_target_semantics_metadata(build_default_target_semantics())
    metadata = DatasetMetadata(
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

    result = require_target_semantics_metadata(metadata, context="unit")
    assert result["version"] == payload["version"]
    assert "horizons" in result


@pytest.mark.unit
def test_require_target_column_in_semantics_when_missing_raises_value_error() -> None:
    """Verify target column must be declared in target semantics labels."""
    payload = build_target_semantics_metadata(build_default_target_semantics())
    metadata = DatasetMetadata(
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

    with pytest.raises(ValueError, match="target_col"):
        require_target_column_in_semantics(metadata, "y", context="unit")


@pytest.mark.unit
def test_require_target_column_in_semantics_allows_primary_label() -> None:
    """Verify target column in labels is accepted."""
    payload = build_target_semantics_metadata(build_default_target_semantics())
    metadata = DatasetMetadata(
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

    require_target_column_in_semantics(metadata, "target_bin_15m", context="unit")


@pytest.mark.unit
def test_require_target_column_in_semantics_allows_legacy_alias() -> None:
    """Verify legacy aliases are accepted when enabled."""
    payload = build_target_semantics_metadata(build_default_target_semantics(legacy_aliases=True))
    metadata = DatasetMetadata(
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

    require_target_column_in_semantics(metadata, "y", context="unit")
