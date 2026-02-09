from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from ml.common.reproducibility import ReproducibilityValue
from ml.common.reproducibility import build_configured_reproducibility_provenance
from ml.data import DatasetMetadata
from ml.data import load_dataset_metadata
from ml.data import require_reproducibility_metadata
from ml.data.metadata import write_dataset_metadata
from ml.data.providers.metadata import InstrumentMetadataProvider
from ml.data.sources.metadata import MockMetadataSource, default_metadata
from ml.data.vintage import VintagePolicy


def test_metadata_defaults_parity() -> None:
    symbols = ["XYZ", "ABC"]
    provider = InstrumentMetadataProvider(source=MockMetadataSource(seed=1))

    # Force empty/unknown paths by asking provider to validate defaults
    df = provider._empty_metadata_frame(symbols)
    assert isinstance(df, pl.DataFrame)
    # Compare row-by-row to canonical default_metadata
    expected = [default_metadata(sym) for sym in symbols]
    for i, sym in enumerate(symbols):
        row = df.row(i, named=True)
        for k, v in expected[i].items():
            assert row[k] == v


def _build_dataset_metadata_with_reproducibility(
    reproducibility: dict[str, ReproducibilityValue] | None,
) -> DatasetMetadata:
    return DatasetMetadata(
        dataset_id="unit_dataset",
        vintage_policy=VintagePolicy.REAL_TIME,
        vintage_cutoff=None,
        build_ts="2026-02-07T00:00:00Z",
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
        reproducibility=reproducibility,
    )


@pytest.mark.unit
def test_dataset_metadata_round_trip_preserves_reproducibility_payload(
    tmp_path: Path,
) -> None:
    payload = build_configured_reproducibility_provenance(
        primary_seed=13,
        deterministic_mode=True,
        context="dataset metadata seed",
    )
    metadata = _build_dataset_metadata_with_reproducibility(payload)

    metadata_path = write_dataset_metadata(metadata, tmp_path)
    loaded = load_dataset_metadata(metadata_path)
    loaded_payload = require_reproducibility_metadata(loaded, context="unit")

    assert loaded_payload["seed"] == 13
    assert loaded_payload["deterministic_mode"] is True
    assert isinstance(loaded_payload["python_version"], str)


@pytest.mark.unit
def test_require_reproducibility_metadata_when_missing_raises_value_error() -> None:
    metadata = _build_dataset_metadata_with_reproducibility(None)

    with pytest.raises(ValueError, match="missing reproducibility"):
        require_reproducibility_metadata(metadata, context="unit")


@pytest.mark.unit
def test_require_reproducibility_metadata_when_seed_invalid_raises_value_error() -> None:
    payload = build_configured_reproducibility_provenance(
        primary_seed=17,
        deterministic_mode=True,
        context="dataset metadata seed",
    )
    payload["seed"] = -1
    metadata = _build_dataset_metadata_with_reproducibility(payload)

    with pytest.raises(ValueError, match=r"seed must be >= 0"):
        require_reproducibility_metadata(metadata, context="unit")
