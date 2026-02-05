from __future__ import annotations

import pytest

from ml.data.dataset_manifest_defaults import build_auto_dataset_manifest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import StorageKind


@pytest.mark.unit
def test_build_auto_manifest_defaults_mbp10() -> None:
    manifest = build_auto_dataset_manifest(
        dataset_id="TEST.MBP10",
        dataset_type=DatasetType.MBP10,
        location="/tmp/catalog",
        storage_kind=StorageKind.PARQUET,
        pipeline_signature="test",
    )

    assert manifest.dataset_type is DatasetType.MBP10
    assert manifest.metadata.get("schema_kind") == "mbp10"
    assert "bids" in manifest.schema
    assert "asks" in manifest.schema
    assert manifest.seq_field == "sequence"
    assert manifest.primary_keys == ["instrument_id", "ts_event", "sequence"]


@pytest.mark.unit
def test_build_auto_manifest_defaults_mbo() -> None:
    manifest = build_auto_dataset_manifest(
        dataset_id="TEST.MBO",
        dataset_type=DatasetType.MBO,
        location="/tmp/catalog",
        storage_kind=StorageKind.PARQUET,
        pipeline_signature="test",
    )

    assert manifest.dataset_type is DatasetType.MBO
    assert manifest.metadata.get("schema_kind") == "mbo"
    assert "order_payload" in manifest.schema
    assert manifest.seq_field == "sequence"
    assert manifest.primary_keys == ["instrument_id", "ts_event", "sequence"]


@pytest.mark.unit
def test_build_auto_manifest_overrides_eqs_mini_mbp10() -> None:
    manifest = build_auto_dataset_manifest(
        dataset_id="EQUS.MINI_MBP10",
        dataset_type=None,
        location="/tmp/catalog",
        storage_kind=StorageKind.PARQUET,
        pipeline_signature="test",
    )

    assert manifest.dataset_type is DatasetType.MBP10
    assert manifest.metadata.get("schema_kind") == "mbp10"
    assert manifest.metadata.get("dataset_family") == "equities_mini"
    assert manifest.retention_days == 365
    assert manifest.partitioning.get("interval") == "monthly"


@pytest.mark.unit
def test_build_auto_manifest_overrides_eqs_mini_mbo() -> None:
    manifest = build_auto_dataset_manifest(
        dataset_id="EQUS.MINI_MBO",
        dataset_type=None,
        location="/tmp/catalog",
        storage_kind=StorageKind.PARQUET,
        pipeline_signature="test",
    )

    assert manifest.dataset_type is DatasetType.MBO
    assert manifest.metadata.get("schema_kind") == "mbo"
    assert manifest.metadata.get("dataset_family") == "equities_mini"
    assert manifest.retention_days == 365
    assert manifest.partitioning.get("interval") == "monthly"
