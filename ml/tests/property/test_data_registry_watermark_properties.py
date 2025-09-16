#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ml.config.events import Source
from ml.registry.data_registry import DataRegistry
from ml.registry.dataclasses import DatasetManifest, DatasetType, StorageKind


@pytest.fixture
def json_registry(tmp_path: Path) -> DataRegistry:
    return DataRegistry(registry_path=tmp_path)


def _manifest(dataset_id: str) -> DatasetManifest:
    schema = {
        "instrument_id": "string",
        "ts_event": "int64",
        "ts_init": "int64",
    }
    return DatasetManifest(
        dataset_id=dataset_id,
        dataset_type=DatasetType.SIGNALS,
        storage_kind=StorageKind.PARQUET,
        location="/tmp",
        partitioning=None,
        retention_days=7,
        schema=schema,
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["instrument_id", "ts_event"],
        schema_hash="0",
        constraints={},
        lineage=[],
        pipeline_signature="test",
        version="1.0.0",
        created_at=0,
        last_modified=0,
        metadata={},
    )


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    ts_list=st.lists(st.integers(min_value=0, max_value=10**6), min_size=1, max_size=50).map(sorted),
    count_list=st.lists(st.integers(min_value=0, max_value=10**3), min_size=1, max_size=50),
)
def test_watermark_monotonic(json_registry: DataRegistry, ts_list: list[int], count_list: list[int]) -> None:
    ds = "signals"
    inst = "EUR/USD"
    src = Source.HISTORICAL
    # Ensure dataset exists for JSON backend (not strictly required but realistic)
    json_registry.register_dataset(_manifest(ds))

    # Update in non-decreasing order; watermark should never go backwards
    last = -1
    for i, t in enumerate(ts_list):
        json_registry.update_watermark(ds, inst, src, last_success_ns=t, count=count_list[i % len(count_list)], completeness_pct=100.0)
        w = json_registry.get_watermark(ds, inst, src)
        assert w is not None
        assert w.last_success_ns >= last
        last = w.last_success_ns
