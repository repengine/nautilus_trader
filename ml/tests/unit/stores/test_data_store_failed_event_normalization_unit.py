"""
DataStore failed event normalization: invalid source normalized to 'live'.
"""

from __future__ import annotations

from typing import Any, Callable, cast

import pytest

from ml.stores.data_store import DataStore
from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategyStore
from ml.registry.protocols import RegistryProtocol
from ml.registry.dataclasses import DataContract, DatasetManifest, DatasetType, StorageKind
from ml.registry.dataclasses import ValidationRule, ValidationRuleType, QualityFlag
from ml.config.events import EventStatus, Source, Stage


class _RegistryCap(RegistryProtocol):
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit_event(
        self,
        dataset_id: str,
        instrument_id: str,
        stage: Stage,
        source: Source,
        run_id: str,
        ts_min: int,
        ts_max: int,
        count: int,
        status: EventStatus,
        error: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.events.append(
            {
                "dataset_id": dataset_id,
                "instrument_id": instrument_id,
                "stage": stage.value,
                "source": source.value if isinstance(source, Source) else str(source),
                "run_id": run_id,
                "ts_min": ts_min,
                "ts_max": ts_max,
                "count": count,
                "status": status.value,
                "error": error,
                "metadata": metadata or {},
            },
        )

    def update_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: Source,
        last_success_ns: int,
        count: int,
        completeness_pct: float,
    ) -> None:
        return None

    def get_manifest(self, dataset_id: str) -> DatasetManifest:
        return DatasetManifest(
            dataset_id=dataset_id,
            dataset_type=DatasetType.FEATURES,
            storage_kind=StorageKind.PARQUET,
            location="",
            partitioning={},
            retention_days=1,
            schema={"instrument_id": "str", "ts_event": "int64"},
            ts_field="ts_event",
            seq_field=None,
            primary_keys=["instrument_id", "ts_event"],
            schema_hash="",
            constraints={},
            lineage=[],
            pipeline_signature="test",
            version="1.0.0",
        )

    def get_contract(self, dataset_id: str) -> DataContract:
        return DataContract(
            contract_id=f"contract-{dataset_id}",
            dataset_id=dataset_id,
            version="1.0.0",
            validation_rules=[
                ValidationRule(
                    rule_type=ValidationRuleType.MONOTONICITY,
                    field_name="ts_event",
                    parameters={"direction": "increasing"},
                    severity=QualityFlag.FAIL,
                    description="ts_event must increase",
                ),
            ],
        )

    def register_dataset(self, manifest: DatasetManifest) -> str:
        return manifest.dataset_id

    def update_manifest(self, dataset_id: str, changes: dict[str, object]) -> None:
        del dataset_id
        del changes
        return None


class _FailStore:
    def write_features(self, *a: Any, **k: Any) -> None:
        raise RuntimeError("store failed")


def test_failed_event_source_normalized_to_live() -> None:
    ds: DataStore = object.__new__(DataStore)
    setattr(ds, "connection_string", "sqlite:///:memory:")
    fail_store = _FailStore()
    setattr(ds, "feature_store", cast(FeatureStore, fail_store))
    setattr(ds, "model_store", cast(ModelStore, fail_store))
    setattr(ds, "strategy_store", cast(StrategyStore, fail_store))
    reg = _RegistryCap()
    setattr(ds, "registry", reg)
    setattr(ds, "_data_registry", reg)
    setattr(ds, "_ensure_dataset_registered", cast(Callable[..., None], lambda **kwargs: None))
    clock = type("_C", (), {"timestamp_ns": lambda self: 100})()
    setattr(ds, "clock", clock)

    from ml.stores.base import FeatureData

    fd = FeatureData(
        feature_set_id="fs",
        instrument_id="X.SIM",
        values={"a": 1.0},
        _ts_event=1,
        _ts_init=1,
    )
    with pytest.raises(RuntimeError, match="Feature write failed: store failed"):
        DataStore.write_features(ds, instrument_id="X.SIM", features=[fd], source="unit")
    # Check last event has normalized source
    assert reg.events and reg.events[-1]["source"] == "live"
    assert reg.events[-1]["status"] == EventStatus.FAILED.value
    assert reg.events[-1]["count"] == 0
