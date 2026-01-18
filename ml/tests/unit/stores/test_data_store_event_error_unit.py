"""
DataStore event emission error path: ensure write does not raise if registry fails.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

from ml.stores.data_store import DataStore
from ml.features.earnings.store import DummyEarningsStore
from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategyStore
from ml.registry.dataclasses import DataContract
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import QualityFlag
from ml.registry.dataclasses import StorageKind
from ml.registry.dataclasses import ValidationRule
from ml.registry.dataclasses import ValidationRuleType
from ml.registry.protocols import RegistryProtocol
from ml.config.events import EventStatus, Source, Stage


class _FlakyRegistry(RegistryProtocol):
    def __init__(self, fail_on: str) -> None:
        self.fail_on = fail_on

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
        del dataset_id, instrument_id, stage, source, run_id, ts_min, ts_max, count, status, error, metadata
        if self.fail_on == "emit":
            raise RuntimeError("emit fail")

    def update_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: Source,
        last_success_ns: int,
        count: int,
        completeness_pct: float,
    ) -> None:
        del dataset_id, instrument_id, source, last_success_ns, count, completeness_pct
        if self.fail_on == "wm":
            raise RuntimeError("wm fail")

    def get_manifest(self, dataset_id: str) -> DatasetManifest:
        return DatasetManifest(
            dataset_id=dataset_id,
            dataset_type=DatasetType.FEATURES,
            storage_kind=StorageKind.PARQUET,
            location="/tmp",
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
        del dataset_id, changes
        return None


class _NoOpStore:
    def write_features(self, *a: Any, **k: Any) -> None:
        return None


def test_write_features_tolerates_registry_emit_error() -> None:
    ds = DataStore(
        connection_string="sqlite:///:memory:",
        registry=_FlakyRegistry("emit"),
        feature_store=cast(FeatureStore, _NoOpStore()),
        model_store=cast(ModelStore, MagicMock(spec=ModelStore)),
        strategy_store=cast(StrategyStore, MagicMock(spec=StrategyStore)),
        earnings_store=DummyEarningsStore(),
        fail_on_validation_error=False,
    )

    from ml.stores.base import FeatureData

    fd = FeatureData(
        feature_set_id="fs",
        instrument_id="X.SIM",
        values={"a": 1.0},
        _ts_event=1,
        _ts_init=1,
    )
    # Should not raise even if registry emit_event fails internally
    event = DataStore.write_features(ds, instrument_id="X.SIM", features=[fd])
    assert event.record_count == 1


def test_write_features_tolerates_watermark_error() -> None:
    ds = DataStore(
        connection_string="sqlite:///:memory:",
        registry=_FlakyRegistry("wm"),
        feature_store=cast(FeatureStore, _NoOpStore()),
        model_store=cast(ModelStore, MagicMock(spec=ModelStore)),
        strategy_store=cast(StrategyStore, MagicMock(spec=StrategyStore)),
        earnings_store=DummyEarningsStore(),
        fail_on_validation_error=False,
    )

    from ml.stores.base import FeatureData

    fd = FeatureData(
        feature_set_id="fs",
        instrument_id="X.SIM",
        values={"a": 1.0},
        _ts_event=1,
        _ts_init=1,
    )
    event = DataStore.write_features(ds, instrument_id="X.SIM", features=[fd])
    assert event.status == "success"
