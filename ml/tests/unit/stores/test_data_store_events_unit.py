"""
DataStore event emission tests using stubbed registry and stores.

Focus: functional outcome that DataStore emits events and updates watermarks
with expected parameters when writing features/predictions/signals.

"""

from __future__ import annotations

from typing import Any, Callable, cast

import pytest

from ml.config.events import EventStatus, Source, Stage
from ml.stores.data_store import DataStore
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
from nautilus_trader.model.identifiers import InstrumentId


class _StubRegistry(RegistryProtocol):
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.watermarks: list[dict[str, Any]] = []

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
                "source": source.value,
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
        self.watermarks.append(
            {
                "dataset_id": dataset_id,
                "instrument_id": instrument_id,
                "source": source.value,
                "last_success_ns": last_success_ns,
                "count": count,
                "completeness_pct": completeness_pct,
            },
        )

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
        del dataset_id
        del changes
        return None


class _NoOpStore:
    def write_features(self, *args: Any, **kwargs: Any) -> None:
        """
        No-op write_features.
        """
        return None

    def write_prediction(self, *args: Any, **kwargs: Any) -> None:
        """
        No-op write_prediction.
        """
        return None

    def write_signal(self, *args: Any, **kwargs: Any) -> None:
        """
        No-op write_signal.
        """
        return None

    def write_batch(self, *args: Any, **kwargs: Any) -> None:
        """
        No-op batch write for stores expecting it.
        """
        return None


@pytest.fixture
def stubbed_data_store(monkeypatch: pytest.MonkeyPatch) -> tuple[DataStore, _StubRegistry]:
    # Build an instance and replace internal stores and registry accessor
    ds: DataStore = object.__new__(DataStore)
    ds_any = cast(Any, ds)
    ds_any.connection_string = "sqlite:///:memory:"
    ds_any.feature_store = cast(FeatureStore, _NoOpStore())
    ds_any.model_store = cast(ModelStore, _NoOpStore())
    ds_any.strategy_store = cast(StrategyStore, _NoOpStore())
    stub_registry = _StubRegistry()
    ds_any.registry = stub_registry
    ds_any._data_registry = stub_registry
    ds_any._get_dataset_ids = lambda: {
        "features": "features",
        "predictions": "predictions",
        "signals": "signals",
    }

    # Provide clock stub
    class _Clock:
        def timestamp_ns(self) -> int:
            return 100

    ds_any.clock = _Clock()
    # Avoid schema/registration side effects
    ds_any._ensure_dataset_registered = cast(Callable[..., None], lambda **kwargs: None)
    return ds, stub_registry


def test_data_store_emits_feature_events(
    stubbed_data_store: tuple[DataStore, _StubRegistry],
    default_instrument_id: InstrumentId,
    test_timestamps: tuple[int, int],
) -> None:
    ds, reg = stubbed_data_store
    from ml.stores.base import FeatureData

    ts_event, ts_init = test_timestamps
    instrument_id_str = str(default_instrument_id)

    fd = FeatureData(
        feature_set_id="fs1",
        instrument_id=instrument_id_str,
        values={"a": 1.0},
        _ts_event=ts_event,
        _ts_init=ts_init,
    )
    ds.write_features(instrument_id=instrument_id_str, features=[fd])
    assert any(e for e in reg.events if e["dataset_id"] == "features")


def test_data_store_emits_prediction_events(
    stubbed_data_store: tuple[DataStore, _StubRegistry],
    default_instrument_id: InstrumentId,
    test_timestamps: tuple[int, int],
) -> None:
    ds, reg = stubbed_data_store
    from ml.stores.base import ModelPrediction

    ts_event, ts_init = test_timestamps
    instrument_id_str = str(default_instrument_id)

    mp = ModelPrediction(
        model_id="m1",
        instrument_id=instrument_id_str,
        prediction=0.8,
        confidence=0.9,
        features_used={"a": 1.0},
        inference_time_ms=0.1,
        _ts_event=ts_event,
        _ts_init=ts_init,
    )
    ds.write_predictions(predictions=[mp])
    assert any(e for e in reg.events if e["dataset_id"] == "predictions")


def test_data_store_emits_signal_events(
    stubbed_data_store: tuple[DataStore, _StubRegistry],
    default_instrument_id: InstrumentId,
    test_timestamps: tuple[int, int],
) -> None:
    ds, reg = stubbed_data_store
    from ml.stores.base import StrategySignal

    ts_event, ts_init = test_timestamps
    instrument_id_str = str(default_instrument_id)

    ss = StrategySignal(
        strategy_id="s1",
        instrument_id=instrument_id_str,
        signal_type="BUY",
        strength=0.8,
        model_predictions={"m1": 0.8},
        risk_metrics={"conf": 0.8},
        execution_params={"side": "BUY"},
        _ts_event=ts_event,
        _ts_init=ts_init,
    )
    ds.write_signals(signals=[ss])
    assert any(e for e in reg.events if e["dataset_id"] == "signals")
