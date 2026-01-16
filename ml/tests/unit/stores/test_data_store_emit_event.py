from __future__ import annotations

from typing import Any, Callable, cast
from unittest.mock import MagicMock

import pytest

from ml.common.message_bus import MessagePublisherProtocol
from ml.config.events import EventStatus, Source, Stage
from ml.features.earnings.store import DummyEarningsStore
from ml.stores.data_store_facade import DataStore
from ml.stores.feature_store_facade import FeatureStore
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

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)


pytestmark = pytest.mark.usefixtures("patch_datastore")


class CapturePublisher(MessagePublisherProtocol):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        self.calls.append((topic, payload))
        return True


class RegistryMockAdapter(RegistryProtocol):
    """
    Tiny adapter to make a MagicMock behave like a typed RegistryProtocol.

    Converts enum arguments to their `.value` strings before forwarding to the
    underlying mock. Keeps tests decoupled from production-only branches.

    """

    def __init__(self, mock: MagicMock) -> None:
        self._mock = mock

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
        self._mock.emit_event(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            stage=stage.value,
            source=source.value,
            run_id=run_id,
            ts_min=ts_min,
            ts_max=ts_max,
            count=count,
            status=status.value,
            error=error,
            metadata=metadata,
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
        # Allow tests to assert watermark behavior if needed
        self._mock.update_watermark(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            source=source.value,
            last_success_ns=last_success_ns,
            count=count,
            completeness_pct=completeness_pct,
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
        del dataset_id, changes
        return None


class TestDataStoreEmitEvent:
    def test_emit_event_attaches_correlation_id_and_normalizes(self) -> None:
        # Arrange
        mock_registry = MagicMock()
        # Avoid DB connections by injecting store mocks
        feature_store = cast(FeatureStore, MagicMock(spec=FeatureStore))
        model_store = cast(ModelStore, MagicMock(spec=ModelStore))
        strategy_store = cast(StrategyStore, MagicMock(spec=StrategyStore))
        capture = CapturePublisher()
        store = DataStore(
            connection_string="sqlite:///:memory:",
            registry=RegistryMockAdapter(mock_registry),
            feature_store=feature_store,
            model_store=model_store,
            strategy_store=strategy_store,
            earnings_store=DummyEarningsStore(),
            publisher=capture,
            enable_publishing=True,
        )

        dataset_id = "features"
        instrument_id = "EURUSD.SIM"
        run_id = "run_123"
        ts_min = 1000
        ts_max = 2000
        count = 42

        # Act
        store.emit_event(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            stage=Stage.FEATURE_COMPUTED,
            source=Source.LIVE,
            run_id=run_id,
            ts_min=ts_min,
            ts_max=ts_max,
            count=count,
            status="success",
        )

        # Assert
        assert mock_registry.emit_event.call_count == 1
        kwargs = mock_registry.emit_event.call_args.kwargs
        assert kwargs["dataset_id"] == dataset_id
        assert kwargs["instrument_id"] == instrument_id
        assert kwargs["stage"] == Stage.FEATURE_COMPUTED.value
        assert kwargs["source"] == Source.LIVE.value
        assert kwargs["run_id"] == run_id
        assert kwargs["ts_min"] == ts_min
        assert kwargs["ts_max"] == ts_max
        assert kwargs["count"] == count
        assert kwargs["status"] == "success"
        metadata = cast(dict[str, Any], kwargs["metadata"])
        assert "correlation_id" in metadata and isinstance(metadata["correlation_id"], str)
        assert len(capture.calls) == 1
        topic, payload = capture.calls[0]
        assert topic.startswith("ml.features.updated.")
        assert payload["dataset_id"] == dataset_id
        assert payload["stage"] == Stage.FEATURE_COMPUTED.value

    def test_emit_event_respects_external_metadata_without_overwriting_correlation(self) -> None:
        mock_registry = MagicMock()
        feature_store = cast(Any, MagicMock())
        model_store = cast(Any, MagicMock())
        strategy_store = cast(Any, MagicMock())
        capture = CapturePublisher()
        store = DataStore(
            connection_string="sqlite:///:memory:",
            registry=RegistryMockAdapter(mock_registry),
            feature_store=feature_store,
            model_store=model_store,
            strategy_store=strategy_store,
            earnings_store=DummyEarningsStore(),
            publisher=capture,
            enable_publishing=True,
        )

        store.emit_event(
            dataset_id="predictions",
            instrument_id="BTCUSDT.BINANCE",
            stage=Stage.PREDICTION_EMITTED,
            source="unknown",  # should normalize to 'live'
            run_id="run_abc",
            ts_min=0,
            ts_max=10,
            count=1,
            status="partial",
            metadata={"custom": 1, "correlation_id": "SHOULD_NOT_WIN"},
        )

        assert mock_registry.emit_event.call_count == 1
        kwargs = mock_registry.emit_event.call_args.kwargs
        assert kwargs["source"] == "live"
        meta = cast(dict[str, Any], kwargs["metadata"])
        assert meta["custom"] == 1
        # Phase 0 behavior: correlation_id is enforced and cannot be overridden by metadata
        assert meta["correlation_id"] != "SHOULD_NOT_WIN"
        assert len(capture.calls) == 1
        topic, payload = capture.calls[0]
        assert topic.startswith("ml.models.created.")
        assert payload["metadata"]["custom"] == 1
        assert payload["metadata"]["correlation_id"] != "SHOULD_NOT_WIN"
