from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

from ml.common.message_bus import MessagePublisherProtocol
from ml.config.events import Stage
from ml.stores.data_store import DataStore
from ml.registry.dataclasses import DataContract, DatasetManifest, DatasetType, StorageKind
from ml.registry.protocols import RegistryProtocol


@contextmanager
def env(vars: dict[str, str]) -> Iterator[None]:
    old = {k: os.environ.get(k) for k in vars}
    try:
        os.environ.update(vars)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class StubRegistry(RegistryProtocol):
    def emit_event(
        self,
        dataset_id: str,
        instrument_id: str,
        stage: Stage,
        source: str,
        run_id: str,
        ts_min: int,
        ts_max: int,
        count: int,
        status: str,
        error: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        return None

    def update_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: str,
        last_success_ns: int,
        count: int,
        completeness_pct: float,
    ) -> None:
        return None

    # Protocol methods unused in this test
    def get_manifest(self, dataset_id: str) -> DatasetManifest:
        return DatasetManifest(
            dataset_id=dataset_id,
            dataset_type=DatasetType.FEATURES,
            storage_kind=StorageKind.POSTGRES,
            location="",
            partitioning={},
            retention_days=1,
            schema={"ts_event": "int64"},
            ts_field="ts_event",
            seq_field=None,
            primary_keys=["ts_event"],
            schema_hash="",
            constraints={},
            lineage=[],
            pipeline_signature="test",
            version="1.0.0",
            metadata={},
        )

    def get_contract(self, dataset_id: str) -> DataContract:
        return DataContract(
            contract_id=f"contract-{dataset_id}",
            dataset_id=dataset_id,
            version="1.0.0",
            validation_rules=[],
        )

    def register_dataset(self, manifest: DatasetManifest) -> str:
        return manifest.dataset_id


class CapturePublisher(MessagePublisherProtocol):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        self.calls.append((topic, payload))
        return True


from pathlib import Path


def test_data_store_stage_first_topics(tmp_path: Path) -> None:
    pub = CapturePublisher()
    with env({"ML_BUS_SCHEME": "stage_first", "ML_BUS_ENABLE": "1"}):
        store = DataStore(
            connection_string=f"sqlite:///{tmp_path}/ds.db",
            registry=StubRegistry(),
            publisher=pub,
            enable_publishing=True,
        )
        store.emit_event(
            dataset_id="features",
            instrument_id="EURUSD.SIM",
            stage=Stage.FEATURE_COMPUTED,
            source="historical",
            run_id="r1",
            ts_min=1,
            ts_max=2,
            count=1,
        )
        assert pub.calls, "Publisher should be called when enabled"
        topic, payload = pub.calls[-1]
        assert topic.startswith("events.ml.FEATURE_COMPUTED."), topic
        assert payload["stage"] == Stage.FEATURE_COMPUTED.value
