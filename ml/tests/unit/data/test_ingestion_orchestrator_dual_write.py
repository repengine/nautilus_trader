from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest
import pandas as pd

# Skip if optional low-level dependencies are not present
pytest.importorskip("msgspec")

from ml.config.events import EventStatus, Source, Stage
from ml.registry.dataclasses import DataContract, DatasetManifest, DatasetType, StorageKind
from ml.registry.protocols import RegistryProtocol
from ml.ml_types import DataFrameLike
from ml.stores.protocols import CoverageProviderProtocol, MarketDataWriterProtocol
from ml.stores.io_raw import RawIngestionWriterProtocol


class _FakeCoverage(CoverageProviderProtocol):
    def read_bucket_coverage(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> set[int]:
        # No coverage to force a single gap
        return set()


class _FakeWriter(MarketDataWriterProtocol):
    def __init__(self) -> None:
        self.calls: int = 0

    def write(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        df: pd.DataFrame,
    ) -> int:
        self.calls += 1
        return len(df.index)


class _FakeRaw(RawIngestionWriterProtocol):
    def __init__(self) -> None:
        self.calls: int = 0
        self.last_type: DatasetType | None = None

    def write(
        self,
        *,
        dataset_type: DatasetType,
        data: DataFrameLike | list[dict[str, object]],
    ) -> int:
        self.calls += 1
        self.last_type = dataset_type
        if isinstance(data, list):
            return len(data)
        if isinstance(data, pd.DataFrame):
            return len(data.index)
        # Polars DataFrame or other DataFrameLike: fall back to len()
        return len(data)


class _FakeRegistry(RegistryProtocol):
    def __init__(self) -> None:
        self.events: list[tuple[str, Stage]] = []

    # Minimal implementations for orchestrator
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
        self.events.append((dataset_id, stage))

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
        # Not used by test
        return DatasetManifest(
            dataset_id=dataset_id,
            dataset_type=DatasetType.BARS,
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
        return cast(DataContract, object())

    def register_dataset(self, manifest: DatasetManifest) -> str:
        return manifest.dataset_id


def _make_fake_ingestor():
    from ml.data.ingest.resume import DatabentoIngestor, IngestState

    class _FakeIngestor(DatabentoIngestor):
        def __init__(self) -> None:
            from ml.data.ingest.resume import BackoffPolicy, DatabentoLikeClient, SleepFn

            self.client = cast(DatabentoLikeClient, object())
            self.policy = BackoffPolicy(max_attempts=1)
            self.sleep_fn = cast(SleepFn | None, None)

        def ingest_time_window(
            self,
            *,
            dataset: str,
            schema: str,
            instrument: str,
            start_ns: int,
            end_ns: int,
            source: str = "historical",
            state: IngestState | None = None,
        ) -> pd.DataFrame:
            return pd.DataFrame({"ts_event": [start_ns + 1]})

    return _FakeIngestor()


def test_dual_write_invokes_both_sinks() -> None:
    try:
        from ml.data.ingest.orchestrator import IngestionOrchestrator
    except Exception:  # pragma: no cover - environment optional deps
        pytest.skip("ml.data.* optional dependencies not installed; skipping dual-write test")

    cov = _FakeCoverage()
    sql_writer = _FakeWriter()
    reg = _FakeRegistry()
    ing = _make_fake_ingestor()
    raw = _FakeRaw()

    class _Loader:
        def load(
            self,
            *,
            dataset_id: str,
            schema: str,
            instrument_id: str,
            start_ns: int,
            end_ns: int,
        ) -> list[object]:
            return [object()]

    orch = IngestionOrchestrator(
        coverage=cov,
        writer=sql_writer,
        registry=reg,
        ingestor=ing,
        raw_writer=raw,
        domain_loader=_Loader(),
    )

    gaps = orch.backfill_gaps(
        dataset_id="EQUS.MINI",
        schema="bars",
        instrument_id="SPY.XNAS",
        lookback_days=0,
        state=None,
    )

    assert len(gaps) >= 1
    assert sql_writer.calls == 1
    assert raw.calls == 1
    assert raw.last_type == DatasetType.BARS
    assert reg.events and reg.events[0][1] == Stage.DATA_INGESTED

def test_backfill_handles_timestamp_ts_event() -> None:
    try:
        from ml.data.ingest.orchestrator import IngestionOrchestrator
    except Exception:  # pragma: no cover - optional dependency guard
        pytest.skip("ml.data.* optional dependencies not installed; skipping timestamp test")

    cov = _FakeCoverage()
    sql_writer = _FakeWriter()
    reg = _FakeRegistry()

    from ml.data.ingest.resume import DatabentoIngestor, IngestState

    class _TimestampIngestor(DatabentoIngestor):
        def __init__(self) -> None:
            from ml.data.ingest.resume import BackoffPolicy, DatabentoLikeClient, SleepFn

            self.client = cast(DatabentoLikeClient, object())
            self.policy = BackoffPolicy(max_attempts=1)
            self.sleep_fn = cast(SleepFn | None, None)

        def ingest_time_window(
            self,
            *,
            dataset: str,
            schema: str,
            instrument: str,
            start_ns: int,
            end_ns: int,
            source: str = "historical",
            state: IngestState | None = None,
        ) -> pd.DataFrame:
            return pd.DataFrame({"ts_event": [pd.Timestamp("2025-01-01T00:00:00Z")]})

    orch = IngestionOrchestrator(
        coverage=cov,
        writer=sql_writer,
        registry=reg,
        ingestor=_TimestampIngestor(),
        raw_writer=None,
    )

    gaps = orch.backfill_gaps(
        dataset_id="EQUS.MINI",
        schema="tbbo",
        instrument_id="SPY.XNAS",
        lookback_days=0,
        state=None,
    )

    assert len(gaps) >= 1
    assert sql_writer.calls == 1
