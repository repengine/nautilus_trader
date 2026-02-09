from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.data.ingest.market_bindings import ResolvedMarketBinding
from ml.data.ingest.orchestrator import BackfillWindowList
from ml.data.ingest.orchestrator import IngestionOrchestrator
from ml.data.ingest.orchestrator import execute_backfill_plan
from ml.data.ingest.resume import DatabentoIngestor
from ml.data.ingest.resume import IngestState
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import StorageKind
from ml.stores.protocols import CoverageProviderProtocol
from ml.stores.protocols import MarketDataWriterProtocol

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)

DAY_NS = 86_400_000_000_000

@dataclass
class _FixtureClient:
    def get_data(
        self,
        dataset: str,
        symbols: list[str],
        schema: str,
        start: Any,
        end: Any,
        **kwargs: Any,
    ) -> pd.DataFrame:
        # Return a simple frame covering the requested window at 3 timestamps
        start_ns = int(pd.Timestamp(start).value)
        end_ns = int(pd.Timestamp(end).value)
        ts = [start_ns, (start_ns + end_ns) // 2, end_ns - 1]
        return pd.DataFrame({"ts_event": ts, "instrument_id": [symbols[0]] * 3})

class _MemCoverage(CoverageProviderProtocol):
    def __init__(self, covered: set[int] | None = None) -> None:
        self.covered = set(covered or set())

    def add_bucket(self, b: int) -> None:
        self.covered.add(int(b))

    def read_bucket_coverage(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> set[int]:
        start_b = start_ns // DAY_NS
        end_b = end_ns // DAY_NS
        return {b for b in self.covered if start_b <= b <= end_b}

class _MemWriter(MarketDataWriterProtocol):
    def __init__(self) -> None:
        self.writes: list[pd.DataFrame] = []

    def write(self, *, dataset_id: str, schema: str, instrument_id: str, df: pd.DataFrame) -> int:
        self.writes.append(df.copy())
        return len(df.index)

class _MemRegistry:
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

    def update_manifest(self, dataset_id: str, changes: dict[str, object]) -> None:
        # Not required for this test scenario
        return None

    # Protocol read methods not used here
    def get_manifest(self, dataset_id: str) -> Any:  # pragma: no cover - unused in this test
        raise NotImplementedError

    def get_contract(self, dataset_id: str) -> Any:  # pragma: no cover - unused in this test
        raise NotImplementedError

    def register_dataset(self, manifest: Any) -> str:  # pragma: no cover - unused
        raise NotImplementedError

class _RegistryWithManifest(_MemRegistry):
    def __init__(self, manifest: DatasetManifest) -> None:
        super().__init__()
        self._manifest = manifest

    def get_manifest(self, dataset_id: str) -> DatasetManifest:
        if dataset_id != self._manifest.dataset_id:
            raise ValueError(dataset_id)
        return self._manifest

@dataclass
class _TrackingIngestor:
    calls: list[tuple[int, int]]

    def ingest_time_window(
        self,
        *,
        dataset: str,
        schema: str,
        instrument: str,
        start_ns: int,
        end_ns: int,
        source: str,
        state: IngestState | None,
    ) -> pd.DataFrame:
        self.calls.append((start_ns, end_ns))
        return pd.DataFrame(
            {
                "ts_event": [start_ns, end_ns - 1],
                "instrument_id": [instrument, instrument],
            },
        )

def test_backfill_gaps_detects_missing_days_and_writes() -> None:
    ingestor = DatabentoIngestor(client=_FixtureClient())
    today_bucket = int(pd.Timestamp.utcnow().value // DAY_NS)
    cov = _MemCoverage(covered={today_bucket})
    writer = _MemWriter()
    reg = _MemRegistry()
    orch = IngestionOrchestrator(coverage=cov, writer=writer, registry=reg, ingestor=ingestor)
    st = IngestState()

    gaps = orch.backfill_gaps(
        dataset_id="tbbo",
        schema="tbbo",
        instrument_id="EURUSD.SIM",
        lookback_days=2,
        state=st,
    )
    assert gaps  # at least yesterday
    # Ensure writer was called for each gap (non-empty)
    assert writer.writes
    # Registry recorded events and watermarks
    assert reg.events and reg.watermarks
    assert all(evt["stage"] == Stage.DATA_INGESTED.value for evt in reg.events)

def test_backfill_gaps_coalesces_long_runs() -> None:
    ingestor = _TrackingIngestor(calls=[])
    cov = _MemCoverage()
    writer = _MemWriter()
    reg = _MemRegistry()
    orch = IngestionOrchestrator(coverage=cov, writer=writer, registry=reg, ingestor=ingestor)

    gaps = orch.backfill_gaps(
        dataset_id="EQUS.MINI",
        schema="ohlcv-1m",
        instrument_id="AAPL.ARCX",
        lookback_days=10,
    )

    assert gaps
    assert len(ingestor.calls) == 1
    span_ns = ingestor.calls[0][1] - ingestor.calls[0][0]
    assert 9 * DAY_NS <= span_ns <= 11 * DAY_NS

def test_coerce_frame_to_manifest_casts_expected_types() -> None:
    manifest = DatasetManifest(
        dataset_id="EQUS.MINI",
        dataset_type=DatasetType.BARS,
        storage_kind=StorageKind.POSTGRES,
        location="sql",
        partitioning={},
        retention_days=365,
        schema={
            "ts_event": "int64",
            "ts_init": "int64",
            "instrument_id": "str",
            "rtype": "str",
            "publisher_id": "str",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
            "symbol": "str",
        },
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["instrument_id", "ts_event"],
        schema_hash="",
        constraints={},
        lineage=[],
        pipeline_signature="test_pipeline",
        version="1.0",
    )

    ingestor = DatabentoIngestor(client=_FixtureClient())
    cov = _MemCoverage()
    writer = _MemWriter()
    reg = _RegistryWithManifest(manifest)
    orch = IngestionOrchestrator(coverage=cov, writer=writer, registry=reg, ingestor=ingestor)

    frame = pd.DataFrame(
        {
            "ts_event": [1680000000000000000, 1680000060000000000],
            "ts_init": [1680000000000000000, 1680000060000000000],
            "instrument_id": ["SPY.ARCX", "SPY.ARCX"],
            "rtype": [33, 33],
            "publisher_id": [95, 95],
            "open": [395.92, 395.93],
            "high": [396.00, 396.10],
            "low": [395.80, 395.85],
            "close": [395.95, 396.05],
            "volume": [955, 450],
            "symbol": ["SPY", "SPY"],
        },
    )

    coerced = orch._coerce_frame_to_manifest(
        dataset_id="EQUS.MINI",
        instrument_id="SPY.XNAS",
        frame=frame,
    )

    assert str(coerced["publisher_id"].dtype).lower() in {"object", "string[python]"}
    assert str(coerced["rtype"].dtype).lower() in {"object", "string[python]"}
    assert str(coerced["volume"].dtype).lower() == "float64"
    assert str(coerced["ts_event"].dtype).lower() == "int64"
    assert all(coerced["instrument_id"] == "SPY.XNAS")

def test_backfill_binding_uses_binding_and_logs_sql_warning(monkeypatch, caplog) -> None:
    ingestor = DatabentoIngestor(client=_FixtureClient())
    cov = _MemCoverage()
    writer = _MemWriter()
    reg = _MemRegistry()
    orch = IngestionOrchestrator(coverage=cov, writer=writer, registry=reg, ingestor=ingestor)

    binding = ResolvedMarketBinding(
        binding_id="binding-001",
        symbol="EURUSD",
        instrument_ids=("EURUSD.SIM",),
        dataset_id="EQUS.MINI",
        descriptor_id="EQUS.MINI",
        schema="tbbo",
        storage_kind=StorageKind.POSTGRES,
        license_start=None,
        license_end=None,
        start=None,
        end=None,
        source="descriptor",
    )

    calls: list[tuple[str, str, str, int]] = []

    def _fake_backfill(
        self,
        *,
        dataset_id: str,
        provider_dataset_id: str | None = None,
        provider_schema: str | None = None,
        schema: str,
        instrument_id: str,
        lookback_days: int,
        state: IngestState | None = None,
        symbol_hint: str | None = None,
    ) -> BackfillWindowList:
        calls.append((dataset_id, schema, instrument_id, lookback_days, provider_schema))
        return BackfillWindowList(
            ((0, DAY_NS),),
            requested=((0, DAY_NS),),
            frames_written=1,
            rows_written=1,
        )

    monkeypatch.setattr(IngestionOrchestrator, "backfill_gaps", _fake_backfill)

    caplog.set_level("WARNING")
    results = orch.backfill_binding(binding=binding, lookback_days=3)

    assert calls == [("EQUS.MINI", "tbbo", "EURUSD.SIM", 3, "tbbo")]
    assert results.get("EURUSD.SIM") == [(0, DAY_NS)]
    assert any("not SQL-backed" in record.message for record in caplog.records)

def test_backfill_binding_warns_on_legacy_source(monkeypatch, caplog) -> None:
    ingestor = DatabentoIngestor(client=_FixtureClient())
    cov = _MemCoverage()
    writer = _MemWriter()
    reg = _MemRegistry()
    orch = IngestionOrchestrator(coverage=cov, writer=writer, registry=reg, ingestor=ingestor)

    binding = ResolvedMarketBinding(
        binding_id="legacy-001",
        symbol="SPY",
        instrument_ids=("SPY.XNAS",),
        dataset_id="LEGACY.BARS",
        descriptor_id=None,
        schema="ohlcv-1m",
        storage_kind=None,
        license_start=None,
        license_end=None,
        start=None,
        end=None,
        source="legacy",
    )

    monkeypatch.setattr(
        IngestionOrchestrator,
        "backfill_gaps",
        lambda self, **kwargs: BackfillWindowList(
            ((0, DAY_NS),),
            requested=((0, DAY_NS),),
            frames_written=1,
            rows_written=1,
        ),
    )

    caplog.set_level("WARNING")
    orch.backfill_binding(binding=binding, lookback_days=1)

    assert any("fallback market binding" in record.message for record in caplog.records)

def test_normalize_time_columns_derives_ts_event_from_timestamp_column() -> None:
    ts = [
        pd.Timestamp("2024-01-01T09:30:00Z", tz=UTC),
        pd.Timestamp("2024-01-01T09:30:01Z", tz=UTC),
    ]
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "bid": [1.0, 1.1],
        },
    )

    normalized = IngestionOrchestrator._normalize_time_columns(frame)

    expected = [ts_value.value for ts_value in ts]
    assert normalized["ts_event"].tolist() == expected
    assert normalized["ts_init"].tolist() == expected

def test_normalize_time_columns_uses_ts_exchange_fallback() -> None:
    ts_exchange = [1_700_000_000_000_000_000, 1_700_000_100_000_000_000]
    frame = pd.DataFrame(
        {
            "ts_exchange": ts_exchange,
            "ask": [2.3, 2.4],
        },
    )

    normalized = IngestionOrchestrator._normalize_time_columns(frame)

    assert normalized["ts_event"].tolist() == ts_exchange
    assert normalized["ts_init"].tolist() == ts_exchange


def test_execute_backfill_plan_processes_each_binding_once(monkeypatch: pytest.MonkeyPatch) -> None:
    binding = ResolvedMarketBinding(
        binding_id="binding-spy",
        symbol="SPY",
        instrument_ids=("SPY.XNAS", "SPY.BATS"),
        dataset_id="EQUS.MINI",
        descriptor_id="EQUS.MINI",
        schema="ohlcv-1m",
        storage_kind=StorageKind.POSTGRES,
        license_start=None,
        license_end=None,
        start=None,
        end=None,
        source="descriptor",
    )

    monkeypatch.setattr(
        IngestionOrchestrator,
        "resolve_market_bindings",
        staticmethod(lambda **_: (binding,)),
    )

    orchestrator_mock = MagicMock()
    orchestrator_mock.backfill_binding.return_value = {
        "SPY.XNAS": BackfillWindowList(((1, 2),), requested=((1, 2),)),
        "SPY.BATS": BackfillWindowList(((3, 4),), requested=((3, 4),)),
    }
    messages: list[str] = []
    summary = execute_backfill_plan(
        orchestrator=orchestrator_mock,
        dataset_id="EQUS.MINI",
        schema="ohlcv-1m",
        instruments=("SPY.XNAS", "SPY.BATS"),
        lookback_days=5,
        binding_dataset_id="EQUS.BINDINGS",
        market_inputs=None,
        state=None,
        emit=messages.append,
    )

    assert summary.total_windows == 2
    assert summary.processed_bindings == ("binding-spy",)
    assert orchestrator_mock.backfill_binding.call_count == 1
    assert orchestrator_mock.backfill_gaps.call_count == 0
    assert "SPY.XNAS: planned 1 day window(s) via binding binding-spy" in messages
    assert "SPY.BATS: planned 1 day window(s) via binding binding-spy" in messages


def test_execute_backfill_plan_warns_when_binding_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        IngestionOrchestrator,
        "resolve_market_bindings",
        staticmethod(lambda **_: ()),
    )

    orchestrator_mock = MagicMock()
    orchestrator_mock.backfill_gaps.return_value = BackfillWindowList(
        ((1, 2), (2, 3)),
        requested=((1, 2), (2, 3)),
    )

    messages: list[str] = []
    summary = execute_backfill_plan(
        orchestrator=orchestrator_mock,
        dataset_id="EQUS.MINI",
        schema="ohlcv-1m",
        instruments=("QQQ.XNAS",),
        lookback_days=3,
        binding_dataset_id="EQUS.BINDINGS",
        market_inputs=None,
        state=None,
        emit=messages.append,
    )

    assert summary.total_windows == 2
    assert summary.processed_bindings == ()
    orchestrator_mock.backfill_gaps.assert_called_once_with(
        dataset_id="EQUS.MINI",
        schema="ohlcv-1m",
        instrument_id="QQQ.XNAS",
        lookback_days=3,
        state=None,
    )
    assert (
        "Warning: no binding resolved for QQQ.XNAS; using legacy dataset EQUS.MINI/ohlcv-1m"
        in messages
    )
    assert "QQQ.XNAS: planned 2 day window(s)" in messages
