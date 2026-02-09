from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from types import SimpleNamespace
from typing import Any
from typing import cast

import pandas as pd
import pytest
from sqlalchemy.exc import IntegrityError

from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.data.ingest.market_bindings import ResolvedMarketBinding
from ml.data.ingest.orchestrator import DAY_NS
from ml.data.ingest.orchestrator import IngestionOrchestrator
from ml.data.ingest.resume import DatabentoIngestor
from ml.data.ingest.resume import IngestState
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import StorageKind
from ml.stores.protocols import CoverageProviderProtocol
from ml.stores.protocols import MarketDataWriterProtocol


pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)


class _Coverage(CoverageProviderProtocol):
    def read_bucket_coverage(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> set[int]:
        del dataset_id, schema, instrument_id, start_ns, end_ns
        return set()


class _Writer(MarketDataWriterProtocol):
    def __init__(self) -> None:
        self.frames: list[pd.DataFrame] = []

    def write(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        df: pd.DataFrame,
    ) -> int:
        del dataset_id, schema, instrument_id
        self.frames.append(df.copy())
        return len(df.index)


class _Registry:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.watermark_calls: int = 0

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
        del error
        self.events.append(
            {
                "dataset_id": dataset_id,
                "instrument_id": instrument_id,
                "stage": stage,
                "source": source,
                "run_id": run_id,
                "ts_min": ts_min,
                "ts_max": ts_max,
                "count": count,
                "status": status,
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
        del dataset_id, instrument_id, source, last_success_ns, count, completeness_pct
        self.watermark_calls += 1
        raise IntegrityError("UPDATE", {}, RuntimeError("missing dataset"))

    def get_manifest(self, dataset_id: str) -> Any:
        del dataset_id
        raise RuntimeError("manifest not required")

    def get_contract(self, dataset_id: str) -> Any:
        del dataset_id
        raise RuntimeError("contract not required")

    def register_dataset(self, manifest: Any) -> str:
        del manifest
        raise RuntimeError("register not required")

    def update_manifest(self, dataset_id: str, changes: dict[str, object]) -> None:
        del dataset_id, changes


class _TimestampIngestor(DatabentoIngestor):
    def __init__(self) -> None:
        from ml.data.ingest.resume import BackoffPolicy
        from ml.data.ingest.resume import DatabentoLikeClient
        from ml.data.ingest.resume import SleepFn

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
        del dataset, schema, instrument, start_ns, end_ns, source, state
        return pd.DataFrame(
            {
                "ts_event": [pd.Timestamp("2025-01-01T00:00:00Z")],
                "source_dataset": ["EQUS.MINI"],
            },
        )


@dataclass(slots=True)
class _ServiceStub:
    frame: pd.DataFrame

    def get_available_range_ns(
        self,
        *,
        dataset: str,
        schema: str,
    ) -> tuple[int | None, int | None]:
        del dataset, schema
        return (None, None)

    def ingest(self, request: object, on_chunk: Any) -> None:
        del request
        from ml.data.ingest.service import IngestionChunk
        from ml.data.ingest.service import IngestionWindow

        chunk = IngestionChunk(
            symbol="SPY",
            window=IngestionWindow(
                start=datetime(2025, 1, 1, tzinfo=UTC),
                end=datetime(2025, 1, 2, tzinfo=UTC),
            ),
            frame=self.frame,
        )
        on_chunk(chunk)


def test_backfill_gaps_emits_metadata_and_handles_integrity_error(monkeypatch: pytest.MonkeyPatch) -> None:
    base_now = 100 * DAY_NS
    monkeypatch.setattr("ml.data.ingest.orchestrator._utc_now_ns", lambda: base_now)

    orch = IngestionOrchestrator(
        coverage=_Coverage(),
        writer=_Writer(),
        registry=cast(Any, _Registry()),
        ingestor=_TimestampIngestor(),
    )

    result = orch.backfill_gaps(
        dataset_id="EQUS.MINI",
        schema="tbbo",
        instrument_id="SPY.XNAS",
        lookback_days=1,
    )

    assert result.persisted_window_count == 1
    registry = cast(_Registry, orch.registry)
    assert registry.events
    metadata = registry.events[0]["metadata"]
    assert metadata == {"source_datasets": ["EQUS.MINI"]}
    assert registry.watermark_calls == 1


def test_backfill_gaps_service_handles_empty_and_missing_ts_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_now = 120 * DAY_NS
    monkeypatch.setattr("ml.data.ingest.orchestrator._utc_now_ns", lambda: base_now)

    writer = _Writer()
    registry = cast(Any, _Registry())

    orch_empty = IngestionOrchestrator(
        coverage=_Coverage(),
        writer=writer,
        registry=registry,
        ingestor=_TimestampIngestor(),
        service=cast(Any, _ServiceStub(frame=pd.DataFrame())),
    )
    empty_result = orch_empty.backfill_gaps(
        dataset_id="EQUS.MINI",
        schema="bars",
        instrument_id="SPY.XNAS",
        lookback_days=1,
    )
    assert empty_result.persisted_window_count == 0

    orch_missing = IngestionOrchestrator(
        coverage=_Coverage(),
        writer=writer,
        registry=registry,
        ingestor=_TimestampIngestor(),
        service=cast(Any, _ServiceStub(frame=pd.DataFrame({"price": [1.0]}))),
    )
    missing_result = orch_missing.backfill_gaps(
        dataset_id="EQUS.MINI",
        schema="bars",
        instrument_id="SPY.XNAS",
        lookback_days=1,
    )
    assert missing_result.persisted_window_count == 0


def test_clamp_window_outside_metadata_range_returns_none() -> None:
    orch = IngestionOrchestrator(
        coverage=_Coverage(),
        writer=_Writer(),
        registry=cast(Any, _Registry()),
        ingestor=_TimestampIngestor(),
        service=cast(Any, _ServiceStub(frame=pd.DataFrame())),
    )

    orch.service = cast(
        Any,
        SimpleServiceRange(
            start_ns=10 * DAY_NS,
            end_ns=11 * DAY_NS,
        ),
    )

    clamped = orch._clamp_window_to_available_range(
        provider_dataset_id="EQUS.MINI",
        provider_schema="bars",
        start_ns=1,
        end_ns=DAY_NS,
    )
    assert clamped is None


@dataclass(slots=True)
class SimpleServiceRange:
    start_ns: int | None
    end_ns: int | None

    def get_available_range_ns(
        self,
        *,
        dataset: str,
        schema: str,
    ) -> tuple[int | None, int | None]:
        del dataset, schema
        return (self.start_ns, self.end_ns)


def test_normalize_time_columns_handles_named_ts_index() -> None:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-01-01T00:00:00Z"),
            pd.Timestamp("2025-01-01T00:00:01Z"),
        ],
        name="ts",
    )
    frame = pd.DataFrame({"price": [1.0, 2.0]}, index=index)

    normalized = IngestionOrchestrator._normalize_time_columns(frame)

    assert "ts_event" in normalized.columns
    assert "ts_init" in normalized.columns
    assert pd.api.types.is_integer_dtype(normalized["ts_event"])
    assert pd.api.types.is_integer_dtype(normalized["ts_init"])


def test_backfill_binding_requires_schema() -> None:
    orch = IngestionOrchestrator(
        coverage=_Coverage(),
        writer=_Writer(),
        registry=cast(Any, _Registry()),
        ingestor=_TimestampIngestor(),
    )

    binding = ResolvedMarketBinding(
        binding_id="bad-binding",
        symbol="SPY",
        instrument_ids=("SPY.XNAS",),
        dataset_id="EQUS.MINI",
        descriptor_id="EQUS.MINI",
        schema="",
        storage_kind=StorageKind.POSTGRES,
        license_start=None,
        license_end=None,
        start=None,
        end=None,
        source="descriptor",
    )

    with pytest.raises(ValueError, match="missing schema"):
        orch.backfill_binding(binding=binding, lookback_days=1)


def test_backfill_window_helpers_cover_schema_and_chunk_branches() -> None:
    from ml.data.ingest.orchestrator import _max_chunk_days_for_schema
    from ml.data.ingest.orchestrator import _split_into_chunks

    assert _max_chunk_days_for_schema("mbp-10") == 31
    assert _max_chunk_days_for_schema("tbbo") == 365
    assert _max_chunk_days_for_schema("trades") == 365
    assert _max_chunk_days_for_schema("ohlcv-1m") == 1095
    assert _max_chunk_days_for_schema("other") == 365

    assert _split_into_chunks(start_ns=10, end_ns=10, max_days=1) == ()
    chunks = _split_into_chunks(start_ns=0, end_ns=2 * DAY_NS + 1, max_days=1)
    assert len(chunks) == 3


def test_schema_to_dataset_type_mapping() -> None:
    from ml.data.ingest.orchestrator import _schema_to_dataset_type

    assert _schema_to_dataset_type("bars") == DatasetType.BARS


def test_backfill_gaps_service_and_ingestor_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    base_now = 140 * DAY_NS
    monkeypatch.setattr("ml.data.ingest.orchestrator._utc_now_ns", lambda: base_now)

    class _FailingService:
        def ingest(self, request: object, on_chunk: Any) -> None:
            del request, on_chunk
            raise RuntimeError("service failed")

    orch_service = IngestionOrchestrator(
        coverage=_Coverage(),
        writer=_Writer(),
        registry=cast(Any, _Registry()),
        ingestor=_TimestampIngestor(),
        service=cast(Any, _FailingService()),
    )
    with pytest.raises(RuntimeError, match="service failed"):
        orch_service.backfill_gaps(
            dataset_id="EQUS.MINI",
            schema="bars",
            instrument_id="SPY.XNAS",
            lookback_days=1,
        )

    class _FailingIngestor(_TimestampIngestor):
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
            del dataset, schema, instrument, start_ns, end_ns, source, state
            raise RuntimeError("ingestor failed")

    orch_ingestor = IngestionOrchestrator(
        coverage=_Coverage(),
        writer=_Writer(),
        registry=cast(Any, _Registry()),
        ingestor=_FailingIngestor(),
    )
    with pytest.raises(RuntimeError, match="ingestor failed"):
        orch_ingestor.backfill_gaps(
            dataset_id="EQUS.MINI",
            schema="bars",
            instrument_id="SPY.XNAS",
            lookback_days=1,
        )


def test_backfill_gaps_raw_writer_disable_and_error(monkeypatch: pytest.MonkeyPatch) -> None:
    base_now = 160 * DAY_NS
    monkeypatch.setattr("ml.data.ingest.orchestrator._utc_now_ns", lambda: base_now)

    class _DisableRaw:
        def __init__(self) -> None:
            self.calls = 0

        def is_enabled(self, dataset_type: DatasetType) -> bool:
            del dataset_type
            return False

        def write(self, *, dataset_type: DatasetType, data: object) -> int:
            del dataset_type, data
            self.calls += 1
            return 0

    raw_disable = _DisableRaw()
    orch_disable = IngestionOrchestrator(
        coverage=_Coverage(),
        writer=_Writer(),
        registry=cast(Any, _Registry()),
        ingestor=_TimestampIngestor(),
        raw_writer=cast(Any, raw_disable),
    )
    result_disable = orch_disable.backfill_gaps(
        dataset_id="EQUS.MINI",
        schema="bars",
        instrument_id="SPY.XNAS",
        lookback_days=1,
    )
    assert result_disable.persisted_window_count == 1
    assert raw_disable.calls == 0

    class _ErrorRaw:
        def is_enabled(self, dataset_type: DatasetType) -> bool:
            del dataset_type
            return True

        def write(self, *, dataset_type: DatasetType, data: object) -> int:
            del dataset_type, data
            raise RuntimeError("raw write failed")

    orch_error = IngestionOrchestrator(
        coverage=_Coverage(),
        writer=_Writer(),
        registry=cast(Any, _Registry()),
        ingestor=_TimestampIngestor(),
        raw_writer=cast(Any, _ErrorRaw()),
    )
    result_error = orch_error.backfill_gaps(
        dataset_id="EQUS.MINI",
        schema="bars",
        instrument_id="SPY.XNAS",
        lookback_days=1,
    )
    assert result_error.persisted_window_count == 1


def test_clamp_window_handles_missing_metadata_method() -> None:
    orch = IngestionOrchestrator(
        coverage=_Coverage(),
        writer=_Writer(),
        registry=cast(Any, _Registry()),
        ingestor=_TimestampIngestor(),
        service=cast(Any, object()),
    )
    clamped = orch._clamp_window_to_available_range(
        provider_dataset_id="EQUS.MINI",
        provider_schema="bars",
        start_ns=10,
        end_ns=100,
    )
    assert clamped == (10, 100)


def test_coerce_frame_to_manifest_covers_type_edge_cases() -> None:
    class _ManifestRegistry(_Registry):
        def get_manifest(self, dataset_id: str) -> object:
            del dataset_id
            return SimpleNamespace(
                schema={
                    "instrument_id": "str",
                    "price": "float64",
                    "size": "int64",
                    "is_valid": "bool",
                    "opaque": 123,
                    "missing_col": "str",
                },
            )

    orch = IngestionOrchestrator(
        coverage=_Coverage(),
        writer=_Writer(),
        registry=cast(Any, _ManifestRegistry()),
        ingestor=_TimestampIngestor(),
    )
    frame = pd.DataFrame(
        {
            "instrument_id": [1],
            "price": ["1.5"],
            "size": ["1"],
            "is_valid": [1],
            "opaque": ["abc"],
        },
    )
    coerced = orch._coerce_frame_to_manifest(
        dataset_id="EQUS.MINI",
        instrument_id="SPY.XNAS",
        frame=frame,
    )

    assert str(coerced["instrument_id"].dtype).lower() in {"object", "string[python]"}
    assert str(coerced["price"].dtype).lower() == "float64"
    assert "int" in str(coerced["size"].dtype).lower()
    assert str(coerced["is_valid"].dtype).lower() in {"bool", "boolean"}


def test_normalize_time_columns_numeric_and_ts_init_paths() -> None:
    frame = pd.DataFrame(
        {
            "ts_event": ["1700000000000000000", "1700000001000000000"],
            "ts_init": ["1700000000000000001", "1700000001000000001"],
        },
    )
    normalized = IngestionOrchestrator._normalize_time_columns(frame)
    assert pd.api.types.is_integer_dtype(normalized["ts_event"])
    assert pd.api.types.is_integer_dtype(normalized["ts_init"])

    frame_without_init = pd.DataFrame({"ts_event": [1700000000000000000, 1700000001000000000]})
    normalized_without_init = IngestionOrchestrator._normalize_time_columns(frame_without_init)
    assert normalized_without_init["ts_init"].tolist() == normalized_without_init["ts_event"].tolist()
