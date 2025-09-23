from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ml.config.events import EventStatus
from ml.config.events import Source

import pandas as pd

from ml.config.events import Stage
from ml.data.ingest.orchestrator import IngestionOrchestrator
from ml.data.ingest.resume import DatabentoIngestor, IngestState
from ml.stores.protocols import CoverageProviderProtocol, MarketDataWriterProtocol

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
