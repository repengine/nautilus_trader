from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from ml.core.integration import MLIntegrationManager
from ml.preprocessing.event_ingestion import EventIngestionConfig


def test_ingest_events_creates_artifact(tmp_path: Path) -> None:
    mgr = object.__new__(MLIntegrationManager)
    cfg = EventIngestionConfig(
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 31, tzinfo=UTC),
        out_dir=tmp_path / "events",
    )

    result = MLIntegrationManager.ingest_events(mgr, cfg)

    assert result.exists()
    assert result.name == "events.parquet"
    assert result.parent == tmp_path / "events"


def test_ingest_events_uses_data_store_when_provided(tmp_path: Path) -> None:
    class _StubStore:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def write_ingestion(
            self,
            *,
            dataset_id: str,
            records: object,
            source: str,
            run_id: str,
            instrument_id: str | None = None,
        ) -> None:
            self.calls.append(
                {
                    "dataset_id": dataset_id,
                    "records": records,
                    "source": source,
                    "run_id": run_id,
                    "instrument_id": instrument_id,
                },
            )

    mgr = object.__new__(MLIntegrationManager)
    setattr(mgr, "data_store", _StubStore())
    economic_stub = tmp_path / "economic.csv"
    economic_stub.write_text(
        "timestamp,event_type,name,importance\n"
        "2024-02-15T13:30:00,economic_release,Retail Sales,HIGH\n",
        encoding="utf-8",
    )
    cfg = EventIngestionConfig(
        start=datetime(2024, 2, 15, tzinfo=UTC),
        end=datetime(2024, 2, 16, tzinfo=UTC),
        out_dir=tmp_path / "events",
        economic_stub_path=economic_stub,
        include_options_expiry=False,
    )

    result = MLIntegrationManager.ingest_events(mgr, cfg)

    assert result.exists()
    store = cast(_StubStore, getattr(mgr, "data_store"))
    assert store.calls
