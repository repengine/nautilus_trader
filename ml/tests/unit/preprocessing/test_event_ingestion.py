from __future__ import annotations

from datetime import UTC
from datetime import datetime
from pathlib import Path

import polars as pl

from ml.preprocessing.event_ingestion import EventIngestionConfig
from ml.preprocessing.event_ingestion import EventIngestionUtility
from ml.config.dataset_ids import EVENTS_CALENDAR_DATASET_ID
from ml.data.coverage.types import GLOBAL_ENTITY_ID


def _write_stub_csv(path: Path, rows: list[dict[str, object]]) -> None:
    df = pl.DataFrame(rows)
    df.write_csv(path)


def _write_release_calendar(path: Path) -> None:
    df = pl.DataFrame(
        {
            "series_id": ["CPI"],
            "observation_ts": [datetime(2024, 2, 1, tzinfo=UTC)],
            "value": [3.1],
            "release_ts": [datetime(2024, 2, 13, 13, 30, tzinfo=UTC)],
            "release_end_ts": [datetime(2024, 3, 13, 13, 30, tzinfo=UTC)],
        },
    )
    df.write_parquet(path)


def test_event_ingestion_creates_normalized_events(tmp_path: Path) -> None:
    economic_stub = tmp_path / "economic.csv"
    _write_stub_csv(
        economic_stub,
        [
            {
                "timestamp": "2024-02-15T13:30:00",
                "event_type": "economic_release",
                "name": "Retail Sales",
                "importance": "HIGH",
            },
        ],
    )

    corporate_stub = tmp_path / "corporate.csv"
    _write_stub_csv(
        corporate_stub,
        [
            {
                "timestamp": "2024-04-22T21:30:00",
                "event_type": "earnings",
                "name": "AAPL Q1",
                "instrument_id": "AAPL",
                "importance": "HIGH",
                "quarter": "Q1",
                "year": 2024,
            },
        ],
    )

    alfred_dir = tmp_path / "alfred"
    series_dir = alfred_dir / "CPI"
    series_dir.mkdir(parents=True, exist_ok=True)
    _write_release_calendar(series_dir / "release_calendar.parquet")

    cfg = EventIngestionConfig(
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 6, 30, tzinfo=UTC),
        out_dir=tmp_path / "events",
        alfred_vintage_dir=alfred_dir,
        economic_series=("CPI",),
        economic_stub_path=economic_stub,
        corporate_source_path=corporate_stub,
    )

    utility = EventIngestionUtility(cfg)
    events_path = utility.ingest()

    assert events_path.exists()
    events_df = pl.read_parquet(events_path)
    event_types = set(events_df.get_column("event_type").to_list())
    assert "fed_meeting" in event_types
    assert "economic_release" in event_types
    assert "earnings" in event_types
    assert "options_expiry" in event_types
    assert "holiday" in event_types

    global_events = events_df.filter(
        pl.col("event_type").is_in(["fed_meeting", "economic_release", "options_expiry", "holiday"]),
    )
    assert not global_events.is_empty()
    assert (
        global_events.get_column("instrument_id")
        .unique()
        .to_list()
        == [GLOBAL_ENTITY_ID]
    )

    # Ensure metadata for corporate event preserved
    earnings_rows = events_df.filter(pl.col("event_type") == "earnings")
    assert not earnings_rows.is_empty()


def test_event_ingestion_writes_sql_when_data_store_provided(tmp_path: Path) -> None:
    economic_stub = tmp_path / "economic.csv"
    _write_stub_csv(
        economic_stub,
        [
            {
                "timestamp": "2024-02-15T13:30:00",
                "event_type": "economic_release",
                "name": "Retail Sales",
                "importance": "HIGH",
            },
        ],
    )
    cfg = EventIngestionConfig(
        start=datetime(2024, 2, 15, tzinfo=UTC),
        end=datetime(2024, 2, 16, tzinfo=UTC),
        out_dir=tmp_path / "events",
        economic_stub_path=economic_stub,
        include_options_expiry=False,
    )

    class _StubStore:
        def __init__(self) -> None:
            self.calls: list[tuple[str, pl.DataFrame]] = []

        def write_ingestion(self, *, dataset_id: str, records: pl.DataFrame, source: str, run_id: str, instrument_id: str | None = None) -> None:  # type: ignore[override]
            self.calls.append((dataset_id, records))

    stub = _StubStore()
    utility = EventIngestionUtility(cfg, data_store=stub, ingest_run_id="event_test")
    path = utility.ingest()

    assert path.exists()
    assert stub.calls
    dataset_id, records = stub.calls[0]
    assert dataset_id == EVENTS_CALENDAR_DATASET_ID
    assert "ts_event" in records.columns
    assert (
        records.get_column("instrument_id").unique().to_list()
        == [GLOBAL_ENTITY_ID]
    )
