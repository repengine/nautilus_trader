from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pandas")

import pandas as pd

from ml.registry.dataclasses import DatasetType
from ml.stores.feature_raw_writer import CompositeRawIngestionWriter
from ml.stores.feature_raw_writer import FeatureDatasetParquetRawWriter
from ml.stores.feature_raw_writer import FeatureValuesParquetMirrorWriter
from ml.stores.raw_protocols import RawIngestionWriterProtocol


def test_feature_raw_writer_persists_events(tmp_path: Path) -> None:
    writer = FeatureDatasetParquetRawWriter(
        events_path=tmp_path / "events.parquet",
        micro_base_dir=tmp_path / "micro",
        l2_base_dir=tmp_path / "l2",
    )
    frame = pd.DataFrame(
        [
            {
                "event_timestamp": datetime(2025, 1, 5, tzinfo=UTC),
                "event_type": "fed_meeting",
                "name": "FOMC",
                "instrument_id": "",
            },
            {
                "event_timestamp": datetime(2025, 1, 6, tzinfo=UTC),
                "event_type": "earnings",
                "name": "AAPL call",
                "instrument_id": "AAPL.XNAS",
            },
        ],
    )

    written = writer.write(dataset_type=DatasetType.EVENTS_CALENDAR, data=frame)

    assert written == 2
    stored = pd.read_parquet(tmp_path / "events.parquet")
    assert len(stored) == 2
    assert set(stored["event_type"]) == {"fed_meeting", "earnings"}


def test_feature_raw_writer_partitions_micro_rows(tmp_path: Path) -> None:
    writer = FeatureDatasetParquetRawWriter(
        events_path=tmp_path / "events.parquet",
        micro_base_dir=tmp_path / "micro",
        l2_base_dir=tmp_path / "l2",
    )
    ts = int(datetime(2025, 3, 15, tzinfo=UTC).timestamp() * 1_000_000_000)
    frame = pd.DataFrame(
        [
            {
                "instrument_id": "SPY.XNAS",
                "timestamp": ts,
                "midprice": 1.0,
                "spread_bps": 2.0,
            },
        ],
    )

    written = writer.write(dataset_type=DatasetType.MICRO_MINUTE_FEATURES, data=frame)

    assert written == 1
    expected = tmp_path / "micro" / "SPY" / "year=2025" / "month=03" / "day=15.parquet"
    assert expected.exists()
    stored = pd.read_parquet(expected)
    assert len(stored) == 1
    assert stored["instrument_id"].iloc[0] == "SPY.XNAS"


def test_feature_values_mirror_writer_partitions(tmp_path: Path) -> None:
    writer = FeatureValuesParquetMirrorWriter(base_dir=tmp_path)
    ts_event = int(datetime(2025, 1, 5, tzinfo=UTC).timestamp() * 1_000_000_000)
    rows = [
        {
            "feature_set_id": "snapshot",
            "instrument_id": "AAPL",
            "ts_event": ts_event,
            "values": {"alpha": 1.5, "beta": 2.0},
        },
    ]

    written = writer.write_rows(rows)

    assert written == 1
    expected = tmp_path / "AAPL" / "year=2025" / "month=01" / "day=05.parquet"
    assert expected.exists()
    stored = pd.read_parquet(expected)
    assert len(stored) == 1
    payload = stored.iloc[0]
    assert payload["feature_set_id"] == "snapshot"
    assert payload["instrument_id"] == "AAPL"


class _FailingWriter(RawIngestionWriterProtocol):
    def write(self, *, dataset_type: DatasetType, data: Any) -> int:  # noqa: D401
        raise ValueError(f"{dataset_type.value} unsupported")


def test_composite_writer_swallows_unsupported_dataset(tmp_path: Path) -> None:
    composite = CompositeRawIngestionWriter((_FailingWriter(),))

    written = composite.write(dataset_type=DatasetType.EVENTS_CALENDAR, data=[])

    assert written == 0
