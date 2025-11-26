#!/usr/bin/env python3

"""
Tests for the EarningsParquetRawWriter helper.
"""

from __future__ import annotations

from pathlib import Path

from ml.registry.dataclasses import DatasetType
from ml.stores.earnings_raw_writer import EarningsParquetRawWriter


def test_earnings_parquet_raw_writer_writes_parquet(tmp_path: Path) -> None:
    writer = EarningsParquetRawWriter(base_path=tmp_path)
    payload = [
        {
            "ticker": "AAPL",
            "period_end": "2024-03-31",
            "filing_date": "2024-05-02",
            "ts_event": 1,
            "ts_init": 2,
            "eps_diluted": 1.52,
            "data_source": "EDGAR",
        },
    ]
    written = writer.write(dataset_type=DatasetType.EARNINGS_ACTUALS, data=payload)
    assert written == 1
    dataset_dir = tmp_path / DatasetType.EARNINGS_ACTUALS.value
    assert dataset_dir.exists()
    files = list(dataset_dir.rglob("*.parquet"))
    assert len(files) == 1


def test_earnings_parquet_raw_writer_rejects_unsupported(tmp_path: Path) -> None:
    writer = EarningsParquetRawWriter(base_path=tmp_path)
    try:
        writer.write(dataset_type=DatasetType.BARS, data=[])
    except ValueError as exc:
        assert "only supports earnings datasets" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ValueError for unsupported dataset type")
