#!/usr/bin/env python3
"""
Test for the plan-backfill CLI command.

This test verifies the plan_backfill function works correctly.

"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from ml.cli.coverage import plan_backfill
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig


@pytest.mark.slow
@pytest.mark.unit
def test_plan_backfill_with_gaps(tmp_path: Path) -> None:
    """
    Test plan_backfill identifies gaps correctly and creates job spec.
    """
    # Setup test registry data
    registry_path = tmp_path / "test_registry"
    registry_path.mkdir(parents=True, exist_ok=True)

    # Create mock registry data with gaps
    registry_data = {
        "manifests": {
            "bars_test": {
                "dataset_id": "bars_test",
                "dataset_type": "bars",
                "storage_kind": "parquet",
                "location": "/data/bars",
                "partitioning": {"by": "ts_event", "interval": "daily"},
                "retention_days": 90,
                "schema": {
                    "instrument_id": "string",
                    "ts_event": "int64",
                    "ts_init": "int64",
                    "open": "float64",
                },
                "ts_field": "ts_event",
                "seq_field": None,
                "primary_keys": ["instrument_id", "ts_event"],
                "schema_hash": "test_hash_1",
                "constraints": {},
                "lineage": [],
                "pipeline_signature": "test_pipeline",
                "version": "1.0.0",
                "created_at": int(datetime(2024, 1, 1).timestamp() * 1e9),
                "last_modified": int(datetime(2024, 1, 1).timestamp() * 1e9),
                "metadata": {"description": "Test BARS dataset"},
            },
            "mbp1_test": {
                "dataset_id": "mbp1_test",
                "dataset_type": "mbp1",
                "storage_kind": "parquet",
                "location": "/data/mbp1",
                "partitioning": {"by": "ts_event", "interval": "daily"},
                "retention_days": 90,
                "schema": {
                    "instrument_id": "string",
                    "ts_event": "int64",
                    "ts_init": "int64",
                    "bid": "float64",
                },
                "ts_field": "ts_event",
                "seq_field": None,
                "primary_keys": ["instrument_id", "ts_event"],
                "schema_hash": "test_hash_2",
                "constraints": {},
                "lineage": [],
                "pipeline_signature": "test_pipeline",
                "version": "1.0.0",
                "created_at": int(datetime(2024, 1, 1).timestamp() * 1e9),
                "last_modified": int(datetime(2024, 1, 1).timestamp() * 1e9),
                "metadata": {"description": "Test MBP1 dataset"},
            },
        },
        "events": [
            # BARS data exists for EUR/USD on 2024-01-15
            {
                "dataset_id": "bars_test",
                "instrument_id": "EUR/USD",
                "stage": "CATALOG_WRITTEN",
                "source": "historical",
                "run_id": "test_run_1",
                "ts_event": int(datetime(2024, 1, 15, 10, 0).timestamp() * 1e9),
                "ts_min": int(datetime(2024, 1, 15, 0, 0).timestamp() * 1e9),
                "ts_max": int(datetime(2024, 1, 15, 23, 59).timestamp() * 1e9),
                "count": 1000,
                "status": "success",
            },
            # BARS data exists for GBP/USD on 2024-01-15
            {
                "dataset_id": "bars_test",
                "instrument_id": "GBP/USD",
                "stage": "CATALOG_WRITTEN",
                "source": "historical",
                "run_id": "test_run_2",
                "ts_event": int(datetime(2024, 1, 15, 10, 0).timestamp() * 1e9),
                "ts_min": int(datetime(2024, 1, 15, 0, 0).timestamp() * 1e9),
                "ts_max": int(datetime(2024, 1, 15, 23, 59).timestamp() * 1e9),
                "count": 500,
                "status": "success",
            },
            # No MBP1 data for EUR/USD or GBP/USD
        ],
        "watermarks": {},
        "lineage": [],
    }

    # Save registry data
    registry_file = registry_path / "data_registry.json"
    with open(registry_file, "w") as f:
        json.dump(registry_data, f)

    # Create output file path
    output_file = tmp_path / "test_backfill.json"

    # Configure persistence for JSON backend
    persistence_config = PersistenceConfig(
        backend=BackendType.JSON,
        json_path=registry_path,
    )

    # Mock datetime.now to return a specific date within 30 days
    with patch("ml.cli.coverage.datetime") as mock_datetime:
        mock_datetime.strptime.side_effect = datetime.strptime
        mock_datetime.now.return_value = datetime(2024, 1, 20)  # 5 days after the target date

        # Run plan_backfill
        plan_backfill(
            from_dataset="BARS",
            to_dataset="MBP1",
            date="2024-01-15",
            instruments=None,
            registry_path=registry_path,
            persistence_config=persistence_config,
            output_file=output_file,
        )

    # Verify output file was created
    assert output_file.exists()

    # Load and verify job specification
    with open(output_file) as f:
        job_spec = json.load(f)

    assert job_spec["source_dataset"] == "BARS"
    assert job_spec["target_dataset"] == "MBP1"
    assert job_spec["date_range"]["date"] == "2024-01-15"
    assert job_spec["status"] == "planned"

    # Should find 2 instruments with gaps
    assert len(job_spec["instruments"]) == 2
    assert "EUR/USD" in job_spec["instruments"]
    assert "GBP/USD" in job_spec["instruments"]

    # Check gap details
    gaps = job_spec["gaps"]
    assert len(gaps) == 2

    eur_gap = next(g for g in gaps if g["instrument_id"] == "EUR/USD")
    assert eur_gap["source_count"] == 1000
    assert eur_gap["target_count"] == 0

    gbp_gap = next(g for g in gaps if g["instrument_id"] == "GBP/USD")
    assert gbp_gap["source_count"] == 500
    assert gbp_gap["target_count"] == 0

    # Check statistics
    stats = job_spec["statistics"]
    assert stats["instruments_with_gaps"] == 2
    assert stats["total_source_records"] == 1500


def test_plan_backfill_no_gaps(tmp_path: Path) -> None:
    """
    Test plan_backfill when no gaps exist.
    """
    # Setup test registry data
    registry_path = tmp_path / "test_registry"
    registry_path.mkdir(parents=True, exist_ok=True)

    # Create mock registry data without gaps (both datasets have data)
    registry_data = {
        "manifests": {
            "bars_test": {
                "dataset_id": "bars_test",
                "dataset_type": "bars",
                "storage_kind": "parquet",
                "location": "/data/bars",
                "partitioning": {"by": "ts_event", "interval": "daily"},
                "retention_days": 90,
                "schema": {
                    "instrument_id": "string",
                    "ts_event": "int64",
                    "ts_init": "int64",
                    "open": "float64",
                },
                "ts_field": "ts_event",
                "seq_field": None,
                "primary_keys": ["instrument_id", "ts_event"],
                "schema_hash": "test_hash_1",
                "constraints": {},
                "lineage": [],
                "pipeline_signature": "test_pipeline",
                "version": "1.0.0",
                "created_at": int(datetime(2024, 1, 1).timestamp() * 1e9),
                "last_modified": int(datetime(2024, 1, 1).timestamp() * 1e9),
                "metadata": {"description": "Test BARS dataset"},
            },
            "mbp1_test": {
                "dataset_id": "mbp1_test",
                "dataset_type": "mbp1",
                "storage_kind": "parquet",
                "location": "/data/mbp1",
                "partitioning": {"by": "ts_event", "interval": "daily"},
                "retention_days": 90,
                "schema": {
                    "instrument_id": "string",
                    "ts_event": "int64",
                    "ts_init": "int64",
                    "bid": "float64",
                },
                "ts_field": "ts_event",
                "seq_field": None,
                "primary_keys": ["instrument_id", "ts_event"],
                "schema_hash": "test_hash_2",
                "constraints": {},
                "lineage": [],
                "pipeline_signature": "test_pipeline",
                "version": "1.0.0",
                "created_at": int(datetime(2024, 1, 1).timestamp() * 1e9),
                "last_modified": int(datetime(2024, 1, 1).timestamp() * 1e9),
                "metadata": {"description": "Test MBP1 dataset"},
            },
        },
        "events": [
            # BARS data exists
            {
                "dataset_id": "bars_test",
                "instrument_id": "EUR/USD",
                "stage": "CATALOG_WRITTEN",
                "source": "historical",
                "run_id": "test_run_1",
                "ts_event": int(datetime(2024, 1, 15, 10, 0).timestamp() * 1e9),
                "count": 1000,
                "status": "success",
            },
            # MBP1 data also exists
            {
                "dataset_id": "mbp1_test",
                "instrument_id": "EUR/USD",
                "stage": "CATALOG_WRITTEN",
                "source": "historical",
                "run_id": "test_run_2",
                "ts_event": int(datetime(2024, 1, 15, 11, 0).timestamp() * 1e9),
                "count": 1000,
                "status": "success",
            },
        ],
        "watermarks": {},
        "lineage": [],
    }

    # Save registry data
    registry_file = registry_path / "data_registry.json"
    with open(registry_file, "w") as f:
        json.dump(registry_data, f)

    # Create output file path
    output_file = tmp_path / "test_backfill_no_gaps.json"

    # Configure persistence for JSON backend
    persistence_config = PersistenceConfig(
        backend=BackendType.JSON,
        json_path=registry_path,
    )

    # Mock datetime.now
    with patch("ml.cli.coverage.datetime") as mock_datetime:
        mock_datetime.strptime.side_effect = datetime.strptime
        mock_datetime.now.return_value = datetime(2024, 1, 20)

        # Run plan_backfill
        plan_backfill(
            from_dataset="BARS",
            to_dataset="MBP1",
            date="2024-01-15",
            instruments=None,
            registry_path=registry_path,
            persistence_config=persistence_config,
            output_file=output_file,
        )

    # Verify output file was created
    assert output_file.exists()

    # Load and verify job specification
    with open(output_file) as f:
        job_spec = json.load(f)

    # Should find no gaps
    assert len(job_spec["instruments"]) == 0
    assert len(job_spec["gaps"]) == 0
    assert job_spec["statistics"]["instruments_with_gaps"] == 0
    assert job_spec["statistics"]["total_source_records"] == 0


def test_plan_backfill_dataset_shortcuts() -> None:
    """
    Test that L1 and L2 shortcuts are properly mapped to BARS and MBP1.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        registry_path = tmp_path / "test_registry"
        registry_path.mkdir(parents=True, exist_ok=True)

        # Create minimal registry data
        registry_data: dict[str, Any] = {
            "manifests": {},
            "events": [],
            "watermarks": {},
            "lineage": [],
        }

        registry_file = registry_path / "data_registry.json"
        with open(registry_file, "w") as f:
            json.dump(registry_data, f)

        output_file = tmp_path / "test_shortcuts.json"

        persistence_config = PersistenceConfig(
            backend=BackendType.JSON,
            json_path=registry_path,
        )

        with patch("ml.cli.coverage.datetime") as mock_datetime:
            mock_datetime.strptime.side_effect = datetime.strptime
            mock_datetime.now.return_value = datetime(2024, 1, 20)

            # Test L1 -> L2 mapping
            plan_backfill(
                from_dataset="L1",
                to_dataset="L2",
                date="2024-01-15",
                registry_path=registry_path,
                persistence_config=persistence_config,
                output_file=output_file,
            )

        with open(output_file) as f:
            job_spec = json.load(f)

        assert job_spec["source_dataset"] == "BARS"  # L1 mapped to BARS
        assert job_spec["target_dataset"] == "MBP1"  # L2 mapped to MBP1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
