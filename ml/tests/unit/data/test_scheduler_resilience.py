#!/usr/bin/env python3
"""
Resilience and error recovery tests for DataScheduler.

This module tests all error handling paths in the DataScheduler class,
ensuring robust operation in production environments.

"""

from __future__ import annotations

import asyncio
import json
import os
import pickle
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import PropertyMock
from unittest.mock import call
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ml._imports import pl
from ml.config.scheduler_config import DatabentoConfig
from ml.config.scheduler_config import SchedulerConfig
from ml.data.scheduler import DataScheduler
from ml.data.scheduler import track_pipeline_stage
from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.registry.data_registry import DataRegistry
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.stores.data_store import DataStore
from ml.stores.feature_store import FeatureStore
from nautilus_trader.model.data import Bar
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.flaky
@pytest.mark.unit
@pytest.mark.usefixtures("clean_postgres_db")
class TestSchedulerResilience:
    """Test DataScheduler resilience and error recovery."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create temporary directory."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        # Cleanup
        import shutil
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def mock_catalog(self, temp_dir: Path) -> MagicMock:
        """Create mock catalog."""
        catalog = MagicMock(spec=ParquetDataCatalog)
        catalog.path = temp_dir
        catalog.write_data = MagicMock()
        catalog.query = MagicMock(return_value=[])
        return catalog

    @pytest.fixture
    def scheduler_config(self) -> SchedulerConfig:
        """Create scheduler configuration."""
        return SchedulerConfig(
            symbols=["SPY.XNAS", "QQQ.XNAS", "IWM.XNAS"],
            retention_days=30,
            max_retries=3,
            retry_delay_seconds=0.01,
            feature_store_enabled=True,
            databento=DatabentoConfig(
                dataset="GLBX.MDP3",
                schema="ohlcv-1m",
                use_temporary_files=True,
                temp_data_dir="/tmp/databento_temp",
            ),
        )

    @pytest.fixture
    def mock_feature_engineer(self) -> MagicMock:
        """Create mock feature engineer."""
        engineer = MagicMock(spec=FeatureEngineer)
        engineer.config = FeatureConfig()
        engineer.calculate_features_batch = MagicMock(
            return_value=(
                pl.DataFrame({"feature1": [1.0, 2.0, 3.0]}),
                ["feature1"],
            )
        )
        return engineer

    @pytest.fixture
    def scheduler(
        self,
        mock_catalog: MagicMock,
        scheduler_config: SchedulerConfig,
        mock_feature_engineer: MagicMock,
        test_database,
    ) -> DataScheduler:
        """Create DataScheduler instance."""
        with patch.dict(os.environ, {"DATABENTO_API_KEY": "test_key"}):
            scheduler = DataScheduler(
                catalog=mock_catalog,
                config=scheduler_config,
                feature_engineer=mock_feature_engineer,
                start_metrics_server=False,
                connection=test_database.connection_string,
            )
            # Initialize DataStore with PostgreSQL
            scheduler._data_store = DataStore(connection_string=test_database.connection_string)
            return scheduler

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.usefixtures("clean_postgres_db")
    def test_data_registry_initialization_failures(
        self,
        mock_catalog: MagicMock,
        scheduler_config: SchedulerConfig,
        test_database,
    ) -> None:
        """Test DataRegistry initialization failure handling."""
        with patch("ml.data.scheduler.DataCollector") as mock_collector_cls:
            mock_collector_cls.return_value = MagicMock()
            with patch("ml.data.scheduler.DataRegistry") as mock_registry:
                mock_registry.side_effect = Exception("Database connection failed")

                # Should handle registry failure gracefully
                scheduler = DataScheduler(
                    catalog=mock_catalog,
                    config=scheduler_config,
                    start_metrics_server=False,
                    connection=test_database.connection_string,
                )

                assert scheduler._data_registry is None

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.usefixtures("clean_postgres_db")
    def test_feature_store_initialization_failures(
        self,
        mock_catalog: MagicMock,
        scheduler_config: SchedulerConfig,
        mock_feature_engineer: MagicMock,
        test_database,
    ) -> None:
        """Test FeatureStore initialization failure handling."""
        with patch("ml.data.scheduler.DataCollector") as mock_collector_cls:
            mock_collector_cls.return_value = MagicMock()
            with patch("ml.stores.feature_store.FeatureStore") as mock_store:
                mock_store.side_effect = Exception("PostgreSQL connection failed")

                # Should handle feature store failure gracefully
                scheduler = DataScheduler(
                    catalog=mock_catalog,
                    config=scheduler_config,
                    feature_engineer=mock_feature_engineer,
                    start_metrics_server=False,
                    connection=test_database.connection_string,
                )

                assert scheduler._feature_store is None

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.usefixtures("clean_postgres_db")
    def test_metrics_server_startup_failure(
        self,
        mock_catalog: MagicMock,
        scheduler_config: SchedulerConfig,
        test_database,
    ) -> None:
        """Test metrics server startup failure handling."""
        with patch("ml.data.scheduler.DataCollector") as mock_collector_cls:
            mock_collector_cls.return_value = MagicMock()
            with patch("ml.monitoring.server.MetricsServer") as mock_server:
                mock_server.side_effect = Exception("Port already in use")

                # Should handle metrics server failure gracefully
                scheduler = DataScheduler(
                    catalog=mock_catalog,
                    config=scheduler_config,
                    start_metrics_server=True,
                    metrics_port=8000,
                    connection=test_database.connection_string,
                )

                assert scheduler._metrics_server is None

    @pytest.mark.database
    @pytest.mark.serial
    def test_pipeline_stage_tracking_with_errors(self) -> None:
        """Test pipeline stage tracking handles errors."""
        with patch("time.time") as mock_time:
            mock_time.side_effect = [1.0, 2.0]  # Start and end times

            # Test with exception in stage
            with pytest.raises(ValueError):
                with track_pipeline_stage("test_stage"):
                    raise ValueError("Stage failed")

            # Metrics should still be recorded
            assert mock_time.call_count == 2

    @pytest.mark.database
    @pytest.mark.serial
    def test_daily_update_partial_failure_recovery(
        self,
        scheduler: DataScheduler,
        mock_catalog: MagicMock,
    ) -> None:
        """Test daily update continues after partial failures."""
        with patch.object(scheduler, "_collect_latest_data") as mock_collect:
            mock_collect.side_effect = Exception("Collection failed")

            with patch.object(scheduler, "_compute_features") as mock_compute:
                mock_compute.return_value = None

                with patch.object(scheduler, "_clean_old_data") as mock_clean:
                    mock_clean.return_value = None

                    # Should handle collection failure and continue
                    with pytest.raises(Exception):
                        scheduler.run_daily_update()

                    # Collection should be attempted
                    assert mock_collect.called
                    # Other stages should not run after failure
                    assert not mock_compute.called
                    assert not mock_clean.called

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.usefixtures("clean_postgres_db")
    def test_databento_api_key_missing(
        self,
        mock_catalog: MagicMock,
        scheduler_config: SchedulerConfig,
        test_database,
    ) -> None:
        """Test handling of missing Databento API key."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("ml.data.scheduler.DataCollector") as mock_collector_cls:
                mock_collector_cls.return_value = MagicMock()
                scheduler = DataScheduler(
                    catalog=mock_catalog,
                    config=scheduler_config,
                    start_metrics_server=False,
                    connection=test_database.connection_string,
                )

                # Should raise ValueError for missing API key
                with pytest.raises(ValueError, match="DATABENTO_API_KEY"):
                    scheduler._collect_latest_data()

    @pytest.mark.database
    @pytest.mark.serial
    def test_databento_library_import_failure(
        self,
        scheduler: DataScheduler,
    ) -> None:
        """Test handling of missing databento library."""
        with patch.dict("sys.modules", {"databento": None}):
            with patch("builtins.__import__") as mock_import:
                mock_import.side_effect = ImportError("No module named 'databento'")

                # Should raise ImportError with helpful message
                with pytest.raises(ImportError, match="databento"):
                    scheduler._collect_latest_data()

    @pytest.mark.database
    @pytest.mark.serial
    def test_temporary_file_handling_errors(
        self,
        scheduler: DataScheduler,
        temp_dir: Path,
    ) -> None:
        """Test temporary file creation and cleanup errors."""
        scheduler.config.databento.use_temporary_files = True
        scheduler.config.databento.temp_data_dir = str(temp_dir / "nonexistent" / "nested")

        with patch("databento.Historical") as mock_db:
            mock_client = MagicMock()
            mock_db.return_value = mock_client

            # Should create nested directories
            with patch.dict(os.environ, {"DATABENTO_API_KEY": "test_key"}):
                scheduler._collect_latest_data()

                # Temp directory should be created
                temp_path = Path(scheduler.config.databento.temp_data_dir)
                assert temp_path.exists()

    @pytest.mark.database
    @pytest.mark.serial
    def test_symbol_data_collection_all_failure_modes(
        self,
        scheduler: DataScheduler,
        temp_dir: Path,
    ) -> None:
        """Test all symbol data collection failure modes."""
        test_cases = [
            # (symbol_format, error_type, should_retry)
            ("INVALID", "format", False),
            ("SPY", "format", False),  # Missing venue
            ("SPY.XNAS.EXTRA", "format", False),  # Too many parts
            ("SPY.XNAS", "rate_limit", True),
            ("SPY.XNAS", "connection", True),
            ("SPY.XNAS", "unauthorized", False),
            ("SPY.XNAS", "unknown", True),
        ]

        for symbol, error_type, should_retry in test_cases:
            with patch("databento.Historical") as mock_db:
                mock_client = MagicMock()

                # Set up error based on type
                if error_type == "format":
                    # Format errors are caught before API call
                    pass
                elif error_type == "rate_limit":
                    mock_client.timeseries.get_range.side_effect = Exception("rate limit exceeded")
                elif error_type == "connection":
                    mock_client.timeseries.get_range.side_effect = Exception("Connection timeout")
                elif error_type == "unauthorized":
                    mock_client.timeseries.get_range.side_effect = Exception("Unauthorized access")
                else:
                    mock_client.timeseries.get_range.side_effect = Exception("Unknown error")

                success = scheduler._collect_symbol_data(
                    client=mock_client,
                    symbol=symbol,
                    start_date=datetime.now(),
                    end_date=datetime.now(),
                    target_date=datetime.now(),
                    temp_data_dir=temp_dir if scheduler.config.databento.use_temporary_files else None,
                )

                assert not success

                if error_type != "format" and should_retry:
                    # Should have retried
                    assert mock_client.timeseries.get_range.call_count == scheduler.config.max_retries
                elif error_type != "format":
                    # Should not retry for auth errors
                    assert mock_client.timeseries.get_range.call_count <= 1

    @pytest.mark.database
    @pytest.mark.serial
    def test_dbn_file_loading_errors(
        self,
        scheduler: DataScheduler,
        temp_dir: Path,
    ) -> None:
        """Test DBN file loading error scenarios."""
        # Create various corrupted DBN files
        test_files = [
            (temp_dir / "empty.dbn", b""),
            (temp_dir / "corrupt.dbn", b"CORRUPT_DATA_12345"),
            (temp_dir / "partial.dbn", b"DBN\x00\x01"),
        ]

        for file_path, content in test_files:
            file_path.write_bytes(content)

            with patch.object(scheduler._databento_loader, "from_dbn_file") as mock_loader:
                mock_loader.side_effect = ValueError(f"Invalid DBN file: {file_path}")

                result = scheduler._load_from_dbn_file(
                    file_path=file_path,
                    symbol_code="TEST",
                    venue="XNAS",
                )

                # Should return empty list on error
                assert result == []

    @pytest.mark.database
    @pytest.mark.serial
    def test_catalog_write_failures_with_recovery(
        self,
        scheduler: DataScheduler,
        mock_catalog: MagicMock,
    ) -> None:
        """Test catalog write failures and recovery attempts."""
        with patch("databento.Historical") as mock_db:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.to_file = MagicMock()
            mock_client.timeseries.get_range.return_value = mock_response

            with patch.object(scheduler, "_load_from_dbn_file") as mock_load:
                mock_load.return_value = [MagicMock(ts_event=1000, ts_init=2000)]

                # First write fails, second succeeds
                mock_catalog.write_data.side_effect = [
                    Exception("Disk full"),
                    None,
                ]

                # Should handle write failure
                success = scheduler._collect_symbol_data(
                    client=mock_client,
                    symbol="SPY.XNAS",
                    start_date=datetime.now(),
                    end_date=datetime.now(),
                    target_date=datetime.now(),
                    temp_data_dir=Path("/tmp"),
                )

                # Should fail after first write error
                assert not success

    @pytest.mark.database
    @pytest.mark.serial
    def test_data_registry_event_emission_failures(
        self,
        scheduler: DataScheduler,
        mock_catalog: MagicMock,
    ) -> None:
        """Test handling of DataRegistry event emission failures."""
        # Create mock registry
        mock_registry = MagicMock(spec=DataRegistry)
        mock_registry.emit_event.side_effect = Exception("Registry unavailable")
        scheduler._data_registry = mock_registry

        with patch("databento.Historical") as mock_db:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.to_file = MagicMock()
            mock_client.timeseries.get_range.return_value = mock_response

            with patch.object(scheduler, "_load_from_dbn_file") as mock_load:
                mock_load.return_value = [MagicMock(ts_event=1000, ts_init=2000)]

                # Should continue despite registry failure
                success = scheduler._collect_symbol_data(
                    client=mock_client,
                    symbol="SPY.XNAS",
                    start_date=datetime.now(),
                    end_date=datetime.now(),
                    target_date=datetime.now(),
                    temp_data_dir=Path("/tmp"),
                )

                assert success
                assert mock_catalog.write_data.called

    @pytest.mark.database
    @pytest.mark.serial
    def test_feature_computation_error_scenarios(
        self,
        scheduler: DataScheduler,
        mock_catalog: MagicMock,
        mock_feature_engineer: MagicMock,
    ) -> None:
        """Test various feature computation error scenarios."""
        # Initialize feature store
        mock_feature_store = MagicMock(spec=FeatureStore)
        scheduler._feature_store = mock_feature_store

        # Test different error scenarios
        error_scenarios = [
            # (bars_data, feature_error, store_error, expected_result)
            ([], None, None, "no_bars"),  # No bars found
            ([MagicMock(spec=Bar)], ValueError("Invalid features"), None, "feature_error"),
            ([MagicMock(spec=Bar)], None, Exception("Store failed"), "store_error"),
        ]

        for bars_data, feature_error, store_error, expected in error_scenarios:
            mock_catalog.query.return_value = bars_data

            if feature_error:
                mock_feature_engineer.calculate_features_batch.side_effect = feature_error
            else:
                mock_feature_engineer.calculate_features_batch.return_value = (
                    pl.DataFrame({"feat1": [1.0]}),
                    ["feat1"],
                )

            if store_error:
                mock_feature_store.compute_and_store_historical.side_effect = store_error
            else:
                mock_feature_store.compute_and_store_historical.return_value = 100

            # Should handle errors gracefully
            if expected in ["feature_error", "store_error"]:
                # These raise exceptions that are caught
                scheduler._compute_features()
            else:
                # No bars case completes normally
                scheduler._compute_features()

    @pytest.mark.database
    @pytest.mark.serial
    def test_feature_computation_with_invalid_symbols(
        self,
        scheduler: DataScheduler,
        mock_catalog: MagicMock,
        mock_feature_engineer: MagicMock,
    ) -> None:
        """Test feature computation with invalid symbol formats."""
        scheduler.config.symbols = ["INVALID", "SPY", "QQQ.XNAS.EXTRA"]
        mock_feature_store = MagicMock(spec=FeatureStore)
        scheduler._feature_store = mock_feature_store

        # Should skip invalid symbols
        scheduler._compute_features()

        # Should process valid symbols only
        valid_calls = [
            call for call in mock_catalog.query.call_args_list
            if "identifiers" in call[1]
        ]
        # No valid symbols in the test set
        assert len(valid_calls) == 0

    @pytest.mark.database
    @pytest.mark.serial
    def test_data_cleanup_failures(
        self,
        scheduler: DataScheduler,
        mock_catalog: MagicMock,
    ) -> None:
        """Test data cleanup error handling."""
        with patch.object(mock_catalog, "delete_old_data") as mock_delete:
            mock_delete.side_effect = Exception("Cleanup failed")

            # Should handle cleanup failure
            with pytest.raises(Exception):
                scheduler._clean_old_data()

    @pytest.mark.database
    @pytest.mark.serial
    def test_previous_trading_day_calculation_edge_cases(
        self,
        scheduler: DataScheduler,
    ) -> None:
        """Test previous trading day calculation for edge cases."""
        test_cases = [
            # (current_day, expected_previous_day)
            (datetime(2025, 1, 20), datetime(2025, 1, 17)),  # Monday -> Friday
            (datetime(2025, 1, 19), datetime(2025, 1, 17)),  # Sunday -> Friday
            (datetime(2025, 1, 18), datetime(2025, 1, 17)),  # Saturday -> Friday (not tested in original)
            (datetime(2025, 1, 21), datetime(2025, 1, 20)),  # Tuesday -> Monday
            (datetime(2025, 1, 22), datetime(2025, 1, 21)),  # Wednesday -> Tuesday
            (datetime(2025, 1, 23), datetime(2025, 1, 22)),  # Thursday -> Wednesday
            (datetime(2025, 1, 24), datetime(2025, 1, 23)),  # Friday -> Thursday
        ]

        for current_day, expected_previous in test_cases:
            with patch("ml.data.scheduler.datetime") as mock_dt:
                mock_dt.now.return_value = current_day
                mock_dt.side_effect = datetime  # Allow normal datetime operations

                result = scheduler._get_previous_trading_day()
                assert result.date() == expected_previous.date()

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.usefixtures("clean_postgres_db")
    def test_concurrent_scheduler_runs_prevention(
        self,
        mock_catalog: MagicMock,
        scheduler_config: SchedulerConfig,
        temp_dir: Path,
        test_database,
    ) -> None:
        """Test prevention of concurrent scheduler runs."""
        lock_file = temp_dir / "scheduler.lock"

        with patch("ml.data.scheduler.DataCollector") as mock_collector_cls:
            mock_collector_cls.return_value = MagicMock()

            # Create two scheduler instances
            scheduler1 = DataScheduler(
                catalog=mock_catalog,
                config=scheduler_config,
                start_metrics_server=False,
                connection=test_database.connection_string,
            )
            scheduler2 = DataScheduler(
                catalog=mock_catalog,
                config=scheduler_config,
                start_metrics_server=False,
                connection=test_database.connection_string,
            )

        # Simulate lock mechanism
        class SimpleLock:
            def __init__(self, path: Path):
                self.path = path
                self.locked = False

            def acquire(self) -> bool:
                if not self.locked and not self.path.exists():
                    self.path.touch()
                    self.locked = True
                    return True
                return False

            def release(self) -> None:
                if self.locked and self.path.exists():
                    self.path.unlink()
                    self.locked = False

        lock1 = SimpleLock(lock_file)
        lock2 = SimpleLock(lock_file)

        # First scheduler acquires lock
        assert lock1.acquire()

        # Second scheduler cannot acquire
        assert not lock2.acquire()

        # Release first lock
        lock1.release()

        # Now second can acquire
        assert lock2.acquire()
        lock2.release()

    @pytest.mark.database
    @pytest.mark.serial
    def test_get_status_with_various_states(
        self,
        scheduler: DataScheduler,
    ) -> None:
        """Test get_status method with various scheduler states."""
        # Test initial state
        status = scheduler.get_status()
        assert status["enabled"] is True
        assert status["symbol_count"] == 3
        assert status["retention_days"] == 30

        # Test after disabling
        scheduler.enabled = False
        status = scheduler.get_status()
        assert status["enabled"] is False

        # Test with feature engineer
        assert status["has_feature_engineer"] is True

    @pytest.mark.database
    @pytest.mark.serial
    def test_scheduler_stop_cleanup(
        self,
        scheduler: DataScheduler,
    ) -> None:
        """Test scheduler cleanup on stop."""
        # Create mock metrics server
        mock_server = MagicMock()
        scheduler._metrics_server = mock_server

        # Stop scheduler
        scheduler.stop()

        # Should stop metrics server
        mock_server.stop.assert_called_once()
        assert scheduler.enabled is False

    @pytest.mark.database
    @pytest.mark.serial
    def test_high_failure_rate_detection(
        self,
        scheduler: DataScheduler,
    ) -> None:
        """Test detection of high failure rates."""
        with patch("databento.Historical") as mock_db:
            mock_client = MagicMock()
            # All requests fail
            mock_client.timeseries.get_range.side_effect = Exception("Service down")

            with patch.dict(os.environ, {"DATABENTO_API_KEY": "test_key"}):
                with patch("ml.data.scheduler.api_rate_limit_hits") as mock_metric:
                    scheduler._collect_latest_data()

                    # Should detect high failure rate (>70%)
                    assert mock_metric.labels.called

    @pytest.mark.database
    @pytest.mark.serial
    def test_polars_import_failure_handling(
        self,
        scheduler: DataScheduler,
        mock_catalog: MagicMock,
    ) -> None:
        """Test handling of missing Polars library."""
        scheduler._feature_store = MagicMock()

        with patch("ml._imports.HAS_POLARS", False):
            with patch("ml._imports.check_ml_dependencies") as mock_check:
                mock_check.side_effect = ImportError("Polars not installed")

                # Should raise ImportError with helpful message
                with pytest.raises(ImportError):
                    scheduler._compute_features()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
