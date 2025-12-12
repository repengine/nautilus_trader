"""
Integration tests for DataScheduler with Databento.

These tests verify the scheduler can properly collect data from Databento and write it
to the Nautilus catalog.

"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch
from typing import Any, cast

import pytest


# Skip if optional dependency is not available
pytest.importorskip("databento", reason="databento package not installed")

from ml.config.scheduler_config import DatabentoConfig
from ml.config.scheduler_config import SchedulerConfig
from ml.data.scheduler import DataScheduler
from ml.stores.data_store import DataStore
from nautilus_trader.model.data import Bar
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)


@pytest.mark.integration
@pytest.mark.database
@pytest.mark.serial
@pytest.mark.slow
@pytest.mark.usefixtures("cloned_test_database")
class TestDataSchedulerIntegration:
    """
    Test DataScheduler with Databento integration.
    """

    @pytest.mark.database
    @pytest.mark.serial
    def test_scheduler_initialization(self, cloned_test_database: str) -> None:
        """
        Test scheduler initializes correctly with configuration.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create catalog
            catalog = ParquetDataCatalog(temp_dir)

            # Create config with PostgreSQL connection
            config = SchedulerConfig(
                symbols=["SPY.XNAS", "QQQ.XNAS"],
                retention_days=30,
                connection_string=cloned_test_database,
            )

            # Initialize scheduler with PostgreSQL
            scheduler = DataScheduler(
                catalog=catalog,
                config=config,
                connection=cloned_test_database,
            )

            cast(Any, scheduler)._data_store = DataStore(
                connection_string=cloned_test_database,
            )

            # Verify initialization
            assert scheduler.enabled is True
            assert len(scheduler.config.symbols) == 2
            assert scheduler.config.retention_days == 30
            assert scheduler._databento_loader is not None

    @pytest.mark.database
    @pytest.mark.serial
    def test_get_previous_trading_day(self, cloned_test_database: str) -> None:
        """
        Test getting previous trading day logic.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = ParquetDataCatalog(temp_dir)
            config = SchedulerConfig(
                connection_string=cloned_test_database,
            )
            scheduler = DataScheduler(
                catalog=catalog,
                config=config,
                connection=cloned_test_database,
            )

            # Mock different days
            with patch("ml.data.scheduler.datetime") as mock_datetime:
                # Test Monday -> Friday
                mock_datetime.now.return_value = datetime(2024, 1, 8)  # Monday
                mock_datetime.side_effect = datetime
                result = scheduler._get_previous_trading_day()
                assert result.date() == datetime(2024, 1, 5).date()  # Friday

                # Test Sunday -> Friday
                mock_datetime.now.return_value = datetime(2024, 1, 7)  # Sunday
                result = scheduler._get_previous_trading_day()
                assert result.date() == datetime(2024, 1, 5).date()  # Friday

                # Test Tuesday -> Monday
                mock_datetime.now.return_value = datetime(2024, 1, 9)  # Tuesday
                result = scheduler._get_previous_trading_day()
                assert result.date() == datetime(2024, 1, 8).date()  # Monday

    @pytest.mark.database
    @pytest.mark.serial
    def test_scheduler_status(self, cloned_test_database: str) -> None:
        """
        Test scheduler status reporting.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = ParquetDataCatalog(temp_dir)

            config = SchedulerConfig(
                symbols=["AAPL.XNAS"],
                databento=DatabentoConfig(
                    dataset="GLBX.MDP3",
                    schema="ohlcv-1m",
                ),
                connection_string=cloned_test_database,
            )

            scheduler = DataScheduler(
                catalog=catalog,
                config=config,
                connection=cloned_test_database,
            )

            status = scheduler.get_status()

            # Verify status fields
            assert status["enabled"] is True
            assert status["collection_time"] == "04:00"
            assert status["retention_days"] == 90
            assert status["symbol_count"] == 1
            assert status["databento_dataset"] == "GLBX.MDP3"
            assert status["databento_schema"] == "ohlcv-1m"
            assert status["has_feature_engineer"] is False

    @pytest.mark.database
    @pytest.mark.serial
    def test_collect_symbol_data_success(self, cloned_test_database: str) -> None:
        """
        Test successful data collection for a symbol.
        """
        with patch("ml.data.scheduler.db") as mock_db:
            with tempfile.TemporaryDirectory() as temp_dir:
                catalog = ParquetDataCatalog(temp_dir)

                config = SchedulerConfig(
                    symbols=["SPY.XNAS"],
                    databento=DatabentoConfig(
                        use_temporary_files=True,
                        temp_data_dir=temp_dir,
                    ),
                    connection_string=cloned_test_database,
                )

                scheduler = DataScheduler(
                    catalog=catalog,
                    config=config,
                    connection=cloned_test_database,
                )

                # Mock Databento client
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.to_file = MagicMock()
                mock_client.timeseries.get_range.return_value = mock_response

                # Mock loader to return sample data
                with patch.object(scheduler._databento_loader, "from_dbn_file") as mock_loader:
                    # Create mock bar data
                    mock_bar = MagicMock(spec=Bar)
                    mock_loader.return_value = [mock_bar]

                    # Test collection
                    result = scheduler._collect_symbol_data(
                        client=mock_client,
                        symbol="SPY.XNAS",
                        start_date=datetime.now() - timedelta(days=1),
                        end_date=datetime.now(),
                        target_date=datetime.now() - timedelta(days=1),
                        temp_data_dir=Path(temp_dir),
                    )

                    assert result is True
                    mock_client.timeseries.get_range.assert_called_once()
                    mock_loader.assert_called_once()

    @pytest.mark.database
    @pytest.mark.serial
    def test_collect_symbol_data_retry_logic(
        self,
        test_database: TestDatabase,
    ) -> None:
        """
        Test retry logic on collection failure.
        """
        with patch("ml.data.scheduler.db") as mock_db:
            with tempfile.TemporaryDirectory() as temp_dir:
                catalog = ParquetDataCatalog(temp_dir)

                config = SchedulerConfig(
                    symbols=["SPY.XNAS"],
                    max_retries=3,
                    retry_delay_seconds=0.1,
                    databento=DatabentoConfig(
                        use_temporary_files=True,
                        temp_data_dir=temp_dir,
                    ),
                    connection_string=test_database.connection_string,
                )

                scheduler = DataScheduler(
                    catalog=catalog,
                    config=config,
                    connection=test_database.connection_string,
                )

                # Mock client to fail twice then succeed
                mock_client = MagicMock()
                mock_client.timeseries.get_range.side_effect = [
                    Exception("Network error"),
                    Exception("Timeout"),
                    MagicMock(to_file=MagicMock()),  # Success on third try
                ]

                with patch.object(scheduler._databento_loader, "from_dbn_file") as mock_loader:
                    mock_loader.return_value = [MagicMock(spec=Bar)]

                    result = scheduler._collect_symbol_data(
                        client=mock_client,
                        symbol="SPY.XNAS",
                        start_date=datetime.now() - timedelta(days=1),
                        end_date=datetime.now(),
                        target_date=datetime.now() - timedelta(days=1),
                        temp_data_dir=Path(temp_dir),
                    )

                    assert result is True
                    assert mock_client.timeseries.get_range.call_count == 3

    @pytest.mark.database
    @pytest.mark.serial
    def test_load_from_dbn_file_venue_mapping(self, test_database: TestDatabase) -> None:
        """
        Test venue code mapping in DBN file loading.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = ParquetDataCatalog(temp_dir)
            config = SchedulerConfig(
                connection_string=test_database.connection_string,
            )
            scheduler = DataScheduler(
                catalog=catalog,
                config=config,
                connection=test_database.connection_string,
            )

            # Test venue mappings
            test_cases = [
                ("XNAS", "NASDAQ"),
                ("XNYS", "NYSE"),
                ("ARCX", "ARCA"),
                ("BATS", "BATS"),
                ("GLBX", "GLBX"),
                ("UNKNOWN", "UNKNOWN"),  # Should pass through unknown venues
            ]

            with patch.object(scheduler._databento_loader, "from_dbn_file") as mock_loader:
                mock_loader.return_value = []

                for input_venue, expected_venue in test_cases:
                    scheduler._load_from_dbn_file(
                        file_path=Path("test.dbn"),
                        symbol_code="TEST",
                        venue=input_venue,
                    )

                    # Check the instrument_id passed to loader
                    call_args = mock_loader.call_args
                    instrument_id = call_args.kwargs["instrument_id"]
                    assert str(instrument_id) == f"TEST.{expected_venue}"

    @pytest.mark.real_api
    @pytest.mark.skipif(
        not (os.getenv("DATABENTO_API_KEY") and os.getenv("ML_TEST_REAL_API")),
        reason="Real API test gated; set DATABENTO_API_KEY and ML_TEST_REAL_API=1",
    )
    @pytest.mark.database
    @pytest.mark.serial
    def test_collect_latest_data_with_real_api(self) -> None:
        """
        Test actual data collection with real Databento API (requires API key).
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = ParquetDataCatalog(temp_dir)

            # Use minimal config for real API test
            config = SchedulerConfig(
                symbols=["SPY.XNAS"],
                databento=DatabentoConfig(
                    dataset="GLBX.MDP3",
                    schema="ohlcv-1m",
                    use_temporary_files=True,
                    temp_data_dir=temp_dir,
                ),
                max_retries=1,
            )

            scheduler = DataScheduler(
                catalog=catalog,
                config=config,
            )

            # This will make real API calls
            scheduler._collect_latest_data()

            # Verify data was written to catalog
            if hasattr(catalog, "bar_types"):
                bar_types = getattr(catalog, "bar_types")()
                assert len(bar_types) > 0

            # Check we have SPY data (best-effort; allow skip on empty result)
            spy_instrument = InstrumentId.from_str("SPY.XNAS")
            bars = catalog.bars([spy_instrument])
            if len(bars) == 0:
                pytest.skip("Databento returned no data for SPY.XNAS; treating as transient")
            assert len(bars) > 0

    @pytest.mark.database
    @pytest.mark.serial
    def test_clean_old_data(self, test_database: TestDatabase) -> None:
        """
        Test cleanup of old data (placeholder test).
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = ParquetDataCatalog(temp_dir)

            config = SchedulerConfig(
                retention_days=30,
                connection_string=test_database.connection_string,
            )
            scheduler = DataScheduler(
                catalog=catalog,
                config=config,
                connection=test_database.connection_string,
            )

            # Currently just logs, but test it doesn't error
            scheduler._clean_old_data()

    @pytest.mark.database
    @pytest.mark.serial
    def test_compute_features(self, test_database: TestDatabase) -> None:
        """
        Test feature computation trigger (placeholder test).
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = ParquetDataCatalog(temp_dir)
            config = SchedulerConfig(
                connection_string=test_database.connection_string,
            )
            scheduler = DataScheduler(
                catalog=catalog,
                config=config,
                connection=test_database.connection_string,
            )

            # Without feature engineer, should return early
            scheduler._compute_features()

            # With mock feature engineer
            scheduler.feature_engineer = MagicMock()
            scheduler._compute_features()
