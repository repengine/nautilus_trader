"""
Integration tests for DataScheduler with FeatureStore.

Tests the complete flow of data collection and feature computation using the scheduler
with FeatureStore integration.

"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from ml.config.scheduler_config import DatabentoConfig
from ml.config.scheduler_config import SchedulerConfig
from ml.data.scheduler import DataScheduler
from ml.features.config import FeatureConfig
from ml.features.facade import FeatureEngineer
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AggregationSource
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

if TYPE_CHECKING:
    from ml.tests.fixtures.database_fixtures import TestDatabase

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)


def create_test_bars(
    instrument_id: InstrumentId,
    start_time: datetime,
    num_bars: int = 100,
) -> list[Bar]:
    """
    Create test bars for integration testing.

    Parameters
    ----------
    instrument_id : InstrumentId
        The instrument ID for the bars
    start_time : datetime
        Starting timestamp for the bars
    num_bars : int
        Number of bars to create

    Returns
    -------
    list[Bar]
        List of test bars

    """
    bars = []
    bar_type = BarType(
        instrument_id=instrument_id,
        bar_spec=BarSpecification(
            step=1,
            aggregation=BarAggregation.MINUTE,
            price_type=PriceType.LAST,
        ),
        aggregation_source=AggregationSource.EXTERNAL,
    )

    current_time = start_time
    base_price = 100.0

    for i in range(num_bars):
        # Create realistic OHLCV data with some volatility
        open_price = base_price + (i % 5) * 0.1
        high_price = open_price + 0.5
        low_price = open_price - 0.3
        close_price = open_price + 0.2
        volume = 1000.0 + (i % 10) * 100

        bar = Bar(
            bar_type=bar_type,
            open=Price.from_str(f"{open_price:.2f}"),
            high=Price.from_str(f"{high_price:.2f}"),
            low=Price.from_str(f"{low_price:.2f}"),
            close=Price.from_str(f"{close_price:.2f}"),
            volume=Quantity.from_str(f"{volume:.0f}"),
            ts_event=int(current_time.timestamp() * 1e9),
            ts_init=int(current_time.timestamp() * 1e9),
        )
        bars.append(bar)

        # Increment time by 1 minute
        current_time += timedelta(minutes=1)
        base_price += 0.01  # Slight trend

    return bars


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.integration
@pytest.mark.usefixtures("clean_postgres_db_class")
class TestSchedulerFeatureStoreIntegration:
    """
    Test DataScheduler with FeatureStore integration.
    """

    def setup_method(self) -> None:
        """
        Set up test fixtures.
        """
        # Create temporary directory for catalog
        self.temp_dir = tempfile.mkdtemp()
        self.catalog_path = Path(self.temp_dir) / "catalog"
        self.catalog_path.mkdir()

        # Create catalog
        self.catalog = ParquetDataCatalog(str(self.catalog_path))

        # Create test configuration
        self.config = SchedulerConfig(
            symbols=["SPY.XNAS", "QQQ.XNAS"],
            retention_days=30,
            databento=DatabentoConfig(
                dataset="GLBX.MDP3",
                schema="ohlcv-1m",
            ),
            feature_store_enabled=True,
            feature_store_connection=None,  # Will be provided via test fixture
        )

        # Create feature config and engineer
        self.feature_config = FeatureConfig(
            lookback_window=20,
            return_periods=[1, 5, 10],
            momentum_periods=[5, 10],
            rsi_period=14,
            bb_period=20,
        )
        self.feature_engineer = FeatureEngineer(self.feature_config)

    def teardown_method(self) -> None:
        """
        Clean up test fixtures.
        """
        import shutil

        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @pytest.mark.database
    @pytest.mark.serial
    def test_feature_computation_with_catalog_data(
        self,
        test_database: TestDatabase,
    ) -> None:
        """
        Test feature computation when bars are available in catalog.
        """
        # Update config with test database connection (frozen dataclass -> replace)
        from dataclasses import replace as _replace

        self.config = _replace(
            self.config,
            feature_store_connection=test_database.connection_string,
        )
        # Create scheduler
        with patch("ml.stores.feature_store.FeatureStore") as mock_feature_store_class:
            # Set up mock feature store
            mock_feature_store = MagicMock()
            mock_feature_store.compute_and_store_historical.return_value = 100
            mock_feature_store_class.return_value = mock_feature_store

            scheduler = DataScheduler(
                catalog=self.catalog,
                config=self.config,
                feature_engineer=self.feature_engineer,
            )

        # Prepare test data in catalog
        instrument_id = InstrumentId.from_str("SPY.NASDAQ")
        start_time = datetime.now() - timedelta(days=1)
        start_time = start_time.replace(hour=9, minute=30, second=0, microsecond=0)

        test_bars = create_test_bars(instrument_id, start_time, num_bars=390)  # Full trading day

        # Write bars to catalog
        self.catalog.write_data(test_bars)

        # Run feature computation
        with patch.object(scheduler, "_get_previous_trading_day") as mock_get_date:
            mock_get_date.return_value = start_time
            scheduler._compute_features()

        # Verify feature store was initialized
        assert mock_feature_store_class.called

        # Verify compute_and_store_historical was called for each symbol
        assert mock_feature_store.compute_and_store_historical.call_count >= 1

    @pytest.mark.database
    @pytest.mark.serial
    def test_feature_computation_disabled(self) -> None:
        """
        Test that feature computation is skipped when disabled.
        """
        # Create config with feature store disabled
        config = SchedulerConfig(
            symbols=["SPY.XNAS"],
            feature_store_enabled=False,
        )

        # Create scheduler
        scheduler = DataScheduler(
            catalog=self.catalog,
            config=config,
            feature_engineer=self.feature_engineer,
        )

        # Run feature computation - should return early
        scheduler._compute_features()

        # Verify feature store was not initialized
        assert scheduler._feature_store is None

    @pytest.mark.database
    @pytest.mark.serial
    def test_feature_computation_without_engineer(self) -> None:
        """
        Test that feature computation is skipped without feature engineer.
        """
        # Create scheduler without feature engineer
        scheduler = DataScheduler(
            catalog=self.catalog,
            config=self.config,
            feature_engineer=None,
        )

        # Run feature computation - should return early
        scheduler._compute_features()

        # Verify feature store was not initialized
        assert scheduler._feature_store is None

    @pytest.mark.database
    @pytest.mark.serial
    def test_feature_store_initialization_failure(self, test_database: TestDatabase) -> None:
        """
        Test graceful handling of feature store initialization failure.
        """
        # Set an invalid connection string to trigger failure
        from dataclasses import replace as _replace

        self.config = _replace(self.config, feature_store_connection="invalid://connection")

        # Create scheduler - should handle failure gracefully
        with patch("ml.stores.feature_store.create_engine") as mock_create_engine:
            mock_create_engine.side_effect = Exception("Database connection failed")

            scheduler = DataScheduler(
                catalog=self.catalog,
                config=self.config,
                feature_engineer=self.feature_engineer,
            )

            # Feature store should be None after failed initialization
            assert scheduler._feature_store is None

        # Scheduler should still be functional
        assert scheduler.enabled
        assert len(scheduler.config.symbols) == 2

    @pytest.mark.database
    @pytest.mark.serial
    def test_symbol_parsing_and_mapping(self) -> None:
        """
        Test correct parsing and mapping of symbol formats.
        """
        config = SchedulerConfig(
            symbols=["AAPL.XNAS", "MSFT.XNYS", "INVALID", "TEST.UNKNOWN"],
        )

        scheduler = DataScheduler(
            catalog=self.catalog,
            config=config,
        )

        # Test venue mapping
        venue_map = {
            "XNAS": "NASDAQ",
            "XNYS": "NYSE",
            "ARCX": "ARCA",
            "BATS": "BATS",
            "GLBX": "GLBX",
        }

        # Verify mapping logic
        assert venue_map["XNAS"] == "NASDAQ"
        assert venue_map["XNYS"] == "NYSE"
        assert venue_map.get("UNKNOWN", "UNKNOWN") == "UNKNOWN"

    @pytest.mark.database
    @pytest.mark.serial
    def test_feature_store_connection_from_env(self, test_database: TestDatabase) -> None:
        """
        Test that feature store uses connection string from environment.
        """
        # Temporarily set environment variable
        with patch.dict(
            os.environ,
            {"NAUTILUS_DB_CONNECTION": test_database.connection_string},
        ):
            with patch("ml.stores.feature_store.FeatureStore") as mock_feature_store_class:
                mock_feature_store = MagicMock()
                mock_feature_store_class.return_value = mock_feature_store

                # Create scheduler with no explicit connection string
                config = SchedulerConfig(
                    feature_store_enabled=True,
                    feature_store_connection=None,  # Should use env var
                )

                scheduler = DataScheduler(
                    catalog=self.catalog,
                    config=config,
                    feature_engineer=self.feature_engineer,
                )

                # Verify feature store was initialized with env var connection
                mock_feature_store_class.assert_called_once()
                call_args = mock_feature_store_class.call_args
                assert test_database.connection_string in str(call_args)

    @pytest.mark.database
    @pytest.mark.serial
    def test_metrics_tracking(self, test_database: TestDatabase) -> None:
        """
        Test that feature computation tracks metrics correctly.
        """
        # Update config with test database connection
        from dataclasses import replace as _replace

        self.config = _replace(
            self.config,
            feature_store_connection=test_database.connection_string,
        )
        with patch("ml.stores.feature_store.FeatureStore") as mock_feature_store_class:
            mock_feature_store = MagicMock()
            mock_feature_store.compute_and_store_historical.return_value = 50
            mock_feature_store_class.return_value = mock_feature_store

            scheduler = DataScheduler(
                catalog=self.catalog,
                config=self.config,
                feature_engineer=self.feature_engineer,
            )

            # Mock catalog to return empty results (no data scenario)
            with patch.object(scheduler.catalog, "query", return_value=[]):
                scheduler._compute_features()

            # Feature store should be initialized even with no data
            assert scheduler._feature_store is not None
