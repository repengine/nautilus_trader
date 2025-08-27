"""
Unit tests for TFT Dataset Builder with FeatureStore integration.

Tests the integration between TFTDatasetBuilder and FeatureStore to ensure proper
training/inference parity.

"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml.config.base import MLFeatureConfig
from ml.data.tft_dataset_builder import TFTDatasetBuilder
from ml.stores.feature_store import FeatureStore
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from ml.tests.fixtures.database_fixtures import TestDatabase


class TestTFTDatasetBuilderWithFeatureStore:
    """
    Test TFT Dataset Builder FeatureStore integration.
    """

    @pytest.fixture
    def mock_catalog(self) -> MagicMock:
        """
        Create mock ParquetDataCatalog.
        """
        return MagicMock(spec=ParquetDataCatalog)

    @pytest.fixture
    def mock_feature_store(self, test_database: TestDatabase) -> MagicMock:
        """
        Create mock FeatureStore.
        """
        mock_store = MagicMock(spec=FeatureStore)
        mock_store.connection_string = test_database.connection_string

        # Mock get_training_data to return sample features
        features = np.random.randn(100, 10).astype(np.float64)
        timestamps = np.arange(100, dtype=np.int64) * int(1e9)  # Nanosecond timestamps
        feature_names = [f"feature_{i}" for i in range(10)]

        mock_store.get_training_data.return_value = (features, timestamps, feature_names)

        return mock_store

    @pytest.fixture
    def feature_config(self) -> MLFeatureConfig:
        """
        Create feature configuration.
        """
        return MLFeatureConfig()

    @pytest.mark.usefixtures("clean_postgres_db")
    def test_init_with_feature_store(
        self,
        mock_catalog: MagicMock,
        mock_feature_store: MagicMock,
        feature_config: MLFeatureConfig,
        test_database: TestDatabase,
    ) -> None:
        """
        Test initialization with FeatureStore.
        """
        symbols = ["AAPL", "MSFT", "GOOGL"]

        builder = TFTDatasetBuilder(
            catalog=mock_catalog,
            symbols=symbols,
            feature_config=feature_config,
            feature_store=mock_feature_store,
        )

        assert builder.catalog == mock_catalog
        assert builder.symbols == symbols
        assert builder.feature_store == mock_feature_store
        assert builder.feature_config == feature_config

    def test_init_without_feature_store(
        self,
        mock_catalog: MagicMock,
        feature_config: MLFeatureConfig,
    ) -> None:
        """
        Test initialization without FeatureStore (backward compatibility).
        """
        symbols = ["SPY", "QQQ"]

        builder = TFTDatasetBuilder(
            catalog=mock_catalog,
            symbols=symbols,
            feature_config=feature_config,
        )

        assert builder.catalog == mock_catalog
        assert builder.symbols == symbols
        assert builder.feature_store is None
        assert builder.feature_config == feature_config

    def test_prepare_training_data_from_store_no_store(
        self,
        mock_catalog: MagicMock,
    ) -> None:
        """
        Test prepare_training_data_from_store raises when no FeatureStore.
        """
        builder = TFTDatasetBuilder(
            catalog=mock_catalog,
            symbols=["AAPL"],
        )

        with pytest.raises(ValueError, match="FeatureStore not configured"):
            builder.prepare_training_data_from_store()

    @pytest.mark.usefixtures("clean_postgres_db")
    @patch("ml.data.tft_dataset_builder.bars_to_dataframe")
    def test_prepare_training_data_from_store_success(
        self,
        mock_bars_to_dataframe: MagicMock,
        mock_catalog: MagicMock,
        mock_feature_store: MagicMock,
        test_database: TestDatabase,
    ) -> None:
        """
        Test successful data preparation from FeatureStore.
        """
        # Setup mock bars data
        mock_bars = pl.DataFrame(
            {
                "ts_event": np.arange(100, dtype=np.int64) * int(1e9),
                "open": np.random.randn(100),
                "high": np.random.randn(100),
                "low": np.random.randn(100),
                "close": np.random.randn(100),
                "volume": np.random.randint(1000, 10000, 100),
            },
        )
        mock_bars_to_dataframe.return_value = mock_bars

        builder = TFTDatasetBuilder(
            catalog=mock_catalog,
            symbols=["AAPL"],
            feature_store=mock_feature_store,
        )

        # Execute
        result = builder.prepare_training_data_from_store(
            instrument_ids=["AAPL.NASDAQ"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
        )

        # Verify
        assert isinstance(result, pl.DataFrame)
        assert len(result) > 0
        assert "instrument_id" in result.columns
        assert "time_index" in result.columns
        assert "y" in result.columns  # Target column

        # Verify FeatureStore was called
        mock_feature_store.get_training_data.assert_called_once()
        call_args = mock_feature_store.get_training_data.call_args
        assert call_args[1]["instrument_id"] == "AAPL.NASDAQ"

    @pytest.mark.usefixtures("clean_postgres_db")
    @patch("ml.data.tft_dataset_builder.bars_to_dataframe")
    def test_prepare_training_data_from_store_no_features(
        self,
        mock_bars_to_dataframe: MagicMock,
        mock_catalog: MagicMock,
        mock_feature_store: MagicMock,
        test_database: TestDatabase,
    ) -> None:
        """
        Test handling when no features found in FeatureStore.
        """
        # Mock empty features
        mock_feature_store.get_training_data.return_value = (
            np.array([]),
            np.array([]),
            [],
        )

        builder = TFTDatasetBuilder(
            catalog=mock_catalog,
            symbols=["AAPL"],
            feature_store=mock_feature_store,
        )

        # Should raise RuntimeError when no features found
        with pytest.raises(RuntimeError, match="No features loaded from FeatureStore"):
            builder.prepare_training_data_from_store(
                instrument_ids=["AAPL.NASDAQ"],
            )

    @pytest.mark.usefixtures("clean_postgres_db")
    @patch("ml.data.tft_dataset_builder.bars_to_dataframe")
    def test_prepare_training_data_auto_selection_with_store(
        self,
        mock_bars_to_dataframe: MagicMock,
        mock_catalog: MagicMock,
        mock_feature_store: MagicMock,
        test_database: TestDatabase,
    ) -> None:
        """
        Test prepare_training_data automatically uses FeatureStore when available.
        """
        # Setup mock bars data
        mock_bars = pl.DataFrame(
            {
                "ts_event": np.arange(100, dtype=np.int64) * int(1e9),
                "open": np.random.randn(100),
                "high": np.random.randn(100),
                "low": np.random.randn(100),
                "close": np.random.randn(100),
                "volume": np.random.randint(1000, 10000, 100),
            },
        )
        mock_bars_to_dataframe.return_value = mock_bars

        builder = TFTDatasetBuilder(
            catalog=mock_catalog,
            symbols=["SPY"],
            feature_store=mock_feature_store,
        )

        # Execute
        result = builder.prepare_training_data()

        # Verify it used FeatureStore
        assert isinstance(result, pl.DataFrame)
        mock_feature_store.get_training_data.assert_called()

    def test_prepare_training_data_fallback_to_direct(
        self,
        mock_catalog: MagicMock,
    ) -> None:
        """
        Test prepare_training_data falls back to direct computation without
        FeatureStore.
        """
        builder = TFTDatasetBuilder(
            catalog=mock_catalog,
            symbols=["SPY"],
            feature_store=None,  # No FeatureStore
        )

        # Mock the direct computation method
        with patch.object(builder, "_build_training_dataset_direct") as mock_direct:
            mock_direct.return_value = pl.DataFrame({"test": [1, 2, 3]})

            result = builder.prepare_training_data()

            # Verify it used direct computation
            mock_direct.assert_called_once()
            assert isinstance(result, pl.DataFrame)

    @pytest.mark.usefixtures("clean_postgres_db")
    def test_build_training_dataset_uses_feature_store(
        self,
        mock_catalog: MagicMock,
        mock_feature_store: MagicMock,
        test_database: TestDatabase,
    ) -> None:
        """
        Test build_training_dataset method prefers FeatureStore when available.
        """
        builder = TFTDatasetBuilder(
            catalog=mock_catalog,
            symbols=["AAPL"],
            feature_store=mock_feature_store,
        )

        # Mock the FeatureStore method
        with patch.object(builder, "prepare_training_data_from_store") as mock_store_method:
            mock_store_method.return_value = pl.DataFrame({"from_store": [1, 2, 3]})

            result = builder.build_training_dataset()

            # Verify it tried to use FeatureStore
            mock_store_method.assert_called_once()
            assert isinstance(result, pl.DataFrame)

    @pytest.mark.usefixtures("clean_postgres_db")
    def test_build_training_dataset_fallback_on_error(
        self,
        mock_catalog: MagicMock,
        mock_feature_store: MagicMock,
        test_database: TestDatabase,
    ) -> None:
        """
        Test build_training_dataset falls back when FeatureStore fails.
        """
        builder = TFTDatasetBuilder(
            catalog=mock_catalog,
            symbols=["AAPL"],
            feature_store=mock_feature_store,
        )

        # Mock FeatureStore method to fail
        with patch.object(builder, "prepare_training_data_from_store") as mock_store_method:
            mock_store_method.side_effect = RuntimeError("FeatureStore error")

            # Mock direct method to succeed
            with patch.object(builder, "_build_training_dataset_direct") as mock_direct:
                mock_direct.return_value = pl.DataFrame({"direct": [1, 2, 3]})

                result = builder.build_training_dataset()

                # Verify fallback occurred
                mock_store_method.assert_called_once()
                mock_direct.assert_called_once()
                assert isinstance(result, pl.DataFrame)

    @pytest.mark.usefixtures("clean_postgres_db")
    def test_pandas_conversion(
        self,
        mock_catalog: MagicMock,
        mock_feature_store: MagicMock,
        test_database: TestDatabase,
    ) -> None:
        """
        Test conversion to pandas when use_polars=False.
        """
        builder = TFTDatasetBuilder(
            catalog=mock_catalog,
            symbols=["AAPL"],
            feature_store=mock_feature_store,
        )

        # Mock the FeatureStore method
        with patch.object(builder, "prepare_training_data_from_store") as mock_store_method:
            mock_store_method.return_value = pl.DataFrame({"test": [1, 2, 3]})

            result = builder.prepare_training_data(use_polars=False)

            # Verify pandas conversion
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3

    @pytest.mark.usefixtures("clean_postgres_db")
    def test_logging_feature_source(
        self,
        mock_catalog: MagicMock,
        mock_feature_store: MagicMock,
        caplog: pytest.LogCaptureFixture,
        test_database: TestDatabase,
    ) -> None:
        """
        Test that feature source is properly logged.
        """
        import logging

        builder = TFTDatasetBuilder(
            catalog=mock_catalog,
            symbols=["AAPL"],
            feature_store=mock_feature_store,
        )

        with patch.object(builder, "prepare_training_data_from_store") as mock_store_method:
            mock_store_method.return_value = pl.DataFrame({"test": [1, 2, 3]})

            # Set log level and clear logs
            caplog.set_level(logging.INFO)
            caplog.clear()

            # Execute
            builder.prepare_training_data()

            # Check logs
            assert "FeatureStore" in caplog.text
            assert "ensures training/inference parity" in caplog.text

    @pytest.mark.usefixtures("clean_postgres_db")
    def test_logging_fallback(
        self,
        mock_catalog: MagicMock,
        mock_feature_store: MagicMock,
        caplog: pytest.LogCaptureFixture,
        test_database: TestDatabase,
    ) -> None:
        """
        Test that fallback is properly logged.
        """
        import logging

        builder = TFTDatasetBuilder(
            catalog=mock_catalog,
            symbols=["AAPL"],
            feature_store=mock_feature_store,
        )

        with patch.object(builder, "prepare_training_data_from_store") as mock_store_method:
            mock_store_method.side_effect = RuntimeError("Test error")

            with patch.object(builder, "_build_training_dataset_direct") as mock_direct:
                mock_direct.return_value = pl.DataFrame({"test": [1]})

                # Set log level and clear logs
                caplog.set_level(logging.WARNING)
                caplog.clear()

                # Execute
                builder.prepare_training_data()

                # Check logs
                assert "Failed to load from FeatureStore" in caplog.text
                assert "Falling back to direct computation" in caplog.text
