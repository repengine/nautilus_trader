"""
Tests for the catalog utility functions.

This test module ensures comprehensive coverage of the catalog utilities that replaced
MLDataLoader, including data loading, error handling, and integration with Nautilus
components.

"""

from datetime import datetime
from unittest.mock import MagicMock
from unittest.mock import Mock

import pytest

from ml._imports import HAS_POLARS
from ml._imports import pl
from ml.data.catalog_utils import bars_to_dataframe
from ml.data.catalog_utils import quotes_to_dataframe
from ml.data.catalog_utils import trades_to_dataframe
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


pytestmark = pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")


@pytest.mark.parallel_safe
@pytest.mark.unit
class TestCatalogUtils:
    """
    Test cases for catalog utility functions.
    """

    def setup_method(self) -> None:
        """
        Set up test fixtures.
        """
        # Create mock catalog
        self.mock_catalog = MagicMock(spec=ParquetDataCatalog)

        # Create test instrument ID
        self.instrument_id = InstrumentId.from_str("EURUSD.SIM")

        # Create test timestamps
        self.start_time = datetime(2023, 1, 1)
        self.end_time = datetime(2023, 1, 2)

    def _create_mock_bar(
        self,
        instrument_id: InstrumentId,
        timestamp: int,
    ) -> Mock:
        """
        Create a mock Bar object.
        """
        bar = Mock(spec=Bar)
        bar_type = Mock(spec=BarType)
        bar_type.instrument_id = instrument_id
        bar.bar_type = bar_type
        bar.ts_event = timestamp
        bar.open = Price.from_str("1.0900")
        bar.high = Price.from_str("1.0910")
        bar.low = Price.from_str("1.0890")
        bar.close = Price.from_str("1.0905")
        bar.volume = Quantity.from_int(100000)
        return bar

    def _create_mock_quote(
        self,
        instrument_id: InstrumentId,
        timestamp: int,
    ) -> Mock:
        """
        Create a mock QuoteTick object.
        """
        quote = Mock(spec=QuoteTick)
        quote.instrument_id = instrument_id
        quote.ts_event = timestamp
        quote.bid_price = Price.from_str("1.0899")
        quote.ask_price = Price.from_str("1.0901")
        quote.bid_size = Quantity.from_int(100000)
        quote.ask_size = Quantity.from_int(100000)
        return quote

    def _create_mock_trade(
        self,
        instrument_id: InstrumentId,
        timestamp: int,
    ) -> Mock:
        """
        Create a mock TradeTick object.
        """
        trade = Mock(spec=TradeTick)
        trade.instrument_id = instrument_id
        trade.ts_event = timestamp
        trade.price = Price.from_str("1.0900")
        trade.size = Quantity.from_int(10000)
        trade.aggressor_side = AggressorSide.BUYER
        return trade

    def test_bars_to_dataframe_basic(self) -> None:
        """
        Test basic bar loading functionality.
        """
        # Setup mock bars
        bars = [
            self._create_mock_bar(self.instrument_id, 1672531200000000000 + i * 60_000_000_000)
            for i in range(10)
        ]
        self.mock_catalog.bars.return_value = bars

        # Load bars
        df = bars_to_dataframe(
            self.mock_catalog,
            ["EURUSD.SIM"],
            start=self.start_time,
            end=self.end_time,
        )

        # Assertions
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 10
        assert set(df.columns) == {
            "instrument_id",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        }
        assert df["instrument_id"][0] == "EURUSD.SIM"

        # Verify catalog was called correctly
        self.mock_catalog.bars.assert_called_once()
        call_args = self.mock_catalog.bars.call_args
        assert call_args.kwargs["start"] == self.start_time
        assert call_args.kwargs["end"] == self.end_time

    def test_bars_to_dataframe_empty(self) -> None:
        """
        Test handling of empty bar data.
        """
        self.mock_catalog.bars.return_value = []

        df = bars_to_dataframe(self.mock_catalog, ["EURUSD.SIM"])

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0
        assert set(df.columns) == {
            "instrument_id",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        }

    def test_quotes_to_dataframe_basic(self) -> None:
        """
        Test basic quote loading functionality.
        """
        # Setup mock quotes
        quotes = [
            self._create_mock_quote(self.instrument_id, 1672531200000000000 + i * 1_000_000_000)
            for i in range(10)
        ]
        self.mock_catalog.quote_ticks.return_value = quotes

        # Load quotes
        df = quotes_to_dataframe(
            self.mock_catalog,
            ["EURUSD.SIM"],
            start=self.start_time,
            end=self.end_time,
        )

        # Assertions
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 10
        assert set(df.columns) == {
            "instrument_id",
            "timestamp",
            "bid",
            "ask",
            "bid_size",
            "ask_size",
        }
        assert df["instrument_id"][0] == "EURUSD.SIM"

    def test_quotes_to_dataframe_empty(self) -> None:
        """
        Test handling of empty quote data.
        """
        self.mock_catalog.quote_ticks.return_value = []

        df = quotes_to_dataframe(self.mock_catalog, ["EURUSD.SIM"])

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0
        assert set(df.columns) == {
            "instrument_id",
            "timestamp",
            "bid",
            "ask",
            "bid_size",
            "ask_size",
        }

    def test_trades_to_dataframe_basic(self) -> None:
        """
        Test basic trade loading functionality.
        """
        # Setup mock trades
        trades = [
            self._create_mock_trade(self.instrument_id, 1672531200000000000 + i * 500_000_000)
            for i in range(10)
        ]
        self.mock_catalog.trade_ticks.return_value = trades

        # Load trades
        df = trades_to_dataframe(
            self.mock_catalog,
            ["EURUSD.SIM"],
            start=self.start_time,
            end=self.end_time,
        )

        # Assertions
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 10
        assert set(df.columns) == {"instrument_id", "timestamp", "price", "size", "aggressor_side"}
        assert df["instrument_id"][0] == "EURUSD.SIM"

    def test_trades_to_dataframe_empty(self) -> None:
        """
        Test handling of empty trade data.
        """
        self.mock_catalog.trade_ticks.return_value = []

        df = trades_to_dataframe(self.mock_catalog, ["EURUSD.SIM"])

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0
        assert set(df.columns) == {"instrument_id", "timestamp", "price", "size", "aggressor_side"}

    def test_multiple_instruments(self) -> None:
        """
        Test loading data for multiple instruments.
        """
        # Setup mock bars for multiple instruments
        eurusd_id = InstrumentId.from_str("EURUSD.SIM")
        gbpusd_id = InstrumentId.from_str("GBPUSD.SIM")

        bars = []
        for i in range(5):
            bars.append(self._create_mock_bar(eurusd_id, 1672531200000000000 + i * 60_000_000_000))
        for i in range(5):
            bars.append(self._create_mock_bar(gbpusd_id, 1672531200000000000 + i * 60_000_000_000))

        self.mock_catalog.bars.return_value = bars

        # Load bars for multiple instruments
        df = bars_to_dataframe(
            self.mock_catalog,
            ["EURUSD.SIM", "GBPUSD.SIM"],
        )

        # Assertions
        assert len(df) == 10
        assert set(df["instrument_id"].unique()) == {"EURUSD.SIM", "GBPUSD.SIM"}

    def test_invalid_instrument_id(self) -> None:
        """
        Test handling of invalid instrument ID format.
        """
        with pytest.raises(ValueError):
            bars_to_dataframe(
                self.mock_catalog,
                ["INVALID"],  # Missing venue suffix
            )

    def test_date_string_formats(self) -> None:
        """
        Test handling of date strings as start/end parameters.
        """
        self.mock_catalog.bars.return_value = []

        # Test with string dates
        df = bars_to_dataframe(
            self.mock_catalog,
            ["EURUSD.SIM"],
            start="2023-01-01",
            end="2023-01-02",
        )

        assert isinstance(df, pl.DataFrame)

        # Verify catalog was called with string dates
        call_args = self.mock_catalog.bars.call_args
        assert call_args.kwargs["start"] == "2023-01-01"
        assert call_args.kwargs["end"] == "2023-01-02"

    def test_data_integrity(self) -> None:
        """
        Test that data values are correctly converted.
        """
        # Create a single bar with specific values
        bar = Mock(spec=Bar)
        bar_type = Mock(spec=BarType)
        bar_type.instrument_id = self.instrument_id
        bar.bar_type = bar_type
        bar.ts_event = 1672531200000000000
        bar.open = Price.from_str("1.0900")
        bar.high = Price.from_str("1.0950")
        bar.low = Price.from_str("1.0850")
        bar.close = Price.from_str("1.0925")
        bar.volume = Quantity.from_int(123456)

        self.mock_catalog.bars.return_value = [bar]

        df = bars_to_dataframe(self.mock_catalog, ["EURUSD.SIM"])

        # Check values are correctly converted
        assert df["open"][0] == 1.0900
        assert df["high"][0] == 1.0950
        assert df["low"][0] == 1.0850
        assert df["close"][0] == 1.0925
        assert df["volume"][0] == 123456
        assert df["timestamp"][0] == 1672531200000000000
