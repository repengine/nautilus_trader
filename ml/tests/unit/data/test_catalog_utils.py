"""
Tests for catalog utilities using property-based testing.

This module tests the catalog utility functions that replace MLDataLoader with direct
ParquetDataCatalog usage.

"""

from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from unittest.mock import MagicMock
from unittest.mock import Mock

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ml._imports import HAS_POLARS
from ml._imports import pl
from ml.data.catalog_utils import bars_to_dataframe
from ml.data.catalog_utils import quotes_to_dataframe
from ml.data.catalog_utils import trades_to_dataframe
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


# Skip tests if Polars not available
pytestmark = pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")


# Hypothesis strategies
instrument_id_strategy = st.text(
    alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    min_size=3,
    max_size=6,
).map(lambda s: f"{s}.SIM")

datetime_strategy = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2025, 12, 31),
)

price_strategy = st.floats(min_value=0.00001, max_value=10000.0, allow_nan=False)
volume_strategy = st.integers(min_value=1, max_value=1000000)


@pytest.mark.property
@pytest.mark.parallel_safe
@pytest.mark.unit
class TestCatalogUtils:
    """
    Test catalog utility functions.
    """

    def setup_method(self) -> None:
        """
        Set up test fixtures.
        """
        self.mock_catalog = MagicMock(spec=ParquetDataCatalog)

    def _create_mock_bar(
        self,
        instrument_id: str,
        timestamp: int,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: int,
    ) -> Mock:
        """
        Create a mock Bar object.
        """
        bar = Mock(spec=Bar)
        bar_type = Mock(spec=BarType)
        bar_type.instrument_id = InstrumentId.from_str(instrument_id)
        bar.bar_type = bar_type
        bar.ts_event = timestamp
        bar.open = Price.from_str(str(open_price))
        bar.high = Price.from_str(str(high_price))
        bar.low = Price.from_str(str(low_price))
        bar.close = Price.from_str(str(close_price))
        bar.volume = Quantity.from_int(volume)
        return bar

    def _create_mock_quote(
        self,
        instrument_id: str,
        timestamp: int,
        bid: float,
        ask: float,
        bid_size: int,
        ask_size: int,
    ) -> Mock:
        """
        Create a mock QuoteTick object.
        """
        quote = Mock(spec=QuoteTick)
        quote.instrument_id = InstrumentId.from_str(instrument_id)
        quote.ts_event = timestamp
        quote.bid_price = Price.from_str(str(bid))
        quote.ask_price = Price.from_str(str(ask))
        quote.bid_size = Quantity.from_int(bid_size)
        quote.ask_size = Quantity.from_int(ask_size)
        return quote

    def _create_mock_trade(
        self,
        instrument_id: str,
        timestamp: int,
        price: float,
        size: int,
        aggressor_side: str,
    ) -> Mock:
        """
        Create a mock TradeTick object.
        """
        trade = Mock(spec=TradeTick)
        trade.instrument_id = InstrumentId.from_str(instrument_id)
        trade.ts_event = timestamp
        trade.price = Price.from_str(str(price))
        trade.size = Quantity.from_int(size)
        trade.aggressor_side = aggressor_side
        return trade

    # ============================================================================
    # bars_to_dataframe tests
    # ============================================================================

    def test_bars_to_dataframe_empty(self) -> None:
        """
        Test bars_to_dataframe with no data.
        """
        self.mock_catalog.bars.return_value = []

        result = bars_to_dataframe(
            self.mock_catalog,
            ["EURUSD.SIM"],
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0
        assert set(result.columns) == {
            "instrument_id",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        }

    @given(
        instrument_ids=st.lists(instrument_id_strategy, min_size=1, max_size=5),
        num_bars=st.integers(min_value=1, max_value=100),
    )
    def test_bars_to_dataframe_with_data(
        self,
        instrument_ids: list[str],
        num_bars: int,
    ) -> None:
        """
        Test bars_to_dataframe with data using property-based testing.
        """
        # Create mock bars
        bars = []
        base_timestamp = 1672531200000000000  # 2023-01-01 in nanoseconds

        for i in range(num_bars):
            instrument_id = instrument_ids[i % len(instrument_ids)]
            timestamp = base_timestamp + i * 60_000_000_000  # 1 minute intervals

            # Ensure high >= low and high >= close/open, low <= close/open
            low = 1.0 + i * 0.001
            high = low + 0.002
            open_price = low + 0.001
            close_price = low + 0.0015

            bar = self._create_mock_bar(
                instrument_id=instrument_id,
                timestamp=timestamp,
                open_price=open_price,
                high_price=high,
                low_price=low,
                close_price=close_price,
                volume=1000 + i,
            )
            bars.append(bar)

        self.mock_catalog.bars.return_value = bars

        result = bars_to_dataframe(
            self.mock_catalog,
            instrument_ids,
        )

        # Property checks
        assert isinstance(result, pl.DataFrame)
        assert len(result) == num_bars
        assert result["timestamp"].is_sorted()  # Timestamps should be ordered
        assert (result["high"] >= result["low"]).all()  # High >= Low invariant
        assert (result["high"] >= result["close"]).all()  # High >= Close
        assert (result["low"] <= result["close"]).all()  # Low <= Close
        assert (result["volume"] > 0).all()  # Volume is positive

    # ============================================================================
    # quotes_to_dataframe tests
    # ============================================================================

    def test_quotes_to_dataframe_empty(self) -> None:
        """
        Test quotes_to_dataframe with no data.
        """
        self.mock_catalog.quote_ticks.return_value = []

        result = quotes_to_dataframe(
            self.mock_catalog,
            ["EURUSD.SIM"],
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0
        assert set(result.columns) == {
            "instrument_id",
            "timestamp",
            "bid",
            "ask",
            "bid_size",
            "ask_size",
        }

    @given(
        instrument_ids=st.lists(instrument_id_strategy, min_size=1, max_size=5),
        num_quotes=st.integers(min_value=1, max_value=100),
    )
    def test_quotes_to_dataframe_with_data(
        self,
        instrument_ids: list[str],
        num_quotes: int,
    ) -> None:
        """
        Test quotes_to_dataframe with data using property-based testing.
        """
        # Create mock quotes
        quotes = []
        base_timestamp = 1672531200000000000

        for i in range(num_quotes):
            instrument_id = instrument_ids[i % len(instrument_ids)]
            timestamp = base_timestamp + i * 1_000_000_000  # 1 second intervals

            # Ensure bid < ask (spread is positive)
            bid = 1.0999 + i * 0.0001
            ask = bid + 0.0002  # 2 pip spread

            quote = self._create_mock_quote(
                instrument_id=instrument_id,
                timestamp=timestamp,
                bid=bid,
                ask=ask,
                bid_size=100 + i,
                ask_size=100 + i,
            )
            quotes.append(quote)

        self.mock_catalog.quote_ticks.return_value = quotes

        result = quotes_to_dataframe(
            self.mock_catalog,
            instrument_ids,
        )

        # Property checks
        assert isinstance(result, pl.DataFrame)
        assert len(result) == num_quotes
        assert result["timestamp"].is_sorted()
        assert (result["ask"] > result["bid"]).all()  # Spread is positive
        assert (result["bid_size"] > 0).all()  # Sizes are positive
        assert (result["ask_size"] > 0).all()

    # ============================================================================
    # trades_to_dataframe tests
    # ============================================================================

    def test_trades_to_dataframe_empty(self) -> None:
        """
        Test trades_to_dataframe with no data.
        """
        self.mock_catalog.trade_ticks.return_value = []

        result = trades_to_dataframe(
            self.mock_catalog,
            ["EURUSD.SIM"],
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0
        assert set(result.columns) == {
            "instrument_id",
            "timestamp",
            "price",
            "size",
            "aggressor_side",
        }

    @given(
        instrument_ids=st.lists(instrument_id_strategy, min_size=1, max_size=5),
        num_trades=st.integers(min_value=1, max_value=100),
    )
    def test_trades_to_dataframe_with_data(
        self,
        instrument_ids: list[str],
        num_trades: int,
    ) -> None:
        """
        Test trades_to_dataframe with data using property-based testing.
        """
        # Create mock trades
        trades = []
        base_timestamp = 1672531200000000000

        for i in range(num_trades):
            instrument_id = instrument_ids[i % len(instrument_ids)]
            timestamp = base_timestamp + i * 500_000_000  # 0.5 second intervals

            trade = self._create_mock_trade(
                instrument_id=instrument_id,
                timestamp=timestamp,
                price=1.1000 + i * 0.0001,
                size=100 + i * 10,
                aggressor_side="BUYER" if i % 2 == 0 else "SELLER",
            )
            trades.append(trade)

        self.mock_catalog.trade_ticks.return_value = trades

        result = trades_to_dataframe(
            self.mock_catalog,
            instrument_ids,
        )

        # Property checks
        assert isinstance(result, pl.DataFrame)
        assert len(result) == num_trades
        assert result["timestamp"].is_sorted()
        assert (result["price"] > 0).all()  # Prices are positive
        assert (result["size"] > 0).all()  # Sizes are positive
        assert result["aggressor_side"].is_in(["BUYER", "SELLER"]).all()

    # ============================================================================
    # Date range tests
    # ============================================================================

    @given(
        start_date=datetime_strategy,
        end_date=datetime_strategy,
    )
    def test_date_range_filtering(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        """
        Test that date range is passed correctly to catalog.
        """
        # Reset mock for each hypothesis example
        self.mock_catalog.reset_mock()

        # Ensure end > start
        if end_date <= start_date:
            end_date = start_date + timedelta(days=1)

        self.mock_catalog.bars.return_value = []

        bars_to_dataframe(
            self.mock_catalog,
            ["EURUSD.SIM"],
            start=start_date,
            end=end_date,
        )

        # Verify catalog was called with correct parameters
        self.mock_catalog.bars.assert_called_once()
        call_kwargs = self.mock_catalog.bars.call_args.kwargs
        assert call_kwargs["start"] == start_date
        assert call_kwargs["end"] == end_date

    # ============================================================================
    # Error handling tests
    # ============================================================================

    def test_invalid_instrument_id_format(self) -> None:
        """
        Test handling of invalid instrument ID format.
        """
        with pytest.raises(ValueError):
            bars_to_dataframe(
                self.mock_catalog,
                ["INVALID"],  # Missing venue suffix
            )
