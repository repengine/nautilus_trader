"""
Unit tests for metadata provider and sources.

Tests instrument metadata loading, caching, and validation.

"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import polars as pl
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from ml.data.providers.metadata import InstrumentMetadataProvider
from ml.data.sources.metadata import CSVMetadataSource
from ml.data.sources.metadata import DatabentoMetadataSource
from ml.data.sources.metadata import MetadataSource
from ml.data.sources.metadata import MockMetadataSource
from ml.data.sources.metadata import NautilusMetadataSource


class TestMockMetadataSource:
    """
    Test mock metadata source.
    """

    def test_mock_source_generates_metadata(self) -> None:
        """
        Test that mock source generates valid metadata.
        """
        source = MockMetadataSource(seed=42)
        instruments = ["AAPL", "GOOGL", "SPY", "TSLA"]

        df = source.fetch_metadata(instruments)

        # Check all instruments present
        assert set(df["instrument_id"].to_list()) == set(instruments)
        assert len(df) == len(instruments)

        # Check required columns
        required_cols = {
            "instrument_id",
            "tick_size",
            "lot_size",
            "exchange",
            "asset_class",
            "currency",
        }
        assert required_cols.issubset(df.columns)

        # Check data types
        assert df["tick_size"].dtype == pl.Float64
        assert df["lot_size"].dtype == pl.Float64
        assert df["instrument_id"].dtype == pl.Utf8

    def test_mock_source_deterministic(self) -> None:
        """
        Test that mock source is deterministic with same seed.
        """
        source1 = MockMetadataSource(seed=123)
        source2 = MockMetadataSource(seed=123)

        instruments = ["AAPL", "GOOGL", "SPY"]

        df1 = source1.fetch_metadata(instruments)
        df2 = source2.fetch_metadata(instruments)

        # Should be identical
        assert df1.equals(df2)

    def test_mock_source_etf_detection(self) -> None:
        """
        Test that mock source correctly identifies ETFs.
        """
        source = MockMetadataSource()

        # Known ETFs
        etfs = ["SPY", "QQQ", "IWM", "VTI", "SOMETHINGETF"]
        df = source.fetch_metadata(etfs)

        # SPY, QQQ, IWM, VTI should be ETFs
        spy_row = df.filter(pl.col("instrument_id") == "SPY")[0]
        assert spy_row["asset_class"][0] == "ETF"

        etf_row = df.filter(pl.col("instrument_id") == "SOMETHINGETF")[0]
        assert etf_row["asset_class"][0] == "ETF"

    @given(
        instruments=st.lists(
            st.text(
                min_size=1,
                max_size=10,
                alphabet=st.characters(min_codepoint=65, max_codepoint=90),
            ),
            min_size=1,
            max_size=100,
        ),
    )
    @settings(max_examples=10)
    def test_mock_source_handles_any_symbols(self, instruments: list[str]) -> None:
        """
        Test mock source handles arbitrary symbols.
        """
        source = MockMetadataSource()
        df = source.fetch_metadata(instruments)

        # Should return data for all requested
        assert len(df) == len(instruments)

        # All should have valid tick sizes
        assert (df["tick_size"] > 0).all()
        assert (df["lot_size"] > 0).all()


class TestCSVMetadataSource:
    """
    Test CSV metadata source.
    """

    def test_csv_source_loads_from_file(self, tmp_path: Path) -> None:
        """
        Test loading metadata from CSV file.
        """
        # Create test CSV
        csv_path = tmp_path / "metadata.csv"
        test_data = pl.DataFrame(
            {
                "instrument_id": ["AAPL", "GOOGL", "MSFT"],
                "tick_size": [0.01, 0.01, 0.01],
                "lot_size": [100.0, 100.0, 100.0],
                "exchange": ["XNAS", "XNAS", "XNAS"],
                "asset_class": ["EQUITY", "EQUITY", "EQUITY"],
                "currency": ["USD", "USD", "USD"],
                "contract_size": [1.0, 1.0, 1.0],
                "min_price_increment": [0.01, 0.01, 0.01],
                "margin_initial": [0.25, 0.25, 0.25],
                "margin_maintenance": [0.25, 0.25, 0.25],
                "fee_class": ["MAKER_TAKER", "MAKER_TAKER", "MAKER_TAKER"],
                "market_segment": ["MAIN", "MAIN", "MAIN"],
            },
        )
        test_data.write_csv(csv_path)

        # Load with source
        source = CSVMetadataSource(csv_path)
        df = source.fetch_metadata(["AAPL", "GOOGL"])

        assert len(df) == 2
        assert set(df["instrument_id"].to_list()) == {"AAPL", "GOOGL"}

    def test_csv_source_handles_missing_file(self) -> None:
        """
        Test CSV source handles missing file gracefully.
        """
        source = CSVMetadataSource("nonexistent.csv")

        # Should fall back to mock
        df = source.fetch_metadata(["AAPL"])
        assert len(df) == 1
        assert df["instrument_id"][0] == "AAPL"

    def test_csv_source_adds_missing_instruments(self, tmp_path: Path) -> None:
        """
        Test CSV source adds defaults for missing instruments.
        """
        # Create CSV with only AAPL
        csv_path = tmp_path / "metadata.csv"
        test_data = pl.DataFrame(
            {
                "instrument_id": ["AAPL"],
                "tick_size": [0.01],
                "lot_size": [100.0],
                "exchange": ["XNAS"],
                "asset_class": ["EQUITY"],
                "currency": ["USD"],
            },
        )
        test_data.write_csv(csv_path)

        source = CSVMetadataSource(csv_path)

        # Request AAPL and GOOGL
        df = source.fetch_metadata(["AAPL", "GOOGL"])

        # Should have both
        assert len(df) == 2
        assert set(df["instrument_id"].to_list()) == {"AAPL", "GOOGL"}

        # GOOGL should have default values
        googl_row = df.filter(pl.col("instrument_id") == "GOOGL")[0]
        assert googl_row["tick_size"][0] > 0  # Should have some default


class TestDatabentoMetadataSource:
    """
    Test Databento metadata source.
    """

    def test_databento_source_without_key(self) -> None:
        """
        Test Databento source falls back when no API key.
        """
        with patch.dict("os.environ", {}, clear=True):
            source = DatabentoMetadataSource()
            df = source.fetch_metadata(["AAPL"])

            # Should fall back to mock
            assert len(df) == 1
            assert df["instrument_id"][0] == "AAPL"

    def test_databento_source_with_api_key(self) -> None:
        """
        Test Databento source with API key (uses mock fallback).
        """
        source = DatabentoMetadataSource(api_key="test_key")

        df = source.fetch_metadata(["AAPL", "GOOGL"])

        # Should return data (even if using mock due to missing databento)
        assert len(df) == 2
        assert set(df["instrument_id"].to_list()) == {"AAPL", "GOOGL"}


class TestNautilusMetadataSource:
    """
    Test Nautilus metadata source.
    """

    def test_nautilus_source_with_instruments(self) -> None:
        """
        Test Nautilus source with pre-loaded instruments.
        """
        # Create mock instruments
        mock_instruments = {}

        # Mock AAPL instrument
        aapl = MagicMock()
        aapl.id.symbol = "AAPL"
        aapl.id.venue = "XNAS"
        aapl.price_increment = 0.01
        aapl.lot_size = 100
        aapl.asset_class = "EQUITY"
        aapl.quote_currency = "USD"
        mock_instruments["AAPL"] = aapl

        source = NautilusMetadataSource(instruments=mock_instruments)
        df = source.fetch_metadata(["AAPL", "GOOGL"])

        # Should have both (GOOGL with defaults)
        assert len(df) == 2

        # AAPL should have real values
        aapl_row = df.filter(pl.col("instrument_id") == "AAPL")[0]
        assert aapl_row["exchange"][0] == "XNAS"
        assert aapl_row["tick_size"][0] == 0.01

        # GOOGL should have defaults
        googl_row = df.filter(pl.col("instrument_id") == "GOOGL")[0]
        assert googl_row["exchange"][0] == "UNKNOWN"


class TestInstrumentMetadataProvider:
    """
    Test the main metadata provider.
    """

    def test_provider_loads_and_caches(self) -> None:
        """
        Test provider loads data and uses cache.
        """
        mock_source = MockMetadataSource()
        provider = InstrumentMetadataProvider(mock_source)

        # First call
        df1 = provider.load_metadata(["AAPL", "GOOGL"])
        assert len(df1) == 2

        # Second call (should use cache)
        with patch.object(mock_source, "fetch_metadata") as mock_fetch:
            df2 = provider.load_metadata(["AAPL", "GOOGL"])
            # Should not call source again
            mock_fetch.assert_not_called()

        assert df1.equals(df2)

    def test_provider_validates_data(self) -> None:
        """
        Test provider validates loaded data.
        """
        # Create source that returns invalid data
        mock_source = MagicMock(spec=MetadataSource)
        mock_source.fetch_metadata.return_value = pl.DataFrame(
            {
                "instrument_id": ["AAPL"],
                "tick_size": [-0.01],  # Invalid: negative
                "lot_size": [100.0],
                "exchange": ["XNAS"],
                "asset_class": ["EQUITY"],
                "currency": ["USD"],
            },
        )

        provider = InstrumentMetadataProvider(mock_source)
        df = provider.load_metadata(["AAPL"])

        # Should return default frame due to validation failure
        assert df["tick_size"][0] == 0.01  # Default value

    def test_provider_handles_empty_data(self) -> None:
        """
        Test provider handles empty data from source.
        """
        mock_source = MagicMock(spec=MetadataSource)
        mock_source.fetch_metadata.return_value = pl.DataFrame()

        provider = InstrumentMetadataProvider(mock_source)
        df = provider.load_metadata(["AAPL"])

        # Should return default frame
        assert len(df) == 1
        assert df["instrument_id"][0] == "AAPL"

    def test_provider_handles_source_errors(self) -> None:
        """
        Test provider handles source errors gracefully.
        """
        mock_source = MagicMock(spec=MetadataSource)
        mock_source.fetch_metadata.side_effect = Exception("API Error")

        provider = InstrumentMetadataProvider(mock_source)
        df = provider.load_metadata(["AAPL", "GOOGL"])

        # Should return default frame
        assert len(df) == 2
        assert set(df["instrument_id"].to_list()) == {"AAPL", "GOOGL"}

    def test_provider_schema(self) -> None:
        """
        Test provider returns correct schema.
        """
        provider = InstrumentMetadataProvider(MockMetadataSource())
        schema = provider.get_schema()

        # Check key fields
        assert schema["instrument_id"] is str
        assert schema["tick_size"] is float
        assert schema["lot_size"] is float
        assert schema["exchange"] is str

    def test_provider_ensures_all_instruments(self) -> None:
        """
        Test provider ensures all requested instruments are returned.
        """
        # Source that only returns AAPL
        mock_source = MagicMock(spec=MetadataSource)
        mock_source.fetch_metadata.return_value = pl.DataFrame(
            {
                "instrument_id": ["AAPL"],
                "tick_size": [0.01],
                "lot_size": [100.0],
                "exchange": ["XNAS"],
                "asset_class": ["EQUITY"],
                "currency": ["USD"],
            },
        )

        provider = InstrumentMetadataProvider(mock_source)
        df = provider.load_metadata(["AAPL", "GOOGL", "MSFT"])

        # Should have all three
        assert len(df) == 3
        assert set(df["instrument_id"].to_list()) == {"AAPL", "GOOGL", "MSFT"}

        # GOOGL and MSFT should have defaults
        googl_row = df.filter(pl.col("instrument_id") == "GOOGL")[0]
        assert googl_row["exchange"][0] == "UNKNOWN"

    @given(
        instruments=st.lists(
            st.text(min_size=3, max_size=6, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            min_size=1,
            max_size=20,
            unique=True,
        ),
    )
    @settings(max_examples=10)
    def test_provider_handles_arbitrary_instruments(self, instruments: list[str]) -> None:
        """Property test: provider handles any instrument list."""
        provider = InstrumentMetadataProvider(MockMetadataSource())
        df = provider.load_metadata(instruments)

        # Should return data for all
        assert len(df) == len(instruments)
        assert set(df["instrument_id"].to_list()) == set(instruments)

        # All should have valid data
        assert (df["tick_size"] > 0).all()
        assert (df["lot_size"] > 0).all()
        assert df["exchange"].null_count() == 0
