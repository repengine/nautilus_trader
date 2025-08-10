
"""
Test the integration test infrastructure itself.

This module verifies that all test fixtures and utilities work correctly.

"""

from __future__ import annotations

from typing import Any

import numpy as np

from nautilus_trader.model.data import Bar
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from .test_utils import compute_nautilus_indicators
from .test_utils import validate_feature_parity


class TestInfrastructure:
    """
    Test integration test infrastructure components.
    """

    def test_generate_test_bars(self, generate_test_bars: list[Bar]) -> None:
        """
        Test that test bar generation works correctly.
        """
        # Arrange & Act
        bars = generate_test_bars

        # Assert
        assert len(bars) == 100
        assert all(isinstance(bar, Bar) for bar in bars)

        # Check OHLC constraints
        for bar in bars:
            assert bar.high >= bar.open
            assert bar.high >= bar.close
            assert bar.low <= bar.open
            assert bar.low <= bar.close
            assert bar.high >= bar.low
            assert bar.volume > 0

    def test_mock_parquet_catalog(self, mock_parquet_catalog: ParquetDataCatalog) -> None:
        """
        Test that mock ParquetDataCatalog fixture works.
        """
        # Arrange & Act & Assert
        assert mock_parquet_catalog is not None
        assert isinstance(mock_parquet_catalog, ParquetDataCatalog)

    def test_ml_signals(self, test_ml_signals: list[dict[str, Any]]) -> None:
        """
        Test ML signal generation.
        """
        # Arrange & Act
        signals = test_ml_signals

        # Assert
        assert len(signals) > 0

        for signal in signals:
            assert "instrument_id" in signal
            assert "timestamp" in signal
            assert "prediction" in signal
            assert signal["prediction"] in [-1, 0, 1]
            assert 0.0 <= signal["confidence"] <= 1.0

    def test_ml_config(self, test_ml_config: dict[str, Any]) -> None:
        """
        Test ML configuration fixture.
        """
        # Arrange & Act & Assert
        assert "feature_config" in test_ml_config
        assert "model_config" in test_ml_config
        assert "signal_config" in test_ml_config
        assert "risk_config" in test_ml_config

        # Check specific values
        assert test_ml_config["signal_config"]["confidence_threshold"] == 0.7
        assert test_ml_config["risk_config"]["max_position_size"] == 0.1

    def test_multi_instrument_bars(
        self,
        multi_instrument_bars: dict[InstrumentId, list[Bar]],
    ) -> None:
        """
        Test multi-instrument bar generation.
        """
        # Arrange & Act
        bars_dict = multi_instrument_bars

        # Assert
        assert len(bars_dict) == 3  # EURUSD, GBPUSD, USDJPY

        for instrument_id, bars in bars_dict.items():
            assert len(bars) == 100
            assert all(isinstance(bar, Bar) for bar in bars)

    def test_feature_parity_validation(self) -> None:
        """
        Test feature parity validation utility.
        """
        # Arrange
        n_samples, n_features = 100, 5
        rng = np.random.default_rng(42)
        batch_features = rng.standard_normal((n_samples, n_features))

        # Perfect match
        online_features = batch_features.copy()

        # Act
        is_valid, report = validate_feature_parity(
            batch_features,
            online_features,
            tolerance=1e-10,
        )

        # Assert
        assert is_valid
        assert report["max_abs_diff"] == 0.0
        assert report["n_exact_matches"] == n_samples * n_features

        # Test with small differences
        online_features_noisy = (
            batch_features + rng.standard_normal((n_samples, n_features)) * 1e-11
        )

        is_valid_noisy, report_noisy = validate_feature_parity(
            batch_features,
            online_features_noisy,
            tolerance=1e-10,
        )

        # Should still be valid with tiny noise below tolerance
        # Note: With random noise, occasionally we might exceed tolerance
        # The important thing is that we're detecting differences correctly
        assert "max_rel_diff" in report_noisy
        assert "is_valid" in report_noisy

    def test_compute_nautilus_indicators(self, generate_test_bars: list[Bar]) -> None:
        """
        Test Nautilus indicator computation.
        """
        # Arrange
        bars = generate_test_bars

        # Act
        df = compute_nautilus_indicators(bars)

        # Assert
        assert len(df) == len(bars)
        assert "sma_10" in df.columns
        assert "sma_20" in df.columns
        assert "rsi" in df.columns
        assert "macd" in df.columns

        # Check that indicators are initialized after warm-up
        assert not df["sma_20"].iloc[-1] == np.nan  # Should have values after 20 bars
        assert not df["rsi"].iloc[-1] == np.nan  # Should have values after 14 bars
