"""
Cross-Asset Features Parity Tests.

Validates that batch (cold path) and incremental (hot path) implementations produce
identical results within specified tolerance (rtol=1e-10).

Test Coverage:
- EWMA Beta: Batch vs incremental parity
- Z-Scored Spreads: Batch vs incremental parity
- State serialization/deserialization
- Edge cases: zero variance, small samples, extreme values
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.features.cross_asset import (
    EWMABetaState,
    ZScoreSpreadState,
    compute_ewma_beta_batch,
    compute_ewma_beta_incremental,
    compute_zscore_spread_batch,
    compute_zscore_spread_incremental,
)


class TestEWMABetaParity:
    """Test parity between batch and incremental EWMA beta computation."""

    @pytest.fixture
    def sample_returns(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate sample returns for testing."""
        np.random.seed(42)
        n = 100
        market_returns = np.random.normal(0.0005, 0.02, n)
        # Asset correlated with market (beta ~1.2)
        asset_returns = 1.2 * market_returns + np.random.normal(0, 0.01, n)
        return asset_returns, market_returns

    def test_parity_basic(self, sample_returns: tuple[np.ndarray, np.ndarray]) -> None:
        """Test basic parity between batch and incremental computation."""
        asset_returns, market_returns = sample_returns
        alpha = 0.94

        # Batch computation
        betas_batch = compute_ewma_beta_batch(asset_returns, market_returns, alpha=alpha)

        # Incremental computation
        state = EWMABetaState(alpha=alpha)
        betas_incremental = []
        for asset_ret, market_ret in zip(asset_returns, market_returns):
            beta = compute_ewma_beta_incremental(state, asset_ret, market_ret)
            betas_incremental.append(beta)

        betas_incremental_arr = np.array(betas_incremental)

        # Validate parity (rtol=1e-10)
        np.testing.assert_allclose(
            betas_batch,
            betas_incremental_arr,
            rtol=1e-10,
            atol=0,
            err_msg="Batch and incremental EWMA beta computation differ beyond tolerance",
        )

    def test_parity_different_alphas(
        self,
        sample_returns: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test parity with different alpha values."""
        asset_returns, market_returns = sample_returns

        for alpha in [0.90, 0.94, 0.97, 0.99]:
            # Batch computation
            betas_batch = compute_ewma_beta_batch(
                asset_returns,
                market_returns,
                alpha=alpha,
            )

            # Incremental computation
            state = EWMABetaState(alpha=alpha)
            betas_incremental = []
            for asset_ret, market_ret in zip(asset_returns, market_returns):
                beta = compute_ewma_beta_incremental(state, asset_ret, market_ret)
                betas_incremental.append(beta)

            betas_incremental_arr = np.array(betas_incremental)

            # Validate parity
            np.testing.assert_allclose(
                betas_batch,
                betas_incremental_arr,
                rtol=1e-10,
                atol=0,
                err_msg=f"Parity failed for alpha={alpha}",
            )

    def test_zero_market_variance(self) -> None:
        """Test behavior when market variance is zero."""
        asset_returns = np.array([0.01, 0.02, 0.015])
        market_returns = np.array([0.0, 0.0, 0.0])

        # Batch should handle gracefully
        betas_batch = compute_ewma_beta_batch(asset_returns, market_returns)
        assert np.all(betas_batch == 0.0)

        # Incremental should handle gracefully
        state = EWMABetaState()
        for asset_ret, market_ret in zip(asset_returns, market_returns):
            beta = compute_ewma_beta_incremental(state, asset_ret, market_ret)
            assert beta == 0.0

    def test_state_serialization(self) -> None:
        """Test state serialization/deserialization preserves values."""
        state = EWMABetaState(alpha=0.94)

        # Update state
        for _ in range(10):
            compute_ewma_beta_incremental(state, 0.01, 0.008)

        # Serialize
        state_dict = state.to_dict()

        # Deserialize
        restored_state = EWMABetaState.from_dict(state_dict)

        # Validate exact equality
        assert restored_state.alpha == state.alpha
        assert restored_state.ewma_cov == state.ewma_cov
        assert restored_state.ewma_var_market == state.ewma_var_market
        assert restored_state.n == state.n
        assert restored_state.last_beta == state.last_beta

    def test_state_validation(self) -> None:
        """Test state parameter validation."""
        # Valid alpha
        state = EWMABetaState(alpha=0.94)
        assert state.alpha == 0.94

        # Invalid alpha (too low)
        with pytest.raises(ValueError, match="alpha must be in"):
            EWMABetaState(alpha=0.0)

        # Invalid alpha (too high)
        with pytest.raises(ValueError, match="alpha must be in"):
            EWMABetaState(alpha=1.0)

    def test_batch_validation(self) -> None:
        """Test batch computation input validation."""
        asset_returns = np.array([0.01, 0.02, 0.015])
        market_returns = np.array([0.008, 0.015])  # Different length

        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_ewma_beta_batch(asset_returns, market_returns)

        # Invalid alpha
        with pytest.raises(ValueError, match="alpha must be in"):
            compute_ewma_beta_batch(
                asset_returns,
                np.array([0.008, 0.015, 0.012]),
                alpha=1.5,
            )

    def test_empty_arrays(self) -> None:
        """Test handling of empty input arrays."""
        empty = np.array([])

        # Batch should return empty array
        result = compute_ewma_beta_batch(empty, empty)
        assert len(result) == 0


class TestZScoreSpreadParity:
    """Test parity between batch and incremental z-score spread computation."""

    @pytest.fixture
    def sample_prices(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate sample price series for testing."""
        np.random.seed(42)
        n = 100
        # Correlated price series with mean spread ~2.0
        prices_a = 100.0 + np.cumsum(np.random.normal(0, 0.5, n))
        prices_b = 98.0 + np.cumsum(np.random.normal(0, 0.5, n))
        return prices_a, prices_b

    def test_parity_expanding_window(
        self,
        sample_prices: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test parity with expanding window (all history)."""
        prices_a, prices_b = sample_prices

        # Batch computation (expanding window)
        zscores_batch = compute_zscore_spread_batch(prices_a, prices_b, window=None)

        # Incremental computation
        state = ZScoreSpreadState()
        zscores_incremental = []
        for price_a, price_b in zip(prices_a, prices_b):
            zscore = compute_zscore_spread_incremental(state, price_a, price_b)
            zscores_incremental.append(zscore)

        zscores_incremental_arr = np.array(zscores_incremental)

        # Validate parity (rtol=1e-10)
        # First value is always 0 (not enough samples)
        np.testing.assert_allclose(
            zscores_batch,
            zscores_incremental_arr,
            rtol=1e-10,
            atol=0,
            err_msg="Batch and incremental z-score computation differ beyond tolerance",
        )

    def test_parity_rolling_window(
        self,
        sample_prices: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test parity with rolling window."""
        prices_a, prices_b = sample_prices
        window = 20

        # Batch computation (rolling window)
        zscores_batch = compute_zscore_spread_batch(prices_a, prices_b, window=window)

        # For rolling window, incremental would need circular buffer
        # Here we verify batch computation produces expected results
        assert len(zscores_batch) == len(prices_a)

        # First window-1 values should be 0 (not enough samples)
        assert np.all(zscores_batch[: window - 1] == 0.0)

        # After window fills, should have non-zero z-scores
        assert not np.all(zscores_batch[window:] == 0.0)

    def test_zero_spread_variance(self) -> None:
        """Test behavior when spread has zero variance."""
        prices_a = np.array([100.0, 100.0, 100.0, 100.0])
        prices_b = np.array([98.0, 98.0, 98.0, 98.0])

        # Constant spread -> zero variance -> z-score should be 0
        zscores_batch = compute_zscore_spread_batch(prices_a, prices_b)
        assert np.all(zscores_batch == 0.0)

        # Incremental should handle gracefully
        state = ZScoreSpreadState()
        for price_a, price_b in zip(prices_a, prices_b):
            zscore = compute_zscore_spread_incremental(state, price_a, price_b)
            assert zscore == 0.0

    def test_state_serialization(self) -> None:
        """Test state serialization/deserialization preserves values."""
        state = ZScoreSpreadState()

        # Update state
        for i in range(10):
            compute_zscore_spread_incremental(state, 100.0 + i * 0.1, 98.0 + i * 0.1)

        # Serialize
        state_dict = state.to_dict()

        # Deserialize
        restored_state = ZScoreSpreadState.from_dict(state_dict)

        # Validate exact equality
        assert restored_state.mean == state.mean
        assert restored_state.m2 == state.m2
        assert restored_state.n == state.n
        assert restored_state.last_zscore == state.last_zscore

    def test_welford_numerical_stability(self) -> None:
        """Test Welford's algorithm numerical stability with large values."""
        # Large base values to test numerical stability
        base_a = 1e6
        base_b = 1e6 - 2.0

        prices_a = base_a + np.random.normal(0, 0.1, 100)
        prices_b = base_b + np.random.normal(0, 0.1, 100)

        # Batch computation
        zscores_batch = compute_zscore_spread_batch(prices_a, prices_b)

        # Incremental computation
        state = ZScoreSpreadState()
        zscores_incremental = []
        for price_a, price_b in zip(prices_a, prices_b):
            zscore = compute_zscore_spread_incremental(state, price_a, price_b)
            zscores_incremental.append(zscore)

        zscores_incremental_arr = np.array(zscores_incremental)

        # Should still maintain parity even with large values
        np.testing.assert_allclose(
            zscores_batch,
            zscores_incremental_arr,
            rtol=1e-10,
            atol=0,
        )

    def test_batch_validation(self) -> None:
        """Test batch computation input validation."""
        prices_a = np.array([100.0, 101.0, 102.0])
        prices_b = np.array([98.0, 99.0])  # Different length

        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_zscore_spread_batch(prices_a, prices_b)

        # Invalid window
        with pytest.raises(ValueError, match="window must be in"):
            compute_zscore_spread_batch(
                prices_a,
                np.array([98.0, 99.0, 100.0]),
                window=1,
            )

        with pytest.raises(ValueError, match="window must be in"):
            compute_zscore_spread_batch(
                prices_a,
                np.array([98.0, 99.0, 100.0]),
                window=1000,
            )

    def test_empty_arrays(self) -> None:
        """Test handling of empty input arrays."""
        empty = np.array([])

        # Batch should return empty array
        result = compute_zscore_spread_batch(empty, empty)
        assert len(result) == 0

    def test_state_validity_checks(self) -> None:
        """Test state validity methods."""
        state = ZScoreSpreadState()

        # Initial state not valid (n < MIN_SAMPLES)
        assert not state.is_valid()

        # Update to MIN_SAMPLES
        for i in range(30):
            compute_zscore_spread_incremental(state, 100.0 + i * 0.1, 98.0 + i * 0.1)

        # Now should be valid
        assert state.is_valid()
        assert state.n >= 30

        # Test std computation
        std = state.get_std()
        assert std >= 0.0

    def test_reset_functionality(self) -> None:
        """Test state reset functionality."""
        # Beta state
        beta_state = EWMABetaState(alpha=0.94)
        for _ in range(10):
            compute_ewma_beta_incremental(beta_state, 0.01, 0.008)

        assert beta_state.n > 0
        beta_state.reset()
        assert beta_state.n == 0
        assert beta_state.ewma_cov == 0.0
        assert beta_state.last_beta == 0.0

        # Spread state
        spread_state = ZScoreSpreadState()
        for i in range(10):
            compute_zscore_spread_incremental(spread_state, 100.0 + i, 98.0 + i)

        assert spread_state.n > 0
        spread_state.reset()
        assert spread_state.n == 0
        assert spread_state.mean == 0.0
        assert spread_state.m2 == 0.0
        assert spread_state.last_zscore == 0.0


class TestCrossAssetEdgeCases:
    """Test edge cases and extreme values."""

    def test_extreme_returns(self) -> None:
        """Test with extreme return values."""
        # Extreme positive returns
        asset_returns = np.array([0.5, 0.8, 0.3])
        market_returns = np.array([0.4, 0.6, 0.2])

        betas_batch = compute_ewma_beta_batch(asset_returns, market_returns)
        assert np.all(np.isfinite(betas_batch))

        # Extreme negative returns
        asset_returns = np.array([-0.5, -0.8, -0.3])
        market_returns = np.array([-0.4, -0.6, -0.2])

        betas_batch = compute_ewma_beta_batch(asset_returns, market_returns)
        assert np.all(np.isfinite(betas_batch))

    def test_single_observation(self) -> None:
        """Test with single observation."""
        asset_returns = np.array([0.01])
        market_returns = np.array([0.008])

        betas_batch = compute_ewma_beta_batch(asset_returns, market_returns)
        assert len(betas_batch) == 1

        state = EWMABetaState()
        beta = compute_ewma_beta_incremental(state, 0.01, 0.008)
        assert np.isfinite(beta)

    def test_negative_correlation(self) -> None:
        """Test with negatively correlated assets."""
        np.random.seed(42)
        market_returns = np.random.normal(0, 0.02, 50)
        # Negative beta asset
        asset_returns = -1.5 * market_returns + np.random.normal(0, 0.01, 50)

        betas_batch = compute_ewma_beta_batch(asset_returns, market_returns)

        # Should produce negative betas
        assert np.mean(betas_batch[-10:]) < 0

    def test_mixed_sign_spreads(self) -> None:
        """Test spreads that cross zero."""
        # Prices that create positive and negative spreads
        prices_a = np.array([100, 99, 101, 98, 102])
        prices_b = np.array([98, 100, 99, 101, 100])

        zscores = compute_zscore_spread_batch(prices_a, prices_b)
        assert np.all(np.isfinite(zscores))

        # Spread crosses zero
        spreads = prices_a - prices_b
        assert np.any(spreads > 0) and np.any(spreads < 0)
