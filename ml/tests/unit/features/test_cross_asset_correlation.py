"""
Parity tests for correlation feature.

Validates that batch and incremental implementations produce identical results
within strict numerical tolerance (rtol=1e-10).
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.features.cross_asset.correlation import (
    compute_correlation_batch,
    compute_correlation_incremental,
)
from ml.features.cross_asset.state import CorrelationState


class TestCorrelationParity:
    """Test batch vs incremental parity for correlation."""

    def test_parity_basic(self) -> None:
        """Batch and incremental produce identical results."""
        # Generate synthetic correlated data
        np.random.seed(42)
        x = np.random.randn(100) * 0.01
        y = x * 0.7 + np.random.randn(100) * 0.003  # Correlated (r ≈ 0.7)

        # Batch computation
        batch_corr = compute_correlation_batch(x, y, window_size=100)

        # Incremental computation
        state = CorrelationState(window_size=100)
        incremental_corr = []
        for i in range(len(x)):
            corr = compute_correlation_incremental(state, x[i], y[i])
            incremental_corr.append(corr)

        incremental_corr_arr = np.array(incremental_corr)

        # Assert parity (strict tolerance)
        np.testing.assert_allclose(batch_corr, incremental_corr_arr, rtol=1e-10)

    def test_parity_negative_correlation(self) -> None:
        """Batch and incremental agree for negative correlation."""
        np.random.seed(123)
        x = np.random.randn(80) * 0.02
        y = -x * 0.8 + np.random.randn(80) * 0.002  # Negative correlation

        # Batch
        batch_corr = compute_correlation_batch(x, y, window_size=80)

        # Incremental
        state = CorrelationState(window_size=80)
        incremental_corr = []
        for i in range(len(x)):
            corr = compute_correlation_incremental(state, x[i], y[i])
            incremental_corr.append(corr)

        # Assert parity
        np.testing.assert_allclose(batch_corr, incremental_corr, rtol=1e-10)

        # Verify negative correlation detected
        assert batch_corr[-1] < -0.5, "Should detect strong negative correlation"

    def test_parity_zero_variance(self) -> None:
        """Both implementations handle zero variance gracefully."""
        # X is constant (zero variance)
        x = np.ones(50) * 100.0
        y = np.random.randn(50)

        # Batch
        batch_corr = compute_correlation_batch(x, y, window_size=50)

        # Incremental
        state = CorrelationState(window_size=50)
        incremental_corr = []
        for i in range(len(x)):
            corr = compute_correlation_incremental(state, x[i], y[i])
            incremental_corr.append(corr)

        # Assert parity
        np.testing.assert_allclose(batch_corr, incremental_corr, rtol=1e-10)

        # Should return 0.0 for undefined correlation
        assert batch_corr[-1] == 0.0
        assert incremental_corr[-1] == 0.0

    def test_parity_small_window(self) -> None:
        """Parity holds for small window sizes."""
        np.random.seed(456)
        x = np.random.randn(30)
        y = x * 0.6 + np.random.randn(30) * 0.5

        window = 10

        # Batch
        batch_corr = compute_correlation_batch(x, y, window_size=window)

        # Incremental
        state = CorrelationState(window_size=window)
        incremental_corr = []
        for i in range(len(x)):
            corr = compute_correlation_incremental(state, x[i], y[i])
            incremental_corr.append(corr)

        # Assert parity
        np.testing.assert_allclose(batch_corr, incremental_corr, rtol=1e-10)

    def test_parity_perfect_correlation(self) -> None:
        """Both implementations detect perfect correlation."""
        x = np.linspace(0, 10, 50)
        y = x * 2.0 + 1.0  # Perfect linear relationship

        # Batch
        batch_corr = compute_correlation_batch(x, y, window_size=50)

        # Incremental
        state = CorrelationState(window_size=50)
        incremental_corr = []
        for i in range(len(x)):
            corr = compute_correlation_incremental(state, x[i], y[i])
            incremental_corr.append(corr)

        # Assert parity
        np.testing.assert_allclose(batch_corr, incremental_corr, rtol=1e-10)

        # Should approach 1.0 (perfect positive correlation)
        assert abs(batch_corr[-1] - 1.0) < 1e-6

    def test_state_serialization(self) -> None:
        """State can be serialized and restored."""
        np.random.seed(789)
        x = np.random.randn(20)
        y = x * 0.5 + np.random.randn(20)

        # Build state incrementally
        state = CorrelationState(window_size=20)
        for i in range(10):
            compute_correlation_incremental(state, x[i], y[i])

        # Serialize
        state_dict = state.to_dict()

        # Create new state from dict
        restored_state = CorrelationState.from_dict(state_dict)

        # Verify state matches
        assert restored_state.n == state.n
        assert restored_state.mean_x == state.mean_x
        assert restored_state.mean_y == state.mean_y
        assert restored_state.m2_x == state.m2_x
        assert restored_state.m2_y == state.m2_y
        assert restored_state.m2_xy == state.m2_xy
        assert restored_state.window_size == state.window_size
        assert restored_state.last_correlation == state.last_correlation

        # Continue with restored state
        for i in range(10, 20):
            compute_correlation_incremental(restored_state, x[i], y[i])

        # Compare with fresh state
        fresh_state = CorrelationState(window_size=20)
        for i in range(20):
            compute_correlation_incremental(fresh_state, x[i], y[i])

        # Should match
        assert abs(restored_state.last_correlation - fresh_state.last_correlation) < 1e-10

    def test_incremental_updates_state(self) -> None:
        """Incremental function updates state correctly."""
        state = CorrelationState(window_size=10)

        assert state.n == 0
        assert state.mean_x == 0.0
        assert state.mean_y == 0.0

        # First update
        corr1 = compute_correlation_incremental(state, 1.0, 2.0)
        assert state.n == 1
        assert corr1 == 0.0  # Not enough samples

        # Second update
        corr2 = compute_correlation_incremental(state, 2.0, 4.0)
        assert state.n == 2
        assert corr2 != 0.0  # Now we have enough samples

        # State accumulates
        assert state.mean_x != 0.0
        assert state.mean_y != 0.0

    def test_batch_empty_input(self) -> None:
        """Batch computation handles empty input."""
        x = np.array([])
        y = np.array([])

        result = compute_correlation_batch(x, y, window_size=10)

        assert len(result) == 0
        assert result.dtype == np.float64

    def test_batch_shape_mismatch(self) -> None:
        """Batch computation raises on shape mismatch."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_correlation_batch(x, y, window_size=10)

    def test_invalid_window_size(self) -> None:
        """Batch computation validates window size."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="window_size must be >= 2"):
            compute_correlation_batch(x, y, window_size=1)

    def test_state_validation(self) -> None:
        """CorrelationState validates parameters."""
        with pytest.raises(ValueError, match="window_size must be >= 2"):
            CorrelationState(window_size=1)

    def test_state_is_valid(self) -> None:
        """State validity check works correctly."""
        from ml.features.cross_asset.state import MIN_SAMPLES

        state = CorrelationState(window_size=100)

        # Initially invalid
        assert not state.is_valid()

        # Process some samples
        for i in range(MIN_SAMPLES - 1):
            compute_correlation_incremental(state, float(i), float(i) * 2.0)

        # Still invalid
        assert not state.is_valid()

        # One more sample
        compute_correlation_incremental(state, 1.0, 2.0)

        # Now valid
        assert state.is_valid()

    def test_state_reset(self) -> None:
        """State reset works correctly."""
        state = CorrelationState(window_size=50)

        # Process some data
        for i in range(20):
            compute_correlation_incremental(state, float(i), float(i) * 1.5)

        assert state.n > 0
        assert state.mean_x != 0.0

        # Reset
        state.reset()

        # Verify reset
        assert state.n == 0
        assert state.mean_x == 0.0
        assert state.mean_y == 0.0
        assert state.m2_x == 0.0
        assert state.m2_y == 0.0
        assert state.m2_xy == 0.0
        assert state.last_correlation == 0.0

    def test_get_correlation_method(self) -> None:
        """State.get_correlation() computes correctly."""
        state = CorrelationState(window_size=50)

        # Insufficient data
        assert state.get_correlation() == 0.0

        # Add some data
        x_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_vals = [2.0, 4.0, 6.0, 8.0, 10.0]  # Perfect correlation

        for x, y in zip(x_vals, y_vals):
            compute_correlation_incremental(state, x, y)

        # Should be close to 1.0
        corr = state.get_correlation()
        assert abs(corr - 1.0) < 1e-6

    def test_parity_with_financial_returns(self) -> None:
        """Test parity with realistic financial return data."""
        # Simulate daily returns for two correlated assets
        np.random.seed(2024)
        n_days = 252  # One trading year

        # Asset 1: Mean return 0.05%, std 1.5%
        returns_1 = np.random.normal(0.0005, 0.015, n_days)

        # Asset 2: Correlated with asset 1 (beta ≈ 0.8)
        beta = 0.8
        idiosyncratic = np.random.normal(0, 0.01, n_days)
        returns_2 = beta * returns_1 + idiosyncratic

        # Batch
        batch_corr = compute_correlation_batch(returns_1, returns_2, window_size=252)

        # Incremental
        state = CorrelationState(window_size=252)
        incremental_corr = []
        for i in range(n_days):
            corr = compute_correlation_incremental(state, returns_1[i], returns_2[i])
            incremental_corr.append(corr)

        # Assert parity
        np.testing.assert_allclose(batch_corr, incremental_corr, rtol=1e-10)

        # Final correlation should be positive and significant
        final_corr = batch_corr[-1]
        assert 0.3 < final_corr < 0.9, f"Expected reasonable correlation, got {final_corr}"
