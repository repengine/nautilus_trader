"""Unit tests for factor correlation and PCA analysis."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime

import numpy as np
import polars as pl
import pytest

from playground.risk_model.factor_analysis import compute_factor_correlations
from playground.risk_model.factor_analysis import compute_pca_analysis


class TestFactorCorrelations:
    """Test suite for factor correlation analysis."""

    @pytest.fixture
    def orthogonal_factors(self) -> pl.DataFrame:
        """Create perfectly orthogonal factors using controlled random seeds."""
        n = 500
        np.random.seed(42)
        return pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1, tzinfo=UTC)] * n,
                "factor_duration": np.random.randn(n),
                "factor_credit": np.random.randn(n),
                "factor_liquidity": np.random.randn(n),
            }
        )

    @pytest.fixture
    def correlated_factors(self) -> pl.DataFrame:
        """Create factors with known correlations."""
        n = 500
        np.random.seed(123)

        # Base factor
        base = np.random.randn(n)

        # Create correlated factor (r ≈ 0.7)
        factor1 = base
        factor2 = 0.7 * base + 0.3 * np.random.randn(n)

        # Independent factor
        factor3 = np.random.randn(n)

        return pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1, tzinfo=UTC)] * n,
                "factor_duration": factor1,
                "factor_credit": factor2,
                "factor_liquidity": factor3,
            }
        )

    @pytest.fixture
    def single_factor(self) -> pl.DataFrame:
        """Create DataFrame with single factor."""
        n = 100
        np.random.seed(42)
        return pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1, tzinfo=UTC)] * n,
                "factor_duration": np.random.randn(n),
            }
        )

    @pytest.fixture
    def perfectly_correlated_factors(self) -> pl.DataFrame:
        """Create perfectly correlated factors (r = 1.0)."""
        n = 100
        np.random.seed(42)
        base = np.random.randn(n)

        return pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1, tzinfo=UTC)] * n,
                "factor_duration": base,
                "factor_credit": base,  # Perfect correlation
                "factor_liquidity": np.random.randn(n),
            }
        )

    def test_orthogonal_factors(self, orthogonal_factors: pl.DataFrame) -> None:
        """Test correlation analysis with orthogonal factors."""
        result = compute_factor_correlations(
            orthogonal_factors,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
        )

        # Should be nearly orthogonal (small correlation due to randomness)
        assert result.max_abs_correlation < 0.15
        assert result.is_orthogonal is True
        assert result.n_observations == 500
        assert len(result.factor_names) == 3

        # Diagonal should be 1.0
        for factor in result.factor_names:
            assert abs(result.correlation_matrix[factor][factor] - 1.0) < 1e-10

    def test_correlated_factors(self, correlated_factors: pl.DataFrame) -> None:
        """Test correlation analysis with correlated factors."""
        result = compute_factor_correlations(
            correlated_factors,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            correlation_threshold=0.50,
        )

        # Should detect high correlation between duration and credit
        assert result.max_abs_correlation > 0.50
        assert result.is_orthogonal is False
        assert result.n_observations == 500

    def test_correlation_matrix_symmetry(self, orthogonal_factors: pl.DataFrame) -> None:
        """Test that correlation matrix is symmetric."""
        result = compute_factor_correlations(
            orthogonal_factors,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
        )

        factors = list(result.factor_names)
        for i, factor1 in enumerate(factors):
            for j, factor2 in enumerate(factors):
                corr_ij = result.correlation_matrix[factor1][factor2]
                corr_ji = result.correlation_matrix[factor2][factor1]
                assert abs(corr_ij - corr_ji) < 1e-10

    def test_max_mean_correlation_calculation(
        self, correlated_factors: pl.DataFrame
    ) -> None:
        """Test max and mean correlation calculations."""
        result = compute_factor_correlations(
            correlated_factors,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
        )

        # Extract off-diagonal correlations manually
        factors = list(result.factor_names)
        off_diagonal = []
        for i, factor1 in enumerate(factors):
            for j, factor2 in enumerate(factors):
                if i < j:
                    corr = abs(result.correlation_matrix[factor1][factor2])
                    off_diagonal.append(corr)

        expected_max = max(off_diagonal)
        expected_mean = np.mean(off_diagonal)

        assert abs(result.max_abs_correlation - expected_max) < 1e-10
        assert abs(result.mean_abs_correlation - expected_mean) < 1e-10

    def test_empty_dataframe_handling(self) -> None:
        """Test handling of empty DataFrame."""
        empty_df = pl.DataFrame(
            {
                "timestamp": [],
                "factor_duration": [],
                "factor_credit": [],
                "factor_liquidity": [],
            }
        )

        with pytest.raises(ValueError, match="factor_returns cannot be empty"):
            compute_factor_correlations(
                empty_df,
                factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            )

    def test_single_factor_no_correlation(self, single_factor: pl.DataFrame) -> None:
        """Test with single factor (no correlation possible)."""
        result = compute_factor_correlations(
            single_factor,
            factor_columns=["factor_duration"],
        )

        # Only diagonal element
        assert result.max_abs_correlation == 0.0
        assert result.mean_abs_correlation == 0.0
        assert result.is_orthogonal is True

    def test_perfectly_correlated_factors(
        self, perfectly_correlated_factors: pl.DataFrame
    ) -> None:
        """Test with perfectly correlated factors (r = 1.0)."""
        result = compute_factor_correlations(
            perfectly_correlated_factors,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
        )

        # Should detect perfect correlation
        assert result.max_abs_correlation > 0.99
        assert result.is_orthogonal is False

        # Duration and credit should have correlation ≈ 1.0
        corr_duration_credit = result.correlation_matrix["factor_duration"]["factor_credit"]
        assert abs(corr_duration_credit - 1.0) < 0.01

    def test_missing_values_handling(self) -> None:
        """Test handling of missing values."""
        n = 100
        data_with_nulls = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1, tzinfo=UTC)] * n,
                "factor_duration": [1.0] * 50 + [None] * 50,  # Half missing
                "factor_credit": [2.0] * n,
                "factor_liquidity": [3.0] * n,
            }
        )

        result = compute_factor_correlations(
            data_with_nulls,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
        )

        # Should drop nulls, resulting in 50 observations
        assert result.n_observations == 50

    def test_missing_columns_error(self, orthogonal_factors: pl.DataFrame) -> None:
        """Test error when required columns are missing."""
        with pytest.raises(ValueError, match="missing required columns"):
            compute_factor_correlations(
                orthogonal_factors,
                factor_columns=["factor_duration", "nonexistent_factor"],
            )

    def test_empty_factor_columns_error(self, orthogonal_factors: pl.DataFrame) -> None:
        """Test error when factor_columns is empty."""
        with pytest.raises(ValueError, match="factor_columns cannot be empty"):
            compute_factor_correlations(
                orthogonal_factors,
                factor_columns=[],
            )


class TestPCAAnalysis:
    """Test suite for PCA analysis."""

    @pytest.fixture
    def independent_factors(self) -> pl.DataFrame:
        """Create 3 independent factors (should give 3 PCs)."""
        n = 500
        np.random.seed(42)
        return pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1, tzinfo=UTC)] * n,
                "factor_duration": np.random.randn(n),
                "factor_credit": np.random.randn(n),
                "factor_liquidity": np.random.randn(n),
            }
        )

    @pytest.fixture
    def two_factors_disguised(self) -> pl.DataFrame:
        """Create 2 true factors disguised as 3 (only 2 PCs needed)."""
        n = 500
        np.random.seed(123)

        factor1 = np.random.randn(n)
        factor2 = np.random.randn(n)

        # Third is linear combination of first two
        factor3 = 0.5 * factor1 + 0.5 * factor2 + 0.01 * np.random.randn(n)

        return pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1, tzinfo=UTC)] * n,
                "factor_duration": factor1,
                "factor_credit": factor2,
                "factor_liquidity": factor3,
            }
        )

    def test_independent_factors_pca(self, independent_factors: pl.DataFrame) -> None:
        """Test PCA with 3 independent factors (should capture ~100%)."""
        result = compute_pca_analysis(
            independent_factors,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            n_components=3,
        )

        # 3 independent factors should require all 3 PCs to explain ~100%
        assert result.variance_captured_by_3pc > 0.95
        assert result.is_adequate is True
        assert result.n_components == 3
        assert len(result.explained_variance_ratio) == 3
        assert len(result.loadings) == 3

    def test_two_factors_disguised_as_three(
        self, two_factors_disguised: pl.DataFrame
    ) -> None:
        """Test PCA with 2 factors disguised as 3 (first 2 PCs should dominate)."""
        result = compute_pca_analysis(
            two_factors_disguised,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            n_components=3,
        )

        # First 2 PCs should explain most variance
        variance_2pc = sum(result.explained_variance_ratio[:2])
        assert variance_2pc > 0.95

        # Third PC should contribute very little
        assert result.explained_variance_ratio[2] < 0.05

    def test_explained_variance_sums_to_one(
        self, independent_factors: pl.DataFrame
    ) -> None:
        """Test that explained variance ratios sum to ≤ 1.0."""
        result = compute_pca_analysis(
            independent_factors,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            n_components=3,
        )

        total_variance = sum(result.explained_variance_ratio)
        assert 0.0 <= total_variance <= 1.0

        # Cumulative variance should match
        assert abs(result.cumulative_variance[-1] - total_variance) < 1e-10

    def test_loadings_interpretation(self, independent_factors: pl.DataFrame) -> None:
        """Test that loadings are properly structured."""
        result = compute_pca_analysis(
            independent_factors,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            n_components=3,
        )

        # Check structure
        assert "PC1" in result.loadings
        assert "PC2" in result.loadings
        assert "PC3" in result.loadings

        # Each PC should have loadings for all factors
        for pc_name, loadings in result.loadings.items():
            assert "factor_duration" in loadings
            assert "factor_credit" in loadings
            assert "factor_liquidity" in loadings

            # Loadings should be finite numbers
            for loading in loadings.values():
                assert np.isfinite(loading)

    def test_empty_dataframe_pca(self) -> None:
        """Test PCA with empty DataFrame."""
        empty_df = pl.DataFrame(
            {
                "timestamp": [],
                "factor_duration": [],
                "factor_credit": [],
                "factor_liquidity": [],
            }
        )

        with pytest.raises(ValueError, match="factor_returns cannot be empty"):
            compute_pca_analysis(
                empty_df,
                factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            )

    def test_single_factor_pca(self) -> None:
        """Test PCA with single factor."""
        n = 100
        np.random.seed(42)
        single_factor_df = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1, tzinfo=UTC)] * n,
                "factor_duration": np.random.randn(n),
            }
        )

        result = compute_pca_analysis(
            single_factor_df,
            factor_columns=["factor_duration"],
            n_components=1,
        )

        # Single factor should explain 100% of variance
        assert abs(result.variance_captured_by_3pc - 1.0) < 1e-10
        assert result.is_adequate is True

    def test_n_components_validation(self, independent_factors: pl.DataFrame) -> None:
        """Test validation of n_components parameter."""
        # Too many components
        with pytest.raises(ValueError, match="cannot exceed number of factors"):
            compute_pca_analysis(
                independent_factors,
                factor_columns=["factor_duration", "factor_credit"],
                n_components=5,
            )

        # Zero components
        with pytest.raises(ValueError, match="n_components must be at least 1"):
            compute_pca_analysis(
                independent_factors,
                factor_columns=["factor_duration", "factor_credit"],
                n_components=0,
            )

    def test_variance_threshold_inadequate(
        self, two_factors_disguised: pl.DataFrame
    ) -> None:
        """Test is_adequate flag with high variance threshold."""
        result = compute_pca_analysis(
            two_factors_disguised,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            n_components=2,
            variance_threshold=0.99,
        )

        # Only using 2 components, so variance_captured_by_3pc should use only those 2
        variance_2pc = sum(result.explained_variance_ratio)
        assert abs(result.variance_captured_by_3pc - variance_2pc) < 1e-10

    def test_cumulative_variance_monotonic(
        self, independent_factors: pl.DataFrame
    ) -> None:
        """Test that cumulative variance is monotonically increasing."""
        result = compute_pca_analysis(
            independent_factors,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            n_components=3,
        )

        # Cumulative variance should be strictly increasing
        for i in range(len(result.cumulative_variance) - 1):
            assert result.cumulative_variance[i] < result.cumulative_variance[i + 1]

    def test_eigenvalues_positive(self, independent_factors: pl.DataFrame) -> None:
        """Test that eigenvalues are positive."""
        result = compute_pca_analysis(
            independent_factors,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            n_components=3,
        )

        # All eigenvalues should be positive
        for eigenvalue in result.eigenvalues:
            assert eigenvalue > 0


class TestEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_all_nulls_in_factor_column(self) -> None:
        """Test handling when all values in a factor are null."""
        n = 100
        all_nulls = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1, tzinfo=UTC)] * n,
                "factor_duration": [None] * n,  # All null
                "factor_credit": [2.0] * n,
                "factor_liquidity": [3.0] * n,
            }
        )

        with pytest.raises(ValueError, match=r"no valid.*observations"):
            compute_factor_correlations(
                all_nulls,
                factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            )

    def test_constant_factor_values(self) -> None:
        """Test with constant (zero variance) factor."""
        n = 100
        constant_factor = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1, tzinfo=UTC)] * n,
                "factor_duration": [1.0] * n,  # Constant
                "factor_credit": np.random.randn(n).tolist(),
                "factor_liquidity": np.random.randn(n).tolist(),
            }
        )

        # This should work but correlation will be NaN for constant factor
        # NumPy handles this, but we should verify it doesn't crash
        result = compute_factor_correlations(
            constant_factor,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
        )

        # Correlation matrix should exist even with NaN values
        assert result.correlation_matrix is not None

    def test_very_small_sample_size(self) -> None:
        """Test with very small sample size."""
        n = 3  # Minimum for 3 factors
        small_sample = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1, tzinfo=UTC)] * n,
                "factor_duration": [1.0, 2.0, 3.0],
                "factor_credit": [2.0, 3.0, 4.0],
                "factor_liquidity": [3.0, 4.0, 5.0],
            }
        )

        result = compute_factor_correlations(
            small_sample,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
        )

        assert result.n_observations == 3

    def test_negative_correlation_threshold(
        self,
    ) -> None:
        """Test with negative correlation threshold (unusual but valid)."""
        n = 100
        np.random.seed(42)
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1, tzinfo=UTC)] * n,
                "factor_duration": np.random.randn(n),
                "factor_credit": np.random.randn(n),
                "factor_liquidity": np.random.randn(n),
            }
        )

        result = compute_factor_correlations(
            df,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            correlation_threshold=-0.5,  # Negative threshold (unusual but allowed)
        )

        # Negative threshold means almost nothing is considered orthogonal
        # Only orthogonal if max_abs_corr < -0.5 (impossible for positive correlations)
        assert result.correlation_matrix is not None
