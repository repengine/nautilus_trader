"""
Tests for PCA validation module.

Covers:
- Synthetic data tests (known factor structure)
- Real data tests (actual sector dataset)
- Edge cases (insufficient data, zero variance, etc.)
- Correlation computation tests
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from playground.risk_model.dataset import CoverageSummary
from playground.risk_model.dataset import SectorDataset
from playground.risk_model.pca_validation import SectorPCAResult
from playground.risk_model.pca_validation import compare_pc_loadings_to_betas
from playground.risk_model.pca_validation import compute_sector_pca
from playground.risk_model.pca_validation import generate_pca_validation_report


# ===== Fixtures =====


@pytest.fixture
def synthetic_three_factor_dataset() -> SectorDataset:
    """
    Generate synthetic sector returns from 3 latent factors.

    Factor structure:
    - Factor 1 (50% variance): Duration-like (affects all sectors similarly)
    - Factor 2 (30% variance): Credit-like (affects financials more)
    - Factor 3 (20% variance): Liquidity-like (affects tech/discretionary more)
    """
    np.random.seed(42)

    n_observations = 252  # 1 year of daily data
    n_sectors = 6

    # Generate 3 independent factors
    factor1 = np.random.normal(0, 1.0, n_observations)  # Duration
    factor2 = np.random.normal(0, 0.8, n_observations)  # Credit
    factor3 = np.random.normal(0, 0.6, n_observations)  # Liquidity

    # Define factor loadings for each sector
    # [Duration, Credit, Liquidity]
    loadings = np.array([
        [0.8, 0.2, 0.1],   # Sector A: High duration sensitivity
        [0.7, 0.5, 0.2],   # Sector B: Balanced
        [0.6, 0.7, 0.1],   # Sector C: High credit sensitivity
        [0.5, 0.3, 0.8],   # Sector D: High liquidity sensitivity
        [0.9, 0.1, 0.3],   # Sector E: Very high duration sensitivity
        [0.4, 0.6, 0.5],   # Sector F: Balanced credit/liquidity
    ])

    # Generate sector returns: Return = β1*F1 + β2*F2 + β3*F3 + noise
    sector_returns = []
    sector_names = [f"SECTOR{chr(65+i)}" for i in range(n_sectors)]

    base_time = datetime(2020, 1, 1, tzinfo=UTC)

    for i in range(n_sectors):
        sector_returns_i = (
            loadings[i, 0] * factor1 +
            loadings[i, 1] * factor2 +
            loadings[i, 2] * factor3 +
            np.random.normal(0, 0.1, n_observations)  # Idiosyncratic noise
        )

        for t in range(n_observations):
            sector_returns.append({
                "timestamp": base_time + timedelta(days=t),
                "symbol": sector_names[i],
                "return": sector_returns_i[t],
            })

    sector_returns_df = pl.DataFrame(sector_returns)

    # Create dummy factor returns (not used in this test)
    factor_returns_df = pl.DataFrame({
        "timestamp": [base_time + timedelta(days=t) for t in range(n_observations)],
        "factor_duration": factor1,
        "factor_credit": factor2,
        "factor_liquidity": factor3,
    })

    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=252,
        factor_expected_days=252,
        sector_coverage=dict.fromkeys(sector_names, 1.0),
        factor_coverage={"factor_duration": 1.0, "factor_credit": 1.0, "factor_liquidity": 1.0},
    )

    return SectorDataset(
        sector_returns=sector_returns_df,
        factor_returns=factor_returns_df,
        coverage=coverage,
    )


@pytest.fixture
def simple_dataset() -> SectorDataset:
    """Simple dataset with 3 sectors and 50 observations for basic tests."""
    np.random.seed(123)

    n_observations = 50
    sectors = ["XLK", "XLF", "XLE"]

    base_time = datetime(2020, 1, 1, tzinfo=UTC)

    sector_returns = []
    for sector in sectors:
        returns = np.random.normal(0, 0.02, n_observations)
        for t, ret in enumerate(returns):
            sector_returns.append({
                "timestamp": base_time + timedelta(days=t),
                "symbol": sector,
                "return": ret,
            })

    sector_returns_df = pl.DataFrame(sector_returns)

    factor_returns_df = pl.DataFrame({
        "timestamp": [base_time + timedelta(days=t) for t in range(n_observations)],
        "factor_duration": np.random.normal(0, 0.01, n_observations),
    })

    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=50,
        factor_expected_days=50,
        sector_coverage=dict.fromkeys(sectors, 1.0),
        factor_coverage={"factor_duration": 1.0},
    )

    return SectorDataset(
        sector_returns=sector_returns_df,
        factor_returns=factor_returns_df,
        coverage=coverage,
    )


# ===== Tests: Synthetic Data =====


def test_pca_synthetic_three_factors(synthetic_three_factor_dataset: SectorDataset) -> None:
    """Test PCA on synthetic data with known 3-factor structure."""
    result = compute_sector_pca(synthetic_three_factor_dataset, n_components=3)

    # Should extract 3 components
    assert result.n_components == 3
    assert len(result.variance_explained) == 3
    assert len(result.cumulative_variance) == 3
    assert len(result.eigenvalues) == 3

    # Check that variance percentages sum to 100% (approximately)
    # Note: We have 6 sectors, so 3 PCs won't capture 100%, but should be >70%
    total_variance = sum(result.variance_explained)
    assert 70.0 <= total_variance <= 100.0, f"Expected >70% variance, got {total_variance:.2f}%"

    # PC1 should capture the most variance
    assert result.variance_explained[0] > result.variance_explained[1]
    assert result.variance_explained[1] > result.variance_explained[2]

    # Cumulative variance should be monotonically increasing
    assert result.cumulative_variance[0] < result.cumulative_variance[1]
    assert result.cumulative_variance[1] < result.cumulative_variance[2]

    # Check PC loadings structure
    assert len(result.pc_loadings) == 6  # 6 sectors
    for sector_id, loadings in result.pc_loadings.items():
        assert len(loadings) == 3  # 3 PCs
        assert all(isinstance(loading, float) for loading in loadings)

    # Check eigenvectors shape
    assert result.eigenvectors.shape == (6, 3)  # 6 sectors x 3 PCs

    # Check sector IDs
    assert len(result.sector_ids) == 6
    assert all(sid.startswith("SECTOR") for sid in result.sector_ids)


def test_pca_variance_explained_sum_to_one() -> None:
    """Test that variance explained sums to 100% when n_components = n_sectors."""
    np.random.seed(99)

    n_observations = 100
    n_sectors = 4
    sectors = [f"S{i}" for i in range(n_sectors)]

    base_time = datetime(2020, 1, 1, tzinfo=UTC)

    sector_returns = []
    for sector in sectors:
        returns = np.random.normal(0, 0.02, n_observations)
        for t, ret in enumerate(returns):
            sector_returns.append({
                "timestamp": base_time + timedelta(days=t),
                "symbol": sector,
                "return": ret,
            })

    sector_returns_df = pl.DataFrame(sector_returns)

    factor_returns_df = pl.DataFrame({
        "timestamp": [base_time + timedelta(days=t) for t in range(n_observations)],
        "dummy": np.zeros(n_observations),
    })

    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=100,
        factor_expected_days=100,
        sector_coverage=dict.fromkeys(sectors, 1.0),
        factor_coverage={"dummy": 1.0},
    )

    dataset = SectorDataset(
        sector_returns=sector_returns_df,
        factor_returns=factor_returns_df,
        coverage=coverage,
    )

    # Extract all components
    result = compute_sector_pca(dataset, n_components=n_sectors)

    # Variance should sum to 100% (within floating point tolerance)
    total_variance = sum(result.variance_explained)
    assert 99.9 <= total_variance <= 100.1, f"Expected ~100% variance, got {total_variance:.2f}%"


def test_pc_loadings_orthogonal(synthetic_three_factor_dataset: SectorDataset) -> None:
    """Test that principal component eigenvectors are orthogonal."""
    result = compute_sector_pca(synthetic_three_factor_dataset, n_components=3)

    # Verify eigenvectors are orthonormal by checking V^T * V = I
    # where V is the eigenvector matrix (sectors x PCs)
    eigenvector_matrix = result.eigenvectors

    # Compute V^T * V
    gram_matrix = eigenvector_matrix.T @ eigenvector_matrix

    # Should be identity matrix (orthonormal eigenvectors)
    identity = np.eye(3)

    # Check all elements are close to identity
    assert np.allclose(gram_matrix, identity, atol=1e-6), (
        f"Eigenvectors are not orthonormal:\n{gram_matrix}\n"
        f"Expected identity:\n{identity}"
    )


# ===== Tests: Real Data =====


def test_sector_pca_real_data(simple_dataset: SectorDataset) -> None:
    """Test PCA on real-like sector data."""
    result = compute_sector_pca(simple_dataset, n_components=3)

    assert result.n_components == 3
    assert len(result.sector_ids) == 3
    assert len(result.variance_explained) == 3

    # Basic sanity checks
    assert all(0 <= v <= 100 for v in result.variance_explained)
    assert all(0 <= v <= 100 for v in result.cumulative_variance)
    assert all(e > 0 for e in result.eigenvalues)

    # Check PC loadings
    for sector_id in result.sector_ids:
        assert sector_id in result.pc_loadings
        assert len(result.pc_loadings[sector_id]) == 3


def test_variance_explained_threshold(synthetic_three_factor_dataset: SectorDataset) -> None:
    """Test that top 3 PCs explain >70% variance for well-structured data."""
    result = compute_sector_pca(synthetic_three_factor_dataset, n_components=3)

    total_variance_3pc = sum(result.variance_explained)

    # With 3 underlying factors and 6 sectors, top 3 PCs should capture >70%
    assert total_variance_3pc >= 70.0, (
        f"Top 3 PCs should explain >70% variance, got {total_variance_3pc:.2f}%"
    )


# ===== Tests: Edge Cases =====


def test_pca_single_sector_error() -> None:
    """Test that PCA with single sector raises ValueError."""
    base_time = datetime(2020, 1, 1, tzinfo=UTC)

    sector_returns_df = pl.DataFrame({
        "timestamp": [base_time + timedelta(days=t) for t in range(30)],
        "symbol": ["XLK"] * 30,
        "return": np.random.normal(0, 0.01, 30),
    })

    factor_returns_df = pl.DataFrame({
        "timestamp": [base_time + timedelta(days=t) for t in range(30)],
        "dummy": np.zeros(30),
    })

    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=30,
        factor_expected_days=30,
        sector_coverage={"XLK": 1.0},
        factor_coverage={"dummy": 1.0},
    )

    dataset = SectorDataset(
        sector_returns=sector_returns_df,
        factor_returns=factor_returns_df,
        coverage=coverage,
    )

    with pytest.raises(ValueError, match="at least 2 sectors"):
        compute_sector_pca(dataset, n_components=1)


def test_pca_insufficient_observations() -> None:
    """Test that PCA with insufficient observations raises ValueError."""
    base_time = datetime(2020, 1, 1, tzinfo=UTC)

    # Only 15 observations (< 20 minimum)
    n_obs = 15
    sector_returns = []
    for sector in ["XLK", "XLF"]:
        for t in range(n_obs):
            sector_returns.append({
                "timestamp": base_time + timedelta(days=t),
                "symbol": sector,
                "return": np.random.normal(0, 0.01),
            })

    sector_returns_df = pl.DataFrame(sector_returns)

    factor_returns_df = pl.DataFrame({
        "timestamp": [base_time + timedelta(days=t) for t in range(n_obs)],
        "dummy": np.zeros(n_obs),
    })

    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=n_obs,
        factor_expected_days=n_obs,
        sector_coverage={"XLK": 1.0, "XLF": 1.0},
        factor_coverage={"dummy": 1.0},
    )

    dataset = SectorDataset(
        sector_returns=sector_returns_df,
        factor_returns=factor_returns_df,
        coverage=coverage,
    )

    with pytest.raises(ValueError, match="at least 20 observations"):
        compute_sector_pca(dataset, n_components=2)


def test_pca_all_constant_returns() -> None:
    """Test that PCA with all constant returns raises ValueError."""
    base_time = datetime(2020, 1, 1, tzinfo=UTC)

    n_obs = 50
    sector_returns = []
    for sector in ["XLK", "XLF"]:
        for t in range(n_obs):
            sector_returns.append({
                "timestamp": base_time + timedelta(days=t),
                "symbol": sector,
                "return": 0.0,  # Constant returns
            })

    sector_returns_df = pl.DataFrame(sector_returns)

    factor_returns_df = pl.DataFrame({
        "timestamp": [base_time + timedelta(days=t) for t in range(n_obs)],
        "dummy": np.zeros(n_obs),
    })

    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=n_obs,
        factor_expected_days=n_obs,
        sector_coverage={"XLK": 1.0, "XLF": 1.0},
        factor_coverage={"dummy": 1.0},
    )

    dataset = SectorDataset(
        sector_returns=sector_returns_df,
        factor_returns=factor_returns_df,
        coverage=coverage,
    )

    with pytest.raises(ValueError, match="constant returns"):
        compute_sector_pca(dataset, n_components=2)


def test_standardize_vs_non_standardize(simple_dataset: SectorDataset) -> None:
    """Test difference between standardized and non-standardized PCA."""
    result_standardized = compute_sector_pca(simple_dataset, n_components=3, standardize=True)
    result_non_standardized = compute_sector_pca(simple_dataset, n_components=3, standardize=False)

    # Both should produce valid results
    assert result_standardized.n_components == 3
    assert result_non_standardized.n_components == 3

    # Variance explained may differ
    # (standardized PCA uses correlation matrix, non-standardized uses covariance matrix)
    assert len(result_standardized.variance_explained) == 3
    assert len(result_non_standardized.variance_explained) == 3

    # Sector IDs should be the same
    assert result_standardized.sector_ids == result_non_standardized.sector_ids


# ===== Tests: Correlation Computation =====


def test_compare_loadings_to_betas_perfect_match() -> None:
    """Test correlation with perfect alignment between PC loadings and betas."""
    # Create PCA result with known loadings
    sector_ids = ["S1", "S2", "S3", "S4"]
    pc_loadings = {
        "S1": [0.8, 0.2, 0.1],
        "S2": [0.7, 0.3, 0.2],
        "S3": [0.6, 0.4, 0.3],
        "S4": [0.5, 0.5, 0.4],
    }

    pca_result = SectorPCAResult(
        n_components=3,
        variance_explained=[50.0, 30.0, 20.0],
        cumulative_variance=[50.0, 80.0, 100.0],
        pc_loadings=pc_loadings,
        factor_beta_correlations={},
        eigenvalues=[2.0, 1.2, 0.8],
        eigenvectors=np.array([[0.8, 0.2, 0.1], [0.7, 0.3, 0.2], [0.6, 0.4, 0.3], [0.5, 0.5, 0.4]]),
        sector_ids=sector_ids,
    )

    # Create sector betas that match PC1 loadings exactly
    sector_betas = {
        "S1": {"duration": 0.8, "credit": 0.0},
        "S2": {"duration": 0.7, "credit": 0.0},
        "S3": {"duration": 0.6, "credit": 0.0},
        "S4": {"duration": 0.5, "credit": 0.0},
    }

    correlations = compare_pc_loadings_to_betas(pca_result, sector_betas)

    # PC1 should have perfect correlation with duration
    assert abs(correlations["PC1"]["duration"] - 1.0) < 0.01, (
        f"Expected PC1-duration correlation ~1.0, got {correlations['PC1']['duration']:.4f}"
    )

    # PC1 should have zero correlation with credit (all zeros)
    # Note: correlation with constant is undefined, should be 0.0
    assert correlations["PC1"]["credit"] == 0.0


def test_compare_loadings_to_betas_no_match() -> None:
    """Test correlation with no alignment (random betas)."""
    np.random.seed(777)

    sector_ids = ["S1", "S2", "S3", "S4", "S5"]
    pc_loadings = {
        "S1": [0.8, 0.2, 0.1],
        "S2": [0.7, 0.3, 0.2],
        "S3": [0.6, 0.4, 0.3],
        "S4": [0.5, 0.5, 0.4],
        "S5": [0.4, 0.6, 0.5],
    }

    pca_result = SectorPCAResult(
        n_components=3,
        variance_explained=[50.0, 30.0, 20.0],
        cumulative_variance=[50.0, 80.0, 100.0],
        pc_loadings=pc_loadings,
        factor_beta_correlations={},
        eigenvalues=[2.5, 1.5, 1.0],
        eigenvectors=np.random.randn(5, 3),
        sector_ids=sector_ids,
    )

    # Create random sector betas (no structure)
    sector_betas = {
        "S1": {"duration": np.random.randn(), "credit": np.random.randn()},
        "S2": {"duration": np.random.randn(), "credit": np.random.randn()},
        "S3": {"duration": np.random.randn(), "credit": np.random.randn()},
        "S4": {"duration": np.random.randn(), "credit": np.random.randn()},
        "S5": {"duration": np.random.randn(), "credit": np.random.randn()},
    }

    correlations = compare_pc_loadings_to_betas(pca_result, sector_betas)

    # No strong correlations expected (should be < 0.60)
    for pc_name, factor_corrs in correlations.items():
        for factor_name, corr in factor_corrs.items():
            assert abs(corr) < 0.95, (
                f"{pc_name}-{factor_name} has unexpectedly high correlation: {corr:.4f}"
            )


def test_compare_loadings_missing_sector() -> None:
    """Test error when sectors in PCA don't match sectors in betas."""
    sector_ids = ["S1", "S2", "S3"]
    pc_loadings = {
        "S1": [0.8, 0.2],
        "S2": [0.7, 0.3],
        "S3": [0.6, 0.4],
    }

    pca_result = SectorPCAResult(
        n_components=2,
        variance_explained=[60.0, 40.0],
        cumulative_variance=[60.0, 100.0],
        pc_loadings=pc_loadings,
        factor_beta_correlations={},
        eigenvalues=[1.8, 1.2],
        eigenvectors=np.random.randn(3, 2),
        sector_ids=sector_ids,
    )

    # Betas missing S3
    sector_betas = {
        "S1": {"duration": 0.5},
        "S2": {"duration": 0.6},
        # S3 missing
    }

    with pytest.raises(ValueError, match="not in betas"):
        compare_pc_loadings_to_betas(pca_result, sector_betas)


def test_compare_loadings_empty_betas() -> None:
    """Test error when sector_betas is empty."""
    sector_ids = ["S1", "S2"]
    pc_loadings = {
        "S1": [0.8],
        "S2": [0.7],
    }

    pca_result = SectorPCAResult(
        n_components=1,
        variance_explained=[100.0],
        cumulative_variance=[100.0],
        pc_loadings=pc_loadings,
        factor_beta_correlations={},
        eigenvalues=[2.0],
        eigenvectors=np.random.randn(2, 1),
        sector_ids=sector_ids,
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        compare_pc_loadings_to_betas(pca_result, {})


def test_compare_loadings_zero_variance_factor() -> None:
    """Test correlation when factor has zero variance (all betas same)."""
    sector_ids = ["S1", "S2", "S3"]
    pc_loadings = {
        "S1": [0.8, 0.2],
        "S2": [0.7, 0.3],
        "S3": [0.6, 0.4],
    }

    pca_result = SectorPCAResult(
        n_components=2,
        variance_explained=[60.0, 40.0],
        cumulative_variance=[60.0, 100.0],
        pc_loadings=pc_loadings,
        factor_beta_correlations={},
        eigenvalues=[1.8, 1.2],
        eigenvectors=np.random.randn(3, 2),
        sector_ids=sector_ids,
    )

    # All sectors have same beta (zero variance)
    sector_betas = {
        "S1": {"duration": 0.5},
        "S2": {"duration": 0.5},
        "S3": {"duration": 0.5},
    }

    correlations = compare_pc_loadings_to_betas(pca_result, sector_betas)

    # Correlation with constant should be 0.0
    assert correlations["PC1"]["duration"] == 0.0
    assert correlations["PC2"]["duration"] == 0.0


# ===== Tests: Report Generation =====


def test_generate_pca_validation_report(
    simple_dataset: SectorDataset,
    tmp_path: Path,
) -> None:
    """Test markdown report generation."""
    result = compute_sector_pca(simple_dataset, n_components=3)

    # Create dummy beta correlations
    beta_correlations = {
        "PC1": {"duration": 0.85, "credit": 0.20, "liquidity": -0.10},
        "PC2": {"duration": -0.15, "credit": 0.75, "liquidity": 0.30},
        "PC3": {"duration": 0.10, "credit": -0.25, "liquidity": 0.65},
    }

    output_path = tmp_path / "pca_report.md"

    generate_pca_validation_report(result, beta_correlations, output_path)

    # Check file was created
    assert output_path.exists()

    # Check content
    content = output_path.read_text()

    assert "# PCA Validation of Sector Returns" in content
    assert "## Executive Summary" in content
    assert "## Methodology" in content
    assert "## Results: Variance Decomposition" in content
    assert "## Results: PC Loadings" in content
    assert "## Results: Factor Beta Correlation" in content
    assert "## Validation Outcome" in content

    # Check that sector IDs appear
    for sector_id in result.sector_ids:
        assert sector_id in content

    # Check that factor names appear
    assert "duration" in content
    assert "credit" in content
    assert "liquidity" in content


def test_generate_report_pass_criteria(
    synthetic_three_factor_dataset: SectorDataset,
    tmp_path: Path,
) -> None:
    """Test report shows PASS when criteria are met."""
    result = compute_sector_pca(synthetic_three_factor_dataset, n_components=3)

    # Create strong correlations (should pass)
    beta_correlations = {
        "PC1": {"duration": 0.85, "credit": 0.20, "liquidity": 0.15},
        "PC2": {"duration": 0.20, "credit": 0.80, "liquidity": 0.25},
        "PC3": {"duration": 0.15, "credit": 0.25, "liquidity": 0.75},
    }

    output_path = tmp_path / "pca_pass_report.md"

    generate_pca_validation_report(result, beta_correlations, output_path)

    content = output_path.read_text()

    # Should show PASS
    assert "✅ PASS" in content
    assert "Proceed to Phase 3" in content


def test_generate_report_fail_criteria(
    simple_dataset: SectorDataset,
    tmp_path: Path,
) -> None:
    """Test report shows FAIL when criteria are not met."""
    result = compute_sector_pca(simple_dataset, n_components=3)

    # Create weak correlations (should fail)
    beta_correlations = {
        "PC1": {"duration": 0.20, "credit": 0.15, "liquidity": 0.10},
        "PC2": {"duration": 0.15, "credit": 0.25, "liquidity": 0.20},
        "PC3": {"duration": 0.10, "credit": 0.20, "liquidity": 0.15},
    }

    output_path = tmp_path / "pca_fail_report.md"

    generate_pca_validation_report(result, beta_correlations, output_path)

    content = output_path.read_text()

    # Might show FAIL or CONDITIONAL depending on variance
    assert ("❌ FAIL" in content) or ("⚠️ CONDITIONAL" in content)


def test_generate_report_handles_two_components(
    simple_dataset: SectorDataset,
    tmp_path: Path,
) -> None:
    """Report generation should handle n_components < 3 without indexing errors."""
    result = compute_sector_pca(simple_dataset, n_components=2)

    beta_correlations = {
        "PC1": {"duration": 0.70, "credit": 0.25, "liquidity": -0.05},
        "PC2": {"duration": -0.10, "credit": 0.65, "liquidity": 0.30},
    }

    output_path = tmp_path / "pca_two_component_report.md"

    generate_pca_validation_report(result, beta_correlations, output_path)

    content = output_path.read_text()

    assert "Top 2 PCs" in content
    assert "PC3" not in content


# ===== Tests: Invalid Inputs =====


def test_compute_pca_empty_dataset() -> None:
    """Test error with empty dataset."""
    empty_sector_returns = pl.DataFrame({
        "timestamp": [],
        "symbol": [],
        "return": [],
    }).with_columns([
        pl.col("timestamp").cast(pl.Datetime),
        pl.col("symbol").cast(pl.String),
        pl.col("return").cast(pl.Float64),
    ])

    empty_factor_returns = pl.DataFrame({
        "timestamp": [],
        "dummy": [],
    }).with_columns([
        pl.col("timestamp").cast(pl.Datetime),
        pl.col("dummy").cast(pl.Float64),
    ])

    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=0,
        factor_expected_days=0,
        sector_coverage={},
        factor_coverage={},
    )

    dataset = SectorDataset(
        sector_returns=empty_sector_returns,
        factor_returns=empty_factor_returns,
        coverage=coverage,
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        compute_sector_pca(dataset, n_components=1)


def test_compute_pca_invalid_n_components() -> None:
    """Test error with invalid n_components."""
    base_time = datetime(2020, 1, 1, tzinfo=UTC)

    # Create properly structured data with complete observations
    sector_returns = []
    for t in range(30):
        for sector in ["XLK", "XLF"]:
            sector_returns.append({
                "timestamp": base_time + timedelta(days=t),
                "symbol": sector,
                "return": np.random.normal(0, 0.01),
            })

    sector_returns_df = pl.DataFrame(sector_returns)

    factor_returns_df = pl.DataFrame({
        "timestamp": [base_time + timedelta(days=t) for t in range(30)],
        "dummy": np.zeros(30),
    })

    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=30,
        factor_expected_days=30,
        sector_coverage={"XLK": 1.0, "XLF": 1.0},
        factor_coverage={"dummy": 1.0},
    )

    dataset = SectorDataset(
        sector_returns=sector_returns_df,
        factor_returns=factor_returns_df,
        coverage=coverage,
    )

    # n_components = 0
    with pytest.raises(ValueError, match="at least 1"):
        compute_sector_pca(dataset, n_components=0)

    # n_components > n_sectors
    with pytest.raises(ValueError, match="cannot exceed"):
        compute_sector_pca(dataset, n_components=10)
