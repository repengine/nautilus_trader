"""
Factor correlation and orthogonality analysis for 3D risk model.

This module validates that the three factors (Duration, Credit, Liquidity) are
independent and capture meaningful dimensions of risk through correlation analysis
and Principal Component Analysis (PCA).

Key capabilities:
- Pairwise factor correlation computation
- Orthogonality testing (|correlation| < threshold)
- PCA dimensionality validation
- Variance decomposition analysis
- Correlation heatmap generation
- Scree plot visualization

Performance: Cold path only (training/validation, not real-time inference)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import structlog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


if TYPE_CHECKING:
    pass

LOGGER = structlog.get_logger(__name__)


@dataclass(slots=True)
class FactorCorrelationAnalysis:
    """
    Correlation matrix and related metrics for factor orthogonality.

    Attributes
    ----------
    correlation_matrix : dict[str, dict[str, float]]
        Nested dict of correlations (factor1 -> factor2 -> correlation).
    max_abs_correlation : float
        Maximum absolute correlation off-diagonal.
    mean_abs_correlation : float
        Mean absolute correlation off-diagonal.
    factor_names : tuple[str, ...]
        Factor column names.
    n_observations : int
        Number of observations used in computation.
    is_orthogonal : bool
        True if max |r| < threshold (factors are independent).
    """

    correlation_matrix: dict[str, dict[str, float]]
    max_abs_correlation: float
    mean_abs_correlation: float
    factor_names: tuple[str, ...]
    n_observations: int
    is_orthogonal: bool


@dataclass(slots=True)
class PCAAnalysis:
    """
    PCA results for factor dimensionality validation.

    Attributes
    ----------
    explained_variance_ratio : list[float]
        Percentage of variance explained by each PC.
    cumulative_variance : list[float]
        Cumulative percentage of variance explained.
    eigenvalues : list[float]
        Eigenvalues of correlation matrix.
    loadings : dict[str, dict[str, float]]
        PC loadings (PC1 -> {factor1: loading, factor2: loading, ...}).
    n_components : int
        Number of principal components computed.
    variance_captured_by_3pc : float
        Percentage of variance captured by first 3 PCs.
    is_adequate : bool
        True if 3 PCs capture > variance_threshold.
    """

    explained_variance_ratio: list[float]
    cumulative_variance: list[float]
    eigenvalues: list[float]
    loadings: dict[str, dict[str, float]]
    n_components: int
    variance_captured_by_3pc: float
    is_adequate: bool


def compute_factor_correlations(
    factor_returns: pl.DataFrame,
    *,
    factor_columns: Sequence[str],
    correlation_threshold: float = 0.50,
) -> FactorCorrelationAnalysis:
    """
    Compute pairwise correlations between factors.

    Parameters
    ----------
    factor_returns : pl.DataFrame
        DataFrame with timestamp and factor columns.
    factor_columns : Sequence[str]
        Factor column names to analyze.
    correlation_threshold : float
        Maximum acceptable |r| for orthogonality (default: 0.50).

    Returns
    -------
    FactorCorrelationAnalysis
        Correlation matrix, max/mean off-diagonal correlations, and orthogonality flag.

    Raises
    ------
    ValueError
        If factor_returns is empty, factor_columns is empty, or required columns missing.

    Notes
    -----
    - Orthogonality is determined by max |r| < correlation_threshold
    - Uses Pearson correlation coefficient
    - Missing values are dropped before computation
    """
    LOGGER.info(
        "Computing factor correlations",
        n_factors=len(factor_columns),
        threshold=correlation_threshold,
    )

    # Validate inputs
    _validate_factor_columns(factor_returns, factor_columns)

    # Extract factor data and drop missing values
    factor_data = factor_returns.select(list(factor_columns)).drop_nulls()

    if factor_data.is_empty():
        msg = "factor_returns contains no valid (non-null) observations"
        raise ValueError(msg)

    n_observations = factor_data.height

    # Compute correlation matrix using numpy
    data_matrix = factor_data.to_numpy()
    corr_matrix = np.corrcoef(data_matrix.T)

    # Handle single factor case (np.corrcoef returns scalar)
    n_factors = len(factor_columns)
    if n_factors == 1:
        corr_matrix = np.array([[1.0]])

    # Convert to nested dict
    correlation_dict: dict[str, dict[str, float]] = {}
    for i, factor1 in enumerate(factor_columns):
        correlation_dict[factor1] = {}
        for j, factor2 in enumerate(factor_columns):
            correlation_dict[factor1][factor2] = float(corr_matrix[i, j])

    # Compute off-diagonal statistics
    off_diagonal: list[float] = []
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            off_diagonal.append(abs(corr_matrix[i, j]))

    max_abs_corr = float(np.max(off_diagonal)) if off_diagonal else 0.0
    mean_abs_corr = float(np.mean(off_diagonal)) if off_diagonal else 0.0

    is_orthogonal = max_abs_corr < correlation_threshold

    LOGGER.info(
        "Factor correlations computed",
        n_observations=n_observations,
        max_abs_correlation=f"{max_abs_corr:.4f}",
        is_orthogonal=is_orthogonal,
    )

    return FactorCorrelationAnalysis(
        correlation_matrix=correlation_dict,
        max_abs_correlation=max_abs_corr,
        mean_abs_correlation=mean_abs_corr,
        factor_names=tuple(factor_columns),
        n_observations=n_observations,
        is_orthogonal=is_orthogonal,
    )


def compute_pca_analysis(
    factor_returns: pl.DataFrame,
    *,
    factor_columns: Sequence[str],
    n_components: int = 3,
    variance_threshold: float = 0.80,
) -> PCAAnalysis:
    """
    Run PCA on factor returns to validate dimensionality.

    Parameters
    ----------
    factor_returns : pl.DataFrame
        DataFrame with timestamp and factor columns.
    factor_columns : Sequence[str]
        Factor column names to analyze.
    n_components : int
        Number of principal components to compute (default: 3).
    variance_threshold : float
        Minimum cumulative variance for adequacy (default: 0.80).

    Returns
    -------
    PCAAnalysis
        Explained variance, eigenvalues, loadings, and adequacy flag.

    Raises
    ------
    ValueError
        If factor_returns is empty, factor_columns is empty, or required columns missing.

    Notes
    -----
    - Data is standardized (zero mean, unit variance) before PCA
    - Adequacy is determined by variance_captured_by_3pc >= variance_threshold
    - Loadings represent factor contributions to each principal component
    """
    LOGGER.info(
        "Computing PCA analysis",
        n_factors=len(factor_columns),
        n_components=n_components,
        variance_threshold=variance_threshold,
    )

    # Validate inputs
    _validate_factor_columns(factor_returns, factor_columns)

    if n_components < 1:
        msg = f"n_components must be at least 1, got {n_components}"
        raise ValueError(msg)

    if n_components > len(factor_columns):
        msg = (
            f"n_components ({n_components}) cannot exceed number of factors "
            f"({len(factor_columns)})"
        )
        raise ValueError(msg)

    # Extract and drop missing values
    factor_data = factor_returns.select(list(factor_columns)).drop_nulls()

    if factor_data.is_empty():
        msg = "factor_returns contains no valid (non-null) observations"
        raise ValueError(msg)

    data_matrix = factor_data.to_numpy()

    # Standardize data (PCA is sensitive to scale)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_matrix)

    # Run PCA
    pca = PCA(n_components=n_components)
    pca.fit(data_scaled)

    # Extract results
    explained_var = pca.explained_variance_ratio_.tolist()
    cumulative_var = np.cumsum(explained_var).tolist()
    eigenvalues = pca.explained_variance_.tolist()

    # Compute loadings (components matrix)
    loadings: dict[str, dict[str, float]] = {}
    for i in range(n_components):
        pc_name = f"PC{i+1}"
        loadings[pc_name] = {
            factor_columns[j]: float(pca.components_[i, j]) for j in range(len(factor_columns))
        }

    # Calculate variance captured by first 3 PCs
    variance_3pc = sum(explained_var[:3]) if len(explained_var) >= 3 else sum(explained_var)

    is_adequate = variance_3pc >= variance_threshold

    LOGGER.info(
        "PCA analysis completed",
        variance_captured_by_3pc=f"{variance_3pc:.4f}",
        is_adequate=is_adequate,
    )

    return PCAAnalysis(
        explained_variance_ratio=explained_var,
        cumulative_variance=cumulative_var,
        eigenvalues=eigenvalues,
        loadings=loadings,
        n_components=n_components,
        variance_captured_by_3pc=float(variance_3pc),
        is_adequate=is_adequate,
    )


def generate_correlation_heatmap(
    correlation_analysis: FactorCorrelationAnalysis,
    *,
    output_path: Path | None = None,
) -> None:
    """
    Generate correlation heatmap using seaborn.

    Parameters
    ----------
    correlation_analysis : FactorCorrelationAnalysis
        Correlation analysis results.
    output_path : Path | None
        Optional path to save the figure (PNG format).

    Notes
    -----
    - Requires matplotlib and seaborn
    - Figure size: 10x8 inches
    - Colormap: coolwarm diverging
    - Annotations show 2 decimal places
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as e:
        msg = "matplotlib and seaborn required for visualization"
        raise ImportError(msg) from e

    # Convert correlation dict to matrix
    factors = list(correlation_analysis.factor_names)
    n_factors = len(factors)
    corr_matrix = np.zeros((n_factors, n_factors))

    for i, factor1 in enumerate(factors):
        for j, factor2 in enumerate(factors):
            corr_matrix[i, j] = correlation_analysis.correlation_matrix[factor1][factor2]

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        xticklabels=factors,
        yticklabels=factors,
        cbar_kws={"label": "Correlation"},
    )
    plt.title("Factor Correlation Matrix")
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        LOGGER.info("Correlation heatmap saved", path=str(output_path))

    plt.close()


def generate_scree_plot(
    pca_analysis: PCAAnalysis,
    *,
    output_path: Path | None = None,
) -> None:
    """
    Generate scree plot showing explained variance by component.

    Parameters
    ----------
    pca_analysis : PCAAnalysis
        PCA analysis results.
    output_path : Path | None
        Optional path to save the figure (PNG format).

    Notes
    -----
    - Requires matplotlib
    - Figure size: 10x6 inches
    - Shows both individual and cumulative variance
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        msg = "matplotlib required for visualization"
        raise ImportError(msg) from e

    n_components = pca_analysis.n_components
    components = list(range(1, n_components + 1))

    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Individual variance explained
    ax1.bar(components, pca_analysis.explained_variance_ratio)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained Ratio")
    ax1.set_title("Individual Variance Explained by Each PC")
    ax1.set_xticks(components)

    # Cumulative variance explained
    ax2.plot(
        components,
        pca_analysis.cumulative_variance,
        marker="o",
        linestyle="-",
        linewidth=2,
    )
    ax2.axhline(y=0.80, color="r", linestyle="--", label="80% threshold")
    ax2.set_xlabel("Number of Principal Components")
    ax2.set_ylabel("Cumulative Variance Explained")
    ax2.set_title("Cumulative Variance Explained")
    ax2.set_xticks(components)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        LOGGER.info("Scree plot saved", path=str(output_path))

    plt.close()


def _validate_factor_columns(
    factor_returns: pl.DataFrame,
    factor_columns: Sequence[str],
) -> None:
    """Validate factor_returns DataFrame and factor_columns."""
    if factor_returns.is_empty():
        msg = "factor_returns cannot be empty"
        raise ValueError(msg)

    if not factor_columns:
        msg = "factor_columns cannot be empty"
        raise ValueError(msg)

    # Check all factor columns exist
    missing_columns = set(factor_columns) - set(factor_returns.columns)
    if missing_columns:
        msg = f"factor_returns missing required columns: {sorted(missing_columns)}"
        raise ValueError(msg)


__all__ = [
    "FactorCorrelationAnalysis",
    "PCAAnalysis",
    "compute_factor_correlations",
    "compute_pca_analysis",
    "generate_correlation_heatmap",
    "generate_scree_plot",
]
