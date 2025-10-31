"""
PCA validation of sector returns against factor betas.

This module implements Principal Component Analysis on sector returns to validate
that our chosen risk factors (Duration, Credit, Liquidity) actually capture the
dominant sources of variation in sector co-movement. By comparing PC loadings to
factor betas, we verify whether the factor model is correctly specified.

Key capabilities:
- PCA on sector return correlation matrix
- Variance decomposition analysis (% explained by each PC)
- Correlation between PC loadings and factor betas
- Sector clustering visualization in PC space
- Validation reporting (pass/fail criteria)

Performance: Cold path only (factor validation, not real-time inference)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import structlog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


if TYPE_CHECKING:
    from playground.risk_model.dataset import SectorDataset

LOGGER = structlog.get_logger(__name__)


@dataclass(slots=True)
class SectorPCAResult:
    """
    Results from PCA on sector returns.

    Attributes
    ----------
    n_components : int
        Number of principal components extracted.
    variance_explained : list[float]
        Percentage of variance explained by each PC (0-100 scale).
    cumulative_variance : list[float]
        Cumulative percentage of variance explained (0-100 scale).
    pc_loadings : dict[str, list[float]]
        Sector loadings on each PC, keyed by sector_id.
        e.g., {"XLK": [0.40, -0.25, 0.10], ...}
    factor_beta_correlations : dict[str, dict[str, float]]
        Correlation between each PC and factor betas.
        e.g., {"PC1": {"duration": 0.85, "credit": 0.20, ...}, ...}
    eigenvalues : list[float]
        Eigenvalues of the correlation matrix (variance per component).
    eigenvectors : np.ndarray
        Eigenvector matrix (n_sectors x n_components).
    sector_ids : list[str]
        Ordered list of sector identifiers.
    """

    n_components: int
    variance_explained: list[float]
    cumulative_variance: list[float]
    pc_loadings: dict[str, list[float]]
    factor_beta_correlations: dict[str, dict[str, float]]
    eigenvalues: list[float]
    eigenvectors: np.ndarray
    sector_ids: list[str]


def compute_sector_pca(
    dataset: SectorDataset,
    n_components: int = 3,
    standardize: bool = True,
) -> SectorPCAResult:
    """
    Run PCA on sector return matrix.

    Parameters
    ----------
    dataset : SectorDataset
        Dataset containing sector returns and factor data.
    n_components : int
        Number of principal components to extract (default 3).
    standardize : bool
        Standardize returns before PCA (default True).

    Returns
    -------
    SectorPCAResult
        PCA results with loadings, variance explained, and correlations.

    Raises
    ------
    ValueError
        If dataset has insufficient sectors (<2), insufficient observations (<20),
        or all returns are constant.

    Notes
    -----
    - Standardization (mean=0, std=1) is recommended for PCA to ensure all
      sectors contribute equally regardless of their volatility.
    - PCA is applied to the correlation matrix (standardized) or covariance
      matrix (non-standardized).
    - PC loadings represent the contribution of each sector to each principal
      component (the eigenvectors).
    """
    LOGGER.info(
        "Computing PCA on sector returns",
        n_components=n_components,
        standardize=standardize,
    )

    # Validate inputs
    if dataset.sector_returns.is_empty():
        msg = "dataset.sector_returns cannot be empty"
        raise ValueError(msg)

    # Pivot to wide format: timestamp x sectors
    sector_return_matrix = (
        dataset.sector_returns
        .pivot(
            index="timestamp",
            on="symbol",
            values="return",
        )
        .drop_nulls()
        .sort("timestamp")
    )

    if sector_return_matrix.is_empty():
        msg = "No complete observations after pivoting sector returns"
        raise ValueError(msg)

    # Extract sector columns (exclude timestamp)
    sector_columns = [col for col in sector_return_matrix.columns if col != "timestamp"]

    # Validate number of sectors
    if len(sector_columns) < 2:
        msg = f"PCA requires at least 2 sectors, got {len(sector_columns)}"
        raise ValueError(msg)

    # Validate number of components
    if n_components < 1:
        msg = f"n_components must be at least 1, got {n_components}"
        raise ValueError(msg)

    if n_components > len(sector_columns):
        msg = (
            f"n_components ({n_components}) cannot exceed number of sectors "
            f"({len(sector_columns)})"
        )
        raise ValueError(msg)

    # Extract return matrix as numpy array
    return_matrix = sector_return_matrix.select(sector_columns).to_numpy()

    # Validate sufficient observations
    n_observations = return_matrix.shape[0]
    if n_observations < 20:
        msg = f"PCA requires at least 20 observations, got {n_observations}"
        raise ValueError(msg)

    LOGGER.info(
        "Sector return matrix prepared",
        n_sectors=len(sector_columns),
        n_observations=n_observations,
    )

    # Check for zero variance (all constant returns)
    variances = np.var(return_matrix, axis=0)
    if np.all(variances == 0):
        msg = "All sectors have constant returns (zero variance)"
        raise ValueError(msg)

    # Standardize data if requested
    if standardize:
        scaler = StandardScaler()
        return_matrix_scaled = scaler.fit_transform(return_matrix)
    else:
        return_matrix_scaled = return_matrix

    # Run PCA
    pca = PCA(n_components=n_components)
    pca.fit(return_matrix_scaled)

    # Extract results
    variance_explained = (pca.explained_variance_ratio_ * 100).tolist()
    cumulative_variance = (np.cumsum(pca.explained_variance_ratio_) * 100).tolist()
    eigenvalues = pca.explained_variance_.tolist()
    eigenvectors = pca.components_.T  # Transpose to get sector x PC format

    # Build PC loadings dict (sector_id -> [PC1, PC2, PC3, ...])
    pc_loadings: dict[str, list[float]] = {}
    for i, sector_id in enumerate(sector_columns):
        pc_loadings[sector_id] = eigenvectors[i, :].tolist()

    LOGGER.info(
        "PCA completed",
        variance_explained_by_top_3=f"{sum(variance_explained[:3]):.2f}%",
        pc1_variance=f"{variance_explained[0]:.2f}%",
    )

    return SectorPCAResult(
        n_components=n_components,
        variance_explained=variance_explained,
        cumulative_variance=cumulative_variance,
        pc_loadings=pc_loadings,
        factor_beta_correlations={},  # Will be filled by compare_pc_loadings_to_betas
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        sector_ids=sector_columns,
    )


def compare_pc_loadings_to_betas(
    pca_result: SectorPCAResult,
    sector_betas: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """
    Compute correlation between PC loadings and factor betas.

    For each PC (1, 2, 3), compute Pearson correlation with:
    - Duration betas across all sectors
    - Credit betas across all sectors
    - Liquidity betas across all sectors

    Parameters
    ----------
    pca_result : SectorPCAResult
        PCA results with sector loadings.
    sector_betas : dict[str, dict[str, float]]
        Sector betas keyed by sector_id, e.g.:
        {"XLK": {"duration": -0.12, "credit": -0.20, "liquidity": 0.05}, ...}

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dict: PC name -> {factor_name: correlation, ...}
        e.g., {"PC1": {"duration": 0.85, "credit": 0.20, "liquidity": -0.10}, ...}

    Raises
    ------
    ValueError
        If sector_betas is empty, if there's a mismatch between sectors in
        pca_result and sector_betas, or if any factor has insufficient variation.

    Notes
    -----
    - Correlations close to +1 or -1 indicate strong alignment between a PC
      and a factor (positive or negative loading pattern matches).
    - Correlations near 0 indicate no alignment.
    - The sign of correlation doesn't matter for validation (|r| > 0.60 is target).
    """
    LOGGER.info(
        "Computing correlations between PC loadings and factor betas",
        n_sectors=len(pca_result.sector_ids),
        n_components=pca_result.n_components,
    )

    # Validate inputs
    if not sector_betas:
        msg = "sector_betas cannot be empty"
        raise ValueError(msg)

    # Check for sector mismatch
    pca_sectors = set(pca_result.sector_ids)
    beta_sectors = set(sector_betas.keys())

    if pca_sectors != beta_sectors:
        missing_in_betas = pca_sectors - beta_sectors
        missing_in_pca = beta_sectors - pca_sectors

        if missing_in_betas:
            msg = f"Sectors in PCA but not in betas: {sorted(missing_in_betas)}"
            raise ValueError(msg)

        if missing_in_pca:
            LOGGER.warning(
                "Sectors in betas but not in PCA (will be ignored)",
                sectors=sorted(missing_in_pca),
            )

    # Extract factor names from first sector's betas
    first_sector = next(iter(sector_betas.values()))
    factor_names = list(first_sector.keys())

    if not factor_names:
        msg = "No factors found in sector_betas"
        raise ValueError(msg)

    # Build correlation matrix: PC x Factor
    correlations: dict[str, dict[str, float]] = {}

    for pc_idx in range(pca_result.n_components):
        pc_name = f"PC{pc_idx + 1}"
        correlations[pc_name] = {}

        # Extract PC loadings for all sectors (in order)
        pc_loadings = np.array([
            pca_result.pc_loadings[sector][pc_idx]
            for sector in pca_result.sector_ids
        ])

        for factor_name in factor_names:
            # Extract factor betas for all sectors (in same order)
            factor_betas = np.array([
                sector_betas[sector][factor_name]
                for sector in pca_result.sector_ids
            ])

            # Compute Pearson correlation
            # Handle edge case: constant values
            if np.std(pc_loadings) == 0 or np.std(factor_betas) == 0:
                LOGGER.warning(
                    "Zero variance detected in correlation computation",
                    pc=pc_name,
                    factor=factor_name,
                )
                corr = 0.0
            else:
                corr = float(np.corrcoef(pc_loadings, factor_betas)[0, 1])

            correlations[pc_name][factor_name] = corr

            LOGGER.debug(
                "PC-Factor correlation computed",
                pc=pc_name,
                factor=factor_name,
                correlation=f"{corr:.4f}",
            )

    LOGGER.info("PC-Factor correlations computed successfully")

    return correlations


def generate_pca_validation_report(
    pca_result: SectorPCAResult,
    beta_correlations: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """
    Generate markdown report with PCA validation results.

    Includes:
    - Variance explained table
    - PC loadings table (sectors × PCs)
    - Correlation matrix (PCs × factor betas)
    - Interpretation and recommendations

    Parameters
    ----------
    pca_result : SectorPCAResult
        PCA results with variance decomposition and loadings.
    beta_correlations : dict[str, dict[str, float]]
        Correlation matrix between PCs and factor betas.
    output_path : Path
        Path to save the markdown report.

    Notes
    -----
    - Report uses markdown tables for compatibility with GitHub/GitLab.
    - Validation criteria: top 3 PCs explain >70% variance, and at least
      2/3 PCs correlate (|r| > 0.60) with factor betas.
    """
    LOGGER.info("Generating PCA validation report", output_path=str(output_path))

    component_count = min(3, pca_result.n_components)
    if component_count < 1:
        msg = "pca_result must include at least one principal component"
        raise ValueError(msg)

    component_labels = [f"PC{i + 1}" for i in range(component_count)]
    variance_label = (
        "Top PC" if component_count == 1 else f"Top {component_count} PCs"
    )

    # Compute validation metrics
    top_k_variance = sum(pca_result.variance_explained[:component_count])
    variance_pass = top_k_variance >= 70.0

    # Check how many PCs have strong correlation (|r| > 0.60) with any factor
    strong_correlations = 0
    for pc_name in component_labels:
        factor_corrs = beta_correlations.get(pc_name, {})
        if not factor_corrs:
            continue
        max_abs_corr = max(abs(corr) for corr in factor_corrs.values())
        if max_abs_corr >= 0.60:
            strong_correlations += 1

    required_strong = max(1, math.ceil(component_count * 2 / 3))
    correlation_pass = strong_correlations >= required_strong

    overall_pass = variance_pass and correlation_pass

    # Build report content
    lines = [
        "# PCA Validation of Sector Returns",
        "",
        "## Executive Summary",
        "",
        f"**Validation Status**: {'✅ PASS' if overall_pass else '❌ FAIL'}",
        "",
        f"- **Variance Explained ({variance_label})**: {top_k_variance:.2f}% (Threshold: 70%)",
        f"  - {'✅ PASS' if variance_pass else '❌ FAIL'}",
        f"- **Strong PC-Factor Correlations**: {strong_correlations}/{component_count} PCs "
        f"(Threshold: {required_strong})",
        f"  - {'✅ PASS' if correlation_pass else '❌ FAIL'}",
        "",
        "---",
        "",
        "## Methodology",
        "",
        "Principal Component Analysis (PCA) was applied to the sector return matrix to:",
        "",
        "1. Identify the dominant sources of variation in sector co-movement",
        "2. Validate that our 3-factor model (Duration, Credit, Liquidity) captures these sources",
        "3. Check if PC loadings align with factor betas",
        "",
        "**Approach**:",
        "- PCA on correlation matrix (standardized returns)",
        f"- {pca_result.n_components} principal components extracted",
        f"- {len(pca_result.sector_ids)} sectors analyzed",
        "",
        "---",
        "",
        "## Results: Variance Decomposition",
        "",
        "| Component | Variance Explained (%) | Cumulative Variance (%) |",
        "|-----------|------------------------|-------------------------|",
    ]

    for i in range(pca_result.n_components):
        lines.append(
            f"| PC{i+1}       | {pca_result.variance_explained[i]:22.2f} | "
            f"{pca_result.cumulative_variance[i]:23.2f} |"
        )

    lines.extend([
        "",
        f"**Interpretation**: The {variance_label.lower()} explain **{top_k_variance:.2f}%** of total variance.",
        "",
        "---",
        "",
        "## Results: PC Loadings",
        "",
        f"Sector loadings on {variance_label.lower()}:",
        "",
    ])

    # PC loadings table
    header = "| Sector |" + "".join(f"  {label:^5} |" for label in component_labels)
    separator = "|--------|" + "".join("---------|" for _ in component_labels)
    lines.append(header)
    lines.append(separator)

    for sector_id in pca_result.sector_ids:
        loadings = pca_result.pc_loadings[sector_id]
        row = f"| {sector_id:6} |" + "".join(f" {loadings[i]:7.3f} |" for i in range(component_count))
        lines.append(row)

    lines.extend([
        "",
        "**Interpretation**:",
    ])

    component_descriptions = {
        "PC1": "Represents the dominant co-movement pattern",
        "PC2": "Represents the second-largest orthogonal variation",
        "PC3": "Represents the third-largest orthogonal variation",
    }
    for label in component_labels:
        description = component_descriptions.get(label, "Orthogonal component")
        lines.append(f"- {label}: {description}")

    lines.extend([
        "",
        "---",
        "",
        "## Results: Factor Beta Correlation",
        "",
        "Correlation between PC loadings and factor betas:",
        "",
    ])

    # Factor correlation table
    # Get factor names from first PC
    if beta_correlations:
        factor_names = list(next(iter(beta_correlations.values())).keys())
        header = "| PC  |" + "".join(f" {name:10} |" for name in factor_names)
        separator = "|-----|" + "".join("------------|" for _ in factor_names)

        lines.append(header)
        lines.append(separator)

        for pc_name in component_labels:
            row = f"| {pc_name} |"
            for factor_name in factor_names:
                corr = beta_correlations.get(pc_name, {}).get(factor_name)
                if corr is None:
                    row += "     N/A     |"
                else:
                    row += f" {corr:10.3f} |"
            lines.append(row)

        valid_corrs = {
            pc_name: factor_corrs
            for pc_name, factor_corrs in beta_correlations.items()
            if pc_name in component_labels and factor_corrs
        }

        if valid_corrs:
            lines.extend([
                "",
                "**Key Findings**:",
                "",
            ])

            # Find best alignment for each PC
            for pc_name in component_labels:
                pc_factor_corrs: dict[str, float] | None = valid_corrs.get(pc_name)
                if not pc_factor_corrs:
                    continue
                best_factor = max(pc_factor_corrs.items(), key=lambda x: abs(x[1]))
                lines.append(
                    f"- **{pc_name}**: Strongest correlation with **{best_factor[0]}** "
                    f"(r = {best_factor[1]:.3f})"
                )
    else:
        lines.extend([
            "",
            "_No factor correlation data supplied._",
        ])

    lines.extend([
        "",
        "---",
        "",
        "## Validation Outcome",
        "",
        f"### Variance Explained: {'✅ PASS' if variance_pass else '❌ FAIL'}",
        "",
        f"The {variance_label.lower()} explain **{top_k_variance:.2f}%** of sector return variance.",
        "",
        "- **Target**: >70%",
        f"- **Result**: {'Sufficient' if variance_pass else 'Insufficient'} "
        f"{'for 3-factor model' if variance_pass else '(consider 4+ factors)'}",
        "",
        f"### Factor Correlation: {'✅ PASS' if correlation_pass else '❌ FAIL'}",
        "",
        f"**{strong_correlations}/{component_count}** PCs show strong correlation (|r| > 0.60) with factor betas.",
        "",
        f"- **Target**: At least {required_strong} PCs",
        f"- **Result**: {'Factors validated' if correlation_pass else 'Weak factor alignment'}",
        "",
        "### Decision",
        "",
    ])

    if overall_pass:
        lines.extend([
            "✅ **PASS**: The 3-factor model (Duration, Credit, Liquidity) is validated.",
            "",
            "**Recommendation**: Proceed to Phase 3 (Backtesting).",
        ])
    elif variance_pass and not correlation_pass:
        lines.extend([
            "⚠️ **CONDITIONAL**: Variance is sufficient, but PC-factor alignment is weak.",
            "",
            "**Recommendation**:",
            "- Re-examine factor definitions (try alternative proxies)",
            "- Consider orthogonalizing factors before computing betas",
            "- Proceed to backtesting with caution",
        ])
    elif not variance_pass and correlation_pass:
        lines.extend([
            "⚠️ **CONDITIONAL**: Factor alignment is good, but variance explained is low.",
            "",
            "**Recommendation**:",
            "- Consider adding a 4th factor (Momentum, Value, Quality)",
            "- Verify sector selection (are all 9 sectors needed?)",
        ])
    else:
        lines.extend([
            "❌ **FAIL**: Both variance and factor correlation criteria not met.",
            "",
            "**Recommendation**:",
            "- Redefine factors using alternative proxies",
            "- Consider non-linear dimensionality reduction (t-SNE, UMAP)",
            "- Revisit factor construction in Phase 2.1.1",
        ])

    lines.extend([
        "",
        "---",
        "",
        "## References",
        "",
        "- Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.",
        "- Connor, G., & Korajczyk, R. A. (1993). A Test for the Number of Factors in "
        "an Approximate Factor Model. *Journal of Finance*, 48(4), 1263-1291.",
        "- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on "
        "stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.",
        "",
    ])

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    LOGGER.info("PCA validation report generated successfully")


__all__ = [
    "SectorPCAResult",
    "compare_pc_loadings_to_betas",
    "compute_sector_pca",
    "generate_pca_validation_report",
]
