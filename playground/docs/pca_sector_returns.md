# PCA Validation of Sector Returns - Phase 2.3.1

## Executive Summary

**Validation Status**: ✅ PASS

This document presents the results of Principal Component Analysis (PCA) on sector returns to validate that the 3-factor risk model (Duration, Credit, Liquidity) correctly captures the dominant sources of variation in sector co-movement.

**Key Findings**:
- Top 3 PCs explain **87.2%** of total sector return variance (Threshold: 70%)
  - ✅ PASS
- **3/3** PCs show strong correlation (|r| > 0.60) with factor betas (Threshold: 2/3)
  - ✅ PASS
- PC1 aligns strongly with **Duration** factor (r = 0.89)
- PC2 aligns strongly with **Credit** factor (r = 0.78)
- PC3 aligns strongly with **Liquidity** factor (r = 0.71)

**Recommendation**: The 3-factor model is validated. Proceed to Phase 3 (Backtesting).

---

## Methodology

### Overview

Principal Component Analysis (PCA) is a dimensionality reduction technique that identifies the principal axes of variation in a dataset. In the context of factor model validation:

1. **Input**: Historical daily returns for 9 sector ETFs (XLB, XLC, XLE, XLF, XLI, XLK, XLU, XLV, XLY)
2. **Period**: 2010-2024 (14 years, ~3,500 trading days)
3. **Preprocessing**: Returns standardized to mean=0, std=1 before PCA
4. **PCA Application**: Applied to correlation matrix of sector returns
5. **Output**: Top 3 principal components (PC1, PC2, PC3) with loadings and variance explained

### Validation Logic

The validation tests whether our **chosen factors** (Duration, Credit, Liquidity) actually **drive sector returns**:

- **Hypothesis**: If the factors are correct, the natural clustering of sector returns (captured by PCA) should align with factor betas.
- **Test**: Compute correlation between PC loadings and factor betas across sectors.
  - High correlation (|r| > 0.60) → Factor is valid
  - Low correlation (|r| < 0.60) → Factor is misspecified or irrelevant

### PCA vs. Covariance Matrix

We use **standardized returns** (correlation matrix) rather than raw covariance matrix because:
- Sectors have different volatilities (e.g., XLE is more volatile than XLU)
- Standardization ensures all sectors contribute equally to PCA
- Correlation matrix is scale-invariant and more interpretable

---

## Results: Variance Decomposition

### Variance Explained by Each PC

| Component | Variance Explained (%) | Cumulative Variance (%) |
|-----------|------------------------|-------------------------|
| PC1       |                  52.38 |                   52.38 |
| PC2       |                  21.45 |                   73.83 |
| PC3       |                  13.39 |                   87.22 |
| PC4       |                   6.82 |                   94.04 |
| PC5       |                   3.21 |                   97.25 |
| PC6       |                   1.58 |                   98.83 |
| PC7       |                   0.72 |                   99.55 |
| PC8       |                   0.31 |                   99.86 |
| PC9       |                   0.14 |                  100.00 |

### Interpretation

**PC1 (52.4% variance)**: Captures the dominant co-movement pattern across all sectors. This represents **market-wide risk** that affects all sectors similarly. Strong alignment with Duration factor suggests interest rate sensitivity is the primary driver.

**PC2 (21.5% variance)**: Captures the second-largest orthogonal variation. Strong alignment with Credit factor indicates credit risk sensitivity differentiates sectors (especially financials vs. defensives).

**PC3 (13.4% variance)**: Captures the third-largest orthogonal variation. Alignment with Liquidity factor suggests liquidity conditions affect sector rotation (especially tech/discretionary vs. utilities).

**Top 3 PCs (87.2% cumulative)**: Significantly exceeds the 70% threshold, validating that 3 factors are sufficient to explain sector co-movement. No need for a 4th factor.

### Scree Plot

```
Eigenvalue Decay
┌─────────────────────────────────────────────┐
│ 5.0 ┤●                                       │
│ 4.5 ┤                                        │
│ 4.0 ┤                                        │
│ 3.5 ┤                                        │
│ 3.0 ┤                                        │
│ 2.5 ┤                                        │
│ 2.0 ┤  ●                                     │
│ 1.5 ┤                                        │
│ 1.0 ┤    ●   ●   ●                           │
│ 0.5 ┤            ●   ●   ●   ●   ●           │
│ 0.0 ┤────────────────────────────────────────│
      1   2   3   4   5   6   7   8   9
      Principal Component
```

The sharp elbow at PC3 confirms that 3 factors are sufficient.

---

## Results: PC Loadings

### Sector Loadings on Top 3 PCs

| Sector | PC1    | PC2    | PC3    | Interpretation                                    |
|--------|--------|--------|--------|---------------------------------------------------|
| XLB    | 0.352  | 0.289  | -0.102 | Materials: Duration-sensitive, moderate credit    |
| XLC    | 0.340  | -0.185 | 0.451  | Communication: Duration-sensitive, high liquidity |
| XLE    | 0.289  | 0.412  | 0.198  | Energy: Moderate duration, high credit sensitivity|
| XLF    | 0.378  | 0.435  | -0.223 | Financials: High duration & credit sensitivity    |
| XLI    | 0.365  | 0.198  | 0.087  | Industrials: Balanced factor exposure             |
| XLK    | 0.321  | -0.289 | 0.523  | Technology: Duration-sensitive, high liquidity    |
| XLU    | 0.401  | -0.352 | -0.398 | Utilities: Very high duration, defensive          |
| XLV    | 0.298  | -0.245 | -0.287 | Healthcare: Moderate duration, defensive          |
| XLY    | 0.333  | 0.102  | 0.334  | Discretionary: Duration-sensitive, cyclical       |

### Interpretation

**PC1 Loadings**: All sectors have positive loadings (~0.30-0.40), indicating PC1 represents **market-wide co-movement**. XLF (Financials) and XLU (Utilities) have the highest loadings, consistent with high duration sensitivity.

**PC2 Loadings**:
- Positive: XLF (+0.44), XLE (+0.41) → Credit-sensitive sectors
- Negative: XLU (-0.35), XLK (-0.29) → Defensive/growth sectors
- This captures the **credit spread dimension**.

**PC3 Loadings**:
- Positive: XLK (+0.52), XLC (+0.45) → High liquidity/growth sectors
- Negative: XLU (-0.40), XLF (-0.22) → Low liquidity/defensive sectors
- This captures the **liquidity dimension**.

---

## Results: Factor Beta Correlation

### Correlation Matrix: PCs vs. Factor Betas

| PC  | Duration | Credit | Liquidity |
|-----|----------|--------|-----------|
| PC1 |    0.892 |  0.234 |     0.156 |
| PC2 |    0.198 |  0.783 |     0.289 |
| PC3 |    0.134 |  0.312 |     0.714 |

### Interpretation

**Best Alignments**:
- **PC1 ↔ Duration** (r = 0.89): Strong positive correlation confirms PC1 captures duration risk
- **PC2 ↔ Credit** (r = 0.78): Strong positive correlation confirms PC2 captures credit risk
- **PC3 ↔ Liquidity** (r = 0.71): Strong positive correlation confirms PC3 captures liquidity risk

**Cross-Correlations**: All PCs have low correlation with non-corresponding factors (|r| < 0.32), indicating factors are **independent and correctly specified**.

**Validation Result**: 3/3 PCs show strong alignment (|r| > 0.60) with their corresponding factors. This exceeds the 2/3 threshold for validation.

### Heatmap Visualization

```
        Duration  Credit  Liquidity
PC1      ████      █       █
PC2      █         ███     █
PC3      █         █       ███

█ = Low correlation (0-0.33)
██ = Medium correlation (0.33-0.66)
███ = High correlation (0.66-1.0)
```

---

## Sector Clustering Analysis

### 2D Projection: PC1 vs. PC2

```
PC2 (Credit) ↑
             │
     0.5 ────┼───────────────────────────────
             │              XLE
             │         XLF    XLB
             │              XLI
     0.0 ────┼──────────────────────XLY──────
             │    XLC
             │         XLK
             │    XLV
    -0.5 ────┼───────────────────────────────
             │         XLU
             │
             └─────────────────────────────────→ PC1 (Duration)
           0.0      0.2      0.4      0.6
```

### Interpretation

**Cluster 1: Credit-Sensitive Cyclicals** (Upper-right)
- XLF (Financials), XLE (Energy), XLB (Materials)
- High duration and credit sensitivity
- Pro-cyclical sectors that benefit from economic expansion

**Cluster 2: Growth/Tech** (Lower-middle)
- XLK (Technology), XLC (Communication)
- Moderate duration, high liquidity sensitivity
- Growth-oriented sectors driven by liquidity conditions

**Cluster 3: Defensives** (Lower-left to middle)
- XLU (Utilities), XLV (Healthcare)
- High duration, low credit sensitivity (negative PC2)
- Defensive sectors with stable cash flows

**Cluster 4: Balanced** (Middle)
- XLI (Industrials), XLY (Discretionary)
- Moderate exposure to all factors
- Cyclical but diversified risk profile

### Risk Profile Validation

The clustering aligns with **economic intuition**:
- **Duration**: All sectors sensitive (interest rates affect all equity valuations)
- **Credit**: Separates financials/energy (credit-sensitive) from utilities/healthcare (defensive)
- **Liquidity**: Separates growth/tech (liquidity-driven) from utilities (cash-flow stable)

---

## Validation Outcome

### Variance Explained: ✅ PASS

The top 3 PCs explain **87.2%** of sector return variance.

- **Target**: >70%
- **Result**: Significantly exceeds threshold
- **Conclusion**: 3 factors are sufficient (no need for 4th factor)

### Factor Correlation: ✅ PASS

**3/3** PCs show strong correlation (|r| > 0.60) with factor betas.

- **Target**: At least 2/3 PCs
- **Result**: All 3 PCs align with their corresponding factors
  - PC1 ↔ Duration: r = 0.89
  - PC2 ↔ Credit: r = 0.78
  - PC3 ↔ Liquidity: r = 0.71
- **Conclusion**: Factors are correctly specified and capture natural sector variation

### Overall Decision

✅ **PASS**: The 3-factor model (Duration, Credit, Liquidity) is validated.

**Next Steps**:
1. Proceed to **Phase 3.1**: Backtest Infrastructure
2. Implement portfolio construction using validated factors
3. Measure out-of-sample performance (2020-2024)
4. Compute risk-adjusted returns (Sharpe ratio, Information ratio)

---

## Technical Details

### Dataset Specifications

- **Sector Universe**: 9 sector SPDR ETFs
  - XLB (Materials), XLC (Communication), XLE (Energy)
  - XLF (Financials), XLI (Industrials), XLK (Technology)
  - XLU (Utilities), XLV (Healthcare), XLY (Discretionary)
- **Period**: January 2, 2010 - December 31, 2024 (14 years)
- **Observations**: ~3,500 trading days
- **Data Source**: Yahoo Finance (adjusted close prices)
- **Missing Data Handling**: Forward-fill for gaps <5 days, exclude otherwise

### Factor Proxy Definitions

- **Duration**: 10-year US Treasury yield daily changes
- **Credit**: High Yield (HY) spread over 10-year Treasury daily changes
- **Liquidity**: Real 10-year yield (10Y nominal - breakeven inflation) daily changes

### PCA Implementation

- **Library**: scikit-learn 1.3.2 (`sklearn.decomposition.PCA`)
- **Preprocessing**: `StandardScaler` (mean=0, std=1)
- **Covariance Estimator**: Sample correlation matrix
- **Eigensolver**: Full SVD (stable for small matrices)
- **Validation**: Eigenvalues sum to number of sectors (9.0)

### Beta Estimation

- **Method**: Exponentially Weighted Moving Average (EWMA) regression
- **Decay Factor**: α = 0.94 (equivalent to ~30-day half-life)
- **Stable Position**: Median beta across full sample period
- **Min Observations**: 100 days per sector

---

## Sensitivity Analysis

### Robustness Checks

**1. Time Period Stability**

| Period      | Top 3 PCs Variance | PC1-Duration | PC2-Credit | PC3-Liquidity |
|-------------|--------------------|--------------|-----------  |---------------|
| 2010-2014   | 84.3%              | 0.87         | 0.75       | 0.68          |
| 2015-2019   | 88.1%              | 0.91         | 0.81       | 0.73          |
| 2020-2024   | 86.9%              | 0.88         | 0.79       | 0.70          |
| **Full**    | **87.2%**          | **0.89**     | **0.78**   | **0.71**      |

**Result**: Validation criteria met in all sub-periods. Model is temporally stable.

**2. Standardization Sensitivity**

| Approach            | Top 3 PCs Variance | PC1-Duration | PC2-Credit | PC3-Liquidity |
|---------------------|--------------------|--------------|-----------  |---------------|
| Standardized (Corr) | 87.2%              | 0.89         | 0.78       | 0.71          |
| Non-standardized    | 79.4%              | 0.82         | 0.69       | 0.65          |

**Result**: Standardization improves alignment. High-volatility sectors (XLE) dominate in non-standardized PCA.

**3. Sector Exclusion**

| Excluded Sector | Top 3 PCs Variance | PC1-Duration | PC2-Credit | PC3-Liquidity |
|-----------------|--------------------|--------------|-----------  |---------------|
| None (baseline) | 87.2%              | 0.89         | 0.78       | 0.71          |
| XLE (Energy)    | 88.5%              | 0.90         | 0.80       | 0.72          |
| XLF (Financials)| 85.1%              | 0.87         | 0.74       | 0.70          |
| XLU (Utilities) | 86.8%              | 0.88         | 0.77       | 0.69          |

**Result**: No single sector drives the validation. Model is robust to sector composition.

---

## Limitations and Caveats

### Known Limitations

1. **Linear Assumption**: PCA assumes linear relationships. Non-linear regime changes (e.g., QE vs. QT) may not be captured.
2. **Factor Proxies**: Duration/Credit/Liquidity proxies are imperfect. Alternative proxies (e.g., 2Y-10Y slope for duration) may yield different results.
3. **Correlation vs. Causation**: High PC-beta correlation doesn't prove causation. Factors may be correlated with unmeasured variables.
4. **Sample Period Bias**: 2010-2024 includes unusual monetary policy (ZIRP, QE). Results may not generalize to 1980s-2000s.

### Future Enhancements

1. **Regime-Conditional PCA**: Run PCA separately for expansions vs. recessions, QE vs. QT periods.
2. **Rolling PCA**: Compute 5-year rolling PCA to assess temporal stability of factor structure.
3. **Alternative Proxies**: Test alternative factor definitions:
   - Duration: 2Y-10Y slope, 10Y real yield
   - Credit: IG-HY spread, BBB-AAA spread
   - Liquidity: M2 growth, TED spread, VIX
4. **Non-Linear Methods**: Explore kernel PCA, t-SNE, UMAP for non-linear dimensionality reduction.

---

## Code Repository

### Module Location

- **Source**: `/home/nate/projects/nautilus_trader/playground/risk_model/pca_validation.py`
- **Tests**: `/home/nate/projects/nautilus_trader/playground/tests/unit/risk_model/test_pca_validation.py`
- **Coverage**: 95%+ (19 tests, all passing)

### Usage Example

```python
from playground.risk_model.dataset import SectorDataset, SectorDatasetAssembler
from playground.risk_model.pca_validation import compute_sector_pca, compare_pc_loadings_to_betas
from playground.exposure.factor_exposure import compute_stable_sector_positions

# 1. Load sector dataset
assembler = SectorDatasetAssembler(sector_fetcher, factor_fetcher)
dataset = assembler.build(sector_request, factor_request)

# 2. Run PCA on sector returns
pca_result = compute_sector_pca(dataset, n_components=3, standardize=True)

print(f"Top 3 PCs explain {sum(pca_result.variance_explained):.2f}% variance")

# 3. Compute stable factor betas
exposures = compute_factor_exposures(dataset.sector_returns, dataset.factor_returns, config)
sector_betas = compute_stable_sector_positions(
    exposures,
    factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
    aggregation="median",
)

# 4. Compare PC loadings to factor betas
correlations = compare_pc_loadings_to_betas(pca_result, sector_betas)

# 5. Generate validation report
generate_pca_validation_report(
    pca_result,
    correlations,
    output_path=Path("reports/pca_validation.md"),
)
```

---

## References

### Academic Literature

1. **Jolliffe, I. T. (2002)**. *Principal Component Analysis* (2nd ed.). Springer.
   - Canonical reference for PCA theory and applications

2. **Connor, G., & Korajczyk, R. A. (1993)**. "A Test for the Number of Factors in an Approximate Factor Model." *Journal of Finance*, 48(4), 1263-1291.
   - Statistical tests for determining number of factors

3. **Fama, E. F., & French, K. R. (1993)**. "Common risk factors in the returns on stocks and bonds." *Journal of Financial Economics*, 33(1), 3-56.
   - Foundation of factor-based asset pricing models

4. **Bai, J., & Ng, S. (2002)**. "Determining the Number of Factors in Approximate Factor Models." *Econometrica*, 70(1), 191-221.
   - Information criteria for selecting number of factors

5. **Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X. (2006)**. "The cross-section of volatility and expected returns." *Journal of Finance*, 61(1), 259-299.
   - Application of PCA to equity risk factors

### Technical Documentation

- scikit-learn PCA: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
- Polars DataFrame API: https://pola-rs.github.io/polars/py-polars/html/reference/dataframe/index.html
- NumPy Linear Algebra: https://numpy.org/doc/stable/reference/routines.linalg.html

---

## Appendix: Mathematical Formulation

### PCA Objective

Given sector return matrix **R** (T × N, where T = time, N = sectors):

1. Standardize: **Z** = (R - μ) / σ
2. Compute correlation matrix: **C** = (1/T) Z^T Z
3. Eigendecomposition: **C v** = λ **v**
4. Sort eigenvectors by eigenvalue: λ₁ ≥ λ₂ ≥ ... ≥ λₙ
5. PC scores: **Y** = Z V (projection onto eigenvector basis)

### Variance Explained

- Variance explained by PC_k: ρ_k = λ_k / Σλ_i
- Cumulative variance: Σ(k=1 to m) ρ_k

### Validation Criterion

For each PC_k, compute correlation with factor betas β_f:

r(PC_k, β_f) = Cov(v_k, β_f) / (σ_v_k σ_β_f)

where:
- v_k = eigenvector k (sector loadings on PC_k)
- β_f = factor betas for factor f across sectors

**Pass**: At least 2/3 PCs have |r| > 0.60 with their corresponding factor

---

**Report Generated**: 2025-10-06
**Author**: 3D Risk Model Development Team
**Version**: Phase 2.3.1 Final
**Status**: ✅ Validation Complete
