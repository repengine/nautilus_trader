# Factor Correlation & Orthogonality Analysis

## Overview

This document explains the methodology and interpretation of factor correlation and orthogonality analysis for the 3D Risk Model. The analysis validates that our three factors (Duration, Credit, Liquidity) are independent and capture meaningful, distinct dimensions of risk.

## Table of Contents

1. [Introduction](#introduction)
2. [Why Factor Independence Matters](#why-factor-independence-matters)
3. [Correlation Analysis](#correlation-analysis)
4. [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
5. [Acceptance Criteria](#acceptance-criteria)
6. [Economic Interpretation](#economic-interpretation)
7. [What to Do If Factors Are Correlated](#what-to-do-if-factors-are-correlated)
8. [Technical Implementation](#technical-implementation)
9. [References](#references)

---

## Introduction

The 3D Risk Model decomposes sector ETF returns into three fundamental risk factors:

1. **Duration Factor**: Interest rate sensitivity (10-year Treasury yield changes)
2. **Credit Factor**: Credit risk premium (high-yield spread changes)
3. **Liquidity Factor**: Market liquidity conditions (VIX changes)

For this decomposition to be valid, these factors must be:
- **Independent (orthogonal)**: Low correlation with each other
- **Comprehensive**: Collectively explain a high proportion of variance in sector returns
- **Economically meaningful**: Each factor represents a distinct risk dimension

This analysis tests these properties through correlation analysis and Principal Component Analysis (PCA).

---

## Why Factor Independence Matters

### Multicollinearity Problems

If factors are highly correlated (|r| > 0.5), we encounter several issues:

1. **Unstable Beta Estimates**: Small changes in data can cause large swings in factor loadings
2. **Inflated Standard Errors**: Difficulty determining which factors are truly significant
3. **Interpretation Challenges**: Cannot isolate the independent effect of each factor
4. **Redundancy**: Two correlated factors may be measuring the same underlying risk

### Example: Good vs Bad Factor Structure

**Good (Independent Factors):**
```
Correlation Matrix:
              Duration  Credit  Liquidity
Duration         1.00    0.15      -0.08
Credit           0.15    1.00       0.12
Liquidity       -0.08    0.12       1.00
```
Maximum |r| = 0.15 → Factors are orthogonal ✓

**Bad (Correlated Factors):**
```
Correlation Matrix:
              Duration  Credit  Liquidity
Duration         1.00    0.75       0.10
Credit           0.75    1.00       0.15
Liquidity        0.10    0.15       1.00
```
Maximum |r| = 0.75 → Duration and Credit are redundant ✗

---

## Correlation Analysis

### Methodology

We compute the Pearson correlation coefficient between all factor pairs:

```
r(X, Y) = Cov(X, Y) / (σ_X × σ_Y)
```

Where:
- `Cov(X, Y)` = Covariance between factors X and Y
- `σ_X`, `σ_Y` = Standard deviations of X and Y
- `r` ranges from -1 (perfect negative correlation) to +1 (perfect positive correlation)

### Interpretation Guidelines

| Absolute Correlation | Interpretation | Action Required |
|---------------------|----------------|-----------------|
| 0.00 - 0.30 | Weak/negligible | None - ideal |
| 0.30 - 0.50 | Moderate | Monitor, acceptable |
| 0.50 - 0.70 | Strong | Review factor construction |
| 0.70 - 1.00 | Very strong | Redesign factors |

### Key Metrics

1. **Correlation Matrix**: Full pairwise correlation table
2. **Max Absolute Correlation**: Highest |r| among all off-diagonal pairs
3. **Mean Absolute Correlation**: Average |r| across all off-diagonal pairs
4. **Orthogonality Flag**: `True` if max |r| < threshold (default: 0.50)

### Statistical Significance

For n observations, the standard error of the correlation coefficient is approximately:

```
SE(r) ≈ 1 / sqrt(n)
```

With 3,500+ daily observations (2010-2024), SE ≈ 0.017, so even small correlations are statistically significant. However, we focus on **economic significance** (|r| < 0.50) rather than statistical significance.

---

## Principal Component Analysis (PCA)

### Purpose

PCA answers the question: "Do we really need all 3 factors, or could we explain the same variance with fewer dimensions?"

If the factors are truly independent and meaningful, we should observe:
- Each principal component (PC) explains roughly equal variance (~33% each)
- All 3 PCs are needed to capture >80% of total variance
- No single PC dominates (which would indicate redundancy)

### Methodology

PCA transforms the original correlated factors into uncorrelated principal components:

1. **Standardize** factor returns (mean=0, variance=1)
2. **Compute covariance matrix** of standardized returns
3. **Extract eigenvalues and eigenvectors**
4. **Transform** data into principal component space

### Key Metrics

1. **Explained Variance Ratio**: Percentage of variance captured by each PC
   ```
   Explained Variance (PC_i) = λ_i / Σλ_j
   ```
   Where λ_i is the eigenvalue of the i-th component

2. **Cumulative Variance**: Running sum of explained variance
   ```
   Cumulative Variance (k) = Σ(Explained Variance (PC_i)) for i=1 to k
   ```

3. **Eigenvalues**: Represent the amount of variance captured by each PC
   - Eigenvalue > 1: Component explains more variance than a single original variable
   - Eigenvalue < 1: Component explains less variance than original variables

4. **Loadings**: Correlation between original factors and principal components
   - High loading (|loading| > 0.5): Factor contributes strongly to that PC
   - Low loading (|loading| < 0.3): Factor has little influence on that PC

### Interpretation Examples

**Scenario 1: Three Independent Factors (Ideal)**
```
PC1: 35% variance
PC2: 33% variance
PC3: 32% variance
Total (3 PCs): 100% variance ✓
```
All three PCs are needed → Factors are independent

**Scenario 2: Two Factors Disguised as Three**
```
PC1: 48% variance
PC2: 47% variance
PC3: 5% variance
Total (2 PCs): 95% variance
```
Only 2 PCs needed → One factor is redundant

**Scenario 3: One Dominant Factor**
```
PC1: 85% variance
PC2: 10% variance
PC3: 5% variance
Total (1 PC): 85% variance
```
Factors are highly correlated → Measure same underlying risk

---

## Acceptance Criteria

### Primary Criteria

✅ **Factor Independence**: Maximum absolute off-diagonal correlation < 0.50

✅ **Dimensionality Adequacy**: First 3 principal components capture >80% of variance

✅ **Economic Interpretation**: Each factor represents a distinct, meaningful risk dimension

### Detailed Thresholds

| Metric | Threshold | Pass Criteria |
|--------|-----------|---------------|
| Max \|r\| (off-diagonal) | 0.50 | < 0.50 |
| Mean \|r\| (off-diagonal) | 0.30 | < 0.30 |
| Variance by 3 PCs | 80% | > 80% |
| Individual PC variance | 20% | Each PC > 20% |

### Traffic Light System

🟢 **Green (Pass)**: Max |r| < 0.50 AND 3 PCs > 80% variance
- Factors are independent and comprehensive
- Proceed with risk model deployment

🟡 **Yellow (Review)**: 0.50 ≤ Max |r| < 0.70 OR 70% < 3 PCs ≤ 80%
- Factors show moderate correlation
- Consider factor refinement
- Document limitations

🔴 **Red (Fail)**: Max |r| ≥ 0.70 OR 3 PCs ≤ 70%
- Factors are redundant or incomplete
- Redesign factor construction
- Do not deploy risk model

---

## Economic Interpretation

### Duration Factor (Interest Rate Sensitivity)

**Proxy**: 10-Year Treasury Yield Changes

**Economic Meaning**: Sensitivity to changes in the general level of interest rates

**Expected Behavior**:
- **Positive exposure (β > 0)**: Sector performs well when rates rise (e.g., Financials)
- **Negative exposure (β < 0)**: Sector suffers when rates rise (e.g., Utilities, Real Estate)

**Why Independent**:
- Interest rate changes are primarily driven by monetary policy and inflation expectations
- Largely orthogonal to credit spreads and liquidity conditions

### Credit Factor (Credit Risk Premium)

**Proxy**: High-Yield Spread Changes (Option-Adjusted)

**Economic Meaning**: Sensitivity to changes in corporate credit risk

**Expected Behavior**:
- **Positive exposure (β > 0)**: Sector benefits from tightening credit spreads (e.g., Consumer Discretionary)
- **Negative exposure (β < 0)**: Sector suffers when credit deteriorates (e.g., high-quality defensive sectors)

**Why Independent**:
- Credit spreads reflect corporate default risk and risk appetite
- Can widen even when rates are stable (2008 crisis)
- Can tighten when rates rise (2004-2006 tightening cycle)

### Liquidity Factor (Market Liquidity Conditions)

**Proxy**: VIX Changes (Implied Volatility)

**Economic Meaning**: Sensitivity to changes in market-wide liquidity and risk aversion

**Expected Behavior**:
- **Negative exposure (β < 0)**: Most sectors suffer when VIX spikes (liquidity dries up)
- **Magnitude varies**: High-beta sectors (Technology, Discretionary) more sensitive than defensives

**Why Independent**:
- VIX captures broad market stress and liquidity provision
- Can spike due to geopolitical events, flash crashes, or positioning unwinds
- Independent from fundamental credit and rate dynamics

### Historical Independence Examples

**2013 Taper Tantrum**:
- Duration factor: Large move (10Y yield +100 bps)
- Credit factor: Minimal move (spreads stable)
- Liquidity factor: Modest spike (VIX +30%)
- **Demonstrates**: Duration and credit can move independently

**2015 Oil Collapse**:
- Duration factor: Minimal move (rates stable)
- Credit factor: Large widening (energy credit spreads)
- Liquidity factor: Elevated (general risk-off)
- **Demonstrates**: Credit stress without rate moves

**2018 Volmageddon**:
- Duration factor: Small move (rates slightly up)
- Credit factor: Small move (spreads slightly wider)
- Liquidity factor: Massive spike (VIX from 10 to 50)
- **Demonstrates**: Pure liquidity event

---

## What to Do If Factors Are Correlated

### Step 1: Diagnose the Problem

Run the correlation and PCA analysis to identify:
1. Which factors are correlated
2. Magnitude of correlation
3. Whether correlation is stable over time

### Step 2: Investigate Root Causes

**Possible Causes**:

1. **Structural Correlation**: Factors are inherently linked (e.g., credit spreads widen when rates fall)
   - **Solution**: Accept moderate correlation if economically justified; document limitation

2. **Proxy Selection**: Chosen proxies measure overlapping risks
   - **Solution**: Use orthogonalized factors or alternative proxies

3. **Sample Period Bias**: Correlation driven by specific historical events
   - **Solution**: Test robustness across subperiods; use longer history

4. **Factor Construction**: Poor standardization or return computation
   - **Solution**: Review winsorization, return method (additive vs multiplicative)

### Step 3: Remediation Strategies

**Option 1: Orthogonalization**

Transform factors to remove correlation while preserving variance:

```python
# Gram-Schmidt orthogonalization
factor2_orth = factor2 - (corr(factor1, factor2) * factor1)
factor3_orth = factor3 - (corr(factor1, factor3) * factor1)
                       - (corr(factor2_orth, factor3) * factor2_orth)
```

**Pros**: Guarantees orthogonality
**Cons**: Loses economic interpretation; arbitrary ordering

**Option 2: Alternative Proxies**

Replace correlated factors with better proxies:

| Original Factor | Alternative Proxy |
|----------------|-------------------|
| Credit (HY spread) | Investment-grade spread, CDS index |
| Liquidity (VIX) | Bid-ask spreads, Amihud illiquidity |
| Duration (10Y yield) | 2-year yield, real rates |

**Option 3: Higher-Frequency Data**

Use daily/intraday data instead of monthly to increase observations and reduce spurious correlation.

**Option 4: Accept and Document**

If correlation is moderate (0.30-0.50) and economically justified, proceed with caveats:
- Report variance inflation factors (VIF) in regression diagnostics
- Use robust standard errors
- Document interpretation challenges

### Step 4: Validate Remediation

After applying fixes, re-run analysis:
- Correlation matrix should show max |r| < 0.50
- PCA should show balanced variance across 3 PCs
- Economic interpretation should remain clear

---

## Technical Implementation

### Code Example: Correlation Analysis

```python
from playground.risk_model.factor_analysis import compute_factor_correlations

# Compute correlations
correlation_analysis = compute_factor_correlations(
    factor_returns=factor_df,
    factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
    correlation_threshold=0.50,
)

# Check results
print(f"Max |r|: {correlation_analysis.max_abs_correlation:.4f}")
print(f"Is orthogonal: {correlation_analysis.is_orthogonal}")

# Access correlation matrix
corr_matrix = correlation_analysis.correlation_matrix
print(corr_matrix["factor_duration"]["factor_credit"])
```

### Code Example: PCA Analysis

```python
from playground.risk_model.factor_analysis import compute_pca_analysis

# Run PCA
pca_results = compute_pca_analysis(
    factor_returns=factor_df,
    factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
    n_components=3,
    variance_threshold=0.80,
)

# Check adequacy
print(f"Variance by 3 PCs: {pca_results.variance_captured_by_3pc:.2%}")
print(f"Is adequate: {pca_results.is_adequate}")

# Examine individual components
for i, var_explained in enumerate(pca_results.explained_variance_ratio, 1):
    print(f"PC{i}: {var_explained:.2%} variance")
```

### Code Example: Visualization

```python
from pathlib import Path
from playground.risk_model.factor_analysis import (
    generate_correlation_heatmap,
    generate_scree_plot,
)

# Generate correlation heatmap
generate_correlation_heatmap(
    correlation_analysis,
    output_path=Path("factor_correlation_heatmap.png"),
)

# Generate scree plot
generate_scree_plot(
    pca_results,
    output_path=Path("factor_scree_plot.png"),
)
```

### Data Requirements

- **Minimum observations**: 100 (preferably 1,000+ for stable estimates)
- **Frequency**: Daily factor returns (2010-2024 gives ~3,500 observations)
- **Handling missing values**: Drop rows with any nulls before analysis
- **Standardization**: PCA requires standardized inputs (mean=0, variance=1)

---

## References

### Academic Literature

1. **Fama, E. F., & French, K. R. (1993)**. "Common risk factors in the returns on stocks and bonds."
   *Journal of Financial Economics*, 33(1), 3-56.
   - Foundational work on multi-factor models

2. **Roll, R., & Ross, S. A. (1980)**. "An empirical investigation of the arbitrage pricing theory."
   *Journal of Finance*, 35(5), 1073-1103.
   - APT framework for factor independence

3. **Connor, G., & Korajczyk, R. A. (1993)**. "A test for the number of factors in an approximate factor model."
   *Journal of Finance*, 48(4), 1263-1291.
   - PCA methodology for determining factor count

4. **Bai, J., & Ng, S. (2002)**. "Determining the number of factors in approximate factor models."
   *Econometrica*, 70(1), 191-221.
   - Advanced techniques for factor dimensionality

### Industry Practice

5. **MSCI Barra (2013)**. "Barra Equity Risk Model Handbook."
   - Commercial risk model methodology and validation

6. **BlackRock (2015)**. "Factor-Based Investing."
   - Practitioner guide to factor construction and testing

### Statistical Methods

7. **Jolliffe, I. T. (2002)**. *Principal Component Analysis* (2nd ed.).
   Springer.
   - Comprehensive PCA reference

8. **Härdle, W. K., & Simar, L. (2015)**. *Applied Multivariate Statistical Analysis* (4th ed.).
   Springer.
   - Correlation analysis and multivariate methods

### Online Resources

9. **QuantLib Documentation**: https://www.quantlib.org/
   - Open-source quantitative finance library

10. **Sklearn PCA Documentation**: https://scikit-learn.org/stable/modules/decomposition.html#pca
    - Implementation details for our PCA analysis

---

## Appendix: Mathematical Details

### Correlation Coefficient Formula

For two factors X and Y with returns x₁, ..., xₙ and y₁, ..., yₙ:

```
r(X,Y) = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / sqrt[Σ(xᵢ - x̄)² × Σ(yᵢ - ȳ)²]
```

### PCA Eigenvalue Decomposition

Given standardized factor matrix F (n × 3), compute covariance matrix:

```
Σ = (1/n) F'F
```

Solve eigenvalue problem:

```
Σv = λv
```

Where:
- λ = eigenvalues (sorted descending)
- v = eigenvectors (principal component loadings)

### Variance Inflation Factor (VIF)

For factor j in a regression with k factors:

```
VIF_j = 1 / (1 - R²_j)
```

Where R²_j is the R² from regressing factor j on all other factors.

**Interpretation**:
- VIF = 1: No correlation with other factors
- VIF = 5: Moderate multicollinearity (threshold)
- VIF > 10: Severe multicollinearity

---

**Document Version**: 1.0
**Last Updated**: 2025-10-06
**Authors**: Claude Code (Anthropic)
**Related Documents**:
- [Factor Methodology](factor_methodology.md)
- [Regression Diagnostics](regression_diagnostics.md)
