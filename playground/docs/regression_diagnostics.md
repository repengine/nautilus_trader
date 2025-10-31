# Regression Diagnostics for the 3D Factor Risk Model

## Overview

This document describes the comprehensive regression diagnostics framework for validating the 3-factor risk model. The diagnostics assess whether the model has sufficient explanatory power and statistical significance to justify its use in portfolio construction and risk management.

## Regression Model Specification

For each sector ETF (XLU, XLK, XLF, XLE, XLY, XLP, XLB, XLI, XLV), we estimate:

```
R_sector,t = α + β_dur*ΔDuration_t + β_cred*ΔCredit_t + β_liq*ΔLiquidity_t + ε_t
```

Where:
- **R_sector,t**: Sector return at time t
- **ΔDuration_t**: Change in 10-Year Treasury Yield (DGS10)
- **ΔCredit_t**: Change in High-Yield OAS Spread (BAMLH0A0HYM2)
- **ΔLiquidity_t**: Change in 10-Year TIPS Real Rate (DFII10)
- **α**: Intercept (sector-specific alpha)
- **β coefficients**: Factor loadings (exposures)
- **ε_t**: Idiosyncratic residual

## Diagnostic Metrics

### Goodness of Fit

#### R² (Coefficient of Determination)
- **Range**: 0 to 1
- **Interpretation**: Proportion of variance in sector returns explained by the factors
- **Target**: R² > 0.30 for at least 70% of sectors
- **Good**: R² > 0.50
- **Excellent**: R² > 0.70

#### Adjusted R²
- Accounts for number of predictors
- More conservative than R²
- Use for comparing models with different numbers of factors

#### F-Statistic
- Tests overall model significance
- **Null hypothesis**: All betas = 0 (model has no explanatory power)
- **Target**: p-value < 0.05 for at least 70% of sectors
- High F-statistic with low p-value indicates model is better than random

### Coefficient Significance

#### Beta Coefficients
- **Duration Beta (β_dur)**: Sector sensitivity to interest rate changes
  - Positive: Sector benefits from rising rates
  - Negative: Sector suffers from rising rates
- **Credit Beta (β_cred)**: Sector sensitivity to credit risk
  - Positive: Sector performs well when credit spreads widen (risk-off)
  - Negative: Sector performs well when spreads tighten (risk-on)
- **Liquidity Beta (β_liq)**: Sector sensitivity to liquidity conditions
  - Positive: Sector benefits from tightening liquidity (rising real rates)
  - Negative: Sector benefits from easing liquidity (falling real rates)

#### T-Statistics and P-Values
- **T-statistic**: Ratio of coefficient to its standard error
- **Rule of thumb**: |t| > 2.0 suggests significance
- **P-value**: Probability coefficient is zero
- **Target**: At least 2 out of 3 factor betas significant (p < 0.05) for 70%+ sectors

#### Standard Errors
- Measure uncertainty in beta estimates
- Smaller standard errors indicate more precise estimates
- Used to compute confidence intervals

### Multicollinearity Detection

#### Variance Inflation Factor (VIF)
- Measures correlation between predictors
- **Formula**: VIF_j = 1 / (1 - R²_j) where R²_j is R² from regressing factor j on other factors
- **Interpretation**:
  - VIF = 1: No correlation
  - VIF < 5: Acceptable
  - VIF > 5: Potential multicollinearity issue
  - VIF > 10: Serious multicollinearity
- **Target**: VIF < 5 for all factors

**Why it matters**: High multicollinearity makes it difficult to isolate individual factor effects and can lead to unstable beta estimates.

### Heteroskedasticity Testing

#### Breusch-Pagan Test
- Tests if error variance is constant over time
- **Null hypothesis**: Homoskedasticity (constant variance)
- **Alternative**: Heteroskedasticity (variance depends on X)
- **Interpretation**:
  - p-value > 0.05: No evidence of heteroskedasticity (good)
  - p-value < 0.05: Heteroskedasticity detected
- **Impact**: Heteroskedasticity doesn't bias coefficients but underestimates standard errors

**Remedies if detected**:
- Use robust standard errors (White's correction)
- Transform variables (log, square root)
- Use weighted least squares

### Autocorrelation Detection

#### Durbin-Watson Statistic
- Tests for serial correlation in residuals
- **Range**: 0 to 4
- **Interpretation**:
  - DW ≈ 2: No autocorrelation (ideal)
  - DW < 1.5: Positive autocorrelation
  - DW > 2.5: Negative autocorrelation
- **Target**: 1.5 ≤ DW ≤ 2.5 for at least 70% of sectors

**Why it matters**: Autocorrelated residuals suggest:
- Missing dynamic effects (lagged variables)
- Model misspecification
- Standard errors are biased (too small)

**Remedies if detected**:
- Add lagged dependent variable
- Include lagged factor returns
- Use Newey-West standard errors

### Residual Analysis

#### Mean
- **Target**: ≈ 0
- Non-zero mean suggests bias in predictions

#### Standard Deviation
- Measures typical prediction error
- Lower is better
- Compare across sectors to identify which are harder to model

#### Skewness
- Measures asymmetry of residual distribution
- **Target**: ≈ 0 (symmetric)
- **Positive skewness**: More extreme positive errors
- **Negative skewness**: More extreme negative errors

#### Kurtosis (Excess)
- Measures tail heaviness relative to normal distribution
- **Target**: ≈ 0 (normal distribution)
- **Positive**: Fat tails (more extreme errors than normal)
- **Negative**: Thin tails (fewer extreme errors)

## Acceptance Criteria

The 3D Factor Risk Model is considered **valid** if it meets these criteria:

### 1. R² Criterion
**At least 70% of sectors have R² > 0.30**

**Rationale**: A minimum R² of 0.30 means the model explains at least 30% of sector return variance, which is substantial given the noisy nature of daily returns.

### 2. Significant Betas Criterion
**At least 70% of sectors have 2 or more significant factor betas (p < 0.05)**

**Rationale**: Each sector should be meaningfully exposed to at least 2 of the 3 factors. This ensures the 3D model captures real factor dynamics rather than noise.

### 3. Multicollinearity Criterion
**All factors have VIF < 5**

**Rationale**: Factors should be sufficiently independent to isolate their individual effects. High VIF undermines the interpretation of betas.

### 4. Autocorrelation Criterion
**At least 70% of sectors have Durbin-Watson in [1.5, 2.5]**

**Rationale**: Residuals should not exhibit strong serial correlation, which would indicate model misspecification or omitted dynamic effects.

## Interpreting Results

### Sector-Specific Patterns

#### High R² Sectors (R² > 0.60)
- Strong factor dependence
- Returns well-explained by macro factors
- Examples: Utilities (XLU), Financials (XLF)

#### Medium R² Sectors (0.30 < R² < 0.60)
- Moderate factor dependence
- Mix of factor-driven and idiosyncratic returns
- Examples: Industrials (XLI), Materials (XLB)

#### Low R² Sectors (R² < 0.30)
- Weak factor dependence
- More idiosyncratic, sector-specific dynamics
- Examples: Technology (XLK) - driven by company earnings, not macro
- **Action**: Consider adding sector-specific factors or using different model

### Factor Significance Patterns

#### All 3 Factors Significant
- Sector is truly multi-dimensional
- Responds to duration, credit, and liquidity
- Best case for 3D model

#### 2 Factors Significant
- Sector primarily driven by 2 dimensions
- Still valid for 3D model
- Non-significant factor may be near zero (no exposure)

#### 1 Factor Significant
- Sector is one-dimensional
- May be better modeled with single-factor or specialized approach
- Check if this is consistent across time periods

#### No Factors Significant
- Model does not explain sector
- Investigate alternative factors
- May be data quality issue

## Troubleshooting Guide

### Problem: Low R² Across Many Sectors

**Possible Causes**:
- Factors not relevant for these sectors
- Time period includes structural breaks
- Data quality issues (missing data, outliers)

**Solutions**:
1. Check factor data quality and alignment
2. Test different time periods
3. Add sector-specific factors (earnings, volatility)
4. Consider regime-specific models

### Problem: High VIF (Multicollinearity)

**Possible Causes**:
- Factors are too correlated (e.g., credit and liquidity often move together)
- Time period has unusual factor correlation

**Solutions**:
1. Use orthogonalized factors (Gram-Schmidt)
2. Drop one of the correlated factors
3. Use PCA to create uncorrelated factor proxies
4. Use ridge regression to stabilize estimates

### Problem: Heteroskedasticity (Low BP p-value)

**Possible Causes**:
- Volatility clustering (GARCH effects)
- Different market regimes (calm vs. crisis)

**Solutions**:
1. Use robust standard errors
2. Add volatility scaling
3. Consider GARCH-type models
4. Split analysis by volatility regime

### Problem: Autocorrelation (DW Outside [1.5, 2.5])

**Possible Causes**:
- Omitted lagged effects
- Slow-moving factors
- Microstructure effects

**Solutions**:
1. Add lagged dependent variable (AR term)
2. Add lagged factor returns
3. Use longer return horizons (weekly instead of daily)
4. Apply Newey-West standard errors

### Problem: Non-Normal Residuals (High Skewness/Kurtosis)

**Possible Causes**:
- Extreme events (crashes, rallies)
- Outliers not handled
- Fat-tailed return distribution

**Solutions**:
1. Winsorize returns at 1st/99th percentile
2. Use robust regression (Huber, M-estimators)
3. Model tail events separately
4. Accept non-normality (common in finance)

## Example Output Interpretation

### Utilities Sector (XLU)

```
R² = 0.72, Adj R² = 0.71, F-stat = 245.8 (p < 0.001)
Beta_duration = 0.85 (t = 12.3, p < 0.001) ✓
Beta_credit = 0.42 (t = 6.8, p < 0.001) ✓
Beta_liquidity = -0.31 (t = -4.2, p < 0.001) ✓
VIF: Duration = 1.15, Credit = 1.22, Liquidity = 1.08
Durbin-Watson = 1.95
BP test p-value = 0.32 (no heteroskedasticity)
```

**Interpretation**:
- **Excellent fit**: 72% of variance explained
- **All 3 factors significant**: True 3D exposure
- **Duration beta = 0.85**: Highly sensitive to rates (bond proxy)
- **Credit beta = 0.42**: Performs well in risk-off (widening spreads)
- **Liquidity beta = -0.31**: Benefits from falling real rates (easing)
- **No issues**: VIF low, DW good, no heteroskedasticity
- **Conclusion**: Utilities are a defensive, bond-like sector responding to all 3 macro factors

### Technology Sector (XLK)

```
R² = 0.28, Adj R² = 0.27, F-stat = 35.2 (p < 0.001)
Beta_duration = -0.45 (t = -5.1, p < 0.001) ✓
Beta_credit = -0.28 (t = -3.2, p = 0.002) ✓
Beta_liquidity = 0.12 (t = 1.3, p = 0.19) ✗
VIF: Duration = 1.18, Credit = 1.31, Liquidity = 1.12
Durbin-Watson = 2.15
BP test p-value = 0.08 (borderline heteroskedasticity)
```

**Interpretation**:
- **Marginal fit**: 28% explained (near threshold)
- **2/3 factors significant**: Acceptable (meets 2/3 rule)
- **Duration beta = -0.45**: Benefits from falling rates (growth stock)
- **Credit beta = -0.28**: Risk-on sector (suffers in crises)
- **Liquidity beta not significant**: Less sensitive to real rates
- **Minor issues**: Borderline heteroskedasticity (consider robust SE)
- **Conclusion**: Tech is growth-oriented with moderate factor dependence, significant idiosyncratic component (company earnings, innovation)

## Advanced Topics

### Rolling Diagnostics

To assess stability over time:
1. Compute diagnostics in rolling 3-year windows
2. Plot R² evolution
3. Check if beta significance changes across regimes

### Regime-Conditional Diagnostics

Compare diagnostics across market regimes:
- Bull vs. bear markets
- Low vs. high volatility
- Rate hiking vs. easing cycles

### Out-of-Sample Validation

- Compute diagnostics on training period (e.g., 2010-2018)
- Validate predictions on test period (2019-2024)
- Check if R² holds out-of-sample

## CLI Automation and Reports

Use the `playground/scripts/run_phase2_regression_diagnostics.py` CLI to regenerate diagnostics whenever the dataset is refreshed. The runner enforces the acceptance criteria defined in `ml.config.playground.PhaseTwoValidationDefaults` (R² ≥ 0.30 with ≥6/9 sectors passing, ≥2/3 significant betas, VIF < 5, Durbin–Watson within 1.5–2.5).

### Usage

```bash
poetry run python playground/scripts/run_phase2_regression_diagnostics.py \
  --dataset-path playground/data/sector_dataset \
  --output-dir playground/reports/phase2/diagnostics \
  --run-tag 20241024_phase2
```

### Outputs

- `sector_regression_diagnostics.csv` / `.parquet`: Per-sector metrics (R², adj-R², t/F statistics, p-values, Durbin–Watson, Breusch–Pagan, VIF).
- `phase2_regression_summary.json`: Aggregated pass/fail status plus summary statistics and the config snapshot for governance and Grafana ingestion.

Each run creates a timestamped (or user-tagged) subdirectory under `playground/reports/phase2/diagnostics`, ensuring reproducible artefacts for audits and nightly monitoring.

## References

### Statistical Methodology
- Breusch, T. S., & Pagan, A. R. (1979). A simple test for heteroscedasticity and random coefficient variation. *Econometrica*, 47(5), 1287-1294.
- Durbin, J., & Watson, G. S. (1950). Testing for serial correlation in least squares regression. *Biometrika*, 37(3/4), 409-428.
- Belsley, D. A., Kuh, E., & Welsch, R. E. (1980). *Regression diagnostics: Identifying influential data and sources of collinearity*. John Wiley & Sons.

### Factor Models
- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.
- Carhart, M. M. (1997). On persistence in mutual fund performance. *Journal of Finance*, 52(1), 57-82.

### Implementation Tools
- Statsmodels Documentation: https://www.statsmodels.org/
- Polars Documentation: https://pola-rs.github.io/polars/
- SciPy Statistical Functions: https://docs.scipy.org/doc/scipy/reference/stats.html

## Appendix: Diagnostic Plots (Planned)

Future enhancements will include:
1. **Actual vs. Predicted Returns**: Scatter plot showing fit quality
2. **Residual Plot**: Residuals vs. fitted values (check for patterns)
3. **QQ Plot**: Test residual normality
4. **Rolling R²**: Time series of explanatory power
5. **Beta Confidence Intervals**: Visualize coefficient uncertainty
