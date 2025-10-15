# Factor Return Methodology

## Overview

Factor returns are computed using **additive returns** (differences) rather than multiplicative returns (percentage changes) to avoid division-by-zero issues when factor levels cross zero. This methodology is critical for the 3D Risk Model's ability to decompose sector returns into Duration, Credit, and Liquidity exposures.

## Factor Definitions

### Duration Factor (X-Axis)
- **Proxy**: 10-Year Treasury Yield (DGS10)
- **FRED Series**: DGS10
- **Calculation**: ΔDuration = DGS10_t - DGS10_{t-1}
- **Rationale**: Yield changes are naturally additive; percentage changes are unintuitive for yields
- **Interpretation**: Positive values indicate rising rates (bond price pressure), negative values indicate falling rates

### Credit Factor (Y-Axis)
- **Proxy**: High-Yield OAS Spread (BAMLH0A0HYM2)
- **FRED Series**: BAMLH0A0HYM2
- **Calculation**: ΔCredit = HY_Spread_t - HY_Spread_{t-1}
- **Rationale**: Spreads can compress to near-zero; additive returns avoid infinities
- **Interpretation**: Positive values indicate widening spreads (credit stress), negative values indicate tightening

### Liquidity Factor (Z-Axis)
- **Proxy**: 10-Year TIPS Real Rate (DFII10)
- **FRED Series**: DFII10
- **Calculation**: ΔLiquidity = Real_Rate_t - Real_Rate_{t-1}
- **Rationale**: Real rates cross zero; additive returns preserve sign changes
- **Interpretation**: Positive values indicate rising real rates (liquidity tightening), negative values indicate falling rates

## Return Calculation Algorithm

The factor return calculation follows this robust pipeline:

```python
# Step 1: Sort by timestamp to ensure correct differencing
factor_features = factor_features.sort("timestamp")

# Step 2: Compute raw returns (additive)
factor_returns = factor_features.with_columns(
    [pl.col(col).diff().alias(col) for col in factor_columns]
)

# Step 3: Replace inf/-inf with large finite values to prevent NaN propagation
for col in factor_columns:
    factor_returns = factor_returns.with_columns(
        pl.when(pl.col(col).is_infinite())
        .then(pl.when(pl.col(col) > 0).then(pl.lit(10.0)).otherwise(pl.lit(-10.0)))
        .otherwise(pl.col(col))
        .alias(col)
    )

# Step 4: Winsorize at 99th percentile to prevent outliers from dominating EWMA
if winsorize_percentile is not None:
    for col in factor_columns:
        non_null = factor_returns.filter(
            pl.col(col).is_not_null() & pl.col(col).is_finite()
        )
        if non_null.height > 10:  # Only winsorize if enough data
            lower = non_null[col].quantile(1 - winsorize_percentile)
            upper = non_null[col].quantile(winsorize_percentile)
            factor_returns = factor_returns.with_columns(
                pl.col(col).clip(lower, upper).alias(col)
            )

# Step 5: Drop null values to ensure clean regression inputs
factor_returns = factor_returns.drop_nulls(subset=factor_columns)
```

## Design Rationale

### Why Additive Returns?

1. **Zero-Crossing Safety**: Financial factors (yields, spreads, real rates) can cross zero. Percentage changes produce infinities:
   - If previous value = 0, then pct_change = (current - 0) / 0 = ∞
   - If previous value is near-zero, pct_change can be extremely volatile

2. **Economic Interpretability**:
   - "10Y yield rose by 50 basis points" is more meaningful than "10Y yield increased by 35%"
   - Spread changes are naturally additive (100 bps widening vs. 100% increase)

3. **Statistical Stability**: Additive returns maintain consistent scale across the distribution, avoiding extreme values from near-zero denominators

### Winsorization Strategy

- **Default**: 99th percentile capping
- **Purpose**: Prevent extreme outliers from dominating EWMA beta calculations
- **Threshold**: Only applied when dataset has >10 observations to avoid over-trimming
- **Alternative**: Could use robust regression (Huber loss) instead, but winsorization is simpler and effective

### Infinity Handling

- **Replacement Values**: ±10.0 for ±∞
- **Rationale**: Large but finite values prevent NaN propagation in downstream calculations while maintaining sign information
- **Alternative Considered**: Could use median absolute deviation (MAD) scaling, but fixed values are more predictable

## Alignment with Original Specification

**Reference**: `/home/nate/projects/nautilus_trader/playground/3D_Risk_Model_Idea.md` (lines 250-252)

Original specification states:
```python
# Calculate factor returns (daily changes)
factor_returns = factor_data.diff()

# X = factor_returns.dropna()  # Calculate factor returns (daily changes)
```

**Current Implementation** (`playground/exposure/factor_exposure.py` line 70):
```python
returns = selected.with_columns(
    [pl.col(col).diff().alias(col) for col in column_list],
)
```

### Compliance Verification
✅ **Method**: Uses `.diff()` for daily changes (additive)
✅ **Null Handling**: Drops nulls after computation (line 101)
✅ **Return Type**: Returns DataFrame with factor return columns
✅ **Specification Compliance**: **CONFIRMED**

## Regression Model

The 3D Risk Model regresses sector returns against factor returns:

```python
R_sector,t = α + β_dur*ΔDuration_t + β_cred*ΔCredit_t + β_liq*ΔLiquidity_t + ε_t
```

Where:
- **R_sector,t**: Sector return at time t (e.g., XLK, XLU daily return)
- **ΔDuration_t**: Change in 10Y yield at time t
- **ΔCredit_t**: Change in HY spread at time t
- **ΔLiquidity_t**: Change in real rate at time t
- **β coefficients**: EWMA rolling betas (exposures) estimated incrementally
- **ε_t**: Idiosyncratic residual (sector-specific return)

### Key Properties
1. **All variables are in return space** (changes, not levels)
2. **Linear regression is appropriate** for return decomposition
3. **EWMA weighting** gives more weight to recent observations
4. **Factors should be uncorrelated** for clean interpretation (verified in Phase 2.1.3)

## Data Quality Checks

Before computing factor returns, the pipeline ensures:

1. **Timestamp Sorting**: Data is sorted chronologically to ensure correct differencing
2. **Finite Value Validation**: Infinite values are replaced with large finite values
3. **Outlier Management**: Winsorization prevents extreme values from distorting betas
4. **Null Removal**: Missing values are dropped to avoid regression errors
5. **Minimum Data Requirement**: Winsorization only applied when n > 10

## Testing Coverage

The test suite (`playground/tests/unit/risk_model/test_factor_returns.py`) verifies:

1. **Correctness**: Additive returns match manual calculations
2. **Edge Cases**: Handles empty DataFrames, missing columns, invalid methods
3. **Robustness**: Winsorization and infinity handling work correctly
4. **Property Invariants**: Returns are always finite, output shape matches input - 1
5. **Method Parity**: Both "difference" and "pct_change" methods work as expected

## Alternative Methodologies Considered

### Percentage Change Returns
- **Pros**: Standard in equity markets, intuitive for prices
- **Cons**: Infinities when denominator crosses zero, extreme volatility near zero
- **Decision**: Use only for assets (prices), not for factors (spreads/yields)

### Log Returns
- **Pros**: Symmetric, additive across time
- **Cons**: Undefined for negative values, less intuitive for spreads
- **Decision**: Not suitable for factors that can be negative

### Robust Regression
- **Pros**: Handles outliers without data modification
- **Cons**: Computationally expensive, harder to interpret
- **Decision**: Winsorization provides sufficient robustness with better performance

## Future Enhancements

1. **Adaptive Winsorization**: Dynamic percentile based on volatility regime
2. **Multivariate Outlier Detection**: Mahalanobis distance for joint factor outliers
3. **Regime-Dependent Returns**: Separate calculations for high/low volatility periods
4. **Factor Orthogonalization**: Gram-Schmidt to ensure factor independence

## References

1. Original 3D Risk Model Specification: `playground/3D_Risk_Model_Idea.md`
2. Implementation: `playground/exposure/factor_exposure.py`
3. FRED Data Sources:
   - DGS10: https://fred.stlouisfed.org/series/DGS10
   - BAMLH0A0HYM2: https://fred.stlouisfed.org/series/BAMLH0A0HYM2
   - DFII10: https://fred.stlouisfed.org/series/DFII10
