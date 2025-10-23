# Backtesting Methodology for 3D Factor Risk Model

**Version:** 1.1
**Date:** October 15, 2025
**Phase:** 3.2.1 - Train/Test Split Design
**Status:** Implementation Complete (walk-forward orchestration delivered)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Temporal Split Design](#temporal-split-design)
3. [Look-Ahead Bias Prevention](#look-ahead-bias-prevention)
4. [Factor Parameter Estimation](#factor-parameter-estimation)
5. [Walk-Forward Analysis](#walk-forward-analysis)
6. [Rolling Beta Estimation](#rolling-beta-estimation)
7. [Implementation Guidelines](#implementation-guidelines)
8. [Validation Procedures](#validation-procedures)
9. [References](#references)

---

## Executive Summary

This document defines the backtesting methodology for validating the 3D Factor Risk Model developed in Phase 2. The methodology is designed to provide a **realistic assessment of out-of-sample performance** while strictly preventing look-ahead bias and data leakage.

### Key Design Principles

1. **Temporal Separation**: Clear chronological split between training and testing periods
2. **No Look-Ahead Bias**: Test data never influences training parameters
3. **Realistic Constraints**: Transaction costs, slippage, and rebalancing schedules
4. **Multiple Validation Modes**: Both static split and walk-forward analysis
5. **Rolling Window Estimation**: Beta estimates updated using only historical data

### Primary Train/Test Split

- **Training Period**: January 1, 2010 to December 31, 2018 (8 years)
- **Testing Period**: January 1, 2019 to December 31, 2024 (6 years)
- **No Gap**: Test period starts immediately after training ends
- **Total Duration**: 15 years of market data

### Success Criteria

- ✅ Zero instances of look-ahead bias (verified via automated checks)
- ✅ Factor parameters estimated exclusively on training data
- ✅ Beta estimates use rolling windows confined to historical data
- ✅ Reproducible results with fixed random seed
- ✅ Documented rationale for all design decisions

---

## Temporal Split Design

### 2.1 Primary Split Rationale

The primary split divides the dataset into an 8-year training period (2010-2018) followed by a 6-year testing period (2019-2024). This design balances several competing objectives:

#### Training Period (8 Years: 2010-2018)

**Duration Justification:**

- **Statistical Sufficiency**: ~2,000 trading days provides robust factor parameter estimates
- **Regime Coverage**: Includes multiple market regimes:
  - Post-financial crisis recovery (2010-2011)
  - Low-volatility growth period (2012-2017)
  - Rising rate environment (2017-2018)
  - "Taper tantrum" volatility spike (2013)
- **Factor Stability**: Sufficient history for stable factor covariance estimation
- **Rolling Window Support**: Allows for 252-day (1-year) rolling beta estimation with adequate warm-up

**Market Conditions Covered:**

| Period | Regime | S&P 500 Return | VIX Average | Key Events |
|--------|--------|----------------|-------------|------------|
| 2010-2011 | Recovery | +26.5% | 22.5 | European debt crisis |
| 2012-2014 | Low Vol Growth | +48.6% | 14.2 | QE3, taper tantrum |
| 2015-2016 | Mixed | +12.7% | 16.8 | Oil crash, Brexit |
| 2017-2018 | Late Cycle | +22.5% | 14.5 | Tax cuts, rate hikes |

**Why Not Longer?**

- Structural regime changes over 10+ years may reduce relevance
- Factor loadings can drift over very long periods
- We want to test on recent data (2019-2024) for practical relevance

**Why Not Shorter?**

- <5 years may not capture sufficient regime diversity
- Factor covariance estimation becomes unstable with <1,000 observations
- Rolling beta windows (252 days) need adequate warm-up period

#### Testing Period (6 Years: 2019-2024)

**Duration Justification:**

- **Out-of-Sample Rigor**: Model has never seen this data during training
- **Regime Testing**: Includes unprecedented market conditions:
  - COVID-19 pandemic and crash (2020)
  - Massive fiscal/monetary stimulus (2020-2021)
  - Rapid rate hike cycle (2022-2023)
  - Inflation surge and normalization
- **Statistical Power**: ~1,500 trading days for robust performance metrics
- **Practical Relevance**: Tests model in recent market conditions

**Market Conditions Covered:**

| Period | Regime | S&P 500 Return | VIX Average | Key Events |
|--------|--------|----------------|-------------|------------|
| 2019 | Late Cycle | +31.5% | 15.3 | Fed pivot, trade war |
| 2020 | Crisis & Recovery | +18.4% | 29.8 | COVID-19 crash |
| 2021 | Stimulus | +28.7% | 17.2 | Meme stocks, SPAC boom |
| 2022 | Tightening | -18.1% | 25.7 | Rate hikes, inflation |
| 2023 | AI Boom | +26.3% | 16.9 | Tech rally, soft landing |
| 2024 | Normalization | +23.3% | 14.5 | Election year |

**Why This Testing Window?**

- Tests model through genuine **unseen regimes** (pandemic, inflation shock)
- Validates factor stability across dramatic macro shifts
- Provides recent performance relevant to current deployment
- Includes enough data for statistically significant conclusions

### 2.2 Timeline Visualization

```
=========================================================================
                    3D FACTOR MODEL TIMELINE
=========================================================================

2010  2011  2012  2013  2014  2015  2016  2017  2018 | 2019  2020  2021  2022  2023  2024
|-----|-----|-----|-----|-----|-----|-----|-----|----|-----|-----|-----|-----|-----|-----|
|                                                     |                                   |
|                TRAINING PERIOD                      |          TESTING PERIOD           |
|                   (8 years)                         |            (6 years)              |
|          Factor Parameters Estimated Here           |    Parameters Fixed & Evaluated   |
|                                                     |                                   |
=========================================================================

Key Properties:
- No temporal overlap between periods
- Test period starts exactly after training ends
- All factor parameters frozen at 2018-12-31
- Beta estimates use rolling 252-day windows WITHIN each period
```

### 2.3 Alternative Split Strategies (Not Chosen)

We considered and rejected several alternative designs:

#### Option A: 50/50 Split (7 years / 7 years)

**Pros:**
- Balanced test set size
- More out-of-sample data for evaluation

**Cons:**
- Training period ends in 2016 (misses 2017-2018 tightening)
- Less training data for stable factor estimation
- **Rejected** due to insufficient training coverage

#### Option B: 70/30 Split (10 years / 4 years)

**Pros:**
- Very stable factor estimates
- Covers post-crisis decade comprehensively

**Cons:**
- Training extends to 2019 (includes start of COVID setup)
- Testing period too short (only 4 years)
- Misses testing on 2019-2020 transition
- **Rejected** due to insufficient test period

#### Option C: Gap Between Train/Test

**Pros:**
- Additional protection against look-ahead
- Mimics real-world deployment lag

**Cons:**
- Wastes valuable data (especially problematic with limited history)
- No theoretical benefit if rolling windows properly implemented
- Makes timeline interpretation more complex
- **Rejected** due to data inefficiency

**Conclusion:** The 8-year training / 6-year testing split optimally balances stability, regime coverage, and out-of-sample rigor.

---

## Look-Ahead Bias Prevention

Look-ahead bias occurs when information from the future inadvertently influences past decisions or model parameters. This is the **most critical failure mode** in backtesting and must be prevented through multiple safeguards.

### 3.1 What Constitutes Look-Ahead Bias?

**Valid (No Bias):**
- Using data from 2010-2018 to estimate factor parameters
- Testing those parameters on 2019-2024 data
- Updating beta estimates using rolling 252-day windows that end on rebalance date

**Invalid (Contains Bias):**
- Using 2019-2024 data to select factors or tune parameters
- Computing factor covariance over full 2010-2024 period
- Using forward-looking information in beta estimation windows
- Rebalancing based on data not available at decision time

### 3.2 Temporal Safeguards

#### Rule 1: Strict Chronological Separation

**Implementation:**
```python
split = TrainTestSplit(
    train_start=datetime(2010, 1, 1, tzinfo=UTC),
    train_end=datetime(2018, 12, 31, tzinfo=UTC),
    test_start=datetime(2019, 1, 1, tzinfo=UTC),  # After train_end
    test_end=datetime(2024, 12, 31, tzinfo=UTC),
)

# Automatic validation in __post_init__
if split.test_start <= split.train_end:
    raise ValueError("Look-ahead bias: test starts before training ends")
```

**Validation:**
- `validate_no_lookahead(split)` checks test_start > train_end
- Automated tests ensure no overlap
- Code review checklist includes temporal verification

#### Rule 2: Factor Parameter Freezing

**What Gets Frozen:**

1. **Factor Definitions:**
   - Duration factor: First principal component of bond sector returns
   - Credit factor: Second principal component of bond sector returns
   - Liquidity factor: Third principal component of equity sector returns

2. **Factor Means & Volatilities:**
   - Estimated from 2010-2018 training data only
   - Fixed for entire 2019-2024 testing period

3. **Factor Covariance Matrix:**
   - Computed from 2010-2018 training data only
   - Never updated using test period data

**Implementation:**
```python
# Training phase (2010-2018)
factor_params = estimate_factor_parameters(
    data=train_data,  # Only 2010-2018
)

# Testing phase (2019-2024)
# Parameters are FROZEN - never re-estimated
for rebalance_date in test_dates:
    weights = compute_portfolio_weights(
        factor_params=factor_params,  # Frozen from training
        current_date=rebalance_date,
    )
```

#### Rule 3: Rolling Window Confinement

Beta estimates use rolling 252-day windows, but these windows must respect temporal boundaries:

**Training Period (2010-2018):**
```python
# Example: Estimating beta on 2015-06-30
beta_window_start = date(2014, 7, 1)   # 252 days before
beta_window_end = date(2015, 6, 30)    # Current date (inclusive)

# This window is entirely within training period ✓
assert beta_window_start >= train_start
assert beta_window_end <= train_end
```

**Testing Period (2019-2024):**
```python
# Example: Estimating beta on 2020-03-31
beta_window_start = date(2019, 4, 1)   # 252 days before
beta_window_end = date(2020, 3, 31)    # Current date (inclusive)

# This window is entirely within testing period ✓
assert beta_window_start >= test_start
assert beta_window_end <= test_end
```

**Forbidden: Cross-Boundary Windows**
```python
# BAD: Window spans training and testing periods
beta_window_start = date(2018, 4, 1)   # In training period
beta_window_end = date(2019, 3, 31)    # In testing period
# This would leak test information into training phase ✗
```

### 3.3 Rebalancing Point-in-Time Constraints

Every rebalancing decision must use **only information available at the decision time**.

#### Monthly Rebalancing Example

**Scenario:** Rebalancing on January 31, 2020 (last trading day of month)

**Available Data:**
- All market data through January 31, 2020
- Rolling beta estimates using [February 1, 2019 to January 31, 2020] window
- Factor parameters from training period (frozen since 2018-12-31)

**Unavailable Data:**
- February 2020 returns (haven't happened yet)
- Realized volatility from February 2020 onward
- Any data after January 31, 2020

**Implementation:**
```python
def compute_target_weights(
    rebalance_date: datetime,
    dataset: SectorDataset,
) -> dict[str, float]:
    """
    Compute portfolio weights using only data available at rebalance_date.

    This function enforces point-in-time constraints.
    """
    # Filter dataset to exclude future data
    available_data = dataset.sector_returns.filter(
        pl.col("timestamp") <= rebalance_date  # Strict inequality
    )

    # Compute rolling beta using past 252 days only
    beta_window_start = rebalance_date - timedelta(days=365)
    beta_data = available_data.filter(
        (pl.col("timestamp") > beta_window_start) &
        (pl.col("timestamp") <= rebalance_date)
    )

    # Estimate betas (using only historical data)
    betas = estimate_sector_betas(beta_data, factor_params)

    # Optimize weights (using frozen factor parameters)
    weights = optimize_portfolio(betas, factor_params)

    return weights
```

### 3.4 Automated Validation Checks

To prevent accidental look-ahead bias, we implement multiple validation layers:

#### Check 1: Timestamp Validation
```python
def validate_no_future_data(
    rebalance_date: datetime,
    data_used: pl.DataFrame,
) -> None:
    """Ensure no data after rebalance_date was used."""
    max_timestamp = data_used["timestamp"].max()
    if max_timestamp > rebalance_date:
        raise ValueError(
            f"Look-ahead bias detected: used data from {max_timestamp}, "
            f"but rebalancing on {rebalance_date}"
        )
```

#### Check 2: Split Overlap Validation
```python
def validate_no_overlap(split: TrainTestSplit) -> None:
    """Ensure test period doesn't overlap with training."""
    if split.test_start <= split.train_end:
        raise ValueError(
            f"Look-ahead bias: test starts {split.test_start} "
            f"before training ends {split.train_end}"
        )
```

#### Check 3: Parameter Provenance Tracking
```python
@dataclass
class FactorParameters:
    """Factor parameters with provenance tracking."""
    means: np.ndarray
    covariances: np.ndarray
    estimated_on: datetime  # Must be <= train_end

    def validate_training_only(self, train_end: datetime) -> None:
        """Ensure parameters estimated from training data only."""
        if self.estimated_on > train_end:
            raise ValueError(
                f"Parameters estimated on {self.estimated_on}, "
                f"after training end {train_end}"
            )
```

### 3.5 Code Review Checklist

Before merging any backtesting code, reviewers must verify:

- [ ] All date filters use `<=` for historical data (not `<`)
- [ ] Rolling windows computed using `[date - lookback, date]` range
- [ ] No global parameter re-estimation in testing loop
- [ ] Rebalancing logic uses only data available at decision time
- [ ] Test assertions verify temporal constraints
- [ ] No forward-filling of missing data across train/test boundary
- [ ] Factor parameters frozen after training phase
- [ ] Documentation clearly states what data is used when

---

## Factor Parameter Estimation

Factor parameters must be estimated exclusively from training data and frozen during testing.

### 4.1 What Parameters Are Estimated?

#### A. Factor Definitions (PCA-Based)

**Duration Factor:**
- First principal component of bond sector returns (TLT, LQD, HYG)
- Captures interest rate sensitivity
- Estimated using covariance matrix from 2010-2018

**Credit Factor:**
- Second principal component of bond sector returns
- Captures credit spread movements
- Orthogonal to duration factor by construction

**Liquidity Factor:**
- Third principal component of equity sector returns
- Captures market liquidity conditions
- Independent of duration and credit factors

**Estimation Procedure:**
```python
# Use ONLY training data (2010-2018)
train_data = dataset.sector_returns.filter(
    (pl.col("timestamp") >= train_start) &
    (pl.col("timestamp") <= train_end)
)

# Compute covariance matrix
bond_returns = train_data.filter(
    pl.col("symbol").is_in(["TLT", "LQD", "HYG"])
).pivot(index="timestamp", columns="symbol", values="return")

covariance = np.cov(bond_returns.to_numpy().T)

# PCA decomposition
eigenvalues, eigenvectors = np.linalg.eigh(covariance)
duration_factor = eigenvectors[:, -1]  # First PC
credit_factor = eigenvectors[:, -2]    # Second PC

# FREEZE these definitions for testing period
```

#### B. Factor Means & Volatilities

**Historical Averages:**
```python
factor_means = {
    "duration": train_factor_returns["duration"].mean(),
    "credit": train_factor_returns["credit"].mean(),
    "liquidity": train_factor_returns["liquidity"].mean(),
}

factor_vols = {
    "duration": train_factor_returns["duration"].std(),
    "credit": train_factor_returns["credit"].std(),
    "liquidity": train_factor_returns["liquidity"].std(),
}
```

**Why Estimate Means?**
- Used in mean-variance optimization
- Required for expected return calculations
- Assumption: Historical average approximates future expected return

**Why Estimate Volatilities?**
- Used in risk targeting and position sizing
- Required for Sharpe ratio maximization
- More stable than return means over long periods

#### C. Factor Covariance Matrix

**Purpose:**
- Captures correlations between factors
- Used in portfolio risk calculations
- Critical for mean-variance optimization

**Estimation:**
```python
# Compute 3×3 covariance matrix
factor_cov = np.cov(
    np.column_stack([
        train_factor_returns["duration"],
        train_factor_returns["credit"],
        train_factor_returns["liquidity"],
    ]).T
)

# Store for use in testing period (frozen)
factor_params = FactorParameters(
    means=factor_means,
    volatilities=factor_vols,
    covariance=factor_cov,
    estimated_on=train_end,
)
```

### 4.2 What Parameters Are NOT Estimated?

#### Rolling Beta Estimates (Updated in Real-Time)

Sector betas are **not** part of the frozen parameters. They are re-estimated at each rebalance using rolling 252-day windows.

**Rationale:**
- Sector factor exposures drift over time
- Rolling estimation captures changing relationships
- Still respects temporal constraints (only uses historical data)

**Example:**
```python
# On each rebalance date in testing period
for rebalance_date in test_rebalance_dates:
    # Compute rolling window (past 252 days)
    window_start = rebalance_date - timedelta(days=365)
    window_data = dataset.sector_returns.filter(
        (pl.col("timestamp") > window_start) &
        (pl.col("timestamp") <= rebalance_date)
    )

    # Estimate betas using THIS WINDOW ONLY
    # (factor parameters remain frozen from training)
    current_betas = estimate_sector_betas(
        returns=window_data,
        factor_params=factor_params,  # Frozen
    )

    # Use current betas for portfolio optimization
    weights = optimize_portfolio(current_betas, factor_params)
```

### 4.3 Parameter Stability Verification

Before using parameters in testing, verify their stability:

#### Subperiod Consistency Check

**Method:** Split training period into subperiods and compare estimates

```python
# Split training data into two halves
mid_point = datetime(2014, 6, 30, tzinfo=UTC)

early_params = estimate_parameters(
    data.filter(pl.col("timestamp") <= mid_point)
)

late_params = estimate_parameters(
    data.filter(pl.col("timestamp") > mid_point)
)

# Check stability (correlation > 0.9)
stability = np.corrcoef(
    early_params.means,
    late_params.means,
)[0, 1]

assert stability > 0.9, f"Parameter instability detected: {stability:.3f}"
```

#### Bootstrap Confidence Intervals

**Method:** Resample training data and estimate parameter distributions

```python
bootstrap_means = []
for _ in range(1000):
    # Bootstrap sample
    sample = train_data.sample(frac=1.0, replace=True)
    params = estimate_parameters(sample)
    bootstrap_means.append(params.means)

# Compute 95% confidence intervals
lower = np.percentile(bootstrap_means, 2.5, axis=0)
upper = np.percentile(bootstrap_means, 97.5, axis=0)

# Verify estimates are well-determined
ci_width = upper - lower
assert np.all(ci_width < 0.1), "Wide confidence intervals detected"
```

---

## Walk-Forward Analysis

Walk-forward analysis provides a more robust validation than a single train/test split by simulating real-world model retraining.

### 5.1 Algorithm Overview

**Concept:** Progressively roll training and testing windows forward through time

**Standard Configuration:**
- Training window: 5 years
- Testing window: 1 year
- Step size: 1 year (roll forward)
- Total folds: 10 (for 2010-2024 data)

**Timeline:**
```
Fold 1:  Train [2010-2014] → Test [2015]
Fold 2:  Train [2011-2015] → Test [2016]
Fold 3:  Train [2012-2016] → Test [2017]
Fold 4:  Train [2013-2017] → Test [2018]
Fold 5:  Train [2014-2018] → Test [2019]
Fold 6:  Train [2015-2019] → Test [2020]
Fold 7:  Train [2016-2020] → Test [2021]
Fold 8:  Train [2017-2021] → Test [2022]
Fold 9:  Train [2018-2022] → Test [2023]
Fold 10: Train [2019-2023] → Test [2024]
```

### 5.2 Implementation

**Python Implementation:**
```python
config = WalkForwardConfig(
    start_date=datetime(2010, 1, 1, tzinfo=UTC),
    end_date=datetime(2024, 12, 31, tzinfo=UTC),
    train_years=5,
    test_years=1,
    step_years=1,
)

walk_forward = run_walk_forward_backtest_suite(
    dataset_path=Path("playground/data/sector_dataset"),
    output_dir=Path("playground/reports/backtesting"),
    walk_forward_config=config,
)

# Aggregate results across all folds (test-period metrics)
aggregate = walk_forward.aggregate_metrics()
summary = walk_forward.summarize_metrics()

print(summary)
# ┌──────────────┬───────────┬──────────────┬─────────────────────┬─────────────────────┐
# │ strategy     ┆ num_folds ┆ sharpe_ratio ┆ annualized_return   ┆ annualized_volatility │
# │ ---          ┆ ---       ┆ ---          ┆ ---                 ┆ ---                 │
# │ str          ┆ i64       ┆ f64          ┆ f64                 ┆ f64                 │
# ╞══════════════╪═══════════╪══════════════╪═════════════════════╪═════════════════════╡
# │ Equal Weight ┆ 10        ┆ 0.71 ± 0.05  ┆ 0.153               ┆ 0.203               │
# │ ...          ┆ ...       ┆ ...          ┆ ...                 ┆ ...                 │
# └──────────────┴───────────┴──────────────┴─────────────────────┴─────────────────────┘

# Persist CSV summaries (done automatically by run_walk_forward_backtest_suite)
walk_forward.write_summaries(Path("playground/reports/backtesting/walk_forward"))

# Per-fold artefacts stored at:
# playground/reports/backtesting/walk_forward/fold_01/
# playground/reports/backtesting/walk_forward/fold_02/
# ...
```

### 5.3 Advantages

1. **Multiple Out-of-Sample Tests:**
   - Each fold provides independent performance estimate
   - Reduces selection bias from single split choice
   - Provides distribution of performance metrics

2. **Regime Diversity:**
   - Tests model across different market conditions
   - Each test year has different macro environment
   - More robust than single test period

3. **Realistic Simulation:**
   - Mimics real-world model retraining schedule
   - Parameters updated annually (as in production)
   - Captures parameter drift effects

### 5.4 Disadvantages

1. **Overlapping Training Data:**
   - Training periods overlap (e.g., 2011-2015 and 2012-2016)
   - Not fully independent folds
   - Serial correlation in results

2. **Computational Cost:**
   - Must re-estimate parameters 10 times
   - 10× longer than single split backtest
   - May require parallel execution

3. **Complexity:**
   - More difficult to debug
   - Harder to interpret aggregated results
   - Requires careful fold management

### 5.5 Alternative Configurations

#### Expanding Window (Anchored)

**Description:** Training window grows over time (always starts at beginning)

```python
# Fold 1: Train [2010-2014] → Test [2015]
# Fold 2: Train [2010-2015] → Test [2016]  # Training expanded
# Fold 3: Train [2010-2016] → Test [2017]  # Training expanded
# ...
```

**Pros:**
- Uses all available historical data
- Parameters more stable (larger training sets)

**Cons:**
- Early data may be less relevant
- Training data becomes stale over time
- Computationally expensive (large training sets)

#### Purged Walk-Forward

**Description:** Add gap between training and testing to prevent data leakage

```python
# Fold 1: Train [2010-2014] → Gap [2014-12-01 to 2014-12-31] → Test [2015]
# Fold 2: Train [2011-2015] → Gap [2015-12-01 to 2015-12-31] → Test [2016]
```

**Pros:**
- Additional protection against look-ahead
- Mimics model deployment lag

**Cons:**
- Wastes valuable test data
- May not be necessary with proper rolling windows

**Recommendation:** Use standard rolling window (no gap) for primary analysis, with expanding window as robustness check.

---

## Rolling Beta Estimation

Sector betas are estimated using rolling 252-day (1-year) windows and updated at each monthly rebalance.

### 6.1 Why Rolling Windows?

**Problem with Static Betas:**
- Sector factor exposures change over time
- Technology sector became less cyclical (2010 → 2024)
- Energy sector credit quality deteriorated (2015-2020)
- Static betas miss these structural shifts

**Solution:**
- Re-estimate betas every month using past 252 trading days
- Captures recent relationship between sectors and factors
- Adapts to changing market structure

### 6.2 Window Size Justification

**252 Trading Days (≈ 1 Year):**

**Why Not Shorter? (e.g., 126 days = 6 months)**
- Estimation noise increases with smaller samples
- Beta standard errors: σ(β) ∝ 1/√n
- 126 days: σ(β) ≈ 0.08
- 252 days: σ(β) ≈ 0.06 (25% improvement)

**Why Not Longer? (e.g., 504 days = 2 years)**
- Captures outdated relationships
- Less responsive to regime changes
- May miss recent structural breaks

**Empirical Evidence:**
- Academic literature supports 1-year windows for beta estimation
- Fama-French use 2-5 year windows (but for academic research, not trading)
- Practitioner consensus: 6-12 months for tactical strategies

**Compromise:**
- 252 days balances estimation precision and adaptiveness
- Sufficient data for stable regression (n=252 > 10×k where k=3 factors)
- Responsive enough to capture regime shifts within 2-3 quarters

### 6.3 Temporal Constraints

**Critical Rule:** Rolling windows must respect train/test boundary

#### Example 1: Training Period Beta Estimation

```python
# Rebalancing on June 30, 2015 (within training period)
rebalance_date = datetime(2015, 6, 30, tzinfo=UTC)

# Rolling window: [July 1, 2014 to June 30, 2015]
window_start = rebalance_date - timedelta(days=365)  # Approx 252 trading days
window_end = rebalance_date

# Verify window is entirely within training period
assert window_start >= train_start  # July 1, 2014 >= Jan 1, 2010 ✓
assert window_end <= train_end      # June 30, 2015 <= Dec 31, 2018 ✓

# Estimate betas using this window
betas = estimate_betas(
    returns=train_data.filter(
        (pl.col("timestamp") > window_start) &
        (pl.col("timestamp") <= window_end)
    ),
    factor_params=factor_params,
)
```

#### Example 2: Testing Period Beta Estimation

```python
# Rebalancing on March 31, 2020 (within testing period)
rebalance_date = datetime(2020, 3, 31, tzinfo=UTC)

# Rolling window: [April 1, 2019 to March 31, 2020]
window_start = rebalance_date - timedelta(days=365)
window_end = rebalance_date

# Verify window is entirely within testing period
assert window_start >= test_start  # April 1, 2019 >= Jan 1, 2019 ✓
assert window_end <= test_end      # March 31, 2020 <= Dec 31, 2024 ✓

# Estimate betas using this window
betas = estimate_betas(
    returns=test_data.filter(
        (pl.col("timestamp") > window_start) &
        (pl.col("timestamp") <= window_end)
    ),
    factor_params=factor_params,  # Still frozen from training!
)
```

#### Example 3: Invalid Cross-Boundary Window (FORBIDDEN)

```python
# Rebalancing on March 31, 2019 (early in testing period)
rebalance_date = datetime(2019, 3, 31, tzinfo=UTC)

# Rolling window would be: [April 1, 2018 to March 31, 2019]
window_start = rebalance_date - timedelta(days=365)
window_end = rebalance_date

# PROBLEM: window_start is in TRAINING period, window_end is in TESTING period
assert window_start < train_end    # April 1, 2018 < Dec 31, 2018
assert window_end > test_start      # March 31, 2019 > Jan 1, 2019

# ✗ This violates temporal separation!
# Solution: Skip this rebalance date OR use shorter window
```

### 6.4 Warm-Up Period Handling

**Problem:** Early rebalances in testing period may lack sufficient history

**Solution 1: Skip Early Rebalances**
```python
# Only rebalance after sufficient data available
min_rebalance_date = test_start + timedelta(days=365)

for rebalance_date in test_rebalance_dates:
    if rebalance_date < min_rebalance_date:
        continue  # Skip this rebalance

    # Proceed with normal beta estimation
    ...
```

**Solution 2: Use Shorter Windows Initially**
```python
# Use adaptive window length
days_since_test_start = (rebalance_date - test_start).days

if days_since_test_start < 252:
    # Use all available testing data
    window_start = test_start
else:
    # Use full 252-day window
    window_start = rebalance_date - timedelta(days=365)

betas = estimate_betas(
    returns=test_data.filter(
        (pl.col("timestamp") > window_start) &
        (pl.col("timestamp") <= rebalance_date)
    ),
    factor_params=factor_params,
)
```

**Recommendation:** Use Solution 1 (skip early rebalances) for primary analysis. This ensures consistent beta estimation methodology throughout the testing period.

### 6.5 Beta Estimation Procedure

**Step-by-Step Algorithm:**

1. **Extract Rolling Window Data**
   ```python
   window_data = sector_returns.filter(
       (pl.col("timestamp") > window_start) &
       (pl.col("timestamp") <= window_end)
   )
   ```

2. **Compute Factor Returns**
   ```python
   # Apply frozen factor definitions to rolling window data
   factor_returns = compute_factor_returns(
       returns=window_data,
       factor_definitions=factor_params.definitions,  # From training
   )
   ```

3. **Run Cross-Sectional Regression**
   ```python
   # For each sector, regress returns on factor returns
   betas = {}
   for sector in sectors:
       sector_returns = window_data.filter(pl.col("symbol") == sector)

       # Regression: R_sector = α + β1*F1 + β2*F2 + β3*F3 + ε
       X = factor_returns[["duration", "credit", "liquidity"]].to_numpy()
       y = sector_returns["return"].to_numpy()

       # OLS estimation
       beta = np.linalg.lstsq(X, y, rcond=None)[0]
       betas[sector] = beta
   ```

4. **Store Beta Estimates**
   ```python
   beta_estimates[rebalance_date] = betas
   ```

**Regression Diagnostics:**
- Check R² > 0.3 (factors explain >30% of sector variance)
- Verify residuals are not serially correlated (Durbin-Watson test)
- Monitor beta stability (shouldn't jump dramatically month-to-month)

---

## Implementation Guidelines

This section provides practical guidance for implementing the backtesting methodology.

### 7.1 Code Organization

**File Structure:**
```
playground/
├── backtest/
│   ├── __init__.py
│   ├── engine.py           # FactorBacktester class
│   ├── benchmarks.py       # 60/40, risk parity, min variance
│   ├── splits.py           # Train/test split utilities (NEW)
│   └── strategies.py       # Factor-based strategies
├── docs/
│   └── backtesting_methodology.md  # This document (NEW)
└── tests/
    └── backtest/
        ├── test_engine.py
        ├── test_benchmarks.py
        └── test_splits.py  # Split validation tests (NEW)
```

### 7.2 Configuration Management

**BacktestConfig Class:**
```python
@dataclass(slots=True)
class BacktestConfig:
    """Configuration for backtest execution."""

    # Time period
    split: TrainTestSplit

    # Rebalancing
    rebalance_frequency: str = "monthly"  # "daily", "weekly", "monthly"
    rebalance_threshold: float = 0.05     # Trigger if weights drift >5%

    # Costs
    transaction_cost_bps: float = 10.0    # 10 basis points per trade
    slippage_bps: float = 0.0             # Market impact

    # Constraints
    position_limits: dict[str, tuple[float, float]] | None = None
    max_leverage: float = 1.0

    # Parameters
    initial_capital: float = 1_000_000.0
    random_seed: int = 42

    # Beta estimation
    beta_window_days: int = 252
    min_beta_observations: int = 200
```

All Phase 3 scripts pull their baseline knobs from
`ml.config.playground.ThreeDRiskBacktestDefaults`. The defaults object exposes
risk-free rates, turnover smoothing overrides, liquidity scaling thresholds, and
split-validation tolerances; the backtest runner, sensitivity utilities, and CLI
entrypoints import it directly so changes propagate through every workflow.

Walk-forward exports write a companion `metadata.json`, and the Phase 3 visuals
script renders a `walk_forward_metadata.txt` summary so nightly reports surface
the exact defaults each run used.
Monitoring helpers in `playground/backtest/monitoring.py` load the metadata file
and emit structlog warnings when the recorded configuration drifts from
`ThreeDRiskBacktestDefaults`, enabling dashboards/alerting layers to hook into
logged anomalies without bespoke parsing.
For scheduled checks, invoke `poetry run python -m playground.scripts.check_walk_forward_metadata`
to perform the validation and return a non-zero exit code when drift is detected.

### 7.3 Execution Workflow

**Standard Backtest Execution:**

```python
# 1. Define split
split = define_train_test_split()

# 2. Estimate factor parameters (training period only)
factor_params = estimate_factor_parameters(
    data=dataset.sector_returns.filter(
        (pl.col("timestamp") >= split.train_start) &
        (pl.col("timestamp") <= split.train_end)
    )
)

# 3. Configure backtest
config = BacktestConfig(
    split=split,
    rebalance_frequency="monthly",
    transaction_cost_bps=10.0,
)

# 4. Initialize backtester
backtester = FactorBacktester(config)

# 5. Run backtest (testing period only)
result = backtester.run_backtest(
    dataset=dataset,
    strategy="3d_factor_rolling",
    factor_params=factor_params,
)

# 6. Analyze results
print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
print(f"Max Drawdown: {result.max_drawdown:.1%}")
print(f"Total Return: {result.total_return:.1%}")
```

**Walk-Forward Execution:**

```python
# 1. Generate splits
splits = walk_forward_splits(
    start_date=datetime(2010, 1, 1, tzinfo=UTC),
    end_date=datetime(2024, 12, 31, tzinfo=UTC),
    train_years=5,
    test_years=1,
)

# 2. Run backtest for each fold
results = []
for i, split in enumerate(splits, 1):
    # Estimate parameters for this fold
    factor_params = estimate_factor_parameters(
        data=dataset.sector_returns.filter(
            (pl.col("timestamp") >= split.train_start) &
            (pl.col("timestamp") <= split.train_end)
        )
    )

    # Run backtest
    config = BacktestConfig(split=split)
    backtester = FactorBacktester(config)
    result = backtester.run_backtest(
        dataset=dataset,
        strategy="3d_factor_rolling",
        factor_params=factor_params,
    )

    results.append(result)
    print(f"Fold {i}: Sharpe {result.sharpe_ratio:.3f}, "
          f"Return {result.total_return:.1%}")

# 3. Aggregate statistics
sharpe_mean = np.mean([r.sharpe_ratio for r in results])
sharpe_std = np.std([r.sharpe_ratio for r in results])
print(f"\nWalk-Forward Summary:")
print(f"  Mean Sharpe: {sharpe_mean:.3f}")
print(f"  Std Sharpe:  {sharpe_std:.3f}")
print(f"  Min Sharpe:  {np.min([r.sharpe_ratio for r in results]):.3f}")
print(f"  Max Sharpe:  {np.max([r.sharpe_ratio for r in results]):.3f}")
```

### 7.4 Logging and Monitoring

**Structured Logging:**
```python
LOGGER = structlog.get_logger(__name__)

LOGGER.info(
    "Starting backtest",
    strategy="3d_factor_rolling",
    train_period=f"{split.train_start.date()} to {split.train_end.date()}",
    test_period=f"{split.test_start.date()} to {split.test_end.date()}",
)

# During rebalancing
LOGGER.debug(
    "Rebalanced portfolio",
    date=rebalance_date.isoformat(),
    num_positions=len(current_weights),
    turnover=turnover_rate,
    transaction_cost=cost,
)

# After completion
LOGGER.info(
    "Backtest completed",
    sharpe_ratio=f"{result.sharpe_ratio:.3f}",
    max_drawdown=f"{result.max_drawdown:.1%}",
    num_rebalances=result.num_rebalances,
)
```

### 7.5 Error Handling

**Common Failure Modes:**

1. **Insufficient Training Data**
   ```python
   if split.train_days < 252:
       raise ValueError(
           f"Insufficient training data: {split.train_days} days "
           f"(minimum 252 required)"
       )
   ```

2. **Missing Data in Rolling Window**
   ```python
   if len(window_data) < min_beta_observations:
       LOGGER.warning(
           "Insufficient data for beta estimation, skipping rebalance",
           date=rebalance_date,
           observations=len(window_data),
           required=min_beta_observations,
       )
       continue  # Skip this rebalance
   ```

3. **Look-Ahead Bias Detection**
   ```python
   max_timestamp = data_used["timestamp"].max()
   if max_timestamp > rebalance_date:
       raise ValueError(
           f"Look-ahead bias detected: used data from {max_timestamp}, "
           f"but rebalancing on {rebalance_date}"
       )
   ```

---

## Validation Procedures

This section defines procedures for validating backtest correctness.

### 8.1 Unit Tests

**Test Coverage Requirements:**
- ✅ Split creation and validation (test_splits.py)
- ✅ Look-ahead bias detection
- ✅ Rolling window computation
- ✅ Walk-forward generation
- ✅ Beta estimation (test_engine.py)
- ✅ Portfolio rebalancing logic
- ✅ Transaction cost calculation

**Example Test:**
```python
def test_no_lookahead_bias():
    """Test that test period never precedes training period."""
    split = define_train_test_split()

    # Verify chronological order
    assert split.train_start < split.train_end
    assert split.test_start > split.train_end  # Critical check
    assert split.test_start < split.test_end

    # Verify validation function
    assert validate_no_lookahead(split)

def test_rolling_window_respects_boundaries():
    """Test that rolling windows don't cross train/test boundary."""
    split = define_train_test_split()

    # Test early date in testing period
    rebalance_date = datetime(2019, 3, 31, tzinfo=UTC)
    window_start = rebalance_date - timedelta(days=365)

    # This window would span boundary - should be handled
    if window_start < split.test_start:
        # Should either skip rebalance or use shorter window
        # Test that code handles this correctly
        ...
```

### 8.2 Integration Tests

**End-to-End Validation:**

```python
def test_full_backtest_no_lookahead():
    """Test complete backtest execution with look-ahead checks."""

    # Setup
    split = define_train_test_split()
    dataset = load_test_dataset()

    # Train parameters
    factor_params = estimate_factor_parameters(
        data=dataset.sector_returns.filter(
            (pl.col("timestamp") >= split.train_start) &
            (pl.col("timestamp") <= split.train_end)
        )
    )

    # Verify parameters estimated from training data only
    assert factor_params.estimated_on <= split.train_end

    # Run backtest
    config = BacktestConfig(split=split)
    backtester = FactorBacktester(config)
    result = backtester.run_backtest(
        dataset=dataset,
        strategy="equal_weight",
    )

    # Verify no data leakage
    assert all(d >= split.test_start for d in result.dates)
    assert all(d <= split.test_end for d in result.dates)
```

### 8.3 Monte Carlo Stress Testing

Phase 3 validation now includes a Monte Carlo stress harness that randomises regime
sequences and applies configurable macro overlays. The runner persists both the
per-path distribution and aggregated summary statistics under
``playground/reports/backtesting/stress/monte_carlo``. Invoke the sweep via:

```bash
poetry run python -m playground.scripts.run_phase3_walk_forward --monte-carlo-stress
```

Core diagnostics recorded for each simulation path encompass Sharpe ratio
distributions, drawdown extremes, terminal value, and overlay activations. Tunable
shock parameters remain centralised under
``ThreeDRiskBacktestDefaults().monte_carlo_stress`` to preserve config-driven
behaviour.

The default overlay catalog spans rates hikes, growth scares, liquidity crunches,
volatility breakouts, cross-asset contagion, compound liquidity-growth cascades,
credit spread widening, inflation repricing shocks, and energy-supply
disruptions. Overlay activations are exported both at the event level and in
category aggregates (`overlay_category_summary.csv`), and baseline metrics are
persisted in `baseline_metrics.csv` for dashboard parity.

### 8.4 Parameter Response Heatmaps

Use the parameter heatmap suite to visualise stability across combinations of
transaction costs, turnover smoothing, liquidity multipliers, and related inputs.
Artefacts are written under ``playground/reports/backtesting/heatmaps`` with both
long-form result tables and pivot-ready CSVs:

```bash
poetry run python -m playground.scripts.run_phase3_walk_forward --parameter-heatmaps
```

The evaluated grids and target strategy are defined in
``ThreeDRiskBacktestDefaults().parameter_heatmaps`` ensuring reproducible,
config-driven sweeps.

Pass ``--heatmap-specs turnover-vs-liquidity-multipliers,transaction-cost-envelope`` to
target specific grids—the CLI automatically enables the heatmap suite whenever
spec slugs are provided. Suite summaries capture evaluation counts and the
best-performing configuration for each specification so dashboards can surface
preferred parameters directly.

### 8.5 Extended Diagnostics & Proxy Datasets

Running the ``--extended-diagnostics`` flag captures tail risk statistics,
turnover histograms, and benchmark deltas for the baseline suite, persisting under
``playground/reports/backtesting/diagnostics``. Proxy dataset validation is driven
via ``--proxy-validation`` and reuses specifications from
``ThreeDRiskBacktestDefaults().proxy_datasets`` so alternative universes and
vintage windows stay in sync with configuration files.
Defaults now include international sectors, factor ETF proxies, a treasury
futures hedge dataset, and a crisis-response 2y/1y vintage window so rate and
liquidity scenarios remain covered.

### 8.6 Monitoring Snapshot Export

Enable ``--monitoring-export`` to emit a consolidated JSON snapshot summarising
available artefacts for dashboards and alerting systems. The export references the
latest walk-forward summaries, Monte Carlo stresses, heatmaps, diagnostics, proxy
status, and vintage simulations, and is written to the backtesting output root as
``phase3_monitoring_snapshot.json``.
Snapshot payloads include overlay event totals, category aggregates, baseline
metrics, parameter heatmap metadata, proxy dataset health, vintage window status,
alert channel mappings, and automation targets so Grafana and PagerDuty consumers
can ingest the data without additional transformation. The CLI also persists
integration-ready payloads under ``playground/reports/backtesting/monitoring/``:

- ``grafana_dashboard_payload.json`` summarises artefact paths, overlay category
  statistics, and baseline metrics for dashboard provisioning.
- ``pagerduty_alert_payload.json`` captures alert rules, automation targets,
  diagnostics metadata, and proxy/vintage health for escalation workflows.

### 8.7 Phase 3 Validation Battery

Use ``--phase3-battery`` to execute the entire Phase 3 validation stack in a
single invocation. The flag runs the walk-forward refresh, Monte Carlo stress
sweep, parameter heatmaps (respecting any ``--heatmap-specs`` overrides), extended
diagnostics, proxy validation, vintage simulations, and emits the monitoring
snapshot. This is the preferred option for nightly CI smoke runs and offline
pre-deployment rehearsals.

### 8.3 Manual Validation Checklist

Before production deployment, manually verify:

- [ ] **Temporal Correctness:**
  - Training period uses 2010-2018 data only
  - Testing period uses 2019-2024 data only
  - No overlap between periods

- [ ] **Parameter Freezing:**
  - Factor parameters estimated once from training data
  - Parameters never updated during testing
  - Beta estimates use rolling windows correctly

- [ ] **Point-in-Time Constraints:**
  - Each rebalance uses only data available at that time
  - No forward-looking information in rolling windows
  - Portfolio weights computed from historical data only

- [ ] **Reproducibility:**
  - Results identical across multiple runs (fixed seed)
  - No randomness in factor estimation
  - Deterministic optimization procedures

- [ ] **Performance Metrics:**
  - Sharpe ratio within reasonable range (0.5 - 2.0)
  - Max drawdown not extreme (<50%)
  - Transaction costs properly accounted for

### 8.4 Regression Tests

**Prevent Future Regressions:**

```python
def test_backtest_results_stable():
    """Test that backtest results are reproducible."""

    # Run backtest twice with same seed
    config = BacktestConfig(random_seed=42)
    backtester1 = FactorBacktester(config)
    result1 = backtester1.run_backtest(dataset, "equal_weight")

    config = BacktestConfig(random_seed=42)
    backtester2 = FactorBacktester(config)
    result2 = backtester2.run_backtest(dataset, "equal_weight")

    # Verify identical results
    np.testing.assert_allclose(
        result1.portfolio_values,
        result2.portfolio_values,
        rtol=1e-10,
    )
    assert result1.sharpe_ratio == result2.sharpe_ratio
```

---

## References

### Academic Literature

1. **Fama, E. F., & French, K. R. (2015).** "A five-factor asset pricing model." *Journal of Financial Economics*, 116(1), 1-22.
   - Establishes multi-factor model framework
   - Discusses parameter estimation from historical data

2. **De Prado, M. L. (2018).** *Advances in Financial Machine Learning.* Wiley.
   - Chapter 7: Cross-validation in finance
   - Chapter 12: Backtesting through cross-validation
   - Purged K-fold methodology

3. **Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2014).** "Pseudo-mathematics and financial charlatanism: The effects of backtest overfitting on out-of-sample performance." *Notices of the AMS*, 61(5), 458-471.
   - Discusses overfitting and selection bias
   - Importance of out-of-sample testing

4. **Harvey, C. R., Liu, Y., & Zhu, H. (2016).** "... and the cross-section of expected returns." *Review of Financial Studies*, 29(1), 5-68.
   - Multiple testing and data snooping
   - Adjustment for researcher degrees of freedom

### Industry Best Practices

5. **CFA Institute (2013).** *Global Investment Performance Standards (GIPS).*
   - Standards for performance reporting
   - Backtest disclosure requirements

6. **SEC Division of Investment Management (2018).** "Compliance Issues Related to Adviser Use of Hypothetical Performance."
   - Regulatory guidance on backtesting
   - Required disclosures and limitations

### Implementation References

7. **Pandas Development Team (2024).** *Pandas Market Calendars Documentation.*
   - Trading calendar implementation
   - Holiday and half-day handling

8. **Lopez de Prado, M. (2020).** "A Data Science Solution to the Multiple-Testing Crisis in Financial Research." *The Journal of Financial Data Science*, 2(1), 99-110.
   - Combinatorial purged cross-validation
   - Symmetric CSCV for time series

### Internal Documentation

9. **3D Risk Model Roadmap (playground/3D_Risk_Model_Roadmap.md)**
   - Phase 2: Factor model design and validation
   - Phase 3: Backtesting and performance validation
   - Phase 3.2.1: Train/test split specification

10. **Phase 2 Executive Summary (playground/docs/phase_2_executive_summary.md)**
    - Factor selection and estimation results
    - Beta stability analysis
    - Rolling vs. stable beta comparison

11. **Factor Methodology (playground/docs/factor_methodology.md)**
    - Duration, credit, and liquidity factor definitions
    - PCA-based factor extraction
    - Orthogonalization procedures

---

## Appendix A: Timeline Diagrams

### Primary Split Visualization

```
========================================================================
                    PRIMARY TRAIN/TEST SPLIT
========================================================================

Training Period: 2010-01-01 to 2018-12-31 (8 years, ~2,000 trading days)
Testing Period:  2019-01-01 to 2024-12-31 (6 years, ~1,500 trading days)

                                     Split Point
                                          ↓
2010    2012    2014    2016    2018 | 2019    2021    2023    2024
|-------|-------|-------|-------|-----+-----|-------|-------|-------|
|                                     |                               |
|          TRAINING PERIOD            |        TESTING PERIOD         |
|          Factor Parameters          |     Parameters FROZEN         |
|          Estimated Here             |     Beta Estimates Rolling    |
|                                     |                               |
========================================================================

Key Events in Training Period:
- 2010-2011: Post-crisis recovery
- 2013: Taper tantrum volatility
- 2015: Oil price crash
- 2017-2018: Rate hike cycle begins

Key Events in Testing Period:
- 2020: COVID-19 pandemic
- 2021-2022: Inflation surge
- 2022-2023: Aggressive rate hikes
- 2024: Election year volatility
```

### Walk-Forward Analysis Visualization

```
========================================================================
                  WALK-FORWARD ANALYSIS (10 FOLDS)
========================================================================

Fold 1:  [2010────2014] → [2015]
Fold 2:       [2011────2015] → [2016]
Fold 3:            [2012────2016] → [2017]
Fold 4:                 [2013────2017] → [2018]
Fold 5:                      [2014────2018] → [2019]
Fold 6:                           [2015────2019] → [2020] ← COVID
Fold 7:                                [2016────2020] → [2021]
Fold 8:                                     [2017────2021] → [2022] ← Hikes
Fold 9:                                          [2018────2022] → [2023]
Fold 10:                                              [2019────2023] → [2024]

Legend:
[────] = 5-year training window
  [x]  = 1-year testing window

Properties:
- Training windows: 5 years (1,260 trading days)
- Testing windows: 1 year (252 trading days)
- Step size: 1 year (252 trading days)
- Total folds: 10
- Training overlap: 80% (4 years shared between consecutive folds)
- Testing overlap: 0% (fully disjoint test periods)
```

### Rolling Beta Estimation Timeline

```
========================================================================
                   ROLLING BETA ESTIMATION
========================================================================

Example: Monthly rebalances in 2020 with 252-day (1-year) windows

Rebalance    Rolling Window (252 trading days)          Beta Estimate
Date         Start → End                                 Used For
------------------------------------------------------------------------
2020-01-31   [2019-02-01 → 2020-01-31]  ──────────────→  Jan 31 rebalance
2020-02-28   [2019-03-01 → 2020-02-28]  ──────────────→  Feb 28 rebalance
2020-03-31   [2019-04-01 → 2020-03-31]  ──────────────→  Mar 31 rebalance
                          ↑ COVID crash occurs mid-window
2020-04-30   [2019-05-01 → 2020-04-30]  ──────────────→  Apr 30 rebalance
2020-05-31   [2019-06-01 → 2020-05-31]  ──────────────→  May 31 rebalance
                          ↑ Window now includes crash
2020-06-30   [2019-07-01 → 2020-06-30]  ──────────────→  Jun 30 rebalance
...

Properties:
- Window length: 252 trading days (≈1 calendar year)
- Update frequency: Monthly (every rebalance)
- Temporal constraint: window_end ≤ rebalance_date (no look-ahead)
- Adaptive: Captures evolving factor exposures
```

---

## Appendix B: Validation Checklist

### Pre-Execution Checklist

Before running any backtest:

- [ ] Dataset loaded and validated
- [ ] Train/test split defined correctly
- [ ] Factor parameters estimated from training data only
- [ ] Configuration parameters set appropriately
- [ ] Random seed fixed for reproducibility
- [ ] Logging configured and enabled

### Post-Execution Checklist

After completing backtest:

- [ ] Results are reproducible (same seed → same results)
- [ ] Performance metrics within expected ranges
- [ ] No date stamps from training period in test results
- [ ] Transaction costs properly accounted for
- [ ] Beta estimates show reasonable stability
- [ ] Rolling windows respect temporal boundaries
- [ ] Documentation updated with findings

### Code Review Checklist

When reviewing backtest code:

- [ ] All date filters use `<=` (not `<`) for inclusive ranges
- [ ] Rolling windows computed as `[date - lookback, date]`
- [ ] No parameter re-estimation in testing loop
- [ ] Point-in-time constraints verified
- [ ] Test assertions validate temporal correctness
- [ ] Factor parameters frozen after training
- [ ] No forward-filling across train/test boundary
- [ ] Documentation clear on data usage

---

**Document Version History:**

- v1.0 (October 8, 2025): Initial methodology documentation for Phase 3.2.1
- Future versions will document walk-forward implementation and regime analysis

**Maintenance:**

This document should be updated when:
- Backtesting methodology changes
- New validation procedures are added
- Issues or edge cases are discovered
- Walk-forward analysis is fully implemented
- Production deployment modifies procedures

**Contact:**

For questions or clarifications regarding this methodology:
- Review Phase 3.2.1 section of 3D_Risk_Model_Roadmap.md
- Consult playground/backtest/splits.py implementation
- Examine playground/tests/backtest/test_splits.py for validation examples
