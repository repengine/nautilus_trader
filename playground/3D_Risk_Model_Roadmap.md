# 3D Factor Risk Model: Development Roadmap

## Document Status
- **Version:** 1.2
- **Last Updated:** 2025-10-17
- **Status:** Phase 2 Complete → Phase 3 (Backtest validation) in steady iteration; Phase 4 (Strategy integration) not started

## Playground Snapshot (2025-10-16)

- Backtest runner exercises stable/rolling betas, turnover smoothing, regime-aware liquidity scaling.  
- Walk-forward harness (10 folds, 5y/1y) and liquidity experiments regenerate nightly; turnover smoothing currently default candidate (Sharpe ≈0.94, TC savings ≈$2.6k).  
- New microbench (`pytest -q playground/tests/performance/test_turnover_smoothing.py -m performance`) keeps `FactorTiltStrategy.compute_weights` under 5 ms.  
- Outstanding for live module: execution limits, monitoring hooks, Nautilus Trader actor wiring, and codifying parameter decisions in `playground/3D_Risk_Model_Idea.md`.

---

## Executive Summary

### Current State (Phase 1: ✅ COMPLETE)
We have successfully built a **working 3D risk visualization prototype** with:
- ✅ Factor data pipeline (FRED API integration for duration, credit, liquidity)
- ✅ EWMA beta calculation engine for sector factor exposures
- ✅ Portfolio optimization using Sharpe ratio weighting
- ✅ Stable coordinate system (sectors stay in clouds, portfolio moves through space)
- ✅ Interactive Three.js 3D visualization
- ✅ JSON payload generation for 2010-2024 historical data

### Remaining Gaps
- ❌ No empirical validation that the model generates alpha
- ❌ No statistical significance testing of factor loadings
- ❌ No out-of-sample backtesting
- ❌ No comparison to industry-standard benchmarks
- ❌ Theoretical justification for stable vs. time-varying betas unclear

### Objective
Transform the current **proof-of-concept** into an **academically rigorous, production-ready** quantitative investment system.

---

## Phase 2: Statistical Validation & Factor Model Testing
**Duration:** 2-3 weeks
**Objective:** Validate that the 3-factor model has explanatory power and statistical significance

### 2.1 Factor Model Specification & Validation

#### Tasks

**2.1.1 Verify Factor Return Calculation**
- [x] Audit current factor construction methodology
  - Confirm using factor CHANGES (returns) not LEVELS
  - Document exact formulas for each factor
  - Verify data alignment and missing data handling
- [x] Compare against original plan specification (3D_Risk_Model_Idea.md lines 251-252)
- [x] Create unit tests for factor return calculations

**Acceptance Criteria:**
- Factor returns match specification: `factor_returns = factor_data.diff()`
- Documentation clearly states: regression on ΔDuration, ΔCredit, ΔLiquidity
- Unit tests achieve 100% coverage on factor calculation functions

**Deliverables:**
- ✅ `playground/docs/factor_methodology.md` (full mathematical specification and alignment notes)
- ✅ `playground/tests/unit/risk_model/test_factor_returns.py` (comprehensive unit coverage)

---

**2.1.2 Regression Diagnostics for Each Sector**

Run full regression diagnostics for each sector ETF:

```python
# For each sector (XLU, XLK, XLF, etc.):
# R_sector,t = α + β_dur*ΔDuration_t + β_cred*ΔCredit_t + β_liq*ΔLiquidity_t + ε_t
```

- [x] Calculate and report for each sector:
  - **Adjusted R²**: Does the model explain variance? (Target: R² > 0.30)
  - **t-statistics**: Are betas statistically significant? (Target: |t| > 2.0, p < 0.05)
  - **F-statistic**: Is the overall model significant?
  - **Durbin-Watson**: Test for autocorrelation in residuals (Target: 1.5 < DW < 2.5)
  - **Breusch-Pagan test**: Test for heteroskedasticity
  - **VIF (Variance Inflation Factor)**: Test for multicollinearity (Target: VIF < 5)

- [x] Create regression diagnostics report with plots:
  - Residual plots (QQ-plot, residuals vs fitted)
  - Actual vs predicted returns scatter
  - Rolling R² over time

**Acceptance Criteria:**
- At least 70% of sectors have R² > 0.30
- At least 2/3 factor betas per sector are significant (p < 0.05)
- No severe heteroskedasticity or autocorrelation issues
- Factors show low multicollinearity (VIF < 5)

**Deliverables:**
- ✅ `playground/docs/regression_diagnostics.md` (full statistical narrative with plots referenced)
- ✅ `playground/risk_model/diagnostics.py` (regression diagnostics implementation)
- ✅ `playground/tests/unit/risk_model/test_diagnostics.py` (unit coverage)

---

**2.1.3 Factor Correlation & Orthogonality Analysis**

Test whether the 3 factors are independent (low correlation):

- [x] Calculate pairwise correlations between factors
  - Target: |correlation| < 0.50 for all pairs
- [x] Run Principal Component Analysis (PCA) on factor returns
  - Do we get ~3 principal components?
  - How much variance does each PC explain?
- [x] Test if factors are better than arbitrary linear combinations

**Acceptance Criteria:**
- Factors show reasonable independence (|r| < 0.50)
- PCA confirms 3 factors capture >80% of variance
- Documentation explains economic interpretation of each factor

**Deliverables:**
- ✅ `playground/docs/factor_correlation_analysis.md`
- ✅ `playground/docs/pca_sector_returns.md`
- ✅ `playground/risk_model/factor_analysis.py`, `playground/risk_model/pca_validation.py`
- ✅ `playground/tests/unit/risk_model/test_factor_analysis.py`, `.../test_pca_validation.py`

---

### 2.2 Stable vs. Time-Varying Beta Analysis

#### Tasks

**2.2.1 Implement Rolling Beta Estimation**

- [x] Build rolling window beta estimation (academic standard approach)
  ```python
  def compute_rolling_betas(
      exposures: pl.DataFrame,
      window_days: int = 252,  # 1-year rolling
  ) -> dict[str, pd.DataFrame]:
      """Compute time-varying betas for each sector"""
  ```
- [x] Compare rolling betas vs stable betas:
  - Visual comparison: plot beta evolution over time
  - Forecast accuracy: which approach better predicts next-period returns?
  - Stability: how much do rolling betas vary?

**Acceptance Criteria:**
- Rolling beta implementation tested against known benchmarks
- Comparison report shows quantitative metrics for both approaches
- Clear recommendation: use stable or rolling betas based on evidence

**Deliverables:**
- ✅ `playground/risk_model/rolling_beta.py`
- ✅ `playground/docs/rolling_beta_analysis.md`
- ✅ `playground/docs/beta_comparison_report.md`
- ✅ `playground/tests/unit/risk_model/test_rolling_beta.py`

---

**2.2.2 Economic Justification for Stable Betas**

- [x] Literature review: When are stable betas appropriate?
  - Sector ETFs vs individual stocks
  - Time horizon considerations
  - Structural break testing
- [x] Perform Chow test for structural breaks in betas
  - Test around major regime changes (2008, 2020, 2022)
- [x] Document theoretical justification

**Acceptance Criteria:**
- Chow test results show whether betas are stable across regimes
- Literature review cites 5+ academic papers
- Clear decision: stable betas justified or not

**Deliverables:**
- ✅ `playground/docs/beta_stability_justification.md`
- ✅ `playground/docs/chow_test_results.md`
- ✅ `playground/risk_model/structural_break_tests.py`
- ✅ `playground/tests/unit/risk_model/test_structural_break_tests.py`

---

### 2.3 Factor Validity via PCA on Sector Returns

Test if our factors actually drive sector returns:

**2.3.1 PCA on Sector Return Matrix**

- [x] Run PCA on historical sector returns (all 9 sectors)
- [x] Extract top 3 principal components
- [x] Compare PC loadings to our factor betas
  - Do PC1, PC2, PC3 align with duration, credit, liquidity?
  - If not, what do they represent?

**Acceptance Criteria:**
- Top 3 PCs explain >70% of sector return variance
- PC loadings correlate (|r| > 0.60) with our factor betas
- If no alignment: iterate on factor definitions

**Deliverables:**
- ✅ `playground/docs/pca_sector_returns.md`
- ✅ `playground/docs/factor_correlation_analysis.md` (alignment summary)
- ✅ `playground/risk_model/pca_validation.py`
- ✅ `playground/tests/unit/risk_model/test_pca_validation.py`

---

## Phase 3: Backtesting & Performance Validation
**Duration:** 3-4 weeks
**Objective:** Prove the model generates risk-adjusted returns competitive with benchmarks

**Status Update (2025-10-17):**
- Backtesting engine, benchmark suite, and attribution pipeline are stable with regression-tested coverage.
- Walk-forward harness (5y/1y, stride 1y) and liquidity mitigation experiments regenerate nightly via `make export-phase3-walk-forward`; turnover smoothing is the leading configuration (Sharpe ≈0.94, TC savings ≈$2.6k).
- Turnover smoothing compute-weight microbench keeps hot-path work under 5 ms (`pytest -q playground/tests/performance/test_turnover_smoothing.py -m performance`).
- Central defaults now live in `ml.config.playground.ThreeDRiskBacktestDefaults`; the backtest runner, sensitivity harness, and Phase 3 CLI consume the shared config and surface missing-baseline diagnostics in tests.
- Walk-forward summaries now emit `metadata.json` capturing the defaults/liquidity config used, and the mitigation suite includes a "Turnover Stress Test" scenario to probe higher transaction-cost risk.
- Accepted parameter values have been codified in `playground/docs/nautilus_strategy_spec.md`, tying the Nautilus integration plan directly to the shared defaults.
- Monitoring helpers now validate walk-forward metadata and log alerts when defaults drift, and the visuals export surfaces the metadata summary path for dashboards.
- Added `check_walk_forward_metadata` CLI to enable cron/Grafana health checks with non-zero exit on drift.
- Remaining Phase 3 focus areas: keep nightly monitoring/reporting aligned with the new metadata outputs and extend deeper validation: long-horizon walk-forward permutations, Monte Carlo stress sweeps, parameter response heatmaps, extra diagnostic metrics, alternate datasets, and automated nightly dashboards/alerts.

### 3.1 Backtest Infrastructure

#### Tasks

**3.1.1 Build Backtesting Engine**

- [x] Implement backtesting framework with:
  - Rolling window optimization (monthly rebalance)
  - Transaction cost modeling (default: 10 bps per trade)
  - Slippage assumptions
  - Position size constraints
  - Rebalancing thresholds

```python
class FactorBacktester:
    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        rebalance_frequency: str = "monthly",
        transaction_cost_bps: float = 10.0,
    ):
        ...

    def run_backtest(
        self,
        strategy: str,  # "3d_factor", "60_40", "risk_parity", etc.
    ) -> BacktestResult:
        ...
```

**Acceptance Criteria:**
- Backtester handles corporate actions (splits, dividends)
- Realistic transaction costs included
- Results reproducible with fixed random seed
- Unit tests validate against hand-calculated examples

**Deliverables:**
- `playground/backtest/engine.py`
- `playground/backtest/strategies.py`
- `playground/tests/backtest/test_engine.py`
- `playground/tests/backtest/test_runner.py`

---

**3.1.2 Implement Benchmark Strategies**

Build comparison strategies:

- [x] **60/40 Portfolio**
  - 60% SPY, 40% AGG
  - Monthly rebalance
- [x] **Risk Parity**
  - Equal risk contribution across asset classes
  - Volatility targeting
- [x] **Minimum Variance**
  - Optimize for lowest portfolio variance
  - Use same sector universe
- [x] **Equal Weight**
  - 1/N across all sectors
  - Monthly rebalance

**Acceptance Criteria:**
- All benchmark strategies implemented and tested
- Historical performance aligns with published benchmarks (where available)
- Same transaction cost assumptions applied

**Deliverables:**
- `playground/backtest/benchmarks.py`
- `playground/tests/backtest/test_benchmarks.py`
- `playground/tests/backtest/test_runner.py`

---

### 3.2 Out-of-Sample Testing

#### Tasks

**3.2.1 Train/Test Split Design**

- [x] Define training and testing periods:
  - **Training:** 2010-01-01 to 2018-12-31 (8 years)
  - **Testing:** 2019-01-01 to 2024-12-31 (6 years)
- [x] Alternative: Walk-forward analysis
  - Train on year 1-5, test year 6
  - Roll forward 1 year, repeat
- [x] Document prevention of look-ahead bias

**Acceptance Criteria:**
- Clear temporal separation (no future data leakage)
- Factor parameters estimated only on training data
- Beta estimates use rolling windows within sample period

**Deliverables:**
- `playground/docs/backtesting_methodology.md`
- `playground/backtest/splits.py`
- `playground/reports/backtesting/walk_forward/aggregate_metrics.csv`

---

**3.2.2 Run Full Backtest Suite**

- [x] Run backtests for all strategies:
  - 3D Factor Model (stable betas)
  - 3D Factor Model (rolling betas)
  - 60/40 benchmark
  - Risk parity
  - Minimum variance
  - Equal weight

- [x] Calculate performance metrics:
  - **Return Metrics:**
    - Annualized return
    - Cumulative return
    - Monthly returns distribution
  - **Risk Metrics:**
    - Annualized volatility
    - Maximum drawdown
    - Value at Risk (95%, 99%)
    - Conditional Value at Risk (CVaR)
  - **Risk-Adjusted Metrics:**
    - Sharpe ratio
    - Sortino ratio
    - Calmar ratio (return / max drawdown)
    - Information ratio (vs benchmark)
  - **Trade Metrics:**
  - Turnover rate
  - Transaction costs (% of returns)
  - Number of rebalances

**Acceptance Criteria:**
- All metrics calculated for train and test periods separately
- 3D Factor Model Sharpe ratio > 0.50 (test period)
- 3D Factor Model beats 60/40 in Sharpe ratio OR Calmar ratio (test period)
- Results consistent across multiple random seeds

**Deliverables:**
- `playground/reports/backtesting/backtest_results_2010_2024.md`
- `playground/reports/backtesting/performance_comparison_table.csv`
- `playground/reports/backtesting/train_vs_test_metrics.csv`
- `playground/reports/backtesting/notes/phase3_summary.md`
- `playground/backtest/performance_metrics.py`
- `playground/scripts/export_phase3_visuals.py`

---

**3.2.3 Regime Analysis**

Test performance across different market regimes:

```python
regimes = {
    "GFC Aftermath": ("2010-01-01", "2011-12-31"),
    "QE Era": ("2012-01-01", "2015-12-31"),
    "Rate Normalization": ("2016-01-01", "2019-12-31"),
    "COVID Crash": ("2020-02-01", "2020-04-30"),
    "Zero Rates": ("2020-05-01", "2021-12-31"),
    "Rate Hiking Cycle": ("2022-01-01", "2023-12-31"),
    "Recent": ("2024-01-01", "2024-12-31"),
}
```

- [x] Calculate Sharpe ratio for each regime
- [x] Identify regimes where model fails
- [x] Analyze why (factor breakdown, correlation shifts)

**Acceptance Criteria:**
- Performance reported for all 7 regimes
- Model works in at least 5/7 regimes (Sharpe > 0)
- Clear documentation of failure modes

**Deliverables:**
- `playground/reports/backtesting/regime_summary.csv`
- `playground/reports/backtesting/regime_comparison.csv`
- `playground/docs/backtest_attribution_visuals.md`
- `playground/backtest/regime_analysis.py`

---

### 3.3 Factor Attribution Analysis

#### Tasks

**3.3.1 Performance Attribution**

Decompose portfolio returns into factor contributions:

- [x] For each month, calculate:
  ```python
  R_portfolio = α + β_dur * R_dur + β_cred * R_cred + β_liq * R_liq + ε
  ```
- [x] Report:
  - Alpha (skill-based returns)
  - Factor contribution to returns
  - Residual (unexplained returns)

**Acceptance Criteria:**
- Attribution sums to total portfolio return
- Statistical significance of alpha tested (t-stat > 2.0)
- Clear visualization of factor contributions over time

**Deliverables:**
- `playground/reports/backtesting/attribution/*.csv`
- `playground/reports/backtesting/visuals/attribution_waterfall.png`
- `playground/scripts/export_phase3_visuals.py`
- `playground/backtest/runner.py`

---

## Phase 4: Refinement & Robustness Testing
**Duration:** 2-3 weeks
**Objective:** Stress-test assumptions and improve model robustness

### 4.1 Sensitivity Analysis

#### Tasks

**4.1.1 Parameter Sensitivity Tests**

Test sensitivity to key parameters:

- [ ] EWMA alpha (current: default, test: 0.90, 0.94, 0.96, 0.98)
- [ ] Rolling window size (test: 126, 252, 504 days)
- [ ] Rebalancing frequency (monthly, quarterly, semi-annual)
- [ ] Transaction costs (5 bps, 10 bps, 20 bps)
- [ ] Min/max weight constraints

**Acceptance Criteria:**
- Performance stable across reasonable parameter ranges
- Optimal parameters identified via grid search
- Results not overly sensitive to arbitrary choices

**Deliverables:**
- `reports/sensitivity_analysis.pdf`
- `playground/backtest/parameter_search.py`

---

**4.1.2 Stress Testing**

Test extreme scenarios:

- [x] 1987 Black Monday (if data available)
- [x] 2008 Financial Crisis
- [x] 2020 COVID crash
- [x] 2022 Bonds+Stocks crash
- [x] Synthetic shocks (factor returns +/- 3 std dev)

**Acceptance Criteria:**
- Maximum drawdown documented for each stress scenario
- Recovery time calculated
- Comparison to benchmark drawdowns

**Deliverables:**
- `reports/stress_test_results.pdf`

---

### 4.2 Data Quality & Robustness

#### Tasks

**4.2.1 Missing Data Handling**

- [ ] Audit current missing data treatment
- [ ] Test alternative imputation methods:
  - Forward fill (current)
  - Linear interpolation
  - Kalman filter
- [ ] Document impact on results

**Acceptance Criteria:**
- Missing data rate documented (< 1% for factors)
- Sensitivity analysis shows <5% impact on Sharpe ratio

**Deliverables:**
- `docs/data_quality_report.md`

---

**4.2.2 Outlier Detection & Treatment**

- [ ] Identify outliers in factor returns (> 3 std dev)
- [ ] Winsorize vs. exclude decision
- [ ] Document impact on regression betas

**Acceptance Criteria:**
- Outlier treatment justified
- Results stable with/without outliers

**Deliverables:**
- `playground/risk_model/outlier_detection.py`

---

### 4.3 Model Improvements (If Needed)

#### Tasks (Conditional on Phase 3 Results)

**4.3.1 If Model Underperforms:**

Potential improvements to test:

- [ ] **Add 4th factor:**
  - Momentum (12-month return)
  - Value (book-to-market)
  - Quality (ROE, debt/equity)
- [ ] **Time-varying factor risk premia:**
  - Estimate expected factor returns
  - Tilt toward factors with high expected returns
- [ ] **Regime-switching model:**
  - Use HMM to detect regimes
  - Different factor weights per regime
- [ ] **Machine learning enhancements:**
  - Random forest for non-linear factor interactions
  - LSTM for time-series forecasting

**Acceptance Criteria:**
- Each improvement tested with out-of-sample validation
- Sharpe ratio improvement > 0.10
- Complexity justified by performance gain

**Deliverables:**
- `reports/model_enhancements.pdf`
- Updated code modules

---

## Phase 5: Production System Development
**Duration:** 4-6 weeks
**Objective:** Build live, automated system for daily operation

### 5.1 Data Pipeline Automation

#### Tasks

**5.1.1 Automated Data Ingestion**

- [ ] Build daily data update pipeline:
  ```python
  class DataPipeline:
      def fetch_fred_data(self) -> pl.DataFrame:
          """Fetch latest FRED factor data"""

      def fetch_sector_prices(self) -> pl.DataFrame:
          """Fetch latest sector ETF prices"""

      def validate_data(self) -> bool:
          """Run data quality checks"""

      def update_database(self) -> None:
          """Persist to database"""
  ```

- [ ] Implement data validation:
  - Check for missing values
  - Check for stale data (> 3 days old)
  - Sanity checks (no returns > 20% in a day)

**Acceptance Criteria:**
- Pipeline runs via cron job (daily 6pm EST)
- Email alerts on data quality failures
- 99.9% uptime over 30 days

**Deliverables:**
- `ml/data/pipelines/risk_model_pipeline.py`
- `scripts/update_risk_model_data.sh`
- `tests/test_data_pipeline.py`

---

**5.1.2 Database Integration**

- [ ] Persist data to PostgreSQL:
  - `ml_factor_returns` table (FRED data)
  - `ml_sector_prices` table (ETF prices)
  - `ml_sector_betas` table (EWMA betas)
  - `ml_portfolio_positions` table (current allocations)

- [ ] Implement data versioning
- [ ] Set up backup strategy

**Acceptance Criteria:**
- Database schema follows CLAUDE.md standards
- All tables include `ts_event`, `ts_init`, `instrument_id`
- Daily backups configured

**Deliverables:**
- `ml/schema/risk_model.sql`
- `ml/stores/risk_model_store.py`

---

### 5.2 Real-Time Portfolio Construction

#### Tasks

**5.2.1 Portfolio Optimization Service**

- [ ] Build service to compute optimal portfolio daily:
  ```python
  class RiskModelPortfolioOptimizer:
      def compute_optimal_weights(
          self,
          current_betas: dict[str, dict[str, float]],
          factor_forecasts: dict[str, float],
          constraints: PortfolioConstraints,
      ) -> dict[str, float]:
          """Return optimal sector weights"""
  ```

**Acceptance Criteria:**
- Optimization runs in < 5 seconds
- Constraints enforced (min/max weights, turnover limits)
- Results logged to database

**Deliverables:**
- `playground/risk_model/optimizer_service.py`

---

**5.2.2 Position Sizing & Rebalancing Logic**

- [ ] Implement rebalancing rules:
  - Trigger: monthly OR weight deviation > 5%
  - Transaction cost awareness
  - Tax loss harvesting considerations (optional)

**Acceptance Criteria:**
- Rebalancing logic tested with historical data
- Turnover matches backtest assumptions

**Deliverables:**
- `playground/risk_model/rebalancer.py`

---

### 5.3 Risk Monitoring Dashboard

#### Tasks

**5.3.1 Build Monitoring Dashboard**

- [ ] Grafana dashboard showing:
  - Current portfolio position in 3D space
  - Factor exposures (duration, credit, liquidity)
  - Daily/weekly/monthly returns
  - Sharpe ratio (rolling 252 days)
  - Current vs target weights
  - Recent trades

**Acceptance Criteria:**
- Dashboard updates in real-time (< 60s latency)
- Accessible via web browser
- Mobile-responsive

**Deliverables:**
- `grafana/dashboards/risk_model_dashboard.json`

---

**5.3.2 Alerting System**

- [ ] Implement alerts for:
  - Maximum drawdown exceeded (> -15%)
  - Factor exposure outside bounds
  - Data pipeline failures
  - Optimization errors

**Acceptance Criteria:**
- Alerts sent via email/Slack
- No false positives over 7 days
- Alert response playbook documented

**Deliverables:**
- `playground/risk_model/alerts.py`
- `docs/alert_response_playbook.md`

---

### 5.4 Compliance & Documentation

#### Tasks

**5.4.1 Model Documentation**

- [ ] Write comprehensive model documentation:
  - Factor definitions
  - Beta estimation methodology
  - Optimization approach
  - Risk management rules
  - Historical performance
  - Known limitations

**Acceptance Criteria:**
- Documentation suitable for regulatory review
- Mathematical notation consistent
- All assumptions explicitly stated

**Deliverables:**
- `docs/risk_model_specification.pdf` (20-30 pages)

---

**5.4.2 Audit Trail**

- [ ] Implement full audit trail:
  - Every optimization logged with inputs/outputs
  - Every trade with rationale
  - Every rebalance with before/after weights

**Acceptance Criteria:**
- Audit trail allows full reconstruction of decisions
- Logs retained for 7 years
- Queryable via SQL

**Deliverables:**
- `ml/stores/audit_trail_store.py`
- `ml/schema/audit_trail.sql`

---

## Phase 6: Academic Publication & External Validation
**Duration:** 6-8 weeks (parallel with Phase 5)
**Objective:** Achieve peer review and industry recognition

### 6.1 Research Paper Preparation

#### Tasks

**6.1.1 Draft Academic Paper**

Paper structure:
1. **Abstract** (200 words)
2. **Introduction** (3-4 pages)
   - Motivation
   - Contribution to literature
   - Preview of results
3. **Literature Review** (4-5 pages)
   - Factor models (Fama-French, Carhart)
   - Sector rotation strategies
   - Risk parity approaches
4. **Methodology** (6-8 pages)
   - Factor construction
   - Beta estimation (EWMA)
   - Portfolio optimization
   - Backtesting approach
5. **Data** (2-3 pages)
   - Data sources
   - Sample period
   - Descriptive statistics
6. **Results** (8-10 pages)
   - Factor model validation
   - Backtest performance
   - Regime analysis
   - Robustness tests
7. **Conclusion** (2 pages)
8. **Appendices**
   - Mathematical derivations
   - Detailed tables

**Acceptance Criteria:**
- Draft complete with all sections
- Figures production-quality
- Citations formatted (APA/Chicago)
- Proofread by 2+ reviewers

**Deliverables:**
- `papers/3d_factor_model_v1.pdf`

---

**6.1.2 Preprint Publication**

- [ ] Submit to SSRN (Social Science Research Network)
- [ ] Submit to arXiv (if allowed for quantitative finance)
- [ ] Share on Twitter/LinkedIn for feedback

**Acceptance Criteria:**
- Preprint published with DOI
- Feedback incorporated from 5+ industry professionals

**Deliverables:**
- Published preprint with link

---

**6.1.3 Peer Review Submission**

Target journals (ranked by prestige):

1. **Tier 1:**
   - Journal of Finance
   - Review of Financial Studies
   - Journal of Financial Economics

2. **Tier 2:**
   - Financial Analysts Journal
   - Journal of Portfolio Management
   - Journal of Investment Strategies

3. **Tier 3:**
   - Journal of Asset Management
   - Quantitative Finance
   - Journal of Risk

- [ ] Submit to target journal
- [ ] Address reviewer comments
- [ ] Revise and resubmit

**Acceptance Criteria:**
- Submission confirmation received
- Paper accepted or major revisions (success)
- If rejected: resubmit to next tier

**Deliverables:**
- Published academic paper (12-18 month timeline)

---

### 6.2 Industry Validation

#### Tasks

**6.2.1 Conference Presentations**

Target conferences:
- CFA Institute Annual Conference
- Quantitative Finance conferences (QuantMinds, etc.)
- Regional CFA society events

- [ ] Submit abstracts
- [ ] Prepare presentation slides
- [ ] Present and collect feedback

**Acceptance Criteria:**
- Accepted to at least 1 conference
- Presentation delivered to 50+ attendees

**Deliverables:**
- `presentations/3d_factor_model_conference.pdf`

---

**6.2.2 Industry Outreach**

- [ ] Share model with:
  - Portfolio managers (request feedback)
  - Risk managers (stress test validation)
  - Quantitative researchers (peer review)
- [ ] Publish blog post / Medium article
- [ ] Create open-source implementation (GitHub)

**Acceptance Criteria:**
- Feedback from 10+ industry professionals
- Blog post reaches 1000+ views
- GitHub repo stars > 50

**Deliverables:**
- Public GitHub repository
- Blog post with backtest results

---

## Success Criteria by Phase

### Phase 2: Statistical Validation
- ✅ R² > 0.30 for 70%+ of sectors
- ✅ Factor betas statistically significant (p < 0.05)
- ✅ Factors show low multicollinearity (VIF < 5)
- ✅ Decision made: stable vs rolling betas

### Phase 3: Backtesting
- ✅ 3D Factor Model Sharpe ratio > 0.50 (out-of-sample)
- ✅ Beats 60/40 in Sharpe OR Calmar ratio
- ✅ Works in 5/7 market regimes
- ✅ Alpha statistically significant (t > 2.0)

### Phase 4: Robustness
- ✅ Performance stable across parameter variations
- ✅ Maximum drawdown < 30% (all periods)
- ✅ Outlier treatment justified

### Phase 5: Production
- ✅ Data pipeline 99.9% uptime over 30 days
- ✅ Optimization runs in < 5 seconds
- ✅ Dashboard operational with zero downtime
- ✅ Full audit trail implemented

### Phase 6: Publication
- ✅ Preprint published on SSRN
- ✅ Submitted to peer-reviewed journal
- ✅ Presented at 1+ industry conference
- ✅ Open-source implementation released

---

## Resource Requirements

### Software Dependencies

**Python Packages:**
```toml
[tool.poetry.dependencies]
python = "^3.11"
polars = "^0.19"
numpy = "^1.24"
pandas = "^2.0"
scikit-learn = "^1.3"
scipy = "^1.11"
statsmodels = "^0.14"  # For regression diagnostics
fredapi = "^0.5"       # FRED API
yfinance = "^0.2"      # Market data
matplotlib = "^3.7"
seaborn = "^0.12"
plotly = "^5.17"       # Interactive plots
cvxpy = "^1.4"         # Convex optimization
pytest = "^7.4"
hypothesis = "^6.88"   # Property-based testing

[tool.poetry.group.backtest]
backtrader = "^1.9"    # Alternative: vectorbt, zipline
quantstats = "^0.0.62" # Performance metrics

[tool.poetry.group.production]
postgresql = "^11.0"
sqlalchemy = "^2.0"
prometheus-client = "^0.17"
grafana-api = "^1.0"
```

**Infrastructure:**
- PostgreSQL database (>= 11.0)
- Grafana for dashboards
- Prometheus for metrics
- Cron for scheduling

### Data Requirements

**Historical Data:**
- FRED API key (free): https://fred.stlouisfed.org/docs/api/api_key.html
- Yahoo Finance (free, via yfinance)
- Optional: Polygon.io ($99-249/mo) for production

**Storage:**
- Local development: ~5 GB
- Production: ~20 GB (7 years audit trail)

### Compute Requirements

**Development:**
- CPU: 4+ cores
- RAM: 16 GB
- Disk: 50 GB SSD

**Production:**
- CPU: 2+ cores
- RAM: 8 GB
- Disk: 100 GB SSD
- Uptime: 99.9%

### Time Investment

**Phase 2:** 80-120 hours (1 person, 2-3 weeks)
**Phase 3:** 120-160 hours (1 person, 3-4 weeks)
**Phase 4:** 60-90 hours (1 person, 2-3 weeks)
**Phase 5:** 120-180 hours (1 person, 4-6 weeks)
**Phase 6:** 150-250 hours (1 person, 6-8 weeks, parallel with Phase 5)

**Total: 530-800 hours (13-20 weeks full-time equivalent)**

---

## Risk Register

### Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Factor model has low R² | High | Medium | Test alternative factor specifications early (Phase 2) |
| Model doesn't beat benchmarks | High | Medium | Implement improvements (Phase 4.3) |
| Data quality issues | Medium | Low | Robust validation pipeline (Phase 5.1) |
| Overfitting to training data | High | Medium | Strict train/test separation, multiple regimes |
| Production system downtime | Medium | Low | Redundancy, monitoring, alerts |

### Academic Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Paper rejected by journals | Medium | High | Submit to multiple tiers, incorporate feedback |
| Results not novel enough | High | Medium | Emphasize unique contributions (stable betas, 3D visualization) |
| Insufficient statistical rigor | High | Low | Follow checklist in Phase 2 |

### Business Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Model underperforms live | High | Medium | Paper trading period before real capital |
| Regulatory concerns | Medium | Low | Full compliance documentation (Phase 5.4) |
| Competition from similar models | Low | High | Publish quickly, establish priority |

---

## Decision Points

### After Phase 2 (Statistical Validation)
**Decision:** Proceed to backtesting OR iterate on factors?

**Criteria:**
- **Proceed if:** R² > 0.30 for 70%+ sectors, betas significant
- **Iterate if:** R² < 0.20 for 50%+ sectors, factors correlated (|r| > 0.70)

---

### After Phase 3 (Backtesting)
**Decision:** Proceed to production OR improve model?

**Criteria:**
- **Proceed if:** Sharpe > 0.50, beats 60/40 in Sharpe OR Calmar
- **Improve if:** Sharpe < 0.30, loses to 60/40 in all metrics
- **Conditional proceed:** 0.30 < Sharpe < 0.50, test improvements in Phase 4.3

---

### After Phase 4 (Robustness)
**Decision:** Deploy to production OR abandon?

**Criteria:**
- **Deploy if:** All Phase 4 success criteria met, improvements yield Sharpe > 0.50
- **Abandon if:** No path to Sharpe > 0.40 after all improvements tested

---

## Appendix A: Acceptance Testing Checklist

### Phase 2 Checklist
- [ ] All factor calculations documented with formulas
- [ ] Regression diagnostics report generated for all 9 sectors
- [ ] R² > 0.30 for 70%+ of sectors (6+ out of 9)
- [ ] VIF < 5 for all factor pairs
- [ ] Stable vs rolling beta decision documented with evidence
- [ ] PCA shows 3 factors explain >70% variance
- [ ] All unit tests pass (pytest coverage > 90%)

### Phase 3 Checklist
- [ ] Backtesting engine handles corporate actions correctly
- [ ] All 5 benchmark strategies implemented and tested
- [ ] Train/test split prevents look-ahead bias (verified by code review)
- [ ] Performance metrics calculated for all strategies
- [ ] 3D Factor Model Sharpe > 0.50 (test period)
- [ ] Regime analysis covers 7 regimes with results documented
- [ ] Factor attribution sums to total return (verified within 1 bps)

### Phase 4 Checklist
- [ ] Sensitivity analysis covers 5+ key parameters
- [ ] Performance stable (Sharpe delta < 0.15) across reasonable ranges
- [ ] Stress test results documented for 5 scenarios
- [ ] Missing data rate < 1%, impact on Sharpe < 5%
- [ ] Outlier treatment justified and tested

### Phase 5 Checklist
- [ ] Data pipeline runs daily via cron
- [ ] Pipeline uptime > 99.9% over 30 days
- [ ] Database schema compliant with CLAUDE.md
- [ ] Optimization completes in < 5 seconds (99th percentile)
- [ ] Grafana dashboard operational, <60s latency
- [ ] Alert system tested, no false positives over 7 days
- [ ] Model documentation complete (20-30 pages)
- [ ] Audit trail allows full reconstruction of all decisions

### Phase 6 Checklist
- [ ] Paper draft complete, 30-40 pages
- [ ] Preprint published on SSRN with DOI
- [ ] Submitted to peer-reviewed journal
- [ ] Conference presentation accepted and delivered
- [ ] GitHub repo published with >50 stars
- [ ] Blog post published with >1000 views

---

## Appendix B: Key References

### Academic Papers
1. Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.
2. Carhart, M. M. (1997). On persistence in mutual fund performance. *Journal of Finance*, 52(1), 57-82.
3. Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). Value and momentum everywhere. *Journal of Finance*, 68(3), 929-985.

### Industry Resources
- CFA Institute: Factor Investing Research
- AQR Capital: Factor Library
- BlackRock: Factor Investing Guides

### Software Documentation
- FRED API: https://fred.stlouisfed.org/docs/api/
- Polars: https://pola-rs.github.io/polars/
- cvxpy: https://www.cvxpy.org/

---

## Document Control

| Version | Date       | Author | Changes |
|---------|------------|--------|---------|
| 1.0     | 2025-10-05 | Claude | Initial roadmap based on Phase 1 completion |

**Next Review Date:** After Phase 2 completion (estimated 2025-10-26)
- [x] Extend walk-forward validation to multiple horizon permutations (vary training years, testing years, stride length) and nested cross-validation runs (multi-horizon artefacts exported via `run_multi_horizon_walk_forward_analysis` with nested summaries).
- [x] Execute Monte Carlo and bootstrapped stress suites (randomized regime orderings, macro shock overlays) with automated reporting.
- [ ] Generate parameter response heatmaps covering turnover smoothing, transaction costs, liquidity multipliers, and beta window lengths.
- [ ] Track additional diagnostics (tail risk metrics, turnover distributions, alternative benchmarks) and include them in nightly exports.
- [ ] Validate robustness on proxy datasets (international sectors, factor ETFs) and vintage simulations to measure adaptation speed after regime breaks.
- [ ] Automate metadata-driven dashboards/alerts summarising the above analyses for nightly monitoring.
