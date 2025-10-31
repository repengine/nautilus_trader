"""
Regime analysis framework for testing strategy performance across market conditions.

This module provides comprehensive tools for analyzing portfolio strategy performance
across different market regimes, identifying failure modes, and understanding how
macroeconomic conditions affect strategy returns.

Key Features:
- Standard market regime definitions (2010-2024)
- Performance metrics calculation per regime
- Failure mode identification and analysis
- Multi-strategy comparison across regimes
- Economic context and key events for each regime
- Markdown report generation

Performance Targets (Cold Path):
- Regime analysis: < 5 seconds for full 15-year backtest
- Report generation: < 1 second
- No performance-critical constraints (offline analysis)

Hot/Cold Path Separation:
- This is a cold-path module (regime analysis is offline)
- No real-time constraints, optimized for correctness and insight

Integration Notes:
- Compatible with BacktestResult from engine.py
- Uses PerformanceMetrics from performance_metrics.py
- Follows Phase 3.2.3 requirements from 3D_Risk_Model_Roadmap.md
- Outputs ready for strategy optimization and risk management

Examples
--------
Basic regime analysis:

>>> from datetime import datetime, UTC
>>> result = backtester.run_backtest(dataset, strategy="equal_weight")
>>> analysis = analyze_strategy_across_regimes(result)
>>> print(f"Success rate: {analysis.success_rate:.1%}")
Success rate: 85.7%
>>> failures = analysis.failure_analysis()
>>> for regime, reason in failures.items():
...     print(f"{regime}: {reason}")

Multi-strategy comparison:

>>> results = {
...     "Equal Weight": ew_result,
...     "60/40": sixty_forty_result,
...     "Risk Parity": rp_result,
... }
>>> comparison = compare_strategies_across_regimes(results)
>>> print(comparison)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import polars as pl
import structlog

from ml.config.playground import ThreeDRiskBacktestDefaults


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from playground.backtest.engine import BacktestResult


LOGGER = structlog.get_logger(__name__)


# ===== Constants =====

TRADING_DAYS_PER_YEAR = 252
MIN_OBSERVATIONS_FOR_REGIME = 20  # Minimum trading days for meaningful metrics
MIN_OBSERVATIONS_FALLBACK = 5  # Threshold where we refuse to compute metrics
BACKTEST_DEFAULTS = ThreeDRiskBacktestDefaults()


# ===== Type Definitions =====


class RegimeData(TypedDict):
    """Filtered regime data structure."""

    returns: NDArray[np.float64]
    dates: list[datetime]
    num_observations: int
    num_rebalances: int


# ===== Regime Definition Classes =====


@dataclass(slots=True, frozen=True)
class MarketRegime:
    """
    Definition of a market regime period with economic context.

    A market regime represents a distinct macroeconomic environment with
    characteristic market behavior, volatility patterns, and factor dynamics.

    Attributes
    ----------
    name : str
        Human-readable regime name (e.g., "GFC Aftermath")
    start : datetime
        Regime start date (inclusive), timezone-aware UTC
    end : datetime
        Regime end date (inclusive), timezone-aware UTC
    description : str
        Economic context describing key characteristics
        Example: "Post-financial crisis recovery with quantitative easing"
    key_events : list[str]
        Notable market events during regime
        Example: ["European debt crisis", "Flash crash May 2010"]

    Properties
    ----------
    duration_days : int
        Number of calendar days in regime

    Methods
    -------
    validate()
        Validate regime dates and duration

    Examples
    --------
    >>> regime = MarketRegime(
    ...     name="COVID Crash",
    ...     start=datetime(2020, 2, 1, tzinfo=UTC),
    ...     end=datetime(2020, 4, 30, tzinfo=UTC),
    ...     description="Pandemic-induced market shock",
    ...     key_events=["WHO declares pandemic March 11", "Circuit breakers triggered"],
    ... )
    >>> print(f"Duration: {regime.duration_days} days")
    Duration: 90 days

    Raises
    ------
    ValueError
        If end date is before start date or dates lack timezone
    """

    name: str
    start: datetime
    end: datetime
    description: str
    key_events: list[str]

    def __post_init__(self) -> None:
        """Validate regime dates."""
        if self.start.tzinfo is None:
            msg = f"Regime {self.name}: start date must be timezone-aware"
            raise ValueError(msg)

        if self.end.tzinfo is None:
            msg = f"Regime {self.name}: end date must be timezone-aware"
            raise ValueError(msg)

        if self.end <= self.start:
            msg = f"Regime {self.name}: end date must be after start date"
            raise ValueError(msg)

    @property
    def duration_days(self) -> int:
        """Calculate regime duration in calendar days."""
        return (self.end - self.start).days + 1  # Inclusive


# ===== Performance Classes =====


@dataclass(slots=True)
class RegimePerformance:
    """
    Performance metrics for a strategy in a single regime.

    This dataclass captures all key performance indicators for evaluating
    how a strategy performed during a specific market regime.

    Attributes
    ----------
    regime : MarketRegime
        Market regime definition
    strategy_name : str
        Name of the strategy being evaluated
    sharpe_ratio : float
        Annualized Sharpe ratio (risk-adjusted return)
    annualized_return : float
        Geometric annualized return (compounded)
    annualized_volatility : float
        Annualized standard deviation of returns
    max_drawdown : float
        Maximum peak-to-trough decline (negative value)
    calmar_ratio : float
        Return-to-drawdown ratio (annualized return / abs(max drawdown))
    positive_months_pct : float
        Percentage of months with positive returns (0.0 to 1.0)
    win_rate : float
        Percentage of days with positive returns (0.0 to 1.0)
    num_observations : int
        Number of trading days in regime
    num_rebalances : int
        Number of portfolio rebalancing events

    Properties
    ----------
    is_successful : bool
        True if Sharpe ratio > 0.0 (beats cash)

    Examples
    --------
    >>> perf = RegimePerformance(
    ...     regime=gfc_regime,
    ...     strategy_name="Equal Weight",
    ...     sharpe_ratio=0.85,
    ...     annualized_return=0.12,
    ...     annualized_volatility=0.15,
    ...     max_drawdown=-0.08,
    ...     calmar_ratio=1.5,
    ...     positive_months_pct=0.75,
    ...     win_rate=0.52,
    ...     num_observations=504,
    ...     num_rebalances=24,
    ... )
    >>> print(f"Success: {perf.is_successful}")
    Success: True
    """

    regime: MarketRegime
    strategy_name: str

    # Core metrics
    sharpe_ratio: float
    annualized_return: float
    annualized_volatility: float
    max_drawdown: float
    calmar_ratio: float

    # Win rate and consistency
    positive_months_pct: float
    win_rate: float

    # Regime-specific context
    num_observations: int
    num_rebalances: int

    @property
    def is_successful(self) -> bool:
        """Check if strategy was successful in this regime (Sharpe > 0)."""
        return self.sharpe_ratio > 0.0


@dataclass(slots=True)
class RegimeAnalysisResult:
    """
    Complete regime analysis across all market regimes.

    This dataclass aggregates performance metrics across all regimes,
    providing tools for cross-regime comparison and failure analysis.

    Attributes
    ----------
    strategy_name : str
        Name of the strategy being analyzed
    regime_performances : dict[str, RegimePerformance]
        Mapping of regime name to performance metrics

    Methods
    -------
    summary_table() -> pl.DataFrame
        Generate summary table of performance across regimes
    failure_analysis() -> dict[str, str]
        Identify and explain regime failures
    success_rate() -> float
        Percentage of regimes with Sharpe > 0

    Examples
    --------
    >>> analysis = analyze_strategy_across_regimes(result)
    >>> summary = analysis.summary_table()
    >>> print(summary)
    >>> failures = analysis.failure_analysis()
    >>> print(f"Success rate: {analysis.success_rate:.1%}")
    """

    strategy_name: str
    regime_performances: dict[str, RegimePerformance]

    def summary_table(self) -> pl.DataFrame:
        """
        Generate summary table of performance across regimes.

        Returns
        -------
        pl.DataFrame
            Summary table with columns:
            - regime_name: str
            - sharpe_ratio: float
            - annualized_return: float
            - annualized_volatility: float
            - max_drawdown: float
            - calmar_ratio: float
            - win_rate: float
            - num_observations: int
            - status: str ("Success" or "Failed")

        Notes
        -----
        Table is sorted chronologically by regime start date.
        """
        rows = []

        for regime_name, perf in self.regime_performances.items():
            row = {
                "regime_name": regime_name,
                "sharpe_ratio": perf.sharpe_ratio,
                "annualized_return": perf.annualized_return,
                "annualized_volatility": perf.annualized_volatility,
                "max_drawdown": perf.max_drawdown,
                "calmar_ratio": perf.calmar_ratio,
                "win_rate": perf.win_rate,
                "num_observations": perf.num_observations,
                "status": "Success" if perf.is_successful else "Failed",
            }
            rows.append(row)

        df = pl.DataFrame(rows)

        # Sort by regime chronologically (preserve order from regime_performances)
        return df

    def failure_analysis(self) -> dict[str, str]:
        """
        Identify and explain regime failures.

        Returns
        -------
        dict[str, str]
            Mapping of regime_name -> failure explanation for failed regimes
            (Sharpe ratio <= 0)

        Examples
        --------
        >>> analysis = analyze_strategy_across_regimes(result)
        >>> failures = analysis.failure_analysis()
        >>> for regime, reason in failures.items():
        ...     print(f"{regime}:")
        ...     print(f"  {reason}")
        COVID Crash:
          High volatility (2.5x normal) overwhelmed returns. Negative Sharpe
          (-0.45) indicates underperformance vs cash. Max drawdown -18.3%
          in only 3 months suggests inadequate downside protection.
        """
        failures = {}

        for regime_name, perf in self.regime_performances.items():
            if not perf.is_successful:
                # Generate detailed failure explanation
                explanation = self._generate_failure_explanation(perf)
                failures[regime_name] = explanation

        return failures

    def _generate_failure_explanation(self, perf: RegimePerformance) -> str:
        """
        Generate detailed explanation for why strategy failed in regime.

        Parameters
        ----------
        perf : RegimePerformance
            Performance metrics for failed regime

        Returns
        -------
        str
            Detailed failure explanation with specific metrics
        """
        explanations = []

        # Negative returns
        if perf.annualized_return < 0:
            explanations.append(
                f"Negative returns ({perf.annualized_return:.2%} annualized) "
                "indicate strategy lost money during this period."
            )

        # High volatility
        if perf.annualized_volatility > 0.25:  # >25% volatility is high
            explanations.append(
                f"High volatility ({perf.annualized_volatility:.1%}) suggests "
                "unstable returns and elevated risk."
            )

        # Large drawdown
        if perf.max_drawdown < -0.15:  # >15% drawdown is significant
            explanations.append(
                f"Large drawdown ({perf.max_drawdown:.1%}) indicates poor "
                "downside protection."
            )

        # Poor win rate
        if perf.win_rate < 0.45:  # <45% win rate is concerning
            explanations.append(
                f"Low win rate ({perf.win_rate:.1%}) shows strategy was wrong "
                "more often than right."
            )

        # Low positive months
        if perf.positive_months_pct < 0.40:  # <40% positive months
            explanations.append(
                f"Only {perf.positive_months_pct:.0%} of months were positive, "
                "indicating inconsistent performance."
            )

        # Combine explanations
        if explanations:
            full_explanation = " ".join(explanations)
        else:
            # Fallback for edge cases
            full_explanation = (
                f"Negative Sharpe ratio ({perf.sharpe_ratio:.2f}) indicates "
                "returns did not adequately compensate for risk taken."
            )

        # Add economic context suggestion
        full_explanation += (
            f" Economic conditions during {perf.regime.name} "
            f"({perf.regime.description}) likely contributed to underperformance."
        )

        return full_explanation

    @property
    def success_rate(self) -> float:
        """
        Calculate percentage of regimes with Sharpe > 0.

        Returns
        -------
        float
            Success rate as decimal (0.0 to 1.0)

        Examples
        --------
        >>> analysis = analyze_strategy_across_regimes(result)
        >>> print(f"Success rate: {analysis.success_rate:.1%}")
        Success rate: 85.7%
        """
        if not self.regime_performances:
            return 0.0

        successful = sum(1 for rp in self.regime_performances.values() if rp.is_successful)
        return successful / len(self.regime_performances)


# ===== Regime Definition Functions =====


def define_market_regimes() -> list[MarketRegime]:
    """
    Define standard market regimes for 2010-2024.

    This function creates the 7 standard market regimes as specified in
    Phase 3.2.3 of the 3D Risk Model Roadmap. Each regime represents a
    distinct macroeconomic environment with characteristic market dynamics.

    Returns
    -------
    list[MarketRegime]
        7 market regimes with economic context, ordered chronologically

    Notes
    -----
    Regime definitions based on major macroeconomic transitions:

    1. **GFC Aftermath (2010-2011)**: Post-financial crisis recovery with
       early QE programs and European debt crisis concerns.

    2. **QE Era (2012-2015)**: Central bank quantitative easing dominance,
       characterized by low rates and asset price inflation.

    3. **Rate Normalization (2016-2019)**: Federal Reserve rate hiking cycle
       and gradual withdrawal of monetary stimulus.

    4. **COVID Crash (Feb-Apr 2020)**: Pandemic-induced market shock with
       unprecedented volatility and economic uncertainty.

    5. **Zero Rates (May 2020-2021)**: Emergency monetary and fiscal stimulus
       supporting rapid market recovery.

    6. **Rate Hiking Cycle (2022-2023)**: Aggressive Fed rate hikes to combat
       inflation, ending easy money era.

    7. **Recent (2024)**: Current market conditions with elevated rates and
       normalization of volatility.

    Examples
    --------
    >>> regimes = define_market_regimes()
    >>> for regime in regimes:
    ...     print(f"{regime.name}: {regime.start.year}-{regime.end.year}")
    GFC Aftermath: 2010-2011
    QE Era: 2012-2015
    Rate Normalization: 2016-2019
    COVID Crash: 2020-2020
    Zero Rates: 2020-2021
    Rate Hiking Cycle: 2022-2023
    Recent: 2024-2024
    """
    regimes = [
        MarketRegime(
            name="GFC Aftermath",
            start=datetime(2010, 1, 1, tzinfo=UTC),
            end=datetime(2011, 12, 31, tzinfo=UTC),
            description="Post-financial crisis recovery with early QE programs",
            key_events=[
                "European debt crisis (Greece, Ireland)",
                "Flash crash May 6, 2010",
                "QE2 announced November 2010",
                "S&P downgrades US credit rating August 2011",
            ],
        ),
        MarketRegime(
            name="QE Era",
            start=datetime(2012, 1, 1, tzinfo=UTC),
            end=datetime(2015, 12, 31, tzinfo=UTC),
            description="Central bank quantitative easing dominance with low rates",
            key_events=[
                "Draghi 'whatever it takes' speech July 2012",
                "Taper tantrum May 2013",
                "Oil price crash 2014-2015 (WTI $100 to $35)",
                "China market turbulence August 2015",
            ],
        ),
        MarketRegime(
            name="Rate Normalization",
            start=datetime(2016, 1, 1, tzinfo=UTC),
            end=datetime(2019, 12, 31, tzinfo=UTC),
            description="Fed rate hiking cycle and monetary policy normalization",
            key_events=[
                "Brexit referendum June 2016",
                "Trump election November 2016",
                "Fed rate hikes begin (0.25% to 2.50%)",
                "Volmageddon February 2018",
                "Trade war tensions 2018-2019",
            ],
        ),
        MarketRegime(
            name="COVID Crash",
            start=datetime(2020, 2, 1, tzinfo=UTC),
            end=datetime(2020, 4, 30, tzinfo=UTC),
            description="Pandemic-induced market shock with unprecedented volatility",
            key_events=[
                "WHO declares pandemic March 11, 2020",
                "Multiple circuit breakers triggered",
                "VIX peaks at 82.69 March 16, 2020",
                "S&P 500 falls 34% in 33 days",
                "Emergency Fed rate cuts to zero",
            ],
        ),
        MarketRegime(
            name="Zero Rates",
            start=datetime(2020, 5, 1, tzinfo=UTC),
            end=datetime(2021, 12, 31, tzinfo=UTC),
            description="Zero interest rate policy with massive fiscal and monetary stimulus",
            key_events=[
                "CARES Act ($2.2T stimulus)",
                "Vaccine approval December 2020",
                "Retail trading boom (GameStop/AMC)",
                "Inflation concerns emerge late 2021",
                "Supply chain disruptions",
            ],
        ),
        MarketRegime(
            name="Rate Hiking Cycle",
            start=datetime(2022, 1, 1, tzinfo=UTC),
            end=datetime(2023, 12, 31, tzinfo=UTC),
            description="Aggressive Fed rate hikes to combat 40-year high inflation",
            key_events=[
                "Fed raises rates 0% to 5.5% (fastest pace since 1980s)",
                "Russia-Ukraine war February 2022",
                "Crypto winter (FTX collapse)",
                "Silicon Valley Bank failure March 2023",
                "Inflation peaks at 9.1% June 2022",
            ],
        ),
        MarketRegime(
            name="Recent",
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 12, 31, tzinfo=UTC),
            description="Higher-for-longer rates with AI boom and market resilience",
            key_events=[
                "AI investment surge (Nvidia/Magnificent 7)",
                "Soft landing narrative strengthens",
                "Fed pivot expectations (rate cuts)",
                "Market concentration concerns",
            ],
        ),
    ]

    LOGGER.info(
        "Defined market regimes",
        num_regimes=len(regimes),
        start_year=regimes[0].start.year,
        end_year=regimes[-1].end.year,
    )

    return regimes


# ===== Core Analysis Functions =====


def analyze_strategy_across_regimes(
    backtest_result: BacktestResult,
    regimes: list[MarketRegime] | None = None,
    risk_free_rate: float | None = None,
) -> RegimeAnalysisResult:
    """
    Analyze strategy performance across market regimes.

    This function computes performance metrics for each regime and identifies
    failure modes. It provides the foundation for understanding how strategy
    performance varies with macroeconomic conditions.

    Parameters
    ----------
    backtest_result : BacktestResult
        Full backtest result (must span all regime periods)
    regimes : list[MarketRegime] | None
        Market regimes to analyze (defaults to standard 7 regimes)
    risk_free_rate : float | None, default None
        Annual risk-free rate for Sharpe calculation. Defaults to
        ``ThreeDRiskBacktestDefaults().risk_free_rate`` when omitted.

    Returns
    -------
    RegimeAnalysisResult
        Performance analysis for each regime with failure identification

    Raises
    ------
    ValueError
        If backtest period doesn't cover all regimes or insufficient data

    Examples
    --------
    >>> result = backtester.run_backtest(dataset, strategy="equal_weight")
    >>> analysis = analyze_strategy_across_regimes(result)
    >>> print(f"Success rate: {analysis.success_rate:.1%}")
    Success rate: 85.7%
    >>> summary = analysis.summary_table()
    >>> print(summary)

    Notes
    -----
    Algorithm:
    1. Filter backtest returns to each regime period
    2. Calculate annualized metrics (return, volatility, Sharpe)
    3. Compute drawdown and win rates
    4. Identify failures (Sharpe <= 0)
    5. Generate failure explanations
    """
    if regimes is None:
        regimes = define_market_regimes()

    LOGGER.info(
        "Analyzing strategy across regimes",
        strategy=backtest_result.strategy_name,
        num_regimes=len(regimes),
        backtest_start=backtest_result.start_date.isoformat(),
        backtest_end=backtest_result.end_date.isoformat(),
    )

    # Validate backtest coverage
    _validate_backtest_coverage(backtest_result, regimes)

    # Calculate performance for each regime
    regime_performances: dict[str, RegimePerformance] = {}

    resolved_risk_free_rate = (
        risk_free_rate if risk_free_rate is not None else BACKTEST_DEFAULTS.risk_free_rate
    )

    for regime in regimes:
        perf = _calculate_regime_performance(
            backtest_result=backtest_result,
            regime=regime,
            risk_free_rate=resolved_risk_free_rate,
        )
        regime_performances[regime.name] = perf

        LOGGER.debug(
            "Regime performance calculated",
            regime=regime.name,
            sharpe=f"{perf.sharpe_ratio:.2f}",
            return_=f"{perf.annualized_return:.2%}",
            status="Success" if perf.is_successful else "Failed",
        )

    result = RegimeAnalysisResult(
        strategy_name=backtest_result.strategy_name,
        regime_performances=regime_performances,
    )

    LOGGER.info(
        "Regime analysis completed",
        strategy=backtest_result.strategy_name,
        success_rate=f"{result.success_rate:.1%}",
        num_failures=len(result.failure_analysis()),
        risk_free_rate=resolved_risk_free_rate,
    )

    return result


def compare_strategies_across_regimes(
    strategy_results: dict[str, BacktestResult],
    regimes: list[MarketRegime] | None = None,
    risk_free_rate: float | None = None,
) -> pl.DataFrame:
    """
    Compare multiple strategies across all market regimes.

    This function generates a comprehensive comparison table showing how
    different strategies perform in each market regime, enabling identification
    of regime-specific strengths and weaknesses.

    Parameters
    ----------
    strategy_results : dict[str, BacktestResult]
        Mapping of strategy_name -> backtest result
    regimes : list[MarketRegime] | None
        Market regimes (defaults to standard 7 regimes)
    risk_free_rate : float | None, default None
        Annual risk-free rate supplied to ``analyze_strategy_across_regimes``.
        Uses ``ThreeDRiskBacktestDefaults().risk_free_rate`` when omitted.

    Returns
    -------
    pl.DataFrame
        Comparison table with columns:
        - regime_name: str
        - strategy_name: str
        - sharpe_ratio: float
        - annualized_return: float
        - annualized_volatility: float
        - max_drawdown: float
        - win_rate: float
        Sorted by regime (chronological) then Sharpe ratio (descending)

    Raises
    ------
    ValueError
        If any strategy result doesn't cover all regimes

    Examples
    --------
    >>> results = {
    ...     "Equal Weight": ew_result,
    ...     "60/40": sixty_forty_result,
    ...     "Risk Parity": rp_result,
    ... }
    >>> comparison = compare_strategies_across_regimes(results)
    >>> print(comparison)
    >>> # Find best strategy per regime
    >>> best = comparison.group_by("regime_name").agg(
    ...     pl.col("strategy_name").filter(
    ...         pl.col("sharpe_ratio") == pl.col("sharpe_ratio").max()
    ...     ).first()
    ... )

    Notes
    -----
    This comparison is useful for:
    - Identifying regime-robust strategies
    - Understanding strategy-regime interactions
    - Portfolio construction (regime rotation)
    - Risk management (regime-dependent hedging)
    """
    if regimes is None:
        regimes = define_market_regimes()

    LOGGER.info(
        "Comparing strategies across regimes",
        num_strategies=len(strategy_results),
        num_regimes=len(regimes),
    )

    rows = []

    for strategy_name, result in strategy_results.items():
        analysis = analyze_strategy_across_regimes(
            result,
            regimes,
            risk_free_rate=risk_free_rate,
        )

        for regime_name, perf in analysis.regime_performances.items():
            row = {
                "regime_name": regime_name,
                "strategy_name": strategy_name,
                "sharpe_ratio": perf.sharpe_ratio,
                "annualized_return": perf.annualized_return,
                "annualized_volatility": perf.annualized_volatility,
                "max_drawdown": perf.max_drawdown,
                "win_rate": perf.win_rate,
            }
            rows.append(row)

    df = pl.DataFrame(rows)

    # Sort by regime chronologically, then by Sharpe descending within regime
    # Create regime order based on input list
    regime_order = {regime.name: i for i, regime in enumerate(regimes)}
    df = df.with_columns(
        pl.col("regime_name").replace(regime_order).alias("_regime_order")
    )
    df = df.sort(["_regime_order", "sharpe_ratio"], descending=[False, True])
    df = df.drop("_regime_order")

    return df


def regime_performance_matrix(
    analyses: dict[str, RegimeAnalysisResult],
    metric: str = "sharpe_ratio",
) -> pl.DataFrame:
    """
    Build a wide-format matrix of regime performance across strategies.

    Parameters
    ----------
    analyses : dict[str, RegimeAnalysisResult]
        Mapping of strategy name -> regime analysis result.
    metric : str, default "sharpe_ratio"
        Metric attribute to pivot (e.g., "sharpe_ratio", "annualized_return").

    Returns
    -------
    pl.DataFrame
        DataFrame indexed by regime with one column per strategy.
    """
    if not analyses:
        return pl.DataFrame(schema={"regime_name": pl.Utf8})

    sample_analysis = next(iter(analyses.values()))
    sample_perf = next(iter(sample_analysis.regime_performances.values()))
    if not hasattr(sample_perf, metric):
        msg = f"RegimePerformance has no attribute '{metric}'"
        raise ValueError(msg)

    rows: list[dict[str, object]] = []
    for strategy_name, analysis in analyses.items():
        for regime_name, perf in analysis.regime_performances.items():
            rows.append({
                "regime_name": regime_name,
                "strategy": strategy_name,
                metric: getattr(perf, metric),
            })

    matrix = pl.DataFrame(rows).pivot(
        index="regime_name",
        on="strategy",
        values=metric,
        aggregate_function="first",
    )
    return matrix.sort("regime_name")


def identify_failure_modes(
    regime_analysis: RegimeAnalysisResult,
    factor_data: pl.DataFrame | None = None,
) -> dict[str, str]:
    """
    Identify and explain why strategy failed in specific regimes.

    This function provides detailed failure analysis, incorporating economic
    context and factor behavior to explain underperformance.

    Parameters
    ----------
    regime_analysis : RegimeAnalysisResult
        Regime performance results
    factor_data : pl.DataFrame | None
        Optional factor returns/levels during regimes for correlation analysis
        Expected columns: timestamp, factor_duration, factor_credit, factor_liquidity

    Returns
    -------
    dict[str, str]
        Mapping of regime_name -> detailed failure explanation

    Examples
    --------
    >>> analysis = analyze_strategy_across_regimes(result)
    >>> failures = identify_failure_modes(analysis)
    >>> print(failures["COVID Crash"])
    Negative returns (-8.2% annualized) indicate strategy lost money during
    this period. High volatility (35.4%) suggests unstable returns and
    elevated risk. Large drawdown (-18.3%) indicates poor downside protection.
    Economic conditions during COVID Crash (Pandemic-induced market shock with
    unprecedented volatility) likely contributed to underperformance.

    Notes
    -----
    Failure modes analyzed:
    - Negative returns (strategy lost money)
    - Excessive volatility (unstable returns)
    - Large drawdowns (poor downside protection)
    - Low win rates (directional errors)
    - Inconsistent monthly returns

    Future enhancements could incorporate:
    - Factor correlation breakdown analysis
    - Regime-specific factor volatility
    - Sector concentration risk
    - Turnover and transaction cost impact
    """
    failures = regime_analysis.failure_analysis()

    # If factor data provided, could enhance with factor-specific analysis
    # (Not implemented in this version, but structure is ready for enhancement)
    if factor_data is not None:
        LOGGER.debug(
            "Factor data provided for failure analysis",
            num_factors=len([col for col in factor_data.columns if col.startswith("factor_")]),
        )
        # Future: Analyze factor correlation breakdowns, volatility spikes, etc.

    return failures


# ===== Report Generation =====


def generate_regime_report(
    analysis: RegimeAnalysisResult,
    output_path: Path,
) -> None:
    """
    Generate comprehensive markdown report for regime analysis.

    Parameters
    ----------
    analysis : RegimeAnalysisResult
        Regime analysis results
    output_path : Path
        Path to output markdown file

    Raises
    ------
    IOError
        If unable to write report file

    Examples
    --------
    >>> analysis = analyze_strategy_across_regimes(result)
    >>> generate_regime_report(
    ...     analysis,
    ...     Path("playground/reports/regime_analysis.md")
    ... )
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = analysis.summary_table()
    failures = analysis.failure_analysis()

    with open(output_path, "w", encoding="utf-8") as f:
        # Header
        f.write(f"# Regime Analysis Report: {analysis.strategy_name}\n\n")
        f.write(f"**Generated:** {datetime.now(UTC).isoformat()}\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"- **Overall Success Rate**: {len(analysis.regime_performances) - len(failures)}/{len(analysis.regime_performances)} ")
        f.write(f"regimes ({analysis.success_rate:.1%})\n")

        # Find best and worst regimes
        best_regime = max(analysis.regime_performances.values(), key=lambda p: p.sharpe_ratio)
        worst_regime = min(analysis.regime_performances.values(), key=lambda p: p.sharpe_ratio)

        f.write(f"- **Best Regime**: {best_regime.regime.name} (Sharpe: {best_regime.sharpe_ratio:.2f})\n")
        f.write(f"- **Worst Regime**: {worst_regime.regime.name} (Sharpe: {worst_regime.sharpe_ratio:.2f})\n")

        if failures:
            f.write(f"- **Failed Regimes**: {', '.join(failures.keys())}\n")
        else:
            f.write("- **Failed Regimes**: None\n")

        f.write("\n")

        # Performance by Regime
        f.write("## Performance by Regime\n\n")

        for regime_name, perf in analysis.regime_performances.items():
            f.write(f"### {regime_name} ({perf.regime.start.year}-{perf.regime.end.year})\n\n")
            f.write(f"**Description**: {perf.regime.description}\n\n")

            # Metrics
            f.write("**Performance Metrics:**\n\n")
            f.write(f"- Sharpe Ratio: {perf.sharpe_ratio:.2f}\n")
            f.write(f"- Annualized Return: {perf.annualized_return:.2%}\n")
            f.write(f"- Annualized Volatility: {perf.annualized_volatility:.2%}\n")
            f.write(f"- Max Drawdown: {perf.max_drawdown:.2%}\n")
            f.write(f"- Calmar Ratio: {perf.calmar_ratio:.2f}\n")
            f.write(f"- Win Rate: {perf.win_rate:.1%}\n")
            f.write(f"- Positive Months: {perf.positive_months_pct:.0%}\n")
            f.write(f"- Observations: {perf.num_observations} days\n")
            f.write(f"- Rebalances: {perf.num_rebalances}\n\n")

            # Status
            status = "✅ **Success**" if perf.is_successful else "❌ **Failed**"
            f.write(f"**Status**: {status}\n\n")

            # Key Events
            if perf.regime.key_events:
                f.write("**Key Events:**\n\n")
                for event in perf.regime.key_events:
                    f.write(f"- {event}\n")
                f.write("\n")

            f.write("---\n\n")

        # Failure Mode Analysis
        if failures:
            f.write("## Failure Mode Analysis\n\n")
            f.write("### Failed Regimes\n\n")

            for regime_name, explanation in failures.items():
                f.write(f"#### {regime_name}\n\n")
                f.write(f"{explanation}\n\n")
        else:
            f.write("## Failure Mode Analysis\n\n")
            f.write("No regime failures detected. Strategy maintained positive ")
            f.write("risk-adjusted returns across all market conditions.\n\n")

        # Cross-Regime Patterns
        f.write("## Cross-Regime Patterns\n\n")

        # Calculate average metrics
        avg_sharpe = np.mean([p.sharpe_ratio for p in analysis.regime_performances.values()])
        avg_return = np.mean([p.annualized_return for p in analysis.regime_performances.values()])
        avg_vol = np.mean([p.annualized_volatility for p in analysis.regime_performances.values()])

        f.write("**Average Performance:**\n\n")
        f.write(f"- Average Sharpe Ratio: {avg_sharpe:.2f}\n")
        f.write(f"- Average Return: {avg_return:.2%}\n")
        f.write(f"- Average Volatility: {avg_vol:.2%}\n\n")

        f.write("**Observations:**\n\n")
        f.write(f"- Strategy succeeded in {analysis.success_rate:.0%} of regimes\n")

        # Identify volatility regimes
        high_vol_regimes = [
            name for name, p in analysis.regime_performances.items()
            if p.annualized_volatility > avg_vol * 1.5
        ]
        if high_vol_regimes:
            f.write(f"- High volatility regimes: {', '.join(high_vol_regimes)}\n")

        # Identify negative return regimes
        negative_regimes = [
            name for name, p in analysis.regime_performances.items()
            if p.annualized_return < 0
        ]
        if negative_regimes:
            f.write(f"- Negative return regimes: {', '.join(negative_regimes)}\n")

        f.write("\n")

        # Recommendations
        f.write("## Recommendations\n\n")
        f.write("Based on regime analysis:\n\n")

        if analysis.success_rate >= 0.8:  # 80%+ success
            f.write("1. ✅ Strategy demonstrates strong regime robustness\n")
            f.write("2. Consider minimal modifications; focus on execution optimization\n")
        elif analysis.success_rate >= 0.6:  # 60-80% success
            f.write("1. ⚠️ Strategy shows moderate regime sensitivity\n")
            f.write("2. Investigate failure modes for defensive overlays\n")
            f.write("3. Consider regime-conditional position sizing\n")
        else:  # <60% success
            f.write("1. ❌ Strategy shows high regime sensitivity\n")
            f.write("2. Fundamental redesign may be needed\n")
            f.write("3. Consider regime-switching framework or separate strategies\n")

        if failures:
            f.write("4. Address specific failure modes identified above\n")
            f.write("5. Implement regime detection for defensive positioning\n")

        f.write("\n")

        # Appendix
        f.write("## Appendix: Summary Table\n\n")
        f.write(_format_polars_to_markdown(summary))
        f.write("\n")

    LOGGER.info("Regime analysis report generated", output_path=str(output_path))


def _format_polars_to_markdown(df: pl.DataFrame) -> str:
    """Format Polars DataFrame as markdown table."""
    if df.is_empty():
        return "*No data*"

    columns = df.columns
    lines = []

    # Header
    header = "| " + " | ".join(columns) + " |"
    lines.append(header)

    # Separator
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines.append(separator)

    # Rows
    for row in df.iter_rows(named=True):
        formatted_values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                if col in {"sharpe_ratio", "calmar_ratio"}:
                    formatted_values.append(f"{value:.2f}")
                elif col in {"annualized_return", "annualized_volatility", "max_drawdown", "win_rate"}:
                    formatted_values.append(f"{value:.1%}")
                else:
                    formatted_values.append(f"{value:.2f}")
            elif isinstance(value, int):
                formatted_values.append(f"{value:,}")
            else:
                formatted_values.append(str(value))

        row_str = "| " + " | ".join(formatted_values) + " |"
        lines.append(row_str)

    return "\n".join(lines)


# ===== Helper Functions =====


def _validate_backtest_coverage(
    backtest_result: BacktestResult,
    regimes: list[MarketRegime],
) -> None:
    """
    Validate that backtest result covers all regime periods.

    Parameters
    ----------
    backtest_result : BacktestResult
        Backtest result to validate
    regimes : list[MarketRegime]
        Regimes that must be covered

    Raises
    ------
    ValueError
        If backtest doesn't cover all regimes
    """
    tolerance = timedelta(days=BACKTEST_DEFAULTS.coverage_tolerance_days)
    for regime in regimes:
        if backtest_result.start_date > regime.start + tolerance:
            msg = (
                f"Backtest starts {backtest_result.start_date} after regime "
                f"{regime.name} starts {regime.start}"
            )
            raise ValueError(msg)

        if backtest_result.end_date < regime.end - tolerance:
            msg = (
                f"Backtest ends {backtest_result.end_date} before regime "
                f"{regime.name} ends {regime.end}"
            )
            raise ValueError(msg)


def _calculate_regime_performance(
    backtest_result: BacktestResult,
    regime: MarketRegime,
    risk_free_rate: float,
) -> RegimePerformance:
    """
    Calculate performance metrics for a single regime.

    Parameters
    ----------
    backtest_result : BacktestResult
        Full backtest result
    regime : MarketRegime
        Regime to analyze
    risk_free_rate : float
        Annual risk-free rate

    Returns
    -------
    RegimePerformance
        Performance metrics for the regime

    Raises
    ------
    ValueError
        If insufficient observations in regime
    """
    # Filter returns to regime period
    regime_data = _filter_to_regime(backtest_result, regime)

    if regime_data["num_observations"] < MIN_OBSERVATIONS_FOR_REGIME:
        if regime_data["num_observations"] < MIN_OBSERVATIONS_FALLBACK:
            msg = (
                f"Insufficient observations in regime {regime.name}: "
                f"need {MIN_OBSERVATIONS_FOR_REGIME}, got {regime_data['num_observations']}"
            )
            raise ValueError(msg)
        LOGGER.warning(
            "Regime analysis proceeding with limited observations",
            regime=regime.name,
            observations=regime_data["num_observations"],
            minimum=MIN_OBSERVATIONS_FOR_REGIME,
        )

    returns_arr = regime_data["returns"]

    # Calculate core metrics
    annualized_return = _calculate_annualized_return(returns_arr, len(returns_arr))
    annualized_volatility = _calculate_annualized_volatility(returns_arr)
    sharpe_ratio = _calculate_sharpe_ratio(returns_arr, risk_free_rate)
    max_drawdown = _calculate_max_drawdown(returns_arr)
    calmar_ratio = _calculate_calmar_ratio(annualized_return, max_drawdown)

    # Calculate win metrics
    positive_days = np.sum(returns_arr > 0)
    win_rate = positive_days / len(returns_arr) if len(returns_arr) > 0 else 0.0

    # Calculate monthly statistics
    positive_months_pct = _calculate_positive_months_pct(
        regime_data["dates"],
        returns_arr,
    )

    return RegimePerformance(
        regime=regime,
        strategy_name=backtest_result.strategy_name,
        sharpe_ratio=sharpe_ratio,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar_ratio,
        positive_months_pct=positive_months_pct,
        win_rate=win_rate,
        num_observations=regime_data["num_observations"],
        num_rebalances=regime_data["num_rebalances"],
    )


def _filter_to_regime(
    backtest_result: BacktestResult,
    regime: MarketRegime,
) -> RegimeData:
    """
    Filter backtest result to regime period.

    Parameters
    ----------
    backtest_result : BacktestResult
        Full backtest result
    regime : MarketRegime
        Regime to filter to

    Returns
    -------
    RegimeData
        Filtered data with keys:
        - returns: NDArray[np.float64] (daily returns)
        - dates: list[datetime]
        - num_observations: int
        - num_rebalances: int (estimated)
    """
    # Find indices within regime
    regime_indices = [
        i for i, date in enumerate(backtest_result.dates)
        if regime.start <= date <= regime.end
    ]

    if not regime_indices:
        msg = f"No data found for regime {regime.name}"
        raise ValueError(msg)

    # Filter returns (returns list is 1 element shorter than dates)
    # Adjust indices for returns array
    returns_indices = [i - 1 for i in regime_indices if i > 0]
    regime_returns: NDArray[np.float64] = np.array(
        [backtest_result.returns[i] for i in returns_indices],
        dtype=np.float64,
    )

    # Filter dates
    regime_dates: list[datetime] = [backtest_result.dates[i] for i in regime_indices]

    # Estimate number of rebalances (rough approximation)
    # Assume monthly rebalancing
    num_months: int = max(1, len(regime_indices) // 21)  # ~21 trading days per month
    num_rebalances: int = num_months

    return RegimeData(
        returns=regime_returns,
        dates=regime_dates,
        num_observations=len(regime_returns),
        num_rebalances=num_rebalances,
    )


def _calculate_annualized_return(returns: NDArray[np.float64], num_days: int) -> float:
    """Calculate annualized return from daily returns."""
    if len(returns) == 0 or num_days <= 0:
        return 0.0

    cumulative = np.prod(1.0 + returns) - 1.0
    n_years = num_days / TRADING_DAYS_PER_YEAR

    if n_years <= 0:
        return 0.0

    annualized = (1.0 + cumulative) ** (1.0 / n_years) - 1.0
    return float(annualized)


def _calculate_annualized_volatility(returns: NDArray[np.float64]) -> float:
    """Calculate annualized volatility from daily returns."""
    if len(returns) < 2:
        return 0.0

    daily_std = float(np.std(returns, ddof=1))
    annualized_vol = daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)

    return float(annualized_vol)


def _calculate_sharpe_ratio(returns: NDArray[np.float64], risk_free_rate: float) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(returns) < 2:
        return 0.0

    rf_daily = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess_returns = returns - rf_daily

    mean_excess = float(np.mean(excess_returns))
    std_excess = float(np.std(excess_returns, ddof=1))

    if std_excess < 1e-10:
        return 0.0

    sharpe = (mean_excess / std_excess) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return float(sharpe)


def _calculate_max_drawdown(returns: NDArray[np.float64]) -> float:
    """Calculate maximum drawdown from daily returns."""
    if len(returns) == 0:
        return 0.0

    cumulative = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(np.min(drawdown))

    return max_drawdown


def _calculate_calmar_ratio(annualized_return: float, max_drawdown: float) -> float:
    """Calculate Calmar ratio (return / abs(drawdown))."""
    if abs(max_drawdown) < 1e-10:
        return 0.0

    calmar = annualized_return / abs(max_drawdown)
    return float(calmar)


def _calculate_positive_months_pct(
    dates: list[datetime],
    returns: NDArray[np.float64],
) -> float:
    """Calculate percentage of months with positive returns."""
    if len(returns) == 0 or len(dates) <= 1:
        return 0.0

    # Group returns by month
    monthly_returns: dict[tuple[int, int], list[float]] = {}

    for date, ret in zip(dates[1:], returns):  # Skip first date (no return)
        month_key = (date.year, date.month)
        if month_key not in monthly_returns:
            monthly_returns[month_key] = []
        monthly_returns[month_key].append(ret)

    if not monthly_returns:
        return 0.0

    # Compute geometric monthly returns
    positive_months = 0
    for daily_rets in monthly_returns.values():
        monthly_return = float(np.prod(np.array(daily_rets) + 1.0) - 1.0)
        if monthly_return > 0:
            positive_months += 1

    return positive_months / len(monthly_returns)


# ===== Public API =====

__all__ = [
    "MarketRegime",
    "RegimeAnalysisResult",
    "RegimePerformance",
    "analyze_strategy_across_regimes",
    "compare_strategies_across_regimes",
    "define_market_regimes",
    "generate_regime_report",
    "identify_failure_modes",
    "regime_performance_matrix",
]
