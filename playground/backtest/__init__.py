"""Backtesting infrastructure for factor-based portfolio strategies."""

from __future__ import annotations

from playground.backtest.benchmarks import MinimumVarianceStrategy
from playground.backtest.benchmarks import RiskParityStrategy
from playground.backtest.benchmarks import SixtyFortyStrategy
from playground.backtest.engine import BacktestConfig
from playground.backtest.engine import BacktestResult
from playground.backtest.engine import FactorBacktester
from playground.backtest.regime_analysis import MarketRegime
from playground.backtest.regime_analysis import RegimeAnalysisResult
from playground.backtest.regime_analysis import RegimePerformance
from playground.backtest.regime_analysis import analyze_strategy_across_regimes
from playground.backtest.regime_analysis import compare_strategies_across_regimes
from playground.backtest.regime_analysis import define_market_regimes
from playground.backtest.regime_analysis import generate_regime_report
from playground.backtest.regime_analysis import identify_failure_modes
from playground.backtest.strategies import EqualWeightStrategy
from playground.backtest.strategies import FactorTiltStrategy


__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "EqualWeightStrategy",
    "FactorBacktester",
    "FactorTiltStrategy",
    "MarketRegime",
    "MinimumVarianceStrategy",
    "RegimeAnalysisResult",
    "RegimePerformance",
    "RiskParityStrategy",
    "SixtyFortyStrategy",
    "analyze_strategy_across_regimes",
    "compare_strategies_across_regimes",
    "define_market_regimes",
    "generate_regime_report",
    "identify_failure_modes",
]
