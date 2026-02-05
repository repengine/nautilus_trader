"""
Example demonstrating the Strategy Registry usage.

This example shows how to:
1. Register strategies with manifests
2. Query strategies by various criteria
3. Track performance metrics
4. Manage strategy lifecycle

"""

import tempfile
import time
from pathlib import Path

from ml.config.constants import Versions
from ml.registry.strategy_registry import MarketRegime
from ml.registry.strategy_registry import StrategyManifest
from ml.registry.strategy_registry import StrategyRegistry
from ml.registry.strategy_registry import StrategyType


def main() -> None:
    """
    Demonstrate Strategy Registry functionality.
    """
    # Create a temporary directory for this demo
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = StrategyRegistry(Path(tmpdir))
        print(f"Created strategy registry at: {tmpdir}")

        # =============================================================================
        # 1. Register a Trend Following Strategy
        # =============================================================================

        trend_strategy = StrategyManifest(
            strategy_id="trend_follow_ma_cross",
            strategy_type=StrategyType.TREND_FOLLOWING,
            version=Versions.DEFAULT_MANIFEST_VERSION,
            # Requirements
            required_models=["lgb_directional_v1"],
            required_features=["price_sma_20", "sma_50", "rsi_14", "volume_ratio_20"],
            # Market conditions
            suitable_regimes=[MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN],
            instrument_types=["FX", "EQUITY"],
            timeframe_range=("15m", "1h"),
            # Risk parameters
            max_position_size=0.1,  # 10% of portfolio
            max_leverage=2.0,
            max_drawdown=0.15,
            stop_loss_type="TRAILING",
            # Performance constraints
            min_sharpe_ratio=1.2,
            min_win_rate=0.45,
            max_correlation_with_portfolio=0.7,
            # Dependencies
            parent_strategy_id=None,
            incompatible_strategies=["mean_revert_bollinger"],  # Can't run with mean reversion
            # Configuration
            config_schema={
                "fast_period": "int",
                "slow_period": "int",
                "atr_multiplier": "float",
            },
            default_config={
                "fast_period": 20,
                "slow_period": 50,
                "atr_multiplier": 2.0,
            },
            # Performance (from backtest)
            backtest_metrics={
                "sharpe_ratio": 1.85,
                "win_rate": 0.62,
                "max_drawdown": 0.08,
                "total_return": 0.45,
            },
            live_metrics=None,
            # Metadata
            created_at=time.time(),
            last_modified=time.time(),
            author="ML Team",
            description="MA crossover with ML confirmation signals",
        )

        # Create dummy strategy file
        strategy_file = Path(tmpdir) / "trend_strategy.py"
        strategy_file.write_text("# Trend following strategy implementation")

        # Register the strategy
        strategy_id = registry.register_strategy(strategy_file, trend_strategy)
        print(f"\n✓ Registered trend strategy: {strategy_id}")

        # =============================================================================
        # 2. Register a Mean Reversion Strategy
        # =============================================================================

        mean_revert_strategy = StrategyManifest(
            strategy_id="mean_revert_bollinger",
            strategy_type=StrategyType.MEAN_REVERSION,
            version="2.1.0",
            required_models=["xgb_range_predictor_v2"],
            required_features=["bollinger_upper", "bollinger_lower", "rsi_14", "atr_14"],
            suitable_regimes=[MarketRegime.RANGING],
            instrument_types=["FX", "CRYPTO"],
            timeframe_range=("5m", "30m"),
            max_position_size=0.05,
            max_leverage=1.5,
            max_drawdown=0.10,
            stop_loss_type="FIXED",
            min_sharpe_ratio=1.0,
            min_win_rate=0.55,
            max_correlation_with_portfolio=0.5,
            parent_strategy_id=None,
            incompatible_strategies=["trend_follow_ma_cross"],
            config_schema={
                "bb_period": "int",
                "bb_std": "float",
                "entry_threshold": "float",
            },
            default_config={
                "bb_period": 20,
                "bb_std": 2.0,
                "entry_threshold": 0.95,
            },
            backtest_metrics={
                "sharpe_ratio": 1.45,
                "win_rate": 0.68,
                "max_drawdown": 0.06,
                "total_return": 0.32,
            },
            live_metrics=None,
            created_at=time.time(),
            last_modified=time.time(),
            author="Quant Team",
            description="Bollinger band mean reversion with ML filters",
        )

        strategy_file2 = Path(tmpdir) / "mean_revert.py"
        strategy_file2.write_text("# Mean reversion strategy")
        registry.register_strategy(strategy_file2, mean_revert_strategy)
        print("✓ Registered mean reversion strategy")

        # =============================================================================
        # 3. Register an Improved Version (Child Strategy)
        # =============================================================================

        improved_trend = StrategyManifest(
            strategy_id="trend_follow_ma_cross_v2",
            strategy_type=StrategyType.TREND_FOLLOWING,
            version="2.0.0",
            required_models=["lgb_directional_v2", "xgb_momentum_v1"],  # Uses ensemble
            required_features=["price_sma_20", "sma_50", "ema_12", "rsi_14", "volume_ratio_20", "atr_14"],
            suitable_regimes=[
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.VOLATILE,
            ],
            instrument_types=["FX", "EQUITY", "CRYPTO"],
            timeframe_range=("5m", "4h"),
            max_position_size=0.15,
            max_leverage=3.0,
            max_drawdown=0.12,
            stop_loss_type="ADAPTIVE",
            min_sharpe_ratio=1.5,
            min_win_rate=0.50,
            max_correlation_with_portfolio=0.6,
            parent_strategy_id="trend_follow_ma_cross",  # Evolution of v1
            incompatible_strategies=["mean_revert_bollinger"],
            config_schema={
                "fast_period": "int",
                "slow_period": "int",
                "atr_multiplier": "float",
                "regime_filter": "bool",
            },
            default_config={
                "fast_period": 20,
                "slow_period": 50,
                "atr_multiplier": 1.5,
                "regime_filter": True,
            },
            backtest_metrics={
                "sharpe_ratio": 2.15,
                "win_rate": 0.58,
                "max_drawdown": 0.07,
                "total_return": 0.62,
            },
            live_metrics=None,
            created_at=time.time(),
            last_modified=time.time(),
            author="ML Team",
            description="Enhanced MA crossover with regime detection and ensemble ML",
        )

        strategy_file3 = Path(tmpdir) / "trend_v2.py"
        strategy_file3.write_text("# Enhanced trend strategy")
        registry.register_strategy(strategy_file3, improved_trend)
        print("✓ Registered improved trend strategy (v2)")

        # =============================================================================
        # 4. Query Strategies by Market Regime
        # =============================================================================

        print("\n" + "=" * 60)
        print("QUERYING STRATEGIES BY MARKET REGIME")
        print("=" * 60)

        # Find strategies for trending markets
        trending_strategies = registry.get_strategies_for_regime(MarketRegime.TRENDING_UP)
        print("\nStrategies for TRENDING_UP markets:")
        for strategy_info in trending_strategies:
            print(
                f"  - {strategy_info.manifest.strategy_id} (Sharpe: {strategy_info.manifest.backtest_metrics.get('sharpe_ratio', 0):.2f})",
            )

        # Find strategies for ranging markets
        ranging_strategies = registry.get_strategies_for_regime(MarketRegime.RANGING)
        print("\nStrategies for RANGING markets:")
        for strategy_info in ranging_strategies:
            print(f"  - {strategy_info.manifest.strategy_id}")

        # =============================================================================
        # 5. Query by Instrument Type
        # =============================================================================

        print("\n" + "=" * 60)
        print("QUERYING STRATEGIES BY INSTRUMENT TYPE")
        print("=" * 60)

        fx_strategies = registry.get_strategies_for_instrument_type("FX")
        print(f"\nFX trading strategies ({len(fx_strategies)} found):")
        for strategy_info in fx_strategies:
            print(f"  - {strategy_info.manifest.strategy_id}: {strategy_info.manifest.description}")

        # =============================================================================
        # 6. Rank Strategies by Performance
        # =============================================================================

        print("\n" + "=" * 60)
        print("RANKING STRATEGIES BY PERFORMANCE")
        print("=" * 60)

        # Rank by Sharpe ratio
        ranked_by_sharpe = registry.get_strategies_ranked_by_performance(
            "sharpe_ratio",
            use_live_metrics=False,  # Use backtest metrics
        )

        print("\nStrategies ranked by Sharpe Ratio:")
        for i, strategy_info in enumerate(ranked_by_sharpe, 1):
            sharpe = strategy_info.manifest.backtest_metrics.get("sharpe_ratio", 0)
            print(f"  {i}. {strategy_info.manifest.strategy_id}: {sharpe:.2f}")

        # Rank by total return
        ranked_by_return = registry.get_strategies_ranked_by_performance(
            "total_return",
            use_live_metrics=False,
        )

        print("\nStrategies ranked by Total Return:")
        for i, strategy_info in enumerate(ranked_by_return, 1):
            total_return = strategy_info.manifest.backtest_metrics.get("total_return", 0)
            print(f"  {i}. {strategy_info.manifest.strategy_id}: {total_return:.1%}")

        # =============================================================================
        # 7. Check Strategy Requirements
        # =============================================================================

        print("\n" + "=" * 60)
        print("VALIDATING STRATEGY REQUIREMENTS")
        print("=" * 60)

        # Simulate available models and features
        available_models = ["lgb_directional_v1", "xgb_range_predictor_v2"]
        available_features = [
            "price_sma_20",
            "sma_50",
            "rsi_14",
            "volume_ratio_20",
            "bollinger_upper",
            "bollinger_lower",
            "atr_14",
        ]

        for strategy_id in [
            "trend_follow_ma_cross",
            "mean_revert_bollinger",
            "trend_follow_ma_cross_v2",
        ]:
            can_run = registry.validate_requirements(
                strategy_id,
                available_models,
                available_features,
            )
            status = "✓ CAN RUN" if can_run else "✗ MISSING REQUIREMENTS"
            print(f"{strategy_id}: {status}")

        # =============================================================================
        # 8. Check Strategy Compatibility
        # =============================================================================

        print("\n" + "=" * 60)
        print("CHECKING STRATEGY COMPATIBILITY")
        print("=" * 60)

        # Check if trend and mean reversion can run together
        can_run_together = registry.check_compatibility(
            "trend_follow_ma_cross",
            ["mean_revert_bollinger"],
        )
        print(f"Can run trend_follow_ma_cross with mean_revert_bollinger: {can_run_together}")

        # Check if v1 and v2 can run together
        can_run_versions = registry.check_compatibility(
            "trend_follow_ma_cross_v2",
            ["trend_follow_ma_cross"],
        )
        print(f"Can run trend v1 and v2 together: {can_run_versions}")

        # =============================================================================
        # 9. Get Strategy Lineage
        # =============================================================================

        print("\n" + "=" * 60)
        print("STRATEGY LINEAGE (Evolution)")
        print("=" * 60)

        lineage = registry.get_strategy_lineage("trend_follow_ma_cross_v2")
        print("\nLineage for trend_follow_ma_cross_v2:")
        for i, strategy_info in enumerate(lineage):
            indent = "  " * i
            print(
                f"{indent}└─ {strategy_info.manifest.strategy_id} (v{strategy_info.manifest.version})",
            )

        # =============================================================================
        # 10. Update Live Performance Metrics
        # =============================================================================

        print("\n" + "=" * 60)
        print("UPDATING LIVE PERFORMANCE")
        print("=" * 60)

        # Simulate live trading results
        live_metrics = {
            "sharpe_ratio": 1.92,
            "win_rate": 0.64,
            "total_trades": 247,
            "total_return": 0.38,
            "max_drawdown": 0.09,
        }

        registry.update_live_metrics("trend_follow_ma_cross", live_metrics)
        print("Updated live metrics for trend_follow_ma_cross")

        # Re-rank using live metrics
        ranked_live = registry.get_strategies_ranked_by_performance(
            "sharpe_ratio",
            use_live_metrics=True,
        )

        print("\nStrategies ranked by LIVE Sharpe Ratio:")
        for i, strategy_info in enumerate(ranked_live, 1):
            metrics = strategy_info.manifest.live_metrics or strategy_info.manifest.backtest_metrics
            sharpe = metrics.get("sharpe_ratio", 0)
            source = "LIVE" if strategy_info.manifest.live_metrics else "BACKTEST"
            print(f"  {i}. {strategy_info.manifest.strategy_id}: {sharpe:.2f} ({source})")


if __name__ == "__main__":
    main()
