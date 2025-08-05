#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Example: XGBoost model training for financial time series prediction.

This example demonstrates how to use the XGBoostTrainer for both single-asset
and multi-asset training scenarios, showcasing key features like feature
engineering, model evaluation, and feature importance analysis.

Requirements:
- polars: pip install polars
- xgboost: pip install xgboost
- scikit-learn: pip install scikit-learn

"""

from pathlib import Path

import numpy as np


try:
    import polars as pl
except ImportError:
    print("Warning: polars not installed. Please install with: pip install polars")
    pl = None

from ml.config.base import MLFeatureConfig
from ml.config.xgboost import XGBoostTrainingConfig
from ml.training.xgboost import XGBoostTrainer


def create_synthetic_data(n_samples: int = 500, ticker: str = "SYNTHETIC") -> pl.DataFrame:
    """
    Create synthetic OHLCV data for demonstration.

    This generates realistic-looking financial data with trends, volatility,
    and various market conditions for testing the ML trainer.

    Parameters
    ----------
    n_samples : int, default 500
        Number of data points to generate.
    ticker : str, default "SYNTHETIC"
        Ticker symbol for the synthetic data.

    Returns
    -------
    pl.DataFrame
        DataFrame with OHLCV data.

    """
    try:
        import polars as pl
    except ImportError:
        raise ImportError("Polars is required. Install with: pip install polars")

    rng = np.random.default_rng(42)

    # Generate price series with trend and volatility
    returns = rng.normal(0.0005, 0.02, n_samples)  # Daily returns
    prices = 100 * np.exp(np.cumsum(returns))

    # Add intraday high/low spread
    spreads = rng.uniform(0.5, 3.0, n_samples)
    highs = prices * (1 + spreads / 200)
    lows = prices * (1 - spreads / 200)

    # Volume with some correlation to price changes
    base_volume = 10000
    volume_multiplier = 1 + np.abs(returns) * 10
    volumes = base_volume * volume_multiplier * rng.uniform(0.5, 2.0, n_samples)

    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2022, 1, 1),
                end=pl.datetime(2023, 12, 31),
                interval="1d",
            )[:n_samples],
            "open": prices,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
            "ticker": [ticker] * n_samples,
        },
    )


def example_single_asset_training():
    """
    Train XGBoost model for single asset prediction.

    This demonstrates the basic workflow for training an XGBoost model on a single
    financial instrument with automatic feature engineering.

    """
    print("=" * 60)
    print("SINGLE ASSET XGBOOST TRAINING EXAMPLE")
    print("=" * 60)

    # Create synthetic data
    print("Creating synthetic market data...")
    data = create_synthetic_data(n_samples=300, ticker="DEMO")
    print(f"Generated {len(data)} bars of data")

    # Configure feature engineering
    feature_config = MLFeatureConfig(
        lookback_window=50,
        return_periods=[1, 5, 10, 20],
        indicators={
            "sma": {"periods": [20, 50]},
            "rsi": {"period": 14},
            "bb": {"period": 20, "std": 2.0},
        },
        normalize_features=True,
        fill_missing_with=0.0,
    )

    # Configure XGBoost training
    config = XGBoostTrainingConfig(
        data_source="synthetic_data",
        target_column="target",  # Will be auto-created
        feature_config=feature_config,
        # XGBoost parameters optimized for financial data
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        min_child_weight=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        # Training settings
        early_stopping_rounds=20,
        train_test_split=0.8,
        random_seed=42,
        # Use CPU-optimized tree method
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="auc",
        # Save trained model
        save_model_path="models/xgboost_demo_single.pkl",
    )

    print("Training configuration:")
    print(f"- Features: {len(feature_config.return_periods)} return periods")
    print(f"- XGBoost: {config.n_estimators} estimators, depth {config.max_depth}")
    print(f"- Early stopping: {config.early_stopping_rounds} rounds")

    # Create trainer and train model
    print("\nInitializing XGBoost trainer...")
    trainer = XGBoostTrainer(config)

    print("Starting model training...")
    results = trainer.train(data)

    # Display results
    print("\nTraining Results:")
    print(f"- Training samples: {results['metrics']['training_samples']}")
    print(f"- Validation samples: {results['metrics']['validation_samples']}")
    print(f"- Features: {results['metrics']['feature_count']}")
    print(f"- Best iteration: {results['metrics']['best_iteration']}")
    print(f"- Training time: {results['metrics']['training_time']:.2f}s")

    # Show validation metrics
    metrics = results["metrics"]
    if "accuracy" in metrics:
        print(f"- Validation accuracy: {metrics['accuracy']:.4f}")
    if "sharpe_ratio" in metrics:
        print(f"- Strategy Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
    if "max_drawdown" in metrics:
        print(f"- Max drawdown: {metrics['max_drawdown']:.4f}")

    # Show top features
    print("\nTop 10 Most Important Features:")
    importance_summary = trainer.get_feature_importance_summary()
    if "top_10_features" in importance_summary:
        for i, (feature, importance) in enumerate(importance_summary["top_10_features"], 1):
            print(f"{i:2d}. {feature:<20} {importance:.4f}")

    print(f"\nModel saved to: {config.save_model_path}")
    return results


def example_multi_asset_training():
    """
    Train XGBoost model for multi-asset portfolio prediction.

    This demonstrates advanced multi-asset training with cross-sectional features and
    sector-relative calculations.

    """
    print("\n" + "=" * 60)
    print("MULTI-ASSET XGBOOST TRAINING EXAMPLE")
    print("=" * 60)

    # Create multi-asset dataset
    tickers = ["TECH_A", "TECH_B", "FINANCE_A", "FINANCE_B", "CONSUMER_A"]
    sector_map = {
        "TECH_A": "Technology",
        "TECH_B": "Technology",
        "FINANCE_A": "Finance",
        "FINANCE_B": "Finance",
        "CONSUMER_A": "Consumer",
    }

    print(f"Creating synthetic data for {len(tickers)} assets...")
    data_dict = {}

    for ticker in tickers:
        # Generate synthetic data for each ticker
        # Using deterministic approach for consistent results
        data_dict[ticker] = create_synthetic_data(250, ticker)

    print(f"Generated data for {len(data_dict)} assets")

    # Configure multi-asset training
    config = XGBoostTrainingConfig(
        data_source="multi_asset_synthetic",
        target_column="returns",  # Will predict forward returns
        # Multi-asset settings
        multi_asset=True,
        sector_map=sector_map,
        cross_sectional_features=True,
        # Feature configuration
        feature_config=MLFeatureConfig(
            lookback_window=30,
            return_periods=[1, 5, 10],
            normalize_features=True,
        ),
        # XGBoost parameters for multi-asset
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,  # Lower learning rate for stability
        subsample=0.7,
        colsample_bytree=0.7,
        # Training settings
        early_stopping_rounds=25,
        objective="binary:logistic",
        eval_metric="auc",
        random_seed=42,
        save_model_path="models/xgboost_demo_multi.pkl",
    )

    print("\nMulti-asset configuration:")
    print(f"- Assets: {len(tickers)}")
    print(f"- Sectors: {len(set(sector_map.values()))}")
    print(f"- Cross-sectional features: {config.cross_sectional_features}")

    # Train model
    trainer = XGBoostTrainer(config)

    print("\nStarting multi-asset training...")
    results = trainer.train(data_dict)

    # Display results
    print("\nMulti-Asset Training Results:")
    print(
        f"- Total samples: {results['metrics']['training_samples'] + results['metrics']['validation_samples']}",
    )
    print(f"- Features: {results['metrics']['feature_count']}")
    print(f"- Training time: {results['metrics']['training_time']:.2f}s")

    if "accuracy" in results["metrics"]:
        print(f"- Portfolio accuracy: {results['metrics']['accuracy']:.4f}")

    # Show cross-sectional features in importance
    print("\nTop Cross-Sectional Features:")
    importance_summary = trainer.get_feature_importance_summary()
    if "xgb_importance" in importance_summary:
        cross_sectional_features = [
            (feature, importance)
            for feature, importance in importance_summary["xgb_importance"].items()
            if "_rank" in feature or "_sector_rel" in feature
        ][:5]

        for i, (feature, importance) in enumerate(cross_sectional_features, 1):
            print(f"{i}. {feature:<25} {importance:.4f}")

    return results


def example_feature_importance_analysis():
    """
    Demonstrate feature importance analysis capabilities.

    This shows how to analyze which features are most important for the model's
    predictions and understand model behavior.

    """
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS EXAMPLE")
    print("=" * 60)

    # Create data and train a model for analysis
    data = create_synthetic_data(n_samples=200)

    # Configuration with SHAP analysis enabled
    config = XGBoostTrainingConfig(
        data_source="importance_analysis",
        n_estimators=50,  # Smaller for faster demo
        max_depth=3,
        enable_shap=False,  # Set to True if SHAP is installed
        feature_config=MLFeatureConfig(
            lookback_window=30,
            return_periods=[1, 5, 10, 20],
            normalize_features=True,
        ),
    )

    trainer = XGBoostTrainer(config)
    _ = trainer.train(data)

    # Get comprehensive importance analysis
    importance_summary = trainer.get_feature_importance_summary()

    print("Feature Importance Analysis:")
    print("-" * 40)

    if "xgb_importance" in importance_summary:
        print("XGBoost Native Importance (Top 10):")
        for i, (feature, importance) in enumerate(importance_summary["top_10_features"], 1):
            bar_length = int(importance * 50 / max(importance_summary["xgb_importance"].values()))
            bar = "█" * bar_length
            print(f"{i:2d}. {feature:<20} {importance:.4f} {bar}")

    # Analyze feature categories
    print("\nFeature Category Analysis:")
    categories = {
        "returns": [f for f in importance_summary.get("xgb_importance", {}) if "return_" in f],
        "technical": [
            f
            for f in importance_summary.get("xgb_importance", {})
            if f in ["rsi", "bb_width", "atr_normalized"]
        ],
        "volume": [f for f in importance_summary.get("xgb_importance", {}) if "volume" in f],
        "momentum": [
            f for f in importance_summary.get("xgb_importance", {}) if "momentum" in f or "ema" in f
        ],
    }

    for category, features in categories.items():
        if features:
            avg_importance = np.mean([importance_summary["xgb_importance"][f] for f in features])
            print(f"- {category.capitalize():<12} features: {avg_importance:.4f} (avg)")

    return importance_summary


def main():
    """
    Run all XGBoost training examples.

    This function demonstrates the complete workflow from data preparation to model
    training and analysis for both single and multi-asset scenarios.

    """
    print("XGBoost Trainer Examples for Nautilus Trader ML")
    print("===============================================")

    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    try:
        # Example 1: Single asset training
        _ = example_single_asset_training()

        # Example 2: Multi-asset training
        _ = example_multi_asset_training()

        # Example 3: Feature importance analysis
        _ = example_feature_importance_analysis()

        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("Generated models:")
        print("- models/xgboost_demo_single.pkl (single asset)")
        print("- models/xgboost_demo_multi.pkl (multi-asset)")
        print("\nNext steps:")
        print("1. Use these models with ML inference actors")
        print("2. Integrate with trading strategies")
        print("3. Backtest with historical data")
        print("4. Deploy for live trading")

    except ImportError as e:
        print(f"\nDependency Error: {e}")
        print("Please install required dependencies:")
        print("pip install polars xgboost scikit-learn")

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
