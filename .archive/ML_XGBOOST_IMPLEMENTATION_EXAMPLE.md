# XGBoost Trainer Implementation Example

## Step-by-Step Migration Example

### 1. Create XGBoost Configuration

```python
# ml/config/xgboost.py
from __future__ import annotations

from typing import Any

from nautilus_trader.common.config import NautilusConfig
from nautilus_trader.common.config import NonNegativeFloat
from nautilus_trader.common.config import NonNegativeInt
from nautilus_trader.common.config import PositiveFloat
from nautilus_trader.common.config import PositiveInt

from ml.config.base import MLTrainingConfig


class XGBoostTrainingConfig(MLTrainingConfig, kw_only=True, frozen=True):
    """
    Configuration for XGBoost model training.

    Extends MLTrainingConfig with XGBoost-specific parameters.
    """

    # Core XGBoost parameters
    n_estimators: PositiveInt = 100
    max_depth: PositiveInt = 6
    learning_rate: PositiveFloat = 0.3
    min_child_weight: NonNegativeFloat = 1.0
    subsample: PositiveFloat = 1.0
    colsample_bytree: PositiveFloat = 1.0
    colsample_bylevel: PositiveFloat = 1.0
    gamma: NonNegativeFloat = 0.0
    reg_alpha: NonNegativeFloat = 0.0
    reg_lambda: NonNegativeFloat = 1.0

    # GPU settings
    tree_method: str = "hist"  # "hist" for CPU, "gpu_hist" for GPU
    gpu_id: NonNegativeInt = 0

    # Training control
    objective: str = "binary:logistic"
    eval_metric: str = "auc"

    # Advanced features
    enable_shap: bool = False
    monotonic_constraints: dict[str, int] | None = None

    # Multi-asset configuration
    multi_asset: bool = False
    sector_map: dict[str, str] | None = None
    cross_sectional_features: bool = True

    # Hyperparameter optimization
    optimize_hyperparams: bool = False
    n_trials: PositiveInt = 100
    optimization_metric: str = "sharpe_ratio"
```

### 2. Implement XGBoost Trainer

```python
# ml/training/xgboost.py
from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ml.config.xgboost import XGBoostTrainingConfig
from ml.features.engineering import compute_technical_features
from ml.training.base import BaseMLTrainer


class XGBoostTrainer(BaseMLTrainer):
    """
    XGBoost trainer for financial time series prediction.

    This trainer supports:
    - Single and multi-asset training
    - GPU acceleration
    - Feature importance tracking
    - Cross-sectional features for portfolio models
    - Monotonic constraints for interpretability
    """

    def __init__(self, config: XGBoostTrainingConfig) -> None:
        """Initialize XGBoost trainer with configuration."""
        super().__init__(config)
        self._xgb_config = config
        self._scaler = None
        self._is_multi_asset = config.multi_asset

        # Lazy imports to make XGBoost optional
        self._xgb = None
        self._shap = None

    def prepare_data(
        self,
        data: pl.DataFrame | dict[str, pl.DataFrame],
        target_col: str = "target",
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Prepare features and target for XGBoost training.

        For single asset: expects pl.DataFrame
        For multi-asset: expects dict[ticker, pl.DataFrame]
        """
        if self._is_multi_asset:
            return self._prepare_multi_asset_data(data, target_col)
        else:
            return self._prepare_single_asset_data(data, target_col)

    def _prepare_single_asset_data(
        self,
        data: pl.DataFrame,
        target_col: str,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Prepare single asset data with technical features."""
        # Compute technical features using Nautilus indicators
        features_df = compute_technical_features(
            data,
            self._feature_config,
        )

        # Create target if not present
        if target_col not in data.columns:
            # Default: predict next bar direction
            target = (
                data["close"].shift(-1) > data["close"]
            ).cast(pl.Int32).to_numpy()[:-1]
            features_df = features_df[:-1]
        else:
            target = data[target_col].to_numpy()

        # Convert to numpy
        feature_names = [col for col in features_df.columns if col != target_col]
        X = features_df.select(feature_names).to_numpy()

        # Scale features if configured
        if self._feature_config.normalize_features:
            from sklearn.preprocessing import StandardScaler
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        metadata = {
            "feature_names": feature_names,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
        }

        return X, target, metadata

    def _prepare_multi_asset_data(
        self,
        data_dict: dict[str, pl.DataFrame],
        target_col: str,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Prepare multi-asset data with cross-sectional features.

        This includes:
        - Individual asset features
        - Cross-sectional ranks
        - Sector-relative features
        """
        all_features = []
        all_targets = []
        metadata_list = []

        # Process each asset
        for ticker, df in data_dict.items():
            # Skip if insufficient data
            if len(df) < self._feature_config.lookback_window:
                continue

            # Compute features for this asset
            features_df = compute_technical_features(df, self._feature_config)

            # Create target
            if target_col not in df.columns:
                target = (
                    df["close"].shift(-5) / df["close"] - 1
                ).to_numpy()
                # Convert to classification (positive/negative returns)
                target = (target > 0.001).astype(int)
            else:
                target = df[target_col].to_numpy()

            # Align lengths
            min_len = min(len(features_df), len(target))
            features_df = features_df[:min_len]
            target = target[:min_len]

            # Add asset metadata
            features_df = features_df.with_columns([
                pl.lit(ticker).alias("ticker"),
                pl.lit(self._xgb_config.sector_map.get(ticker, "unknown")).alias("sector"),
            ])

            all_features.append(features_df)
            all_targets.append(target)

        # Combine all assets
        combined_df = pl.concat(all_features)
        combined_targets = np.concatenate(all_targets)

        # Add cross-sectional features if configured
        if self._xgb_config.cross_sectional_features:
            combined_df = self._add_cross_sectional_features(combined_df)

        # Extract feature columns (exclude metadata)
        feature_cols = [
            col for col in combined_df.columns
            if col not in ["ticker", "sector", "timestamp"]
        ]

        X = combined_df.select(feature_cols).to_numpy()

        # Scale features
        if self._feature_config.normalize_features:
            from sklearn.preprocessing import StandardScaler
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        metadata = {
            "feature_names": feature_cols,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "n_assets": len(data_dict),
            "asset_metadata": combined_df.select(["ticker", "sector"]),
        }

        return X, combined_targets, metadata

    def _add_cross_sectional_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add cross-sectional ranking and sector-relative features."""
        # Features to rank cross-sectionally
        rank_features = ["return_5", "return_20", "rsi", "volume_ratio"]

        # Add timestamp if not present (for grouping)
        if "timestamp" not in df.columns:
            df = df.with_row_count("timestamp")

        # Calculate cross-sectional ranks
        for feature in rank_features:
            if feature in df.columns:
                df = df.with_columns([
                    pl.col(feature).rank().over("timestamp").alias(f"{feature}_rank")
                ])

        # Calculate sector-relative features
        if "sector" in df.columns:
            for feature in ["return_5", "return_20"]:
                if feature in df.columns:
                    # Sector mean
                    df = df.with_columns([
                        pl.col(feature).mean().over(["timestamp", "sector"]).alias(f"{feature}_sector_mean")
                    ])
                    # Relative to sector
                    df = df.with_columns([
                        (pl.col(feature) - pl.col(f"{feature}_sector_mean")).alias(f"{feature}_sector_rel")
                    ])

        return df

    def _train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Train XGBoost model with specified parameters."""
        # Import XGBoost (lazy import)
        if self._xgb is None:
            try:
                import xgboost as xgb
                self._xgb = xgb
            except ImportError:
                raise ImportError(
                    "XGBoost is required for training. Install with: pip install xgboost"
                )

        # Prepare XGBoost parameters
        xgb_params = {
            "n_estimators": self._xgb_config.n_estimators,
            "max_depth": self._xgb_config.max_depth,
            "learning_rate": self._xgb_config.learning_rate,
            "min_child_weight": self._xgb_config.min_child_weight,
            "subsample": self._xgb_config.subsample,
            "colsample_bytree": self._xgb_config.colsample_bytree,
            "gamma": self._xgb_config.gamma,
            "reg_alpha": self._xgb_config.reg_alpha,
            "reg_lambda": self._xgb_config.reg_lambda,
            "objective": self._xgb_config.objective,
            "eval_metric": self._xgb_config.eval_metric,
            "tree_method": self._xgb_config.tree_method,
            "random_state": self._xgb_config.random_seed,
            "n_jobs": -1,
            "verbosity": 0,
        }

        # Add GPU parameters if using GPU
        if self._xgb_config.tree_method == "gpu_hist":
            xgb_params["gpu_id"] = self._xgb_config.gpu_id
            xgb_params["predictor"] = "gpu_predictor"

        # Apply monotonic constraints if specified
        if self._xgb_config.monotonic_constraints:
            constraints = self._create_monotonic_constraints(
                self._feature_names,
                self._xgb_config.monotonic_constraints,
            )
            xgb_params["monotone_constraints"] = constraints

        # Create and train model
        model = self._xgb.XGBClassifier(**xgb_params)

        # Train with early stopping
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=self._xgb_config.early_stopping_rounds,
            verbose=False,
        )

        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(model)

        # Calculate SHAP values if enabled
        shap_results = {}
        if self._xgb_config.enable_shap:
            shap_results = self._calculate_shap_values(model, X_val)

        return {
            "model": model,
            "metrics": {
                "best_iteration": model.best_iteration,
                "best_score": model.best_score,
            },
            "feature_importance": feature_importance,
            "shap_results": shap_results,
        }

    def _create_monotonic_constraints(
        self,
        feature_names: list[str],
        constraints_dict: dict[str, int],
    ) -> str:
        """Create monotonic constraints string for XGBoost."""
        constraints = []
        for feature in feature_names:
            if feature in constraints_dict:
                constraints.append(str(constraints_dict[feature]))
            else:
                constraints.append("0")
        return f"({','.join(constraints)})"

    def _calculate_feature_importance(self, model) -> dict[str, float]:
        """Calculate feature importance scores."""
        importance_dict = {}

        # Get importance from model
        for feature, importance in zip(self._feature_names, model.feature_importances_):
            importance_dict[feature] = float(importance)

        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    def _calculate_shap_values(
        self,
        model,
        X_sample: np.ndarray,
        max_samples: int = 1000,
    ) -> dict[str, Any]:
        """Calculate SHAP values for model explainability."""
        if self._shap is None:
            try:
                import shap
                self._shap = shap
            except ImportError:
                print("SHAP not available. Skipping SHAP analysis.")
                return {}

        # Limit samples for efficiency
        n_samples = min(len(X_sample), max_samples)
        X_shap = X_sample[:n_samples]

        # Create explainer and calculate SHAP values
        explainer = self._shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)

        # For binary classification, use positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Create importance dictionary
        shap_importance = {}
        for feature, importance in zip(self._feature_names, mean_abs_shap):
            shap_importance[feature] = float(importance)

        return {
            "shap_values": shap_values,
            "shap_importance": dict(sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)),
            "expected_value": explainer.expected_value,
        }
```

### 3. Example Usage

```python
# examples/train_xgboost_example.py
import polars as pl
from pathlib import Path

from nautilus_trader.model.identifiers import InstrumentId
from ml.config.xgboost import XGBoostTrainingConfig
from ml.config.base import MLFeatureConfig
from ml.training.xgboost import XGBoostTrainer


def train_single_asset_xgboost():
    """Example: Train XGBoost for single asset prediction."""

    # Load data (example with Polars)
    data = pl.read_parquet("data/AAPL_1h_bars.parquet")

    # Configure features
    feature_config = MLFeatureConfig(
        lookback_window=100,
        indicators={
            "sma": {"periods": [20, 50]},
            "rsi": {"period": 14},
            "bb": {"period": 20, "std": 2.0},
        },
        normalize_features=True,
    )

    # Configure XGBoost training
    config = XGBoostTrainingConfig(
        data_source="data/AAPL_1h_bars.parquet",
        target_column="target",
        feature_config=feature_config,
        # XGBoost parameters
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        # Use GPU if available
        tree_method="gpu_hist",
        # Enable SHAP for explainability
        enable_shap=True,
        # Save model
        save_model_path="models/xgboost_aapl.pkl",
    )

    # Create trainer and train
    trainer = XGBoostTrainer(config)
    results = trainer.train(data)

    print(f"Training completed!")
    print(f"Best iteration: {results['metrics']['best_iteration']}")
    print(f"Validation accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"Sharpe ratio: {results['metrics']['sharpe_ratio']:.4f}")

    # Show top features
    print("\nTop 10 Features:")
    for i, (feature, importance) in enumerate(results['feature_importance'].items()):
        if i >= 10:
            break
        print(f"{i+1}. {feature}: {importance:.4f}")


def train_multi_asset_xgboost():
    """Example: Train XGBoost for multi-asset portfolio prediction."""

    # Load data for multiple assets
    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
    data_dict = {}

    for ticker in tickers:
        data_dict[ticker] = pl.read_parquet(f"data/{ticker}_1h_bars.parquet")

    # Define sector mapping
    sector_map = {
        "AAPL": "Technology",
        "GOOGL": "Technology",
        "MSFT": "Technology",
        "AMZN": "Consumer",
        "META": "Technology",
    }

    # Configure multi-asset XGBoost
    config = XGBoostTrainingConfig(
        data_source="data/multi_asset",
        target_column="returns",  # Predict returns
        feature_config=MLFeatureConfig(
            lookback_window=50,
            normalize_features=True,
        ),
        # Multi-asset settings
        multi_asset=True,
        sector_map=sector_map,
        cross_sectional_features=True,
        # XGBoost parameters
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        # Optimize hyperparameters
        optimize_hyperparams=True,
        n_trials=50,
        optimization_metric="sharpe_ratio",
    )

    # Train model
    trainer = XGBoostTrainer(config)
    results = trainer.train(data_dict)

    print("Multi-asset training completed!")
    print(f"Portfolio Sharpe: {results['metrics']['sharpe_ratio']:.4f}")

    # Show cross-sectional features
    print("\nTop Cross-Sectional Features:")
    for feature, importance in results['feature_importance'].items():
        if "_rank" in feature or "_sector_rel" in feature:
            print(f"{feature}: {importance:.4f}")


if __name__ == "__main__":
    # Train single asset model
    train_single_asset_xgboost()

    # Train multi-asset model
    train_multi_asset_xgboost()
```

### 4. Integration with ML Actor

```python
# ml/actors/xgboost_inference.py
from ml.actors.base import BaseMLActor
from ml.config.base import MLActorConfig
import numpy as np


class XGBoostInferenceActor(BaseMLActor):
    """Actor for real-time XGBoost inference."""

    def _load_model(self) -> None:
        """Load XGBoost model and metadata."""
        import pickle

        with open(self.config.model_path, "rb") as f:
            model_data = pickle.load(f)

        self._model = model_data["model"]
        self._feature_names = model_data["feature_names"]
        self._scaler = model_data.get("scaler")

        # Verify model type
        if not hasattr(self._model, "predict_proba"):
            raise ValueError("Model must have predict_proba method")

    def _compute_features(self, bar) -> np.ndarray:
        """Compute features from bar data."""
        # Update indicators
        for indicator in self._indicators.values():
            indicator.update(bar.close)

        # Build feature vector in correct order
        features = []
        for feature_name in self._feature_names:
            if hasattr(self, f"_get_{feature_name}"):
                value = getattr(self, f"_get_{feature_name}")()
                features.append(value)

        return np.array(features, dtype=np.float32)

    def _predict(self, features: np.ndarray) -> tuple[float, float]:
        """Make prediction with XGBoost model."""
        # Scale features if scaler available
        if self._scaler is not None:
            features = self._scaler.transform(features.reshape(1, -1))

        # Get probability predictions
        proba = self._model.predict_proba(features.reshape(1, -1))[0]

        # Binary classification: return probability of positive class
        prediction = 1 if proba[1] > 0.5 else -1
        confidence = proba[1] if prediction == 1 else proba[0]

        return prediction, confidence
```

### 5. Key Migration Notes

1. **Configuration**: Use msgspec-based configs (frozen=True)
2. **Data Format**: Use Polars throughout training pipeline
3. **Feature Engineering**: Reuse Nautilus indicators for consistency
4. **Hot Path**: Keep inference lightweight (numpy only)
5. **Dependencies**: Make heavy libraries optional (lazy imports)
6. **Testing**: Ensure feature parity between training and inference

This implementation maintains all the advanced features from the OLD trainer while conforming to the new architecture patterns.
