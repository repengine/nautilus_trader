# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
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
XGBoost trainer for ML model training.

This module provides XGBoost-specific implementation of the BaseMLTrainer, supporting
both single-asset and multi-asset training with advanced features like SHAP analysis,
feature importance tracking, and hyperparameter optimization.

"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from ml.config.xgboost import XGBoostTrainingConfig
from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.training.base import BaseMLTrainer


try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    pl = None

try:
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    StandardScaler = None


# Explicitly export for mypy
__all__ = [
    "HAS_POLARS",
    "HAS_SKLEARN",
    "StandardScaler",
    "XGBoostTrainer",
]


class XGBoostTrainer(BaseMLTrainer):
    """
    XGBoost trainer for financial time series prediction.

    This trainer extends BaseMLTrainer with XGBoost-specific functionality,
    supporting both single-asset and multi-asset training scenarios with
    advanced ML features.

    Features:
    - Single and multi-asset training
    - GPU acceleration support
    - Feature importance analysis
    - SHAP value computation (optional)
    - Cross-sectional features for portfolio models
    - Monotonic constraints for interpretability
    - Hyperparameter optimization (optional)

    Parameters
    ----------
    config : XGBoostTrainingConfig
        Configuration for XGBoost training.

    """

    def __init__(self, config: XGBoostTrainingConfig) -> None:
        """
        Initialize XGBoost trainer.

        Parameters
        ----------
        config : XGBoostTrainingConfig
            Configuration for XGBoost training.

        """
        super().__init__(config)
        self._xgb_config = config
        # Convert MLFeatureConfig to FeatureConfig
        if self._feature_config is not None:
            feature_config = FeatureConfig(
                lookback_window=self._feature_config.lookback_window,
                normalize_features=self._feature_config.normalize_features,
                fill_missing_with=self._feature_config.fill_missing_with,
            )
        else:
            feature_config = None
        self._feature_engineer = FeatureEngineer(feature_config)
        self._scaler: Any = None
        self._is_multi_asset = config.multi_asset

        # Lazy imports for optional dependencies
        self._xgb = None
        self._shap = None
        self._optuna = None

    def prepare_data(
        self,
        data: Any,  # pl.DataFrame when polars is available
        target_col: str = "target",
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Prepare features and target for XGBoost training.

        This method handles both single-asset and multi-asset scenarios,
        using the FeatureEngineer for consistent feature computation.

        Parameters
        ----------
        data : Any
            Training data. For single-asset: pl.DataFrame.
            For multi-asset: dict[str, pl.DataFrame].
        target_col : str, default "target"
            Name of the target column.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, dict[str, Any]]
            Tuple containing:
            - X: Feature matrix
            - y: Target array
            - metadata: Dictionary with feature names and other metadata

        """
        if not HAS_POLARS:
            raise ImportError(
                "Polars is required for training. Install with: pip install polars",
            )

        if self._is_multi_asset:
            return self._prepare_multi_asset_data(data, target_col)
        else:
            return self._prepare_single_asset_data(data, target_col)

    def _prepare_single_asset_data(
        self,
        data: Any,  # pl.DataFrame
        target_col: str,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Prepare single asset data with technical features.

        Parameters
        ----------
        data : pl.DataFrame
            Input DataFrame with OHLCV data.
        target_col : str
            Target column name.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, dict[str, Any]]
            Features, targets, and metadata.

        """
        print(f"Preparing single asset data with {len(data)} samples")

        # Calculate features using FeatureEngineer
        features_df, scaler = self._feature_engineer.calculate_features_batch(
            data,
            fit_scaler=self._feature_config.normalize_features,
        )

        self._scaler = scaler

        # Create target if not present
        if target_col not in data.columns:
            print(f"Target column '{target_col}' not found, creating default target")
            # Default: predict next bar direction (returns > 0)
            target_series = (data["close"].shift(-1) > data["close"]).cast(pl.Int32)
            # Remove last row (NaN from shift) and convert to numpy
            target_slice = target_series[:-1]
            if hasattr(target_slice, "to_numpy"):
                target = target_slice.to_numpy()
            else:
                # Already a numpy array (can happen in tests)
                target = target_slice
            features_df = features_df[:-1]
        else:
            target = data[target_col].to_numpy()

        # Convert features to numpy
        feature_names = self._feature_engineer.get_feature_names()
        X = features_df.select(feature_names).to_numpy()

        # Handle any remaining NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

        # Ensure X and y have the same number of samples
        min_len = min(len(X), len(target))
        X = X[:min_len]
        target = target[:min_len]

        # Ensure target is binary for classification
        if self._xgb_config.objective == "binary:logistic":
            target = (target > 0).astype(int)

        metadata = {
            "feature_names": feature_names,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "target_type": (
                "classification"
                if self._xgb_config.objective == "binary:logistic"
                else "regression"
            ),
            "scaler": self._scaler,
        }

        print(f"Features shape: {X.shape}, Target shape: {target.shape}")

        # Only print distribution for classification, stats for regression
        if self._xgb_config.objective == "binary:logistic":
            print(f"Target distribution: {np.bincount(target.astype(int))}")
        else:
            print(f"Target stats: mean={target.mean():.4f}, std={target.std():.4f}")

        return X, target, metadata

    def _prepare_multi_asset_data(
        self,
        data_dict: dict[str, Any],  # dict[str, pl.DataFrame]
        target_col: str,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Prepare multi-asset data with cross-sectional features.

        Parameters
        ----------
        data_dict : dict[str, pl.DataFrame]
            Dictionary mapping asset symbols to their DataFrames.
        target_col : str
            Target column name.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, dict[str, Any]]
            Combined features, targets, and metadata.

        """
        print(f"Preparing multi-asset data for {len(data_dict)} assets")

        all_features = []
        all_targets = []
        asset_metadata = []

        # Process each asset
        for ticker, df in data_dict.items():
            if len(df) < self._feature_config.lookback_window:
                print(
                    f"Skipping {ticker}: insufficient data "
                    f"({len(df)} < {self._feature_config.lookback_window})",
                )
                continue

            print(f"Processing {ticker}: {len(df)} samples")

            # Calculate features for this asset
            features_df, _ = self._feature_engineer.calculate_features_batch(df)

            # Create target
            if target_col not in df.columns:
                # Default: predict 5-bar forward returns
                returns = df["close"].shift(-5) / df["close"] - 1
                target = (returns > 0.001).cast(pl.Int32).to_numpy()
            else:
                target = df[target_col].to_numpy()

            # Align lengths
            min_len = min(len(features_df), len(target))
            features_df = features_df[:min_len]
            target = target[:min_len]

            # Add asset metadata columns
            sector = "unknown"
            if self._xgb_config.sector_map is not None:
                sector = self._xgb_config.sector_map.get(ticker, "unknown")

            features_df = features_df.with_columns(
                [
                    pl.lit(ticker).alias("ticker"),
                    pl.lit(sector).alias("sector"),
                ],
            )

            all_features.append(features_df)
            all_targets.append(target)
            asset_metadata.append(
                {
                    "ticker": ticker,
                    "sector": sector,
                    "n_samples": len(features_df),
                },
            )

        if not all_features:
            raise ValueError("No assets had sufficient data for training")

        # Combine all assets
        combined_df = pl.concat(all_features)
        combined_targets = np.concatenate(all_targets)

        # Add cross-sectional features if configured
        if self._xgb_config.cross_sectional_features:
            combined_df = self._add_cross_sectional_features(combined_df)

        # Extract feature columns (exclude metadata)
        feature_names = [
            col for col in combined_df.columns if col not in ["ticker", "sector", "timestamp"]
        ]

        X = combined_df.select(feature_names).to_numpy()

        # Scale features if configured
        if self._feature_config.normalize_features:
            if not HAS_SKLEARN:
                raise ImportError("sklearn is required for feature scaling")
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        combined_targets = np.nan_to_num(combined_targets, nan=0.0, posinf=0.0, neginf=0.0)

        # Ensure target is binary for classification
        if self._xgb_config.objective == "binary:logistic":
            combined_targets = (combined_targets > 0).astype(int)

        metadata = {
            "feature_names": feature_names,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "n_assets": len(data_dict),
            "asset_metadata": asset_metadata,
            "target_type": (
                "classification"
                if self._xgb_config.objective == "binary:logistic"
                else "regression"
            ),
            "scaler": self._scaler,
        }

        print(f"Combined features shape: {X.shape}, Target shape: {combined_targets.shape}")

        return X, combined_targets, metadata

    def _add_cross_sectional_features(self, df: Any) -> Any:  # pl.DataFrame -> pl.DataFrame
        """
        Add cross-sectional ranking and sector-relative features.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame with features for all assets.

        Returns
        -------
        pl.DataFrame
            DataFrame with additional cross-sectional features.

        """
        # Features to rank cross-sectionally
        rank_features = ["return_5", "return_20", "rsi", "volume_ratio_5"]

        # Add timestamp for grouping if not present
        if "timestamp" not in df.columns:
            df = df.with_row_count("timestamp")

        # Calculate cross-sectional ranks
        for feature in rank_features:
            if feature in df.columns:
                df = df.with_columns(
                    [
                        pl.col(feature).rank().over("timestamp").alias(f"{feature}_rank"),
                    ],
                )

        # Calculate sector-relative features
        if "sector" in df.columns:
            for feature in ["return_5", "return_20"]:
                if feature in df.columns:
                    # Sector mean
                    df = df.with_columns(
                        [
                            pl.col(feature)
                            .mean()
                            .over(["timestamp", "sector"])
                            .alias(f"{feature}_sector_mean"),
                        ],
                    )
                    # Relative to sector
                    df = df.with_columns(
                        [
                            (pl.col(feature) - pl.col(f"{feature}_sector_mean")).alias(
                                f"{feature}_sector_rel",
                            ),
                        ],
                    )

        return df

    def _train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Train XGBoost model with specified parameters.

        Parameters
        ----------
        X_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training targets.
        X_val : np.ndarray
            Validation features.
        y_val : np.ndarray
            Validation targets.
        **kwargs : Any
            Additional training parameters.

        Returns
        -------
        dict[str, Any]
            Dictionary containing trained model and metrics.

        """
        # Import XGBoost (lazy import)
        if self._xgb is None:
            try:
                import xgboost as xgb

                self._xgb = xgb
            except ImportError:
                raise ImportError(
                    "XGBoost is required for training. Install with: pip install xgboost",
                )

        print("Training XGBoost model...")

        # Get XGBoost parameters
        xgb_params = self._xgb_config.get_xgb_params()

        # Apply monotonic constraints if specified
        if self._xgb_config.monotonic_constraints:
            constraints = self._create_monotonic_constraints(
                self._feature_names,
                self._xgb_config.monotonic_constraints,
            )
            xgb_params["monotone_constraints"] = constraints

        # Create and train model
        if self._xgb_config.objective == "binary:logistic":
            model = self._xgb.XGBClassifier(**xgb_params)  # type: ignore
        else:
            model = self._xgb.XGBRegressor(**xgb_params)  # type: ignore

        # Train with early stopping
        start_time = time.time()
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=self._xgb_config.early_stopping_rounds,
            verbose=False,
        )
        training_time = time.time() - start_time

        print(f"XGBoost training completed in {training_time:.2f}s")
        print(f"Best iteration: {model.best_iteration}")
        print(f"Best score: {model.best_score:.4f}")

        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(model)

        # Calculate SHAP values if enabled
        shap_results = {}
        if self._xgb_config.enable_shap:
            print("Computing SHAP values...")
            shap_results = self._calculate_shap_values(model, X_val)

        return {
            "model": model,
            "metrics": {
                "best_iteration": model.best_iteration,
                "best_score": model.best_score,
                "training_time": training_time,
            },
            "feature_importance": feature_importance,
            "shap_results": shap_results,
        }

    def _create_monotonic_constraints(
        self,
        feature_names: list[str],
        constraints_dict: dict[str, int],
    ) -> str:
        """
        Create monotonic constraints string for XGBoost.

        Parameters
        ----------
        feature_names : list[str]
            List of feature names in order.
        constraints_dict : dict[str, int]
            Dictionary mapping feature names to constraint values.

        Returns
        -------
        str
            Monotonic constraints string.

        """
        constraints = []
        for feature in feature_names:
            if feature in constraints_dict:
                constraints.append(str(constraints_dict[feature]))
            else:
                constraints.append("0")
        return f"({','.join(constraints)})"

    def _calculate_feature_importance(self, model: Any) -> dict[str, float]:
        """
        Calculate feature importance scores.

        Parameters
        ----------
        model : XGBoost model
            Trained XGBoost model.

        Returns
        -------
        dict[str, float]
            Dictionary of feature names to importance scores.

        """
        importance_dict = {}

        # Get importance from model
        for feature, importance in zip(self._feature_names, model.feature_importances_):
            importance_dict[feature] = float(importance)

        # Sort by importance (descending)
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    def _calculate_shap_values(
        self,
        model: Any,
        X_sample: np.ndarray,
        max_samples: int = 1000,
    ) -> dict[str, Any]:
        """
        Calculate SHAP values for model explainability.

        Parameters
        ----------
        model : XGBoost model
            Trained XGBoost model.
        X_sample : np.ndarray
            Sample data for SHAP calculation.
        max_samples : int, default 1000
            Maximum number of samples to use for SHAP computation.

        Returns
        -------
        dict[str, Any]
            Dictionary containing SHAP values and importance scores.

        """
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
        explainer = self._shap.TreeExplainer(model)  # type: ignore
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
            "shap_importance": dict(
                sorted(shap_importance.items(), key=lambda x: x[1], reverse=True),
            ),
            "expected_value": (
                explainer.expected_value if hasattr(explainer, "expected_value") else 0.0
            ),
        }

    def save_model(self, path: str | Path) -> None:
        """
        Save the trained XGBoost model with enhanced metadata.

        Parameters
        ----------
        path : str | Path
            Path where to save the model.

        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before saving")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Enhanced model data for XGBoost
        model_data = {
            "model": self._model,
            "feature_names": self._feature_names,
            "training_metrics": self._training_metrics,
            "scaler": self._scaler,
            "config": {
                "xgb_params": self._xgb_config.get_xgb_params(),
                "multi_asset": self._xgb_config.multi_asset,
                "feature_config": self._feature_config,
            },
        }

        with open(save_path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"XGBoost model saved to {save_path}")

    def get_feature_importance_summary(self) -> dict[str, Any]:
        """
        Get a comprehensive feature importance summary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing various importance metrics.

        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted to get feature importance")

        summary: dict[str, Any] = {}

        # Native XGBoost importance
        if hasattr(self._model, "feature_importances_"):
            importance_dict = {}
            for feature, importance in zip(
                self._feature_names,
                self._model.feature_importances_,
            ):
                importance_dict[feature] = float(importance)
            summary["xgb_importance"] = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True),
            )

        # SHAP importance if available
        if (
            "shap_results" in self._training_metrics
            and "shap_importance" in self._training_metrics["shap_results"]
        ):
            summary["shap_importance"] = self._training_metrics["shap_results"]["shap_importance"]

        # Top features
        if "xgb_importance" in summary:
            top_10_items = list(summary["xgb_importance"].items())[:10]
            summary["top_10_features"] = top_10_items

        return summary
