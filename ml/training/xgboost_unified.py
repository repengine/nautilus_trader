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
Unified XGBoost trainer with advanced ML features.

This module provides the UnifiedXGBoostTrainer which extends the base XGBoostTrainer
with advanced features including GPU acceleration, hyperparameter optimization,
MLflow tracking, feature decay monitoring, and comprehensive model export capabilities.

"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ml._imports import HAS_MLFLOW, HAS_OPTUNA, HAS_ONNX, HAS_POLARS, HAS_XGBOOST
from ml._imports import check_ml_dependencies
from ml._imports import mlflow, optuna, ort, pl, xgb
from ml.config.xgboost_unified import UnifiedXGBoostConfig
from ml.monitoring.collectors.model import ModelLifecycleCollector
from ml.monitoring._config import MonitoringConfig
from ml.training.mlflow_tracker import MLflowXGBoostTracker
from ml.training.optuna_optimizer import XGBoostOptunaOptimizer
from ml.training.xgboost import XGBoostTrainer


class UnifiedXGBoostTrainer(XGBoostTrainer):
    """
    Unified XGBoost trainer with advanced ML features.

    This trainer extends the base XGBoostTrainer with comprehensive features
    migrated from the OLD system, including GPU acceleration, hyperparameter
    optimization, MLflow tracking, and advanced monitoring capabilities.

    Features:
    - GPU acceleration with automatic validation
    - Optuna hyperparameter optimization with pruning
    - MLflow experiment tracking and model registry
    - Feature importance decay tracking over time
    - ONNX model export for high-performance inference
    - Cross-validation with multiple strategies
    - Comprehensive Prometheus metrics integration
    - Automatic model validation and quality checks

    Parameters
    ----------
    config : UnifiedXGBoostConfig
        Configuration for unified XGBoost training.

    """

    def __init__(self, config: UnifiedXGBoostConfig) -> None:
        """
        Initialize unified XGBoost trainer.

        Parameters
        ----------
        config : UnifiedXGBoostConfig
            Unified configuration for advanced XGBoost training.

        """
        super().__init__(config)
        self._unified_config = config

        # Feature decay tracking
        self._importance_history: list[dict[str, float]] = []
        self._feature_decay_alerts: list[str] = []

        # Monitoring
        self._metrics_collector: ModelLifecycleCollector | None
        if config.enable_monitoring:
            monitoring_config = MonitoringConfig()  # Use default monitoring config
            self._metrics_collector = ModelLifecycleCollector(monitoring_config)
        else:
            self._metrics_collector = None

        # Lazy-loaded components
        self._mlflow_tracker: MLflowXGBoostTracker | None = None
        self._optuna_optimizer: XGBoostOptunaOptimizer | None = None

        # Validation metadata
        self._validation_metadata: dict[str, Any] = {}

        # Environment validation
        if config.gpu_config.enabled or config.optuna_config.enabled or config.mlflow_config.enabled:
            warnings = config.validate_environment()
            for warning in warnings:
                print(f"⚠️ {warning}")

    def train(
        self,
        data: Any,  # pl.DataFrame or dict[str, pl.DataFrame]
        validation_data: Any | None = None,
        target_col: str = "target",
        optimize_hyperparams: bool | None = None,
        cv_validate: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Train XGBoost model with unified advanced features.

        Parameters
        ----------
        data : Any
            Training data (DataFrame for single asset, dict for multi-asset).
        target_col : str, default "target"
            Target column name.
        optimize_hyperparams : bool | None, optional
            Override config setting for hyperparameter optimization.
        cv_validate : bool, default False
            Perform cross-validation in addition to train/test split.

        Returns
        -------
        dict[str, Any]
            Comprehensive training results including:
            - model: Trained XGBoost model
            - metrics: Training and validation metrics
            - feature_importance: Feature importance scores
            - optimization_results: Hyperparameter optimization results (if enabled)
            - mlflow_run_id: MLflow run ID (if enabled)
            - cv_results: Cross-validation results (if enabled)
            - onnx_path: ONNX export path (if enabled)

        """
        training_start_time = time.time()
        print("🚀 Starting unified XGBoost training...")

        # Initialize tracking if enabled
        if self._unified_config.mlflow_config.enabled:
            self._init_mlflow_tracking()

        # Prepare data
        print("📊 Preparing training data...")
        X, y, metadata = self.prepare_data(data, target_col)
        self._feature_names = metadata["feature_names"]
        
        # Validate GPU setup if enabled
        if self._unified_config.gpu_config.enabled:
            self._validate_gpu_setup()

        # Handle validation data - use provided validation_data or split automatically
        if validation_data is not None:
            X_val, y_val, _ = self.prepare_data(validation_data, target_col)
            X_train, y_train = X, y
        else:
            # Split data (80/20 train/validation)
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

        print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")

        # Cross-validation if requested
        cv_results = {}
        if cv_validate:
            print("🔄 Performing cross-validation...")
            cv_results = self._perform_cross_validation(X, y)

        # Hyperparameter optimization
        best_params = self._unified_config.get_unified_xgb_params()
        optimization_results = {}

        should_optimize = (
            optimize_hyperparams
            if optimize_hyperparams is not None
            else self._unified_config.optuna_config.enabled
        )

        if should_optimize:
            print("🔍 Starting hyperparameter optimization...")
            optimization_results = self._optimize_hyperparameters(X_train, y_train, X_val, y_val)
            best_params.update(optimization_results.get("best_params", {}))

        # Train final model
        print("🎯 Training final model with optimized parameters...")
        training_results = self._train_model_unified(X_train, y_train, X_val, y_val, best_params)

        # Track feature importance decay
        if self._unified_config.track_feature_decay:
            self._track_feature_decay(training_results["feature_importance"])

        # Export to ONNX if configured
        onnx_path = None
        if self._unified_config.export_onnx:
            print("📦 Exporting model to ONNX...")
            onnx_path = self._export_to_onnx(training_results["model"])

        # Log to MLflow
        mlflow_run_id = None
        if self._unified_config.mlflow_config.enabled:
            print("📝 Logging to MLflow...")
            mlflow_run_id = self._log_mlflow_run(
                model=training_results["model"],
                params=best_params,
                metrics=training_results["metrics"],
                importance=training_results["feature_importance"],
                cv_results=cv_results,
                optimization_results=optimization_results,
            )

        # Record monitoring metrics
        if self._metrics_collector is not None:
            self._record_training_metrics(training_results, optimization_results)

        # Compile comprehensive results
        training_time = time.time() - training_start_time
        results = {
            **training_results,
            **optimization_results,
            "cv_results": cv_results,
            "mlflow_run_id": mlflow_run_id,
            "onnx_path": onnx_path,
            "feature_decay_alerts": self._feature_decay_alerts.copy(),
            "total_training_time": training_time,
            "metadata": metadata,
        }

        # Store results for later use
        self._model = training_results["model"]
        self._is_fitted = True
        self._training_metrics = results

        print(f"✅ Training completed in {training_time:.1f}s")
        self._print_training_summary(results)

        return results

    def _validate_gpu_setup(self) -> None:
        """Validate GPU setup and configuration."""
        if not self._unified_config.gpu_config.validate_gpu:
            return

        try:
            # Test GPU availability
            import subprocess
            result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError("nvidia-smi not available")
            
            gpu_count = len([line for line in result.stdout.split('\n') if 'GPU' in line])
            if self._unified_config.gpu_config.device_id >= gpu_count:
                raise ValueError(f"GPU {self._unified_config.gpu_config.device_id} not available. Found {gpu_count} GPUs.")
            
            print(f"✅ GPU {self._unified_config.gpu_config.device_id} validated")

        except Exception as e:
            print(f"⚠️ GPU validation failed: {e}")
            print("Falling back to CPU training...")
            # Override GPU settings - create new config with GPU disabled
            from ml.config.xgboost_unified import GPUConfig
            disabled_gpu_config = GPUConfig(enabled=False)
            
            # Create a new config with disabled GPU (working with frozen structs)
            # Use type: ignore since mypy has trouble with the **kwargs pattern for msgspec structs
            self._unified_config = self._unified_config.__class__(
                data_source=self._unified_config.data_source,
                n_estimators=self._unified_config.n_estimators,
                max_depth=self._unified_config.max_depth,
                learning_rate=self._unified_config.learning_rate,
                objective=self._unified_config.objective,
                enable_monitoring=self._unified_config.enable_monitoring,
                feature_config=self._unified_config.feature_config,
                track_feature_decay=self._unified_config.track_feature_decay,
                feature_decay_threshold=self._unified_config.feature_decay_threshold,
                cv_strategy=self._unified_config.cv_strategy,
                cv_folds=self._unified_config.cv_folds,
                export_onnx=self._unified_config.export_onnx,
                gpu_config=disabled_gpu_config,  # Disable GPU
                optuna_config=self._unified_config.optuna_config,
                mlflow_config=self._unified_config.mlflow_config,
            )

    def _init_mlflow_tracking(self) -> None:
        """Initialize MLflow tracking."""
        self._mlflow_tracker = MLflowXGBoostTracker(self._unified_config.mlflow_config)

    def _perform_cross_validation(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """
        Perform cross-validation with the configured strategy.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target array.

        Returns
        -------
        dict[str, Any]
            Cross-validation results.

        """
        try:
            from sklearn.model_selection import TimeSeriesSplit, KFold
            from sklearn.metrics import accuracy_score, mean_squared_error
        except ImportError:
            print("⚠️ sklearn required for cross-validation. Skipping.")
            return {}

        # Select CV strategy
        if self._unified_config.cv_strategy == "time_series":
            cv = TimeSeriesSplit(n_splits=self._unified_config.cv_folds)
        elif self._unified_config.cv_strategy == "standard":
            cv = KFold(n_splits=self._unified_config.cv_folds, shuffle=True, random_state=self._unified_config.random_seed)
        else:
            print(f"⚠️ Unsupported CV strategy: {self._unified_config.cv_strategy}")
            return {}

        scores = defaultdict(list)
        fold_results = []

        base_params = self._unified_config.get_unified_xgb_params()

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            print(f"  Fold {fold + 1}/{self._unified_config.cv_folds}")

            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]

            # Train fold model
            if self._unified_config.objective == "binary:logistic":
                model = xgb.XGBClassifier(**base_params)
            else:
                model = xgb.XGBRegressor(**base_params)

            model.fit(
                X_fold_train,
                y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                early_stopping_rounds=50,
                verbose=False,
            )

            # Evaluate fold
            if hasattr(model, "predict_proba"):
                predictions = model.predict_proba(X_fold_val)[:, 1]
                accuracy = accuracy_score(y_fold_val, (predictions > 0.5).astype(int))
                scores["accuracy"].append(accuracy)
            else:
                predictions = model.predict(X_fold_val)
                mse = mean_squared_error(y_fold_val, predictions)
                scores["mse"].append(mse)
                scores["rmse"].append(np.sqrt(mse))

            # Calculate Sharpe ratio
            sharpe = self._calculate_sharpe_ratio(y_fold_val, predictions)
            scores["sharpe"].append(sharpe)

            fold_results.append({
                "fold": fold + 1,
                "train_size": len(X_fold_train),
                "val_size": len(X_fold_val),
                "best_iteration": model.best_iteration,
                "predictions": predictions[:10].tolist(),  # Sample predictions
            })

        # Aggregate results
        cv_summary = {}
        for metric, values in scores.items():
            cv_summary[f"{metric}_mean"] = np.mean(values)
            cv_summary[f"{metric}_std"] = np.std(values)
            cv_summary[f"{metric}_scores"] = values

        return {
            "strategy": self._unified_config.cv_strategy,
            "n_folds": self._unified_config.cv_folds,
            "summary": cv_summary,
            "fold_results": fold_results,
        }

    def _optimize_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        if self._optuna_optimizer is None:
            self._optuna_optimizer = XGBoostOptunaOptimizer(self._unified_config.optuna_config)

        base_params = self._unified_config.get_unified_xgb_params()

        # Select metric function
        metric_function = self._get_optimization_metric_function()

        # Create objective
        objective = self._optuna_optimizer.create_objective_function(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            base_params=base_params,
            metric_function=metric_function,
            early_stopping_rounds=self._unified_config.early_stopping_rounds,
        )

        # Run optimization
        results = self._optuna_optimizer.optimize(objective)

        print(f"🎯 Optimization complete: {results['n_trials']} trials")
        print(f"Best {self._unified_config.optuna_config.metric}: {results['best_value']:.4f}")
        print(f"Best params: {results['best_params']}")

        return results

    def _get_optimization_metric_function(self) -> Callable[[np.ndarray, np.ndarray], float]:
        """Get the metric function for optimization."""
        metric = self._unified_config.optuna_config.metric

        if metric == "sharpe_ratio":
            return self._calculate_sharpe_ratio
        elif metric == "accuracy":
            return self._calculate_accuracy
        elif metric == "auc":
            return self._calculate_auc
        elif metric == "rmse":
            return lambda y_true, y_pred: -np.sqrt(np.mean((y_true - y_pred) ** 2))  # Negative for maximization
        else:
            # Default to Sharpe ratio
            return self._calculate_sharpe_ratio

    def _train_model_unified(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Train XGBoost model with unified parameters and monitoring."""
        if not HAS_XGBOOST:
            check_ml_dependencies(["xgboost"])

        # Apply monotonic constraints if specified
        if self._unified_config.monotonic_constraints:
            constraints = self._create_monotonic_constraints_string()
            params["monotone_constraints"] = constraints
            print(f"Applied monotonic constraints: {constraints}")

        # Create model
        if self._unified_config.objective == "binary:logistic":
            model = xgb.XGBClassifier(**params)
        else:
            model = xgb.XGBRegressor(**params)

        # Train with monitoring
        start_time = time.time()
        
        callbacks: list[Any] = []
        
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=self._unified_config.early_stopping_rounds,
            verbose=self._unified_config.gpu_config.enabled,  # Show progress for GPU training
            callbacks=callbacks if callbacks else None,
        )

        training_time = time.time() - start_time

        # Generate predictions for metrics
        if hasattr(model, "predict_proba"):
            val_preds = model.predict_proba(X_val)[:, 1]
        else:
            val_preds = model.predict(X_val)

        # Calculate comprehensive metrics
        metrics = {
            "training_time": training_time,
            "best_iteration": model.best_iteration if hasattr(model, 'best_iteration') else 0,
            "best_score": model.best_score if hasattr(model, 'best_score') else 0.0,
            "n_features": X_train.shape[1],
            "n_train_samples": X_train.shape[0],
            "n_val_samples": X_val.shape[0],
            "val_accuracy": self._calculate_accuracy(y_val, val_preds),
            "val_sharpe": self._calculate_sharpe_ratio(y_val, val_preds),
        }

        # Add AUC for classification
        if self._unified_config.objective == "binary:logistic":
            metrics["val_auc"] = self._calculate_auc(y_val, val_preds)

        # Calculate feature importance
        importance = self._calculate_enhanced_importance(model)

        # SHAP analysis if enabled
        shap_results = {}
        if self._unified_config.enable_shap:
            print("🔍 Computing SHAP values...")
            shap_results = self._calculate_shap_values(model, X_val[:1000])  # Sample for efficiency

        return {
            "model": model,
            "metrics": metrics,
            "feature_importance": importance,
            "shap_results": shap_results,
        }

    def _create_monotonic_constraints_string(self) -> str:
        """Create monotonic constraints string for XGBoost."""
        if not self._unified_config.monotonic_constraints:
            return "()"

        constraints = []
        for feature in self._feature_names:
            constraint_value = self._unified_config.monotonic_constraints.get(feature, 0)
            constraints.append(str(constraint_value))

        return f"({','.join(constraints)})"

    def _calculate_enhanced_importance(self, model: Any) -> dict[str, float]:
        """Calculate enhanced feature importance with multiple methods."""
        importance_dict = {}

        # Get native XGBoost importance
        if hasattr(model, "feature_importances_"):
            for feature, score in zip(self._feature_names, model.feature_importances_):
                importance_dict[feature] = float(score)

        # Sort by importance (descending)
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    def _track_feature_decay(self, current_importance: dict[str, float]) -> None:
        """Track feature importance decay over time."""
        if not self._importance_history:
            self._importance_history.append(current_importance)
            return

        # Calculate historical average (sliding window)
        window_size = min(
            len(self._importance_history), self._unified_config.feature_history_window
        )
        recent_history = self._importance_history[-window_size:]

        # Calculate average importance
        avg_importance: dict[str, float] = defaultdict(float)
        for hist in recent_history:
            for feature, score in hist.items():
                avg_importance[feature] += score / window_size

        # Check for decay
        self._feature_decay_alerts.clear()
        threshold = self._unified_config.feature_decay_threshold

        for feature, current_score in current_importance.items():
            historical_score = avg_importance.get(feature, 0)
            if historical_score > 0:
                decay_ratio = (historical_score - current_score) / historical_score
                if decay_ratio > threshold:
                    self._feature_decay_alerts.append(feature)
                    print(
                        f"⚠️ Feature decay alert: '{feature}' "
                        f"importance declined by {decay_ratio:.1%}"
                    )

        # Update history
        self._importance_history.append(current_importance)

        # Trim history if needed
        max_history = self._unified_config.feature_history_window * 2
        if len(self._importance_history) > max_history:
            self._importance_history = self._importance_history[-max_history:]

    def _calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate prediction accuracy."""
        if self._unified_config.objective == "binary:logistic":
            y_pred_binary = (y_pred > 0.5).astype(int)
            return float(np.mean(y_true == y_pred_binary))
        else:
            # For regression, use R² score
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    def _calculate_auc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate AUC score."""
        try:
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(y_true, y_pred))
        except ImportError:
            print("⚠️ sklearn required for AUC calculation")
            return 0.0
        except Exception:
            return 0.0

    def _calculate_sharpe_ratio(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        risk_free_rate: float = 0.0,
    ) -> float:
        """Calculate Sharpe ratio for predictions."""
        if self._unified_config.objective == "binary:logistic":
            # Binary predictions as signals
            signals = (y_pred > 0.5).astype(float) * 2 - 1  # -1 or 1
            returns = signals * y_true
        else:
            # Regression predictions as returns
            returns = y_pred * y_true

        # Calculate Sharpe ratio
        excess_returns = returns - risk_free_rate
        if len(excess_returns) > 1 and np.std(excess_returns) > 0:
            sharpe = np.mean(excess_returns) / np.std(excess_returns)
            # Annualize (assuming daily data)
            return float(sharpe * np.sqrt(252))
        return 0.0

    def _export_to_onnx(self, model: Any) -> str | None:
        """Export model to ONNX format."""
        try:
            check_ml_dependencies(["onnx"])
            import onnxmltools
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError:
            print("⚠️ ONNX tools not available. Skipping export.")
            return None

        try:
            # Define input type
            initial_types = [("float_input", FloatTensorType([None, len(self._feature_names)]))]

            # Convert to ONNX
            onnx_model = onnxmltools.convert_xgboost(
                model, initial_types=initial_types, target_opset=12
            )

            # Ensure output directory exists
            output_path = Path(self._unified_config.onnx_output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save ONNX model
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            # Save feature metadata
            metadata_path = output_path.with_suffix(".json")
            metadata = {
                "feature_names": self._feature_names,
                "model_type": "xgboost",
                "objective": self._unified_config.objective,
                "n_features": len(self._feature_names),
                "export_timestamp": int(time.time()),
                "nautilus_version": "2.0.0",
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"✅ Model exported to ONNX: {output_path}")
            return str(output_path)

        except Exception as e:
            print(f"⚠️ ONNX export failed: {e}")
            return None

    def _log_mlflow_run(
        self,
        model: Any,
        params: dict[str, Any],
        metrics: dict[str, float],
        importance: dict[str, float],
        cv_results: dict[str, Any] | None = None,
        optimization_results: dict[str, Any] | None = None,
    ) -> str | None:
        """Log comprehensive training run to MLflow."""
        if self._mlflow_tracker is None:
            return None

        # Prepare artifacts
        artifacts = {
            "feature_names": self._feature_names,
            "feature_decay_alerts": self._feature_decay_alerts,
            "config_summary": {
                "multi_asset": self._unified_config.multi_asset,
                "gpu_enabled": self._unified_config.gpu_config.enabled,
                "optuna_enabled": self._unified_config.optuna_config.enabled,
                "cv_strategy": self._unified_config.cv_strategy,
                "track_decay": self._unified_config.track_feature_decay,
            },
        }

        # Add CV results
        if cv_results:
            artifacts["cv_results"] = cv_results

        # Add optimization results
        if optimization_results:
            artifacts["optimization_results"] = {
                k: v for k, v in optimization_results.items() if k != "study"  # Skip study object
            }

        # Create tags
        tags = {
            "model_version": "unified",
            "gpu_training": str(self._unified_config.gpu_config.enabled),
            "hyperopt": str(self._unified_config.optuna_config.enabled),
            "n_features": str(len(self._feature_names)),
        }

        return self._mlflow_tracker.log_training_run(
            model=model,
            params=params,
            metrics=metrics,
            feature_importance=importance,
            feature_names=self._feature_names,
            artifacts=artifacts,
            tags=tags,
        )

    def _record_training_metrics(
        self,
        training_results: dict[str, Any],
        optimization_results: dict[str, Any] | None = None,
    ) -> None:
        """Record training metrics to monitoring collector."""
        if self._metrics_collector is None:
            return

        metrics = training_results.get("metrics", {})
        model_id = f"xgboost_unified_{int(time.time())}"

        # Record model training using available ModelLifecycleCollector method
        self._metrics_collector.record_model_training(
            model=model_id,
            training_duration=metrics.get("training_time", 0.0),
            training_samples=metrics.get("n_train_samples", 0),
            training_score=metrics.get("val_accuracy"),
            validation_score=metrics.get("val_sharpe"),
            phase="training",
            metric_type="accuracy",
        )

        # Note: Feature importance recording would need a separate collector
        # or extension of the ModelLifecycleCollector if needed in production

    def _print_training_summary(self, results: dict[str, Any]) -> None:
        """Print comprehensive training summary."""
        print("\n" + "=" * 60)
        print("📊 UNIFIED XGBOOST TRAINING SUMMARY")
        print("=" * 60)

        # Basic metrics
        metrics = results.get("metrics", {})
        print(f"Training Time: {results.get('total_training_time', 0):.1f}s")
        print(f"Validation Accuracy: {metrics.get('val_accuracy', 0):.4f}")
        print(f"Validation Sharpe: {metrics.get('val_sharpe', 0):.4f}")
        if "val_auc" in metrics:
            print(f"Validation AUC: {metrics['val_auc']:.4f}")

        # Feature information
        print(f"Features: {metrics.get('n_features', 0)}")
        if self._feature_decay_alerts:
            print(f"⚠️ Feature Decay Alerts: {len(self._feature_decay_alerts)}")

        # Optimization results
        if "best_params" in results:
            print(f"Optimization Trials: {results.get('n_trials', 0)}")
            print(f"Best Value: {results.get('best_value', 0):.4f}")

        # Export information
        if results.get("onnx_path"):
            print(f"ONNX Export: {results['onnx_path']}")
        if results.get("mlflow_run_id"):
            print(f"MLflow Run: {results['mlflow_run_id']}")

        print("=" * 60)

    def get_feature_decay_summary(self) -> dict[str, Any]:
        """
        Get summary of feature importance decay tracking.

        Returns
        -------
        dict[str, Any]
            Feature decay tracking summary.

        """
        if not self._unified_config.track_feature_decay:
            return {"tracking_enabled": False}

        return {
            "tracking_enabled": True,
            "history_length": len(self._importance_history),
            "current_alerts": self._feature_decay_alerts.copy(),
            "decay_threshold": self._unified_config.feature_decay_threshold,
            "history_window": self._unified_config.feature_history_window,
        }

    def get_model_metadata(self) -> dict[str, Any]:
        """
        Get comprehensive model metadata.

        Returns
        -------
        dict[str, Any]
            Model metadata including configuration and performance.

        """
        if not self._is_fitted:
            return {"fitted": False}

        return {
            "fitted": True,
            "model_type": "xgboost_unified",
            "config": {
                "gpu_enabled": self._unified_config.gpu_config.enabled,
                "optuna_enabled": self._unified_config.optuna_config.enabled,
                "mlflow_enabled": self._unified_config.mlflow_config.enabled,
                "multi_asset": self._unified_config.multi_asset,
                "objective": self._unified_config.objective,
            },
            "features": {
                "n_features": len(self._feature_names) if self._feature_names else 0,
                "feature_names": self._feature_names,
                "decay_alerts": self._feature_decay_alerts,
            },
            "performance": self._training_metrics.get("metrics", {}),
            "training_time": self._training_metrics.get("total_training_time", 0),
        }


# Explicit exports
__all__ = [
    "UnifiedXGBoostTrainer",
]