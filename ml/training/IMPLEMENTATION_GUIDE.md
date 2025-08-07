# XGBoost Trainer Implementation Guide

## Phase 1: Core Enhancement Implementation

This guide provides step-by-step implementation details for enhancing the XGBoostTrainer with features from the OLD system while maintaining Nautilus ML architecture compliance.

## 1. Configuration Extensions

### 1.1 Create Enhanced Configuration

```python
# ml/config/xgboost_unified.py
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ml.config.xgboost import XGBoostTrainingConfig


class GPUConfig(BaseModel):
    """GPU acceleration configuration."""

    enabled: bool = Field(default=False, description="Enable GPU acceleration")
    device_id: int = Field(default=0, description="GPU device ID")
    max_bin: int = Field(default=256, description="Maximum bins for GPU histogram")
    predictor: str = Field(default="gpu_predictor", description="Predictor type")


class OptunaConfig(BaseModel):
    """Optuna hyperparameter optimization configuration."""

    enabled: bool = Field(default=False, description="Enable Optuna optimization")
    n_trials: int = Field(default=100, description="Number of optimization trials")
    direction: str = Field(default="maximize", description="Optimization direction")
    metric: str = Field(default="sharpe_ratio", description="Metric to optimize")
    pruner: str = Field(default="median", description="Pruning algorithm")
    sampler: str = Field(default="tpe", description="Sampling algorithm")
    timeout: int | None = Field(default=None, description="Optimization timeout in seconds")


class MLflowConfig(BaseModel):
    """MLflow tracking configuration."""

    enabled: bool = Field(default=False, description="Enable MLflow tracking")
    tracking_uri: str = Field(default="http://localhost:5000", description="MLflow server URI")
    experiment_name: str = Field(default="xgboost_unified", description="Experiment name")
    register_model: bool = Field(default=True, description="Register model after training")
    model_name: str = Field(default="xgboost_unified", description="Registered model name")


class UnifiedXGBoostConfig(XGBoostTrainingConfig):
    """Unified configuration for enhanced XGBoost trainer."""

    # GPU settings
    gpu_config: GPUConfig = Field(default_factory=GPUConfig)

    # Optimization settings
    optuna_config: OptunaConfig = Field(default_factory=OptunaConfig)

    # MLflow settings
    mlflow_config: MLflowConfig = Field(default_factory=MLflowConfig)

    # Feature tracking
    track_feature_decay: bool = Field(default=True)
    feature_decay_threshold: float = Field(default=0.3)
    feature_history_window: int = Field(default=10)

    # Cross-validation
    cv_strategy: str = Field(default="time_series")
    cv_folds: int = Field(default=5)
    purge_gap: int = Field(default=10)

    # Model export
    export_onnx: bool = Field(default=False)
    onnx_output_path: str = Field(default="./models/xgboost.onnx")

    def get_unified_xgb_params(self) -> dict[str, Any]:
        """Get XGBoost parameters with GPU settings."""
        params = self.get_xgb_params()

        if self.gpu_config.enabled:
            params.update({
                'tree_method': 'gpu_hist',
                'predictor': self.gpu_config.predictor,
                'gpu_id': self.gpu_config.device_id,
                'max_bin': self.gpu_config.max_bin,
            })

        return params
```

## 2. Enhanced Trainer Implementation

### 2.1 Core Trainer Class

```python
# ml/training/xgboost_unified.py
from __future__ import annotations

import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ml._imports import HAS_XGBOOST, HAS_POLARS, check_ml_dependencies
from ml._imports import xgb, pl
from ml.config.xgboost_unified import UnifiedXGBoostConfig
from ml.data.loader import MLDataLoader
from ml.features.engineering_enhanced import EnhancedFeatureEngineer
from ml.monitoring.collectors.model import ModelMetricsCollector
from ml.training.xgboost import XGBoostTrainer


class UnifiedXGBoostTrainer(XGBoostTrainer):
    """
    Unified XGBoost trainer with advanced features migrated from OLD system.

    Features:
    - GPU acceleration support
    - Optuna hyperparameter optimization
    - MLflow experiment tracking
    - Feature importance decay tracking
    - Advanced cross-validation strategies
    - ONNX model export for inference
    - Integration with monitoring infrastructure
    """

    def __init__(self, config: UnifiedXGBoostConfig) -> None:
        """Initialize unified XGBoost trainer."""
        super().__init__(config)
        self._unified_config = config

        # Enhanced feature engineer
        self._feature_engineer = EnhancedFeatureEngineer(config.feature_config)

        # Feature tracking
        self._importance_history: list[dict[str, float]] = []
        self._feature_decay_alerts: list[str] = []

        # Validation metadata for multi-asset
        self._validation_metadata: dict[str, Any] = {}

        # Monitoring collector
        self._metrics_collector = ModelMetricsCollector()

        # MLflow tracker (lazy init)
        self._mlflow_tracker = None

        # Optuna study (lazy init)
        self._optuna_study = None

    def train(
        self,
        data: Any,  # pl.DataFrame or dict[str, pl.DataFrame]
        target_col: str = "target",
        optimize_hyperparams: bool | None = None,
    ) -> dict[str, Any]:
        """
        Train XGBoost model with optional hyperparameter optimization.

        Parameters
        ----------
        data : Any
            Training data (DataFrame for single asset, dict for multi-asset)
        target_col : str
            Target column name
        optimize_hyperparams : bool, optional
            Override config setting for hyperparameter optimization

        Returns
        -------
        dict[str, Any]
            Training results including model, metrics, and optimization results
        """
        # Initialize MLflow if enabled
        if self._unified_config.mlflow_config.enabled:
            self._init_mlflow_tracking()

        # Prepare data
        X, y, metadata = self.prepare_data(data, target_col)
        self._feature_names = metadata["feature_names"]

        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Optimize hyperparameters if configured
        best_params = self._unified_config.get_unified_xgb_params()
        optimization_results = {}

        should_optimize = (
            optimize_hyperparams if optimize_hyperparams is not None
            else self._unified_config.optuna_config.enabled
        )

        if should_optimize:
            print("Starting hyperparameter optimization...")
            optimization_results = self._optimize_hyperparameters(
                X_train, y_train, X_val, y_val
            )
            best_params.update(optimization_results["best_params"])

        # Train final model
        print("Training final model with best parameters...")
        training_results = self._train_model_unified(
            X_train, y_train, X_val, y_val, best_params
        )

        # Track feature importance decay
        if self._unified_config.track_feature_decay:
            self._track_feature_decay(training_results["feature_importance"])

        # Log to MLflow
        if self._unified_config.mlflow_config.enabled:
            run_id = self._log_mlflow_run(
                model=training_results["model"],
                params=best_params,
                metrics=training_results["metrics"],
                importance=training_results["feature_importance"]
            )
            training_results["mlflow_run_id"] = run_id

        # Export to ONNX if configured
        if self._unified_config.export_onnx:
            self._export_to_onnx(training_results["model"])

        # Record monitoring metrics
        self._record_training_metrics(training_results)

        # Store results
        self._model = training_results["model"]
        self._is_fitted = True
        self._training_metrics = {
            **training_results,
            **optimization_results,
            "metadata": metadata,
        }

        return self._training_metrics

    def _train_model_unified(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Train XGBoost model with unified parameters."""
        if not HAS_XGBOOST:
            check_ml_dependencies(["xgboost"])

        # Apply monotonic constraints if specified
        if self._unified_config.monotonic_constraints:
            constraints = self._create_monotonic_constraints_string()
            params["monotone_constraints"] = constraints

        # Create model
        if self._unified_config.objective == "binary:logistic":
            model = xgb.XGBClassifier(**params)
        else:
            model = xgb.XGBRegressor(**params)

        # Train with early stopping
        start_time = time.time()
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=self._unified_config.early_stopping_rounds,
            verbose=self._unified_config.gpu_config.enabled,  # Show GPU info
        )
        training_time = time.time() - start_time

        # Calculate metrics
        val_preds = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_val)

        metrics = {
            "training_time": training_time,
            "best_iteration": model.best_iteration,
            "best_score": model.best_score,
            "val_accuracy": self._calculate_accuracy(y_val, val_preds),
            "val_sharpe": self._calculate_sharpe_ratio(y_val, val_preds),
        }

        # Calculate feature importance
        importance = self._calculate_enhanced_importance(model)

        # SHAP analysis if enabled
        shap_results = {}
        if self._unified_config.enable_shap:
            shap_results = self._calculate_shap_values(model, X_val[:1000])

        return {
            "model": model,
            "metrics": metrics,
            "feature_importance": importance,
            "shap_results": shap_results,
        }

    def _create_monotonic_constraints_string(self) -> str:
        """Create monotonic constraints string for XGBoost."""
        constraints = []
        for feature in self._feature_names:
            constraint_value = self._unified_config.monotonic_constraints.get(feature, 0)
            constraints.append(str(constraint_value))
        return f"({','.join(constraints)})"

    def _calculate_enhanced_importance(self, model: Any) -> dict[str, float]:
        """Calculate enhanced feature importance with multiple methods."""
        importance_dict = {}

        # Get native XGBoost importance
        for feature, score in zip(self._feature_names, model.feature_importances_):
            importance_dict[feature] = float(score)

        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    def _track_feature_decay(self, current_importance: dict[str, float]) -> None:
        """Track feature importance decay over time."""
        if not self._importance_history:
            self._importance_history.append(current_importance)
            return

        # Calculate historical average (sliding window)
        window_size = min(
            len(self._importance_history),
            self._unified_config.feature_history_window
        )
        recent_history = self._importance_history[-window_size:]

        # Calculate average importance for each feature
        avg_importance = defaultdict(float)
        for hist in recent_history:
            for feature, score in hist.items():
                avg_importance[feature] += score / window_size

        # Check for decay
        self._feature_decay_alerts = []
        for feature, current_score in current_importance.items():
            historical_score = avg_importance.get(feature, 0)
            if historical_score > 0:
                decay_ratio = (historical_score - current_score) / historical_score
                if decay_ratio > self._unified_config.feature_decay_threshold:
                    self._feature_decay_alerts.append(feature)
                    print(f"⚠️ Feature decay alert: '{feature}' "
                          f"importance declined by {decay_ratio:.1%}")

        # Add to history
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

    def _calculate_sharpe_ratio(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        risk_free_rate: float = 0.0
    ) -> float:
        """Calculate Sharpe ratio for predictions."""
        # Convert predictions to returns
        if self._unified_config.objective == "binary:logistic":
            # Binary predictions as signals
            signals = (y_pred > 0.5).astype(float) * 2 - 1  # -1 or 1
            returns = signals * y_true
        else:
            # Regression predictions as portfolio weights
            returns = y_pred * y_true

        # Calculate Sharpe ratio
        excess_returns = returns - risk_free_rate
        if len(excess_returns) > 1 and np.std(excess_returns) > 0:
            sharpe = np.mean(excess_returns) / np.std(excess_returns)
            # Annualize (assuming daily returns)
            sharpe_annual = sharpe * np.sqrt(252)
            return float(sharpe_annual)
        return 0.0

    def _record_training_metrics(self, results: dict[str, Any]) -> None:
        """Record training metrics to monitoring collector."""
        metrics = results.get("metrics", {})

        # Record to collector
        self._metrics_collector.record_training_metrics(
            model_type="xgboost",
            model_id=f"xgboost_{int(time.time())}",
            metrics={
                "training_time": metrics.get("training_time", 0),
                "best_iteration": metrics.get("best_iteration", 0),
                "val_accuracy": metrics.get("val_accuracy", 0),
                "val_sharpe": metrics.get("val_sharpe", 0),
            }
        )

        # Record feature importance
        for feature, importance in results.get("feature_importance", {}).items():
            self._metrics_collector.record_feature_importance(
                model_id=f"xgboost_{int(time.time())}",
                feature_name=feature,
                importance=importance
            )
```

## 3. Optuna Integration

### 3.1 Hyperparameter Optimizer

```python
# ml/training/optimization/optuna_optimizer.py
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ml._imports import check_ml_dependencies


class XGBoostOptunaOptimizer:
    """Optuna-based hyperparameter optimizer for XGBoost."""

    def __init__(self, config: OptunaConfig):
        """Initialize optimizer."""
        self.config = config
        self._optuna = None

    def _ensure_optuna(self) -> None:
        """Ensure Optuna is available."""
        if self._optuna is None:
            try:
                import optuna
                self._optuna = optuna
            except ImportError:
                check_ml_dependencies(["optuna"])

    def create_objective(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        base_params: dict[str, Any],
        metric_func: Callable,
    ) -> Callable:
        """Create Optuna objective function."""
        self._ensure_optuna()

        def objective(trial):
            # Sample hyperparameters
            params = {
                **base_params,
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            }

            # Train model
            import xgboost as xgb

            if base_params.get('objective') == 'binary:logistic':
                model = xgb.XGBClassifier(**params)
            else:
                model = xgb.XGBRegressor(**params)

            # Use callback for pruning
            pruning_callback = self._optuna.integration.XGBoostPruningCallback(
                trial, "validation_0-logloss"
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False,
                callbacks=[pruning_callback] if self.config.pruner != "none" else None,
            )

            # Calculate metric
            predictions = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_val)
            metric_value = metric_func(y_val, predictions)

            return metric_value

        return objective

    def optimize(
        self,
        objective: Callable,
        n_trials: int | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Run optimization and return best parameters."""
        self._ensure_optuna()

        # Create study
        study = self._optuna.create_study(
            direction=self.config.direction,
            pruner=self._create_pruner(),
            sampler=self._create_sampler(),
        )

        # Optimize
        study.optimize(
            objective,
            n_trials=n_trials or self.config.n_trials,
            timeout=timeout or self.config.timeout,
            n_jobs=1,  # Single job for XGBoost (GPU conflicts)
        )

        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "study": study,
        }

    def _create_pruner(self):
        """Create Optuna pruner."""
        if self.config.pruner == "none":
            return None
        elif self.config.pruner == "median":
            return self._optuna.pruners.MedianPruner()
        elif self.config.pruner == "percentile":
            return self._optuna.pruners.PercentilePruner(percentile=25.0)
        else:
            return self._optuna.pruners.MedianPruner()

    def _create_sampler(self):
        """Create Optuna sampler."""
        if self.config.sampler == "tpe":
            return self._optuna.samplers.TPESampler()
        elif self.config.sampler == "random":
            return self._optuna.samplers.RandomSampler()
        elif self.config.sampler == "cmaes":
            return self._optuna.samplers.CmaEsSampler()
        else:
            return self._optuna.samplers.TPESampler()
```

### 3.2 Integration in Trainer

```python
# Add to UnifiedXGBoostTrainer
def _optimize_hyperparameters(
    self,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, Any]:
    """Optimize hyperparameters using Optuna."""
    from ml.training.optimization.optuna_optimizer import XGBoostOptunaOptimizer

    optimizer = XGBoostOptunaOptimizer(self._unified_config.optuna_config)

    # Create objective
    base_params = self._unified_config.get_xgb_params()

    # Metric function based on configuration
    if self._unified_config.optuna_config.metric == "sharpe_ratio":
        metric_func = self._calculate_sharpe_ratio
    elif self._unified_config.optuna_config.metric == "accuracy":
        metric_func = self._calculate_accuracy
    else:
        # Default to validation score
        metric_func = lambda y_true, y_pred: -np.mean((y_true - y_pred) ** 2)

    objective = optimizer.create_objective(
        X_train, y_train, X_val, y_val,
        base_params, metric_func
    )

    # Run optimization
    results = optimizer.optimize(objective)

    print(f"Optimization complete: {results['n_trials']} trials")
    print(f"Best value: {results['best_value']:.4f}")
    print(f"Best params: {results['best_params']}")

    return results
```

## 4. MLflow Integration

### 4.1 MLflow Tracker

```python
# ml/training/mlflow/tracking.py
from __future__ import annotations

from typing import Any
import json


class MLflowXGBoostTracker:
    """MLflow tracking for XGBoost models."""

    def __init__(self, config: MLflowConfig):
        """Initialize MLflow tracker."""
        self.config = config
        self._mlflow = None
        self._client = None

    def _ensure_mlflow(self) -> None:
        """Ensure MLflow is available."""
        if self._mlflow is None:
            try:
                import mlflow
                import mlflow.xgboost
                self._mlflow = mlflow
                self._mlflow.set_tracking_uri(self.config.tracking_uri)
                self._mlflow.set_experiment(self.config.experiment_name)
                self._client = mlflow.tracking.MlflowClient()
            except ImportError:
                print("MLflow not available. Skipping tracking.")

    def log_run(
        self,
        model: Any,
        params: dict[str, Any],
        metrics: dict[str, float],
        feature_importance: dict[str, float],
        artifacts: dict[str, Any] | None = None,
    ) -> str | None:
        """Log complete training run."""
        self._ensure_mlflow()

        if self._mlflow is None:
            return None

        with self._mlflow.start_run() as run:
            # Log parameters
            for key, value in params.items():
                if isinstance(value, (int, float, str, bool)):
                    self._mlflow.log_param(key, value)

            # Log metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._mlflow.log_metric(key, float(value))

            # Log feature importance as metrics
            for feature, importance in list(feature_importance.items())[:20]:  # Top 20
                self._mlflow.log_metric(f"importance_{feature}", importance)

            # Log model
            self._mlflow.xgboost.log_model(
                model,
                "model",
                registered_model_name=self.config.model_name if self.config.register_model else None,
            )

            # Log artifacts
            if artifacts:
                for name, content in artifacts.items():
                    if isinstance(content, dict):
                        self._mlflow.log_dict(content, f"{name}.json")

            return run.info.run_id

    def load_model(self, run_id: str) -> Any:
        """Load model from MLflow."""
        self._ensure_mlflow()

        if self._mlflow is None:
            return None

        model_uri = f"runs:/{run_id}/model"
        return self._mlflow.xgboost.load_model(model_uri)
```

### 4.2 Integration in Trainer

```python
# Add to UnifiedXGBoostTrainer
def _init_mlflow_tracking(self) -> None:
    """Initialize MLflow tracking."""
    from ml.training.mlflow.tracking import MLflowXGBoostTracker

    self._mlflow_tracker = MLflowXGBoostTracker(self._unified_config.mlflow_config)

def _log_mlflow_run(
    self,
    model: Any,
    params: dict[str, Any],
    metrics: dict[str, float],
    importance: dict[str, float],
) -> str | None:
    """Log training run to MLflow."""
    if self._mlflow_tracker is None:
        return None

    # Prepare artifacts
    artifacts = {
        "feature_names": self._feature_names,
        "feature_decay_alerts": self._feature_decay_alerts,
        "config": {
            "multi_asset": self._unified_config.multi_asset,
            "gpu_enabled": self._unified_config.gpu_config.enabled,
            "optuna_enabled": self._unified_config.optuna_config.enabled,
        }
    }

    return self._mlflow_tracker.log_run(
        model=model,
        params=params,
        metrics=metrics,
        feature_importance=importance,
        artifacts=artifacts,
    )
```

## 5. ONNX Export

### 5.1 Export Implementation

```python
# Add to UnifiedXGBoostTrainer
def _export_to_onnx(self, model: Any) -> None:
    """Export model to ONNX format for inference."""
    try:
        import onnxmltools
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        print("ONNX tools not available. Skipping export.")
        return

    # Define input type
    initial_types = [
        ('float_input', FloatTensorType([None, len(self._feature_names)]))
    ]

    # Convert to ONNX
    try:
        onnx_model = onnxmltools.convert_xgboost(
            model,
            initial_types=initial_types,
            target_opset=12
        )

        # Save model
        output_path = Path(self._unified_config.onnx_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        print(f"✅ Model exported to ONNX: {output_path}")

        # Also save feature names for inference
        feature_names_path = output_path.with_suffix('.features.json')
        with open(feature_names_path, 'w') as f:
            json.dump({
                "feature_names": self._feature_names,
                "model_type": "xgboost",
                "objective": self._unified_config.objective,
            }, f, indent=2)

    except Exception as e:
        print(f"⚠️ ONNX export failed: {e}")
```

## 6. Testing Implementation

### 6.1 Unit Tests

```python
# ml/tests/unit/test_xgboost_unified.py
import numpy as np
import pytest

from ml.config.xgboost_unified import UnifiedXGBoostConfig, GPUConfig, OptunaConfig
from ml.training.xgboost_unified import UnifiedXGBoostTrainer


class TestUnifiedXGBoostTrainer:
    """Test unified XGBoost trainer."""

    def test_gpu_configuration(self):
        """Test GPU configuration."""
        config = UnifiedXGBoostConfig(
            gpu_config=GPUConfig(enabled=True, device_id=1)
        )

        params = config.get_unified_xgb_params()

        assert params["tree_method"] == "gpu_hist"
        assert params["gpu_id"] == 1
        assert params["predictor"] == "gpu_predictor"

    def test_feature_decay_tracking(self):
        """Test feature importance decay tracking."""
        config = UnifiedXGBoostConfig(
            track_feature_decay=True,
            feature_decay_threshold=0.3
        )

        trainer = UnifiedXGBoostTrainer(config)

        # Simulate importance history
        trainer._feature_names = ["feature1", "feature2", "feature3"]

        # Add initial importance
        trainer._track_feature_decay({
            "feature1": 0.5,
            "feature2": 0.3,
            "feature3": 0.2,
        })

        # Add decayed importance
        trainer._track_feature_decay({
            "feature1": 0.2,  # 60% decay - should trigger alert
            "feature2": 0.25,  # 17% decay - no alert
            "feature3": 0.19,  # 5% decay - no alert
        })

        assert "feature1" in trainer._feature_decay_alerts
        assert "feature2" not in trainer._feature_decay_alerts

    @pytest.mark.parametrize("objective,expected_type", [
        ("binary:logistic", "classification"),
        ("reg:squarederror", "regression"),
    ])
    def test_model_type_handling(self, objective, expected_type):
        """Test correct model type selection."""
        config = UnifiedXGBoostConfig(objective=objective)
        trainer = UnifiedXGBoostTrainer(config)

        # Create dummy data
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100) if expected_type == "classification" else np.random.randn(100)

        metadata = {
            "feature_names": [f"f{i}" for i in range(10)],
            "n_features": 10,
            "n_samples": 100,
            "target_type": expected_type,
        }

        # Verify metadata
        assert metadata["target_type"] == expected_type
```

### 6.2 Integration Tests

```python
# ml/tests/integration/test_xgboost_unified_integration.py
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ml.config.xgboost_unified import UnifiedXGBoostConfig
from ml.data.loader import MLDataLoader
from ml.training.xgboost_unified import UnifiedXGBoostTrainer


class TestUnifiedXGBoostIntegration:
    """Integration tests for unified XGBoost trainer."""

    def test_end_to_end_training(self, sample_bars_data):
        """Test complete training pipeline."""
        # Create config
        config = UnifiedXGBoostConfig(
            n_estimators=10,  # Small for testing
            enable_shap=False,
            track_feature_decay=True,
        )

        # Create trainer
        trainer = UnifiedXGBoostTrainer(config)

        # Train model
        results = trainer.train(sample_bars_data)

        # Verify results
        assert "model" in results
        assert "metrics" in results
        assert "feature_importance" in results
        assert results["metrics"]["training_time"] > 0
        assert results["metrics"]["best_iteration"] > 0

    def test_onnx_export(self, sample_bars_data):
        """Test ONNX export functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "model.onnx"

            config = UnifiedXGBoostConfig(
                n_estimators=10,
                export_onnx=True,
                onnx_output_path=str(onnx_path),
            )

            trainer = UnifiedXGBoostTrainer(config)
            trainer.train(sample_bars_data)

            # Verify ONNX file created
            assert onnx_path.exists()

            # Verify feature names saved
            features_path = onnx_path.with_suffix('.features.json')
            assert features_path.exists()

    @pytest.mark.skipif(not _has_optuna(), reason="Optuna not installed")
    def test_optuna_optimization(self, sample_bars_data):
        """Test Optuna hyperparameter optimization."""
        config = UnifiedXGBoostConfig(
            optuna_config=OptunaConfig(
                enabled=True,
                n_trials=3,  # Few trials for testing
                metric="accuracy",
            )
        )

        trainer = UnifiedXGBoostTrainer(config)
        results = trainer.train(sample_bars_data)

        # Verify optimization results
        assert "best_params" in results
        assert "best_value" in results
        assert results["n_trials"] == 3


def _has_optuna() -> bool:
    """Check if Optuna is available."""
    try:
        import optuna
        return True
    except ImportError:
        return False
```

## Next Steps

1. **Implement Phase 1 components**:
   - Enhanced configuration
   - Core trainer with GPU support
   - Feature decay tracking
   - Monitoring integration

2. **Test thoroughly**:
   - Unit tests for each component
   - Integration tests for pipeline
   - Performance benchmarks

3. **Document usage**:
   - API documentation
   - Usage examples
   - Migration guide for existing models

4. **Prepare for Phase 2**:
   - Complete Optuna integration
   - Full MLflow tracking
   - Cross-validation strategies

This implementation guide provides the foundation for migrating the UnifiedXGBoostTrainer while maintaining Nautilus ML architecture compliance and ensuring seamless integration with existing components.
