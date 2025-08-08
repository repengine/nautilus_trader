"""
Neural Forecast trainer supporting time series transformer models with Optuna and GPU
support.
"""

import pickle
import urllib.error
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import requests.exceptions
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error


warnings.filterwarnings("ignore")

from ..config.settings import Settings
from ..data.unified_loader import UnifiedNautilusDataLoader
from ..features.feature_engineering import FeatureConfig
from ..features.feature_engineering import FeatureEngineerV2

# Moved to conditional import to avoid circular dependency
from ..resource_management.trainer_mixin import ResourceManagedTrainerMixin
from ..utils.dataframe_converter import DataFrameConverter
from ..utils.mlflow_utils import MLflowManager
from .base_trainer import BaseTrainer


# Try to import Neural Forecast libraries
try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATS  # N-BEATS
    from neuralforecast.models import NHITS  # N-HiTS
    from neuralforecast.models import TFT  # Temporal Fusion Transformer
    from neuralforecast.models import Autoformer  # Autoformer
    from neuralforecast.models import DLinear  # DLinear
    from neuralforecast.models import Informer  # Informer
    from neuralforecast.models import NLinear  # NLinear
    from neuralforecast.models import PatchTST  # PatchTST
    from neuralforecast.models import TimesNet  # TimesNet
    from neuralforecast.models import TSMixer  # TSMixer

    NEURALFORECAST_AVAILABLE = True
except ImportError:
    NEURALFORECAST_AVAILABLE = False
    print("Warning: NeuralForecast not available. Install with: pip install neuralforecast")


class NeuralForecastTrainer(ResourceManagedTrainerMixin, BaseTrainer):
    """
    Neural Forecast trainer supporting multiple transformer models.

    Supports:
    - TFT (Temporal Fusion Transformer)
    - Informer
    - Autoformer
    - PatchTST
    - NBEATS
    - NHITS
    - TimesNet
    - TSMixer
    - DLinear/NLinear

    Features:
    - Multi-horizon predictions
    - Time series specific features (lags, date features)
    - GPU acceleration with PyTorch
    - Optuna hyperparameter optimization
    - MLflow experiment tracking
    - Proper train/validation/test splits for time series

    """

    def __init__(self, config: dict[str, Any], settings: Settings | None = None):
        self.mlflow_manager = MLflowManager(settings)
        """
        Initialize Neural Forecast trainer.

        Args:
            config: Configuration dictionary containing:
                - model_type: One of ['TFT', 'Informer', 'Autoformer', 'PatchTST', etc]
                - forecast_horizon: Number of steps to forecast
                - input_size: Input sequence length
                - feature_config: FeatureConfig object
                - use_time_features: Whether to add time-based features
                - use_static_features: Whether to use static features
                - freq: Frequency of the data ('H', 'D', etc)
                - gpu_id: GPU device ID (default 0)
                - enable_early_stopping: Whether to use early stopping
            settings: Nautilus ML settings object

        """
        super().__init__(config, settings)

        if not NEURALFORECAST_AVAILABLE:
            raise ImportError(
                "NeuralForecast is required. Install with: pip install neuralforecast",
            )

        self.model = None
        self.scaler = None
        self.feature_names: list[str] | None = None
        self.feature_engineer = FeatureEngineerV2(config.get("feature_config", FeatureConfig()))

        # Neural Forecast specific
        self.model_type = config.get("model_type", "TFT")
        self.forecast_horizon = config.get("forecast_horizon", 24)  # Default 24 hours
        self.input_size = config.get("input_size", 168)  # Default 1 week for hourly
        self.freq = config.get("freq", "H")  # Hourly by default
        self.use_time_features = config.get("use_time_features", True)
        self.use_static_features = config.get("use_static_features", False)
        self.gpu_id = config.get("gpu_id", 0)
        self.enable_early_stopping = config.get("enable_early_stopping", True)

        # Model registry
        self.model_registry = {
            "TFT": TFT,
            "Informer": Informer,
            "Autoformer": Autoformer,
            "PatchTST": PatchTST,
            "NBEATS": NBEATS,
            "NHITS": NHITS,
            "TimesNet": TimesNet,
            "TSMixer": TSMixer,
            "DLinear": DLinear,
            "NLinear": NLinear,
        }

    def prepare_data(
        self,
        data: pl.DataFrame,
        target_col: str = "close",
        fit_scaler: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """
        Prepare data for Neural Forecast models.

        Args:
            data: Polars DataFrame with OHLCV data and timestamp
            target_col: Name of target column (default 'close')
            fit_scaler: Whether to fit and store a new scaler (training only)

        Returns:
            X: Features DataFrame in NeuralForecast format
            y: Target values (same as X for NeuralForecast)
            feature_names: List of feature names

        """
        # Ensure we have required columns
        required_cols = ["timestamp", target_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Calculate technical features using feature engineer
        features_df, scaler = self.feature_engineer.calculate_features_batch(
            data,
            fit_scaler=fit_scaler,
        )
        if fit_scaler:
            self.scaler = scaler
        feature_names: list[str] = self.feature_engineer.get_feature_names()
        self.feature_names = feature_names

        # Convert to pandas for NeuralForecast
        # NeuralForecast expects specific format: unique_id, ds, y, [exogenous features]
        df = DataFrameConverter.polars_to_pandas(data)
        features_pd = DataFrameConverter.polars_to_pandas(features_df)

        # Create NeuralForecast formatted DataFrame
        nf_df = pd.DataFrame(
            {
                "unique_id": "series_1",  # Single series for now
                "ds": pd.to_datetime(df["timestamp"]),
                "y": df[target_col].values,
            },
        )

        # Add exogenous features
        for i, feature in enumerate(feature_names):
            nf_df[feature] = features_pd.iloc[:, i]

        # Sort by timestamp
        nf_df = nf_df.sort_values("ds").reset_index(drop=True)

        # For NeuralForecast, X and y are the same DataFrame
        return nf_df, nf_df, feature_names

    def _create_model(self, params: dict[str, Any]) -> Any:
        """
        Create a Neural Forecast model with given parameters.

        Args:
            params: Model parameters

        Returns:
            NeuralForecast model instance

        """
        model_class = self.model_registry.get(self.model_type)
        if not model_class:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Common parameters for all models
        common_params = {
            "h": self.forecast_horizon,
            "input_size": self.input_size,
            "random_seed": params.get("random_seed", 42),
            "batch_size": params.get("batch_size", 32),
            "learning_rate": params.get("learning_rate", 1e-3),
            "max_steps": params.get("max_steps", 1000),
            "early_stop_patience_steps": (
                params.get("early_stop_patience_steps", 50) if self.enable_early_stopping else -1
            ),
            "val_check_steps": params.get("val_check_steps", 100),
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        }

        # Model-specific parameters
        if self.model_type == "TFT":
            model_params = {
                **common_params,
                "hidden_size": params.get("hidden_size", 128),
                "n_head": params.get("n_head", 4),
                "dropout": params.get("dropout", 0.1),
                "attn_dropout": params.get("attention_dropout", 0.0),
                "hidden_continuous_size": params.get("hidden_continuous_size", 8),
                "hist_exog_list": self.feature_names if self.feature_names else None,
            }
        elif self.model_type in ["Informer", "Autoformer"]:
            model_params = {
                **common_params,
                "hidden_size": params.get("hidden_size", 128),
                "n_head": params.get("n_head", 8),
                "dropout": params.get("dropout", 0.1),
                "conv_hidden_size": params.get("conv_hidden_size", 32),
                "encoder_layers": params.get("encoder_layers", 2),
                "decoder_layers": params.get("decoder_layers", 1),
                "hist_exog_list": self.feature_names if self.feature_names else None,
            }
        elif self.model_type == "PatchTST":
            model_params = {
                **common_params,
                "patch_len": params.get("patch_len", 16),
                "stride": params.get("stride", 8),
                "hidden_size": params.get("hidden_size", 128),
                "n_heads": params.get("n_head", 16),
                "dropout": params.get("dropout", 0.1),
                "hist_exog_list": self.feature_names if self.feature_names else None,
            }
        elif self.model_type in ["NBEATS", "NHITS"]:
            model_params = {
                **common_params,
                "n_blocks": params.get("n_blocks", [1, 1, 1]),
                "mlp_units": params.get("mlp_units", [[512, 512], [512, 512], [512, 512]]),
                "dropout_prob_theta": params.get("dropout", 0.1),
                "hist_exog_list": self.feature_names if self.feature_names else None,
            }
        elif self.model_type == "TimesNet":
            model_params = {
                **common_params,
                "hidden_size": params.get("hidden_size", 64),
                "conv_hidden_size": params.get("conv_hidden_size", 64),
                "num_kernels": params.get("num_kernels", 6),
                "top_k": params.get("top_k", 5),
                "dropout": params.get("dropout", 0.1),
                "hist_exog_list": self.feature_names if self.feature_names else None,
            }
        elif self.model_type == "TSMixer":
            model_params = {
                **common_params,
                "n_block": params.get("n_blocks", 2),
                "hidden_size": params.get("hidden_size", 128),
                "dropout": params.get("dropout", 0.1),
                "ff_dim": params.get("ff_dim", 256),
                "hist_exog_list": self.feature_names if self.feature_names else None,
            }
        elif self.model_type in ["DLinear", "NLinear"]:
            model_params = {
                **common_params,
                "hist_exog_list": self.feature_names if self.feature_names else None,
            }
        else:
            model_params = common_params

        # Add GPU device if available
        if torch.cuda.is_available() and "accelerator" in model_params:
            model_params["devices"] = [self.gpu_id]

        return model_class(**model_params)

    def train(
        self,
        train_data: pl.DataFrame,
        val_data: pl.DataFrame,
        optimize_hyperparams: bool = True,
    ) -> dict[str, Any]:
        """
        Train Neural Forecast model with optional hyperparameter optimization.

        Args:
            train_data: Training data DataFrame
            val_data: Validation data DataFrame
            optimize_hyperparams: Whether to run Optuna optimization

        Returns:
            Dictionary with model, metrics, and metadata

        """
        # Set up MLflow experiment
        experiment_name = (
            f"{self.settings.mlflow.experiment_name}_neural_forecast_{self.model_type}"
        )
        self.mlflow_manager.set_experiment(experiment_name)

        with self.mlflow_manager.start_run(
            run_name=f"neural_forecast_{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags={"model_type": "neural_forecast", "forecast_model": self.model_type},
        ):
            # Log configuration
            self.mlflow_manager.log_params(
                {
                    "model_type": self.model_type,
                    "forecast_horizon": self.forecast_horizon,
                    "input_size": self.input_size,
                    "freq": self.freq,
                    "use_time_features": self.use_time_features,
                    "optimize_hyperparams": optimize_hyperparams,
                    "gpu_available": torch.cuda.is_available(),
                },
                prefix="training",
            )

            # Log feature config separately
            if "feature_config" in self.config:
                self.mlflow_manager.log_params(
                    self.config["feature_config"],
                    prefix="feature_config",
                )

            if optimize_hyperparams:
                results = self._optimize_and_train(train_data, val_data)
            else:
                results = self._train_with_params(train_data, val_data, self.config)

            # Log results
            self.mlflow_manager.log_metrics(results["metrics"])
            self._save_artifacts(results)

            return results

    def _optimize_and_train(
        self,
        train_data: pl.DataFrame,
        val_data: pl.DataFrame,
    ) -> dict[str, Any]:
        """
        Run Optuna optimization and train with best parameters.

        Args:
            train_data: Training data
            val_data: Validation data

        Returns:
            Results dictionary with optimized model and metrics

        """
        # Import here to avoid circular import
        from ..optimization import HyperparameterOptimizer
        from ..optimization import OptimizerConfig

        optimizer_config = OptimizerConfig(
            n_trials=self.config.get("n_trials", 50),
            n_jobs=self.config.get(
                "n_jobs",
                1,
            ),  # NeuralForecast doesn't support parallel trials well
            metric_name=self.config.get("metric_name", "val_mae"),
            direction="minimize",
        )

        optimizer = HyperparameterOptimizer(optimizer_config, self.settings)

        # Set model type in study attrs
        optimizer.study_attrs = {"model_type": self.model_type}

        # Run optimization
        opt_results = optimizer.optimize_model(
            model_type="neural_forecast",
            train_data=train_data,
            val_data=val_data,
            trainer_class=NeuralForecastTrainer,
            base_config=self.config,
        )

        print("\nOptimization complete!")
        print(f"Best parameters: {opt_results.best_params}")
        print(f"Best {optimizer_config.metric_name}: {opt_results.best_value:.4f}")

        # Train final model with best parameters
        best_config = {**self.config, **opt_results.best_params}
        final_results = self._train_with_params(train_data, val_data, best_config)

        # Add optimization info
        final_results["optimization"] = {
            "best_params": opt_results.best_params,
            "best_value": opt_results.best_value,
            "n_trials": opt_results.n_trials,
            "param_importance": opt_results.param_importance,
        }

        return final_results

    def _train_with_params(
        self,
        train_data: pl.DataFrame,
        val_data: pl.DataFrame,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Train model with specific parameters.

        Args:
            train_data: Training data
            val_data: Validation data
            params: Model parameters

        Returns:
            Results dictionary with model, metrics, and artifacts

        """
        # Prepare data
        print(f"Preparing data for {self.model_type}...")
        train_df, _, feature_names = self.prepare_data(train_data, fit_scaler=True)
        val_df, _, _ = self.prepare_data(val_data, fit_scaler=False)

        # Apply stored training scaler to validation data
        if self.scaler is not None:
            val_df[feature_names] = self.scaler.transform(val_df[feature_names])

        # Create model
        print(f"Creating {self.model_type} model...")
        model = self._create_model(params)

        # Create NeuralForecast object
        nf = NeuralForecast(models=[model], freq=self.freq)

        # Fit model
        print(f"Training {self.model_type} model...")
        nf.fit(df=train_df, val_size=len(val_df))

        # Generate predictions on validation set
        print("Generating predictions...")
        forecasts = nf.predict(df=train_df)

        # Calculate metrics
        metrics = self._calculate_metrics(val_df, forecasts)

        # Feature importance (if available)
        feature_importance = self._get_feature_importance(model, feature_names)

        return {
            "model": nf,
            "metrics": metrics,
            "params": params,
            "scaler": self.scaler,
            "feature_names": feature_names,
            "feature_importance": feature_importance,
            "model_type": self.model_type,
            "forecast_horizon": self.forecast_horizon,
            "input_size": self.input_size,
        }

    def _calculate_metrics(
        self,
        actual_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Calculate forecasting metrics.

        Args:
            actual_df: Actual values DataFrame
            forecast_df: Forecast values DataFrame

        Returns:
            Dictionary of metrics

        """
        # Align actual and forecast data
        merged = actual_df.merge(
            forecast_df,
            on=["unique_id", "ds"],
            suffixes=("_actual", "_forecast"),
        )

        if len(merged) == 0:
            print("Warning: No overlapping timestamps for evaluation")
            return {"val_mse": np.nan, "val_mae": np.nan, "val_mape": np.nan, "val_rmse": np.nan}

        # Get model column name (e.g., 'TFT', 'Informer', etc)
        model_col = [col for col in forecast_df.columns if col not in ["unique_id", "ds"]][0]

        y_true = merged["y"].values
        y_pred = merged[model_col].values

        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        # MAPE with protection against division by zero
        mask = y_true != 0
        if mask.sum() > 0:
            mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask])
        else:
            mape = np.nan

        # Directional accuracy
        if len(y_true) > 1:
            direction_true = np.diff(y_true) > 0
            direction_pred = np.diff(y_pred) > 0
            directional_accuracy = (direction_true == direction_pred).mean()
        else:
            directional_accuracy = np.nan

        # Per-horizon metrics
        horizon_metrics = {}
        for h in range(1, min(self.forecast_horizon + 1, len(y_true))):
            if h < len(y_true):
                horizon_metrics[f"mae_h{h}"] = mean_absolute_error(y_true[:h], y_pred[:h])

        metrics = {
            "val_mse": float(mse),
            "val_mae": float(mae),
            "val_rmse": float(rmse),
            "val_mape": float(mape) if not np.isnan(mape) else 0.0,
            "directional_accuracy": (
                float(directional_accuracy) if not np.isnan(directional_accuracy) else 0.0
            ),
            **horizon_metrics,
        }

        return metrics

    def _get_feature_importance(self, model: Any, feature_names: list[str]) -> pd.DataFrame | None:
        """
        Extract feature importance if available for the model.

        Args:
            model: Trained model
            feature_names: List of feature names

        Returns:
            DataFrame with feature importance or None

        """
        # Most transformer models don't have direct feature importance
        # This is a placeholder for models that might support it
        # For transformers, we might use attention weights or other interpretability methods

        # For now, return None as most models don't support this directly
        return None

    def _save_artifacts(self, results: dict[str, Any]):
        """
        Save model artifacts to MLflow.

        Args:
            results: Results dictionary from training

        """
        # Log model
        model_name = (
            f"neural_forecast_{self.model_type}_{self.config.get('strategy_name', 'default')}"
        )

        # For NeuralForecast, we need to save the entire NeuralForecast object
        self.mlflow_manager.log_model(
            model=results["model"].models[0],  # Log the underlying PyTorch model
            artifact_path="model",
            model_type="pytorch",
            registered_model_name=model_name,
        )

        # Log scaler
        scaler_path = Path("scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(results["scaler"], f)
        self.mlflow_manager.log_artifact(scaler_path, description="Feature scaler")
        scaler_path.unlink()

        # Log feature config
        config_path = Path("feature_config.pkl")
        with open(config_path, "wb") as f:
            pickle.dump(self.feature_engineer.config, f)
        self.mlflow_manager.log_artifact(
            config_path,
            description="Feature engineering configuration",
        )
        config_path.unlink()

        # Log model config
        model_config = {
            "model_type": self.model_type,
            "forecast_horizon": self.forecast_horizon,
            "input_size": self.input_size,
            "freq": self.freq,
            "feature_names": results["feature_names"],
            "params": results["params"],
        }
        self.mlflow_manager.log_dict_as_json(model_config, "model_config", artifact_path="config")

    def save_model_bundle(
        self,
        results: dict[str, Any],
        strategy_name: str,
        version: str | None = None,
    ) -> Path:
        """
        Save complete model bundle for production deployment.

        Args:
            results: Results dictionary from training
            strategy_name: Name for the strategy
            version: Version string (auto-generated if None)

        Returns:
            Path to saved model bundle

        """
        if version is None:
            version = datetime.now().strftime("v%Y%m%d_%H%M%S")

        # Create production wrapper
        model_data = {
            "model": results["model"],
            "scaler": results["scaler"],
            "feature_config": self.feature_engineer.config,
            "feature_names": results["feature_names"],
            "version": version,
            "strategy_name": strategy_name,
            "test_metrics": results["metrics"],
            "trained_at": datetime.now().isoformat(),
            "model_params": results["params"],
            "model_type": self.model_type,
            "forecast_horizon": self.forecast_horizon,
            "input_size": self.input_size,
            "freq": self.freq,
            "framework": "neural_forecast",
        }

        # Save versioned model
        output_path = Path(self.settings.data.processed_data_path) / "models"
        output_path.mkdir(exist_ok=True, parents=True)

        model_filename = f"{strategy_name}_{version}.pkl"
        model_path = output_path / model_filename

        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        # Also save as latest
        latest_path = output_path / f"{strategy_name}_latest.pkl"
        with open(latest_path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"\nModel saved to: {model_path}")
        print(f"Also saved as: {latest_path}")

        return model_path


# Convenience functions for common use cases


def train_tft_model(
    instrument: str,
    data_path: Path,
    forecast_horizon: int = 24,
    input_size: int = 168,
    optimize: bool = True,
    n_trials: int = 50,
    enable_gpu: bool = True,
    catalog_path: Path | None = None,
) -> dict[str, Any]:
    """
    Train Temporal Fusion Transformer for time series forecasting.

    Args:
        instrument: Trading instrument symbol
        data_path: Path to data directory
        forecast_horizon: Number of steps to forecast
        input_size: Length of input sequence
        optimize: Whether to run hyperparameter optimization
        n_trials: Number of Optuna trials
        enable_gpu: Whether to use GPU acceleration
        catalog_path: Optional path to Nautilus catalog

    Returns:
        Training results dictionary

    """
    # Initialize data loader
    loader = UnifiedNautilusDataLoader(catalog_path=catalog_path)

    # Load data using unified loader
    try:
        if catalog_path and catalog_path.exists():
            data = loader.load_bars(instrument, source="catalog")
        else:
            parquet_file = data_path / f"{instrument}_hourly.parquet"
            if not parquet_file.exists():
                raise FileNotFoundError(f"Data file not found: {parquet_file}")
            # Try loading bars directly if catalog doesn't exist
            data = loader.load_bars(instrument, bar_aggregation="1-HOUR")
    except (
        FileNotFoundError,
        OSError,
        requests.exceptions.RequestException,
        urllib.error.URLError,
        ConnectionError,
        ValueError,
        TypeError,
        ZeroDivisionError,
        FloatingPointError,
        RuntimeError,
    ) as e:
        print(f"Error loading data: {e}")
        raise

    # Ensure timestamp column
    if "timestamp" not in data.columns and "index" in data.columns:
        data = data.rename({"index": "timestamp"})

    # Split data
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # Configure trainer
    config = {
        "instrument": instrument,
        "model_type": "TFT",
        "forecast_horizon": forecast_horizon,
        "input_size": input_size,
        "n_trials": n_trials,
        "metric_name": "val_mae",
        "feature_config": FeatureConfig(),
        "strategy_name": f"{instrument}_tft_forecast",
        "freq": "H",  # Hourly
        "use_time_features": True,
        "gpu_id": 0 if enable_gpu else None,
    }

    # Train
    trainer = NeuralForecastTrainer(config)
    results = trainer.train(train_data, val_data, optimize_hyperparams=optimize)

    # Save model bundle
    trainer.save_model_bundle(results, config["strategy_name"])

    return results


def train_multi_model_ensemble(
    instrument: str,
    data_path: Path,
    models: list[str] | None = None,
    forecast_horizon: int = 24,
    optimize: bool = False,
    n_trials: int = 20,
    catalog_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Train multiple Neural Forecast models for ensemble.

    Args:
        instrument: Trading instrument symbol
        data_path: Path to data directory
        models: List of model types to train
        forecast_horizon: Number of steps to forecast
        optimize: Whether to run hyperparameter optimization
        n_trials: Number of Optuna trials per model
        catalog_path: Optional path to Nautilus catalog

    Returns:
        Dictionary mapping model name to results

    """
    # Initialize data loader
    loader = UnifiedNautilusDataLoader(catalog_path=catalog_path)

    # Load data
    if models is None:
        models = ["TFT", "Informer", "PatchTST"]

    try:
        if catalog_path and catalog_path.exists():
            data = loader.load_bars(instrument, source="catalog")
        else:
            parquet_file = data_path / f"{instrument}_hourly.parquet"
            if not parquet_file.exists():
                raise FileNotFoundError(f"Data file not found: {parquet_file}")
            # Try loading bars directly if catalog doesn't exist
            data = loader.load_bars(instrument, bar_aggregation="1-HOUR")
    except (
        FileNotFoundError,
        OSError,
        requests.exceptions.RequestException,
        urllib.error.URLError,
        ConnectionError,
        ValueError,
        TypeError,
        ZeroDivisionError,
        FloatingPointError,
        RuntimeError,
    ) as e:
        print(f"Error loading data: {e}")
        raise

    # Ensure timestamp column
    if "timestamp" not in data.columns and "index" in data.columns:
        data = data.rename({"index": "timestamp"})

    # Split data
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    all_results = {}

    for model_type in models:
        print(f"\n{'='*60}")
        print(f"Training {model_type} model...")
        print(f"{'='*60}")

        # Configure trainer
        config = {
            "instrument": instrument,
            "model_type": model_type,
            "forecast_horizon": forecast_horizon,
            "input_size": 168,  # 1 week for hourly data
            "n_trials": n_trials,
            "metric_name": "val_mae",
            "feature_config": FeatureConfig(),
            "strategy_name": f"{instrument}_{model_type.lower()}_forecast",
            "freq": "H",
            "use_time_features": True,
            "gpu_id": 0,
        }

        try:
            # Train
            trainer = NeuralForecastTrainer(config)
            results = trainer.train(train_data, val_data, optimize_hyperparams=optimize)

            # Save model bundle
            trainer.save_model_bundle(results, config["strategy_name"])

            all_results[model_type] = results

            print(f"\n{model_type} Results:")
            print(f"  MAE: {results['metrics']['val_mae']:.4f}")
            print(f"  RMSE: {results['metrics']['val_rmse']:.4f}")
            print(f"  MAPE: {results['metrics']['val_mape']:.2%}")

        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
            print(f"Error training {model_type}: {e}")
            all_results[model_type] = {"error": str(e)}

    return all_results


def train_lightweight_forecast(
    instrument: str,
    data_path: Path,
    model_type: str = "DLinear",
    forecast_horizon: int = 24,
    catalog_path: Path | None = None,
) -> dict[str, Any]:
    """
    Train lightweight linear models (DLinear/NLinear) for fast inference.

    Args:
        instrument: Trading instrument symbol
        data_path: Path to data directory
        model_type: 'DLinear' or 'NLinear'
        forecast_horizon: Number of steps to forecast
        catalog_path: Optional path to Nautilus catalog

    Returns:
        Training results dictionary

    """
    return train_tft_model(
        instrument=instrument,
        data_path=data_path,
        forecast_horizon=forecast_horizon,
        optimize=False,  # These models have few hyperparameters
        n_trials=0,
        catalog_path=catalog_path,
    )
