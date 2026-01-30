#!/usr/bin/env python3

"""
Builder classes for ML test fixtures.

This module provides builder patterns for creating test objects with sensible defaults
that can be easily customized.

"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from ml.config.actors import MLSignalActorConfig
from ml.config.base import MLActorConfig
from ml.config.base import MLFeatureConfig
from ml.config.base import MLStrategyConfig
from ml.strategies.analytics import AnalyticsConfig as _AnalyticsConfig
from ml.strategies.execution import ExecutionConfig as _ExecutionConfig
from ml.strategies.portfolio import PortfolioConfig as _PortfolioConfig
from ml.strategies.risk import RiskConfig as _RiskConfig
from ml.strategies.sizing import SizingConfig as _SizingConfig
from ml.config.registry import ModelRegistryConfig
from ml.registry import FeatureManifest
from ml.registry import ModelManifest
from ml.registry import StrategyManifest
from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategyStore
from ml.tests.fixtures.model_factory import TestModelFactory
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import ComponentId
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Venue


if TYPE_CHECKING:
    from ml.stores.data_store import DataStore


class MLConfigBuilder:
    """
    Builder for ML configuration objects with sensible defaults.
    """

    @staticmethod
    def actor_config(**overrides: Any) -> MLActorConfig:
        """
        Create MLActorConfig with defaults and optional overrides.

        Parameters
        ----------
        **overrides
            Any configuration fields to override defaults

        Returns
        -------
        MLActorConfig
            Configured ML actor config

        """
        # Create temporary model if not provided
        if "model_path" not in overrides:
            model_path = TestModelFactory.create_onnx_model(n_features=10, n_outputs=1)
            overrides["model_path"] = str(model_path)

        defaults = {
            "model_id": "test_model",
            "bar_type": BarType.from_str("EUR/USD.SIM-1-MINUTE-MID-INTERNAL"),
            "instrument_id": InstrumentId.from_str("EUR/USD.SIM"),
            "batch_size": 1,
            "warm_up_period": 10,
            "prediction_threshold": 0.5,
            "use_dummy_stores": True,  # Use dummy stores for testing
        }

        # Merge defaults with overrides
        config_dict = {**defaults, **overrides}

        return MLActorConfig(**config_dict)

    @staticmethod
    def signal_config(**overrides: Any) -> MLSignalActorConfig:
        """
        Create MLSignalActorConfig with defaults and optional overrides.

        Parameters
        ----------
        **overrides
            Any configuration fields to override defaults

        Returns
        -------
        MLSignalActorConfig
            Configured signal actor config

        """
        # Create temporary model if not provided
        if "model_path" not in overrides:
            model_path = TestModelFactory.create_onnx_model(n_features=10, n_outputs=1)
            overrides["model_path"] = str(model_path)

        # Create feature config if not provided
        if "feature_config" not in overrides:
            overrides["feature_config"] = MLFeatureConfig(
                lookback_window=20,
                indicators={
                    "sma": {"period": 20},
                    "rsi": {"period": 14},
                },
            )

        defaults = {
            "model_id": "test_signal_model",
            "bar_type": BarType.from_str("EUR/USD.SIM-1-MINUTE-MID-INTERNAL"),
            "instrument_id": InstrumentId.from_str("EUR/USD.SIM"),
            "batch_size": 1,
            "warm_up_period": 10,
            "prediction_threshold": 0.5,
            "use_dummy_stores": True,  # Use dummy stores for testing
            "signal_strategy": "threshold",
        }

        config_dict = {**defaults, **overrides}

        return MLSignalActorConfig(**config_dict)

    @staticmethod
    def strategy_config(
        **overrides: Any,
    ) -> MLStrategyConfig:
        """
        Create MLStrategyConfig with defaults and optional overrides.

        Parameters
        ----------
        **overrides
            Any configuration fields to override defaults

        Returns
        -------
        MLStrategyConfig
            Configured trading strategy config

        """
        defaults = {
            "strategy_id": "test_strategy",
            "ml_signal_source": "test_signal_actor",
            "instrument_id": InstrumentId.from_str("EUR/USD.SIM"),
            "position_size_pct": 0.02,
            "min_confidence": 0.6,
            "max_positions": 3,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "execute_trades": False,  # Safe default for testing
            "use_strategy_store": True,
            "persist_all_signals": True,
        }

        # Allow explicit typed sub-configs; fall back to overrides for backward compatibility
        for key in (
            "sizing_config",
            "risk_config",
            "execution_config",
            "portfolio_config",
            "analytics_config",
        ):
            if key in overrides and overrides[key] is None:
                del overrides[key]

        config_dict: dict[str, Any] = {**defaults, **overrides}

        return MLStrategyConfig(**config_dict)

    @staticmethod
    def feature_config(**overrides: Any) -> MLFeatureConfig:
        """
        Create MLFeatureConfig with defaults and optional overrides.
        """
        defaults = {
            "lookback_window": 20,
            "indicators": {
                "sma": {"period": 20},
                "rsi": {"period": 14},
                "bb": {"period": 20, "std_dev": 2},
            },
            "normalize_features": True,
            "fill_missing_with": 0.0,
            "average_volume": 1000000.0,
        }

        config_dict = {**defaults, **overrides}

        return MLFeatureConfig(**config_dict)


class MockBuilder:
    """
    Builder for common mock objects used in tests.
    """

    @staticmethod
    def model_registry(
        model_id: str = "test_model",
        version: str = "1.0.0",
        architecture: str = "xgboost",
        **kwargs: Any,
    ) -> MagicMock:
        """
        Create a fully configured mock model registry.

        Parameters
        ----------
        model_id : str
            Model identifier
        version : str
            Model version
        architecture : str
            Model architecture
        **kwargs
            Additional manifest fields

        Returns
        -------
        MagicMock
            Configured mock registry

        """
        mock_registry = MagicMock()

        from ml.registry.base import ModelRole, DataRequirements
        import time

        # Create model manifest
        manifest_dict = {
            "model_id": model_id,
            "role": kwargs.get("role", ModelRole.INFERENCE),
            "data_requirements": kwargs.get("data_requirements", DataRequirements.L1_ONLY),
            "architecture": architecture,
            "feature_schema": kwargs.get(
                "feature_schema",
                {"feature_1": "float", "feature_2": "float"},
            ),
            "feature_schema_hash": kwargs.get("feature_schema_hash", "abc123"),
            "version": version,
            "created_at": kwargs.get("created_at", time.time()),
            "last_modified": kwargs.get("last_modified", time.time()),
            "performance_metrics": kwargs.get("performance_metrics", {"accuracy": 0.95}),
            "training_config": kwargs.get("training_config", {"n_estimators": 100}),
        }

        mock_model_info = MagicMock()
        mock_model_info.manifest = ModelManifest(**manifest_dict)
        mock_model_info.path = Path(f"/tmp/{model_id}.onnx")
        mock_model_info.metadata = {"mock": True}

        # Configure registry methods
        mock_registry.get_model = MagicMock(return_value=mock_model_info)
        mock_registry.list_models = MagicMock(return_value=[f"{model_id}_v{version}"])
        mock_registry.register_model = MagicMock(return_value=f"{model_id}_v{version}")
        mock_registry.load_model = MagicMock(return_value=mock_model_info)
        mock_registry.exists = MagicMock(return_value=True)

        return mock_registry

    @staticmethod
    def feature_registry(
        feature_set_id: str = "test_features",
        version: str = "1.0.0",
        feature_names: list[str] | None = None,
        **kwargs: Any,
    ) -> MagicMock:
        """
        Create a fully configured mock feature registry.
        """
        mock_registry = MagicMock()

        if feature_names is None:
            feature_names = ["sma_20", "rsi_14", "volume_ratio"]

        from ml.registry.feature_registry import FeatureRole
        from ml.registry.base import DataRequirements
        import time

        manifest_dict = {
            "feature_set_id": feature_set_id,
            "name": kwargs.get("name", f"Mock {feature_set_id}"),
            "version": version,
            "role": kwargs.get("role", FeatureRole.INFERENCE_SUPPORT),
            "data_requirements": kwargs.get("data_requirements", DataRequirements.L1_ONLY),
            "feature_names": feature_names,
            "feature_dtypes": kwargs.get("feature_dtypes", ["float32"] * len(feature_names)),
            "schema_hash": kwargs.get("schema_hash", "def456"),
            "pipeline_signature": kwargs.get("pipeline_signature", "pipeline_sig_123"),
            "pipeline_version": kwargs.get("pipeline_version", "1.0.0"),
            "created_at": kwargs.get("created_at", time.time()),
            "last_modified": kwargs.get("last_modified", time.time()),
        }

        mock_feature_info = MagicMock()
        mock_feature_info.manifest = FeatureManifest(**manifest_dict)

        # Configure registry methods
        mock_registry.get_feature_set = MagicMock(return_value=mock_feature_info)
        mock_registry.list_feature_sets = MagicMock(return_value=[f"{feature_set_id}_v{version}"])
        mock_registry.register_feature_set = MagicMock(return_value=f"{feature_set_id}_v{version}")
        mock_registry.exists = MagicMock(return_value=True)

        return mock_registry

    @staticmethod
    def strategy_registry(
        strategy_id: str = "test_strategy",
        version: str = "1.0.0",
        **kwargs: Any,
    ) -> MagicMock:
        """
        Create a fully configured mock strategy registry.
        """
        from ml.registry.strategy_registry import StrategyType, MarketRegime
        import time

        mock_registry = MagicMock()

        manifest_dict = {
            "strategy_id": strategy_id,
            "strategy_type": kwargs.get("strategy_type", StrategyType.TREND_FOLLOWING),
            "version": version,
            "required_models": kwargs.get("required_models", ["test_model"]),
            "required_features": kwargs.get("required_features", ["test_features"]),
            "suitable_regimes": kwargs.get("suitable_regimes", [MarketRegime.TRENDING_UP]),
            "instrument_types": kwargs.get("instrument_types", ["FX", "CRYPTO"]),
            "timeframe_range": kwargs.get("timeframe_range", ("1m", "1h")),
            "max_position_size": kwargs.get("max_position_size", 0.1),
            "max_leverage": kwargs.get("max_leverage", 2.0),
            "max_drawdown": kwargs.get("max_drawdown", 0.15),
            "stop_loss_type": kwargs.get("stop_loss_type", "fixed"),
            "min_sharpe_ratio": kwargs.get("min_sharpe_ratio", 0.5),
            "min_win_rate": kwargs.get("min_win_rate", 0.4),
            "max_correlation_with_portfolio": kwargs.get("max_correlation_with_portfolio", 0.7),
            "parent_strategy_id": kwargs.get("parent_strategy_id", None),
            "incompatible_strategies": kwargs.get("incompatible_strategies", []),
            "config_schema": kwargs.get("config_schema", {"position_size": "float"}),
            "default_config": kwargs.get("default_config", {"position_size": 0.02}),
            "backtest_metrics": kwargs.get("backtest_metrics", {"sharpe_ratio": 1.5}),
            "live_metrics": kwargs.get("live_metrics", None),
            "created_at": kwargs.get("created_at", time.time()),
            "last_modified": kwargs.get("last_modified", time.time()),
            "author": kwargs.get("author", "test_author"),
            "description": kwargs.get("description", f"Mock strategy {strategy_id}"),
        }

        mock_strategy_info = MagicMock()
        mock_strategy_info.manifest = StrategyManifest(**manifest_dict)

        # Configure registry methods
        mock_registry.get_strategy = MagicMock(return_value=mock_strategy_info)
        mock_registry.list_strategies = MagicMock(return_value=[f"{strategy_id}_v{version}"])
        mock_registry.register_strategy = MagicMock(return_value=f"{strategy_id}_v{version}")
        mock_registry.exists = MagicMock(return_value=True)

        return mock_registry

    @staticmethod
    def all_registries() -> dict[str, MagicMock]:
        """
        Create a complete set of mock registries.

        Returns
        -------
        dict[str, MagicMock]
            Dictionary with all four registry mocks

        """
        return {
            "model_registry": MockBuilder.model_registry(),
            "feature_registry": MockBuilder.feature_registry(),
            "strategy_registry": MockBuilder.strategy_registry(),
            "data_registry": MagicMock(),  # Simple mock for data registry
        }

    @staticmethod
    def store_with_data(
        store_type: str = "feature",
        data: Any = None,
    ) -> MagicMock:
        """
        Create a mock store pre-populated with test data.

        Parameters
        ----------
        store_type : str
            Type of store ('feature', 'model', 'strategy', 'data')
        data : Any
            Data to return from read operations

        Returns
        -------
        MagicMock
            Configured mock store

        """
        mock_store = MagicMock()

        if store_type == "feature":
            if data is None:
                data = {"sma_20": 1.09, "rsi": 55.5, "volume": 12345}
            mock_store.read_features = MagicMock(return_value=data)
            mock_store.write_features = MagicMock(return_value=True)
            mock_store.get_latest_features = MagicMock(return_value=data)

        elif store_type == "model":
            if data is None:
                data = [{"prediction": 0.65, "confidence": 0.8}]
            mock_store.read_predictions = MagicMock(return_value=data)
            mock_store.write_predictions = MagicMock(return_value=True)
            mock_store.get_latest_predictions = MagicMock(return_value=data)

        elif store_type == "strategy":
            if data is None:
                data = [{"signal": 1, "confidence": 0.7}]
            mock_store.read_signals = MagicMock(return_value=data)
            mock_store.write_signals = MagicMock(return_value=True)
            mock_store.get_active_signals = MagicMock(return_value=data)

        else:  # data store
            if data is None:
                data = {"type": "generic", "value": 42}
            mock_store.read = MagicMock(return_value=data)
            mock_store.write = MagicMock(return_value=True)
            mock_store.query = MagicMock(return_value=[data])

        # Common methods
        mock_store.exists = MagicMock(return_value=True)
        mock_store.delete = MagicMock(return_value=True)

        return mock_store


class DataBuilder:
    """
    Builder for test data generation.
    """

    @staticmethod
    def feature_data(
        n_samples: int = 100,
        n_features: int = 10,
        feature_names: list[str] | None = None,
        as_dataframe: bool = False,
    ) -> np.ndarray | pd.DataFrame:
        """
        Generate synthetic feature data.

        Parameters
        ----------
        n_samples : int
            Number of samples
        n_features : int
            Number of features
        feature_names : list[str], optional
            Names for features (for DataFrame)
        as_dataframe : bool
            Return as pandas DataFrame

        Returns
        -------
        np.ndarray or pd.DataFrame
            Generated feature data

        """
        rng = np.random.default_rng(42)
        data = rng.standard_normal((n_samples, n_features)).astype(np.float32)

        if as_dataframe:
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(n_features)]
            return pd.DataFrame(data, columns=feature_names)

        return data

    @staticmethod
    def predictions(
        n_samples: int = 100,
        n_outputs: int = 1,
        bounded: bool = True,
    ) -> np.ndarray:
        """
        Generate synthetic prediction data.

        Parameters
        ----------
        n_samples : int
            Number of samples
        n_outputs : int
            Number of outputs per sample
        bounded : bool
            Whether to bound predictions to [0, 1]

        Returns
        -------
        np.ndarray
            Generated predictions

        """
        rng = np.random.default_rng(42)

        if bounded:
            # Generate predictions in [0, 1] range
            predictions = rng.uniform(0, 1, (n_samples, n_outputs)).astype(np.float32)
        else:
            predictions = rng.standard_normal((n_samples, n_outputs)).astype(np.float32)

        return predictions.squeeze() if n_outputs == 1 else predictions

    @staticmethod
    def time_series(
        n_points: int = 100,
        start_time: int | None = None,
        interval_ns: int = 60_000_000_000,  # 1 minute in nanoseconds
    ) -> np.ndarray:
        """
        Generate time series timestamps.

        Parameters
        ----------
        n_points : int
            Number of time points
        start_time : int, optional
            Starting timestamp in nanoseconds (current time if None)
        interval_ns : int
            Interval between points in nanoseconds

        Returns
        -------
        np.ndarray
            Array of timestamps

        """
        if start_time is None:
            start_time = int(time.time() * 1e9)

        return np.arange(start_time, start_time + n_points * interval_ns, interval_ns)

    @staticmethod
    def ohlcv_data(
        n_bars: int = 100,
        start_price: float = 100.0,
        volatility: float = 0.02,
        as_dataframe: bool = True,
    ) -> pd.DataFrame | dict:
        """
        Generate synthetic OHLCV data.

        Parameters
        ----------
        n_bars : int
            Number of bars
        start_price : float
            Starting price
        volatility : float
            Price volatility
        as_dataframe : bool
            Return as DataFrame

        Returns
        -------
        pd.DataFrame or dict
            Generated OHLCV data

        """
        rng = np.random.default_rng(42)

        # Generate price series using random walk
        returns = rng.normal(0, volatility, n_bars)
        prices = start_price * np.exp(np.cumsum(returns))

        # Generate OHLCV
        data = {
            "open": prices * (1 + rng.normal(0, volatility / 10, n_bars)),
            "high": prices * (1 + np.abs(rng.normal(0, volatility / 5, n_bars))),
            "low": prices * (1 - np.abs(rng.normal(0, volatility / 5, n_bars))),
            "close": prices,
            "volume": rng.uniform(1000, 10000, n_bars),
        }

        # Ensure high >= max(open, close) and low <= min(open, close)
        data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
        data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

        if as_dataframe:
            return pd.DataFrame(data)

        return data

    @staticmethod
    def signal_data(
        n_signals: int = 10,
        instrument_id: str = "EUR/USD.SIM",
        start_time: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generate synthetic signal data.

        Parameters
        ----------
        n_signals : int
            Number of signals
        instrument_id : str
            Instrument identifier
        start_time : int, optional
            Starting timestamp

        Returns
        -------
        list[dict]
            List of signal dictionaries

        """
        if start_time is None:
            start_time = int(time.time() * 1e9)

        rng = np.random.default_rng(42)
        signals = []

        for i in range(n_signals):
            signal = {
                "instrument_id": instrument_id,
                "ts_event": start_time + i * 60_000_000_000,  # 1 minute intervals
                "ts_init": start_time + i * 60_000_000_000 + 1000,
                "signal": int(rng.choice([-1, 0, 1])),
                "confidence": float(rng.uniform(0.5, 1.0)),
                "features": {
                    "sma_20": float(rng.normal(1.09, 0.01)),
                    "rsi": float(rng.uniform(30, 70)),
                },
            }
            signals.append(signal)

        return signals


class RegistryBuilder:
    """
    Builder for registry-related test objects.
    """

    @staticmethod
    def model_manifest(**overrides: Any) -> ModelManifest:
        """
        Create a model manifest with defaults and overrides.
        """
        from ml.registry.base import ModelRole, DataRequirements
        import time

        defaults = {
            "model_id": "test_model",
            "role": ModelRole.INFERENCE,
            "data_requirements": DataRequirements.L1_ONLY,
            "architecture": "xgboost",
            "feature_schema": {"feature_1": "float", "feature_2": "float"},
            "feature_schema_hash": "abc123",
            "version": "1.0.0",
            "created_at": time.time(),
            "last_modified": time.time(),
            "performance_metrics": {"accuracy": 0.95},
            "training_config": {"n_estimators": 100},
        }

        manifest_dict = {**defaults, **overrides}
        return ModelManifest(**manifest_dict)

    @staticmethod
    def feature_manifest(**overrides: Any) -> FeatureManifest:
        """
        Create a feature manifest with defaults and overrides.
        """
        from ml.registry.feature_registry import FeatureRole
        from ml.registry.base import DataRequirements
        import time

        defaults = {
            "feature_set_id": "test_features",
            "name": "Test Features",
            "version": "1.0.0",
            "role": FeatureRole.INFERENCE_SUPPORT,
            "data_requirements": DataRequirements.L1_ONLY,
            "feature_names": ["sma_20", "rsi_14"],
            "feature_dtypes": ["float32", "float32"],
            "schema_hash": "def456",
            "pipeline_signature": "pipeline_sig_123",
            "pipeline_version": "1.0.0",
            "created_at": time.time(),
            "last_modified": time.time(),
        }

        manifest_dict = {**defaults, **overrides}
        return FeatureManifest(**manifest_dict)

    @staticmethod
    def strategy_manifest(**overrides: Any) -> StrategyManifest:
        """
        Create a strategy manifest with defaults and overrides.
        """
        from ml.registry.strategy_registry import StrategyType, MarketRegime
        import time

        defaults = {
            "strategy_id": "test_strategy",
            "strategy_type": StrategyType.TREND_FOLLOWING,
            "version": "1.0.0",
            "required_models": ["test_model"],
            "required_features": ["test_features"],
            "suitable_regimes": [MarketRegime.TRENDING_UP],
            "instrument_types": ["FX", "CRYPTO"],
            "timeframe_range": ("1m", "1h"),
            "max_position_size": 0.1,
            "max_leverage": 2.0,
            "max_drawdown": 0.15,
            "stop_loss_type": "fixed",
            "min_sharpe_ratio": 0.5,
            "min_win_rate": 0.4,
            "max_correlation_with_portfolio": 0.7,
            "parent_strategy_id": None,
            "incompatible_strategies": [],
            "config_schema": {"position_size": "float"},
            "default_config": {"position_size": 0.02},
            "backtest_metrics": {"sharpe_ratio": 1.5},
            "live_metrics": None,
            "created_at": time.time(),
            "last_modified": time.time(),
            "author": "test_author",
            "description": "Test strategy",
        }

        manifest_dict = {**defaults, **overrides}
        return StrategyManifest(**manifest_dict)
