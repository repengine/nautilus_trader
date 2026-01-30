#!/usr/bin/env python3

"""
Test model factory for creating minimal but valid ML models.

This factory provides consistent, lightweight models for testing without relying on
pickle or creating invalid/empty files.

"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from ml._imports import HAS_LIGHTGBM
from ml._imports import HAS_ONNX
from ml._imports import HAS_XGBOOST
from ml._imports import check_ml_dependencies
from ml._imports import lgb
from ml._imports import ort
from ml._imports import xgb


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")


class TestModelFactory:
    """
    Factory for creating minimal but valid test models.

    All models created by this factory are:
    - Minimal in size (fast to create and load)
    - Valid for their respective frameworks
    - Saved in production-safe formats (no pickle)
    - Include proper metadata

    """

    @staticmethod
    def create_minimal_xgboost_model(
        n_features: int = 10,
        model_type: Literal["classification", "regression"] = "classification",
        output_path: Path | None = None,
        n_samples: int = 20,
    ) -> Path:
        """
        Create a minimal valid XGBoost model for testing.

        Parameters
        ----------
        n_features : int, default 10
            Number of input features
        model_type : str, default "classification"
            Type of model to create
        output_path : Path, optional
            Where to save the model (temp file if not provided)
        n_samples : int, default 20
            Number of training samples

        Returns
        -------
        Path
            Path to the saved model file

        """
        if not HAS_XGBOOST:
            check_ml_dependencies(["xgboost"])

        # Create minimal training data
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n_samples, n_features)).astype(np.float32)

        cpu_params: dict[str, object] = {
            "device": "cpu",
            "tree_method": "hist",
            "predictor": "cpu_predictor",
        }

        if model_type == "classification":
            y = rng.integers(0, 2, n_samples)
            model = xgb.XGBClassifier(
                n_estimators=2,  # Minimal trees
                max_depth=2,  # Shallow trees
                random_state=42,
                verbosity=0,
                **cpu_params,
            )
        else:
            y = rng.standard_normal(n_samples).astype(np.float32)
            model = xgb.XGBRegressor(
                n_estimators=2,
                max_depth=2,
                random_state=42,
                verbosity=0,
                **cpu_params,
            )

        # Train minimal model
        model.fit(X, y)

        # Determine save path
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(
                suffix=".json",
                delete=False,
            )
            output_path = Path(temp_file.name)
            temp_file.close()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure JSON format for security
        if not str(output_path).endswith(".json"):
            output_path = output_path.with_suffix(".json")

        # Save model in JSON format
        model.save_model(str(output_path))

        # Save metadata
        metadata_path = output_path.with_suffix(".json.meta")
        metadata = {
            "model_type": "xgboost",
            "model_class": model_type,
            "n_features": n_features,
            "n_samples": n_samples,
            "test_model": True,
            "feature_names": [f"f{i}" for i in range(n_features)],
            "input_shape": [None, n_features],
            "output_shape": [None, 1] if model_type == "regression" else [None, 2],
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return output_path

    @staticmethod
    def create_minimal_lightgbm_model(
        n_features: int = 10,
        model_type: Literal["classification", "regression"] = "classification",
        output_path: Path | None = None,
        n_samples: int = 20,
    ) -> Path:
        """
        Create a minimal valid LightGBM model for testing.

        Parameters
        ----------
        n_features : int, default 10
            Number of input features
        model_type : str, default "classification"
            Type of model to create
        output_path : Path, optional
            Where to save the model (temp file if not provided)
        n_samples : int, default 20
            Number of training samples

        Returns
        -------
        Path
            Path to the saved model file

        """
        if not HAS_LIGHTGBM:
            check_ml_dependencies(["lightgbm"])

        # Create minimal training data
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n_samples, n_features)).astype(np.float32)

        if model_type == "classification":
            y = rng.integers(0, 2, n_samples)
            model = lgb.LGBMClassifier(
                n_estimators=2,
                max_depth=2,
                random_state=42,
                verbosity=-1,
            )
        else:
            y = rng.standard_normal(n_samples).astype(np.float32)
            model = lgb.LGBMRegressor(
                n_estimators=2,
                max_depth=2,
                random_state=42,
                verbosity=-1,
            )

        # Train minimal model
        model.fit(X, y)

        # Determine save path
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(
                suffix=".txt",
                delete=False,
            )
            output_path = Path(temp_file.name)
            temp_file.close()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model in text format
        model.booster_.save_model(str(output_path))

        # Save metadata
        metadata_path = output_path.with_suffix(".txt.meta")
        metadata = {
            "model_type": "lightgbm",
            "model_class": model_type,
            "n_features": n_features,
            "n_samples": n_samples,
            "test_model": True,
            "feature_names": [f"f{i}" for i in range(n_features)],
            "input_shape": [None, n_features],
            "output_shape": [None, 1] if model_type == "regression" else [None, 2],
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return output_path

    @staticmethod
    def create_onnx_model(
        n_features: int = 10,
        n_outputs: int = 1,
        output_path: Path | None = None,
    ) -> Path:
        """
        Create a minimal ONNX model for testing.

        Parameters
        ----------
        n_features : int, default 10
            Number of input features
        n_outputs : int, default 1
            Number of outputs
        output_path : Path, optional
            Where to save the model (temp file if not provided)

        Returns
        -------
        Path
            Path to the saved model file

        """
        if not HAS_ONNX:
            check_ml_dependencies(["onnxruntime"])

        if not HAS_XGBOOST:
            check_ml_dependencies(["xgboost"])

        # Create a minimal XGBoost model first
        temp_xgb = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        temp_xgb_path = Path(temp_xgb.name)
        temp_xgb.close()

        # Create XGBoost model
        xgb_path = TestModelFactory.create_minimal_xgboost_model(
            n_features=n_features,
            model_type="regression" if n_outputs == 1 else "classification",
            output_path=temp_xgb_path,
        )

        # Determine save path
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(
                suffix=".onnx",
                delete=False,
            )
            output_path = Path(temp_file.name)
            temp_file.close()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to ONNX
        try:
            from onnxmltools import convert_xgboost
            from onnxmltools.convert.common.data_types import FloatTensorType

            # Load the XGBoost model
            booster = xgb.Booster()
            booster.load_model(str(xgb_path))

            # Define input type
            initial_type = [
                ("float_input", FloatTensorType([None, n_features])),
            ]

            # Convert to ONNX
            onnx_model = convert_xgboost(
                booster,
                initial_types=initial_type,
                target_opset=12,
            )

            # Save ONNX model
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

        except ImportError:
            # Fallback: Create a simple ONNX model using numpy operations
            import onnx

            # Create a simple linear model: output = input @ weights + bias
            from numpy.random import default_rng
            from onnx import TensorProto
            from onnx import helper

            _rng = default_rng(0)
            weights = _rng.standard_normal((n_features, n_outputs)).astype(np.float32)
            bias = _rng.standard_normal(n_outputs).astype(np.float32)

            # Create ONNX graph
            input_tensor = helper.make_tensor_value_info(
                "input",
                TensorProto.FLOAT,
                [None, n_features],
            )
            output_tensor = helper.make_tensor_value_info(
                "output",
                TensorProto.FLOAT,
                [None, n_outputs],
            )

            weights_tensor = helper.make_tensor(
                "weights",
                TensorProto.FLOAT,
                [n_features, n_outputs],
                weights.flatten(),
            )
            bias_tensor = helper.make_tensor("bias", TensorProto.FLOAT, [n_outputs], bias)

            matmul_node = helper.make_node("MatMul", ["input", "weights"], ["matmul_output"])
            add_node = helper.make_node("Add", ["matmul_output", "bias"], ["output"])

            graph = helper.make_graph(
                [matmul_node, add_node],
                "test_model",
                [input_tensor],
                [output_tensor],
                [weights_tensor, bias_tensor],
            )

            model = helper.make_model(graph)
            onnx.save(model, str(output_path))

        finally:
            # Clean up temporary XGBoost model
            if temp_xgb_path.exists():
                temp_xgb_path.unlink()
            meta_path = temp_xgb_path.with_suffix(".json.meta")
            if meta_path.exists():
                meta_path.unlink()

        # Save metadata
        metadata_path = output_path.with_suffix(".onnx.meta")
        metadata = {
            "model_type": "onnx",
            "n_features": n_features,
            "n_outputs": n_outputs,
            "test_model": True,
            "input_shape": [None, n_features],
            "output_shape": [None, n_outputs],
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return output_path

    @staticmethod
    def create_sklearn_model(
        n_features: int = 10,
        model_type: Literal["classification", "regression"] = "classification",
        output_path: Path | None = None,
        n_samples: int = 20,
    ) -> Path:
        """
        Create a minimal scikit-learn model for testing.

        Parameters
        ----------
        n_features : int, default 10
            Number of input features
        model_type : str, default "classification"
            Type of model to create
        output_path : Path, optional
            Where to save the model (temp file if not provided)
        n_samples : int, default 20
            Number of training samples

        Returns
        -------
        Path
            Path to the saved model file

        """
        # Security: Only import joblib in test environments
        import os

        if not (
            os.getenv("PYTEST_CURRENT_TEST")
            or os.getenv("ML_TESTING")
            or os.getenv("ML_ALLOW_JOBLIB")
        ):
            raise RuntimeError("JobLib usage only allowed in test environments")

        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import RandomForestRegressor

        # Create minimal training data
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n_samples, n_features)).astype(np.float32)

        if model_type == "classification":
            y = rng.integers(0, 2, n_samples)
            model = RandomForestClassifier(
                n_estimators=2,
                max_depth=2,
                random_state=42,
            )
        else:
            y = rng.standard_normal(n_samples).astype(np.float32)
            model = RandomForestRegressor(
                n_estimators=2,
                max_depth=2,
                random_state=42,
            )

        # Train minimal model
        model.fit(X, y)

        # Determine save path
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(
                suffix=".joblib",
                delete=False,
            )
            output_path = Path(temp_file.name)
            temp_file.close()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Coerce to .joblib extension for safety
        if output_path.suffix.lower() != ".joblib":
            output_path = output_path.with_suffix(".joblib")

        # Save model using joblib (safer than pickle)
        joblib.dump(model, output_path)

        # Save metadata
        metadata_path = output_path.with_suffix(".joblib.meta")
        metadata = {
            "model_type": "sklearn",
            "model_class": model.__class__.__name__,
            "n_features": n_features,
            "n_samples": n_samples,
            "test_model": True,
            "feature_names": [f"f{i}" for i in range(n_features)],
            "input_shape": [None, n_features],
            "output_shape": [None, 1] if model_type == "regression" else [None, 2],
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return output_path

    @staticmethod
    def validate_model(model_path: Path) -> dict[str, Any]:
        """
        Validate that a test model is properly formed.

        Parameters
        ----------
        model_path : Path
            Path to the model file

        Returns
        -------
        dict[str, Any]
            Validation results including model info and any issues

        """
        results: dict[str, Any] = {
            "valid": False,
            "path": str(model_path),
            "exists": model_path.exists(),
            "size": 0,
            "format": None,
            "metadata": None,
            "issues": [],
        }

        if not model_path.exists():
            cast(list[str], results["issues"]).append("Model file does not exist")
            return results

        # Check file size
        results["size"] = model_path.stat().st_size
        if results["size"] == 0:
            cast(list[str], results["issues"]).append("Model file is empty")
            return results

        # Determine format
        suffix = model_path.suffix.lower()
        results["format"] = suffix

        # Check for metadata
        meta_paths = [
            model_path.with_suffix(f"{suffix}.meta"),
            model_path.with_suffix(".meta"),
            model_path.with_suffix(".json.meta"),
        ]

        for meta_path in meta_paths:
            if meta_path.exists():
                with open(meta_path) as f:
                    results["metadata"] = json.load(f)
                break

        # Validate based on format
        if suffix == ".json":
            try:
                with open(model_path) as f:
                    data = json.load(f)
                    if not data:
                        cast(list[str], results["issues"]).append("JSON model is empty")
                    else:
                        results["valid"] = True
            except json.JSONDecodeError as e:
                cast(list[str], results["issues"]).append(f"Invalid JSON: {e}")

        elif suffix == ".onnx":
            if HAS_ONNX and ort is not None:
                try:
                    session = ort.InferenceSession(str(model_path))
                    results["valid"] = True
                except Exception as e:
                    cast(list[str], results["issues"]).append(f"Invalid ONNX model: {e}")
            else:
                cast(list[str], results["issues"]).append(
                    "ONNX runtime not available for validation",
                )

        elif suffix in [".pkl", ".pickle"]:
            cast(list[str], results["issues"]).append(
                "Pickle format not allowed for security reasons",
            )

        elif suffix == ".joblib":
            try:
                # Security: Only import joblib in test environments
                import os

                if not (
                    os.getenv("PYTEST_CURRENT_TEST")
                    or os.getenv("ML_TESTING")
                    or os.getenv("ML_ALLOW_JOBLIB")
                ):
                    cast(list[str], results["issues"]).append(
                        "JobLib usage only allowed in test environments",
                    )
                    return results

                import joblib

                model = joblib.load(model_path)
                if model is not None:
                    results["valid"] = True
            except Exception as e:
                cast(list[str], results["issues"]).append(f"Invalid joblib model: {e}")

        else:
            # Assume valid for other formats
            results["valid"] = True

        return results


class TestDataFactory:
    """
    Factory for creating test data.
    """

    @staticmethod
    def create_feature_data(
        n_samples: int = 100,
        n_features: int = 10,
        seed: int = 42,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """
        Create synthetic feature data for testing.

        Parameters
        ----------
        n_samples : int, default 100
            Number of samples
        n_features : int, default 10
            Number of features
        seed : int, default 42
            Random seed for reproducibility

        Returns
        -------
        np.ndarray
            Feature matrix of shape (n_samples, n_features)

        """
        rng = np.random.default_rng(seed)

        # Create features with different distributions
        features: list[np.ndarray[Any, np.dtype[np.float64]]] = []

        for i in range(n_features):
            if i % 3 == 0:
                # Normal distribution
                feature = rng.standard_normal(n_samples)
            elif i % 3 == 1:
                # Uniform distribution
                feature = rng.uniform(-1, 1, n_samples)
            else:
                # Exponential distribution
                feature = rng.exponential(1, n_samples)

            features.append(feature)

        return np.column_stack(features).astype(np.float32)

    @staticmethod
    def create_target_data(
        n_samples: int = 100,
        target_type: Literal["binary", "multiclass", "regression"] = "binary",
        n_classes: int = 3,
        seed: int = 42,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """
        Create synthetic target data for testing.

        Parameters
        ----------
        n_samples : int, default 100
            Number of samples
        target_type : str, default "binary"
            Type of target variable
        n_classes : int, default 3
            Number of classes for multiclass
        seed : int, default 42
            Random seed

        Returns
        -------
        np.ndarray
            Target array

        """
        rng = np.random.default_rng(seed)

        if target_type == "binary":
            return rng.integers(0, 2, n_samples)
        elif target_type == "multiclass":
            return rng.integers(0, n_classes, n_samples)
        else:  # regression
            return rng.standard_normal(n_samples).astype(np.float32)

    @staticmethod
    def bars(
        n: int = 100,
        instrument_id: Any | None = None,
        bar_type: Any | None = None,
        start_date: Any | None = None,
    ) -> list[Any]:
        """
        Generate test Bar objects with realistic OHLCV data.

        Creates a list of Bar objects with correlated price movement, realistic
        OHLC relationships, and monotonically increasing timestamps.

        Parameters
        ----------
        n : int, default 100
            Number of bars to generate (must be > 0)
        instrument_id : InstrumentId | str | None, default None
            Instrument identifier. If None, defaults to EUR/USD.SIM.
            Can be InstrumentId object or string representation.
        bar_type : BarType | str | None, default None
            Bar type specification. If None, defaults to 1-MINUTE-LAST-EXTERNAL.
            Can be BarType object or string representation.
        start_date : datetime | None, default None
            Starting timestamp for bar generation. If None, defaults to 2024-01-01.

        Returns
        -------
        list[Bar]
            List of Bar objects with correlated price movement

        Raises
        ------
        ValueError
            If n <= 0

        Examples
        --------
        >>> factory = TestDataFactory()
        >>> bars = factory.bars(n=50, instrument_id="BTC/USD.SIM")
        >>> assert len(bars) == 50
        >>> assert all(float(b.high) >= float(b.low) for b in bars)
        >>> # Verify timestamp monotonicity
        >>> timestamps = [b.ts_event for b in bars]
        >>> assert timestamps == sorted(timestamps)

        """
        from datetime import datetime

        from nautilus_trader.core.datetime import dt_to_unix_nanos
        from nautilus_trader.model.data import Bar
        from nautilus_trader.model.data import BarType
        from nautilus_trader.model.identifiers import InstrumentId
        from nautilus_trader.model.objects import Price
        from nautilus_trader.model.objects import Quantity

        if n <= 0:
            raise ValueError("n must be greater than 0")

        # Default values
        if instrument_id is None:
            instrument_id = InstrumentId.from_str("EUR/USD.SIM")
        elif isinstance(instrument_id, str):
            instrument_id = InstrumentId.from_str(instrument_id)

        if bar_type is None:
            bar_type = BarType.from_str(f"{instrument_id}-1-MINUTE-LAST-EXTERNAL")
        elif isinstance(bar_type, str):
            bar_type = BarType.from_str(bar_type)

        if start_date is None:
            start_date = datetime(2024, 1, 1, 0, 0, 0)

        # Generate bars with realistic OHLCV data
        bars: list[Bar] = []
        base_timestamp = dt_to_unix_nanos(start_date)
        interval_ns = 60_000_000_000  # 1 minute in nanoseconds

        # Start price around 1.0900 for EUR/USD (or scale for other instruments)
        current_price = 1.0900

        rng = np.random.default_rng(42)  # Fixed seed for reproducibility

        for i in range(n):
            # Generate realistic price movement
            drift = 0.00001
            volatility = 0.0001
            returns = rng.normal(drift, volatility, 4)

            open_price = current_price
            high_price = open_price + abs(returns[0]) * 2
            low_price = open_price - abs(returns[1]) * 2
            close_price = open_price + returns[2]

            # Ensure OHLC relationships
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)

            # Generate volume
            volume = float(rng.uniform(1000, 5000)) * (1 + abs(returns[3]) * 10)

            bar = Bar(
                bar_type=bar_type,
                open=Price(open_price, precision=5),
                high=Price(high_price, precision=5),
                low=Price(low_price, precision=5),
                close=Price(close_price, precision=5),
                volume=Quantity(volume, precision=0),
                ts_event=base_timestamp + i * interval_ns,
                ts_init=base_timestamp + i * interval_ns + 1000,
            )

            bars.append(bar)
            current_price = close_price

        return bars

    @staticmethod
    def features(
        n: int = 50,
        n_features: int = 10,
        instrument: str = "EUR/USD",
        seed: int = 42,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """
        Generate synthetic feature data for testing.

        This is a convenience wrapper around create_feature_data() with common
        defaults for typical test scenarios.

        Parameters
        ----------
        n : int, default 50
            Number of samples (must be > 0)
        n_features : int, default 10
            Number of features (must be > 0)
        instrument : str, default "EUR/USD"
            Instrument identifier (for documentation, not used in generation)
        seed : int, default 42
            Random seed for reproducibility (must be >= 0)

        Returns
        -------
        np.ndarray
            Feature array of shape (n, n_features) with dtype float32

        Raises
        ------
        ValueError
            If n <= 0 or n_features <= 0 or seed < 0

        Examples
        --------
        >>> factory = TestDataFactory()
        >>> features = factory.features(n=100, n_features=5)
        >>> assert features.shape == (100, 5)
        >>> assert features.dtype == np.float32
        >>> assert not np.any(np.isnan(features))

        """
        if n <= 0:
            raise ValueError("n_samples must be greater than 0")
        if n_features <= 0:
            raise ValueError("n_features must be greater than 0")
        if seed < 0:
            raise ValueError("seed must be non-negative")

        return TestDataFactory.create_feature_data(
            n_samples=n,
            n_features=n_features,
            seed=seed,
        )

    @staticmethod
    def predictions(
        n: int = 20,
        instrument: str = "EUR/USD.SIM",
        start_timestamp: int | None = None,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """
        Generate test prediction dictionaries.

        Creates prediction dictionaries with realistic structure for ML inference
        testing. Each prediction includes instrument_id, timestamp, prediction
        value, and confidence score.

        Parameters
        ----------
        n : int, default 20
            Number of predictions to generate (must be > 0)
        instrument : str, default "EUR/USD.SIM"
            Instrument identifier string
        start_timestamp : int | None, default None
            Starting timestamp in nanoseconds. If None, uses current time.
        seed : int, default 42
            Random seed for reproducibility

        Returns
        -------
        list[dict[str, Any]]
            List of prediction dictionaries with keys:
                - instrument_id: str (instrument identifier)
                - timestamp: int (nanoseconds since epoch)
                - prediction: float in [0, 1]
                - confidence: float in [0, 1]

        Raises
        ------
        ValueError
            If n <= 0 or instrument is empty

        Examples
        --------
        >>> factory = TestDataFactory()
        >>> preds = factory.predictions(n=10)
        >>> assert len(preds) == 10
        >>> assert all(0 <= p["confidence"] <= 1 for p in preds)
        >>> assert all(0 <= p["prediction"] <= 1 for p in preds)

        """
        import time

        if n <= 0:
            raise ValueError("n must be greater than 0")
        if not instrument:
            raise ValueError("instrument must not be empty")

        rng = np.random.default_rng(seed)

        # Default start timestamp
        if start_timestamp is None:
            start_timestamp = int(time.time() * 1e9)

        predictions: list[dict[str, Any]] = []
        interval_ns = 60_000_000_000  # 1 minute between predictions

        for i in range(n):
            # Generate prediction in [0, 1] range
            prediction = float(rng.uniform(0, 1))

            # Generate confidence in [0, 1] range
            # Higher confidence for predictions closer to extremes (0 or 1)
            confidence = float(max(prediction, 1.0 - prediction) * rng.uniform(0.5, 1.0))
            confidence = min(max(confidence, 0.0), 1.0)  # Clamp to [0, 1]

            pred_dict: dict[str, Any] = {
                "instrument_id": instrument,
                "timestamp": start_timestamp + i * interval_ns,
                "prediction": prediction,
                "confidence": confidence,
            }
            predictions.append(pred_dict)

        return predictions


__all__ = ["TestDataFactory", "TestModelFactory"]
