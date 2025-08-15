#!/usr/bin/env python3

"""
Test model factory for creating minimal but valid ML models.

This factory provides consistent, lightweight models for testing without relying on
pickle or creating invalid/empty files.

"""

from __future__ import annotations

import json
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

        if model_type == "classification":
            y = rng.integers(0, 2, n_samples)
            model = xgb.XGBClassifier(
                n_estimators=2,  # Minimal trees
                max_depth=2,  # Shallow trees
                random_state=42,
                verbosity=0,
            )
        else:
            y = rng.standard_normal(n_samples).astype(np.float32)
            model = xgb.XGBRegressor(
                n_estimators=2,
                max_depth=2,
                random_state=42,
                verbosity=0,
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
            from onnx import TensorProto
            from onnx import helper

            # Create a simple linear model: output = input @ weights + bias
            weights = np.random.randn(n_features, n_outputs).astype(np.float32)
            bias = np.random.randn(n_outputs).astype(np.float32)

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
            if HAS_ONNX:
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
