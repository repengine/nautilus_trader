#!/usr/bin/env python3

"""
Production model loader with automatic format detection and security.

This module provides the ProductionModelLoader which auto-detects model formats
and loads them with the appropriate wrapper for a unified interface.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ml._imports import HAS_LIGHTGBM
from ml._imports import HAS_ONNX
from ml._imports import HAS_XGBOOST
from ml._imports import check_ml_dependencies
from ml._imports import lgb
from ml._imports import ort
from ml._imports import xgb
from ml.models import LightGBMModel
from ml.models import ModelMetadata
from ml.models import ModelType
from ml.models import ONNXModel
from ml.models import XGBoostModel
from ml.models import create_model_wrapper

if TYPE_CHECKING:
    from ml.models import BaseModel


class ModelLoader(ABC):
    """
    Abstract base class for model loading strategies.
    """

    @abstractmethod
    def load_model(self, path: str) -> tuple[Any, dict[str, Any]]:
        """
        Load model and return model with metadata.

        Parameters
        ----------
        path : str
            Path to model file.

        Returns
        -------
        tuple[Any, dict[str, Any]]
            Tuple of (model, metadata).

        """

    def get_model_version(self, path: str) -> str:
        """
        Get model version without loading.

        Parameters
        ----------
        path : str
            Path to model file.

        Returns
        -------
        str
            Model version string.

        """
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        stat = model_path.stat()
        # Create version hash from file size and modification time
        version_string = f"{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(version_string.encode()).hexdigest()[:8]  # noqa: S324


class ProductionModelLoader(ModelLoader):
    """
    Production-grade model loader with format detection and security.

    Supports ONNX, XGBoost native, and LightGBM native formats.
    Explicitly does NOT support pickle for security reasons.
    """

    def load_model(self, path: str) -> tuple[Any, dict[str, Any]]:
        """
        Load model with automatic format detection.

        Parameters
        ----------
        path : str
            Path to the model file.

        Returns
        -------
        tuple[Any, dict[str, Any]]
            The model and its metadata dictionary.

        Raises
        ------
        FileNotFoundError
            If the model file doesn't exist.
        ValueError
            If the model format is not supported or is pickle.

        """
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Detect and load based on extension
        suffix = model_path.suffix.lower()

        if suffix == ".onnx":
            return self._load_onnx_model(model_path)
        elif suffix in {".json", ".ubj", ".xgb"}:
            return self._load_xgboost_model(model_path)
        elif suffix in {".txt", ".lgb"}:
            return self._load_lightgbm_model(model_path)
        elif suffix in {".pkl", ".pickle"}:
            raise ValueError(
                f"Pickle format not supported for security reasons. "
                f"Please export your model to ONNX or native format. "
                f"File: {path}",
            )
        elif suffix in {".joblib"}:
            # Joblib is also considered insecure for production
            raise ValueError(
                f"Joblib format not supported for security reasons. "
                f"Please export your model to ONNX or native format. "
                f"File: {path}",
            )
        else:
            raise ValueError(
                f"Unsupported model format: {suffix}. "
                f"Supported formats: .onnx, .json (XGBoost), .txt (LightGBM)",
            )

    def load_model_with_wrapper(self, path: str) -> Any:
        """
        Load model and return wrapped with appropriate model class.

        Parameters
        ----------
        path : str
            Path to the model file.

        Returns
        -------
        BaseModel
            The wrapped model with unified interface.

        """
        model, metadata = self.load_model(path)
        return create_model_wrapper(model, Path(path), metadata)

    def _load_onnx_model(self, model_path: Path) -> tuple[Any, dict[str, Any]]:
        """
        Load ONNX model.

        Parameters
        ----------
        model_path : Path
            Path to the ONNX model file.

        Returns
        -------
        tuple[Any, dict[str, Any]]
            The ONNX model and metadata.

        """
        if not HAS_ONNX:
            check_ml_dependencies(["onnx"])

        # ONNX Runtime compatibility: use default providers
        model = ort.InferenceSession(
            str(model_path),
            providers=ort.get_available_providers(),
        )

        metadata = {
            "path": str(model_path),
            "size_bytes": model_path.stat().st_size,
            "modified_time": model_path.stat().st_mtime,
            "version": self.get_model_version(str(model_path)),
            "type": "onnx",
            "input_names": [inp.name for inp in model.get_inputs()],
            "output_names": [out.name for out in model.get_outputs()],
            "input_shapes": [inp.shape for inp in model.get_inputs()],
            "output_shapes": [out.shape for out in model.get_outputs()],
        }

        # Load additional metadata if available
        metadata = self._load_metadata_json(model_path, metadata)

        return model, metadata

    def _load_xgboost_model(self, model_path: Path) -> tuple[Any, dict[str, Any]]:
        """
        Load XGBoost model in native format.

        Parameters
        ----------
        model_path : Path
            Path to the XGBoost model file.

        Returns
        -------
        tuple[Any, dict[str, Any]]
            The XGBoost model and metadata.

        """
        if not HAS_XGBOOST:
            check_ml_dependencies(["xgboost"])

        # Try loading as Booster first (raw model)
        try:
            model = xgb.Booster()
            model.load_model(str(model_path))
            is_booster = True
        except Exception:
            # Try loading as sklearn-like model
            import xgboost as xgb_module

            # XGBoost sklearn models can be loaded directly
            model = xgb_module.XGBClassifier()
            model.load_model(str(model_path))
            is_booster = False

        # Create base metadata
        metadata = self._create_metadata(model_path, "xgboost")

        # Add XGBoost specific metadata
        if is_booster:
            metadata["n_features"] = model.num_features()
            metadata["feature_names"] = (
                model.feature_names if hasattr(model, "feature_names") else []
            )
        else:
            # sklearn-like interface
            metadata["n_features"] = model.n_features_in_ if hasattr(model, "n_features_in_") else None
            metadata["feature_names"] = (
                model.feature_names_in_.tolist()
                if hasattr(model, "feature_names_in_")
                else []
            )

        # Load additional metadata if available
        metadata = self._load_metadata_json(model_path, metadata)

        return model, metadata

    def _load_lightgbm_model(self, model_path: Path) -> tuple[Any, dict[str, Any]]:
        """
        Load LightGBM model in native format.

        Parameters
        ----------
        model_path : Path
            Path to the LightGBM model file.

        Returns
        -------
        tuple[Any, dict[str, Any]]
            The LightGBM model and metadata.

        """
        if not HAS_LIGHTGBM:
            check_ml_dependencies(["lightgbm"])

        # Load as Booster (LightGBM native format is always Booster)
        model = lgb.Booster(model_file=str(model_path))
        is_booster = True

        # Create base metadata
        metadata = self._create_metadata(model_path, "lightgbm")

        # Add LightGBM specific metadata
        metadata["n_features"] = model.num_feature()
        metadata["feature_names"] = model.feature_name()

        # Load additional metadata if available
        metadata = self._load_metadata_json(model_path, metadata)

        return model, metadata

    def _create_metadata(self, model_path: Path, model_type: str) -> dict[str, Any]:
        """
        Create standardized metadata for any model.

        Parameters
        ----------
        model_path : Path
            Path to the model file.
        model_type : str
            Type of the model.

        Returns
        -------
        dict[str, Any]
            Base metadata dictionary.

        """
        return {
            "path": str(model_path),
            "type": model_type,
            "size_bytes": model_path.stat().st_size,
            "modified_time": model_path.stat().st_mtime,
            "version": self.get_model_version(str(model_path)),
            "format_version": self._get_format_version(model_type),
        }

    def _load_metadata_json(
        self,
        model_path: Path,
        base_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Load additional metadata from JSON file if it exists.

        The JSON file is created by save_model_with_metadata() and contains
        training metadata and other information.

        Parameters
        ----------
        model_path : Path
            Path to the model file.
        base_metadata : dict[str, Any]
            Base metadata dictionary to update.

        Returns
        -------
        dict[str, Any]
            Updated metadata dictionary.

        """
        # Check for metadata file created by save_model_with_metadata
        metadata_path = model_path.with_suffix(model_path.suffix + ".meta.json")
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    saved_metadata = json.load(f)

                # Merge with base metadata, preserving runtime values
                if "training_metadata" in saved_metadata:
                    base_metadata["training_metadata"] = saved_metadata["training_metadata"]
                if "input_shape" in saved_metadata:
                    base_metadata["input_shape"] = saved_metadata["input_shape"]
                if "output_shape" in saved_metadata:
                    base_metadata["output_shape"] = saved_metadata["output_shape"]
                if "model_id" in saved_metadata:
                    base_metadata["model_id"] = saved_metadata["model_id"]
                if "feature_names" in saved_metadata and not base_metadata.get("feature_names"):
                    base_metadata["feature_names"] = saved_metadata["feature_names"]

            except Exception:
                # If we can't load metadata, continue with base metadata
                pass

        return base_metadata

    def _get_format_version(self, model_type: str) -> str:
        """
        Get the version of the model format/framework.

        Parameters
        ----------
        model_type : str
            Type of the model.

        Returns
        -------
        str
            Framework version string.

        """
        if model_type == "onnx":
            return ort.__version__ if ort else "unknown"
        elif model_type == "xgboost":
            return xgb.__version__ if xgb else "unknown"
        elif model_type == "lightgbm":
            return lgb.__version__ if lgb else "unknown"
        return "unknown"


class SecurityError(Exception):
    """
    Raised when attempting to load insecure model formats.
    """

    pass


__all__ = [
    "ModelLoader",
    "ProductionModelLoader",
    "SecurityError",
]