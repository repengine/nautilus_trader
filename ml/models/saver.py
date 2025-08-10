#!/usr/bin/env python3

"""
Model saving utilities that ensure consistent metadata.

This module provides functions to save models with proper metadata,
ensuring they can be loaded correctly by the unified model system.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ml._imports import HAS_LIGHTGBM
from ml._imports import HAS_ONNX
from ml._imports import HAS_XGBOOST
from ml.models import ModelMetadata
from ml.models import ModelType
from ml.models import detect_model_type


def save_model_with_metadata(
    model: Any,
    path: str | Path,
    input_shape: tuple[int, ...] | None = None,
    output_shape: tuple[int, ...] | None = None,
    training_metadata: dict[str, Any] | None = None,
    force_pickle: bool = False,
) -> Path:
    """
    Save a model with comprehensive metadata.

    This ensures the model can be loaded correctly by the unified model system.

    Parameters
    ----------
    model : Any
        The model to save
    path : str | Path
        Path where to save the model
    input_shape : Optional[tuple[int, ...]]
        Expected input shape (important for validation)
    output_shape : Optional[tuple[int, ...]]
        Expected output shape
    training_metadata : Optional[dict[str, Any]]
        Additional metadata from training (hyperparameters, metrics, etc.)
    force_pickle : bool
        Force saving as pickle even if better format available

    Returns
    -------
    Path
        Path where the model was saved

    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Detect model type
    model_type = detect_model_type(model)

    # Save model in appropriate format
    if model_type == ModelType.XGBOOST and not force_pickle:
        model_path = _save_xgboost_model(model, path)
    elif model_type == ModelType.LIGHTGBM and not force_pickle:
        model_path = _save_lightgbm_model(model, path)
    elif model_type == ModelType.ONNX:
        model_path = _save_onnx_model(model, path)
    else:
        model_path = _save_pickle_model(model, path)

    # Create metadata
    metadata = ModelMetadata(
        model_type=model_type,
        path=str(model_path),
        version=_generate_version(model),
        size_bytes=model_path.stat().st_size,
        modified_time=model_path.stat().st_mtime,
        input_shape=input_shape,
        output_shape=output_shape,
        training_metadata=training_metadata or {},
    )

    # Save metadata alongside model
    metadata_path = model_path.with_suffix(model_path.suffix + ".meta.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)

    return model_path


def _save_xgboost_model(model: Any, path: Path) -> Path:
    """Save XGBoost model in native format."""
    if HAS_XGBOOST:
        model_path = path.with_suffix(".xgb")
        model.save_model(str(model_path))
        return model_path
    else:
        return _save_pickle_model(model, path)


def _save_lightgbm_model(model: Any, path: Path) -> Path:
    """Save LightGBM model in native format."""
    if HAS_LIGHTGBM:
        model_path = path.with_suffix(".lgb")
        model.booster_.save_model(str(model_path))
        return model_path
    else:
        return _save_pickle_model(model, path)


def _save_onnx_model(model: Any, path: Path) -> Path:
    """Save ONNX model."""
    if HAS_ONNX:
        import onnx
        model_path = path.with_suffix(".onnx")
        onnx.save(model, str(model_path))
        return model_path
    else:
        raise ValueError("Cannot save ONNX model without onnxruntime installed")


def _save_pickle_model(model: Any, path: Path) -> Path:
    """Save model as pickle."""
    model_path = path.with_suffix(".pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return model_path


def _generate_version(model: Any) -> str:
    """Generate a version hash for the model."""
    import hashlib

    # Create a version based on model characteristics
    version_parts = [
        model.__class__.__name__,
        str(type(model)),
    ]

    # Add model-specific information
    if hasattr(model, "n_estimators"):
        version_parts.append(f"n_estimators={model.n_estimators}")
    if hasattr(model, "max_depth"):
        version_parts.append(f"max_depth={model.max_depth}")
    if hasattr(model, "get_params"):
        params = model.get_params()
        version_parts.append(f"params={len(params)}")

    version_string = "|".join(version_parts)
    return hashlib.md5(version_string.encode()).hexdigest()[:8]


def convert_to_onnx(
    model: Any,
    sample_input: NDArray[np.float32],
    output_path: str | Path,
    opset_version: int = 11,
) -> Path:
    """
    Convert a model to ONNX format.

    Parameters
    ----------
    model : Any
        Model to convert (XGBoost, LightGBM, sklearn)
    sample_input : NDArray[np.float32]
        Sample input for shape inference
    output_path : str | Path
        Where to save the ONNX model
    opset_version : int
        ONNX opset version to use

    Returns
    -------
    Path
        Path to saved ONNX model

    """
    output_path = Path(output_path).with_suffix(".onnx")
    model_type = detect_model_type(model)

    if model_type == ModelType.XGBOOST:
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed")
        from onnxmltools import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType

        initial_type = [("float_input", FloatTensorType([None, sample_input.shape[-1]]))]
        onnx_model = convert_xgboost(model, initial_types=initial_type, target_opset=opset_version)

    elif model_type == ModelType.LIGHTGBM:
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed")
        from onnxmltools import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType

        initial_type = [("float_input", FloatTensorType([None, sample_input.shape[-1]]))]
        onnx_model = convert_lightgbm(model, initial_types=initial_type, target_opset=opset_version)

    elif model_type == ModelType.SKLEARN:
        from skl2onnx import to_onnx

        onnx_model = to_onnx(model, sample_input[:1], target_opset=opset_version)

    else:
        raise ValueError(f"Cannot convert {model_type} to ONNX")

    # Save ONNX model
    import onnx
    onnx.save(onnx_model, str(output_path))

    # Save metadata
    metadata = ModelMetadata(
        model_type=ModelType.ONNX,
        path=str(output_path),
        version=_generate_version(model),
        size_bytes=output_path.stat().st_size,
        modified_time=output_path.stat().st_mtime,
        input_shape=sample_input.shape,
        output_shape=None,  # Will be determined when loaded
        input_names=["float_input"],
        output_names=None,  # Will be determined when loaded
    )

    metadata_path = output_path.with_suffix(".onnx.meta.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)

    return output_path
