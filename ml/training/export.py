#!/usr/bin/env python3

"""
Artifact export utilities for training → inference handoff.

This module consolidates model saving and ONNX conversion used by trainers. It
intentionally keeps dependencies minimal and writes a simple sidecar JSON with technical
metadata (NOT the registry manifest).

"""

from __future__ import annotations

import json
from abc import ABC
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ml._imports import HAS_LIGHTGBM
from ml._imports import HAS_ONNX
from ml._imports import HAS_ONNX_CORE
from ml._imports import HAS_XGBOOST
from ml._imports import lgb
from ml._imports import onnx
from ml._imports import onnxmltools
from ml._imports import skl2onnx
from ml._imports import xgb
from ml.config.constants import SUFFIX_ONNX
from ml.config.constants import Versions
from ml.config.names import ONNX_INPUT_NAME


class ModelType(Enum):
    ONNX = "onnx"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    SKLEARN = "sklearn"
    UNKNOWN = "unknown"


def detect_model_type(model: Any, file_path: Path | None = None) -> ModelType:
    """
    Best-effort detection of model type from object or file extension.
    """
    if file_path is not None:
        suffix = file_path.suffix.lower()
        if suffix == SUFFIX_ONNX:
            return ModelType.ONNX
        if suffix in {".json", ".ubj", ".xgb"}:
            return ModelType.XGBOOST
        if suffix in {".txt", ".lgb"}:
            return ModelType.LIGHTGBM

    # Object-based detection
    try:
        if HAS_ONNX and hasattr(model, "run") and hasattr(model, "get_inputs"):
            return ModelType.ONNX
    except Exception:
        import logging as _logging
        _logging.getLogger(__name__).debug("ONNX detection failed", exc_info=True)

    if HAS_XGBOOST and xgb is not None:
        try:
            if isinstance(model, xgb.Booster) or model.__class__.__name__.startswith("XGB"):
                return ModelType.XGBOOST
        except Exception:
            import logging as _logging
            _logging.getLogger(__name__).debug("XGBoost detection failed", exc_info=True)

    if HAS_LIGHTGBM and lgb is not None:
        try:
            if isinstance(model, lgb.Booster) or model.__class__.__name__.startswith("LGBM"):
                return ModelType.LIGHTGBM
        except Exception:
            import logging as _logging
            _logging.getLogger(__name__).debug("LightGBM detection failed", exc_info=True)

    # Fallback: sklearn-like
    if hasattr(model, "predict"):
        return ModelType.SKLEARN

    return ModelType.UNKNOWN


# Unified default opset for ONNX exports used by training layer
# Delegate to central Versions to avoid drift across modules.
DEFAULT_ONNX_OPSET = Versions.ONNX_OPSET


def save_model_with_metadata(
    model: Any,
    path: str | Path,
    input_shape: tuple[int, ...] | None = None,
    output_shape: tuple[int, ...] | None = None,
    training_metadata: dict[str, Any] | None = None,
    force_pickle: bool = False,
) -> Path:
    """
    Save a model artifact plus technical metadata sidecar.

    Notes
    -----
    This metadata is strictly file-level context to aid loaders. Deployment
    identity lives in the registry ModelManifest.

    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model_type = detect_model_type(model)

    if model_type == ModelType.XGBOOST and not force_pickle:
        model_path = _save_xgboost_model(model, path)
    elif model_type == ModelType.LIGHTGBM and not force_pickle:
        model_path = _save_lightgbm_model(model, path)
    elif model_type == ModelType.ONNX:
        model_path = _save_onnx_model(model, path)
    else:
        raise ValueError(
            f"Unsupported model type '{model_type.value}' for direct save. "
            "Export to ONNX or use a framework-native saver before calling this.",
        )

    metadata_dict = {
        "model_type": model_type.value,
        "path": str(model_path),
        "version": _generate_version(model),
        "size_bytes": model_path.stat().st_size,
        "modified_time": model_path.stat().st_mtime,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "training_metadata": training_metadata or {},
    }

    metadata_path = model_path.with_suffix(model_path.suffix + ".meta.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata_dict, f, indent=2)

    return model_path


def _save_xgboost_model(model: Any, path: Path) -> Path:
    if HAS_XGBOOST:
        model_path = path.with_suffix(".xgb")
        model.save_model(str(model_path))
        return model_path
    else:
        raise ImportError("XGBoost not installed; cannot save XGBoost model")


def _save_lightgbm_model(model: Any, path: Path) -> Path:
    if not HAS_LIGHTGBM:
        raise ImportError("LightGBM not installed; cannot save LightGBM model")

    # Support both sklearn wrapper (has booster_) and raw Booster
    try:
        assert lgb is not None
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to access LightGBM Booster: {exc}")

    booster = getattr(model, "booster_", model)
    model_path = path.with_suffix(".lgb")
    try:
        if isinstance(booster, lgb.Booster):
            booster.save_model(str(model_path))
            return model_path
    except Exception:
        import logging as _logging
        _logging.getLogger(__name__).debug("LightGBM save_model (primary) failed", exc_info=True)
    # Fallback: try common save_model API or pickle
    try:
        booster.save_model(str(model_path))
        return model_path
    except Exception as exc:
        raise RuntimeError(f"LightGBM save_model failed: {exc}")


def _save_onnx_model(model: Any, path: Path) -> Path:
    if HAS_ONNX_CORE and onnx is not None:
        model_path = path.with_suffix(SUFFIX_ONNX)
        onnx.save(model, str(model_path))
        return model_path
    else:
        raise ValueError("Cannot save ONNX model without 'onnx' installed")


# Removed: pickle saving is deprecated and unsupported.


def _generate_version(model: Any) -> str:
    import hashlib

    parts = [model.__class__.__name__, str(type(model))]
    if hasattr(model, "n_estimators"):
        parts.append(f"n_estimators={model.n_estimators}")
    if hasattr(model, "max_depth"):
        parts.append(f"max_depth={model.max_depth}")
    if hasattr(model, "get_params"):
        try:
            params = model.get_params()
            parts.append(f"params={len(params)}")
        except Exception:
            import logging as _logging
            _logging.getLogger(__name__).debug("get_params failed for version generation", exc_info=True)
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:8]


def convert_to_onnx(
    model: Any,
    sample_input: NDArray[np.float32],
    output_path: str | Path,
    opset_version: int = DEFAULT_ONNX_OPSET,
) -> Path:
    """
    Convert supported models to ONNX format.

    Writes a small metadata sidecar with input/output hints.

    """
    output_path = Path(output_path).with_suffix(".onnx")
    model_type = detect_model_type(model)

    if model_type == ModelType.XGBOOST:
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed")
        assert onnxmltools is not None
        from onnxmltools.convert.common.data_types import FloatTensorType

        initial_type = [(ONNX_INPUT_NAME, FloatTensorType([None, sample_input.shape[-1]]))]
        onnx_model = onnxmltools.convert_xgboost(
            model,
            initial_types=initial_type,
            target_opset=opset_version,
        )

    elif model_type == ModelType.LIGHTGBM:
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed")
        assert onnxmltools is not None
        from onnxmltools.convert.common.data_types import FloatTensorType

        initial_type = [(ONNX_INPUT_NAME, FloatTensorType([None, sample_input.shape[-1]]))]
        onnx_model = onnxmltools.convert_lightgbm(
            model,
            initial_types=initial_type,
            target_opset=opset_version,
        )

    elif model_type == ModelType.SKLEARN:
        assert skl2onnx is not None
        onnx_model = skl2onnx.to_onnx(model, sample_input[:1], target_opset=opset_version)

    else:
        raise ValueError(f"Cannot convert {model_type} to ONNX")

    assert HAS_ONNX_CORE and onnx is not None
    onnx.save(onnx_model, str(output_path))

    metadata_dict = {
        "model_type": ModelType.ONNX.value,
        "path": str(output_path),
        "version": _generate_version(model),
        "size_bytes": output_path.stat().st_size,
        "modified_time": output_path.stat().st_mtime,
        "input_shape": sample_input.shape,
        "output_shape": None,
        "input_names": [ONNX_INPUT_NAME],
        "output_names": None,
    }
    metadata_path = output_path.with_suffix(".onnx.meta.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata_dict, f, indent=2)

    return output_path


def convert_to_torchscript(
    model: Any,
    sample_input: NDArray[np.float32] | None,
    output_path: str | Path,
) -> Path:
    """
    Trace or script a PyTorch module to TorchScript and save to .pt.

    Notes
    -----
    - This is a generic helper; the caller is responsible for providing a model
      that accepts the provided `sample_input`.
    - For complex models that expect dict inputs (e.g., TFT), callers should
      wrap the model in a small adapter module that accepts a tensor.

    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError("PyTorch is required for TorchScript export") from exc

    output_path = Path(output_path).with_suffix(".pt")
    model.eval()

    def _jit_trace(mod: object, example: object) -> Any:
        # Wrapper to keep mypy strict happy around untyped torch APIs.
        return torch.jit.trace(mod, example)  # type: ignore[no-untyped-call]

    def _jit_script(mod: object) -> Any:
        # Wrapper to keep mypy strict happy around untyped torch APIs.
        return torch.jit.script(mod)

    with torch.inference_mode():
        if sample_input is not None:
            scripted = _jit_trace(model, torch.as_tensor(sample_input))
        else:
            scripted = _jit_script(model)
        scripted.save(str(output_path))
    return output_path


__all__ = [
    "DEFAULT_ONNX_OPSET",
    "ModelExportMixin",
    "ModelType",
    "TrainingActorContract",
    "convert_to_onnx",
    "convert_to_torchscript",
    "detect_model_type",
    "save_model_with_metadata",
]


# ---------------------------------------------------------------------------
# Model export mixin and training actor contract (unified here)
# ---------------------------------------------------------------------------


class ModelExportMixin(ABC):
    """
    Mixin class that ensures models are exported in production-ready formats.

    This should be used by all training classes to ensure compatibility with
    ProductionModelLoader in inference actors.

    """

    @abstractmethod
    def get_model(self) -> Any:  # pragma: no cover - interface
        ...

    @abstractmethod
    def get_feature_names(self) -> list[str]:  # pragma: no cover - interface
        ...

    @abstractmethod
    def get_training_metadata(self) -> dict[str, Any]:  # pragma: no cover - interface
        ...

    def save_for_production(
        self,
        path: str | Path,
        format: str = "auto",
        include_metadata: bool = True,
        opset: int = DEFAULT_ONNX_OPSET,
    ) -> Path:
        """
        Save model in production-ready format, delegating to unified helpers.
        """
        model = self.get_model()
        feature_names = self.get_feature_names()

        training_metadata = self.get_training_metadata() if include_metadata else {}
        training_metadata["feature_names"] = feature_names
        training_metadata["trainer_class"] = self.__class__.__name__

        # Determine format
        if format == "auto":
            model_type = detect_model_type(model)
            if model_type in {ModelType.XGBOOST, ModelType.LIGHTGBM}:
                format = "native"
            else:
                format = "onnx"

        save_path = Path(path)
        if format == "onnx":
            n_features = len(feature_names)
            sample_input = np.random.randn(1, n_features).astype(np.float32)
            return convert_to_onnx(
                model=model,
                sample_input=sample_input,
                output_path=save_path,
                opset_version=opset,
            )
        else:
            return save_model_with_metadata(
                model=model,
                path=save_path,
                input_shape=(1, len(feature_names)),
                training_metadata=training_metadata,
                force_pickle=False,
            )

    def validate_inference_compatibility(
        self,
        model_path: str | Path,
        test_features: NDArray[np.float32] | None = None,
    ) -> bool:
        """
        Validate that a saved ONNX model can be loaded and run.
        """
        try:
            model_path = Path(model_path)
            if model_path.suffix.lower() != ".onnx":
                return False

            # Ensure dependencies
            if not HAS_ONNX:
                from ml._imports import check_ml_dependencies as _check

                _check(["onnx"])

            from ml._imports import ort

            session = ort.InferenceSession(str(model_path))
            # Basic metadata check
            _ = [i.name for i in session.get_inputs()]
            _ = [o.name for o in session.get_outputs()]

            if test_features is not None:
                input_name = session.get_inputs()[0].name
                outputs = session.run(None, {input_name: test_features.astype(np.float32)})
                if outputs is None:
                    return False
            return True
        except Exception:
            return False


class TrainingActorContract(ABC):
    """
    Contract ensuring training outputs are compatible with inference actors.
    """

    @abstractmethod
    def get_required_features(self) -> list[str]:  # pragma: no cover - interface
        ...

    @abstractmethod
    def get_model_input_shape(self) -> tuple[int, ...]:  # pragma: no cover - interface
        ...

    @abstractmethod
    def export_for_actor(
        self,
        actor_model_path: str | Path,
        actor_config_path: str | Path | None = None,
    ) -> dict[str, Any]:  # pragma: no cover - interface
        ...

    def generate_actor_config(self) -> dict[str, Any]:  # pragma: no cover - stub
        """Placeholder: example configuration for MLSignalActor."""
        return {
            "model_path": "path/to/model.onnx",  # To be filled by caller
            "feature_config": {
                "indicators": {},
                "lookback_window": 20,
                "normalize_features": True,
            },
            "prediction_threshold": 0.5,
            "warm_up_period": 50,
        }
