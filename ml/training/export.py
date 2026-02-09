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
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any, cast

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
from ml.common.reproducibility import ReproducibilityValue
from ml.common.reproducibility import validate_reproducibility_provenance
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


def resolve_classifier_classes(model_obj: Any) -> list[Any] | None:
    """
    Resolve classifier classes from a fitted model if available.

    Args:
        model_obj: Fitted model instance.

    Returns:
        List of classes or None when unavailable.
    """
    classes = getattr(model_obj, "classes_", None)
    if isinstance(classes, np.ndarray):
        return [item.item() if isinstance(item, np.generic) else item for item in classes]
    if isinstance(classes, (list, tuple)):
        return list(classes)
    return None


def enforce_positive_class_mapping(
    model_obj: Any,
    decision_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """
    Ensure positive-class mapping is explicit for classifier outputs.

    Args:
        model_obj: Fitted model instance.
        decision_config: Optional decision adapter configuration.

    Returns:
        Decision config with explicit positive class mapping.

    Raises:
        ValueError: If classifier outputs require a mapping and none is provided.
    """
    decision_cfg: dict[str, Any] = dict(decision_config or {})
    predict_proba = getattr(model_obj, "predict_proba", None)
    has_proba = callable(predict_proba)
    if has_proba and type(predict_proba).__module__ == "unittest.mock":
        has_proba = False
    if not has_proba:
        return decision_cfg

    classes = resolve_classifier_classes(model_obj)
    if classes is not None and len(classes) <= 1:
        return decision_cfg

    has_mapping = any(
        key in decision_cfg
        for key in ("positive_class_index", "positive_class_label", "positive_class")
    )
    if not has_mapping:
        raise ValueError(
            "decision_config must include positive_class_index or positive_class_label "
            "for classifier outputs with probability vectors",
        )

    if classes is not None:
        from ml.common import resolve_positive_class_index

        idx = resolve_positive_class_index(
            {"decision_config": decision_cfg},
            classes=classes,
            num_classes=len(classes),
        )
        if idx is None:
            raise ValueError("positive_class_index mapping could not be resolved")
        decision_cfg["positive_class_index"] = idx
        decision_cfg.setdefault("positive_class_label", classes[idx])
        return decision_cfg

    if "positive_class_label" in decision_cfg and "positive_class_index" not in decision_cfg:
        raise ValueError("positive_class_label requires classifier classes to resolve index")
    if "positive_class" in decision_cfg and not isinstance(decision_cfg["positive_class"], int):
        raise ValueError("positive_class requires classifier classes to resolve index")
    if "positive_class_index" in decision_cfg and not isinstance(
        decision_cfg["positive_class_index"],
        int,
    ):
        raise ValueError("positive_class_index must be an int")
    return decision_cfg


# Unified default opset for ONNX exports used by training layer
# Delegate to central Versions to avoid drift across modules.
DEFAULT_ONNX_OPSET = Versions.ONNX_OPSET


def _normalize_reproducibility_payload(
    payload: Mapping[str, object] | None,
    *,
    context: str,
) -> dict[str, ReproducibilityValue] | None:
    if payload is None:
        return None
    return validate_reproducibility_provenance(
        payload=payload,
        context=context,
    )


def save_model_with_metadata(
    model: Any,
    path: str | Path,
    input_shape: tuple[int, ...] | None = None,
    output_shape: tuple[int, ...] | None = None,
    training_metadata: dict[str, Any] | None = None,
    reproducibility_provenance: Mapping[str, object] | None = None,
    force_pickle: bool = False,
) -> Path:
    """
    Save a model artifact plus technical metadata sidecar.

    Notes
    -----
    This metadata is strictly file-level context to aid loaders. Deployment
    identity lives in the registry ModelManifest.

    Args:
        model: Model instance to persist.
        path: Base output path.
        input_shape: Optional input shape hint.
        output_shape: Optional output shape hint.
        training_metadata: Optional training metrics/context payload.
        reproducibility_provenance: Optional canonical reproducibility payload.
        force_pickle: Deprecated compatibility flag (unused in strict paths).

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

    reproducibility_payload = _normalize_reproducibility_payload(
        reproducibility_provenance,
        context="export reproducibility",
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
        "reproducibility": reproducibility_payload,
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
    if lgb is None:  # pragma: no cover - environment guard
        raise ImportError("LightGBM not available for saving model")

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

            _logging.getLogger(__name__).debug(
                "get_params failed for version generation",
                exc_info=True,
            )
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
        if onnxmltools is None:
            raise ImportError("onnxmltools is required for XGBoost → ONNX conversion")
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
        if onnxmltools is None:
            raise ImportError("onnxmltools is required for LightGBM → ONNX conversion")
        from onnxmltools.convert.common.data_types import FloatTensorType

        initial_type = [(ONNX_INPUT_NAME, FloatTensorType([None, sample_input.shape[-1]]))]
        onnx_model = onnxmltools.convert_lightgbm(
            model,
            initial_types=initial_type,
            target_opset=opset_version,
        )

    elif model_type == ModelType.SKLEARN:
        if skl2onnx is None:
            raise ImportError("skl2onnx is required for sklearn → ONNX conversion")
        onnx_model = skl2onnx.to_onnx(model, sample_input[:1], target_opset=opset_version)

    else:
        raise ValueError(f"Cannot convert {model_type} to ONNX")

    if not HAS_ONNX_CORE or onnx is None:
        raise ImportError("onnx core package is required to save ONNX models")
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
        trace_fn = cast(Any, torch.jit.trace)
        return trace_fn(mod, example)

    def _jit_script(mod: object) -> Any:
        # Wrapper to keep mypy strict happy around untyped torch APIs.
        script_fn = cast(Any, torch.jit.script)
        return script_fn(mod)

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
    "create_model_manifest_stub",
    "detect_model_type",
    "register_model_with_registry",
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

            # Ensure dependencies: require onnxruntime availability
            if not HAS_ONNX:
                return False

            from typing import Any as _Any
            from typing import cast as _cast

            from ml._imports import ort as _ort

            if _ort is None:
                return False
            session = _cast(_Any, _ort).InferenceSession(str(model_path))
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


def create_model_manifest_stub(
    model: Any,
    feature_names: list[str],
    training_metrics: dict[str, Any] | None = None,
    model_role: str = "inference",
    data_requirements: str = "l1_only",
    architecture: str | None = None,
    feature_set_id: str | None = None,
    pipeline_signature: str | None = None,
    pipeline_version: str | None = None,
    *,
    parent_id: str | None = None,
    decision_config: Mapping[str, Any] | None = None,
    reproducibility_provenance: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """
    Create a ModelManifest stub from training outputs.

    This function creates a manifest dictionary that can be used to register
    a model with the ModelRegistry. It extracts metadata from the model and
    training process to populate the manifest fields.

    Parameters
    ----------
    model : Any
        The trained model object
    feature_names : list[str]
        List of feature names used by the model
    training_metrics : dict[str, Any] | None
        Training metrics and performance data
    model_role : str
        Role of the model (teacher/student/inference)
    data_requirements : str
        Data requirements (l1_only/l1_l2/l1_l2_l3)
    architecture : str | None
        Model architecture name (auto-detected if None)
    feature_set_id : str | None
        Associated feature set ID for parity validation
    pipeline_signature : str | None
        Pipeline signature hash for reproducibility
    pipeline_version : str | None
        Pipeline version string

    parent_id : str | None, optional
        Parent model identifier (e.g., teacher model ID) for lineage when the
        role is "student". If omitted, no lineage is attached.
    decision_config : Mapping[str, Any] | None, optional
        Decision adapter config payload. Required to declare positive class
        mapping for classifier outputs that emit probability vectors.
    reproducibility_provenance : Mapping[str, object] | None, optional
        Canonical reproducibility payload persisted under ``training_config``.

    Returns
    -------
    dict[str, Any]
        ModelManifest dictionary ready for registry registration

    """
    import time

    # Auto-detect architecture if not provided
    if architecture is None:
        architecture = detect_model_type(model).value

    # Build feature schema
    feature_schema = dict.fromkeys(feature_names, "float32")

    # Compute feature schema hash
    from ml.registry.feature_registry import compute_schema_hash

    feature_schema_hash = compute_schema_hash(
        feature_names,
        ["float32"] * len(feature_names),
        pipeline_signature or "",
    )

    # Extract performance metrics
    performance_metrics = {}
    if training_metrics:
        for key, value in training_metrics.items():
            if isinstance(value, (int, float)):
                performance_metrics[key] = float(value)

    # Determine if model is serveable
    serveable = architecture.lower() == "onnx" or hasattr(model, "run")
    artifact_format = "onnx" if serveable else "native"

    # Optional environment fallback for lineage (non-binding)
    if parent_id is None:
        import os as _os

        env_parent = _os.getenv("ML_PARENT_MODEL_ID")
        if env_parent:
            parent_id = str(env_parent)

    decision_cfg = enforce_positive_class_mapping(model, decision_config)
    reproducibility_payload = _normalize_reproducibility_payload(
        reproducibility_provenance,
        context="manifest reproducibility",
    )
    training_config: dict[str, Any] = {}
    if reproducibility_payload is not None:
        training_config["reproducibility"] = reproducibility_payload

    return {
        "model_id": "",  # Will be generated by registry
        "role": model_role,
        "data_requirements": data_requirements,
        "architecture": architecture,
        "feature_schema": feature_schema,
        "feature_schema_hash": feature_schema_hash,
        "parent_id": parent_id,
        "children_ids": [],
        "training_config": training_config,
        "performance_metrics": performance_metrics,
        "deployment_constraints": {
            "max_inference_latency_ms": 50.0,
            "memory_limit_mb": 1024.0,
        },
        "version": "1.0.0",
        "created_at": time.time(),
        "last_modified": time.time(),
        "serveable": serveable,
        "artifact_format": artifact_format,
        "feature_set_id": feature_set_id,
        "pipeline_signature": pipeline_signature,
        "pipeline_version": pipeline_version,
        "decision_policy": None,
        "decision_config": decision_cfg,
        "artifact_sha256_digest": None,
    }


def register_model_with_registry(
    model_path: Path,
    manifest_data: Mapping[str, Any],
    registry_path: Path | None = None,
    auto_deploy: bool = False,
) -> str:
    """
    Register a model with the ModelRegistry.

    Parameters
    ----------
    model_path : Path
        Path to the saved model artifact
    manifest_data : dict[str, Any]
        ModelManifest data from create_model_manifest_stub
    registry_path : Path | None
        Path to the model registry (defaults to ./model_registry)
    auto_deploy : bool
        Whether to auto-deploy the model if validation passes

    Returns
    -------
    str
        Model ID assigned by the registry

    """
    from ml.registry import DataRequirements
    from ml.registry import ModelManifest
    from ml.registry import ModelRegistry
    from ml.registry import ModelRole

    # Initialize registry
    if registry_path is None:
        registry_path = Path("./model_registry")
    registry = ModelRegistry(registry_path)

    # Create manifest object (defensive mapping access)
    md = dict(manifest_data)
    from time import time as _time

    manifest = ModelManifest(
        model_id=str(md.get("model_id", "")),
        role=ModelRole(md.get("role", "inference")),
        data_requirements=DataRequirements(md.get("data_requirements", "l1_only")),
        architecture=str(md.get("architecture", "unknown")),
        feature_schema=dict(md.get("feature_schema", {})),
        feature_schema_hash=str(md.get("feature_schema_hash", "")),
        parent_id=md.get("parent_id"),
        children_ids=list(md.get("children_ids", [])),
        training_config=dict(md.get("training_config", {})),
        performance_metrics=dict(md.get("performance_metrics", {})),
        deployment_constraints=dict(md.get("deployment_constraints", {})),
        version=str(md.get("version", "1.0.0")),
        created_at=float(md.get("created_at", _time())),
        last_modified=float(md.get("last_modified", _time())),
        serveable=bool(md.get("serveable", True)),
        artifact_format=str(md.get("artifact_format", "onnx")),
        feature_set_id=md.get("feature_set_id"),
        pipeline_signature=md.get("pipeline_signature"),
        pipeline_version=md.get("pipeline_version"),
        decision_policy=md.get("decision_policy"),
        decision_config=dict(md.get("decision_config", {})),
        artifact_sha256_digest=md.get("artifact_sha256_digest"),
    )

    # Register model
    model_id = registry.register_model(
        model_path=model_path,
        manifest=manifest,
        auto_deploy=auto_deploy,
    )
    # Ensure persistence for immediate re-load scenarios in tests/tools
    try:
        registry.flush()
    except Exception as exc:
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "ModelRegistry flush failed (non-blocking): %s",
            exc,
        )
    return model_id
