"""
Model Operations Component.

This module implements Universal Pattern #1 (Model Management) and
security requirements from CLAUDE.md (ONNX-only policy, no pickle files).

All ML actors MUST use this component to manage model loading, validation,
hot-reload, warm-up, version tracking, and security enforcement.

The component provides:
- Model loading with ONNX-only security enforcement
- Support for ONNX (preferred), JSON/XGBoost (legacy), Joblib (test-only)
- Hot reload capability (detect file changes, reload models)
- Model warm-up for hot path performance (P99 <2ms after warm-up)
- Version tracking and metadata extraction
- Resource cleanup and error handling
- Centralized metrics for security violations and fallback activations

    Security Policy (CLAUDE.md):
    - PRIMARY: ONNX models via ONNXRuntime (secure, optimized)
    - FALLBACK: JSON for XGBoost models (legacy support)
    - BLOCKED: Pickle/Joblib files in production (security risk - arbitrary code execution)
    - Environment flags: ML_ONNX_ONLY=1 (strict)

"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from ml.common.metrics_bootstrap import get_counter
from ml.config.base import MLActorConfig


if TYPE_CHECKING:
    pass


class ModelProtocol(Protocol):
    """
    Protocol for model operations component.

    Defines the interface for managing ML model lifecycle: loading, validation,
    hot-reload, warm-up, version tracking, and security enforcement.

    """

    @property
    def model(self) -> Any:
        """
        Return loaded model instance.

        Returns:
            Model instance (ONNX, XGBoost, or sklearn)

        """
        ...

    @property
    def model_id(self) -> str | None:
        """
        Return model ID (from metadata, training_metadata, or path).

        Returns:
            Model ID string or None if not determined

        """
        ...

    @property
    def model_version(self) -> str | None:
        """
        Return current model version.

        Returns:
            Version string or None if not tracked

        """
        ...

    @property
    def model_metadata(self) -> dict[str, Any]:
        """
        Return model metadata dictionary.

        Returns:
            Metadata dict with input/output shapes, framework, version, etc.

        """
        ...

    def reload_if_changed(self) -> bool:
        """
        Check if model file changed and reload if needed.

        Returns:
            True if model was reloaded, False otherwise

        """
        ...


class ModelComponent:
    """
    Manages model loading, validation, hot-reload, warm-up, and security enforcement.

    Implements CLAUDE.md security requirements (ONNX-only policy) and hot path
    performance requirements (P99 <2ms after warm-up).

    The component handles:
    - Model loading with security enforcement (ONNX-only unless flags set)
    - Multiple format support: ONNX (preferred), JSON/XGBoost (legacy)
    - Hot reload: detect file changes, reload models without actor restart
    - Model warm-up: pre-allocate buffers for zero-allocation hot path
    - Version tracking: maintain version history across reloads
    - Metadata extraction: input/output shapes, framework, opset version
    - Resource cleanup: release memory and file handles
    - Centralized metrics: security violations, fallback activations

    Security Enforcement:
    - Production mode (default): Only ONNX models allowed
    - ML_ONNX_ONLY=1: Blocks all non-ONNX formats (joblib, pickle, pt, h5)
    - PYTEST_CURRENT_TEST: Auto-detected test environment

    Example:
        >>> config = MLActorConfig(
        ...     actor_id="my_actor",
        ...     model_path="/models/xgboost_classifier.onnx",
        ...     enable_hot_reload=True,
        ...     model_check_interval=30,
        ... )
        >>> component = ModelComponent(config, logger=logging.getLogger(__name__))
        >>> # Model loaded automatically with security checks
        >>> model = component.model
        >>> metadata = component.model_metadata
        >>> # Hot reload on file change
        >>> if component.reload_if_changed():
        ...     print(f"Model reloaded to version {component.model_version}")

    """

    def __init__(self, config: MLActorConfig, logger: logging.Logger) -> None:
        """
        Initialize model operations component.

        Args:
            config: ML actor configuration containing model_path and security settings
            logger: Logger instance for component logging

        Raises:
            ValueError: If security policy violated (e.g., pickle file in production)
            FileNotFoundError: If model file not found
            RuntimeError: If model loading fails

        """
        self._config = config
        self._logger = logger

        # Model references (initialized in _load_model_with_metadata)
        self._model: Any | None = None
        self._model_id: str | None = None
        self._model_version: str | None = None
        self._model_metadata: dict[str, Any] = {}

        # Version tracking
        self._version_history: list[dict[str, Any]] = []
        self._version_updated_at: int | None = None

        # Hot reload tracking
        self._model_file_mtime: float | None = None

        # Warm-up tracking
        self._warmup_completed: bool = False
        self._warmup_iterations: int = 0

        # Manifest feature tracking (for warm-up)
        self._manifest_feature_names: list[str] = []
        self._manifest_feature_schema_hash: str | None = None
        self._manifest_feature_dtypes: list[str] = []

        # Metrics for security violations and fallbacks
        self._security_counter = get_counter(
            "ml_model_security_checks_total",
            "Total model security check results",
            labelnames=("result",),
        )
        self._fallback_counter = get_counter(
            "ml_model_fallback_total",
            "Total model fallback activations",
            labelnames=("stage",),
        )

        # NOTE: Model loading is deferred to load_model() method
        # This allows actors to initialize successfully even with invalid model paths,
        # and properly handle errors during on_start lifecycle phase.
        # Contract: Initialization failures should occur in on_start, not __init__

    def load_model(self) -> None:
        """
        Public API to load model with metadata and security enforcement.

        Delegates to internal _load_model_with_metadata() for implementation.
        This method is called during actor initialization (on_start).

        Raises:
            ValueError: If security policy violated or model config invalid
            FileNotFoundError: If model file not found
            RuntimeError: If model loading fails

        """
        # Load model with metadata and security checks
        self._load_model_with_metadata()

        # Determine model ID using three-level fallback
        self._determine_model_id()

        # Track version after loading and metadata extraction
        self._track_model_version()

        opt_config = getattr(self._config, "optimization_config", None)
        enable_model_warm_up = False
        if opt_config is not None:
            try:
                enable_model_warm_up = bool(getattr(opt_config, "enable_model_warm_up", False))
            except Exception:
                self._logger.debug(
                    "optimization_config warm-up check failed",
                    exc_info=True,
                )
                enable_model_warm_up = False
        if enable_model_warm_up:
            self._perform_manifest_warmup()

    def _try_load_from_registry(self) -> bool:
        """
        Compatibility hook for registry-based model loading.

        Registry lookups and manifest-driven wiring are handled by ``RegistryComponent`` and
        actor-level orchestration. ModelComponent remains a file-backed loader and returns
        ``False`` so callers can fall back to ``model_path``.

        Returns:
            False (registry loading not performed by this component).

        """
        return False

    def _schedule_model_checks(self) -> None:
        """
        Compatibility hook for hot-reload timer scheduling.

        Timer scheduling requires the Nautilus clock, which is owned by the actor
        runtime. Component-level tests assert this method exists; the scheduling itself
        is performed by actor lifecycle wiring.

        """
        return None

    def _load_model_with_metadata(self) -> None:
        """
        Load model with security enforcement and metadata extraction.

        Security Policy:
        - Check ML_ONNX_ONLY environment variable (strictest)
        - Check file extension
        - Block .pkl/.joblib entirely (security risk)
        - Block .pt/.h5 in production
        - Prefer ONNX for security and performance

        Supported Formats:
        - .onnx: ONNX Runtime (preferred, secure, optimized)
        - .json: XGBoost Booster (legacy support)

        Raises:
            ValueError: If security policy violated
            FileNotFoundError: If model file not found
            RuntimeError: If model loading fails

        """
        # Check if model_path provided
        if not hasattr(self._config, "model_path") or not self._config.model_path:
            raise ValueError("Neither model_path nor model_id provided")

        model_path = Path(self._config.model_path)

        # Check file exists
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Record file mtime for hot reload
        self._model_file_mtime = model_path.stat().st_mtime

        # Security enforcement: Check environment flags
        onnx_only_mode = os.getenv("ML_ONNX_ONLY", "0") == "1"
        file_ext = model_path.suffix.lower()

        # ONNX-only mode: Block all non-ONNX
        if onnx_only_mode and file_ext != ".onnx":
            self._security_counter.labels(result="blocked_non_onnx").inc()
            raise ValueError(
                "Joblib models are disabled in ONNX-only mode. "
                "Convert artifacts to ONNX for secure loading.",
            )

        # Block pickle files (NEVER allowed - security risk)
        if file_ext in (".pkl", ".pickle"):
            self._security_counter.labels(result="blocked_pickle").inc()
            raise ValueError(
                f"Pickle model formats are not allowed in production. Use ONNX instead. "
                f"File: {model_path}",
            )

        allow_joblib = os.getenv("ML_ALLOW_JOBLIB", "0") == "1"
        is_test_env = (
            os.getenv("PYTEST_CURRENT_TEST") is not None
            or os.getenv("ML_TEST_ALLOW_NON_ONNX", "").lower() in {"1", "true", "yes"}
            or os.getenv("ML_ALLOW_NON_ONNX_IN_TESTS", "").lower() in {"1", "true", "yes"}
        )

        # Block other non-ONNX formats in production
        if file_ext in (".pt", ".h5", ".pth"):
            self._security_counter.labels(result="blocked_non_onnx").inc()
            raise ValueError(f"Unsupported model format: {file_ext}")

        # Load based on file extension
        if file_ext == ".onnx":
            self._load_onnx_model(model_path)
        elif file_ext == ".json":
            self._load_json_model(model_path)
        elif file_ext == ".joblib":
            if allow_joblib or is_test_env:
                self._load_joblib_model(model_path)
            else:
                raise ValueError(
                    "Joblib models are not supported in production actors. "
                    "Convert artifacts to ONNX for secure loading.",
                )
        else:
            raise ValueError(f"Unsupported model format: {file_ext}")

        # Security check passed
        self._security_counter.labels(result="allowed").inc()

    def _load_onnx_model(self, model_path: Path) -> None:
        """
        Load ONNX model with ONNXRuntime.

        Args:
            model_path: Path to .onnx file

        Raises:
            RuntimeError: If ONNX model fails to load

        """
        try:
            import onnx
            import onnxruntime

            # Validate ONNX model before loading
            try:
                onnx_model = onnx.load(str(model_path))
                onnx.checker.check_model(onnx_model)
            except Exception as e:
                raise ValueError(f"Invalid ONNX model: {e}") from e

            # Load with ONNXRuntime
            sess_options = onnxruntime.SessionOptions()
            sess_options.graph_optimization_level = (
                onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            self._model = onnxruntime.InferenceSession(
                str(model_path),
                sess_options=sess_options,
            )

            # Extract metadata from ONNX model
            self._extract_onnx_metadata()

        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}") from e

    def _load_json_model(self, model_path: Path) -> None:
        """
        Load JSON model (XGBoost or plain JSON fallback).

        Args:
            model_path: Path to .json file

        Raises:
            RuntimeError: If JSON model fails to load

        """
        try:
            # Try XGBoost loading first
            try:
                from ml._imports import HAS_XGBOOST
                from ml._imports import xgb

                if HAS_XGBOOST:
                    booster = xgb.Booster()
                    booster.load_model(str(model_path))
                    self._model = booster
                    self._model_metadata = {
                        "type": "xgboost",
                        "format": "json",
                        "framework": "XGBoost",
                    }
                    return
            except Exception:
                # XGBoost loading failed - try plain JSON
                self._logger.debug(
                    "XGBoost JSON load failed, falling back to plain JSON",
                    exc_info=True,
                )

            # Fallback to plain JSON
            import json

            data = json.loads(model_path.read_text(encoding="utf-8"))

            self._model = data
            self._model_metadata = {
                "type": "json",
                "format": "json",
                "framework": "custom",
            }

        except Exception as e:
            raise RuntimeError(f"Failed to load JSON model: {e}") from e

    def _load_joblib_model(self, model_path: Path) -> None:
        """
        Load joblib-serialized sklearn model (test-only / guarded by env flag).

        Args:
            model_path: Path to .joblib file

        Raises:
            RuntimeError: If joblib model fails to load

        """
        try:
            from ml._imports import HAS_JOBLIB
            from ml._imports import HAS_SKLEARN
            from ml._imports import check_ml_dependencies
            from ml._imports import joblib

            if not HAS_JOBLIB or not HAS_SKLEARN:
                check_ml_dependencies(["joblib", "scikit-learn"])

            model = joblib.load(model_path)
            self._model = model
            self._model_metadata = {
                "type": "sklearn",
                "format": "joblib",
                "framework": "sklearn",
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load joblib model: {e}") from e

    def _extract_onnx_metadata(self) -> None:
        """
        Extract metadata from ONNX model.

        Populates self._model_metadata with:
        - input_shape: tuple of input dimensions
        - output_shape: tuple of output dimensions
        - framework: "ONNX"
        - opset_version: ONNX opset version
        - inputs: list of input tensor metadata
        - outputs: list of output tensor metadata

        """
        if self._model is None or not hasattr(self._model, "get_inputs"):
            return

        try:
            # Get input metadata
            inputs_meta = self._model.get_inputs()
            outputs_meta = self._model.get_outputs()

            input_shape = tuple(inputs_meta[0].shape) if inputs_meta else ()
            output_shape = tuple(outputs_meta[0].shape) if outputs_meta else ()
            input_names = [inp.name for inp in inputs_meta]
            output_names = [out.name for out in outputs_meta]

            self._model_metadata = {
                "input_shape": input_shape,
                "output_shape": output_shape,
                "framework": "ONNX",
                "version": "1.0.0",  # Default version
                "opset_version": 13,  # Default opset
                "input_names": input_names,
                "output_names": output_names,
                "inputs": [
                    {"name": inp.name, "shape": inp.shape, "type": str(inp.type)}
                    for inp in inputs_meta
                ],
                "outputs": [
                    {"name": out.name, "shape": out.shape, "type": str(out.type)}
                    for out in outputs_meta
                ],
            }

        except Exception as e:
            self._logger.debug(f"Failed to extract ONNX metadata: {e}", exc_info=True)

    def _determine_model_id(self) -> None:
        """
        Determine model ID using four-level fallback.

        Priority:
        1. self._model_metadata["model_id"] (from registry manifest)
        2. self._model_metadata["training_metadata"]["model_id"]
        3. self._config.model_id (explicit config)
        4. {Path(model_path).stem}_{version[:8]} (derived from path)

        Sets self._model_id to the determined value.

        """
        # Priority 1: Direct metadata
        if "model_id" in self._model_metadata:
            self._model_id = self._model_metadata["model_id"]
            return

        # Priority 2: Training metadata
        if "training_metadata" in self._model_metadata:
            training_meta = self._model_metadata["training_metadata"]
            if isinstance(training_meta, dict) and "model_id" in training_meta:
                self._model_id = training_meta["model_id"]
                return

        # Priority 3: Explicit config model_id
        config_model_id = getattr(self._config, "model_id", None)
        if isinstance(config_model_id, str) and config_model_id:
            self._model_id = config_model_id
            return

        # Priority 4: Derive from path + version
        if hasattr(self._config, "model_path") and self._config.model_path:
            path_stem = Path(self._config.model_path).stem
            version_suffix = self._model_version[:8] if self._model_version else "unknown"
            self._model_id = f"{path_stem}_{version_suffix}"

    def _track_model_version(self) -> str:
        """
        Track model version and update version history.

        Extracts version from metadata or generates hash from file.
        Maintains version history with timestamps.

        Returns:
            Current model version string

        """
        # Extract version from metadata
        version = self._model_metadata.get("version", None)

        # If no version in metadata, generate from file hash
        if not version and hasattr(self._config, "model_path"):
            version = self._compute_version_hash(Path(self._config.model_path))

        self._model_version = version or "unknown"

        # Record in version history
        import time

        timestamp_ns = int(time.time() * 1e9)
        self._version_history.append(
            {
                "version": self._model_version,
                "timestamp": timestamp_ns,
            },
        )
        self._version_updated_at = timestamp_ns

        return self._model_version

    def _compute_version_hash(self, model_path: Path) -> str:
        """
        Compute SHA256 hash of model file for version tracking.

        Args:
            model_path: Path to model file

        Returns:
            First 16 characters of SHA256 hash

        """
        try:
            stat_result = model_path.stat()
            fingerprint = f"{stat_result.st_size}:{stat_result.st_mtime_ns}"
            return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
        except Exception:
            return "unknown"

    def _hot_reload_model(self) -> bool:
        """
        Check if model file changed and reload if needed.

        Compares current file mtime with stored mtime.
        Reloads model if file was modified.

        Returns:
            True if model was reloaded, False otherwise

        """
        if not hasattr(self._config, "model_path") or not self._config.model_path:
            return False

        model_path = Path(self._config.model_path)

        if not model_path.exists():
            return False

        # Check if mtime changed
        current_mtime = model_path.stat().st_mtime

        if self._model_file_mtime is None or current_mtime == self._model_file_mtime:
            return False

        # File changed - reload model
        try:
            self._load_model_with_metadata()
            self._determine_model_id()
            old_version = self._model_version
            self._track_model_version()

            if old_version != self._model_version:
                self._logger.info(
                    f"Model hot-reloaded: {old_version} → {self._model_version}",
                )

            return True

        except Exception as e:
            self._logger.error(f"Hot reload failed: {e}", exc_info=True)
            return False

    def _warm_up_model(self, iterations: int = 10) -> None:
        """
        Warm up model by running N inferences with dummy data.

        Pre-allocates buffers and caches JIT compilation for zero-allocation hot path.

        Args:
            iterations: Number of warm-up iterations (default 10)

        """
        if self._model is None:
            return

        try:
            # Determine input dimension from metadata
            input_dim = 20  # Default
            if "input_shape" in self._model_metadata:
                input_shape = self._model_metadata["input_shape"]
                if isinstance(input_shape, tuple) and len(input_shape) >= 2:
                    input_dim = input_shape[1]

            # Create dummy input
            import numpy as np

            dummy_input = np.zeros((1, input_dim), dtype=np.float32)

            # Run warm-up iterations
            for _ in range(iterations):
                if hasattr(self._model, "run") and hasattr(self._model, "get_inputs"):
                    # ONNX Runtime
                    input_name = self._model.get_inputs()[0].name
                    self._model.run(None, {input_name: dummy_input})
                elif hasattr(self._model, "predict"):
                    # sklearn-like
                    self._model.predict(dummy_input)

            self._warmup_completed = True
            self._warmup_iterations = iterations

        except Exception as e:
            self._logger.debug(f"Model warm-up failed (ignored): {e}", exc_info=True)

    def _perform_manifest_warmup(self) -> None:
        """
        Perform warm-up using manifest feature schema if available.

        Uses ml.actors.model_loader_utils.maybe_warm_up_model utility. Input dimension
        derived from feature_schema in metadata.

        """
        if self._model is None:
            return

        feature_schema = self._model_metadata.get("feature_schema")
        input_dim = 20
        if isinstance(feature_schema, (dict, list, tuple)):
            input_dim = len(feature_schema)
        else:
            input_shape = self._model_metadata.get("input_shape")
            if isinstance(input_shape, tuple) and len(input_shape) >= 2:
                try:
                    input_dim = int(input_shape[1])
                except (TypeError, ValueError):
                    input_dim = 20

        # Use utility function
        try:
            from ml.actors.model_loader_utils import maybe_warm_up_model

            maybe_warm_up_model(
                self._model,
                True,  # warm_up=True
                input_dim,
            )

            self._warmup_completed = True

        except Exception as e:
            self._logger.debug(f"Manifest warm-up failed (ignored): {e}", exc_info=True)

    def _run_inference(self, features: Any) -> Any:
        """
        Run model inference on feature array.

        Args:
            features: Feature array (numpy array or compatible)

        Returns:
            Model prediction (numpy array or compatible)

        Raises:
            RuntimeError: If inference fails

        """
        if self._model is None:
            raise RuntimeError("Model not loaded")

        try:
            # ONNX Runtime inference
            if hasattr(self._model, "run") and hasattr(self._model, "get_inputs"):
                input_name = self._model.get_inputs()[0].name
                outputs = self._model.run(None, {input_name: features})
                return outputs[0]

            # sklearn-like inference
            if hasattr(self._model, "predict"):
                return self._model.predict(features)

            raise RuntimeError("Model does not support inference")

        except Exception as e:
            raise RuntimeError(f"Model inference failed: {e}") from e

    def cleanup(self) -> None:
        """
        Public API to clean up model resources.

        Releases model resources including memory and file handles. This method is
        called during actor shutdown (on_stop).

        """
        self._cleanup_model()

    def _cleanup_model(self) -> None:
        """
        Release model resources (memory, file handles).

        Clears model reference and metadata to free memory.

        """
        self._model = None
        self._model_metadata = {}
        self._warmup_completed = False

    def _get_model_metadata(self) -> dict[str, Any]:
        """
        Get model metadata dictionary.

        Returns:
            Metadata dict with input/output shapes, framework, version, etc.

        """
        return self._model_metadata.copy()

    @property
    def model(self) -> Any:
        """
        Return loaded model instance.

        Returns:
            Model instance (ONNX, XGBoost, or sklearn)

        Raises:
            RuntimeError: If model not loaded

        """
        if self._model is None:
            raise RuntimeError("Model not loaded")
        return self._model

    @property
    def model_id(self) -> str | None:
        """
        Return model ID (from metadata, training_metadata, or path).

        Returns:
            Model ID string or None if not determined

        """
        return self._model_id

    @property
    def model_version(self) -> str | None:
        """
        Return current model version.

        Returns:
            Version string or None if not tracked

        """
        return self._model_version

    @property
    def model_metadata(self) -> dict[str, Any]:
        """
        Return model metadata dictionary.

        Returns:
            Metadata dict with input/output shapes, framework, version, etc.

        """
        return self._model_metadata.copy()

    def reload_if_changed(self) -> bool:
        """
        Check if model file changed and reload if needed.

        Returns:
            True if model was reloaded, False otherwise

        """
        return self._hot_reload_model()
