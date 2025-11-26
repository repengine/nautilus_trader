"""
Persistence component for BaseMLTrainer decomposition.

This module provides the PersistenceComponent which encapsulates model persistence
logic from BaseMLTrainer (lines 1125-1161, 1265-1494), including:
- ONNX model export
- Model saving with registry integration
- Model loading from registry or file
- Model manifest creation
- Feature importance extraction

Following Universal ML Architecture Pattern 2: Protocol-First Interface Design.

"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from ml._imports import HAS_ONNX
from ml._imports import check_ml_dependencies


if TYPE_CHECKING:
    import numpy.typing as npt

    from ml.config.base import MLTrainingConfig
    from ml.registry import ModelManifest


logger = logging.getLogger(__name__)


class PersistenceTrainerProtocol(Protocol):
    """
    Protocol for trainer interaction with persistence component.

    Defines the interface that any trainer must implement to work with
    the PersistenceComponent. This follows Protocol-First design
    (Pattern 2) to enable structural typing without inheritance coupling.

    Attributes
    ----------
    _config : MLTrainingConfig
        Training configuration.
    _feature_names : list[str]
        List of feature names used in training.
    _training_metrics : dict[str, Any]
        Dictionary containing training metrics.
    _is_fitted : bool
        Whether the model has been fitted.
    _model : Any
        The trained model object.

    """

    _config: MLTrainingConfig
    _feature_names: list[str]
    _training_metrics: dict[str, Any]
    _is_fitted: bool
    _model: Any

    def _convert_to_onnx(self, model: Any, path: Path) -> None:
        """
        Convert model to ONNX format (model-specific implementation).

        Parameters
        ----------
        model : Any
            Trained model to convert.
        path : Path
            Path to save the ONNX model.

        """
        ...

    def _config_to_dict(self) -> dict[str, Any]:
        """
        Convert config to dictionary for logging/persistence.

        Returns
        -------
        dict[str, Any]
            Configuration as dictionary.

        """
        ...


class PersistenceComponent:
    """
    Component responsible for model persistence operations.

    This component encapsulates persistence logic from BaseMLTrainer
    (lines 1125-1161, 1265-1494), including:
    - ONNX model export
    - Model saving with registry integration
    - Model loading from registry or file
    - Model manifest creation
    - Feature importance extraction

    The component delegates model-specific operations to the trainer instance
    through the PersistenceTrainerProtocol interface.

    Parameters
    ----------
    trainer : PersistenceTrainerProtocol
        The trainer instance that implements the PersistenceTrainerProtocol.

    Example
    -------
    >>> from ml.training.common import PersistenceComponent
    >>> # trainer is an instance implementing PersistenceTrainerProtocol
    >>> persistence = PersistenceComponent(trainer)
    >>> persistence.save_model("/path/to/model.onnx")

    """

    def __init__(self, trainer: PersistenceTrainerProtocol) -> None:
        """
        Initialize the persistence component with a trainer reference.

        Parameters
        ----------
        trainer : PersistenceTrainerProtocol
            The trainer instance for delegation.

        """
        self._trainer = trainer

    def export_to_onnx(self, path: str | Path) -> None:
        """
        Export trained model to ONNX format.

        This method exports the trainer's fitted model to ONNX format for
        production inference. ONNX models can be loaded by ProductionModelLoader
        and used with ONNX Runtime for efficient inference.

        Parameters
        ----------
        path : str | Path
            Path to save the ONNX model.

        Raises
        ------
        ValueError
            If the model has not been fitted.
        ImportError
            If ONNX dependencies are not available.

        Example
        -------
        >>> persistence.export_to_onnx("/models/my_model.onnx")

        """
        if not self._trainer._is_fitted:
            raise ValueError("Model must be fitted before exporting to ONNX")

        if not HAS_ONNX:
            check_ml_dependencies(["onnx"])

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Delegate to trainer's model-specific ONNX conversion
        self._trainer._convert_to_onnx(self._trainer._model, save_path)
        self._log_info(f"Model exported to ONNX: {save_path}")

    def save_model(self, path: str | Path) -> None:
        """
        Save the trained model to disk in production format.

        This method saves the model using the ModelRegistry system, creating
        a model manifest and registering the model for deployment tracking.
        The actual save format depends on the model type (XGBoost JSON,
        LightGBM TXT, ONNX, etc.).

        Parameters
        ----------
        path : str | Path
            Path where to save the model.

        Raises
        ------
        ValueError
            If the model has not been fitted.

        Example
        -------
        >>> persistence.save_model("/models/my_model")

        """
        if not self._trainer._is_fitted:
            raise ValueError("Model must be fitted before saving")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Use registry for saving models
        from ml.registry import ModelRegistry
        from ml.training.export import save_model_with_metadata

        # Determine the registry path from config or use a default
        registry_path = getattr(
            self._trainer._config,
            "registry_path",
            None,
        )
        if registry_path is None:
            registry_path = Path("./model_registry")
        registry = ModelRegistry(registry_path)

        # Create model manifest
        manifest = self._create_model_manifest(save_path)

        # Save model artifact first
        artifact_path = save_model_with_metadata(
            model=self._trainer._model,
            path=save_path,
            training_metadata=self._trainer._training_metrics,
        )

        # Register with registry
        model_id = registry.register_model(
            model_path=artifact_path,
            manifest=manifest,
            auto_deploy=getattr(self._trainer._config, "auto_deploy", False),
        )

        self._log_info(f"Model registered with ID: {model_id} at {artifact_path}")

    def _create_model_manifest(self, save_path: Path) -> ModelManifest:
        """
        Create a model manifest from training metadata.

        This method creates a complete ModelManifest object for registry
        registration, extracting metadata from the trainer's configuration
        and training metrics.

        Parameters
        ----------
        save_path : Path
            Path where the model will be saved.

        Returns
        -------
        ModelManifest
            Complete model manifest for registry registration.

        Example
        -------
        >>> manifest = persistence._create_model_manifest(Path("/models/model.onnx"))
        >>> print(manifest.architecture)

        """
        from ml.registry import DataRequirements
        from ml.registry import ModelManifest
        from ml.registry import ModelRole
        from ml.registry.feature_registry import compute_schema_hash

        # Determine model role based on config
        role = getattr(self._trainer._config, "model_role", ModelRole.INFERENCE)
        if isinstance(role, str):
            role = ModelRole(role)

        # Determine data requirements based on config
        data_requirements = getattr(
            self._trainer._config,
            "data_requirements",
            DataRequirements.L1_ONLY,
        )
        if isinstance(data_requirements, str):
            data_requirements = DataRequirements(data_requirements)

        # Build feature schema from feature names
        feature_schema: dict[str, str] = {}
        if self._trainer._feature_names:
            # Assume float32 for all features unless specified otherwise
            feature_dtypes = getattr(
                self._trainer._config,
                "feature_dtypes",
                None,
            )
            if feature_dtypes is None:
                feature_dtypes = ["float32"] * len(self._trainer._feature_names)
            feature_schema = dict(zip(self._trainer._feature_names, feature_dtypes))

        # Compute feature schema hash
        feature_schema_hash = ""
        if feature_schema:
            pipeline_signature = getattr(self._trainer._config, "pipeline_signature", "")
            feature_schema_hash = compute_schema_hash(
                list(feature_schema.keys()),
                list(feature_schema.values()),
                pipeline_signature,
            )

        # Extract performance metrics
        performance_metrics: dict[str, float] = {}
        if self._trainer._training_metrics:
            # Filter out non-numeric values
            for key, value in self._trainer._training_metrics.items():
                if isinstance(value, (int, float)):
                    performance_metrics[key] = float(value)

        # Determine if model is serveable (ONNX format for hot path)
        serveable = save_path.suffix.lower() == ".onnx" or getattr(
            self._trainer._config,
            "export_onnx",
            False,
        )
        artifact_format = "onnx" if serveable else "native"

        # Get trainer class name for architecture field
        trainer_class = self._trainer.__class__.__name__
        architecture = trainer_class.replace("Trainer", "")

        return ModelManifest(
            model_id="",  # Will be generated by registry
            role=role,
            data_requirements=data_requirements,
            architecture=architecture,
            feature_schema=feature_schema,
            feature_schema_hash=feature_schema_hash,
            parent_id=getattr(self._trainer._config, "parent_model_id", None),
            training_config=self._trainer._config_to_dict(),
            performance_metrics=performance_metrics,
            deployment_constraints={
                "max_inference_latency_ms": getattr(
                    self._trainer._config,
                    "max_inference_latency_ms",
                    50.0,
                ),
                "memory_limit_mb": getattr(
                    self._trainer._config,
                    "memory_limit_mb",
                    1024.0,
                ),
            },
            version=getattr(self._trainer._config, "model_version", "1.0.0"),
            created_at=time.time(),
            last_modified=time.time(),
            serveable=serveable,
            artifact_format=artifact_format,
            feature_set_id=getattr(self._trainer._config, "feature_set_id", None),
            pipeline_signature=getattr(self._trainer._config, "pipeline_signature", None),
            pipeline_version=getattr(self._trainer._config, "pipeline_version", None),
            decision_policy=getattr(self._trainer._config, "decision_policy", None),
            decision_config=getattr(self._trainer._config, "decision_config", {}),
        )

    def load_model(self, path: str | Path) -> None:
        """
        Load a trained model from disk using ProductionModelLoader.

        This method supports loading models by either:
        - Model ID: Looks up the model in the registry
        - File path: Loads directly from the specified file

        Parameters
        ----------
        path : str | Path
            Path to the saved model or model ID in registry.

        Raises
        ------
        ValueError
            If model ID is not found in registry.
        FileNotFoundError
            If model file does not exist.
        RuntimeError
            If model loading fails.

        Example
        -------
        >>> persistence.load_model("/models/my_model.onnx")
        >>> # Or by registry ID
        >>> persistence.load_model("model_abc123")

        """
        # Use registry for loading models by ID or path
        from ml.registry import ModelRegistry

        # Determine the registry path from config or use a default
        registry_path = getattr(
            self._trainer._config,
            "registry_path",
            None,
        )
        if registry_path is None:
            registry_path = Path("./model_registry")
        registry = ModelRegistry(registry_path)

        # Check if path is a model ID or file path
        path_str = str(path)
        if "/" not in path_str and "." not in path_str:
            # Looks like a model ID
            model_info = registry.get_model(path_str)
            if model_info is None:
                raise ValueError(f"Model ID not found in registry: {path_str}")

            # Load model from registry
            model = registry.load_model(path_str)
            if model is None:
                raise RuntimeError(f"Failed to load model {path_str} from registry")

            self._trainer._model = model
            self._trainer._feature_names = list(model_info.manifest.feature_schema.keys())
            self._trainer._training_metrics = model_info.manifest.performance_metrics
            self._trainer._is_fitted = True

            self._log_info(f"Model loaded from registry: {path_str}")
        else:
            # Fallback to file path loading for backward compatibility
            load_path = Path(path)
            if not load_path.exists():
                raise FileNotFoundError(f"Model file not found: {load_path}")

            # Use ProductionModelLoader with supported formats (no pickle)
            from ml.actors.base import ProductionModelLoader

            loader = ProductionModelLoader()
            model, metadata = loader.load_model(str(load_path))

            self._trainer._model = model
            self._trainer._feature_names = metadata.get("feature_names", [])
            self._trainer._training_metrics = metadata.get("training_metrics", {})
            self._trainer._is_fitted = True

            self._log_info(f"Model loaded from file: {load_path}")

    def get_feature_importance(self) -> dict[str, float] | None:
        """
        Get feature importance from the trained model.

        This method extracts feature importance scores from the trained model
        if available. It supports various model types including scikit-learn
        models (feature_importances_) and XGBoost (get_score).

        Returns
        -------
        dict[str, float] | None
            Dictionary mapping feature names to importance scores,
            or None if feature importance is not available.

        Example
        -------
        >>> importance = persistence.get_feature_importance()
        >>> if importance:
        ...     for name, score in sorted(importance.items(), key=lambda x: -x[1]):
        ...         print(f"{name}: {score:.4f}")

        """
        if not self._trainer._is_fitted or self._trainer._model is None:
            return None

        # Try to get feature importance from the model
        importance: npt.NDArray[np.float64] | None = None

        if hasattr(self._trainer._model, "feature_importances_"):
            importance = np.asarray(
                self._trainer._model.feature_importances_,
                dtype=np.float64,
            )
        elif hasattr(self._trainer._model, "get_score"):
            # XGBoost Booster style
            importance_dict = self._trainer._model.get_score(importance_type="gain")
            if importance_dict:
                # Convert to array format
                importance = np.zeros(len(self._trainer._feature_names), dtype=np.float64)
                for fname, imp in importance_dict.items():
                    if fname in self._trainer._feature_names:
                        idx = self._trainer._feature_names.index(fname)
                        importance[idx] = imp

        if importance is not None and len(self._trainer._feature_names) == len(importance):
            return dict(zip(self._trainer._feature_names, importance.tolist()))

        return None

    def _log_info(self, message: str, *args: object, **kwargs: Any) -> None:
        """
        Log info message.

        Parameters
        ----------
        message : str
            The message to log.
        *args : object
            Positional arguments for message formatting.
        **kwargs : Any
            Keyword arguments for logger.

        """
        logger.info(message, *args, **kwargs)

    def _log_warning(self, message: str, *args: object, **kwargs: Any) -> None:
        """
        Log warning message.

        Parameters
        ----------
        message : str
            The message to log.
        *args : object
            Positional arguments for message formatting.
        **kwargs : Any
            Keyword arguments for logger.

        """
        logger.warning(message, *args, **kwargs)

    def _log_error(self, message: str, *args: object, **kwargs: Any) -> None:
        """
        Log error message.

        Parameters
        ----------
        message : str
            The message to log.
        *args : object
            Positional arguments for message formatting.
        **kwargs : Any
            Keyword arguments for logger.

        """
        logger.error(message, *args, **kwargs)


__all__ = [
    "PersistenceComponent",
    "PersistenceTrainerProtocol",
]
