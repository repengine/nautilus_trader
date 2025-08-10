#!/usr/bin/env python3

"""
Model export utilities ensuring training/inference compatibility.

This module provides standardized methods for exporting trained models
in formats compatible with ProductionModelLoader, ensuring seamless
integration between training and inference pipelines.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ml._imports import HAS_XGBOOST
from ml.models import ModelType
from ml.models import detect_model_type
from ml.models.saver import convert_to_onnx
from ml.models.saver import save_model_with_metadata


class ModelExportMixin(ABC):
    """
    Mixin class that ensures models are exported in production-ready formats.

    This should be used by all training classes to ensure compatibility
    with ProductionModelLoader in inference actors.
    """

    @abstractmethod
    def get_model(self) -> Any:
        """
        Get the trained model instance.

        Returns
        -------
        Any
            The trained model

        """
        ...

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """
        Get the feature names used in training.

        Returns
        -------
        list[str]
            List of feature names in order

        """
        ...

    @abstractmethod
    def get_training_metadata(self) -> dict[str, Any]:
        """
        Get training metadata (metrics, config, etc).

        Returns
        -------
        dict[str, Any]
            Training metadata dictionary

        """
        ...

    def save_for_production(
        self,
        path: str | Path,
        format: str = "auto",
        include_metadata: bool = True,
    ) -> Path:
        """
        Save model in production-ready format.

        This ensures the model can be loaded by ProductionModelLoader
        in inference actors.

        Parameters
        ----------
        path : str | Path
            Path where to save the model
        format : str, default "auto"
            Model format: "auto", "onnx", "native"
        include_metadata : bool, default True
            Whether to include training metadata

        Returns
        -------
        Path
            Path to the saved model

        """
        model = self.get_model()
        feature_names = self.get_feature_names()

        # Prepare metadata
        training_metadata = self.get_training_metadata() if include_metadata else {}
        training_metadata["feature_names"] = feature_names
        training_metadata["trainer_class"] = self.__class__.__name__

        # Determine format
        if format == "auto":
            model_type = detect_model_type(model)
            if model_type in {ModelType.XGBOOST, ModelType.LIGHTGBM}:
                # Use native format for these
                format = "native"
            else:
                # Default to ONNX for cross-platform compatibility
                format = "onnx"

        save_path = Path(path)

        if format == "onnx":
            # Convert to ONNX
            n_features = len(feature_names)
            sample_input = np.random.randn(1, n_features).astype(np.float32)

            return convert_to_onnx(
                model=model,
                sample_input=sample_input,
                output_path=save_path,
            )
        else:
            # Save in native format
            return save_model_with_metadata(
                model=model,
                path=save_path,
                input_shape=(1, len(feature_names)),  # Use 1 instead of None for batch dimension
                training_metadata=training_metadata,
                force_pickle=False,  # Never use pickle
            )

    def validate_inference_compatibility(
        self,
        model_path: str | Path,
        test_features: NDArray[np.float32] | None = None,
    ) -> bool:
        """
        Validate that a saved model can be loaded for inference.

        This tests the full pipeline: save -> load -> predict.

        Parameters
        ----------
        model_path : str | Path
            Path to the saved model
        test_features : Optional[NDArray[np.float32]]
            Test features for prediction validation

        Returns
        -------
        bool
            True if model is inference-compatible

        """
        from ml.models.loader import ProductionModelLoader

        try:
            # Try loading with production loader
            loader = ProductionModelLoader()
            model, metadata = loader.load_model(str(model_path))

            # Validate metadata
            if "type" not in metadata:
                return False

            # Try prediction if test features provided
            if test_features is not None:
                model_type = metadata.get("type", "unknown")

                if model_type == "xgboost":
                    # XGBoost models require DMatrix
                    if HAS_XGBOOST:
                        from ml._imports import xgb
                        dtest = xgb.DMatrix(test_features)
                        prediction = model.predict(dtest)
                        if prediction is None:
                            return False
                    else:
                        return False
                elif model_type == "lightgbm":
                    # LightGBM models can handle numpy arrays directly
                    prediction = model.predict(test_features)
                    if prediction is None:
                        return False
                elif hasattr(model, "run"):
                    # ONNX model
                    input_name = model.get_inputs()[0].name
                    outputs = model.run(None, {input_name: test_features.astype(np.float32)})
                    if outputs is None:
                        return False
                elif hasattr(model, "predict"):
                    # Generic wrapped model
                    prediction = model.predict(test_features)
                    if prediction is None:
                        return False
                else:
                    return False

            return True

        except Exception as e:
            print(f"Validation failed: {e}")
            return False


class TrainingActorContract(ABC):
    """
    Contract ensuring training outputs are compatible with inference actors.

    All training classes should implement this contract to ensure
    their outputs can be loaded by ProductionModelLoader.
    """

    @abstractmethod
    def get_required_features(self) -> list[str]:
        """
        Get the exact feature names required for inference.

        These must match what the inference actor will compute.

        Returns
        -------
        list[str]
            Ordered list of feature names

        """
        ...

    @abstractmethod
    def get_model_input_shape(self) -> tuple[int, ...]:
        """
        Get the expected input shape for the model.

        Returns
        -------
        tuple[int, ...]
            Model input shape (batch_size, n_features)

        """
        ...

    @abstractmethod
    def export_for_actor(
        self,
        actor_model_path: str | Path,
        actor_config_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Export model and config for use in inference actor.

        Parameters
        ----------
        actor_model_path : str | Path
            Path where to save the model for the actor
        actor_config_path : Optional[str | Path]
            Path where to save actor configuration

        Returns
        -------
        dict[str, Any]
            Export summary with paths and metadata

        """
        ...

    def generate_actor_config(self) -> dict[str, Any]:
        """
        Generate configuration for MLSignalActor.

        Returns
        -------
        dict[str, Any]
            Actor configuration dictionary

        """
        return {
            "model_path": "path/to/model.onnx",  # To be filled
            "feature_config": {
                "indicators": self._get_indicator_config(),
                "lookback_window": 20,
                "normalize_features": True,
            },
            "prediction_threshold": 0.5,
            "warm_up_period": 50,
        }

    @abstractmethod
    def _get_indicator_config(self) -> dict[str, Any]:
        """
        Get indicator configuration matching training features.

        Returns
        -------
        dict[str, Any]
            Indicator configuration

        """
        ...
