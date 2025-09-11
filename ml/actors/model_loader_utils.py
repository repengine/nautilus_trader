"""
Shared model loading utilities for ML actors.
"""  # ruff: noqa: I001

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ml.registry.base import DataRequirements
from ml.registry.base import ModelManifest
from ml.registry.base import ModelRole


def maybe_warm_up_model(model: Any, warm_up: bool, input_dim: int) -> None:
    """
    Optionally run a single forward pass to warm up the model runtime.

    Parameters
    ----------
    model : Any
        Loaded model (ONNX session or framework model)
    warm_up : bool
        Whether to perform warm-up
    input_dim : int
        Number of features to use for the dummy input

    """
    if not warm_up or model is None:
        return
    try:  # pragma: no cover - warm up is environment specific
        x = np.zeros((1, int(input_dim)), dtype=np.float32)
        # ONNXRuntime-like
        if hasattr(model, "run") and hasattr(model, "get_inputs"):
            inp_name = model.get_inputs()[0].name
            model.run(None, {inp_name: x})
        # Scikit-learn-like
        elif hasattr(model, "predict"):
            model.predict(x)
    except Exception as exc:
        logger.debug("Model warm-up failed (ignored): %s", exc)


def assert_features_parity(
    manifest_feature_names: list[str] | None,
    model_metadata: dict[str, Any] | None,
    actual_feature_names: list[str],
) -> None:
    """
    Validate that runtime feature names/dtypes are compatible with the model manifest.
    """
    names = manifest_feature_names or []
    if not names:
        return
    manifest_schema = None
    if isinstance(model_metadata, dict):
        manifest_schema = model_metadata.get("feature_schema")
    if not manifest_schema:
        manifest_schema = dict.fromkeys(names, "float32")

    from typing import cast

    manifest_schema_typed = cast(dict[str, str], manifest_schema)
    tmp_manifest = ModelManifest(
        model_id="__validation__",
        role=ModelRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="unknown",
        feature_schema=manifest_schema_typed,
        feature_schema_hash=(
            model_metadata.get("feature_schema_hash", "")
            if isinstance(model_metadata, dict)
            else ""
        ),
    )
    # Import locally to avoid cycles
    from ml.registry.utils import assert_features_compatible

    actual_dtypes = ["float32"] * len(actual_feature_names)
    assert_features_compatible(tmp_manifest, actual_feature_names, actual_dtypes)


__all__ = ["assert_features_parity", "maybe_warm_up_model"]
logger = logging.getLogger(__name__)
