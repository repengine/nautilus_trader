"""
Deployment security helpers (cold path).

Currently enforces ONNX-only model artifact usage for actors in production.
"""

from __future__ import annotations

from pathlib import Path


def assert_allowed_model_path(path: str) -> None:
    """
    Validate model artifact path for deployment.

    Only ONNX models are allowed. Raises ValueError with actionable message
    if the path is not permitted.
    """
    suffix = Path(path).suffix.lower()
    if suffix != ".onnx":
        raise ValueError(
            "Only ONNX model artifacts are permitted in production (.*.onnx). "
            f"Refused: {path}",
        )


__all__ = ["assert_allowed_model_path"]
