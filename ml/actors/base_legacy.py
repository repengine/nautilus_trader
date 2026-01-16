"""
Legacy compatibility shim for BaseMLInferenceActor.

During the transition period, parity tests import this module to compare
legacy and current implementations. It currently re-exports the facade.
"""

from __future__ import annotations

from ml.actors.base import BaseMLInferenceActor


__all__ = [
    "BaseMLInferenceActor",
]
