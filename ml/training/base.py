"""
Compatibility shim for BaseMLTrainer.

Redirects legacy imports to the facade-only implementation.
"""

from __future__ import annotations

from ml.training.base_facade import BaseMLTrainerFacade as BaseMLTrainer


__all__ = [
    "BaseMLTrainer",
]
