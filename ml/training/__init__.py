"""
ML model training infrastructure for Nautilus Trader.

Note: Avoid importing heavy dependencies at package import time. Expose public
symbols via lazy getattr to keep CLI utilities lightweight when only specific
submodules are used (e.g., teacher CLI).
"""

__all__ = ["BaseMLTrainer"]


def __getattr__(name: str):  # pragma: no cover - import side-effect utility
    if name == "BaseMLTrainer":
        from .base import BaseMLTrainer  # local import to avoid heavy deps unless needed

        return BaseMLTrainer
    raise AttributeError(name)
