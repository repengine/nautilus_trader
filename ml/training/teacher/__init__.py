"""
Teacher models and CLIs for training pipelines.

Public API exposes the CLI module for tests and orchestrators via a lazy attribute to
avoid import-time overhead.

"""

from __future__ import annotations

from importlib import import_module as _import_module
from typing import Any


__all__ = [
    "QuickTFTTrainConfig",
    "QuickTFTTrainResult",
    "hpo_tft",
    "tft_cli",
    "train_tft_quick",
]


def __getattr__(name: str) -> Any:
    if name == "tft_cli":
        return _import_module("ml.training.teacher.tft_cli")
    if name == "hpo_tft":
        return _import_module("ml.training.teacher.hpo_tft")
    if name in {"QuickTFTTrainConfig", "QuickTFTTrainResult", "train_tft_quick"}:
        module = _import_module("ml.training.teacher.quick")
        return getattr(module, name)
    raise AttributeError(name)
