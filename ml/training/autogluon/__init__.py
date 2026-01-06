"""
AutoGluon TimeSeries training module.

This module provides trainers for Chronos foundation models using
AutoGluon's TimeSeriesPredictor API.

"""

from __future__ import annotations

from ml.training.autogluon.chronos_trainer import ChronosTrainer


__all__ = [
    "ChronosTrainer",
]
