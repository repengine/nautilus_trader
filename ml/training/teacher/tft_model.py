"""
Deprecated compatibility shim for TFT teacher imports.
"""

from __future__ import annotations

import warnings

from ml.training.teacher.tft_teacher import TFTTeacher
from ml.training.teacher.tft_teacher import TFTTeacherConfig


warnings.warn(
    "ml.training.teacher.tft_model is deprecated; use ml.training.teacher.tft_teacher instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["TFTTeacher", "TFTTeacherConfig"]
