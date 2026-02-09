from __future__ import annotations

import importlib

import pytest

from ml.training.teacher import QuickTFTTrainConfig as TeacherPackageQuickTFTTrainConfig
from ml.training.teacher import QuickTFTTrainResult as TeacherPackageQuickTFTTrainResult
from ml.training.teacher import train_tft_quick as teacher_package_train_tft_quick
from ml.training.teacher.quick import QuickTFTTrainConfig as CanonicalQuickTFTTrainConfig
from ml.training.teacher.quick import QuickTFTTrainResult as CanonicalQuickTFTTrainResult
from ml.training.teacher.quick import _DEFAULT_DATA_DIRS as canonical_default_data_dirs
from ml.training.teacher.quick import _DEFAULT_SYMBOLS as canonical_default_symbols
from ml.training.teacher.quick import train_tft_quick as canonical_train_tft_quick


def test_quick_training_teacher_symbols_remain_canonical() -> None:
    assert TeacherPackageQuickTFTTrainConfig is CanonicalQuickTFTTrainConfig
    assert TeacherPackageQuickTFTTrainResult is CanonicalQuickTFTTrainResult
    assert teacher_package_train_tft_quick is canonical_train_tft_quick

    assert canonical_default_data_dirs
    assert canonical_default_symbols


def test_task_quick_shim_module_is_retired() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("ml.tasks.training.quick")
