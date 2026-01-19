from __future__ import annotations

from types import SimpleNamespace

import pytest

from ml._imports import HAS_TORCH
from ml._imports import check_ml_dependencies
from ml._imports import torch as torch_module
from ml.training.teacher.tft_teacher import TFTTeacher
from ml.training.teacher.tft_teacher import TFTTeacherConfig


@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
def test_apply_optimizer_and_scheduler_when_onecycle_returns_scheduler() -> None:
    if torch_module is None:
        check_ml_dependencies(["torch"])
        pytest.skip("torch import guard triggered")
    torch = torch_module

    class _StubModel:
        def __init__(self) -> None:
            self.trainer = SimpleNamespace(estimated_stepping_batches=5)
            self._param = torch.nn.Parameter(torch.zeros(1))

        def parameters(self):
            return [self._param]

    teacher = TFTTeacher(
        TFTTeacherConfig(),
        max_encoder_length=1,
        max_prediction_length=1,
        dataloader_workers=0,
        batch_size=1,
        optimizer="adam",
        lr_scheduler="onecycle",
    )
    teacher._tft = _StubModel()  # type: ignore[attribute-defined-outside-init]
    teacher._apply_optimizer_and_scheduler()

    result = teacher._tft.configure_optimizers()
    assert isinstance(result, dict)
    assert isinstance(result["lr_scheduler"], torch.optim.lr_scheduler.OneCycleLR)


@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
def test_apply_optimizer_and_scheduler_when_cosine_returns_scheduler() -> None:
    if torch_module is None:
        check_ml_dependencies(["torch"])
        pytest.skip("torch import guard triggered")
    torch = torch_module

    class _StubModel:
        def __init__(self) -> None:
            self.trainer = SimpleNamespace(estimated_stepping_batches=4)
            self._param = torch.nn.Parameter(torch.zeros(1))

        def parameters(self):
            return [self._param]

    teacher = TFTTeacher(
        TFTTeacherConfig(),
        max_encoder_length=1,
        max_prediction_length=1,
        dataloader_workers=0,
        batch_size=1,
        optimizer="adam",
        lr_scheduler="cosine",
    )
    teacher._tft = _StubModel()  # type: ignore[attribute-defined-outside-init]
    teacher._apply_optimizer_and_scheduler()

    result = teacher._tft.configure_optimizers()
    assert isinstance(result, dict)
    assert isinstance(result["lr_scheduler"], torch.optim.lr_scheduler.CosineAnnealingLR)
