"""
Custom loss wrappers for TFT training.

This module provides a minimal BCE-with-logits wrapper compatible with the
TemporalFusionTransformer "loss" argument. The wrapper exposes a callable that
accepts ``y_pred`` and ``y_true`` and returns a scalar loss tensor.

Example:
    >>> import torch
    >>> from ml.training.teacher.losses import BCEWithLogitsLossPF
    >>> loss_fn = BCEWithLogitsLossPF()
    >>> y_pred = torch.tensor([[0.0],[1.0],[2.0]])
    >>> y_true = torch.tensor([[0.0],[1.0],[1.0]])
    >>> loss = loss_fn(y_pred, y_true)
    >>> float(loss) > 0
    True

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


try:  # pragma: no cover - optional heavy dep guard
    import torch
except Exception as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for BCEWithLogitsLossPF") from exc


@dataclass
class BCEWithLogitsLossPF:
    """
    Binary cross-entropy with logits loss wrapper for PF models.

    This class mimics the minimal callable interface expected by
    :class:`pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer`.

    Attributes:
        pos_weight: Optional positive class weight to address imbalance.

    Note:
        This is a minimal wrapper and does not provide advanced metric logging
        like :mod:`pytorch_forecasting.metrics`. It is intended for use as the
        ``loss=`` argument to ``from_dataset``.

    """

    pos_weight: float | None = None
    _loss: torch.nn.modules.loss.BCEWithLogitsLoss | None = None

    def __post_init__(self) -> None:
        weight: torch.Tensor | None
        if self.pos_weight is not None:
            weight = torch.tensor([float(self.pos_weight)], dtype=torch.float32)
        else:
            weight = None
        self._loss = torch.nn.BCEWithLogitsLoss(pos_weight=weight)

    def __call__(self, y_pred: Any, y_true: Any) -> torch.Tensor:
        """
        Compute BCE-with-logits loss.

        Args:
            y_pred: Predicted logits (Tensor [..., 1]).
            y_true: Targets (Tensor [..., 1] or [...]).

        Returns:
            Scalar loss tensor.

        """
        yp = torch.as_tensor(y_pred)
        yt = torch.as_tensor(y_true)
        if yt.ndim == yp.ndim - 1:
            yt = yt.unsqueeze(-1)
        assert self._loss is not None
        out: torch.Tensor = self._loss(yp.float(), yt.float())
        return out
