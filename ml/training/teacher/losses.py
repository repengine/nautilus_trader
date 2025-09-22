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

from typing import TYPE_CHECKING, Any, cast


try:  # pragma: no cover - optional heavy dep guard
    import torch
    from pytorch_forecasting.metrics import MultiHorizonMetric as _MultiHorizonMetricRuntime
except Exception as exc:  # pragma: no cover
    raise ImportError("PyTorch + pytorch-forecasting are required for BCEWithLogitsLossPF") from exc

if TYPE_CHECKING:
    from pytorch_forecasting.metrics import MultiHorizonMetric as MultiHorizonMetricBase
else:
    MultiHorizonMetricBase = cast(type[Any], _MultiHorizonMetricRuntime)


class BCEWithLogitsLossPF(MultiHorizonMetricBase):  # type: ignore[misc]
    """
    BCE-with-logits loss as a Pytorch Forecasting Metric.

    Implements ``loss(y_pred, target)`` returning element-wise loss (no reduction).

    Attributes:
        pos_weight: Optional positive class weight to address imbalance.
        reduction: "mean" (default) or "none" compatible with MultiHorizonMetric.

    """

    def __init__(self, pos_weight: float | None = None, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        self.pos_weight = pos_weight
        self.register_buffer(
            "_pos_weight_tensor",
            (
                torch.tensor([float(pos_weight)], dtype=torch.float32)
                if pos_weight is not None
                else None
            ),
            persistent=False,
        )

    def loss(self, y_pred: Any, target: torch.Tensor) -> torch.Tensor:
        # Accept either dict outputs (with key 'prediction') or direct tensor outputs
        if isinstance(y_pred, dict):
            pred = y_pred["prediction"].float()
        else:
            pred = torch.as_tensor(y_pred).float()
        tgt = target.float()
        if tgt.ndim == pred.ndim - 1:
            tgt = tgt.unsqueeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred,
            tgt,
            weight=None,
            pos_weight=self._pos_weight_tensor,
            reduction="none",
        )
        return loss
