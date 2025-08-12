"""
TFTTeacher (PyTorch Forecasting)
--------------------------------
A minimal wrapper to train a Temporal Fusion Transformer teacher on rich features,
then score a student window to produce calibrated probabilities (and logits) for
distillation.

This is intentionally light; in real use you'll build a proper DataModule around
your parquet store and feature engineering pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Optional heavy deps (import when available)
try:
    import torch
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss
    from pytorch_lightning import Trainer
except Exception:  # pragma: no cover
    torch = None
    Trainer = None
    TemporalFusionTransformer = None
    TimeSeriesDataSet = None
    QuantileLoss = None

# Optional calibration
try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover
    IsotonicRegression = None
    LogisticRegression = None


@dataclass
class TFTTeacherConfig:
    max_epochs: int = 5
    gpus: int = 0
    hidden_size: int = 32
    attention_head_size: int = 4
    learning_rate: float = 1e-3
    dropout: float = 0.1
    loss: str = "bce"  # 'bce' for classification; for regression use 'quantile'
    seed: int = 42


class TFTTeacher:
    def __init__(self, cfg: TFTTeacherConfig):
        self.cfg = cfg
        self.model = None
        self._calibrator = None  # isotonic or platt
        self._calibrator_kind = None

    def fit(self, dataset: TimeSeriesDataSet) -> TFTTeacher:
        if TemporalFusionTransformer is None:
            raise ImportError("pytorch-forecasting and lightning are required.")
        torch.manual_seed(self.cfg.seed)
        self.model = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=self.cfg.learning_rate,
            hidden_size=self.cfg.hidden_size,
            attention_head_size=self.cfg.attention_head_size,
            dropout=self.cfg.dropout,
            loss=("bce" if self.cfg.loss == "bce" else QuantileLoss()),
            log_interval=10,
            reduce_on_plateau_patience=2,
        )
        trainer = Trainer(
            max_epochs=self.cfg.max_epochs,
            accelerator=("gpu" if self.cfg.gpus > 0 else "cpu"),
            devices=self.cfg.gpus if self.cfg.gpus > 0 else None,
        )
        trainer.fit(
            self.model,
            train_dataloaders=dataset.to_dataloader(train=True, batch_size=128, num_workers=0),
            val_dataloaders=dataset.to_dataloader(train=False, batch_size=256, num_workers=0),
        )
        return self

    def calibrate(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        # You'll pass pre-processed tensors; for brevity we assume you already ran the model to get p_val
        # Here we just fit a Platt or Isotonic on (p_val, y_val)
        # Replace this with your own flow that extracts probabilities from the TFT model.
        if LogisticRegression is not None:
            lr = LogisticRegression(solver="lbfgs")
            lr.fit(X_val.reshape(-1, 1), y_val.astype(int))
            self._calibrator = lr
            self._calibrator_kind = "platt"
        elif IsotonicRegression is not None:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(X_val, y_val.astype(float))
            self._calibrator = iso
            self._calibrator_kind = "isotonic"

    def predict_proba(self, p_raw: np.ndarray) -> np.ndarray:
        # Apply optional calibrator to raw probs.
        p = p_raw.astype(np.float32)
        if self._calibrator is not None:
            if self._calibrator_kind == "platt":
                p = self._calibrator.predict_proba(p.reshape(-1, 1))[:, 1]
            else:
                p = self._calibrator.transform(p)
        return p.reshape(-1, 1)
