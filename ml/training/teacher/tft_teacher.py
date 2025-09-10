from __future__ import annotations

import types
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ml._imports import HAS_PANDAS
from ml._imports import check_ml_dependencies
from ml._imports import pd
from ml.training.teacher.base import BaseTeacher
from ml.training.teacher.base import TeacherConfig


@dataclass(frozen=True)
class TFTTeacherConfig(TeacherConfig):
    architecture: str = "TFT"
    #: Loss function name: "poisson" (default) or "bce".
    loss_name: str = "poisson"


class TFTTeacher(BaseTeacher):
    """
    Temporal Fusion Transformer teacher using PyTorch Forecasting.

    Notes
    -----
    - Heavy dependencies are imported lazily to keep import-time light.
    - Designed for binary classification with logits output.

    """

    def __init__(
        self,
        config: TFTTeacherConfig,
        max_encoder_length: int = 30,
        max_prediction_length: int = 1,
        static_categoricals: Sequence[str] | None = None,
        static_reals: Sequence[str] | None = None,
        time_varying_known_reals: Sequence[str] | None = None,
        time_varying_unknown_reals: Sequence[str] | None = None,
        time_idx_col: str = "time_index",
        group_id_col: str = "instrument_id",
        target_col: str = "y",
        max_epochs: int = 1,
        hidden_size: int = 16,
        lstm_layers: int = 1,
        attention_head_size: int = 2,
        dropout: float = 0.1,
        dataloader_workers: int = 0,
        pretrained_state_path: str | None = None,
    ) -> None:
        super().__init__(config)
        self.max_encoder_length = int(max_encoder_length)
        self.max_prediction_length = int(max_prediction_length)
        self.static_categoricals = list(static_categoricals or [])
        self.static_reals = list(static_reals or [])
        self.time_varying_known_reals = list(time_varying_known_reals or [])
        self.time_varying_unknown_reals = list(time_varying_unknown_reals or [])
        self.time_idx_col = time_idx_col
        self.group_id_col = group_id_col
        self.target_col = target_col
        self.max_epochs = int(max_epochs)
        self.hidden_size = int(hidden_size)
        self.lstm_layers = int(lstm_layers)
        self.attention_head_size = int(attention_head_size)
        self.dropout = float(dropout)
        self.dataloader_workers = int(dataloader_workers)
        self.pretrained_state_path = pretrained_state_path

        # Runtime state
        self._training_dataset: Any | None = None
        self._tft: Any | None = None
        self._trainer: Any | None = None

    # --- public API ---
    def fit(self, df: Any) -> TFTTeacher:
        if not HAS_PANDAS:
            check_ml_dependencies(["pandas"])  # pragma: no cover - guarded
        assert pd is not None

        # Local imports to avoid heavy deps at import time
        pl_module: types.ModuleType
        try:  # pragma: no cover - exercised in integration test
            # Prefer Lightning 2.x when available for better NumPy 2.0 compatibility;
            # fall back to pytorch_lightning 1.x if necessary.
            try:
                import lightning.pytorch as _lpl

                pl_module = _lpl
            except Exception:  # pragma: no cover
                import pytorch_lightning as _pl

                pl_module = _pl
            import torch
            from pytorch_forecasting import TemporalFusionTransformer
            from pytorch_forecasting import TimeSeriesDataSet

            # Use a supported loss for single-output regression to 0/1 targets
            from pytorch_forecasting.metrics import PoissonLoss

            from ml.training.teacher.losses import BCEWithLogitsLossPF
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "pytorch-forecasting + pytorch-lightning are required for TFT training",
            ) from exc

        df = pd.DataFrame(df).copy()
        if self.time_idx_col not in df.columns:
            raise ValueError(f"Missing required time index column: {self.time_idx_col}")
        if self.group_id_col not in df.columns:
            raise ValueError(f"Missing required group id column: {self.group_id_col}")
        if self.target_col not in df.columns:
            raise ValueError(f"Missing required target column: {self.target_col}")

        # Split by time: last 20% as validation cutoff
        df_sorted = df.sort_values(self.time_idx_col)
        n = len(df_sorted)
        cutoff_idx = int(n * 0.8)
        df_train = df_sorted.iloc[:cutoff_idx]
        _df_val = df_sorted.iloc[cutoff_idx:]

        # Fill missing numeric values to satisfy TimeSeriesDataSet requirements
        try:
            num_cols = df_train.select_dtypes(include=["number"]).columns
        except Exception:
            num_cols = [c for c in df_train.columns if pd.api.types.is_numeric_dtype(df_train[c])]
        if len(num_cols) > 0:
            df_train.loc[:, num_cols] = df_train.loc[:, num_cols].fillna(0)
            df_sorted.loc[:, num_cols] = df_sorted.loc[:, num_cols].fillna(0)

        # Default unknown reals: use all features not in control columns
        if not self.time_varying_unknown_reals:
            control_cols = {self.time_idx_col, self.group_id_col, self.target_col}
            control_cols.update(self.static_categoricals)
            control_cols.update(self.static_reals)
            control_cols.update(self.time_varying_known_reals)
            # Only include numeric columns as unknown reals to avoid string/time fields like 'timestamp'
            try:
                numeric_cols = set(df.select_dtypes(include=["number"]).columns)
            except Exception:
                numeric_cols = {c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
            self.time_varying_unknown_reals = [
                c for c in df.columns if c in numeric_cols and c not in control_cols
            ]

        training = TimeSeriesDataSet(
            df_train,
            time_idx=self.time_idx_col,
            target=self.target_col,
            group_ids=[self.group_id_col],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=self.static_categoricals,
            static_reals=self.static_reals,
            time_varying_known_reals=self.time_varying_known_reals,
            time_varying_unknown_reals=self.time_varying_unknown_reals,
            allow_missing_timesteps=True,
        )

        # Use the full sorted dataset for validation to ensure sufficient encoder history
        val = TimeSeriesDataSet.from_dataset(
            training,
            df_sorted,
            predict=False,
            stop_randomization=True,
        )

        train_loader = training.to_dataloader(
            train=True,
            batch_size=64,
            num_workers=self.dataloader_workers,
        )
        val_loader = val.to_dataloader(
            train=False,
            batch_size=64,
            num_workers=self.dataloader_workers,
        )

        from typing import cast as _cast

        cfg = _cast(TFTTeacherConfig, self.config)
        # Define model with selected loss.
        loss_obj: Any
        if cfg.loss_name.lower() == "bce":
            loss_obj = BCEWithLogitsLossPF()
        else:
            loss_obj = PoissonLoss()
        self._tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=3e-4,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
            dropout=self.dropout,
            output_size=1,
            loss=loss_obj,
            attention_head_size=self.attention_head_size,
            log_interval=100,
        )
        # Optional warm-start from a pretrained state dict (best-effort partial load)
        if self.pretrained_state_path:
            try:
                import torch  # local import for typing

                state = torch.load(self.pretrained_state_path, map_location="cpu")
                _missing, _unexpected = self._tft.load_state_dict(state, strict=False)
            except Exception:
                pass

        callbacks = None
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        self._trainer = pl_module.Trainer(
            max_epochs=self.max_epochs,
            gradient_clip_val=1.0,
            enable_progress_bar=False,
            logger=False,
            callbacks=([] if callbacks is None else [callbacks]),
            accelerator=accelerator,
            devices=1,
            enable_checkpointing=False,
        )
        self._trainer.fit(self._tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

        self._training_dataset = training
        self._is_fitted = True
        return self

    def predict_logits(self, df: Any) -> npt.NDArray[np.float64]:
        if not self._is_fitted or self._tft is None:
            raise RuntimeError("TFTTeacher must be fitted before prediction")

        try:  # pragma: no cover - integration exercised in tests
            import numpy as _np
            from pytorch_forecasting import TimeSeriesDataSet
        except Exception as exc:  # pragma: no cover
            raise ImportError("pytorch-forecasting is required for prediction") from exc

        if not HAS_PANDAS:
            check_ml_dependencies(["pandas"])  # pragma: no cover
        assert pd is not None
        df = pd.DataFrame(df).copy()

        training = self._training_dataset
        assert training is not None
        dataset = TimeSeriesDataSet.from_dataset(
            training,
            df,
            predict=True,
            stop_randomization=True,
        )
        loader = dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

        # Use default prediction mode for broader compatibility; convert to logits below
        raw = self._tft.predict(loader)
        preds = None
        # Common PF return types: dict, list[dict], tensor/ndarray
        if isinstance(raw, dict):
            if "prediction" in raw:
                preds = raw["prediction"]
            elif "predictions" in raw:
                preds = raw["predictions"]
        elif isinstance(raw, list) and len(raw) > 0:
            # Concatenate per-batch outputs
            vals: list[_np.ndarray] = []
            for item in raw:
                if isinstance(item, dict) and ("prediction" in item or "predictions" in item):
                    v = item.get("prediction", item.get("predictions"))
                    vals.append(_np.asarray(v))
                else:
                    vals.append(_np.asarray(item))
            preds = _np.concatenate(vals, axis=0)
        elif isinstance(raw, _np.ndarray):
            preds = raw
        else:
            try:
                import torch as _torch  # type: ignore

                if isinstance(raw, _torch.Tensor):
                    preds = raw.detach().cpu().numpy()
            except Exception:
                pass
        if preds is None:
            raise RuntimeError("Unexpected TFT prediction output format")

        arr: npt.NDArray[_np.float64] = _np.asarray(preds, dtype=_np.float64).reshape(-1)
        # Heuristic: if values look like probabilities in (0,1), convert to logits
        if _np.all(arr >= 0.0) and _np.all(arr <= 1.0):
            eps = 1e-6
            p = _np.clip(arr, eps, 1.0 - eps)
            arr = _np.log(p / (1.0 - p))
        return arr.astype(np.float64)

    def feature_schema(self) -> dict[str, str]:
        return dict.fromkeys(self.time_varying_unknown_reals, "float32")
