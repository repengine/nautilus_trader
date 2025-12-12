from __future__ import annotations

import logging
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


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TFTTeacherConfig(TeacherConfig):
    architecture: str = "TFT"
    #: Loss function name: "poisson" (default) or "bce".
    loss_name: str = "poisson"
    #: Optional positive class weight for BCE to handle class imbalance.
    pos_weight: float | None = None


@dataclass(frozen=True, slots=True)
class StreamingRowMetadata:
    """
    Metadata for streaming row processing.

    Attributes
    ----------
    row_ids : npt.NDArray[np.str_]
        Unique row identifiers (instrument + time).
    instrument_ids : npt.NDArray[np.str_]
        Instrument identifiers.
    time_indices : npt.NDArray[np.int64]
        Time indices aligned with model inputs.
    """

    row_ids: npt.NDArray[np.str_]
    instrument_ids: npt.NDArray[np.str_]
    time_indices: npt.NDArray[np.int64]

    @property
    def size(self) -> int:
        """Return the number of rows represented."""
        return int(self.row_ids.shape[0])

    @classmethod
    def empty(cls) -> StreamingRowMetadata:
        """Return an empty metadata instance."""
        return cls(
            row_ids=np.array([], dtype=np.str_),
            instrument_ids=np.array([], dtype=np.str_),
            time_indices=np.array([], dtype=np.int64),
        )


@dataclass(frozen=True, slots=True)
class StreamingFitResult:
    """
    Result of streaming fit operation.

    Attributes
    ----------
    model_path : str | None
        Path to saved model, if any.
    metrics : dict[str, float]
        Training and validation metrics.
    epochs_completed : int
        Number of epochs completed.
    rows_processed : int
        Total rows processed during training.
    """

    model_path: str | None = None
    metrics: dict[str, float] | None = None
    epochs_completed: int = 0
    rows_processed: int = 0
    z_train: npt.NDArray[np.float64] | npt.NDArray[np.float32] | None = None
    z_val: npt.NDArray[np.float64] | npt.NDArray[np.float32] | None = None
    y_val: npt.NDArray[np.float64] | npt.NDArray[np.float32] | None = None
    train_rows: StreamingRowMetadata | None = None
    val_rows: StreamingRowMetadata | None = None
    val_returns: npt.NDArray[np.float64] | npt.NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        # Use object.__setattr__ for frozen dataclass
        if self.metrics is None:
            object.__setattr__(self, "metrics", {})


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
        learning_rate: float = 3e-4,
        batch_size: int = 64,
        accelerator: str = "auto",
        devices: int = 1,
        precision: str = "32",
        optimizer: str | None = None,
        lr_scheduler: str | None = None,
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
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self._accelerator = str(accelerator)
        self._devices = int(devices)
        self._precision = str(precision)
        self._logger = logging.getLogger(__name__)
        self._optimizer_name = optimizer
        self._lr_scheduler_name = lr_scheduler

        # Runtime state
        self._training_dataset: Any | None = None
        self._tft: Any | None = None
        self._trainer: Any | None = None

    # --- public API ---
    def fit(self, df: Any) -> TFTTeacher:
        if not HAS_PANDAS:
            check_ml_dependencies(["pandas"])  # pragma: no cover - guarded
        if pd is None:
            raise RuntimeError("pandas is required for TFTTeacher.fit")

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
            min_encoder_length=1,
            min_prediction_length=1,
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

        # DataLoader tuning: pin memory, persistent workers, prefetch
        loader_kwargs: dict[str, Any] = {"pin_memory": True}
        if self.dataloader_workers > 0:
            loader_kwargs.update(
                {
                    "persistent_workers": True,
                    "prefetch_factor": 2,
                },
            )
        try:
            train_loader = training.to_dataloader(
                train=True,
                batch_size=self.batch_size,
                num_workers=self.dataloader_workers,
                **loader_kwargs,
            )
            val_loader = val.to_dataloader(
                train=False,
                batch_size=self.batch_size,
                num_workers=self.dataloader_workers,
                **loader_kwargs,
            )
        except TypeError:
            # Older PF/DataLoader may not support these kwargs; fall back safely
            train_loader = training.to_dataloader(
                train=True,
                batch_size=self.batch_size,
                num_workers=self.dataloader_workers,
            )
            val_loader = val.to_dataloader(
                train=False,
                batch_size=self.batch_size,
                num_workers=self.dataloader_workers,
            )

        from typing import cast as _cast

        cfg = _cast(TFTTeacherConfig, self.config)
        # Define model with selected loss.
        loss_obj: Any
        if cfg.loss_name.lower() == "bce":
            loss_obj = BCEWithLogitsLossPF(pos_weight=cfg.pos_weight)
        else:
            loss_obj = PoissonLoss()
        self._tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=self.learning_rate,
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
                # Use hardened loader with optional checksum (if caller sets via env)
                from ml.training.safe_torch import safe_torch_load

                expected = None  # Could wire from config later
                state = safe_torch_load(self.pretrained_state_path, expected_sha256=expected)
                _missing, _unexpected = self._tft.load_state_dict(state, strict=False)
            except Exception as exc:
                logger.debug("Optional TFT warm-start failed: %s", exc)

        callbacks = None
        # Force CPU for stability across environments; avoids CUDA kernel asserts in PF paths
        # Normalize precision for Lightning 2.x ("16" → "16-mixed", "bf16" → "bf16-mixed")
        precision_arg: Any = self._precision
        try:  # pragma: no cover - environment/version specific
            import lightning.pytorch as _lpl  # type: ignore[unused-ignore]

            if isinstance(precision_arg, str):
                if precision_arg == "16":
                    precision_arg = "16-mixed"
                elif precision_arg.lower() == "bf16":
                    precision_arg = "bf16-mixed"
        except Exception as exc:
            # Use value as-is for pytorch_lightning 1.x
            logger.debug("Lightning precision normalization skipped: %s", exc)

        self._trainer = pl_module.Trainer(
            max_epochs=self.max_epochs,
            gradient_clip_val=1.0,
            enable_progress_bar=False,
            logger=False,
            callbacks=([] if callbacks is None else [callbacks]),
            accelerator=self._accelerator,
            devices=self._devices,
            enable_checkpointing=False,
            precision=precision_arg,
        )
        # Workaround: Some PF versions raise during validation due to interpretability logging
        # (e.g., integer_histogram index bounds). Disable interpretability/logging by overriding
        # create_log on the instance to return an empty dict.
        try:
            import types as _types

            if hasattr(self._tft, "create_log"):
                self._tft.create_log = _types.MethodType(lambda *_a, **_k: {}, self._tft)
            if hasattr(self._tft, "on_epoch_end"):
                self._tft.on_epoch_end = _types.MethodType(lambda *_a, **_k: None, self._tft)
        except Exception as exc:
            logger.debug("Disabled interpretability hooks failed: %s", exc)
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
        if pd is None:
            raise RuntimeError("pandas is required for prediction")
        df = pd.DataFrame(df).copy()

        training = self._training_dataset
        if training is None:
            raise RuntimeError("Training dataset is not initialized")
        dataset = TimeSeriesDataSet.from_dataset(
            training,
            df,
            predict=False,
            stop_randomization=True,
        )
        loader = dataset.to_dataloader(train=False, batch_size=self.batch_size, num_workers=0)

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
            vals: list[npt.NDArray[np.float64]] = []
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
                import torch as _torch

                if isinstance(raw, _torch.Tensor):
                    preds = raw.detach().cpu().numpy()
            except Exception as exc:
                logger.debug("Torch tensor to numpy conversion failed: %s", exc)
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

    # Optional convenience: predict logits with aligned targets using PF return_x
    def predict_logits_with_targets(
        self,
        df: Any,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if not self._is_fitted or self._tft is None:
            raise RuntimeError("TFTTeacher must be fitted before prediction")

        try:
            import numpy as _np
            from pytorch_forecasting import TimeSeriesDataSet
        except Exception as exc:  # pragma: no cover
            raise ImportError("pytorch-forecasting is required for prediction") from exc

        if not HAS_PANDAS:
            check_ml_dependencies(["pandas"])  # pragma: no cover
        if pd is None:
            raise RuntimeError("pandas is required for prediction")
        df = pd.DataFrame(df).copy()

        training = self._training_dataset
        if training is None:
            raise RuntimeError("Training dataset is not initialized")
        dataset = TimeSeriesDataSet.from_dataset(
            training,
            df,
            predict=False,
            stop_randomization=True,
        )
        loader = dataset.to_dataloader(train=False, batch_size=self.batch_size, num_workers=0)

        out = self._tft.predict(loader, mode="raw", return_x=True)
        preds_any = None
        x_any = None
        # Handle common return structures: tuple(preds, x) or list of tuples
        if isinstance(out, tuple) and len(out) == 2:
            preds_any, x_any = out
            if isinstance(preds_any, dict):
                preds_any = preds_any.get("prediction") or preds_any.get("predictions")
        elif isinstance(out, list) and len(out) > 0:
            first = out[0]
            if isinstance(first, tuple) and len(first) == 2:
                preds_list = []
                y_list = []
                for preds_i, x_i in out:
                    try:
                        import torch as _torch

                        if isinstance(preds_i, _torch.Tensor):
                            preds_list.append(preds_i.detach().cpu().numpy())
                        else:
                            preds_list.append(_np.asarray(preds_i))
                    except Exception:
                        preds_list.append(_np.asarray(preds_i))
                    # x_i may be a dict with decoder_target
                    if isinstance(x_i, dict):
                        y_i = x_i.get("decoder_target") or x_i.get("target")
                    else:
                        y_i = getattr(x_i, "decoder_target", None)
                    if y_i is None:
                        raise RuntimeError("Unable to locate decoder_target in PF return_x output")
                    try:
                        import torch as _torch

                        if isinstance(y_i, _torch.Tensor):
                            y_list.append(y_i.detach().cpu().numpy())
                        else:
                            y_list.append(_np.asarray(y_i))
                    except Exception:
                        y_list.append(_np.asarray(y_i))
                preds = _np.concatenate(preds_list, axis=0)
                y = _np.concatenate(y_list, axis=0)
                # Convert to logits if probabilities
                arr: npt.NDArray[np.float64] = _np.asarray(preds, dtype=_np.float64).reshape(-1)
                if _np.all(arr >= 0.0) and _np.all(arr <= 1.0):
                    eps = 1e-6
                    p = _np.clip(arr, eps, 1.0 - eps)
                    arr = _np.log(p / (1.0 - p))
                y_arr: npt.NDArray[np.float64] = _np.asarray(y, dtype=_np.float64).reshape(-1)
                return arr.astype(_np.float64), y_arr.astype(_np.float64)
        else:
            preds_any = out
            x_any = None

        try:
            import torch as _torch

            if isinstance(preds_any, _torch.Tensor):
                preds_np = preds_any.detach().cpu().numpy()
            else:
                preds_np = _np.asarray(preds_any)
        except Exception:
            preds_np = _np.asarray(preds_any)
        # Extract decoder_target if available
        y_np = None
        if x_any is not None:
            if isinstance(x_any, dict):
                y_np = x_any.get("decoder_target") or x_any.get("target")
            else:
                y_np = getattr(x_any, "decoder_target", None)
        if y_np is None:
            # As a fallback, cannot align targets — raise for caller to degrade gracefully
            raise RuntimeError("predict_logits_with_targets could not extract decoder_target")

        arr_logits: npt.NDArray[np.float64] = _np.asarray(preds_np, dtype=_np.float64).reshape(-1)
        if _np.all(arr_logits >= 0.0) and _np.all(arr_logits <= 1.0):
            eps = 1e-6
            p = _np.clip(arr_logits, eps, 1.0 - eps)
            arr_logits = _np.log(p / (1.0 - p))
        try:
            import torch as _torch

            if isinstance(y_np, _torch.Tensor):
                y_np = y_np.detach().cpu().numpy()
        except Exception as exc:
            self._logger.debug("Torch tensor to numpy conversion failed: %s", exc)
        y_arr_aligned: npt.NDArray[np.float64] = _np.asarray(y_np, dtype=_np.float64).reshape(-1)
        return arr_logits.astype(_np.float64), y_arr_aligned.astype(_np.float64)

    def fit_streaming(
        self,
        *,
        parquet_path: Any,
        train_loader: Any,
        val_loader: Any,
        train_metadata: Any,
        val_metadata: Any,
        full_metadata: Any,
        streaming_config: Any,
        callbacks: Sequence[Any] | None = None,
        checkpoint_path: Any | None = None,
    ) -> StreamingFitResult:
        """Placeholder streaming fit to satisfy strict typing when streaming mode is unused."""
        _ = (
            parquet_path,
            train_loader,
            val_loader,
            train_metadata,
            val_metadata,
            full_metadata,
            streaming_config,
            callbacks,
            checkpoint_path,
        )
        return StreamingFitResult(
            z_train=np.array([], dtype=np.float32),
            z_val=np.array([], dtype=np.float32),
            y_val=np.array([], dtype=np.float32),
            train_rows=StreamingRowMetadata.empty(),
            val_rows=StreamingRowMetadata.empty(),
            val_returns=np.array([], dtype=np.float32),
        )
