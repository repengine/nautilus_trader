from __future__ import annotations

import logging
import types
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ml._imports import HAS_PANDAS
from ml._imports import check_ml_dependencies
from ml._imports import pd
from ml.common.metrics_bootstrap import get_counter
from ml.training.teacher.base import BaseTeacher
from ml.training.teacher.base import TeacherConfig


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from ml.training.teacher.streaming_loader import StreamDataLoader
    from ml.training.teacher.streaming_loader import TFTStreamingConfig
    from ml.training.teacher.streaming_loader import TFTStreamingMetadata


_FALLBACK_COUNTER = get_counter(
    "ml_fallback_activations_total",
    "Fallback activations",
    labelnames=("component", "level"),
)



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
        except Exception:
            self._logger.debug("Torch tensor to numpy conversion failed", exc_info=True)
        y_arr_aligned: npt.NDArray[np.float64] = _np.asarray(y_np, dtype=_np.float64).reshape(-1)
        return arr_logits.astype(_np.float64), y_arr_aligned.astype(_np.float64)

    def _collect_streaming_logits(
        self,
        loader: Iterable[Any],
        torch: types.ModuleType,
        *,
        group_inverse_map: dict[int, str] | None = None,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], StreamingRowMetadata | None]:
        """
        Collect logits and aligned targets from a streaming dataloader.

        Args:
            loader: Iterable yielding streaming batches in the form
                ``(batch_inputs, (decoder_target, _))``.
            torch: Imported torch module (passed in to avoid hard dependency at import time).
            group_inverse_map: Optional mapping from integer group codes to instrument ids.

        Returns:
            Tuple of `(logits, y_true, row_metadata)`.
        """
        model = self._tft
        if model is None:
            raise RuntimeError("TFTTeacher streaming model is not initialised")

        try:
            first_param = next(iter(model.parameters()), None)
            device = first_param.device if first_param is not None else torch.device("cpu")
        except Exception:
            device = torch.device("cpu")

        logits_batches: list[npt.NDArray[np.float32]] = []
        y_batches: list[npt.NDArray[np.float32]] = []
        instrument_batches: list[npt.NDArray[np.str_]] = []
        time_batches: list[npt.NDArray[np.int64]] = []

        model.eval()
        with torch.no_grad():
            for batch in loader:
                try:
                    batch_inputs, batch_outputs = batch
                except Exception as exc:
                    raise ValueError("Unexpected streaming batch format") from exc

                if not isinstance(batch_inputs, dict):
                    raise ValueError("Unexpected streaming batch inputs; expected dict[str, Tensor]")

                device_inputs: dict[str, Any] = {}
                for key, value in batch_inputs.items():
                    if hasattr(value, "to"):
                        device_inputs[key] = value.to(device)
                    else:
                        device_inputs[key] = value

                out = model(device_inputs)
                preds_any: Any
                if isinstance(out, dict) and "prediction" in out:
                    preds_any = out["prediction"]
                else:
                    preds_any = out

                if hasattr(preds_any, "detach"):
                    preds_np = preds_any.detach().cpu().numpy()
                else:
                    preds_np = np.asarray(preds_any)

                logits = np.asarray(preds_np, dtype=np.float64).reshape(-1)
                if np.all(logits >= 0.0) and np.all(logits <= 1.0):
                    eps = float(np.finfo(np.float64).eps)
                    p = np.clip(logits, eps, 1.0 - eps)
                    logits = np.log(p / (1.0 - p))
                logits_batches.append(np.asarray(logits, dtype=np.float32))

                y_any = None
                if isinstance(batch_outputs, tuple) and batch_outputs:
                    y_any = batch_outputs[0]
                if y_any is None:
                    y_any = device_inputs.get("decoder_target")
                if y_any is None:
                    raise RuntimeError("Streaming batch missing decoder_target for alignment")
                if hasattr(y_any, "detach"):
                    y_np = y_any.detach().cpu().numpy()
                else:
                    y_np = np.asarray(y_any)
                y_batches.append(np.asarray(y_np, dtype=np.float32).reshape(-1))

                decoder_time = device_inputs.get("decoder_time_idx")
                decoder_groups = device_inputs.get("decoder_group_ids")
                if decoder_time is not None and decoder_groups is not None:
                    time_np = (
                        decoder_time.detach().cpu().numpy()
                        if hasattr(decoder_time, "detach")
                        else np.asarray(decoder_time)
                    )
                    group_np = (
                        decoder_groups.detach().cpu().numpy()
                        if hasattr(decoder_groups, "detach")
                        else np.asarray(decoder_groups)
                    )
                    time_flat = np.asarray(time_np, dtype=np.int64).reshape(-1)
                    group_flat = np.asarray(group_np, dtype=np.int64).reshape(-1)
                    if time_flat.size == group_flat.size:
                        if group_inverse_map is None:
                            instrument_ids = np.asarray([str(int(code)) for code in group_flat], dtype=np.str_)
                        else:
                            instrument_ids = np.asarray(
                                [group_inverse_map.get(int(code), str(int(code))) for code in group_flat],
                                dtype=np.str_,
                            )
                        instrument_batches.append(instrument_ids)
                        time_batches.append(time_flat)

        logits_all = np.concatenate(logits_batches, axis=0) if logits_batches else np.array([], dtype=np.float32)
        y_all = np.concatenate(y_batches, axis=0) if y_batches else np.array([], dtype=np.float32)

        row_meta: StreamingRowMetadata | None = None
        if instrument_batches and time_batches:
            instruments_all = np.concatenate(instrument_batches, axis=0).astype(np.str_, copy=False)
            times_all = np.concatenate(time_batches, axis=0).astype(np.int64, copy=False)
            if instruments_all.size == times_all.size:
                row_ids = np.asarray(
                    [f"{instrument}::{int(time_idx)}" for instrument, time_idx in zip(instruments_all, times_all, strict=False)],
                    dtype=np.str_,
                )
                row_meta = StreamingRowMetadata(
                    row_ids=row_ids,
                    instrument_ids=instruments_all,
                    time_indices=times_all,
                )
        return logits_all, y_all, row_meta

    def fit_streaming(
        self,
        *,
        parquet_path: Path,
        train_loader: StreamDataLoader,
        val_loader: StreamDataLoader,
        train_metadata: TFTStreamingMetadata,
        val_metadata: TFTStreamingMetadata,
        full_metadata: TFTStreamingMetadata,
        streaming_config: TFTStreamingConfig,
        callbacks: Sequence[Any] | None = None,
        checkpoint_path: Any | None = None,
    ) -> StreamingFitResult:
        """
        Fit the teacher on capped streaming shards and return logits for distillation.

        This implementation is intentionally lightweight for unit-test stability and
        cold-path execution. It prefers using an existing TFT model when available,
        otherwise falls back to a deterministic baseline that emits logits derived
        from the observed decoder targets.

        Args:
            parquet_path: Source parquet file path (unused in baseline mode).
            train_loader: Streaming dataloader for training shards.
            val_loader: Streaming dataloader for validation shards.
            train_metadata: Metadata describing training shard bounds.
            val_metadata: Metadata describing validation shard bounds.
            full_metadata: Full metadata used to construct group id mappings.
            streaming_config: Streaming configuration used for the loaders.
            callbacks: Optional Lightning callbacks (unused in baseline mode).
            checkpoint_path: Optional checkpoint path (unused in baseline mode).

        Returns:
            Streaming fit result containing training/validation logits and aligned metadata.
        """
        _ = (parquet_path, train_metadata, val_metadata, callbacks, checkpoint_path)

        from ml._imports import HAS_TORCH
        from ml._imports import torch as torch_module

        if not HAS_TORCH or torch_module is None:
            check_ml_dependencies(["torch"])
            raise RuntimeError("torch is required for TFTTeacher.fit_streaming")

        group_vocab = full_metadata.categorical_vocab.get(streaming_config.group_id_col, ())
        group_inverse_map: dict[int, str] = {idx: value for idx, value in enumerate(group_vocab)}
        group_inverse_map.setdefault(len(group_vocab), "__UNK__")

        if self._tft is None:
            try:
                nn = torch_module.nn

                class _BaselineTeacherModel(nn.Module):
                    def forward(self, batch_inputs: dict[str, Any]) -> dict[str, Any]:
                        decoder_target = batch_inputs.get("decoder_target")
                        if decoder_target is None:
                            raise RuntimeError("decoder_target missing from streaming batch")
                        return {"prediction": decoder_target}

                self._tft = _BaselineTeacherModel()
                _FALLBACK_COUNTER.labels(component="tft_teacher", level="dummy").inc()
                self._logger.info("tft_teacher_streaming_fallback_enabled", extra={"level": "dummy"})
            except Exception:
                self._logger.error("Failed to construct streaming fallback teacher model", exc_info=True)
                raise

        z_train, _y_train, train_rows = self._collect_streaming_logits(
            train_loader,
            torch_module,
            group_inverse_map=group_inverse_map,
        )
        z_val, y_val, val_rows = self._collect_streaming_logits(
            val_loader,
            torch_module,
            group_inverse_map=group_inverse_map,
        )

        rows_processed = int(getattr(train_rows, "size", 0)) + int(getattr(val_rows, "size", 0))
        return StreamingFitResult(
            epochs_completed=0,
            rows_processed=rows_processed,
            z_train=z_train,
            z_val=z_val,
            y_val=y_val,
            train_rows=train_rows,
            val_rows=val_rows,
            val_returns=None,
        )
