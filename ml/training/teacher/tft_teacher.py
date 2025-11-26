from __future__ import annotations

import logging
import types
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

from ml._imports import HAS_PANDAS
from ml._imports import HAS_TORCH
from ml._imports import check_ml_dependencies
from ml._imports import pd
from ml.training.teacher.base import BaseTeacher
from ml.training.teacher.base import TeacherConfig


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torch import Tensor as TorchTensor
    from torch.utils.data import DataLoader as TorchDataLoader

    from ml.training.teacher.streaming_loader import TFTStreamingConfig
    from ml.training.teacher.streaming_loader import TFTStreamingMetadata

    StreamingBatch = tuple[dict[str, TorchTensor], tuple[TorchTensor, None]]
else:  # pragma: no cover - optional dependency fallback
    TorchDataLoader = Any
    StreamingBatch = Any


def _to_int_time_indices(values: npt.NDArray[np.generic]) -> npt.NDArray[np.int64]:
    """Return integer time indices while preserving exact parquet values."""
    if np.issubdtype(values.dtype, np.floating):
        floored = np.rint(values.astype(np.float64, copy=False))
        return floored.astype(np.int64, copy=False)
    return values.astype(np.int64, copy=False)


@dataclass(frozen=True)
class TFTTeacherConfig(TeacherConfig):
    architecture: str = "TFT"
    #: Loss function name: "poisson" (default) or "bce".
    loss_name: str = "poisson"
    #: Optional positive class weight for BCE to handle class imbalance.
    pos_weight: float | None = None


@dataclass(frozen=True)
class StreamingRowMetadata:
    """Row-level identifiers used to align logits across ensemble members."""

    row_ids: npt.NDArray[np.str_]
    instrument_ids: npt.NDArray[np.str_]
    time_indices: npt.NDArray[np.int64]

    @property
    def size(self) -> int:
        return int(self.row_ids.size)

    @classmethod
    def empty(cls) -> StreamingRowMetadata:
        empty_str = np.empty(0, dtype=np.str_)
        empty_int = np.empty(0, dtype=np.int64)
        return cls(
            row_ids=empty_str,
            instrument_ids=empty_str,
            time_indices=empty_int,
        )


@dataclass(frozen=True)
class StreamingFitResult:
    """
    Container for logits generated during streaming training.

    Attributes:
        z_train: Logits for the training shard set, flattened to 1-D.
        z_val: Logits for the validation shard set, flattened to 1-D.
        y_val: Validation targets aligned with ``z_val``.
        train_rows: Row metadata describing the training logits order (optional).
        val_rows: Row metadata describing the validation logits order (optional).
        val_returns: Optional forward returns aligned with ``z_val``.
    """

    z_train: npt.NDArray[np.float64]
    z_val: npt.NDArray[np.float64]
    y_val: npt.NDArray[np.float64]
    train_rows: StreamingRowMetadata | None = None
    val_rows: StreamingRowMetadata | None = None
    val_returns: npt.NDArray[np.float64] | None = None


@dataclass(slots=True)
class _StreamingState:
    parquet_path: Path
    train_metadata: Any
    val_metadata: Any
    full_metadata: Any
    config: Any


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
        self.optimizer = optimizer.strip().lower() if optimizer else None
        self.lr_scheduler = lr_scheduler.strip().lower() if lr_scheduler else None

        # Runtime state
        self._training_dataset: Any | None = None
        self._tft: Any | None = None
        self._trainer: Any | None = None
        self._streaming_state: _StreamingState | None = None
        self._decoder_group_ids_seen: bool = False
        self._decoder_group_shape_mismatch: bool = False
        self._decoder_group_debug_logged: bool = False

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

    def fit_streaming(
        self,
        parquet_path: Path,
        train_loader: TorchDataLoader[StreamingBatch],
        val_loader: TorchDataLoader[StreamingBatch],
        *,
        train_metadata: TFTStreamingMetadata,
        val_metadata: TFTStreamingMetadata,
        full_metadata: TFTStreamingMetadata,
        streaming_config: TFTStreamingConfig,
        callbacks: Sequence[Any] | None = None,
        checkpoint_path: Path | str | None = None,
    ) -> StreamingFitResult:
        """
        Train using streaming dataloaders to avoid materialising the full dataset.

        Args:
            parquet_path: Path to the source parquet dataset used for streaming loaders.
            train_loader: Streaming dataloader supplying training batches.
            val_loader: Streaming dataloader supplying validation batches.
            train_metadata: Metadata describing the training shard selection.
            val_metadata: Metadata describing the validation shard selection.
            full_metadata: Combined metadata prior to train/validation splits.
            streaming_config: Streaming configuration applied to loader construction.
            callbacks: Optional Lightning callbacks invoked during training (e.g., checkpointing).
            checkpoint_path: Optional checkpoint file used to resume the Lightning trainer.
        """
        if not HAS_TORCH:
            check_ml_dependencies(["torch"])
        try:
            import torch
        except Exception as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("PyTorch is required for streaming training") from exc

        try:
            from pytorch_forecasting import TemporalFusionTransformer
            from pytorch_forecasting import TimeSeriesDataSet
            from pytorch_forecasting.metrics import PoissonLoss
        except Exception as exc:  # pragma: no cover
            raise ImportError("pytorch-forecasting is required for streaming training") from exc

        from ml.training.teacher.losses import BCEWithLogitsLossPF

        cfg = cast(TFTTeacherConfig, self.config)

        categorical_vocab = full_metadata.categorical_vocab
        cat_cardinalities = {
            name: len(vocab) + 1 for name, vocab in categorical_vocab.items()
        }
        embedding_sizes = {
            name: (cardinality, min(50, (cardinality + 1) // 2))
            for name, cardinality in cat_cardinalities.items()
        }

        if cfg.loss_name.lower() == "bce":
            loss_obj: Any = BCEWithLogitsLossPF(pos_weight=cfg.pos_weight)
        else:
            loss_obj = PoissonLoss()

        static_reals = list(streaming_config.static_reals)
        known_reals = list(streaming_config.time_varying_known_reals)
        unknown_reals = list(streaming_config.time_varying_unknown_reals)
        static_cats = list(streaming_config.static_categoricals)

        template_dataset = self._build_streaming_template_dataset(
            metadata=full_metadata,
            streaming_config=streaming_config,
            static_categoricals=tuple(static_cats),
            static_reals=tuple(static_reals),
            known_reals=tuple(known_reals),
            unknown_reals=tuple(unknown_reals),
            dataset_cls=TimeSeriesDataSet,
        )

        model_kwargs: dict[str, Any] = {
            "hidden_size": self.hidden_size,
            "lstm_layers": self.lstm_layers,
            "dropout": self.dropout,
            "output_size": 1,
            "loss": loss_obj,
            "attention_head_size": self.attention_head_size,
            "learning_rate": self.learning_rate,
            "log_interval": 100,
            "embedding_sizes": embedding_sizes,
        }
        if self.optimizer:
            model_kwargs["optimizer"] = self.optimizer
        if self.lr_scheduler:
            model_kwargs["lr_scheduler"] = self.lr_scheduler
        try:
            model = TemporalFusionTransformer.from_dataset(
                template_dataset,
                **model_kwargs,
            )
        except TypeError:
            fallback_kwargs = {
                key: value for key, value in model_kwargs.items() if key not in {"optimizer", "lr_scheduler"}
            }
            if len(fallback_kwargs) == len(model_kwargs):
                raise
            self._logger.debug(
                "TemporalFusionTransformer.from_dataset rejected optimizer or scheduler arguments; retrying without extras",
                exc_info=True,
            )
            model = TemporalFusionTransformer.from_dataset(
                template_dataset,
                **fallback_kwargs,
            )

        precision_arg: Any = self._precision
        trainer_cls: Any
        try:  # pragma: no cover - prefer Lightning 2.x
            import lightning.pytorch as _lpl

            if isinstance(precision_arg, str):
                if precision_arg == "16":
                    precision_arg = "16-mixed"
                elif precision_arg.lower() == "bf16":
                    precision_arg = "bf16-mixed"
            trainer_cls = _lpl.Trainer
        except Exception:  # pragma: no cover - fall back to PL 1.x
            import pytorch_lightning as _pl

            trainer_cls = _pl.Trainer

        callbacks_param = list(callbacks) if callbacks is not None else None
        trainer = trainer_cls(
            max_epochs=self.max_epochs,
            gradient_clip_val=1.0,
            enable_progress_bar=False,
            logger=False,
            callbacks=callbacks_param,
            enable_checkpointing=bool(callbacks_param),
            accelerator=self._accelerator,
            devices=self._devices,
            precision=precision_arg,
        )

        self._tft = model
        self._trainer = trainer
        fit_kwargs: dict[str, Any] = {
            "train_dataloaders": train_loader,
            "val_dataloaders": val_loader,
        }
        if checkpoint_path is not None:
            ckpt_path_str = str(checkpoint_path)
            if ckpt_path_str:
                fit_kwargs["ckpt_path"] = ckpt_path_str
                logger.info(
                    "lightning_resume_from_checkpoint",
                    extra={"checkpoint_path": ckpt_path_str},
                )

        trainer.fit(model, **fit_kwargs)

        group_inverse_map = self._build_group_inverse_map(full_metadata, streaming_config)
        train_logits, _, train_rows = self._collect_streaming_logits(
            train_loader,
            torch,
            group_inverse_map=group_inverse_map,
        )
        val_logits, val_targets, val_rows = self._collect_streaming_logits(
            val_loader,
            torch,
            group_inverse_map=group_inverse_map,
        )

        self._streaming_state = _StreamingState(
            parquet_path=parquet_path,
            train_metadata=train_metadata,
            val_metadata=val_metadata,
            full_metadata=full_metadata,
            config=streaming_config,
        )
        self._is_fitted = True
        return StreamingFitResult(
            z_train=train_logits,
            z_val=val_logits,
            y_val=val_targets,
            train_rows=train_rows,
            val_rows=val_rows,
        )

    def _collect_streaming_logits(
        self,
        loader: TorchDataLoader[StreamingBatch],
        torch_mod: Any,
        *,
        group_inverse_map: Mapping[int, str],
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        StreamingRowMetadata | None,
    ]:
        """Return concatenated logits and targets from a streaming DataLoader."""
        if self._tft is None:
            raise RuntimeError("Streaming model not initialised")

        model_device: Any | None = None
        parameters_method = getattr(self._tft, "parameters", None)
        if callable(parameters_method):
            try:
                first_param = next(parameters_method())
                model_device = getattr(first_param, "device", None)
            except StopIteration:
                model_device = None
            except TypeError:
                model_device = None
        if model_device is None and hasattr(torch_mod, "device"):
            try:
                model_device = torch_mod.device("cpu")
            except Exception:  # pragma: no cover - defensive guard
                model_device = None
        device_type = getattr(model_device, "type", "cpu") if model_device is not None else "cpu"
        non_blocking = device_type != "cpu"

        preds: list[npt.NDArray[np.float64]] = []
        targets: list[npt.NDArray[np.float64]] = []
        row_instruments: list[str] = []
        row_times: list[int] = []
        metadata_enabled = True
        self._decoder_group_ids_seen = False
        self._decoder_group_shape_mismatch = False
        self._decoder_group_debug_logged = False
        self._tft.eval()
        with torch_mod.no_grad():
            for batch_inputs, (decoder_target, _) in loader:
                metadata_payload: tuple[list[str], list[int]] | None = None
                if metadata_enabled:
                    metadata_payload = self._extract_row_metadata(
                        batch_inputs,
                        group_inverse_map=group_inverse_map,
                    )
                    if metadata_payload is None:
                        logger.debug(
                            "streaming worker could not extract row metadata from batch",
                            extra={"available_keys": tuple(batch_inputs.keys())},
                        )
                if model_device is not None:
                    batch_inputs = {
                        key: (
                            tensor.to(model_device, non_blocking=non_blocking)
                            if hasattr(tensor, "to") and getattr(tensor, "device", None) != model_device
                            else tensor
                        )
                        for key, tensor in batch_inputs.items()
                    }
                    if hasattr(decoder_target, "to") and getattr(decoder_target, "device", None) != model_device:
                        decoder_target = decoder_target.to(model_device, non_blocking=non_blocking)
                output = self._tft(batch_inputs)
                prediction = output["prediction"]
                preds.append(
                    prediction.detach().cpu().numpy().reshape(-1),
                )
                targets.append(
                    decoder_target.detach().cpu().numpy().reshape(-1),
                )
                if metadata_enabled and metadata_payload is not None:
                    instruments, times = metadata_payload
                    if instruments and times and len(instruments) == len(times):
                        row_instruments.extend(instruments)
                        row_times.extend(times)
                    else:
                        metadata_enabled = False
                        logger.debug(
                            "streaming worker received empty or mismatched row metadata payload",
                            extra={
                                "instrument_count": len(instruments),
                                "time_count": len(times),
                            },
                        )
                elif metadata_payload is None:
                    metadata_enabled = False

        if preds:
            logits = np.concatenate(preds, axis=0).astype(np.float64, copy=False)
            y = np.concatenate(targets, axis=0).astype(np.float64, copy=False)
        else:
            logits = np.empty(0, dtype=np.float64)
            y = np.empty(0, dtype=np.float64)
        row_metadata: StreamingRowMetadata | None
        if logits.size == 0:
            row_metadata = StreamingRowMetadata.empty()
        elif metadata_enabled and len(row_instruments) == logits.size:
            row_metadata = self._build_row_metadata(row_instruments, row_times)
        else:
            logger.debug(
                "streaming worker falling back to metadata-derived alignment",
                extra={
                    "metadata_enabled": metadata_enabled,
                    "collected_instruments": len(row_instruments),
                    "logit_count": logits.size,
                },
            )
            row_metadata = self._fallback_row_metadata(loader, group_inverse_map, logits.size)
        if not self._decoder_group_ids_seen:
            logger.info(
                "streaming worker decoder group ids unavailable; reverting to legacy metadata extraction",
                extra={
                    "collected_rows": len(row_instruments),
                    "logit_count": logits.size,
                    "shape_mismatch_detected": self._decoder_group_shape_mismatch,
                },
            )
        return logits, y, row_metadata

    def _extract_row_metadata(
        self,
        batch_inputs: dict[str, Any],
        *,
        group_inverse_map: Mapping[int, str],
    ) -> tuple[list[str], list[int]] | None:
        time_tensor = batch_inputs.get("decoder_time_idx")
        if time_tensor is None:
            logger.debug("streaming worker batch missing decoder time metadata")
            return None
        decoder_group_tensor = batch_inputs.get("decoder_group_ids")
        group_tensor = batch_inputs.get("groups") if decoder_group_tensor is None else None
        try:
            detached_times = time_tensor.detach().cpu().numpy()
        except Exception:
            logger.debug(
                "streaming worker failed to detach decoder metadata tensors",
                exc_info=True,
            )
            return None
        if detached_times.ndim != 2:
            detached_times = np.reshape(detached_times, (detached_times.shape[0], -1))
        time_values = _to_int_time_indices(detached_times)
        instruments: list[str] = []
        times: list[int] = []
        if decoder_group_tensor is not None:
            try:
                decoder_group_values = decoder_group_tensor.detach().cpu().numpy()
            except Exception:
                logger.debug(
                    "streaming worker failed to detach decoder group ids",
                    exc_info=True,
                )
                return None
            if decoder_group_values.ndim != 2:
                decoder_group_values = np.reshape(decoder_group_values, (decoder_group_values.shape[0], -1))
            if decoder_group_values.shape != time_values.shape:
                self._decoder_group_shape_mismatch = True
                logger.debug(
                    "streaming worker decoder group ids shape mismatch",
                    extra={
                        "group_shape": tuple(decoder_group_values.shape),
                        "time_shape": tuple(time_values.shape),
                    },
                )
                return None
            self._decoder_group_ids_seen = True
            if not self._decoder_group_debug_logged:
                sample_codes = decoder_group_values[0][: min(5, decoder_group_values.shape[1])].astype(int, copy=False).tolist()
                sample_instruments = [group_inverse_map.get(int(code), "__UNK__") for code in sample_codes]
                sample_raw_times = time_values[0][: min(5, time_values.shape[1])].astype(int, copy=False).tolist()
                logger.info(
                    "streaming worker decoder group id sample (codes=%s, mapped=%s, raw_times=%s)",
                    sample_codes,
                    sample_instruments,
                    sample_raw_times,
                    extra={
                        "group_codes": sample_codes,
                        "mapped_instruments": sample_instruments,
                        "raw_times": sample_raw_times,
                    },
                )
                self._decoder_group_debug_logged = True
            for row_codes, row_times in zip(decoder_group_values, time_values, strict=False):
                seq_codes = np.asarray(row_codes, dtype=np.int64)
                seq_times = np.asarray(row_times, dtype=np.int64)
                if seq_codes.size != seq_times.size:
                    self._decoder_group_shape_mismatch = True
                    logger.debug(
                        "streaming worker decoder group ids length mismatch",
                        extra={
                            "codes": int(seq_codes.size),
                            "times": int(seq_times.size),
                        },
                    )
                    return None
                instruments.extend(
                    [group_inverse_map.get(int(code), "__UNK__") for code in seq_codes.tolist()]
                )
                times.extend(seq_times.tolist())
        else:
            if group_tensor is None:
                logger.debug("streaming worker batch missing group metadata")
                return None
            try:
                group_values = group_tensor.detach().cpu().numpy()
            except Exception:
                logger.debug(
                    "streaming worker failed to detach group metadata tensor",
                    exc_info=True,
                )
                return None
            group_flat = np.reshape(group_values, (group_values.shape[0], -1))
            if group_flat.shape[1] == 0:
                logger.debug("streaming worker received empty group metadata payload")
                return None
            group_codes = group_flat[:, 0]
            for code, row_times in zip(group_codes, time_values, strict=False):
                instrument = group_inverse_map.get(int(code), "__UNK__")
                seq_times = np.asarray(row_times, dtype=np.int64)
                instruments.extend([instrument] * seq_times.size)
                times.extend(seq_times.tolist())
        if len(instruments) != len(times):
            logger.debug(
                "streaming worker produced mismatched instrument/time metadata counts",
                extra={
                    "instrument_count": len(instruments),
                    "time_count": len(times),
                },
            )
            return None
        return instruments, times

    def _build_row_metadata(
        self,
        instruments: list[str],
        times: list[int],
    ) -> StreamingRowMetadata:
        instrument_array = np.asarray(instruments, dtype=np.str_)
        time_array = np.asarray(times, dtype=np.int64)
        row_ids = np.asarray(
            [f"{instrument}::{time_value}" for instrument, time_value in zip(instruments, times, strict=False)],
            dtype=np.str_,
        )
        return StreamingRowMetadata(
            row_ids=row_ids,
            instrument_ids=instrument_array,
            time_indices=time_array,
        )

    def _fallback_row_metadata(
        self,
        loader: TorchDataLoader[StreamingBatch],
        group_inverse_map: Mapping[int, str],
        expected_size: int,
    ) -> StreamingRowMetadata | None:
        """
        Derive row ordering from shard metadata when batches omit decoder metadata.
        """
        instruments: list[str] = []
        times: list[int] = []
        for batch_inputs, _ in loader:
            time_tensor = batch_inputs.get("decoder_time_idx")
            decoder_group_tensor = batch_inputs.get("decoder_group_ids")
            if time_tensor is None:
                continue
            try:
                detached_times = time_tensor.detach().cpu().numpy()
            except Exception:
                logger.debug("fallback metadata extraction failed to detach time tensor", exc_info=True)
                return None
            if detached_times.ndim != 2:
                detached_times = np.reshape(detached_times, (detached_times.shape[0], -1))
            time_values = _to_int_time_indices(detached_times)
            if decoder_group_tensor is not None:
                try:
                    decoder_group_values = decoder_group_tensor.detach().cpu().numpy()
                except Exception:
                    logger.debug("fallback metadata extraction failed to detach decoder group ids", exc_info=True)
                    return None
                if decoder_group_values.ndim != 2:
                    decoder_group_values = np.reshape(decoder_group_values, (decoder_group_values.shape[0], -1))
            else:
                group_tensor = batch_inputs.get("groups")
                if group_tensor is None:
                    continue
                try:
                    base_group_values = group_tensor.detach().cpu().numpy()
                except Exception:
                    logger.debug("fallback metadata extraction failed to detach group tensor", exc_info=True)
                    return None
                base_group_values = np.reshape(base_group_values, (base_group_values.shape[0], -1))
                if base_group_values.shape[1] == 0:
                    continue
                first_codes = base_group_values[:, 0]
                decoder_group_values = np.tile(first_codes[:, None], (1, time_values.shape[1]))
            time_flat = np.reshape(time_values, (time_values.shape[0], -1))
            if decoder_group_values.shape != time_flat.shape:
                logger.debug(
                    "fallback metadata extraction decoder ids shape mismatch",
                    extra={
                        "group_shape": tuple(decoder_group_values.shape),
                        "time_shape": tuple(time_flat.shape),
                    },
                )
                return None
            for row_codes, row_times in zip(decoder_group_values, time_flat, strict=False):
                seq_times = np.asarray(row_times, dtype=np.int64)
                instruments.extend(
                    [group_inverse_map.get(int(code), "__UNK__") for code in np.asarray(row_codes, dtype=np.int64)]
                )
                times.extend(seq_times.tolist())
        if len(instruments) != expected_size or len(times) != expected_size:
            logger.debug(
                "fallback metadata size mismatch",
                extra={
                    "instrument_count": len(instruments),
                    "time_count": len(times),
                    "expected": expected_size,
                },
            )
            return None
        return self._build_row_metadata(instruments, times)

    def _build_group_inverse_map(
        self,
        metadata: TFTStreamingMetadata,
        config: TFTStreamingConfig,
    ) -> dict[int, str]:
        vocab = metadata.categorical_vocab.get(config.group_id_col)
        if not vocab:
            return {}
        mapping: dict[str, int] = {}
        for idx, raw_value in enumerate(vocab):
            mapping[str(raw_value)] = idx
        if "__UNK__" not in mapping:
            mapping["__UNK__"] = len(mapping)
        inverse: dict[int, str] = {code: value for value, code in mapping.items()}
        inverse.setdefault(mapping["__UNK__"], "__UNK__")
        return inverse

    def _build_streaming_template_dataset(
        self,
        *,
        metadata: TFTStreamingMetadata,
        streaming_config: TFTStreamingConfig,
        static_categoricals: Sequence[str],
        static_reals: Sequence[str],
        known_reals: Sequence[str],
        unknown_reals: Sequence[str],
        dataset_cls: Any,
    ) -> Any:
        """Create a minimal TimeSeriesDataSet that mirrors the streaming configuration."""
        if pd is None:
            raise RuntimeError("pandas is required for streaming template dataset construction")
        if not metadata.instrument_row_counts:
            raise ValueError("Streaming metadata must include instrument row counts")

        encoder_len = max(1, streaming_config.max_encoder_length)
        decoder_len = max(1, streaming_config.max_prediction_length)
        total_len = encoder_len + decoder_len
        instrument_id = sorted(metadata.instrument_row_counts.keys())[0]

        template_data: dict[str, list[Any]] = {
            streaming_config.time_idx_col: list(range(total_len)),
            streaming_config.group_id_col: [instrument_id] * total_len,
            streaming_config.target_col: [0.0] * total_len,
        }

        for column in static_categoricals:
            if column == streaming_config.group_id_col:
                value = instrument_id
            else:
                vocab = metadata.categorical_vocab.get(column)
                value = vocab[0] if vocab else "__UNK__"
            template_data[column] = [value] * total_len

        def _numeric_value(name: str) -> float:
            stats = metadata.numeric_stats.get(name)
            if stats is None or stats.count <= 0:
                return 0.0
            return float(stats.mean)

        for column in static_reals:
            template_data[column] = [_numeric_value(column)] * total_len

        for column in known_reals:
            mean_value = _numeric_value(column)
            template_data[column] = [float(mean_value)] * total_len
        for column in unknown_reals:
            mean_value = _numeric_value(column)
            template_data[column] = [float(mean_value)] * total_len

        frame = pd.DataFrame(template_data)
        dataset = dataset_cls(
            frame,
            time_idx=streaming_config.time_idx_col,
            target=streaming_config.target_col,
            group_ids=[streaming_config.group_id_col],
            max_encoder_length=streaming_config.max_encoder_length,
            max_prediction_length=streaming_config.max_prediction_length,
            min_encoder_length=1,
            min_prediction_length=1,
            static_categoricals=list(static_categoricals),
            static_reals=list(static_reals),
            time_varying_known_reals=list(known_reals),
            time_varying_unknown_reals=list(unknown_reals),
            allow_missing_timesteps=True,
            add_encoder_length=False,
        )
        return dataset

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
