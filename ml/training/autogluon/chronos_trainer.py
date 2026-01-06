"""
Chronos trainer for time series forecasting using AutoGluon.

This module provides a trainer that wraps AutoGluon's TimeSeriesPredictor
with Chronos foundation model presets for efficient time series forecasting
on massive datasets.

"""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ml._imports import HAS_AUTOGLUON
from ml._imports import HAS_PANDAS
from ml._imports import TimeSeriesDataFrame
from ml._imports import TimeSeriesPredictor
from ml._imports import check_ml_dependencies
from ml._imports import pd
from ml.config.autogluon import AutoGluonDataConfig
from ml.config.autogluon import ChronosTrainingConfig
from ml.data.autogluon_adapter import convert_to_timeseries_dataframe
from ml.data.autogluon_adapter import convert_to_timeseries_pandas
from ml.training.autogluon.soft_label_generator import build_distillation_dataset


if TYPE_CHECKING:
    import polars as _pl

    from ml.config.autogluon import ChronosDistillationConfig


__all__ = [
    "ChronosTrainer",
]


logger = logging.getLogger(__name__)


def _fit_supports_arg(predictor: TimeSeriesPredictor, name: str) -> bool:
    try:
        return name in inspect.signature(predictor.fit).parameters
    except (TypeError, ValueError):
        return False


def _cpu_hyperparameters(preset: str) -> dict[str, dict[str, Any]]:
    if preset == "chronos2":
        return {"Chronos2": {"model_path": "autogluon/chronos-2", "device": "cpu"}}
    return {"Chronos": {"model_path": preset, "device": "cpu"}}


class ChronosTrainer:
    """
    Trainer for Chronos foundation models via AutoGluon TimeSeries.

    This trainer provides a high-level interface for training time series
    forecasting models using AutoGluon's Chronos presets, which leverage
    pretrained transformer models for zero-shot or fine-tuned forecasting.

    Key features:
    - Supports Chronos-2 (best accuracy) and Chronos-Bolt (250x faster)
    - Native covariate support (macro, calendar, technical indicators)
    - Handles 4M+ row datasets efficiently (unlike PyTorch Forecasting TFT)
    - Teacher-student distillation within same model family
    - Forward return regression for granular trading signals

    Parameters
    ----------
    config : ChronosTrainingConfig
        Training configuration including preset, prediction length, etc.

    Examples
    --------
    >>> from ml.config.autogluon import ChronosTrainingConfig
    >>> config = ChronosTrainingConfig(
    ...     preset="chronos2",
    ...     prediction_length=15,
    ...     time_limit=1800,
    ... )
    >>> trainer = ChronosTrainer(config)
    >>> result = trainer.train(df)

    """

    def __init__(self, config: ChronosTrainingConfig) -> None:
        """
        Initialize the Chronos trainer.

        Parameters
        ----------
        config : ChronosTrainingConfig
            Training configuration.

        """
        if not HAS_AUTOGLUON:
            check_ml_dependencies(["autogluon"])

        self._config = config
        self._predictor: TimeSeriesPredictor | None = None
        self._is_fitted = False
        self._training_metrics: dict[str, Any] = {}
        self._feature_names: list[str] = []

    @property
    def config(self) -> ChronosTrainingConfig:
        """Return the training configuration."""
        return self._config

    @property
    def predictor(self) -> TimeSeriesPredictor | None:
        """Return the underlying TimeSeriesPredictor."""
        return self._predictor

    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been trained."""
        return self._is_fitted

    def train(
        self,
        data: _pl.DataFrame | Any,
        validation_data: _pl.DataFrame | Any | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Train a Chronos model on the provided dataset.

        Parameters
        ----------
        data : pl.DataFrame | TimeSeriesDataFrame
            Training data. Can be a Polars DataFrame (converted automatically)
            or an already-converted TimeSeriesDataFrame.
        validation_data : pl.DataFrame | TimeSeriesDataFrame, optional
            Validation data for evaluation. If None, uses internal split.
        **kwargs : Any
            Additional arguments passed to TimeSeriesPredictor.fit().

        Returns
        -------
        dict[str, Any]
            Training results including:
            - predictor: The trained TimeSeriesPredictor
            - metrics: Evaluation metrics
            - training_time: Time taken for training
            - feature_names: List of covariate features used

        """
        start_time = time.perf_counter()

        # Convert to TimeSeriesDataFrame if needed
        if isinstance(data, TimeSeriesDataFrame):
            train_tsdf = data
        else:
            logger.info("Converting training data to TimeSeriesDataFrame...")
            train_tsdf = convert_to_timeseries_dataframe(data, self._config)

        # Convert validation data if provided
        val_tsdf = None
        if validation_data is not None:
            if isinstance(validation_data, TimeSeriesDataFrame):
                val_tsdf = validation_data
            else:
                logger.info("Converting validation data to TimeSeriesDataFrame...")
                val_tsdf = convert_to_timeseries_dataframe(validation_data, self._config)

        logger.info(
            "Training Chronos model with preset='%s', prediction_length=%d, time_limit=%ds, "
            "num_val_windows=%d, refit_every_n_windows=%d, refit_full=%s, skip_model_selection=%s",
            self._config.preset,
            self._config.prediction_length,
            self._config.time_limit,
            self._config.num_val_windows,
            self._config.refit_every_n_windows,
            self._config.refit_full,
            self._config.skip_model_selection,
        )
        if self._config.tuning_config is not None:
            logger.info(
                "AutoGluon tuning enabled: num_trials=%d, scheduler=%s, searcher=%s",
                self._config.tuning_config.num_trials,
                self._config.tuning_config.scheduler,
                self._config.tuning_config.searcher,
            )

        # Prepare known covariates
        data_config = self._config.get_data_config()
        known_covariates = list(data_config.known_covariates) if data_config.known_covariates else None

        # Create TimeSeriesPredictor
        self._predictor = TimeSeriesPredictor(
            prediction_length=self._config.prediction_length,
            freq=self._config.freq,
            target="target",
            known_covariates_names=known_covariates,
            eval_metric=self._config.eval_metric.upper(),
            verbosity=self._config.verbosity,
            path=self._config.save_path,
        )

        # Fit the model
        fit_kwargs: dict[str, Any] = {
            "train_data": train_tsdf,
            "presets": self._config.preset,
            "time_limit": self._config.time_limit,
            "random_seed": self._config.random_seed,
            "enable_ensemble": self._config.enable_ensemble,
            "num_val_windows": self._config.num_val_windows,
            "refit_every_n_windows": self._config.refit_every_n_windows,
            "refit_full": self._config.refit_full,
            "skip_model_selection": self._config.skip_model_selection,
        }

        if val_tsdf is not None:
            fit_kwargs["tuning_data"] = val_tsdf

        if self._config.tuning_config is not None:
            fit_kwargs["hyperparameter_tune_kwargs"] = (
                self._config.tuning_config.as_autogluon_kwargs()
            )

        use_gpu = self._config.enable_gpu and self._config.num_gpus > 0
        if use_gpu:
            if _fit_supports_arg(self._predictor, "num_gpus"):
                fit_kwargs["num_gpus"] = self._config.num_gpus
            else:
                logger.info("AutoGluon fit() has no num_gpus argument; skipping GPU config")
        else:
            fit_kwargs.setdefault("hyperparameters", _cpu_hyperparameters(self._config.preset))

        # Add any additional kwargs
        fit_kwargs.update(kwargs)

        self._predictor.fit(**fit_kwargs)
        self._is_fitted = True

        # Calculate training time
        training_time = time.perf_counter() - start_time

        # Get evaluation metrics
        if val_tsdf is not None:
            try:
                eval_metrics = self._predictor.evaluate(val_tsdf)
                logger.info(f"Validation metrics: {eval_metrics}")
            except Exception as e:
                logger.warning(
                    f"Failed to evaluate on validation set: {e}",
                    exc_info=True,
                )
                eval_metrics = {}
        else:
            eval_metrics = {}

        # Get leaderboard for model comparison
        try:
            leaderboard = self._predictor.leaderboard()
            logger.info(f"Model leaderboard:\n{leaderboard.to_string()}")
        except Exception:
            leaderboard = None

        # Store training metadata
        self._training_metrics = {
            "training_time_seconds": training_time,
            "preset": self._config.preset,
            "prediction_length": self._config.prediction_length,
            "num_time_series": len(train_tsdf.item_ids) if hasattr(train_tsdf, "item_ids") else "unknown",
            "num_rows": len(train_tsdf),
            **eval_metrics,
        }

        self._feature_names = list(known_covariates) if known_covariates else []

        logger.info(f"Training completed in {training_time:.2f}s")

        return {
            "predictor": self._predictor,
            "metrics": self._training_metrics,
            "training_time": training_time,
            "feature_names": self._feature_names,
            "leaderboard": leaderboard,
        }

    def predict(
        self,
        data: _pl.DataFrame | Any,
        **kwargs: Any,
    ) -> npt.NDArray[np.float32]:
        """
        Generate predictions using the trained model.

        Parameters
        ----------
        data : pl.DataFrame | TimeSeriesDataFrame
            Data to generate predictions for.
        **kwargs : Any
            Additional arguments passed to predictor.predict().

        Returns
        -------
        np.ndarray
            Predictions as float32 array (mean forecast values).

        Raises
        ------
        ValueError
            If model has not been trained.

        """
        if not self._is_fitted or self._predictor is None:
            raise ValueError("Model must be trained before prediction")

        # Convert to TimeSeriesDataFrame if needed
        if isinstance(data, TimeSeriesDataFrame):
            tsdf = data
        else:
            tsdf = convert_to_timeseries_dataframe(data, self._config)

        data_config = self._config.get_data_config()
        known_covariates = list(data_config.known_covariates) if data_config.known_covariates else []
        provided_covariates = kwargs.get("known_covariates")
        if known_covariates and provided_covariates is None:
            kwargs["known_covariates"] = self._build_known_covariates(
                tsdf,
                known_covariates=known_covariates,
            )

        # Generate predictions
        predictions = self._predictor.predict(tsdf, **kwargs)

        # Extract mean predictions
        if hasattr(predictions, "values"):
            # For DataFrames, extract mean column
            if "mean" in predictions.columns:
                result: npt.NDArray[np.float32] = predictions["mean"].values.astype(np.float32)
                return result
            result = predictions.values.astype(np.float32)
            return result

        return np.asarray(predictions, dtype=np.float32)

    def generate_soft_labels(
        self,
        data: _pl.DataFrame | Any,
        **kwargs: Any,
    ) -> npt.NDArray[np.float64]:
        """
        Generate soft labels for student distillation.

        This method produces teacher predictions that can be used as
        soft targets for training a faster student model.

        Parameters
        ----------
        data : pl.DataFrame | TimeSeriesDataFrame
            Data to generate soft labels for.
        **kwargs : Any
            Additional arguments passed to prediction.

        Returns
        -------
        np.ndarray
            Soft labels as float64 array (mean predictions).

        """
        predictions = self.predict(data, **kwargs)
        return predictions.astype(np.float64)

    def _build_known_covariates(
        self,
        tsdf: TimeSeriesDataFrame,
        *,
        known_covariates: Sequence[str],
    ) -> TimeSeriesDataFrame:
        """
        Build a future known covariates frame for prediction.

        Args:
            tsdf: TimeSeriesDataFrame with historical data.
            known_covariates: Names of covariate columns.

        Returns:
            TimeSeriesDataFrame with future known covariates.
        """
        if not HAS_PANDAS or pd is None:
            check_ml_dependencies(["pandas"])
            raise ImportError("Pandas not available")

        df_pandas = tsdf.to_data_frame().reset_index()
        missing = [cov for cov in known_covariates if cov not in df_pandas.columns]
        if missing:
            raise ValueError(f"Missing known covariate columns: {missing}")

        freq = self._config.freq
        offset = pd.tseries.frequencies.to_offset(freq)
        rows: list[dict[str, Any]] = []

        for item_id, group in df_pandas.groupby("item_id", sort=False):
            group_sorted = group.sort_values("timestamp")
            last_row = group_sorted.iloc[-1]
            last_ts = last_row["timestamp"]
            if pd.isna(last_ts):
                continue
            future_index = pd.date_range(
                start=last_ts + offset,
                periods=int(self._config.prediction_length),
                freq=freq,
            )
            base_values = {cov: last_row[cov] for cov in known_covariates}
            for ts in future_index:
                rows.append({"item_id": item_id, "timestamp": ts, **base_values})

        if not rows:
            raise ValueError("Unable to construct known covariates for prediction")

        covariates_df = convert_to_timeseries_pandas(
            pd.DataFrame(rows),
            AutoGluonDataConfig(known_covariates=tuple(known_covariates)),
        )
        return TimeSeriesDataFrame.from_data_frame(
            covariates_df,
            id_column="item_id",
            timestamp_column="timestamp",
        )

    def save(self, path: str | Path | None = None) -> Path:
        """
        Save the trained predictor to disk.

        Parameters
        ----------
        path : str | Path, optional
            Path to save the predictor. If None, uses config save_path.

        Returns
        -------
        Path
            Path where predictor was saved.

        Raises
        ------
        ValueError
            If model has not been trained or no path specified.

        """
        if not self._is_fitted or self._predictor is None:
            raise ValueError("Model must be trained before saving")

        save_path = Path(path) if path else Path(self._config.save_path) if self._config.save_path else None

        if save_path is None:
            raise ValueError("No save path specified")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        predictor_path = getattr(self._predictor, "path", None)
        if predictor_path is not None and str(predictor_path) != str(save_path):
            try:
                setattr(self._predictor, "path", str(save_path))
            except Exception:
                logger.warning(
                    "Failed to update predictor path before save",
                    exc_info=True,
                )

        self._predictor.save()

        logger.info(f"Predictor saved to {save_path}")
        return save_path

    @classmethod
    def load(cls, path: str | Path) -> ChronosTrainer:
        """
        Load a trained predictor from disk.

        Parameters
        ----------
        path : str | Path
            Path to the saved predictor.

        Returns
        -------
        ChronosTrainer
            Trainer with loaded predictor.

        """
        if not HAS_AUTOGLUON:
            check_ml_dependencies(["autogluon"])

        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Predictor not found at {load_path}")

        predictor = TimeSeriesPredictor.load(str(load_path))

        # Create a minimal config (actual config stored with predictor)
        config = ChronosTrainingConfig(
            prediction_length=predictor.prediction_length,
            freq=predictor.freq or "min",
            save_path=str(load_path),
        )

        trainer = cls(config)
        trainer._predictor = predictor
        trainer._is_fitted = True

        logger.info(f"Predictor loaded from {load_path}")
        return trainer

    def persist(self) -> None:
        """
        Persist models in memory for fast repeated inference.

        This keeps models loaded in GPU/CPU memory to avoid
        reloading overhead on each prediction call.

        """
        if self._predictor is not None and self._config.persist_models:
            try:
                self._predictor.persist()
                logger.info("Models persisted in memory for fast inference")
            except Exception as e:
                logger.warning(f"Failed to persist models: {e}", exc_info=True)

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the trained model.

        Returns
        -------
        dict[str, Any]
            Model information including architecture, metrics, etc.

        """
        if not self._is_fitted or self._predictor is None:
            return {"status": "not_fitted"}

        info: dict[str, Any] = {
            "status": "fitted",
            "preset": self._config.preset,
            "prediction_length": self._config.prediction_length,
            "freq": self._config.freq,
            "eval_metric": self._config.eval_metric,
            "training_metrics": self._training_metrics,
            "feature_names": self._feature_names,
        }

        # Add model-specific info
        try:
            info["model_best"] = self._predictor.model_best
            info["leaderboard"] = self._predictor.leaderboard().to_dict()
        except Exception:
            pass

        return info


def train_teacher_student(
    data: _pl.DataFrame | Any,
    distillation_config: ChronosDistillationConfig,
    validation_data: _pl.DataFrame | Any | None = None,
) -> dict[str, Any]:
    """
    Train a Chronos teacher and distill to a student model.

    This function implements the full distillation pipeline:
    1. Train Chronos-2 teacher (best accuracy)
    2. Generate soft labels from teacher predictions
    3. Train Chronos-Bolt student on soft labels

    Parameters
    ----------
    data : pl.DataFrame
        Training dataset.
    distillation_config : ChronosDistillationConfig
        Configuration for teacher and student training.
    validation_data : pl.DataFrame, optional
        Validation dataset.

    Returns
    -------
    dict[str, Any]
        Results including:
        - teacher: Trained teacher ChronosTrainer
        - student: Trained student ChronosTrainer
        - soft_labels: Teacher predictions used as soft targets
        - metrics: Combined training metrics

    """
    logger.info("Starting Chronos teacher-student distillation pipeline")

    if not distillation_config.enable_distillation:
        logger.info("Distillation disabled; training teacher only")
        teacher = ChronosTrainer(distillation_config.teacher_config)
        teacher_result = teacher.train(data, validation_data=validation_data)
        return {
            "teacher": teacher,
            "student": None,
            "soft_labels": None,
            "soft_label_stats": None,
            "teacher_metrics": teacher_result["metrics"],
            "student_metrics": None,
        }

    # Train teacher
    logger.info(f"Training teacher with preset '{distillation_config.teacher_config.preset}'...")
    teacher = ChronosTrainer(distillation_config.teacher_config)
    teacher_result = teacher.train(data, validation_data=validation_data)

    if teacher.predictor is None:
        raise RuntimeError("Teacher training completed without a predictor")

    logger.info("Generating rolling soft labels from teacher predictions...")
    distilled = build_distillation_dataset(
        data,
        teacher.predictor,
        teacher_config=distillation_config.teacher_config,
        distillation_config=distillation_config,
    )

    coverage = distilled.stats.coverage
    logger.info(
        "Soft label coverage: %.3f (%d/%d eligible, %d sampled, %d/%d series)",
        coverage,
        distilled.stats.generated,
        distilled.stats.eligible_candidates,
        distilled.stats.total_candidates,
        distilled.stats.used_series,
        distilled.stats.total_series,
    )
    if coverage < distillation_config.min_soft_label_coverage:
        raise ValueError(
            "Soft label coverage below threshold: "
            f"{coverage:.3f} < {distillation_config.min_soft_label_coverage:.3f}"
        )

    # Export soft labels if configured
    if distillation_config.export_soft_labels:
        soft_labels_path = (
            Path(distillation_config.soft_labels_path)
            if distillation_config.soft_labels_path
            else Path(distillation_config.output_dir) / "soft_labels.parquet"
        )
        soft_labels_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if soft_labels_path.name.endswith(".csv") or soft_labels_path.name.endswith(".csv.gz"):
                distilled.labels.to_csv(soft_labels_path, index=False)
            else:
                distilled.labels.to_parquet(soft_labels_path, index=False)
            logger.info(f"Soft labels saved to {soft_labels_path}")
        except Exception as exc:
            logger.error(
                "Failed to export soft labels to %s: %s",
                soft_labels_path,
                exc,
                exc_info=True,
            )
            raise

    # Ensure student covariates match teacher (target column excluded)
    teacher_data_config = distillation_config.teacher_config.get_data_config()
    student_data_config = distillation_config.student_config.get_data_config()
    if (
        student_data_config.item_id_column != teacher_data_config.item_id_column
        or student_data_config.timestamp_column != teacher_data_config.timestamp_column
        or student_data_config.known_covariates != teacher_data_config.known_covariates
        or student_data_config.past_covariates != teacher_data_config.past_covariates
        or student_data_config.static_features != teacher_data_config.static_features
    ):
        raise ValueError("Student data_config must match teacher data_config for distillation")

    distilled_target = distillation_config.distilled_target_column
    student_data = AutoGluonDataConfig(
        item_id_column=teacher_data_config.item_id_column,
        timestamp_column=teacher_data_config.timestamp_column,
        target_column=distilled_target,
        known_covariates=teacher_data_config.known_covariates,
        past_covariates=teacher_data_config.past_covariates,
        static_features=teacher_data_config.static_features,
    )

    student_payload = distillation_config.student_config.dict()
    student_payload["target_column"] = distilled_target
    student_payload["data_config"] = student_data
    student_config = ChronosTrainingConfig(**student_payload)

    # Train student
    logger.info(f"Training student with preset '{student_config.preset}'...")
    student = ChronosTrainer(student_config)
    student_result = student.train(distilled.data, validation_data=validation_data)

    # Persist student for fast inference
    student.persist()

    logger.info("Distillation pipeline complete")

    return {
        "teacher": teacher,
        "student": student,
        "soft_labels": distilled.labels,
        "soft_label_stats": distilled.stats,
        "teacher_metrics": teacher_result["metrics"],
        "student_metrics": student_result["metrics"],
    }
