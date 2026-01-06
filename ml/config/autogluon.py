"""
AutoGluon TimeSeries configuration for Chronos foundation models.

This module provides configuration classes for training time series models
using AutoGluon's Chronos presets, supporting both teacher training and
student distillation.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from nautilus_trader.common.config import NautilusConfig
from nautilus_trader.common.config import NonNegativeFloat
from nautilus_trader.common.config import NonNegativeInt
from nautilus_trader.common.config import PositiveFloat
from nautilus_trader.common.config import PositiveInt


if TYPE_CHECKING:
    pass


__all__ = [
    "AutoGluonDataConfig",
    "ChronosDistillationConfig",
    "ChronosTrainingConfig",
    "ChronosTuningConfig",
]


# Valid Chronos presets for AutoGluon TimeSeries
ChronosPreset = Literal[
    "chronos_tiny",
    "chronos_mini",
    "chronos_small",
    "chronos_base",
    "chronos_large",
    "chronos2",  # Chronos-2 (120M params, best accuracy)
    "bolt_tiny",
    "bolt_mini",
    "bolt_small",  # Chronos-Bolt (250x faster inference)
    "bolt_base",
]


class AutoGluonDataConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Configuration for AutoGluon TimeSeriesDataFrame conversion.

    Parameters
    ----------
    item_id_column : str, default "instrument_id"
        Column name for time series identifier (maps to item_id).
    timestamp_column : str, default "ts_event"
        Column name for timestamps (nanoseconds, converted to datetime).
    target_column : str, default "forward_return"
        Column name for prediction target.
    known_covariates : tuple[str, ...], default ()
        Feature columns known at prediction time (calendar, macro).
    past_covariates : tuple[str, ...], default ()
        Feature columns only known historically (returns, volume).
    static_features : tuple[str, ...], default ()
        Time-invariant features (asset class, exchange).

    """

    item_id_column: str = "instrument_id"
    timestamp_column: str = "ts_event"
    target_column: str = "forward_return"
    known_covariates: tuple[str, ...] = ()
    past_covariates: tuple[str, ...] = ()
    static_features: tuple[str, ...] = ()


class ChronosTuningConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Configuration for AutoGluon hyperparameter tuning.

    Parameters
    ----------
    num_trials : PositiveInt
        Number of HPO trials to run.
    scheduler : str, default "local"
        Scheduler name passed to AutoGluon (e.g., "local", "ray").
    searcher : str, default "random"
        Searcher name passed to AutoGluon (e.g., "random", "bayes").

    """

    num_trials: PositiveInt
    scheduler: str = "local"
    searcher: str = "random"

    def __post_init__(self) -> None:
        """Validate tuning configuration values."""
        if not self.scheduler:
            raise ValueError("scheduler must be a non-empty string")
        if not self.searcher:
            raise ValueError("searcher must be a non-empty string")

    def as_autogluon_kwargs(self) -> dict[str, int | str]:
        """Return AutoGluon hyperparameter_tune_kwargs payload."""
        return {
            "num_trials": int(self.num_trials),
            "scheduler": self.scheduler,
            "searcher": self.searcher,
        }


class ChronosTrainingConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Configuration for Chronos model training via AutoGluon TimeSeries.

    This config supports both teacher training (chronos2 preset) and
    student training (bolt_small preset) with the same interface.

    Parameters
    ----------
    prediction_length : PositiveInt, default 15
        Forecast horizon in time steps (minutes for minute-frequency data).
    freq : str, default "min"
        Time series frequency. Use "min" for minute, "h" for hourly, "D" for daily.
    target_column : str, default "forward_return"
        Column name for the regression target.
    eval_metric : str, default "RMSE"
        Evaluation metric for model selection. Options: RMSE, MAE, MAPE, MASE, etc.
    preset : ChronosPreset, default "chronos2"
        AutoGluon preset for model architecture:
        - "chronos2": Best accuracy, 120M params (teacher)
        - "bolt_small": 250x faster inference (student)
    time_limit : PositiveInt, default 3600
        Maximum training time in seconds.
    enable_ensemble : bool, default True
        Whether to build ensembles when model selection is enabled.
    num_val_windows : PositiveInt, default 1
        Number of rolling validation windows for tuning/backtesting.
    refit_every_n_windows : PositiveInt, default 1
        Refit cadence for rolling windows (1 = refit every window).
    refit_full : bool, default False
        Whether to refit the best model on the full dataset.
    skip_model_selection : bool, default False
        Whether to skip model selection (False enables tuning).
    data_config : AutoGluonDataConfig, optional
        Configuration for data conversion. Uses defaults if None and must
        keep target_column aligned with this config (ts_event required).
    tuning_config : ChronosTuningConfig, optional
        Hyperparameter tuning configuration for AutoGluon model selection.
    enable_gpu : bool, default True
        Whether to use GPU acceleration if available.
    num_gpus : NonNegativeInt, default 1
        Number of GPUs to use. Set to 0 for CPU-only training.
    persist_models : bool, default True
        Whether to keep models in memory for fast inference.
    random_seed : NonNegativeInt, default 42
        Random seed for reproducibility.
    save_path : str, optional
        Path to save the trained predictor.
    verbosity : NonNegativeInt, default 2
        Logging verbosity (0=silent, 1=warnings, 2=info, 3=debug).

    Examples
    --------
    >>> config = ChronosTrainingConfig(
    ...     prediction_length=15,
    ...     preset="chronos2",
    ...     time_limit=1800,
    ... )

    """

    # AutoGluon TimeSeries settings
    prediction_length: PositiveInt = 15
    freq: str = "min"
    target_column: str = "forward_return"
    eval_metric: str = "RMSE"

    # Chronos-specific settings
    preset: str = "chronos2"  # Type as str for msgspec, validated in __post_init__
    time_limit: PositiveInt = 3600
    enable_ensemble: bool = True
    num_val_windows: PositiveInt = 1
    refit_every_n_windows: PositiveInt = 1
    refit_full: bool = False
    skip_model_selection: bool = False

    # Data configuration
    data_config: AutoGluonDataConfig | None = None
    tuning_config: ChronosTuningConfig | None = None

    # Hardware settings
    enable_gpu: bool = True
    num_gpus: NonNegativeInt = 1
    persist_models: bool = True

    # Reproducibility
    random_seed: NonNegativeInt = 42

    # Output
    save_path: str | None = None
    verbosity: NonNegativeInt = 2

    def __post_init__(self) -> None:
        """Validate configuration values."""
        valid_presets = {
            "chronos_tiny",
            "chronos_mini",
            "chronos_small",
            "chronos_base",
            "chronos_large",
            "chronos2",
            "bolt_tiny",
            "bolt_mini",
            "bolt_small",
            "bolt_base",
        }
        if self.preset not in valid_presets:
            raise ValueError(
                f"Invalid preset '{self.preset}'. Must be one of: {sorted(valid_presets)}"
            )

        valid_metrics = {"RMSE", "MAE", "MAPE", "MASE", "SMAPE", "WAPE", "MSE"}
        # Handle special case for sMAPE (lowercase 's')
        metric_upper = self.eval_metric.upper()
        if metric_upper not in valid_metrics and self.eval_metric not in {"sMAPE", "smape"}:
            raise ValueError(
                f"Invalid eval_metric '{self.eval_metric}'. Must be one of: {sorted(valid_metrics)}"
            )
        if self.data_config is not None:
            if self.data_config.target_column != self.target_column:
                raise ValueError(
                    "data_config.target_column must match ChronosTrainingConfig.target_column"
                )
            if self.data_config.timestamp_column != "ts_event":
                raise ValueError(
                    "Chronos training requires ts_event as the timestamp column"
                )
        if self.tuning_config is not None and self.skip_model_selection:
            raise ValueError("tuning_config requires skip_model_selection=False")

    def get_data_config(self) -> AutoGluonDataConfig:
        """Return data config, using defaults if not specified."""
        if self.data_config is not None:
            return self.data_config
        return AutoGluonDataConfig(target_column=self.target_column)


class ChronosDistillationConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Configuration for Chronos teacher-student distillation.

    Supports distillation from Chronos-2 teacher to Chronos-Bolt student
    for production inference with 250x speedup.

    Parameters
    ----------
    teacher_config : ChronosTrainingConfig
        Configuration for teacher model (typically chronos2 preset).
    student_config : ChronosTrainingConfig
        Configuration for student model (typically bolt_small preset).
    enable_distillation : bool, default True
        Whether to perform knowledge distillation from teacher to student.
    soft_label_temperature : PositiveFloat, default 1.0
        Temperature for soft label generation (higher = softer labels).
    distillation_alpha : NonNegativeFloat, default 0.5
        Weight for distillation loss vs. hard label loss (0-1).
    label_strategy : str, default "blend"
        Distillation target strategy ("teacher_only" or "blend").
    soft_target_column : str, default "soft_target"
        Column name for teacher soft labels.
    distilled_target_column : str, default "distilled_target"
        Column name for the final student target (after blending, if enabled).
    forecast_step : PositiveInt, default 1
        Forecast step to align soft labels (1 = next timestep).
    min_history : PositiveInt, default 75
        Minimum history length per series before generating a label.
    stride : PositiveInt, default 15
        Step size between forecast cutoffs per series.
    max_windows_per_series : PositiveInt, optional
        Cap the number of forecast windows per series (None for no cap).
    max_series : PositiveInt, optional
        Cap the number of series used for distillation (None for all).
    sample_fraction : PositiveFloat, optional
        Optional fraction of windows to sample per series (0 < f <= 1).
    window_sampling_strategy : str, default "uniform"
        Window sampling strategy ("uniform" or "contiguous").
    min_soft_label_coverage : NonNegativeFloat, default 0.05
        Minimum fraction of eligible rows that must receive soft labels.
    export_soft_labels : bool, default True
        Whether to export teacher predictions as soft labels.
    soft_labels_path : str, optional
        Path to save/load soft labels. If None, uses output_dir/soft_labels.parquet.
    output_dir : str, default "reports/experiments/chronos"
        Base directory for experiment outputs.

    Notes
    -----
    Defaults assume ``prediction_length=15``; adjust rolling parameters when
    changing horizons to keep coverage consistent.

    Examples
    --------
    >>> teacher = ChronosTrainingConfig(preset="chronos2", time_limit=3600)
    >>> student = ChronosTrainingConfig(preset="bolt_small", time_limit=1800)
    >>> distill = ChronosDistillationConfig(
    ...     teacher_config=teacher,
    ...     student_config=student,
    ... )

    """

    teacher_config: ChronosTrainingConfig
    student_config: ChronosTrainingConfig
    enable_distillation: bool = True
    soft_label_temperature: PositiveFloat = 1.0
    distillation_alpha: NonNegativeFloat = 0.5
    label_strategy: str = "blend"
    soft_target_column: str = "soft_target"
    distilled_target_column: str = "distilled_target"
    forecast_step: PositiveInt = 1
    min_history: PositiveInt = 75
    stride: PositiveInt = 15
    max_windows_per_series: PositiveInt | None = None
    max_series: PositiveInt | None = None
    sample_fraction: PositiveFloat | None = None
    window_sampling_strategy: str = "uniform"
    min_soft_label_coverage: NonNegativeFloat = 0.05
    export_soft_labels: bool = True
    soft_labels_path: str | None = None
    output_dir: str = "reports/experiments/chronos"

    def __post_init__(self) -> None:
        """Validate distillation configuration."""
        if self.distillation_alpha > 1.0:
            raise ValueError(
                f"distillation_alpha must be between 0 and 1, got {self.distillation_alpha}"
            )
        if self.label_strategy not in {"teacher_only", "blend"}:
            raise ValueError(
                "label_strategy must be 'teacher_only' or 'blend', "
                f"got {self.label_strategy}"
            )
        if self.sample_fraction is not None and self.sample_fraction > 1.0:
            raise ValueError(
                f"sample_fraction must be <= 1.0, got {self.sample_fraction}"
            )
        if self.window_sampling_strategy not in {"uniform", "contiguous"}:
            raise ValueError(
                "window_sampling_strategy must be 'uniform' or 'contiguous', "
                f"got {self.window_sampling_strategy}"
            )
        if self.min_soft_label_coverage > 1.0:
            raise ValueError(
                f"min_soft_label_coverage must be <= 1.0, got {self.min_soft_label_coverage}"
            )
