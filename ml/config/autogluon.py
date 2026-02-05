"""
AutoGluon TimeSeries configuration for Chronos foundation models.

This module provides configuration classes for training time series models
using AutoGluon's Chronos presets, supporting both teacher training and
student distillation.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ml.common.validation_strategies import require_holdout_strategy
from nautilus_trader.common.config import NautilusConfig
from nautilus_trader.common.config import NonNegativeFloat
from nautilus_trader.common.config import NonNegativeInt
from nautilus_trader.common.config import PositiveFloat
from nautilus_trader.common.config import PositiveInt


if TYPE_CHECKING:
    pass


__all__ = [
    "AutoGluonDataConfig",
    "ChronosBaselineStrategy",
    "ChronosDistillationConfig",
    "ChronosEvaluationConfig",
    "ChronosFineTuneConfig",
    "ChronosOnnxDistillationConfig",
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

ChronosBaselineStrategy = Literal["global_mean", "per_item_mean"]


class ChronosEvaluationConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Configuration for Chronos evaluation helpers.

    Parameters
    ----------
    train_fraction : NonNegativeFloat, default 0.7
        Fraction of timestamps for training split.
    val_fraction : NonNegativeFloat, default 0.15
        Fraction of timestamps for validation split.
    test_fraction : NonNegativeFloat, default 0.15
        Fraction of timestamps for test split.
    min_rows_per_split : PositiveInt, default 50
        Minimum rows required per split.
    min_rows_per_series_split : NonNegativeInt, default 0
        Minimum rows required per series in each split (0 disables filtering).
    item_id_column : str, default "instrument_id"
        Column identifying the series.
    timestamp_column : str, default "ts_event"
        Timestamp column used for sorting/splitting.
    target_column : str, default "forward_return"
        Target column for evaluation.
    baseline_strategy : ChronosBaselineStrategy, default "per_item_mean"
        Baseline strategy for evaluation.
    feature_exclude_columns : tuple[str, ...]
        Columns to drop during sanitization.
    feature_exclude_suffixes : tuple[str, ...]
        Suffixes that trigger feature exclusion.
    drop_non_numeric_features : bool, default True
        Whether to drop non-numeric features.
    drop_constant_features : bool, default True
        Whether to drop constant numeric features.
    filter_market_hours : bool, default True
        Whether to filter to market hours before splitting.
    market_hours_column : str, default "is_market_hours"
        Column used for market hours filtering.
    market_hours_value : bool, default True
        Value indicating a row is within market hours.
    report_dir_name : str, default "reports"
        Subdirectory used for evaluation reports.
    report_filename : str, default "chronos_evaluation.json"
        Filename used for evaluation reports.
    validation_strategy : str, default "time_window"
        Validation strategy ("time_window" or "purged").
    purge_gap : NonNegativeInt, default 0
        Gap between train/validation folds for purged splits.
    embargo_pct : NonNegativeFloat, default 0.0
        Embargo percentage for purged validation splits.
    cv_splits : PositiveInt, default 5
        Number of purged splits to compute when validation_strategy is "purged".

    Notes
    -----
    When ``validation_strategy="purged"``, ``train_fraction`` and ``val_fraction``
    are ignored and the last purged split is used for train/validation. The
    ``test_fraction`` still controls the holdout window.
    """

    train_fraction: NonNegativeFloat = 0.7
    val_fraction: NonNegativeFloat = 0.15
    test_fraction: NonNegativeFloat = 0.15
    min_rows_per_split: PositiveInt = 50
    min_rows_per_series_split: NonNegativeInt = 0
    item_id_column: str = "instrument_id"
    timestamp_column: str = "ts_event"
    target_column: str = "forward_return"
    baseline_strategy: ChronosBaselineStrategy = "per_item_mean"
    feature_exclude_columns: tuple[str, ...] = ("is_weekend", "is_market_hours")
    feature_exclude_suffixes: tuple[str, ...] = ("_vintage_ts",)
    drop_non_numeric_features: bool = True
    drop_constant_features: bool = True
    filter_market_hours: bool = True
    market_hours_column: str = "is_market_hours"
    market_hours_value: bool = True
    report_dir_name: str = "reports"
    report_filename: str = "chronos_evaluation.json"
    validation_strategy: str = "time_window"
    purge_gap: NonNegativeInt = 0
    embargo_pct: NonNegativeFloat = 0.0
    cv_splits: PositiveInt = 5

    def __post_init__(self) -> None:
        """Validate split fractions and column names."""
        if not self.item_id_column:
            raise ValueError("item_id_column must be a non-empty string")
        if not self.timestamp_column:
            raise ValueError("timestamp_column must be a non-empty string")
        if not self.target_column:
            raise ValueError("target_column must be a non-empty string")
        total = float(self.train_fraction) + float(self.val_fraction) + float(self.test_fraction)
        if abs(total - 1.0) > 1e-6:
            raise ValueError("train_fraction + val_fraction + test_fraction must equal 1.0")
        if float(self.train_fraction) <= 0.0:
            raise ValueError("train_fraction must be > 0")
        if float(self.val_fraction) <= 0.0:
            raise ValueError("val_fraction must be > 0")
        if float(self.test_fraction) <= 0.0:
            raise ValueError("test_fraction must be > 0")
        normalized_strategy = require_holdout_strategy(str(self.validation_strategy))
        if normalized_strategy == "purged" and int(self.cv_splits) < 2:
            raise ValueError("cv_splits must be >= 2 for purged validation")
        if float(self.embargo_pct) < 0.0 or float(self.embargo_pct) >= 1.0:
            raise ValueError("embargo_pct must be in [0.0, 1.0)")


class ChronosOnnxDistillationConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Configuration for Chronos -> ONNX distillation.

    Parameters
    ----------
    distillation_config : ChronosDistillationConfig
        Teacher/student distillation settings.
    output_dir : str
        Directory for distillation artifacts.
    feature_registry_dir : str
        Feature registry root directory.
    feature_set_id : str
        Feature set identifier used for alignment.
    registry_dir : str
        Model registry root directory.
    model_id : str
        Identifier for the distilled student model.
    parent_id : str
        Identifier for the parent teacher model.
    train_fraction : NonNegativeFloat, default 0.8
        Fraction of rows used for training when splitting teacher labels.
    output_transform : str, default "identity"
        Transform applied to teacher outputs ("identity" or "sigmoid").
    hard_label_column : str | None, default "y"
        Column containing hard labels for validation metrics.
    require_hard_labels : bool, default False
        Whether to require hard labels for evaluation.
    filtered_features_filename : str, default "features_filtered.npz"
        Filename for filtered feature matrix output.
    teacher_preds_filename : str, default "teacher_preds.npz"
        Filename for teacher prediction output.
    student_output_subdir : str, default "student"
        Subdirectory for student artifacts.
    """

    distillation_config: ChronosDistillationConfig
    output_dir: str
    feature_registry_dir: str
    feature_set_id: str
    registry_dir: str
    model_id: str
    parent_id: str
    train_fraction: NonNegativeFloat = 0.8
    output_transform: str = "identity"
    hard_label_column: str | None = "y"
    require_hard_labels: bool = False
    filtered_features_filename: str = "features_filtered.npz"
    teacher_preds_filename: str = "teacher_preds.npz"
    student_output_subdir: str = "student"
    use_val_for_distill: bool = False
    student_objective: str = "binary"
    kd_lambda: NonNegativeFloat = 0.5
    early_stopping: NonNegativeInt = 50
    opset: PositiveInt = 17

    def __post_init__(self) -> None:
        """Validate distillation configuration values."""
        if not self.output_dir:
            raise ValueError("output_dir must be a non-empty string")
        if not self.feature_registry_dir:
            raise ValueError("feature_registry_dir must be a non-empty string")
        if not self.feature_set_id:
            raise ValueError("feature_set_id must be a non-empty string")
        if not self.registry_dir:
            raise ValueError("registry_dir must be a non-empty string")
        if not self.model_id:
            raise ValueError("model_id must be a non-empty string")
        if not self.parent_id:
            raise ValueError("parent_id must be a non-empty string")
        if float(self.train_fraction) <= 0.0 or float(self.train_fraction) >= 1.0:
            raise ValueError("train_fraction must be between 0 and 1")
        if self.output_transform not in {"identity", "sigmoid"}:
            raise ValueError("output_transform must be 'identity' or 'sigmoid'")


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


class ChronosFineTuneConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Configuration for Chronos fine-tuning search spaces.

    Parameters
    ----------
    learning_rate_bounds : tuple[PositiveFloat, PositiveFloat], default (1e-5, 1e-3)
        Lower/upper bounds for learning rate tuning.
    weight_decay_bounds : tuple[NonNegativeFloat, NonNegativeFloat], default (0.0, 1e-2)
        Lower/upper bounds for weight decay tuning.
    """

    learning_rate_bounds: tuple[PositiveFloat, PositiveFloat] = (1e-5, 1e-3)
    weight_decay_bounds: tuple[NonNegativeFloat, NonNegativeFloat] = (0.0, 1e-2)

    def __post_init__(self) -> None:
        """Validate fine-tune search space bounds."""
        lr_low, lr_high = self.learning_rate_bounds
        if float(lr_low) <= 0.0 or float(lr_high) <= 0.0:
            raise ValueError("learning_rate_bounds must be > 0")
        if float(lr_low) >= float(lr_high):
            raise ValueError("learning_rate_bounds lower must be < upper")

        wd_low, wd_high = self.weight_decay_bounds
        if float(wd_low) < 0.0 or float(wd_high) < 0.0:
            raise ValueError("weight_decay_bounds must be >= 0")
        if float(wd_low) >= float(wd_high):
            raise ValueError("weight_decay_bounds lower must be < upper")


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
    fine_tune : bool, default False
        Whether to enable Chronos fine-tuning (required for tuning search spaces).
    fine_tune_config : ChronosFineTuneConfig, optional
        Search space configuration for fine-tuning.
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
    fine_tune: bool = False
    fine_tune_config: ChronosFineTuneConfig | None = None

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

    def get_fine_tune_config(self) -> ChronosFineTuneConfig | None:
        """
        Return fine-tune config if fine-tuning is enabled.

        Returns
        -------
        ChronosFineTuneConfig | None
            Fine-tune configuration or None when fine-tuning is disabled.
        """
        if not self.fine_tune and self.tuning_config is None:
            return None
        if self.fine_tune_config is not None:
            return self.fine_tune_config
        return ChronosFineTuneConfig()


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
