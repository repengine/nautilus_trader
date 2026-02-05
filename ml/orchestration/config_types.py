"""
Typed configuration structures for the ML pipeline orchestrator.

The dataclasses defined here mirror the legacy argparse configuration but expose a
small, frozen public API suitable for reuse by the scheduler, dashboard, and other
cold-path tooling.

Example
-------
>>> from ml.orchestration.config_types import DatasetBuildConfig
>>> cfg = DatasetBuildConfig(
...     data_dir="data/tier1",
...     symbols="SPY",
...     out_dir="ml_out/spy",
...     target_semantics={
...         "version": "v1",
...         "horizons": [{"minutes": 15}],
...         "binary": {"enabled": True, "threshold_bps": 10.0, "return_basis": "raw"},
...     },
... )
>>> cfg.dataset_id
'tft_dataset'

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING, Final

from ml.common.validation_strategies import DEFAULT_HOLDOUT_STRATEGY
from ml.common.validation_strategies import require_holdout_strategy
from ml.config import WatermarkWindowConfig
from ml.config import earnings_window_defaults
from ml.config import macro_window_defaults
from ml.config.feature_cache import FeatureCachePolicy
from ml.config.feature_cache import normalize_feature_cache_policy
from ml.config.market_data import MarketDatasetInput
from ml.data import DatasetValidationConfig
from ml.data.vintage import VintagePolicy
from ml.registry.dataclasses import DatasetType


if TYPE_CHECKING:  # pragma: no cover - typing only
    from ml.config.scheduler_config import SchedulerConfig

__all__ = [
    "DEFAULT_LOOKBACK_YEARS",
    "DEFAULT_MACRO_SERIES",
    "AutoFillUniverseConfig",
    "DatasetBuildConfig",
    "EarningsCoordinatorConfig",
    "HPOConfig",
    "IntegrationConfig",
    "MacroIngestionConfig",
    "OrchestratorConfig",
    "PreIngestionOptions",
    "PromotionsConfig",
    "StudentDistillConfig",
    "TeacherTrainConfig",
]


@dataclass(slots=True, frozen=True)
class DatasetBuildConfig:
    """
    Configuration for dataset construction.
    """

    data_dir: str
    symbols: str
    out_dir: str
    dataset_id: str = "tft_dataset"
    market_dataset_id: str | None = None
    market_inputs: tuple[MarketDatasetInput, ...] | None = None
    instrument_ids: tuple[str, ...] | None = None
    include_macro: bool = False
    macro_lag_days: int = 1
    include_micro: bool = False
    include_l2: bool = False
    micro_cache_policy: FeatureCachePolicy = "cache_first"
    l2_cache_policy: FeatureCachePolicy = "cache_first"
    include_events: bool = False
    include_calendar: bool = False
    include_earnings: bool = False
    earnings_lag_days: int = 1
    include_macro_deltas: bool = False
    include_calendar_lags: bool = False
    include_clustering_tags: bool = False
    include_context_features: bool = False
    fred_vintage_dir: str | None = None
    events_dir: str | None = None
    student_mode: bool = False
    horizon_minutes: int = 15
    threshold: float = 0.001
    target_semantics: dict[str, object] | None = None
    lookback_periods: int = 30
    emit_dataset_events: bool = False
    start_iso: str | None = None
    end_iso: str | None = None
    chunk_days: int = 0
    write_csv: bool | None = None
    csv_max_rows: int = 1_000_000
    csv_sample_rows: int = 0
    register_features: bool = False
    feature_registry_dir: str | None = None
    feature_role: str = "teacher"
    auto_refresh_macro: bool = True
    macro_staleness_hours: int = 24
    macro_series_ids: tuple[str, ...] | None = None
    macro_fred_path: str | None = None
    validation: DatasetValidationConfig | None = None
    vintage_policy: VintagePolicy = VintagePolicy.REAL_TIME
    vintage_as_of: str | None = None
    include_macro_revisions: bool = False
    macro_revision_mode: str = "core"
    macro_revision_windows: tuple[int, ...] | None = None
    convert_vintage_to_age: bool = False

    def __post_init__(self) -> None:
        """
        Normalize cache policy tokens.
        """
        object.__setattr__(
            self,
            "micro_cache_policy",
            normalize_feature_cache_policy(
                self.micro_cache_policy,
                label="micro_cache_policy",
            ),
        )
        object.__setattr__(
            self,
            "l2_cache_policy",
            normalize_feature_cache_policy(
                self.l2_cache_policy,
                label="l2_cache_policy",
            ),
        )


@dataclass(slots=True, frozen=True)
class MacroIngestionConfig:
    """
    Configuration for FRED/ALFRED macro data ingestion.

    Used by IngestionCoordinator to determine where to store macro data
    and how fresh it needs to be.

    Attributes
    ----------
    fred_path : str
        Path to the FRED indicators parquet file.
    vintage_dir : str | None
        Directory for ALFRED vintage releases. None disables ALFRED refresh.
    max_staleness_hours : int
        Maximum age in hours before data is considered stale and refreshed.
    series_ids : tuple[str, ...] | None
        Optional subset of series IDs to refresh. None refreshes all configured.
    watermark_config : WatermarkWindowConfig
        Watermark window configuration for SQL ingestion filters.

    Example
    -------
    >>> config = MacroIngestionConfig(
    ...     fred_path="data/features/macro/fred_indicators.parquet",
    ...     vintage_dir="data/features/macro/fred/vintages",
    ...     max_staleness_hours=24,
    ... )

    """

    fred_path: str = "data/features/macro/fred_indicators_ml_format.parquet"
    vintage_dir: str | None = "data/features/macro/fred/vintages"
    max_staleness_hours: int = 24
    series_ids: tuple[str, ...] | None = None
    watermark_config: WatermarkWindowConfig = field(default_factory=macro_window_defaults)

    @classmethod
    def from_dataset_config(cls, cfg: DatasetBuildConfig) -> MacroIngestionConfig:
        """
        Create MacroIngestionConfig from a DatasetBuildConfig.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset build configuration containing macro settings.

        Returns
        -------
        MacroIngestionConfig
            Macro ingestion configuration extracted from dataset config.

        """
        return cls(
            fred_path=cfg.macro_fred_path or "data/features/macro/fred_indicators_ml_format.parquet",
            vintage_dir=cfg.fred_vintage_dir,
            max_staleness_hours=cfg.macro_staleness_hours,
            series_ids=cfg.macro_series_ids,
        )


@dataclass(slots=True, frozen=True)
class EarningsCoordinatorConfig:
    """
    Configuration for earnings data ingestion via IngestionCoordinator.

    Used by IngestionCoordinator to control earnings ingestion behavior.
    The actual EarningsIngestionConfig is constructed internally using these
    settings combined with the symbol provided at call time.

    Attributes
    ----------
    edgar_quarters : int
        Number of quarters to fetch from EDGAR (10-Q/10-K filings).
    enable_yahoo : bool
        Whether to fetch Yahoo Finance consensus estimates.
    edgar_rate_limit : float
        Delay in seconds between EDGAR API calls.
    yahoo_rate_limit : float
        Delay in seconds between Yahoo Finance API calls.
    sec_identity : str | None
        Optional SEC identity string for EDGAR API.
    skip_tickers : tuple[str, ...] | None
        Tickers to skip (e.g., ETFs without earnings).
    watermark_config : WatermarkWindowConfig
        Watermark window configuration for incremental ingestion.

    Example
    -------
    >>> config = EarningsCoordinatorConfig(
    ...     edgar_quarters=8,
    ...     enable_yahoo=True,
    ... )

    """

    edgar_quarters: int = 8
    enable_yahoo: bool = True
    edgar_rate_limit: float = 1.0
    yahoo_rate_limit: float = 0.5
    sec_identity: str | None = None
    skip_tickers: tuple[str, ...] | None = None
    watermark_config: WatermarkWindowConfig = field(default_factory=earnings_window_defaults)


@dataclass(slots=True, frozen=True)
class AutoFillUniverseConfig:
    """
    Configuration for catalog auto-fill ingestion.
    """

    enabled: bool = False
    dataset_id: str = "EQUS.MINI"
    include_bars: bool = True
    # L1 quotes are canonicalized as the "quotes" schema (provider schema is tbbo).
    include_tbbo: bool = True
    include_trades: bool = True
    include_l2: bool = False
    include_l3: bool = False
    l2_dataset_id: str = "DBEQ.BASIC"
    l2_schema: str = "mbp-1"
    l2_days: int | None = None
    l2_progress_file: str | None = None
    disable_dataset_l2_ingest: bool = True
    instrument_ids: tuple[str, ...] | None = None
    l3_dataset_id: str | None = None
    l3_schema: str | None = None
    l3_days: int | None = None

    def __post_init__(self) -> None:
        """
        Validate schema overrides for auto-fill ingestion.
        """
        from ml.schema import schema_spec_for

        if self.include_l2:
            schema_spec_for(self.l2_schema)
        if self.include_l3 and self.l3_schema is not None:
            schema_spec_for(self.l3_schema)


@dataclass(slots=True, frozen=True)
class HPOConfig:
    """
    Hyper-parameter optimisation configuration.
    """

    enabled: bool = False
    epochs: int = 2
    batch_size: int = 32
    tail_rows: int = 5000
    limit_groups: int = 50
    workers: int = 2
    backend: str = "optuna"
    metric: str = "prx"
    direction: str | None = None
    optuna_trials: int = 20
    optuna_timeout: int | None = None
    loss: str = "bce"
    pos_weight: str = "auto"


@dataclass(slots=True, frozen=True)
class TeacherTrainConfig:
    """
    Teacher training configuration.

    Attributes
    ----------
    enabled
        Toggle teacher training on/off.
    model_id
        Model identifier used for registry/export.
    feature_registry_dir
        Optional feature registry directory override.
    feature_set_id
        Optional feature set identifier override.
    max_epochs
        Maximum training epochs for TFT teacher.
    batch_size
        Training batch size for the teacher DataLoader.
    dataloader_workers
        Number of DataLoader worker processes.
    accelerator
        Lightning accelerator selection ("auto", "cpu", "gpu").
    devices
        Number of devices used by Lightning.
    precision
        Training precision (e.g. "32", "16-mixed", "bf16").
    max_encoder_length
        Encoder lookback window length for TFT.
    max_prediction_length
        Prediction horizon length for TFT.
    hidden_size
        Hidden size for TFT encoder/decoder.
    lstm_layers
        Number of LSTM layers in TFT.
    attention_head_size
        Attention head size for TFT.
    dropout
        Dropout probability used in TFT.
    learning_rate
        Optimizer learning rate.
    loss
        Loss function ("poisson" or "bce").
    pos_weight
        Optional class weight for BCE loss ("auto" or float).
    seed
        Optional RNG seed for deterministic training.
    tail_rows
        Per-group tail cap (0 disables).
    limit_groups
        Top-N group cap by row count (0 disables).
    val_days
        Validation window size in days (0 disables time-window validation).
    validation_strategy
        Validation strategy ("time_window" or "purged").
    embargo_hours
        Embargo window in hours for purged splits.
    embargo_pct
        Optional embargo percentage override for purged splits.
    purge_gap
        Purge gap (rows) between train/validation folds.
    cv_splits
        Number of cross-validation splits.
    test_fraction
        Hold-out fraction used for train/validation split.
    target_col
        Target column name in the training dataset (must be declared in target_semantics).
    time_index_col
        Time index column name.
    timestamp_col
        Timestamp column name (for purged splits).
    group_id_col
        Group identifier column name (instrument identifier).
    static_categoricals
        Static categorical feature names.
    static_reals
        Static real-valued feature names.
    known_future_reals
        Known-future real feature names.
    save_interpretability
        Persist interpretability artifacts if available.
    export_torchscript
        Export TorchScript artifact for the teacher.
    export_safetensors
        Export safetensors weights for the teacher.
    pretrained_state_path
        Optional pretrained state dict path for warm start.
    register_teacher
        Register the teacher model artifact.
    decision_policy
        Optional decision-policy adapter reference.
    decision_config
        Optional decision-policy configuration payload.
    prefer_parquet
        Prefer parquet datasets when available.
    """

    enabled: bool = True
    model_id: str = "teacher_model"
    feature_registry_dir: str | None = None
    feature_set_id: str | None = None
    max_epochs: int = 5
    batch_size: int = 64
    dataloader_workers: int = 0
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "32"
    max_encoder_length: int = 30
    max_prediction_length: int = 1
    hidden_size: int = 16
    lstm_layers: int = 1
    attention_head_size: int = 2
    dropout: float = 0.1
    learning_rate: float = 3e-4
    loss: str = "poisson"
    pos_weight: str | float | None = None
    seed: int | None = None
    tail_rows: int = 0
    limit_groups: int = 0
    val_days: int = 0
    validation_strategy: str = DEFAULT_HOLDOUT_STRATEGY
    embargo_hours: float = 24.0
    embargo_pct: float | None = None
    purge_gap: int = 0
    cv_splits: int = 5
    test_fraction: float = 0.2
    target_col: str = "y"
    time_index_col: str = "time_index"
    timestamp_col: str = "timestamp"
    group_id_col: str = "instrument_id"
    static_categoricals: tuple[str, ...] = ()
    static_reals: tuple[str, ...] = ()
    known_future_reals: tuple[str, ...] = ()
    save_interpretability: bool = False
    export_torchscript: bool = False
    export_safetensors: bool = False
    pretrained_state_path: str | None = None
    register_teacher: bool = False
    decision_policy: str | None = None
    decision_config: Mapping[str, object] | str | None = None
    prefer_parquet: bool = True

    def __post_init__(self) -> None:
        """
        Validate training configuration ranges.
        """
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.dataloader_workers < 0:
            raise ValueError("dataloader_workers must be >= 0")
        if self.devices < 1:
            raise ValueError("devices must be >= 1")
        if self.max_encoder_length < 1:
            raise ValueError("max_encoder_length must be >= 1")
        if self.max_prediction_length < 1:
            raise ValueError("max_prediction_length must be >= 1")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        if self.lstm_layers < 1:
            raise ValueError("lstm_layers must be >= 1")
        if self.attention_head_size < 1:
            raise ValueError("attention_head_size must be >= 1")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0.0")
        if self.tail_rows < 0:
            raise ValueError("tail_rows must be >= 0")
        if self.limit_groups < 0:
            raise ValueError("limit_groups must be >= 0")
        if self.val_days < 0:
            raise ValueError("val_days must be >= 0")
        normalized_strategy = require_holdout_strategy(str(self.validation_strategy))
        object.__setattr__(self, "validation_strategy", normalized_strategy)
        if normalized_strategy == "time_window" and self.val_days <= 0:
            raise ValueError("validation_strategy=time_window requires val_days > 0")
        if self.embargo_hours < 0.0:
            raise ValueError("embargo_hours must be >= 0.0")
        if self.embargo_pct is not None and not 0.0 <= float(self.embargo_pct) < 1.0:
            raise ValueError("embargo_pct must be in [0.0, 1.0) when provided")
        if self.purge_gap < 0:
            raise ValueError("purge_gap must be >= 0")
        if self.cv_splits < 0:
            raise ValueError("cv_splits must be >= 0")
        if normalized_strategy == "purged" and self.cv_splits < 2:
            raise ValueError("validation_strategy=purged requires cv_splits >= 2")
        if self.test_fraction < 0.0 or self.test_fraction >= 1.0:
            raise ValueError("test_fraction must be in [0.0, 1.0)")
        if not str(self.target_col).strip():
            raise ValueError("target_col must be non-empty")
        if not str(self.time_index_col).strip():
            raise ValueError("time_index_col must be non-empty")
        if not str(self.timestamp_col).strip():
            raise ValueError("timestamp_col must be non-empty")
        if not str(self.group_id_col).strip():
            raise ValueError("group_id_col must be non-empty")
        if not str(self.accelerator).strip():
            raise ValueError("accelerator must be non-empty")
        if not str(self.precision).strip():
            raise ValueError("precision must be non-empty")
        if not str(self.loss).strip():
            raise ValueError("loss must be non-empty")


@dataclass(slots=True, frozen=True)
class StudentDistillConfig:
    """
    Student distillation configuration.
    """

    enabled: bool = False
    model_id: str = "student_model"
    parent_model_id: str | None = None
    model_registry_dir: str | None = None
    feature_registry_dir: str | None = None
    feature_set_id: str | None = None
    objective: str = "logit_mse"
    kd_lambda: float = 0.5
    early_stopping: int = 200
    opset: int | None = None
    use_val_for_distill: bool = False


@dataclass(slots=True, frozen=True)
class IntegrationConfig:
    """
    Integration manager attachment configuration.
    """

    enabled: bool = False
    db_connection: str | None = None
    auto_start_postgres: bool = False
    auto_migrate: bool = False
    ensure_healthy: bool = True
    strict_protocol_validation: bool | None = None
    run_validators: bool = True


@dataclass(slots=True, frozen=True)
class PreIngestionOptions:
    """
    Options for the pre-ingestion scheduler stage.
    """

    use_orchestrator: bool = True
    dual_write: bool = True
    dual_write_bars: bool = True
    dual_write_tbbo: bool = True
    dual_write_trades: bool = True
    dual_write_mbp: bool = True
    start_metrics_server: bool = False
    metrics_port: int | None = None

    def dual_write_dataset_types(self) -> dict[DatasetType, bool]:
        """
        Return dataset-type toggles for dual-write mirroring.
        """
        return {
            DatasetType.BARS: self.dual_write_bars,
            DatasetType.TBBO: self.dual_write_tbbo,
            DatasetType.TRADES: self.dual_write_trades,
            DatasetType.MBP1: self.dual_write_mbp,
            DatasetType.MBP10: self.dual_write_mbp,
            DatasetType.MBO: self.dual_write_mbp,
        }


@dataclass(slots=True, frozen=True)
class PromotionsConfig:
    """
    Promotion and feature refresh configuration.
    """

    auto_register_model: bool = False
    gates_json: str | None = None
    auto_promote: bool = False
    deploy_target: str | None = None
    auto_register_features: bool = False
    feature_metrics_json: str | None = None
    refresh_features: bool = False


@dataclass(slots=True, frozen=True)
class OrchestratorConfig:
    """
    Composite orchestrator configuration consumed by the legacy runner.
    """

    dataset: DatasetBuildConfig
    hpo: HPOConfig
    teacher: TeacherTrainConfig
    student: StudentDistillConfig = field(default_factory=StudentDistillConfig)
    promotions: PromotionsConfig | None = None
    pre_ingestion: SchedulerConfig | None = None
    pre_ingestion_options: PreIngestionOptions | None = None
    auto_fill: AutoFillUniverseConfig | None = None
    integration: IntegrationConfig | None = None


DEFAULT_LOOKBACK_YEARS: Final[int] = 7
DEFAULT_MACRO_SERIES: Final[tuple[str, ...]] = (
    "CPIAUCSL",
    "PCEPI",
    "PAYEMS",
    "UNRATE",
    "GDP",
    "FEDFUNDS",
)
