"""
Typed configuration structures for the ML pipeline orchestrator.

The dataclasses defined here mirror the legacy argparse configuration but expose a
small, frozen public API suitable for reuse by the scheduler, dashboard, and other
cold-path tooling.

Example
-------
>>> from ml.orchestration.config_types import DatasetBuildConfig
>>> cfg = DatasetBuildConfig(data_dir="data/tier1", symbols="SPY", out_dir="ml_out/spy")
>>> cfg.dataset_id
'tft_dataset'

"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING, Final

from ml.config.market_data import MarketDatasetInput
from ml.data import DatasetValidationConfig
from ml.data.vintage import VintagePolicy


if TYPE_CHECKING:  # pragma: no cover - typing only
    from ml.config.scheduler_config import SchedulerConfig

__all__ = [
    "DEFAULT_LOOKBACK_YEARS",
    "DEFAULT_MACRO_SERIES",
    "AutoFillUniverseConfig",
    "DatasetBuildConfig",
    "HPOConfig",
    "IntegrationConfig",
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
    lookback_periods: int = 30
    emit_dataset_events: bool = False
    start_iso: str | None = None
    end_iso: str | None = None
    chunk_days: int = 0
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


@dataclass(slots=True, frozen=True)
class AutoFillUniverseConfig:
    """
    Configuration for catalog auto-fill ingestion.
    """

    enabled: bool = False
    dataset_id: str = "EQUS.MINI"
    include_bars: bool = True
    include_tbbo: bool = True
    include_trades: bool = True
    include_l2: bool = False
    include_l3: bool = False
    l2_dataset_id: str = "DBEQ.BASIC"
    l2_schema: str = "mbp-10"
    l2_days: int | None = None
    l2_progress_file: str | None = None
    disable_dataset_l2_ingest: bool = True
    instrument_ids: tuple[str, ...] | None = None
    l3_dataset_id: str | None = None
    l3_schema: str | None = None
    l3_days: int | None = None


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
    """

    enabled: bool = True
    model_id: str = "teacher_model"
    feature_registry_dir: str | None = None
    feature_set_id: str | None = None
    max_epochs: int = 5


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
    start_metrics_server: bool = False
    metrics_port: int | None = None


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
