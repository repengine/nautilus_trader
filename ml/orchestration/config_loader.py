"""
Configuration loader for stage-aware pipeline orchestration.

The refactor introduces a typed configuration surface so orchestrator runs can be
controlled via TOML/JSON documents instead of lengthy CLI invocations.  This loader
parses the configuration file, applies optional environment overrides, and returns a
structured :class:`OrchestratorRunConfig` instance that downstream orchestration code
can consume.

Example
-------
>>> from pathlib import Path
>>> from ml.orchestration.config_loader import load_orchestrator_run_config, Stage
>>> run_cfg = load_orchestrator_run_config(Path("tests/data/orchestrator/minimal.json"))
>>> run_cfg.stage is Stage.FULL
True

"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from enum import StrEnum
from pathlib import Path
from typing import Any


try:  # Python 3.11 includes tomllib
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - alt interpreter fallback
    tomllib = None  # type: ignore[assignment]

from ml.config.market_data import MarketDatasetInput
from ml.config.market_data import coerce_storage_kind
from ml.orchestration.config_types import AutoFillUniverseConfig
from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.config_types import HPOConfig
from ml.orchestration.config_types import IntegrationConfig
from ml.orchestration.config_types import OrchestratorConfig
from ml.orchestration.config_types import PromotionsConfig
from ml.orchestration.config_types import StudentDistillConfig
from ml.orchestration.config_types import TeacherTrainConfig


__all__ = [
    "IngestionStageConfig",
    "OrchestratorRunConfig",
    "Stage",
    "TrainingStageConfig",
    "load_orchestrator_config",
    "load_orchestrator_run_config",
    "to_pipeline_args",
]


class Stage(StrEnum):
    """
    Valid pipeline execution stages.
    """

    INGEST = "ingest"
    DATASET = "dataset"
    TRAIN = "train"
    FULL = "full"


@dataclass(slots=True, frozen=True)
class IngestionStageConfig:
    """
    Configuration for the ingestion stage.

    Attributes
    ----------
    enabled
        Whether ingestion should run before other stages.
    dataset_id
        Market dataset identifier to ingest (e.g. ``"EQUS.MINI"``).
    schema
        Schema to request from the ingestion orchestrator (``"bars"``, ``"tbbo"`` or
        ``"trades"``).
    instruments
        Nautilus instrument identifiers to ingest.  Defaults to ``("SPY.NYSE",)`` to
        mirror the legacy CLI default.
    lookback_days
        Historical lookback window per instrument.
    coverage_mode
        Coverage provider mode (``"catalog"`` or ``"sql"``).
    write_mode
        Ingestion write mode token string (e.g. ``"sql+parquet"``).
    catalog_path
        Optional parquet catalog location.
    symbols
        Optional explicit symbol universe used for resolving market bindings.
    instrument_ids
        Explicit instrument identifiers used when binding descriptors supply
        templates.  Defaults to ``None`` which falls back to ``instruments``.
    market_dataset_id
        Legacy dataset identifier used when no descriptor bindings are resolved.
    market_inputs
        Optional list of :class:`~ml.config.market_data.MarketDatasetInput` entries
        used to drive descriptor-based binding resolution.

    """

    enabled: bool = False
    dataset_id: str = "EQUS.MINI"
    schema: str = "bars"
    instruments: tuple[str, ...] = ("SPY.NYSE",)
    lookback_days: int = 7
    coverage_mode: str = "catalog"
    write_mode: str = "parquet"
    catalog_path: str | None = None
    symbols: tuple[str, ...] | None = None
    instrument_ids: tuple[str, ...] | None = None
    market_dataset_id: str | None = None
    market_inputs: tuple[MarketDatasetInput, ...] | None = None


@dataclass(slots=True, frozen=True)
class TrainingStageConfig:
    """
    Training stage configuration wrapper.
    """

    teacher: TeacherTrainConfig = field(default_factory=TeacherTrainConfig)
    student: StudentDistillConfig = field(default_factory=StudentDistillConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)


@dataclass(slots=True, frozen=True)
class OrchestratorRunConfig:
    """
    Top-level configuration structure returned by the loader.
    """

    stage: Stage = Stage.FULL
    dataset: DatasetBuildConfig | None = None
    ingestion: IngestionStageConfig | None = None
    training: TrainingStageConfig = field(default_factory=TrainingStageConfig)
    promotions: PromotionsConfig | None = None
    auto_fill: AutoFillUniverseConfig | None = None
    integration: IntegrationConfig | None = None

    def compose_orchestrator_config(self) -> OrchestratorConfig:
        """
        Collapse the run config into the legacy :class:`OrchestratorConfig`.

        Returns
        -------
        OrchestratorConfig
            Composite configuration compatible with the existing orchestrator
            implementation.

        """
        if self.dataset is None:
            raise ValueError("Dataset configuration is required to compose orchestrator config")

        return OrchestratorConfig(
            dataset=self.dataset,
            hpo=self.training.hpo,
            teacher=self.training.teacher,
            student=self.training.student,
            promotions=self.promotions,
            pre_ingestion=None,
            pre_ingestion_options=None,
            auto_fill=self.auto_fill,
            integration=self.integration,
        )


def load_orchestrator_run_config(
    path: str | Path,
    *,
    env: Mapping[str, str] | None = None,
) -> OrchestratorRunConfig:
    """
    Load stage-aware orchestrator configuration from ``path``.

    Parameters
    ----------
    path
        Path to a JSON or TOML document.
    env
        Optional environment mapping.  Keys prefixed with ``ML_ORCH__`` override nested
        configuration keys.  For example ``ML_ORCH__DATASET__SYMBOLS="SPY,QQQ"`` updates
        ``dataset.symbols`` before conversion to dataclasses.

    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")

    payload = _load_raw_payload(resolved)
    env_map = env if env is not None else os.environ
    overrides = _collect_overrides(env_map)
    if overrides:
        _apply_overrides(payload, overrides)

    stage = _parse_stage(payload.get("stage"))
    dataset_cfg = (
        _coerce_dataset(payload.get("dataset")) if payload.get("dataset") is not None else None
    )
    ingestion_cfg = _coerce_ingestion(payload.get("ingestion"))
    teacher_cfg = _coerce_dataclass(payload.get("teacher"), TeacherTrainConfig)
    student_cfg = _coerce_dataclass(payload.get("student"), StudentDistillConfig)
    hpo_cfg = _coerce_dataclass(payload.get("hpo"), HPOConfig)
    training_cfg = _coerce_training(payload.get("training"))
    if training_cfg is None:
        training_cfg = TrainingStageConfig(
            teacher=teacher_cfg or TeacherTrainConfig(),
            student=student_cfg or StudentDistillConfig(),
            hpo=hpo_cfg or HPOConfig(),
        )
    else:
        training_cfg = TrainingStageConfig(
            teacher=teacher_cfg or training_cfg.teacher,
            student=student_cfg or training_cfg.student,
            hpo=hpo_cfg or training_cfg.hpo,
        )
    promotions_cfg = _coerce_dataclass(payload.get("promotions"), PromotionsConfig)
    auto_fill_cfg = _coerce_dataclass(payload.get("auto_fill"), AutoFillUniverseConfig)
    integration_cfg = _coerce_dataclass(payload.get("integration"), IntegrationConfig)

    return OrchestratorRunConfig(
        stage=stage,
        dataset=dataset_cfg,
        ingestion=ingestion_cfg,
        training=training_cfg,
        promotions=promotions_cfg,
        auto_fill=auto_fill_cfg,
        integration=integration_cfg,
    )


def _load_raw_payload(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".toml", ".tml"}:
        if tomllib is None:  # pragma: no cover - defensive
            raise ValueError("TOML support unavailable; install tomllib for this interpreter")
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    else:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    if not isinstance(data, Mapping):
        raise ValueError("Configuration payload must be a mapping")
    return dict(data)


def _parse_stage(raw: Any) -> Stage:
    if raw is None:
        return Stage.FULL
    if isinstance(raw, Stage):
        return raw
    try:
        return Stage(str(raw).lower())
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unsupported stage value: {raw!r}") from exc


def _collect_overrides(env: Mapping[str, str] | None) -> dict[tuple[str, ...], str]:
    if not env:
        return {}
    overrides: dict[tuple[str, ...], str] = {}
    for key, value in env.items():
        if not key.startswith("ML_ORCH__"):
            continue
        segments = tuple(segment for segment in key.split("__") if segment)[1:]
        if not segments:
            continue
        overrides[tuple(segment.lower() for segment in segments)] = value
    return overrides


def _apply_overrides(target: dict[str, Any], overrides: Mapping[tuple[str, ...], str]) -> None:
    for path_segments, raw_value in overrides.items():
        cursor: dict[str, Any] = target
        for segment in path_segments[:-1]:
            nested = cursor.setdefault(segment, {})
            if not isinstance(nested, dict):
                raise ValueError(f"Cannot override segment '{segment}' in non-mapping payload")
            cursor = nested
        last = path_segments[-1]
        cursor[last] = _parse_override_value(raw_value)


def _parse_override_value(value: str) -> Any:
    trimmed = value.strip()
    if not trimmed:
        return ""
    try:
        return json.loads(trimmed)
    except json.JSONDecodeError:
        lowered = trimmed.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        if lowered == "null":
            return None
        return trimmed


def _coerce_training(payload: Any) -> TrainingStageConfig | None:
    if payload is None:
        return None
    if isinstance(payload, TrainingStageConfig):
        return payload
    if not isinstance(payload, Mapping):
        raise ValueError("training section must be a mapping")
    teacher = _coerce_dataclass(payload.get("teacher"), TeacherTrainConfig) or TeacherTrainConfig()
    student = (
        _coerce_dataclass(payload.get("student"), StudentDistillConfig) or StudentDistillConfig()
    )
    hpo = _coerce_dataclass(payload.get("hpo"), HPOConfig) or HPOConfig()
    return TrainingStageConfig(teacher=teacher, student=student, hpo=hpo)


def _coerce_dataset(payload: Any) -> DatasetBuildConfig:
    if isinstance(payload, DatasetBuildConfig):
        return payload
    if not isinstance(payload, Mapping):
        raise ValueError("dataset section must be a mapping")
    kwargs = dict(payload)
    if "market_inputs" in kwargs:
        kwargs["market_inputs"] = _coerce_market_inputs(kwargs["market_inputs"])
    if "instrument_ids" in kwargs:
        kwargs["instrument_ids"] = tuple(
            str(item) for item in _ensure_sequence(kwargs["instrument_ids"])
        )
    if "macro_series_ids" in kwargs and kwargs["macro_series_ids"] is not None:
        kwargs["macro_series_ids"] = tuple(
            str(item) for item in _ensure_sequence(kwargs["macro_series_ids"])
        )
    return DatasetBuildConfig(**kwargs)


def _coerce_ingestion(payload: Any) -> IngestionStageConfig | None:
    if payload is None:
        return None
    if isinstance(payload, IngestionStageConfig):
        return payload
    if not isinstance(payload, Mapping):
        raise ValueError("ingestion section must be a mapping")
    kwargs = dict(payload)
    if "instruments" in kwargs and kwargs["instruments"] is not None:
        kwargs["instruments"] = tuple(
            str(item).strip()
            for item in _ensure_sequence(kwargs["instruments"])
            if str(item).strip()
        )
    if "instrument_ids" in kwargs and kwargs["instrument_ids"] is not None:
        kwargs["instrument_ids"] = tuple(
            str(item).strip()
            for item in _ensure_sequence(kwargs["instrument_ids"])
            if str(item).strip()
        )
    if "symbols" in kwargs and kwargs["symbols"] is not None:
        kwargs["symbols"] = tuple(
            str(item).strip().upper()
            for item in _ensure_sequence(kwargs["symbols"])
            if str(item).strip()
        )
    if "market_inputs" in kwargs:
        kwargs["market_inputs"] = _coerce_market_inputs(kwargs["market_inputs"])
        if not kwargs["market_inputs"]:
            kwargs["market_inputs"] = None
    if "market_dataset_id" in kwargs and kwargs["market_dataset_id"] is not None:
        kwargs["market_dataset_id"] = str(kwargs["market_dataset_id"])
    if "dataset_id" in kwargs and kwargs["dataset_id"] is not None:
        kwargs["dataset_id"] = str(kwargs["dataset_id"])
    if "schema" in kwargs and kwargs["schema"] is not None:
        kwargs["schema"] = str(kwargs["schema"])
    if "catalog_path" in kwargs and kwargs["catalog_path"] is not None:
        kwargs["catalog_path"] = str(kwargs["catalog_path"])
    return IngestionStageConfig(**kwargs)


def _coerce_dataclass(payload: Any, cls: type[Any]) -> Any:
    if payload is None:
        return None
    if isinstance(payload, cls):
        return payload
    if not isinstance(payload, Mapping):
        raise ValueError(f"{cls.__name__} section must be a mapping")
    kwargs = dict(payload)
    for item in fields(cls):
        if item.name not in kwargs:
            continue
        value = kwargs[item.name]
        if value is None:
            continue
        if _is_tuple_field(item.type):
            kwargs[item.name] = tuple(_ensure_sequence(value))
    return cls(**kwargs)


def _coerce_market_inputs(payload: Any) -> tuple[MarketDatasetInput, ...]:
    if payload is None:
        return ()
    items = _ensure_sequence(payload)
    inputs: list[MarketDatasetInput] = []
    for item in items:
        if isinstance(item, MarketDatasetInput):
            inputs.append(item)
            continue
        if not isinstance(item, Mapping):
            raise ValueError("market_inputs entries must be mappings")
        storage_kind = item.get("storage_kind_override") or item.get("storage_kind")
        storage_kind_parsed = coerce_storage_kind(storage_kind)
        inputs.append(
            MarketDatasetInput(
                descriptor_id=item.get("descriptor_id"),
                dataset_id=item.get("dataset_id"),
                symbols=tuple(item.get("symbols", ()) or ()),
                schema_override=item.get("schema_override") or item.get("schema"),
                storage_kind_override=storage_kind_parsed,
                start=item.get("start"),
                end=item.get("end"),
            ),
        )
    return tuple(inputs)


def _ensure_sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, list | tuple):
        return tuple(value)
    if value is None:
        return ()
    return (value,)


def _is_tuple_field(tp: Any) -> bool:
    origin = getattr(tp, "__origin__", None)
    return origin is tuple or tp is tuple


def load_orchestrator_config(
    path: str | None,
    *,
    env: Mapping[str, str] | None = None,
) -> OrchestratorConfig:
    """
    Compatibility wrapper returning the legacy orchestrator config.
    """
    if path is None:
        raise ValueError("Config path is required for orchestrator runs")
    run_cfg = load_orchestrator_run_config(path, env=env)
    return run_cfg.compose_orchestrator_config()


def to_pipeline_args(
    cfg: OrchestratorConfig,
    *,
    ingestion: IngestionStageConfig | None = None,
) -> list[str]:
    """
    Translate an :class:`OrchestratorConfig` into CLI arguments.
    """
    dataset = cfg.dataset
    args: list[str] = [
        "--data_dir",
        dataset.data_dir,
        "--symbols",
        dataset.symbols,
        "--out_dir",
        dataset.out_dir,
        "--horizon_minutes",
        str(dataset.horizon_minutes),
        "--threshold",
        str(dataset.threshold),
        "--lookback_periods",
        str(dataset.lookback_periods),
    ]

    if dataset.include_macro:
        args.append("--include_macro")
        args += ["--macro_lag_days", str(dataset.macro_lag_days)]
    if dataset.include_micro:
        args.append("--include_micro")
    if dataset.include_l2:
        args.append("--include_l2")
    if dataset.include_events:
        args.append("--include_events")
    if dataset.include_calendar:
        args.append("--include_calendar")
    if dataset.instrument_ids:
        args += ["--instrument_ids", ",".join(dataset.instrument_ids)]
    if dataset.market_dataset_id:
        args += ["--market_dataset_id", dataset.market_dataset_id]
    if dataset.market_inputs:
        payload = _market_inputs_to_payload(dataset.market_inputs)
        args += ["--market_inputs_json", json.dumps(payload)]
    if dataset.student_mode:
        args.append("--student_mode")
    if dataset.emit_dataset_events:
        args.append("--emit_dataset_events")
    if dataset.start_iso:
        args += ["--start_iso", dataset.start_iso]
    if dataset.end_iso:
        args += ["--end_iso", dataset.end_iso]
    if dataset.chunk_days:
        args += ["--chunk_days", str(dataset.chunk_days)]
    if dataset.fred_vintage_dir:
        args += ["--fred_vintage_dir", dataset.fred_vintage_dir]
    if dataset.events_dir:
        args += ["--events_dir", dataset.events_dir]
    if dataset.feature_registry_dir:
        args += ["--feature_registry_dir", dataset.feature_registry_dir]
    if dataset.register_features:
        args.append("--dataset_register_features")
    if not dataset.auto_refresh_macro:
        args.append("--skip_macro_refresh")
    if dataset.macro_staleness_hours != 24:
        args += ["--macro_freshness_hours", str(dataset.macro_staleness_hours)]
    if dataset.macro_series_ids:
        args += ["--macro_series_ids", ",".join(dataset.macro_series_ids)]
    if dataset.macro_fred_path:
        args += ["--macro_fred_path", dataset.macro_fred_path]
    if dataset.validation is not None:
        args += ["--validation_min_rows", str(dataset.validation.min_rows)]
        if dataset.validation.min_positive_rate is not None:
            args += ["--validation_min_positive_rate", str(dataset.validation.min_positive_rate)]
        if dataset.validation.max_positive_rate is not None:
            args += ["--validation_max_positive_rate", str(dataset.validation.max_positive_rate)]
        if dataset.validation.min_feature_coverage is not None:
            args += [
                "--validation_min_feature_coverage",
                str(dataset.validation.min_feature_coverage),
            ]
    if dataset.vintage_policy:
        args += ["--vintage_policy", dataset.vintage_policy.value]
    if dataset.vintage_as_of:
        args += ["--vintage_as_of", dataset.vintage_as_of]

    if cfg.hpo.enabled:
        args.append("--hpo")
        args += [
            "--hpo_epochs",
            str(cfg.hpo.epochs),
            "--hpo_batch_size",
            str(cfg.hpo.batch_size),
            "--hpo_tail_rows",
            str(cfg.hpo.tail_rows),
            "--hpo_limit_groups",
            str(cfg.hpo.limit_groups),
        ]

    if cfg.teacher.enabled:
        args.append("--train")
        args += ["--teacher_model_id", cfg.teacher.model_id]
        if cfg.teacher.feature_registry_dir:
            args += ["--feature_registry_dir", cfg.teacher.feature_registry_dir]
        if cfg.teacher.feature_set_id:
            args += ["--feature_set_id", cfg.teacher.feature_set_id]
        args += ["--max_epochs", str(cfg.teacher.max_epochs)]

    if cfg.student.enabled:
        args.append("--distill_student")
        args += ["--student_model_id", cfg.student.model_id]
        if cfg.student.parent_model_id:
            args += ["--student_parent_model_id", cfg.student.parent_model_id]
        if cfg.student.model_registry_dir:
            args += ["--student_model_registry_dir", cfg.student.model_registry_dir]
        if cfg.student.feature_registry_dir:
            args += ["--student_feature_registry_dir", cfg.student.feature_registry_dir]
        if cfg.student.feature_set_id:
            args += ["--student_feature_set_id", cfg.student.feature_set_id]
        args += ["--student_objective", cfg.student.objective]
        args += ["--student_kd_lambda", str(cfg.student.kd_lambda)]
        args += ["--student_early_stopping", str(cfg.student.early_stopping)]
        if cfg.student.opset is not None:
            args += ["--student_opset", str(cfg.student.opset)]
        if cfg.student.use_val_for_distill:
            args.append("--student_use_val_for_distill")

    if cfg.promotions is not None:
        if cfg.promotions.auto_register_model:
            args.append("--auto_register_model")
        if cfg.promotions.gates_json:
            args += ["--gates_json", cfg.promotions.gates_json]
        if cfg.promotions.auto_promote:
            args.append("--auto_promote")
        if cfg.promotions.deploy_target:
            args += ["--deploy_target", cfg.promotions.deploy_target]
        if cfg.promotions.auto_register_features:
            args.append("--auto_register_features")
        if cfg.promotions.feature_metrics_json:
            args += ["--feature_metrics_json", cfg.promotions.feature_metrics_json]
        if cfg.promotions.refresh_features:
            args.append("--refresh_features")

    if cfg.auto_fill and cfg.auto_fill.enabled:
        args.append("--auto_fill_universe")
        if cfg.auto_fill.dataset_id:
            args += ["--auto_fill_dataset_id", cfg.auto_fill.dataset_id]
        if cfg.auto_fill.instrument_ids:
            args += [
                "--auto_fill_instrument_ids",
                ",".join(cfg.auto_fill.instrument_ids),
            ]
        if not cfg.auto_fill.include_l2:
            args.append("--auto_fill_skip_l2")
        if cfg.auto_fill.l2_dataset_id:
            args += ["--auto_fill_l2_dataset_id", cfg.auto_fill.l2_dataset_id]
        if cfg.auto_fill.l2_schema:
            args += ["--auto_fill_l2_schema", cfg.auto_fill.l2_schema]
        if cfg.auto_fill.l2_days is not None:
            args += ["--auto_fill_l2_days", str(cfg.auto_fill.l2_days)]
        if cfg.auto_fill.l2_progress_file:
            args += ["--auto_fill_l2_progress_file", cfg.auto_fill.l2_progress_file]
        if cfg.auto_fill.include_l3:
            args.append("--auto_fill_include_l3")
        if cfg.auto_fill.l3_dataset_id:
            args += ["--auto_fill_l3_dataset_id", cfg.auto_fill.l3_dataset_id]
        if cfg.auto_fill.l3_schema:
            args += ["--auto_fill_l3_schema", cfg.auto_fill.l3_schema]
        if cfg.auto_fill.l3_days is not None:
            args += ["--auto_fill_l3_days", str(cfg.auto_fill.l3_days)]
        if not cfg.auto_fill.disable_dataset_l2_ingest:
            args.append("--auto_fill_allow_dataset_l2_ingest")

    if cfg.integration and cfg.integration.enabled:
        args.append("--attach-runtime")
        if cfg.integration.db_connection:
            args += ["--runtime-db-connection", cfg.integration.db_connection]
        if cfg.integration.auto_start_postgres:
            args.append("--runtime-auto-start-db")
        if cfg.integration.auto_migrate:
            args.append("--runtime-auto-migrate")
        if not cfg.integration.ensure_healthy:
            args.append("--runtime-no-ensure-healthy")
        if cfg.integration.strict_protocol_validation:
            args.append("--runtime-strict-protocol-validation")
        if not cfg.integration.run_validators:
            args.append("--runtime-skip-validators")

    if ingestion is not None:
        if ingestion.enabled:
            args.append("--ingest")
        args += ["--dataset_id", ingestion.dataset_id]
        args += ["--schema", ingestion.schema]
        if ingestion.instruments:
            args += ["--instruments", ",".join(ingestion.instruments)]
        args += ["--lookback_days", str(ingestion.lookback_days)]
        args += ["--coverage_mode", ingestion.coverage_mode]
        args += ["--write_mode", ingestion.write_mode]
        if ingestion.catalog_path:
            args += ["--catalog_path", ingestion.catalog_path]

    return args


def _market_inputs_to_payload(
    inputs: tuple[MarketDatasetInput, ...],
) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for item in inputs:
        entry: dict[str, object] = {}
        if item.descriptor_id is not None:
            entry["descriptor_id"] = item.descriptor_id
        if item.dataset_id is not None:
            entry["dataset_id"] = item.dataset_id
        if item.symbols is not None:
            entry["symbols"] = list(item.symbols)
        if item.schema_override is not None:
            entry["schema"] = item.schema_override
        if item.storage_kind_override is not None:
            entry["storage_kind"] = item.storage_kind_override.value
        if item.start is not None:
            entry["start"] = item.start
        if item.end is not None:
            entry["end"] = item.end
        payload.append(entry)
    return payload
