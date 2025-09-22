#!/usr/bin/env python3

from __future__ import annotations


# ruff: noqa: E402  # Allow module docstring preceding imports per project style

"""
Configuration loader for the pipeline orchestrator (cold path).

This module parses a lightweight JSON or TOML configuration file into the
frozen dataclasses defined in ``ml.orchestration.pipeline_orchestrator`` and
converts that configuration back into CLI arguments for the orchestrator CLI.

Notes
-----
- The loader is intentionally minimal and typed. It validates presence and
  types of expected fields but does not attempt to cover every optional flag.
- Promotions- and deployment-related flags are passed via the orchestrator
  CLI directly and are not part of the OrchestratorConfig dataclass.

"""

from dataclasses import is_dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

from ml.data import DatasetValidationConfig
from ml.orchestration.pipeline_orchestrator import AutoFillUniverseConfig
from ml.orchestration.pipeline_orchestrator import DatasetBuildConfig
from ml.orchestration.pipeline_orchestrator import HPOConfig
from ml.orchestration.pipeline_orchestrator import IntegrationConfig
from ml.orchestration.pipeline_orchestrator import OrchestratorConfig
from ml.orchestration.pipeline_orchestrator import PromotionsConfig
from ml.orchestration.pipeline_orchestrator import StudentDistillConfig
from ml.orchestration.pipeline_orchestrator import TeacherTrainConfig


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _coerce_bool(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off", ""}:
            return False
    return default


def _as_int(val: Any, *, default: int | None = None) -> int:
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str) and val.strip():
        return int(val)
    if default is None:
        raise ValueError("Missing required integer value")
    return default


def _as_float(val: Any, *, default: float | None = None) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str) and val.strip():
        return float(val)
    if default is None:
        raise ValueError("Missing required float value")
    return default


def _as_str(val: Any, *, default: str | None = None) -> str:
    if isinstance(val, str):
        return val
    if default is None:
        raise ValueError("Missing required string value")
    return default


def _as_tuple(val: Any) -> tuple[str, ...] | None:
    if val is None:
        return None
    if isinstance(val, (list, tuple, set)):
        items = [str(item).strip() for item in val if str(item).strip()]
        return tuple(items) or None
    if isinstance(val, str):
        items = [piece.strip() for piece in val.split(",") if piece.strip()]
        return tuple(items) or None
    raise TypeError(f"Expected iterable or str for macro series ids, got {type(val)!r}")


def _load_json_or_toml(path: Path) -> dict[str, Any]:
    text = _load_text(path)
    if path.suffix.lower() in {".json"}:
        import json

        return dict(json.loads(text))
    if path.suffix.lower() in {".toml", ".tml", ".toml.txt"}:
        try:  # Python 3.11+
            import tomllib as toml
        except Exception:  # pragma: no cover - fallback for older runtimes
            import tomli as toml  # type: ignore[no-redef]
        return dict(toml.loads(text))
    raise ValueError(f"Unsupported config format for: {path}")


def _build_dataset_cfg(data: dict[str, Any]) -> DatasetBuildConfig:
    macro_series = _as_tuple(data.get("macro_series_ids"))
    return DatasetBuildConfig(
        data_dir=_as_str(data.get("data_dir", "data/tier1")),
        symbols=_as_str(data.get("symbols", "SPY.NYSE")),
        out_dir=_as_str(data.get("out_dir", "ml_out")),
        instrument_ids=_as_tuple(data.get("instrument_ids")),
        include_macro=_coerce_bool(data.get("include_macro", False)),
        macro_lag_days=_as_int(data.get("macro_lag_days", 1)),
        include_micro=_coerce_bool(data.get("include_micro", False)),
        include_l2=_coerce_bool(data.get("include_l2", False)),
        horizon_minutes=_as_int(data.get("horizon_minutes", 15)),
        threshold=_as_float(data.get("threshold", 0.001)),
        lookback_periods=_as_int(data.get("lookback_periods", 30)),
        auto_refresh_macro=_coerce_bool(data.get("auto_refresh_macro", True)),
        macro_staleness_hours=_as_int(data.get("macro_staleness_hours", 24)),
        macro_series_ids=macro_series,
        macro_fred_path=data.get("macro_fred_path"),
        validation=_build_validation_cfg(data.get("validation"), macro_series),
    )


def _build_auto_fill_cfg(data: Any) -> AutoFillUniverseConfig | None:
    if not isinstance(data, dict):
        return None

    enabled = _coerce_bool(data.get("enabled", False))
    dataset_id = _as_str(data.get("dataset_id", "EQUS.MINI"))
    include_l2 = _coerce_bool(data.get("include_l2", False))
    include_l3 = _coerce_bool(data.get("include_l3", False))
    l2_dataset = data.get("l2_dataset_id", "DBEQ.BASIC")
    l2_schema = data.get("l2_schema", "mbp-10")
    l2_days = data.get("l2_days")
    l3_days = data.get("l3_days")
    instrument_ids = _as_tuple(data.get("instrument_ids"))
    cfg = AutoFillUniverseConfig(
        enabled=enabled,
        dataset_id=dataset_id,
        include_bars=_coerce_bool(data.get("include_bars", True)),
        include_tbbo=_coerce_bool(data.get("include_tbbo", True)),
        include_trades=_coerce_bool(data.get("include_trades", True)),
        include_l2=include_l2,
        include_l3=include_l3,
        l2_dataset_id=_as_str(l2_dataset),
        l2_schema=_as_str(l2_schema),
        l2_days=None if l2_days is None else _as_int(l2_days),
        l2_progress_file=data.get("l2_progress_file"),
        disable_dataset_l2_ingest=_coerce_bool(
            data.get("disable_dataset_l2_ingest", True),
        ),
        instrument_ids=instrument_ids,
        l3_dataset_id=data.get("l3_dataset_id"),
        l3_schema=data.get("l3_schema"),
        l3_days=None if l3_days is None else _as_int(l3_days),
    )
    return cfg


def _build_hpo_cfg(data: dict[str, Any]) -> HPOConfig:
    return HPOConfig(
        enabled=_coerce_bool(data.get("enabled", False)),
        epochs=_as_int(data.get("epochs", 2)),
        batch_size=_as_int(data.get("batch_size", 32)),
        tail_rows=_as_int(data.get("tail_rows", 5000)),
        limit_groups=_as_int(data.get("limit_groups", 50)),
    )


def _build_validation_cfg(
    data: Any,
    macro_series: tuple[str, ...] | None,
) -> DatasetValidationConfig | None:
    if not isinstance(data, dict):
        if macro_series:
            return DatasetValidationConfig(require_macro_series=macro_series)
        return None
    cfg = DatasetValidationConfig()
    modified = False
    if "min_rows" in data:
        cfg = replace(cfg, min_rows=_as_int(data["min_rows"]))
        modified = True
    if "min_positive_rate" in data:
        cfg = replace(cfg, min_positive_rate=float(data["min_positive_rate"]))
        modified = True
    if "max_positive_rate" in data:
        cfg = replace(cfg, max_positive_rate=float(data["max_positive_rate"]))
        modified = True
    if "min_feature_coverage" in data:
        cfg = replace(cfg, min_feature_coverage=float(data["min_feature_coverage"]))
        modified = True
    if "require_macro_series" in data:
        macro_arg = _as_tuple(data.get("require_macro_series"))
        cfg = replace(cfg, require_macro_series=macro_arg)
        modified = True
    elif macro_series:
        cfg = replace(cfg, require_macro_series=macro_series)
        modified = True
    return cfg if modified else None


def _build_teacher_cfg(data: dict[str, Any]) -> TeacherTrainConfig:
    return TeacherTrainConfig(
        enabled=_coerce_bool(data.get("enabled", True)),
        model_id=_as_str(data.get("model_id", "teacher_model")),
        feature_registry_dir=data.get("feature_registry_dir"),
        feature_set_id=data.get("feature_set_id"),
        max_epochs=_as_int(data.get("max_epochs", 5)),
    )


def _build_student_cfg(data: dict[str, Any]) -> StudentDistillConfig:
    return StudentDistillConfig(
        enabled=_coerce_bool(data.get("enabled", False)),
        model_id=_as_str(data.get("model_id", "student_model")),
        parent_model_id=data.get("parent_model_id"),
        model_registry_dir=data.get("model_registry_dir"),
        feature_registry_dir=data.get("feature_registry_dir"),
        feature_set_id=data.get("feature_set_id"),
        objective=_as_str(data.get("objective", "logit_mse")),
        kd_lambda=float(data.get("kd_lambda", 0.5)),
        early_stopping=_as_int(data.get("early_stopping", 200)),
        opset=data.get("opset"),
        use_val_for_distill=_coerce_bool(data.get("use_val_for_distill", False)),
    )


def _build_integration_cfg(data: Any) -> IntegrationConfig | None:
    if not isinstance(data, dict):
        return None

    enabled = _coerce_bool(data.get("enabled", False))
    db_connection = data.get("db_connection")
    auto_start_postgres = _coerce_bool(data.get("auto_start_postgres", False))
    auto_migrate = _coerce_bool(data.get("auto_migrate", False))
    ensure_healthy = _coerce_bool(data.get("ensure_healthy", True))
    strict_raw = data.get("strict_protocol_validation")
    strict_protocol_validation: bool | None
    if strict_raw is None:
        strict_protocol_validation = None
    else:
        strict_protocol_validation = _coerce_bool(strict_raw, default=True)
    run_validators = _coerce_bool(data.get("run_validators", True))

    return IntegrationConfig(
        enabled=enabled,
        db_connection=db_connection,
        auto_start_postgres=auto_start_postgres,
        auto_migrate=auto_migrate,
        ensure_healthy=ensure_healthy,
        strict_protocol_validation=strict_protocol_validation,
        run_validators=run_validators,
    )


def load_orchestrator_config(path: str | None) -> OrchestratorConfig:
    """
    Load an OrchestratorConfig from a JSON/TOML file.

    The expected top-level keys are ``dataset``, ``hpo``, and ``teacher``; any
    additional keys are ignored by this loader.

    """
    # Provide a reasonable default if path is None (suitable for local runs/tests)
    if path is None:
        ds = DatasetBuildConfig(data_dir="data/tier1", symbols="SPY.NYSE", out_dir="ml_out")
        return OrchestratorConfig(
            dataset=ds,
            hpo=HPOConfig(),
            teacher=TeacherTrainConfig(),
            student=StudentDistillConfig(),
        )

    cfg_path = Path(path)
    data = _load_json_or_toml(cfg_path)
    dataset_cfg = _build_dataset_cfg(data.get("dataset", {}))
    hpo_cfg = _build_hpo_cfg(data.get("hpo", {}))
    teacher_cfg = _build_teacher_cfg(data.get("teacher", {}))
    student_cfg = _build_student_cfg(data.get("student", {}))
    integration_cfg = _build_integration_cfg(data.get("integration"))
    auto_fill_cfg = _build_auto_fill_cfg(data.get("auto_fill"))
    # Optional promotions/feature refresh
    promotions_cfg: PromotionsConfig | None = None
    if isinstance(data.get("promotions"), dict):
        p = data["promotions"]
        promotions_cfg = PromotionsConfig(
            auto_register_model=_coerce_bool(p.get("auto_register_model", False)),
            gates_json=p.get("gates_json"),
            auto_promote=_coerce_bool(p.get("auto_promote", False)),
            deploy_target=p.get("deploy_target"),
            auto_register_features=_coerce_bool(p.get("auto_register_features", False)),
            feature_metrics_json=p.get("feature_metrics_json"),
            refresh_features=_coerce_bool(p.get("refresh_features", False)),
        )
    return OrchestratorConfig(
        dataset=dataset_cfg,
        hpo=hpo_cfg,
        teacher=teacher_cfg,
        student=student_cfg,
        promotions=promotions_cfg,
        integration=integration_cfg if integration_cfg and integration_cfg.enabled else None,
        auto_fill=auto_fill_cfg if auto_fill_cfg and auto_fill_cfg.enabled else None,
    )


def to_pipeline_args(cfg: OrchestratorConfig) -> list[str]:
    """
    Convert OrchestratorConfig to CLI arguments for ml.cli.pipeline_orchestrator.

    Only includes dataset/HPO/teacher arguments. The orchestrator CLI controls ingestion
    and writer/coverage choices independently.

    """
    if not is_dataclass(cfg):  # defensive typing guard
        raise TypeError("cfg must be a dataclass OrchestratorConfig")

    args: list[str] = [
        "--data_dir",
        cfg.dataset.data_dir,
        "--symbols",
        cfg.dataset.symbols,
        "--out_dir",
        cfg.dataset.out_dir,
        "--horizon_minutes",
        str(cfg.dataset.horizon_minutes),
        "--threshold",
        str(cfg.dataset.threshold),
        "--lookback_periods",
        str(cfg.dataset.lookback_periods),
    ]
    if cfg.dataset.include_macro:
        args += ["--include_macro", "--macro_lag_days", str(cfg.dataset.macro_lag_days)]
    if cfg.dataset.include_micro:
        args += ["--include_micro"]
    if cfg.dataset.include_l2:
        args += ["--include_l2"]

    # HPO flags
    if cfg.hpo.enabled:
        args += [
            "--hpo",
            "--hpo_epochs",
            str(cfg.hpo.epochs),
            "--hpo_batch_size",
            str(cfg.hpo.batch_size),
            "--hpo_tail_rows",
            str(cfg.hpo.tail_rows),
            "--hpo_limit_groups",
            str(cfg.hpo.limit_groups),
        ]

    # Teacher flags
    if cfg.teacher.enabled:
        args += [
            "--train",
            "--teacher_model_id",
            cfg.teacher.model_id,
            "--max_epochs",
            str(cfg.teacher.max_epochs),
        ]
        if cfg.teacher.feature_registry_dir is not None:
            args += ["--feature_registry_dir", cfg.teacher.feature_registry_dir]
        if cfg.teacher.feature_set_id is not None:
            args += ["--feature_set_id", cfg.teacher.feature_set_id]

    if cfg.student.enabled:
        args += [
            "--distill_student",
            "--student_model_id",
            cfg.student.model_id,
        ]
        if cfg.student.parent_model_id:
            args += ["--student_parent_model_id", cfg.student.parent_model_id]
        if cfg.student.model_registry_dir:
            args += [
                "--student_model_registry_dir",
                cfg.student.model_registry_dir,
            ]
        if cfg.student.feature_registry_dir:
            args += [
                "--student_feature_registry_dir",
                cfg.student.feature_registry_dir,
            ]
        if cfg.student.feature_set_id:
            args += ["--student_feature_set_id", cfg.student.feature_set_id]
        if cfg.student.objective != "logit_mse":
            args += ["--student_objective", cfg.student.objective]
        if cfg.student.kd_lambda != 0.5:
            args += ["--student_kd_lambda", str(cfg.student.kd_lambda)]
        if cfg.student.early_stopping != 200:
            args += [
                "--student_early_stopping",
                str(cfg.student.early_stopping),
            ]
        if cfg.student.opset is not None:
            args += ["--student_opset", str(cfg.student.opset)]
    if cfg.student.use_val_for_distill:
        args += ["--student_use_val_for_distill"]

    # Promotions and feature registration flags (optional)
    prom = getattr(cfg, "promotions", None)
    if isinstance(prom, PromotionsConfig):
        if prom.auto_register_model:
            args += ["--auto_register_model"]
        if prom.gates_json:
            args += ["--gates_json", prom.gates_json]
        if prom.auto_promote:
            args += ["--auto_promote"]
        if prom.deploy_target:
            args += ["--deploy_target", prom.deploy_target]
        if prom.auto_register_features:
            args += ["--auto_register_features"]
        if prom.feature_metrics_json:
            args += ["--feature_metrics_json", prom.feature_metrics_json]
        if prom.refresh_features:
            args += ["--refresh_features"]

    integ_obj = cfg.integration
    if isinstance(integ_obj, IntegrationConfig) and integ_obj.enabled:
        args.append("--attach-runtime")
        if integ_obj.db_connection:
            args += ["--runtime-db-connection", integ_obj.db_connection]
        if integ_obj.auto_start_postgres:
            args.append("--runtime-auto-start-db")
        if integ_obj.auto_migrate:
            args.append("--runtime-auto-migrate")
        if not integ_obj.ensure_healthy:
            args.append("--runtime-no-ensure-healthy")
        if integ_obj.strict_protocol_validation:
            args.append("--runtime-strict-protocol-validation")
        if not integ_obj.run_validators:
            args.append("--runtime-skip-validators")

    auto_fill_cfg = getattr(cfg, "auto_fill", None)
    if isinstance(auto_fill_cfg, AutoFillUniverseConfig) and auto_fill_cfg.enabled:
        args.append("--auto_fill_universe")
        if auto_fill_cfg.dataset_id:
            args += ["--auto_fill_dataset_id", auto_fill_cfg.dataset_id]
        if auto_fill_cfg.instrument_ids:
            args += [
                "--auto_fill_instrument_ids",
                ",".join(auto_fill_cfg.instrument_ids),
            ]
        if auto_fill_cfg.l2_days is not None:
            args += ["--auto_fill_l2_days", str(auto_fill_cfg.l2_days)]
        if not auto_fill_cfg.include_l2:
            args.append("--auto_fill_skip_l2")
        if auto_fill_cfg.l2_dataset_id:
            args += ["--auto_fill_l2_dataset_id", auto_fill_cfg.l2_dataset_id]
        if auto_fill_cfg.l2_schema:
            args += ["--auto_fill_l2_schema", auto_fill_cfg.l2_schema]
        if auto_fill_cfg.l2_progress_file:
            args += [
                "--auto_fill_l2_progress_file",
                auto_fill_cfg.l2_progress_file,
            ]
        if auto_fill_cfg.include_l3:
            args.append("--auto_fill_include_l3")
            if auto_fill_cfg.l3_dataset_id:
                args += [
                    "--auto_fill_l3_dataset_id",
                    auto_fill_cfg.l3_dataset_id,
                ]
            if auto_fill_cfg.l3_schema:
                args += ["--auto_fill_l3_schema", auto_fill_cfg.l3_schema]
            if auto_fill_cfg.l3_days is not None:
                args += ["--auto_fill_l3_days", str(auto_fill_cfg.l3_days)]
        if not auto_fill_cfg.disable_dataset_l2_ingest:
            args.append("--auto_fill_allow_dataset_l2_ingest")

    # Normalize to strings for safety
    return [str(x) for x in args]


__all__ = ["load_orchestrator_config", "to_pipeline_args"]
