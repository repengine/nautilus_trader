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
from pathlib import Path
from typing import Any

from ml.orchestration.pipeline_orchestrator import DatasetBuildConfig
from ml.orchestration.pipeline_orchestrator import HPOConfig
from ml.orchestration.pipeline_orchestrator import OrchestratorConfig
from ml.orchestration.pipeline_orchestrator import PromotionsConfig
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
    return DatasetBuildConfig(
        data_dir=_as_str(data.get("data_dir", "data/tier1")),
        symbols=_as_str(data.get("symbols", "SPY.NYSE")),
        out_dir=_as_str(data.get("out_dir", "ml_out")),
        include_macro=_coerce_bool(data.get("include_macro", False)),
        macro_lag_days=_as_int(data.get("macro_lag_days", 1)),
        include_micro=_coerce_bool(data.get("include_micro", False)),
        include_l2=_coerce_bool(data.get("include_l2", False)),
        horizon_minutes=_as_int(data.get("horizon_minutes", 15)),
        threshold=_as_float(data.get("threshold", 0.001)),
        lookback_periods=_as_int(data.get("lookback_periods", 30)),
    )


def _build_hpo_cfg(data: dict[str, Any]) -> HPOConfig:
    return HPOConfig(
        enabled=_coerce_bool(data.get("enabled", False)),
        epochs=_as_int(data.get("epochs", 2)),
        batch_size=_as_int(data.get("batch_size", 32)),
        tail_rows=_as_int(data.get("tail_rows", 5000)),
        limit_groups=_as_int(data.get("limit_groups", 50)),
    )


def _build_teacher_cfg(data: dict[str, Any]) -> TeacherTrainConfig:
    return TeacherTrainConfig(
        enabled=_coerce_bool(data.get("enabled", True)),
        model_id=_as_str(data.get("model_id", "teacher_model")),
        feature_registry_dir=data.get("feature_registry_dir"),
        feature_set_id=data.get("feature_set_id"),
        max_epochs=_as_int(data.get("max_epochs", 5)),
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
        return OrchestratorConfig(dataset=ds, hpo=HPOConfig(), teacher=TeacherTrainConfig())

    cfg_path = Path(path)
    data = _load_json_or_toml(cfg_path)
    dataset_cfg = _build_dataset_cfg(data.get("dataset", {}))
    hpo_cfg = _build_hpo_cfg(data.get("hpo", {}))
    teacher_cfg = _build_teacher_cfg(data.get("teacher", {}))
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
    return OrchestratorConfig(dataset=dataset_cfg, hpo=hpo_cfg, teacher=teacher_cfg, promotions=promotions_cfg)


def to_pipeline_args(cfg: OrchestratorConfig) -> list[str]:
    """
    Convert OrchestratorConfig to CLI arguments for ml.cli.pipeline_orchestrator.

    Only includes dataset/HPO/teacher arguments. The orchestrator CLI controls
    ingestion and writer/coverage choices independently.
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

    # Normalize to strings for safety
    return [str(x) for x in args]


__all__ = ["load_orchestrator_config", "to_pipeline_args"]
