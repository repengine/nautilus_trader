from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import ml.cli.sanity_check as sanity_cli
from ml.cli.check_pipeline_health import main as canonical_check_pipeline_health_main
from ml.cli.coverage import CoverageReporter as CanonicalCoverageReporter
from ml.cli.coverage import _auto_persistence_config as canonical_auto_persistence_config
from ml.cli.coverage import _backend_override_persistence_config as canonical_backend_override_persistence_config
from ml.cli.coverage import _coerce_registry_path as canonical_coerce_registry_path
from ml.cli.coverage import _resolve_registry_path as canonical_resolve_registry_path
from ml.cli.coverage import apply_backfill as canonical_apply_backfill
from ml.cli.coverage import main as canonical_coverage_main
from ml.cli.coverage import plan_backfill as canonical_plan_backfill
from ml.cli.observability_backfill import main as canonical_observability_backfill_main
from ml.core.common.health_monitoring import (
    aggregate_integration_health as canonical_aggregate_integration_health,
)
from ml.monitoring.health import ComponentHealth as CanonicalComponentHealth
from ml.monitoring.health import HealthStatus as CanonicalHealthStatus
from ml.monitoring.health import PipelineHealthChecker as CanonicalPipelineHealthChecker
from ml.monitoring.health import Thresholds as CanonicalThresholds
from ml.monitoring.health import format_human_output as canonical_format_human_output
from ml.monitoring.health import format_json_output as canonical_format_json_output
from ml.monitoring.health import run_pipeline_health_checks as canonical_run_pipeline_health_checks
from ml.observability.backfill import _load_jsonl_files as canonical_load_jsonl_files
from ml.observability.backfill import _load_name_shards as canonical_load_name_shards
from ml.tools.sanity_check import main as canonical_sanity_check_main
from ml.training.teacher.hpo_tft import HAS_OPTUNA as canonical_hpo_has_optuna
from ml.training.teacher.hpo_tft import main as canonical_hpo_main


REPO_ROOT = Path(__file__).resolve().parents[4]
_RETIRED_TASK_PACKAGE_INIT_MODULES: tuple[Path, ...] = (
    REPO_ROOT / "ml" / "tasks" / "__init__.py",
    REPO_ROOT / "ml" / "tasks" / "caches" / "__init__.py",
    REPO_ROOT / "ml" / "tasks" / "datasets" / "__init__.py",
    REPO_ROOT / "ml" / "tasks" / "dev" / "__init__.py",
    REPO_ROOT / "ml" / "tasks" / "ingest" / "__init__.py",
    REPO_ROOT / "ml" / "tasks" / "monitoring" / "__init__.py",
    REPO_ROOT / "ml" / "tasks" / "observability" / "__init__.py",
    REPO_ROOT / "ml" / "tasks" / "pipelines" / "__init__.py",
    REPO_ROOT / "ml" / "tasks" / "training" / "__init__.py",
)
_RETIRED_TASK_MODULE_SHIMS: tuple[Path, ...] = (
    REPO_ROOT / "ml" / "tasks" / "caches" / "hydration.py",
    REPO_ROOT / "ml" / "tasks" / "datasets" / "production.py",
    REPO_ROOT / "ml" / "tasks" / "datasets" / "report.py",
    REPO_ROOT / "ml" / "tasks" / "datasets" / "splits.py",
    REPO_ROOT / "ml" / "tasks" / "datasets" / "tft.py",
    REPO_ROOT / "ml" / "tasks" / "datasets" / "tft_cli.py",
    REPO_ROOT / "ml" / "tasks" / "db.py",
    REPO_ROOT / "ml" / "tasks" / "dev" / "sanity_check.py",
    REPO_ROOT / "ml" / "tasks" / "ingest" / "alternative.py",
    REPO_ROOT / "ml" / "tasks" / "ingest" / "backfill.py",
    REPO_ROOT / "ml" / "tasks" / "ingest" / "l2.py",
    REPO_ROOT / "ml" / "tasks" / "ingest" / "recent.py",
    REPO_ROOT / "ml" / "tasks" / "ingest" / "supplementary.py",
    REPO_ROOT / "ml" / "tasks" / "ingest" / "yahoo.py",
    REPO_ROOT / "ml" / "tasks" / "monitoring" / "coverage.py",
    REPO_ROOT / "ml" / "tasks" / "monitoring" / "health.py",
    REPO_ROOT / "ml" / "tasks" / "observability" / "backfill.py",
    REPO_ROOT / "ml" / "tasks" / "observability" / "flush.py",
    REPO_ROOT / "ml" / "tasks" / "registry.py",
    REPO_ROOT / "ml" / "tasks" / "pipelines" / "runner.py",
    REPO_ROOT / "ml" / "tasks" / "pipelines" / "scheduler.py",
    REPO_ROOT / "ml" / "tasks" / "training" / "hpo_tft.py",
    REPO_ROOT / "ml" / "tasks" / "training" / "quick.py",
)
_RETIRED_TASK_MODULE_IMPORTS: tuple[str, ...] = (
    "ml.tasks.caches.hydration",
    "ml.tasks.datasets.production",
    "ml.tasks.datasets.report",
    "ml.tasks.datasets.splits",
    "ml.tasks.datasets.tft",
    "ml.tasks.datasets.tft_cli",
    "ml.tasks.db",
    "ml.tasks.dev.sanity_check",
    "ml.tasks.ingest.alternative",
    "ml.tasks.ingest.backfill",
    "ml.tasks.ingest.l2",
    "ml.tasks.ingest.recent",
    "ml.tasks.ingest.supplementary",
    "ml.tasks.ingest.yahoo",
    "ml.tasks.monitoring.coverage",
    "ml.tasks.monitoring.health",
    "ml.tasks.observability.backfill",
    "ml.tasks.observability.flush",
    "ml.tasks.registry",
    "ml.tasks.pipelines.runner",
    "ml.tasks.pipelines.scheduler",
    "ml.tasks.training.hpo_tft",
    "ml.tasks.training.quick",
)


def test_retired_task_package_shims_have_no_init_modules() -> None:
    lingering = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in _RETIRED_TASK_PACKAGE_INIT_MODULES
        if path.exists()
    ]
    assert not lingering, f"Task package __init__ shims should be retired: {lingering}"


def test_retired_leaf_task_module_shims_are_removed() -> None:
    lingering = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in _RETIRED_TASK_MODULE_SHIMS
        if path.exists()
    ]
    assert not lingering, f"Task module shims should be retired: {lingering}"

    for module_name in _RETIRED_TASK_MODULE_IMPORTS:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)


def test_sanity_cli_uses_canonical_main() -> None:
    assert sanity_cli.sanity_main is canonical_sanity_check_main


def test_monitoring_and_observability_canonical_owners_remain_available() -> None:
    assert callable(canonical_coverage_main)
    assert callable(canonical_plan_backfill)
    assert callable(canonical_apply_backfill)
    assert CanonicalCoverageReporter.__name__ == "CoverageReporter"
    assert callable(canonical_coerce_registry_path)
    assert callable(canonical_resolve_registry_path)
    assert callable(canonical_auto_persistence_config)
    assert callable(canonical_backend_override_persistence_config)

    assert callable(canonical_check_pipeline_health_main)
    assert callable(canonical_aggregate_integration_health)
    assert CanonicalComponentHealth.__name__ == "ComponentHealth"
    assert CanonicalHealthStatus.__name__ == "HealthStatus"
    assert CanonicalPipelineHealthChecker.__name__ == "PipelineHealthChecker"
    assert CanonicalThresholds.__name__ == "Thresholds"
    assert callable(canonical_format_human_output)
    assert callable(canonical_format_json_output)
    assert callable(canonical_run_pipeline_health_checks)

    assert callable(canonical_observability_backfill_main)
    assert callable(canonical_load_jsonl_files)
    assert callable(canonical_load_name_shards)


def test_teacher_hpo_symbols_remain_available() -> None:
    assert callable(canonical_hpo_main)
    assert isinstance(canonical_hpo_has_optuna, bool)
