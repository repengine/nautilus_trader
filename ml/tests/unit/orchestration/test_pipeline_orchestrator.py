#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Callable

import json
import numpy as np
import pandas as pd
import pytest

from ml.config.coverage import CoveragePolicy
from ml.orchestration.pipeline_orchestrator import AutoFillUniverseConfig
from ml.orchestration.pipeline_orchestrator import DatasetBuildConfig
from ml.orchestration.pipeline_orchestrator import HPOConfig
from ml.orchestration.pipeline_orchestrator import IntegrationConfig
from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator
from ml.orchestration.pipeline_orchestrator import OrchestratorConfig
from ml.orchestration.pipeline_orchestrator import StudentDistillConfig
from ml.orchestration.pipeline_orchestrator import TeacherTrainConfig
from ml.orchestration.pipeline_orchestrator import _build_auto_fill_config_from_args
from ml.orchestration.pipeline_orchestrator import parse_args


@dataclass
class _Coverage:
    def read_bucket_coverage(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> set[int]:
        return set()


@dataclass
class _Writer:
    def write(self, *, dataset_id: str, schema: str, instrument_id: str, df: pd.DataFrame) -> int:
        return len(df.index) if df is not None and not df.empty else 0


@dataclass
class _Registry:
    def emit_event(self, **kwargs: Any) -> None:  # pragma: no cover - stub
        return None

    def update_watermark(self, **kwargs: Any) -> None:  # pragma: no cover - stub
        return None


@dataclass
class _Ingestor:
    def ingest_time_window(self, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame({"ts_event": [1, 2]})


class _CliWrapper:
    def __init__(self, fn: Callable[[list[str] | None], int]) -> None:
        self._fn = fn

    def __call__(self, argv: list[str] | None = None) -> int:
        return self._fn(argv)


def _ok(_: list[str] | None = None) -> int:
    return 0


def test_pipeline_orchestrator_runs_all_phases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    coverage = _Coverage()
    writer = _Writer()
    registry = _Registry()
    ingestor = _Ingestor()

    # record calls
    called: dict[str, int] = {"build": 0, "hpo": 0, "train": 0, "distill": 0}
    feature_registry_dir = tmp_path / "features"
    feature_registry_dir.mkdir(parents=True, exist_ok=True)
    model_registry_dir = tmp_path / "models"
    model_registry_dir.mkdir(parents=True, exist_ok=True)

    def _build(argv: list[str] | None = None) -> int:
        called["build"] += 1
        # simulate dataset.csv emitted
        out_dir = None
        if argv is not None and "--out_dir" in argv:
            out_dir = Path(argv[argv.index("--out_dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "dataset.csv").write_text("id,ts_event\n1,1\n")
            np.savez(
                out_dir / "features_npz.npz",
                X_train=np.array([[0.1]], dtype=np.float32),
                X_val=np.array([[0.2]], dtype=np.float32),
                feature_names=np.array(["f1"], dtype=object),
            )
            sidecar = {
                "feature_registry_dir": str(feature_registry_dir),
                "feature_set_id": "fs1",
                "feature_names": ["f1"],
            }
            (out_dir / "feature_registration.json").write_text(
                json.dumps(sidecar),
                encoding="utf-8",
            )
        return 0

    def _hpo(argv: list[str] | None = None) -> int:
        called["hpo"] += 1
        return 0

    def _train(argv: list[str] | None = None) -> int:
        called["train"] += 1
        if argv is not None and "--out_dir" in argv:
            out_dir = Path(argv[argv.index("--out_dir") + 1])
            np.savez(out_dir / "teacher_preds.npz", q_train=np.array([0.3], dtype=np.float32))
        return 0

    def _distill(argv: list[str] | None = None) -> int:
        called["distill"] += 1
        return 0

    monkeypatch.setattr("ml.training.distillation.cli.main", _distill)

    orch = MLPipelineOrchestrator(
        coverage=coverage,
        writer=writer,
        registry=registry,
        ingestor=ingestor,
        build_main=_CliWrapper(_build),
        hpo_main=_CliWrapper(_hpo),
        teacher_main=_CliWrapper(_train),
    )

    cfg = OrchestratorConfig(
        dataset=DatasetBuildConfig(
            data_dir=str(tmp_path),
            symbols="SPY.NYSE",
            out_dir=str(tmp_path / "out"),
            include_micro=True,
            feature_registry_dir=str(feature_registry_dir),
        ),
        hpo=HPOConfig(enabled=True, epochs=1, batch_size=8, tail_rows=100, limit_groups=10),
        teacher=TeacherTrainConfig(enabled=True, model_id="teacher_X", max_epochs=1),
        student=StudentDistillConfig(
            enabled=True,
            model_id="student_X",
            model_registry_dir=str(model_registry_dir),
        ),
    )
    rc = orch.run(cfg)
    assert rc == 0
    assert called == {"build": 1, "hpo": 1, "train": 1, "distill": 1}


def test_pipeline_orchestrator_attach_runtime_sets_components(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    coverage = _Coverage()
    writer = _Writer()

    called: dict[str, int] = {"build": 0}

    def _build(argv: list[str] | None = None) -> int:
        called["build"] += 1
        if argv is not None and "--out_dir" in argv:
            out_dir = Path(argv[argv.index("--out_dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "dataset.csv").write_text("id,ts_event\n1,1\n", encoding="utf-8")
        return 0

    orch_metrics_calls: list[int] = []
    orch_events_calls: list[int] = []

    def _fake_metrics() -> int:
        orch_metrics_calls.append(1)
        return 0

    def _fake_events() -> int:
        orch_events_calls.append(1)
        return 0

    monkeypatch.setattr("tools.validate_metrics_bootstrap.main", _fake_metrics)
    monkeypatch.setattr("tools.validate_event_constants.main", _fake_events)

    factory_calls: list[dict[str, Any]] = []
    manager_instances: list[_Manager] = []

    class _Manager:
        def __init__(self, **kwargs: Any) -> None:
            factory_calls.append(kwargs)
            self.data_registry = object()
            self.feature_registry = object()
            self.model_registry = object()
            self.strategy_registry = object()
            self.feature_store = object()
            self.model_store = object()
            self.strategy_store = object()
            self.data_store = object()
            self.partition_manager = object()
            manager_instances.append(self)  # pragma: no mutate - tracking

    def _factory(**kwargs: Any) -> _Manager:
        return _Manager(**kwargs)

    orch = MLPipelineOrchestrator(
        coverage=coverage,
        writer=writer,
        data_registry=None,
        ingestor=None,
        build_main=_CliWrapper(_build),
        hpo_main=None,
        teacher_main=_CliWrapper(_ok),
        model_registry=None,
        feature_registry=None,
        integration_manager_factory=_factory,
    )

    cfg = OrchestratorConfig(
        dataset=DatasetBuildConfig(
            data_dir=str(tmp_path),
            symbols="SPY.NYSE",
            out_dir=str(tmp_path / "out"),
        ),
        hpo=HPOConfig(),
        teacher=TeacherTrainConfig(enabled=False),
        integration=IntegrationConfig(
            enabled=True,
            db_connection="postgresql://example",
            auto_start_postgres=True,
            auto_migrate=True,
            ensure_healthy=False,
            strict_protocol_validation=True,
            run_validators=True,
        ),
    )

    rc = orch.run(cfg)
    assert rc == 0
    assert called == {"build": 1}
    assert len(factory_calls) == 1
    assert factory_calls[0]["auto_start_postgres"] is True
    assert factory_calls[0]["auto_migrate"] is True
    assert factory_calls[0]["ensure_healthy"] is False
    assert factory_calls[0]["db_connection"] == "postgresql://example"
    assert factory_calls[0]["strict_protocol_validation"] is True

    manager = manager_instances[0]
    assert orch.data_registry is manager.data_registry
    assert orch.feature_registry is manager.feature_registry
    assert orch.model_registry is manager.model_registry
    assert orch.strategy_registry is manager.strategy_registry
    assert orch.feature_store is manager.feature_store
    assert orch.model_store is manager.model_store
    assert orch.strategy_store is manager.strategy_store
    assert orch.data_store is manager.data_store
    assert orch.partition_manager is manager.partition_manager
    assert orch_metrics_calls and orch_events_calls


def test_pipeline_orchestrator_attach_runtime_skips_validators_when_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    coverage = _Coverage()
    writer = _Writer()

    def _build(argv: list[str] | None = None) -> int:
        if argv is not None and "--out_dir" in argv:
            out_dir = Path(argv[argv.index("--out_dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "dataset.csv").write_text("id,ts_event\n1,1\n", encoding="utf-8")
        return 0

    metrics_called = False
    events_called = False

    def _metrics() -> int:
        nonlocal metrics_called
        metrics_called = True
        return 0

    def _events() -> int:
        nonlocal events_called
        events_called = True
        return 0

    monkeypatch.setattr("tools.validate_metrics_bootstrap.main", _metrics)
    monkeypatch.setattr("tools.validate_event_constants.main", _events)

    class _Manager:
        def __init__(self) -> None:
            self.data_registry = object()
            self.feature_registry = object()
            self.model_registry = object()
            self.strategy_registry = object()
            self.feature_store = object()
            self.model_store = object()
            self.strategy_store = object()
            self.data_store = object()
            self.partition_manager = object()

    orch = MLPipelineOrchestrator(
        coverage=coverage,
        writer=writer,
        data_registry=None,
        ingestor=None,
        build_main=_CliWrapper(_build),
        hpo_main=None,
        teacher_main=_CliWrapper(_ok),
        integration_manager_factory=lambda **_: _Manager(),
    )

    cfg = OrchestratorConfig(
        dataset=DatasetBuildConfig(
            data_dir=str(tmp_path),
            symbols="QQQ.NASDAQ",
            out_dir=str(tmp_path / "skip"),
        ),
        hpo=HPOConfig(),
        teacher=TeacherTrainConfig(enabled=False),
        integration=IntegrationConfig(
            enabled=True,
            run_validators=False,
        ),
    )

    rc = orch.run(cfg)
    assert rc == 0
    assert metrics_called is False
    assert events_called is False


def test_auto_fill_universe_backfills_expected_schemas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    coverage = _Coverage()
    writer = _Writer()
    ingestor = _Ingestor()

    out_dir = tmp_path / "out"

    def _build(argv: list[str] | None = None) -> int:
        if argv is None:
            return 0
        if "--out_dir" in argv:
            target = Path(argv[argv.index("--out_dir") + 1])
            target.mkdir(parents=True, exist_ok=True)
            (target / "dataset.csv").write_text("id,ts_event\n1,1\n", encoding="utf-8")
        return 0

    orch = MLPipelineOrchestrator(
        coverage=coverage,
        writer=writer,
        data_registry=_Registry(),
        ingestor=ingestor,
        build_main=_CliWrapper(_build),
        hpo_main=None,
        teacher_main=_CliWrapper(_ok),
    )

    backfill_calls: list[tuple[str, str, str, int]] = []

    def _fake_backfill(
        self: MLPipelineOrchestrator,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        lookback_days: int,
    ) -> list[tuple[int, int]]:
        backfill_calls.append((dataset_id, schema, instrument_id, lookback_days))
        return [(0, 1)]

    monkeypatch.setattr(MLPipelineOrchestrator, "backfill", _fake_backfill)

    l2_configs: list[object] = []

    def _fake_populate_l2(config: object) -> object:
        l2_configs.append(config)
        return object()

    monkeypatch.setattr(
        "ml.orchestration.pipeline_orchestrator.populate_l2_efficient",
        _fake_populate_l2,
    )

    policy = CoveragePolicy(
        l0_max_lookback_days=14,
        l1_max_lookback_days=7,
        l2_max_lookback_days=3,
        l3_max_lookback_days=2,
    )
    monkeypatch.setattr(
        "ml.orchestration.pipeline_orchestrator.CoveragePolicy.from_env",
        staticmethod(lambda: policy),
    )

    cfg = OrchestratorConfig(
        dataset=DatasetBuildConfig(
            data_dir=str(tmp_path),
            symbols="SPY.NYSE",
            out_dir=str(out_dir),
            include_l2=True,
            instrument_ids=("SPY.NYSE",),
        ),
        hpo=HPOConfig(),
        teacher=TeacherTrainConfig(enabled=False),
        auto_fill=AutoFillUniverseConfig(
            enabled=True,
            include_l2=True,
        ),
    )

    rc = orch.run(cfg)
    assert rc == 0
    schemas = {(schema, lookback) for _, schema, _, lookback in backfill_calls}
    assert ("bars", 14) in schemas
    assert ("tbbo", 7) in schemas
    assert ("trades", 7) in schemas
    assert {instrument for _, _, instrument, _ in backfill_calls} == {"SPY.NYSE"}
    assert l2_configs, "Expected L2 auto-fill to run"
    l2_config = l2_configs[0]
    assert getattr(l2_config, "days") == 3
    assert Path(getattr(l2_config, "data_dir")).samefile(tmp_path)


def test_build_auto_fill_config_from_args_handles_cli(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--auto_fill_universe",
            "--auto_fill_dataset_id",
            "EQUS.PRO",
            "--auto_fill_l2_days",
            "45",
            "--auto_fill_skip_l2",
            "--auto_fill_allow_dataset_l2_ingest",
            "--auto_fill_l2_dataset_id",
            "DBEQ.PRO",
            "--auto_fill_l2_schema",
            "mbp-1",
            "--auto_fill_l2_progress_file",
            str(tmp_path / "progress.json"),
        ],
    )

    dataset_cfg = DatasetBuildConfig(
        data_dir=str(tmp_path),
        symbols="SPY.NYSE",
        out_dir=str(tmp_path / "out"),
    )

    auto_fill_cfg = _build_auto_fill_config_from_args(args, dataset_cfg)
    assert auto_fill_cfg.enabled is True
    assert auto_fill_cfg.dataset_id == "EQUS.PRO"
    assert auto_fill_cfg.l2_days == 45
    assert auto_fill_cfg.include_l2 is False
    assert auto_fill_cfg.disable_dataset_l2_ingest is False
    assert auto_fill_cfg.l2_dataset_id == "DBEQ.PRO"
    assert auto_fill_cfg.l2_schema == "mbp-1"
    assert auto_fill_cfg.l2_progress_file == str(tmp_path / "progress.json")


def test_auto_fill_skips_without_databento(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    coverage = _Coverage()
    writer = _Writer()

    # Ensure environment lacks the Databento API key so ingestion is unavailable
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)

    def _build(argv: list[str] | None = None) -> int:
        if argv is not None and "--out_dir" in argv:
            out_dir = Path(argv[argv.index("--out_dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "dataset.csv").write_text("id,ts_event\n1,1\n", encoding="utf-8")
        return 0

    orch = MLPipelineOrchestrator(
        coverage=coverage,
        writer=writer,
        data_registry=_Registry(),
        ingestor=None,
        build_main=_CliWrapper(_build),
        hpo_main=None,
        teacher_main=_CliWrapper(_ok),
    )

    def _fail_backfill(
        self: MLPipelineOrchestrator,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        lookback_days: int,
    ) -> list[tuple[int, int]]:
        raise AssertionError("backfill should not be invoked when Databento is unavailable")

    monkeypatch.setattr(MLPipelineOrchestrator, "backfill", _fail_backfill)

    cfg = OrchestratorConfig(
        dataset=DatasetBuildConfig(
            data_dir=str(tmp_path),
            symbols="SPY.NYSE",
            out_dir=str(tmp_path / "out"),
        ),
        hpo=HPOConfig(),
        teacher=TeacherTrainConfig(enabled=False),
        auto_fill=AutoFillUniverseConfig(
            enabled=True,
            include_bars=True,
            include_tbbo=True,
            include_trades=True,
            include_l2=False,
        ),
    )

    rc = orch.run(cfg)
    assert rc == 0
