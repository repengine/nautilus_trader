#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from ml.config.events import EventStatus, Source, Stage
from ml.orchestration import config_loader as _cfg
from ml.orchestration.config_loader import OrchestratorRunConfig
from ml.orchestration.config_loader import Stage as OrchestratorStage
from ml.orchestration.config_loader import TrainingStageConfig
from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.config_types import HPOConfig
from ml.orchestration.config_types import IntegrationConfig
from ml.orchestration.config_types import TeacherTrainConfig
from ml.orchestration.scheduler import _EmitEventProtocol as _EmitProto
from ml.orchestration.scheduler import compute_next_run, run_forever
from ml.tests.utils.targets import build_default_target_semantics_payload

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)

def test_compute_next_run_daily_vs_interval() -> None:
    now = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    nr1 = compute_next_run("13:15Z", None, now)
    assert nr1.hour == 13 and nr1.minute == 15 and nr1.date() == now.date()
    nr2 = compute_next_run("11:00Z", None, now)
    assert (nr2 - now) >= timedelta(hours=23, minutes=0)
    nr3 = compute_next_run(None, 10, now)
    assert (nr3 - now) == timedelta(minutes=10)

class _OnceSleeper:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(
        self,
        seconds: float,
    ) -> None:  # pragma: no cover - behavior verified via side effects
        self.calls.append(seconds)
        if len(self.calls) > 1:
            raise RuntimeError("stop")

def _write_cfg(path: Path, out_dir: Path) -> None:
    payload = {
        "dataset": {
            "data_dir": "data/tier1",
            "symbols": "SPY",
            "out_dir": str(out_dir),
            "target_semantics": build_default_target_semantics_payload(),
        },
        "hpo": {"enabled": False},
        "teacher": {"enabled": False},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

def test_lock_behavior_and_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_orchestrator_env: None,
) -> None:
    del isolated_orchestrator_env
    # Prepare config
    cfg_file = tmp_path / "cfg.json"
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_cfg(cfg_file, out_dir)

    # Environment for scheduler
    monkeypatch.setenv("ORCH_INTERVAL_MIN", "1")
    monkeypatch.setenv("ORCH_CONFIG", str(cfg_file))
    monkeypatch.setenv("ORCH_DRY_RUN", "1")  # do not actually invoke

    lock_path = tmp_path / "lock"
    monkeypatch.setenv("ORCH_LOCK_PATH", str(lock_path))

    # Invoke counter (should not be called in dry-run)
    called: dict[str, int] = {"n": 0}

    def _invoke(_: list[str]) -> int:
        called["n"] += 1
        return 0

    sleeper = _OnceSleeper()

    with pytest.raises(RuntimeError):
        run_forever(_cfg, _invoke, sleeper)
    # Dry-run means invoke not called
    assert called["n"] == 0

    # Non-stale lock prevents run
    lock_path.write_text("locked", encoding="utf-8")
    called["n"] = 0
    with pytest.raises(RuntimeError):
        run_forever(_cfg, _invoke, sleeper)
    assert called["n"] == 0

    # Stale lock should be cleared
    os.utime(lock_path, (0, 0))  # epoch mtime
    called["n"] = 0
    monkeypatch.delenv("ORCH_DRY_RUN", raising=False)
    sleeper2 = _OnceSleeper()
    with pytest.raises(RuntimeError):
        run_forever(_cfg, _invoke, sleeper2)
    assert called["n"] == 1

def test_event_emission_success_and_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_orchestrator_env: None,
) -> None:
    del isolated_orchestrator_env
    cfg_file = tmp_path / "cfg.json"
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_cfg(cfg_file, out_dir)

    monkeypatch.setenv("ORCH_INTERVAL_MIN", "1")
    monkeypatch.setenv("ORCH_CONFIG", str(cfg_file))
    monkeypatch.delenv("ORCH_DRY_RUN", raising=False)

    statuses: list[str] = []

    def _emit(
        _registry: Any,
        *,
        dataset_id: str,
        instrument_id: str,
        stage: Stage,
        source: Source,
        run_id: str,
        ts_min: int,
        ts_max: int,
        count: int,
        status: EventStatus,
        error: str | None = None,
        metadata: dict[str, object] | None = None,
        dataset_type: str | None = None,
        component: str | None = None,
    ) -> None:
        _ = (
            dataset_id,
            instrument_id,
            stage,
            source,
            run_id,
            ts_min,
            ts_max,
            count,
            error,
            metadata,
            dataset_type,
            component,
        )
        statuses.append(status.value)

    # Success case
    def _ok(_: list[str]) -> int:  # noqa: ARG001
        return 0

    sl1 = _OnceSleeper()
    with pytest.raises(RuntimeError):
        run_forever(_cfg, _ok, sl1, emit_event=cast(_EmitProto, _emit))
    assert statuses[-1] == "success"

    # Fail case
    def _fail(_: list[str]) -> int:  # noqa: ARG001
        return 2

    sl2 = _OnceSleeper()
    with pytest.raises(RuntimeError):
        run_forever(_cfg, _fail, sl2, emit_event=cast(_EmitProto, _emit))
    assert statuses[-1] == "failed"

def test_skip_if_outputs_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_orchestrator_env: None,
) -> None:
    del isolated_orchestrator_env
    cfg_file = tmp_path / "cfg.json"
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_cfg(cfg_file, out_dir)
    # Pre-create output file to trigger skip
    (out_dir / "dataset.csv").write_text("id,ts\n1,1\n", encoding="utf-8")

    monkeypatch.setenv("ORCH_INTERVAL_MIN", "1")
    monkeypatch.setenv("ORCH_CONFIG", str(cfg_file))
    monkeypatch.delenv("ORCH_FORCE", raising=False)

    called = {"n": 0}

    def _invoke(_: list[str]) -> int:
        called["n"] += 1
        return 0

    sl = _OnceSleeper()
    with pytest.raises(RuntimeError):
        run_forever(_cfg, _invoke, sl)
    assert called["n"] == 0

def test_run_forever_passes_stage_argument(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text("{}", encoding="utf-8")

    run_cfg = OrchestratorRunConfig(
        stage=OrchestratorStage.DATASET,
        dataset=DatasetBuildConfig(
            data_dir="data/tier1",
            symbols="SPY.NYSE",
            out_dir=str(out_dir),
            target_semantics=build_default_target_semantics_payload(),
        ),
        training=TrainingStageConfig(
            teacher=TeacherTrainConfig(enabled=False),
            hpo=HPOConfig(),
        ),
    )

    class _Loader:
        def load_orchestrator_run_config(self, path: str | None) -> OrchestratorRunConfig:
            assert path == str(cfg_path)
            return run_cfg

    class _StubManager:
        def __init__(self, **_: Any) -> None:
            self.data_registry = object()

    monkeypatch.setattr("ml.core.integration.MLIntegrationManager", _StubManager)

    lock_path = tmp_path / "lock"
    monkeypatch.setenv("ORCH_INTERVAL_MIN", "1")
    monkeypatch.setenv("ORCH_CONFIG", str(cfg_path))
    monkeypatch.setenv("ORCH_FORCE", "1")
    monkeypatch.delenv("ORCH_DRY_RUN", raising=False)
    monkeypatch.setenv("ORCH_LOCK_PATH", str(lock_path))

    invoke_args: list[list[str]] = []

    def _invoke(args: list[str]) -> int:
        invoke_args.append(list(args))
        return 0

    sleeper = _OnceSleeper()

    with pytest.raises(RuntimeError):
        run_forever(_Loader(), _invoke, sleeper)

    assert invoke_args, "invoke_pipeline was not called"
    final_args = invoke_args[-1]
    assert final_args[:2] == ["--config", str(cfg_path)]
    assert final_args[-2:] == ["--stage", OrchestratorStage.DATASET.value]
