from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any

import pytest

from ml.orchestration.scheduler import PipelineScheduleConfig
from ml.orchestration.scheduler import run_pipeline_schedule

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)


def _pipeline_stub(_: Sequence[str] | None = None) -> int:
    return 0


def _sleep_stub(_: float) -> None:
    return None


def test_run_pipeline_schedule_sets_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, Any] = {}

    def _fake_run_forever(
        *,
        config_loader: Any,
        invoke_pipeline: Callable[[Sequence[str] | None], int],
        sleep_fn: Callable[[float], None],
    ) -> None:
        recorded["config_loader"] = config_loader
        recorded["invoke_pipeline"] = invoke_pipeline
        recorded["sleep_fn"] = sleep_fn

    monkeypatch.setattr("ml.orchestration.scheduler.run_forever", _fake_run_forever)

    run_pipeline_schedule(
        PipelineScheduleConfig(
            schedule_time="02:30Z",
            config_path="path/to/config.json",
            dry_run=True,
        ),
        invoke_pipeline=_pipeline_stub,
        sleep_fn=_sleep_stub,
    )

    assert os.environ["ORCH_SCHEDULE_TIME"] == "02:30Z"
    assert os.environ["ORCH_CONFIG"] == "path/to/config.json"
    assert os.environ["ORCH_DRY_RUN"] == "1"
    assert recorded["invoke_pipeline"] is _pipeline_stub


def test_run_pipeline_schedule_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ml.orchestration.scheduler.run_forever", lambda **_: None)

    run_pipeline_schedule(
        PipelineScheduleConfig(interval_minutes=30, force=True),
        invoke_pipeline=_pipeline_stub,
        sleep_fn=_sleep_stub,
    )

    assert os.environ["ORCH_INTERVAL_MIN"] == "30"
    assert os.environ["ORCH_FORCE"] == "1"


def test_run_pipeline_schedule_rejects_both_interval_and_time() -> None:
    with pytest.raises(ValueError):
        run_pipeline_schedule(
            PipelineScheduleConfig(schedule_time="01:00Z", interval_minutes=60),
            invoke_pipeline=_pipeline_stub,
            sleep_fn=_sleep_stub,
        )


def test_task_pipeline_scheduler_shim_module_is_retired() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("ml.tasks.pipelines.scheduler")
