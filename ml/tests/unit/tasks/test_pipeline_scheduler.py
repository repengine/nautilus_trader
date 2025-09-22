from __future__ import annotations

import os
from typing import Any, Callable, Sequence

import pytest

from ml.tasks.pipelines import PipelineScheduleConfig
from ml.tasks.pipelines import run_pipeline_schedule


def _pipeline_stub(_: Sequence[str] | None = None) -> int:
    return 0


def _sleep_stub(_: float) -> None:
    return None


@pytest.fixture(autouse=True)
def _clear_env() -> None:
    for key in ["ORCH_SCHEDULE_TIME", "ORCH_INTERVAL_MIN", "ORCH_CONFIG", "ORCH_DRY_RUN", "ORCH_FORCE"]:
        os.environ.pop(key, None)


def test_run_pipeline_schedule_sets_environment(monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr("ml.tasks.pipelines.scheduler._run_forever", _fake_run_forever)

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


def test_run_pipeline_schedule_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ml.tasks.pipelines.scheduler._run_forever", lambda **_: None)

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
