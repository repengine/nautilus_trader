"""Task wrapper around the pipeline scheduler loop."""

from __future__ import annotations

import os
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass

from ml.orchestration import config_loader as _config_loader
from ml.orchestration.scheduler import run_forever as _run_forever


@dataclass(slots=True, frozen=True)
class PipelineScheduleConfig:
    """Configuration for invoking the pipeline scheduler."""

    schedule_time: str | None = None
    interval_minutes: int | None = None
    config_path: str | None = None
    dry_run: bool = False
    force: bool = False


def run_pipeline_schedule(
    config: PipelineScheduleConfig,
    *,
    invoke_pipeline: Callable[[Sequence[str] | None], int],
    sleep_fn: Callable[[float], None],
) -> None:
    """Run the orchestrator scheduler with explicit configuration."""
    if config.schedule_time and config.interval_minutes is not None:
        raise ValueError("Specify either schedule_time or interval_minutes, not both")

    env_updates: dict[str, str] = {}
    if config.schedule_time:
        env_updates["ORCH_SCHEDULE_TIME"] = config.schedule_time
        os.environ.pop("ORCH_INTERVAL_MIN", None)
    elif config.interval_minutes is not None:
        env_updates["ORCH_INTERVAL_MIN"] = str(int(config.interval_minutes))
        os.environ.pop("ORCH_SCHEDULE_TIME", None)

    if config.config_path:
        env_updates["ORCH_CONFIG"] = config.config_path
    if config.dry_run:
        env_updates["ORCH_DRY_RUN"] = "1"
    else:
        os.environ.pop("ORCH_DRY_RUN", None)
    if config.force:
        env_updates["ORCH_FORCE"] = "1"
    else:
        os.environ.pop("ORCH_FORCE", None)

    for key, value in env_updates.items():
        os.environ[key] = value

    _run_forever(
        config_loader=_config_loader,
        invoke_pipeline=invoke_pipeline,
        sleep_fn=sleep_fn,
    )


__all__ = ["PipelineScheduleConfig", "run_pipeline_schedule"]
