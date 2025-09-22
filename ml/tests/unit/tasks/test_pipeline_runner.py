from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ml.tasks.pipelines.runner import PipelineRunConfig
from ml.tasks.pipelines.runner import run_pipeline


def test_run_pipeline_initialises_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    monkeypatch.setattr("ml.tasks.pipelines.runner.load_config", lambda _: {"catalog_path": "./data"})

    class _DummyRunner:
        def __init__(self, config: dict[str, Any], dry_run: bool) -> None:
            captured["config"] = config
            captured["dry_run"] = dry_run

        def setup_ml_system(self) -> Any:
            captured["setup_called"] = True
            return SimpleNamespace(config=SimpleNamespace(symbols=["SPY"]))

    def _fake_execute(runner: Any, mode: str, start: str | None, end: str | None) -> None:
        captured["mode"] = mode
        captured["start"] = start
        captured["end"] = end
        captured["runner"] = runner

    monkeypatch.setattr("ml.tasks.pipelines.runner.MLPipelineRunner", _DummyRunner)
    monkeypatch.setattr("ml.tasks.pipelines.runner._execute_pipeline_mode", _fake_execute)

    run_pipeline(
        PipelineRunConfig(
            mode="daily",
            dry_run=True,
            config_path="config.json",
        ),
    )

    assert captured["config"] == {"catalog_path": "./data"}
    assert captured["dry_run"] is True
    assert captured["setup_called"] is True
    assert captured["mode"] == "daily"
