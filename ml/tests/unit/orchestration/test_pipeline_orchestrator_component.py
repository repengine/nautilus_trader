from __future__ import annotations

import importlib
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)


@pytest.fixture(name="reload_component")
def _reload_component_fixture(monkeypatch: pytest.MonkeyPatch) -> Any:
    """
    Ensure ``pipeline_orchestrator_component`` is reloaded with desired environment.
    """

    def _loader(env_value: str) -> Any:
        monkeypatch.setenv("ML_USE_LEGACY_PIPELINE_ORCHESTRATOR", env_value)
        module = importlib.import_module("ml.orchestration.pipeline_orchestrator_component")
        return importlib.reload(module)

    yield _loader
    monkeypatch.delenv("ML_USE_LEGACY_PIPELINE_ORCHESTRATOR", raising=False)


def test_component_orchestrator_uses_legacy_when_flag_enabled(
    monkeypatch: pytest.MonkeyPatch,
    reload_component: Any,
) -> None:
    calls: dict[str, Any] = {}

    class _DummyLegacy:
        def __init__(self, **kwargs: Any) -> None:
            calls["init_kwargs"] = kwargs

        def run(self, cfg: Any) -> int:
            calls["run_cfg"] = cfg
            return 0

    component = reload_component("1")
    import ml.orchestration.pipeline_orchestrator as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "MLPipelineOrchestrator",
        _DummyLegacy,
    )
    orchestrator = component.MLPipelineOrchestrator(connection_string="postgresql://example")
    assert orchestrator._use_legacy is True  # type: ignore[attr-defined]
    dummy_cfg = cast("OrchestratorConfig", SimpleNamespace())
    result = orchestrator.run(dummy_cfg)
    assert result == 0
    assert getattr(orchestrator, "_legacy", None) is not None
    assert getattr(orchestrator._legacy, "connection_string") == "postgresql://example"  # type: ignore[attr-defined]
    assert calls["run_cfg"] is dummy_cfg


def test_component_orchestrator_defaults_to_component(monkeypatch: pytest.MonkeyPatch, reload_component: Any) -> None:
    component = reload_component("0")
    orchestrator = component.MLPipelineOrchestrator()
    assert orchestrator._use_legacy is False  # type: ignore[attr-defined]
if TYPE_CHECKING:
    from ml.orchestration.config_types import OrchestratorConfig
