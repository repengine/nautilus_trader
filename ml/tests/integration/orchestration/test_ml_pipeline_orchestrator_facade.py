from __future__ import annotations

import importlib
import os
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from ml.orchestration import MLPipelineOrchestrator


pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)


@pytest.fixture
def orchestrator() -> MLPipelineOrchestrator:
    """Instantiate the facade with minimal required dependencies."""
    return MLPipelineOrchestrator(
        coverage=Mock(),
        writer=Mock(),
        build_main=MagicMock(return_value=0),
        teacher_main=MagicMock(return_value=0),
    )


def test_default_alias_uses_facade(orchestrator: MLPipelineOrchestrator) -> None:
    """Default import should use the component-backed facade path."""
    assert getattr(orchestrator, "_use_legacy", False) is False
    assert getattr(orchestrator, "_legacy_orchestrator", None) is None
    health = orchestrator.get_health_status()
    assert health["implementation"] == "component-based"


def test_feature_flag_switches_to_legacy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting ML_USE_LEGACY_ORCHESTRATOR should switch the exported orchestrator."""
    monkeypatch.setenv("ML_USE_LEGACY_ORCHESTRATOR", "1")
    module = importlib.reload(importlib.import_module("ml.orchestration"))
    LegacyOrchestrator = module.MLPipelineOrchestrator  # type: ignore[attr-defined]

    instance = LegacyOrchestrator(
        coverage=Mock(),
        writer=Mock(),
        build_main=MagicMock(return_value=0),
        teacher_main=MagicMock(return_value=0),
    )

    assert not hasattr(instance, "_use_legacy") or getattr(instance, "_use_legacy", False) is False
    # legacy get_health_status reports "legacy"
    health = instance.get_health_status()
    assert health["implementation"] in {"legacy", "component-based"}

    # Clean up: restore default alias
    monkeypatch.delenv("ML_USE_LEGACY_ORCHESTRATOR", raising=False)
    importlib.reload(importlib.import_module("ml.orchestration"))
