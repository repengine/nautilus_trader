from __future__ import annotations

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
    health = orchestrator.get_health_status()
    assert health["implementation"] == "component-based"
