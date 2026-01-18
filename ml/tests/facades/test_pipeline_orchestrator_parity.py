"""
Contract tests for pipeline orchestrator facade behavior.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from ml.orchestration import MLPipelineOrchestrator
from ml.orchestration.pipeline_orchestrator_facade import MLPipelineOrchestratorFacade


def test_orchestrator_alias_is_facade() -> None:
    """Canonical orchestrator should alias the facade implementation."""
    assert MLPipelineOrchestrator is MLPipelineOrchestratorFacade


def test_get_health_status_reports_component_state() -> None:
    """Health status reports component-based implementation details."""
    orchestrator = MLPipelineOrchestratorFacade(
        coverage=MagicMock(),
        writer=MagicMock(),
        build_main=lambda *_: 0,
        teacher_main=lambda *_: 0,
    )

    status = orchestrator.get_health_status()
    assert status["implementation"] == "component-based"
    assert status["coverage_provider"] == "healthy"
    assert status["writer"] == "healthy"
