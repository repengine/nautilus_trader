#!/usr/bin/env python3
"""
System Validation Smoke Tests for MLPipelineOrchestrator (component-only).
"""

from __future__ import annotations

from unittest.mock import Mock

from ml.orchestration import MLPipelineOrchestrator


def _build_orchestrator() -> MLPipelineOrchestrator:
    mock_coverage = Mock()
    mock_writer = Mock()
    mock_build_main = Mock(return_value=0)
    mock_teacher_main = Mock(return_value=0)
    return MLPipelineOrchestrator(
        coverage=mock_coverage,
        writer=mock_writer,
        build_main=mock_build_main,
        teacher_main=mock_teacher_main,
    )


def test_import_and_initialize() -> None:
    """Import and initialize orchestrator in component mode."""
    orchestrator = _build_orchestrator()
    health = orchestrator.get_health_status()
    assert health["implementation"] == "component-based"


def test_stage_controller_present() -> None:
    """StageController is initialized."""
    orchestrator = _build_orchestrator()
    assert orchestrator._stage_controller is not None
