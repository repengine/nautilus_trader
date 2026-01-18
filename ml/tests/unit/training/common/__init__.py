"""
Unit tests for training components.

This package contains unit tests for the decomposed training components
extracted from BaseMLTrainer.

Test modules:
    test_training_orchestrator: Tests for TrainingOrchestratorComponent
    test_data_preparation: Tests for DataPreparationComponent

"""

from __future__ import annotations

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

__all__ = ("pytest_plugins",)
