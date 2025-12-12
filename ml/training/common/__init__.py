"""
Training components for BaseMLTrainer decomposition.

This package contains decomposed components from the BaseMLTrainer class,
following the Universal ML Architecture Pattern 2 (Protocol-First Interface Design).

Components:
    TrainingOrchestratorComponent: Orchestrates the complete training pipeline
    DataPreparationComponent: Handles data preparation and splitting
    CrossValidationComponent: Handles cross-validation strategies (time-series, purged)
    HyperparameterComponent: Handles Optuna hyperparameter optimization
    EvaluationComponent: Handles model evaluation and trading metrics
    PersistenceComponent: Handles model persistence (save/load/ONNX export)
    MLflowTrackingComponent: Handles MLflow experiment tracking (deprecated)

"""

from __future__ import annotations

from ml.training.common.cross_validation import CrossValidationComponent
from ml.training.common.cross_validation import CVTrainerProtocol
from ml.training.common.data_preparation import DataPreparationComponent
from ml.training.common.data_preparation import DataPreparationTrainerProtocol
from ml.training.common.evaluation import EvaluationComponent
from ml.training.common.evaluation import EvaluationTrainerProtocol
from ml.training.common.hyperparameter import HyperparameterComponent
from ml.training.common.hyperparameter import HyperparameterTrainerProtocol
from ml.training.common.mlflow_tracking import MLflowTrackingComponent
from ml.training.common.mlflow_tracking import MLflowTrainerProtocol
from ml.training.common.persistence import PersistenceComponent
from ml.training.common.persistence import PersistenceTrainerProtocol
from ml.training.common.training_orchestrator import TrainingOrchestratorComponent


__all__ = [
    "CVTrainerProtocol",
    "CrossValidationComponent",
    "DataPreparationComponent",
    "DataPreparationTrainerProtocol",
    "EvaluationComponent",
    "EvaluationTrainerProtocol",
    "HyperparameterComponent",
    "HyperparameterTrainerProtocol",
    "MLflowTrackingComponent",
    "MLflowTrainerProtocol",
    "PersistenceComponent",
    "PersistenceTrainerProtocol",
    "TrainingOrchestratorComponent",
]
