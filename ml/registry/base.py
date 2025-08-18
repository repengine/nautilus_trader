#!/usr/bin/env python3

"""
Abstract base class for model registry implementations.

This module defines the contract that all model registries must follow, ensuring
consistent model lifecycle management across different storage backends.

"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Any


class DeploymentStatus(Enum):
    """
    Model deployment status.
    """

    INACTIVE = "inactive"  # Model registered but not deployed
    ACTIVE = "active"  # Model actively serving predictions
    TESTING = "testing"  # Model in A/B test or shadow mode
    RETIRED = "retired"  # Model retired from production
    FAILED = "failed"  # Model deployment failed


class ModelRole(Enum):
    """
    Model role in the system.
    """

    TEACHER = "teacher"  # Teacher model using rich L2/L3 data
    STUDENT = "student"  # Student model distilled for L1-only inference
    INFERENCE = "inference"  # Direct inference model (no distillation)
    ENSEMBLE = "ensemble"  # Ensemble of multiple models
    FEATURE = "feature"  # Feature engineering model


class DataRequirements(Enum):
    """
    Data requirements for model operation.
    """

    L1_ONLY = "l1_only"  # Only L1 market data (trades/quotes)
    L1_L2 = "l1_l2"  # L1 + L2 (order book)
    L1_L2_L3 = "l1_l2_l3"  # L1 + L2 + L3 (detailed order flow)
    HISTORICAL = "historical"  # Historical data only (no streaming)
    STREAMING = "streaming"  # Real-time streaming data


@dataclass
class ModelManifest:
    """
    Self-describing model manifest with complete metadata.

    Every model carries this manifest to declare its capabilities,
    requirements, and relationships - enabling the registry to
    automatically understand and manage ALL model types.

    Attributes
    ----------
    model_id : str
        Unique identifier for the model
    role : ModelRole
        Model's role in the system (teacher/student/inference/etc)
    data_requirements : DataRequirements
        Data requirements for model operation
    architecture : str
        Model architecture (XGBoost, LightGBM, TFT, etc)
    feature_schema : dict[str, str]
        Feature names and types expected by the model
    feature_schema_hash : str
        Hash of feature schema for validation
    parent_id : Optional[str]
        Parent model ID for lineage tracking (e.g., teacher for a student)
    children_ids : list[str]
        Child model IDs (e.g., students distilled from this teacher)
    training_config : dict[str, Any]
        Configuration used for training
    performance_metrics : dict[str, float]
        Model performance metrics
    deployment_constraints : dict[str, Any]
        Constraints for deployment (latency, memory, etc)
    version : str
        Semantic version of the model
    created_at : float
        Unix timestamp of creation
    last_modified : float
        Unix timestamp of last modification

    """

    model_id: str
    role: ModelRole
    data_requirements: DataRequirements
    architecture: str
    feature_schema: dict[str, str]
    feature_schema_hash: str
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    training_config: dict[str, Any] = field(default_factory=dict)
    performance_metrics: dict[str, float] = field(default_factory=dict)
    deployment_constraints: dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    created_at: float = 0.0
    last_modified: float = 0.0
    # Serving and artifact details
    serveable: bool = True  # True for hot-path models; False for cold-path reference models
    artifact_format: str = "onnx"  # onnx|torchscript|none
    # Linkage to feature registry and pipeline identity
    feature_set_id: str | None = None
    pipeline_signature: str | None = None
    pipeline_version: str | None = None


@dataclass
class ModelInfo:
    """
    Complete model information including manifest and deployment details.

    Attributes
    ----------
    manifest : ModelManifest
        Self-describing model manifest
    model_path : Path
        Path to the model file
    deployment_status : DeploymentStatus
        Current deployment status
    deployed_to : list[str]
        List of deployment targets (actors/strategies)
    performance_history : list[dict[str, Any]]
        Performance metrics over time
    metadata : dict[str, Any]
        Additional metadata not in manifest

    """

    manifest: ModelManifest
    model_path: Path
    deployment_status: DeploymentStatus
    deployed_to: list[str]
    performance_history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelRegistry(ABC):
    """
    Abstract base class for model registry implementations.

    The registry is responsible for:
    - Tracking all trained models
    - Managing model deployments
    - Monitoring model performance
    - Coordinating A/B tests
    - Handling rollbacks

    """

    @abstractmethod
    def register_model(
        self,
        model_path: Path,
        manifest: ModelManifest,
        auto_deploy: bool = False,
        quality_gates: list[Any] | None = None,
        enforce_quality: bool = False,
    ) -> str:
        """
        Register a new model in the registry.

        Parameters
        ----------
        model_path : Path
            Path to the model file
        manifest : ModelManifest
            Self-describing model manifest
        auto_deploy : bool
            Whether to automatically deploy if validation passes
        quality_gates : list[Any] | None
            Quality gates to validate before registration
        enforce_quality : bool
            If True, raise error on quality gate failure

        Returns
        -------
        str
            Unique model ID

        """
        ...

    @abstractmethod
    def deploy_model(
        self,
        model_id: str,
        target: str,
        config: dict[str, Any] | None = None,
    ) -> bool:
        """
        Deploy a model to a target (actor/strategy).

        Parameters
        ----------
        model_id : str
            Model ID to deploy
        target : str
            Deployment target (e.g., "ml_signal_actor")
        config : Optional[dict[str, Any]]
            Deployment configuration

        Returns
        -------
        bool
            True if deployment successful

        """
        ...

    @abstractmethod
    def get_active_models(self) -> list[ModelInfo]:
        """
        Get all currently deployed models.

        Returns
        -------
        list[ModelInfo]
            List of active model information

        """
        ...

    @abstractmethod
    def get_all_models(self) -> list[ModelInfo]:
        """
        Get all registered models.

        Returns
        -------
        list[ModelInfo]
            List of all model information

        """
        ...

    @abstractmethod
    def get_model(self, model_id: str) -> ModelInfo | None:
        """
        Get information about a specific model.

        Parameters
        ----------
        model_id : str
            Model ID to retrieve

        Returns
        -------
        Optional[ModelInfo]
            Model information if found

        """
        ...

    @abstractmethod
    def track_performance(
        self,
        model_id: str,
        metrics: dict[str, Any],
    ) -> None:
        """
        Track model performance metrics.

        Parameters
        ----------
        model_id : str
            Model ID
        metrics : dict[str, Any]
            Performance metrics to track

        """
        ...

    @abstractmethod
    def get_performance_history(
        self,
        model_id: str,
    ) -> list[dict[str, Any]]:
        """
        Get performance history for a model.

        Parameters
        ----------
        model_id : str
            Model ID

        Returns
        -------
        list[dict[str, Any]]
            Performance history

        """
        ...

    @abstractmethod
    def rollback(
        self,
        target: str,
        to_model_id: str,
    ) -> bool:
        """
        Rollback to a previous model version.

        Parameters
        ----------
        target : str
            Deployment target
        to_model_id : str
            Model ID to rollback to

        Returns
        -------
        bool
            True if rollback successful

        """
        ...

    @abstractmethod
    def retire_model(self, model_id: str) -> bool:
        """
        Retire a model from production.

        Parameters
        ----------
        model_id : str
            Model ID to retire

        Returns
        -------
        bool
            True if retirement successful

        """
        ...

    @abstractmethod
    def configure_ab_test(
        self,
        models: list[str],
        split_ratio: float,
        duration_hours: int,
        target: str,
    ) -> dict[str, Any] | None:
        """
        Configure A/B test between models.

        Parameters
        ----------
        models : list[str]
            List of model IDs to test
        split_ratio : float
            Traffic split ratio (0.0 to 1.0)
        duration_hours : int
            Test duration in hours
        target : str
            Deployment target

        Returns
        -------
        Optional[dict[str, Any]]
            A/B test configuration if successful

        """
        ...

    @abstractmethod
    def compare_models(
        self,
        model_ids: list[str],
        metric: str,
    ) -> dict[str, Any] | None:
        """
        Compare performance between models.

        Parameters
        ----------
        model_ids : list[str]
            List of model IDs to compare
        metric : str
            Metric to compare on

        Returns
        -------
        Optional[dict[str, Any]]
            Comparison results

        """
        ...
