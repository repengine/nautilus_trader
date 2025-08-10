#!/usr/bin/env python3

"""
Abstract base class for model registry implementations.

This module defines the contract that all model registries must follow,
ensuring consistent model lifecycle management across different storage backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class DeploymentStatus(Enum):
    """Model deployment status."""
    
    INACTIVE = "inactive"       # Model registered but not deployed
    ACTIVE = "active"           # Model actively serving predictions
    TESTING = "testing"         # Model in A/B test or shadow mode
    RETIRED = "retired"         # Model retired from production
    FAILED = "failed"           # Model deployment failed


@dataclass
class ModelInfo:
    """
    Information about a registered model.
    
    Attributes
    ----------
    model_id : str
        Unique identifier for the model
    model_path : Path
        Path to the model file
    version : str
        Model version (semantic versioning)
    metadata : dict[str, Any]
        Model metadata (features, metrics, etc.)
    deployment_status : DeploymentStatus
        Current deployment status
    deployed_to : list[str]
        List of deployment targets (actors/strategies)
    created_at : float
        Unix timestamp of registration
    last_modified : float
        Unix timestamp of last modification
    performance_history : list[dict[str, Any]]
        Performance metrics over time
    """
    
    model_id: str
    model_path: Path
    version: str
    metadata: dict[str, Any]
    deployment_status: DeploymentStatus
    deployed_to: list[str]
    created_at: float
    last_modified: float
    performance_history: list[dict[str, Any]] = field(default_factory=list)


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
        metadata: dict[str, Any],
        version: Optional[str] = None,
    ) -> str:
        """
        Register a new model in the registry.
        
        Parameters
        ----------
        model_path : Path
            Path to the model file
        metadata : dict[str, Any]
            Model metadata (features, training metrics, etc.)
        version : Optional[str]
            Model version (auto-generated if not provided)
            
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
        config: Optional[dict[str, Any]] = None,
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
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
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
    ) -> Optional[dict[str, Any]]:
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
    ) -> Optional[dict[str, Any]]:
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