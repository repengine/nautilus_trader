#!/usr/bin/env python3

"""
Local file-based model registry implementation.

This module provides a JSON-based registry for environments without
external model registry services like MLflow.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

from ml.registry.base import DeploymentStatus, ModelInfo, ModelRegistry


logger = logging.getLogger(__name__)


class LocalModelRegistry(ModelRegistry):
    """
    Local file-based model registry using JSON for persistence.
    
    This registry stores all model information in a local JSON file,
    providing a lightweight solution for model lifecycle management
    without external dependencies.
    
    Thread-safe for concurrent operations.
    """
    
    def __init__(self, registry_path: Path) -> None:
        """
        Initialize local model registry.
        
        Parameters
        ----------
        registry_path : Path
            Directory path for registry storage
        """
        self.registry_path = registry_path
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.registry_file = self.registry_path / "registry.json"
        self._lock = threading.Lock()
        
        # Initialize or load registry
        self._load_registry()
        
        logger.info(f"Initialized LocalModelRegistry at {registry_path}")
    
    def _load_registry(self) -> None:
        """Load registry from disk or create new one."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
                self._models: dict[str, ModelInfo] = {
                    model_id: self._dict_to_model_info(model_data)
                    for model_id, model_data in data.get("models", {}).items()
                }
                self._ab_tests: dict[str, dict[str, Any]] = data.get("ab_tests", {})
                self._deployments: dict[str, list[str]] = data.get("deployments", {})
        else:
            self._models = {}
            self._ab_tests = {}
            self._deployments = {}  # target -> model_ids
            self._save_registry()
    
    def _save_registry(self) -> None:
        """Save registry to disk."""
        data = {
            "models": {
                model_id: self._model_info_to_dict(model_info)
                for model_id, model_info in self._models.items()
            },
            "ab_tests": self._ab_tests,
            "deployments": self._deployments,
            "last_updated": time.time(),
        }
        
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _model_info_to_dict(self, model_info: ModelInfo) -> dict[str, Any]:
        """Convert ModelInfo to dictionary for JSON serialization."""
        return {
            "model_id": model_info.model_id,
            "model_path": str(model_info.model_path),
            "version": model_info.version,
            "metadata": model_info.metadata,
            "deployment_status": model_info.deployment_status.value,
            "deployed_to": model_info.deployed_to,
            "created_at": model_info.created_at,
            "last_modified": model_info.last_modified,
            "performance_history": model_info.performance_history,
        }
    
    def _dict_to_model_info(self, data: dict[str, Any]) -> ModelInfo:
        """Convert dictionary to ModelInfo."""
        return ModelInfo(
            model_id=data["model_id"],
            model_path=Path(data["model_path"]),
            version=data["version"],
            metadata=data["metadata"],
            deployment_status=DeploymentStatus(data["deployment_status"]),
            deployed_to=data["deployed_to"],
            created_at=data["created_at"],
            last_modified=data["last_modified"],
            performance_history=data.get("performance_history", []),
        )
    
    def _generate_model_id(self) -> str:
        """Generate unique model ID."""
        timestamp = int(time.time() * 1000000)  # Microsecond precision
        return f"model_{timestamp}"
    
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
            Model metadata
        version : Optional[str]
            Model version (auto-generated if not provided)
            
        Returns
        -------
        str
            Unique model ID
        """
        with self._lock:
            # Validate model file exists
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Generate ID and version
            model_id = self._generate_model_id()
            if version is None:
                # Auto-generate version based on existing models
                existing_versions = [
                    m.version for m in self._models.values()
                    if m.model_path.stem == model_path.stem
                ]
                if existing_versions:
                    latest = max(existing_versions)
                    major, minor, patch = latest.split('.')
                    version = f"{major}.{minor}.{int(patch) + 1}"
                else:
                    version = "1.0.0"
            
            # Create model info
            model_info = ModelInfo(
                model_id=model_id,
                model_path=model_path,
                version=version,
                metadata=metadata,
                deployment_status=DeploymentStatus.INACTIVE,
                deployed_to=[],
                created_at=time.time(),
                last_modified=time.time(),
                performance_history=[],
            )
            
            # Store and save
            self._models[model_id] = model_info
            self._save_registry()
            
            logger.info(f"Registered model {model_id} (version {version}) at {model_path}")
            return model_id
    
    def deploy_model(
        self,
        model_id: str,
        target: str,
        config: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        Deploy a model to a target.
        
        Parameters
        ----------
        model_id : str
            Model ID to deploy
        target : str
            Deployment target
        config : Optional[dict[str, Any]]
            Deployment configuration
            
        Returns
        -------
        bool
            True if deployment successful
        """
        with self._lock:
            if model_id not in self._models:
                logger.error(f"Model {model_id} not found in registry")
                return False
            
            model_info = self._models[model_id]
            
            # Update deployment status
            model_info.deployment_status = DeploymentStatus.ACTIVE
            model_info.deployed_to.append(target)
            model_info.last_modified = time.time()
            
            # Track deployment
            if target not in self._deployments:
                self._deployments[target] = []
            
            # Remove previous model for this target if exists
            self._deployments[target] = [model_id]
            
            # Store deployment config in metadata
            if config:
                model_info.metadata["deployment_config"] = config
            
            self._save_registry()
            
            logger.info(f"Deployed model {model_id} to {target}")
            return True
    
    def get_active_models(self) -> list[ModelInfo]:
        """Get all currently deployed models."""
        with self._lock:
            return [
                model_info for model_info in self._models.values()
                if model_info.deployment_status == DeploymentStatus.ACTIVE
            ]
    
    def get_all_models(self) -> list[ModelInfo]:
        """Get all registered models."""
        with self._lock:
            return list(self._models.values())
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        with self._lock:
            return self._models.get(model_id)
    
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
            Performance metrics
        """
        with self._lock:
            if model_id not in self._models:
                logger.error(f"Model {model_id} not found in registry")
                return
            
            # Add timestamp if not present
            if "timestamp" not in metrics:
                metrics["timestamp"] = time.time()
            
            # Append to history
            self._models[model_id].performance_history.append(metrics)
            self._models[model_id].last_modified = time.time()
            
            self._save_registry()
            
            logger.debug(f"Tracked performance for model {model_id}: {metrics}")
    
    def get_performance_history(
        self,
        model_id: str,
    ) -> list[dict[str, Any]]:
        """Get performance history for a model."""
        with self._lock:
            if model_id not in self._models:
                return []
            return self._models[model_id].performance_history.copy()
    
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
        with self._lock:
            if to_model_id not in self._models:
                logger.error(f"Model {to_model_id} not found in registry")
                return False
            
            # Deactivate current model for target
            if target in self._deployments:
                for current_model_id in self._deployments[target]:
                    if current_model_id in self._models:
                        current_model = self._models[current_model_id]
                        current_model.deployment_status = DeploymentStatus.INACTIVE
                        if target in current_model.deployed_to:
                            current_model.deployed_to.remove(target)
            
            # Activate rollback model
            rollback_model = self._models[to_model_id]
            rollback_model.deployment_status = DeploymentStatus.ACTIVE
            if target not in rollback_model.deployed_to:
                rollback_model.deployed_to.append(target)
            rollback_model.last_modified = time.time()
            
            # Update deployments
            self._deployments[target] = [to_model_id]
            
            self._save_registry()
            
            logger.info(f"Rolled back {target} to model {to_model_id}")
            return True
    
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
        with self._lock:
            if model_id not in self._models:
                logger.error(f"Model {model_id} not found in registry")
                return False
            
            model_info = self._models[model_id]
            model_info.deployment_status = DeploymentStatus.RETIRED
            model_info.last_modified = time.time()
            
            # Remove from all deployments
            for target in list(model_info.deployed_to):
                if target in self._deployments and model_id in self._deployments[target]:
                    self._deployments[target].remove(model_id)
            
            model_info.deployed_to.clear()
            
            self._save_registry()
            
            logger.info(f"Retired model {model_id}")
            return True
    
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
            List of model IDs to test (expects 2 models)
        split_ratio : float
            Traffic split ratio for first model
        duration_hours : int
            Test duration in hours
        target : str
            Deployment target
            
        Returns
        -------
        Optional[dict[str, Any]]
            A/B test configuration
        """
        with self._lock:
            if len(models) != 2:
                logger.error("A/B test requires exactly 2 models")
                return None
            
            model_a_id, model_b_id = models
            
            if model_a_id not in self._models or model_b_id not in self._models:
                logger.error("One or more models not found in registry")
                return None
            
            # Create A/B test config
            ab_config = {
                "model_a": model_a_id,
                "model_b": model_b_id,
                "split_ratio": split_ratio,
                "duration_hours": duration_hours,
                "target": target,
                "start_time": time.time(),
                "end_time": time.time() + (duration_hours * 3600),
                "status": "active",
            }
            
            # Deploy both models
            for model_id in models:
                model_info = self._models[model_id]
                model_info.deployment_status = DeploymentStatus.TESTING
                if target not in model_info.deployed_to:
                    model_info.deployed_to.append(target)
                model_info.last_modified = time.time()
            
            # Store A/B test config
            test_id = f"ab_test_{int(time.time())}"
            self._ab_tests[test_id] = ab_config
            
            # Update deployments to include both models
            self._deployments[target] = models
            
            self._save_registry()
            
            logger.info(f"Configured A/B test {test_id} for models {models} on {target}")
            return ab_config
    
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
        with self._lock:
            results = []
            
            for model_id in model_ids:
                if model_id not in self._models:
                    logger.warning(f"Model {model_id} not found, skipping")
                    continue
                
                model_info = self._models[model_id]
                
                # Get latest metric value
                metric_value = None
                for perf in reversed(model_info.performance_history):
                    if metric in perf:
                        metric_value = perf[metric]
                        break
                
                if metric_value is not None:
                    results.append({
                        "model_id": model_id,
                        "version": model_info.version,
                        metric: metric_value,
                    })
            
            if not results:
                return None
            
            # Sort by metric (descending)
            results.sort(key=lambda x: x[metric], reverse=True)
            
            comparison = {
                "metric": metric,
                "rankings": results,
                "best_model": results[0]["model_id"] if results else None,
            }
            
            return comparison