#!/usr/bin/env python3

"""
Model deployment orchestration and management.

This module provides advanced deployment patterns including hot reload,
gradual rollout, and coordinated multi-model deployments.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from ml.registry.base import DeploymentStatus, ModelRegistry


logger = logging.getLogger(__name__)


class ModelDeploymentManager:
    """
    Manages model deployments and orchestrates updates.
    
    This class provides:
    - Hot reload capabilities
    - Gradual rollout strategies
    - Deployment validation
    - Coordinated multi-model deployments
    - Rollback orchestration
    """
    
    def __init__(self, registry: ModelRegistry) -> None:
        """
        Initialize deployment manager.
        
        Parameters
        ----------
        registry : ModelRegistry
            Model registry instance
        """
        self.registry = registry
        self._deployments: dict[str, dict[str, Any]] = {}
        self._deployment_counter = 0
        
        logger.info("Initialized ModelDeploymentManager")
    
    def _generate_deployment_id(self) -> str:
        """Generate unique deployment ID."""
        self._deployment_counter += 1
        return f"deployment_{int(time.time())}_{self._deployment_counter}"
    
    def deploy(
        self,
        model_id: str,
        config: dict[str, Any],
    ) -> Optional[str]:
        """
        Deploy a model with given configuration.
        
        Parameters
        ----------
        model_id : str
            Model ID to deploy
        config : dict[str, Any]
            Deployment configuration including:
            - target: Deployment target (required)
            - instruments: List of instruments
            - max_positions: Maximum positions
            - Other deployment parameters
            
        Returns
        -------
        Optional[str]
            Deployment ID if successful
        """
        # Validate model exists
        model_info = self.registry.get_model(model_id)
        if model_info is None:
            logger.error(f"Model {model_id} not found in registry")
            return None
        
        # Validate configuration
        if "target" not in config:
            logger.error("Deployment config must include 'target'")
            return None
        
        target = config["target"]
        
        # Deploy through registry
        success = self.registry.deploy_model(
            model_id=model_id,
            target=target,
            config=config,
        )
        
        if not success:
            logger.error(f"Failed to deploy model {model_id}")
            return None
        
        # Track deployment
        deployment_id = self._generate_deployment_id()
        self._deployments[deployment_id] = {
            "deployment_id": deployment_id,
            "model_id": model_id,
            "config": config,
            "target": target,
            "created_at": time.time(),
            "is_active": True,
            "hot_reload_history": [],
        }
        
        logger.info(f"Created deployment {deployment_id} for model {model_id}")
        return deployment_id
    
    def get_deployment_status(
        self,
        deployment_id: str,
    ) -> Optional[dict[str, Any]]:
        """
        Get status of a deployment.
        
        Parameters
        ----------
        deployment_id : str
            Deployment ID
            
        Returns
        -------
        Optional[dict[str, Any]]
            Deployment status information
        """
        if deployment_id not in self._deployments:
            logger.warning(f"Deployment {deployment_id} not found")
            return None
        
        deployment = self._deployments[deployment_id].copy()
        
        # Add model status
        model_info = self.registry.get_model(deployment["model_id"])
        if model_info:
            deployment["model_status"] = model_info.deployment_status.value
            deployment["model_version"] = model_info.version
        
        return deployment
    
    def hot_reload(
        self,
        deployment_id: str,
        new_model_id: str,
    ) -> bool:
        """
        Hot reload a deployment with a new model version.
        
        This performs a zero-downtime model update by:
        1. Validating the new model
        2. Deploying it to the same target
        3. Updating deployment tracking
        4. Retiring the old model
        
        Parameters
        ----------
        deployment_id : str
            Deployment ID to update
        new_model_id : str
            New model ID to deploy
            
        Returns
        -------
        bool
            True if hot reload successful
        """
        if deployment_id not in self._deployments:
            logger.error(f"Deployment {deployment_id} not found")
            return False
        
        deployment = self._deployments[deployment_id]
        old_model_id = deployment["model_id"]
        target = deployment["target"]
        config = deployment["config"]
        
        # Validate new model
        new_model_info = self.registry.get_model(new_model_id)
        if new_model_info is None:
            logger.error(f"New model {new_model_id} not found")
            return False
        
        # Validate compatibility (check feature names if available)
        old_model_info = self.registry.get_model(old_model_id)
        if old_model_info and new_model_info:
            old_features = old_model_info.metadata.get("features", [])
            new_features = new_model_info.metadata.get("features", [])
            
            if old_features and new_features and old_features != new_features:
                logger.warning(
                    f"Feature mismatch during hot reload: "
                    f"old={old_features}, new={new_features}"
                )
        
        # Perform hot reload
        logger.info(f"Hot reloading {deployment_id}: {old_model_id} -> {new_model_id}")
        
        # Deploy new model
        success = self.registry.deploy_model(
            model_id=new_model_id,
            target=target,
            config=config,
        )
        
        if not success:
            logger.error(f"Failed to deploy new model {new_model_id}")
            return False
        
        # Retire old model
        self.registry.retire_model(old_model_id)
        
        # Update deployment tracking
        deployment["model_id"] = new_model_id
        deployment["last_updated"] = time.time()
        deployment["hot_reload_history"].append({
            "from_model": old_model_id,
            "to_model": new_model_id,
            "timestamp": time.time(),
        })
        
        logger.info(f"Successfully hot reloaded {deployment_id} to {new_model_id}")
        return True
    
    def gradual_rollout(
        self,
        deployment_id: str,
        new_model_id: str,
        stages: list[float],
        stage_duration_minutes: int,
    ) -> str:
        """
        Perform gradual rollout of a new model.
        
        Parameters
        ----------
        deployment_id : str
            Current deployment ID
        new_model_id : str
            New model to roll out
        stages : list[float]
            Traffic percentages for each stage (e.g., [0.1, 0.25, 0.5, 1.0])
        stage_duration_minutes : int
            Duration of each stage in minutes
            
        Returns
        -------
        str
            Rollout ID for tracking
        """
        if deployment_id not in self._deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment = self._deployments[deployment_id]
        old_model_id = deployment["model_id"]
        target = deployment["target"]
        
        # Create rollout plan
        rollout_id = f"rollout_{int(time.time())}"
        rollout_plan = {
            "rollout_id": rollout_id,
            "deployment_id": deployment_id,
            "old_model": old_model_id,
            "new_model": new_model_id,
            "stages": stages,
            "stage_duration_minutes": stage_duration_minutes,
            "current_stage": 0,
            "started_at": time.time(),
            "status": "planned",
        }
        
        # For each stage, configure A/B test
        for i, traffic_split in enumerate(stages):
            if traffic_split == 1.0:
                # Final stage - full deployment
                logger.info(f"Rollout {rollout_id} stage {i+1}: Full deployment")
                # In production, this would trigger the hot reload
            else:
                # Configure A/B test with traffic split
                logger.info(
                    f"Rollout {rollout_id} stage {i+1}: "
                    f"{traffic_split*100:.0f}% traffic to new model"
                )
                
                self.registry.configure_ab_test(
                    models=[old_model_id, new_model_id],
                    split_ratio=1.0 - traffic_split,  # Ratio for old model
                    duration_hours=int(stage_duration_minutes / 60),
                    target=target,
                )
        
        logger.info(f"Created gradual rollout plan {rollout_id}")
        return rollout_id
    
    def validate_deployment(
        self,
        deployment_id: str,
    ) -> dict[str, Any]:
        """
        Validate a deployment's health and configuration.
        
        Parameters
        ----------
        deployment_id : str
            Deployment ID to validate
            
        Returns
        -------
        dict[str, Any]
            Validation results
        """
        if deployment_id not in self._deployments:
            return {
                "is_valid": False,
                "errors": [f"Deployment {deployment_id} not found"],
            }
        
        deployment = self._deployments[deployment_id]
        model_id = deployment["model_id"]
        
        errors = []
        warnings = []
        
        # Check model exists and is active
        model_info = self.registry.get_model(model_id)
        if model_info is None:
            errors.append(f"Model {model_id} not found in registry")
        elif model_info.deployment_status not in [
            DeploymentStatus.ACTIVE,
            DeploymentStatus.TESTING,
        ]:
            warnings.append(
                f"Model status is {model_info.deployment_status.value}, "
                f"expected ACTIVE or TESTING"
            )
        
        # Check model file exists
        if model_info and not model_info.model_path.exists():
            errors.append(f"Model file not found: {model_info.model_path}")
        
        # Check performance metrics
        if model_info:
            perf_history = self.registry.get_performance_history(model_id)
            if not perf_history:
                warnings.append("No performance metrics tracked")
            elif len(perf_history) < 5:
                warnings.append(f"Limited performance history: {len(perf_history)} entries")
        
        # Check configuration
        config = deployment["config"]
        if "target" not in config:
            errors.append("Missing 'target' in deployment config")
        
        validation_result = {
            "is_valid": len(errors) == 0,
            "deployment_id": deployment_id,
            "model_id": model_id,
            "errors": errors,
            "warnings": warnings,
            "checked_at": time.time(),
        }
        
        return validation_result
    
    def undeploy(
        self,
        deployment_id: str,
    ) -> bool:
        """
        Undeploy a model deployment.
        
        Parameters
        ----------
        deployment_id : str
            Deployment ID to remove
            
        Returns
        -------
        bool
            True if undeployment successful
        """
        if deployment_id not in self._deployments:
            logger.error(f"Deployment {deployment_id} not found")
            return False
        
        deployment = self._deployments[deployment_id]
        model_id = deployment["model_id"]
        
        # Retire the model
        success = self.registry.retire_model(model_id)
        
        if success:
            # Mark deployment as inactive
            deployment["is_active"] = False
            deployment["undeployed_at"] = time.time()
            
            logger.info(f"Undeployed {deployment_id}")
            return True
        
        return False