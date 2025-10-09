#!/usr/bin/env python3

"""Canary deployment and gradual rollout management."""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from ml.registry.base import DeploymentStatus
from ml.registry.base import ModelInfo
from ml.registry.dataclasses import CanaryConfig
from ml.registry.dataclasses import CanaryDeployment
from ml.registry.dataclasses import RolloutPlan


logger = logging.getLogger(__name__)


class CanaryDeploymentManagerProtocol(Protocol):
    """Protocol for canary deployment operations."""

    def start_canary_deployment(
        self,
        model_id: str,
        target: str,
        config: CanaryConfig,
        baseline_model_id: str | None = None,
    ) -> str:
        """Start a canary deployment for a model."""
        ...

    def get_canary_deployment(self, deployment_id: str) -> CanaryDeployment | None:
        """Get canary deployment by ID."""
        ...

    def update_canary_metrics(
        self,
        deployment_id: str,
        metric_value: float,
        latency_ms: float | None = None,
        error_occurred: bool = False,
    ) -> None:
        """Update metrics for a canary deployment."""
        ...

    def evaluate_canary(self, deployment_id: str) -> tuple[bool, str]:
        """Evaluate if canary should be promoted."""
        ...

    def evaluate_canary_for_rollback(self, deployment_id: str) -> tuple[bool, str]:
        """Evaluate if canary should be rolled back."""
        ...

    def auto_promote_canary(self, deployment_id: str) -> bool:
        """Automatically promote a canary to full production."""
        ...

    def start_gradual_rollout(
        self,
        current_model_id: str,
        new_model_id: str,
        target: str,
        stages: list[float],
        stage_duration_minutes: int,
    ) -> str:
        """Start gradual rollout of a new model."""
        ...

    def get_rollout_status(self, rollout_id: str) -> dict[str, Any] | None:
        """Get rollout status."""
        ...

    def advance_rollout_stage(self, rollout_id: str) -> bool:
        """Advance to next rollout stage."""
        ...


class CanaryDeploymentManager:
    """
    Manages canary deployments and gradual rollouts.

    Handles canary deployment lifecycle, metric tracking, automatic
    promotion/rollback, and multi-stage gradual rollouts.

    Parameters
    ----------
    models : dict[str, ModelInfo]
        Models registry (reference)
    ab_testing_manager : ABTestingManager
        A/B testing manager for traffic splitting
    deploy_callback : callable | None
        Callback to deploy a model
    retire_callback : callable | None
        Callback to retire a model
    save_callback : callable | None
        Callback to save registry state

    """

    def __init__(
        self,
        models: dict[str, ModelInfo],
        ab_testing_manager: Any,
        deploy_callback: Any = None,
        retire_callback: Any = None,
        save_callback: Any = None,
    ) -> None:
        """
        Initialize canary deployment manager.

        Parameters
        ----------
        models : dict[str, ModelInfo]
            Models registry (reference)
        ab_testing_manager : ABTestingManager
            A/B testing manager for traffic splitting
        deploy_callback : callable | None
            Callback to deploy a model
        retire_callback : callable | None
            Callback to retire a model
        save_callback : callable | None
            Callback to trigger registry save

        """
        self._models = models
        self._ab_testing_mgr = ab_testing_manager
        self._deploy_callback = deploy_callback
        self._retire_callback = retire_callback
        self._save_callback = save_callback
        self._canary_deployments: dict[str, CanaryDeployment] = {}
        self._rollout_plans: dict[str, RolloutPlan] = {}

    def start_canary_deployment(
        self,
        model_id: str,
        target: str,
        config: CanaryConfig,
        baseline_model_id: str | None = None,
    ) -> str:
        """
        Start a canary deployment for a model.

        Parameters
        ----------
        model_id : str
            Model to deploy as canary
        target : str
            Deployment target
        config : CanaryConfig
            Canary configuration
        baseline_model_id : str | None
            Baseline model for comparison (current prod if None)

        Returns
        -------
        str
            Canary deployment ID

        """
        if model_id not in self._models:
            raise ValueError(f"Model {model_id} not found")

        # Generate deployment ID
        deployment_id = f"canary_{int(time.time())}_{model_id}"

        # Get baseline performance if needed
        baseline_performance = None
        if baseline_model_id:
            if baseline_model_id in self._models:
                baseline_metrics = self._models[baseline_model_id].manifest.performance_metrics
                baseline_performance = baseline_metrics.get(config.success_metric)
        else:
            # Find current production model for target
            for m_id, m_info in self._models.items():
                if (
                    target in m_info.deployed_to
                    and m_info.deployment_status == DeploymentStatus.ACTIVE
                ):
                    baseline_model_id = m_id
                    baseline_performance = m_info.manifest.performance_metrics.get(
                        config.success_metric,
                    )
                    break

        # Create canary deployment
        canary = CanaryDeployment(
            deployment_id=deployment_id,
            model_id=model_id,
            target=target,
            config=config,
            baseline_model_id=baseline_model_id,
            baseline_performance=baseline_performance,
        )

        # Store canary deployment
        self._canary_deployments[deployment_id] = canary

        # Update model status
        model_info = self._models[model_id]
        model_info.deployment_status = DeploymentStatus.TESTING
        model_info.metadata["canary_deployment"] = deployment_id

        # Trigger save
        if self._save_callback:
            self._save_callback()

        logger.info(f"Started canary deployment {deployment_id} for model {model_id}")

        return deployment_id

    def get_canary_deployment(self, deployment_id: str) -> CanaryDeployment | None:
        """
        Get canary deployment by ID.

        Parameters
        ----------
        deployment_id : str
            Canary deployment ID

        Returns
        -------
        CanaryDeployment | None
            Canary deployment or None if not found

        """
        return self._canary_deployments.get(deployment_id)

    def update_canary_metrics(
        self,
        deployment_id: str,
        metric_value: float,
        latency_ms: float | None = None,
        error_occurred: bool = False,
    ) -> None:
        """
        Update metrics for a canary deployment.

        Parameters
        ----------
        deployment_id : str
            Canary deployment ID
        metric_value : float
            Value of the success metric
        latency_ms : float | None
            Response latency
        error_occurred : bool
            Whether an error occurred

        """
        canary = self._canary_deployments.get(deployment_id)
        if canary:
            canary.record_metric(metric_value, latency_ms, error_occurred)

    def evaluate_canary(self, deployment_id: str) -> tuple[bool, str]:
        """
        Evaluate if canary should be promoted.

        Parameters
        ----------
        deployment_id : str
            Canary deployment ID

        Returns
        -------
        tuple[bool, str]
            (should_promote, reason)

        """
        canary = self._canary_deployments.get(deployment_id)
        if not canary:
            return False, "deployment_not_found"

        return canary.should_promote()

    def evaluate_canary_for_rollback(self, deployment_id: str) -> tuple[bool, str]:
        """
        Evaluate if canary should be rolled back.

        Parameters
        ----------
        deployment_id : str
            Canary deployment ID

        Returns
        -------
        tuple[bool, str]
            (should_rollback, reason)

        """
        canary = self._canary_deployments.get(deployment_id)
        if not canary:
            return False, "deployment_not_found"

        return canary.should_rollback()

    def auto_promote_canary(self, deployment_id: str) -> bool:
        """
        Automatically promote a canary to full production.

        Parameters
        ----------
        deployment_id : str
            Canary deployment ID

        Returns
        -------
        bool
            True if promotion successful

        """
        canary = self._canary_deployments.get(deployment_id)
        if not canary:
            return False

        # Promote model to full deployment
        success = False
        if self._deploy_callback:
            success = self._deploy_callback(
                model_id=canary.model_id,
                target=canary.target,
                config={"traffic_percentage": 100.0},
            )

        if success:
            canary.status = "promoted"
            # Retire baseline if exists
            if canary.baseline_model_id and self._retire_callback:
                self._retire_callback(canary.baseline_model_id)

            logger.info(f"Promoted canary {deployment_id} to full production")

        return success

    def start_gradual_rollout(
        self,
        current_model_id: str,
        new_model_id: str,
        target: str,
        stages: list[float],
        stage_duration_minutes: int,
    ) -> str:
        """
        Start gradual rollout of a new model.

        Parameters
        ----------
        current_model_id : str
            Currently deployed model
        new_model_id : str
            New model to roll out
        target : str
            Deployment target
        stages : list[float]
            Traffic percentages for each stage
        stage_duration_minutes : int
            Duration of each stage

        Returns
        -------
        str
            Rollout ID

        """
        if current_model_id not in self._models or new_model_id not in self._models:
            raise ValueError("One or both models not found")

        rollout_id = f"rollout_{int(time.time())}"

        # Create rollout plan
        rollout = RolloutPlan(
            rollout_id=rollout_id,
            current_model_id=current_model_id,
            new_model_id=new_model_id,
            target=target,
            stages=stages,
            stage_duration_minutes=stage_duration_minutes,
        )

        # Store rollout plan
        self._rollout_plans[rollout_id] = rollout

        # Start first stage
        if stages and self._ab_testing_mgr:
            self._ab_testing_mgr.configure_ab_test(
                models=[current_model_id, new_model_id],
                split_ratio=1.0 - stages[0],
                duration_hours=max(
                    1,
                    stage_duration_minutes * 60,
                ),  # Convert to hours (stage_duration is already in minutes * 60)
                target=target,
            )

        logger.info(f"Started gradual rollout {rollout_id}")
        return rollout_id

    def get_rollout_status(self, rollout_id: str) -> dict[str, Any] | None:
        """
        Get rollout status.

        Parameters
        ----------
        rollout_id : str
            Rollout ID

        Returns
        -------
        dict[str, Any] | None
            Rollout status

        """
        rollout = self._rollout_plans.get(rollout_id)
        if not rollout:
            return None

        return {
            "rollout_id": rollout.rollout_id,
            "current_stage": rollout.current_stage,
            "stages": rollout.stages,
            "traffic_split": rollout.get_current_traffic_split(),
            "status": rollout.status,
        }

    def advance_rollout_stage(self, rollout_id: str) -> bool:
        """
        Advance to next rollout stage.

        Parameters
        ----------
        rollout_id : str
            Rollout ID

        Returns
        -------
        bool
            True if advanced successfully

        """
        rollout = self._rollout_plans.get(rollout_id)
        if not rollout:
            return False

        if rollout.advance_stage():
            # Configure next stage
            new_split = rollout.get_current_traffic_split()
            if self._ab_testing_mgr:
                self._ab_testing_mgr.configure_ab_test(
                    models=[rollout.current_model_id, rollout.new_model_id],
                    split_ratio=1.0 - new_split,
                    duration_hours=max(
                        1,
                        int(rollout.stage_duration_minutes / 60),
                    ),  # Convert minutes to hours, min 1
                    target=rollout.target,
                )
            logger.info(f"Advanced rollout {rollout_id} to stage {rollout.current_stage}")
            return True

        return False
