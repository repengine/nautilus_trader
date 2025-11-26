#!/usr/bin/env python3

"""
DeploymentManagerComponent - Manages model deployment, canary releases, and gradual rollouts.

This component is extracted from the ModelRegistry god class following the
established TDD decomposition pattern. It handles:
- Model deployment to targets
- Rollback operations
- Model retirement
- Hot reload for zero-downtime updates
- Canary deployments with metric tracking
- Gradual rollouts with staged traffic management

Thread-safety: All operations are protected by acquiring the persistence lock.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from ml.registry.base import DeploymentStatus
from ml.registry.dataclasses import CanaryConfig
from ml.registry.dataclasses import CanaryDeployment
from ml.registry.dataclasses import RolloutPlan


if TYPE_CHECKING:
    from ml.registry.common.model_persistence import ModelPersistenceComponent


logger = logging.getLogger(__name__)


class DeploymentManagerComponent:
    """
    Manages model deployment, canary releases, and gradual rollouts.

    This component handles the deployment lifecycle of models including:
    - Basic deployment to targets
    - Rollback to previous versions
    - Model retirement
    - Hot reload for seamless model swapping
    - Canary deployments with metric-based promotion/rollback
    - Gradual rollouts with staged traffic percentages

    Attributes
    ----------
    _persistence : ModelPersistenceComponent
        The persistence component for data access and locking.
    _canary_deployments : dict[str, CanaryDeployment]
        Active canary deployments indexed by deployment ID.
    _rollout_plans : dict[str, RolloutPlan]
        Active rollout plans indexed by rollout ID.

    Thread Safety
    -------------
    All public methods acquire the persistence component's lock for thread-safe
    concurrent access. This ensures consistency with other components sharing
    the same persistence layer.

    Example
    -------
    >>> persistence = ModelPersistenceComponent(config, Path("/tmp/registry"))
    >>> deployment_mgr = DeploymentManagerComponent(persistence)
    >>> deployment_mgr.deploy_model("model_123", "ml_signal_actor")
    True
    """

    def __init__(
        self,
        persistence: ModelPersistenceComponent,
    ) -> None:
        """
        Initialize deployment manager with persistence layer.

        Parameters
        ----------
        persistence : ModelPersistenceComponent
            The persistence component providing model storage and locking.

        Example
        -------
        >>> persistence = ModelPersistenceComponent(config, Path("/tmp/registry"))
        >>> deployment_mgr = DeploymentManagerComponent(persistence)
        """
        self._persistence = persistence
        self._canary_deployments: dict[str, CanaryDeployment] = {}
        self._rollout_plans: dict[str, RolloutPlan] = {}

        logger.debug("Initialized DeploymentManagerComponent")

    # -------------------------------------------------------------------------
    # Basic Deployment Operations
    # -------------------------------------------------------------------------

    def deploy_model(
        self,
        model_id: str,
        target: str,
        config: dict[str, Any] | None = None,
    ) -> bool:
        """
        Deploy a model to a target.

        Parameters
        ----------
        model_id : str
            Model ID to deploy.
        target : str
            Deployment target (e.g., "ml_signal_actor").
        config : dict[str, Any] | None
            Optional deployment configuration to store in metadata.

        Returns
        -------
        bool
            True if deployment successful, False if model not found.

        Example
        -------
        >>> success = deployment_mgr.deploy_model(
        ...     "model_123",
        ...     "ml_signal_actor",
        ...     config={"traffic_percentage": 100.0}
        ... )
        >>> assert success is True
        """
        with self._persistence._lock:
            model_info = self._persistence.get_model(model_id)
            if model_info is None:
                logger.error("Model %s not found in registry", model_id)
                return False

            # Update deployment status
            model_info.deployment_status = DeploymentStatus.ACTIVE
            if target not in model_info.deployed_to:
                model_info.deployed_to.append(target)
            model_info.manifest.last_modified = time.time()

            # Track deployment - replace previous model for this target
            deployments = self._persistence.deployments
            deployments[target] = [model_id]

            # Store deployment config in metadata
            if config:
                model_info.metadata["deployment_config"] = config

            self._persistence.save_registry()

            logger.info("Deployed model %s to %s", model_id, target)
            return True

    def rollback(
        self,
        target: str,
        to_model_id: str,
    ) -> bool:
        """
        Rollback to a previous model version.

        Deactivates any current model deployed to the target and activates
        the rollback model.

        Parameters
        ----------
        target : str
            Deployment target.
        to_model_id : str
            Model ID to rollback to.

        Returns
        -------
        bool
            True if rollback successful, False if rollback model not found.

        Example
        -------
        >>> success = deployment_mgr.rollback("ml_signal_actor", "model_v1")
        >>> assert success is True
        """
        with self._persistence._lock:
            rollback_model = self._persistence.get_model(to_model_id)
            if rollback_model is None:
                logger.error("Model %s not found in registry", to_model_id)
                return False

            # Deactivate current model for target
            deployments = self._persistence.deployments
            if target in deployments:
                for current_model_id in deployments[target]:
                    current_model = self._persistence.get_model(current_model_id)
                    if current_model is not None:
                        current_model.deployment_status = DeploymentStatus.INACTIVE
                        if target in current_model.deployed_to:
                            current_model.deployed_to.remove(target)

            # Activate rollback model
            rollback_model.deployment_status = DeploymentStatus.ACTIVE
            if target not in rollback_model.deployed_to:
                rollback_model.deployed_to.append(target)
            rollback_model.manifest.last_modified = time.time()

            # Update deployments
            deployments[target] = [to_model_id]

            self._persistence.save_registry()

            logger.info("Rolled back %s to model %s", target, to_model_id)
            return True

    def retire_model(self, model_id: str) -> bool:
        """
        Retire a model from production.

        Sets the model status to RETIRED and removes it from all deployment targets.

        Parameters
        ----------
        model_id : str
            Model ID to retire.

        Returns
        -------
        bool
            True if retirement successful, False if model not found.

        Example
        -------
        >>> success = deployment_mgr.retire_model("model_123")
        >>> model_info = persistence.get_model("model_123")
        >>> assert model_info.deployment_status == DeploymentStatus.RETIRED
        """
        with self._persistence._lock:
            model_info = self._persistence.get_model(model_id)
            if model_info is None:
                logger.error("Model %s not found in registry", model_id)
                return False

            model_info.deployment_status = DeploymentStatus.RETIRED
            model_info.manifest.last_modified = time.time()

            # Remove from all deployments
            deployments = self._persistence.deployments
            for target in list(model_info.deployed_to):
                if target in deployments and model_id in deployments[target]:
                    deployments[target].remove(model_id)

            model_info.deployed_to.clear()

            self._persistence.save_registry()

            logger.info("Retired model %s", model_id)
            return True

    def hot_reload_model(
        self,
        target: str,
        new_model_id: str,
    ) -> bool:
        """
        Hot reload a deployment with a new model.

        Atomically swaps the current model for a new one, retiring the old model.
        Logs a warning if feature schemas don't match between models.

        Parameters
        ----------
        target : str
            Deployment target.
        new_model_id : str
            New model to deploy.

        Returns
        -------
        bool
            True if successful, False if new model not found.

        Example
        -------
        >>> success = deployment_mgr.hot_reload_model("ml_signal_actor", "model_v2")
        >>> assert success is True
        """
        with self._persistence._lock:
            new_model = self._persistence.get_model(new_model_id)
            if new_model is None:
                logger.error("Model %s not found", new_model_id)
                return False

            # Find current model for target
            current_model_id: str | None = None
            for m_id in list(self._persistence.models.keys()):
                m_info = self._persistence.get_model(m_id)
                if m_info is not None and target in m_info.deployed_to:
                    if m_info.deployment_status == DeploymentStatus.ACTIVE:
                        current_model_id = m_id
                        break

            if current_model_id is None:
                # No current model, just deploy new one
                # Release lock temporarily for deploy call
                pass

        # Deploy outside the lock since deploy_model acquires it
        if current_model_id is None:
            return self.deploy_model(new_model_id, target)

        # Validate feature compatibility
        with self._persistence._lock:
            current_model = self._persistence.get_model(current_model_id)
            new_model = self._persistence.get_model(new_model_id)

            if current_model is not None and new_model is not None:
                if (
                    current_model.manifest.feature_schema_hash
                    != new_model.manifest.feature_schema_hash
                ):
                    logger.warning(
                        "Feature schema mismatch during hot reload: "
                        "current=%s, new=%s",
                        current_model.manifest.feature_schema_hash,
                        new_model.manifest.feature_schema_hash,
                    )

        # Deploy new model
        success = self.deploy_model(new_model_id, target)

        if success:
            # Retire old model
            self.retire_model(current_model_id)
            logger.info(
                "Hot reloaded %s: %s -> %s",
                target,
                current_model_id,
                new_model_id,
            )

        return success

    # -------------------------------------------------------------------------
    # Canary Deployment Operations
    # -------------------------------------------------------------------------

    def start_canary_deployment(
        self,
        model_id: str,
        target: str,
        config: CanaryConfig,
        baseline_model_id: str | None = None,
    ) -> str:
        """
        Start a canary deployment for a model.

        Creates a canary deployment that routes a percentage of traffic to the
        new model while monitoring metrics for automatic promotion or rollback.

        Parameters
        ----------
        model_id : str
            Model to deploy as canary.
        target : str
            Deployment target.
        config : CanaryConfig
            Canary deployment configuration.
        baseline_model_id : str | None
            Baseline model for comparison. If None, uses current production model.

        Returns
        -------
        str
            Canary deployment ID.

        Raises
        ------
        ValueError
            If model not found.

        Example
        -------
        >>> config = CanaryConfig(traffic_percentage=5.0, min_samples=100)
        >>> deployment_id = deployment_mgr.start_canary_deployment(
        ...     "model_v2",
        ...     "ml_signal_actor",
        ...     config,
        ... )
        """
        with self._persistence._lock:
            model_info = self._persistence.get_model(model_id)
            if model_info is None:
                raise ValueError(f"Model {model_id} not found")

            # Generate deployment ID
            deployment_id = f"canary_{int(time.time())}_{model_id}"

            # Get baseline performance if needed
            baseline_performance: float | None = None
            resolved_baseline_id = baseline_model_id

            if baseline_model_id:
                baseline_model = self._persistence.get_model(baseline_model_id)
                if baseline_model is not None:
                    baseline_performance = baseline_model.manifest.performance_metrics.get(
                        config.success_metric
                    )
            else:
                # Find current production model for target
                for m_id in list(self._persistence.models.keys()):
                    m_info = self._persistence.get_model(m_id)
                    if m_info is not None and target in m_info.deployed_to:
                        if m_info.deployment_status == DeploymentStatus.ACTIVE:
                            resolved_baseline_id = m_id
                            baseline_performance = m_info.manifest.performance_metrics.get(
                                config.success_metric
                            )
                            break

            # Create canary deployment
            canary = CanaryDeployment(
                deployment_id=deployment_id,
                model_id=model_id,
                target=target,
                config=config,
                baseline_model_id=resolved_baseline_id,
                baseline_performance=baseline_performance,
            )

            # Store canary deployment
            self._canary_deployments[deployment_id] = canary

            # Update model status
            model_info.deployment_status = DeploymentStatus.TESTING
            model_info.metadata["canary_deployment"] = deployment_id

            self._persistence.save_registry()
            logger.info(
                "Started canary deployment %s for model %s",
                deployment_id,
                model_id,
            )

            return deployment_id

    def get_canary_deployment(self, deployment_id: str) -> CanaryDeployment | None:
        """
        Get canary deployment by ID.

        Parameters
        ----------
        deployment_id : str
            The canary deployment ID.

        Returns
        -------
        CanaryDeployment | None
            The canary deployment if found, None otherwise.

        Example
        -------
        >>> canary = deployment_mgr.get_canary_deployment("canary_123_model_v2")
        >>> if canary:
        ...     print(canary.status)
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

        Records metric observations for the canary which are used to make
        promotion/rollback decisions.

        Parameters
        ----------
        deployment_id : str
            Canary deployment ID.
        metric_value : float
            Value of the success metric.
        latency_ms : float | None
            Response latency in milliseconds.
        error_occurred : bool
            Whether an error occurred.

        Example
        -------
        >>> deployment_mgr.update_canary_metrics(
        ...     "canary_123_model_v2",
        ...     metric_value=0.95,
        ...     latency_ms=2.5,
        ...     error_occurred=False,
        ... )
        """
        with self._persistence._lock:
            canary = self._canary_deployments.get(deployment_id)
            if canary:
                canary.record_metric(metric_value, latency_ms, error_occurred)

    def evaluate_canary(self, deployment_id: str) -> tuple[bool, str]:
        """
        Evaluate if canary should be promoted.

        Checks if the canary has met the promotion criteria based on the
        configured thresholds and monitoring duration.

        Parameters
        ----------
        deployment_id : str
            Canary deployment ID.

        Returns
        -------
        tuple[bool, str]
            (should_promote, reason) where reason explains the decision.

        Example
        -------
        >>> should_promote, reason = deployment_mgr.evaluate_canary("canary_123")
        >>> if should_promote:
        ...     deployment_mgr.auto_promote_canary("canary_123")
        """
        with self._persistence._lock:
            canary = self._canary_deployments.get(deployment_id)
            if not canary:
                return False, "deployment_not_found"

            return canary.should_promote()

    def evaluate_canary_for_rollback(self, deployment_id: str) -> tuple[bool, str]:
        """
        Evaluate if canary should be rolled back.

        Checks if the canary is performing poorly enough to warrant rollback
        based on error rates and performance degradation.

        Parameters
        ----------
        deployment_id : str
            Canary deployment ID.

        Returns
        -------
        tuple[bool, str]
            (should_rollback, reason) where reason explains the decision.

        Example
        -------
        >>> should_rollback, reason = deployment_mgr.evaluate_canary_for_rollback(
        ...     "canary_123"
        ... )
        >>> if should_rollback:
        ...     # Handle rollback
        ...     pass
        """
        with self._persistence._lock:
            canary = self._canary_deployments.get(deployment_id)
            if not canary:
                return False, "deployment_not_found"

            return canary.should_rollback()

    def auto_promote_canary(self, deployment_id: str) -> bool:
        """
        Automatically promote a canary to full production.

        Deploys the canary model at 100% traffic and retires the baseline model.

        Parameters
        ----------
        deployment_id : str
            Canary deployment ID.

        Returns
        -------
        bool
            True if promotion successful.

        Example
        -------
        >>> success = deployment_mgr.auto_promote_canary("canary_123_model_v2")
        >>> assert success is True
        """
        canary = self._canary_deployments.get(deployment_id)
        if not canary:
            return False

        # Promote model to full deployment
        success = self.deploy_model(
            model_id=canary.model_id,
            target=canary.target,
            config={"traffic_percentage": 100.0},
        )

        if success:
            with self._persistence._lock:
                canary.status = "promoted"

            # Retire baseline if exists
            if canary.baseline_model_id:
                self.retire_model(canary.baseline_model_id)

            logger.info("Promoted canary %s to full production", deployment_id)

        return success

    # -------------------------------------------------------------------------
    # Gradual Rollout Operations
    # -------------------------------------------------------------------------

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

        Creates a rollout plan that incrementally increases traffic to the new
        model through defined stages.

        Parameters
        ----------
        current_model_id : str
            Currently deployed model.
        new_model_id : str
            New model to roll out.
        target : str
            Deployment target.
        stages : list[float]
            Traffic percentages for each stage (e.g., [0.1, 0.25, 0.5, 1.0]).
        stage_duration_minutes : int
            Duration of each stage in minutes.

        Returns
        -------
        str
            Rollout ID.

        Raises
        ------
        ValueError
            If one or both models not found.

        Example
        -------
        >>> rollout_id = deployment_mgr.start_gradual_rollout(
        ...     "model_v1",
        ...     "model_v2",
        ...     "ml_signal_actor",
        ...     stages=[0.1, 0.25, 0.5, 1.0],
        ...     stage_duration_minutes=60,
        ... )
        """
        with self._persistence._lock:
            current_model = self._persistence.get_model(current_model_id)
            new_model = self._persistence.get_model(new_model_id)

            if current_model is None or new_model is None:
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

            # Configure initial traffic split via A/B test mechanism
            # Note: In full facade integration, this would call configure_ab_test
            # For now, we store the split info in the rollout plan
            if stages:
                initial_split = stages[0]
                rollout.stage_results.append({
                    "stage": 0,
                    "traffic_split": initial_split,
                    "started_at": time.time(),
                })

            logger.info("Started gradual rollout %s", rollout_id)
            return rollout_id

    def get_rollout_status(self, rollout_id: str) -> dict[str, Any] | None:
        """
        Get rollout status.

        Parameters
        ----------
        rollout_id : str
            The rollout ID.

        Returns
        -------
        dict[str, Any] | None
            Rollout status dict if found, None otherwise.

        Example
        -------
        >>> status = deployment_mgr.get_rollout_status("rollout_123")
        >>> if status:
        ...     print(f"Stage: {status['current_stage']}")
        """
        with self._persistence._lock:
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

        Moves the rollout to the next traffic percentage stage.

        Parameters
        ----------
        rollout_id : str
            The rollout ID.

        Returns
        -------
        bool
            True if advanced, False if already at final stage or not found.

        Example
        -------
        >>> success = deployment_mgr.advance_rollout_stage("rollout_123")
        >>> if success:
        ...     print("Advanced to next stage")
        """
        with self._persistence._lock:
            rollout = self._rollout_plans.get(rollout_id)
            if not rollout:
                return False

            if rollout.advance_stage():
                # Record stage advancement
                new_split = rollout.get_current_traffic_split()
                rollout.stage_results.append({
                    "stage": rollout.current_stage,
                    "traffic_split": new_split,
                    "started_at": time.time(),
                })

                logger.info(
                    "Advanced rollout %s to stage %d (traffic: %.1f%%)",
                    rollout_id,
                    rollout.current_stage,
                    new_split * 100,
                )
                return True

            return False
