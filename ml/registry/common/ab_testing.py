#!/usr/bin/env python3

"""
ABTestingComponent - Manages A/B testing and statistical model comparison.

This component is extracted from the ModelRegistry god class following the
established TDD decomposition pattern. It handles:
- A/B test configuration and lifecycle management
- Statistical model comparison using Welch's t-test
- Metric tracking and analysis for A/B tests
- Performance ranking and comparison

Thread-safety: All operations use the persistence component's lock for shared state.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from ml.config.registry import RegistryPolicyConfig
from ml.registry.base import DeploymentStatus
from ml.registry.common.model_persistence import ModelPersistenceComponent
from ml.registry.statistics import welch_t_test


logger = logging.getLogger(__name__)


class ABTestingComponent:
    """
    Manages A/B testing and statistical model comparison.

    This component provides functionality for:
    - Configuring A/B tests between exactly 2 models
    - Comparing model performance by metrics
    - Statistical comparison using Welch's t-test
    - Running A/B tests with metric tracking
    - Analyzing A/B test results

    Attributes
    ----------
    _persistence : ModelPersistenceComponent
        The persistence component for model storage operations.
    _policy : RegistryPolicyConfig
        Policy configuration including ab_models_required.
    _ab_test_metrics : dict[str, dict[str, list[float]]]
        In-memory storage for A/B test metrics by test_id and model_id.

    Thread Safety
    -------------
    All operations that access shared state use the persistence component's
    RLock for thread-safe concurrent access.

    Example
    -------
    >>> from ml.registry.common.ab_testing import ABTestingComponent
    >>> from ml.registry.common.model_persistence import ModelPersistenceComponent
    >>> persistence = ModelPersistenceComponent(config, registry_path)
    >>> ab_testing = ABTestingComponent(persistence)
    >>> result = ab_testing.configure_ab_test(
    ...     models=["model_a", "model_b"],
    ...     split_ratio=0.5,
    ...     duration_hours=24,
    ...     target="production",
    ... )
    """

    def __init__(
        self,
        persistence: ModelPersistenceComponent,
        policy_config: RegistryPolicyConfig | None = None,
    ) -> None:
        """
        Initialize A/B testing component.

        Parameters
        ----------
        persistence : ModelPersistenceComponent
            The persistence component for model storage operations.
        policy_config : RegistryPolicyConfig | None
            Policy configuration. If None, uses defaults.

        Example
        -------
        >>> component = ABTestingComponent(persistence)
        >>> component = ABTestingComponent(persistence, RegistryPolicyConfig(ab_models_required=3))
        """
        self._persistence = persistence
        self._policy = policy_config or RegistryPolicyConfig()
        self._ab_test_metrics: dict[str, dict[str, list[float]]] = {}

        logger.debug(
            "Initialized ABTestingComponent with ab_models_required=%d",
            self._policy.ab_models_required,
        )

    def configure_ab_test(
        self,
        models: list[str],
        split_ratio: float,
        duration_hours: int,
        target: str,
    ) -> dict[str, Any] | None:
        """
        Configure A/B test between exactly 2 models.

        This method sets up an A/B test configuration, validates that exactly
        the required number of models (default 2) are provided, verifies all
        models exist in the registry, and sets their deployment status to TESTING.

        Parameters
        ----------
        models : list[str]
            List of model IDs to test (expects exactly ab_models_required models).
        split_ratio : float
            Traffic split ratio for the first model (0.0 to 1.0).
        duration_hours : int
            Test duration in hours.
        target : str
            Deployment target identifier.

        Returns
        -------
        dict[str, Any] | None
            A/B test configuration dict containing model_a, model_b, split_ratio,
            duration_hours, target, start_time, end_time, and status.
            Returns None if validation fails.

        Example
        -------
        >>> result = component.configure_ab_test(
        ...     models=["model_0", "model_1"],
        ...     split_ratio=0.5,
        ...     duration_hours=24,
        ...     target="production",
        ... )
        >>> assert result["model_a"] == "model_0"
        >>> assert result["status"] == "active"
        """
        with self._persistence._lock:
            required = int(self._policy.ab_models_required)
            if len(models) != required:
                logger.error("A/B test requires exactly %d models", required)
                return None

            model_a_id, model_b_id = models

            # Verify both models exist
            model_a = self._persistence.get_model(model_a_id)
            model_b = self._persistence.get_model(model_b_id)

            if model_a is None or model_b is None:
                logger.error("One or more models not found in registry")
                return None

            # Create A/B test config
            ab_config: dict[str, Any] = {
                "model_a": model_a_id,
                "model_b": model_b_id,
                "split_ratio": split_ratio,
                "duration_hours": duration_hours,
                "target": target,
                "start_time": time.time(),
                "end_time": time.time() + (duration_hours * 3600),
                "status": "active",
            }

            # Set both models to TESTING status
            model_a.deployment_status = DeploymentStatus.TESTING
            if target not in model_a.deployed_to:
                model_a.deployed_to.append(target)
            model_a.manifest.last_modified = time.time()

            model_b.deployment_status = DeploymentStatus.TESTING
            if target not in model_b.deployed_to:
                model_b.deployed_to.append(target)
            model_b.manifest.last_modified = time.time()

            # Store A/B test config
            test_id = f"ab_test_{int(time.time())}"
            self._persistence.ab_tests[test_id] = ab_config

            # Update deployments
            self._persistence.deployments[target] = models

            # Persist changes
            self._persistence.save_registry()

            logger.info(
                "Configured A/B test %s for models %s on %s",
                test_id,
                models,
                target,
            )
            return ab_config

    def compare_models(
        self,
        model_ids: list[str],
        metric: str,
    ) -> dict[str, Any] | None:
        """
        Compare performance between models by metric.

        Retrieves the latest metric value from each model's performance_history
        and ranks them in descending order.

        Parameters
        ----------
        model_ids : list[str]
            List of model IDs to compare.
        metric : str
            The metric name to compare on (e.g., "accuracy", "f1_score").

        Returns
        -------
        dict[str, Any] | None
            Comparison results containing:
            - metric: The metric name
            - rankings: List of dicts with model_id, version, and metric value,
              sorted by metric descending
            - best_model: The model_id with the highest metric value
            Returns None if no models have the metric or model_ids is empty.

        Example
        -------
        >>> result = component.compare_models(
        ...     model_ids=["model_0", "model_1", "model_2"],
        ...     metric="accuracy",
        ... )
        >>> assert result["best_model"] == "model_2"
        >>> assert result["rankings"][0]["accuracy"] > result["rankings"][1]["accuracy"]
        """
        with self._persistence._lock:
            if not model_ids:
                return None

            results: list[dict[str, Any]] = []

            for model_id in model_ids:
                model_info = self._persistence.get_model(model_id)
                if model_info is None:
                    logger.warning("Model %s not found, skipping", model_id)
                    continue

                # Get latest metric value from performance_history
                metric_value = None
                for perf in reversed(model_info.performance_history):
                    if metric in perf:
                        metric_value = perf[metric]
                        break

                if metric_value is not None:
                    results.append(
                        {
                            "model_id": model_id,
                            "version": model_info.manifest.version,
                            metric: metric_value,
                        },
                    )

            if not results:
                return None

            # Sort by metric (descending)
            results.sort(key=lambda x: x[metric], reverse=True)

            comparison: dict[str, Any] = {
                "metric": metric,
                "rankings": results,
                "best_model": results[0]["model_id"] if results else None,
            }

            return comparison

    def compare_models_statistically(
        self,
        model_ids: list[str],
        metric: str,
    ) -> dict[str, Any] | None:
        """
        Statistical comparison between exactly 2 models using Welch's t-test.

        Collects all metric samples from each model's performance_history and
        performs Welch's t-test to determine statistical significance.

        Parameters
        ----------
        model_ids : list[str]
            List of exactly 2 model IDs to compare.
        metric : str
            The metric name to compare (e.g., "accuracy").

        Returns
        -------
        dict[str, Any] | None
            Statistical comparison results containing:
            - model_a: First model ID
            - model_b: Second model ID
            - metric: The metric compared
            - t_statistic: The t-test statistic
            - p_value_approx: Approximate p-value
            - statistically_significant: Boolean indicating significance
            - relative_improvement: Percentage improvement of model_b over model_a
            Returns None if not exactly 2 models or insufficient samples.

        Example
        -------
        >>> result = component.compare_models_statistically(
        ...     model_ids=["model_a", "model_b"],
        ...     metric="accuracy",
        ... )
        >>> assert "p_value_approx" in result
        >>> assert "statistically_significant" in result
        """
        with self._persistence._lock:
            if len(model_ids) != 2:
                logger.error("Statistical comparison requires exactly 2 models")
                return None

            model_a_id, model_b_id = model_ids

            # Collect metric samples from performance_history
            samples_a: list[float] = []
            samples_b: list[float] = []

            model_a = self._persistence.get_model(model_a_id)
            if model_a is not None:
                for perf in model_a.performance_history:
                    if metric in perf:
                        samples_a.append(perf[metric])

            model_b = self._persistence.get_model(model_b_id)
            if model_b is not None:
                for perf in model_b.performance_history:
                    if metric in perf:
                        samples_b.append(perf[metric])

            if not samples_a or not samples_b:
                return None

            # Perform Welch's t-test
            test_result = welch_t_test(
                np.array(samples_a, dtype=np.float64),
                np.array(samples_b, dtype=np.float64),
            )

            # Add model identification to results
            test_result["model_a"] = model_a_id
            test_result["model_b"] = model_b_id
            test_result["metric"] = metric

            # Ensure relative_improvement is present (may be missing in error cases)
            if "relative_improvement" not in test_result:
                test_result["relative_improvement"] = 0.0

            return test_result

    def run_ab_test(
        self,
        model_a_id: str,
        model_b_id: str,
        split_ratio: float,
        duration_hours: float,
        target: str,
    ) -> str:
        """
        Start an A/B test between two models.

        Configures the A/B test and initializes metric tracking for both models.

        Parameters
        ----------
        model_a_id : str
            Control model ID.
        model_b_id : str
            Treatment model ID.
        split_ratio : float
            Traffic split ratio for control model (0.0 to 1.0).
        duration_hours : float
            Test duration in hours.
        target : str
            Deployment target identifier.

        Returns
        -------
        str
            The A/B test ID. Returns empty string if configuration fails.

        Example
        -------
        >>> test_id = component.run_ab_test(
        ...     model_a_id="model_0",
        ...     model_b_id="model_1",
        ...     split_ratio=0.5,
        ...     duration_hours=24.0,
        ...     target="production",
        ... )
        >>> assert test_id != ""
        >>> assert test_id in component._ab_test_metrics
        """
        with self._persistence._lock:
            # Use configure_ab_test for initial setup
            config = self.configure_ab_test(
                models=[model_a_id, model_b_id],
                split_ratio=split_ratio,
                duration_hours=int(duration_hours),
                target=target,
            )

            if config:
                test_id = f"ab_test_{int(time.time())}"
                # Initialize metric tracking for both models
                self._ab_test_metrics[test_id] = {
                    model_a_id: [],
                    model_b_id: [],
                }
                return test_id

            return ""

    def track_ab_test_metric(
        self,
        test_id: str,
        model_id: str,
        metric_value: float,
    ) -> None:
        """
        Track a metric value for an A/B test.

        Appends the metric value to the model's list in the A/B test tracking.
        Silently ignores if test_id or model_id is not found.

        Parameters
        ----------
        test_id : str
            The A/B test identifier.
        model_id : str
            The model identifier.
        metric_value : float
            The metric value to track.

        Example
        -------
        >>> test_id = component.run_ab_test(...)
        >>> component.track_ab_test_metric(test_id, "model_0", 0.85)
        >>> component.track_ab_test_metric(test_id, "model_0", 0.86)
        >>> assert len(component._ab_test_metrics[test_id]["model_0"]) == 2
        """
        with self._persistence._lock:
            if test_id in self._ab_test_metrics:
                if model_id in self._ab_test_metrics[test_id]:
                    self._ab_test_metrics[test_id][model_id].append(metric_value)

    def analyze_ab_test(self, test_id: str) -> dict[str, Any] | None:
        """
        Analyze A/B test results with statistical significance.

        Calculates mean values for control and treatment groups and performs
        Welch's t-test to determine statistical significance.

        Parameters
        ----------
        test_id : str
            The A/B test identifier.

        Returns
        -------
        dict[str, Any] | None
            Analysis results containing:
            - test_id: The test identifier
            - control_model: Control model ID
            - treatment_model: Treatment model ID
            - control_mean: Mean of control samples
            - treatment_mean: Mean of treatment samples
            - relative_improvement: Percentage improvement
            - statistical_significance: Boolean indicating significance
            - p_value: Approximate p-value
            Returns None if test not found or insufficient samples.

        Example
        -------
        >>> result = component.analyze_ab_test(test_id)
        >>> assert result["test_id"] == test_id
        >>> assert "control_mean" in result
        >>> assert "statistical_significance" in result
        """
        with self._persistence._lock:
            if test_id not in self._ab_test_metrics:
                return None

            metrics = self._ab_test_metrics[test_id]
            model_ids = list(metrics.keys())

            if len(model_ids) != 2:
                return None

            control_id = model_ids[0]
            treatment_id = model_ids[1]

            control_samples = metrics[control_id]
            treatment_samples = metrics[treatment_id]

            if not control_samples or not treatment_samples:
                return None

            control_mean = float(np.mean(control_samples))
            treatment_mean = float(np.mean(treatment_samples))

            # Perform statistical test
            test_result = welch_t_test(
                np.array(control_samples, dtype=np.float64),
                np.array(treatment_samples, dtype=np.float64),
            )

            return {
                "test_id": test_id,
                "control_model": control_id,
                "treatment_model": treatment_id,
                "control_mean": control_mean,
                "treatment_mean": treatment_mean,
                "relative_improvement": test_result["relative_improvement"],
                "statistical_significance": test_result["statistically_significant"],
                "p_value": test_result["p_value_approx"],
            }
