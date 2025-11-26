#!/usr/bin/env python3
# ruff: noqa: E402  # Deprecation warning must execute before imports

"""
A/B testing configuration and statistical analysis (DEPRECATED).

.. deprecated::
    This is the legacy implementation. Use package-level imports instead:

        # Old (deprecated):
        from ml.registry.ab_testing_manager import ABTestingManager

        # New (preferred):
        from ml.registry import ABTestingComponent
        # or for backwards compatibility:
        from ml.registry import ABTestingManager  # alias to ABTestingComponent

    The canonical implementation is :class:`ml.registry.common.ab_testing.ABTestingComponent`
    which provides thread-safety via RLock and uses the persistence component pattern.
"""

from __future__ import annotations

import warnings


warnings.warn(
    "ml.registry.ab_testing_manager is deprecated. "
    "Use 'from ml.registry import ABTestingComponent' instead.",
    DeprecationWarning,
    stacklevel=2,
)

import logging
import time
from typing import Any, Protocol

from ml.registry.base import DeploymentStatus
from ml.registry.base import ModelInfo
from ml.registry.statistics import welch_t_test


logger = logging.getLogger(__name__)


class ABTestingManagerProtocol(Protocol):
    """
    Protocol for A/B testing operations.
    """

    def configure_ab_test(
        self,
        models: list[str],
        split_ratio: float,
        duration_hours: int,
        target: str,
    ) -> dict[str, Any] | None:
        """
        Configure A/B test between models.
        """
        ...

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
        """
        ...

    def track_ab_test_metric(
        self,
        test_id: str,
        model_id: str,
        metric_value: float,
    ) -> None:
        """
        Track metric for A/B test.
        """
        ...

    def analyze_ab_test(self, test_id: str) -> dict[str, Any] | None:
        """
        Analyze A/B test results.
        """
        ...

    def compare_models(
        self,
        model_ids: list[str],
        metric: str,
    ) -> dict[str, Any] | None:
        """
        Compare performance between models.
        """
        ...

    def compare_models_statistically(
        self,
        model_ids: list[str],
        metric: str,
    ) -> dict[str, Any] | None:
        """
        Perform statistical comparison using Welch's t-test.
        """
        ...


class ABTestingManager:
    """
    Manages A/B testing between models.

    Configures A/B tests, tracks metrics, performs statistical analysis
    using Welch's t-test, and provides result analysis.

    Parameters
    ----------
    models : dict[str, ModelInfo]
        Models registry (reference)
    deployments : dict[str, list[str]]
        Deployments registry (reference)
    ab_models_required : int
        Number of models required for A/B test (default: 2)
    save_callback : callable | None
        Callback to save registry state

    """

    def __init__(
        self,
        models: dict[str, ModelInfo],
        deployments: dict[str, list[str]],
        ab_models_required: int = 2,
        save_callback: Any = None,
    ) -> None:
        """
        Initialize A/B testing manager.

        Parameters
        ----------
        models : dict[str, ModelInfo]
            Models registry (reference)
        deployments : dict[str, list[str]]
            Deployments registry (reference)
        ab_models_required : int
            Number of models required for A/B test
        save_callback : callable | None
            Callback to trigger registry save

        """
        self._models = models
        self._deployments = deployments
        self._ab_models_required = ab_models_required
        self._save_callback = save_callback
        self._ab_tests: dict[str, dict[str, Any]] = {}
        self._ab_test_metrics: dict[str, dict[str, list[float]]] = {}

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
            List of model IDs to test (expects 2 models)
        split_ratio : float
            Traffic split ratio for first model
        duration_hours : int
            Test duration in hours
        target : str
            Deployment target

        Returns
        -------
        dict[str, Any] | None
            A/B test configuration

        """
        if len(models) != self._ab_models_required:
            logger.error("A/B test requires exactly %d models", self._ab_models_required)
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
            model_info.manifest.last_modified = time.time()

        # Store A/B test config
        test_id = f"ab_test_{int(time.time())}"
        self._ab_tests[test_id] = ab_config

        # Update deployments to include both models
        self._deployments[target] = models

        # Trigger save
        if self._save_callback:
            self._save_callback()

        logger.info(f"Configured A/B test {test_id} for models {models} on {target}")
        return ab_config

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
        dict[str, Any] | None
            Comparison results

        """
        results = []

        for model_id in model_ids:
            if model_id not in self._models:
                logger.warning("Model %s not found, skipping", model_id)
                continue

            model_info = self._models[model_id]

            # Get latest metric value
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

        comparison = {
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
        Perform statistical comparison between models.

        Parameters
        ----------
        model_ids : list[str]
            Model IDs to compare
        metric : str
            Metric to compare

        Returns
        -------
        dict[str, Any] | None
            Statistical comparison results

        """
        if len(model_ids) != 2:
            logger.error("Statistical comparison requires exactly 2 models")
            return None

        model_a_id, model_b_id = model_ids

        # Collect metric samples
        samples_a = []
        samples_b = []

        if model_a_id in self._models:
            for perf in self._models[model_a_id].performance_history:
                if metric in perf:
                    samples_a.append(perf[metric])

        if model_b_id in self._models:
            for perf in self._models[model_b_id].performance_history:
                if metric in perf:
                    samples_b.append(perf[metric])

        if not samples_a or not samples_b:
            return None

        # Perform Welch's t-test
        import numpy as np

        test_result = welch_t_test(
            np.array(samples_a),
            np.array(samples_b),
        )

        test_result["model_a"] = model_a_id
        test_result["model_b"] = model_b_id
        test_result["metric"] = metric

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

        Parameters
        ----------
        model_a_id : str
            Control model ID
        model_b_id : str
            Treatment model ID
        split_ratio : float
            Traffic split for control (0.0 to 1.0)
        duration_hours : float
            Test duration
        target : str
            Deployment target

        Returns
        -------
        str
            A/B test ID

        """
        # Use existing configure_ab_test
        config = self.configure_ab_test(
            models=[model_a_id, model_b_id],
            split_ratio=split_ratio,
            duration_hours=int(duration_hours),
            target=target,
        )

        if config:
            test_id = f"ab_test_{int(time.time())}"
            # Store A/B test metrics tracking
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
        Track metric for A/B test.

        Parameters
        ----------
        test_id : str
            A/B test ID
        model_id : str
            Model ID
        metric_value : float
            Metric value

        """
        if test_id in self._ab_test_metrics:
            if model_id in self._ab_test_metrics[test_id]:
                self._ab_test_metrics[test_id][model_id].append(metric_value)

    def analyze_ab_test(self, test_id: str) -> dict[str, Any] | None:
        """
        Analyze A/B test results.

        Parameters
        ----------
        test_id : str
            A/B test ID

        Returns
        -------
        dict[str, Any] | None
            Analysis results

        """
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

        import numpy as np

        control_mean = np.mean(control_samples)
        treatment_mean = np.mean(treatment_samples)

        # Perform statistical test
        test_result = welch_t_test(
            np.array(control_samples),
            np.array(treatment_samples),
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
