# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Advanced model registry management with A/B testing and deployment capabilities.

This module extends the MLflowManager with advanced model registry operations including
A/B testing framework, canary deployments, rollback capabilities, and performance
tracking.

"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from ml.tracking.mlflow_manager import MLflowManager
from ml.tracking.mlflow_manager import ModelStage


if TYPE_CHECKING:
    from ml.config.shared import MLflowConfig

# Configure module logger
logger = logging.getLogger(__name__)


class ModelRegistry(MLflowManager):
    """
    Advanced model registry with A/B testing and deployment management.

    This class extends MLflowManager with sophisticated model deployment patterns
    including A/B testing, canary deployments, automated rollbacks, and performance
    tracking for production model management.

    Features:
    - A/B testing framework with traffic splitting
    - Canary deployment management
    - Automated rollback based on performance metrics
    - Model performance tracking and comparison
    - Shadow mode testing
    - Gradual rollout strategies
    - Model validation gates

    Parameters
    ----------
    config : MLflowConfig
        Configuration for MLflow tracking and registry operations.

    """

    def __init__(self, config: MLflowConfig) -> None:
        """
        Initialize model registry with advanced capabilities.

        Parameters
        ----------
        config : MLflowConfig
            MLflow configuration settings.

        """
        super().__init__(config)
        self._ab_tests: dict[str, dict[str, Any]] = {}
        self._canary_deployments: dict[str, dict[str, Any]] = {}
        self._performance_history: dict[str, list[dict[str, Any]]] = {}

    def setup_ab_test(
        self,
        test_name: str,
        model_a_name: str,
        model_b_name: str,
        model_a_version: str | None = None,
        model_b_version: str | None = None,
        traffic_split: float = 0.5,
        success_metric: str = "accuracy",
        min_samples: int = 1000,
        significance_level: float = 0.05,
        max_duration_hours: int = 168,  # 1 week
    ) -> str:
        """
        Set up A/B test between two model versions.

        Parameters
        ----------
        test_name : str
            Unique identifier for the A/B test.
        model_a_name : str
            Name of the first model (control).
        model_b_name : str
            Name of the second model (treatment).
        model_a_version : str | None, optional
            Specific version of model A. Uses latest production if None.
        model_b_version : str | None, optional
            Specific version of model B. Uses latest staging if None.
        traffic_split : float, default 0.5
            Fraction of traffic to send to model B (0.0 to 1.0).
        success_metric : str, default "accuracy"
            Metric to optimize for during the test.
        min_samples : int, default 1000
            Minimum samples required before test can conclude.
        significance_level : float, default 0.05
            Statistical significance level for test conclusion.
        max_duration_hours : int, default 168
            Maximum test duration in hours.

        Returns
        -------
        str
            A/B test ID for tracking.

        """
        self._ensure_initialized()

        # Get model versions if not specified
        if model_a_version is None:
            try:
                versions_a = self._client.get_latest_versions(
                    model_a_name,
                    stages=[ModelStage.PRODUCTION.value],
                )
                model_a_version = versions_a[0].version if versions_a else None
            except Exception as e:
                raise ValueError(f"Could not get production version for {model_a_name}: {e}")

        if model_b_version is None:
            try:
                versions_b = self._client.get_latest_versions(
                    model_b_name,
                    stages=[ModelStage.STAGING.value],
                )
                model_b_version = versions_b[0].version if versions_b else None
            except Exception as e:
                raise ValueError(f"Could not get staging version for {model_b_name}: {e}")

        if not model_a_version or not model_b_version:
            raise ValueError("Both model versions must be specified or available")

        # Create test configuration
        test_id = f"ab_test_{test_name}_{int(time.time())}"

        test_config = {
            "test_id": test_id,
            "test_name": test_name,
            "status": "active",
            "created_at": time.time(),
            "model_a": {
                "name": model_a_name,
                "version": model_a_version,
                "role": "control",
            },
            "model_b": {
                "name": model_b_name,
                "version": model_b_version,
                "role": "treatment",
            },
            "config": {
                "traffic_split": traffic_split,
                "success_metric": success_metric,
                "min_samples": min_samples,
                "significance_level": significance_level,
                "max_duration_hours": max_duration_hours,
            },
            "results": {
                "model_a_samples": 0,
                "model_b_samples": 0,
                "model_a_metric_sum": 0.0,
                "model_b_metric_sum": 0.0,
                "model_a_metric_values": [],
                "model_b_metric_values": [],
            },
        }

        # Store test configuration
        self._ab_tests[test_id] = test_config

        # Tag models with A/B test info
        try:
            self._client.set_model_version_tag(
                model_a_name,
                model_a_version,
                "ab_test_id",
                test_id,
            )
            self._client.set_model_version_tag(
                model_a_name,
                model_a_version,
                "ab_test_role",
                "control",
            )
            self._client.set_model_version_tag(
                model_b_name,
                model_b_version,
                "ab_test_id",
                test_id,
            )
            self._client.set_model_version_tag(
                model_b_name,
                model_b_version,
                "ab_test_role",
                "treatment",
            )
        except Exception as e:
            logger.warning(f"Could not tag models with A/B test info: {e}")

        logger.info(f"A/B test setup: {test_name} (ID: {test_id})")
        logger.info(f"Control: {model_a_name} v{model_a_version}")
        logger.info(f"Treatment: {model_b_name} v{model_b_version}")
        logger.info(f"Traffic split: {traffic_split * 100:.1f}% to treatment")

        return test_id

    def record_ab_test_result(
        self,
        test_id: str,
        model_name: str,
        metric_value: float,
        sample_count: int = 1,
    ) -> dict[str, Any]:
        """
        Record A/B test results for a model.

        Parameters
        ----------
        test_id : str
            A/B test identifier.
        model_name : str
            Name of the model that generated the result.
        metric_value : float
            Value of the success metric.
        sample_count : int, default 1
            Number of samples this result represents.

        Returns
        -------
        dict[str, Any]
            Updated test status and statistics.

        """
        if test_id not in self._ab_tests:
            raise ValueError(f"A/B test {test_id} not found")

        test_config = self._ab_tests[test_id]

        if test_config["status"] != "active":
            raise ValueError(f"A/B test {test_id} is not active")

        # Determine which model this result is for
        model_a_name = test_config["model_a"]["name"]
        model_b_name = test_config["model_b"]["name"]

        if model_name == model_a_name:
            test_config["results"]["model_a_samples"] += sample_count
            test_config["results"]["model_a_metric_sum"] += metric_value * sample_count
            test_config["results"]["model_a_metric_values"].append(metric_value)
        elif model_name == model_b_name:
            test_config["results"]["model_b_samples"] += sample_count
            test_config["results"]["model_b_metric_sum"] += metric_value * sample_count
            test_config["results"]["model_b_metric_values"].append(metric_value)
        else:
            raise ValueError(f"Model {model_name} is not part of A/B test {test_id}")

        # Check if test should conclude
        return self._check_ab_test_conclusion(test_id)

    def _check_ab_test_conclusion(self, test_id: str) -> dict[str, Any]:
        """
        Check if A/B test should conclude based on results.
        """
        test_config = self._ab_tests[test_id]
        results = test_config["results"]
        config = test_config["config"]

        # Calculate current statistics
        model_a_samples = results["model_a_samples"]
        model_b_samples = results["model_b_samples"]
        total_samples = model_a_samples + model_b_samples

        model_a_mean = (
            results["model_a_metric_sum"] / model_a_samples if model_a_samples > 0 else 0.0
        )
        model_b_mean = (
            results["model_b_metric_sum"] / model_b_samples if model_b_samples > 0 else 0.0
        )

        # Check duration
        duration_hours = (time.time() - test_config["created_at"]) / 3600

        status = {
            "test_id": test_id,
            "status": test_config["status"],
            "duration_hours": duration_hours,
            "total_samples": total_samples,
            "model_a_samples": model_a_samples,
            "model_b_samples": model_b_samples,
            "model_a_mean": model_a_mean,
            "model_b_mean": model_b_mean,
            "difference": model_b_mean - model_a_mean,
            "relative_improvement": (
                (model_b_mean - model_a_mean) / model_a_mean * 100 if model_a_mean != 0 else 0.0
            ),
        }

        # Check conclusion criteria
        should_conclude = False
        conclusion_reason = None

        # Time-based conclusion
        if duration_hours >= config["max_duration_hours"]:
            should_conclude = True
            conclusion_reason = "max_duration_reached"

        # Sample-based conclusion with statistical test
        elif (
            total_samples >= config["min_samples"]
            and model_a_samples >= 100
            and model_b_samples >= 100
        ):
            # Perform t-test
            model_a_values = np.array(results["model_a_metric_values"])
            model_b_values = np.array(results["model_b_metric_values"])

            if len(model_a_values) > 1 and len(model_b_values) > 1:
                # Welch's t-test
                var_a = np.var(model_a_values, ddof=1)
                var_b = np.var(model_b_values, ddof=1)

                if var_a > 0 and var_b > 0:
                    pooled_se = np.sqrt(var_a / len(model_a_values) + var_b / len(model_b_values))
                    t_stat = (model_b_mean - model_a_mean) / pooled_se

                    # Degrees of freedom (Welch's formula)
                    df = (var_a / len(model_a_values) + var_b / len(model_b_values)) ** 2 / (
                        (var_a / len(model_a_values)) ** 2 / (len(model_a_values) - 1)
                        + (var_b / len(model_b_values)) ** 2 / (len(model_b_values) - 1)
                    )

                    # Critical value (approximate for large samples)
                    critical_value = 1.96  # For alpha = 0.05, two-tailed
                    if df < 30:
                        critical_value = 2.0  # Conservative estimate

                    p_value_approx = 2 * (1 - 0.5 * (1 + np.tanh(abs(t_stat) / np.sqrt(2))))

                    status.update(
                        {
                            "t_statistic": t_stat,
                            "degrees_of_freedom": df,
                            "p_value_approx": p_value_approx,
                            "statistically_significant": abs(t_stat) > critical_value,
                        },
                    )

                    if abs(t_stat) > critical_value:
                        should_conclude = True
                        conclusion_reason = "statistical_significance"

        if should_conclude:
            test_config["status"] = "concluded"
            test_config["concluded_at"] = time.time()
            test_config["conclusion_reason"] = conclusion_reason
            status["status"] = "concluded"
            status["conclusion_reason"] = conclusion_reason

            # Determine winner
            if model_b_mean > model_a_mean:
                winner = test_config["model_b"]
                status["winner"] = "model_b"
            else:
                winner = test_config["model_a"]
                status["winner"] = "model_a"

            logger.info(f"A/B test {test_config['test_name']} concluded:")
            logger.info(f"Winner: {winner['name']} v{winner['version']}")
            logger.info(f"Improvement: {status['relative_improvement']:.2f}%")

        return status

    def setup_canary_deployment(
        self,
        deployment_name: str,
        model_name: str,
        model_version: str,
        traffic_percentage: float = 5.0,
        success_metric: str = "accuracy",
        baseline_threshold: float = 0.95,
        monitoring_duration_hours: int = 24,
        auto_promote: bool = True,
        auto_rollback: bool = True,
    ) -> str:
        """
        Set up canary deployment for gradual model rollout.

        Parameters
        ----------
        deployment_name : str
            Unique name for the canary deployment.
        model_name : str
            Name of the model to deploy.
        model_version : str
            Version of the model to deploy.
        traffic_percentage : float, default 5.0
            Percentage of traffic to route to canary (0.0 to 100.0).
        success_metric : str, default "accuracy"
            Metric to monitor for deployment success.
        baseline_threshold : float, default 0.95
            Minimum acceptable performance relative to production baseline.
        monitoring_duration_hours : int, default 24
            Hours to monitor before auto-promotion.
        auto_promote : bool, default True
            Whether to automatically promote if metrics are good.
        auto_rollback : bool, default True
            Whether to automatically rollback if metrics are bad.

        Returns
        -------
        str
            Canary deployment ID.

        """
        self._ensure_initialized()

        deployment_id = f"canary_{deployment_name}_{int(time.time())}"

        # Get baseline performance (current production model)
        baseline_performance = None
        try:
            prod_versions = self._client.get_latest_versions(
                model_name,
                stages=[ModelStage.PRODUCTION.value],
            )
            if prod_versions:
                prod_run = self._client.get_run(prod_versions[0].run_id)
                baseline_performance = prod_run.data.metrics.get(success_metric)
        except Exception as e:
            logger.warning(f"Could not get baseline performance: {e}")

        deployment_config = {
            "deployment_id": deployment_id,
            "deployment_name": deployment_name,
            "status": "active",
            "created_at": time.time(),
            "model": {
                "name": model_name,
                "version": model_version,
            },
            "config": {
                "traffic_percentage": traffic_percentage,
                "success_metric": success_metric,
                "baseline_threshold": baseline_threshold,
                "monitoring_duration_hours": monitoring_duration_hours,
                "auto_promote": auto_promote,
                "auto_rollback": auto_rollback,
            },
            "baseline": {
                "performance": baseline_performance,
                "metric": success_metric,
            },
            "metrics": {
                "sample_count": 0,
                "metric_sum": 0.0,
                "metric_values": [],
                "errors": 0,
                "latency_sum": 0.0,
                "latency_values": [],
            },
        }

        self._canary_deployments[deployment_id] = deployment_config

        # Tag model version
        try:
            self._client.set_model_version_tag(
                model_name,
                model_version,
                "canary_deployment_id",
                deployment_id,
            )
            self._client.set_model_version_tag(
                model_name,
                model_version,
                "deployment_status",
                "canary",
            )
        except Exception as e:
            logger.warning(f"Could not tag model with canary info: {e}")

        logger.info(f"Canary deployment started: {deployment_name}")
        logger.info(f"Model: {model_name} v{model_version}")
        logger.info(f"Traffic: {traffic_percentage}%")

        return deployment_id

    def record_canary_metrics(
        self,
        deployment_id: str,
        metric_value: float,
        latency_ms: float | None = None,
        error_occurred: bool = False,
    ) -> dict[str, Any]:
        """
        Record performance metrics for canary deployment.

        Parameters
        ----------
        deployment_id : str
            Canary deployment identifier.
        metric_value : float
            Value of the success metric.
        latency_ms : float | None, optional
            Response latency in milliseconds.
        error_occurred : bool, default False
            Whether an error occurred during prediction.

        Returns
        -------
        dict[str, Any]
            Updated deployment status and metrics.

        """
        if deployment_id not in self._canary_deployments:
            raise ValueError(f"Canary deployment {deployment_id} not found")

        deployment = self._canary_deployments[deployment_id]

        if deployment["status"] != "active":
            raise ValueError(f"Canary deployment {deployment_id} is not active")

        # Update metrics
        metrics = deployment["metrics"]
        metrics["sample_count"] += 1

        if not error_occurred:
            metrics["metric_sum"] += metric_value
            metrics["metric_values"].append(metric_value)
        else:
            metrics["errors"] += 1

        if latency_ms is not None:
            metrics["latency_sum"] += latency_ms
            metrics["latency_values"].append(latency_ms)

        # Check deployment status
        return self._check_canary_status(deployment_id)

    def _check_canary_status(self, deployment_id: str) -> dict[str, Any]:
        """
        Check canary deployment status and decide on promotion/rollback.
        """
        deployment = self._canary_deployments[deployment_id]
        metrics = deployment["metrics"]
        config = deployment["config"]
        baseline = deployment["baseline"]

        # Calculate current statistics
        sample_count = metrics["sample_count"]
        error_count = metrics["errors"]
        success_count = sample_count - error_count

        current_mean = metrics["metric_sum"] / success_count if success_count > 0 else 0.0
        error_rate = error_count / sample_count if sample_count > 0 else 0.0

        avg_latency = (
            metrics["latency_sum"] / len(metrics["latency_values"])
            if metrics["latency_values"]
            else 0.0
        )

        # Duration check
        duration_hours = (time.time() - deployment["created_at"]) / 3600

        status = {
            "deployment_id": deployment_id,
            "status": deployment["status"],
            "duration_hours": duration_hours,
            "sample_count": sample_count,
            "success_count": success_count,
            "error_count": error_count,
            "error_rate": error_rate,
            "current_performance": current_mean,
            "baseline_performance": baseline["performance"],
            "average_latency_ms": avg_latency,
        }

        # Performance relative to baseline
        if baseline["performance"] is not None and baseline["performance"] > 0:
            relative_performance = current_mean / baseline["performance"]
            status["relative_performance"] = relative_performance
        else:
            relative_performance = 1.0
            status["relative_performance"] = relative_performance

        # Decision logic
        should_rollback = False
        should_promote = False

        # Rollback conditions
        if config["auto_rollback"] and sample_count >= 100:  # Minimum samples for decision
            # Performance degradation
            if relative_performance < config["baseline_threshold"]:
                should_rollback = True
                status["decision_reason"] = "performance_degradation"

            # High error rate
            elif error_rate > 0.05:  # 5% error rate threshold
                should_rollback = True
                status["decision_reason"] = "high_error_rate"

        # Promotion conditions
        elif (
            config["auto_promote"]
            and duration_hours >= config["monitoring_duration_hours"]
            and sample_count >= 500  # Sufficient samples
            and relative_performance >= config["baseline_threshold"]
            and error_rate <= 0.01  # Low error rate
        ):
            should_promote = True
            status["decision_reason"] = "successful_monitoring_period"

        if should_rollback:
            deployment["status"] = "rolled_back"
            status["status"] = "rolled_back"
            logger.info(f"Canary deployment {deployment['deployment_name']} rolled back")
            logger.info(f"Reason: {status['decision_reason']}")

            # Remove canary tags
            try:
                model = deployment["model"]
                self._client.delete_model_version_tag(
                    model["name"],
                    model["version"],
                    "deployment_status",
                )
            except Exception as e:
                logger.warning(f"Could not remove canary tags: {e}")

        elif should_promote:
            deployment["status"] = "promoted"
            status["status"] = "promoted"
            logger.info(f"Canary deployment {deployment['deployment_name']} promoted to production")

            # Promote to production
            try:
                model = deployment["model"]
                self.transition_model_stage(
                    model["name"],
                    model["version"],
                    ModelStage.PRODUCTION,
                    archive_existing=True,
                )

                # Update tags
                self._client.set_model_version_tag(
                    model["name"],
                    model["version"],
                    "deployment_status",
                    "production",
                )
            except Exception as e:
                logger.warning(f"Error during promotion: {e}")

        return status

    def rollback_model(
        self,
        model_name: str,
        target_version: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """
        Rollback model to a previous version.

        Parameters
        ----------
        model_name : str
            Name of the model to rollback.
        target_version : str | None, optional
            Version to rollback to. Uses previous production if None.
        reason : str | None, optional
            Reason for rollback.

        Returns
        -------
        dict[str, Any]
            Rollback operation results.

        """
        self._ensure_initialized()

        try:
            # Get current production version
            prod_versions = self._client.get_latest_versions(
                model_name,
                stages=[ModelStage.PRODUCTION.value],
            )

            if not prod_versions:
                raise ValueError(f"No production version found for {model_name}")

            current_version = prod_versions[0].version

            # Determine target version
            if target_version is None:
                # Get all versions and find the previous production version
                all_versions = self._client.search_model_versions(
                    filter_string=f"name='{model_name}'",
                    order_by=["version_number DESC"],
                )

                # Find previous production version
                target_version = None
                for version in all_versions:
                    if version.version != current_version and "previous_production" in (
                        version.tags or {}
                    ):
                        target_version = version.version
                        break

                if target_version is None:
                    # Fallback to previous version number
                    for version in all_versions:
                        if version.version != current_version:
                            target_version = version.version
                            break

            if not target_version:
                raise ValueError("Could not determine target version for rollback")

            # Tag current version as previous
            self._client.set_model_version_tag(
                model_name,
                current_version,
                "previous_production",
                "true",
            )

            # Archive current production
            self.transition_model_stage(
                model_name,
                current_version,
                ModelStage.ARCHIVED,
            )

            # Promote target version to production
            self.transition_model_stage(
                model_name,
                target_version,
                ModelStage.PRODUCTION,
            )

            # Log rollback event
            rollback_info = {
                "model_name": model_name,
                "from_version": current_version,
                "to_version": target_version,
                "reason": reason or "manual_rollback",
                "timestamp": time.time(),
            }

            # Tag target version with rollback info
            self._client.set_model_version_tag(
                model_name,
                target_version,
                "rollback_from",
                current_version,
            )
            if reason:
                self._client.set_model_version_tag(
                    model_name,
                    target_version,
                    "rollback_reason",
                    reason,
                )

            logger.info(f"Rollback: {model_name} v{current_version} -> v{target_version}")
            if reason:
                logger.info(f"Reason: {reason}")

            return rollback_info

        except Exception as e:
            logger.info(f"Error during rollback: {e}")
            raise

    def get_deployment_history(self, model_name: str) -> list[dict[str, Any]]:
        """
        Get deployment history for a model.

        Parameters
        ----------
        model_name : str
            Name of the model.

        Returns
        -------
        list[dict[str, Any]]
            List of deployment events and transitions.

        """
        self._ensure_initialized()

        try:
            # Get all model versions
            versions = self._client.search_model_versions(
                filter_string=f"name='{model_name}'",
                order_by=["version_number DESC"],
            )

            history = []

            for version in versions:
                # Get version history
                version_info = {
                    "version": version.version,
                    "current_stage": version.current_stage,
                    "creation_timestamp": version.creation_timestamp,
                    "description": version.description,
                    "tags": dict(version.tags) if version.tags else {},
                    "run_id": version.run_id,
                }

                # Get run metrics for this version
                try:
                    run = self._client.get_run(version.run_id)
                    version_info["run_metrics"] = dict(run.data.metrics)
                    version_info["run_params"] = dict(run.data.params)
                except Exception as e:
                    logger.warning(f"Could not get run info for version {version.version}: {e}")

                history.append(version_info)

            return history

        except Exception as e:
            logger.info(f"Error getting deployment history: {e}")
            raise

    def validate_model_quality(
        self,
        model_name: str,
        version: str,
        quality_gates: dict[str, float],
    ) -> dict[str, Any]:
        """
        Validate model quality against defined gates.

        Parameters
        ----------
        model_name : str
            Name of the model to validate.
        version : str
            Version of the model to validate.
        quality_gates : dict[str, float]
            Quality thresholds (metric_name -> minimum_value).

        Returns
        -------
        dict[str, Any]
            Validation results and pass/fail status.

        """
        self._ensure_initialized()

        try:
            # Get model version
            model_version = self._client.get_model_version(model_name, version)

            # Get run metrics
            run = self._client.get_run(model_version.run_id)
            run_metrics = dict(run.data.metrics)

            validation_results: dict[str, Any] = {
                "model_name": model_name,
                "version": version,
                "validation_timestamp": time.time(),
                "gates_passed": 0,
                "gates_failed": 0,
                "overall_pass": True,
                "gate_results": {},
            }

            for gate_name, threshold in quality_gates.items():
                actual_value = run_metrics.get(gate_name)

                if actual_value is None:
                    gate_result = {
                        "threshold": threshold,
                        "actual": None,
                        "passed": False,
                        "reason": "metric_not_found",
                    }
                    validation_results["gates_failed"] = int(validation_results["gates_failed"]) + 1
                    validation_results["overall_pass"] = False

                elif actual_value >= threshold:
                    gate_result = {
                        "threshold": threshold,
                        "actual": actual_value,
                        "passed": True,
                        "margin": actual_value - threshold,
                    }
                    validation_results["gates_passed"] = int(validation_results["gates_passed"]) + 1

                else:
                    gate_result = {
                        "threshold": threshold,
                        "actual": actual_value,
                        "passed": False,
                        "shortfall": threshold - actual_value,
                    }
                    validation_results["gates_failed"] = int(validation_results["gates_failed"]) + 1
                    validation_results["overall_pass"] = False

                # Ensure gate_results is a dict
                if "gate_results" not in validation_results or not isinstance(
                    validation_results["gate_results"],
                    dict,
                ):
                    validation_results["gate_results"] = {}
                validation_results["gate_results"][gate_name] = gate_result

            # Tag model with validation results
            self._client.set_model_version_tag(
                model_name,
                version,
                "quality_validation",
                "pass" if validation_results["overall_pass"] else "fail",
            )
            self._client.set_model_version_tag(
                model_name,
                version,
                "validation_timestamp",
                str(int(float(validation_results["validation_timestamp"]))),
            )

            logger.info(f"Quality validation for {model_name} v{version}:")
            logger.info(f"Gates passed: {validation_results['gates_passed']}")
            logger.info(f"Gates failed: {validation_results['gates_failed']}")
            logger.info(f"Overall: {'PASS' if validation_results['overall_pass'] else 'FAIL'}")

            return validation_results

        except Exception as e:
            logger.info(f"Error during quality validation: {e}")
            raise
