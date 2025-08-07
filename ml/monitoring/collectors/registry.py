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
Central registry for all ML metrics collectors.

This module provides a unified interface for managing all ML metrics collectors,
providing easy access and configuration management for the entire monitoring system.

"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any, Self

from ml._imports import HAS_PROMETHEUS
from ml.monitoring._config import MonitoringConfig
from ml.monitoring.collectors.base import BaseMetricsCollector
from ml.monitoring.collectors.data import DataQualityCollector
from ml.monitoring.collectors.features import FeatureEngineeringCollector
from ml.monitoring.collectors.model import ModelLifecycleCollector
from ml.monitoring.collectors.performance import PerformanceDegradationMonitor
from ml.monitoring.collectors.resources import ResourceUtilizationCollector
from ml.monitoring.server import MetricsServer


if TYPE_CHECKING:
    pass


class MLMetricsRegistry:
    """
    Central registry for all ML metrics collectors.

    This registry provides a unified interface for managing and accessing all
    specialized metrics collectors in the ML monitoring system. It handles
    initialization, configuration, and provides easy access to specific collectors.

    Key Features
    ------------
    - Centralized management of all collectors
    - Unified configuration across collectors
    - Easy access to specific collector types
    - Health monitoring and status reporting
    - Integrated metrics server management

    Parameters
    ----------
    config : MonitoringConfig
        Configuration for all collectors and the metrics server.
    enable_background_monitoring : bool, default True
        Whether to enable background resource monitoring.

    """

    def __init__(
        self,
        config: MonitoringConfig,
        enable_background_monitoring: bool = True,
    ) -> None:
        """
        Initialize the ML metrics registry.

        Parameters
        ----------
        config : MonitoringConfig
            Configuration for metrics collection.
        enable_background_monitoring : bool, default True
            Whether to enable background resource monitoring.

        """
        self.config = config
        self._enable_background_monitoring = enable_background_monitoring

        # Initialize all collectors with shared config
        from ml.monitoring.collector import MLMetricsCollector

        self.ml_metrics = MLMetricsCollector(config)
        self.model_lifecycle = ModelLifecycleCollector(config)
        self.data_quality = DataQualityCollector(config)
        self.feature_engineering = FeatureEngineeringCollector(config)
        self.performance = PerformanceDegradationMonitor(config)
        self.resources = ResourceUtilizationCollector(config)

        # Collector mapping for easy access
        self._collectors: dict[str, BaseMetricsCollector] = {
            "ml": self.ml_metrics,
            "model": self.model_lifecycle,
            "data": self.data_quality,
            "features": self.feature_engineering,
            "performance": self.performance,
            "resources": self.resources,
        }

        # Metrics server for unified endpoint
        self.server = MetricsServer(config)

        # State tracking
        self._started = False

    def start(self) -> None:
        """
        Start the metrics registry and all associated services.

        This starts the metrics server and any background monitoring threads.

        """
        if self._started:
            return

        # Start metrics server
        self.server.start()

        # Start background resource monitoring if enabled
        if self._enable_background_monitoring and self.resources.enabled:
            self.resources.start_monitoring()

        self._started = True

    def stop(self) -> None:
        """
        Stop the metrics registry and all associated services.
        """
        if not self._started:
            return

        # Stop background monitoring
        if self._enable_background_monitoring:
            self.resources.stop_monitoring()

        # Stop metrics server
        self.server.stop()

        self._started = False

    def get_collector(self, collector_type: str) -> BaseMetricsCollector:
        """
        Get a specific collector by type.

        Parameters
        ----------
        collector_type : str
            Type of collector to retrieve. Valid types:
            - "ml": Core ML metrics collector
            - "model": Model lifecycle collector
            - "data": Data quality collector
            - "features": Feature engineering collector
            - "performance": Performance degradation monitor
            - "resources": Resource utilization collector

        Returns
        -------
        BaseMetricsCollector
            The requested collector instance.

        Raises
        ------
        ValueError
            If collector_type is not recognized.

        """
        if collector_type not in self._collectors:
            available_types = list(self._collectors.keys())
            raise ValueError(
                f"Unknown collector type '{collector_type}'. "
                f"Available types: {available_types}",
            )

        return self._collectors[collector_type]

    def list_collectors(self) -> list[str]:
        """
        Get list of available collector types.

        Returns
        -------
        List[str]
            List of available collector type identifiers.

        """
        return list(self._collectors.keys())

    def get_all_collectors(self) -> dict[str, BaseMetricsCollector]:
        """
        Get all collectors as a dictionary.

        Returns
        -------
        Dict[str, BaseMetricsCollector]
            Dictionary mapping collector types to instances.

        """
        return self._collectors.copy()

    def health_check(self) -> dict[str, Any]:
        """
        Perform comprehensive health check on all collectors.

        Returns
        -------
        Dict[str, Any]
            Health status for the entire registry including:
            - Overall status
            - Individual collector health
            - Server status
            - Configuration summary

        """
        collector_health = {}
        enabled_collectors = 0
        total_metrics = 0

        for collector_type, collector in self._collectors.items():
            # Handle collectors that may not have health_check method (like MLMetricsCollector)
            if hasattr(collector, "health_check"):
                health = collector.health_check()
            else:
                # Fallback health check for legacy collectors
                health = {
                    "enabled": getattr(collector, "enabled", True),
                    "metrics_count": getattr(collector, "get_metric_count", lambda: 0)(),
                    "prometheus_available": HAS_PROMETHEUS,
                    "config_valid": True,
                    "collector_type": collector.__class__.__name__,
                }

            collector_health[collector_type] = health

            if health["enabled"]:
                enabled_collectors += 1
            total_metrics += health["metrics_count"]

        return {
            "status": "healthy" if self._started else "stopped",
            "started": self._started,
            "enabled_collectors": enabled_collectors,
            "total_collectors": len(self._collectors),
            "total_metrics": total_metrics,
            "server_running": (
                self.server.is_running if hasattr(self.server, "is_running") else None
            ),
            "server_port": self.config.metrics_port,
            "background_monitoring": self._enable_background_monitoring,
            "collectors": collector_health,
        }

    def reset_all_metrics(self) -> None:
        """
        Reset all metrics across all collectors.

        This method is primarily used for testing scenarios where metrics need to be
        cleared between test runs.

        """
        for collector in self._collectors.values():
            collector.reset_metrics()

    def get_metrics_summary(self) -> dict[str, Any]:
        """
        Get summary of metrics across all collectors.

        Returns
        -------
        Dict[str, Any]
            Summary statistics for all collectors.

        """
        summary: dict[str, Any] = {
            "registry_status": "running" if self._started else "stopped",
            "total_collectors": len(self._collectors),
            "enabled_collectors": sum(1 for c in self._collectors.values() if c.enabled),
        }

        # Add collector-specific summaries
        for collector_type, collector in self._collectors.items():
            if not collector.enabled:
                continue

            collector_summary: dict[str, Any] = {}

            if collector_type == "data" and hasattr(collector, "get_data_quality_summary"):
                # Data quality example - in practice you'd need to specify instruments
                collector_summary = {"type": "data_quality", "enabled": True}
            elif collector_type == "features" and hasattr(collector, "get_feature_stats"):
                # Feature engineering example
                collector_summary = {"type": "feature_engineering", "enabled": True}
            elif collector_type == "performance" and hasattr(collector, "get_performance_summary"):
                # Performance monitoring example
                collector_summary = {"type": "performance_monitoring", "enabled": True}
            elif collector_type == "resources" and hasattr(collector, "get_resource_summary"):
                # Resource utilization
                try:
                    collector_summary = collector.get_resource_summary()
                    collector_summary["type"] = "resource_utilization"
                except Exception:
                    collector_summary = {
                        "type": "resource_utilization",
                        "error": "collection_failed",
                    }
            else:
                collector_summary = {
                    "type": collector_type,
                    "enabled": collector.enabled,
                    "metrics_count": collector.get_metric_count(),
                }

            summary[f"{collector_type}_collector"] = collector_summary

        return summary

    def configure_collector(
        self,
        collector_type: str,
        **kwargs: Any,
    ) -> None:
        """
        Configure a specific collector with additional parameters.

        Parameters
        ----------
        collector_type : str
            Type of collector to configure.
        **kwargs : Any
            Configuration parameters specific to the collector.

        Raises
        ------
        ValueError
            If collector_type is not recognized.
        NotImplementedError
            If the collector doesn't support runtime configuration.

        """
        self.get_collector(collector_type)

        # For now, most collectors don't support runtime reconfiguration
        # This could be extended in the future for dynamic configuration
        raise NotImplementedError(
            f"Runtime configuration not supported for {collector_type} collector. "
            "Configuration must be set at initialization time.",
        )

    def get_prometheus_registry(self) -> Any:
        """
        Get the Prometheus registry used by all collectors.

        Returns
        -------
        prometheus_client.CollectorRegistry | None
            The Prometheus registry, or None if Prometheus is not available.

        """
        if not self.config.enabled:
            return None

        try:
            from prometheus_client import REGISTRY

            return REGISTRY
        except ImportError:
            return None

    def export_metrics(self) -> str:
        """
        Export all metrics in Prometheus format.

        Returns
        -------
        str
            Metrics in Prometheus text format, or empty string if not enabled.

        """
        if not self.config.enabled:
            return ""

        try:
            from prometheus_client import generate_latest

            return generate_latest().decode("utf-8")
        except ImportError:
            return ""

    @property
    def enabled(self) -> bool:
        """
        Check if the registry is enabled.

        Returns
        -------
        bool
            True if monitoring is enabled and at least one collector is enabled.

        """
        return self.config.enabled and any(c.enabled for c in self._collectors.values())

    @property
    def started(self) -> bool:
        """
        Check if the registry is started.

        Returns
        -------
        bool
            True if the registry has been started.

        """
        return self._started

    def __enter__(self) -> Self:
        """
        Context manager entry.
        """
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """
        Context manager exit.
        """
        self.stop()

    def __repr__(self) -> str:
        """
        Return string representation of the registry.
        """
        enabled_count = sum(1 for c in self._collectors.values() if c.enabled)
        return (
            f"MLMetricsRegistry("
            f"started={self._started}, "
            f"enabled_collectors={enabled_count}/{len(self._collectors)}, "
            f"server_port={self.config.metrics_port})"
        )
