"""
Health Monitoring Component.

This module provides health monitoring extracted from MLIntegrationManager
as part of the god-class decomposition effort (Phase 3.6.4). The component handles:

- Health checks for stores and registries
- Health aggregation by domain
- Protocol compliance validation
- Partition health checks

The component follows Protocol-First Interface Design and can be used independently
or composed via the MLIntegrationManagerFacade.

Example
-------
>>> from ml.core.common.health_monitoring import HealthMonitoringComponent
>>> component = HealthMonitoringComponent(
...     feature_store=feature_store,
...     model_store=model_store,
...     # ... other stores and registries
... )
>>> health = component.check_health()
>>> print(health)

"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from ml.common.protocols import MLComponentProtocol


logger = logging.getLogger(__name__)


@dataclass
class HealthMonitoringComponent:
    """
    Manages health monitoring for all ML components.

    Provides per-component health checks, domain-level health aggregation,
    and protocol compliance validation for stores and registries.

    This component implements the health monitoring responsibilities
    extracted from MLIntegrationManager. It follows the Protocol-First
    Interface Design pattern for component interaction.

    Attributes
    ----------
    feature_store : object | None
        The FeatureStore instance to monitor.
    model_store : object | None
        The ModelStore instance to monitor.
    strategy_store : object | None
        The StrategyStore instance to monitor.
    data_store : object | None
        The DataStore instance to monitor.
    feature_registry : object | None
        The FeatureRegistry instance to monitor.
    model_registry : object | None
        The ModelRegistry instance to monitor.
    strategy_registry : object | None
        The StrategyRegistry instance to monitor.
    data_registry : object | None
        The DataRegistry instance to monitor.
    partition_manager : object | None
        The PartitionManager instance to monitor.
    is_postgres_running : Callable[[], bool]
        Function to check if PostgreSQL is running.

    Example
    -------
    >>> component = HealthMonitoringComponent(
    ...     feature_store=feature_store,
    ...     model_store=model_store,
    ...     strategy_store=strategy_store,
    ...     data_store=data_store,
    ...     feature_registry=feature_registry,
    ...     model_registry=model_registry,
    ...     strategy_registry=strategy_registry,
    ...     data_registry=data_registry,
    ...     is_postgres_running=lambda: True,
    ... )
    >>> health = component.check_health()
    >>> component.ensure_healthy()

    """

    # Components to monitor (injected)
    feature_store: object | None = None
    model_store: object | None = None
    strategy_store: object | None = None
    data_store: object | None = None
    feature_registry: object | None = None
    model_registry: object | None = None
    strategy_registry: object | None = None
    data_registry: object | None = None
    partition_manager: object | None = None

    # Database check function (injected)
    is_postgres_running: Callable[[], bool] = field(default=lambda: False)

    def ensure_healthy(self) -> None:
        """
        Ensure all components are healthy.

        Checks health of all components and raises RuntimeError if any
        component is unhealthy. Logs success message when all components
        are healthy.

        Raises
        ------
        RuntimeError
            If one or more components are unhealthy, with the list of
            unhealthy component names in the error message.

        Example
        -------
        >>> component = HealthMonitoringComponent(...)
        >>> component.ensure_healthy()  # Raises if unhealthy
        >>> print("All components healthy!")

        """
        health = self.check_health()

        unhealthy = [k for k, v in health.items() if not v]
        if unhealthy:
            raise RuntimeError(f"Unhealthy components: {unhealthy}")

        logger.warning("All ML components are healthy!")

    def validate_protocol_compliance(self, strict: bool | None = None) -> None:
        """
        Validate MLComponentProtocol compliance for core components.

        Checks that all stores and registries implement the MLComponentProtocol
        and that their protocol methods work correctly without raising exceptions.

        Parameters
        ----------
        strict : bool | None
            If True, raise RuntimeError on violations. If None, reads from
            environment variable ``ML_STRICT_PROTOCOL_VALIDATION`` (defaults
            to False). When not strict, logs a warning instead of raising.

        Raises
        ------
        RuntimeError
            If strict mode is enabled and protocol compliance issues are found.

        Example
        -------
        >>> component = HealthMonitoringComponent(...)
        >>> component.validate_protocol_compliance(strict=True)  # Raises on issues
        >>> component.validate_protocol_compliance(strict=False)  # Logs warnings

        """
        if strict is None:
            strict = os.getenv("ML_STRICT_PROTOCOL_VALIDATION", "").lower() in {
                "1",
                "true",
                "yes",
            }

        components: dict[str, Any] = {
            "feature_store": self.feature_store,
            "model_store": self.model_store,
            "strategy_store": self.strategy_store,
            "data_store": self.data_store,
            "feature_registry": self.feature_registry,
            "model_registry": self.model_registry,
            "strategy_registry": self.strategy_registry,
            "data_registry": self.data_registry,
        }

        violations: dict[str, list[str]] = {}

        for name, comp in components.items():
            issues: list[str] = []
            if comp is None or not isinstance(comp, MLComponentProtocol):
                issues.append("does_not_implement_protocol")
            else:
                try:
                    _ = comp.get_health_status()
                except Exception as e:  # pragma: no cover - defensive
                    issues.append(f"health_status_error:{e}")
                try:
                    _ = comp.get_performance_metrics()
                except Exception as e:  # pragma: no cover - defensive
                    issues.append(f"performance_metrics_error:{e}")
                try:
                    config_issues = comp.validate_configuration()
                    if config_issues:
                        issues.extend([f"config:{i}" for i in config_issues])
                except Exception as e:  # pragma: no cover - defensive
                    issues.append(f"validate_configuration_error:{e}")

            if issues:
                violations[name] = issues

        if violations:
            msg = f"Protocol compliance issues: {violations}"
            if strict:
                raise RuntimeError(msg)
            logger.warning(msg)

    def aggregate_health(self) -> dict[str, object]:
        """
        Aggregate component health into domain and system summaries.

        Groups components by domain (data, features, model, strategy) and
        provides an overall system health status. For each component that
        implements MLComponentProtocol, retrieves detailed health status
        and performance metrics.

        Returns
        -------
        dict[str, object]
            A structured health summary with keys:
            - ``components``: per-component health and metrics (when available)
            - ``domains``: aggregated health per domain (data, features, model, strategy)
            - ``system``: overall status with list of unhealthy components

        Example
        -------
        >>> component = HealthMonitoringComponent(...)
        >>> summary = component.aggregate_health()
        >>> print(summary["system"]["healthy"])  # True/False
        >>> print(summary["domains"]["data"]["healthy"])  # Data domain health

        """

        def _comp_health(comp: object) -> dict[str, object]:
            healthy = True
            health: dict[str, object] | None = None
            metrics: dict[str, float] | None = None
            if isinstance(comp, MLComponentProtocol):
                try:
                    health = comp.get_health_status()
                except Exception:
                    healthy = False
                try:
                    metrics = comp.get_performance_metrics()
                except Exception:
                    metrics = None
            return {"healthy": healthy, "health": health or {}, "metrics": metrics or {}}

        components: dict[str, dict[str, object]] = {}
        comp_map: dict[str, object | None] = {
            "feature_store": self.feature_store,
            "model_store": self.model_store,
            "strategy_store": self.strategy_store,
            "data_store": self.data_store,
            "feature_registry": self.feature_registry,
            "model_registry": self.model_registry,
            "strategy_registry": self.strategy_registry,
            "data_registry": self.data_registry,
        }

        for name, comp in comp_map.items():
            components[name] = (
                _comp_health(comp)
                if comp is not None
                else {
                    "healthy": False,
                    "health": {},
                    "metrics": {},
                }
            )

        def _domain_healthy(keys: list[str]) -> bool:
            return all(components[k]["healthy"] for k in keys if k in components)

        domains: dict[str, dict[str, object]] = {
            "data": {
                "components": ["data_store", "data_registry"],
                "healthy": _domain_healthy(["data_store", "data_registry"]),
            },
            "features": {
                "components": ["feature_store", "feature_registry"],
                "healthy": _domain_healthy(["feature_store", "feature_registry"]),
            },
            "model": {
                "components": ["model_store", "model_registry"],
                "healthy": _domain_healthy(["model_store", "model_registry"]),
            },
            "strategy": {
                "components": ["strategy_store", "strategy_registry"],
                "healthy": _domain_healthy(["strategy_store", "strategy_registry"]),
            },
        }

        unhealthy_components = [
            name for name, info in components.items() if not info["healthy"]
        ]
        system: dict[str, object] = {
            "healthy": len(unhealthy_components) == 0,
            "unhealthy": unhealthy_components,
        }

        return {"components": components, "domains": domains, "system": system}

    def check_health(self) -> dict[str, bool]:
        """
        Check health of all components.

        Performs health checks on PostgreSQL, all stores, all registries,
        DataStore, and the partition manager. Each check is done independently
        with exceptions handled gracefully.

        Returns
        -------
        dict[str, bool]
            Health status of each component with keys:
            - ``postgres``: PostgreSQL database availability
            - ``feature_store``: FeatureStore health
            - ``model_store``: ModelStore health
            - ``strategy_store``: StrategyStore health
            - ``feature_registry``: FeatureRegistry health
            - ``model_registry``: ModelRegistry health
            - ``strategy_registry``: StrategyRegistry health
            - ``data_registry``: DataRegistry health
            - ``data_store``: DataStore health
            - ``partitions``: Partition manager health

        Example
        -------
        >>> component = HealthMonitoringComponent(...)
        >>> health = component.check_health()
        >>> if health["postgres"]:
        ...     print("PostgreSQL is running")

        """
        health: dict[str, bool] = {}

        # Check database
        health["postgres"] = self.is_postgres_running()

        # Check stores
        health["feature_store"] = self.check_store_health(self.feature_store)
        health["model_store"] = self.check_store_health(self.model_store)
        health["strategy_store"] = self.check_store_health(self.strategy_store)

        # Check registries
        health["feature_registry"] = self.check_registry_health(
            self.feature_registry,
            "list_features",
        )
        health["model_registry"] = self.check_registry_health(
            self.model_registry, "list_models"
        )
        health["strategy_registry"] = self.check_registry_health(
            self.strategy_registry,
            "list_strategies",
        )
        health["data_registry"] = self.check_registry_health(
            self.data_registry, "list_datasets"
        )

        # Check DataStore
        health["data_store"] = self.check_data_store_health()

        # Check partitions
        health["partitions"] = self.check_partition_health()

        return health

    def check_store_health(self, store: object | None) -> bool:
        """
        Check health of a store component.

        Attempts to call ``get_statistics()`` on the store. If not available,
        falls back to ``is_healthy()`` method. Returns False on any exception.

        Parameters
        ----------
        store : object | None
            The store to check. Returns False if None.

        Returns
        -------
        bool
            True if the store is healthy, False otherwise.

        Example
        -------
        >>> component = HealthMonitoringComponent(feature_store=fs)
        >>> is_healthy = component.check_store_health(fs)

        """
        if store is None:
            return False
        try:
            # Prefer get_statistics() if available, else try is_healthy()
            if hasattr(store, "get_statistics") and callable(store.get_statistics):
                store.get_statistics()
                return True
            return bool(getattr(store, "is_healthy", lambda: False)())
        except Exception:
            return False

    def check_registry_health(
        self,
        registry: object | None,
        method_name: str,
    ) -> bool:
        """
        Check health of a registry component.

        Verifies that the registry has the specified list method and that
        it can be called successfully. For ``list_datasets``, only checks
        method existence without invoking (avoids expensive queries).

        Parameters
        ----------
        registry : object | None
            The registry to check. Returns False if None.
        method_name : str
            The list method to check (e.g., ``list_features``, ``list_models``).

        Returns
        -------
        bool
            True if the registry is healthy, False otherwise.

        Example
        -------
        >>> component = HealthMonitoringComponent(feature_registry=fr)
        >>> is_healthy = component.check_registry_health(fr, "list_features")

        """
        if registry is None:
            return False
        try:
            method = getattr(registry, method_name, None)
            if method_name == "list_datasets":
                return bool(method and callable(method))
            return bool(method and callable(method) and method())
        except Exception:
            return False

    def check_data_store_health(self) -> bool:
        """
        Check health of DataStore component.

        Verifies that the DataStore exists and has a ``registry`` attribute,
        indicating it was properly wired with a DataRegistry.

        Returns
        -------
        bool
            True if the DataStore is healthy, False otherwise.

        Example
        -------
        >>> component = HealthMonitoringComponent(data_store=ds)
        >>> is_healthy = component.check_data_store_health()

        """
        try:
            return bool(self.data_store and hasattr(self.data_store, "registry"))
        except Exception:
            return False

    def check_partition_health(self) -> bool:
        """
        Check health of partition manager.

        Verifies that the partition manager exists and can return partition
        statistics with at least one partition.

        Returns
        -------
        bool
            True if the partition manager is healthy, False otherwise.

        Example
        -------
        >>> component = HealthMonitoringComponent(partition_manager=pm)
        >>> is_healthy = component.check_partition_health()

        """
        try:
            if self.partition_manager is None:
                return False
            get_stats = getattr(self.partition_manager, "get_partition_stats", None)
            if get_stats is None or not callable(get_stats):
                return False
            stats = get_stats()
            return len(stats) > 0
        except Exception:
            return False


__all__ = ["HealthMonitoringComponent"]
