#!/usr/bin/env python3

"""
Runtime attachment for ML pipeline orchestrator.

This module provides runtime integration management including:
- Integration manager factory initialization
- Store and registry attachment
- Validator execution

This component is extracted from the MLPipelineOrchestrator god class to provide
focused, testable runtime attachment functionality.

"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from ml.orchestration.config_types import IntegrationConfig


if TYPE_CHECKING:
    from ml.registry.protocols import RegistryProtocol


logger = logging.getLogger(__name__)


# ========================================================================
# Protocol Definitions
# ========================================================================


class IntegrationManagerProtocol(Protocol):
    """
    Protocol for ML integration managers.

    Defines the interface for accessing stores and registries from an integration manager.
    """

    data_registry: object | None
    feature_registry: object | None
    model_registry: object | None
    strategy_registry: object | None
    data_store: object | None
    feature_store: object | None
    model_store: object | None
    strategy_store: object | None
    partition_manager: object | None


class RuntimeAttacherProtocol(Protocol):
    """
    Protocol for runtime attachment operations.
    """

    def attach_runtime(
        self,
        integration_cfg: IntegrationConfig | None,
        *,
        dataset_out_dir: Path,
    ) -> IntegrationManagerProtocol | None:
        """
        Attach ML integration runtime components.

        Parameters
        ----------
        integration_cfg : IntegrationConfig | None
            Integration configuration
        dataset_out_dir : Path
            Dataset output directory

        Returns
        -------
        IntegrationManagerProtocol | None
            The attached integration manager or None if not attached

        """
        ...


# ========================================================================
# RuntimeAttacher Implementation
# ========================================================================


class RuntimeAttacher:
    """
    Attaches ML integration runtime components to the pipeline.

    Handles integration manager initialization, store/registry attachment,
    and validator execution.

    This component is extracted from the MLPipelineOrchestrator god class to
    provide focused, testable runtime attachment functionality.

    Parameters
    ----------
    integration_manager_factory : Callable[..., IntegrationManagerProtocol] | None
        Factory callable for creating integration managers
    data_registry : RegistryProtocol | None
        Data registry instance (for event emission)

    """

    def __init__(
        self,
        *,
        integration_manager_factory: Callable[..., IntegrationManagerProtocol] | None = None,
        data_registry: RegistryProtocol | None = None,
        integration_manager: IntegrationManagerProtocol | None = None,
        validators: list[object] | None = None,
    ) -> None:
        """
        Initialize runtime attacher.

        Parameters
        ----------
        integration_manager_factory : Callable[..., IntegrationManagerProtocol] | None
            Factory for creating integration managers
        data_registry : RegistryProtocol | None
            Data registry for event emission

        """
        self._integration_manager_factory = integration_manager_factory
        self._integration_manager: IntegrationManagerProtocol | None = integration_manager
        self._data_registry = data_registry
        self.validators = validators

        # Attached components (populated by attach_runtime)
        self.data_registry: object | None = data_registry
        self.feature_registry: object | None = None
        self.model_registry: object | None = None
        self.strategy_registry: object | None = None
        self.feature_store: object | None = None
        self.model_store: object | None = None
        self.strategy_store: object | None = None
        self.data_store: object | None = None
        self.partition_manager: object | None = None

        logger.debug("Initialized RuntimeAttacher")

    @property
    def integration_manager(self) -> IntegrationManagerProtocol | None:
        """Return the current integration manager."""
        return self._integration_manager

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def attach_runtime(
        self,
        integration_cfg: IntegrationConfig | None,
        *,
        dataset_out_dir: Path,
    ) -> IntegrationManagerProtocol | None:
        """
        Attach ML integration runtime components.

        Initializes the integration manager and populates store/registry references.

        Parameters
        ----------
        integration_cfg : IntegrationConfig | None
            Integration configuration
        dataset_out_dir : Path
            Dataset output directory

        Returns
        -------
        IntegrationManagerProtocol | None
            The attached integration manager or None if not attached

        """
        if integration_cfg is None or not integration_cfg.enabled:
            return None

        logger.info(
            "Attaching ML integration runtime (validators=%s, out_dir=%s)",
            integration_cfg.run_validators,
            dataset_out_dir,
        )

        if self._integration_manager is None:
            factory = self._integration_manager_factory
            if factory is None:
                from ml.core.integration import MLIntegrationManager as _MLIntegrationManager

                factory = cast(
                    Callable[..., IntegrationManagerProtocol],
                    _MLIntegrationManager,
                )

            kwargs: dict[str, Any] = {
                "auto_start_postgres": integration_cfg.auto_start_postgres,
                "auto_migrate": integration_cfg.auto_migrate,
                "ensure_healthy": integration_cfg.ensure_healthy,
            }
            if integration_cfg.db_connection is not None:
                kwargs["db_connection"] = integration_cfg.db_connection
            if integration_cfg.strict_protocol_validation is not None:
                kwargs["strict_protocol_validation"] = integration_cfg.strict_protocol_validation

            manager = factory(**kwargs)
            self._integration_manager = manager
        else:
            manager = self._integration_manager

        # Attach components from manager
        for attr in (
            "data_registry",
            "feature_registry",
            "model_registry",
            "strategy_registry",
            "feature_store",
            "model_store",
            "strategy_store",
            "data_store",
            "partition_manager",
        ):
            if getattr(self, attr, None) is None:
                setattr(self, attr, getattr(manager, attr, None))

        if integration_cfg.run_validators:
            self.run_validators()

        return manager

    def run_validators(self) -> None:
        """
        Run runtime validators (metrics and events).

        Raises
        ------
        RuntimeError
            If validation fails

        """
        from tools import validate_event_constants as event_mod
        from tools import validate_metrics_bootstrap as metrics_mod

        metrics_rc = metrics_mod.main()
        if metrics_rc != 0:
            raise RuntimeError("metrics bootstrap validation failed")

        events_rc = event_mod.main()
        if events_rc != 0:
            raise RuntimeError("event constants validation failed")

        logger.info("Runtime validators succeeded")

    def emit_feature_refresh_event(self, metrics_path: Path) -> None:
        """
        Emit a feature refresh event.

        Parameters
        ----------
        metrics_path : Path
            Path to metrics file

        """
        try:
            from ml.common.event_emitter import emit_dataset_event
            from ml.config.events import EventStatus
            from ml.config.events import Source
            from ml.config.events import Stage
        except Exception:
            logger.debug("Failed to import event emitter dependencies; skipping event emission", exc_info=True)
            return

        feature_set_id = "unknown"
        metadata: dict[str, object] = {}
        if metrics_path.exists():
            try:
                data = json.loads(metrics_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    feature_set_id = str(data.get("feature_set_id", feature_set_id))
                    metadata = {k: v for k, v in data.items() if isinstance(k, str)}
            except Exception:
                logger.debug("Failed to parse metrics payload; continuing without metadata", exc_info=True)

        meta_payload = dict(metadata)
        meta_payload["feature_set_id"] = feature_set_id

        try:
            registry_obj = self.data_registry
            if registry_obj is None:
                return
            data_registry = cast("RegistryProtocol", registry_obj)
            emit_dataset_event(
                data_registry,
                dataset_id="features",
                instrument_id="GLOBAL",
                stage=Stage.FEATURE_COMPUTED,
                source=Source.HISTORICAL,
                run_id=f"refresh_{feature_set_id}",
                ts_min=0,
                ts_max=0,
                count=1,
                status=EventStatus.SUCCESS,
                metadata=meta_payload,
                dataset_type="features",
                component="pipeline_orchestrator.refresh_features",
            )
        except Exception:
            logger.debug("Failed to emit feature refresh event", exc_info=True)

    def get_attached_components(self) -> dict[str, object | None]:
        """
        Get all attached components.

        Returns
        -------
        dict[str, object | None]
            Dictionary of component name to component instance

        """
        return {
            "data_registry": self.data_registry,
            "feature_registry": self.feature_registry,
            "model_registry": self.model_registry,
            "strategy_registry": self.strategy_registry,
            "feature_store": self.feature_store,
            "model_store": self.model_store,
            "strategy_store": self.strategy_store,
            "data_store": self.data_store,
            "partition_manager": self.partition_manager,
        }
