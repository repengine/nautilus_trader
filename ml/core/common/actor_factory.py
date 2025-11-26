"""
Actor Factory Component.

This module provides actor factory and lifecycle management extracted from
MLIntegrationManager as part of the god-class decomposition effort (Phase 3.6.6).
The component handles:

- Actor creation with db_connection injection
- Graceful shutdown with store flushing
- Message publisher configuration for DataStore
- Configuration stubs for message bus, event emission, domain bookkeeping

The component follows Protocol-First Interface Design and can be used independently
or composed via the MLIntegrationManagerFacade.

Example
-------
>>> from ml.core.common.actor_factory import ActorFactoryComponent
>>> component = ActorFactoryComponent(
...     db_connection="postgresql://postgres:postgres@localhost:5432/nautilus",
...     feature_store=feature_store,
...     model_store=model_store,
...     strategy_store=strategy_store,
...     data_store=data_store,
... )
>>> actor = component.create_integrated_actor(MyActorClass, config)
>>> component.shutdown()

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING, Any, cast


if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


logger = logging.getLogger(__name__)


@dataclass
class ActorFactoryComponent:
    """
    Manages actor creation, shutdown, and message publisher configuration.

    This component implements the actor factory responsibilities extracted from
    MLIntegrationManager. It provides:

    - Actor instantiation with db_connection injection
    - Graceful shutdown with store flushing
    - Message bus and event configuration stubs (no-op for TDD compatibility)
    - Cross-domain event emission

    Attributes
    ----------
    db_connection : str | None
        Database connection string to inject into actor configs.
    feature_store : object
        Feature store instance for flushing on shutdown.
    model_store : object
        Model store instance for flushing on shutdown.
    strategy_store : object
        Strategy store instance for flushing on shutdown.
    data_store : object | None
        Data store instance for flushing on shutdown and publisher attachment.

    Example
    -------
    >>> component = ActorFactoryComponent(
    ...     db_connection="postgresql://localhost:5432/nautilus",
    ...     feature_store=feature_store,
    ...     model_store=model_store,
    ...     strategy_store=strategy_store,
    ...     data_store=data_store,
    ... )
    >>> actor = component.create_integrated_actor(MyActor, config)
    >>> component.set_message_publisher(publisher)
    >>> component.shutdown()

    """

    # Injected dependencies
    db_connection: str | None = None
    feature_store: object = field(default=None)
    model_store: object = field(default=None)
    strategy_store: object = field(default=None)
    data_store: object | None = field(default=None)

    def create_integrated_actor(
        self,
        actor_class: type[Any],
        config: object,
    ) -> object:
        """
        Create an actor with automatic integration.

        Attempts to attach the db_connection to the config before instantiation.
        If the config is frozen (immutable dataclass), logs the exception and
        continues with the original config.

        Parameters
        ----------
        actor_class : type[Any]
            The actor class to instantiate.
        config : object
            Actor configuration object, ideally with a db_connection attribute.

        Returns
        -------
        object
            Instantiated actor with stores automatically connected via its
            base class initialization.

        Example
        -------
        >>> component = ActorFactoryComponent(db_connection="postgresql://...")
        >>> actor = component.create_integrated_actor(MyActor, config)
        >>> assert actor.config.db_connection == component.db_connection

        """
        # Ensure config has the database connection
        if not hasattr(config, "db_connection"):
            # Best-effort attach db_connection for consumers expecting it
            try:
                setattr(config, "db_connection", self.db_connection)
            except Exception:
                logger.exception("Failed to attach db_connection to config")

        # Create actor - stores are automatically initialized by the base class
        actor = actor_class(config=config)

        return actor

    def shutdown(self) -> None:
        """
        Gracefully shutdown all components.

        Flushes all pending writes to stores. Handles exceptions gracefully
        to ensure all stores are attempted even if one fails.

        Example
        -------
        >>> component = ActorFactoryComponent(
        ...     feature_store=fs,
        ...     model_store=ms,
        ...     strategy_store=ss,
        ...     data_store=ds,
        ... )
        >>> component.shutdown()

        """
        # Flush all pending writes, handling exceptions gracefully
        stores_to_flush = [
            ("feature_store", self.feature_store),
            ("model_store", self.model_store),
            ("strategy_store", self.strategy_store),
            ("data_store", self.data_store),
        ]

        for store_name, store in stores_to_flush:
            if store is not None and hasattr(store, "flush"):
                try:
                    store.flush()
                except Exception:
                    logger.exception(
                        "Failed to flush %s during shutdown",
                        store_name,
                    )

        logger.info("ML integration manager shutdown complete")

    # -------------------------------------------------------------------------
    # TDD prototype convenience hooks (no-op stubs)
    # -------------------------------------------------------------------------

    def configure_message_bus(
        self,
        *,
        backend: str | None = None,
        topic_prefix: str | None = None,
        retention_hours: int | None = None,
        max_size_mb: int | None = None,
    ) -> None:
        """
        No-op configuration stub for message bus (for tests).

        Parameters
        ----------
        backend : str | None
            Message bus backend type (unused).
        topic_prefix : str | None
            Prefix for topic names (unused).
        retention_hours : int | None
            Message retention period in hours (unused).
        max_size_mb : int | None
            Maximum queue size in MB (unused).

        Returns
        -------
        None

        """
        _ = (backend, topic_prefix, retention_hours, max_size_mb)
        return None

    def configure_event_emission(
        self,
        *,
        batching_enabled: bool | None = None,
        batch_size: int | None = None,
        flush_interval_ms: int | None = None,
        correlation_strategy: str | None = None,
    ) -> None:
        """
        No-op configuration stub for event emission (for tests).

        Parameters
        ----------
        batching_enabled : bool | None
            Whether to enable batching (unused).
        batch_size : int | None
            Batch size for events (unused).
        flush_interval_ms : int | None
            Flush interval in milliseconds (unused).
        correlation_strategy : str | None
            Strategy for event correlation (unused).

        Returns
        -------
        None

        """
        _ = (batching_enabled, batch_size, flush_interval_ms, correlation_strategy)
        return None

    def configure_event_system(self, **_: object) -> None:
        """
        No-op aggregate configuration for event system (for tests).

        Parameters
        ----------
        **_ : object
            Ignored keyword arguments.

        Returns
        -------
        None

        """
        return None

    def configure_domain_bookkeeping(self, _config: object) -> None:
        """
        No-op configuration stub for domain bookkeeping (for tests).

        Parameters
        ----------
        _config : object
            Configuration object (unused).

        Returns
        -------
        None

        """
        return None

    def emit_cross_domain_event(self, _event: dict[str, object]) -> None:
        """
        No-op cross-domain event emitter stub (for tests).

        Parameters
        ----------
        _event : dict[str, object]
            Event data to emit (unused).

        Returns
        -------
        None

        """
        return None

    def emit_cascade(
        self,
        source_event: dict[str, object],
        target_domain: str,
        *,
        delay_ns: int | None = None,
    ) -> dict[str, object]:
        """
        Create a cascaded event preserving correlation and timestamp order.

        This adapter delegates to a light helper in ``ml.common.cascade`` to
        avoid deep coupling and keep hot paths unaffected.

        Parameters
        ----------
        source_event : dict[str, object]
            The source event with correlation_id and other metadata.
        target_domain : str
            The target domain for the cascaded event.
        delay_ns : int | None
            Optional delay in nanoseconds for the target event.

        Returns
        -------
        dict[str, object]
            The cascaded event with preserved correlation and updated domain.

        Example
        -------
        >>> source = {
        ...     "domain": "features",
        ...     "event_type": "feature_computed",
        ...     "correlation_id": "abc123",
        ...     "instrument_id": "BTC.USD",
        ...     "ts_event": 1000000000,
        ...     "event_id": "evt_001",
        ...     "payload": {"feature_name": "sma_20"},
        ... }
        >>> result = component.emit_cascade(source, "model", delay_ns=100)
        >>> assert result["correlation_id"] == "abc123"
        >>> assert result["domain"] == "model"

        """
        from ml.common.cascade import EventDict
        from ml.common.cascade import emit_cascade as _emit_cascade

        ev: EventDict = EventDict(
            domain=cast(str, source_event.get("domain", "")),
            event_type=cast(str, source_event.get("event_type", "")),
            correlation_id=cast(str, source_event.get("correlation_id", "")),
            instrument_id=cast(str, source_event.get("instrument_id", "")),
            ts_event=int(cast(Any, source_event.get("ts_event", 0))),
            source_event_id=cast(
                str,
                source_event.get("event_id", source_event.get("source_event_id", "unknown")),
            ),
            payload=cast(dict[str, Any], source_event.get("payload", {}) or {}),
        )
        out = _emit_cascade(ev, target_domain, delay_ns)
        return dict(out)

    def set_message_publisher(self, publisher: object) -> None:
        """
        Configure the message publisher for ML stores which support it.

        Currently applies to ``DataStore`` only. Safe to call at any time;
        if the data_store is not initialized or doesn't have a publisher
        attribute, this method is a no-op.

        Parameters
        ----------
        publisher : object
            The message publisher to attach to the data store.

        Example
        -------
        >>> component = ActorFactoryComponent(data_store=data_store)
        >>> component.set_message_publisher(my_publisher)
        >>> assert data_store.publisher is my_publisher

        """
        # Avoid strict typing dependency; use duck-typing and assign when attribute exists
        if self.data_store is not None and hasattr(self.data_store, "publisher"):
            try:
                cast(Any, self.data_store).publisher = publisher
            except Exception:
                logger.debug("Failed to attach publisher to data_store", exc_info=True)


__all__ = ["ActorFactoryComponent"]
