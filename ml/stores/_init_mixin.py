"""
Shared initialization mixin for SQL-backed ML stores.

Centralizes common constructor wiring used by ModelStore and StrategyStore:
- Persistence configuration and connection string resolution
- Batch/flush interval settings and clock assignment
- Message bus publisher initialization
- Optional persistence manager injection (for tests)
- Engine + metadata initialization with table setup

This mixin is intentionally conservative and leaves buffer allocation
(`_write_buffer`) to concrete store classes to keep element types precise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal


if TYPE_CHECKING:  # pragma: no cover - typing only
    from ml.common.message_bus import MessagePublisherProtocol
    from ml.registry.persistence import PersistenceConfig
    from nautilus_trader.common.clock import Clock


class StoreInitMixin:
    """
    Mixin providing `_init_store_common` for shared constructor logic.
    """

    def _init_store_common(
        self: Any,
        *,
        connection_string: str | None,
        persistence_config: PersistenceConfig | None,
        batch_size: int,
        flush_interval_ms: int,
        flush_interval_seconds: float | None,
        clock: Clock | None,
        enable_publishing: bool,
        publisher: MessagePublisherProtocol | None,
        publish_mode: Literal["batch", "row", "both"],
        persistence_manager: object | None,
    ) -> None:
        """
        Initialize shared store state and engine/tables.

        Parameters
        ----------
        connection_string : str | None
            Optional PostgreSQL connection string.
        persistence_config : PersistenceConfig | None
            Optional persistence configuration. When not provided and the
            connection string looks like PostgreSQL, a minimal one is built.
        batch_size : int
            Max batch size before auto-flush.
        flush_interval_ms : int
            Maximum time between flushes (ms); overridden by seconds when given.
        flush_interval_seconds : float | None
            Alternative flush interval in seconds.
        clock : Clock | None
            Optional clock for timestamping.
        enable_publishing : bool
            Whether to enable bus publishing.
        publisher : MessagePublisherProtocol | None
            Publisher implementation when publishing is enabled.
        publish_mode : {"batch", "row", "both"}
            Publish mode for bus events.
        persistence_manager : object | None
            Optional injected persistence/session provider for tests.
        """
        # Resolve persistence config from connection string when appropriate
        cfg = persistence_config
        if connection_string and not cfg and (
            "postgresql://" in connection_string or "postgres://" in connection_string
        ):
            from ml.registry.persistence import BackendType
            from ml.registry.persistence import PersistenceConfig as _PC

            cfg = _PC(backend=BackendType.POSTGRES, connection_string=connection_string)

        # Wire persistence manager and connection string
        if cfg is not None:
            from ml.registry.persistence import PersistenceManager

            setattr(self, "persistence", PersistenceManager(cfg))
            setattr(self, "connection_string", cfg.connection_string)
        else:
            setattr(self, "persistence", None)
            # Fallback default consistent with existing code
            setattr(
                self,
                "connection_string",
                connection_string or "postgresql://postgres:postgres@localhost:5432/nautilus",
            )

        # Batch/flush/clock
        setattr(self, "batch_size", int(batch_size))
        if flush_interval_seconds is not None:
            setattr(self, "flush_interval_ms", int(flush_interval_seconds * 1000))
        else:
            setattr(self, "flush_interval_ms", int(flush_interval_ms))
        setattr(self, "clock", clock)

        # Bus publisher
        self._init_bus_publishing(
            enable_publishing=enable_publishing,
            publisher=publisher,
            publish_mode=publish_mode,
        )

        # Allow tests to inject a mock persistence manager directly
        if persistence_manager is not None:
            try:
                setattr(self, "persistence", persistence_manager)
            except Exception:
                # Keep silent; tests may pass incompatible doubles
                pass

        # Common timing field for buffered stores (buffer itself is store-specific)
        setattr(self, "_last_flush_ns", 0)

        # Initialize engine + metadata and call subclass _setup_tables()
        self._init_engine_and_tables()
