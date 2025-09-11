"""
Shared engine/metadata initialization mixin for stores.

Centralizes engine acquisition, metadata creation, and table setup with consistent pool
status logging.

"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import MetaData
from sqlalchemy.engine import Engine

from ml.core.db_engine import EngineManager


logger = logging.getLogger(__name__)


class EngineInitMixin:
    """
    Mixin to initialize SQLAlchemy engine + metadata and call `_setup_tables()`.

    Expects the consumer to define `connection_string` attribute and `_setup_tables()` method.

    """

    connection_string: str | None
    engine: Engine
    metadata: MetaData

    def _init_engine_and_tables(self) -> None:
        if not self.connection_string:
            return
        # Initialize engine and metadata, then call subclass table setup
        self.engine = EngineManager.get_engine(self.connection_string)
        self.metadata = MetaData()
        self._setup_tables()  # type: ignore[attr-defined]
        # Optional: pool status logging (best-effort)
        try:
            status: dict[str, Any] | None = EngineManager.get_pool_status(self.connection_string)
            if status:
                logger.debug("Engine pool status: %s", status)
        except Exception as exc:
            logger.debug("Pool status unavailable: %s", exc)
