"""
Common buffered-store behaviors for SQL-backed ML stores.

This mixin centralizes flush semantics, time-based flush checks, and health checks
for stores that buffer writes and then persist in batches. It assumes the concrete
class defines the following attributes and methods:

- Attributes:
  - `_write_buffer`: list[Any]
  - `_last_flush_ns`: int
  - `flush_interval_ms`: int
  - `clock`: object with `timestamp_ns()` or `None`
  - `engine`: SQLAlchemy Engine

- Methods provided by the concrete store:
  - `write_batch(data: list[Any], *[, emit_events: bool]) -> None`
  - `_emit_events(items: list[Any]) -> None`  # per-store registry/metrics emission

The mixin is intentionally light on types where stores differ (e.g. write_batch
signature) and uses defensive calls to preserve backwards compatibility.
"""

from __future__ import annotations

from typing import Any


class BufferedStoreMixin:
    """
    Mixin providing buffered flush behavior, time-based flush decision, and health check.
    """

    # Expected attributes on subclasses (documented for type checkers/readers)
    _write_buffer: list[Any]
    _last_flush_ns: int
    flush_interval_ms: int
    clock: Any | None
    engine: Any  # SQLAlchemy Engine (runtime type)

    def _should_flush_by_time(self) -> bool:
        """
        Return True if enough time has elapsed based on `flush_interval_ms`.
        """
        if not self.clock or not self._last_flush_ns:
            return False

        try:
            elapsed_ms = (self.clock.timestamp_ns() - self._last_flush_ns) / 1e6
            return bool(elapsed_ms >= float(self.flush_interval_ms))
        except Exception:
            return False

    def flush(self) -> None:
        """
        Flush pending buffered items to storage and emit registry/metrics events.

        This is off the hot path. Publish to external buses should already happen
        inside the store's write path; `_emit_events` is intended for DataRegistry
        and metrics notifications.
        """
        if not getattr(self, "_write_buffer", None):  # empty or not present
            return

        buffer_copy = list(self._write_buffer)

        # Attempt to call write_batch with `emit_events=False` when supported
        write_batch = getattr(self, "write_batch")
        try:
            # Best-effort: some stores accept `emit_events`; others do not.
            write_batch(buffer_copy, emit_events=False)
        except TypeError:
            write_batch(buffer_copy)

        # Emit registry/metrics events after successful storage
        try:
            emit = getattr(self, "_emit_events", None)
            if callable(emit):
                emit(buffer_copy)
        finally:
            # Clear buffer and update flush time
            self._write_buffer.clear()
            if self.clock:
                try:
                    self._last_flush_ns = int(self.clock.timestamp_ns())
                except Exception:
                    self._last_flush_ns = 0

    def is_healthy(self) -> bool:
        """
        Basic connectivity check against the underlying engine.
        """
        try:
            if getattr(self, "engine", None):
                from sqlalchemy import text  # local import to avoid module-level dependency

                with self.engine.connect() as conn:
                    result = conn.execute(text("SELECT 1"))
                    return result is not None
            return True  # No engine implies in-memory/dummy; consider healthy.
        except Exception:
            return False
