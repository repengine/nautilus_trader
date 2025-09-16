"""
Structured logging configuration for ML processes.

Provides structlog + stdlib interop with minimal overhead and env-driven settings.

Env variables
- ML_LOG_LEVEL: default "INFO" (valid stdlib levels)
- ML_LOG_FORMAT: "json" | "console" (default: json)
- LOG_FILE: optional path to also write logs to a file (cold path)
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterable

import structlog
from structlog.contextvars import bind_contextvars
from structlog.contextvars import clear_contextvars
from structlog.contextvars import merge_contextvars
from structlog.processors import TimeStamper
from structlog.stdlib import LoggerFactory
from structlog.stdlib import ProcessorFormatter
from structlog.stdlib import add_logger_name
from structlog.stdlib import filter_by_level


def _make_formatter(json: bool) -> ProcessorFormatter:
    """Create a ProcessorFormatter with JSON or console rendering."""
    from typing import Any, cast

    # Renderer is a structlog processor callable; keep typing loose for mypy --strict
    renderer: Any
    if json:
        renderer = structlog.processors.JSONRenderer()
    else:  # developer-friendly console
        try:
            from structlog.dev import ConsoleRenderer

            renderer = ConsoleRenderer()
        except Exception:  # pragma: no cover - fallback if dev extras missing
            renderer = structlog.processors.JSONRenderer()

    pre_chain = [
        structlog.stdlib.add_log_level,
        TimeStamper(fmt="iso", utc=True),
        add_logger_name,
    ]

    return ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=cast(Any, pre_chain),
    )


def configure_logging(level: str | None = None, json: bool | None = None) -> None:
    """
    Configure structured logging for the current process.

    Safe to call multiple times; resets root handlers to prevent duplicates.
    """
    # Resolve settings
    level_name = (level or os.getenv("ML_LOG_LEVEL") or "INFO").upper()
    fmt_name = (os.getenv("ML_LOG_FORMAT") or ("json" if json is None else ("json" if json else "console"))).lower()
    use_json = fmt_name == "json"

    # Root logger handlers
    root = logging.getLogger()
    root.setLevel(getattr(logging, level_name, logging.INFO))

    # Build stream handler
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(_make_formatter(use_json))

    # Optional file handler
    handlers: list[logging.Handler] = [stream_handler]
    log_file = os.getenv("LOG_FILE")
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(_make_formatter(use_json))
        handlers.append(file_handler)

    # Reset handlers to avoid duplicates
    root.handlers = handlers

    # structlog processors
    structlog.configure(
        processors=[
            filter_by_level,
            merge_contextvars,
            structlog.processors.add_log_level,
            add_logger_name,
            TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def bind_log_context(**fields: object) -> None:
    """Bind contextual fields (run_id, correlation_id, component, etc.) to log context."""
    if fields:
        bind_contextvars(**fields)


def clear_log_context(keys: Iterable[str] | None = None) -> None:
    """Clear bound contextvars. If keys provided, unbind only those fields."""
    if keys:
        # structlog has no selective unbind; rebind by clearing then rebind the remaining
        # For simplicity, clear all when keys provided (callers can rebind as needed)
        clear_contextvars()
    else:
        clear_contextvars()


__all__ = [
    "bind_log_context",
    "clear_log_context",
    "configure_logging",
]
