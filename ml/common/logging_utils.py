"""
Logging helpers to preserve legacy keyword arguments on stdlib loggers.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any


_KNOWN_LOG_KWARGS = {"exc_info", "stack_info", "stacklevel", "extra"}


class KeywordLogger:
    """
    Lightweight wrapper that redirects unexpected kwargs into ``extra``.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _prepare_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        extra = dict(kwargs.get("extra") or {})
        for key in list(kwargs.keys()):
            if key in _KNOWN_LOG_KWARGS:
                continue
            extra[key] = kwargs.pop(key)
        if extra:
            kwargs["extra"] = extra
        return kwargs

    def _call(self, method: Callable[..., Any], msg: object, *args: object, **kwargs: Any) -> None:
        prepared = self._prepare_kwargs(dict(kwargs))
        method(msg, *args, **prepared)

    def debug(self, msg: object, *args: object, **kwargs: Any) -> None:
        self._call(self._logger.debug, msg, *args, **kwargs)

    def info(self, msg: object, *args: object, **kwargs: Any) -> None:
        self._call(self._logger.info, msg, *args, **kwargs)

    def warning(self, msg: object, *args: object, **kwargs: Any) -> None:
        self._call(self._logger.warning, msg, *args, **kwargs)

    def error(self, msg: object, *args: object, **kwargs: Any) -> None:
        self._call(self._logger.error, msg, *args, **kwargs)

    def exception(self, msg: object, *args: object, **kwargs: Any) -> None:
        self._call(self._logger.exception, msg, *args, **kwargs)

    def critical(self, msg: object, *args: object, **kwargs: Any) -> None:
        self._call(self._logger.critical, msg, *args, **kwargs)

    def log(self, level: int, msg: object, *args: object, **kwargs: Any) -> None:
        self._call(lambda m, *a, **kw: self._logger.log(level, m, *a, **kw), msg, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._logger, name)


def ensure_keyword_logger(logger: logging.Logger | KeywordLogger) -> KeywordLogger:
    if isinstance(logger, KeywordLogger):
        return logger
    wrapped = KeywordLogger(logger)
    for name in ("debug", "info", "warning", "error", "exception", "critical", "log"):
        if hasattr(logger, name):
            try:
                setattr(logger, name, getattr(wrapped, name))
            except AttributeError:
                # Some logger implementations expose read-only methods.
                pass
    return wrapped


def log_best_effort(
    logger: logging.Logger | KeywordLogger,
    level: str,
    msg: object,
    *args: object,
    **kwargs: Any,
) -> None:
    """
    Attempt to log without raising if the underlying logger rejects kwargs.
    """
    target = logger if isinstance(logger, KeywordLogger) else ensure_keyword_logger(logger)
    method = getattr(target, level, None)
    if callable(method):
        try:
            method(msg, *args, **kwargs)
        except Exception:
            logging.getLogger(__name__).debug("Best-effort log failed", exc_info=True)


__all__ = ["KeywordLogger", "ensure_keyword_logger", "log_best_effort"]
