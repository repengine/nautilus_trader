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
        try:
            method(msg, *args, **prepared)
            return
        except TypeError:
            fallback = dict(prepared)
            removed = False
            for key in ("exc_info", "stack_info", "stacklevel"):
                if key in fallback:
                    fallback.pop(key)
                    removed = True
            if removed:
                try:
                    method(msg, *args, **fallback)
                    return
                except Exception:
                    logging.getLogger(__name__).debug("KeywordLogger fallback failed", exc_info=True)
                    return
            logging.getLogger(__name__).debug("KeywordLogger suppressed logging error", exc_info=True)

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



class KeywordLoggerMixin:
    """
    Mixin providing a KeywordLogger-backed ``log`` property.

    Classes inheriting from this mixin receive a logger that gracefully handles
    keyword arguments such as ``exc_info`` even when the underlying logger does not
    support them.
    """

    _raw_logger: logging.Logger | KeywordLogger
    _keyword_logger: KeywordLogger

    @property
    def log(self) -> KeywordLogger:
        """
        Return the keyword-safe logger, creating one on demand if needed.
        """
        if hasattr(self, "_keyword_logger"):
            return self._keyword_logger

        raw_logger = getattr(self, "_raw_logger", None)
        if isinstance(raw_logger, KeywordLogger):
            underlying = getattr(raw_logger, "_logger", None)
            if underlying is not None:
                self._raw_logger = underlying
            self._keyword_logger = raw_logger
            return self._keyword_logger

        if raw_logger is None:
            raw_logger = logging.getLogger(self.__class__.__name__)
            self._raw_logger = raw_logger

        self._keyword_logger = KeywordLogger(raw_logger)
        return self._keyword_logger

    @log.setter
    def log(self, value: logging.Logger | KeywordLogger) -> None:
        """
        Store the raw logger (if available) and expose a keyword-safe wrapper.
        """
        if isinstance(value, KeywordLogger):
            underlying = getattr(value, "_logger", None)
            if underlying is not None:
                self._raw_logger = underlying
            else:
                self._raw_logger = logging.getLogger(self.__class__.__name__)
            self._keyword_logger = value
            return

        self._raw_logger = value
        self._keyword_logger = KeywordLogger(value)


__all__ = ["KeywordLogger", "KeywordLoggerMixin", "ensure_keyword_logger", "log_best_effort"]
