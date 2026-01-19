"""
Message bus configuration and environment parsing.

This module centralizes configuration for the optional ML message bus and defines typed
helpers to construct publisher instances. Publishing is disabled by default and enabled
via environment flags to preserve hot-path budgets.

"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final, Literal


BusBackend = Literal["noop", "redis", "file"]
TopicScheme = Literal["domain_op", "stage_first"]


def _env_truthy(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class MessageBusConfig:
    """
    Message bus configuration parsed from environment or provided explicitly.

    Attributes
    ----------
    enabled : bool
        Whether publishing is enabled. Default False.
    backend : BusBackend
        Publishing backend implementation. Default "noop".
    scheme : TopicScheme
        Topic naming scheme: "domain_op" (canonical) or "stage_first". Default "domain_op".
    topic_prefix : str
        Prefix for stage-first topics. Default "events.ml".
    redis_url : str
        Redis connection URL for Redis Streams. Default "redis://localhost:6379/0".
    redis_stream : str
        Stream name to append events to. Default "ml-events".
    redis_maxlen : int | None
        Optional approximate max length for the stream (XADD MAXLEN ~). If None, unbounded.
    file_path : str | None
        JSONL file path for the file-backed publisher (optional).

    """

    enabled: bool = False
    backend: BusBackend = "noop"
    scheme: TopicScheme = "domain_op"
    topic_prefix: str = "events.ml"
    redis_url: str = "redis://localhost:6379/0"
    redis_stream: str = "ml-events"
    redis_maxlen: int | None = None
    file_path: str | None = None

    @staticmethod
    def from_env() -> MessageBusConfig:
        """
        Construct configuration from environment variables.

        Environment variables
        ---------------------
        - ML_BUS_ENABLE: bool (default: false)
        - ML_BUS_BACKEND: "noop" | "redis" (default: "noop")
        - ML_BUS_SCHEME: "domain_op" | "stage_first" (default: "domain_op")
        - ML_BUS_TOPIC_PREFIX: str (default: "events.ml")
        - ML_BUS_REDIS_URL: str (default: "redis://localhost:6379/0")
        - ML_BUS_REDIS_STREAM: str (default: "ml-events")
        - ML_BUS_REDIS_MAXLEN: int (optional; default: unset)
        - ML_BUS_FILE_PATH: str (optional; JSONL path for file backend)

        """
        enabled = _env_truthy("ML_BUS_ENABLE", default=False)
        backend_env = (os.getenv("ML_BUS_BACKEND") or "noop").strip().lower()
        if backend_env == "redis":
            backend: BusBackend = "redis"
        elif backend_env == "file":
            backend = "file"
        else:
            backend = "noop"
        scheme_env = (os.getenv("ML_BUS_SCHEME") or "domain_op").strip().lower()
        scheme: TopicScheme = "stage_first" if scheme_env == "stage_first" else "domain_op"
        topic_prefix = os.getenv("ML_BUS_TOPIC_PREFIX", "events.ml").strip() or "events.ml"
        redis_url = os.getenv("ML_BUS_REDIS_URL", "redis://localhost:6379/0").strip()
        redis_stream = os.getenv("ML_BUS_REDIS_STREAM", "ml-events").strip() or "ml-events"
        redis_maxlen_str = os.getenv("ML_BUS_REDIS_MAXLEN")
        redis_maxlen: int | None = None
        if redis_maxlen_str:
            try:
                redis_maxlen = int(redis_maxlen_str)
            except ValueError:
                redis_maxlen = None
        file_path = os.getenv("ML_BUS_FILE_PATH")
        file_path = file_path.strip() if file_path else None

        return MessageBusConfig(
            enabled=enabled,
            backend=backend,
            scheme=scheme,
            topic_prefix=topic_prefix,
            redis_url=redis_url,
            redis_stream=redis_stream,
            redis_maxlen=redis_maxlen,
            file_path=file_path,
        )


DEFAULT_CONFIG: Final[MessageBusConfig] = MessageBusConfig()
