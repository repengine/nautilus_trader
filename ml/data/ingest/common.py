"""
Common ingestion utilities (shared by CLIs and services).

Provides small, dependency-light helpers for retry/backoff interop, simple rate-
limiting, and durable JSON progress persistence.

"""

from __future__ import annotations

from pathlib import Path
from time import monotonic
from time import sleep
from typing import Any

from ml.data.ingest.resume import BackoffPolicy  # re-export
from ml.data.ingest.resume import IngestState  # re-export
from ml.data.ingest.resume import SleepFn  # re-export


__all__ = [
    "BackoffPolicy",
    "IngestState",
    "RateLimiter",
    "SleepFn",
    "load_progress_json",
    "save_progress_json",
]


class RateLimiter:
    """
    Minimal rate limiter for client-side pacing.

    Usage:
        rl = RateLimiter(per_minute=120)
        ...
        rl.wait()  # call before each request

    """

    def __init__(self, *, per_minute: int) -> None:
        self._interval = 60.0 / float(max(1, per_minute))
        self._next_allowed = 0.0

    def wait(self) -> None:
        now = monotonic()
        if self._next_allowed == 0.0:
            self._next_allowed = now
            return
        if now < self._next_allowed:
            sleep(self._next_allowed - now)
        self._next_allowed = monotonic() + self._interval


def load_progress_json(path: str | Path) -> dict[str, Any]:
    """
    Load a JSON file into a dict; returns empty dict on error or missing file.
    """
    p = Path(path)
    try:
        if p.exists():
            import json

            return dict(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        pass
    return {}


def save_progress_json(path: str | Path, data: dict[str, Any]) -> None:
    """
    Safely persist a small JSON object (atomic replace when possible).
    """
    import json

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        # Best-effort fallback
        try:
            p.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        except Exception:
            pass
