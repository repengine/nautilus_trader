"""
Environment helpers (dotenv loading, etc.).

Utilities in this module centralize how the project loads environment variables
from ``.env`` files so that CLI scripts and services behave consistently.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


_LOADED_FILES: set[Path] = set()


def _iter_candidate_dirs() -> Iterable[Path]:
    """
    Yield directories to search for an env file, starting from CWD upwards.
    """
    cwd = Path.cwd().resolve()
    seen: set[Path] = set()
    for directory in (cwd, *cwd.parents):
        if directory in seen:
            continue
        seen.add(directory)
        yield directory


def load_project_dotenv(filename: str = ".env", *, override: bool = False) -> Path | None:
    """
    Load the nearest ``.env`` file (starting from cwd) exactly once.

    Args:
        filename: Env file name to search for. Defaults to ``.env``.
        override: When True, override existing env vars (mirrors python-dotenv).

    Returns:
        Path to the loaded env file, or None if not found / dotenv unavailable.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dependency optional at runtime
        return None

    for directory in _iter_candidate_dirs():
        candidate = directory / filename
        if candidate in _LOADED_FILES:
            return candidate
        if candidate.exists():
            load_dotenv(candidate, override=override)
            _LOADED_FILES.add(candidate)
            return candidate
    return None


__all__ = ["load_project_dotenv"]
