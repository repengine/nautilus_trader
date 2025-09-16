"""
Project-wide startup customizations for Python.

This module is imported automatically by the Python site module (if present on
sys.path). We use it solely to silence a noisy third-party deprecation warning
emitted during import of the `fs` package (PyFilesystem2) before pytest can
apply its own warning filters.

Keeping this narrowly scoped prevents masking useful warnings.
"""

from __future__ import annotations

import warnings


def _silence_pyfilesystem_pkg_resources_warning() -> None:
    # Ignore only the specific deprecation message emitted by fs/__init__.py
    # at import time, where pytest's filters may not yet be active.
    try:
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=r"^pkg_resources is deprecated as an API.*",
            module=r"^fs(\.|$)",
        )
        # Additional fallback: ignore the exact message regardless of module
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=r"^pkg_resources is deprecated as an API.*",
        )
    except Exception:
        # Best-effort; never fail interpreter startup
        pass


_silence_pyfilesystem_pkg_resources_warning()
