"""
Helpers for probing optional ML imports.

These utilities provide a typed and side-effect free way to check whether modules can be
imported in the current environment without raising during module import time. This is
primarily used by the test suite to verify that our public ML package remains
importable.

"""

from __future__ import annotations

import importlib


def probe_import(module_name: str) -> tuple[bool, str]:
    """
    Attempt to import ``module_name`` and return a success flag and message.

    The import is performed lazily via :func:`importlib.import_module`. Any
    exception raised while importing is captured and returned to the caller as
    a string so that diagnostics can be surfaced without propagating the
    original exception to the test harness.

    """
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        return False, f"Failed to import {module_name!r}: {exc}"

    return True, ""


__all__ = ["probe_import"]
