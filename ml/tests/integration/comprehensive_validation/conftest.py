#!/usr/bin/env python3
"""
Integration test configuration for comprehensive validation suite.

Marks all tests in this directory as integration to ensure they run in the serial
integration phase (not in the parallel non-integration phase).

"""

from __future__ import annotations

import pytest

# Apply the integration marker to all tests in this package
pytestmark = pytest.mark.integration


def pytest_ignore_collect(path, config):  # type: ignore[override]
    """
    Ignore collecting this suite when running with -m 'not integration'.
    """
    markexpr = getattr(config.option, "markexpr", "") or ""
    # If the mark expression contains 'not integration', skip collecting files here
    if "not integration" in markexpr:
        return True
    return False
