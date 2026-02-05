from __future__ import annotations

import importlib


def test_strict_conformance_check_module_importable() -> None:
    """
    Ensure the strict conformance module imports without side effects.
    """
    module = importlib.import_module("ml.stores._strict_conformance_check")
    assert module is not None
