"""Regression tests for ML import helpers."""

from __future__ import annotations

import builtins
import importlib
import sys
from types import ModuleType

import pytest


def test_purge_module_removes_submodules() -> None:
    """Ensure the local purge helper clears modules and descendants."""
    from ml import _imports

    sys.modules["test_pkg"] = ModuleType("test_pkg")
    sys.modules["test_pkg.child"] = ModuleType("test_pkg.child")

    _imports._purge_module("test_pkg")

    assert "test_pkg" not in sys.modules
    assert "test_pkg.child" not in sys.modules


def test_nautilus_core_import_propagates_unexpected_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify unexpected import errors are surfaced and do not poison sys.modules."""
    # Ensure the module is loaded before patching so we can restore it afterwards.
    import ml._imports  # noqa: F401  # Imported for side effects

    original_import = builtins.__import__

    def _raising_import(name: str, *args: object, **kwargs: object) -> ModuleType:
        if name == "nautilus_trader.backtest.engine":
            raise RuntimeError("boom")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _raising_import)

    # Drop cached module so the guarded import path executes again.
    sys.modules.pop("ml._imports", None)

    with pytest.raises(RuntimeError):
        importlib.import_module("ml._imports")

    # Restore the original import function before reloading the module.
    monkeypatch.setattr(builtins, "__import__", original_import)

    # Re-import to restore the module for subsequent tests.
    sys.modules.pop("ml._imports", None)
    importlib.import_module("ml._imports")


def test_nautilus_core_flag_matches_engine_availability() -> None:
    """Check the exported flag reflects the availability of the native engine."""
    from ml import _imports

    if not _imports.HAS_NAUTILUS_CORE:
        pytest.skip("Nautilus Trader core extensions not available in this environment")

    engine_module = importlib.import_module("nautilus_trader.backtest.engine")
    assert hasattr(engine_module, "BacktestEngine")
