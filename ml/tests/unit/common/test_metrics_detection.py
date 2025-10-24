from __future__ import annotations

import importlib
import sys
from types import ModuleType

from pytest import MonkeyPatch


def test_metrics_detection_true_when_module_present(monkeypatch: MonkeyPatch) -> None:
    """Test metrics detection when prometheus_client is available.

    NOTE: This test intentionally uses importlib.reload() to test module-level
    detection logic. This is acceptable because we're testing import-time behavior,
    not toggling feature flags. The reload is necessary to re-execute the module's
    detection code after monkeypatching sys.modules.
    """
    # Inject a dummy prometheus_client module
    dummy = ModuleType("prometheus_client")
    monkeypatch.setitem(sys.modules, "prometheus_client", dummy)
    mod = importlib.import_module("ml.common.metrics_detection")
    # Re-import to ensure detection saw the injected module
    importlib.reload(mod)
    assert isinstance(mod.HAS_METRICS_BACKEND, bool)
    assert mod.HAS_METRICS_BACKEND is True


def test_metrics_detection_false_when_module_missing(monkeypatch: MonkeyPatch) -> None:
    """Test metrics detection when prometheus_client is missing.

    NOTE: This test intentionally uses importlib.reload() to test module-level
    detection logic. This is acceptable because we're testing import-time behavior,
    not toggling feature flags. The reload is necessary to re-execute the module's
    detection code after monkeypatching importlib.import_module.
    """
    # Force importlib.import_module to raise for prometheus_client
    orig_import_module = importlib.import_module

    def fake_import(name: str, package: str | None = None) -> ModuleType:
        if name == "prometheus_client":
            raise ImportError("not installed")
        return orig_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    mod = importlib.import_module("ml.common.metrics_detection")
    importlib.reload(mod)
    assert mod.HAS_METRICS_BACKEND is False
