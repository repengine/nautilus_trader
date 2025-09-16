from __future__ import annotations

import importlib
import sys
from types import ModuleType


def test_metrics_detection_true_when_module_present(monkeypatch) -> None:
    # Inject a dummy prometheus_client module
    dummy = ModuleType("prometheus_client")
    monkeypatch.setitem(sys.modules, "prometheus_client", dummy)
    mod = importlib.import_module("ml.common.metrics_detection")
    # Re-import to ensure detection saw the injected module
    importlib.reload(mod)
    assert isinstance(mod.HAS_METRICS_BACKEND, bool)
    assert mod.HAS_METRICS_BACKEND is True


def test_metrics_detection_false_when_module_missing(monkeypatch) -> None:
    # Force importlib.import_module to raise for prometheus_client
    orig_import_module = importlib.import_module

    def fake_import(name: str, package: str | None = None):  # type: ignore[no-untyped-def]
        if name == "prometheus_client":
            raise ImportError("not installed")
        return orig_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    mod = importlib.import_module("ml.common.metrics_detection")
    importlib.reload(mod)
    assert mod.HAS_METRICS_BACKEND is False
