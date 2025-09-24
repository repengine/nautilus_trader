from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from ml.core.integration import MLIntegrationManager


@contextmanager
def _env(var: str, value: str) -> Iterator[None]:
    old = os.environ.get(var)
    os.environ[var] = value
    try:
        yield
    finally:
        if old is None:
            del os.environ[var]
        else:
            os.environ[var] = old


def test_integration_manager_fallback_to_dummy(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force Postgres check to fail
    monkeypatch.setattr(MLIntegrationManager, "_is_postgres_running", lambda self: False)

    with _env("ML_ALLOW_DUMMY", "1"):
        mgr = MLIntegrationManager(config=None, ensure_healthy=False)
        # Stores/registries should be present even in dummy mode
        assert hasattr(mgr, "feature_store")
        assert hasattr(mgr, "model_store")
        assert hasattr(mgr, "strategy_store")
        assert hasattr(mgr, "data_registry")
