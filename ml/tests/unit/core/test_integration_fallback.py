from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ml.core.integration import MLIntegrationManager
from ml.tests.utils.db import build_postgres_url


TEST_DB_CONNECTION = build_postgres_url()


def _patch_facade_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_candidates = MagicMock(urls=(TEST_DB_CONNECTION,))
    monkeypatch.setattr(
        "ml.core.integration_facade.collect_postgres_candidates",
        lambda *args, **kwargs: mock_candidates,
    )
    monkeypatch.setattr(
        "ml.core.common.database_lifecycle.DatabaseLifecycleComponent.is_postgres_running",
        lambda self: False,
    )
    monkeypatch.setattr(
        "ml.core.common.store_initialization.StoreInitializationComponent.enable_file_fallback",
        lambda self: False,
    )
    monkeypatch.setattr(
        "ml.core.integration_facade.MLIntegrationManagerFacade._maybe_run_backfill_on_start",
        lambda self: None,
    )


def test_integration_manager_fallback_to_dummy(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_facade_dependencies(monkeypatch)

    mgr = MLIntegrationManager(config=None, ensure_healthy=False)

    assert hasattr(mgr, "feature_store")
    assert hasattr(mgr, "model_store")
    assert hasattr(mgr, "strategy_store")
    assert hasattr(mgr, "data_registry")


def test_integration_manager_uses_candidate_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = "postgresql://postgres:postgres@localhost:5433/nautilus"
    mock_candidates = MagicMock(urls=(expected,))
    monkeypatch.setattr(
        "ml.core.integration_facade.collect_postgres_candidates",
        lambda *args, **kwargs: mock_candidates,
    )
    monkeypatch.setattr(
        "ml.core.common.database_lifecycle.DatabaseLifecycleComponent.is_postgres_running",
        lambda self: False,
    )
    monkeypatch.setattr(
        "ml.core.common.store_initialization.StoreInitializationComponent.enable_file_fallback",
        lambda self: False,
    )
    monkeypatch.setattr(
        "ml.core.integration_facade.MLIntegrationManagerFacade._maybe_run_backfill_on_start",
        lambda self: None,
    )

    mgr = MLIntegrationManager(config=None, ensure_healthy=False)

    assert mgr.db_connection == expected
