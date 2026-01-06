"""
Store initialization invariants for the facade-only integration manager.

Legacy cast-introspection tests were replaced with behavior checks focused on
fallback initialization paths and typed container outputs.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ml.core.integration import MLIntegrationManager
from ml.core.integration import init_ml_stores_and_registries
from ml.tests.utils.db import build_postgres_url


TEST_DB_CONNECTION = build_postgres_url()


def _patch_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_candidates = MagicMock(urls=(TEST_DB_CONNECTION,))
    monkeypatch.setattr(
        "ml.core.integration_facade.collect_postgres_candidates",
        lambda *args, **kwargs: mock_candidates,
    )


def test_dummy_fallback_initializes_dummy_stores(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_candidates(monkeypatch)
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

    from ml.stores.base import DummyStore
    assert isinstance(mgr.feature_store, DummyStore)
    assert isinstance(mgr.model_store, DummyStore)
    assert isinstance(mgr.strategy_store, DummyStore)
    assert isinstance(mgr.data_store, DummyStore)

    for registry in (
        mgr.feature_registry,
        mgr.model_registry,
        mgr.strategy_registry,
        mgr.data_registry,
    ):
        assert registry is not None
        assert "Registry" in type(registry).__name__


def test_file_fallback_initializes_file_stores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_candidates(monkeypatch)
    monkeypatch.setattr(
        "ml.core.common.database_lifecycle.DatabaseLifecycleComponent.is_postgres_running",
        lambda self: False,
    )
    monkeypatch.setattr(
        "ml.core.integration_facade.MLIntegrationManagerFacade._maybe_run_backfill_on_start",
        lambda self: None,
    )
    monkeypatch.setenv("ML_FILE_STORE_PATH", str(tmp_path))

    mgr = MLIntegrationManager(config=None, ensure_healthy=False)

    from ml.stores.file_backed import FileFeatureStore
    from ml.stores.file_backed import FileModelStore
    from ml.stores.file_backed import FileStrategyStore

    assert isinstance(mgr.feature_store, FileFeatureStore)
    assert isinstance(mgr.model_store, FileModelStore)
    assert isinstance(mgr.strategy_store, FileStrategyStore)


def test_init_ml_stores_and_registries_dummy_mode() -> None:
    config = SimpleNamespace(use_dummy_stores=True, db_connection=None)
    stores = init_ml_stores_and_registries(config)

    from ml.stores.base import DummyStore
    assert isinstance(stores.feature_store, DummyStore)
    assert isinstance(stores.model_store, DummyStore)
    assert isinstance(stores.strategy_store, DummyStore)
    assert isinstance(stores.data_store, DummyStore)

    for registry in (
        stores.feature_registry,
        stores.model_registry,
        stores.strategy_registry,
        stores.data_registry,
    ):
        assert registry is not None
        assert "Registry" in type(registry).__name__
