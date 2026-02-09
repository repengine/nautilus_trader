"""
Facade invariants for MLIntegrationManager.

Legacy parity tests have been replaced with facade-only checks now that the
component-facade implementation is the single supported path.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from ml.tests.utils.db import build_postgres_url


TEST_DB_CONNECTION = build_postgres_url()


def _create_facade_manager(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ensure_healthy: bool = False,
) -> Any:
    import ml.core.integration_facade as integration_facade
    from ml.core.common.database_lifecycle import DatabaseLifecycleComponent
    from ml.core.common.store_initialization import StoreInitializationComponent
    from ml.core.integration_facade import MLIntegrationManagerFacade

    mock_candidates = MagicMock(urls=(TEST_DB_CONNECTION,))
    monkeypatch.setattr(
        integration_facade,
        "collect_postgres_candidates",
        lambda *args, **kwargs: mock_candidates,
    )
    monkeypatch.setattr(DatabaseLifecycleComponent, "is_postgres_running", lambda self: False)
    monkeypatch.setattr(StoreInitializationComponent, "enable_file_fallback", lambda self: False)
    monkeypatch.setattr(MLIntegrationManagerFacade, "_init_partition_manager", lambda self: None)
    monkeypatch.setattr(
        MLIntegrationManagerFacade,
        "_maybe_run_backfill_on_start",
        lambda self: None,
    )

    return MLIntegrationManagerFacade(
        db_connection=TEST_DB_CONNECTION,
        auto_start_postgres=False,
        auto_migrate=False,
        ensure_healthy=ensure_healthy,
        strict_protocol_validation=False,
    )


@pytest.fixture
def facade_manager(monkeypatch: pytest.MonkeyPatch) -> Any:
    return _create_facade_manager(monkeypatch)


def test_check_health_has_expected_keys(facade_manager: Any) -> None:
    health = facade_manager.check_health()

    expected_keys = {
        "postgres",
        "feature_store",
        "model_store",
        "strategy_store",
        "feature_registry",
        "model_registry",
        "strategy_registry",
        "data_registry",
        "data_store",
        "partitions",
    }
    assert expected_keys.issubset(set(health.keys()))


def test_aggregate_health_structure(facade_manager: Any) -> None:
    summary = facade_manager.aggregate_health()

    assert "components" in summary
    assert "domains" in summary
    assert "system" in summary

    expected_domains = {"data", "features", "model", "strategy"}
    assert expected_domains == set(summary["domains"].keys())

    assert "healthy" in summary["system"]
    assert "unhealthy" in summary["system"]


def test_fallback_initializes_stores_and_registries(facade_manager: Any) -> None:
    for attr in [
        "feature_store",
        "model_store",
        "strategy_store",
        "feature_registry",
        "model_registry",
        "strategy_registry",
        "data_registry",
    ]:
        assert getattr(facade_manager, attr, None) is not None


def test_config_stubs_return_none(facade_manager: Any) -> None:
    assert facade_manager.configure_message_bus() is None
    assert facade_manager.configure_event_emission() is None
    assert facade_manager.configure_event_system() is None
    assert facade_manager.configure_domain_bookkeeping(MagicMock()) is None
    assert facade_manager.start_end_to_end_tracking() is None
    assert facade_manager.start_health_checks() is None
    assert facade_manager.emit_cross_domain_event({}) is None


def test_emit_cascade_preserves_correlation(facade_manager: Any) -> None:
    source_event = {
        "domain": "features",
        "event_type": "feature_computed",
        "correlation_id": "test-correlation-123",
        "instrument_id": "BTC.USD",
        "ts_event": 1_000_000_000,
        "event_id": "evt_001",
        "payload": {"feature_name": "price_sma_20"},
    }

    result = facade_manager.emit_cascade(source_event, "model", delay_ns=100)

    assert result["correlation_id"] == source_event["correlation_id"]
    assert result["domain"] == "model"


def test_collect_observability_dataframes_keys(facade_manager: Any) -> None:
    dfs = facade_manager.collect_observability_dataframes()
    assert set(dfs.keys()) == {"latency", "metrics", "correlation", "health"}


def test_get_observability_async_status_defaults(facade_manager: Any) -> None:
    status = facade_manager.get_observability_async_status()
    assert status["running"] is False
    assert status["queue_size"] == 0


def test_public_attributes_exist(facade_manager: Any) -> None:
    for attr in [
        "feature_store",
        "model_store",
        "strategy_store",
        "data_store",
        "feature_registry",
        "model_registry",
        "strategy_registry",
        "data_registry",
        "partition_manager",
        "db_connection",
        "auto_start_postgres",
        "auto_migrate",
    ]:
        assert hasattr(facade_manager, attr)
