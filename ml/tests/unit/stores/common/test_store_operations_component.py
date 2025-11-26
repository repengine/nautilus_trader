#!/usr/bin/env python3

"""
Unit tests for StoreOperationsComponent (Phase 2.4.6).

Tests health monitoring, metrics aggregation, graceful shutdown, progressive
fallback chains, circuit breaker logic, and resource cleanup.

"""

import time
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from ml.stores.common.store_operations import StoreOperationsComponent


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def mock_feature_store():
    """Create mock FeatureStore."""
    store = MagicMock()
    store.get_health_status.return_value = {"healthy": True}
    store.get_performance_metrics.return_value = {"write_count": 100.0}
    store.close = MagicMock()
    return store


@pytest.fixture
def mock_model_store():
    """Create mock ModelStore."""
    store = MagicMock()
    store.get_health_status.return_value = {"healthy": True}
    store.get_performance_metrics.return_value = {"prediction_count": 50.0}
    store.close = MagicMock()
    return store


@pytest.fixture
def mock_strategy_store():
    """Create mock StrategyStore."""
    store = MagicMock()
    store.get_health_status.return_value = {"healthy": True}
    store.get_performance_metrics.return_value = {"signal_count": 25.0}
    store.close = MagicMock()
    return store


@pytest.fixture
def mock_earnings_store():
    """Create mock EarningsStore."""
    store = MagicMock()
    store.get_health_status.return_value = {"healthy": True}
    store.get_performance_metrics.return_value = {"earnings_count": 10.0}
    store.close = MagicMock()
    return store


@pytest.fixture
def mock_data_registry():
    """Create mock DataRegistry."""
    registry = MagicMock()
    registry.get_health_status.return_value = {"healthy": True}
    return registry


@pytest.fixture
def store_operations(
    mock_feature_store,
    mock_model_store,
    mock_strategy_store,
    mock_earnings_store,
    mock_data_registry,
):
    """Create StoreOperationsComponent with mock stores."""
    return StoreOperationsComponent(
        connection_string="postgresql://localhost/test",
        feature_store=mock_feature_store,
        model_store=mock_model_store,
        strategy_store=mock_strategy_store,
        earnings_store=mock_earnings_store,
        data_registry=mock_data_registry,
    )


# =========================================================================
# Health Check Tests
# =========================================================================


def test_health_check_all_healthy(store_operations):
    """Test health check when all components are healthy."""
    health = store_operations.health_check()

    assert health["healthy"] is True
    assert "components" in health
    assert health["components"]["feature_store"]["status"] == "healthy"
    assert health["components"]["model_store"]["status"] == "healthy"
    assert health["components"]["strategy_store"]["status"] == "healthy"
    assert health["components"]["earnings_store"]["status"] == "healthy"
    assert health["fallback_active"] is False
    assert health["fallback_reason"] is None
    assert health["circuit_breakers_open"] == []


def test_health_check_with_failures(mock_feature_store):
    """Test health check when some components fail."""
    # Make feature store unhealthy
    mock_feature_store.get_health_status.return_value = {"healthy": False}

    operations = StoreOperationsComponent(
        connection_string="postgresql://localhost/test",
        feature_store=mock_feature_store,
        model_store=None,  # Missing store
        strategy_store=None,
        earnings_store=None,
    )

    health = operations.health_check()

    # Overall health should be False if any critical component is unhealthy
    assert health["healthy"] is False
    assert health["components"]["feature_store"]["status"] == "degraded"
    assert health["components"]["model_store"]["status"] == "unavailable"
    assert health["components"]["strategy_store"]["status"] == "unavailable"


# =========================================================================
# Metrics Tests
# =========================================================================


def test_get_metrics_aggregation(store_operations):
    """Test metrics aggregation from all components."""
    # Record some operations
    store_operations._record_operation_latency("write_ingestion", 10.5)
    store_operations._record_operation_latency("write_features", 5.2)
    store_operations._record_operation_latency("read_features", 2.1)

    metrics = store_operations.get_metrics()

    assert "operation_count_total" in metrics
    assert metrics["operation_count_total"] == 3.0
    assert "avg_operation_latency_ms" in metrics
    assert metrics["avg_operation_latency_ms"] > 0.0
    assert "p95_operation_latency_ms" in metrics
    assert metrics["fallback_active"] == 0.0
    assert metrics["circuit_breakers_open"] == 0.0

    # Check component metrics
    assert "feature_store_write_count" in metrics
    assert metrics["feature_store_write_count"] == 100.0
    assert "model_store_prediction_count" in metrics
    assert metrics["model_store_prediction_count"] == 50.0


# =========================================================================
# Shutdown Tests
# =========================================================================


def test_close_graceful_shutdown(
    store_operations,
    mock_feature_store,
    mock_model_store,
    mock_strategy_store,
    mock_earnings_store,
):
    """Test graceful shutdown closes all stores."""
    store_operations.close()

    # Verify all stores were closed
    mock_feature_store.close.assert_called_once()
    mock_model_store.close.assert_called_once()
    mock_strategy_store.close.assert_called_once()
    mock_earnings_store.close.assert_called_once()


# =========================================================================
# Store Initialization Tests
# =========================================================================


@patch("ml.stores.earnings_store.EarningsStore")
def test_initialize_stores_success(mock_earnings_store_class):
    """Test successful store initialization."""
    operations = StoreOperationsComponent(
        connection_string="postgresql://localhost/test",
    )

    operations._initialize_stores()

    # Should create earnings store if it was None
    # (Test depends on implementation details)


@patch("ml.stores.earnings_store.EarningsStore", side_effect=Exception("DB connection failed"))
def test_initialize_stores_with_fallback(mock_earnings_store_class):
    """Test store initialization with fallback when primary fails."""
    operations = StoreOperationsComponent(
        connection_string="postgresql://localhost/test",
    )

    with patch.object(operations, "_try_file_earnings_store", return_value=MagicMock()):
        operations._initialize_stores()

    # Fallback should be activated
    assert operations._fallback_active is True


# =========================================================================
# Fallback Chain Tests
# =========================================================================


def test_initialize_fallback_chain(store_operations):
    """Test fallback chain initialization."""
    store_operations._initialize_fallback_chain()

    # Verify circuit breakers initialized
    assert "feature_store" in store_operations._circuit_breaker_failures
    assert store_operations._circuit_breaker_failures["feature_store"] == 0
    assert store_operations._circuit_breaker_open["feature_store"] is False


def test_activate_fallback_with_reason(store_operations):
    """Test fallback activation with reason."""
    assert store_operations._fallback_active is False

    store_operations._activate_fallback("connection_lost")

    assert store_operations._fallback_active is True
    assert store_operations._fallback_reason == "connection_lost"
    assert store_operations._primary_connection_lost_at is not None


def test_restore_primary_success(store_operations):
    """Test successful primary connection restoration."""
    # Activate fallback first
    store_operations._activate_fallback("test_failure")
    assert store_operations._fallback_active is True

    # Mock successful connection restoration
    with patch("ml.core.db_engine.EngineManager") as mock_engine_manager:
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute = MagicMock()
        mock_engine.connect.return_value = mock_conn
        mock_engine_manager.get_engine.return_value = mock_engine

        restored = store_operations._restore_primary()

    assert restored is True
    assert store_operations._fallback_active is False
    assert store_operations._fallback_reason is None


def test_restore_primary_failure(store_operations):
    """Test primary restoration failure."""
    # Activate fallback first
    store_operations._activate_fallback("test_failure")

    # Mock failed connection restoration
    with patch("ml.core.db_engine.EngineManager") as mock_engine_manager:
        mock_engine_manager.get_engine.side_effect = Exception("Connection failed")

        restored = store_operations._restore_primary()

    assert restored is False
    assert store_operations._fallback_active is True


# =========================================================================
# Metrics Emission Tests
# =========================================================================


def test_emit_health_metric(store_operations):
    """Test health metric emission."""
    # Should not raise
    store_operations._emit_health_metric("healthy", "feature_store")


def test_record_operation_latency(store_operations):
    """Test operation latency recording."""
    store_operations._record_operation_latency("write_ingestion", 10.5)

    assert "write_ingestion" in store_operations._operation_latencies
    assert len(store_operations._operation_latencies["write_ingestion"]) == 1
    assert store_operations._operation_latencies["write_ingestion"][0] == 10.5
    assert store_operations._operation_counts["write_ingestion"] == 1

    # Record more operations
    for i in range(10):
        store_operations._record_operation_latency("write_ingestion", 5.0 + i)

    assert len(store_operations._operation_latencies["write_ingestion"]) == 11
    assert store_operations._operation_counts["write_ingestion"] == 11


# =========================================================================
# Circuit Breaker Tests
# =========================================================================


def test_circuit_breaker_opens_after_threshold(store_operations):
    """Test circuit breaker opens after threshold failures."""
    store_operations._initialize_fallback_chain()

    # Record failures
    for _ in range(5):
        store_operations._record_circuit_breaker_failure("feature_store")

    # Circuit breaker should be open
    assert store_operations._check_circuit_breaker("feature_store") is True
    assert store_operations._circuit_breaker_open["feature_store"] is True


def test_circuit_breaker_reset(store_operations):
    """Test circuit breaker reset."""
    store_operations._initialize_fallback_chain()

    # Open circuit breaker
    for _ in range(5):
        store_operations._record_circuit_breaker_failure("feature_store")

    assert store_operations._check_circuit_breaker("feature_store") is True

    # Reset circuit breaker
    store_operations._reset_circuit_breaker("feature_store")

    assert store_operations._check_circuit_breaker("feature_store") is False
    assert store_operations._circuit_breaker_failures["feature_store"] == 0
