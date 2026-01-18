#!/usr/bin/env python3
"""
Store-related fixtures and helpers for ML tests.

This module centralizes shared store bundles, DataStore toggle helpers, and mock
persistence utilities so pytest discovery remains fast while keeping legacy fixtures
available during the transition to isolated bundles.

"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Protocol, cast
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

if TYPE_CHECKING:
    from ml.tests.fixtures.database_fixtures import TestDatabase


@pytest.fixture(params=[True])
def datastore_variant(request: pytest.FixtureRequest) -> bool:
    """
    Parameterize tests across DataStore implementations (Legacy removed, always True).
    """
    return True


@pytest.fixture
def component_data_store_factory() -> Callable[..., ContextManager[ModuleType]]:
    """
    Provide a context manager that yields the component DataStore module.

    Legacy DataStore implementations are no longer supported, so the factory
    always yields the component facade regardless of the requested mode.
    """

    @contextmanager
    def _factory(*, use_component: bool = True) -> Generator[ModuleType, None, None]:
        del use_component
        module = import_module("ml.stores.data_store")
        yield module

    return _factory


@pytest.fixture
def datastore_module(
    datastore_variant: bool,
    component_data_store_factory: Callable[..., ContextManager[ModuleType]],
) -> Generator[ModuleType, None, None]:
    """
    Provide the active DataStore module.
    """
    with component_data_store_factory(use_component=datastore_variant) as module:
        yield module


@pytest.fixture
def patch_datastore(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """
    No-op fixture for backward compatibility during refactor.
    """
    yield


@pytest.fixture
def use_component_datastore(
    datastore_variant: bool,
    patch_datastore: None,
) -> bool:
    """
    Expose whether the component DataStore implementation is active for the test.
    """
    return True


@pytest.fixture
def datastore_class(datastore_module: ModuleType) -> type[Any]:
    """
    Provide the active DataStore class for tests needing direct instantiation.
    """
    return cast(type[Any], getattr(datastore_module, "DataStore"))


@dataclass(slots=True)
class ModuleStoreBundle:
    """
    Bundle of shared store instances for Postgres-backed tests.
    """

    feature_store: Any
    model_store: Any
    strategy_store: Any
    persistence_manager: MagicMock
    engine: Engine


def _truncate_store_tables(engine: Engine) -> None:
    """
    Truncate primary store tables to isolate tests.
    """

    tables = (
        "ml_feature_values",
        "ml_model_predictions",
        "ml_strategy_signals",
    )
    with engine.begin() as conn:
        for table in tables:
            try:
                conn.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
            except Exception as exc:
                logging.getLogger(__name__).debug(
                    "Table truncation failed: %s",
                    exc,
                    exc_info=True,
                )


@pytest.fixture(scope="function")
def module_store_bundle(
    cloned_test_database: str,
) -> Generator[ModuleStoreBundle, None, None]:
    """
    Create shared Feature/Model/Strategy stores backed by PostgreSQL.
    """

    from ml.core.db_engine import EngineManager as _EM
    from ml.stores.feature_store import FeatureStore as _FeatureStore
    from ml.stores.model_store import ModelStore as _ModelStore
    from ml.stores.strategy_store import StrategyStore as _StrategyStore

    # Clear singleton engine cache before creating stores
    _EM.dispose_all()

    persistence_manager = MagicMock()
    persistence_manager.connection_string = cloned_test_database
    persistence_manager.session = MagicMock()

    store_kwargs: dict[str, Any] = {
        "connection_string": cloned_test_database,
        "batch_size": 1,
        "flush_interval_seconds": 1.0,
        "persistence_manager": persistence_manager,
    }

    engine = _EM.get_engine(cloned_test_database)
    with engine.begin() as conn:
        try:
            conn.execute(
                text(
                    """
CREATE OR REPLACE FUNCTION ensure_partition_exists()
RETURNS TRIGGER AS $$
BEGIN
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
                    """,
                ),
            )
            conn.execute(
                text(
                    """
CREATE OR REPLACE FUNCTION ml_registry.ensure_partition_exists()
RETURNS TRIGGER AS $$
BEGIN
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
                    """,
                ),
            )
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Partition helpers setup failed: %s",
                exc,
                exc_info=True,
            )

    feature_store = _FeatureStore(**store_kwargs)
    model_store = _ModelStore(**store_kwargs)
    strategy_store = _StrategyStore(**store_kwargs)

    bundle = ModuleStoreBundle(
        feature_store=feature_store,
        model_store=model_store,
        strategy_store=strategy_store,
        persistence_manager=persistence_manager,
        engine=engine,
    )

    try:
        yield bundle
    finally:
        for store in (feature_store, model_store, strategy_store):
            try:
                store.flush()
            except Exception as exc:
                logging.getLogger(__name__).debug(
                    "Store flush failed: %s",
                    exc,
                    exc_info=True,
                )
            timer = getattr(store, "_timer", None)
            if timer is not None:
                try:
                    timer.cancel()
                    timer.join()
                except Exception as exc:
                    logging.getLogger(__name__).debug(
                        "Timer cancellation failed: %s",
                        exc,
                        exc_info=True,
                    )
            if hasattr(store, "reset"):
                try:
                    store.reset()
                except Exception as exc:
                    logging.getLogger(__name__).debug(
                        "Store reset failed: %s",
                        exc,
                        exc_info=True,
                    )
            if hasattr(store, "close"):
                try:
                    store.close()
                except Exception as exc:
                    logging.getLogger(__name__).debug(
                        "Store close failed: %s",
                        exc,
                        exc_info=True,
                    )

        persistence_manager.reset_mock()
        _EM.dispose_all()


@pytest.fixture
def store_bundle(module_store_bundle: ModuleStoreBundle) -> ModuleStoreBundle:
    """
    Reset shared stores before each test and return the bundle.
    """

    for store in (
        module_store_bundle.feature_store,
        module_store_bundle.model_store,
        module_store_bundle.strategy_store,
    ):
        try:
            store.flush()
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Failed to flush store before test: %s",
                exc,
                exc_info=True,
            )
    _truncate_store_tables(module_store_bundle.engine)
    return module_store_bundle


@pytest.fixture(scope="function")
def fresh_store_bundle(
    cloned_test_database: str,
) -> Generator[ModuleStoreBundle, None, None]:
    """
    Provide fresh store instances per test with complete isolation.
    """

    from ml.core.db_engine import EngineManager as _EM
    from ml.stores.feature_store import FeatureStore as _FeatureStore
    from ml.stores.model_store import ModelStore as _ModelStore
    from ml.stores.strategy_store import StrategyStore as _StrategyStore

    logger = logging.getLogger(__name__)

    persistence_manager = MagicMock()
    persistence_manager.connection_string = cloned_test_database
    persistence_manager.session = MagicMock()

    engine: Engine | None = None
    try:
        engine = _EM.get_engine(cloned_test_database)
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE OR REPLACE FUNCTION create_partitions_if_needed()
                    RETURNS trigger AS $$
                    BEGIN RETURN NEW; END;
                    $$ LANGUAGE plpgsql;
                    """,
                ),
            )
    except Exception as exc:
        logger.debug(
            "fresh_store_bundle: partition function setup failed: %s",
            exc,
            exc_info=True,
        )

    store_kwargs: dict[str, Any] = {
        "connection_string": cloned_test_database,
        "batch_size": 1,
        "flush_interval_seconds": 1.0,
        "persistence_manager": persistence_manager,
    }

    feature_store = _FeatureStore(**store_kwargs)
    model_store = _ModelStore(**store_kwargs)
    strategy_store = _StrategyStore(**store_kwargs)

    bundle = ModuleStoreBundle(
        feature_store=feature_store,
        model_store=model_store,
        strategy_store=strategy_store,
        persistence_manager=persistence_manager,
        engine=engine,
    )

    try:
        yield bundle
    finally:
        try:
            bundle.feature_store.flush()
            bundle.model_store.flush()
            bundle.strategy_store.flush()
        except Exception as exc:
            logger.debug(
                "fresh_store_bundle: flush failed during cleanup: %s",
                exc,
                exc_info=True,
            )

        try:
            if hasattr(bundle.feature_store, "_timer"):
                bundle.feature_store._timer.cancel()
                bundle.feature_store._timer.join()
            if hasattr(bundle.model_store, "_timer"):
                bundle.model_store._timer.cancel()
                bundle.model_store._timer.join()
            if hasattr(bundle.strategy_store, "_timer"):
                bundle.strategy_store._timer.cancel()
                bundle.strategy_store._timer.join()
        except Exception as exc:
            logger.debug(
                "fresh_store_bundle: timer cancellation failed: %s",
                exc,
                exc_info=True,
            )

        try:
            if hasattr(bundle.feature_store, "reset"):
                bundle.feature_store.reset()
            if hasattr(bundle.model_store, "reset"):
                bundle.model_store.reset()
            if hasattr(bundle.strategy_store, "reset"):
                bundle.strategy_store.reset()
        except Exception as exc:
            logger.debug(
                "fresh_store_bundle: state reset failed: %s",
                exc,
                exc_info=True,
            )

        try:
            if hasattr(bundle.feature_store, "close"):
                bundle.feature_store.close()
            if hasattr(bundle.model_store, "close"):
                bundle.model_store.close()
            if hasattr(bundle.strategy_store, "close"):
                bundle.strategy_store.close()
        except Exception as exc:
            logger.debug(
                "fresh_store_bundle: connection close failed: %s",
                exc,
                exc_info=True,
            )

        try:
            if engine is not None:
                _truncate_store_tables(engine)
        except Exception as exc:
            logger.debug(
                "fresh_store_bundle: table truncation failed: %s",
                exc,
                exc_info=True,
            )


@pytest.fixture
def data_processor(module_test_database: TestDatabase) -> Any:
    """
    Provide a DataProcessor bound to the shared PostgreSQL database.
    """

    from ml.stores.data_processor import DataProcessor as _DataProcessor

    return _DataProcessor(
        connection_string=module_test_database.connection_string,
        outlier_threshold=3.0,
        staleness_threshold_seconds=60,
    )


@pytest.fixture
def feature_store(store_bundle: ModuleStoreBundle) -> Any:
    """
    Provide a reset FeatureStore instance for tests.
    """

    return store_bundle.feature_store


@pytest.fixture
def model_store(store_bundle: ModuleStoreBundle) -> Any:
    """
    Provide a reset ModelStore instance for tests.
    """

    return store_bundle.model_store


@pytest.fixture
def strategy_store(store_bundle: ModuleStoreBundle) -> Any:
    """
    Provide a reset StrategyStore instance for tests.
    """

    return store_bundle.strategy_store


@pytest.fixture
def mock_persistence_manager(store_bundle: ModuleStoreBundle) -> MagicMock:
    """
    Return the shared persistence manager with call history reset.
    """

    store_bundle.persistence_manager.reset_mock()
    return store_bundle.persistence_manager


@pytest.fixture
def data_store_session(test_database: TestDatabase) -> Generator[Session, None, None]:
    """
    Backward-compatible session fixture bound to TestDatabase for store tests.
    """

    with test_database.get_session() as session:
        yield session


@pytest.fixture
def store_integration_metrics_database(
    test_database: TestDatabase,
    clean_postgres_db: None,
) -> Generator[TestDatabase, None, None]:
    """
    Seed deterministic metrics data for StoreIntegrationService integration tests.
    """

    now_ns = time.time_ns()
    five_minutes_ns = 300 * 1_000_000_000

    with test_database.engine.begin() as conn:
        for table_name in (
            "ml_positions",
            "ml_data_events",
            "ml_model_predictions",
            "ml_strategy_signals",
            "ml_risk_limits",
        ):
            conn.execute(text(f"DELETE FROM {table_name}"))

        conn.execute(
            text(
                """
                INSERT INTO ml_strategy_signals (
                    strategy_id,
                    instrument_id,
                    ts_event,
                    ts_init,
                    signal_type,
                    strength,
                    model_predictions,
                    risk_metrics,
                    execution_params,
                    is_live
                ) VALUES
                    ('strat-alpha', 'EUR/USD', :ts1, :ts1, 'BUY', 0.6, '{}'::jsonb, '{}'::jsonb, '{}'::jsonb, TRUE),
                    ('strat-alpha', 'EUR/USD', :ts2, :ts2, 'SELL', -0.3, '{}'::jsonb, '{}'::jsonb, '{}'::jsonb, TRUE),
                    ('strat-beta', 'AAPL', :ts3, :ts3, 'BUY', 0.4, '{}'::jsonb, '{}'::jsonb, '{}'::jsonb, TRUE)
                """,
            ),
            {
                "ts1": now_ns - 2_000_000_000,
                "ts2": now_ns - 1_500_000_000,
                "ts3": now_ns - 1_000_000_000,
            },
        )

        conn.execute(
            text(
                """
                INSERT INTO ml_model_predictions (
                    model_id,
                    instrument_id,
                    ts_event,
                    ts_init,
                    prediction,
                    confidence,
                    features_used,
                    inference_time_ms,
                    is_live
                ) VALUES
                    ('model-alpha', 'EUR/USD', :ts1, :ts1, 0.45, 0.9, '{}'::jsonb, 5.0, TRUE),
                    ('model-beta', 'AAPL', :ts2, :ts2, -0.12, 0.7, '{}'::jsonb, 8.0, FALSE)
                """,
            ),
            {
                "ts1": now_ns - 2_000_000_000,
                "ts2": now_ns - 1_500_000_000,
            },
        )

        conn.execute(
            text(
                """
                INSERT INTO ml_positions (
                    strategy_id,
                    instrument_id,
                    quantity,
                    side,
                    entry_price,
                    current_price,
                    unrealized_pnl,
                    realized_pnl,
                    position_value,
                    exposure,
                    var_95,
                    entry_time,
                    last_update
                ) VALUES
                    ('strat-alpha', 'EUR/USD', 1.0, 'LONG', 100.0, 110.0, 25.0, 15.0, 10000.0, 5000.0, 100.0, :ts1, :ts1),
                    ('strat-beta', 'AAPL', 2.0, 'LONG', 200.0, 195.0, -5.0, 20.0, 15000.0, 8000.0, 200.0, :ts2, :ts2)
                """,
            ),
            {
                "ts1": now_ns - 2_000_000_000,
                "ts2": now_ns - 1_500_000_000,
            },
        )

        conn.execute(
            text(
                """
                INSERT INTO ml_risk_limits (
                    strategy_id,
                    max_exposure,
                    max_position_size,
                    max_positions,
                    max_drawdown,
                    max_var,
                    max_leverage,
                    max_daily_trades,
                    max_order_size,
                    is_active
                ) VALUES ('strat-alpha', 100000.0, 50000.0, 10, 0.15, 20000.0, 5.0, 50, 10000.0, TRUE)
                ON CONFLICT (strategy_id) DO UPDATE SET max_drawdown = EXCLUDED.max_drawdown
                """,
            ),
        )

        conn.execute(
            text(
                """
                INSERT INTO ml_data_events (
                    dataset_id,
                    instrument_id,
                    stage,
                    source,
                    run_id,
                    ts_min,
                    ts_max,
                    ts_event,
                    count,
                    seq_min,
                    seq_max,
                    status
                ) VALUES
                    ('EQUS.MINI.BARS', 'AAPL', 'INGESTED', 'live', 'run-bars', :ts_base, :ts_base, :ts_now, 600, NULL, NULL, 'success'),
                    ('EQUS.MINI.QUOTES', 'AAPL', 'INGESTED', 'live', 'run-quotes', :ts_base, :ts_base, :ts_now, 300, NULL, NULL, 'success'),
                    ('EQUS.MINI.BOOK', 'AAPL', 'INGESTED', 'live', 'run-book', :ts_base, :ts_base, :ts_now, 150, NULL, NULL, 'failed')
                """,
            ),
            {"ts_base": now_ns - five_minutes_ns, "ts_now": now_ns - 1_000_000_000},
        )

    yield test_database


@pytest.fixture
def component_feature_store(
    cloned_test_database: str,
    real_engine_manager: None,
) -> Generator[Any, None, None]:
    """
    Provide a ComponentFeatureStore instance backed by the shared PostgreSQL database.
    """

    from ml.stores import ComponentFeatureStore as _ComponentFeatureStore

    store = _ComponentFeatureStore(connection_string=cloned_test_database)
    logger = logging.getLogger(__name__)

    try:
        yield store
    finally:
        try:
            store.flush()
        except Exception as exc:
            logger.debug("ComponentFeatureStore flush failed: %s", exc, exc_info=True)
        if hasattr(store, "close"):
            try:
                store.close()
            except Exception as exc:
                logger.debug("ComponentFeatureStore close failed: %s", exc, exc_info=True)


__all__ = [
    "ModuleStoreBundle",
    "component_data_store_factory",
    "component_feature_store",
    "data_processor",
    "data_store_session",
    "datastore_class",
    "datastore_module",
    "datastore_variant",
    "feature_store",
    "fresh_store_bundle",
    "mock_persistence_manager",
    "model_store",
    "module_store_bundle",
    "patch_datastore",
    "store_bundle",
    "store_integration_metrics_database",
    "strategy_store",
    "use_component_datastore",
]
