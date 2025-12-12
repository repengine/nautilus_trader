"""
Unit tests for SchedulerInitComponent.

Tests extracted initialization logic from DataScheduler:
- Connection string resolution (resolve_connection)
- DataRegistry initialization with Postgres/JSON fallback (init_data_registry)
- FeatureStore initialization (init_feature_store)

Test count: 12
Coverage target: 95%

"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from ml.data.common.scheduler_init import SchedulerInitComponent
from ml.data.common.scheduler_init import SchedulerInitProtocol
from ml.tests.utils.db import build_postgres_url


DEFAULT_CONNECTION = build_postgres_url(
    user="user",
    password="pass",
    host="host",
    database="db",
)
LEGACY_CONNECTION = build_postgres_url(
    user="legacy",
    password="pass",
    host="host",
    database="db",
)
EXPLICIT_CONNECTION = build_postgres_url(
    user="explicit",
    password="pass",
    host="host",
    database="db",
)
ENV_CONNECTION = build_postgres_url(
    user="env",
    password="pass",
    host="envhost",
    database="envdb",
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def component() -> SchedulerInitComponent:
    """Create a SchedulerInitComponent instance for testing."""
    return SchedulerInitComponent()


@pytest.fixture
def minimal_config() -> Any:
    """Create minimal SchedulerConfig for testing."""
    from ml.config.scheduler_config import SchedulerConfig

    return SchedulerConfig(
        symbols=["SPY.XNAS"],
        retention_days=30,
        feature_store_enabled=False,
    )


@pytest.fixture
def config_with_feature_store_connection() -> Any:
    """Create SchedulerConfig with feature_store_connection."""
    from ml.config.scheduler_config import SchedulerConfig

    return SchedulerConfig(
        symbols=["SPY.XNAS"],
        feature_store_enabled=True,
        feature_store_connection=DEFAULT_CONNECTION,
    )


@pytest.fixture
def config_with_connection_string() -> Any:
    """Create SchedulerConfig with connection_string (legacy)."""
    from ml.config.scheduler_config import SchedulerConfig

    return SchedulerConfig(
        symbols=["SPY.XNAS"],
        feature_store_enabled=True,
        connection_string=LEGACY_CONNECTION,
    )


@pytest.fixture
def mock_feature_engineer() -> MagicMock:
    """Create mock FeatureEngineer with config attribute."""
    from ml.features.engineering import FeatureConfig

    engineer = MagicMock()
    engineer.config = FeatureConfig()
    return engineer


# -----------------------------------------------------------------------------
# Connection Resolution Tests (4 tests)
# -----------------------------------------------------------------------------


class TestResolveConnection:
    """Tests for resolve_connection method."""

    def test_resolve_connection_from_parameter(
        self,
        component: SchedulerInitComponent,
        minimal_config: Any,
    ) -> None:
        """Test connection resolved from explicit parameter."""
        connection = EXPLICIT_CONNECTION

        result = component.resolve_connection(minimal_config, connection)

        assert result == connection

    def test_resolve_connection_from_config_feature_store_connection(
        self,
        component: SchedulerInitComponent,
        config_with_feature_store_connection: Any,
    ) -> None:
        """Test connection resolved from config.feature_store_connection."""
        result = component.resolve_connection(
            config_with_feature_store_connection,
            None,
        )

        assert result == DEFAULT_CONNECTION

    def test_resolve_connection_from_config_connection_string(
        self,
        component: SchedulerInitComponent,
        config_with_connection_string: Any,
    ) -> None:
        """Test connection resolved from config.connection_string (legacy)."""
        result = component.resolve_connection(
            config_with_connection_string,
            None,
        )

        assert result == LEGACY_CONNECTION

    def test_resolve_connection_none_when_all_missing(
        self,
        component: SchedulerInitComponent,
        minimal_config: Any,
    ) -> None:
        """Test connection is None when no sources available."""
        result = component.resolve_connection(minimal_config, None)

        assert result is None


# -----------------------------------------------------------------------------
# DataRegistry Initialization Tests (3 tests)
# -----------------------------------------------------------------------------


class TestInitDataRegistry:
    """Tests for init_data_registry method."""

    def test_init_data_registry_postgres_backend(
        self,
        component: SchedulerInitComponent,
    ) -> None:
        """Test DataRegistry initialized with Postgres backend when connection provided."""
        # Patch at the source module where DataRegistry is defined
        with patch(
            "ml.registry.data_registry.DataRegistry"
        ) as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry_class.return_value = mock_registry

            result = component.init_data_registry(
                DEFAULT_CONNECTION
            )

            assert result is mock_registry
            # Verify Postgres backend was requested
            call_args = mock_registry_class.call_args
            assert call_args is not None
            persistence_config = call_args.kwargs.get("persistence_config")
            assert persistence_config is not None
            assert persistence_config.backend.value == "postgres"

    def test_init_data_registry_json_fallback(
        self,
        component: SchedulerInitComponent,
    ) -> None:
        """Test DataRegistry uses JSON backend when no connection provided."""
        with patch(
            "ml.registry.data_registry.DataRegistry"
        ) as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry_class.return_value = mock_registry

            result = component.init_data_registry(None)

            assert result is mock_registry
            # Verify JSON backend was requested
            call_args = mock_registry_class.call_args
            assert call_args is not None
            persistence_config = call_args.kwargs.get("persistence_config")
            assert persistence_config is not None
            assert persistence_config.backend.value == "json"

    def test_init_data_registry_failure_returns_none(
        self,
        component: SchedulerInitComponent,
    ) -> None:
        """Test init_data_registry returns None on failure (no exception raised)."""
        with patch(
            "ml.registry.data_registry.DataRegistry"
        ) as mock_registry_class:
            mock_registry_class.side_effect = RuntimeError("DB connection failed")

            result = component.init_data_registry(
                DEFAULT_CONNECTION
            )

            assert result is None


# -----------------------------------------------------------------------------
# FeatureStore Initialization Tests (5 tests)
# -----------------------------------------------------------------------------


class TestInitFeatureStore:
    """Tests for init_feature_store method."""

    def test_init_feature_store_success(
        self,
        component: SchedulerInitComponent,
        mock_feature_engineer: MagicMock,
    ) -> None:
        """Test FeatureStore initialized successfully."""
        from ml.config.scheduler_config import SchedulerConfig

        config = SchedulerConfig(feature_store_enabled=True)

        with patch("ml.stores.feature_store.FeatureStore") as mock_fs_class:
            mock_store = MagicMock()
            mock_fs_class.return_value = mock_store

            result = component.init_feature_store(
                config,
                DEFAULT_CONNECTION,
                mock_feature_engineer,
            )

            assert result is mock_store
            mock_fs_class.assert_called_once()

    def test_init_feature_store_from_env(
        self,
        component: SchedulerInitComponent,
        mock_feature_engineer: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test FeatureStore uses NAUTILUS_DB_CONNECTION env var."""
        from ml.config.scheduler_config import SchedulerConfig

        config = SchedulerConfig(feature_store_enabled=True)
        env_connection = ENV_CONNECTION
        monkeypatch.setenv("NAUTILUS_DB_CONNECTION", env_connection)

        with patch("ml.stores.feature_store.FeatureStore") as mock_fs_class:
            mock_store = MagicMock()
            mock_fs_class.return_value = mock_store

            result = component.init_feature_store(
                config,
                None,  # No explicit connection
                mock_feature_engineer,
            )

            assert result is mock_store
            # Verify env var connection was used
            call_args = mock_fs_class.call_args
            assert call_args is not None
            assert call_args.kwargs["connection_string"] == env_connection

    def test_init_feature_store_disabled_returns_none(
        self,
        component: SchedulerInitComponent,
        mock_feature_engineer: MagicMock,
    ) -> None:
        """Test init_feature_store returns None when disabled in config."""
        from ml.config.scheduler_config import SchedulerConfig

        config = SchedulerConfig(feature_store_enabled=False)

        result = component.init_feature_store(
            config,
            DEFAULT_CONNECTION,
            mock_feature_engineer,
        )

        assert result is None

    def test_init_feature_store_no_engineer_returns_none(
        self,
        component: SchedulerInitComponent,
    ) -> None:
        """Test init_feature_store returns None when no feature engineer provided."""
        from ml.config.scheduler_config import SchedulerConfig

        config = SchedulerConfig(feature_store_enabled=True)

        result = component.init_feature_store(
            config,
            DEFAULT_CONNECTION,
            None,  # No feature engineer
        )

        assert result is None

    def test_init_feature_store_failure_returns_none(
        self,
        component: SchedulerInitComponent,
        mock_feature_engineer: MagicMock,
    ) -> None:
        """Test init_feature_store returns None on failure (no exception raised)."""
        from ml.config.scheduler_config import SchedulerConfig

        config = SchedulerConfig(feature_store_enabled=True)

        with patch("ml.stores.feature_store.FeatureStore") as mock_fs_class:
            mock_fs_class.side_effect = RuntimeError("FeatureStore init failed")

            result = component.init_feature_store(
                config,
                DEFAULT_CONNECTION,
                mock_feature_engineer,
            )

            assert result is None


# -----------------------------------------------------------------------------
# Protocol Compliance Test
# -----------------------------------------------------------------------------


class TestProtocolCompliance:
    """Tests for protocol compliance."""

    def test_component_satisfies_protocol(
        self,
        component: SchedulerInitComponent,
    ) -> None:
        """Test SchedulerInitComponent satisfies SchedulerInitProtocol."""
        # Protocol compliance is structural in Python - verify methods exist
        assert hasattr(component, "resolve_connection")
        assert hasattr(component, "init_data_registry")
        assert hasattr(component, "init_feature_store")
        assert callable(component.resolve_connection)
        assert callable(component.init_data_registry)
        assert callable(component.init_feature_store)
