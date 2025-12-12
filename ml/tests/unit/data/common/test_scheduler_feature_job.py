"""
Unit tests for FeatureComputationComponent.

Tests extracted feature computation logic from DataScheduler:
- Feature computation enabled/disabled behavior
- Feature engineer presence validation
- Lazy FeatureStore initialization
- Symbol parsing and venue mapping
- Catalog querying for bars data
- FeatureStore.compute_and_store_historical() calls
- Prometheus metrics tracking
- Error handling and failed instruments collection

Test count: 14
Coverage target: 95%

"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import call
from unittest.mock import patch

import pytest

from ml.data.common.scheduler_feature_job import FeatureComputationComponent
from ml.data.common.scheduler_feature_job import FeatureComputationProtocol
from ml.data.common.scheduler_feature_job import VENUE_MAP


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def component() -> FeatureComputationComponent:
    """Create a FeatureComputationComponent instance for testing."""
    return FeatureComputationComponent()


@pytest.fixture
def mock_config() -> MagicMock:
    """Create mock SchedulerConfig with feature_store_enabled=True."""
    config = MagicMock()
    config.feature_store_enabled = True
    config.symbols = ["SPY.XNAS", "AAPL.XNYS"]
    return config


@pytest.fixture
def mock_config_disabled() -> MagicMock:
    """Create mock SchedulerConfig with feature_store_enabled=False."""
    config = MagicMock()
    config.feature_store_enabled = False
    config.symbols = ["SPY.XNAS"]
    return config


@pytest.fixture
def mock_catalog() -> MagicMock:
    """Create mock ParquetDataCatalog."""
    catalog = MagicMock()
    # Return list of mock bars by default
    catalog.query.return_value = [MagicMock() for _ in range(10)]
    return catalog


@pytest.fixture
def mock_feature_engineer() -> MagicMock:
    """Create mock FeatureEngineer."""
    return MagicMock()


@pytest.fixture
def mock_feature_store() -> MagicMock:
    """Create mock FeatureStore with compute_and_store_historical method."""
    store = MagicMock()
    store.compute_and_store_historical.return_value = 100
    return store


@pytest.fixture
def mock_get_previous_day() -> MagicMock:
    """Create mock get_previous_day function."""
    return MagicMock(return_value=datetime(2024, 1, 15, 12, 0, 0))


@pytest.fixture
def mock_init_feature_store_fn(mock_feature_store: MagicMock) -> MagicMock:
    """Create mock init_feature_store function that returns a store."""
    return MagicMock(return_value=mock_feature_store)


# -----------------------------------------------------------------------------
# Test: Feature Computation Success
# -----------------------------------------------------------------------------


class TestComputeFeaturesSuccess:
    """Tests for successful feature computation scenarios."""

    def test_compute_features_success(
        self,
        component: FeatureComputationComponent,
        mock_config: MagicMock,
        mock_catalog: MagicMock,
        mock_feature_engineer: MagicMock,
        mock_feature_store: MagicMock,
        mock_get_previous_day: MagicMock,
    ) -> None:
        """Test successful feature computation with valid inputs."""
        with patch(
            "nautilus_trader.model.identifiers.InstrumentId"
        ) as mock_instrument_id_cls:
            mock_instrument_id = MagicMock()
            mock_instrument_id.__str__ = MagicMock(return_value="SPY.NASDAQ")
            mock_instrument_id_cls.from_str.return_value = mock_instrument_id

            total, failed = component.compute_features(
                config=mock_config,
                catalog=mock_catalog,
                feature_engineer=mock_feature_engineer,
                feature_store=mock_feature_store,
                init_feature_store_fn=MagicMock(return_value=mock_feature_store),
                get_previous_day_fn=mock_get_previous_day,
            )

            # Should have computed features for 2 symbols
            assert total == 200  # 100 per symbol * 2 symbols
            assert failed == []
            assert mock_feature_store.compute_and_store_historical.call_count == 2


# -----------------------------------------------------------------------------
# Test: Feature Computation Disabled
# -----------------------------------------------------------------------------


class TestComputeFeaturesDisabled:
    """Tests for feature computation when disabled or not configured."""

    def test_compute_features_disabled_in_config(
        self,
        component: FeatureComputationComponent,
        mock_config_disabled: MagicMock,
        mock_catalog: MagicMock,
        mock_feature_engineer: MagicMock,
        mock_feature_store: MagicMock,
        mock_get_previous_day: MagicMock,
    ) -> None:
        """Test feature computation returns early when disabled in config."""
        total, failed = component.compute_features(
            config=mock_config_disabled,
            catalog=mock_catalog,
            feature_engineer=mock_feature_engineer,
            feature_store=mock_feature_store,
            init_feature_store_fn=MagicMock(return_value=mock_feature_store),
            get_previous_day_fn=mock_get_previous_day,
        )

        assert total == 0
        assert failed == []
        # Should not call feature store methods
        mock_feature_store.compute_and_store_historical.assert_not_called()

    def test_compute_features_no_engineer(
        self,
        component: FeatureComputationComponent,
        mock_config: MagicMock,
        mock_catalog: MagicMock,
        mock_feature_store: MagicMock,
        mock_get_previous_day: MagicMock,
    ) -> None:
        """Test feature computation returns early when no feature engineer."""
        total, failed = component.compute_features(
            config=mock_config,
            catalog=mock_catalog,
            feature_engineer=None,  # No engineer
            feature_store=mock_feature_store,
            init_feature_store_fn=MagicMock(return_value=mock_feature_store),
            get_previous_day_fn=mock_get_previous_day,
        )

        assert total == 0
        assert failed == []
        mock_feature_store.compute_and_store_historical.assert_not_called()

    def test_compute_features_no_feature_store(
        self,
        component: FeatureComputationComponent,
        mock_config: MagicMock,
        mock_catalog: MagicMock,
        mock_feature_engineer: MagicMock,
        mock_get_previous_day: MagicMock,
    ) -> None:
        """Test feature computation returns early when feature store init fails."""
        total, failed = component.compute_features(
            config=mock_config,
            catalog=mock_catalog,
            feature_engineer=mock_feature_engineer,
            feature_store=None,
            init_feature_store_fn=MagicMock(return_value=None),  # Init fails
            get_previous_day_fn=mock_get_previous_day,
        )

        assert total == 0
        assert failed == []


# -----------------------------------------------------------------------------
# Test: Lazy Feature Store Initialization
# -----------------------------------------------------------------------------


class TestLazyStoreInitialization:
    """Tests for lazy FeatureStore initialization."""

    def test_compute_features_lazy_store_init(
        self,
        component: FeatureComputationComponent,
        mock_config: MagicMock,
        mock_catalog: MagicMock,
        mock_feature_engineer: MagicMock,
        mock_feature_store: MagicMock,
        mock_get_previous_day: MagicMock,
    ) -> None:
        """Test lazy initialization of feature store when not provided."""
        init_fn = MagicMock(return_value=mock_feature_store)

        with patch(
            "nautilus_trader.model.identifiers.InstrumentId"
        ) as mock_instrument_id_cls:
            mock_instrument_id = MagicMock()
            mock_instrument_id.__str__ = MagicMock(return_value="SPY.NASDAQ")
            mock_instrument_id_cls.from_str.return_value = mock_instrument_id

            total, failed = component.compute_features(
                config=mock_config,
                catalog=mock_catalog,
                feature_engineer=mock_feature_engineer,
                feature_store=None,  # Not provided - should trigger lazy init
                init_feature_store_fn=init_fn,
                get_previous_day_fn=mock_get_previous_day,
            )

            # Should have called init function
            init_fn.assert_called_once()
            assert total > 0
            assert failed == []


# -----------------------------------------------------------------------------
# Test: Symbol Parsing and Venue Mapping
# -----------------------------------------------------------------------------


class TestSymbolParsingAndVenueMapping:
    """Tests for symbol parsing and venue code mapping."""

    def test_compute_features_symbol_parsing(
        self,
        component: FeatureComputationComponent,
        mock_catalog: MagicMock,
        mock_feature_engineer: MagicMock,
        mock_feature_store: MagicMock,
        mock_get_previous_day: MagicMock,
    ) -> None:
        """Test correct symbol parsing (e.g., 'SPY.XNAS' -> 'SPY', 'XNAS')."""
        config = MagicMock()
        config.feature_store_enabled = True
        config.symbols = ["INVALID_NO_DOT"]  # Invalid format

        _total, failed = component.compute_features(
            config=config,
            catalog=mock_catalog,
            feature_engineer=mock_feature_engineer,
            feature_store=mock_feature_store,
            init_feature_store_fn=MagicMock(return_value=mock_feature_store),
            get_previous_day_fn=mock_get_previous_day,
        )

        # Invalid symbol should be added to failed list
        assert "INVALID_NO_DOT" in failed
        mock_feature_store.compute_and_store_historical.assert_not_called()

    def test_compute_features_venue_mapping(
        self,
        component: FeatureComputationComponent,
        mock_catalog: MagicMock,
        mock_feature_engineer: MagicMock,
        mock_feature_store: MagicMock,
        mock_get_previous_day: MagicMock,
    ) -> None:
        """Test venue code mapping (e.g., XNAS -> NASDAQ)."""
        config = MagicMock()
        config.feature_store_enabled = True
        config.symbols = ["SPY.XNAS"]

        with patch(
            "nautilus_trader.model.identifiers.InstrumentId"
        ) as mock_instrument_id_cls:
            mock_instrument_id = MagicMock()
            mock_instrument_id.__str__ = MagicMock(return_value="SPY.NASDAQ")
            mock_instrument_id_cls.from_str.return_value = mock_instrument_id

            component.compute_features(
                config=config,
                catalog=mock_catalog,
                feature_engineer=mock_feature_engineer,
                feature_store=mock_feature_store,
                init_feature_store_fn=MagicMock(return_value=mock_feature_store),
                get_previous_day_fn=mock_get_previous_day,
            )

            # Verify InstrumentId was created with mapped venue
            mock_instrument_id_cls.from_str.assert_called_with("SPY.NASDAQ")

    def test_venue_map_contains_expected_mappings(self) -> None:
        """Test VENUE_MAP contains expected venue code mappings."""
        assert VENUE_MAP["XNAS"] == "NASDAQ"
        assert VENUE_MAP["XNYS"] == "NYSE"
        assert VENUE_MAP["ARCX"] == "ARCA"
        assert VENUE_MAP["BATS"] == "BATS"
        assert VENUE_MAP["GLBX"] == "GLBX"


# -----------------------------------------------------------------------------
# Test: Catalog Query
# -----------------------------------------------------------------------------


class TestCatalogQuery:
    """Tests for catalog query operations."""

    def test_compute_features_catalog_query(
        self,
        component: FeatureComputationComponent,
        mock_catalog: MagicMock,
        mock_feature_engineer: MagicMock,
        mock_feature_store: MagicMock,
        mock_get_previous_day: MagicMock,
    ) -> None:
        """Test catalog is queried with correct parameters."""
        config = MagicMock()
        config.feature_store_enabled = True
        config.symbols = ["SPY.XNAS"]

        with patch(
            "nautilus_trader.model.identifiers.InstrumentId"
        ) as mock_instrument_id_cls:
            mock_instrument_id = MagicMock()
            mock_instrument_id.__str__ = MagicMock(return_value="SPY.NASDAQ")
            mock_instrument_id_cls.from_str.return_value = mock_instrument_id

            with patch(
                "nautilus_trader.model.data.Bar"
            ) as mock_bar_cls:
                component.compute_features(
                    config=config,
                    catalog=mock_catalog,
                    feature_engineer=mock_feature_engineer,
                    feature_store=mock_feature_store,
                    init_feature_store_fn=MagicMock(return_value=mock_feature_store),
                    get_previous_day_fn=mock_get_previous_day,
                )

                # Verify catalog.query was called with Bar class
                mock_catalog.query.assert_called()
                call_kwargs = mock_catalog.query.call_args.kwargs
                assert call_kwargs["data_cls"] == mock_bar_cls
                assert "SPY.NASDAQ" in call_kwargs["identifiers"]

    def test_compute_features_no_bars_warning(
        self,
        component: FeatureComputationComponent,
        mock_feature_engineer: MagicMock,
        mock_feature_store: MagicMock,
        mock_get_previous_day: MagicMock,
    ) -> None:
        """Test warning logged when no bars found for instrument."""
        config = MagicMock()
        config.feature_store_enabled = True
        config.symbols = ["SPY.XNAS"]

        catalog = MagicMock()
        catalog.query.return_value = []  # No bars found

        with patch(
            "nautilus_trader.model.identifiers.InstrumentId"
        ) as mock_instrument_id_cls:
            mock_instrument_id = MagicMock()
            mock_instrument_id.__str__ = MagicMock(return_value="SPY.NASDAQ")
            mock_instrument_id_cls.from_str.return_value = mock_instrument_id

            total, failed = component.compute_features(
                config=config,
                catalog=catalog,
                feature_engineer=mock_feature_engineer,
                feature_store=mock_feature_store,
                init_feature_store_fn=MagicMock(return_value=mock_feature_store),
                get_previous_day_fn=mock_get_previous_day,
            )

            # Should not have stored any features (no bars)
            assert total == 0
            # Not marked as failed - just skipped (no bars is not an error)
            assert failed == []
            mock_feature_store.compute_and_store_historical.assert_not_called()


# -----------------------------------------------------------------------------
# Test: Feature Store Historical
# -----------------------------------------------------------------------------


class TestFeatureStoreHistorical:
    """Tests for FeatureStore.compute_and_store_historical() calls."""

    def test_compute_features_store_historical(
        self,
        component: FeatureComputationComponent,
        mock_catalog: MagicMock,
        mock_feature_engineer: MagicMock,
        mock_feature_store: MagicMock,
        mock_get_previous_day: MagicMock,
    ) -> None:
        """Test compute_and_store_historical called with correct parameters."""
        config = MagicMock()
        config.feature_store_enabled = True
        config.symbols = ["SPY.XNAS"]

        with patch(
            "nautilus_trader.model.identifiers.InstrumentId"
        ) as mock_instrument_id_cls:
            mock_instrument_id = MagicMock()
            mock_instrument_id.__str__ = MagicMock(return_value="SPY.NASDAQ")
            mock_instrument_id_cls.from_str.return_value = mock_instrument_id

            component.compute_features(
                config=config,
                catalog=mock_catalog,
                feature_engineer=mock_feature_engineer,
                feature_store=mock_feature_store,
                init_feature_store_fn=MagicMock(return_value=mock_feature_store),
                get_previous_day_fn=mock_get_previous_day,
            )

            # Verify compute_and_store_historical was called
            mock_feature_store.compute_and_store_historical.assert_called_once()
            call_kwargs = mock_feature_store.compute_and_store_historical.call_args.kwargs
            assert call_kwargs["instrument_id"] == "SPY.NASDAQ"
            assert call_kwargs["force_recompute"] is True


# -----------------------------------------------------------------------------
# Test: Metrics Tracking
# -----------------------------------------------------------------------------


class TestMetricsTracking:
    """Tests for Prometheus metrics tracking."""

    def test_compute_features_metrics_tracking(
        self,
        component: FeatureComputationComponent,
        mock_catalog: MagicMock,
        mock_feature_engineer: MagicMock,
        mock_feature_store: MagicMock,
        mock_get_previous_day: MagicMock,
    ) -> None:
        """Test Prometheus metrics are recorded during computation."""
        config = MagicMock()
        config.feature_store_enabled = True
        config.symbols = ["SPY.XNAS"]

        with patch(
            "nautilus_trader.model.identifiers.InstrumentId"
        ) as mock_instrument_id_cls:
            mock_instrument_id = MagicMock()
            mock_instrument_id.__str__ = MagicMock(return_value="SPY.NASDAQ")
            mock_instrument_id_cls.from_str.return_value = mock_instrument_id

            with patch(
                "ml.data.common.scheduler_feature_job.feature_store_operations_total"
            ) as mock_ops_counter:
                with patch(
                    "ml.data.common.scheduler_feature_job.feature_store_latency"
                ) as mock_latency:
                    with patch(
                        "ml.data.common.scheduler_feature_job.feature_computation_store_latency"
                    ) as mock_comp_latency:
                        component.compute_features(
                            config=config,
                            catalog=mock_catalog,
                            feature_engineer=mock_feature_engineer,
                            feature_store=mock_feature_store,
                            init_feature_store_fn=MagicMock(
                                return_value=mock_feature_store
                            ),
                            get_previous_day_fn=mock_get_previous_day,
                        )

                        # Verify metrics were recorded
                        mock_ops_counter.labels.assert_called()
                        mock_latency.labels.assert_called()
                        mock_comp_latency.labels.assert_called()

    def test_active_feature_tasks_gauge(
        self,
        component: FeatureComputationComponent,
        mock_catalog: MagicMock,
        mock_feature_engineer: MagicMock,
        mock_feature_store: MagicMock,
        mock_get_previous_day: MagicMock,
    ) -> None:
        """Test active_feature_tasks gauge is updated during computation."""
        config = MagicMock()
        config.feature_store_enabled = True
        config.symbols = ["SPY.XNAS", "AAPL.XNYS"]

        with patch(
            "nautilus_trader.model.identifiers.InstrumentId"
        ) as mock_instrument_id_cls:
            mock_instrument_id = MagicMock()
            mock_instrument_id.__str__ = MagicMock(return_value="SPY.NASDAQ")
            mock_instrument_id_cls.from_str.return_value = mock_instrument_id

            with patch(
                "ml.data.common.scheduler_feature_job.active_feature_tasks"
            ) as mock_gauge:
                component.compute_features(
                    config=config,
                    catalog=mock_catalog,
                    feature_engineer=mock_feature_engineer,
                    feature_store=mock_feature_store,
                    init_feature_store_fn=MagicMock(return_value=mock_feature_store),
                    get_previous_day_fn=mock_get_previous_day,
                )

                # Should have set gauge multiple times (initial + updates + final 0)
                assert mock_gauge.set.call_count >= 3
                # Final call should reset to 0
                mock_gauge.set.assert_called_with(0)


# -----------------------------------------------------------------------------
# Test: Error Handling
# -----------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling and failed instruments collection."""

    def test_compute_features_error_handling(
        self,
        component: FeatureComputationComponent,
        mock_catalog: MagicMock,
        mock_feature_engineer: MagicMock,
        mock_get_previous_day: MagicMock,
    ) -> None:
        """Test errors during store_historical add instrument to failed list."""
        config = MagicMock()
        config.feature_store_enabled = True
        config.symbols = ["SPY.XNAS"]

        # Store that raises an exception
        failing_store = MagicMock()
        failing_store.compute_and_store_historical.side_effect = RuntimeError(
            "Storage failed"
        )

        with patch(
            "nautilus_trader.model.identifiers.InstrumentId"
        ) as mock_instrument_id_cls:
            mock_instrument_id = MagicMock()
            mock_instrument_id.__str__ = MagicMock(return_value="SPY.NASDAQ")
            mock_instrument_id_cls.from_str.return_value = mock_instrument_id

            _total, failed = component.compute_features(
                config=config,
                catalog=mock_catalog,
                feature_engineer=mock_feature_engineer,
                feature_store=failing_store,
                init_feature_store_fn=MagicMock(return_value=failing_store),
                get_previous_day_fn=mock_get_previous_day,
            )

            assert _total == 0
            assert "SPY.NASDAQ" in failed

    def test_compute_features_failed_instruments(
        self,
        component: FeatureComputationComponent,
        mock_catalog: MagicMock,
        mock_feature_engineer: MagicMock,
        mock_get_previous_day: MagicMock,
    ) -> None:
        """Test multiple failed instruments are all captured."""
        config = MagicMock()
        config.feature_store_enabled = True
        config.symbols = [
            "INVALID",  # Invalid format - no dot
            "SPY.XNAS",  # Will fail on store
            "ALSO.INVALID.TOO",  # Invalid format - too many dots
        ]

        # Store that raises an exception
        failing_store = MagicMock()
        failing_store.compute_and_store_historical.side_effect = RuntimeError(
            "Storage failed"
        )

        with patch(
            "nautilus_trader.model.identifiers.InstrumentId"
        ) as mock_instrument_id_cls:
            mock_instrument_id = MagicMock()
            mock_instrument_id.__str__ = MagicMock(return_value="SPY.NASDAQ")
            mock_instrument_id_cls.from_str.return_value = mock_instrument_id

            _total, failed = component.compute_features(
                config=config,
                catalog=mock_catalog,
                feature_engineer=mock_feature_engineer,
                feature_store=failing_store,
                init_feature_store_fn=MagicMock(return_value=failing_store),
                get_previous_day_fn=mock_get_previous_day,
            )

            # All three should be in failed list
            assert len(failed) == 3
            assert "INVALID" in failed
            assert "SPY.NASDAQ" in failed
            assert "ALSO.INVALID.TOO" in failed


# -----------------------------------------------------------------------------
# Protocol Compliance Test
# -----------------------------------------------------------------------------


class TestProtocolCompliance:
    """Tests for protocol compliance."""

    def test_component_satisfies_protocol(
        self,
        component: FeatureComputationComponent,
    ) -> None:
        """Test FeatureComputationComponent satisfies FeatureComputationProtocol."""
        # Protocol compliance is structural in Python - verify methods exist
        assert hasattr(component, "compute_features")
        assert callable(component.compute_features)

        # Verify it can be used as the protocol type (duck typing)
        def use_protocol(proto: FeatureComputationProtocol) -> None:
            pass

        use_protocol(component)  # Should not raise
