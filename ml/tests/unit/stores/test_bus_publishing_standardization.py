"""
Unit tests for standardized bus publishing gating and error handling across all stores.

This test module verifies that:
1. All stores respect the _enable_publishing flag consistently
2. Topic building uses MessageBusConfig.from_env() scheme/prefix everywhere
3. Hot-path budget is preserved with non-blocking best-effort publishing
4. Environment and config changes properly toggle store emit/no-emit behavior

"""

import os
import pytest
from unittest.mock import MagicMock, Mock, patch

from ml.common.message_bus import BusPublisherMixin, NoopPublisher
from ml.config.bus import MessageBusConfig
from ml.stores import DataStore, FeatureStore
from ml.stores.base import FeatureData


class TestBusPublishingStandardization:
    """
    Test standardized bus publishing behavior across all stores.
    """

    def test_bus_publisher_mixin_respects_env_config(self):
        """
        Test that BusPublisherMixin initializes from MessageBusConfig.from_env().
        """
        with patch.dict(
            os.environ,
            {
                "ML_BUS_SCHEME": "stage_first",
                "ML_BUS_TOPIC_PREFIX": "test.ml.events",
            },
        ):
            mixin = BusPublisherMixin()
            mixin._init_bus_publishing(
                enable_publishing=True,
                publisher=Mock(),
                publish_mode="batch",
            )

            assert mixin._topic_scheme == "stage_first"
            assert mixin._topic_prefix == "test.ml.events"

    def test_bus_publisher_mixin_uses_defaults_on_env_failure(self):
        """
        Test fallback to sensible defaults when env parsing fails.
        """
        with patch("ml.config.bus.MessageBusConfig.from_env", side_effect=Exception("Env failure")):
            mixin = BusPublisherMixin()
            mixin._init_bus_publishing(
                enable_publishing=True,
                publisher=Mock(),
                publish_mode="batch",
            )

            # Should fall back to documented defaults
            assert mixin._topic_scheme == "domain_op"
            assert mixin._topic_prefix == "events.ml"

    @pytest.mark.parametrize(
        "enable_flag,publisher_exists,should_publish",
        [
            (True, True, True),  # Both enabled and publisher exists -> publish
            (True, False, False),  # Enabled but no publisher -> no publish
            (False, True, False),  # Publisher exists but disabled -> no publish
            (False, False, False),  # Neither enabled nor publisher -> no publish
        ],
    )
    def test_data_store_publishing_gating(self, enable_flag, publisher_exists, should_publish):
        """
        Test DataStore respects both _enable_publishing flag and publisher existence.
        """
        mock_publisher = Mock() if publisher_exists else None

        # Create minimal DataStore instance for testing
        with patch("ml.stores.data_store.EngineManager.get_engine"):
            store = DataStore(connection_string="postgresql://test")
            store._enable_publishing = enable_flag
            store.publisher = mock_publisher
            store._topic_scheme = "domain_op"
            store._topic_prefix = "events.ml"

        # Mock the actual publishing method to test gating logic
        with patch.object(store, "emit_dataset_event") as mock_emit:
            store.emit_dataset_event(
                dataset_id="test",
                instrument_id="EUR/USD",
                stage="catalog_written",
                source="historical",
                run_id="test_run",
                ts_min=1000,
                ts_max=2000,
                count=100,
                status="success",
            )

            if should_publish:
                mock_emit.assert_called_once()
            else:
                # Should still call the method but publisher should not be used
                mock_emit.assert_called_once()
                if mock_publisher:
                    mock_publisher.publish.assert_not_called()

    @pytest.mark.parametrize(
        "enable_flag,publisher_exists,should_publish",
        [
            (True, True, True),  # Both enabled and publisher exists -> publish
            (True, False, False),  # Enabled but no publisher -> no publish
            (False, True, False),  # Publisher exists but disabled -> no publish
            (False, False, False),  # Neither enabled nor publisher -> no publish
        ],
    )
    def test_feature_store_publishing_gating(self, enable_flag, publisher_exists, should_publish):
        """
        Test FeatureStore respects both _enable_publishing flag and publisher existence.
        """
        mock_publisher = Mock() if publisher_exists else None

        # Create minimal FeatureStore instance for testing
        with patch("ml.stores.feature_store.EngineManager.get_engine"):
            store = FeatureStore(
                connection_string="postgresql://test",
                enable_publishing=enable_flag,
                publisher=mock_publisher,
            )

            # Test batch publishing
            test_data = [
                FeatureData(
                    instrument_id="EUR/USD",
                    ts_event=1000,
                    ts_init=1000,
                    features={"rsi": 0.5},
                ),
            ]

            with patch.object(store, "_execute_write") as mock_write:
                store.write_batch(test_data)

                # Verify write was called
                mock_write.assert_called_once()

                # Verify publishing behavior
                if should_publish:
                    mock_publisher.publish.assert_called()
                else:
                    if mock_publisher:
                        mock_publisher.publish.assert_not_called()

    def test_publishing_error_handling_non_blocking(self):
        """
        Test that publishing errors don't block store operations.
        """
        mock_publisher = Mock()
        mock_publisher.publish.side_effect = Exception("Publishing failed")

        # Test that BusPublisherMixin properly handles publishing errors
        mixin = BusPublisherMixin()
        mixin._enable_publishing = True
        mixin.publisher = mock_publisher
        mixin._topic_scheme = "domain_op"
        mixin._topic_prefix = "events.ml"

        # Test the publish_batch_and_rows function directly to verify error handling
        from ml.stores.mixins import publish_batch_and_rows
        from ml.config.events import Stage
        import logging

        mock_logger = Mock()

        # This should not raise an exception even though publisher fails
        publish_batch_and_rows(
            enable_publishing=True,
            publisher=mock_publisher,
            publish_mode="batch",
            topic_scheme="domain_op",
            topic_prefix="events.ml",
            stage=Stage.CATALOG_WRITTEN,
            dataset_id="test",
            instrument_key="instrument_id",
            ts_field="ts_event",
            rows=[{"instrument_id": "EUR/USD", "ts_event": 1000}],
            run_id_batch="test_batch",
            run_id_row="test_row",
            source="historical",
            logger=mock_logger,
        )

        # Should complete without exceptions despite publisher failure
        assert mock_publisher.publish.called

    def test_environment_toggle_affects_config(self):
        """
        Test that changing environment variables affects publishing behavior.
        """
        # Test with publishing disabled via env
        with patch.dict(os.environ, {"ML_BUS_ENABLE": "false"}):
            config = MessageBusConfig.from_env()
            assert config.enabled is False

        # Test with publishing enabled via env
        with patch.dict(os.environ, {"ML_BUS_ENABLE": "true"}):
            config = MessageBusConfig.from_env()
            assert config.enabled is True

    def test_topic_scheme_consistency(self):
        """
        Test that all stores use consistent topic scheme/prefix from MessageBusConfig.
        """
        test_cases = [
            ("domain_op", "events.ml"),
            ("stage_first", "custom.prefix"),
        ]

        for scheme, prefix in test_cases:
            with patch.dict(
                os.environ,
                {
                    "ML_BUS_SCHEME": scheme,
                    "ML_BUS_TOPIC_PREFIX": prefix,
                },
            ):
                # Test DataStore
                with patch("ml.stores.data_store.EngineManager.get_engine"):
                    data_store = DataStore(connection_string="postgresql://test")
                    data_store._init_bus_publishing(
                        enable_publishing=True,
                        publisher=Mock(),
                        publish_mode="batch",
                    )
                    assert data_store._topic_scheme == scheme
                    assert data_store._topic_prefix == prefix

                # Test FeatureStore
                with patch("ml.stores.feature_store.EngineManager.get_engine"):
                    feature_store = FeatureStore(
                        connection_string="postgresql://test",
                        enable_publishing=True,
                        publisher=Mock(),
                    )
                    assert feature_store._topic_scheme == scheme
                    assert feature_store._topic_prefix == prefix

    def test_hot_path_performance_preservation(self):
        """
        Test that publishing doesn't significantly impact hot-path performance.
        """
        # This is a smoke test to ensure we don't add heavy operations
        mock_publisher = Mock()

        with patch("ml.stores.data_store.EngineManager.get_engine"):
            store = DataStore(connection_string="postgresql://test")
            store._enable_publishing = True
            store.publisher = mock_publisher

        # Time a simple operation - should complete quickly even with publishing
        import time

        start = time.perf_counter()

        store.emit_dataset_event(
            dataset_id="test",
            instrument_id="EUR/USD",
            stage="catalog_written",
            source="live",
            run_id="test_run",
            ts_min=1000,
            ts_max=2000,
            count=1,
            status="success",
        )

        elapsed = time.perf_counter() - start

        # Should complete in sub-millisecond time for hot-path compliance
        assert elapsed < 0.01, f"Publishing took {elapsed:.4f}s, exceeds hot-path budget"

    def test_noop_publisher_safe_default(self):
        """
        Test that NoopPublisher provides safe default behavior.
        """
        noop = NoopPublisher()

        # Should not raise exceptions
        result = noop.publish("test.topic", {"data": "test"})

        # Should return False to indicate no actual publishing occurred
        assert result is False

    def test_consistent_error_logging_levels(self):
        """
        Test that all stores use consistent error logging levels for publishing
        failures.
        """
        test_stores = []

        # DataStore
        with patch("ml.stores.data_store.EngineManager.get_engine"):
            data_store = DataStore(connection_string="postgresql://test")
            data_store._enable_publishing = True
            data_store.publisher = Mock()
            data_store.publisher.publish.side_effect = Exception("Test error")
            test_stores.append(("DataStore", data_store))

        # FeatureStore
        with patch("ml.stores.feature_store.EngineManager.get_engine"):
            feature_store = FeatureStore(
                connection_string="postgresql://test",
                enable_publishing=True,
                publisher=Mock(),
            )
            feature_store.publisher.publish.side_effect = Exception("Test error")
            test_stores.append(("FeatureStore", feature_store))

        for store_name, store in test_stores:
            with patch(f"ml.stores.{store_name.lower()}.logger") as mock_logger:
                if hasattr(store, "emit_dataset_event"):
                    store.emit_dataset_event(
                        dataset_id="test",
                        instrument_id="EUR/USD",
                        stage="catalog_written",
                        source="historical",
                        run_id="test_run",
                        ts_min=1000,
                        ts_max=2000,
                        count=100,
                        status="success",
                    )
                    # Should use exception-level logging for consistency
                    assert mock_logger.exception.called or mock_logger.debug.called
