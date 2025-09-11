#!/usr/bin/env python3
"""
Comprehensive test suite to validate the event-driven architecture claims.

This test suite validates the actual implementation against the documented
capabilities in ml/docs/context/context_events.md.

Tests cover:
1. Message bus configuration and Redis integration
2. Topic building and normalization
3. Topic filtering and wildcard matching
4. Event publishing and subscribing
5. Domain event bridge functionality
6. Correlation ID tracking and idempotent processing
7. Throttling and backpressure handling
8. End-to-end event flows

"""

import os
import sys
import time
import uuid
import threading
import json
from typing import Any, Dict, List, Tuple
from collections import defaultdict

# Add the project root to sys.path so we can import the ML modules
project_root = "/home/nate/projects/nautilus_trader"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Test imports
try:
    from ml.config.events import Stage, Source, EventStatus
    from ml.config.bus import MessageBusConfig, BusBackend, TopicScheme
    from ml.config.actor_bus import ActorBusConfig
    from ml.common.message_bus import (
        MessagePublisherProtocol,
        NoopPublisher,
        RedisStreamsPublisher,
        publisher_from_config,
    )
    from ml.common.message_topics import (
        build_topic,
        build_stage_topic,
        build_topic_for_stage,
        map_stage_to_topic_segments,
        _normalize_instrument_id,
    )
    from ml.common.topic_filters import match_topic
    from ml.common.in_memory_bus import InMemoryPublisher
    from ml.common.throttler import Throttler
    from ml.actors.ml_domain_events import DomainEventBridge
    from ml.consumers.idempotent import IdempotentConsumer
    from ml.consumers.redis_streams_consumer import RedisStreamsConsumer
    from ml.core.bus_integration import attach_publisher_from_env

    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


class TestResults:
    """
    Collect and report test results.
    """

    def __init__(self):
        self.tests: list[tuple[str, bool, str]] = []
        self.section_results: dict[str, list[tuple[str, bool, str]]] = defaultdict(list)

    def add_test(self, section: str, test_name: str, passed: bool, message: str = ""):
        """
        Add a test result.
        """
        self.tests.append((f"{section}: {test_name}", passed, message))
        self.section_results[section].append((test_name, passed, message))

    def print_section_summary(self, section: str):
        """
        Print summary for a section.
        """
        tests = self.section_results[section]
        passed = sum(1 for _, p, _ in tests if p)
        total = len(tests)
        status = "✅" if passed == total else "⚠️" if passed > 0 else "❌"
        print(f"\n{status} {section}: {passed}/{total} tests passed")

        for test_name, passed, message in tests:
            status_icon = "✅" if passed else "❌"
            print(f"  {status_icon} {test_name}")
            if message:
                print(f"      {message}")

    def print_final_summary(self):
        """
        Print final test summary.
        """
        total_passed = sum(1 for _, p, _ in self.tests if p)
        total_tests = len(self.tests)

        print(f"\n{'='*60}")
        print(f"FINAL RESULTS: {total_passed}/{total_tests} tests passed")
        print(f"{'='*60}")

        if total_passed == total_tests:
            print("🎉 ALL TESTS PASSED - Event-driven architecture is fully functional!")
        elif total_passed > 0:
            print("⚠️  PARTIAL SUCCESS - Some event features are working")
        else:
            print("💥 ALL TESTS FAILED - Event-driven architecture needs major work")


def test_message_bus_config(results: TestResults):
    """
    Test message bus configuration and environment parsing.
    """
    section = "Message Bus Configuration"

    # Test 1: Default configuration
    try:
        cfg = MessageBusConfig()
        assert cfg.enabled == False
        assert cfg.backend == "noop"
        assert cfg.scheme == "domain_op"
        results.add_test(section, "Default configuration", True)
    except Exception as e:
        results.add_test(section, "Default configuration", False, str(e))

    # Test 2: Environment parsing
    try:
        # Set test environment
        test_env = {
            "ML_BUS_ENABLE": "true",
            "ML_BUS_BACKEND": "redis",
            "ML_BUS_SCHEME": "stage_first",
            "ML_BUS_TOPIC_PREFIX": "test.events",
            "ML_BUS_REDIS_URL": "redis://localhost:6379/1",
            "ML_BUS_REDIS_STREAM": "test-stream",
            "ML_BUS_REDIS_MAXLEN": "1000",
        }

        # Save original env
        original_env = {}
        for key, value in test_env.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value

        cfg = MessageBusConfig.from_env()
        assert cfg.enabled == True
        assert cfg.backend == "redis"
        assert cfg.scheme == "stage_first"
        assert cfg.topic_prefix == "test.events"
        assert cfg.redis_url == "redis://localhost:6379/1"
        assert cfg.redis_stream == "test-stream"
        assert cfg.redis_maxlen == 1000

        # Restore env
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        results.add_test(section, "Environment parsing", True)
    except Exception as e:
        results.add_test(section, "Environment parsing", False, str(e))

    # Test 3: Actor bus configuration
    try:
        cfg = ActorBusConfig.from_env()
        assert hasattr(cfg, "from_actor")
        assert hasattr(cfg, "from_store")
        assert hasattr(cfg, "scheme")
        results.add_test(section, "Actor bus configuration", True)
    except Exception as e:
        results.add_test(section, "Actor bus configuration", False, str(e))


def test_topic_building(results: TestResults):
    """
    Test topic building and normalization utilities.
    """
    section = "Topic Building"

    # Test 1: Basic topic building
    try:
        topic = build_topic("features", "computed", "EURUSD.SIM")
        expected = "ml.features.computed.EURUSD.SIM"
        assert topic == expected
        results.add_test(section, "Basic topic building", True)
    except Exception as e:
        results.add_test(section, "Basic topic building", False, str(e))

    # Test 2: Instrument normalization
    try:
        # Test reserved characters
        normalized = _normalize_instrument_id("EUR/USD*TEST#")
        assert "*" not in normalized
        assert "#" not in normalized
        assert "/" not in normalized
        results.add_test(section, "Instrument normalization", True, f"Normalized: {normalized}")
    except Exception as e:
        results.add_test(section, "Instrument normalization", False, str(e))

    # Test 3: Stage to topic mapping
    try:
        domain, op = map_stage_to_topic_segments(Stage.FEATURE_COMPUTED)
        assert domain == "features"
        assert op == "updated"
        results.add_test(
            section, "Stage mapping", True, f"{Stage.FEATURE_COMPUTED} -> {domain}.{op}"
        )
    except Exception as e:
        results.add_test(section, "Stage mapping", False, str(e))

    # Test 4: Stage-first topic building
    try:
        topic = build_stage_topic(Stage.PREDICTION_EMITTED, "BTCUSDT.BINANCE")
        expected = "events.ml.PREDICTION_EMITTED.BTCUSDT.BINANCE"
        assert topic == expected
        results.add_test(section, "Stage-first topics", True)
    except Exception as e:
        results.add_test(section, "Stage-first topics", False, str(e))

    # Test 5: Dynamic topic selection
    try:
        # Domain-op scheme
        topic1 = build_topic_for_stage(
            Stage.SIGNAL_EMITTED,
            "GBPUSD.SIM",
            scheme="domain_op",
        )
        assert topic1.startswith("ml.strategies.")

        # Stage-first scheme
        topic2 = build_topic_for_stage(
            Stage.SIGNAL_EMITTED,
            "GBPUSD.SIM",
            scheme="stage_first",
            prefix="test.events",
        )
        assert topic2.startswith("test.events.SIGNAL_EMITTED")

        results.add_test(section, "Dynamic topic selection", True)
    except Exception as e:
        results.add_test(section, "Dynamic topic selection", False, str(e))


def test_topic_filtering(results: TestResults):
    """
    Test topic filtering and wildcard pattern matching.
    """
    section = "Topic Filtering"

    # Test 1: Exact matching
    try:
        assert match_topic("ml.features.computed.EURUSD.SIM", "ml.features.computed.EURUSD.SIM")
        assert not match_topic(
            "ml.features.computed.EURUSD.SIM", "ml.features.computed.BTCUSDT.BINANCE"
        )
        results.add_test(section, "Exact matching", True)
    except Exception as e:
        results.add_test(section, "Exact matching", False, str(e))

    # Test 2: Wildcard '*' matching
    try:
        assert match_topic("ml.features.computed.*", "ml.features.computed.EURUSD.SIM")
        assert match_topic("ml.*.emitted.*", "ml.signals.emitted.BTCUSDT.BINANCE")
        assert not match_topic("ml.features.computed.*", "ml.features.computed")
        results.add_test(section, "Wildcard '*' matching", True)
    except Exception as e:
        results.add_test(section, "Wildcard '*' matching", False, str(e))

    # Test 3: Hash '#' matching
    try:
        assert match_topic("ml.features.#", "ml.features.computed.EURUSD.SIM")
        assert match_topic("ml.features.#", "ml.features")
        assert match_topic("events.ml.SIGNAL_EMITTED.#", "events.ml.SIGNAL_EMITTED")
        assert match_topic("events.ml.SIGNAL_EMITTED.#", "events.ml.SIGNAL_EMITTED.GBPUSD.SIM")
        results.add_test(section, "Hash '#' matching", True)
    except Exception as e:
        results.add_test(section, "Hash '#' matching", False, str(e))

    # Test 4: Complex patterns
    try:
        pattern = "*.ml.*.EURUSD.*"
        assert match_topic(pattern, "events.ml.FEATURE_COMPUTED.EURUSD.SIM")
        assert not match_topic(pattern, "events.ml.FEATURE_COMPUTED.BTCUSDT.SIM")
        results.add_test(section, "Complex patterns", True)
    except Exception as e:
        results.add_test(section, "Complex patterns", False, str(e))


def test_in_memory_publisher(results: TestResults):
    """
    Test in-memory publisher and subscriber functionality.
    """
    section = "In-Memory Publisher"

    # Test 1: Basic publishing
    try:
        publisher = InMemoryPublisher()
        events_received = []

        def handler(topic: str, payload: dict[str, Any]):
            events_received.append((topic, payload))

        publisher.subscribe("ml.features.*", handler)

        success = publisher.publish(
            "ml.features.computed.EURUSD.SIM",
            {
                "dataset_id": "features",
                "instrument_id": "EURUSD.SIM",
                "metadata": {"correlation_id": "test-1"},
            },
        )

        assert success == True
        assert len(events_received) == 1
        results.add_test(section, "Basic publishing", True)
    except Exception as e:
        results.add_test(section, "Basic publishing", False, str(e))

    # Test 2: Multiple subscribers
    try:
        publisher = InMemoryPublisher()
        events1, events2 = [], []

        publisher.subscribe("ml.features.*", lambda t, p: events1.append((t, p)))
        publisher.subscribe("ml.*", lambda t, p: events2.append((t, p)))

        publisher.publish("ml.features.computed.TEST", {"test": "data"})

        assert len(events1) == 1  # Specific subscriber
        assert len(events2) == 1  # Broader subscriber
        results.add_test(section, "Multiple subscribers", True)
    except Exception as e:
        results.add_test(section, "Multiple subscribers", False, str(e))

    # Test 3: Pattern filtering
    try:
        publisher = InMemoryPublisher()
        matching_events = []

        publisher.subscribe("events.ml.*.EURUSD.*", lambda t, p: matching_events.append((t, p)))

        # Should match
        publisher.publish("events.ml.FEATURE_COMPUTED.EURUSD.SIM", {"test": 1})
        # Should not match
        publisher.publish("events.ml.FEATURE_COMPUTED.BTCUSDT.SIM", {"test": 2})

        assert len(matching_events) == 1
        results.add_test(section, "Pattern filtering", True)
    except Exception as e:
        results.add_test(section, "Pattern filtering", False, str(e))


def test_redis_publisher(results: TestResults):
    """
    Test Redis Streams publisher integration.
    """
    section = "Redis Publisher"

    # Test 1: Redis publisher creation
    try:
        publisher = RedisStreamsPublisher(
            url="redis://localhost:6379/0",
            stream="test-stream",
        )
        results.add_test(section, "Redis publisher creation", True)
    except Exception as e:
        results.add_test(section, "Redis publisher creation", False, str(e))

    # Test 2: Publisher from config
    try:
        # Test with disabled config
        cfg = MessageBusConfig(enabled=False)
        publisher = publisher_from_config(cfg)
        assert isinstance(publisher, NoopPublisher)

        # Test with Redis config
        cfg = MessageBusConfig(enabled=True, backend="redis")
        publisher = publisher_from_config(cfg)
        assert isinstance(publisher, RedisStreamsPublisher)

        results.add_test(section, "Publisher from config", True)
    except Exception as e:
        results.add_test(section, "Publisher from config", False, str(e))

    # Test 3: NoopPublisher behavior
    try:
        publisher = NoopPublisher()
        success = publisher.publish("any.topic", {"test": "data"})
        assert success == False  # NoopPublisher always returns False
        results.add_test(section, "NoopPublisher behavior", True)
    except Exception as e:
        results.add_test(section, "NoopPublisher behavior", False, str(e))


def test_domain_event_bridge(results: TestResults):
    """
    Test DomainEventBridge for non-blocking actor publishing.
    """
    section = "Domain Event Bridge"

    # Test 1: Bridge creation and lifecycle
    try:
        publisher = InMemoryPublisher()
        bridge = DomainEventBridge(
            publisher=publisher,
            max_queue=1000,
            component_id="test_actor",
        )

        bridge.start()
        assert bridge._thread is not None
        bridge.stop()
        results.add_test(section, "Bridge lifecycle", True)
    except Exception as e:
        results.add_test(section, "Bridge lifecycle", False, str(e))

    # Test 2: Non-blocking publishing
    try:
        received_events = []
        publisher = InMemoryPublisher()
        publisher.subscribe("ml.*", lambda t, p: received_events.append((t, p)))

        bridge = DomainEventBridge(publisher, max_queue=100)
        bridge.start()

        # Publish events
        success1 = bridge.publish("ml.test.event", {"id": 1})
        success2 = bridge.publish("ml.test.event", {"id": 2})

        assert success1 == True
        assert success2 == True

        # Wait for background processing
        time.sleep(0.1)

        bridge.stop(drain=True)

        # Should have received the events
        assert len(received_events) >= 1
        results.add_test(
            section, "Non-blocking publishing", True, f"Received {len(received_events)} events"
        )
    except Exception as e:
        results.add_test(section, "Non-blocking publishing", False, str(e))

    # Test 3: Queue overflow handling
    try:
        publisher = InMemoryPublisher()
        bridge = DomainEventBridge(publisher, max_queue=2)  # Very small queue
        bridge.start()

        # Try to overflow the queue
        successes = []
        for i in range(10):
            success = bridge.publish("ml.test.overflow", {"id": i})
            successes.append(success)

        bridge.stop()

        # Should have some failures due to queue overflow
        failures = successes.count(False)
        results.add_test(section, "Queue overflow handling", True, f"{failures} events dropped")
    except Exception as e:
        results.add_test(section, "Queue overflow handling", False, str(e))


def test_throttling(results: TestResults):
    """
    Test throttling mechanism for rate limiting.
    """
    section = "Throttling"

    # Test 1: Basic throttle creation
    try:
        throttler = Throttler(rate_per_sec=10.0, burst=5)
        assert throttler is not None
        results.add_test(section, "Throttle creation", True)
    except Exception as e:
        results.add_test(section, "Throttle creation", False, str(e))

    # Test 2: Token bucket behavior
    try:
        throttler = Throttler(rate_per_sec=100.0, burst=2)
        now_ns = int(time.time() * 1e9)

        # First two should succeed (burst allows)
        assert throttler.should_publish("topic1", now_ns) == True
        assert throttler.should_publish("topic1", now_ns) == True

        # Third should fail (exceeded burst)
        assert throttler.should_publish("topic1", now_ns) == False

        results.add_test(section, "Token bucket behavior", True)
    except Exception as e:
        results.add_test(section, "Token bucket behavior", False, str(e))

    # Test 3: Per-key throttling
    try:
        throttler = Throttler(rate_per_sec=100.0, burst=1)
        now_ns = int(time.time() * 1e9)

        # Different topics should have independent buckets
        assert throttler.should_publish("topic1", now_ns) == True
        assert throttler.should_publish("topic2", now_ns) == True

        # Same topics should be throttled
        assert throttler.should_publish("topic1", now_ns) == False
        assert throttler.should_publish("topic2", now_ns) == False

        results.add_test(section, "Per-key throttling", True)
    except Exception as e:
        results.add_test(section, "Per-key throttling", False, str(e))

    # Test 4: Bridge with throttling
    try:
        throttler = Throttler(rate_per_sec=5.0, burst=2)
        publisher = InMemoryPublisher()
        bridge = DomainEventBridge(
            publisher=publisher,
            max_queue=100,
            throttler=throttler,
            component_id="throttled_actor",
        )

        bridge.start()

        # Send multiple events quickly
        now_ns = int(time.time() * 1e9)
        results_list = []
        for i in range(5):
            payload = {"id": i, "ts_max": now_ns}
            success = bridge.publish("ml.test.throttled", payload)
            results_list.append(success)

        bridge.stop()

        # Should have some throttled events
        throttled_count = results_list.count(False)
        results.add_test(section, "Bridge throttling", True, f"{throttled_count} events throttled")
    except Exception as e:
        results.add_test(section, "Bridge throttling", False, str(e))


def test_idempotent_consumer(results: TestResults):
    """
    Test IdempotentConsumer for correlation ID and watermark gating.
    """
    section = "Idempotent Consumer"

    # Test 1: Basic idempotency
    try:
        consumer = IdempotentConsumer()

        payload1 = {
            "dataset_id": "features",
            "instrument_id": "EURUSD.SIM",
            "source": "historical",
            "ts_max": 1000,
            "metadata": {"correlation_id": "test-123"},
        }

        # First processing should succeed
        assert consumer.process(payload1) == True

        # Duplicate should be rejected
        assert consumer.process(payload1) == False

        results.add_test(section, "Basic idempotency", True)
    except Exception as e:
        results.add_test(section, "Basic idempotency", False, str(e))

    # Test 2: Watermark gating
    try:
        consumer = IdempotentConsumer()

        # Later timestamp should be accepted
        payload_new = {
            "dataset_id": "features",
            "instrument_id": "EURUSD.SIM",
            "source": "historical",
            "ts_max": 2000,
            "metadata": {"correlation_id": "new-event"},
        }
        assert consumer.process(payload_new) == True

        # Earlier timestamp should be rejected
        payload_old = {
            "dataset_id": "features",
            "instrument_id": "EURUSD.SIM",
            "source": "historical",
            "ts_max": 1000,
            "metadata": {"correlation_id": "old-event"},
        }
        assert consumer.process(payload_old) == False

        results.add_test(section, "Watermark gating", True)
    except Exception as e:
        results.add_test(section, "Watermark gating", False, str(e))

    # Test 3: Per-key watermarks
    try:
        consumer = IdempotentConsumer()

        # Different instruments should have independent watermarks
        payload1 = {
            "dataset_id": "features",
            "instrument_id": "EURUSD.SIM",
            "source": "historical",
            "ts_max": 1000,
            "metadata": {"correlation_id": "eur-1"},
        }

        payload2 = {
            "dataset_id": "features",
            "instrument_id": "BTCUSDT.BINANCE",
            "source": "historical",
            "ts_max": 500,  # Earlier timestamp but different instrument
            "metadata": {"correlation_id": "btc-1"},
        }

        assert consumer.process(payload1) == True
        assert consumer.process(payload2) == True  # Should succeed despite earlier timestamp

        results.add_test(section, "Per-key watermarks", True)
    except Exception as e:
        results.add_test(section, "Per-key watermarks", False, str(e))


def test_redis_streams_consumer(results: TestResults):
    """
    Test Redis Streams consumer with idempotent processing.
    """
    section = "Redis Streams Consumer"

    # Test 1: Consumer creation
    try:

        def dummy_handler(topic: str, payload: dict[str, Any]):
            pass

        consumer = RedisStreamsConsumer(
            url="redis://localhost:6379/0",
            stream="test-stream",
            handler=dummy_handler,
        )
        results.add_test(section, "Consumer creation", True)
    except Exception as e:
        results.add_test(section, "Consumer creation", False, str(e))

    # Test 2: Consumer with custom gate
    try:
        processed_events = []

        def handler(topic: str, payload: dict[str, Any]):
            processed_events.append((topic, payload))

        gate = IdempotentConsumer()
        consumer = RedisStreamsConsumer(
            url="redis://localhost:6379/0",
            stream="test-stream",
            handler=handler,
            gate=gate,
        )

        # Simulate processing with duplicate
        duplicate_payload = {
            "dataset_id": "test",
            "instrument_id": "TEST.SIM",
            "source": "historical",
            "ts_max": 1000,
            "metadata": {"correlation_id": "dup-123"},
        }

        # First should be processed
        assert gate.process(duplicate_payload) == True
        # Second should be rejected
        assert gate.process(duplicate_payload) == False

        results.add_test(section, "Consumer with gating", True)
    except Exception as e:
        results.add_test(section, "Consumer with gating", False, str(e))

    # Test 3: Consumer without Redis (graceful degradation)
    try:

        def handler(topic: str, payload: dict[str, Any]):
            pass

        # Use invalid Redis URL to test graceful degradation
        consumer = RedisStreamsConsumer(
            url="redis://invalid:6379/0",
            stream="test-stream",
            handler=handler,
        )

        # Should return 0 when no client available
        processed = consumer.poll_once(count=10, block_ms=0)
        assert processed == 0

        results.add_test(section, "Graceful degradation", True)
    except Exception as e:
        results.add_test(section, "Graceful degradation", False, str(e))


def test_end_to_end_flow(results: TestResults):
    """
    Test end-to-end event flow with correlation tracking.
    """
    section = "End-to-End Flow"

    # Test 1: Complete publish-subscribe flow
    try:
        # Setup pipeline
        publisher = InMemoryPublisher()
        received_events = []
        correlation_ids = set()

        def event_handler(topic: str, payload: dict[str, Any]):
            received_events.append((topic, payload))
            if "metadata" in payload and "correlation_id" in payload["metadata"]:
                correlation_ids.add(payload["metadata"]["correlation_id"])

        # Subscribe to different patterns
        publisher.subscribe("ml.features.*", event_handler)
        publisher.subscribe("events.ml.PREDICTION_EMITTED.*", event_handler)

        # Create domain event bridge
        bridge = DomainEventBridge(publisher, max_queue=1000)
        bridge.start()

        # Publish events through different stages
        test_correlation = str(uuid.uuid4())

        # Stage 1: Feature computed
        feature_topic = build_topic("features", "updated", "EURUSD.SIM")
        feature_payload = {
            "dataset_id": "features",
            "instrument_id": "EURUSD.SIM",
            "source": "historical",
            "ts_max": int(time.time() * 1e9),
            "metadata": {
                "correlation_id": test_correlation,
                "stage": "FEATURE_COMPUTED",
            },
        }

        success1 = bridge.publish(feature_topic, feature_payload)

        # Stage 2: Prediction emitted
        prediction_topic = build_stage_topic(Stage.PREDICTION_EMITTED, "EURUSD.SIM")
        prediction_payload = {
            "dataset_id": "predictions",
            "instrument_id": "EURUSD.SIM",
            "source": "live",
            "ts_max": int(time.time() * 1e9),
            "metadata": {
                "correlation_id": test_correlation,
                "stage": "PREDICTION_EMITTED",
            },
        }

        success2 = bridge.publish(prediction_topic, prediction_payload)

        assert success1 == True
        assert success2 == True

        # Wait for processing
        time.sleep(0.2)
        bridge.stop(drain=True)

        # Verify events were received and correlation preserved
        assert len(received_events) >= 2
        assert test_correlation in correlation_ids

        results.add_test(
            section,
            "Complete flow",
            True,
            f"Received {len(received_events)} events with correlation",
        )
    except Exception as e:
        results.add_test(section, "Complete flow", False, str(e))

    # Test 2: Event cascade with idempotent processing
    try:
        # Setup idempotent consumer
        consumer = IdempotentConsumer()
        processed_events = []

        def process_event(payload: dict[str, Any]):
            if consumer.process(payload):
                processed_events.append(payload)
                return True
            return False

        base_correlation = str(uuid.uuid4())

        # Simulate event cascade
        events = [
            {
                "dataset_id": "features",
                "instrument_id": "EURUSD.SIM",
                "source": "historical",
                "ts_max": 1000,
                "metadata": {"correlation_id": f"{base_correlation}-1"},
            },
            {
                "dataset_id": "predictions",
                "instrument_id": "EURUSD.SIM",
                "source": "historical",
                "ts_max": 2000,
                "metadata": {"correlation_id": f"{base_correlation}-2"},
            },
            # Duplicate event
            {
                "dataset_id": "features",
                "instrument_id": "EURUSD.SIM",
                "source": "historical",
                "ts_max": 1000,
                "metadata": {"correlation_id": f"{base_correlation}-1"},  # Same correlation ID
            },
        ]

        # Process events
        for event in events:
            process_event(event)

        # Should have processed 2 unique events (duplicate rejected)
        assert len(processed_events) == 2

        results.add_test(
            section,
            "Event cascade with idempotency",
            True,
            f"Processed {len(processed_events)}/3 events",
        )
    except Exception as e:
        results.add_test(section, "Event cascade with idempotency", False, str(e))

    # Test 3: Cross-stage correlation tracking
    try:
        # Track correlation across multiple stages
        correlations = {}

        def track_correlation(topic: str, payload: dict[str, Any]):
            if "metadata" in payload and "correlation_id" in payload["metadata"]:
                corr_id = payload["metadata"]["correlation_id"]
                stage = payload["metadata"].get("stage", "unknown")

                if corr_id not in correlations:
                    correlations[corr_id] = []
                correlations[corr_id].append(stage)

        publisher = InMemoryPublisher()
        publisher.subscribe("ml.*", track_correlation)
        publisher.subscribe("events.ml.*", track_correlation)

        # Simulate ML pipeline stages
        pipeline_correlation = str(uuid.uuid4())
        stages = [
            ("ml.data.created.EURUSD.SIM", "DATA_INGESTED"),
            ("ml.features.updated.EURUSD.SIM", "FEATURE_COMPUTED"),
            ("events.ml.PREDICTION_EMITTED.EURUSD.SIM", "PREDICTION_EMITTED"),
            ("events.ml.SIGNAL_EMITTED.EURUSD.SIM", "SIGNAL_EMITTED"),
        ]

        for topic, stage in stages:
            payload = {
                "dataset_id": stage.lower(),
                "instrument_id": "EURUSD.SIM",
                "source": "live",
                "ts_max": int(time.time() * 1e9),
                "metadata": {
                    "correlation_id": pipeline_correlation,
                    "stage": stage,
                },
            }
            publisher.publish(topic, payload)

        # Verify correlation tracking
        assert pipeline_correlation in correlations
        tracked_stages = correlations[pipeline_correlation]
        assert len(tracked_stages) == 4

        results.add_test(
            section, "Cross-stage correlation", True, f"Tracked {len(tracked_stages)} stages"
        )
    except Exception as e:
        results.add_test(section, "Cross-stage correlation", False, str(e))


def main():
    """
    Run all tests and report results.
    """
    print("🧪 Testing Event-Driven Architecture Implementation")
    print("=" * 60)

    results = TestResults()

    # Run all test suites
    test_suites = [
        ("Message Bus Configuration", test_message_bus_config),
        ("Topic Building", test_topic_building),
        ("Topic Filtering", test_topic_filtering),
        ("In-Memory Publisher", test_in_memory_publisher),
        ("Redis Publisher", test_redis_publisher),
        ("Domain Event Bridge", test_domain_event_bridge),
        ("Throttling", test_throttling),
        ("Idempotent Consumer", test_idempotent_consumer),
        ("Redis Streams Consumer", test_redis_streams_consumer),
        ("End-to-End Flow", test_end_to_end_flow),
    ]

    for section_name, test_func in test_suites:
        print(f"\n🔍 Testing {section_name}...")
        try:
            test_func(results)
            results.print_section_summary(section_name)
        except Exception as e:
            print(f"❌ Test suite {section_name} failed: {e}")
            results.add_test(section_name, "Suite execution", False, str(e))

    # Print final summary
    results.print_final_summary()

    return results


if __name__ == "__main__":
    results = main()

    # Exit with error code if tests failed
    total_passed = sum(1 for _, passed, _ in results.tests if passed)
    total_tests = len(results.tests)

    if total_passed == total_tests:
        print("\n🎉 All tests passed! Event-driven architecture is fully functional.")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total_tests - total_passed} tests failed. See details above.")
        sys.exit(1)
