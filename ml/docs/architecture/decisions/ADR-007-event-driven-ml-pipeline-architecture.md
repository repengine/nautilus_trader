# ADR-007: Event-Driven ML Pipeline Architecture

**Status: ACCEPTED**

**Date: 2025-09-10**

**Context: Real-time ML Pipeline with Event Correlation and Observability**

## Summary

This ADR establishes the event-driven architecture for the ML pipeline, implementing comprehensive event correlation, topic scheme flexibility, and non-blocking actor integration to support real-time trading decisions with full observability and fault tolerance.

## Context

The comprehensive system review revealed the need for a production-ready event-driven architecture that can handle high-frequency trading data while maintaining strict latency requirements. The system must support multiple deployment environments, provide comprehensive observability, and ensure reliable event delivery without blocking critical trading paths.

### Key Requirements Identified

1. **Real-time Event Processing**: Sub-millisecond event handling in hot paths
2. **Event Correlation**: Complete lineage tracking across pipeline stages
3. **Topic Flexibility**: Support for different routing schemes across environments
4. **Non-blocking Integration**: Actor-side event publishing without hot path impact
5. **Comprehensive Observability**: Off hot-path event persistence and analysis
6. **Production Reliability**: Graceful degradation and fallback strategies

## Decision

We implement a comprehensive Event-Driven ML Pipeline Architecture with the following components:

### 1. Dual Topic Scheme Architecture

**Pattern**: Support canonical domain-operation and stage-first routing schemes for different deployment scenarios.

```python
from enum import Enum
from typing import Protocol

class TopicScheme(str, Enum):
    DOMAIN_OP = "domain_op"      # ml.features.computed, ml.models.predicted
    STAGE_FIRST = "stage_first"  # computed.ml.features, predicted.ml.models

class TopicBuilder(Protocol):
    """Protocol for building message topics."""

    def build_topic_for_stage(self,
                             stage: Stage,
                             source: Source,
                             scheme: TopicScheme = TopicScheme.DOMAIN_OP,
                             prefix: str = "") -> str:
        """Build topic name according to scheme."""
        ...

# ✅ IMPLEMENTATION: Flexible topic building
def build_topic_for_stage(stage: Stage,
                         source: Source,
                         scheme: TopicScheme = TopicScheme.DOMAIN_OP,
                         prefix: str = "") -> str:
    """Build topic according to specified scheme."""
    base_parts = []

    if prefix:
        base_parts.append(prefix)

    if scheme == TopicScheme.DOMAIN_OP:
        # Format: ml.features.computed, ml.models.predicted
        base_parts.extend([source.value, stage.value])
    elif scheme == TopicScheme.STAGE_FIRST:
        # Format: computed.ml.features, predicted.ml.models
        base_parts.extend([stage.value, source.value])

    return ".".join(base_parts)

# Environment-driven topic configuration
@frozen
class MessageBusConfig(NautilusConfig):
    """Message bus configuration with environment integration."""

    enabled: bool = False
    backend: BusBackend = BusBackend.NOOP
    scheme: TopicScheme = TopicScheme.DOMAIN_OP
    topic_prefix: str = ""

    # Redis configuration
    redis_url: str = "redis://localhost:6379/0"
    redis_stream: str = "ml_events"
    redis_maxlen: int | None = 10000

    @classmethod
    def from_env(cls) -> "MessageBusConfig":
        """Load configuration from environment variables."""
        return cls(
            enabled=os.getenv("ML_BUS_ENABLE", "false").lower() == "true",
            backend=BusBackend(os.getenv("ML_BUS_BACKEND", "noop")),
            scheme=TopicScheme(os.getenv("ML_BUS_SCHEME", "domain_op")),
            topic_prefix=os.getenv("ML_BUS_PREFIX", ""),
            redis_url=os.getenv("ML_BUS_REDIS_URL", "redis://localhost:6379/0"),
            redis_stream=os.getenv("ML_BUS_REDIS_STREAM", "ml_events"),
            redis_maxlen=int(os.getenv("ML_BUS_REDIS_MAXLEN", "10000")) if os.getenv("ML_BUS_REDIS_MAXLEN") else None,
        )
```

### 2. Event Status Standardization

**Pattern**: Unified event status tracking with strongly-typed enums across all components.

```python
from ml.config.events import Stage, Source, EventStatus

# ✅ USAGE: Standardized event emission
class DataStore:
    """Data store with standardized event emission."""

    def write_bars(self, bars: list[Bar]) -> None:
        """Write bars with event emission."""
        try:
            self._persist_bars(bars)

            # ✅ REQUIRED: Standardized event emission
            self._emit_event(
                stage=Stage.DATA_INGESTED,
                source=Source.LIVE,
                status=EventStatus.SUCCESS,
                payload={
                    "bars_count": len(bars),
                    "instruments": [bar.instrument_id.value for bar in bars],
                    "ts_event": bars[-1].ts_event,
                    "correlation_id": self._generate_correlation_id()
                }
            )

        except Exception as e:
            self._emit_event(
                stage=Stage.DATA_INGESTED,
                source=Source.LIVE,
                status=EventStatus.FAILED,
                payload={
                    "error": str(e),
                    "bars_attempted": len(bars),
                    "correlation_id": self._generate_correlation_id()
                }
            )
            raise
```

### 3. Non-Blocking Actor Event Bridge

**Pattern**: Actor-side event publishing with background flushing to prevent hot path blocking.

```python
import asyncio
from collections import deque
from threading import Thread, Event as ThreadEvent
from dataclasses import dataclass
from typing import Any

@dataclass
class DomainEvent:
    """Domain event for non-blocking publishing."""

    stage: Stage
    source: Source
    status: EventStatus
    payload: dict[str, Any]
    ts_event: int
    correlation_id: str

class DomainEventBridge:
    """Non-blocking event bridge for ML actors."""

    def __init__(self, bus_config: MessageBusConfig):
        self.config = bus_config
        self.enabled = bus_config.enabled and bus_config.backend != BusBackend.NOOP

        if self.enabled:
            # ✅ REQUIRED: Non-blocking queue for hot path
            self.event_queue: deque[DomainEvent] = deque(maxlen=1000)
            self.publisher = self._create_publisher()
            self.flush_event = ThreadEvent()
            self.background_thread = Thread(target=self._background_flush, daemon=True)
            self.background_thread.start()

        # Metrics for monitoring
        from ml.common.metrics_bootstrap import get_counter, get_gauge
        self.events_published = get_counter(
            "ml_actor_events_published_total",
            "Total events published by actors",
            labels=["stage", "source", "status"]
        )
        self.queue_depth = get_gauge(
            "ml_actor_event_queue_depth",
            "Current event queue depth"
        )

    def publish_event(self, stage: Stage, source: Source,
                     status: EventStatus, payload: dict[str, Any]) -> None:
        """Publish event without blocking hot path."""
        if not self.enabled:
            return

        # ✅ REQUIRED: Zero-allocation hot path
        event = DomainEvent(
            stage=stage,
            source=source,
            status=status,
            payload=payload,
            ts_event=time.time_ns(),
            correlation_id=self._generate_correlation_id()
        )

        try:
            # ✅ REQUIRED: Non-blocking append
            self.event_queue.append(event)
            self.queue_depth.set(len(self.event_queue))

            # Signal background flush without blocking
            self.flush_event.set()

        except Exception:
            # ✅ REQUIRED: Silent failure in hot path
            # Log error in background thread, never block trading
            pass

    def _background_flush(self) -> None:
        """Background thread for event publishing."""
        while True:
            try:
                self.flush_event.wait(timeout=0.1)  # 100ms max delay
                self.flush_event.clear()

                # Batch flush events
                events_to_flush = []
                while self.event_queue and len(events_to_flush) < 100:
                    events_to_flush.append(self.event_queue.popleft())

                if events_to_flush:
                    self._flush_events(events_to_flush)
                    self.queue_depth.set(len(self.event_queue))

            except Exception as e:
                # Log error but continue processing
                logger.error(f"Event flush error: {e}")

    def _flush_events(self, events: list[DomainEvent]) -> None:
        """Flush events to message bus."""
        for event in events:
            try:
                topic = build_topic_for_stage(
                    stage=event.stage,
                    source=event.source,
                    scheme=self.config.scheme,
                    prefix=self.config.topic_prefix
                )

                self.publisher.publish(topic, event.payload)

                # Record metrics
                self.events_published.inc(labels={
                    "stage": event.stage.value,
                    "source": event.source.value,
                    "status": event.status.value
                })

            except Exception as e:
                logger.error(f"Failed to publish event {event.correlation_id}: {e}")

# ✅ INTEGRATION: Actor integration
class MLSignalActor(BaseMLInferenceActor):
    """ML actor with event bridge integration."""

    def __init__(self, config: MLSignalActorConfig):
        super().__init__(config)

        # ✅ REQUIRED: Non-blocking event bridge
        bus_config = MessageBusConfig.from_env()
        self.event_bridge = DomainEventBridge(bus_config)

    def on_bar(self, bar: Bar) -> None:
        """Bar handler with event publishing."""
        # Hot path processing
        features = self.compute_features(bar)
        prediction = self.model.predict(features)

        # ✅ REQUIRED: Non-blocking event publishing
        if prediction > self.config.prediction_threshold:
            self.event_bridge.publish_event(
                stage=Stage.PREDICTION_EMITTED,
                source=Source.HISTORICAL,
                status=EventStatus.SUCCESS,
                payload={
                    "instrument_id": bar.instrument_id.value,
                    "prediction": prediction,
                    "confidence": self._compute_confidence(prediction),
                    "ts_event": bar.ts_event,
                    "model_id": self.config.model_id
                }
            )
```

### 4. Event Correlation and Lineage Tracking

**Pattern**: Deterministic correlation IDs for complete event lineage across pipeline stages.

```python
import hashlib
from typing import Any

class EventCorrelationManager:
    """Manages event correlation and lineage tracking."""

    @staticmethod
    def generate_correlation_id(instrument_id: str, ts_event: int,
                              stage: Stage, additional_context: str = "") -> str:
        """Generate deterministic correlation ID."""
        # ✅ REQUIRED: Deterministic correlation based on content
        content = f"{instrument_id}:{ts_event}:{stage.value}:{additional_context}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @staticmethod
    def create_event_payload(base_payload: dict[str, Any],
                           correlation_id: str,
                           parent_correlation_id: str | None = None) -> dict[str, Any]:
        """Create standardized event payload with correlation."""
        payload = {
            **base_payload,
            "correlation_id": correlation_id,
            "ts_init": time.time_ns(),
        }

        if parent_correlation_id:
            payload["parent_correlation_id"] = parent_correlation_id

        return payload

# ✅ USAGE: Complete lineage tracking
class FeatureStore:
    """Feature store with event correlation."""

    def write_features(self, instrument_id: str, features: dict[str, float],
                      ts_event: int, ts_init: int) -> None:
        """Write features with correlation tracking."""

        # Generate correlation ID
        correlation_id = EventCorrelationManager.generate_correlation_id(
            instrument_id=instrument_id,
            ts_event=ts_event,
            stage=Stage.FEATURE_COMPUTED,
            additional_context=f"features_{len(features)}"
        )

        try:
            self._persist_features(instrument_id, features, ts_event, ts_init)

            # Emit success event with correlation
            payload = EventCorrelationManager.create_event_payload(
                base_payload={
                    "instrument_id": instrument_id,
                    "feature_count": len(features),
                    "ts_event": ts_event,
                    "feature_names": list(features.keys())
                },
                correlation_id=correlation_id
            )

            self._emit_event(Stage.COMPUTED, Source.FEATURES, EventStatus.SUCCESS, payload)

        except Exception as e:
            # Emit failure event with same correlation
            error_payload = EventCorrelationManager.create_event_payload(
                base_payload={
                    "instrument_id": instrument_id,
                    "error": str(e),
                    "feature_count": len(features),
                    "ts_event": ts_event
                },
                correlation_id=correlation_id
            )

            self._emit_event(Stage.COMPUTED, Source.FEATURES, EventStatus.FAILED, error_payload)
            raise
```

### 5. Idempotent Event Processing

**Pattern**: Consumer templates with watermark gating and deduplication to handle duplicate events.

```python
from typing import Set
import time

class IdempotentEventConsumer:
    """Base class for idempotent event processing."""

    def __init__(self, consumer_id: str, watermark_ttl_seconds: int = 3600):
        self.consumer_id = consumer_id
        self.watermark_ttl = watermark_ttl_seconds

        # Correlation ID tracking for deduplication
        self.processed_correlations: dict[str, float] = {}  # correlation_id -> timestamp
        self.processing_lock = threading.Lock()

        # Metrics
        from ml.common.metrics_bootstrap import get_counter
        self.events_processed = get_counter(
            "ml_events_processed_total",
            "Total events processed",
            labels=["consumer_id", "stage", "status"]
        )
        self.duplicate_events = get_counter(
            "ml_duplicate_events_total",
            "Duplicate events detected",
            labels=["consumer_id"]
        )

    def process_event(self, event: dict[str, Any]) -> bool:
        """Process event with idempotent guarantees."""
        correlation_id = event.get("correlation_id")
        if not correlation_id:
            logger.warning("Event missing correlation_id, processing anyway")
            return self._handle_event(event)

        # ✅ REQUIRED: Idempotent processing
        with self.processing_lock:
            current_time = time.time()

            # Clean expired watermarks
            self._cleanup_watermarks(current_time)

            # Check if already processed
            if correlation_id in self.processed_correlations:
                self.duplicate_events.inc(labels={"consumer_id": self.consumer_id})
                logger.debug(f"Skipping duplicate event {correlation_id}")
                return True

            # Process event
            try:
                result = self._handle_event(event)

                if result:
                    # Mark as processed
                    self.processed_correlations[correlation_id] = current_time

                    self.events_processed.inc(labels={
                        "consumer_id": self.consumer_id,
                        "stage": event.get("stage", "unknown"),
                        "status": "success"
                    })

                return result

            except Exception as e:
                self.events_processed.inc(labels={
                    "consumer_id": self.consumer_id,
                    "stage": event.get("stage", "unknown"),
                    "status": "error"
                })
                logger.error(f"Event processing failed for {correlation_id}: {e}")
                raise

    def _handle_event(self, event: dict[str, Any]) -> bool:
        """Override this method for specific event handling logic."""
        raise NotImplementedError("Subclasses must implement _handle_event")

    def _cleanup_watermarks(self, current_time: float) -> None:
        """Remove expired correlation IDs."""
        cutoff_time = current_time - self.watermark_ttl
        expired_ids = [
            correlation_id for correlation_id, timestamp
            in self.processed_correlations.items()
            if timestamp < cutoff_time
        ]

        for correlation_id in expired_ids:
            del self.processed_correlations[correlation_id]

# ✅ EXAMPLE: Feature validation consumer
class FeatureValidationConsumer(IdempotentEventConsumer):
    """Consumer for validating computed features."""

    def __init__(self):
        super().__init__("feature_validator")
        self.feature_validator = FeatureValidator()

    def _handle_event(self, event: dict[str, Any]) -> bool:
        """Validate feature computation events."""
        if event.get("stage") != Stage.COMPUTED.value:
            return True  # Not our event type

        if event.get("source") != Source.FEATURES.value:
            return True  # Not feature events

        # Extract feature data
        instrument_id = event.get("instrument_id")
        feature_names = event.get("feature_names", [])

        # Validate features
        validation_result = self.feature_validator.validate_features(
            instrument_id=instrument_id,
            feature_names=feature_names
        )

        if not validation_result.is_valid:
            logger.warning(
                f"Feature validation failed for {instrument_id}: "
                f"{validation_result.errors}"
            )

            # Could emit validation failure event here

        return True
```

### 6. Production-Ready Message Bus Integration

**Pattern**: Multiple backend support with Redis streams for production and in-memory for testing.

```python
from abc import ABC, abstractmethod
import redis
from typing import Callable, Any

class MessageBusPublisher(ABC):
    """Abstract message bus publisher."""

    @abstractmethod
    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        """Publish message to topic."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close publisher resources."""
        pass

class RedisStreamPublisher(MessageBusPublisher):
    """Redis streams publisher for production."""

    def __init__(self, redis_url: str, stream_name: str, maxlen: int | None = None):
        self.redis_client = redis.from_url(redis_url)
        self.stream_name = stream_name
        self.maxlen = maxlen

        # Test connection
        self.redis_client.ping()

    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        """Publish to Redis stream."""
        stream_data = {
            "topic": topic,
            "payload": json.dumps(payload),
            "ts_published": time.time_ns()
        }

        self.redis_client.xadd(
            name=self.stream_name,
            fields=stream_data,
            maxlen=self.maxlen,
            approximate=True
        )

    def close(self) -> None:
        """Close Redis connection."""
        self.redis_client.close()

class InMemoryPublisher(MessageBusPublisher):
    """In-memory publisher for testing."""

    def __init__(self):
        self.published_events: list[tuple[str, dict[str, Any]]] = []
        self.subscribers: dict[str, list[Callable]] = {}

    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        """Publish to in-memory subscribers."""
        self.published_events.append((topic, payload))

        # Notify subscribers
        for subscriber in self.subscribers.get(topic, []):
            try:
                subscriber(payload)
            except Exception as e:
                logger.error(f"Subscriber error for topic {topic}: {e}")

    def subscribe(self, topic: str, callback: Callable[[dict], None]) -> None:
        """Subscribe to topic for testing."""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)

    def close(self) -> None:
        """Clean up in-memory state."""
        self.published_events.clear()
        self.subscribers.clear()

# ✅ FACTORY: Environment-driven publisher creation
def create_message_bus_publisher(config: MessageBusConfig) -> MessageBusPublisher:
    """Create publisher based on configuration."""
    if not config.enabled or config.backend == BusBackend.NOOP:
        return NoOpPublisher()

    elif config.backend == BusBackend.REDIS:
        return RedisStreamPublisher(
            redis_url=config.redis_url,
            stream_name=config.redis_stream,
            maxlen=config.redis_maxlen
        )

    elif config.backend == BusBackend.MEMORY:
        return InMemoryPublisher()

    else:
        raise ValueError(f"Unsupported message bus backend: {config.backend}")
```

## Implementation Guidelines

### 1. Event Publishing Requirements

All ML components MUST:

- **Use EventStatus Enum**: Never use string literals for status
- **Generate Correlation IDs**: Deterministic IDs for lineage tracking
- **Non-blocking Publishing**: Use DomainEventBridge in actors
- **Standardized Payloads**: Include required fields (correlation_id, ts_event, ts_init)
- **Error Event Emission**: Always emit events for failures

### 2. Topic Scheme Configuration

```python
# Production: Domain-operation scheme
export ML_BUS_SCHEME="domain_op"
# Topics: ml.features.computed, ml.models.predicted

# Monitoring: Stage-first scheme
export ML_BUS_SCHEME="stage_first"
# Topics: computed.ml.features, predicted.ml.models
```

### 3. Event Consumer Patterns

```python
class MyEventConsumer(IdempotentEventConsumer):
    """Example consumer implementation."""

    def __init__(self):
        super().__init__("my_consumer")

    def _handle_event(self, event: dict[str, Any]) -> bool:
        """Handle specific event types."""
        stage = event.get("stage")
        source = event.get("source")

        if stage == Stage.PREDICTED.value and source == Source.MODELS.value:
            return self._handle_prediction_event(event)

        return True  # Ignore other events
```

### 4. Testing Integration

```python
class TestEventDrivenPipeline:
    """Test event-driven pipeline components."""

    def test_event_correlation(self):
        """Test event correlation across stages."""
        # Configure in-memory bus
        config = MessageBusConfig(
            enabled=True,
            backend=BusBackend.MEMORY,
            scheme=TopicScheme.DOMAIN_OP
        )

        publisher = create_message_bus_publisher(config)

        # Test event flow
        correlation_id = EventCorrelationManager.generate_correlation_id(
            instrument_id="EUR/USD.SIM",
            ts_event=time.time_ns(),
            stage=Stage.COMPUTED
        )

        # Verify correlation preserved across stages
        assert correlation_id in published_events
```

## Consequences

### Benefits

1. **Real-time Performance**: Non-blocking event publishing preserves hot path latency
2. **Complete Observability**: Full event lineage and correlation tracking
3. **Deployment Flexibility**: Multiple topic schemes support different environments
4. **Fault Tolerance**: Graceful degradation and retry mechanisms
5. **Testing Support**: In-memory bus enables comprehensive testing

### Trade-offs

1. **System Complexity**: Additional infrastructure components required
2. **Memory Usage**: Event queues and correlation tracking consume memory
3. **Eventual Consistency**: Asynchronous processing may introduce delays
4. **Debugging Complexity**: Distributed event flows harder to trace

### Mitigation Strategies

1. **Monitoring**: Comprehensive metrics and health checks
2. **Circuit Breakers**: Automatic fallback when event system fails
3. **Resource Limits**: Queue size limits and TTL for correlation tracking
4. **Documentation**: Clear event flow diagrams and troubleshooting guides

## Related ADRs

- **ADR-003**: Hot/Cold Path Separation (event publishing patterns)
- **ADR-004**: Progressive Fallback Chains (event system resilience)
- **ADR-005**: Centralized Metrics Bootstrap (event metrics)
- **ADR-006**: Production Security Architecture (event security)

## Status

**ACCEPTED** - This ADR establishes the event-driven architecture for all ML pipeline components.

All ML components MUST use the standardized event patterns. The system supports both production Redis streams and testing in-memory buses with automatic environment-based configuration.
