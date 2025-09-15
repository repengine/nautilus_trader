"""
ML Event Consumers.

This module provides consumer implementations for event-driven ML pipelines,
following the Universal ML Architecture Patterns. All consumers implement
protocol-first interfaces and support idempotent replay for reliable processing.

Components
----------
Core Protocols:
    ConsumerProtocol: Protocol interface for message consumers
    Envelope: Canonical event envelope for cross-domain processing
    StageLike: Type alias for pipeline stages

Consumer Implementations:
    AggregatingConsumer: Watermark-gated aggregator with timestamp ordering
    LineageWriter: Correlation/lineage persistence to ObservabilityService
    RetriableConsumer: Bounded retry wrapper with DLQ publishing
    IdempotentConsumer: Template for idempotent processing with watermarks
    RedisStreamsConsumer: Redis Streams consumer with idempotent gating

Utility Types:
    RetryPolicy: Configuration for retry behavior
    ConsumerKey: Type alias for consumer partition keys
    OnEvent: Callback type for event handlers
    TopicMapper: Function type for topic transformation

Usage
-----
Basic consumer pattern::

    from ml.consumers import ConsumerProtocol, Envelope

    class MyConsumer:
        def handle(self, topic: str, envelope: Envelope) -> None:
            # Process envelope
            pass

Aggregating with watermarks::

    from ml.consumers import AggregatingConsumer, Envelope

    consumer = AggregatingConsumer()
    consumer.handle("events.features", envelope)
    flushed = consumer.advance_watermark("EUR/USD", watermark_ns)

Retry with DLQ::

    from ml.consumers import RetriableConsumer, RetryPolicy

    policy = RetryPolicy(max_attempts=5)
    consumer = RetriableConsumer(handler, dlq_publisher, policy)

Notes
-----
- All consumers support idempotent replay via event ID tracking
- Aggregating consumers enforce timestamp ordering per instrument
- Redis consumers are cold-path only, not for hot-path processing
- Lineage writers integrate with ObservabilityService for tracing

See Also
--------
ml.common.message_bus : Message bus protocols and publishers
ml.config.events : Event stage enumeration and types
ml.observability.service : Observability and lineage tracking

"""

from __future__ import annotations

from .aggregator import AggregatingConsumer
from .aggregator import TopicMapper
from .idempotent import ConsumerKey
from .idempotent import IdempotentConsumer
from .lineage_writer import LineageWriter
from .protocols import ConsumerProtocol
from .protocols import Envelope
from .protocols import StageLike
from .redis_streams_consumer import OnEvent
from .redis_streams_consumer import RedisStreamsConsumer
from .retry import RetriableConsumer
from .retry import RetryPolicy


__all__ = [
    "AggregatingConsumer",
    "ConsumerKey",
    "ConsumerProtocol",
    "Envelope",
    "IdempotentConsumer",
    "LineageWriter",
    "OnEvent",
    "RedisStreamsConsumer",
    "RetriableConsumer",
    "RetryPolicy",
    "StageLike",
    "TopicMapper",
]
