# Events & Message Bus Context

This document describes the ML events topic schemes, configuration flags, and the actor‑side publishing bridge. It complements the event‑driven pipeline checklist and the integration architecture documents.

## Topic Schemes

- Canonical: `ml.{domain}.{operation}.{instrument_id}`
  - Built with `ml.common.message_topics.build_topic(domain, operation, instrument_id)`
  - Stage → (domain, operation) mapping via `map_stage_to_topic_segments(Stage)`
- Stage‑first (optional): `{prefix}.{STAGE}[.{instrument_id}]`
  - Built with `ml.common.message_topics.build_stage_topic(Stage, instrument_id=None, prefix='events.ml')`
  - Select dynamically with `build_topic_for_stage(stage, instrument_id, scheme='domain_op'|'stage_first', prefix='events.ml')`

All builders normalize `instrument_id` to safe characters: `A-Za-z0-9_.-` (reserved `/*#+$` are replaced with `.`).

## Status Semantics

Status values are standardized via the `EventStatus` enum (`ml.config.events`). Emitters
serialize `.value` to payloads and databases. Valid values:

- `success`
- `failed`
- `partial`

Registry operations use `status=success` and place lifecycle hints such as deprecation in
`metadata` (for example, `{"deprecated": true}`), keeping contracts uniform.

## Wildcard Filters

Topic filters (used by the in-memory publisher and example consumers) use dot-separated tokens with:

- `*` matches exactly one token
- `#` matches zero or more tokens

Examples:

- `ml.features.updated.*.*` matches `ml.features.updated.EURUSD.SIM`
- `events.ml.FEATURE_COMPUTED.#` matches both `events.ml.FEATURE_COMPUTED` and `events.ml.FEATURE_COMPUTED.EURUSD.SIM`

## Configuration (Environment)

Publishing is disabled by default. Enable and configure via environment:

- `ML_BUS_ENABLE`: `1|true|yes` to enable (default: disabled)
- `ML_BUS_BACKEND`: `noop|redis` (default: `noop`)
- `ML_BUS_SCHEME`: `domain_op|stage_first` (default: `domain_op`)
- `ML_BUS_TOPIC_PREFIX`: stage‑first prefix (default: `events.ml`)
- `ML_BUS_REDIS_URL`: Redis URL (default: `redis://localhost:6379/0`)
- `ML_BUS_REDIS_STREAM`: Stream name (default: `ml-events`)
- `ML_BUS_REDIS_MAXLEN`: Approximate stream max length (optional)
- `ML_BUS_FROM_ACTOR`: Publish from actor thread using DomainEventBridge (default: off)
- `ML_BUS_FROM_STORE`: Publish from store path (default: off)
- `ML_BUS_THROTTLE_ENABLE`: Enable publish throttling (default: off)
- `ML_BUS_THROTTLE_RATE`: Tokens per second per topic (default: 100.0)
- `ML_BUS_THROTTLE_BURST`: Burst tokens per topic (default: 100)

Parser: `ml.config.bus.MessageBusConfig.from_env()`

## Publisher Adapters

- Protocol: `ml.common.message_bus.MessagePublisherProtocol`
- Default: `NoopPublisher` (safe)
- Optional: `RedisStreamsPublisher` (XADD `{topic, payload}` JSON payload)
- Factory: `ml.common.message_bus.publisher_from_config(MessageBusConfig)`

## Actor Bridge (Non‑Blocking)

`ml.actors.ml_domain_events.DomainEventBridge` provides an actor‑side enqueue API with a background flusher:

```python
from ml.common.throttler import Throttler

throttler = Throttler(rate_per_sec=100.0, burst=100) if throttle_enabled else None
bridge = DomainEventBridge(publisher, max_queue=4096, throttler=throttler)
bridge.start()
bridge.publish(topic, payload)   # O(1) enqueue on actor thread
bridge.stop(drain=True)
```

The bridge drops events when the queue is full (backpressure); callers can inspect the boolean return. Downstream observability can aggregate drop counts.

## Integration

- `ActorBusConfig.from_env()` resolves actor vs store path, scheme, and throttling parameters.
- `MLIntegrationManager.set_message_publisher(...)` attaches a publisher to the `DataStore` when enabled.
- Store‑level publishing remains off unless a publisher is configured; actor‑level publishing can use the bridge for strict actor‑thread boundaries.

## Tests

- Unit tests cover topic builders, env parsing, publisher behavior (with a dummy Redis client), and bridge enqueue/flush logic.
- Added wildcard filter tests and an in-memory pub/sub e2e test with an idempotent consumer template.

## Consumer Template

`ml.consumers.idempotent.IdempotentConsumer` demonstrates correlation-id deduplication and watermark gating (non-decreasing `ts_max` per `(dataset_id, instrument_id, source)`).

Example:

```python
from ml.common.in_memory_bus import InMemoryPublisher
from ml.consumers.idempotent import IdempotentConsumer
from ml.common.message_topics import build_stage_topic

bus = InMemoryPublisher()
consumer = IdempotentConsumer()
bus.subscribe("events.ml.FEATURE_COMPUTED.#", lambda t, p: consumer.process(p))

topic = build_stage_topic("FEATURE_COMPUTED", "EURUSD.SIM")
payload = {"dataset_id": "features", "instrument_id": "EURUSD.SIM", "source": "historical", "ts_max": 100, "metadata": {"correlation_id": "CID-1"}}
bus.publish(topic, payload)
```

See also:

- `ml/docs/implementation/event_driven_ml_pipeline_checklist.md` (Phase 1)
- `ml/docs/architecture/event_driven_ml_pipeline_exploration.md`
