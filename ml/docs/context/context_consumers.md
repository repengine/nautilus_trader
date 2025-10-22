# Context: Consumers Module

**Last Updated**: 2025-10-19
**Module Size**: ~1,517 lines across 10 files
**Purpose**: Event-driven consumer patterns for ML pipeline message processing

---

## Overview

The `ml/consumers/` module provides production-ready consumer implementations for the event-driven ML pipeline. It implements protocol-first consumer interfaces with idempotent replay, watermark-based ordering, and resilient processing patterns. The module serves as the backbone for consuming ML pipeline events from Redis Streams and in-memory message buses.

**Core Philosophy:**
- **Protocol-First Design**: All consumers implement `ConsumerProtocol` for structural typing
- **Idempotent Replay**: Correlation ID tracking prevents duplicate processing
- **Watermark Ordering**: Monotonic timestamp progression per consumer partition key
- **Progressive Fallback**: Graceful degradation when external dependencies unavailable
- **Cold-Path Only**: Consumers are designed for cold-path processing, not hot-path inference

**Key Components:**

```
ml/consumers/
├── protocols.py                      # Protocol interfaces and envelope type (64 lines)
├── idempotent.py                     # Idempotent consumer template (91 lines)
├── redis_streams_consumer.py         # Redis Streams consumer (85 lines)
├── aggregator.py                     # Watermark-gated aggregating consumer (172 lines)
├── lineage_writer.py                 # Lineage persistence consumer (44 lines)
├── retry.py                          # Retry/DLQ consumer wrapper (69 lines)
├── streaming_training.py             # Streaming training consumer (592 lines)
├── streaming_training_service.py     # Service wrapper for persistence (71 lines)
├── streaming_training_worker.py      # Long-running worker for streaming events (209 lines)
└── __init__.py                       # Public API exports (120 lines)
```

---

## Architecture

### Consumer Hierarchy

```
ConsumerProtocol (Protocol)
    ├── AggregatingConsumer (watermark-gated buffering)
    ├── LineageWriter (observability tracking)
    └── RetriableConsumer (retry/DLQ wrapper)

RedisStreamsConsumer (Redis integration)
    └── Uses IdempotentConsumer for gating

StreamingTrainingConsumer (streaming training events)
    └── Uses StreamingTrainingStateStore for persistence
```

### Event Flow Pattern

```
Redis Stream / In-Memory Bus
    │
    ├─→ RedisStreamsConsumer
    │       └─→ IdempotentConsumer (dedup + watermark)
    │               └─→ Handler (user callback)
    │
    ├─→ AggregatingConsumer
    │       ├─→ Buffer events per instrument
    │       ├─→ Sort by ts_event
    │       └─→ Flush on watermark advance
    │               └─→ Downstream publisher
    │
    ├─→ LineageWriter
    │       └─→ ObservabilityService
    │
    └─→ StreamingTrainingConsumer
            └─→ StreamingTrainingStateStore
                    ├─→ Plans
                    ├─→ Results
                    └─→ Heartbeats
```

### Universal ML Architecture Pattern Alignment

**Pattern 2: Protocol-First Interface Design** ✓
- All consumers implement `ConsumerProtocol` (`protocols.py:52-61`)
- Structural typing enables duck-typed testing with DummyConsumer
- No concrete class coupling between components

**Pattern 5: Centralized Metrics Bootstrap** ✓
- Uses `ml.common.metrics_bootstrap.get_counter/get_gauge` (`streaming_training.py:22-41`)
- Never imports `prometheus_client` directly
- Metrics registered at module import time

---

## Core Components

### 1. ConsumerProtocol (`protocols.py`)

**Purpose**: Define the minimal protocol interface for all message consumers.

**Interface Definition** (`protocols.py:52-61`):
```python
class ConsumerProtocol(Protocol):
    """Protocol for message consumers."""

    def handle(self, topic: str, envelope: Envelope) -> None:
        """Process a message delivered on a topic."""
        ...
```

**Canonical Event Envelope** (`protocols.py:20-49`):

The `Envelope` TypedDict is the canonical event structure passed between pipeline stages:

```python
class Envelope(TypedDict):
    id: str                    # Unique event ID (UUID recommended)
    parent_id: str | None      # Parent event for lineage tracking
    instrument_id: str         # Instrument identifier for routing
    ts_event: int              # Event timestamp (nanoseconds)
    stage: StageLike           # Pipeline stage (e.g., "FEATURE_COMPUTED")
    correlation_id: str        # Correlation ID tying event chain
    payload: dict[str, object] # Opaque payload content
```

**Stage Type Alias** (`protocols.py:17`):
```python
StageLike: TypeAlias = Stage | str
```

Supports both `ml.config.events.Stage` enum values and raw strings for flexibility.

**Design Notes:**
- TypedDict provides static type checking without runtime overhead
- Minimal required fields enable cross-domain event processing
- Envelope pattern separates metadata from business logic payload

---

### 2. IdempotentConsumer (`idempotent.py`)

**Purpose**: Template implementation for duplicate detection and watermark-based ordering guarantees.

**Class Definition** (`idempotent.py:28-89`):
```python
@dataclass
class IdempotentConsumer:
    seen: set[str] = field(default_factory=set)
    watermarks: MutableMapping[ConsumerKey, int] = field(default_factory=dict)
```

**Consumer Key Structure** (`idempotent.py:24`):
```python
ConsumerKey = tuple[str, str, str]  # (dataset_id, instrument_id, source)
```

The consumer key partitions events by dataset, instrument, and source for independent watermark tracking.

**Core Methods:**

**should_process** (`idempotent.py:44-69`):
```python
def should_process(self, payload: Mapping[str, Any]) -> bool:
    """Return True if event should be processed under idempotency/watermarks."""
    # 1. Extract correlation_id from metadata
    # 2. Check if already seen (duplicate)
    # 3. Check watermark ordering per consumer key
    # 4. Return True only if both checks pass
```

**process** (`idempotent.py:71-88`):
```python
def process(self, payload: Mapping[str, Any]) -> bool:
    """Apply idempotency + watermark gating and update state if accepted."""
    if not self.should_process(payload):
        return False
    # Update seen set and watermark
    self.seen.add(correlation_id)
    self.watermarks[key] = ts_max
    return True
```

**Required Payload Structure:**

Events must contain these fields for proper gating (`idempotent.py:48-61`):
- `metadata.correlation_id: str` — Unique event correlation for deduplication
- `dataset_id: str` — Dataset type (features, predictions, signals)
- `instrument_id: str` — Instrument identifier
- `source: str` — Data source (historical, live, backfill)
- `ts_max: int` — Watermark timestamp in nanoseconds (must be non-decreasing)

**Error Handling** (`idempotent.py:68-69`):
- Catches all exceptions and returns `False` for safety
- Defensive guard prevents malformed payloads from breaking consumer loops

**Limitations:**
- In-memory storage only (see limitations in Production Considerations)
- No persistence across process restarts
- Memory grows unbounded with correlation IDs

---

### 3. RedisStreamsConsumer (`redis_streams_consumer.py`)

**Purpose**: Production Redis Streams integration with built-in idempotent gating and resilient processing.

**Class Definition** (`redis_streams_consumer.py:24-83`):
```python
class RedisStreamsConsumer:
    """Minimal Redis Streams consumer that gates events and invokes a handler."""

    def __init__(
        self,
        *,
        url: str,                              # Redis connection URL
        stream: str,                           # Stream name to consume
        handler: OnEvent,                      # Event handler callback
        gate: IdempotentConsumer | None = None # Optional custom gating
    ) -> None
```

**Handler Callback Type** (`redis_streams_consumer.py:21`):
```python
OnEvent = Callable[[str, dict[str, Any]], None]
```

**Core Method: poll_once** (`redis_streams_consumer.py:50-82`):
```python
def poll_once(self, *, count: int = 100, block_ms: int = 0, last_id: str = "$") -> int:
    """Read a batch via XREAD and process accepted events. Returns processed count."""
    # Parameters:
    #   count: Max messages per batch (default: 100)
    #   block_ms: XREAD blocking timeout in ms (0 = non-blocking)
    #   last_id: Stream position ("$" = latest)
    # Returns:
    #   Number of events successfully processed
```

**Processing Flow** (`redis_streams_consumer.py:59-82`):
1. Read batch from Redis stream via `XREAD`
2. For each entry, extract `topic` and `payload` JSON
3. Parse JSON payload (fallback to empty dict on error)
4. Apply idempotent gating via `self._gate.process(payload)`
5. Only invoke handler for accepted events
6. Log handler exceptions and continue (resilient processing)
7. Return total processed count

**Redis Stream Message Format** (`redis_streams_consumer.py:67-68`):

The consumer expects Redis stream entries with these fields:
- `topic`: Message topic string for routing
- `payload`: JSON-encoded event payload

**Error Handling Strategy:**

**Connection Errors** (`redis_streams_consumer.py:57-62`):
- Returns 0 when Redis client unavailable
- Safe fallback allows graceful degradation

**JSON Parse Errors** (`redis_streams_consumer.py:70-72`):
- Uses empty dict as fallback payload
- Prevents malformed JSON from breaking consumer

**Handler Exceptions** (`redis_streams_consumer.py:78-81`):
- Logs at debug level with `exc_info=True`
- Continues processing remaining events (resilient loop)

**Stream Read Errors** (`redis_streams_consumer.py:61-62`):
- Returns 0 and continues
- Suitable for retry loops in production

**Redis Client Initialization** (`redis_streams_consumer.py:42-46`):
```python
try:
    import redis
    self._client = redis.Redis.from_url(url, decode_responses=True)
except Exception:  # pragma: no cover
    self._client = None  # Graceful fallback when redis unavailable
```

**Design Notes:**
- Minimal, safe example implementation (not production-hardened)
- No consumer group support (single consumer only)
- No acknowledgment or dead-letter queue
- Cold-path only (not for hot-path processing)

---

### 4. AggregatingConsumer (`aggregator.py`)

**Purpose**: Aggregate and emit envelopes in timestamp order under watermark gating with idempotent replay.

**Class Definition** (`aggregator.py:32-172`):
```python
@dataclass(slots=True)
class AggregatingConsumer:
    downstream: MessagePublisherProtocol | None = None
    topic_mapper: TopicMapper | None = None
    scheme: str = "domain_op"
    prefix: str = "aggregated.ml"

    _buffer: dict[str, list[Envelope]] = field(default_factory=dict)
    _last_emitted_ts: dict[str, int] = field(default_factory=dict)
    _processed_ids: set[str] = field(default_factory=set)
```

**TopicMapper Type** (`aggregator.py:29`):
```python
TopicMapper = Callable[[str], str]
```

**Core Methods:**

**handle** (`aggregator.py:62-74`):
```python
def handle(self, topic: str, envelope: Envelope) -> None:
    """Buffer envelope for its instrument; ignore duplicates by id."""
    eid = envelope["id"]
    if eid in self._processed_ids:
        aggregator_duplicates_total.inc()
        return
    inst = envelope["instrument_id"]
    buf = self._buffer.setdefault(inst, [])
    buf.append(envelope)
    aggregator_buffer_size.labels(instrument=inst).set(len(buf))
```

**advance_watermark** (`aggregator.py:76-172`):
```python
def advance_watermark(self, instrument_id: str, watermark_ns: int) -> list[Envelope]:
    """
    Advance instrument watermark and flush eligible envelopes in order.

    Returns the flushed envelopes in strictly non-decreasing timestamp order.
    Enforces idempotency (same id is never emitted twice).
    """
    # Processing steps:
    # 1. Retrieve buffer for instrument
    # 2. Sort buffer by ts_event (stable sort)
    # 3. Iterate and flush events <= watermark_ns
    # 4. Enforce monotonic non-decreasing timestamps
    # 5. Remove flushed items from buffer
    # 6. Update metrics (buffer size, flushed count, watermark lag)
    # 7. Forward to downstream publisher if configured
    # 8. Return flushed envelopes
```

**Watermark Ordering Guarantees** (`aggregator.py:87-104`):

The aggregator enforces strict ordering:
- Events sorted by `ts_event` within each instrument buffer
- Only events with `ts_event <= watermark_ns` are flushed
- Monotonic non-decreasing timestamps enforced per instrument
- Out-of-order events skipped until watermark progresses

**Downstream Publishing** (`aggregator.py:119-170`):

When `downstream` publisher is configured, flushed events are forwarded:
1. Use custom `topic_mapper` if provided
2. Otherwise, use canonical `build_topic_for_stage` with Stage enum parsing
3. Publish envelope as payload with all metadata fields
4. Best-effort publishing (logs errors but continues)

**Topic Building Logic** (`aggregator.py:122-147`):
```python
if self.topic_mapper:
    out_topic = self.topic_mapper(ev.get("stage", "events"))
else:
    # Try to parse stage string to Stage enum
    try:
        stage = Stage(stage_str) if stage_str else None
    except ValueError:
        stage = None  # Fallback for invalid stage strings

    if stage:
        out_topic = build_topic_for_stage(
            stage=stage,
            instrument_id=instrument_id,
            scheme=self.scheme,
            prefix=self.prefix,
        )
    else:
        # Fallback for events without valid stage
        out_topic = build_topic("events", "updated", instrument_id)
```

**Metrics Integration** (`aggregator.py:21-24`):

The aggregator exposes these Prometheus metrics:
- `nautilus_ml_aggregator_buffer_size{instrument}` — Current buffer size per instrument
- `nautilus_ml_aggregator_flushed_total{instrument}` — Cumulative flushed event count
- `nautilus_ml_aggregator_duplicates_total` — Duplicate events dropped
- `nautilus_ml_aggregator_watermark_lag_seconds{instrument}` — Watermark lag in seconds

**Metric Updates** (`aggregator.py:68, 74, 111-116`):
- Buffer size updated on every `handle()` and `advance_watermark()`
- Flushed count incremented on successful flush
- Watermark lag computed as `(watermark_ns - last_ts) / 1e9`

**Error Handling** (`aggregator.py:159-170`):
- Downstream publish wrapped in try/except
- Uses `log_best_effort` for graceful error logging with `exc_info=True`
- Aggregator state already updated before publish (no rollback on failure)

---

### 5. LineageWriter (`lineage_writer.py`)

**Purpose**: Write correlation/lineage entries from envelopes to `ObservabilityService` for tracing.

**Class Definition** (`lineage_writer.py:14-42`):
```python
@dataclass(slots=True)
class LineageWriter:
    """
    Write correlation/lineage entries from envelopes to ObservabilityService.

    Idempotent: deduplicates by event id to avoid duplicate writes on replay.
    """
    service: ObservabilityService
    _seen_ids: set[str] = field(default_factory=set)
```

**Core Method: handle** (`lineage_writer.py:26-41`):
```python
def handle(self, _topic: str, envelope: Envelope) -> None:
    eid = envelope["id"]
    if eid in self._seen_ids:
        return  # Skip duplicates
    self._seen_ids.add(eid)

    # Map envelope to correlation row
    self.service.add_correlation(
        correlation_id=envelope["correlation_id"],
        event_id=envelope["id"],
        parent_event_id=envelope["parent_id"],
        instrument_id=envelope["instrument_id"],
        domain="ml",  # unified ML domain
        lineage_depth=0 if envelope["parent_id"] is None else 1,
        ts_event=int(envelope["ts_event"]),
        propagation_path=[envelope["stage"]],
    )
```

**Lineage Depth Logic** (`lineage_writer.py:38`):
- Root events (no parent): `lineage_depth=0`
- Child events (has parent): `lineage_depth=1`

**Integration with ObservabilityService:**

The lineage writer integrates with `ml.observability.service.ObservabilityService` to persist correlation metadata to PostgreSQL (`observability.event_correlation` table). This enables:
- Tracing event chains across pipeline stages
- Debugging pipeline failures by correlation ID
- Analyzing lineage for data quality issues

**Design Notes:**
- Extremely simple (44 lines total)
- Uses in-memory set for deduplication (same limitation as IdempotentConsumer)
- Topic parameter ignored (focus on envelope content only)

---

### 6. RetriableConsumer (`retry.py`)

**Purpose**: Wrap a handler with bounded synchronous retries and DLQ publishing on final failure.

**Configuration** (`retry.py:16-17`):
```python
@dataclass(slots=True)
class RetryPolicy:
    max_attempts: int = 3
```

**Class Definition** (`retry.py:23-67`):
```python
@dataclass(slots=True)
class RetriableConsumer:
    """
    Wrap a handler with bounded synchronous retries and DLQ publishing.

    For simplicity and deterministic testing, failures are retried immediately up
    to ``policy.max_attempts``. On final failure, the envelope is published to the
    DLQ with topic ``dlq.{stage}``.
    """
    handler: HandlerFunc
    dlq: MessagePublisherProtocol
    policy: RetryPolicy = field(default_factory=RetryPolicy)

    _attempts: dict[str, int] = field(default_factory=dict)
```

**HandlerFunc Type** (`retry.py:20`):
```python
HandlerFunc = Callable[[str, Envelope], None]
```

**Core Method: handle** (`retry.py:40-66`):
```python
def handle(self, topic: str, envelope: Envelope) -> None:
    eid = envelope["id"]
    attempts = self._attempts.get(eid, 0)

    # Retry loop (synchronous)
    while attempts < self.policy.max_attempts:
        try:
            self.handler(topic, envelope)
            # Success; reset attempt counter
            self._attempts.pop(eid, None)
            return
        except Exception:
            attempts += 1
            self._attempts[eid] = attempts

    # Final failure: publish to DLQ
    dlq_topic = f"dlq.{envelope['stage']}"
    self.dlq.publish(
        dlq_topic,
        {
            "id": envelope["id"],
            "parent_id": envelope["parent_id"],
            "instrument_id": envelope["instrument_id"],
            "ts_event": envelope["ts_event"],
            "stage": envelope["stage"],
            "correlation_id": envelope["correlation_id"],
            "payload": envelope["payload"],
            "attempts": attempts,
        },
    )
```

**Retry Behavior:**
- Synchronous immediate retries (no backoff)
- Deterministic for testing purposes
- Not production-hardened (real systems need exponential backoff, jitter)

**DLQ Topic Pattern** (`retry.py:53`):
```python
dlq_topic = f"dlq.{envelope['stage']}"
```

Example: `dlq.FEATURE_COMPUTED`, `dlq.PREDICTION_EMITTED`

**Design Notes:**
- Simple synchronous implementation for testing/examples
- Production systems should use async retries with backoff
- No circuit breaker or rate limiting

---

### 7. StreamingTrainingConsumer (`streaming_training.py`)

**Purpose**: Consume streaming training payloads (plans, results, heartbeats) and persist state for monitoring.

**State Store Protocol** (`streaming_training.py:207-238`):
```python
class StreamingTrainingStateStore(Protocol):
    """Protocol describing state persistence for streaming training consumers."""

    def record_plan(self, record: StreamingPlanRecord) -> None: ...
    def record_result(self, record: StreamingResultRecord) -> None: ...
    def record_heartbeat(self, record: StreamingHeartbeatRecord) -> None: ...
    def get_plan(self, plan_id: str) -> StreamingPlanRecord | None: ...
    def get_result(self, plan_id: str) -> StreamingResultRecord | None: ...
    def latest_heartbeat(self, worker_id: str) -> StreamingHeartbeatRecord | None: ...
    def outstanding_plan_ids(self) -> tuple[str, ...]: ...
    def outstanding_plan_ids_for_dataset(self, dataset_id: str) -> tuple[str, ...]: ...
    def snapshot(self) -> Mapping[str, Any]: ...
    def restore(self, snapshot: Mapping[str, Any]) -> None: ...
```

**Record Types:**

**StreamingPlanRecord** (`streaming_training.py:66-110`):
```python
@dataclass(slots=True)
class StreamingPlanRecord:
    plan_id: str
    dataset_id: str
    status: EventStatus
    created_at: datetime
    caps: Mapping[str, float | int | None]
    limits: Mapping[str, int]
    metadata_summary: Mapping[str, int]
    streaming_config: Mapping[str, Any]
    correlation_id: str
    topic: str
```

Represents a dataset training plan emitted on the streaming pipeline bus.

**StreamingResultRecord** (`streaming_training.py:113-157`):
```python
@dataclass(slots=True)
class StreamingResultRecord:
    plan_id: str
    dataset_id: str
    status: EventStatus
    completed_at: datetime
    model_id: str
    metrics: Mapping[str, float]
    artifact_paths: Mapping[str, str]
    telemetry: Mapping[str, Any]
    correlation_id: str
    topic: str
```

Represents a completed streaming training job result.

**StreamingHeartbeatRecord** (`streaming_training.py:160-204`):
```python
@dataclass(slots=True)
class StreamingHeartbeatRecord:
    plan_id: str
    dataset_id: str
    status: EventStatus
    worker_id: str
    progress_pct: float
    rss_mb: float
    shards_processed: int
    timestamp: datetime
    correlation_id: str
    topic: str
```

Represents a worker heartbeat update during streaming training.

**State Store Implementations:**

**InMemoryStreamingTrainingStateStore** (`streaming_training.py:241-308`):
- Simple in-memory implementation using dicts
- No persistence across process restarts
- Suitable for testing and development

**FileBackedStreamingTrainingStateStore** (`streaming_training.py:311-382`):
- Delegates to in-memory store with JSON file persistence
- Atomic writes via temp file + replace (`streaming_training.py:372-376`)
- Auto-restore on initialization (`streaming_training.py:323-333`)
- Safe error handling with warnings (`streaming_training.py:329-332, 377-382`)

**Consumer Class** (`streaming_training.py:385-556`):
```python
class StreamingTrainingConsumer:
    """
    Consume streaming training payloads and persist state.

    Args:
        state_store: Optional backing store (defaults to in-memory)
        observability: Optional observability sink for metrics mirroring
    """

    def __init__(
        self,
        state_store: StreamingTrainingStateStore | None = None,
        *,
        observability: ObservabilitySink | None = None,
    ) -> None
```

**Core Method: handle** (`streaming_training.py:416-446`):
```python
def handle(self, topic: str, payload: dict[str, Any]) -> None:
    payload_type = str(payload.get("payload_type", "")).strip()
    correlation_id = str(payload.get("correlation_id", "")).strip()

    if not payload_type or not correlation_id:
        return  # Ignore invalid payloads

    if correlation_id in self._seen_correlations:
        return  # Idempotent deduplication

    try:
        if payload_type == "streaming_plan":
            plan_record = self._parse_plan(topic, payload)
            self._state_store.record_plan(plan_record)
            self._update_backlog_metric(plan_record.dataset_id)
        elif payload_type == "streaming_result":
            result_record = self._parse_result(topic, payload)
            self._state_store.record_result(result_record)
            self._update_backlog_metric(result_record.dataset_id)
        elif payload_type == "streaming_heartbeat":
            heartbeat_record = self._parse_heartbeat(topic, payload)
            self._state_store.record_heartbeat(heartbeat_record)
            self._record_heartbeat_metrics(heartbeat_record)
        else:
            logger.debug("ignoring unsupported payload_type %s", payload_type)
            return

        self._seen_correlations.add(correlation_id)
    except Exception:  # pragma: no cover
        logger.warning("failed to process streaming payload", ...)
```

**Payload Type Routing:**
- `streaming_plan` → `record_plan()` → update backlog metric
- `streaming_result` → `record_result()` → update backlog metric
- `streaming_heartbeat` → `record_heartbeat()` → update worker metrics

**Metrics Integration:**

The consumer exposes these Prometheus metrics (`streaming_training.py:22-41`):
```python
_BACKLOG_GAUGE = get_gauge(
    "ml_tft_streaming_training_backlog",
    "Number of outstanding streaming training plans awaiting completion.",
    labelnames=("dataset_id",),
)
_WORKER_PROGRESS_GAUGE = get_gauge(
    "ml_tft_streaming_worker_progress_pct",
    "Latest progress percentage reported by streaming workers.",
    labelnames=("worker_id",),
)
_WORKER_RSS_GAUGE = get_gauge(
    "ml_tft_streaming_worker_rss_mb",
    "Latest resident set size (MB) reported by streaming workers.",
    labelnames=("worker_id",),
)
_WORKER_COUNT_GAUGE = get_gauge(
    "ml_tft_streaming_workers_active",
    "Current number of active streaming workers per dataset.",
    labelnames=("dataset_id",),
)
```

**Backlog Metric Update** (`streaming_training.py:493-505`):
```python
def _update_backlog_metric(self, dataset_id: str) -> None:
    outstanding = len(self._state_store.outstanding_plan_ids_for_dataset(dataset_id))
    dataset_key = dataset_id or "UNKNOWN"
    _BACKLOG_GAUGE.labels(dataset_id=dataset_key).set(float(outstanding))

    # Mirror to observability sink if configured
    if self._observability is not None:
        timestamp_ns = int(datetime.utcnow().timestamp() * 1_000_000_000)
        self._observability.add_metric(
            metric_name="ml_tft_streaming_training_backlog",
            metric_type="gauge",
            value=float(outstanding),
            timestamp=timestamp_ns,
            labels={"dataset_id": dataset_key},
        )
```

**Heartbeat Metrics Update** (`streaming_training.py:507-555`):

Records worker progress, RSS, and active worker count:
```python
def _record_heartbeat_metrics(self, record: StreamingHeartbeatRecord) -> None:
    worker_key = record.worker_id or "UNKNOWN"
    _WORKER_PROGRESS_GAUGE.labels(worker_id=worker_key).set(record.progress_pct)
    _WORKER_RSS_GAUGE.labels(worker_id=worker_key).set(record.rss_mb)

    # Count active workers by iterating heartbeat snapshot
    dataset_key = record.dataset_id or "UNKNOWN"
    snapshot = self._state_store.snapshot()
    heartbeats_raw = snapshot.get("heartbeats", {})
    active_workers: set[str] = set()
    # ... (worker counting logic)
    _WORKER_COUNT_GAUGE.labels(dataset_id=dataset_key).set(float(len(active_workers)))

    # Mirror to observability sink if configured
    if self._observability is not None:
        # ... (mirror metrics)
```

**Helper Function: attach_streaming_training_monitor** (`streaming_training.py:565-580`):
```python
def attach_streaming_training_monitor(
    bus: SubscriptionBus,
    *,
    state_path: Path,
    observability: ObservabilitySink | None = None,
    topic_pattern: str = "events.ml.#",
) -> StreamingTrainingConsumer:
    """Attach a streaming training consumer to the provided bus."""
    store = FileBackedStreamingTrainingStateStore(state_path)
    consumer = StreamingTrainingConsumer(state_store=store, observability=observability)

    def _handler(topic: str, payload: dict[str, Any]) -> None:
        consumer.handle(topic, dict(payload))

    bus.subscribe(topic_pattern, _handler)
    return consumer
```

Convenient helper for wiring consumer to message bus with file-backed persistence.

---

### 8. StreamingTrainingPersistenceService (`streaming_training_service.py`)

**Purpose**: Service wrapper for streaming training persistence with Redis consumer creation.

**Class Definition** (`streaming_training_service.py:18-69`):
```python
@dataclass(slots=True)
class StreamingTrainingPersistenceService:
    """Persist streaming training events to a durable state store."""

    state_path: Path
    state_store: StreamingTrainingStateStore
    consumer: StreamingTrainingConsumer
```

**Factory Method** (`streaming_training_service.py:26-42`):
```python
@classmethod
def create(
    cls,
    *,
    state_path: Path,
    observability: ObservabilitySink | None = None,
    state_store: StreamingTrainingStateStore | None = None,
) -> StreamingTrainingPersistenceService:
    """Instantiate the service with a file-backed state store."""
    resolved_path = state_path.expanduser()
    store = state_store or FileBackedStreamingTrainingStateStore(resolved_path)
    consumer = StreamingTrainingConsumer(state_store=store, observability=observability)
    return cls(
        state_path=resolved_path,
        state_store=store,
        consumer=consumer,
    )
```

**Core Methods:**

**handle** (`streaming_training_service.py:44-46`):
```python
def handle(self, topic: str, payload: Mapping[str, Any]) -> None:
    """Persist a single streaming training payload."""
    self.consumer.handle(topic, dict(payload))
```

**snapshot** (`streaming_training_service.py:48-50`):
```python
def snapshot(self) -> Mapping[str, Any]:
    """Return the current state snapshot."""
    return self.state_store.snapshot()
```

**create_stream_consumer** (`streaming_training_service.py:52-68`):
```python
def create_stream_consumer(
    self,
    config: MessageBusConfig | None = None,
) -> RedisStreamsConsumer:
    """Build a Redis streams consumer wired to this persistence service."""
    cfg = config or MessageBusConfig.from_env()
    if not cfg.enabled or cfg.backend != "redis":
        raise RuntimeError("streaming persistence requires redis backend")

    def _handler(topic: str, event: dict[str, Any]) -> None:
        self.handle(topic, event)

    return RedisStreamsConsumer(
        url=cfg.redis_url,
        stream=cfg.redis_stream,
        handler=_handler,
    )
```

**Design Notes:**
- Thin wrapper around `StreamingTrainingConsumer`
- Provides factory for Redis consumer creation
- Validates message bus configuration before creating consumer

---

### 9. StreamingTrainingPersistenceWorker (`streaming_training_worker.py`)

**Purpose**: Long-running worker that polls Redis Streams and persists streaming training events.

**Class Definition** (`streaming_training_worker.py:72-206`):
```python
@dataclass(slots=True)
class StreamingTrainingPersistenceWorker:
    """
    Run the streaming training persistence loop against Redis Streams.

    Args:
        config: Configuration controlling polling cadence and persistence paths.
        message_bus_config: Message bus configuration for Redis endpoints.
        observability: Optional observability sink for mirroring backlog metrics.
        state_store: Optional state store implementation (defaults to file store).
        consumer_factory: Optional factory for custom consumer (primarily for tests).
    """

    config: StreamingPersistenceConfig
    message_bus_config: MessageBusConfig = field(default_factory=MessageBusConfig.from_env)
    observability: ObservabilitySink | None = None
    state_store: StreamingTrainingStateStore | None = None
    consumer_factory: ConsumerFactory | None = None
```

**ConsumerFactory Type** (`streaming_training_worker.py:65-68`):
```python
ConsumerFactory = Callable[
    [StreamingTrainingPersistenceService, MessageBusConfig],
    _PollableConsumer,
]
```

**Core Methods:**

**poll_once** (`streaming_training_worker.py:107-126`):
```python
def poll_once(self) -> int:
    """Poll Redis Streams a single time using configured limits."""
    if not self.config.enabled:
        return 0
    consumer = self._ensure_consumer()
    if consumer is None:
        return 0
    try:
        processed = consumer.poll_once(
            count=int(self.config.batch_size),
            block_ms=int(self.config.block_ms),
        )
    except Exception:
        logger.warning("streaming persistence poll failed", ...)
        return 0
    return processed
```

**run_forever** (`streaming_training_worker.py:128-141`):
```python
def run_forever(self) -> None:
    """Start the persistence loop until :meth:`stop` is invoked."""
    if not self.config.enabled:
        logger.info("streaming persistence worker disabled", ...)
        return
    self._stop_event.clear()
    idle_interval = float(self.config.poll_interval_seconds)
    while not self._stop_event.is_set():
        processed = self.poll_once()
        if processed == 0 and idle_interval > 0.0:
            self._stop_event.wait(timeout=idle_interval)
```

**stop** (`streaming_training_worker.py:143-145`):
```python
def stop(self) -> None:
    """Signal the worker loop to exit."""
    self._stop_event.set()
```

**Lazy Initialization:**

**_ensure_service** (`streaming_training_worker.py:152-164`):
- Creates `StreamingTrainingPersistenceService` on first access
- Uses configured state path and observability sink
- Auto-discovers observability service if not provided

**_ensure_consumer** (`streaming_training_worker.py:166-186`):
- Creates Redis consumer on first access
- Uses custom factory if provided (for testing)
- Logs warning and returns None on initialization failure

**_get_observability_sink** (`streaming_training_worker.py:188-206`):
- Lazy-loads `ObservabilityService` on first access
- Wraps in adapter to match `ObservabilitySink` protocol
- Returns None on failure (graceful degradation)

**ObservabilityService Adapter** (`streaming_training_worker.py:27-52`):
```python
@dataclass(slots=True)
class _ObservabilityServiceAdapter:
    service: ObservabilityService

    def add_metric(
        self,
        *,
        metric_name: str,
        metric_type: str,
        value: float,
        timestamp: int,
        labels: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
    ) -> None:
        # Normalize labels to dict format
        normalized: dict[str, Any]
        if labels is None:
            normalized = {}
        elif isinstance(labels, Mapping):
            normalized = {str(key): val for key, val in labels.items()}
        else:
            normalized = {str(key): val for key, val in labels}

        self.service.add_metric(
            metric_name=metric_name,
            metric_type=metric_type,
            value=value,
            timestamp=timestamp,
            labels=normalized,
        )
```

Adapts `ObservabilityService.add_metric` to match `ObservabilitySink` protocol.

---

## Usage Patterns

### Pattern 1: Basic Idempotent Consumer

**File**: `ml/tests/unit/consumers/test_idempotent_consumer.py`

```python
from ml.consumers.idempotent import IdempotentConsumer

# Create consumer with in-memory state
consumer = IdempotentConsumer()

def process_event(payload: dict[str, Any]) -> None:
    if consumer.process(payload):
        # Event accepted - process it
        dataset_id = payload["dataset_id"]
        instrument_id = payload["instrument_id"]
        print(f"Processing {dataset_id} for {instrument_id}")
    else:
        # Event rejected (duplicate or out-of-order)
        print("Event rejected by idempotent consumer")

# Example event
event = {
    "dataset_id": "features",
    "instrument_id": "EURUSD.SIM",
    "source": "historical",
    "ts_max": 1640995200000000000,  # nanoseconds
    "metadata": {"correlation_id": "unique-event-123"},
    "data": {"rsi": 65.4, "macd": 0.002}
}

process_event(event)  # First time: processed
process_event(event)  # Second time: rejected (duplicate)
```

### Pattern 2: Redis Streams Consumer

**File**: `ml/cli/events_consumer.py`

```python
from ml.consumers.redis_streams_consumer import RedisStreamsConsumer
from ml.common.topic_filters import match_topic

def handle_feature_event(topic: str, payload: dict[str, Any]) -> None:
    """Handler for feature computation events."""
    if match_topic("events.ml.FEATURE_COMPUTED.*", topic):
        instrument = payload.get("instrument_id", "unknown")
        print(f"Features computed for {instrument}")

# Create consumer
consumer = RedisStreamsConsumer(
    url="redis://localhost:6379/0",
    stream="ml-events",
    handler=handle_feature_event
)

# Poll for events (non-blocking)
processed = consumer.poll_once(count=50, block_ms=0)
print(f"Processed {processed} events")

# Poll with blocking (wait up to 5 seconds)
processed = consumer.poll_once(count=100, block_ms=5000)
```

### Pattern 3: Aggregating Consumer with Watermarks

**File**: `ml/tests/property/test_consumers_aggregator_properties.py`

```python
from ml.common.in_memory_bus import InMemoryPublisher
from ml.consumers.aggregator import AggregatingConsumer
from ml.consumers.protocols import Envelope

# Setup downstream bus
bus = InMemoryPublisher()

# Create aggregating consumer
agg = AggregatingConsumer(
    downstream=bus,
    scheme="domain_op",
    prefix="aggregated.ml"
)

# Create envelope
envelope: Envelope = {
    "id": "e1",
    "parent_id": None,
    "instrument_id": "EURUSD.SIM",
    "ts_event": 100,
    "stage": "FEATURE_COMPUTED",
    "correlation_id": "c1",
    "payload": {"x": 1}
}

# Buffer event
agg.handle("events.ml.FEATURE_COMPUTED", envelope)

# Advance watermark and flush
flushed = agg.advance_watermark("EURUSD.SIM", watermark_ns=200)
print(f"Flushed {len(flushed)} events")
```

### Pattern 4: Lineage Writer Integration

**File**: `ml/tests/integration/test_lineage_writer_integration.py`

```python
from ml.consumers.lineage_writer import LineageWriter
from ml.observability.service import ObservabilityService
from ml.consumers.protocols import Envelope

# Create observability service
svc = ObservabilityService()
writer = LineageWriter(service=svc)

# Create envelope with lineage
envelope: Envelope = {
    "id": "e1",
    "parent_id": None,
    "instrument_id": "EURUSD.SIM",
    "ts_event": 1640995200000000000,
    "stage": "FEATURE_COMPUTED",
    "correlation_id": "c1",
    "payload": {"features": {"rsi": 65.4}}
}

# Write lineage
writer.handle("events.ml.FEATURE_COMPUTED", envelope)

# Query lineage
correlations = svc.get_correlations_by_id("c1")
print(f"Found {len(correlations)} correlation entries")
```

### Pattern 5: Retry/DLQ Consumer

**File**: `ml/consumers/retry.py` (docstring example)

```python
from ml.common.in_memory_bus import InMemoryPublisher
from ml.consumers.retry import RetriableConsumer, RetryPolicy
from ml.consumers.protocols import Envelope

# Setup DLQ bus
dlq_bus = InMemoryPublisher()

def failing_handler(topic: str, envelope: Envelope) -> None:
    # This handler will fail 3 times then go to DLQ
    raise RuntimeError("Processing failed")

# Create retriable consumer with custom policy
policy = RetryPolicy(max_attempts=5)
rc = RetriableConsumer(
    handler=failing_handler,
    dlq=dlq_bus,
    policy=policy
)

# Handle envelope (will retry 5 times then DLQ)
envelope: Envelope = {
    "id": "e1",
    "parent_id": None,
    "instrument_id": "EURUSD.SIM",
    "ts_event": 100,
    "stage": "FEATURE_COMPUTED",
    "correlation_id": "c1",
    "payload": {}
}
rc.handle("events.ml.FEATURE_COMPUTED", envelope)

# Check DLQ
# Event will be in DLQ topic: "dlq.FEATURE_COMPUTED"
```

### Pattern 6: Streaming Training Consumer

**File**: `ml/tests/unit/consumers/test_streaming_training_consumer.py`

```python
from pathlib import Path
from ml.consumers.streaming_training import (
    StreamingTrainingConsumer,
    FileBackedStreamingTrainingStateStore,
)

# Create file-backed state store
state_path = Path("/tmp/streaming_state.json")
store = FileBackedStreamingTrainingStateStore(state_path)
consumer = StreamingTrainingConsumer(state_store=store)

# Handle plan event
plan_payload = {
    "payload_type": "streaming_plan",
    "plan_id": "plan-123",
    "dataset_id": "dataset",
    "status": "SUCCESS",
    "correlation_id": "c1",
    "payload": {
        "created_at": "2025-01-01T00:00:00Z",
        "caps": {"max_total_rows": 100},
        "limits": {},
        "metadata_summary": {},
        "streaming_config": {}
    }
}
consumer.handle("events.ml.DATASET_PLANNED.dataset", plan_payload)

# Check outstanding plans
outstanding = store.outstanding_plan_ids()
print(f"Outstanding plans: {outstanding}")  # ('plan-123',)

# Handle result event
result_payload = {
    "payload_type": "streaming_result",
    "plan_id": "plan-123",
    "dataset_id": "dataset",
    "status": "SUCCESS",
    "correlation_id": "c2",
    "payload": {
        "completed_at": "2025-01-01T01:00:00Z",
        "model_id": "model-456",
        "metrics": {"val_loss": 0.123},
        "artifact_paths": {},
        "telemetry": {}
    }
}
consumer.handle("events.ml.MODEL_TRAINING_COMPLETED.dataset", result_payload)

# Check outstanding plans (should be empty now)
outstanding = store.outstanding_plan_ids()
print(f"Outstanding plans: {outstanding}")  # ()
```

### Pattern 7: Streaming Persistence Worker (Production)

**File**: `ml/cli/streaming_persistence_worker.py`

```python
from ml.config.bus import MessageBusConfig
from ml.config.streaming_pipeline import StreamingPersistenceConfig
from ml.consumers.streaming_training_worker import StreamingTrainingPersistenceWorker

# Configuration from environment
config = StreamingPersistenceConfig.from_env()
bus_config = MessageBusConfig.from_env()

# Create worker
worker = StreamingTrainingPersistenceWorker(
    config=config,
    message_bus_config=bus_config,
)

# Run forever (blocking)
worker.run_forever()

# Or poll once for testing
processed = worker.poll_once()
print(f"Processed {processed} events")
```

**CLI Usage:**
```bash
# Run with defaults from environment
uv run -m ml.cli.streaming_persistence_worker

# Override state path
uv run -m ml.cli.streaming_persistence_worker \
  --state-path /custom/path/state.json

# Adjust polling parameters
uv run -m ml.cli.streaming_persistence_worker \
  --batch-size 50 \
  --block-ms 5000 \
  --poll-interval 2.0

# Force enable/disable
uv run -m ml.cli.streaming_persistence_worker --enable
uv run -m ml.cli.streaming_persistence_worker --disable
```

### Pattern 8: Multi-Pattern Consumer with Routing

```python
from ml.consumers.redis_streams_consumer import RedisStreamsConsumer
from ml.common.topic_filters import match_topic

class EventRouter:
    def __init__(self):
        self.feature_count = 0
        self.prediction_count = 0
        self.signal_count = 0

    def route_event(self, topic: str, payload: dict[str, Any]) -> None:
        """Route events to appropriate handlers based on topic patterns."""
        if match_topic("events.ml.FEATURE_COMPUTED.*", topic):
            self._handle_feature(topic, payload)
        elif match_topic("events.ml.PREDICTION_EMITTED.*", topic):
            self._handle_prediction(topic, payload)
        elif match_topic("events.ml.SIGNAL_EMITTED.*", topic):
            self._handle_signal(topic, payload)
        else:
            print(f"Unhandled topic: {topic}")

    def _handle_feature(self, topic: str, payload: dict[str, Any]) -> None:
        self.feature_count += 1
        instrument = payload.get("instrument_id", "unknown")
        feature_count = len(payload.get("features", {}))
        print(f"Features: {feature_count} computed for {instrument}")

    def _handle_prediction(self, topic: str, payload: dict[str, Any]) -> None:
        self.prediction_count += 1
        prediction = payload.get("prediction", 0.0)
        confidence = payload.get("confidence", 0.0)
        print(f"Prediction: {prediction} (confidence: {confidence})")

    def _handle_signal(self, topic: str, payload: dict[str, Any]) -> None:
        self.signal_count += 1
        signal_type = payload.get("signal_type", "unknown")
        strength = payload.get("strength", 0.0)
        print(f"Signal: {signal_type} with strength {strength}")

# Setup router-based consumer
router = EventRouter()
consumer = RedisStreamsConsumer(
    url="redis://localhost:6379/0",
    stream="ml-events",
    handler=router.route_event
)

# Process events in loop
while True:
    processed = consumer.poll_once(count=100, block_ms=1000)
    if processed == 0:
        break
    print(f"Processed {processed} events")
    print(f"Counts - Features: {router.feature_count}, "
          f"Predictions: {router.prediction_count}, "
          f"Signals: {router.signal_count}")
```

---

## Integration Points

### Message Bus Integration

**Redis Streams** (`redis_streams_consumer.py`):
- Production event streaming via `RedisStreamsConsumer`
- Reads from Redis streams with `XREAD` command
- Configurable batch sizes and blocking timeouts
- Built-in idempotent gating via `IdempotentConsumer`

**In-Memory Bus** (`ml.common.in_memory_bus`):
- Testing and development via `InMemoryPublisher`
- Synchronous event delivery (no network overhead)
- Pattern-based subscriptions with wildcards

**Topic Routing** (`ml.common.topic_filters`):
- Wildcard pattern matching for flexible subscription
- `*` matches exactly one token
- `#` matches zero or more tokens
- Example: `events.ml.FEATURE_COMPUTED.#` matches all feature computation events

**CLI Integration**:
- `ml/cli/events_consumer.py` — Debug consumer for Redis Streams
- `ml/cli/streaming_persistence_worker.py` — Production worker for streaming training

### Pipeline Event Integration

**Event Structure:**

All consumers expect events with this structure:
```python
{
    "dataset_id": "features|predictions|signals",
    "instrument_id": "EURUSD.SIM",
    "source": "historical|live|backfill",
    "ts_max": 1640995200000000000,  # nanosecond watermark
    "metadata": {
        "correlation_id": "unique-event-id",
        "run_id": "optional-batch-id"
    },
    "data": {
        # Event-specific payload
    }
}
```

**Correlation Tracking:**
- Correlation IDs link events across stores and pipeline stages
- `LineageWriter` persists correlation metadata to `observability.event_correlation` table
- Enables tracing event chains for debugging and lineage analysis

**Watermark Synchronization:**
- Ensures monotonic processing aligned with store watermarks
- Prevents out-of-order processing across pipeline stages
- Critical for maintaining training/inference parity

### Stores and Registries Integration

**Event Correlation** (`lineage_writer.py`):
- Links events across pipeline stages via correlation IDs
- Persists to `ObservabilityService` for tracing
- Enables debugging pipeline failures by correlation ID

**Watermark Synchronization** (`aggregator.py`):
- Aligns consumer watermarks with store watermarks
- Ensures monotonic processing order per instrument
- Prevents data races during backfill and recomputation

**Idempotency** (`idempotent.py`):
- Prevents duplicate processing during store recomputation
- Tracks correlation IDs to filter replayed events
- Critical for backfill safety and recovery

### Observability Integration

**Prometheus Metrics:**

**Aggregator Metrics** (`ml/common/metrics.py:248-267`):
```python
nautilus_ml_aggregator_buffer_size{instrument}        # Current buffer size
nautilus_ml_aggregator_flushed_total{instrument}      # Cumulative flushed count
nautilus_ml_aggregator_duplicates_total               # Duplicate events dropped
nautilus_ml_aggregator_watermark_lag_seconds{instrument}  # Watermark lag
```

**Streaming Training Metrics** (`streaming_training.py:22-41`):
```python
ml_tft_streaming_training_backlog{dataset_id}        # Outstanding plans
ml_tft_streaming_worker_progress_pct{worker_id}      # Worker progress percentage
ml_tft_streaming_worker_rss_mb{worker_id}            # Worker RSS (MB)
ml_tft_streaming_workers_active{dataset_id}          # Active worker count
```

**Grafana Dashboards:**
- Dashboard row: "Consumers / Aggregator"
- Buffer Size: `nautilus_ml_aggregator_buffer_size`
- Flushed Rate: `sum by (instrument)(rate(nautilus_ml_aggregator_flushed_total[$interval]))`
- Duplicates Rate: `rate(nautilus_ml_aggregator_duplicates_total[$interval])`
- Watermark Lag (max): `max(nautilus_ml_aggregator_watermark_lag_seconds)`

**Alerts:**
- `MLAggregatorDuplicatesHigh`: sustained duplicates > 1/s
- `MLAggregatorBufferHigh`: max buffer > 5k for 10m
- `MLAggregatorWatermarkLagHigh`: max lag > 5m for 10m

**ObservabilitySink Protocol** (`streaming_training.py:44-56`):
```python
class ObservabilitySink(Protocol):
    """Minimal protocol for observability collectors."""

    def add_metric(
        self,
        *,
        metric_name: str,
        metric_type: str,
        value: float,
        timestamp: int,
        labels: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
    ) -> None: ...
```

Used by `StreamingTrainingConsumer` to mirror metrics to `ObservabilityService`.

---

## Testing Strategy

### Unit Tests

**Location**: `ml/tests/unit/consumers/`

**Coverage:**
- `test_idempotent_consumer.py` — IdempotentConsumer deduplication and watermark gating
- `test_redis_streams_consumer.py` — RedisStreamsConsumer with mocked Redis client
- `test_streaming_training_consumer.py` — StreamingTrainingConsumer state management
- `test_streaming_training_service.py` — StreamingTrainingPersistenceService factory methods
- `test_streaming_training_worker.py` — StreamingTrainingPersistenceWorker polling and lifecycle

**Testing Patterns:**
- Mock Redis client for RedisStreamsConsumer tests
- In-memory state store for StreamingTrainingConsumer tests
- Monkeypatch for IdempotentConsumer in integration tests

### Property Tests

**Location**: `ml/tests/property/`

**Files:**
- `test_consumer_idempotent_properties.py` — Hypothesis-based property tests
- `test_consumers_aggregator_properties.py` — Aggregator watermark properties

**Properties Tested:**
- Idempotency: same event never processed twice
- Watermark ordering: events flushed in non-decreasing timestamp order
- Buffer consistency: buffer size metric matches actual buffer size

### Contract Tests

**Location**: `ml/tests/contracts/`

**Files:**
- `test_consumer_idempotency_contracts.py` — Idempotency guarantees across consumer types

**Contracts Tested:**
- All consumers implement `ConsumerProtocol`
- Envelope structure matches TypedDict schema
- Correlation ID deduplication across restarts

### Integration Tests

**Location**: `ml/tests/integration/consumers/`

**Files:**
- `test_streaming_persistence_integration.py` — End-to-end Redis → Worker → State persistence
- `test_lineage_writer_integration.py` — LineageWriter → ObservabilityService integration
- `test_ingest_aggregate_lineage_integration.py` — Full pipeline integration

**Integration Scenarios:**
- Redis Streams → Worker → File state persistence
- Aggregator → Downstream publisher → LineageWriter
- Full cascade: Ingest → Aggregator → LineageWriter → ObservabilityService

### Performance Tests

**Location**: `ml/tests/performance/`

**Files:**
- `test_streaming_persistence_microbench.py` — Streaming persistence throughput benchmarks

**Benchmarks:**
- Streaming training consumer throughput (events/sec)
- State store snapshot persistence latency
- Redis consumer poll latency

---

## Configuration

### Environment Variables

**Message Bus Config** (`ml/config/bus.py`):
```bash
ML_BUS_ENABLED=true
ML_BUS_BACKEND=redis
ML_BUS_REDIS_URL=redis://localhost:6379/0
ML_BUS_REDIS_STREAM=ml-events
ML_BUS_TOPIC_PREFIX=events.ml
ML_BUS_SCHEME=domain_op
```

**Streaming Persistence Config** (`ml/config/streaming_pipeline.py`):
```bash
ML_STREAMING_PERSISTENCE_ENABLED=true
ML_STREAMING_PERSISTENCE_STATE_PATH=~/.cache/nautilus/streaming_state.json
ML_STREAMING_PERSISTENCE_BATCH_SIZE=100
ML_STREAMING_PERSISTENCE_BLOCK_MS=5000
ML_STREAMING_PERSISTENCE_POLL_INTERVAL_SECONDS=1.0
```

### Configuration Classes

**StreamingPersistenceConfig** (`ml/config/streaming_pipeline.py`):
```python
@dataclass(frozen=True)
class StreamingPersistenceConfig:
    enabled: bool = True
    state_path: str = "~/.cache/nautilus/streaming_state.json"
    batch_size: int = 100
    block_ms: int = 5000
    poll_interval_seconds: float = 1.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> StreamingPersistenceConfig: ...
```

**MessageBusConfig** (`ml/config/bus.py`):
```python
@dataclass(frozen=True)
class MessageBusConfig:
    enabled: bool = True
    backend: str = "redis"
    redis_url: str = "redis://localhost:6379/0"
    redis_stream: str = "ml-events"
    topic_prefix: str = "events.ml"
    scheme: str = "domain_op"

    @classmethod
    def from_env(cls) -> MessageBusConfig: ...
```

---

## Common Patterns

### Pattern: Idempotent Event Processing

**Problem**: Prevent duplicate processing during backfill, replay, or network failures.

**Solution**: Use `IdempotentConsumer` with correlation ID tracking.

**Implementation**:
```python
from ml.consumers.idempotent import IdempotentConsumer

consumer = IdempotentConsumer()

def handle_event(payload: dict[str, Any]) -> None:
    if consumer.process(payload):
        # Event accepted - safe to process
        process_business_logic(payload)
```

### Pattern: Watermark-Gated Aggregation

**Problem**: Buffer events and flush in timestamp order when watermark advances.

**Solution**: Use `AggregatingConsumer` with per-instrument buffering.

**Implementation**:
```python
from ml.consumers.aggregator import AggregatingConsumer

agg = AggregatingConsumer(downstream=publisher)

# Buffer events
agg.handle(topic, envelope)

# Advance watermark and flush
flushed = agg.advance_watermark("EURUSD.SIM", watermark_ns)
```

### Pattern: Correlation Lineage Tracking

**Problem**: Track event lineage across pipeline stages for debugging.

**Solution**: Use `LineageWriter` to persist correlation metadata.

**Implementation**:
```python
from ml.consumers.lineage_writer import LineageWriter
from ml.observability.service import ObservabilityService

writer = LineageWriter(service=ObservabilityService())
writer.handle(topic, envelope)
```

### Pattern: Retry with DLQ

**Problem**: Retry transient failures and send persistent failures to DLQ.

**Solution**: Use `RetriableConsumer` with bounded retries.

**Implementation**:
```python
from ml.consumers.retry import RetriableConsumer, RetryPolicy

policy = RetryPolicy(max_attempts=5)
consumer = RetriableConsumer(handler, dlq_bus, policy)
consumer.handle(topic, envelope)
```

### Pattern: Streaming Training Monitoring

**Problem**: Monitor streaming training jobs with backlog and worker metrics.

**Solution**: Use `StreamingTrainingConsumer` with `ObservabilitySink`.

**Implementation**:
```python
from ml.consumers.streaming_training import StreamingTrainingConsumer

consumer = StreamingTrainingConsumer(
    state_store=file_store,
    observability=observability_sink
)
consumer.handle(topic, payload)
```

---

## Known Limitations

### 1. In-Memory State Constraints

**IdempotentConsumer** (`idempotent.py`):
- Correlation ID set grows unbounded (no eviction)
- No persistence across process restarts
- Memory usage grows linearly with event count

**Mitigation:**
- Consider periodic state snapshots for recovery
- Implement LRU eviction for correlation IDs
- Use external deduplication store (Redis, PostgreSQL)

**LineageWriter** (`lineage_writer.py`):
- Same in-memory deduplication limitations
- State lost on process restart

**Mitigation:**
- Rely on `ObservabilityService` database constraints for deduplication
- Accept occasional duplicate writes (database enforces uniqueness)

### 2. RedisStreamsConsumer Limitations

**Single Consumer Mode** (`redis_streams_consumer.py`):
- No Redis consumer groups support
- No acknowledgment or offset tracking
- No automatic retry on failure

**Mitigation:**
- Use Redis consumer groups for distributed processing (requires custom implementation)
- Implement acknowledgment tracking for guaranteed delivery
- Add retry logic in handler or use `RetriableConsumer`

**Error Handling** (`redis_streams_consumer.py:78-81`):
- Handler exceptions logged but not retried
- Failed events not tracked for replay
- No dead-letter queue integration

**Mitigation:**
- Wrap handler in `RetriableConsumer` for automatic retries
- Implement custom DLQ publishing in handler
- Use external monitoring to detect failed events

### 3. AggregatingConsumer Memory Growth

**Buffer Growth** (`aggregator.py:56`):
- Unbounded buffer per instrument if watermark never advances
- No buffer size limits or eviction policy
- Out-of-memory risk for slow watermark progression

**Mitigation:**
- Implement buffer size limits with overflow handling
- Add metrics alerting for high buffer sizes
- Periodically flush stale buffers

**Watermark Lag** (`aggregator.py:114-116`):
- No automatic watermark advancement
- Requires external watermark management
- Stale events can accumulate indefinitely

**Mitigation:**
- Implement periodic forced flush for stale buffers
- Monitor watermark lag metric and alert on high values
- Use time-based watermarks as fallback

### 4. StreamingTrainingConsumer State Persistence

**File Locking** (`streaming_training.py:370-376`):
- No file locking during atomic write
- Concurrent processes can corrupt state
- Temp file replace not atomic on all filesystems

**Mitigation:**
- Use file locking (`fcntl.flock`) before write
- Consider database-backed state store (PostgreSQL)
- Run single worker process per state file

**Snapshot Overhead** (`streaming_training.py:372-376`):
- Full snapshot written on every state change
- I/O overhead scales with state size
- Potential performance bottleneck for high-frequency updates

**Mitigation:**
- Batch state updates before persisting
- Use incremental append-only log instead of full snapshot
- Consider database-backed state store for better performance

### 5. Retry Policy Limitations

**Synchronous Retries** (`retry.py:43-51`):
- Blocks consumer loop during retries
- No exponential backoff or jitter
- Not suitable for production (deterministic for testing only)

**Mitigation:**
- Implement async retries with backoff
- Use external retry queue (Redis, SQS)
- Add circuit breaker for cascading failures

**DLQ Publishing** (`retry.py:52-66`):
- Best-effort publishing (no retry on DLQ failure)
- No DLQ overflow protection
- No DLQ consumer for replay

**Mitigation:**
- Add retry logic for DLQ publishing
- Implement DLQ consumer for manual replay
- Monitor DLQ depth and alert on overflow

---

## Performance Considerations

### Cold-Path Design

**All consumers are cold-path only** (not suitable for hot-path inference):
- Designed for background processing, not real-time inference
- Acceptable latency: 100ms-1s per event batch
- P99 latency target: <1s for batch processing

**Hot-Path Alternatives:**
- Use `MLSignalActor` or `MLTradingStrategy` for hot-path inference
- Actors subscribe to Nautilus message bus (not Redis Streams)
- Pre-load models and features at startup

### Throughput Benchmarks

**RedisStreamsConsumer** (`test_streaming_persistence_microbench.py`):
- Throughput: ~1,000-10,000 events/sec (depends on handler complexity)
- Latency: ~1-10ms per event (with idempotent gating)
- Batch size impact: larger batches improve throughput (up to 100-500 events)

**AggregatingConsumer** (`test_consumers_aggregator_properties.py`):
- Buffer overhead: O(1) per instrument
- Flush latency: O(N log N) for N buffered events (sort overhead)
- Watermark advance: <10ms for typical buffer sizes (<1000 events)

**StreamingTrainingConsumer** (`test_streaming_persistence_microbench.py`):
- Throughput: ~100-1,000 events/sec (limited by file I/O)
- State snapshot latency: 10-100ms (depends on state size)
- Recommended batch size: 50-100 events

### Memory Usage

**IdempotentConsumer**:
- Memory growth: ~100 bytes per correlation ID
- Typical usage: 1-10 MB for 10,000-100,000 events
- Unbounded growth without eviction

**AggregatingConsumer**:
- Memory growth: ~500 bytes per buffered event
- Typical usage: 1-50 MB for 10-100k buffered events
- Per-instrument buffers scale independently

**StreamingTrainingConsumer**:
- Memory growth: ~1 KB per plan/result/heartbeat
- Typical usage: 1-10 MB for 1,000-10,000 records
- File-backed store reduces memory footprint

### Optimization Strategies

**Batch Processing**:
- Use larger batch sizes for better throughput (100-500 events)
- Balance batch size vs. latency requirements
- Monitor memory usage for very large batches

**Idempotent Gating**:
- Consider bloom filters for scalable deduplication
- Implement LRU eviction for correlation ID set
- Use external deduplication store (Redis, PostgreSQL)

**Watermark Management**:
- Advance watermarks periodically (not per-event)
- Use time-based watermarks for predictable flushing
- Monitor watermark lag and alert on high values

---

## Production Deployment

### Deployment Checklist

**Redis Configuration**:
- Enable Redis persistence (RDB or AOF)
- Configure stream retention policy (`XTRIM`)
- Set appropriate maxmemory limits
- Enable Redis Sentinel or Cluster for HA

**Worker Configuration**:
- Run worker as systemd service or Docker container
- Configure state path to persistent volume
- Set appropriate batch size and polling interval
- Enable observability sink for metrics

**Monitoring**:
- Configure Prometheus scraping for consumer metrics
- Set up Grafana dashboards for visualization
- Configure alerts for buffer overflows and watermark lag
- Monitor DLQ depth and processing lag

**Error Handling**:
- Configure log aggregation (e.g., ELK, Splunk)
- Set up error alerting (e.g., PagerDuty, Opsgenie)
- Implement runbook for common failure modes
- Test recovery procedures

### Scaling Considerations

**Horizontal Scaling**:
- Use Redis consumer groups for distributed processing
- Partition consumers by instrument or dataset
- Monitor Redis stream lag and add consumers as needed

**Vertical Scaling**:
- Increase batch size for higher throughput
- Adjust polling interval for lower latency
- Allocate more memory for larger buffers

**Backpressure Handling**:
- Monitor buffer sizes and watermark lag
- Implement buffer overflow protection (drop or DLQ)
- Add circuit breakers for downstream failures

---

## Future Enhancements

### Planned Features

**Consumer Groups** (Redis Streams):
- Implement Redis consumer groups for distributed processing
- Add acknowledgment tracking for guaranteed delivery
- Support consumer rebalancing on failure

**External State Stores**:
- PostgreSQL-backed state store for StreamingTrainingConsumer
- Redis-backed deduplication for IdempotentConsumer
- S3/GCS snapshot backups for disaster recovery

**Advanced Retry Policies**:
- Exponential backoff with jitter
- Circuit breaker integration
- Async retries with separate worker pool

**Performance Optimizations**:
- Bloom filters for scalable deduplication
- Incremental state snapshots (append-only log)
- Zero-copy event forwarding (reduce allocations)

### Open Questions

**State Management**:
- Should we use database-backed state stores by default?
- How to handle state migration across schema changes?
- What's the right eviction policy for correlation IDs?

**Retry Semantics**:
- Should retries be synchronous or asynchronous?
- What's the right DLQ replay strategy?
- How to prevent infinite retry loops?

**Observability**:
- Should we track per-event processing latency?
- What metrics are most useful for debugging?
- How to correlate consumer metrics with pipeline metrics?

---

## References

### Internal Documentation

- `ml/docs/architecture/event_driven_streaming_plan.md` — Event-driven streaming training architecture
- `ml/docs/ops/streaming_scaling_experiments.md` — Streaming persistence scaling experiments
- `ml/common/message_topics.py` — Topic naming conventions and builders
- `ml/config/events.py` — Event status and stage enumerations

### Related Modules

- `ml/common/in_memory_bus.py` — In-memory message bus for testing
- `ml/common/topic_filters.py` — Wildcard topic pattern matching
- `ml/observability/service.py` — Observability and lineage tracking
- `ml/config/bus.py` — Message bus configuration
- `ml/config/streaming_pipeline.py` — Streaming pipeline configuration

### External Dependencies

- **redis**: Redis client for Redis Streams integration
- **prometheus_client**: Metrics collection (via `ml.common.metrics_bootstrap`)

### Testing Files

- `ml/tests/unit/consumers/` — Unit tests for all consumer modules
- `ml/tests/property/` — Property tests for idempotency and aggregation
- `ml/tests/contracts/` — Contract tests for consumer protocols
- `ml/tests/integration/consumers/` — Integration tests for Redis and observability
- `ml/tests/performance/` — Performance benchmarks for streaming persistence

---

## Summary

The `ml/consumers/` module provides the foundation for reliable, ordered event processing in the ML pipeline. It implements protocol-first consumer interfaces with idempotent replay, watermark-based ordering, and progressive fallback chains. The module is designed for cold-path processing and integrates seamlessly with Redis Streams, in-memory buses, and observability systems.

**Key Takeaways:**
1. **Protocol-First Design**: All consumers implement `ConsumerProtocol` for structural typing
2. **Idempotent Replay**: Correlation ID tracking prevents duplicate processing
3. **Watermark Ordering**: Monotonic timestamp progression ensures correct event ordering
4. **Cold-Path Only**: Consumers are designed for background processing, not hot-path inference
5. **Production-Ready**: Built-in metrics, error handling, and graceful degradation

**When to Use:**
- Use `IdempotentConsumer` for simple deduplication and watermark gating
- Use `RedisStreamsConsumer` for production event streaming from Redis
- Use `AggregatingConsumer` for buffering and flushing events in timestamp order
- Use `LineageWriter` for correlation tracking and lineage persistence
- Use `StreamingTrainingConsumer` for monitoring streaming training jobs
- Use `RetriableConsumer` for bounded retries with DLQ fallback

**When NOT to Use:**
- Do NOT use consumers for hot-path inference (use actors/strategies instead)
- Do NOT use in-memory state stores for large-scale production (use database-backed stores)
- Do NOT use synchronous retry policies for production (implement async retries)
