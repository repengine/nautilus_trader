"""
Observability module for ML pipeline monitoring and telemetry.

This module provides comprehensive observability capabilities for ML operations, including
latency tracking, metrics collection, event correlation, and health monitoring. All components
are designed for cold-path operations and should never be used in hot-path/real-time code.

The observability system supports multiple persistence backends (file, SQL database) and
provides both synchronous and asynchronous processing capabilities for different deployment
scenarios.

Key Components
--------------
- **ObservabilityService**: Main facade for collecting observability data
- **ObservabilityFlusher**: Background scheduler for periodic data persistence
- **ObservabilityAsyncWorker**: Async worker with bounded queue for high-throughput scenarios
- **Persistence layers**: File-based and database persistence with retention management
- **Pipeline builders**: DTO builders for contract-compliant DataFrames
- **Correlation analysis**: Network analysis tools for event relationships

Architecture Patterns
---------------------
This module strictly follows the Universal ML Architecture Patterns:

1. **Cold Path Only**: All operations are off-hot-path with no real-time constraints
2. **Progressive Fallback**: File fallback when database unavailable, dummy modes for testing
3. **Centralized Metrics**: Uses ml.common.metrics_bootstrap for all metric collection
4. **Protocol-First**: Clean interfaces with structural typing support
5. **Configuration-Driven**: Environment-based configuration via ObservabilityConfig

Usage Patterns
--------------
For background processing::

    from ml.observability import ObservabilityService, ObservabilityFlusher

    service = ObservabilityService()
    service.add_latency_stage(
        correlation_id="req_123",
        instrument_id="EUR/USD.SIM",
        pipeline_stage="feature_computation",
        ts_stage_start=start_ns,
        ts_stage_end=end_ns
    )

    flusher = ObservabilityFlusher(
        service=service,
        base_path=Path("./observability"),
        sink="file"
    )
    flusher.flush_once()

For high-throughput async scenarios::

    from ml.observability import ObservabilityAsyncWorker

    worker = ObservabilityAsyncWorker(
        service=service,
        sink="db",
        db_connection_string="postgresql://...",
        queue_maxsize=8192
    )
    worker.start()

    # From hot path - non-blocking enqueue
    success = worker.enqueue_latency(
        correlation_id="req_456",
        instrument_id="EUR/USD.SIM",
        pipeline_stage="model_inference",
        ts_stage_start=start_ns,
        ts_stage_end=end_ns
    )

For database management::

    from ml.observability import apply_observability_indices, ObservabilityDBPersistor

    # Apply performance optimizations
    apply_observability_indices(engine)

    # Persist with retention
    persistor = ObservabilityDBPersistor(connection_string)
    persistor.apply_retention(retention_days=30)

Environment Configuration
------------------------
The observability system can be configured via environment variables:

- ML_OBS_SINK: "file" or "db"
- ML_OBS_BASE_PATH: Base directory for file outputs
- ML_OBS_DB_URL: Database connection string
- ML_OBS_INTERVAL_SECONDS: Flush interval
- ML_OBS_ASYNC_ENABLE: Enable async worker mode

Performance Characteristics
--------------------------
- **Enqueue operations**: O(1) with bounded memory usage
- **Queue backpressure**: Drops items when full, emits metrics for monitoring
- **Persistence**: Batched writes with configurable intervals
- **Database optimizations**: BRIN indices, monthly partitioning for PostgreSQL
- **Memory efficiency**: Streaming processing, no unbounded growth

Integration Points
-----------------
- **MetricsManager**: For internal metric emission via bootstrap
- **EngineManager**: For database connection pooling and management
- **MLIntegrationManager**: For system-wide observability setup
- **ObservabilityConfig**: For environment-based configuration

Thread Safety
-------------
- ObservabilityService: Not thread-safe, use separate instances per thread
- ObservabilityAsyncWorker: Thread-safe enqueue operations
- Persistence layers: Thread-safe when using separate instances
- Configuration: Immutable, thread-safe after creation

Notes
-----
- All timestamps are in nanoseconds since epoch for consistency with Nautilus core
- DataFrame schemas are validated via pipeline builders for contract compliance
- Database schemas support both SQLite (development) and PostgreSQL (production)
- File outputs support JSONL and CSV formats with optional daily rotation
- Correlation analysis supports both directed and undirected graph operations

"""

from __future__ import annotations

from ml.observability.async_db_persistence import ObservabilityAsyncDBPersistor

# Async processing components
from ml.observability.async_worker import ObservabilityAsyncWorker

# Bootstrap and configuration helpers
from ml.observability.bootstrap import auto_start_if_configured
from ml.observability.correlation import connected_components

# Correlation analysis utilities
from ml.observability.correlation import prune_edges
from ml.observability.db_persistence import ObservabilityDBPersistor

# Database management and migrations
from ml.observability.migrations import apply_observability_indices
from ml.observability.migrations import apply_observability_monthly_partitions

# Persistence layers
from ml.observability.persistence import ObservabilityPersistor
from ml.observability.pipeline import aggregate_metrics_by_window
from ml.observability.pipeline import build_event_correlation
from ml.observability.pipeline import build_health_scores

# Pipeline builders for DataFrame construction
from ml.observability.pipeline import build_latency_watermarks
from ml.observability.pipeline import build_metrics_collection
from ml.observability.pipeline import scale_health_scores
from ml.observability.scheduler import ObservabilityFlusher

# Core observability service and scheduling
from ml.observability.service import ObservabilityService

# Distributed tracing (optional, cold-path only)
from ml.observability.tracing import extract_and_link_trace_context
from ml.observability.tracing import get_trace_context
from ml.observability.tracing import inject_trace_context
from ml.observability.tracing import is_tracing_enabled
from ml.observability.tracing import trace_cold_path
from ml.observability.tracing import trace_cold_path_decorator
from ml.observability.tracing import trace_inference


__all__ = [
    "ObservabilityAsyncDBPersistor",
    "ObservabilityAsyncWorker",
    "ObservabilityDBPersistor",
    "ObservabilityFlusher",
    "ObservabilityPersistor",
    "ObservabilityService",
    "aggregate_metrics_by_window",
    "apply_observability_indices",
    "apply_observability_monthly_partitions",
    "auto_start_if_configured",
    "build_event_correlation",
    "build_health_scores",
    "build_latency_watermarks",
    "build_metrics_collection",
    "connected_components",
    "extract_and_link_trace_context",
    "get_trace_context",
    "inject_trace_context",
    "is_tracing_enabled",
    "prune_edges",
    "scale_health_scores",
    "trace_cold_path",
    "trace_cold_path_decorator",
    "trace_inference",
]
