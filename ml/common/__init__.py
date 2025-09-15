"""
Common utilities for ML components.

This module exposes shared utilities, protocols, and patterns used across the ML domain.
It enforces Universal ML Architecture Pattern 5 (Centralized Metrics Bootstrap) by
exposing metrics utilities as the primary public API.

The module provides:
- Pattern 5: Centralized metrics bootstrap utilities (MANDATORY)
- Pattern 2: Common protocols and mixins for structural typing
- Shared utilities for timestamps, message topics, and safe math
- Event handling and correlation utilities
- Message bus abstractions for pub/sub patterns

Usage:
    # Pattern 5: Always use centralized metrics (MANDATORY)
    from ml.common import get_counter, get_histogram, get_gauge, MetricsManager

    # Pattern 2: Protocol-first interface design
    from ml.common import MLComponentProtocol, MLComponentMixin

    # Shared utilities
    from ml.common import (
        build_topic, sanitize_timestamp_ns, safe_divide,
        emit_dataset_event, make_correlation_id
    )
"""

# ============================================================================
# PATTERN 5: CENTRALIZED METRICS BOOTSTRAP (MANDATORY)
# ============================================================================
# NEVER import prometheus_client directly - use these instead

from ml.common.cascade import EventDict
from ml.common.cascade import emit_cascade

# Correlation and cascading utilities
from ml.common.correlation import make_correlation_id
from ml.common.event_emitter import emit_dataset_event
from ml.common.event_emitter import emit_dataset_event_and_watermark

# Event handling utilities
from ml.common.events_util import SourceStr
from ml.common.events_util import build_bus_payload
from ml.common.events_util import to_source_enum
from ml.common.events_util import to_source_str
from ml.common.in_memory_bus import Handler
from ml.common.in_memory_bus import InMemoryPublisher
from ml.common.message_bus import BusPublisherMixin
from ml.common.message_bus import MessagePublisherProtocol
from ml.common.message_bus import NoopPublisher
from ml.common.message_bus import RedisStreamsPublisher
from ml.common.message_bus import publisher_from_config
from ml.common.message_topics import build_stage_topic

# ============================================================================
# SHARED UTILITIES - CROSS-DOMAIN
# ============================================================================
# Message bus and topic utilities
from ml.common.message_topics import build_topic
from ml.common.message_topics import build_topic_for_stage
from ml.common.message_topics import map_stage_to_topic_segments
from ml.common.metrics_bootstrap import HAS_METRICS_BACKEND
from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_gauge
from ml.common.metrics_bootstrap import get_histogram

# Convenience metrics export utilities
from ml.common.metrics_export import CONTENT_TYPE_LATEST
from ml.common.metrics_export import generate_latest
from ml.common.metrics_manager import MetricsManager
from ml.common.observability_utils import is_observability_enabled
from ml.common.observability_utils import record_stage_boundary
from ml.common.precision import MAX_PRICE_DECIMALS
from ml.common.precision import clamp_price_str

# ============================================================================
# PATTERN 2: PROTOCOL-FIRST INTERFACE DESIGN
# ============================================================================
# Universal component protocol and mixin for structural typing
from ml.common.protocols import MLComponentMixin
from ml.common.protocols import MLComponentProtocol

# Safe math operations
from ml.common.safe_math import safe_divide
from ml.common.safe_math import safe_divide_expr

# Security utilities
from ml.common.security import ArtifactIntegrityError
from ml.common.security import calculate_file_sha256
from ml.common.security import secure_onnx_load
from ml.common.security import verify_artifact_integrity

# Rate limiting utilities
from ml.common.throttler import Throttler

# Timestamp and precision utilities
from ml.common.timestamps import normalize_timestamp_ns
from ml.common.timestamps import sanitize_timestamp_ns

# Topic filtering for pub/sub
from ml.common.topic_filters import match_topic

# Trace context utilities for event consumers
from ml.common.trace_context import extract_and_link_from_event
from ml.common.trace_context import get_correlation_and_trace_context


# ============================================================================
# PUBLIC API SURFACE
# ============================================================================

__all__ = [
    "CONTENT_TYPE_LATEST",
    "HAS_METRICS_BACKEND",
    "MAX_PRICE_DECIMALS",
    "ArtifactIntegrityError",
    "BusPublisherMixin",
    "EventDict",
    "Handler",
    "InMemoryPublisher",
    "MLComponentMixin",
    "MLComponentProtocol",
    "MessagePublisherProtocol",
    "MetricsManager",
    "NoopPublisher",
    "RedisStreamsPublisher",
    "SourceStr",
    "Throttler",
    "build_bus_payload",
    "build_stage_topic",
    "build_topic",
    "build_topic_for_stage",
    "calculate_file_sha256",
    "clamp_price_str",
    "emit_cascade",
    "emit_dataset_event",
    "emit_dataset_event_and_watermark",
    "extract_and_link_from_event",
    "generate_latest",
    "get_correlation_and_trace_context",
    "get_counter",
    "get_gauge",
    "get_histogram",
    "is_observability_enabled",
    "make_correlation_id",
    "map_stage_to_topic_segments",
    "match_topic",
    "normalize_timestamp_ns",
    "publisher_from_config",
    "record_stage_boundary",
    "safe_divide",
    "safe_divide_expr",
    "sanitize_timestamp_ns",
    "secure_onnx_load",
    "to_source_enum",
    "to_source_str",
    "verify_artifact_integrity",
]
