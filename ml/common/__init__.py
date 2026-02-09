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

import importlib
from typing import TYPE_CHECKING

# ============================================================================
# PATTERN 5: CENTRALIZED METRICS BOOTSTRAP (MANDATORY)
# ============================================================================
# NEVER import prometheus_client directly - use these instead
from ml.common.cascade import EventDict
from ml.common.cascade import emit_cascade

# Causality guard utilities
from ml.common.causality_guard import CausalityAction
from ml.common.causality_guard import CausalityGuard
from ml.common.causality_guard import CausalityGuardResult
from ml.common.causality_guard import CausalityViolation

# Correlation and cascading utilities
from ml.common.correlation import make_correlation_id
from ml.common.databento_credentials import CredentialResolution
from ml.common.databento_credentials import CredentialSource
from ml.common.databento_credentials import resolve_databento_api_key
from ml.common.db_connections import ConnectionCandidates
from ml.common.db_connections import ConnectionRole
from ml.common.db_connections import collect_postgres_candidates
from ml.common.db_connections import select_first_working_connection
from ml.common.db_utils import get_default_pool_config
from ml.common.db_utils import get_or_create_engine
from ml.common.error_handlers import db_operation_handler
from ml.common.error_handlers import registry_operation_handler
from ml.common.error_handlers import with_db_error_handling
from ml.common.error_handlers import with_fallback
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
from ml.common.output_semantics import OutputSemanticsValidationResult
from ml.common.output_semantics import OutputSemanticsValidator
from ml.common.output_semantics import validate_output_semantics
from ml.common.precision import MAX_PRICE_DECIMALS
from ml.common.precision import clamp_price_str

# ============================================================================
# PATTERN 2: PROTOCOL-FIRST INTERFACE DESIGN
# ============================================================================
# Universal component protocol and mixin for structural typing
from ml.common.protocols import MLComponentMixin
from ml.common.protocols import MLComponentProtocol
from ml.common.reproducibility import DeterministicSeedResult
from ml.common.reproducibility import ReproducibilityHelper
from ml.common.reproducibility import apply_reproducibility_seed
from ml.common.reproducibility import build_configured_reproducibility_provenance
from ml.common.reproducibility import build_reproducibility_provenance
from ml.common.reproducibility import validate_reproducibility_provenance

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


if TYPE_CHECKING:
    from ml.common.decision_metadata import decision_metadata_from_model_metadata
    from ml.common.decision_metadata import normalize_decision_metadata
    from ml.common.decision_metadata import resolve_decision_horizon_ms
    from ml.common.prediction_surface import decision_from_probability
    from ml.common.prediction_surface import neutral_band_bounds
    from ml.common.prediction_surface import normalize_prediction_batch
    from ml.common.prediction_surface import normalize_prediction_output
    from ml.common.prediction_surface import resolve_output_is_logits
    from ml.common.prediction_surface import resolve_positive_class_index
    from ml.common.resource_monitor import current_rss_mb
    from ml.common.symbol_utils import resolve_symbol_data_dir
    from ml.common.symbol_utils import resolve_symbol_data_dir_candidates
    from ml.common.symbol_utils import resolve_symbol_data_dir_exact
    from ml.common.symbol_utils import select_latest_symbol_file
    from ml.common.watermark_window import WatermarkRegistryProtocol
    from ml.common.watermark_window import WatermarkWindowResult
    from ml.common.watermark_window import resolve_watermark_start_date
    from ml.common.watermark_window import resolve_watermark_start_datetime


_LAZY_COMMON_EXPORTS: dict[str, tuple[str, str]] = {
    "decision_metadata_from_model_metadata": (
        "ml.common.decision_metadata",
        "decision_metadata_from_model_metadata",
    ),
    "normalize_decision_metadata": ("ml.common.decision_metadata", "normalize_decision_metadata"),
    "resolve_decision_horizon_ms": ("ml.common.decision_metadata", "resolve_decision_horizon_ms"),
    "decision_from_probability": ("ml.common.prediction_surface", "decision_from_probability"),
    "neutral_band_bounds": ("ml.common.prediction_surface", "neutral_band_bounds"),
    "normalize_prediction_batch": ("ml.common.prediction_surface", "normalize_prediction_batch"),
    "normalize_prediction_output": ("ml.common.prediction_surface", "normalize_prediction_output"),
    "resolve_output_is_logits": ("ml.common.prediction_surface", "resolve_output_is_logits"),
    "resolve_positive_class_index": ("ml.common.prediction_surface", "resolve_positive_class_index"),
    "current_rss_mb": ("ml.common.resource_monitor", "current_rss_mb"),
    "resolve_symbol_data_dir": ("ml.common.symbol_utils", "resolve_symbol_data_dir"),
    "resolve_symbol_data_dir_candidates": ("ml.common.symbol_utils", "resolve_symbol_data_dir_candidates"),
    "resolve_symbol_data_dir_exact": ("ml.common.symbol_utils", "resolve_symbol_data_dir_exact"),
    "select_latest_symbol_file": ("ml.common.symbol_utils", "select_latest_symbol_file"),
    "WatermarkRegistryProtocol": ("ml.common.watermark_window", "WatermarkRegistryProtocol"),
    "WatermarkWindowResult": ("ml.common.watermark_window", "WatermarkWindowResult"),
    "resolve_watermark_start_date": ("ml.common.watermark_window", "resolve_watermark_start_date"),
    "resolve_watermark_start_datetime": ("ml.common.watermark_window", "resolve_watermark_start_datetime"),
}


def __getattr__(name: str) -> object:
    target = _LAZY_COMMON_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


# ============================================================================
# PUBLIC API SURFACE
# ============================================================================

__all__ = [
    "CONTENT_TYPE_LATEST",
    "HAS_METRICS_BACKEND",
    "MAX_PRICE_DECIMALS",
    "ArtifactIntegrityError",
    "BusPublisherMixin",
    "CausalityAction",
    "CausalityGuard",
    "CausalityGuardResult",
    "CausalityViolation",
    "ConnectionCandidates",
    "ConnectionRole",
    "CredentialResolution",
    "CredentialSource",
    "DeterministicSeedResult",
    "EventDict",
    "Handler",
    "InMemoryPublisher",
    "MLComponentMixin",
    "MLComponentProtocol",
    "MessagePublisherProtocol",
    "MetricsManager",
    "NoopPublisher",
    "OutputSemanticsValidationResult",
    "OutputSemanticsValidator",
    "RedisStreamsPublisher",
    "ReproducibilityHelper",
    "SourceStr",
    "Throttler",
    "WatermarkRegistryProtocol",
    "WatermarkWindowResult",
    "apply_reproducibility_seed",
    "build_bus_payload",
    "build_configured_reproducibility_provenance",
    "build_reproducibility_provenance",
    "build_stage_topic",
    "build_topic",
    "build_topic_for_stage",
    "calculate_file_sha256",
    "clamp_price_str",
    "collect_postgres_candidates",
    "current_rss_mb",
    "db_operation_handler",
    "decision_from_probability",
    "decision_metadata_from_model_metadata",
    "emit_cascade",
    "emit_dataset_event",
    "emit_dataset_event_and_watermark",
    "extract_and_link_from_event",
    "generate_latest",
    "get_correlation_and_trace_context",
    "get_counter",
    "get_default_pool_config",
    "get_gauge",
    "get_histogram",
    "get_or_create_engine",
    "is_observability_enabled",
    "make_correlation_id",
    "map_stage_to_topic_segments",
    "match_topic",
    "neutral_band_bounds",
    "normalize_decision_metadata",
    "normalize_prediction_batch",
    "normalize_prediction_output",
    "normalize_timestamp_ns",
    "publisher_from_config",
    "record_stage_boundary",
    "registry_operation_handler",
    "resolve_databento_api_key",
    "resolve_decision_horizon_ms",
    "resolve_output_is_logits",
    "resolve_positive_class_index",
    "resolve_symbol_data_dir",
    "resolve_symbol_data_dir_candidates",
    "resolve_symbol_data_dir_exact",
    "resolve_watermark_start_date",
    "resolve_watermark_start_datetime",
    "safe_divide",
    "safe_divide_expr",
    "sanitize_timestamp_ns",
    "secure_onnx_load",
    "select_first_working_connection",
    "select_latest_symbol_file",
    "to_source_enum",
    "to_source_str",
    "validate_output_semantics",
    "validate_reproducibility_provenance",
    "verify_artifact_integrity",
    "with_db_error_handling",
    "with_fallback",
]
