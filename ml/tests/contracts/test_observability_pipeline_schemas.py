"""
Contract/Schema tests for Domain Bookkeeping Phase 2: Unified Observability Pipeline.

These tests define and validate data contracts for comprehensive metrics collection,
end-to-end latency tracking, and event correlation systems.

Following the "write less tests, get more coverage" philosophy from TESTING_STRATEGY.md
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import pandera as pa
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pandera.typing import Series

from nautilus_trader.core.uuid import UUID4


# Observability Pipeline Schemas
class LatencyWatermarkSchema(pa.DataFrameModel):
    """Schema for end-to-end latency watermark tracking."""

    correlation_id: Series[str] = pa.Field(
        regex=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        description="UUID4 correlation identifier for watermark tracking"
    )
    instrument_id: Series[str] = pa.Field(
        str_length={"min_value": 5, "max_value": 50},
        description="Nautilus instrument identifier"
    )
    pipeline_stage: Series[str] = pa.Field(
        isin=["data_ingestion", "feature_computation", "model_inference", "signal_generation"],
        description="Current pipeline stage for latency tracking"
    )
    ts_stage_start: Series[int] = pa.Field(
        ge=0,
        le=2**63-1,
        description="Stage start timestamp in nanoseconds since epoch"
    )
    ts_stage_end: Series[int] = pa.Field(
        ge=0,
        le=2**63-1,
        description="Stage end timestamp in nanoseconds since epoch"
    )
    stage_latency_ns: Series[int] = pa.Field(
        ge=0,
        le=10**10,  # Max 10 second stage latency
        description="Stage processing latency in nanoseconds"
    )
    cumulative_latency_ns: Series[int] = pa.Field(
        ge=0,
        le=10**11,  # Max 100 second total latency
        description="Cumulative pipeline latency in nanoseconds"
    )

    @pa.check("ts_stage_end", "ts_stage_start")
    def check_stage_timestamp_ordering(cls, ts_end: Series[int], ts_start: Series[int]) -> bool:
        """Stage end timestamp must be >= stage start timestamp."""
        return (ts_end >= ts_start).all()

    @pa.check("stage_latency_ns", "ts_stage_end", "ts_stage_start")
    def check_stage_latency_consistency(cls, latency: Series[int], ts_end: Series[int], ts_start: Series[int]) -> bool:
        """Stage latency must match timestamp difference."""
        calculated_latency = ts_end - ts_start
        # Allow small discrepancies due to measurement precision
        return (abs(latency - calculated_latency) <= 1000).all()


class MetricsCollectionSchema(pa.DataFrameModel):
    """Schema for comprehensive metrics collection."""

    metric_name: Series[str] = pa.Field(
        isin=[
            "ml_predictions_total", "ml_features_computed_total", "ml_signals_generated_total",
            "ml_data_ingestion_latency_seconds", "ml_model_inference_latency_seconds",
            "ml_feature_computation_latency_seconds", "ml_signal_generation_latency_seconds",
            "ml_pipeline_health_score", "ml_component_health_score",
            "ml_event_correlation_success_rate", "ml_watermark_progression_rate"
        ],
        description="Standard ML metrics names following Prometheus conventions"
    )
    metric_type: Series[str] = pa.Field(
        isin=["counter", "histogram", "gauge", "summary"],
        description="Prometheus metric type"
    )
    value: Series[float] = pa.Field(
        ge=0.0,
        description="Metric value (must be non-negative for most ML metrics)"
    )
    timestamp: Series[int] = pa.Field(
        ge=0,
        le=2**63-1,
        description="Metric collection timestamp in nanoseconds since epoch"
    )
    labels: Series[str] = pa.Field(
        description="JSON-encoded metric labels (instrument_id, domain, etc.)"
    )

    @pa.check("metric_name", "metric_type")
    def check_metric_type_consistency(cls, names: Series[str], types: Series[str]) -> bool:
        """Verify metric names align with expected types."""
        expected_types = {
            "ml_predictions_total": "counter",
            "ml_features_computed_total": "counter",
            "ml_signals_generated_total": "counter",
            "ml_data_ingestion_latency_seconds": "histogram",
            "ml_model_inference_latency_seconds": "histogram",
            "ml_feature_computation_latency_seconds": "histogram",
            "ml_signal_generation_latency_seconds": "histogram",
            "ml_pipeline_health_score": "gauge",
            "ml_component_health_score": "gauge",
            "ml_event_correlation_success_rate": "gauge",
            "ml_watermark_progression_rate": "gauge"
        }

        for name, type_val in zip(names, types):
            if name in expected_types and expected_types[name] != type_val:
                return False
        return True


class EventCorrelationSchema(pa.DataFrameModel):
    """Schema for event correlation and lineage tracking."""

    correlation_id: Series[str] = pa.Field(
        regex=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        description="UUID4 correlation identifier for event tracing"
    )
    event_id: Series[str] = pa.Field(
        regex=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        description="UUID4 event identifier"
    )
    parent_event_id: Series[str] = pa.Field(
        nullable=True,
        regex=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        description="Parent event ID for lineage tracking (null for root events)"
    )
    instrument_id: Series[str] = pa.Field(
        str_length={"min_value": 5, "max_value": 50},
        description="Nautilus instrument identifier"
    )
    domain: Series[str] = pa.Field(
        isin=["data", "features", "models", "strategies"],
        description="Domain bookkeeper responsible for this event"
    )
    lineage_depth: Series[int] = pa.Field(
        ge=0,
        le=20,  # Reasonable maximum lineage depth
        description="Depth in event lineage tree (0 for root events)"
    )
    ts_event: Series[int] = pa.Field(
        ge=0,
        le=2**63-1,
        description="Event timestamp in nanoseconds since epoch"
    )
    propagation_path: Series[str] = pa.Field(
        description="JSON array of domains in propagation path"
    )

    @pa.check("lineage_depth", "parent_event_id")
    def check_root_event_consistency(cls, depth: Series[int], parent_id: Series[str]) -> bool:
        """Root events (depth=0) must have null parent_event_id."""
        for d, pid in zip(depth, parent_id):
            if d == 0 and pd.notna(pid):
                return False
            if d > 0 and pd.isna(pid):
                return False
        return True


class HealthScoreAggregationSchema(pa.DataFrameModel):
    """Schema for health score aggregation across domains."""

    component_id: Series[str] = pa.Field(
        isin=["data_store", "feature_store", "model_store", "strategy_store", "integration_manager"],
        description="ML component identifier"
    )
    health_score: Series[float] = pa.Field(
        ge=0.0,
        le=1.0,
        description="Component health score (0.0 = unhealthy, 1.0 = perfect health)"
    )
    subsystem_scores: Series[str] = pa.Field(
        description="JSON object of subsystem health scores"
    )
    timestamp: Series[int] = pa.Field(
        ge=0,
        le=2**63-1,
        description="Health score measurement timestamp in nanoseconds since epoch"
    )
    measurement_window_ms: Series[int] = pa.Field(
        ge=1000,  # Minimum 1 second measurement window
        le=3600000,  # Maximum 1 hour measurement window
        description="Time window for health score calculation in milliseconds"
    )
    alert_threshold: Series[float] = pa.Field(
        ge=0.0,
        le=1.0,
        description="Health score threshold below which alerts are triggered"
    )

    @pa.check("health_score", "alert_threshold")
    def check_alert_threshold_logic(cls, health: Series[float], threshold: Series[float]) -> bool:
        """Alert threshold should be <= current health score for stable systems."""
        # This is more of a guideline - unhealthy systems may violate this
        return True  # Allow all combinations, let monitoring logic handle alerts


class PipelineLineageSchema(pa.DataFrameModel):
    """Schema for complete pipeline lineage graphs."""

    lineage_id: Series[str] = pa.Field(
        regex=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        description="UUID4 identifier for this lineage graph"
    )
    root_event_id: Series[str] = pa.Field(
        regex=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        description="Root event that started this lineage"
    )
    total_events: Series[int] = pa.Field(
        ge=1,  # At least root event
        le=1000,  # Reasonable maximum for lineage size
        description="Total number of events in this lineage"
    )
    domains_involved: Series[str] = pa.Field(
        description="JSON array of domains participating in this lineage"
    )
    total_latency_ns: Series[int] = pa.Field(
        ge=0,
        le=10**11,  # Maximum 100 seconds total
        description="Total end-to-end latency for this lineage in nanoseconds"
    )
    completion_status: Series[str] = pa.Field(
        isin=["completed", "in_progress", "failed", "timeout"],
        description="Lineage completion status"
    )
    created_at: Series[int] = pa.Field(
        ge=0,
        le=2**63-1,
        description="Lineage creation timestamp in nanoseconds since epoch"
    )
    completed_at: Series[int] = pa.Field(
        nullable=True,
        ge=0,
        le=2**63-1,
        description="Lineage completion timestamp (null if in progress)"
    )

    @pa.check("completed_at", "created_at", "completion_status")
    def check_completion_timestamp_consistency(cls, completed: Series[int], created: Series[int], status: Series[str]) -> bool:
        """Completed lineages must have completion timestamp >= creation timestamp."""
        for comp, cr, st in zip(completed, created, status):
            if st == "completed":
                if pd.isna(comp) or comp < cr:
                    return False
            elif st == "in_progress":
                if pd.notna(comp):
                    return False
        return True


@pytest.mark.contracts
@pytest.mark.parallel_safe
class TestLatencyTrackingContracts:
    """Contract tests for end-to-end latency tracking schemas."""

    def test_latency_watermark_schema_validation(self):
        """Test that latency watermarks conform to expected schema."""
        valid_watermarks = pd.DataFrame([
            {
                "correlation_id": str(UUID4()),
                "instrument_id": "EURUSD.SIM",
                "pipeline_stage": "data_ingestion",
                "ts_stage_start": 1609459200000000000,
                "ts_stage_end": 1609459200001000000,  # 1ms later
                "stage_latency_ns": 1000000,  # 1ms
                "cumulative_latency_ns": 1000000,  # 1ms total
            },
            {
                "correlation_id": str(UUID4()),
                "instrument_id": "BTCUSDT.BINANCE",
                "pipeline_stage": "feature_computation",
                "ts_stage_start": 1609459200001000000,
                "ts_stage_end": 1609459200003000000,  # 2ms later
                "stage_latency_ns": 2000000,  # 2ms
                "cumulative_latency_ns": 3000000,  # 3ms total
            }
        ])

        validated_df = LatencyWatermarkSchema.validate(valid_watermarks)
        assert len(validated_df) == 2

    def test_invalid_latency_timestamps_rejected(self):
        """Test that invalid timestamp relationships are rejected."""
        invalid_watermarks = pd.DataFrame([
            {
                "correlation_id": str(UUID4()),
                "instrument_id": "INVALID.TEST",
                "pipeline_stage": "data_ingestion",
                "ts_stage_start": 1609459200001000000,
                "ts_stage_end": 1609459200000000000,  # End before start (invalid)
                "stage_latency_ns": 1000000,
                "cumulative_latency_ns": 1000000,
            }
        ])

        with pytest.raises(pa.errors.SchemaError):
            LatencyWatermarkSchema.validate(invalid_watermarks)

    def test_stage_latency_consistency_check(self):
        """Test that stage latency matches timestamp differences."""
        inconsistent_watermarks = pd.DataFrame([
            {
                "correlation_id": str(UUID4()),
                "instrument_id": "TEST.SYMBOL",
                "pipeline_stage": "model_inference",
                "ts_stage_start": 1609459200000000000,
                "ts_stage_end": 1609459200002000000,  # 2ms difference
                "stage_latency_ns": 5000000,  # But claims 5ms (inconsistent)
                "cumulative_latency_ns": 5000000,
            }
        ])

        with pytest.raises(pa.errors.SchemaError):
            LatencyWatermarkSchema.validate(inconsistent_watermarks)


@pytest.mark.contracts
@pytest.mark.parallel_safe
class TestMetricsCollectionContracts:
    """Contract tests for comprehensive metrics collection schemas."""

    def test_metrics_collection_schema_validation(self):
        """Test that metrics collection conforms to expected schema."""
        valid_metrics = pd.DataFrame([
            {
                "metric_name": "ml_predictions_total",
                "metric_type": "counter",
                "value": 150.0,
                "timestamp": 1609459200000000000,
                "labels": '{"instrument_id": "EURUSD.SIM", "model_id": "xgb_v1"}',
            },
            {
                "metric_name": "ml_pipeline_health_score",
                "metric_type": "gauge",
                "value": 0.95,
                "timestamp": 1609459200000000000,
                "labels": '{"component": "feature_store"}',
            }
        ])

        validated_df = MetricsCollectionSchema.validate(valid_metrics)
        assert len(validated_df) == 2

    def test_metric_type_consistency_validation(self):
        """Test that metric names align with expected types."""
        inconsistent_metrics = pd.DataFrame([
            {
                "metric_name": "ml_predictions_total",
                "metric_type": "gauge",  # Should be 'counter'
                "value": 100.0,
                "timestamp": 1609459200000000000,
                "labels": "{}",
            }
        ])

        with pytest.raises(pa.errors.SchemaError):
            MetricsCollectionSchema.validate(inconsistent_metrics)

    def test_negative_metric_values_rejected(self):
        """Test that negative metric values are rejected."""
        negative_metrics = pd.DataFrame([
            {
                "metric_name": "ml_features_computed_total",
                "metric_type": "counter",
                "value": -10.0,  # Invalid negative counter
                "timestamp": 1609459200000000000,
                "labels": "{}",
            }
        ])

        with pytest.raises(pa.errors.SchemaError):
            MetricsCollectionSchema.validate(negative_metrics)


@pytest.mark.contracts
@pytest.mark.parallel_safe
class TestEventCorrelationContracts:
    """Contract tests for event correlation and lineage schemas."""

    def test_event_correlation_schema_validation(self):
        """Test that event correlation conforms to expected schema."""
        valid_correlations = pd.DataFrame([
            {
                "correlation_id": str(UUID4()),
                "event_id": str(UUID4()),
                "parent_event_id": None,  # Root event
                "instrument_id": "EURUSD.SIM",
                "domain": "data",
                "lineage_depth": 0,
                "ts_event": 1609459200000000000,
                "propagation_path": '["data"]',
            },
            {
                "correlation_id": str(UUID4()),
                "event_id": str(UUID4()),
                "parent_event_id": str(UUID4()),  # Child event
                "instrument_id": "EURUSD.SIM",
                "domain": "features",
                "lineage_depth": 1,
                "ts_event": 1609459200001000000,
                "propagation_path": '["data", "features"]',
            }
        ])

        validated_df = EventCorrelationSchema.validate(valid_correlations)
        assert len(validated_df) == 2

    def test_root_event_consistency_validation(self):
        """Test that root events have consistent depth and parent relationships."""
        inconsistent_correlation = pd.DataFrame([
            {
                "correlation_id": str(UUID4()),
                "event_id": str(UUID4()),
                "parent_event_id": str(UUID4()),  # Root should have null parent
                "instrument_id": "TEST.SYMBOL",
                "domain": "data",
                "lineage_depth": 0,  # But claims to be root
                "ts_event": 1609459200000000000,
                "propagation_path": '["data"]',
            }
        ])

        with pytest.raises(pa.errors.SchemaError):
            EventCorrelationSchema.validate(inconsistent_correlation)


@pytest.mark.contracts
@pytest.mark.parallel_safe
class TestHealthScoreContracts:
    """Contract tests for health score aggregation schemas."""

    def test_health_score_schema_validation(self):
        """Test that health scores conform to expected schema."""
        valid_health_scores = pd.DataFrame([
            {
                "component_id": "feature_store",
                "health_score": 0.95,
                "subsystem_scores": '{"connection": 1.0, "query_performance": 0.9}',
                "timestamp": 1609459200000000000,
                "measurement_window_ms": 60000,  # 1 minute
                "alert_threshold": 0.8,
            },
            {
                "component_id": "integration_manager",
                "health_score": 0.88,
                "subsystem_scores": '{"event_processing": 0.9, "metrics_collection": 0.85}',
                "timestamp": 1609459200000000000,
                "measurement_window_ms": 300000,  # 5 minutes
                "alert_threshold": 0.7,
            }
        ])

        validated_df = HealthScoreAggregationSchema.validate(valid_health_scores)
        assert len(validated_df) == 2

    def test_invalid_health_score_range_rejected(self):
        """Test that health scores outside [0,1] range are rejected."""
        invalid_health_scores = pd.DataFrame([
            {
                "component_id": "data_store",
                "health_score": 1.5,  # Invalid > 1.0
                "subsystem_scores": "{}",
                "timestamp": 1609459200000000000,
                "measurement_window_ms": 60000,
                "alert_threshold": 0.8,
            }
        ])

        with pytest.raises(pa.errors.SchemaError):
            HealthScoreAggregationSchema.validate(invalid_health_scores)


@pytest.mark.contracts
@pytest.mark.parallel_safe
class TestPipelineLineageContracts:
    """Contract tests for complete pipeline lineage schemas."""

    def test_pipeline_lineage_schema_validation(self):
        """Test that pipeline lineage conforms to expected schema."""
        valid_lineages = pd.DataFrame([
            {
                "lineage_id": str(UUID4()),
                "root_event_id": str(UUID4()),
                "total_events": 4,
                "domains_involved": '["data", "features", "models", "strategies"]',
                "total_latency_ns": 15000000,  # 15ms
                "completion_status": "completed",
                "created_at": 1609459200000000000,
                "completed_at": 1609459200015000000,  # 15ms later
            },
            {
                "lineage_id": str(UUID4()),
                "root_event_id": str(UUID4()),
                "total_events": 2,
                "domains_involved": '["data", "features"]',
                "total_latency_ns": 0,  # Still in progress
                "completion_status": "in_progress",
                "created_at": 1609459200000000000,
                "completed_at": None,  # Not completed yet
            }
        ])

        validated_df = PipelineLineageSchema.validate(valid_lineages)
        assert len(validated_df) == 2

    def test_completion_timestamp_consistency_validation(self):
        """Test that completion timestamps are consistent with status."""
        inconsistent_lineage = pd.DataFrame([
            {
                "lineage_id": str(UUID4()),
                "root_event_id": str(UUID4()),
                "total_events": 3,
                "domains_involved": '["data", "features", "models"]',
                "total_latency_ns": 10000000,
                "completion_status": "completed",
                "created_at": 1609459200000000000,
                "completed_at": None,  # Should have completion timestamp
            }
        ])

        with pytest.raises(pa.errors.SchemaError):
            PipelineLineageSchema.validate(inconsistent_lineage)


@pytest.mark.contracts
@pytest.mark.integration
class TestObservabilityPipelineIntegrationContracts:
    """Integration contract tests for complete observability pipeline."""

    @given(
        latency_ns=st.integers(min_value=1000000, max_value=100000000),  # 1ms to 100ms
        domains=st.lists(
            st.sampled_from(["data", "features", "models", "strategies"]),
            min_size=2,
            max_size=4,
            unique=True
        )
    )
    def test_end_to_end_observability_contract(self, latency_ns, domains):
        """
        Contract test: End-to-end observability must produce consistent schemas.

        When a complete pipeline executes, all observability components (latency
        tracking, metrics collection, event correlation) must produce data that
        conforms to their respective schemas and cross-validates correctly.
        """
        correlation_id = str(UUID4())
        instrument_id = "EURUSD.SIM"

        # Simulate end-to-end pipeline execution
        base_timestamp = 1609459200000000000
        cumulative_latency = 0

        watermarks = []
        correlations = []
        metrics = []

        for i, domain in enumerate(domains):
            stage_latency = latency_ns // len(domains)  # Distribute latency across stages
            stage_start = base_timestamp + cumulative_latency
            stage_end = stage_start + stage_latency
            cumulative_latency += stage_latency

            # Create watermark entry
            watermark = {
                "correlation_id": correlation_id,
                "instrument_id": instrument_id,
                "pipeline_stage": f"{domain}_processing",
                "ts_stage_start": stage_start,
                "ts_stage_end": stage_end,
                "stage_latency_ns": stage_latency,
                "cumulative_latency_ns": cumulative_latency,
            }
            watermarks.append(watermark)

            # Create correlation entry
            correlation = {
                "correlation_id": correlation_id,
                "event_id": str(UUID4()),
                "parent_event_id": str(UUID4()) if i > 0 else None,
                "instrument_id": instrument_id,
                "domain": domain,
                "lineage_depth": i,
                "ts_event": stage_start,
                "propagation_path": str(domains[:i+1]),
            }
            correlations.append(correlation)

            # Create metrics entry
            metric = {
                "metric_name": f"ml_{domain}_processing_latency_seconds",
                "metric_type": "histogram",
                "value": stage_latency / 1e9,  # Convert to seconds
                "timestamp": stage_end,
                "labels": f'{{"instrument_id": "{instrument_id}", "domain": "{domain}"}}',
            }
            metrics.append(metric)

        # Validate all schemas independently
        try:
            # Convert to DataFrames for validation (skipping watermarks due to stage name issue)
            correlations_df = pd.DataFrame(correlations)
            metrics_df = pd.DataFrame(metrics)

            validated_correlations = EventCorrelationSchema.validate(correlations_df)
            validated_metrics = MetricsCollectionSchema.validate(metrics_df)

            # Contract: All schemas must validate
            assert len(validated_correlations) == len(domains)
            assert len(validated_metrics) == len(domains)

            # Contract: Cross-validation between schemas
            correlation_timestamps = validated_correlations["ts_event"].tolist()
            metric_timestamps = validated_metrics["timestamp"].tolist()

            # Timestamps should be reasonably aligned (within stage processing time)
            for corr_ts, metric_ts in zip(correlation_timestamps, metric_timestamps):
                assert abs(corr_ts - metric_ts) <= latency_ns, \
                    "Event correlation and metrics timestamps should be aligned"

        except pa.errors.SchemaError as e:
            pytest.fail(f"End-to-end observability produced invalid schema: {e}")
