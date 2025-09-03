from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera as pa

from ml.observability.persistence import ObservabilityPersistor
from ml.observability.service import ObservabilityService
from ml.tests.contracts.test_observability_pipeline_schemas import (
    EventCorrelationSchema,
    HealthScoreAggregationSchema,
    LatencyWatermarkSchema,
    MetricsCollectionSchema,
)
from nautilus_trader.core.uuid import UUID4


class TestPersistedObservabilityContracts:
    def test_persisted_jsonl_files_conform_to_contracts(self, tmp_path: Path) -> None:
        svc = ObservabilityService()

        # Latency stages (schema expects these canonical stage names)
        svc.add_latency_stage(
            correlation_id=str(UUID4()),
            instrument_id="EURUSD.SIM",
            pipeline_stage="data_ingestion",
            ts_stage_start=1609459200000000000,
            ts_stage_end=1609459200001000000,  # +1ms
        )
        svc.add_latency_stage(
            correlation_id=str(UUID4()),
            instrument_id="BTCUSDT.BINANCE",
            pipeline_stage="feature_computation",
            ts_stage_start=1609459200001000000,
            ts_stage_end=1609459200003000000,  # +2ms
        )

        # Metrics
        svc.add_metric(
            metric_name="ml_model_inference_latency_seconds",
            metric_type="histogram",
            value=0.002,
            timestamp=1609459200002000000,
            labels={"actor_id": "a1", "model_id": "m1"},
        )

        # Correlation: root and child
        root_event = str(UUID4())
        child_event = str(UUID4())
        corr_id = str(UUID4())
        svc.add_correlation(
            correlation_id=corr_id,
            event_id=root_event,
            parent_event_id=None,
            instrument_id="EURUSD.SIM",
            domain="data",
            lineage_depth=0,
            ts_event=1609459200000000000,
            propagation_path=["data"],
        )
        svc.add_correlation(
            correlation_id=corr_id,
            event_id=child_event,
            parent_event_id=root_event,
            instrument_id="EURUSD.SIM",
            domain="features",
            lineage_depth=1,
            ts_event=1609459200001000000,
            propagation_path=["data", "features"],
        )

        # Health
        svc.add_health(
            component_id="feature_store",
            health_score=0.95,
            subsystem_scores={"connection": 1.0, "query_performance": 0.9},
            timestamp=1609459200005000000,
            measurement_window_ms=60000,
        )

        # Persist as JSONL
        tables = {
            "latency": svc.latency_watermarks_df(),
            "metrics": svc.metrics_collection_df(),
            "correlation": svc.event_correlation_df(),
            "health": svc.health_scores_df(),
        }
        sink = ObservabilityPersistor(base_path=tmp_path, file_format="jsonl")
        written = sink.persist(tables)

        # Validate with Pandera contracts by reading JSONL
        def _read_jsonl(path: Path) -> pd.DataFrame:
            # Avoid pandas auto-date conversion on columns named 'timestamp'
            return pd.read_json(path, lines=True, convert_dates=False)

        if "latency" in written:
            LatencyWatermarkSchema.validate(_read_jsonl(written["latency"]))
        if "metrics" in written:
            MetricsCollectionSchema.validate(_read_jsonl(written["metrics"]))
        if "correlation" in written:
            EventCorrelationSchema.validate(_read_jsonl(written["correlation"]))
        if "health" in written:
            HealthScoreAggregationSchema.validate(_read_jsonl(written["health"]))
