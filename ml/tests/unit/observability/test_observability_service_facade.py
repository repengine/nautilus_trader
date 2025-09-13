from __future__ import annotations

from typing import Any

import numpy as np

from ml.core.integration import MLIntegrationManager
from ml.observability.service import ObservabilityService


class TestObservabilityServiceFacade:
    def test_service_builds_contract_dataframes(self, default_instrument_id) -> None:
        svc = ObservabilityService()

        # Populate latency rows
        svc.add_latency_stage(
            correlation_id="corr-1",
            instrument_id=str(default_instrument_id),
            pipeline_stage="data_ingested",
            ts_stage_start=1,
            ts_stage_end=6,
        )
        svc.add_latency_stage(
            correlation_id="corr-1",
            instrument_id=str(default_instrument_id),
            pipeline_stage="feature_computed",
            ts_stage_start=6,
            ts_stage_end=10,
        )

        # Metrics row
        svc.add_metric(
            metric_name="ml_inference_latency_seconds",
            metric_type="histogram",
            value=0.002,
            timestamp=10,
            labels={"actor_id": "a1", "model_id": "m1"},
        )

        # Correlation row
        svc.add_correlation(
            correlation_id="corr-1",
            event_id="e1",
            parent_event_id=None,
            instrument_id=str(default_instrument_id),
            domain="data",
            lineage_depth=0,
            ts_event=1,
            propagation_path=["data", "features", "models", "strategies"],
        )

        # Health row
        svc.add_health(
            component_id="data_store",
            health_score=0.98,
            subsystem_scores={"db": 1.0, "registry": 0.96},
            timestamp=10,
            measurement_window_ms=1000,
        )

        df_latency = svc.latency_watermarks_df()
        df_metrics = svc.metrics_collection_df()
        df_corr = svc.event_correlation_df()
        df_health = svc.health_scores_df()

        # Basic contract checks
        assert set(["correlation_id", "pipeline_stage", "stage_latency_ns"]).issubset(
            set(df_latency.columns),
        )
        assert set(["metric_name", "metric_type", "labels"]).issubset(set(df_metrics.columns))
        assert set(["correlation_id", "event_id", "lineage_depth"]).issubset(set(df_corr.columns))
        assert set(["component_id", "health_score"]).issubset(set(df_health.columns))

        # Invariants from DTO builders
        assert int(df_latency.loc[0, "stage_latency_ns"]) >= 0
        assert np.all((df_health["health_score"] >= 0.0) & (df_health["health_score"] <= 1.0))


class TestIntegrationObservability:
    def test_integration_manager_initializes_and_collects(self, default_instrument_id) -> None:
        # Avoid heavy __init__ by constructing via __new__ like other tests
        mgr = object.__new__(MLIntegrationManager)  # type: ignore[misc]

        # Initialize service via the integration method
        MLIntegrationManager.initialize_observability_pipeline(mgr)
        svc = getattr(mgr, "observability_service", None)
        assert svc is not None

        # Add minimal rows through the service
        svc.add_latency_stage(
            correlation_id="c2",
            instrument_id=str(default_instrument_id),
            pipeline_stage="prediction_emitted",
            ts_stage_start=10,
            ts_stage_end=15,
        )
        svc.add_health(
            component_id="model_store",
            health_score=0.9,
            subsystem_scores={"db": 0.9},
            timestamp=20,
            measurement_window_ms=500,
        )

        # Collect materialized DataFrames
        tables = MLIntegrationManager.collect_observability_dataframes(mgr)
        assert set(tables.keys()) == {"latency", "metrics", "correlation", "health"}
        # Verify latency table has our entry
        lat = tables["latency"]
        assert getattr(lat, "empty", True) is False
