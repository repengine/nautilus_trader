from __future__ import annotations

import json
from pathlib import Path

from ml.core.integration import MLIntegrationManager
from ml.observability.persistence import ObservabilityPersistor
from ml.observability.service import ObservabilityService


class TestObservabilityPersistorJsonl:
    def test_persist_writes_non_empty_tables_as_jsonl(self, tmp_path: Path) -> None:
        svc = ObservabilityService()

        # Add at least one row to each category
        svc.add_latency_stage(
            correlation_id="c1",
            instrument_id="EURUSD.SIM",
            pipeline_stage="data_ingested",
            ts_stage_start=1,
            ts_stage_end=3,
        )
        svc.add_metric(
            metric_name="ml_predictions_total",
            metric_type="counter",
            value=1.0,
            timestamp=2,
            labels={"actor_id": "a1"},
        )
        svc.add_correlation(
            correlation_id="c1",
            event_id="e1",
            parent_event_id=None,
            instrument_id="EURUSD.SIM",
            domain="data",
            lineage_depth=0,
            ts_event=1,
            propagation_path=["data", "features"],
        )
        svc.add_health(
            component_id="data_store",
            health_score=0.99,
            subsystem_scores={"db": 1.0},
            timestamp=3,
            measurement_window_ms=1000,
        )

        tables = {
            "latency": svc.latency_watermarks_df(),
            "metrics": svc.metrics_collection_df(),
            "correlation": svc.event_correlation_df(),
            "health": svc.health_scores_df(),
        }

        sink = ObservabilityPersistor(base_path=tmp_path, file_format="jsonl")
        written = sink.persist(tables)

        # All four tables should be written
        assert set(written.keys()) == {"latency", "metrics", "correlation", "health"}
        for name, path in written.items():
            assert path.exists()
            # JSONL has one JSON object per line
            lines = path.read_text().strip().splitlines()
            assert len(lines) == len(tables[name])  # type: ignore[index]
            # Verify first line parses
            json.loads(lines[0])


class TestIntegrationFlush:
    def test_integration_manager_flushes_observability_to_path(self, tmp_path: Path) -> None:
        mgr = object.__new__(MLIntegrationManager)  # type: ignore[misc]

        # Initialize service and add a couple of rows
        MLIntegrationManager.initialize_observability_pipeline(mgr)
        svc = mgr.observability_service  # type: ignore[attr-defined]
        assert svc is not None

        svc.add_latency_stage(
            correlation_id="c2",
            instrument_id="EURUSD.SIM",
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

        # Act: flush to disk as JSONL
        out = MLIntegrationManager.flush_observability_to_path(
            mgr,
            base_path=tmp_path,
            file_format="jsonl",
        )

        # Assert files exist for non-empty tables
        assert out["latency"].exists()  # type: ignore[index]
        assert out["health"].exists()  # type: ignore[index]
        # metrics and correlation may be empty in this scenario and may be absent
