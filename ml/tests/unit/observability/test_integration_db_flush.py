from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.core.integration import MLIntegrationManager


class TestIntegrationDBFlush:
    def test_flush_observability_to_db(self, tmp_path: Path) -> None:
        mgr = object.__new__(MLIntegrationManager)  # type: ignore[misc]
        MLIntegrationManager.initialize_observability_pipeline(mgr)
        svc = mgr.observability_service  # type: ignore[attr-defined]
        assert svc is not None

        # Add minimal rows
        svc.add_latency_stage(
            correlation_id="c1",
            instrument_id="EURUSD.SIM",
            pipeline_stage="data_ingestion",
            ts_stage_start=1,
            ts_stage_end=2,
        )
        svc.add_metric(
            metric_name="ml_predictions_total",
            metric_type="counter",
            value=1.0,
            timestamp=1,
            labels={"instrument_id": "EURUSD.SIM"},
        )
        svc.add_correlation(
            correlation_id="c1",
            event_id="e1",
            parent_event_id=None,
            instrument_id="EURUSD.SIM",
            domain="data",
            lineage_depth=0,
            ts_event=1,
            propagation_path=["data"],
        )
        svc.add_health(
            component_id="data_store",
            health_score=0.9,
            subsystem_scores={"db": 1.0},
            timestamp=2,
            measurement_window_ms=100,
        )

        db = tmp_path / "obs.db"
        out = MLIntegrationManager.flush_observability_to_db(mgr, connection_string=f"sqlite:///{db}")
        assert out.get("latency", 0) >= 1
        assert out.get("metrics", 0) >= 1
        assert out.get("correlation", 0) >= 1
        assert out.get("health", 0) >= 1

        # Spot check data persisted
        import sqlalchemy as sa

        eng = sa.create_engine(f"sqlite:///{db}")
        with eng.connect() as conn:
            lat_df = pd.read_sql("select * from obs_latency_watermarks", conn)
            assert not lat_df.empty

