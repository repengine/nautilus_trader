from __future__ import annotations

from pathlib import Path

from ml.core.integration import MLIntegrationManager
from ml.observability.scheduler import ObservabilityFlusher
from ml.observability.service import ObservabilityService


class TestObservabilityFlusher:
    def test_tick_flushes_when_interval_elapsed(self, tmp_path: Path, default_instrument_id) -> None:
        svc = ObservabilityService()
        # Add a row so flush writes something
        svc.add_latency_stage(
            correlation_id="c1",
            instrument_id=str(default_instrument_id),
            pipeline_stage="data_ingested",
            ts_stage_start=1,
            ts_stage_end=2,
        )

        # Fake time provider
        times = [0.0, 10.0]

        def now() -> float:
            return times[0]

        flusher = ObservabilityFlusher(
            service=svc,
            base_path=tmp_path,
            file_format="jsonl",
            interval_seconds=5.0,
            now=now,
        )

        # Not yet due
        wrote = flusher.tick()
        assert wrote == {}

        # Advance time
        times[0] = 10.0
        wrote = flusher.tick()
        assert "latency" in wrote
        assert wrote["latency"].exists()


class TestIntegrationFlusher:
    def test_integration_manager_single_flush(self, tmp_path: Path) -> None:
        mgr = object.__new__(MLIntegrationManager)  # type: ignore[misc]
        MLIntegrationManager.initialize_observability_pipeline(mgr)
        svc = mgr.observability_service  # type: ignore[attr-defined]
        assert svc is not None
        svc.add_health(
            component_id="strategy_store",
            health_score=0.95,
            subsystem_scores={"db": 0.95},
            timestamp=5,
            measurement_window_ms=100,
        )

        # Start with single flush (no background thread)
        out = MLIntegrationManager.start_observability_flush(
            mgr,
            base_path=tmp_path,
            interval_seconds=0.0,
            file_format="jsonl",
        )
        assert out is not None
        assert out.get("health") is not None
        assert out["health"].exists()  # type: ignore[index]
