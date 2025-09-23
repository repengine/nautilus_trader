from __future__ import annotations

from pathlib import Path

from ml.config.observability import ObservabilityConfig
from ml.core.integration import MLIntegrationManager
from ml.tests.utils.stubs import build_integration_manager_stub


def test_start_observability_from_config_db_sink(tmp_path: Path) -> None:
    mgr = build_integration_manager_stub()
    MLIntegrationManager.initialize_observability_pipeline(mgr)
    cfg = ObservabilityConfig(
        sink="db",
        base_path=str(tmp_path),
        db_connection_string=f"sqlite:///{tmp_path}/obs.db",
        interval_seconds=0.01,
    )
    MLIntegrationManager.start_observability_from_config(mgr, cfg)
    assert getattr(mgr, "_obs_flusher", None) is not None
    # Stop background thread
    MLIntegrationManager.stop_observability_flush(mgr)
