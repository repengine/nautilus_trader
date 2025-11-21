"""
Unit tests for ``SimpleControlPanel`` resilience helpers.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml.dashboard.control_simple import SimpleControlPanel

pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry", "mock_tracing_backend")


class _SlowIntegration:
    def __init__(self, *, delay: float, store: object | None = None) -> None:
        time.sleep(delay)
        self.data_store = store
        self.model_store = None
        self.feature_store = None
        self.strategy_store = None


def test_simple_control_panel_integration_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Panel should fall back when integration bootstrap exceeds timeout.
    """

    def _slow_manager(**_: object) -> _SlowIntegration:
        return _SlowIntegration(delay=0.05)

    monkeypatch.setattr("ml.dashboard.control_simple.MLIntegrationManager", _slow_manager)

    panel = SimpleControlPanel(integration_timeout=0.01)

    status = panel.get_system_status()

    assert panel._integration is None  # type: ignore[attr-defined]
    assert status["stores"]["data"] == {"healthy": False, "fallback": True}


def test_simple_control_panel_health_timeout() -> None:
    """
    Slow store health probes should degrade to fallback state.
    """

    class _SlowStore:
        def health_check(self) -> dict[str, object]:
            time.sleep(0.05)
            return {"healthy": True, "fallback": False}

    panel = SimpleControlPanel(integration_timeout=0.0, health_timeout=0.01)
    panel._integration = SimpleNamespace(  # type: ignore[attr-defined]
        data_store=_SlowStore(),
        model_store=None,
        feature_store=None,
        strategy_store=None,
    )

    stores = panel.get_system_status()["stores"]

    assert stores["data"] == {"healthy": False, "fallback": True}
    assert stores["model"] == {"healthy": False, "fallback": True}


def test_simple_control_panel_pipeline_job_tracking(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    panel = SimpleControlPanel(state_path=state_dir / "panel.json")
    result = panel.trigger_pipeline("dataset", {"param": 1}, job_id="job123", status="queued")

    assert result["job_id"] == "job123"
    status = panel.get_system_status()
    runs = status["pipelines"]["runs"]
    assert "job123" in runs
    assert runs["job123"]["job_id"] == "job123"
    assert runs["job123"]["status"] == "queued"
