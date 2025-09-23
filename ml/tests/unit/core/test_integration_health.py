from __future__ import annotations

from ml.common.protocols import MLComponentMixin
from ml.core.integration import MLIntegrationManager
from ml.tests.utils.stubs import build_integration_manager_stub


class _StubComp(MLComponentMixin):
    def __init__(self, healthy: bool = True) -> None:
        self._ok = healthy

    def get_health_status(self) -> dict[str, object]:  # type: ignore[override]
        return {"status": "healthy" if self._ok else "unhealthy"}


def test_aggregate_health_summaries() -> None:
    mgr = build_integration_manager_stub()

    # Provide stub components
    mgr.feature_store = _StubComp(True)
    mgr.model_store = _StubComp(True)
    mgr.strategy_store = _StubComp(True)
    mgr.data_store = _StubComp(True)
    mgr.feature_registry = _StubComp(True)
    mgr.model_registry = _StubComp(True)
    mgr.strategy_registry = _StubComp(True)
    mgr.data_registry = _StubComp(True)

    summary = MLIntegrationManager.aggregate_health(mgr)

    assert isinstance(summary, dict)
    assert summary["system"]["healthy"] is True  # type: ignore[index]
    assert summary["domains"]["data"]["healthy"] is True  # type: ignore[index]
    assert summary["domains"]["features"]["healthy"] is True  # type: ignore[index]
    assert summary["domains"]["model"]["healthy"] is True  # type: ignore[index]
    assert summary["domains"]["strategy"]["healthy"] is True  # type: ignore[index]
