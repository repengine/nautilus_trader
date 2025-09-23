from __future__ import annotations

import logging
from typing import Any

import pytest

from ml.common.protocols import MLComponentProtocol
from ml.core.integration import MLIntegrationManager
from ml.tests.utils.stubs import build_integration_manager_stub


class GoodComp(MLComponentProtocol):
    def get_health_status(self) -> dict[str, Any]:
        return {"status": "ok"}

    def get_performance_metrics(self) -> dict[str, float]:
        return {"metric": 1.0}

    def validate_configuration(self) -> list[str]:
        return []


class BadComp:  # Does not implement MLComponentProtocol
    pass


class TestIntegrationProtocolValidation:
    def _make_manager_with_components(self) -> MLIntegrationManager:
        mgr = build_integration_manager_stub()
        mgr.feature_store = GoodComp()
        mgr.model_store = BadComp()
        mgr.strategy_store = None
        mgr.data_store = GoodComp()
        mgr.feature_registry = GoodComp()
        mgr.model_registry = GoodComp()
        mgr.strategy_registry = BadComp()
        mgr.data_registry = None
        return mgr

    def test_validation_warns_when_non_strict(self, caplog: pytest.LogCaptureFixture) -> None:
        mgr = self._make_manager_with_components()
        with caplog.at_level(logging.WARNING):
            mgr._validate_protocol_compliance(strict=False)
        assert any("Protocol compliance issues" in rec.message for rec in caplog.records)

    def test_validation_raises_when_strict(self) -> None:
        mgr = self._make_manager_with_components()
        with pytest.raises(RuntimeError):
            mgr._validate_protocol_compliance(strict=True)
