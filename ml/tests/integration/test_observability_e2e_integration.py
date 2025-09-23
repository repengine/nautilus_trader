"""
End-to-end observability integration tests exercising store hooks.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Sequence, cast

import pytest

from ml.common.observability_utils import record_stage_boundary
from ml.core.integration import MLIntegrationManager
from ml.stores.base import FeatureData, ModelPrediction, StrategySignal

if TYPE_CHECKING:  # pragma: no cover - hints only
    import pandas as pd


class _FeatureStoreStub:
    """Feature store stub that records observability boundaries."""

    def __init__(self) -> None:
        self._observability_service: Any | None = None

    def write_features(
        self,
        feature_set_id: str | None = None,
        instrument_id: str | None = None,
        features: Any | None = None,
        ts_event: int | None = None,
        ts_init: int | None = None,
        data: Sequence[FeatureData] | None = None,
        *,
        publish_bus: bool = True,
    ) -> None:
        entries = list(data or [])
        if not entries:
            return
        ts_start = time.time_ns()
        ts_end = time.time_ns()
        record_stage_boundary(
            getattr(self, "_observability_service", None),
            component="feature_store",
            instrument_id=entries[0].instrument_id,
            stage="feature_storage",
            ts_stage_start=ts_start,
            ts_stage_end=ts_end,
            row_count=len(entries),
        )


class _ModelStoreStub:
    """Model store stub that records observability boundaries."""

    def __init__(self) -> None:
        self._observability_service: Any | None = None

    def write_batch(
        self,
        data: Sequence[ModelPrediction],
        emit_events: bool = True,
        publish_bus: bool = True,
    ) -> None:
        entries = list(data)
        if not entries:
            return
        ts_start = time.time_ns()
        ts_end = time.time_ns()
        record_stage_boundary(
            getattr(self, "_observability_service", None),
            component="model_store",
            instrument_id=entries[0].instrument_id,
            stage="model_prediction_storage",
            ts_stage_start=ts_start,
            ts_stage_end=ts_end,
            row_count=len(entries),
        )


class _StrategyStoreStub:
    """Strategy store stub that records observability boundaries."""

    def __init__(self) -> None:
        self._observability_service: Any | None = None

    def write_batch(
        self,
        data: Sequence[StrategySignal],
        emit_events: bool = True,
        publish_bus: bool = True,
    ) -> None:
        entries = list(data)
        if not entries:
            return
        ts_start = time.time_ns()
        ts_end = time.time_ns()
        record_stage_boundary(
            getattr(self, "_observability_service", None),
            component="strategy_store",
            instrument_id=entries[0].instrument_id,
            stage="strategy_signal_storage",
            ts_stage_start=ts_start,
            ts_stage_end=ts_end,
            row_count=len(entries),
        )


@pytest.fixture
def integration_manager_with_observability(monkeypatch: pytest.MonkeyPatch) -> MLIntegrationManager:
    monkeypatch.setenv("ML_OBSERVABILITY_ENABLED", "1")
    monkeypatch.setenv("ML_ALLOW_DUMMY", "1")

    mgr = MLIntegrationManager(
        auto_start_postgres=False,
        auto_migrate=False,
        ensure_healthy=False,
    )

    mgr.feature_store = _FeatureStoreStub()
    mgr.model_store = _ModelStoreStub()
    mgr.strategy_store = _StrategyStoreStub()
    mgr.data_store = None

    mgr.initialize_observability_pipeline()
    mgr._inject_observability_service_into_stores()

    return mgr


class TestObservabilityE2EIntegration:
    """Exercise observability helpers end-to-end via the integration manager."""

    def test_feature_store_observability_hook(self, integration_manager_with_observability: MLIntegrationManager) -> None:
        mgr = integration_manager_with_observability
        feature_store = cast(_FeatureStoreStub, mgr.feature_store)
        assert getattr(feature_store, "_observability_service") is mgr.observability_service

        feature_data = FeatureData(
            feature_set_id="test_features",
            instrument_id="EUR/USD.SIM",
            _ts_event=time.time_ns(),
            _ts_init=time.time_ns(),
            values={"rsi_14": 65.0},
        )
        feature_store.write_features(data=[feature_data])

        latency_rows = cast(list[dict[str, Any]], getattr(mgr.observability_service, "_latency_rows"))
        assert any(row.get("pipeline_stage") == "feature_storage" for row in latency_rows)

    def test_model_store_observability_hook(self, integration_manager_with_observability: MLIntegrationManager) -> None:
        mgr = integration_manager_with_observability
        model_store = cast(_ModelStoreStub, mgr.model_store)
        assert getattr(model_store, "_observability_service") is mgr.observability_service

        prediction = ModelPrediction(
            model_id="test_model_v1",
            instrument_id="EUR/USD.SIM",
            _ts_event=time.time_ns(),
            _ts_init=time.time_ns(),
            prediction=0.75,
            confidence=0.85,
            features_used={"rsi_14": 65.0},
            inference_time_ms=2.5,
        )
        model_store.write_batch([prediction])

        latency_rows = cast(list[dict[str, Any]], getattr(mgr.observability_service, "_latency_rows"))
        assert any(row.get("pipeline_stage") == "model_prediction_storage" for row in latency_rows)

    def test_strategy_store_observability_hook(self, integration_manager_with_observability: MLIntegrationManager) -> None:
        mgr = integration_manager_with_observability
        strategy_store = cast(_StrategyStoreStub, mgr.strategy_store)
        assert getattr(strategy_store, "_observability_service") is mgr.observability_service

        signal = StrategySignal(
            strategy_id="test_strategy",
            instrument_id="EUR/USD.SIM",
            _ts_event=time.time_ns(),
            _ts_init=time.time_ns(),
            signal_type="BUY",
            strength=0.8,
            model_predictions={"test_model": 0.75},
            risk_metrics={"var": 0.02},
            execution_params={"stop_loss": 0.02},
        )
        strategy_store.write_batch([signal])

        latency_rows = cast(list[dict[str, Any]], getattr(mgr.observability_service, "_latency_rows"))
        assert any(row.get("pipeline_stage") == "strategy_signal_storage" for row in latency_rows)
