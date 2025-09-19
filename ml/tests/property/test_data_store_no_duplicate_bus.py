from __future__ import annotations

from typing import Any, cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ml.common.message_bus import MessagePublisherProtocol
from ml.stores.base import FeatureData, ModelPrediction, StrategySignal
from ml.stores.data_store import DataStore


class _CapturePublisher(MessagePublisherProtocol):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        self.calls.append((topic, payload))
        return True


class _StubRegistry:
    def __init__(self) -> None:
        self._datasets: set[str] = set()

    def emit_event(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        stage: Any,
        source: Any,
        run_id: str,
        ts_min: int,
        ts_max: int,
        count: int,
        status: Any,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        return None

    def update_watermark(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        source: Any,
        last_success_ns: int,
        count: int,
        completeness_pct: float,
    ) -> None:
        return None

    # Minimal DataRegistry API for DataStore auto-registration
    def get_manifest(self, dataset_id: str) -> Any:  # noqa: D401
        if dataset_id not in self._datasets:
            raise ValueError("not found")
        return {"dataset_id": dataset_id}

    def register_dataset(self, manifest: Any) -> None:  # noqa: D401
        self._datasets.add(getattr(manifest, "dataset_id", str(manifest)))


class _StubFeatureStore:
    """Stub with typed write_features to capture publish_bus flag."""

    def __init__(self) -> None:
        self.publish_flags: list[bool] = []

    def write_features(
        self,
        *,
        feature_set_id: str,
        instrument_id: str,
        features: dict[str, float],
        ts_event: int,
        ts_init: int,
        publish_bus: bool = True,
    ) -> None:
        self.publish_flags.append(bool(publish_bus))


class _StubModelStore:
    def write_batch(self, data: list[ModelPrediction], emit_events: bool = True, publish_bus: bool = True) -> None:  # noqa: D401
        return None


class _StubStrategyStore:
    def write_batch(self, data: list[StrategySignal], emit_events: bool = True, publish_bus: bool = True) -> None:  # noqa: D401
        return None


@given(n=st.integers(min_value=1, max_value=5))
@pytest.mark.property
def test_no_duplicate_publish_for_features(n: int) -> None:
    feature_store = _StubFeatureStore()
    model_store = _StubModelStore()
    strategy_store = _StubStrategyStore()
    pub = _CapturePublisher()

    store = cast(
        Any,
        DataStore(
            connection_string="sqlite:///:memory:",
            registry=_StubRegistry(),
            feature_store=cast(Any, feature_store),
            model_store=cast(Any, model_store),
            strategy_store=cast(Any, strategy_store),
            publisher=pub,
            enable_publishing=True,
        ),
    )

    # Build feature records
    items = [
        FeatureData(
            feature_set_id="core",
            instrument_id="EURUSD.SIM",
            values={"f": float(i)},
            ts_event=1000 + i,
            ts_init=1000 + i,
        )
        for i in range(n)
    ]

    store.write_features(instrument_id="EURUSD.SIM", features=items, source="computed", run_id="run_f")

    # Underlying FeatureStore must be suppressed
    assert all(flag is False for flag in feature_store.publish_flags)

    # Exactly one bus publish from DataStore
    assert len(pub.calls) == 1


@given(n=st.integers(min_value=1, max_value=5))
@pytest.mark.property
def test_no_duplicate_publish_for_predictions(n: int) -> None:
    feature_store = _StubFeatureStore()
    model_store = _StubModelStore()
    strategy_store = _StubStrategyStore()
    pub = _CapturePublisher()

    store = cast(
        Any,
        DataStore(
            connection_string="sqlite:///:memory:",
            registry=_StubRegistry(),
            feature_store=cast(Any, feature_store),
            model_store=cast(Any, model_store),
            strategy_store=cast(Any, strategy_store),
            publisher=pub,
            enable_publishing=True,
        ),
    )

    preds = [
        ModelPrediction(
            model_id="m",
            instrument_id="EURUSD.SIM",
            prediction=0.1 * i,
            confidence=0.5,
            features_used={},
            inference_time_ms=0.1,
            _ts_event=1000 + i,
            _ts_init=1000 + i,
        )
        for i in range(n)
    ]
    store.write_predictions(predictions=preds, source="inference", run_id="run_p")
    assert len(pub.calls) == 1


@given(n=st.integers(min_value=1, max_value=5))
@pytest.mark.property
def test_no_duplicate_publish_for_signals(n: int) -> None:
    feature_store = _StubFeatureStore()
    model_store = _StubModelStore()
    strategy_store = _StubStrategyStore()
    pub = _CapturePublisher()

    store = cast(
        Any,
        DataStore(
            connection_string="sqlite:///:memory:",
            registry=_StubRegistry(),
            feature_store=cast(Any, feature_store),
            model_store=cast(Any, model_store),
            strategy_store=cast(Any, strategy_store),
            publisher=pub,
            enable_publishing=True,
        ),
    )

    sigs = [
        StrategySignal(
            strategy_id="s",
            instrument_id="EURUSD.SIM",
            signal_type="BUY",
            strength=0.2,
            model_predictions={},
            risk_metrics={},
            execution_params={},
            _ts_event=1000 + i,
            _ts_init=1000 + i,
        )
        for i in range(n)
    ]
    store.write_signals(signals=sigs, source="strategy", run_id="run_s")
    assert len(pub.calls) == 1
