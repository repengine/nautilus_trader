from __future__ import annotations

from collections.abc import Generator
from typing import Any, cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ml.common.message_bus import MessagePublisherProtocol
from ml.stores.base import FeatureData, ModelPrediction, StrategySignal
from ml.stores.data_store_facade import DataStore
from ml.config.events import EventStatus, Source, Stage
from ml.registry.dataclasses import DataContract, DatasetManifest
from ml.registry.dataclasses import QualityFlag
from ml.registry.dataclasses import ValidationRule
from ml.registry.dataclasses import ValidationRuleType


class _CapturePublisher(MessagePublisherProtocol):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        self.calls.append((topic, payload))
        return True


class _StubRegistry:
    def __init__(self) -> None:
        self._datasets: set[str] = set()
        self._manifests: dict[str, DatasetManifest] = {}
        self._contracts: dict[str, DataContract] = {}
        self.events: list[dict[str, Any]] = []
        self.watermarks: list[dict[str, Any]] = []

    def emit_event(
        self,
        dataset_id: str,
        instrument_id: str,
        stage: Stage,
        source: Source,
        run_id: str,
        ts_min: int,
        ts_max: int,
        count: int,
        status: EventStatus,
        error: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.events.append(
            {
                "dataset_id": dataset_id,
                "instrument_id": instrument_id,
                "stage": stage,
                "source": source,
                "status": status,
                "count": count,
                "ts_min": ts_min,
                "ts_max": ts_max,
                "metadata": metadata or {},
            },
        )

    def update_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: Source,
        last_success_ns: int,
        count: int,
        completeness_pct: float,
    ) -> None:
        self.watermarks.append(
            {
                "dataset_id": dataset_id,
                "instrument_id": instrument_id,
                "source": source,
                "last_success_ns": last_success_ns,
                "count": count,
                "completeness_pct": completeness_pct,
            },
        )

    # Minimal DataRegistry API for DataStore auto-registration
    def get_manifest(self, dataset_id: str) -> DatasetManifest:
        try:
            return self._manifests[dataset_id]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError("manifest not found") from exc

    def get_contract(self, dataset_id: str) -> DataContract:
        return self._ensure_contract(dataset_id)

    def register_dataset(self, manifest: DatasetManifest) -> str:
        self._datasets.add(manifest.dataset_id)
        self._manifests[manifest.dataset_id] = manifest
        self._ensure_contract(manifest.dataset_id)
        return manifest.dataset_id

    def update_manifest(
        self,
        dataset_id: str,
        changes: dict[str, object],
    ) -> None:
        if dataset_id in self._manifests:
            self._manifests[dataset_id] = self._manifests[dataset_id]

    def _ensure_contract(self, dataset_id: str) -> DataContract:
        if dataset_id not in self._contracts:
            self._contracts[dataset_id] = DataContract(
                contract_id=f"{dataset_id}-contract",
                dataset_id=dataset_id,
                version="1.0.0",
                validation_rules=[
                    ValidationRule(
                        rule_type=ValidationRuleType.REGEX,
                        field_name="stub",
                        parameters={"pattern": ".*"},
                        severity=QualityFlag.WARN,
                        description="stub",
                    ),
                ],
            )
        return self._contracts[dataset_id]


class _StubFeatureStore:
    """
    Stub with typed write_features to capture publish_bus flag.
    """

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
    def write_batch(
        self,
        data: list[ModelPrediction],
        emit_events: bool = True,
        publish_bus: bool = True,
    ) -> None:  # noqa: D401
        return None


class _StubStrategyStore:
    def write_batch(
        self,
        data: list[StrategySignal],
        emit_events: bool = True,
        publish_bus: bool = True,
    ) -> None:  # noqa: D401
        return None


@given(n=st.integers(min_value=1, max_value=5))
@pytest.mark.property
def test_no_duplicate_publish_for_features(n: int) -> None:
    feature_store = _StubFeatureStore()
    model_store = _StubModelStore()
    strategy_store = _StubStrategyStore()
    pub = _CapturePublisher()
    registry = _StubRegistry()

    store = cast(
        Any,
        DataStore(
            connection_string="sqlite:///:memory:",
            registry=registry,
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

    store.write_features(
        instrument_id="EURUSD.SIM",
        features=items,
        source="computed",
        run_id="run_f",
    )

    # Underlying FeatureStore must be suppressed
    assert all(flag is False for flag in feature_store.publish_flags)

    # Exactly one bus publish from DataStore
    assert len(pub.calls) == 1
    assert len(registry.events) == 1
    assert len(registry.watermarks) == 1


@given(n=st.integers(min_value=1, max_value=5))
@pytest.mark.property
def test_no_duplicate_publish_for_predictions(n: int) -> None:
    feature_store = _StubFeatureStore()
    model_store = _StubModelStore()
    strategy_store = _StubStrategyStore()
    pub = _CapturePublisher()
    registry = _StubRegistry()

    store = cast(
        Any,
        DataStore(
            connection_string="sqlite:///:memory:",
            registry=registry,
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
    assert len(registry.events) == 1
    assert len(registry.watermarks) == 1


@given(n=st.integers(min_value=1, max_value=5))
@pytest.mark.property
def test_no_duplicate_publish_for_signals(n: int) -> None:
    feature_store = _StubFeatureStore()
    model_store = _StubModelStore()
    strategy_store = _StubStrategyStore()
    pub = _CapturePublisher()
    registry = _StubRegistry()

    store = cast(
        Any,
        DataStore(
            connection_string="sqlite:///:memory:",
            registry=registry,
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
    assert len(registry.events) == 1
    assert len(registry.watermarks) == 1
