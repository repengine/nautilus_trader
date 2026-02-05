"""
Unit tests for signal actor factory routing.
"""

from __future__ import annotations

from pathlib import Path

import msgspec
import pytest
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId

from ml.actors.multi_signal import MultiInstrumentSignalActor
from ml.actors.multi_signal import MultiInstrumentSignalActorConfig
from ml.actors.signal import create_signal_actor
from ml.actors.signal_facade_impl import MLSignalActorFacade
from ml.config.actors import MLSignalActorConfig
from ml.config.base import MLFeatureConfig
from ml.registry.base import DeploymentStatus
from ml.registry.base import ModelInfo


pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry")


@pytest.mark.unit
def test_create_signal_actor_returns_facade(
    base_signal_config: MLSignalActorConfig,
) -> None:
    actor = create_signal_actor(base_signal_config)
    assert isinstance(actor, MLSignalActorFacade)


@pytest.mark.unit
def test_create_signal_actor_routes_multi_instrument(
    default_bar_type: BarType,
    default_instrument_id: InstrumentId,
    dummy_onnx_model: Path,
    base_feature_config: MLFeatureConfig,
) -> None:
    config = MultiInstrumentSignalActorConfig(
        model_id="multi-model",
        model_path=str(dummy_onnx_model),
        bar_type=default_bar_type,
        instrument_id=default_instrument_id,
        feature_config=base_feature_config,
        batch_size=1,
        warm_up_period=5,
        prediction_threshold=0.5,
        use_dummy_stores=True,
        signal_strategy="threshold",
        max_batch_size=4,
        feature_dim=2,
    )

    actor = create_signal_actor(config)
    assert isinstance(actor, MultiInstrumentSignalActor)


@pytest.mark.unit
def test_create_signal_actor_routes_multi_from_manifest_metadata(
    base_signal_config: MLSignalActorConfig,
    sample_model_manifest,
    dummy_onnx_model: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = ModelInfo(
        manifest=sample_model_manifest,
        model_path=dummy_onnx_model,
        deployment_status=DeploymentStatus.ACTIVE,
        deployed_to=[],
        metadata={"universe_instrument_ids": ["EUR/USD.SIM"]},
    )

    class _FakeRegistryFacade:
        def __init__(self, registry_path: Path) -> None:
            self.registry_path = registry_path

        def get_model(self, model_id: str) -> ModelInfo | None:
            return info

    monkeypatch.setattr(
        "ml.registry.model_registry_facade.ModelRegistryFacade",
        _FakeRegistryFacade,
    )

    config = msgspec.structs.replace(
        base_signal_config,
        registry_path=str(tmp_path),
    )

    actor = create_signal_actor(config)
    assert isinstance(actor, MultiInstrumentSignalActor)


@pytest.mark.unit
def test_create_signal_actor_uses_facade_when_manifest_has_no_universe(
    base_signal_config: MLSignalActorConfig,
    sample_model_manifest,
    dummy_onnx_model: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = ModelInfo(
        manifest=sample_model_manifest,
        model_path=dummy_onnx_model,
        deployment_status=DeploymentStatus.ACTIVE,
        deployed_to=[],
        metadata={"universe_instrument_ids": []},
    )

    class _FakeRegistryFacade:
        def __init__(self, registry_path: Path) -> None:
            self.registry_path = registry_path

        def get_model(self, model_id: str) -> ModelInfo | None:
            return info

    monkeypatch.setattr(
        "ml.registry.model_registry_facade.ModelRegistryFacade",
        _FakeRegistryFacade,
    )

    config = msgspec.structs.replace(
        base_signal_config,
        registry_path=str(tmp_path),
    )

    actor = create_signal_actor(config)
    assert isinstance(actor, MLSignalActorFacade)
