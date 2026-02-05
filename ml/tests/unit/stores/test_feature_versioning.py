from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ml.registry.base import DataRequirements
from ml.stores.feature_versioning import FeatureVersioning


@dataclass(frozen=True)
class DummyConfig:
    """
    Minimal config object for hashing tests.
    """

    lookback: int
    name: str


class StubPipelineRunner:
    """
    Minimal pipeline runner stub for feature name and signature checks.
    """

    def __init__(self, signature: str, feature_names: list[str]) -> None:
        self._signature = signature
        self._feature_names = feature_names
        self.signature_calls = 0
        self.feature_name_calls = 0

    def compute_signature(self) -> str:
        self.signature_calls += 1
        return self._signature

    def compute_feature_names(self) -> list[str]:
        self.feature_name_calls += 1
        return list(self._feature_names)


def test_compute_config_hash_stable_for_equivalent_config() -> None:
    """
    Hash should be stable for equivalent configs and change on mutation.
    """
    cfg_a = DummyConfig(lookback=20, name="alpha")
    cfg_b = DummyConfig(lookback=20, name="alpha")
    cfg_c = DummyConfig(lookback=30, name="alpha")

    hash_a = FeatureVersioning(cfg_a).compute_config_hash()
    hash_b = FeatureVersioning(cfg_b).compute_config_hash()
    hash_c = FeatureVersioning(cfg_c).compute_config_hash()

    assert hash_a == hash_b
    assert hash_a != hash_c
    assert len(hash_a) == 16
    assert all(ch in "0123456789abcdef" for ch in hash_a)


def test_feature_set_id_prefers_pipeline_signature() -> None:
    """
    Feature set IDs should use pipeline signatures when available.
    """
    runner = StubPipelineRunner(signature="abcdef0123456789", feature_names=["f1"])
    versioning = FeatureVersioning(
        DummyConfig(lookback=10, name="beta"),
        pipeline_runner_offline=runner,
    )

    assert versioning.get_feature_set_id() == "fs_abcdef012345"


def test_feature_names_use_pipeline_runners_when_provided() -> None:
    """
    Pipeline runners should be used directly when supplied.
    """
    offline_runner = StubPipelineRunner(signature="sig-off", feature_names=["a", "b"])
    online_runner = StubPipelineRunner(signature="sig-on", feature_names=["x"])

    versioning = FeatureVersioning(
        DummyConfig(lookback=5, name="gamma"),
        pipeline_runner_offline=offline_runner,
        pipeline_runner_online=online_runner,
    )

    assert versioning.get_feature_names() == ["a", "b"]
    assert versioning.get_feature_names(online=True) == ["x"]


def test_feature_names_offline_fallback_uses_feature_engineer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Offline fallback should delegate to FeatureEngineer when no runner is provided.
    """

    class StubEngineer:
        def __init__(self, config: Any) -> None:
            self.config = config

        def get_feature_names(self) -> list[str]:
            return ["f_off_1", "f_off_2"]

    import ml.features as features

    monkeypatch.setattr(features, "FeatureEngineer", StubEngineer)

    versioning = FeatureVersioning(DummyConfig(lookback=1, name="delta"))
    assert versioning.get_feature_names() == ["f_off_1", "f_off_2"]


def test_feature_names_online_fallback_builds_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Online fallback should build a pipeline and respect L1_ONLY constraints.
    """

    class StubEngineer:
        def __init__(self, config: Any) -> None:
            self.config = config

        def build_pipeline_spec_from_config(self) -> object:
            return object()

    class StubPipelineRunner:
        last_allowable: Any = None

        def __init__(self, spec: object, allowable: Any) -> None:
            self.__class__.last_allowable = allowable

        def compute_feature_names(self) -> list[str]:
            return ["f_on_1"]

    import ml.features as features
    import ml.features.pipeline as pipeline

    monkeypatch.setattr(features, "FeatureEngineer", StubEngineer)
    monkeypatch.setattr(pipeline, "PipelineRunner", StubPipelineRunner)

    versioning = FeatureVersioning(DummyConfig(lookback=2, name="epsilon"))
    assert versioning.get_feature_names(online=True) == ["f_on_1"]
    assert StubPipelineRunner.last_allowable == DataRequirements.L1_ONLY
