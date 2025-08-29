from __future__ import annotations

import pytest

from ml.actors.signal import MLSignalActor
from ml.actors.signal import MLSignalActorConfig
from ml.features.engineering import FeatureConfig
from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.identifiers import InstrumentId


def _make_actor() -> MLSignalActor:
    cfg = FeatureConfig()
    instrument_id = InstrumentId.from_str("TEST.XY")
    bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
    bar_type = BarType(instrument_id, bar_spec, AggressorSide.BUYER)
    acfg = MLSignalActorConfig(
        model_path="/tmp/dummy.onnx",  # not used for this unit test path
        model_id="dummy",
        bar_type=bar_type,
        instrument_id=instrument_id,
        feature_config=cfg,
        use_registry_features=False,
    )
    return MLSignalActor(acfg)


@pytest.mark.parallel_safe
@pytest.mark.unit
def test_model_manifest_feature_dtype_mismatch_raises() -> None:
    actor = _make_actor()
    names = actor._feature_engineer.config.get_feature_names()
    # Simulate manifest-based metadata injected by registry loader
    actor._manifest_feature_names = list(names)
    actor._manifest_feature_schema_hash = "abc123"
    # Make first dtype incompatible
    manifest_schema = {n: ("float64" if i == 0 else "float32") for i, n in enumerate(names)}
    actor._model_metadata = {"feature_schema": manifest_schema}

    with pytest.raises(ValueError):
        # Call internal validation used by actor during load
        # It should detect dtype mismatch and raise
        # We indirectly trigger via the public method path; tests are allowed to use internals here
        actual_names = actor._feature_engineer.config.get_feature_names()
        actual_dtypes = ["float32"] * len(actual_names)
        from ml.registry.base import DataRequirements
        from ml.registry.base import ModelRole
        from ml.registry.model_registry import ModelManifest
        from ml.registry.utils import assert_features_compatible

        tmp_manifest = ModelManifest(
            model_id="__validation__",
            role=ModelRole.STUDENT,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="unknown",
            feature_schema=manifest_schema,
            feature_schema_hash=actor._manifest_feature_schema_hash,
        )
        assert_features_compatible(tmp_manifest, actual_names, actual_dtypes)


def test_model_manifest_feature_dtype_match_passes() -> None:
    actor = _make_actor()
    names = actor._feature_engineer.config.get_feature_names()
    actor._manifest_feature_names = list(names)
    actor._manifest_feature_schema_hash = "abc123"
    manifest_schema = dict.fromkeys(names, "float32")
    actor._model_metadata = {"feature_schema": manifest_schema}

    # Should not raise
    actual_names = actor._feature_engineer.config.get_feature_names()
    actual_dtypes = ["float32"] * len(actual_names)
    from ml.registry.base import DataRequirements
    from ml.registry.base import ModelRole
    from ml.registry.model_registry import ModelManifest
    from ml.registry.utils import assert_features_compatible

    tmp_manifest = ModelManifest(
        model_id="__validation__",
        role=ModelRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="unknown",
        feature_schema=manifest_schema,
        feature_schema_hash=actor._manifest_feature_schema_hash,
    )
    assert_features_compatible(tmp_manifest, actual_names, actual_dtypes)
