from __future__ import annotations

from pathlib import Path

import pytest

from ml.actors.signal import MLSignalActor
from ml.actors.signal import MLSignalActorConfig
from ml.features.engineering import FeatureConfig
from ml.registry.base import DataRequirements
from ml.registry.feature_registry import FeatureManifest
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.feature_registry import FeatureRole
from ml.registry.feature_registry import compute_schema_hash
from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.identifiers import InstrumentId


@pytest.mark.parallel_safe
@pytest.mark.unit
def test_actor_validates_feature_manifest(tmp_path: Path) -> None:
    # Prepare feature manifest matching default FeatureConfig
    cfg = FeatureConfig()
    names = cfg.get_feature_names()
    dtypes = ["float32"] * len(names)
    sig = "sig-match"
    schema_hash = compute_schema_hash(names, dtypes, sig)
    manifest = FeatureManifest(
        feature_set_id="",
        name="default",
        version="1.0.0",
        role=FeatureRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        feature_names=names,
        feature_dtypes=dtypes,
        schema_hash=schema_hash,
        pipeline_signature=sig,
        pipeline_version="0.1.0",
    )
    reg = FeatureRegistry(tmp_path)
    fid = reg.register_feature_set(manifest)
    # Actor config using registry-based features
    instrument_id = InstrumentId.from_str("TEST.XY")
    bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
    bar_type = BarType(instrument_id, bar_spec, AggressorSide.BUYER)
    acfg = MLSignalActorConfig(
        model_path="/tmp/dummy.onnx",
        model_id="dummy",
        bar_type=bar_type,
        instrument_id=instrument_id,
        feature_config=cfg,
        feature_set_id=fid,
        registry_path=str(tmp_path),
        use_registry_features=True,
    )
    # Construct actor; it should validate schema successfully
    actor = MLSignalActor(acfg)
