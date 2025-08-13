from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from ml.features.pipeline import PipelineRunner
from ml.features.pipeline import PipelineSpec
from ml.features.pipeline import TransformSpec
from ml.features.pipeline import register_transform
from ml.registry.base import DataRequirements


def test_pipeline_names_and_signature_stability() -> None:
    spec = PipelineSpec(
        transforms=[
            TransformSpec(name="returns", params={"periods": [1, 5]}),
            TransformSpec(name="momentum", params={"periods": [5]}),
            TransformSpec(name="volatility"),
            TransformSpec(name="volume_ratio", params={"periods": [5, 10]}),
            TransformSpec(name="core_indicators"),
        ],
    )
    runner = PipelineRunner(spec, allowable=DataRequirements.L1_ONLY)
    names = runner.compute_feature_names()
    assert names[:2] == ["return_1", "return_5"]
    sig1 = runner.compute_signature()
    sig2 = PipelineRunner(spec, allowable=DataRequirements.L1_ONLY).compute_signature()
    assert sig1 == sig2


def test_pipeline_gating_for_l2() -> None:
    # Register a dummy L2 transform for testing gating
    class L2Only:
        name = "dummy_l2"

        def feature_names(self, params: Mapping[str, Any]) -> list[str]:
            return ["l2_feature"]

        def requires(self) -> DataRequirements:
            return DataRequirements.L1_L2

    register_transform(L2Only())

    spec = PipelineSpec(transforms=[TransformSpec(name="dummy_l2")])
    with pytest.raises(ValueError):
        PipelineRunner(spec, allowable=DataRequirements.L1_ONLY)


def test_pipeline_signature_changes_on_params() -> None:
    spec1 = PipelineSpec(
        transforms=[TransformSpec(name="returns", params={"periods": [1, 5]})],
    )
    spec2 = PipelineSpec(
        transforms=[TransformSpec(name="returns", params={"periods": [1, 10]})],
    )
    r1 = PipelineRunner(spec1, allowable=DataRequirements.L1_ONLY)
    r2 = PipelineRunner(spec2, allowable=DataRequirements.L1_ONLY)
    assert r1.compute_signature() != r2.compute_signature()


def test_optional_transforms_gated_under_l1_only() -> None:
    # Optional transforms require L1_L2
    for name in ("keltner", "obv", "microstructure", "trade_flow"):
        spec = PipelineSpec(transforms=[TransformSpec(name=name)])
        with pytest.raises(ValueError):
            PipelineRunner(spec, allowable=DataRequirements.L1_ONLY)
        # Should compile when allowable is L1_L2
        PipelineRunner(spec, allowable=DataRequirements.L1_L2)
