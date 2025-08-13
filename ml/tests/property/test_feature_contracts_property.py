from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ml.features.pipeline import PipelineRunner
from ml.features.pipeline import PipelineSpec
from ml.features.pipeline import TransformSpec
from ml.features.pipeline import register_transform
from ml.registry.base import DataRequirements


class _L2Only:
    name = "_prop_l2"

    def feature_names(self, params: Mapping[str, Any]) -> list[str]:
        return ["l2_prop"]

    def requires(self) -> DataRequirements:
        return DataRequirements.L1_L2


register_transform(_L2Only())


@given(include_l2=st.booleans())
def test_student_contract_l1_only_property(include_l2: bool) -> None:
    tfs = [TransformSpec(name="returns", params={"periods": [1]})]
    if include_l2:
        tfs.append(TransformSpec(name="_prop_l2"))
    spec = PipelineSpec(transforms=tfs)

    if include_l2:
        with pytest.raises(ValueError):
            PipelineRunner(spec, allowable=DataRequirements.L1_ONLY)
    else:
        PipelineRunner(spec, allowable=DataRequirements.L1_ONLY)
