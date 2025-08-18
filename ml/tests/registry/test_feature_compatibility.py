from __future__ import annotations

import pytest

from ml.registry.base import DataRequirements
from ml.registry.base import ModelManifest
from ml.registry.base import ModelRole
from ml.registry.utils import assert_features_compatible


def test_assert_features_compatible_names_and_types() -> None:
    manifest = ModelManifest(
        model_id="m1",
        role=ModelRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="LightGBM",
        feature_schema={"f1": "float32", "f2": "float32"},
        feature_schema_hash="hash",
    )
    # Exact match
    assert_features_compatible(manifest, ["f1", "f2"], ["float32", "float32"])

    # Wrong order
    with pytest.raises(ValueError):
        assert_features_compatible(manifest, ["f2", "f1"], ["float32", "float32"])

    # Wrong dtype
    with pytest.raises(ValueError):
        assert_features_compatible(manifest, ["f1", "f2"], ["float64", "float32"])
