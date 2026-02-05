from __future__ import annotations

import pytest

from ml.actors.model_loader_utils import assert_features_parity
from ml.actors.model_loader_utils import maybe_warm_up_model


def test_assert_features_parity_accepts_matching_features() -> None:
    assert_features_parity(
        ["f1", "f2"],
        {"feature_schema": {"f1": "float32", "f2": "float32"}},
        ["f1", "f2"],
    )


def test_assert_features_parity_raises_on_mismatch() -> None:
    with pytest.raises(ValueError, match="Feature names/order mismatch"):
        assert_features_parity(
            ["f1", "f2"],
            {"feature_schema": {"f1": "float32", "f2": "float32"}},
            ["f2", "f1"],
        )


def test_assert_features_parity_no_manifest_is_noop() -> None:
    assert_features_parity(None, None, ["f1"])


def test_maybe_warm_up_model_noop_when_disabled() -> None:
    class _Model:
        def predict(self, _x):  # pragma: no cover - should not be called
            raise AssertionError("predict should not run")

    maybe_warm_up_model(_Model(), warm_up=False, input_dim=4)
