from __future__ import annotations

import types
from typing import Any

import pytest

from ml.actors.model_loader_utils import assert_features_parity, maybe_warm_up_model


class _OnnxLikeSession:
    def __init__(self) -> None:
        self._ran: bool = False

    def get_inputs(self):  # type: ignore[no-untyped-def]
        class _Inp:
            name = "x"

        return [_Inp()]

    def run(self, _outs: Any, _feed: dict[str, Any]):  # type: ignore[no-untyped-def]
        self._ran = True


class _SklearnLikeModel:
    def __init__(self) -> None:
        self._pred_called: bool = False

    def predict(self, _x: Any) -> None:  # type: ignore[no-untyped-def]
        self._pred_called = True


def test_maybe_warm_up_model_onnx_like() -> None:
    sess = _OnnxLikeSession()
    maybe_warm_up_model(sess, warm_up=True, input_dim=4)
    assert sess._ran is True


def test_maybe_warm_up_model_sklearn_like() -> None:
    model = _SklearnLikeModel()
    maybe_warm_up_model(model, warm_up=True, input_dim=3)
    assert model._pred_called is True


def test_maybe_warm_up_model_noop_when_disabled() -> None:
    model = _SklearnLikeModel()
    maybe_warm_up_model(model, warm_up=False, input_dim=3)
    assert model._pred_called is False


def test_maybe_warm_up_model_handles_exceptions() -> None:
    class _Bad:
        def get_inputs(self):  # type: ignore[no-untyped-def]
            class _Inp:
                name = "x"

            return [_Inp()]

        def run(self, *_a: Any, **_k: Any) -> None:  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

    bad = _Bad()
    # Should not raise
    maybe_warm_up_model(bad, warm_up=True, input_dim=2)


def test_assert_features_parity_happy_path() -> None:
    names = ["f1", "f2", "f3"]
    meta = {"feature_schema": {"f1": "float32", "f2": "float32", "f3": "float32"}}
    # Should not raise
    assert_features_parity(names, meta, names)


def test_assert_features_parity_no_manifest_names_is_noop() -> None:
    # No manifest names: function returns without validation
    assert_features_parity(None, None, ["a", "b"])  # type: ignore[arg-type]


def test_assert_features_parity_raises_on_mismatch() -> None:
    names = ["f1", "f2", "f3"]
    meta = {"feature_schema": {"f1": "float32", "f2": "float32", "f3": "float32"}}
    with pytest.raises(ValueError):
        assert_features_parity(names, meta, ["f2", "f1", "f3"])  # wrong order

