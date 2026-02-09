from __future__ import annotations

from types import SimpleNamespace

import pytest

from ml.training.non_distilled.lightgbm import LightGBMTrainer
from ml.training.non_distilled.xgboost import XGBoostTrainer


pytestmark = pytest.mark.unit


def _make_xgb_config(*, random_seed: int | None) -> SimpleNamespace:
    return SimpleNamespace(
        objective="binary:logistic",
        eval_metric="auc",
        max_depth=6,
        learning_rate=0.3,
        subsample=1.0,
        colsample_bytree=1.0,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        min_child_weight=1.0,
        scale_pos_weight=None,
        random_seed=random_seed,
    )


def _make_lgb_config(
    *,
    random_state: int | None,
    random_seed: int | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        num_leaves=31,
        max_depth=6,
        learning_rate=0.1,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        bagging_freq=0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        min_child_samples=20,
        scale_pos_weight=None,
        random_state=random_state,
        random_seed=random_seed,
    )


def test_xgboost_get_model_params_uses_configured_seed() -> None:
    trainer = object.__new__(XGBoostTrainer)
    trainer._xgb_config = _make_xgb_config(random_seed=123)

    params = trainer._get_model_params()

    assert params["seed"] == 123


def test_xgboost_get_model_params_when_seed_missing_raises_value_error() -> None:
    trainer = object.__new__(XGBoostTrainer)
    trainer._xgb_config = _make_xgb_config(random_seed=None)

    with pytest.raises(ValueError, match="xgboost random seed must be configured"):
        trainer._get_model_params()


def test_lightgbm_get_model_params_prefers_random_state_seed() -> None:
    trainer = object.__new__(LightGBMTrainer)
    trainer._lgb_config = _make_lgb_config(
        random_state=321,
        random_seed=77,
    )

    params = trainer._get_model_params()

    assert params["seed"] == 321


def test_lightgbm_get_model_params_uses_random_seed_fallback() -> None:
    trainer = object.__new__(LightGBMTrainer)
    trainer._lgb_config = _make_lgb_config(
        random_state=None,
        random_seed=55,
    )

    params = trainer._get_model_params()

    assert params["seed"] == 55


def test_lightgbm_get_model_params_when_seed_missing_raises_value_error() -> None:
    trainer = object.__new__(LightGBMTrainer)
    trainer._lgb_config = _make_lgb_config(
        random_state=None,
        random_seed=None,
    )

    with pytest.raises(ValueError, match="lightgbm random seed must be configured"):
        trainer._get_model_params()
