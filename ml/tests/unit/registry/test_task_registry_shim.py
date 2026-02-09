from __future__ import annotations

import importlib

import pytest

import ml.registry as registry_package
from ml.registry.feature_operations import FeaturePromotionGate as CanonicalFeaturePromotionGate
from ml.registry.feature_operations import deprecate_feature_set as canonical_deprecate_feature_set
from ml.registry.feature_operations import promote_feature_set as canonical_promote_feature_set
from ml.registry.feature_operations import register_default_feature_set as canonical_register_default_feature_set


def test_task_registry_shim_module_is_retired() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("ml.tasks.registry")


def test_registry_package_exports_feature_operations_symbols() -> None:
    assert registry_package.FeaturePromotionGate is CanonicalFeaturePromotionGate
    assert registry_package.register_default_feature_set is canonical_register_default_feature_set
    assert registry_package.promote_feature_set is canonical_promote_feature_set
    assert registry_package.deprecate_feature_set is canonical_deprecate_feature_set
