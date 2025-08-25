"""
Tests for canonicalization of feature names between config, pipeline, and manifest.
"""

from __future__ import annotations

from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.features.engineering import build_pipeline_spec_from_feature_config
from ml.features.pipeline import PipelineRunner
from ml.registry.base import DataRequirements
from ml.registry.feature_registry import FeatureRole


def _pipeline_names(cfg: FeatureConfig) -> list[str]:
    allowable = (
        DataRequirements.L1_L2
        if (cfg.include_microstructure or cfg.include_trade_flow)
        else DataRequirements.L1_ONLY
    )
    spec = build_pipeline_spec_from_feature_config(cfg)
    return PipelineRunner(spec, allowable=allowable).compute_feature_names()


def test_feature_names_parity_default() -> None:
    cfg = FeatureConfig()
    names_cfg = cfg.get_feature_names()
    names_pipe = _pipeline_names(cfg)
    assert names_cfg == names_pipe


def test_feature_names_parity_with_microstructure() -> None:
    cfg = FeatureConfig(include_microstructure=True)
    names_cfg = cfg.get_feature_names()
    names_pipe = _pipeline_names(cfg)
    assert names_cfg == names_pipe


def test_feature_names_parity_with_trade_flow() -> None:
    cfg = FeatureConfig(include_trade_flow=True)
    names_cfg = cfg.get_feature_names()
    names_pipe = _pipeline_names(cfg)
    assert names_cfg == names_pipe


def test_manifest_schema_matches_config_names() -> None:
    cfg = FeatureConfig()
    eng = FeatureEngineer(cfg)
    manifest = eng.generate_feature_manifest(
        name="unit-test-feature-set",
        version="0.0.1",
        role=FeatureRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
    )
    assert manifest.feature_names == cfg.get_feature_names()

