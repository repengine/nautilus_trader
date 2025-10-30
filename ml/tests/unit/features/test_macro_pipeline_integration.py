"""
Test macro feature integration with pipeline framework.
"""

from __future__ import annotations

import pytest

from ml.features.pipeline import PipelineRunner
from ml.features.pipeline import PipelineSpec
from ml.features.pipeline import TransformSpec
from ml.registry.base import DataRequirements


class TestMacroPipelineIntegration:
    """Test macro features in declarative pipeline."""

    def test_macro_transform_registered(self) -> None:
        """Test that macro transform is registered in catalog."""
        from ml.features.pipeline import _CATALOG

        assert "macro" in _CATALOG
        transform = _CATALOG["macro"]
        assert transform.name == "macro"
        # Compare by value to handle potential module reload issues
        assert transform.requires().value == "l1_only"

    def test_macro_feature_names_minimal_mode(self) -> None:
        """Test feature name generation for minimal mode."""
        spec = PipelineSpec(
            transforms=[
                TransformSpec(
                    name="macro",
                    params={
                        "series_ids": ["PAYEMS", "UNRATE"],
                        "include_revisions": True,
                        "revision_mode": "minimal",
                    },
                ),
            ],
        )

        runner = PipelineRunner(spec, allowable=DataRequirements.L1_ONLY)
        feature_names = runner.compute_feature_names()

        # Should have: base + prior_1m + revision_1m for each series
        expected = [
            "PAYEMS__value_real_time",
            "PAYEMS_prior_1m",
            "PAYEMS_revision_1m",
            "UNRATE__value_real_time",
            "UNRATE_prior_1m",
            "UNRATE_revision_1m",
        ]

        assert feature_names == expected

    def test_macro_feature_names_core_mode(self) -> None:
        """Test feature name generation for core mode."""
        spec = PipelineSpec(
            transforms=[
                TransformSpec(
                    name="macro",
                    params={
                        "series_ids": ["CPIAUCSL"],
                        "include_revisions": True,
                        "revision_mode": "core",
                    },
                ),
            ],
        )

        runner = PipelineRunner(spec, allowable=DataRequirements.L1_ONLY)
        feature_names = runner.compute_feature_names()

        # Should have minimal + mom_1m, pct_1m, net_signal_1m
        expected = [
            "CPIAUCSL__value_real_time",
            "CPIAUCSL_prior_1m",
            "CPIAUCSL_revision_1m",
            "CPIAUCSL_mom_1m",
            "CPIAUCSL_pct_1m",
            "CPIAUCSL_net_signal_1m",
        ]

        assert feature_names == expected

    def test_macro_feature_names_full_mode(self) -> None:
        """Test feature name generation for full mode."""
        spec = PipelineSpec(
            transforms=[
                TransformSpec(
                    name="macro",
                    params={
                        "series_ids": ["PAYEMS"],
                        "include_revisions": True,
                        "revision_mode": "full",
                    },
                ),
            ],
        )

        runner = PipelineRunner(spec, allowable=DataRequirements.L1_ONLY)
        feature_names = runner.compute_feature_names()

        # Should have all features
        expected = [
            "PAYEMS__value_real_time",
            "PAYEMS_prior_1m",
            "PAYEMS_revision_1m",
            "PAYEMS_mom_1m",
            "PAYEMS_pct_1m",
            "PAYEMS_net_signal_1m",
            "PAYEMS_prior_3m",
            "PAYEMS_prior_12m",
            "PAYEMS_mom_3m",
            "PAYEMS_mom_12m",
            "PAYEMS_pct_12m",
            "PAYEMS_revision_3m",
        ]

        assert feature_names == expected

    def test_macro_without_revisions(self) -> None:
        """Test feature names when revisions disabled."""
        spec = PipelineSpec(
            transforms=[
                TransformSpec(
                    name="macro",
                    params={
                        "series_ids": ["PAYEMS", "UNRATE"],
                        "include_revisions": False,  # Disabled
                    },
                ),
            ],
        )

        runner = PipelineRunner(spec, allowable=DataRequirements.L1_ONLY)
        feature_names = runner.compute_feature_names()

        # Should only have base values
        expected = [
            "PAYEMS__value_real_time",
            "UNRATE__value_real_time",
        ]

        assert feature_names == expected

    def test_macro_combined_with_technical(self) -> None:
        """Test macro features combined with technical indicators."""
        spec = PipelineSpec(
            transforms=[
                TransformSpec(name="returns", params={"periods": [1, 5]}),
                TransformSpec(name="volatility", params={}),
                TransformSpec(
                    name="macro",
                    params={
                        "series_ids": ["PAYEMS"],
                        "include_revisions": True,
                        "revision_mode": "minimal",
                    },
                ),
            ],
        )

        runner = PipelineRunner(spec, allowable=DataRequirements.L1_ONLY)
        feature_names = runner.compute_feature_names()

        # Should have returns + volatility + macro
        assert "return_1" in feature_names
        assert "return_5" in feature_names
        assert "volatility_5" in feature_names
        assert "volatility_20" in feature_names
        assert "PAYEMS__value_real_time" in feature_names
        assert "PAYEMS_prior_1m" in feature_names
        assert "PAYEMS_revision_1m" in feature_names

    def test_pipeline_signature_includes_macro(self) -> None:
        """Test that pipeline signature changes with macro params."""
        spec1 = PipelineSpec(
            transforms=[
                TransformSpec(
                    name="macro",
                    params={
                        "series_ids": ["PAYEMS"],
                        "include_revisions": False,
                    },
                ),
            ],
        )

        spec2 = PipelineSpec(
            transforms=[
                TransformSpec(
                    name="macro",
                    params={
                        "series_ids": ["PAYEMS"],
                        "include_revisions": True,  # Different
                        "revision_mode": "core",
                    },
                ),
            ],
        )

        runner1 = PipelineRunner(spec1, allowable=DataRequirements.L1_ONLY)
        runner2 = PipelineRunner(spec2, allowable=DataRequirements.L1_ONLY)

        # Signatures should be different
        sig1 = runner1.compute_signature()
        sig2 = runner2.compute_signature()

        assert sig1 != sig2

    def test_empty_series_ids(self) -> None:
        """Test that empty series_ids produces no features."""
        spec = PipelineSpec(
            transforms=[
                TransformSpec(
                    name="macro",
                    params={
                        "series_ids": [],  # Empty
                        "include_revisions": True,
                    },
                ),
            ],
        )

        runner = PipelineRunner(spec, allowable=DataRequirements.L1_ONLY)
        feature_names = runner.compute_feature_names()

        assert feature_names == []

    def test_macro_data_requirements(self) -> None:
        """Test that macro features only require L1 data."""
        spec = PipelineSpec(
            transforms=[
                TransformSpec(
                    name="macro",
                    params={"series_ids": ["PAYEMS"]},
                ),
            ],
        )

        # Should work with L1_ONLY
        runner = PipelineRunner(spec, allowable=DataRequirements.L1_ONLY)
        feature_names = runner.compute_feature_names()
        assert len(feature_names) > 0

        # Should also work with higher levels
        runner2 = PipelineRunner(spec, allowable=DataRequirements.L1_L2)
        feature_names2 = runner2.compute_feature_names()
        assert feature_names == feature_names2
