"""
Test FeatureConfig integration with macro and calendar features.
"""

from __future__ import annotations

import pytest

from ml.features.engineering import FeatureConfig
from ml.features.engineering import build_pipeline_spec_from_feature_config
from ml.features.macro_composites import get_composite_feature_names


class TestFeatureConfigMacroIntegration:
    """Test that FeatureConfig properly wires macro/calendar into pipeline."""

    def test_default_config_no_macro(self) -> None:
        """Test that macro features are OFF by default."""
        cfg = FeatureConfig()

        spec = build_pipeline_spec_from_feature_config(cfg)
        transform_names = [t.name for t in spec.transforms]

        assert "macro" not in transform_names
        assert "calendar" not in transform_names

    def test_enable_macro_features(self) -> None:
        """Test enabling macro features."""
        cfg = FeatureConfig(
            include_macro=True,
            macro_series_ids=["PAYEMS", "UNRATE"],
            include_macro_revisions=False,
        )

        spec = build_pipeline_spec_from_feature_config(cfg)
        transform_names = [t.name for t in spec.transforms]

        assert "macro" in transform_names

        # Find macro transform and check params
        macro_transform = next(t for t in spec.transforms if t.name == "macro")
        assert macro_transform.params["series_ids"] == ["PAYEMS", "UNRATE"]
        assert macro_transform.params["include_revisions"] is False
        assert macro_transform.params["min_coverage"] is None

    def test_enable_macro_with_revisions(self) -> None:
        """Test enabling macro features with revisions."""
        cfg = FeatureConfig(
            include_macro=True,
            macro_series_ids=["CPIAUCSL", "PCEPI"],
            include_macro_revisions=True,
            macro_revision_mode="core",
        )

        spec = build_pipeline_spec_from_feature_config(cfg)

        macro_transform = next(t for t in spec.transforms if t.name == "macro")
        assert macro_transform.params["series_ids"] == ["CPIAUCSL", "PCEPI"]
        assert macro_transform.params["include_revisions"] is True
        assert macro_transform.params["revision_mode"] == "core"
        assert macro_transform.params["min_coverage"] is None

    def test_macro_min_coverage_propagates(self) -> None:
        """Macro min coverage config should propagate to pipeline spec."""
        cfg = FeatureConfig(
            include_macro=True,
            macro_series_ids=["PAYEMS", "UNRATE"],
            macro_min_coverage=0.87,
        )

        spec = build_pipeline_spec_from_feature_config(cfg)
        macro_transform = next(t for t in spec.transforms if t.name == "macro")
        assert macro_transform.params["min_coverage"] == pytest.approx(0.87)

    def test_enable_macro_composites(self) -> None:
        """Test enabling macro composites when macro features are active."""
        cfg = FeatureConfig(
            include_macro=True,
            macro_series_ids=["DGS10", "DGS2", "BAMLH0A0HYM2"],
            include_macro_composites=True,
        )

        spec = build_pipeline_spec_from_feature_config(cfg)
        transform_names = [t.name for t in spec.transforms]

        assert transform_names.count("macro") == 1
        assert transform_names.count("macro_composites") == 1

        macro_index = transform_names.index("macro")
        composites_index = transform_names.index("macro_composites")
        assert composites_index > macro_index

        feature_names = cfg.get_feature_names()
        expected_names = set(get_composite_feature_names())
        assert expected_names <= set(feature_names)

    def test_macro_composites_without_macro_raises(self) -> None:
        """Enabling composites without macro features should fail fast."""
        with pytest.raises(ValueError):
            FeatureConfig(
                include_macro=False,
                include_macro_composites=True,
            )

    def test_macro_composites_without_series_ids_raises(self) -> None:
        """Enabling composites without series IDs should fail."""
        with pytest.raises(ValueError):
            FeatureConfig(
                include_macro=True,
                macro_series_ids=[],
                include_macro_composites=True,
            )

    def test_enable_calendar_features(self) -> None:
        """Test enabling calendar features."""
        cfg = FeatureConfig(
            include_calendar=True,
            calendar_encoding="cyclic",
        )

        spec = build_pipeline_spec_from_feature_config(cfg)
        transform_names = [t.name for t in spec.transforms]

        assert "calendar" in transform_names

        calendar_transform = next(t for t in spec.transforms if t.name == "calendar")
        assert calendar_transform.params["encoding"] == "cyclic"

    def test_combined_technical_macro_calendar(self) -> None:
        """Test all features together."""
        cfg = FeatureConfig(
            # Technical (default enabled)
            return_periods=[1, 5, 10],
            # Macro
            include_macro=True,
            macro_series_ids=["PAYEMS", "UNRATE", "CPIAUCSL"],
            include_macro_revisions=True,
            macro_revision_mode="minimal",
            # Calendar
            include_calendar=True,
            calendar_encoding="cyclic",
        )

        spec = build_pipeline_spec_from_feature_config(cfg)
        transform_names = [t.name for t in spec.transforms]

        # Should have technical + macro + calendar
        assert "returns" in transform_names
        assert "volatility" in transform_names
        assert "core_indicators" in transform_names
        assert "macro" in transform_names
        assert "calendar" in transform_names

    def test_macro_feature_names_generated(self) -> None:
        """Test that macro features are included in feature name enumeration."""
        cfg = FeatureConfig(
            include_macro=True,
            macro_series_ids=["PAYEMS"],
            include_macro_revisions=True,
            macro_revision_mode="minimal",
        )

        feature_names = cfg.get_feature_names()

        # Should include macro features
        assert "PAYEMS__value_real_time" in feature_names
        assert "PAYEMS_prior_1m" in feature_names
        assert "PAYEMS_revision_1m" in feature_names

    def test_calendar_feature_names_generated(self) -> None:
        """Test that calendar features are included in feature name enumeration."""
        cfg = FeatureConfig(
            include_calendar=True,
            calendar_encoding="cyclic",
        )

        feature_names = cfg.get_feature_names()

        # Should include calendar features
        assert "hour_sin" in feature_names
        assert "hour_cos" in feature_names
        assert "dow_sin" in feature_names
        assert "dow_cos" in feature_names

    def test_full_feature_config_comprehensive(self) -> None:
        """Test comprehensive feature config like TFT would use."""
        cfg = FeatureConfig(
            # Price features
            return_periods=[1, 5, 10, 20],
            momentum_periods=[5, 10, 20],
            # Macro features (23 series with revisions)
            include_macro=True,
            macro_series_ids=[
                "PAYEMS", "UNRATE", "INDPRO", "CFNAI",
                "CPIAUCSL", "PCEPI", "PPIACO",
                "DGS2", "DGS10", "FEDFUNDS",
            ],
            include_macro_revisions=True,
            macro_revision_mode="core",
            # Calendar features
            include_calendar=True,
            calendar_encoding="cyclic",
        )

        spec = build_pipeline_spec_from_feature_config(cfg)
        feature_names = cfg.get_feature_names()

        # Verify diversity
        assert len(spec.transforms) >= 7  # returns, momentum, volatility, volume_ratio, core_indicators, macro, calendar
        assert len(feature_names) > 100  # Technical + 10 macro series * 6 features + calendar

        # Spot check
        assert "return_1" in feature_names
        assert any("rsi" in name.lower() for name in feature_names)  # RSI feature present
        assert "PAYEMS__value_real_time" in feature_names
        assert "PAYEMS_revision_1m" in feature_names
        assert "hour_sin" in feature_names

    def test_empty_macro_series_ids(self) -> None:
        """Test that empty macro_series_ids is handled correctly."""
        cfg = FeatureConfig(
            include_macro=True,
            macro_series_ids=[],  # Empty!
        )

        spec = build_pipeline_spec_from_feature_config(cfg)

        # Should still have macro transform (but it will produce no features)
        macro_transform = next(t for t in spec.transforms if t.name == "macro")
        assert macro_transform.params["series_ids"] == []

        feature_names = cfg.get_feature_names()
        # No macro features should appear
        assert not any("PAYEMS" in name for name in feature_names)

    def test_backward_compatibility_without_macro(self) -> None:
        """Test that existing configs without macro fields still work."""
        # Create config without macro fields (simulating old code)
        cfg = FeatureConfig(
            return_periods=[1, 5],
            enable_returns=True,
        )

        # Should not crash
        spec = build_pipeline_spec_from_feature_config(cfg)
        feature_names = cfg.get_feature_names()

        assert "return_1" in feature_names
        assert "macro" not in [t.name for t in spec.transforms]
