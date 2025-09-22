"""
Unit tests for known-future feature transforms.

Tests calendar, event schedule, and macro indicator transforms for TFT models.

"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from ml.features.engineering import FeatureConfig
from ml.features.pipeline import PipelineRunner
from ml.features.pipeline import PipelineSpec
from ml.features.pipeline import TransformSpec
from ml.features.pipeline import _CalendarTransform
from ml.features.pipeline import _EventScheduleTransform
from ml.features.pipeline import _MacroIndicatorsTransform
from ml.features.pipeline import _StaticCovariatesTransform
from ml.registry.base import DataRequirements


@pytest.mark.property
@pytest.mark.parallel_safe
@pytest.mark.unit
class TestCalendarTransform:
    """
    Test calendar-based known-future features.
    """

    def test_calendar_transform_cyclic_encoding(self) -> None:
        """
        Test calendar transform with cyclic encoding.
        """
        transform = _CalendarTransform()

        params = {
            "encoding": "cyclic",
            "granularity": "hour",
        }

        features = transform.feature_names(params)

        # Check cyclic features exist
        assert "hour_sin" in features
        assert "hour_cos" in features
        assert "dow_sin" in features
        assert "dow_cos" in features
        assert "month_sin" in features
        assert "month_cos" in features

        # Check calendar indicators
        assert "is_weekend" in features
        assert "is_month_start" in features
        assert "days_to_month_end" in features

    def test_calendar_transform_fourier_encoding(self) -> None:
        """
        Test calendar transform with Fourier encoding.
        """
        transform = _CalendarTransform()

        params = {
            "encoding": "fourier",
            "granularity": "hour",
            "n_harmonics": 3,
        }

        features = transform.feature_names(params)

        # Check Fourier harmonics
        for h in range(1, 4):
            assert f"hour_sin_{h}" in features
            assert f"hour_cos_{h}" in features

    def test_calendar_transform_onehot_encoding(self) -> None:
        """
        Test calendar transform with one-hot encoding.
        """
        transform = _CalendarTransform()

        params = {
            "encoding": "onehot",
            "granularity": "hour",
        }

        features = transform.feature_names(params)

        # Check one-hot encoded hours
        for h in range(24):
            assert f"hour_{h}" in features

        # Check one-hot encoded days
        for d in range(7):
            assert f"dow_{d}" in features

        # Check one-hot encoded months
        for m in range(1, 13):
            assert f"month_{m}" in features

    def test_calendar_transform_minute_granularity(self) -> None:
        """
        Test calendar transform with minute granularity.
        """
        transform = _CalendarTransform()

        params = {
            "encoding": "cyclic",
            "granularity": "minute",
        }

        features = transform.feature_names(params)

        # Should include minute features
        assert "minute_sin" in features
        assert "minute_cos" in features

    def test_calendar_transform_requires_l1_only(self) -> None:
        """
        Test that calendar transform only requires L1 data.
        """
        transform = _CalendarTransform()
        assert transform.requires() == DataRequirements.L1_ONLY

    @given(
        encoding=st.sampled_from(["cyclic", "fourier", "onehot"]),
        granularity=st.sampled_from(["minute", "hour", "day"]),
        n_harmonics=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=20, deadline=5000)
    def test_calendar_transform_property_feature_names_unique(
        self,
        encoding: str,
        granularity: str,
        n_harmonics: int,
    ) -> None:
        """Property: all feature names should be unique."""
        transform = _CalendarTransform()

        params = {
            "encoding": encoding,
            "granularity": granularity,
            "n_harmonics": n_harmonics,
        }

        features = transform.feature_names(params)

        # All features should be unique
        assert len(features) == len(set(features))


class TestEventScheduleTransform:
    """
    Test scheduled event features.
    """

    def test_event_schedule_default_params(self) -> None:
        """
        Test event schedule with default parameters.
        """
        transform = _EventScheduleTransform()

        features = transform.feature_names({})

        # Check time-to-event features
        assert "hours_to_earnings" in features
        assert "hours_to_fed_meeting" in features
        assert "has_earnings_in_24h" in features
        assert "has_fed_meeting_in_week" in features

        # Check proximity features
        assert "earnings_within_1h" in features
        assert "fed_meeting_within_24h" in features

        # Check aggregate features
        assert "total_events_24h" in features
        assert "event_density_week" in features

        # Check special conditions
        assert "is_triple_witching" in features
        assert "is_fomc_week" in features

    def test_event_schedule_custom_events(self) -> None:
        """
        Test event schedule with custom event types.
        """
        transform = _EventScheduleTransform()

        params = {
            "event_types": ["earnings", "dividend"],
            "horizon_hours": [2, 12],
        }

        features = transform.feature_names(params)

        # Should have features for custom events
        assert "hours_to_earnings" in features
        assert "hours_to_dividend" in features

        # Should have custom horizons
        assert "earnings_within_2h" in features
        assert "dividend_within_12h" in features

        # Should not have features for excluded events
        assert "hours_to_fed_meeting" not in features

    def test_event_schedule_requires_l1_only(self) -> None:
        """
        Test that event schedule only requires L1 data.
        """
        transform = _EventScheduleTransform()
        assert transform.requires() == DataRequirements.L1_ONLY

    @given(
        n_event_types=st.integers(min_value=1, max_value=10),
        n_horizons=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=20, deadline=5000)
    def test_event_schedule_property_feature_count(
        self,
        n_event_types: int,
        n_horizons: int,
    ) -> None:
        """Property: feature count should match expected formula."""
        transform = _EventScheduleTransform()

        event_types = [f"event_{i}" for i in range(n_event_types)]
        horizon_hours = list(range(1, n_horizons + 1))

        params = {
            "event_types": event_types,
            "horizon_hours": horizon_hours,
        }

        features = transform.feature_names(params)

        # Calculate expected count
        # Per event: hours_to, has_in_24h, has_in_week + horizons
        per_event_count = 3 + n_horizons
        event_features = n_event_types * per_event_count

        # Aggregate features (4) + special conditions (5)
        fixed_features = 9

        expected_min = event_features + fixed_features

        # Should have at least the expected number
        assert len(features) >= expected_min


def test_pipeline_runner_blocks_l2_transforms_for_student_mode() -> None:
    spec = PipelineSpec(transforms=[TransformSpec(name="microstructure", params={})])
    with pytest.raises(ValueError):
        PipelineRunner(spec, allowable=DataRequirements.L1_ONLY)


def test_feature_config_upgrades_requirements_for_microstructure() -> None:
    cfg = FeatureConfig(include_microstructure=True, data_requirements=DataRequirements.L1_ONLY)
    assert cfg.resolved_data_requirements() == DataRequirements.L1_L2


def test_feature_config_rejects_incompatible_requirements() -> None:
    with pytest.raises(ValueError):
        FeatureConfig(include_microstructure=True, data_requirements=DataRequirements.HISTORICAL)


def test_feature_config_allows_teacher_requirements() -> None:
    cfg = FeatureConfig(include_microstructure=True, data_requirements=DataRequirements.L1_L2)
    assert cfg.data_requirements == DataRequirements.L1_L2


class TestMacroIndicatorsTransform:
    """
    Test macroeconomic indicator features.
    """

    def test_macro_indicators_default_params(self) -> None:
        """
        Test macro indicators with default parameters.
        """
        transform = _MacroIndicatorsTransform()

        features = transform.feature_names({})

        # Check level features
        assert "vix" in features
        assert "treasury_10y" in features
        assert "fed_funds_rate" in features

        # Check change features
        assert "vix_change_1d" in features
        assert "treasury_10y_change_5d" in features

        # Check z-score features
        assert "vix_zscore_20d" in features
        assert "dxy_zscore_60d" in features

        # Check regime indicators
        assert "vix_regime" in features
        assert "yield_curve_regime" in features
        assert "rate_cycle_phase" in features

    def test_macro_indicators_custom_indicators(self) -> None:
        """
        Test macro indicators with custom selection.
        """
        transform = _MacroIndicatorsTransform()

        params = {
            "indicators": ["vix", "dxy"],
            "transformations": ["level", "change"],
        }

        features = transform.feature_names(params)

        # Should have level and change for selected indicators
        assert "vix" in features
        assert "dxy" in features
        assert "vix_change_1d" in features
        assert "dxy_change_5d" in features

        # Should not have z-score (not in transformations)
        assert "vix_zscore_20d" not in features

        # Should not have excluded indicators
        assert "treasury_10y" not in features

    def test_macro_indicators_requires_l1_only(self) -> None:
        """
        Test that macro indicators only require L1 data.
        """
        transform = _MacroIndicatorsTransform()
        assert transform.requires() == DataRequirements.L1_ONLY

    @given(
        n_indicators=st.integers(min_value=1, max_value=10),
        transformations=st.lists(
            st.sampled_from(["level", "change", "z_score"]),
            min_size=1,
            max_size=3,
            unique=True,
        ),
    )
    @settings(max_examples=20, deadline=5000)
    def test_macro_indicators_property_transformation_applied(
        self,
        n_indicators: int,
        transformations: list[str],
    ) -> None:
        """Property: all transformations should be applied to all indicators."""
        transform = _MacroIndicatorsTransform()

        indicators = [f"indicator_{i}" for i in range(n_indicators)]

        params = {
            "indicators": indicators,
            "transformations": transformations,
        }

        features = transform.feature_names(params)

        # Check that transformations are applied
        for indicator in indicators:
            for transform_type in transformations:
                if transform_type == "level":
                    assert indicator in features
                elif transform_type == "change":
                    assert f"{indicator}_change_1d" in features
                elif transform_type == "z_score":
                    assert f"{indicator}_zscore_20d" in features


class TestPipelineIntegration:
    """
    Test integration of known-future transforms in pipeline.
    """

    def test_pipeline_with_known_future_transforms(self) -> None:
        """
        Test pipeline can include known-future transforms.
        """
        spec = PipelineSpec(
            transforms=[
                TransformSpec(name="returns"),
                TransformSpec(name="calendar", params={"encoding": "cyclic"}),
                TransformSpec(name="event_schedule"),
                TransformSpec(name="macro_indicators"),
            ],
        )

        runner = PipelineRunner(spec, DataRequirements.L1_ONLY)

        # Should compile successfully
        features = runner.compute_feature_names()

        # Should have features from all transforms
        assert any("return_" in f for f in features)
        assert any("hour_sin" in f for f in features)
        assert any("hours_to_" in f for f in features)
        assert any("vix" in f for f in features)

    def test_pipeline_signature_with_known_future(self) -> None:
        """
        Test pipeline signature includes known-future transforms.
        """
        spec1 = PipelineSpec(
            transforms=[
                TransformSpec(name="calendar", params={"encoding": "cyclic"}),
            ],
        )

        spec2 = PipelineSpec(
            transforms=[
                TransformSpec(name="calendar", params={"encoding": "onehot"}),
            ],
        )

        runner1 = PipelineRunner(spec1, DataRequirements.L1_ONLY)
        runner2 = PipelineRunner(spec2, DataRequirements.L1_ONLY)

        # Different params should produce different signatures
        sig1 = runner1.compute_signature()
        sig2 = runner2.compute_signature()

        assert sig1 != sig2

    def test_known_future_transforms_l1_compatible(self) -> None:
        """
        Test all known-future transforms work with L1-only data.
        """
        spec = PipelineSpec(
            transforms=[
                TransformSpec(name="calendar"),
                TransformSpec(name="event_schedule"),
                TransformSpec(name="macro_indicators"),
            ],
        )

        # Should work with L1_ONLY requirement
        runner = PipelineRunner(spec, DataRequirements.L1_ONLY)
        features = runner.compute_feature_names()

        # Should produce features
        assert len(features) > 0

        # All transforms should be included
        assert len(runner._transforms) == 3


class TestStaticCovariatesTransform:
    """
    Test static covariate features for TFT models.
    """

    def test_static_covariates_default_params(self) -> None:
        """
        Test static covariates with default parameters.
        """
        transform = _StaticCovariatesTransform()

        features = transform.feature_names({})

        # Check numeric features
        assert "tick_size" in features
        assert "lot_size" in features
        assert "contract_size" in features
        assert "min_price_increment" in features
        assert "margin_initial" in features
        assert "margin_maintenance" in features

        # Check categorical features
        assert "exchange" in features
        assert "asset_class" in features
        assert "currency" in features
        assert "fee_class" in features
        assert "market_segment" in features

    def test_static_covariates_custom_params(self) -> None:
        """
        Test static covariates with custom parameters.
        """
        transform = _StaticCovariatesTransform()

        params = {
            "numeric_features": ["tick_size", "lot_size"],
            "categorical_features": ["exchange", "asset_class"],
        }

        features = transform.feature_names(params)

        # Should only have specified features
        assert set(features) == {"tick_size", "lot_size", "exchange", "asset_class"}

    def test_static_covariates_requires_l1_only(self) -> None:
        """
        Test that static covariates only require L1 data.
        """
        transform = _StaticCovariatesTransform()
        assert transform.requires() == DataRequirements.L1_ONLY

    @given(
        n_numeric=st.integers(min_value=0, max_value=10),
        n_categorical=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=20, deadline=5000)
    def test_static_covariates_property_correct_count(
        self,
        n_numeric: int,
        n_categorical: int,
    ) -> None:
        """Property: feature count should match inputs."""
        transform = _StaticCovariatesTransform()

        numeric_features = [f"numeric_{i}" for i in range(n_numeric)]
        categorical_features = [f"categorical_{i}" for i in range(n_categorical)]

        params = {
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
        }

        features = transform.feature_names(params)

        # Should have exactly the right number of features
        assert len(features) == n_numeric + n_categorical

        # All features should be unique
        assert len(features) == len(set(features))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
