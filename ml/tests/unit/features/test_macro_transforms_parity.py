"""
Test macro feature transform parity (batch vs. real-time).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from ml.features.macro_cache import MacroDataCache
from ml.features.macro_transforms import MacroFeatureTransform


class TestMacroTransformParity:
    """Test batch/real-time parity for macro features."""

    @pytest.fixture
    def vintage_dir(self) -> Path:
        """Path to ALFRED vintages."""
        return Path("data/fred/vintages")

    @pytest.fixture
    def test_series(self) -> list[str]:
        """Test with 2 series we know have vintages."""
        return ["CPIAUCSL", "PCEPI"]

    def test_cache_loads_successfully(
        self,
        vintage_dir: Path,
        test_series: list[str],
    ) -> None:
        """Test that cache loads vintage data."""
        if not vintage_dir.exists():
            pytest.skip("No vintage data available")

        cache = MacroDataCache(
            vintage_base_dir=vintage_dir,
            series_ids=test_series,
            enable_revisions=True,
        )

        assert cache.is_loaded()

        # Check coverage
        coverage = cache.get_coverage()
        assert all(coverage.values()), f"Missing coverage: {coverage}"

    def test_realtime_features_match_structure(
        self,
        vintage_dir: Path,
        test_series: list[str],
    ) -> None:
        """Test that real-time features have correct structure."""
        if not vintage_dir.exists():
            pytest.skip("No vintage data available")

        transform = MacroFeatureTransform(
            macro_series_ids=test_series,
            vintage_base_dir=vintage_dir,
            include_revisions=True,
            revision_mode="core",
        )

        # Get real-time features
        features = transform.compute_realtime()

        # Check feature names match get_feature_names()
        expected_names = set(transform.get_feature_names())
        actual_names = set(features.keys())

        # May have fewer (if some series missing data), but should not have extra
        extra_features = actual_names - expected_names
        assert not extra_features, f"Unexpected features: {extra_features}"

    def test_feature_names_match_mode(
        self,
        vintage_dir: Path,
    ) -> None:
        """Test that feature names respect revision mode."""
        if not vintage_dir.exists():
            pytest.skip("No vintage data available")

        series = ["CPIAUCSL"]

        # Minimal mode
        transform_minimal = MacroFeatureTransform(
            macro_series_ids=series,
            vintage_base_dir=vintage_dir,
            include_revisions=True,
            revision_mode="minimal",
        )
        names_minimal = transform_minimal.get_feature_names()

        # Should have: current, prior_1m, revision_1m
        expected_minimal = [
            "CPIAUCSL__value_real_time",
            "CPIAUCSL_prior_1m",
            "CPIAUCSL_revision_1m",
        ]
        assert set(names_minimal) == set(expected_minimal)

        # Core mode
        transform_core = MacroFeatureTransform(
            macro_series_ids=series,
            vintage_base_dir=vintage_dir,
            include_revisions=True,
            revision_mode="core",
        )
        names_core = transform_core.get_feature_names()

        # Should have minimal + mom_1m, pct_1m, net_signal_1m
        expected_core = expected_minimal + [
            "CPIAUCSL_mom_1m",
            "CPIAUCSL_pct_1m",
            "CPIAUCSL_net_signal_1m",
        ]
        assert set(names_core) == set(expected_core)

    def test_batch_computation_runs(
        self,
        vintage_dir: Path,
        test_series: list[str],
    ) -> None:
        """Test that batch computation doesn't crash."""
        if not vintage_dir.exists():
            pytest.skip("No vintage data available")

        transform = MacroFeatureTransform(
            macro_series_ids=test_series,
            vintage_base_dir=vintage_dir,
            include_revisions=True,
            revision_mode="core",
        )

        # Create dummy market data
        df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 10, 1, 9, 30),
                datetime(2024, 10, 1, 9, 31),
                datetime(2024, 10, 1, 9, 32),
            ],
            "close": [100.0, 101.0, 102.0],
        })

        # Compute batch features
        result = transform.compute_batch(df)

        # Should have original columns + macro columns
        assert "timestamp" in result.columns
        assert "close" in result.columns

        # Should have some macro features
        macro_cols = [col for col in result.columns if any(s in col for s in test_series)]
        assert len(macro_cols) > 0, "No macro columns added"

    def test_cache_refresh(
        self,
        vintage_dir: Path,
        test_series: list[str],
    ) -> None:
        """Test cache refresh mechanism."""
        if not vintage_dir.exists():
            pytest.skip("No vintage data available")

        transform = MacroFeatureTransform(
            macro_series_ids=test_series,
            vintage_base_dir=vintage_dir,
            include_revisions=True,
        )

        # Initial features
        features1 = transform.compute_realtime()

        # Refresh cache
        transform.refresh_cache()

        # Get features again
        features2 = transform.compute_realtime()

        # Should have same keys (data might differ if new releases)
        assert features1.keys() == features2.keys()

    def test_transform_config_serialization(
        self,
        vintage_dir: Path,
    ) -> None:
        """Test transform configuration can be serialized."""
        transform = MacroFeatureTransform(
            macro_series_ids=["PAYEMS", "UNRATE"],
            vintage_base_dir=vintage_dir,
            include_revisions=True,
            revision_mode="core",
            lag_days=1,
        )

        config = transform.get_transform_config()

        assert config["transform_type"] == "macro_features"
        assert config["macro_series_ids"] == ["PAYEMS", "UNRATE"]
        assert config["include_revisions"] is True
        assert config["revision_mode"] == "core"
        assert config["lag_days"] == 1
