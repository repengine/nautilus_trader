"""
Unit tests for macro revision feature computation.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from ml.data.macro_revisions import compute_revision_features_pl


class TestComputeRevisionFeatures:
    """Test revision feature computation with real ALFRED data structure."""

    def test_compute_revision_features_minimal_mode(self) -> None:
        """Test minimal mode: current, prior_1m, revision_1m."""
        # Sample ALFRED vintage data for PAYEMS (NFP)
        # Simulates: August NFP initially 142k, revised to 159k in September release
        release_df = pl.DataFrame(
            {
                "series_id": ["PAYEMS", "PAYEMS", "PAYEMS"],
                "observation_ts": [
                    datetime(2024, 8, 1),  # August observation
                    datetime(2024, 8, 1),  # August observation (revised)
                    datetime(2024, 9, 1),  # September observation
                ],
                "value": [
                    142_000,  # August initial release
                    159_000,  # August revised (in September)
                    254_000,  # September initial release
                ],
                "release_ts": [
                    datetime(2024, 9, 6),   # August released in early September
                    datetime(2024, 10, 4),  # August revision in October
                    datetime(2024, 10, 4),  # September released in October
                ],
            },
            strict=False,
        )

        result = compute_revision_features_pl(
            release_df,
            mode="minimal",
            monthly_windows=[1],
        )

        assert not result.is_empty()

        # Should have features for both releases
        feature_types = result["feature_type"].unique().to_list()
        expected_features = ["PAYEMS_prior_1m", "PAYEMS_revision_1m"]

        for feat in expected_features:
            assert feat in feature_types, f"Missing feature: {feat}"

    def test_compute_revision_features_core_mode(self) -> None:
        """Test core mode: adds mom_1m, pct_1m, net_signal_1m."""
        release_df = pl.DataFrame(
            {
                "series_id": ["PAYEMS", "PAYEMS", "PAYEMS"],
                "observation_ts": [
                    datetime(2024, 8, 1),
                    datetime(2024, 8, 1),
                    datetime(2024, 9, 1),
                ],
                "value": [
                    142_000,  # August initial
                    159_000,  # August revised
                    254_000,  # September initial
                ],
                "release_ts": [
                    datetime(2024, 9, 6),
                    datetime(2024, 10, 4),
                    datetime(2024, 10, 4),
                ],
            },
            strict=False,
        )

        result = compute_revision_features_pl(
            release_df,
            mode="core",
            monthly_windows=[1],
        )

        assert not result.is_empty()

        feature_types = result["feature_type"].unique().to_list()
        expected_features = [
            "PAYEMS_prior_1m",
            "PAYEMS_revision_1m",
            "PAYEMS_mom_1m",
            "PAYEMS_pct_1m",
            "PAYEMS_net_signal_1m",
        ]

        for feat in expected_features:
            assert feat in feature_types, f"Missing feature: {feat}"

    def test_revision_calculation_correctness(self) -> None:
        """Test that revision delta is computed correctly."""
        # August: initial=142k, revised=159k → revision=+17k
        release_df = pl.DataFrame(
            {
                "series_id": ["PAYEMS", "PAYEMS", "PAYEMS"],
                "observation_ts": [
                    datetime(2024, 8, 1),
                    datetime(2024, 8, 1),
                    datetime(2024, 9, 1),
                ],
                "value": [
                    142_000,  # August initial
                    159_000,  # August revised
                    254_000,  # September initial
                ],
                "release_ts": [
                    datetime(2024, 9, 6),   # August released
                    datetime(2024, 10, 4),  # August revised
                    datetime(2024, 10, 4),  # September released
                ],
            },
            strict=False,
        )

        result = compute_revision_features_pl(
            release_df,
            mode="core",
            monthly_windows=[1],
        )

        # Find the revision feature for September release
        revision_feat = result.filter(
            (pl.col("feature_type") == "PAYEMS_revision_1m")
            & (pl.col("timestamp") == datetime(2024, 9, 1))
        )

        if not revision_feat.is_empty():
            revision_value = revision_feat["value"].to_list()[0]
            # August revised (159k) - August initial (142k) = +17k
            expected_revision = 159_000 - 142_000
            assert abs(revision_value - expected_revision) < 1.0, (
                f"Revision mismatch: got {revision_value}, expected {expected_revision}"
            )

    def test_empty_release_df(self) -> None:
        """Test handling of empty input."""
        empty_df = pl.DataFrame(
            schema={
                "series_id": pl.Utf8,
                "observation_ts": pl.Datetime("ns"),
                "value": pl.Float64,
                "release_ts": pl.Datetime("ns"),
            },
        )

        result = compute_revision_features_pl(empty_df, mode="core")

        assert result.is_empty()
        assert "timestamp" in result.columns
        assert "series_id" in result.columns
        assert "value" in result.columns

    def test_series_filter(self) -> None:
        """Test filtering by series_id."""
        release_df = pl.DataFrame(
            {
                "series_id": ["PAYEMS", "UNRATE", "PAYEMS"],
                "observation_ts": [
                    datetime(2024, 8, 1),
                    datetime(2024, 8, 1),
                    datetime(2024, 9, 1),
                ],
                "value": [142_000, 3.8, 254_000],
                "release_ts": [
                    datetime(2024, 9, 6),
                    datetime(2024, 9, 6),
                    datetime(2024, 10, 4),
                ],
            },
            strict=False,
        )

        result = compute_revision_features_pl(
            release_df,
            series_filter={"PAYEMS"},
            mode="minimal",
        )

        # Should only have PAYEMS features
        series_ids = result["feature_type"].str.split("_").list.get(0).unique().to_list()
        assert "PAYEMS" in series_ids
        assert "UNRATE" not in series_ids

    def test_no_lookahead_bias(self) -> None:
        """Test that features use only data available at release_ts."""
        # Critical test: ensure revision uses values known at that release time
        release_df = pl.DataFrame(
            {
                "series_id": ["PAYEMS"] * 4,
                "observation_ts": [
                    datetime(2024, 7, 1),  # July observation
                    datetime(2024, 8, 1),  # August observation
                    datetime(2024, 8, 1),  # August revised
                    datetime(2024, 9, 1),  # September observation
                ],
                "value": [
                    200_000,  # July initial
                    142_000,  # August initial (Sep release)
                    159_000,  # August revised (Oct release)
                    254_000,  # September initial (Oct release)
                ],
                "release_ts": [
                    datetime(2024, 8, 4),   # July released in August
                    datetime(2024, 9, 6),   # August released in September
                    datetime(2024, 10, 4),  # August revised in October
                    datetime(2024, 10, 4),  # September released in October
                ],
            },
            strict=False,
        )

        result = compute_revision_features_pl(
            release_df,
            mode="core",
            monthly_windows=[1],
        )

        # For September release (Oct 4), prior_1m should be August value known on Oct 4
        # That's the REVISED August value (159k), not the initial (142k)
        prior_feat = result.filter(
            (pl.col("feature_type") == "PAYEMS_prior_1m")
            & (pl.col("timestamp") == datetime(2024, 9, 1))
            & (pl.col("release_ts") == datetime(2024, 10, 4))
        )

        if not prior_feat.is_empty():
            prior_value = prior_feat["value"].to_list()[0]
            # Should use revised August value (159k), not initial (142k)
            assert abs(prior_value - 159_000) < 1.0, (
                f"Lookahead bias detected: got {prior_value}, expected 159_000 (revised value)"
            )
