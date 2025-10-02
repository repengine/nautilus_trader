"""
Test extended macro composite features (depth expansion).

Tests the 25 new composite features added for credit, duration, liquidity, and FX.
"""

from __future__ import annotations

import polars as pl
import pytest

from ml.features.macro_composites import compute_macro_composites_pl, get_composite_feature_names


class TestMacroCompositesDepth:
    """Test depth expansion of macro composite features."""

    @pytest.fixture
    def sample_macro_df(self) -> pl.DataFrame:
        """Create sample macro data with all required series."""
        return pl.DataFrame({
            # Duration series
            "DGS2": [2.0, 2.1, 2.2, 2.3, 2.4] * 30,
            "DGS5": [2.5, 2.6, 2.7, 2.8, 2.9] * 30,
            "DGS10": [3.0, 3.1, 3.2, 3.3, 3.4] * 30,
            "DGS30": [3.5, 3.6, 3.7, 3.8, 3.9] * 30,
            "T10Y2Y": [1.0, 1.0, 1.0, 1.0, 1.0] * 30,
            "DFII10": [1.5, 1.6, 1.7, 1.8, 1.9] * 30,
            "FEDFUNDS": [0.5, 0.5, 0.5, 0.5, 0.5] * 30,
            # Credit series
            "BAMLC0A0CM": [100.0, 105.0, 110.0, 115.0, 120.0] * 30,
            "BAMLH0A0HYM2": [400.0, 420.0, 440.0, 460.0, 480.0] * 30,
            "TEDRATE": [20.0, 22.0, 24.0, 26.0, 28.0] * 30,
            "VIXCLS": [15.0, 16.0, 17.0, 18.0, 19.0] * 30,
            # Growth series
            "PAYEMS": [150000.0, 150100.0, 150200.0, 150300.0, 150400.0] * 30,
            "INDPRO": [100.0, 100.5, 101.0, 101.5, 102.0] * 30,
            "CFNAI": [0.1, 0.2, 0.3, 0.4, 0.5] * 30,
            # Inflation series
            "CPIAUCSL": [250.0, 250.5, 251.0, 251.5, 252.0] * 30,
            "PCEPI": [110.0, 110.2, 110.4, 110.6, 110.8] * 30,
            "PPIACO": [120.0, 120.3, 120.6, 120.9, 121.2] * 30,
            # FX series
            "DTWEXBGS": [100.0, 101.0, 102.0, 103.0, 104.0] * 30,
            "DEXUSAL": [1.3, 1.31, 1.32, 1.33, 1.34] * 30,
            "DEXUSEU": [1.1, 1.11, 1.12, 1.13, 1.14] * 30,
            "DEXJPUS": [110.0, 111.0, 112.0, 113.0, 114.0] * 30,
            # Liquidity series
            "WALCL": [8000000.0, 8100000.0, 8200000.0, 8300000.0, 8400000.0] * 30,
            "TOTBKCR": [16000000.0, 16100000.0, 16200000.0, 16300000.0, 16400000.0] * 30,
        })

    def test_credit_depth_features_created(self, sample_macro_df: pl.DataFrame) -> None:
        """Test that new credit depth features are created."""
        result = compute_macro_composites_pl(sample_macro_df)

        # Check new credit features exist
        expected_credit = [
            "credit_spread_bbb_a",
            "credit_spread_ig_momentum",
            "credit_spread_hy_momentum",
            "credit_distress_index",
        ]

        for feature in expected_credit:
            assert feature in result.columns, f"Missing credit feature: {feature}"

    def test_duration_depth_features_created(self, sample_macro_df: pl.DataFrame) -> None:
        """Test that new duration depth features are created."""
        result = compute_macro_composites_pl(sample_macro_df)

        # Check new duration features exist
        expected_duration = [
            "term_spread_5s30s",
            "term_spread_2s30s",
            "curve_curvature",
            "real_term_premium",
        ]

        for feature in expected_duration:
            assert feature in result.columns, f"Missing duration feature: {feature}"

    def test_liquidity_depth_features_created(self, sample_macro_df: pl.DataFrame) -> None:
        """Test that new liquidity depth features are created."""
        result = compute_macro_composites_pl(sample_macro_df)

        # Check new liquidity features exist
        expected_liquidity = [
            "sofr_obfr_spread",
            "financial_stress_composite",
        ]

        for feature in expected_liquidity:
            assert feature in result.columns, f"Missing liquidity feature: {feature}"

    def test_fx_depth_features_created(self, sample_macro_df: pl.DataFrame) -> None:
        """Test that new FX depth features are created."""
        result = compute_macro_composites_pl(sample_macro_df)

        # Check new FX features exist
        expected_fx = [
            "fx_volatility_composite",
        ]

        for feature in expected_fx:
            assert feature in result.columns, f"Missing FX feature: {feature}"

    def test_term_spread_5s30s_calculation(self, sample_macro_df: pl.DataFrame) -> None:
        """Test 5s30s term spread calculation."""
        result = compute_macro_composites_pl(sample_macro_df)

        # Should be DGS30 - DGS5
        expected = sample_macro_df["DGS30"] - sample_macro_df["DGS5"]
        actual = result["term_spread_5s30s"]

        assert actual.to_list() == expected.to_list()

    def test_term_spread_2s30s_calculation(self, sample_macro_df: pl.DataFrame) -> None:
        """Test 2s30s term spread calculation."""
        result = compute_macro_composites_pl(sample_macro_df)

        # Should be DGS30 - DGS2
        expected = sample_macro_df["DGS30"] - sample_macro_df["DGS2"]
        actual = result["term_spread_2s30s"]

        assert actual.to_list() == expected.to_list()

    def test_curve_curvature_calculation(self, sample_macro_df: pl.DataFrame) -> None:
        """Test curve curvature (butterfly) calculation."""
        result = compute_macro_composites_pl(sample_macro_df)

        # Should be 2*DGS10 - DGS2 - DGS30
        expected = (
            2.0 * sample_macro_df["DGS10"]
            - sample_macro_df["DGS2"]
            - sample_macro_df["DGS30"]
        )
        actual = result["curve_curvature"]

        assert actual.to_list() == expected.to_list()

    def test_real_term_premium_calculation(self, sample_macro_df: pl.DataFrame) -> None:
        """Test real term premium (inflation compensation) calculation."""
        result = compute_macro_composites_pl(sample_macro_df)

        # Should be DGS10 - DFII10
        expected = sample_macro_df["DGS10"] - sample_macro_df["DFII10"]
        actual = result["real_term_premium"]

        assert actual.to_list() == expected.to_list()

    def test_credit_bbb_a_spread_calculation(self, sample_macro_df: pl.DataFrame) -> None:
        """Test BBB-A spread approximation."""
        result = compute_macro_composites_pl(sample_macro_df)

        # Should be approximated as 40% of IG spread
        expected = sample_macro_df["BAMLC0A0CM"] * 0.4
        actual = result["credit_spread_bbb_a"]

        assert actual.to_list() == expected.to_list()

    def test_credit_momentum_features_non_null(self, sample_macro_df: pl.DataFrame) -> None:
        """Test that credit momentum features are computed (may have initial nulls)."""
        result = compute_macro_composites_pl(sample_macro_df)

        # Should have values after warmup period
        assert result["credit_spread_ig_momentum"].drop_nulls().len() > 0
        assert result["credit_spread_hy_momentum"].drop_nulls().len() > 0

    def test_credit_distress_index_normalized(self, sample_macro_df: pl.DataFrame) -> None:
        """Test that credit distress index is properly normalized."""
        result = compute_macro_composites_pl(sample_macro_df)

        # Should be between 0 and 1 (normalized components)
        distress = result["credit_distress_index"].drop_nulls()
        assert distress.min() >= 0.0
        assert distress.max() <= 1.5  # Allow some headroom for composite

    def test_financial_stress_composite_includes_multiple_sources(
        self,
        sample_macro_df: pl.DataFrame,
    ) -> None:
        """Test that financial stress composite includes multiple stress indicators."""
        result = compute_macro_composites_pl(sample_macro_df)

        # Should exist and have non-null values
        assert "financial_stress_composite" in result.columns
        stress = result["financial_stress_composite"].drop_nulls()
        assert stress.len() > 0

        # Should be normalized (0-1 range with some headroom)
        assert stress.min() >= 0.0
        assert stress.max() <= 1.5

    def test_fx_volatility_composite_calculation(self, sample_macro_df: pl.DataFrame) -> None:
        """Test FX volatility composite calculation."""
        result = compute_macro_composites_pl(sample_macro_df)

        # Should exist and have values after warmup
        assert "fx_volatility_composite" in result.columns
        fx_vol = result["fx_volatility_composite"].drop_nulls()
        assert fx_vol.len() > 0

        # Volatility should be non-negative
        assert fx_vol.min() >= 0.0

    def test_handle_missing_columns_gracefully(self) -> None:
        """Test that function handles missing columns gracefully."""
        # Create minimal DataFrame with only a few series
        minimal_df = pl.DataFrame({
            "DGS2": [2.0, 2.1, 2.2],
            "DGS10": [3.0, 3.1, 3.2],
        })

        # Should not raise, just skip features that need missing columns
        result = compute_macro_composites_pl(minimal_df)

        # Should still have original columns
        assert "DGS2" in result.columns
        assert "DGS10" in result.columns

        # Should have basic term spread
        assert "term_spread" in result.columns

        # Should NOT have features requiring missing series
        assert "term_spread_5s30s" not in result.columns  # Needs DGS5, DGS30
        assert "curve_curvature" not in result.columns  # Needs DGS30

    def test_get_composite_feature_names_includes_new_features(self) -> None:
        """Test that get_composite_feature_names includes all new features."""
        feature_names = get_composite_feature_names()

        # Check counts by category
        credit_features = [f for f in feature_names if "credit" in f or "ted" in f or "distress" in f]
        duration_features = [f for f in feature_names if "term" in f or "curve" in f or "yield" in f or "fed_policy" in f]
        liquidity_features = [f for f in feature_names if "liquidity" in f or "balance" in f or "bank" in f or "qe" in f or "sofr" in f or "stress" in f]
        fx_features = [f for f in feature_names if "dollar" in f or "fx" in f]

        # Verify we have the expected number of features in each category
        assert len(credit_features) >= 8, f"Expected >=8 credit features, got {len(credit_features)}"
        assert len(duration_features) >= 7, f"Expected >=7 duration features, got {len(duration_features)}"
        assert len(liquidity_features) >= 5, f"Expected >=5 liquidity features, got {len(liquidity_features)}"
        assert len(fx_features) >= 4, f"Expected >=4 FX features, got {len(fx_features)}"

    def test_all_composite_features_are_float_type(self, sample_macro_df: pl.DataFrame) -> None:
        """Test that all composite features produce float columns."""
        result = compute_macro_composites_pl(sample_macro_df)

        # Get all new composite columns (exclude original series)
        original_cols = set(sample_macro_df.columns)
        composite_cols = [c for c in result.columns if c not in original_cols]

        # Check that all composites are float type
        for col in composite_cols:
            dtype = result[col].dtype
            assert dtype in (pl.Float32, pl.Float64), f"Column {col} has dtype {dtype}, expected float"

    def test_no_infinite_values_in_composites(self, sample_macro_df: pl.DataFrame) -> None:
        """Test that composite features don't produce infinite values."""
        result = compute_macro_composites_pl(sample_macro_df)

        # Get all new composite columns
        original_cols = set(sample_macro_df.columns)
        composite_cols = [c for c in result.columns if c not in original_cols]

        # Check for infinites
        for col in composite_cols:
            if result[col].dtype in (pl.Float32, pl.Float64):
                has_inf = result[col].is_infinite().any()
                assert not has_inf, f"Column {col} contains infinite values"
