"""
Test macro feature transform parity (batch vs. real-time).
"""

from __future__ import annotations

from datetime import datetime, timedelta
import math
from pathlib import Path

import logging
import polars as pl
import pytest

from ml.data.validation import MacroCoverageError
from ml.features.macro_cache import MacroDataCache
from ml.features.macro_cache import MacroSeriesSnapshot
from ml.features.macro_transforms import MacroFeatureTransform
from ml.features.macro_composites import get_composite_feature_names


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

    def test_macro_composites_batch_and_realtime_parity(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Macro composites produce matching feature sets in batch and realtime."""

        base_series = [
            "BAMLC0A0CM",
            "BAMLH0A0HYM2",
            "TEDRATE",
            "VIXCLS",
            "DGS10",
            "DGS2",
            "DGS5",
            "DGS30",
            "DFII10",
            "FEDFUNDS",
            "WALCL",
            "SOFR",
            "OBFR",
            "TOTBKCR",
            "PAYEMS",
            "INDPRO",
            "CFNAI",
            "CPIAUCSL",
            "PCEPI",
            "PPIACO",
            "DTWEXBGS",
        ]

        expected_composites = {
            "credit_spread_ig",
            "credit_spread_hy",
            "term_spread",
            "fed_policy_stance",
            "growth_momentum",
            "inflation_momentum",
            "dollar_strength",
        }
        composite_feature_names = set(get_composite_feature_names())
        assert expected_composites <= composite_feature_names

        row_count = 120
        timestamps = [datetime(2024, 1, 1) + timedelta(days=index) for index in range(row_count)]
        base_frame = {
            "timestamp": timestamps,
            "close": [100.0 + float(index) for index in range(row_count)],
        }
        for idx, series_id in enumerate(base_series):
            base_frame[series_id] = [float(idx) + float(index) for index in range(row_count)]

        macro_df = pl.DataFrame(base_frame)

        def _fake_join_fred_asof(*_args: object, **_kwargs: object) -> pl.DataFrame:
            return macro_df

        monkeypatch.setattr("ml.data.fred_join.join_fred_asof", _fake_join_fred_asof)

        transform = MacroFeatureTransform(
            macro_series_ids=list(base_series),
            vintage_base_dir=tmp_path,
            include_composites=True,
        )

        batch_result = transform.compute_batch(
            pl.DataFrame({
                "timestamp": timestamps,
                "close": [100.0 + float(index) for index in range(row_count)],
            }),
        )

        for composite in expected_composites:
            assert composite in batch_result.columns

        observation_ts = datetime(2024, 5, 1)
        release_ts = datetime(2024, 6, 1)
        snapshots: dict[str, MacroSeriesSnapshot] = {}
        for idx, series_id in enumerate(base_series):
            history = tuple(float(idx) + float(value) for value in range(200))
            snapshots[series_id] = MacroSeriesSnapshot(
                series_id=series_id,
                current_value=history[-1],
                observation_ts=observation_ts,
                release_ts=release_ts,
                prior_1m_value=history[-30],
                prior_3m_value=history[-90],
                prior_12m_value=history[-180],
                history=history,
            )

        class _StubCache:
            def __init__(self, snaps: dict[str, MacroSeriesSnapshot]) -> None:
                self._snapshots = snaps

            def is_loaded(self) -> bool:
                return True

            def get_all_features(self, mode: str = "core") -> dict[str, float]:
                return {
                    f"{series_id}__value_real_time": snapshot.current_value
                    for series_id, snapshot in self._snapshots.items()
                }

            def get_snapshot(self, series_id: str) -> MacroSeriesSnapshot | None:
                return self._snapshots.get(series_id)

        transform._cache = _StubCache(snapshots)  # type: ignore[attr-defined]
        realtime_features = transform.compute_realtime()

        for composite in expected_composites:
            assert composite in realtime_features
            assert not math.isnan(realtime_features[composite])

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
        assert config["include_composites"] is False
        assert config["composite_history_window"] == 400

    def test_realtime_composites_with_cache_stub(self) -> None:
        """Real-time composite outputs are emitted when cache data available."""

        now = datetime(2024, 5, 1)

        def snapshot(
            series_id: str,
            *,
            current: float,
            prior_1m: float | None = None,
            prior_3m: float | None = None,
            prior_12m: float | None = None,
            history: tuple[float, ...] | None = None,
        ) -> MacroSeriesSnapshot:
            return MacroSeriesSnapshot(
                series_id=series_id,
                current_value=current,
                observation_ts=now,
                release_ts=now,
                prior_1m_value=prior_1m,
                prior_3m_value=prior_3m,
                prior_12m_value=prior_12m,
                revision_1m=None,
                revision_3m=None,
                initial_value=current,
                history=history or (current,),
            )

        fedfunds_history = tuple(2.0 + 0.01 * idx for idx in range(40))
        dexusal_history = tuple(95.0 + 0.05 * idx for idx in range(40))
        dexuseu_history = tuple(100.0 + 0.1 * idx + (0.2 if idx % 2 else 0.0) for idx in range(40))
        dexjpus_history = tuple(110.0 + 0.03 * ((idx % 5) ** 2) for idx in range(40))

        snapshots = {
            "BAMLC0A0CM": snapshot(
                "BAMLC0A0CM",
                current=150.0,
                prior_1m=145.0,
                prior_3m=120.0,
                history=tuple(110.0 + idx for idx in range(40)),
            ),
            "BAMLH0A0HYM2": snapshot(
                "BAMLH0A0HYM2",
                current=450.0,
                prior_1m=430.0,
                prior_3m=410.0,
                history=tuple(390.0 + idx * 2.0 for idx in range(40)),
            ),
            "TEDRATE": snapshot("TEDRATE", current=40.0, history=tuple(35.0 + idx for idx in range(40))),
            "VIXCLS": snapshot("VIXCLS", current=18.0, history=tuple(15.0 + 0.5 * idx for idx in range(40))),
            "T10Y2Y": snapshot("T10Y2Y", current=0.5),
            "DGS10": snapshot("DGS10", current=3.5, history=tuple(3.0 + 0.01 * idx for idx in range(40))),
            "DGS2": snapshot("DGS2", current=2.0, history=tuple(1.8 + 0.005 * idx for idx in range(40))),
            "DGS5": snapshot("DGS5", current=2.5),
            "DGS30": snapshot("DGS30", current=4.0),
            "DFII10": snapshot("DFII10", current=1.5),
            "FEDFUNDS": snapshot(
                "FEDFUNDS",
                current=fedfunds_history[-1],
                history=fedfunds_history,
            ),
            "WALCL": snapshot("WALCL", current=8_500_000.0),
            "TOTBKCR": snapshot(
                "TOTBKCR",
                current=17_000_000.0,
                prior_3m=16_000_000.0,
            ),
            "PAYEMS": snapshot(
                "PAYEMS",
                current=152_000.0,
                prior_1m=151_000.0,
            ),
            "INDPRO": snapshot(
                "INDPRO",
                current=110.0,
                prior_1m=109.0,
            ),
            "CFNAI": snapshot("CFNAI", current=0.25),
            "CPIAUCSL": snapshot(
                "CPIAUCSL",
                current=300.0,
                prior_12m=285.0,
            ),
            "PCEPI": snapshot(
                "PCEPI",
                current=290.0,
                prior_12m=275.0,
            ),
            "PPIACO": snapshot(
                "PPIACO",
                current=260.0,
                prior_12m=250.0,
            ),
            "DTWEXBGS": snapshot(
                "DTWEXBGS",
                current=115.0,
                prior_3m=110.0,
            ),
            "DEXUSAL": snapshot("DEXUSAL", current=dexusal_history[-1], history=dexusal_history),
            "DEXUSEU": snapshot("DEXUSEU", current=dexuseu_history[-1], history=dexuseu_history),
            "DEXJPUS": snapshot("DEXJPUS", current=dexjpus_history[-1], history=dexjpus_history),
        }

        class _FakeCache:
            def __init__(self) -> None:
                self._features = {"PAYEMS__value_real_time": 152_000.0}

            def is_loaded(self) -> bool:
                return True

            def get_all_features(self, *, mode: str) -> dict[str, float]:
                return dict(self._features)

            def get_snapshot(self, series_id: str) -> MacroSeriesSnapshot | None:
                return snapshots.get(series_id)

        transform = MacroFeatureTransform(
            macro_series_ids=["PAYEMS"],
            vintage_base_dir=Path("/tmp"),
            include_composites=True,
            include_revisions=False,
        )
        transform._cache = _FakeCache()

        features = transform.compute_realtime()

        assert "PAYEMS__value_real_time" in features
        assert features["credit_spread_hy_ig"] == pytest.approx(300.0)
        assert features["liquidity_index"] == pytest.approx(0.3, rel=1e-6)

        expected_payems_mom = (152_000.0 - 151_000.0) / 151_000.0
        expected_indpro_mom = (110.0 - 109.0) / 109.0
        expected_growth = (
            (expected_payems_mom * 100.0)
            + (expected_indpro_mom * 100.0)
            + 0.25
        ) / 3.0
        assert features["growth_momentum"] == pytest.approx(expected_growth, rel=1e-6)

        expected_cpi_yoy = (300.0 - 285.0) / 285.0
        expected_pce_yoy = (290.0 - 275.0) / 275.0
        expected_ppi_yoy = (260.0 - 250.0) / 250.0
        expected_inflation = (
            expected_cpi_yoy * 100.0
            + expected_pce_yoy * 100.0
            + expected_ppi_yoy * 100.0
        ) / 3.0
        assert features["inflation_momentum"] == pytest.approx(expected_inflation, rel=1e-6)
        assert features["stagflation_risk"] == 0.0
        assert features["goldilocks_score"] == 0.0

        fedfunds_tail = fedfunds_history[-30:]
        mean_fedfunds = sum(fedfunds_tail) / len(fedfunds_tail)
        variance_fedfunds = sum((val - mean_fedfunds) ** 2 for val in fedfunds_tail) / len(
            fedfunds_tail,
        )
        expected_sofr = 40.0 + math.sqrt(variance_fedfunds)
        assert features["sofr_obfr_spread"] == pytest.approx(expected_sofr, rel=1e-6)

        assert "fx_volatility_composite" in features
        assert features["fx_volatility_composite"] >= 0.0

    def test_missing_composite_series_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Missing prerequisite series should trigger warning and NaN composites."""

        class _EmptyCache:
            def is_loaded(self) -> bool:
                return True

            def get_all_features(self, *, mode: str) -> dict[str, float]:
                return {}

            def get_snapshot(self, series_id: str) -> MacroSeriesSnapshot | None:
                return None

        transform = MacroFeatureTransform(
            macro_series_ids=["PAYEMS"],
            vintage_base_dir=Path("/tmp"),
            include_composites=True,
        )
        transform._cache = _EmptyCache()

        caplog.set_level(logging.WARNING)

        features = transform.compute_realtime()

        assert math.isnan(features["credit_spread_ig"])
        assert any("Macro composite prerequisite missing" in record.message for record in caplog.records)

    def test_macro_coverage_validator_passes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Macro coverage validation succeeds when thresholds are met."""

        timestamps = [datetime(2024, 1, 1) + timedelta(days=index) for index in range(10)]
        payload = pl.DataFrame(
            {
                "timestamp": timestamps,
                "PAYEMS": [float(index + 1) for index in range(10)],
                "UNRATE": [5.0 + 0.1 * float(index) for index in range(10)],
            },
        )

        def _join_stub(*_args: object, **_kwargs: object) -> pl.DataFrame:
            return payload

        monkeypatch.setattr("ml.data.fred_join.join_fred_asof", _join_stub)

        transform = MacroFeatureTransform(
            macro_series_ids=["PAYEMS", "UNRATE"],
            vintage_base_dir=tmp_path,
            include_revisions=False,
            min_coverage=0.8,
        )

        result = transform.compute_batch(
            pl.DataFrame({"timestamp": timestamps}),
        )

        assert "PAYEMS" in result.columns
        coverage_map = transform._last_coverage  # type: ignore[attr-defined]
        assert coverage_map is not None
        assert coverage_map["PAYEMS"] >= 0.8

    def test_macro_coverage_validator_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Macro coverage validation raises when coverage falls below threshold."""

        timestamps = [datetime(2024, 1, 1) + timedelta(days=index) for index in range(10)]
        sparse_values = [float(index + 1) if index % 2 == 0 else None for index in range(10)]
        payload = pl.DataFrame(
            {
                "timestamp": timestamps,
                "PAYEMS": sparse_values,
                "UNRATE": [5.0 + 0.1 * float(index) for index in range(10)],
            },
        )

        def _join_sparse(*_args: object, **_kwargs: object) -> pl.DataFrame:
            return payload

        monkeypatch.setattr("ml.data.fred_join.join_fred_asof", _join_sparse)

        transform = MacroFeatureTransform(
            macro_series_ids=["PAYEMS", "UNRATE"],
            vintage_base_dir=tmp_path,
            include_revisions=False,
            min_coverage=0.8,
        )

        with pytest.raises(MacroCoverageError):
            transform.compute_batch(
                pl.DataFrame({"timestamp": timestamps}),
            )
