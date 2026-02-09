#!/usr/bin/env python3
"""
Consolidated Strategy Registry Tests.

This file consolidates tests from:
- ml/tests/test_strategy_registry.py (merged and removed)
- ml/tests/unit/registry/test_strategy_registry.py (original)

Consolidation performed on 2025-08-25.

"""

from __future__ import annotations

import tempfile
from datetime import UTC
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.registry.strategy_registry import StrategyManifest
from ml.registry.strategy_registry import MarketRegime
from ml.registry.strategy_registry import StrategyInfo
from ml.registry.strategy_registry import StrategyRegistry
from ml.registry.strategy_registry import StrategyType
from ml.tests.builders import RegistryBuilder


# =================================================================================================
# Hypothesis Strategies for Generating Test Data
# =================================================================================================


@st.composite
def strategy_type_strategy(draw: st.DrawFn) -> StrategyType:
    """
    Generate valid StrategyType enum values.
    """
    return draw(st.sampled_from(list(StrategyType)))


@st.composite
def market_regime_strategy(draw: st.DrawFn) -> MarketRegime:
    """
    Generate valid MarketRegime enum values.
    """
    return draw(st.sampled_from(list(MarketRegime)))


@st.composite
def strategy_manifest_strategy(draw: st.DrawFn):
    """
    Generate valid StrategyManifest instances.
    """
    # Use a simple alphabet for strategy IDs
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
    strategy_id = draw(st.text(min_size=1, max_size=50, alphabet=alphabet))

    # Ensure unique strategy_id by adding a counter
    strategy_id = f"strategy_{strategy_id}_{draw(st.integers(0, 1000))}"

    return RegistryBuilder.strategy_manifest(
        strategy_id=strategy_id,
        strategy_type=draw(strategy_type_strategy()),
        version=f"{draw(st.integers(0, 10))}.{draw(st.integers(0, 10))}.{draw(st.integers(0, 10))}",
        # Requirements
        required_models=draw(
            st.one_of(
                st.none(),
                st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=5),
            ),
        ),
        required_features=draw(
            st.lists(st.text(min_size=1, max_size=30, alphabet=alphabet), min_size=0, max_size=10),
        ),
        # Market conditions
        suitable_regimes=draw(
            st.lists(market_regime_strategy(), min_size=1, max_size=len(MarketRegime), unique=True),
        ),
        instrument_types=draw(
            st.lists(
                st.sampled_from(["FX", "EQUITY", "CRYPTO", "COMMODITY"]),
                min_size=1,
                max_size=4,
                unique=True,
            ),
        ),
        timeframe_range=(
            draw(st.sampled_from(["1m", "5m", "15m", "30m"])),
            draw(st.sampled_from(["30m", "1h", "4h", "1d"])),
        ),
        # Risk parameters
        max_position_size=draw(st.floats(min_value=0.01, max_value=1.0)),
        max_leverage=draw(st.floats(min_value=1.0, max_value=10.0)),
        max_drawdown=draw(st.floats(min_value=0.01, max_value=0.5)),
        stop_loss_type=draw(st.sampled_from(["FIXED", "TRAILING", "ADAPTIVE"])),
        # Performance constraints
        min_sharpe_ratio=draw(st.floats(min_value=-2.0, max_value=5.0)),
        min_win_rate=draw(st.floats(min_value=0.0, max_value=1.0)),
        max_correlation_with_portfolio=draw(st.floats(min_value=0.0, max_value=1.0)),
        # Dependencies
        parent_strategy_id=draw(st.one_of(st.none(), st.text(min_size=1, max_size=50))),
        incompatible_strategies=draw(st.lists(st.text(min_size=1, max_size=50), max_size=5)),
        # Configuration
        config_schema=draw(
            st.dictionaries(
                st.text(min_size=1, max_size=20),
                st.sampled_from(["int", "float", "str", "bool"]),
                max_size=5,
            ),
        ),
        default_config=draw(
            st.dictionaries(
                st.text(min_size=1, max_size=20),
                st.one_of(
                    st.integers(),
                    st.floats(allow_nan=False, allow_infinity=False),
                    st.text(),
                    st.booleans(),
                ),
                max_size=5,
            ),
        ),
        # Performance metrics
        backtest_metrics=draw(
            st.dictionaries(
                st.text(min_size=1, max_size=20),
                st.floats(allow_nan=False, allow_infinity=False),
                max_size=10,
            ),
        ),
        live_metrics=draw(
            st.one_of(
                st.none(),
                st.dictionaries(
                    st.text(min_size=1, max_size=20),
                    st.floats(allow_nan=False, allow_infinity=False),
                    max_size=10,
                ),
            ),
        ),
        # Metadata
        created_at=draw(st.floats(min_value=1600000000, max_value=2000000000)),
        last_modified=draw(st.floats(min_value=1600000000, max_value=2000000000)),
        author=draw(st.text(min_size=1, max_size=50)),
        description=draw(st.text(min_size=0, max_size=200)),
    )


# =================================================================================================
# Helper Functions
# =================================================================================================


def _write_dummy_strategy(path: Path) -> None:
    path.write_text(
        """
def run():
    return "ok"
""",
    )


# =================================================================================================
# Basic Tests
# =================================================================================================


@pytest.mark.property
@pytest.mark.slow
@pytest.mark.unit
def test_strategy_registry_register_and_query() -> None:
    """
    Basic test for strategy registration and querying.
    """
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        reg = StrategyRegistry(base)

        strategy_file = base / "my_strategy.py"
        _write_dummy_strategy(strategy_file)

        manifest = RegistryBuilder.strategy_manifest(
            strategy_id="momentum_v1",
            strategy_type=StrategyType.MOMENTUM,
            version="1.0.0",
            required_models=["student_v1"],
            required_features=["student_features_v1"],
            suitable_regimes=[MarketRegime.TRENDING_UP, MarketRegime.VOLATILE],
            instrument_types=["FX", "CRYPTO"],
            timeframe_range=("1m", "1h"),
            max_position_size=1000.0,
            max_leverage=1.0,
            max_drawdown=0.2,
            stop_loss_type="atr",
            min_sharpe_ratio=0.5,
            min_win_rate=0.45,
            max_correlation_with_portfolio=0.8,
            parent_strategy_id=None,
            incompatible_strategies=[],
            config_schema={"threshold": "float"},
            default_config={"threshold": 0.7},
            backtest_metrics={"sharpe": 1.2},
            live_metrics=None,
            created_at=0.0,
            last_modified=0.0,
            author="test",
            description="test strategy",
        )

        sid = reg.register_strategy(strategy_file, manifest)
        info = reg.get_strategy(sid)
        assert isinstance(info, StrategyInfo)
        assert info.manifest.strategy_type == StrategyType.MOMENTUM

        # Regime filtering
        candidates = reg.get_strategies_for_regime(MarketRegime.TRENDING_UP)
        assert any(s.manifest.strategy_id == sid for s in candidates)

        # Requirements validation
        assert reg.validate_requirements(sid, ["student_v1"], ["student_features_v1"]) is True
        assert reg.validate_requirements(sid, [], []) is False


# =================================================================================================
# Comprehensive Test Suite
# =================================================================================================


def test_list_strategies_returns_registered_entries(tmp_path: Path) -> None:
    reg = StrategyRegistry(tmp_path)

    strategy_file = tmp_path / "list_strategy.py"
    _write_dummy_strategy(strategy_file)

    manifest = RegistryBuilder.strategy_manifest(
        strategy_id="list_strategy",
        strategy_type=StrategyType.MOMENTUM,
        version="1.0.0",
        required_models=[],
        required_features=[],
        suitable_regimes=[MarketRegime.TRENDING_UP],
        instrument_types=["FX"],
        timeframe_range=("1m", "5m"),
        max_position_size=100.0,
        max_leverage=1.0,
        max_drawdown=0.1,
        stop_loss_type="fixed",
        min_sharpe_ratio=0.5,
        min_win_rate=0.5,
        max_correlation_with_portfolio=0.9,
        parent_strategy_id=None,
        incompatible_strategies=[],
        config_schema={},
        default_config={},
        backtest_metrics={"sharpe": 1.0},
        live_metrics=None,
        created_at=0.0,
        last_modified=0.0,
        author="tester",
        description="list test",
    )

    reg.register_strategy(strategy_file, manifest)

    strategies = reg.list_strategies()
    strategy_ids = {info.manifest.strategy_id for info in strategies}
    assert "list_strategy" in strategy_ids


def test_strategy_manifest_round_trip_and_strategy_info_to_dict() -> None:
    manifest = RegistryBuilder.strategy_manifest(
        strategy_id="strategy_manifest_roundtrip",
        strategy_type=StrategyType.MOMENTUM,
        version="1.0.0",
        suitable_regimes=[MarketRegime.TRENDING_UP],
        timeframe_range=("1m", "5m"),
    )
    payload = manifest.to_dict()
    payload["timeframe_range"] = ["1m", "5m"]
    rebuilt = StrategyManifest.from_dict(payload)

    assert rebuilt.timeframe_range == ("1m", "5m")
    assert rebuilt.strategy_type == StrategyType.MOMENTUM

    info_payload = StrategyInfo(manifest=rebuilt, file_path=Path("/tmp/strategy.py")).to_dict()
    assert info_payload["manifest"]["strategy_id"] == "strategy_manifest_roundtrip"
    assert info_payload["file_path"] == "/tmp/strategy.py"


def test_postgres_session_none_paths_return_defaults(tmp_path: Path) -> None:
    registry = StrategyRegistry(tmp_path)
    registry.backend = BackendType.POSTGRES
    registry.persistence = SimpleNamespace(get_session=lambda: None)

    manifest = RegistryBuilder.strategy_manifest(
        strategy_id="session_none_strategy",
        strategy_type=StrategyType.TREND_FOLLOWING,
        version="1.0.0",
    )

    assert registry.get_strategy("missing") is None
    assert registry.list_strategies() == []
    assert registry.is_registered("missing") is False
    assert registry._health_snapshot() == (0, None)
    registry._save_strategy_to_db(manifest, tmp_path / "missing.py")


def test_postgres_health_snapshot_returns_count_and_latest(tmp_path: Path) -> None:
    registry = StrategyRegistry(tmp_path)
    registry.backend = BackendType.POSTGRES

    session = MagicMock()
    count_query = MagicMock()
    count_query.count.return_value = 3
    latest_query = MagicMock()
    latest_ts = datetime(2025, 1, 2, tzinfo=UTC)
    latest_query.order_by.return_value.first.return_value = (latest_ts,)
    session.query.side_effect = [count_query, latest_query]
    registry.persistence = SimpleNamespace(get_session=lambda: session)

    count, last_modified = registry._health_snapshot()
    assert count == 3
    assert last_modified == latest_ts.timestamp()
    session.close.assert_called_once()


def test_postgres_health_snapshot_returns_none_when_latest_missing(tmp_path: Path) -> None:
    registry = StrategyRegistry(tmp_path)
    registry.backend = BackendType.POSTGRES

    session = MagicMock()
    count_query = MagicMock()
    count_query.count.return_value = 1
    latest_query = MagicMock()
    latest_query.order_by.return_value.first.return_value = None
    session.query.side_effect = [count_query, latest_query]
    registry.persistence = SimpleNamespace(get_session=lambda: session)

    assert registry._health_snapshot() == (1, None)
    session.close.assert_called_once()


def test_db_to_strategy_info_parses_defaults_and_metadata_path(tmp_path: Path) -> None:
    registry = StrategyRegistry(tmp_path)
    db_strategy = SimpleNamespace(
        strategy_id="db_strategy",
        strategy_type=StrategyType.ARBITRAGE.value,
        version="2.0.0",
        required_models=None,
        required_features=["feat_a"],
        suitable_regimes=[MarketRegime.VOLATILE.value],
        instrument_types=["FX"],
        timeframe_range="1m,1h",
        max_position_size=1.5,
        max_leverage=2.0,
        max_drawdown=0.3,
        stop_loss_type="fixed",
        min_sharpe_ratio=1.1,
        min_win_rate=0.55,
        max_correlation_with_portfolio=0.7,
        parent_strategy_id=None,
        incompatible_strategies=["legacy"],
        config_schema={"lookback": "int"},
        default_config={"lookback": 14},
        backtest_metrics={"sharpe": 1.2},
        live_metrics={"sharpe": 1.1},
        created_at=None,
        last_modified=None,
        author="db-author",
        description="db-desc",
        extra_metadata={"file_path": str(tmp_path / "db_strategy.py")},
    )

    info = registry._db_to_strategy_info(db_strategy)
    assert info.manifest.strategy_id == "db_strategy"
    assert info.manifest.strategy_type == StrategyType.ARBITRAGE
    assert info.manifest.timeframe_range == ("1m", "1h")
    assert info.file_path == tmp_path / "db_strategy.py"


def test_save_strategy_to_db_handles_update_insert_and_errors(tmp_path: Path) -> None:
    registry = StrategyRegistry(tmp_path)
    registry.backend = BackendType.POSTGRES
    manifest = RegistryBuilder.strategy_manifest(
        strategy_id="persisted_strategy",
        strategy_type=StrategyType.MEAN_REVERSION,
        version="1.2.3",
        suitable_regimes=[MarketRegime.RANGING],
        instrument_types=["EQUITY"],
        timeframe_range=("5m", "1h"),
    )
    strategy_file = tmp_path / "persisted_strategy.py"

    existing_row = SimpleNamespace()
    update_session = MagicMock()
    update_session.query.return_value.filter_by.return_value.first.return_value = existing_row
    registry.persistence = SimpleNamespace(get_session=lambda: update_session)
    registry._save_strategy_to_db(manifest, strategy_file)

    assert existing_row.version == "1.2.3"
    assert existing_row.timeframe_range == "5m,1h"
    update_session.commit.assert_called_once()
    update_session.close.assert_called_once()

    insert_session = MagicMock()
    insert_session.query.return_value.filter_by.return_value.first.return_value = None
    registry.persistence = SimpleNamespace(get_session=lambda: insert_session)
    with pytest.raises(
        RuntimeError,
        match="Failed to save strategy to database: 'extra_metadata' is an invalid keyword argument for StrategyTable",
    ):
        registry._save_strategy_to_db(manifest, strategy_file)
    insert_session.rollback.assert_called_once()
    insert_session.close.assert_called_once()

    failing_session = MagicMock()
    failing_session.query.return_value.filter_by.return_value.first.return_value = SimpleNamespace()
    failing_session.commit.side_effect = RuntimeError("database down")
    registry.persistence = SimpleNamespace(get_session=lambda: failing_session)
    with pytest.raises(RuntimeError, match="Failed to save strategy to database: database down"):
        registry._save_strategy_to_db(manifest, strategy_file)
    failing_session.rollback.assert_called_once()
    failing_session.close.assert_called_once()


class TestStrategyRegistry:
    """
    Test suite for StrategyRegistry.
    """

    def test_registry_initialization(self) -> None:
        """
        Test that registry initializes correctly.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = StrategyRegistry(Path(tmpdir))
            assert registry.base_path == Path(tmpdir)
            assert (Path(tmpdir) / "strategies").exists()
            assert (Path(tmpdir) / "strategies" / "registry.json").exists()

    @given(manifest=strategy_manifest_strategy())
    def test_register_strategy(self, manifest: StrategyManifest) -> None:
        """
        Test strategy registration with various manifests.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = StrategyRegistry(Path(tmpdir))

            # Create a dummy strategy file
            strategy_path = Path(tmpdir) / f"{manifest.strategy_id}.py"
            strategy_path.write_text("# Dummy strategy implementation")

            # Register strategy
            strategy_id = registry.register_strategy(strategy_path, manifest)
            assert strategy_id == manifest.strategy_id

            # Verify strategy is registered
            assert registry.is_registered(strategy_id)

            # Verify files created
            strategy_dir = Path(tmpdir) / "strategies" / strategy_id
            assert strategy_dir.exists()
            assert (strategy_dir / "manifest.json").exists()
            assert (strategy_dir / f"{manifest.strategy_id}.py").exists()

    @given(manifest=strategy_manifest_strategy())
    def test_get_strategy(self, manifest: StrategyManifest) -> None:
        """
        Test retrieving registered strategies.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = StrategyRegistry(Path(tmpdir))

            # Register strategy
            strategy_path = Path(tmpdir) / f"{manifest.strategy_id}.py"
            strategy_path.write_text("# Dummy strategy")
            registry.register_strategy(strategy_path, manifest)

            # Retrieve strategy
            retrieved = registry.get_strategy(manifest.strategy_id)
            assert retrieved is not None
            assert retrieved.manifest.strategy_id == manifest.strategy_id
            assert retrieved.manifest.strategy_type == manifest.strategy_type

    @given(
        manifests=st.lists(
            strategy_manifest_strategy(),
            min_size=2,
            max_size=10,
            unique_by=lambda m: m.strategy_id,
        ),
    )
    def test_filter_by_regime(self, manifests: list[StrategyManifest]) -> None:
        """
        Test filtering strategies by market regime.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = StrategyRegistry(Path(tmpdir))

            # Register all strategies
            for manifest in manifests:
                strategy_path = Path(tmpdir) / f"{manifest.strategy_id}.py"
                strategy_path.write_text("# Dummy")
                registry.register_strategy(strategy_path, manifest)

            # Test filtering for each regime
            for regime in MarketRegime:
                filtered = registry.get_strategies_for_regime(regime)

                # Verify all returned strategies support the regime
                for strategy_info in filtered:
                    assert regime in strategy_info.manifest.suitable_regimes

                # Verify we got all strategies that support this regime
                expected_count = sum(1 for m in manifests if regime in m.suitable_regimes)
                assert len(filtered) == expected_count

    @given(
        manifests=st.lists(
            strategy_manifest_strategy(),
            min_size=2,
            max_size=10,
            unique_by=lambda m: m.strategy_id,
        ),
    )
    def test_filter_by_instrument_type(self, manifests: list[StrategyManifest]) -> None:
        """
        Test filtering strategies by instrument type.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = StrategyRegistry(Path(tmpdir))

            # Register all strategies
            for manifest in manifests:
                strategy_path = Path(tmpdir) / f"{manifest.strategy_id}.py"
                strategy_path.write_text("# Dummy")
                registry.register_strategy(strategy_path, manifest)

            # Test filtering for each instrument type
            for instrument_type in ["FX", "EQUITY", "CRYPTO", "COMMODITY"]:
                filtered = registry.get_strategies_for_instrument_type(instrument_type)

                # Verify all returned strategies support the instrument type
                for strategy_info in filtered:
                    assert instrument_type in strategy_info.manifest.instrument_types

    @given(manifest=strategy_manifest_strategy())
    def test_update_performance_metrics(self, manifest: StrategyManifest) -> None:
        """
        Test updating strategy performance metrics.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = StrategyRegistry(Path(tmpdir))

            # Register strategy
            strategy_path = Path(tmpdir) / f"{manifest.strategy_id}.py"
            strategy_path.write_text("# Dummy")
            registry.register_strategy(strategy_path, manifest)

            # Update live metrics
            new_metrics = {
                "sharpe_ratio": 2.1,
                "win_rate": 0.65,
                "total_trades": 100,
            }

            registry.update_live_metrics(manifest.strategy_id, new_metrics)

            # Retrieve and verify
            updated = registry.get_strategy(manifest.strategy_id)
            assert updated is not None
            assert updated.manifest.live_metrics == new_metrics

    @given(
        manifests=st.lists(
            strategy_manifest_strategy(),
            min_size=3,
            max_size=10,
            unique_by=lambda m: m.strategy_id,
        ),
        performance_metric=st.sampled_from(["sharpe_ratio", "win_rate", "total_return"]),
    )
    def test_rank_by_performance(
        self,
        manifests: list[StrategyManifest],
        performance_metric: str,
    ) -> None:
        """
        Test ranking strategies by performance.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = StrategyRegistry(Path(tmpdir))

            # Register strategies with varying performance
            for i, manifest in enumerate(manifests):
                # Add performance metric to backtest metrics
                manifest.backtest_metrics[performance_metric] = float(i)

                strategy_path = Path(tmpdir) / f"{manifest.strategy_id}.py"
                strategy_path.write_text("# Dummy")
                registry.register_strategy(strategy_path, manifest)

            # Get ranked strategies
            ranked = registry.get_strategies_ranked_by_performance(
                performance_metric,
                use_live_metrics=False,
            )

            # Verify ordering (should be descending)
            for i in range(len(ranked) - 1):
                current_value = ranked[i].manifest.backtest_metrics.get(performance_metric, 0)
                next_value = ranked[i + 1].manifest.backtest_metrics.get(performance_metric, 0)
                assert current_value >= next_value

    def test_validate_requirements(self) -> None:
        """
        Test strategy requirement validation.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = StrategyRegistry(Path(tmpdir))

            # Create manifest with specific requirements
            manifest = RegistryBuilder.strategy_manifest(
                strategy_id="test_strategy",
                strategy_type=StrategyType.TREND_FOLLOWING,
                version="1.0.0",
                required_models=["model_1", "model_2"],
                required_features=["feature_1", "feature_2"],
                suitable_regimes=[MarketRegime.TRENDING_UP],
                instrument_types=["FX"],
                timeframe_range=("5m", "1h"),
                max_position_size=0.1,
                max_leverage=2.0,
                max_drawdown=0.1,
                stop_loss_type="FIXED",
                min_sharpe_ratio=1.0,
                min_win_rate=0.5,
                max_correlation_with_portfolio=0.5,
                parent_strategy_id=None,
                incompatible_strategies=[],
                config_schema={},
                default_config={},
                backtest_metrics={"sharpe_ratio": 1.5},
                live_metrics=None,
                created_at=1700000000,
                last_modified=1700000000,
                author="test",
                description="Test strategy",
            )

            strategy_path = Path(tmpdir) / "test_strategy.py"
            strategy_path.write_text("# Dummy")
            registry.register_strategy(strategy_path, manifest)

            # Test with all requirements met
            assert registry.validate_requirements(
                "test_strategy",
                available_models=["model_1", "model_2", "model_3"],
                available_features=["feature_1", "feature_2", "feature_3"],
            )

            # Test with missing model
            assert not registry.validate_requirements(
                "test_strategy",
                available_models=["model_1"],
                available_features=["feature_1", "feature_2"],
            )

            # Test with missing feature
            assert not registry.validate_requirements(
                "test_strategy",
                available_models=["model_1", "model_2"],
                available_features=["feature_1"],
            )

    @given(
        manifests=st.lists(
            strategy_manifest_strategy(),
            min_size=2,
            max_size=5,
            unique_by=lambda m: m.strategy_id,
        ),
        compatibility_matrix=st.dictionaries(
            st.integers(0, 4),
            st.lists(st.integers(0, 4), min_size=0, max_size=4),
            max_size=5,
        ),
    )
    def test_check_compatibility(
        self,
        manifests: list[StrategyManifest],
        compatibility_matrix: dict[int, list[int]],
    ) -> None:
        """
        Test checking strategy compatibility.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = StrategyRegistry(Path(tmpdir))

            # Set up incompatibilities based on matrix
            for i, manifest in enumerate(manifests):
                if i in compatibility_matrix:
                    incompatible_indices = compatibility_matrix[i]
                    manifest.incompatible_strategies = [
                        manifests[j].strategy_id
                        for j in incompatible_indices
                        if j < len(manifests) and j != i
                    ]

            # Register all strategies
            for manifest in manifests:
                strategy_path = Path(tmpdir) / f"{manifest.strategy_id}.py"
                strategy_path.write_text("# Dummy")
                registry.register_strategy(strategy_path, manifest)

            # Test compatibility checks
            for i, manifest in enumerate(manifests):
                active_strategies = [
                    manifests[j].strategy_id for j in range(len(manifests)) if j != i
                ]

                is_compatible = registry.check_compatibility(
                    manifest.strategy_id,
                    active_strategies,
                )

                # Check if any active strategy is in the incompatible list
                should_be_compatible = not any(
                    active_id in manifest.incompatible_strategies for active_id in active_strategies
                )

                assert is_compatible == should_be_compatible

    def test_get_strategy_lineage(self) -> None:
        """
        Test retrieving strategy lineage (parent-child relationships).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = StrategyRegistry(Path(tmpdir))

            # Create a lineage: parent -> child -> grandchild
            parent = RegistryBuilder.strategy_manifest(
                strategy_id="parent_strategy",
                strategy_type=StrategyType.TREND_FOLLOWING,
                version="1.0.0",
                required_models=None,
                required_features=[],
                suitable_regimes=[MarketRegime.TRENDING_UP],
                instrument_types=["FX"],
                timeframe_range=("5m", "1h"),
                max_position_size=0.1,
                max_leverage=2.0,
                max_drawdown=0.1,
                stop_loss_type="FIXED",
                min_sharpe_ratio=1.0,
                min_win_rate=0.5,
                max_correlation_with_portfolio=0.5,
                parent_strategy_id=None,
                incompatible_strategies=[],
                config_schema={},
                default_config={},
                backtest_metrics={},
                live_metrics=None,
                created_at=1700000000,
                last_modified=1700000000,
                author="test",
                description="Parent strategy",
            )

            child = RegistryBuilder.strategy_manifest(
                strategy_id="child_strategy",
                strategy_type=StrategyType.TREND_FOLLOWING,
                version="2.0.0",
                required_models=None,
                required_features=[],
                suitable_regimes=[MarketRegime.TRENDING_UP],
                instrument_types=["FX"],
                timeframe_range=("5m", "1h"),
                max_position_size=0.1,
                max_leverage=2.0,
                max_drawdown=0.1,
                stop_loss_type="FIXED",
                min_sharpe_ratio=1.0,
                min_win_rate=0.5,
                max_correlation_with_portfolio=0.5,
                parent_strategy_id="parent_strategy",
                incompatible_strategies=[],
                config_schema={},
                default_config={},
                backtest_metrics={},
                live_metrics=None,
                created_at=1700001000,
                last_modified=1700001000,
                author="test",
                description="Child strategy",
            )

            grandchild = RegistryBuilder.strategy_manifest(
                strategy_id="grandchild_strategy",
                strategy_type=StrategyType.TREND_FOLLOWING,
                version="3.0.0",
                required_models=None,
                required_features=[],
                suitable_regimes=[MarketRegime.TRENDING_UP],
                instrument_types=["FX"],
                timeframe_range=("5m", "1h"),
                max_position_size=0.1,
                max_leverage=2.0,
                max_drawdown=0.1,
                stop_loss_type="FIXED",
                min_sharpe_ratio=1.0,
                min_win_rate=0.5,
                max_correlation_with_portfolio=0.5,
                parent_strategy_id="child_strategy",
                incompatible_strategies=[],
                config_schema={},
                default_config={},
                backtest_metrics={},
                live_metrics=None,
                created_at=1700002000,
                last_modified=1700002000,
                author="test",
                description="Grandchild strategy",
            )

            # Register strategies
            for manifest in [parent, child, grandchild]:
                strategy_path = Path(tmpdir) / f"{manifest.strategy_id}.py"
                strategy_path.write_text("# Dummy")
                registry.register_strategy(strategy_path, manifest)

            # Test lineage retrieval
            lineage = registry.get_strategy_lineage("grandchild_strategy")
            assert len(lineage) == 3
            assert lineage[0].manifest.strategy_id == "parent_strategy"
            assert lineage[1].manifest.strategy_id == "child_strategy"
            assert lineage[2].manifest.strategy_id == "grandchild_strategy"


def _db_strategy_row(
    strategy_id: str,
    *,
    timeframe_range: str | None = "1m,5m",
) -> SimpleNamespace:
    return SimpleNamespace(
        strategy_id=strategy_id,
        strategy_type=StrategyType.MOMENTUM.value,
        version="1.0.0",
        required_models=["model_a"],
        required_features=["feature_a"],
        suitable_regimes=[MarketRegime.TRENDING_UP.value],
        instrument_types=["FX"],
        timeframe_range=timeframe_range,
        max_position_size=1.0,
        max_leverage=1.0,
        max_drawdown=0.1,
        stop_loss_type="fixed",
        min_sharpe_ratio=0.5,
        min_win_rate=0.5,
        max_correlation_with_portfolio=0.8,
        parent_strategy_id=None,
        incompatible_strategies=[],
        config_schema={},
        default_config={},
        backtest_metrics={},
        live_metrics=None,
        created_at=None,
        last_modified=None,
        author="tester",
        description="db strategy",
        extra_metadata={"file_path": "/tmp/db_strategy.py"},
    )


def test_strategy_manifest_from_dict_keeps_tuple_timeframe_range() -> None:
    payload = RegistryBuilder.strategy_manifest(
        strategy_id="tuple_timeframe",
        strategy_type=StrategyType.MOMENTUM,
        version="1.0.0",
        suitable_regimes=[MarketRegime.TRENDING_UP],
        timeframe_range=("1m", "15m"),
    ).to_dict()
    payload["timeframe_range"] = ("1m", "15m")

    rebuilt = StrategyManifest.from_dict(payload)

    assert rebuilt.timeframe_range == ("1m", "15m")


def test_strategy_registry_initialization_with_explicit_json_config(tmp_path: Path) -> None:
    config = PersistenceConfig(backend=BackendType.JSON, json_path=tmp_path / "explicit_json")
    registry = StrategyRegistry(tmp_path, persistence_config=config)
    assert registry.backend == BackendType.JSON


def test_register_strategy_uses_postgres_save_path(tmp_path: Path) -> None:
    registry = StrategyRegistry(tmp_path)
    registry.backend = BackendType.POSTGRES
    registry.persistence = SimpleNamespace(log_audit=lambda **_: None)

    strategy_path = tmp_path / "postgres_strategy.py"
    _write_dummy_strategy(strategy_path)
    manifest = RegistryBuilder.strategy_manifest(
        strategy_id="postgres_save_strategy",
        strategy_type=StrategyType.TREND_FOLLOWING,
        version="1.0.0",
    )

    with patch.object(registry, "_save_strategy_to_db") as save_to_db_mock:
        strategy_id = registry.register_strategy(strategy_path, manifest)

    assert strategy_id == "postgres_save_strategy"
    save_to_db_mock.assert_called_once()


def test_postgres_query_paths_for_get_list_and_is_registered(tmp_path: Path) -> None:
    registry = StrategyRegistry(tmp_path)
    registry.backend = BackendType.POSTGRES

    session_get = MagicMock()
    session_get.query.return_value.filter_by.return_value.first.return_value = _db_strategy_row(
        "db_strategy_get",
    )
    registry.persistence = SimpleNamespace(get_session=lambda: session_get)
    fetched = registry.get_strategy("db_strategy_get")
    assert fetched is not None
    assert fetched.manifest.strategy_id == "db_strategy_get"
    session_get.close.assert_called_once()

    session_list = MagicMock()
    session_list.query.return_value.order_by.return_value.all.return_value = [
        _db_strategy_row("db_strategy_list", timeframe_range=None),
    ]
    registry.persistence = SimpleNamespace(get_session=lambda: session_list)
    listed = registry.list_strategies()
    assert len(listed) == 1
    assert listed[0].manifest.timeframe_range == ("", "")
    session_list.close.assert_called_once()

    session_is_registered_true = MagicMock()
    session_is_registered_true.query.return_value.filter_by.return_value.first.return_value = (
        _db_strategy_row("db_strategy_true")
    )
    registry.persistence = SimpleNamespace(get_session=lambda: session_is_registered_true)
    assert registry.is_registered("db_strategy_true") is True
    session_is_registered_true.close.assert_called_once()

    session_is_registered_false = MagicMock()
    session_is_registered_false.query.return_value.filter_by.return_value.first.return_value = None
    registry.persistence = SimpleNamespace(get_session=lambda: session_is_registered_false)
    assert registry.is_registered("db_strategy_missing") is False
    session_is_registered_false.close.assert_called_once()


def test_missing_strategy_paths_for_metrics_requirements_and_compatibility(tmp_path: Path) -> None:
    registry = StrategyRegistry(tmp_path)

    with pytest.raises(ValueError, match="Strategy missing not found"):
        registry.update_live_metrics("missing", {"sharpe_ratio": 1.2})

    assert registry.validate_requirements("missing", [], []) is False
    assert registry.check_compatibility("missing", ["active_strategy"]) is False


def test_json_health_snapshot_handles_manifest_errors_and_empty_registry(tmp_path: Path) -> None:
    registry = StrategyRegistry(tmp_path)
    broken_manifest_dir = registry.strategies_dir / "broken"
    broken_manifest_dir.mkdir(parents=True, exist_ok=True)
    broken_manifest_path = broken_manifest_dir / "manifest.json"
    broken_manifest_path.write_text("{invalid_json")

    registry._save_registry(
        {
            "broken_strategy": {
                "manifest_path": str(broken_manifest_path),
                "file_path": str(broken_manifest_dir / "broken.py"),
                "registered_at": 1.0,
            },
        },
    )

    count, last_modified = registry._health_snapshot()
    assert count == 1
    assert last_modified is None

    registry.registry_file.unlink()
    assert registry._load_registry() == {}

    registry.backend = BackendType.POSTGRES
    with patch.object(registry, "_json_save") as json_save_mock:
        registry._save_registry({"ignored": {}})
    json_save_mock.assert_not_called()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
