"""
Test suite for Strategy Registry using TDD and property-based testing.

This module tests the strategy registry functionality including:
- Strategy registration and retrieval
- Manifest validation
- Strategy filtering by various criteria
- Performance tracking
- Lifecycle management

"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ml.registry.strategy_registry import MarketRegime
from ml.registry.strategy_registry import StrategyManifest

# Import the modules we'll build
from ml.registry.strategy_registry import StrategyRegistry
from ml.registry.strategy_registry import StrategyType


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
def strategy_manifest_strategy(draw: st.DrawFn) -> StrategyManifest:
    """
    Generate valid StrategyManifest instances.
    """
    # Use a simple alphabet for strategy IDs
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
    strategy_id = draw(st.text(min_size=1, max_size=50, alphabet=alphabet))

    # Ensure unique strategy_id by adding a counter
    strategy_id = f"strategy_{strategy_id}_{draw(st.integers(0, 1000))}"

    return StrategyManifest(
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
                min_size=1,
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
# Test StrategyRegistry
# =================================================================================================


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
            manifest = StrategyManifest(
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
            parent = StrategyManifest(
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

            child = StrategyManifest(
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

            grandchild = StrategyManifest(
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


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
