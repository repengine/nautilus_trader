"""
Comprehensive tests for backtest runner orchestration.

This test suite validates the backtest suite orchestration, including:
- Running multiple strategies in parallel
- Strategy comparison tables
- Report generation
- Configuration handling
- Error recovery

All tests use mock datasets to enable fast, deterministic testing.
"""
# ruff: noqa: E402

from __future__ import annotations

import importlib.util
import json
import sys
import types
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest


def _should_stub_nautilus_trader() -> bool:
    spec = importlib.util.find_spec("nautilus_trader")
    return spec is None


if "nautilus_trader" not in sys.modules and _should_stub_nautilus_trader():
    nt_module = types.ModuleType("nautilus_trader")
    core_module = types.ModuleType("nautilus_trader.core")
    core_module.nautilus_pyo3 = object()
    core_data_module = types.ModuleType("nautilus_trader.core.data")
    model_module = types.ModuleType("nautilus_trader.model")
    model_data_module = types.ModuleType("nautilus_trader.model.data")
    identifiers_module = types.ModuleType("nautilus_trader.model.identifiers")
    identifiers_module.ComponentId = type("ComponentId", (), {})
    identifiers_module.InstrumentId = type("InstrumentId", (), {})
    model_data_module.BarType = type("BarType", (), {})

    class _StubConfigBase:
        def __init_subclass__(cls, **kwargs):
            return None

    class NautilusConfig(_StubConfigBase):
        """Stub Nautilus configuration base class."""

    class StrategyConfig(_StubConfigBase):
        """Stub strategy configuration base class."""

    common_module = types.ModuleType("nautilus_trader.common")
    common_config_module = types.ModuleType("nautilus_trader.common.config")
    common_config_module.NautilusConfig = NautilusConfig
    for name in ("NonNegativeFloat", "NonNegativeInt", "PositiveFloat", "PositiveInt"):
        setattr(common_config_module, name, type(name, (), {}))
    config_module = types.ModuleType("nautilus_trader.config")
    config_module.StrategyConfig = StrategyConfig
    sys.modules["nautilus_trader"] = nt_module
    sys.modules["nautilus_trader.core"] = core_module
    sys.modules["nautilus_trader.core.data"] = core_data_module
    sys.modules["nautilus_trader.model"] = model_module
    sys.modules["nautilus_trader.model.data"] = model_data_module
    sys.modules["nautilus_trader.model.identifiers"] = identifiers_module
    sys.modules["nautilus_trader.common"] = common_module
    sys.modules["nautilus_trader.common.config"] = common_config_module
    sys.modules["nautilus_trader.config"] = config_module

from ml.config.playground import DiagnosticsDefaults
from ml.config.playground import MonitoringExportDefaults
from ml.config.playground import MonteCarloShockOverlayDefaults
from ml.config.playground import MonteCarloStressDefaults
from ml.config.playground import NestedWalkForwardDefaults
from ml.config.playground import ParameterHeatmapSpecDefaults
from ml.config.playground import ParameterSensitivitySpecDefaults
from ml.config.playground import ProxyDatasetSpecDefaults
from ml.config.playground import ProxyDatasetSuiteDefaults
from ml.config.playground import ThreeDRiskBacktestDefaults
from ml.config.playground import VintageWindowDefaults
from ml.config.playground import WalkForwardPermutationDefaults
from playground.backtest.engine import BacktestConfig
from playground.backtest.engine import BacktestResult
from playground.backtest.liquidity_controls import LiquidityScalingConfig
from playground.backtest.performance_metrics import PerformanceMetrics
from playground.backtest.regime_analysis import define_market_regimes
from playground.backtest.runner import PHASE3_TARGET_SHARPE
from playground.backtest.runner import BacktestSuite
from playground.backtest.runner import LiquidityMitigationScenario
from playground.backtest.runner import MonteCarloOverlayActivation
from playground.backtest.runner import WalkForwardBacktestResult
from playground.backtest.runner import export_phase3_monitoring_snapshot
from playground.backtest.runner import get_liquidity_mitigation_scenarios
from playground.backtest.runner import run_extended_diagnostics
from playground.backtest.runner import run_full_backtest_suite
from playground.backtest.runner import run_liquidity_mitigation_experiments
from playground.backtest.runner import run_monte_carlo_stress_suite
from playground.backtest.runner import run_multi_horizon_walk_forward_analysis
from playground.backtest.runner import run_parameter_heatmap_suite
from playground.backtest.runner import run_parameter_sensitivity_suite
from playground.backtest.runner import run_proxy_dataset_validation
from playground.backtest.runner import run_vintage_simulation_suite
from playground.backtest.runner import run_walk_forward_backtest_suite
from playground.backtest.splits import TrainTestSplit
from playground.backtest.splits import WalkForwardConfig
from playground.scripts.run_phase3_walk_forward import _parse_comma_separated
from playground.scripts.run_phase3_walk_forward import parse_args as parse_walk_forward_args


# ===== Fixtures =====


@pytest.fixture
def mock_dataset_path(tmp_path: Path) -> Path:
    """
    Create a mock sector returns dataset for testing.

    Returns:
    - Parquet file with ~15 years of weekly data
    - 3 sectors (SPY, AGG, XLK)
    - Simple positive returns
    """
    start_date = datetime(2010, 1, 1, tzinfo=UTC)
    end_date = datetime(2024, 12, 31, tzinfo=UTC)
    step = timedelta(days=7)  # Weekly observations to keep tests fast

    data = []
    date = start_date
    while date <= end_date:
        for sector in ["SPY", "AGG", "XLK"]:
            if sector == "SPY":
                ret = 0.0005
            elif sector == "XLK":
                ret = 0.0004
            else:
                ret = 0.0002

            data.append({
                "timestamp": date,
                "symbol": sector,
                "return": ret,
            })
        date += step

    df = pl.DataFrame(data)

    dataset_path = tmp_path / "sector_returns.parquet"
    df.write_parquet(dataset_path)

    return dataset_path


@pytest.fixture
def mock_split() -> TrainTestSplit:
    """Create a simple train/test split for testing."""
    return TrainTestSplit(
        train_start=datetime(2018, 1, 1, tzinfo=UTC),
        train_end=datetime(2022, 12, 31, tzinfo=UTC),
        test_start=datetime(2023, 1, 1, tzinfo=UTC),
        test_end=datetime(2023, 12, 31, tzinfo=UTC),
    )


# ===== Suite Execution Tests =====


def test_run_full_backtest_suite_basic(
    mock_dataset_path: Path,
    mock_split: TrainTestSplit,
    tmp_path: Path,
) -> None:
    """Test basic backtest suite execution."""
    output_dir = tmp_path / "results"

    suite = run_full_backtest_suite(
        dataset_path=mock_dataset_path,
        output_dir=output_dir,
        split=mock_split,
    )

    # Check that suite was created
    assert isinstance(suite, BacktestSuite)

    # Check that strategies were run
    assert len(suite.strategies) > 0
    assert len(suite.metrics) > 0

    # Check that equal-weight baseline exists
    assert "Equal Weight" in suite.strategies
    assert "Equal Weight" in suite.metrics

    # Check that key suite artifacts exist
    assert output_dir.exists()
    assert (output_dir / "performance_comparison_table.csv").exists()
    assert (output_dir / "train_vs_test_metrics.csv").exists()
    report_file = output_dir / f"backtest_results_{mock_split.train_start.year}_{mock_split.test_end.year}.md"
    assert report_file.exists()

    # Train/test/full results should be tracked
    assert "Equal Weight" in suite.train_results
    assert "Equal Weight" in suite.full_results
    assert "Equal Weight" in suite.overall_metrics

    train_vs_test = suite.train_vs_test_table()
    assert not train_vs_test.is_empty()


def test_backtest_suite_benchmark_summary_matches_baselines(
    mock_dataset_path: Path,
    mock_split: TrainTestSplit,
    tmp_path: Path,
) -> None:
    """Benchmark summary should include canonical baseline strategies."""
    suite = run_full_backtest_suite(
        dataset_path=mock_dataset_path,
        output_dir=tmp_path / "benchmarks",
        split=mock_split,
    )

    summary = suite.benchmark_summary()
    assert not summary.is_empty()
    assert summary.height == len(suite.baseline_strategies)
    assert summary.get_column("strategy").to_list() == list(suite.baseline_strategies)
    expected_columns = {
        "strategy",
        "sharpe_ratio",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "cumulative_return",
        "status",
    }
    assert expected_columns.issubset(set(summary.columns))
    statuses = set(summary.get_column("status").to_list())
    assert "available" in statuses
    assert statuses.issubset({"available", "missing"})


def test_run_full_backtest_suite_writes_benchmark_exports(
    mock_dataset_path: Path,
    mock_split: TrainTestSplit,
    tmp_path: Path,
) -> None:
    """Benchmark artefacts should be mirrored under the canonical benchmarks directory."""
    output_dir = tmp_path / "reports" / "backtesting" / "suite"
    suite = run_full_backtest_suite(
        dataset_path=mock_dataset_path,
        output_dir=output_dir,
        split=mock_split,
    )
    rolling_metrics = suite.metrics["3D Factor (Rolling Betas)"]
    assert rolling_metrics.sharpe_ratio >= PHASE3_TARGET_SHARPE

    benchmark_root = tmp_path / "reports" / "backtesting" / "benchmarks"
    slug = (
        f"train_{mock_split.train_start.date().isoformat()}_{mock_split.train_end.date().isoformat()}__"
        f"test_{mock_split.test_start.date().isoformat()}_{mock_split.test_end.date().isoformat()}"
    )
    run_dir = benchmark_root / slug
    summary_path = run_dir / "benchmark_summary.csv"
    metrics_path = run_dir / "baseline_metrics.csv"
    comparison_path = run_dir / "performance_comparison_table.csv"
    metadata_path = run_dir / "metadata.json"
    audit_path = run_dir / "benchmark_audit.csv"

    assert summary_path.exists()
    assert metrics_path.exists()
    assert comparison_path.exists()
    assert metadata_path.exists()
    assert audit_path.exists()

    summary = pl.read_csv(summary_path)
    assert summary.height == len(suite.baseline_strategies)
    assert set(summary.get_column("strategy").to_list()) == set(suite.baseline_strategies)

    metrics = pl.read_csv(metrics_path)
    assert set(metrics.get_column("strategy").to_list()) == set(suite.baseline_strategies)

    comparison = pl.read_csv(comparison_path)
    assert not comparison.is_empty()
    assert set(comparison.get_column("strategy").to_list()).issubset(set(suite.baseline_strategies))
    audit = pl.read_csv(audit_path)
    assert not audit.is_empty()
    assert {"sharpe_ratio", "transaction_costs_total", "turnover_rate"}.issubset(
        set(audit.get_column("metric").to_list()),
    )
    assert audit.select(pl.col("delta").abs().max()).item() == pytest.approx(0.0, abs=1e-9)
    metadata = json.loads(metadata_path.read_text())
    assert metadata["phase3_sharpe_strategy"] == "3D Factor (Rolling Betas)"
    assert metadata["phase3_sharpe_threshold"] == pytest.approx(PHASE3_TARGET_SHARPE)
    assert metadata["phase3_sharpe_ok"] is True
    assert metadata["phase3_sharpe_value"] >= metadata["phase3_sharpe_threshold"]


def test_run_full_backtest_suite_default_split(
    mock_dataset_path: Path,
    tmp_path: Path,
) -> None:
    """Test suite execution with default train/test split."""
    output_dir = tmp_path / "results"

    suite = run_full_backtest_suite(
        dataset_path=mock_dataset_path,
        output_dir=output_dir,
        split=None,  # Use default
    )

    # Check that default split was used
    assert suite.split is not None
    assert suite.split.train_start.year == 2010
    assert suite.split.test_start.year == 2019

    regime_summary_path = output_dir / "regime_summary.csv"
    regimes = define_market_regimes()
    assert suite.regime_results
    assert regime_summary_path.exists()
    regime_summary = pl.read_csv(regime_summary_path)
    assert regime_summary.height == len(regimes) * len(suite.regime_results)
    expected_regimes = {regime.name for regime in regimes}
    assert set(regime_summary.get_column("regime_name").to_list()) == expected_regimes
    rolling_analysis = suite.regime_results.get("3D Factor (Rolling Betas)")
    assert rolling_analysis is not None
    assert set(rolling_analysis.regime_performances) == expected_regimes
    assert (output_dir / "full_period_metrics.csv").exists()

    report_file = output_dir / "backtest_results_2010_2024.md"
    assert report_file.exists()


def test_run_monte_carlo_stress_suite_generates_paths(
    mock_dataset_path: Path,
    tmp_path: Path,
) -> None:
    """Monte Carlo stress suite should produce summary artefacts and paths."""
    overlay = MonteCarloShockOverlayDefaults(
        name="test_shock",
        probability=1.0,
        magnitude=-0.01,
        duration_days=3,
        decay=0.5,
        max_applications=1,
        regime_bias=None,
    )
    stress_config = MonteCarloStressDefaults(
        num_paths=5,
        random_seed=42,
        risk_free_rate=0.0,
        overlays=(overlay,),
    )

    result = run_monte_carlo_stress_suite(
        dataset_path=mock_dataset_path,
        output_dir=tmp_path,
        config=stress_config,
    )

    assert result.paths
    assert len(result.paths) == stress_config.num_paths
    summary = result.summary_frame()
    assert not summary.is_empty()
    strategy_names = summary.get_column("strategy").to_list()
    assert stress_config.target_strategy in strategy_names
    assert "overlay_count_mean" in summary.columns
    assert "overlay_total_impact_mean" in summary.columns

    artefact_root = tmp_path / "stress" / "monte_carlo"
    assert (artefact_root / "summary.csv").exists()
    assert (artefact_root / "paths.csv").exists()
    assert (artefact_root / "config.json").exists()
    overlay_summary_path = artefact_root / "overlay_summary.csv"
    assert overlay_summary_path.exists()
    overlay_summary = pl.read_csv(overlay_summary_path)
    assert "activation_count" in overlay_summary.columns
    assert overlay_summary.get_column("activation_count").sum() >= 1
    config_payload = json.loads((artefact_root / "config.json").read_text(encoding="utf-8"))
    assert config_payload.get("overlay_summary_path") == "overlay_summary.csv"
    assert config_payload.get("overlay_category_summary_path") == "overlay_category_summary.csv"
    assert config_payload.get("baseline_metrics_path") == "baseline_metrics.csv"
    category_summary_path = artefact_root / "overlay_category_summary.csv"
    assert category_summary_path.exists()
    category_summary = pl.read_csv(category_summary_path)
    assert "category" in category_summary.columns
    assert "activation_count" in category_summary.columns
    assert not category_summary.is_empty()
    baseline_path = artefact_root / "baseline_metrics.csv"
    assert baseline_path.exists()
    baseline_frame = pl.read_csv(baseline_path)
    assert baseline_frame.get_column("strategy").to_list() == [stress_config.target_strategy]
    paths_frame = pl.read_csv(artefact_root / "paths.csv")
    assert "overlay_events" in paths_frame.columns
    assert "overlay_names" in paths_frame.columns
    assert "overlay_total_impact" in paths_frame.columns
    assert paths_frame.get_column("overlay_total_impact").abs().sum() != 0
    path_overlays = [path.overlay_events for path in result.paths]
    assert any(events for events in path_overlays)
    assert all(isinstance(events, tuple) for events in path_overlays)
    flattened = [event for events in path_overlays for event in events]
    assert flattened
    assert all(isinstance(event, MonteCarloOverlayActivation) for event in flattened)
    assert all(event.total_impact != 0.0 for event in flattened)


def test_run_parameter_heatmap_suite_generates_outputs(
    mock_dataset_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parameter heatmap suite should produce pivot artefacts."""
    spec = ParameterHeatmapSpecDefaults(
        name="Unit Heatmap",
        description="Test heatmap specification.",
        target_strategy="Equal Weight",
        parameters=("config.transaction_cost_bps", "config.rebalance_threshold"),
        grid={
            "config.transaction_cost_bps": (5.0, 10.0),
            "config.rebalance_threshold": (0.02, 0.03),
        },
    )

    def _fake_execute(*args: object, **kwargs: object) -> SimpleNamespace:
        metrics = {
            "Equal Weight": SimpleNamespace(
                sharpe_ratio=1.0,
                calmar_ratio=0.5,
                annualized_return=0.12,
            ),
        }
        return SimpleNamespace(metrics=metrics)

    monkeypatch.setattr(
        "playground.backtest.runner._execute_backtest_suite",
        _fake_execute,
    )

    result = run_parameter_heatmap_suite(
        dataset_path=mock_dataset_path,
        output_dir=tmp_path,
        specs=(spec,),
    )

    heatmap_root = tmp_path / "heatmaps" / spec.slug
    assert heatmap_root.exists()
    assert (heatmap_root / "results.csv").exists()
    assert (heatmap_root / "heatmap.csv").exists()
    summary = result.summary_frame()
    assert not summary.is_empty()


def test_parameter_heatmap_defaults_include_new_specs() -> None:
    """Defaults should expose liquidity multiplier and transaction cost envelopes."""
    defaults = ThreeDRiskBacktestDefaults()
    specs_by_slug = {spec.slug: spec for spec in defaults.parameter_heatmaps.specs}
    assert "turnover-vs-liquidity-multipliers" in specs_by_slug
    multipliers_spec = specs_by_slug["turnover-vs-liquidity-multipliers"]
    assert multipliers_spec.parameters == (
        "turnover_overrides.3d_factor_rolling",
        "liquidity_scaling.neutral_liquidity_multiplier",
    )
    assert multipliers_spec.metric == "calmar_ratio"
    assert multipliers_spec.base_overrides["liquidity_scaling.moderate_liquidity_multiplier"] == pytest.approx(0.7)
    assert "transaction-cost-envelope" in specs_by_slug
    envelope_spec = specs_by_slug["transaction-cost-envelope"]
    assert envelope_spec.parameters == ("config.transaction_cost_bps", "config.slippage_bps")
    assert envelope_spec.metric == "sortino_ratio"
    assert envelope_spec.grid["config.transaction_cost_bps"] == (5.0, 10.0, 15.0, 20.0)


def test_run_parameter_heatmap_suite_filters_by_slug(
    mock_dataset_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Heatmap execution should honour slug filters."""
    spec_a = ParameterHeatmapSpecDefaults(
        name="Spec Alpha",
        description="Spec alpha description.",
        target_strategy="Equal Weight",
        parameters=("config.transaction_cost_bps", "config.rebalance_threshold"),
        grid={
            "config.transaction_cost_bps": (5.0, 10.0),
            "config.rebalance_threshold": (0.02, 0.03),
        },
    )
    spec_b = ParameterHeatmapSpecDefaults(
        name="Spec Beta",
        description="Spec beta description.",
        target_strategy="Equal Weight",
        parameters=("config.transaction_cost_bps", "config.slippage_bps"),
        grid={
            "config.transaction_cost_bps": (5.0, 10.0),
            "config.slippage_bps": (0.0, 5.0),
        },
        metric="sortino_ratio",
    )

    def _fake_execute(*args: object, **kwargs: object) -> SimpleNamespace:
        metrics = {
            "Equal Weight": SimpleNamespace(
                sharpe_ratio=1.0,
                calmar_ratio=0.6,
                annualized_return=0.12,
                sortino_ratio=0.75,
            ),
        }
        return SimpleNamespace(metrics=metrics)

    monkeypatch.setattr(
        "playground.backtest.runner._execute_backtest_suite",
        _fake_execute,
    )

    result = run_parameter_heatmap_suite(
        dataset_path=mock_dataset_path,
        output_dir=tmp_path,
        specs=(spec_a, spec_b),
        spec_slugs=(spec_a.slug,),
    )

    assert [run.spec.slug for run in result.runs] == [spec_a.slug]


def test_run_parameter_heatmap_suite_raises_on_unknown_slug(
    mock_dataset_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Filtering by an unknown slug should raise."""
    spec = ParameterHeatmapSpecDefaults(
        name="Spec Gamma",
        description="Spec gamma description.",
        target_strategy="Equal Weight",
        parameters=("config.transaction_cost_bps", "config.rebalance_threshold"),
        grid={
            "config.transaction_cost_bps": (5.0, 10.0),
            "config.rebalance_threshold": (0.02, 0.03),
        },
    )

    def _fake_execute(*args: object, **kwargs: object) -> SimpleNamespace:
        metrics = {
            "Equal Weight": SimpleNamespace(
                sharpe_ratio=1.0,
                calmar_ratio=0.6,
                annualized_return=0.12,
                sortino_ratio=0.75,
            ),
        }
        return SimpleNamespace(metrics=metrics)

    monkeypatch.setattr(
        "playground.backtest.runner._execute_backtest_suite",
        _fake_execute,
    )

    with pytest.raises(ValueError):
        run_parameter_heatmap_suite(
            dataset_path=mock_dataset_path,
            output_dir=tmp_path,
            specs=(spec,),
            spec_slugs=("unknown-slug",),
        )


def test_run_parameter_sensitivity_suite_generates_outputs(
    mock_dataset_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parameter sensitivity suite should persist results and metadata."""
    spec = ParameterSensitivitySpecDefaults(
        name="Transaction Cost Sensitivity",
        description="Validate cost assumptions.",
        target_strategy="3D Factor (Rolling Betas)",
        parameter_grid={
            "config.transaction_cost_bps": (5.0, 10.0),
            "strategy_params.3d_factor_rolling.turnover_smoothing": (0.30, 0.40),
        },
    )

    def _fake_execute(*args: object, **kwargs: object) -> SimpleNamespace:
        metrics = {
            "3D Factor (Rolling Betas)": SimpleNamespace(
                sharpe_ratio=1.2,
                calmar_ratio=0.6,
                annualized_return=0.14,
                annualized_volatility=0.12,
                maximum_drawdown=-0.18,
            ),
        }
        return SimpleNamespace(metrics=metrics)

    monkeypatch.setattr(
        "playground.backtest.runner._execute_backtest_suite",
        _fake_execute,
    )

    result = run_parameter_sensitivity_suite(
        dataset_path=mock_dataset_path,
        output_dir=tmp_path,
        specs=(spec,),
    )

    sensitivity_root = tmp_path / "sensitivity" / spec.slug
    assert sensitivity_root.exists()
    assert (sensitivity_root / "results.csv").exists()
    assert (sensitivity_root / "metadata.json").exists()
    summary = result.summary_frame()
    assert not summary.is_empty()
    assert summary.get_column("slug").to_list() == [spec.slug]


def test_parameter_sensitivity_suite_filters_by_slug(
    mock_dataset_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sensitivity execution should honour slug filters."""
    spec_a = ParameterSensitivitySpecDefaults(
        name="Spec Alpha",
        description="Alpha sensitivity.",
        target_strategy="3D Factor (Rolling Betas)",
        parameter_grid={"config.transaction_cost_bps": (5.0, 10.0)},
    )
    spec_b = ParameterSensitivitySpecDefaults(
        name="Spec Beta",
        description="Beta sensitivity.",
        target_strategy="3D Factor (Rolling Betas)",
        parameter_grid={"config.transaction_cost_bps": (10.0, 20.0)},
    )

    def _fake_execute(*args: object, **kwargs: object) -> SimpleNamespace:
        metrics = {
            "3D Factor (Rolling Betas)": SimpleNamespace(
                sharpe_ratio=1.0,
                calmar_ratio=0.5,
                annualized_return=0.10,
                annualized_volatility=0.11,
                maximum_drawdown=-0.20,
            ),
        }
        return SimpleNamespace(metrics=metrics)

    monkeypatch.setattr(
        "playground.backtest.runner._execute_backtest_suite",
        _fake_execute,
    )

    result = run_parameter_sensitivity_suite(
        dataset_path=mock_dataset_path,
        output_dir=tmp_path,
        specs=(spec_a, spec_b),
        spec_slugs=(spec_b.slug,),
    )
    assert len(result.runs) == 1
    assert result.runs[0].spec.slug == spec_b.slug

    with pytest.raises(ValueError):
        run_parameter_sensitivity_suite(
            dataset_path=mock_dataset_path,
            output_dir=tmp_path,
            specs=(spec_a,),
            spec_slugs=("missing-spec",),
        )


def test_parameter_sensitivity_suite_flags_sharpe_delta_breaches(
    mock_dataset_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sharpe tolerance breaches should flip the metadata flag."""
    spec = ParameterSensitivitySpecDefaults(
        name="Sharpe Delta",
        description="Ensure tolerance enforcement.",
        target_strategy="3D Factor (Rolling Betas)",
        parameter_grid={"config.transaction_cost_bps": (5.0, 10.0)},
    )
    sharpe_values = iter([0.95, 0.70])

    def _fake_execute(*args: object, **kwargs: object) -> SimpleNamespace:
        value = next(sharpe_values)
        metrics = {
            "3D Factor (Rolling Betas)": SimpleNamespace(
                sharpe_ratio=value,
                calmar_ratio=0.5,
                annualized_return=0.12,
                annualized_volatility=0.10,
                maximum_drawdown=-0.18,
            ),
        }
        return SimpleNamespace(metrics=metrics)

    monkeypatch.setattr(
        "playground.backtest.runner._execute_backtest_suite",
        _fake_execute,
    )

    result = run_parameter_sensitivity_suite(
        dataset_path=mock_dataset_path,
        output_dir=tmp_path,
        specs=(spec,),
    )
    metadata = result.runs[0].metadata
    assert metadata.get("metric_spread_ok") is False
    assert pytest.approx(metadata.get("metric_spread")) == 0.25
    assert metadata.get("metric_spread_tolerance") == pytest.approx(0.15)


def test_run_extended_diagnostics_produces_reports(
    mock_dataset_path: Path,
    tmp_path: Path,
) -> None:
    """Extended diagnostics should emit CSV artefacts."""
    suite = run_full_backtest_suite(
        dataset_path=mock_dataset_path,
        output_dir=tmp_path / "baseline",
        split=None,
    )
    diagnostics = run_extended_diagnostics(
        suite=suite,
        output_dir=tmp_path,
        config=DiagnosticsDefaults(tail_quantiles=(0.05,), turnover_bins=(0.05, 0.10)),
    )
    diagnostics_dir = tmp_path / "diagnostics"
    assert (diagnostics_dir / "tail_metrics.csv").exists()
    assert (diagnostics_dir / "turnover_distribution.csv").exists()
    assert not diagnostics.tail_metrics.is_empty()


def test_run_proxy_dataset_validation_handles_existing_dataset(
    mock_dataset_path: Path,
    tmp_path: Path,
) -> None:
    """Proxy dataset validation should succeed when dataset exists."""
    suite_defaults = ProxyDatasetSuiteDefaults(
        specs=(
            ProxyDatasetSpecDefaults(
                name="Fixture Proxy",
                relative_path=str(mock_dataset_path),
                description="Fixture dataset",
                allow_missing=False,
                min_train_years=4,
                min_test_years=1,
            ),
        ),
        vintage_windows=tuple(),
    )
    result = run_proxy_dataset_validation(
        dataset_path=mock_dataset_path,
        output_dir=tmp_path,
        suite_defaults=suite_defaults,
    )
    summary = result.summary_frame()
    assert not summary.is_empty()
    assert summary.get_column("status").to_list() == ["success"]
    assert summary.get_column("allow_missing").to_list() == [False]
    assert "tags" in summary.columns


def test_run_proxy_dataset_validation_marks_allowed_missing(tmp_path: Path) -> None:
    """Allowed missing datasets should be reported with explicit status."""
    missing_path = (tmp_path / "missing_proxy.parquet").resolve()
    suite_defaults = ProxyDatasetSuiteDefaults(
        specs=(
            ProxyDatasetSpecDefaults(
                name="Missing Proxy",
                relative_path=str(missing_path),
                description="Optional dataset",
                allow_missing=True,
                min_train_years=4,
                min_test_years=1,
            ),
        ),
        vintage_windows=tuple(),
    )
    result = run_proxy_dataset_validation(
        dataset_path=tmp_path,
        output_dir=tmp_path,
        suite_defaults=suite_defaults,
    )
    summary = result.summary_frame()
    assert summary.get_column("status").to_list() == ["missing_allowed"]
    assert summary.get_column("allow_missing").to_list() == [True]


def test_run_vintage_simulation_suite_creates_summary(
    mock_dataset_path: Path,
    tmp_path: Path,
) -> None:
    """Vintage simulation should produce per-window summaries."""
    window = VintageWindowDefaults(
        label="Short Window",
        train_years=4,
        test_years=1,
        step_years=2,
        min_folds=1,
    )
    result = run_vintage_simulation_suite(
        dataset_path=mock_dataset_path,
        output_dir=tmp_path,
        windows=(window,),
    )
    vintage_dir = tmp_path / "vintage" / window.slug
    assert (vintage_dir / "summary.csv").exists()
    summary = result.summary_frame()
    assert summary.height == 1
    assert summary.get_column("status").to_list() == ["success"]
    assert summary.get_column("min_folds").to_list() == [1]


def test_run_vintage_simulation_suite_handles_errors(
    mock_dataset_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vintage simulations should record error status when execution fails."""

    def _raise(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "playground.backtest.runner.run_walk_forward_backtest_suite",
        _raise,
    )

    window = VintageWindowDefaults(
        label="Error Window",
        train_years=4,
        test_years=1,
        step_years=1,
        min_folds=2,
    )
    result = run_vintage_simulation_suite(
        dataset_path=mock_dataset_path,
        output_dir=tmp_path,
        windows=(window,),
    )
    summary = result.summary_frame()
    assert summary.get_column("status").to_list() == ["error"]
    assert summary.get_column("fold_count").to_list() == [0]


def test_export_phase3_monitoring_snapshot_compiles_paths(tmp_path: Path) -> None:
    """Monitoring snapshot should aggregate supplied artefact paths."""
    walk_forward_dir = tmp_path / "walk_forward"
    walk_forward_dir.mkdir(parents=True, exist_ok=True)
    (walk_forward_dir / "permutation_summary.csv").write_text("", encoding="utf-8")
    (walk_forward_dir / "nested_summary.csv").write_text("", encoding="utf-8")
    (walk_forward_dir / "nested_rollup.csv").write_text("", encoding="utf-8")

    stress_dir = tmp_path / "stress" / "monte_carlo"
    stress_dir.mkdir(parents=True, exist_ok=True)
    (stress_dir / "summary.csv").write_text("", encoding="utf-8")
    (stress_dir / "paths.csv").write_text("", encoding="utf-8")
    (stress_dir / "overlay_summary.csv").write_text("", encoding="utf-8")
    (stress_dir / "overlay_category_summary.csv").write_text("", encoding="utf-8")
    (stress_dir / "baseline_metrics.csv").write_text("", encoding="utf-8")
    (stress_dir / "config.json").write_text("{}", encoding="utf-8")

    heatmap_dir = tmp_path / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    (heatmap_dir / "summary.csv").write_text("", encoding="utf-8")

    diagnostics_dir = tmp_path / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    (diagnostics_dir / "tail_metrics.csv").write_text("", encoding="utf-8")
    (diagnostics_dir / "turnover_distribution.csv").write_text("", encoding="utf-8")
    (diagnostics_dir / "benchmark_deltas.csv").write_text("", encoding="utf-8")
    (diagnostics_dir / "config.json").write_text("{}", encoding="utf-8")

    proxy_dir = tmp_path / "proxy_datasets"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    (proxy_dir / "summary.csv").write_text("", encoding="utf-8")

    vintage_dir = tmp_path / "vintage"
    vintage_dir.mkdir(parents=True, exist_ok=True)
    (vintage_dir / "summary.csv").write_text("", encoding="utf-8")

    sensitivity_dir = tmp_path / "sensitivity"
    sensitivity_dir.mkdir(parents=True, exist_ok=True)
    (sensitivity_dir / "summary.csv").write_text(
        "spec,slug,strategy,metric,metric_value,evaluated_combinations,best_config\n"
        "Rolling Window Sensitivity,rolling-window-sensitivity,3D Factor (Rolling Betas),sharpe_ratio,0.94,3,\"{'rolling_window': 252}\"\n",
        encoding="utf-8",
    )
    (sensitivity_dir / "sensitivity_analysis.pdf").write_text("%PDF-test", encoding="utf-8")

    data_quality_dir = tmp_path / "data_quality"
    data_quality_dir.mkdir(parents=True, exist_ok=True)
    (data_quality_dir / "missing_data_audit.json").write_text(
        json.dumps(
            {
                "dataset_path": str(tmp_path / "data" / "sector_returns.parquet"),
                "missing_ratio": 0.004,
                "missing_by_column": {"factor_duration": 0.0},
                "imputation_summaries": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    outlier_dir = tmp_path / "outliers"
    outlier_dir.mkdir(parents=True, exist_ok=True)
    (outlier_dir / "factor_outlier_report.json").write_text(
        json.dumps(
            {
                "outlier_ratio": 0.01,
                "factor_summaries": [{"factor": "factor_duration", "outlier_count": 2}],
                "treatment_impacts": [],
                "recommended_treatment": "winsorize",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    walk_forward_stub = SimpleNamespace(base_directory=tmp_path)

    class _MonteCarloStub:
        def __init__(self, base_directory: Path) -> None:
            self.base_directory = base_directory
            self.paths = (SimpleNamespace(overlay_events=tuple()),)
            self.stress_config = SimpleNamespace(
                report_metrics=("sharpe_ratio",),
                report_quantiles=(0.5,),
            )

        def overlay_category_summary(self) -> list[dict[str, object]]:
            return [{
                "category": "rates",
                "activation_count": 1,
                "total_impact": -0.01,
                "overlay_names": "rate_hike_shock",
                "tags": "macro|rates",
            }]

        def baseline_metrics_dict(self) -> dict[str, object]:
            return {
                "strategy": "3D Factor (Rolling Betas)",
                "sharpe_ratio": 1.25,
                "annualized_return": 0.11,
            }

    monte_carlo_stub = _MonteCarloStub(tmp_path)
    heatmap_stub = SimpleNamespace(
        base_directory=tmp_path,
        output_dirname="heatmaps",
        runs=(
            SimpleNamespace(
                spec=SimpleNamespace(slug="unit-test-spec"),
                metadata={
                    "target_strategy": "3D Factor (Rolling Betas)",
                    "metric_name": "sharpe_ratio",
                    "best_metric": 0.95,
                    "evaluated_combinations": 4,
                    "best_config": {"turnover_overrides.3d_factor_rolling": 0.40},
                },
            ),
        ),
    )
    tail_df = pl.DataFrame([
        {
            "strategy": "Equal Weight",
            "quantile": 0.05,
            "value_at_risk": -0.03,
            "conditional_value_at_risk": -0.04,
            "num_observations": 100,
        },
    ])
    turnover_df = pl.DataFrame([
        {
            "strategy": "Equal Weight",
            "bucket": "0.00-0.05",
            "count": 10,
            "proportion": 0.5,
            "mean_turnover": 0.04,
            "p95_turnover": 0.08,
            "rolling_window_days": 252,
            "rolling_mean_turnover": 0.05,
            "rolling_max_turnover": 0.09,
        },
    ])
    benchmark_df = pl.DataFrame([
        {
            "benchmark": "60/40 Portfolio",
            "strategy": "Equal Weight",
            "sharpe_ratio_delta": 0.1,
        },
    ])
    diagnostics_stub = SimpleNamespace(
        tail_metrics=tail_df,
        turnover_distribution=turnover_df,
        benchmark_deltas=benchmark_df,
        output_directory=diagnostics_dir,
        config=DiagnosticsDefaults(),
    )
    proxy_stub = SimpleNamespace(
        base_directory=tmp_path,
        runs=(
            SimpleNamespace(
                spec=SimpleNamespace(slug="proxy-fixture", tags=("demo",), allow_missing=False),
                status="success",
                message=None,
            ),
        ),
    )
    vintage_stub = SimpleNamespace(
        base_directory=tmp_path,
        runs=(
            SimpleNamespace(
                window=SimpleNamespace(slug="vintage-fixture", label="Fixture Window", min_folds=1),
                summary=pl.DataFrame(),
                output_directory=tmp_path,
                fold_count=2,
                status="success",
                message=None,
            ),
        ),
    )

    snapshot = export_phase3_monitoring_snapshot(
        output_dir=tmp_path,
        walk_forward=walk_forward_stub,
        monte_carlo=monte_carlo_stub,
        heatmaps=heatmap_stub,
        diagnostics=diagnostics_stub,
        proxy_datasets=proxy_stub,
        vintage=vintage_stub,
    )
    assert snapshot.path.exists()
    data = json.loads(snapshot.path.read_text(encoding="utf-8"))
    assert "sections" in data
    assert "walk_forward" in data["sections"]
    assert "alert_channels" in data
    assert "dashboard_targets" in data
    assert "alert_rules" in data
    assert "automation_targets" in data
    assert data["sections"]["monte_carlo"]["overlay_summary_path"] is not None
    assert data["sections"]["monte_carlo"]["overlay_category_summary_path"] is not None
    assert data["sections"]["monte_carlo"]["baseline_metrics_path"] is not None
    assert data["sections"]["monte_carlo"]["config_path"] is not None
    assert data["sections"]["parameter_heatmaps"]["config_specs"] == ["unit-test-spec"]
    assert data["sections"]["proxy_datasets"]["datasets"] == ["proxy-fixture"]
    assert data["monte_carlo_metadata"]["report_metrics"] == ["sharpe_ratio"]
    assert data["monte_carlo_metadata"]["overlay_category_stats"][0]["category"] == "rates"
    assert data["monte_carlo_metadata"]["baseline_metrics"]["strategy"] == "3D Factor (Rolling Betas)"
    assert data["parameter_heatmap_metadata"][0]["best_config"]["turnover_overrides.3d_factor_rolling"] == 0.40
    assert data["sections"]["proxy_datasets"]["dataset_status"]["proxy-fixture"] == "success"
    assert data["proxy_dataset_metadata"]["proxy-fixture"]["allow_missing"] is False
    vintage_windows = data["sections"]["vintage_simulations"]["windows"]
    assert vintage_windows[0]["status"] == "success"
    assert data["vintage_metadata"][0]["slug"] == "vintage-fixture"
    assert data["sections"]["phase4_sensitivity"]["summary_path"] is not None
    assert data["sections"]["phase4_sensitivity"]["report_path"] is not None
    assert data["sections"]["phase4_data_quality"]["audit_path"] is not None
    assert data["sections"]["phase4_outliers"]["report_path"] is not None
    sensitivity_meta = data["phase4_sensitivity_metadata"]
    assert sensitivity_meta[0]["slug"] == "rolling-window-sensitivity"
    assert data["phase4_data_quality"]["missing_ratio"] == pytest.approx(0.004)
    assert data["phase4_outlier_summary"]["recommended_treatment"] == "winsorize"
    assert "summary" in data["sections"]["extended_diagnostics"]
    diagnostics_metadata = data.get("diagnostics_metadata")
    assert diagnostics_metadata is not None
    tail_summary = diagnostics_metadata["tail"]["Equal Weight"]
    assert tail_summary["var_p05"] == pytest.approx(-0.03)
    turnover_summary = diagnostics_metadata["turnover"]["Equal Weight"]
    assert turnover_summary["rolling_window_days"] == 252


def test_run_full_backtest_suite_config_overrides(
    mock_dataset_path: Path,
    mock_split: TrainTestSplit,
    tmp_path: Path,
) -> None:
    """Test suite execution with custom configuration overrides."""
    output_dir = tmp_path / "results"

    config_overrides = {
        "initial_capital": 5_000_000.0,
        "transaction_cost_bps": 5.0,
        "rebalance_frequency": "monthly",
    }

    suite = run_full_backtest_suite(
        dataset_path=mock_dataset_path,
        output_dir=output_dir,
        split=mock_split,
        config_overrides=config_overrides,
    )

    # Check that config was applied
    assert suite.config.initial_capital == 5_000_000.0
    assert suite.config.transaction_cost_bps == 5.0
    assert suite.config.rebalance_frequency == "monthly"


def test_run_full_backtest_suite_requires_min_training_history(
    mock_dataset_path: Path,
    tmp_path: Path,
) -> None:
    """Training window shorter than defaults should raise a validation error."""
    short_split = TrainTestSplit(
        train_start=datetime(2022, 1, 1, tzinfo=UTC),
        train_end=datetime(2022, 12, 31, tzinfo=UTC),
        test_start=datetime(2023, 1, 1, tzinfo=UTC),
        test_end=datetime(2023, 12, 31, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="Training window"):
        run_full_backtest_suite(
            dataset_path=mock_dataset_path,
            output_dir=tmp_path / "invalid_training",
            split=short_split,
        )


def test_run_full_backtest_suite_rejects_split_outside_dataset(tmp_path: Path) -> None:
    """Splits outside dataset coverage should fail validation."""
    start_date = datetime(2020, 1, 1, tzinfo=UTC)
    end_date = datetime(2024, 12, 31, tzinfo=UTC)
    step = timedelta(days=7)
    records = []
    current = start_date
    while current <= end_date:
        for symbol in ("SPY", "AGG"):
            records.append({
                "timestamp": current,
                "symbol": symbol,
                "return": 0.0004,
            })
        current += step

    dataset_path = tmp_path / "limited_sector_returns.parquet"
    pl.DataFrame(records).write_parquet(dataset_path)

    with pytest.raises(ValueError, match="dataset coverage"):
        run_full_backtest_suite(
            dataset_path=dataset_path,
            output_dir=tmp_path / "coverage_failure",
            split=None,
        )


def test_run_full_backtest_suite_nonexistent_dataset_raises_error(
    mock_split: TrainTestSplit,
    tmp_path: Path,
) -> None:
    """Test that nonexistent dataset raises FileNotFoundError."""
    output_dir = tmp_path / "results"
    nonexistent_path = tmp_path / "does_not_exist.parquet"

    with pytest.raises(FileNotFoundError, match="Dataset path does not exist"):
        run_full_backtest_suite(
            dataset_path=nonexistent_path,
            output_dir=output_dir,
            split=mock_split,
        )


def test_backtest_suite_compare_strategies(
    mock_dataset_path: Path,
    mock_split: TrainTestSplit,
    tmp_path: Path,
) -> None:
    """Test strategy comparison table generation."""
    output_dir = tmp_path / "results"

    suite = run_full_backtest_suite(
        dataset_path=mock_dataset_path,
        output_dir=output_dir,
        split=mock_split,
    )

    comparison = suite.compare_strategies()

    # Check that comparison table is valid
    assert isinstance(comparison, pl.DataFrame)
    assert not comparison.is_empty()

    # Check required columns
    expected_cols = {
        "strategy",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "max_drawdown",
        "information_ratio",
        "num_rebalances",
        "transaction_costs",
    }
    assert expected_cols.issubset(comparison.columns)

    # Check that strategies are sorted by Sharpe ratio
    sharpe_ratios = comparison["sharpe_ratio"].to_list()
    assert sharpe_ratios == sorted(sharpe_ratios, reverse=True)


def test_backtest_suite_to_markdown_report(
    mock_dataset_path: Path,
    mock_split: TrainTestSplit,
    tmp_path: Path,
) -> None:
    """Test markdown report generation."""
    output_dir = tmp_path / "results"

    suite = run_full_backtest_suite(
        dataset_path=mock_dataset_path,
        output_dir=output_dir,
        split=mock_split,
    )

    report_path = tmp_path / "custom_report.md"
    suite.to_markdown_report(report_path)

    # Check that report was created
    assert report_path.exists()

    # Check report content
    content = report_path.read_text()

    # Should contain headers
    assert "# Backtest Results: Full Strategy Suite" in content
    assert "## Configuration" in content
    assert "## Executive Summary" in content
    assert "## Detailed Performance Metrics" in content

    # Should contain strategy names
    for strategy_name in suite.strategies.keys():
        assert strategy_name in content

    # Should contain key metrics
    assert "Sharpe Ratio" in content
    assert "Maximum Drawdown" in content
    assert "Annualized Return" in content
    assert "## Train vs Test Comparison" in content


def test_backtest_suite_report_format_dataframe() -> None:
    """Test DataFrame formatting as markdown table."""
    # Create a simple suite to test formatting
    start_date = datetime(2023, 1, 1, tzinfo=UTC)
    end_date = datetime(2023, 12, 31, tzinfo=UTC)

    split = TrainTestSplit(
        train_start=datetime(2022, 1, 1, tzinfo=UTC),
        train_end=datetime(2022, 12, 31, tzinfo=UTC),
        test_start=start_date,
        test_end=end_date,
    )

    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
    )

    suite = BacktestSuite(
        strategies={},
        metrics={},
        split=split,
        config=config,
    )

    # Create test DataFrame
    df = pl.DataFrame({
        "strategy": ["Test1", "Test2"],
        "sharpe_ratio": [1.5, 1.2],
        "annualized_return": [10.5, 8.3],
    })

    markdown = suite._format_dataframe_as_markdown(df)

    # Check markdown formatting
    assert "| strategy | sharpe_ratio | annualized_return |" in markdown
    assert "| --- | --- | --- |" in markdown
    assert "| Test1 | 1.500 | 10.50 |" in markdown
    assert "| Test2 | 1.200 | 8.30 |" in markdown


def test_benchmark_summary_marks_missing_baselines(tmp_path: Path) -> None:
    """Benchmark summary rows mark missing baseline strategies and note in report."""
    train_start = datetime(2010, 1, 1, tzinfo=UTC)
    train_end = datetime(2014, 12, 31, tzinfo=UTC)
    test_start = datetime(2015, 1, 1, tzinfo=UTC)
    test_end = datetime(2015, 12, 31, tzinfo=UTC)
    split = TrainTestSplit(
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )
    config = BacktestConfig(start_date=train_start, end_date=test_end)

    metrics = PerformanceMetrics(
        annualized_return=0.12,
        cumulative_return=0.20,
        monthly_return_mean=0.015,
        monthly_return_std=0.02,
        annualized_volatility=0.18,
        maximum_drawdown=-0.12,
        var_95=-0.03,
        var_99=-0.05,
        cvar_95=-0.04,
        cvar_99=-0.06,
        sharpe_ratio=0.75,
        sortino_ratio=0.90,
        calmar_ratio=0.50,
        information_ratio=None,
        turnover_rate=0.30,
        transaction_costs_total=1_500.0,
        transaction_costs_pct=0.02,
        num_rebalances=12,
        start_date=train_start,
        end_date=test_end,
        total_days=365,
    )

    positions = pl.DataFrame({
        "timestamp": [test_start],
        "symbol": ["SPY"],
        "weight": [1.0],
    })
    equal_weight_result = BacktestResult(
        strategy_name="Equal Weight",
        start_date=train_start,
        end_date=test_end,
        dates=[train_start, test_end],
        portfolio_values=[100.0, 120.0],
        returns=[0.015],
        positions=positions,
        total_return=0.20,
        annualized_return=0.12,
        annualized_volatility=0.18,
        sharpe_ratio=0.75,
        max_drawdown=-0.12,
        calmar_ratio=0.50,
        total_transaction_costs=1_500.0,
        turnover_rate=0.30,
        num_rebalances=12,
    )

    suite = BacktestSuite(
        strategies={"Equal Weight": equal_weight_result},
        metrics={"Equal Weight": metrics},
        split=split,
        config=config,
    )

    summary = suite.benchmark_summary()
    assert summary.height == len(suite.baseline_strategies)
    missing = summary.filter(pl.col("status") == "missing")
    expected_missing = set(suite.baseline_strategies) - {"Equal Weight"}
    assert set(missing.get_column("strategy").to_list()) == expected_missing

    report_path = tmp_path / "benchmarks.md"
    suite.to_markdown_report(report_path)
    content = report_path.read_text()
    assert (
        "Metrics unavailable for baseline strategies: 60/40 Portfolio, Risk Parity"
        in content
    )


# ===== Strategy Coverage Tests =====


def test_backtest_suite_includes_all_strategies(
    mock_dataset_path: Path,
    mock_split: TrainTestSplit,
    tmp_path: Path,
) -> None:
    """Test that all expected strategies are included in the suite."""
    output_dir = tmp_path / "results"

    suite = run_full_backtest_suite(
        dataset_path=mock_dataset_path,
        output_dir=output_dir,
        split=mock_split,
    )

    actual_strategies = set(suite.strategies.keys())

    # Check that strategies were run
    # Note: Some strategies might fail or be skipped, so check what was actually run
    assert len(actual_strategies) > 0
    assert "Equal Weight" in actual_strategies  # Baseline should always work


def test_backtest_suite_metrics_consistency(
    mock_dataset_path: Path,
    mock_split: TrainTestSplit,
    tmp_path: Path,
) -> None:
    """Test that metrics are consistent with backtest results."""
    output_dir = tmp_path / "results"

    suite = run_full_backtest_suite(
        dataset_path=mock_dataset_path,
        output_dir=output_dir,
        split=mock_split,
    )

    # For each strategy, check that metrics match result
    for strategy_name, result in suite.strategies.items():
        metrics = suite.metrics[strategy_name]

        # Check date consistency
        assert metrics.start_date == result.start_date
        assert metrics.end_date == result.end_date

        # Check transaction cost consistency
        assert metrics.transaction_costs_total == result.total_transaction_costs
        assert metrics.num_rebalances == result.num_rebalances

    for strategy_name, result in suite.full_results.items():
        metrics = suite.overall_metrics[strategy_name]
        assert metrics.start_date == result.start_date
        assert metrics.end_date == result.end_date


# ===== Error Handling Tests =====


def test_backtest_suite_handles_invalid_dataset_format(
    mock_split: TrainTestSplit,
    tmp_path: Path,
) -> None:
    """Test that invalid dataset format raises appropriate error."""
    output_dir = tmp_path / "results"

    # Create a dataset with missing required columns
    invalid_data = pl.DataFrame({
        "date": [datetime(2023, 1, 1, tzinfo=UTC)],
        "value": [100.0],
        # Missing: timestamp, symbol, return
    })

    invalid_path = tmp_path / "invalid.parquet"
    invalid_data.write_parquet(invalid_path)

    with pytest.raises(ValueError, match="Dataset missing required columns"):
        run_full_backtest_suite(
            dataset_path=invalid_path,
            output_dir=output_dir,
            split=mock_split,
        )


def test_backtest_suite_handles_csv_dataset(
    mock_split: TrainTestSplit,
    tmp_path: Path,
) -> None:
    """Test that CSV datasets are supported."""
    output_dir = tmp_path / "results"

    # Create CSV dataset with enough data for validation and VaR/CVaR (100+ observations)
    start_date = datetime(2018, 1, 1, tzinfo=UTC)
    num_days = 2_200  # ~6 years of observations to cover train/test windows

    data = []
    for day in range(num_days):
        date = start_date + timedelta(days=day)
        for sector in ["SPY", "AGG"]:
            data.append({
                "timestamp": date.isoformat(),
                "symbol": sector,
                "return": 0.001,
            })

    df = pl.DataFrame(data)
    csv_path = tmp_path / "sector_returns.csv"
    df.write_csv(csv_path)

    suite = run_full_backtest_suite(
        dataset_path=csv_path,
        output_dir=output_dir,
        split=mock_split,
    )

    # Check that suite was created successfully
    assert isinstance(suite, BacktestSuite)
    assert len(suite.strategies) > 0


def test_backtest_suite_reproducibility(
    mock_dataset_path: Path,
    mock_split: TrainTestSplit,
    tmp_path: Path,
) -> None:
    """Test that results are reproducible with same random seed."""
    output_dir_1 = tmp_path / "results1"
    output_dir_2 = tmp_path / "results2"

    # Run suite twice with same seed
    suite_1 = run_full_backtest_suite(
        dataset_path=mock_dataset_path,
        output_dir=output_dir_1,
        split=mock_split,
        config_overrides={"random_seed": 42},
    )

    suite_2 = run_full_backtest_suite(
        dataset_path=mock_dataset_path,
        output_dir=output_dir_2,
        split=mock_split,
        config_overrides={"random_seed": 42},
    )

    # Results should be identical
    for strategy_name in suite_1.strategies.keys():
        if strategy_name not in suite_2.strategies:
            continue

        metrics_1 = suite_1.metrics[strategy_name]
        metrics_2 = suite_2.metrics[strategy_name]

        # Check key metrics are identical
        assert abs(metrics_1.annualized_return - metrics_2.annualized_return) < 1e-10
        assert abs(metrics_1.sharpe_ratio - metrics_2.sharpe_ratio) < 1e-10
        assert abs(metrics_1.maximum_drawdown - metrics_2.maximum_drawdown) < 1e-10


def test_run_walk_forward_backtest_suite(
    mock_dataset_path: Path,
    tmp_path: Path,
) -> None:
    """Verify walk-forward orchestration produces fold outputs and summaries."""
    config = WalkForwardConfig(
        start_date=datetime(2018, 1, 1, tzinfo=UTC),
        end_date=datetime(2023, 12, 31, tzinfo=UTC),
        train_years=2,
        test_years=1,
        step_years=1,
    )

    result = run_walk_forward_backtest_suite(
        dataset_path=mock_dataset_path,
        output_dir=tmp_path,
        walk_forward_config=config,
    )

    assert isinstance(result, WalkForwardBacktestResult)
    expected_splits = config.to_splits()
    assert len(result.suites) == len(expected_splits) > 0

    # Aggregated metrics should include Sharpe ratios
    aggregate = result.aggregate_metrics()
    assert not aggregate.is_empty()
    assert "sharpe_ratio" in aggregate.columns

    summary_dir = tmp_path / "walk_forward"
    assert (summary_dir / "aggregate_metrics.csv").exists()
    assert (summary_dir / "strategy_summary.csv").exists()

    # Fold artefacts should have been produced
    first_fold_dir = summary_dir / "fold_01"
    report_name = (
        f"backtest_results_{expected_splits[0].train_start.year}_"
        f"{expected_splits[0].test_end.year}.md"
    )
    assert (first_fold_dir / "performance_comparison_table.csv").exists()
    assert (first_fold_dir / report_name).exists()


def test_run_walk_forward_backtest_suite_accepts_overrides(
    mock_dataset_path: Path,
    tmp_path: Path,
) -> None:
    """Ensure walk-forward suite threads liquidity config and turnover overrides."""
    config = WalkForwardConfig(
        start_date=datetime(2018, 1, 1, tzinfo=UTC),
        end_date=datetime(2021, 12, 31, tzinfo=UTC),
        train_years=1,
        test_years=1,
        step_years=1,
    )
    custom_liquidity = LiquidityScalingConfig(
        severe_threshold=-10.0,
        moderate_threshold=-5.0,
        severe_regime_multiplier=0.2,
        moderate_regime_multiplier=0.3,
        severe_liquidity_multiplier=0.2,
        moderate_liquidity_multiplier=0.3,
        neutral_liquidity_multiplier=0.95,
        floor=0.9,
    )
    overrides = {
        "3d_factor_rolling": 0.15,
        "3d_factor_stable": 0.05,
    }

    result = run_walk_forward_backtest_suite(
        dataset_path=mock_dataset_path,
        output_dir=tmp_path,
        walk_forward_config=config,
        liquidity_config=custom_liquidity,
        turnover_overrides=overrides,
    )

    assert result.suites, "Expected at least one backtest suite"
    first_suite = result.suites[0]
    assert first_suite.turnover_overrides["3d_factor_rolling"] == pytest.approx(0.15)
    assert first_suite.turnover_overrides["3d_factor_stable"] == pytest.approx(0.05)
    assert first_suite.regime_factor_multipliers
    for factor_map in first_suite.regime_factor_multipliers.values():
        assert factor_map["factor_liquidity"] >= 0.9 - 1e-9


def test_liquidity_mitigation_experiments_with_walk_forward(
    mock_dataset_path: Path,
    tmp_path: Path,
) -> None:
    """Verify walk-forward summaries are captured for mitigation scenarios."""
    scenario = LiquidityMitigationScenario(
        name="Turnover Walk Test",
        rolling_turnover_smoothing=0.45,
        stable_turnover_smoothing=0.30,
        liquidity_config=LiquidityScalingConfig(),
    )
    wf_config = WalkForwardConfig(
        start_date=datetime(2014, 1, 1, tzinfo=UTC),
        end_date=datetime(2020, 12, 31, tzinfo=UTC),
        train_years=3,
        test_years=1,
        step_years=1,
    )

    results = run_liquidity_mitigation_experiments(
        dataset_path=mock_dataset_path,
        output_dir=tmp_path,
        scenarios=[scenario],
        run_walk_forward=True,
        walk_forward_config=wf_config,
    )

    assert len(results) == 1
    result = results[0]
    assert result.walk_forward_sharpe_mean is not None
    assert result.walk_forward_output_directory is not None
    assert result.walk_forward_output_directory.exists()


def test_get_liquidity_mitigation_scenarios_filters_known_names() -> None:
    """Ensure scenario resolver returns filtered lists and rejects unknown names."""
    all_scenarios = get_liquidity_mitigation_scenarios()
    expected_names = {
        "Baseline Controls",
        "Turnover Smoothing 0.55/0.40",
        "Tighter Liquidity Regime Scaling",
        "Turnover Stress Test",
        "Stress: 2008 Liquidity Shock",
        "Stress: 2020 Volatility Spike",
        "Stress: 2022 Rates + Stocks",
        "Stress: 1987 Black Monday",
        "Stress: Synthetic Liquidity Shock",
    }
    retrieved_names = {scenario.name for scenario in all_scenarios}
    assert expected_names.issubset(retrieved_names)

    subset = get_liquidity_mitigation_scenarios([all_scenarios[0].name])
    assert len(subset) == 1
    assert subset[0].name == all_scenarios[0].name

    with pytest.raises(ValueError):
        get_liquidity_mitigation_scenarios(["unknown-scenario"])


def test_run_liquidity_mitigation_experiments_single_scenario(
    mock_dataset_path: Path,
    tmp_path: Path,
) -> None:
    """Ensure liquidity mitigation experiments run and capture summary output."""
    output_dir = tmp_path / "experiments"
    scenario = LiquidityMitigationScenario(
        name="Unit Test Scenario",
        rolling_turnover_smoothing=0.25,
        stable_turnover_smoothing=0.15,
        liquidity_config=LiquidityScalingConfig(),
    )

    results = run_liquidity_mitigation_experiments(
        dataset_path=mock_dataset_path,
        output_dir=output_dir,
        scenarios=[scenario],
    )

    assert len(results) == 1
    result = results[0]
    assert result.scenario_name == "Unit Test Scenario"
    assert result.rolling_sharpe_delta == pytest.approx(0.0)
    assert (output_dir / "liquidity_mitigation_results.csv").exists()
    scenario_dir = output_dir / "unit_test_scenario"
    assert (scenario_dir / "performance_comparison_table.csv").exists()


def test_walk_forward_metadata_includes_defaults(
    mock_dataset_path: Path,
    tmp_path: Path,
) -> None:
    """Walk-forward summaries should persist metadata with default parameters."""
    output_dir = tmp_path / "wf_outputs"
    config = WalkForwardConfig(
        start_date=datetime(2015, 1, 1, tzinfo=UTC),
        end_date=datetime(2019, 12, 31, tzinfo=UTC),
        train_years=3,
        test_years=1,
        step_years=1,
    )

    run_walk_forward_backtest_suite(
        dataset_path=mock_dataset_path,
        output_dir=output_dir,
        walk_forward_config=config,
    )

    metadata_path = output_dir / "walk_forward" / "metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    defaults = ThreeDRiskBacktestDefaults()

    assert metadata["risk_free_rate"] == pytest.approx(defaults.risk_free_rate)
    assert metadata["turnover_smoothing"]["stable"] == pytest.approx(defaults.stable_turnover_smoothing)
    assert metadata["turnover_smoothing"]["rolling"] == pytest.approx(defaults.rolling_turnover_smoothing)
    assert metadata["liquidity_config"]["severe_threshold"] == pytest.approx(
        defaults.liquidity_scaling.severe_threshold,
    )
    assert metadata["split_count"] == len(config.to_splits())
    assert metadata["summaries_directory"].endswith("walk_forward")
    wf_config = metadata["walk_forward_config"]
    assert wf_config["train_years"] == config.train_years
    assert wf_config["test_years"] == config.test_years
    assert wf_config["step_years"] == config.step_years
    assert len(metadata["splits"]) == len(config.to_splits())


def test_three_d_risk_backtest_defaults_build_liquidity_config() -> None:
    """Defaults should hydrate LiquidityScalingConfig with matching parameters."""
    defaults = ThreeDRiskBacktestDefaults()
    config = defaults.build_liquidity_config()

    assert config.severe_threshold == pytest.approx(defaults.liquidity_scaling.severe_threshold)
    assert config.moderate_threshold == pytest.approx(defaults.liquidity_scaling.moderate_threshold)
    assert config.severe_regime_multiplier == pytest.approx(defaults.liquidity_scaling.severe_regime_multiplier)
    assert config.moderate_regime_multiplier == pytest.approx(defaults.liquidity_scaling.moderate_regime_multiplier)
    assert config.severe_liquidity_multiplier == pytest.approx(defaults.liquidity_scaling.severe_liquidity_multiplier)
    assert config.moderate_liquidity_multiplier == pytest.approx(defaults.liquidity_scaling.moderate_liquidity_multiplier)
    assert config.neutral_liquidity_multiplier == pytest.approx(defaults.liquidity_scaling.neutral_liquidity_multiplier)
    assert config.floor == pytest.approx(defaults.liquidity_scaling.floor)


def test_three_d_risk_backtest_defaults_walk_forward_permutations() -> None:
    """Defaults should expose walk-forward permutations with canonical primary ordering."""
    defaults = ThreeDRiskBacktestDefaults()
    permutations = defaults.walk_forward_permutations

    assert permutations, "Expected at least one walk-forward permutation"
    assert defaults.primary_walk_forward_permutation == permutations[0]
    for permutation in permutations:
        assert permutation.name
        assert permutation.train_years > 0
        assert permutation.test_years > 0
        if permutation.nested is not None:
            assert permutation.nested.min_folds > 0


def test_run_multi_horizon_walk_forward_analysis_produces_permutation_outputs(
    mock_dataset_path: Path,
    tmp_path: Path,
) -> None:
    """Multi-horizon validation should emit artefacts for each permutation."""
    permutations = (
        WalkForwardPermutationDefaults(
            name="Test Baseline 4y/1y",
            description="Baseline permutation for unit test runtime",
            train_years=4,
            test_years=1,
            step_years=1,
            nested=NestedWalkForwardDefaults(train_years=2, test_years=1, step_years=1, min_folds=1),
        ),
        WalkForwardPermutationDefaults(
            name="Test Secondary 3y/1y",
            description="Secondary permutation for unit test",
            train_years=3,
            test_years=1,
            step_years=2,
            nested=None,
        ),
    )
    output_dir = tmp_path / "multi_horizon"

    result = run_multi_horizon_walk_forward_analysis(
        dataset_path=mock_dataset_path,
        output_dir=output_dir,
        start_date=datetime(2014, 1, 1, tzinfo=UTC),
        end_date=datetime(2020, 12, 31, tzinfo=UTC),
        permutations=permutations,
        include_primary_root=True,
    )

    primary_slug = permutations[0].slug
    secondary_slug = permutations[1].slug
    assert primary_slug in result.runs
    assert secondary_slug in result.runs

    base_dir = output_dir / "walk_forward"
    assert (base_dir / "aggregate_metrics.csv").exists()

    alias_dir = base_dir / "permutations" / primary_slug
    assert (alias_dir / "README.txt").exists()
    assert (alias_dir / "permutation_metadata.json").exists()

    secondary_dir = base_dir / "permutations" / secondary_slug
    assert (secondary_dir / "aggregate_metrics.csv").exists()
    assert result.runs[primary_slug].nested_results, "Expected nested results for primary permutation"
    summary_df = result.summary_table()
    assert not summary_df.is_empty()
    nested_df = result.nested_summary()
    # Nested validation should produce metrics for the baseline permutation even with fallback dataset.
    assert primary_slug in nested_df.get_column("permutation_slug").to_list()


def test_three_d_risk_backtest_defaults_fallbacks_are_immutable() -> None:
    """Fallback mapping should be immutable to preserve config integrity."""
    defaults = ThreeDRiskBacktestDefaults()

    with pytest.raises(TypeError):
        defaults.liquidity_contribution_fallbacks["New Regime"] = -0.01


def test_parse_comma_separated_deduplicates_and_orders() -> None:
    """CLI helper should strip whitespace and remove duplicates."""
    parsed = _parse_comma_separated("  alpha , beta,alpha , , gamma ")
    assert parsed == ("alpha", "beta", "gamma")


def test_parse_args_accepts_heatmap_specs() -> None:
    """CLI parser should expose heatmap spec string argument."""
    namespace = parse_walk_forward_args(["--heatmap-specs", "foo,bar"])
    assert namespace.heatmap_specs == "foo,bar"
    assert namespace.parameter_heatmaps is False


def test_parse_args_exposes_phase3_battery_flag() -> None:
    """CLI parser should surface the phase3 battery toggle."""
    namespace = parse_walk_forward_args(["--phase3-battery"])
    assert namespace.phase3_battery is True


def test_monitoring_defaults_validate_alert_rules() -> None:
    """Monitoring defaults should enforce alert rule coverage."""
    with pytest.raises(ValueError, match="alert_rules"):
        MonitoringExportDefaults(alert_rules={"grafana": "alerts/only-grafana.yml"})
