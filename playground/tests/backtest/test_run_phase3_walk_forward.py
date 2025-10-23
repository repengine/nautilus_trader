"""
Tests for the Phase 3 walk-forward CLI helpers.

These tests focus on verifying that the script wires shared configuration
defaults through to the backtest harness so all entrypoints stay consistent.
"""

from __future__ import annotations

import sys
import types
from datetime import datetime
from pathlib import Path

import pytest


if "nautilus_trader" not in sys.modules:
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

from ml.config.playground import ThreeDRiskBacktestDefaults
from playground.scripts.run_phase3_walk_forward import main as run_phase3_main
from playground.scripts.run_phase3_walk_forward import refresh_phase3_walk_forward


def test_refresh_phase3_walk_forward_uses_shared_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the walk-forward refresh uses the canonical defaults for liquidity and turnover."""
    defaults = ThreeDRiskBacktestDefaults()
    captured: dict[str, object] = {}

    class _FakeResult:
        def __init__(self, directory: Path) -> None:
            self.base_directory = directory

        def write_summary(self) -> None:
            captured["summary_written"] = True

    def fake_run_multi_horizon_walk_forward_analysis(
        *,
        dataset_path: Path,
        output_dir: Path,
        start_date: datetime,
        end_date: datetime,
        permutations,
        config_overrides,
        liquidity_config,
        turnover_overrides,
        include_primary_root: bool,
    ) -> object:
        captured["dataset_path"] = dataset_path
        captured["output_dir"] = output_dir
        captured["start_date"] = start_date
        captured["end_date"] = end_date
        captured["liquidity_config"] = liquidity_config
        captured["turnover_overrides"] = turnover_overrides
        captured["include_primary_root"] = include_primary_root
        return _FakeResult(output_dir)

    monkeypatch.setattr(
        "playground.scripts.run_phase3_walk_forward.run_multi_horizon_walk_forward_analysis",
        fake_run_multi_horizon_walk_forward_analysis,
    )
    # Prevent accidental execution of liquidity experiments when the flag is disabled.
    monkeypatch.setattr(
        "playground.scripts.run_phase3_walk_forward.run_liquidity_mitigation_experiments",
        lambda **_: [],
    )

    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    output_dir = tmp_path / "reports"

    result = refresh_phase3_walk_forward(
        dataset_path=dataset_path,
        output_dir=output_dir,
    )

    assert captured["dataset_path"] == dataset_path
    assert captured["output_dir"] == output_dir
    assert isinstance(result, _FakeResult)
    assert captured.get("summary_written") is True
    liquidity_config = captured["liquidity_config"]
    assert liquidity_config is not None
    assert liquidity_config.severe_threshold == pytest.approx(defaults.liquidity_scaling.severe_threshold)
    assert liquidity_config.moderate_threshold == pytest.approx(defaults.liquidity_scaling.moderate_threshold)

    turnover_overrides = captured["turnover_overrides"]
    assert turnover_overrides == {
        "3d_factor_stable": defaults.stable_turnover_smoothing,
        "3d_factor_rolling": defaults.rolling_turnover_smoothing,
    }


def test_main_runs_heatmaps_when_specs_requested(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Providing --heatmap-specs should trigger the heatmap suite even without the toggle."""
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    output_dir = tmp_path / "reports"
    output_dir.mkdir()

    calls: dict[str, object] = {}

    def fake_refresh_phase3_walk_forward(
        *,
        dataset_path: Path,
        output_dir: Path,
        config,
    ) -> object:
        calls["refresh_dataset_path"] = dataset_path
        calls["refresh_output_dir"] = output_dir
        return types.SimpleNamespace(base_directory=output_dir)

    def fake_run_parameter_heatmap_suite(
        *,
        dataset_path: Path,
        output_dir: Path,
        spec_slugs,
    ) -> object:
        calls["heatmap_spec_slugs"] = spec_slugs
        return types.SimpleNamespace(
            base_directory=output_dir,
            output_dirname="heatmaps",
            runs=(
                types.SimpleNamespace(
                    spec=types.SimpleNamespace(slug="turnover-vs-liquidity-multipliers"),
                    metadata={
                        "target_strategy": "stub",
                        "metric_name": "sharpe_ratio",
                        "best_metric": 1.0,
                        "evaluated_combinations": 1,
                        "best_config": {},
                    },
                ),
            ),
        )

    monkeypatch.setattr(
        "playground.scripts.run_phase3_walk_forward.refresh_phase3_walk_forward",
        fake_refresh_phase3_walk_forward,
    )
    monkeypatch.setattr(
        "playground.scripts.run_phase3_walk_forward.run_parameter_heatmap_suite",
        fake_run_parameter_heatmap_suite,
    )
    monkeypatch.setattr(
        "playground.scripts.run_phase3_walk_forward.run_monte_carlo_stress_suite",
        lambda **_: types.SimpleNamespace(
            paths=(),
            stress_config=types.SimpleNamespace(report_metrics=(), report_quantiles=()),
        ),
    )
    monkeypatch.setattr(
        "playground.scripts.run_phase3_walk_forward.run_extended_diagnostics",
        lambda **_: None,
    )
    monkeypatch.setattr(
        "playground.scripts.run_phase3_walk_forward.run_proxy_dataset_validation",
        lambda **_: types.SimpleNamespace(runs=()),
    )
    monkeypatch.setattr(
        "playground.scripts.run_phase3_walk_forward.run_vintage_simulation_suite",
        lambda **_: types.SimpleNamespace(runs=()),
    )
    monkeypatch.setattr(
        "playground.scripts.run_phase3_walk_forward.run_liquidity_mitigation_experiments",
        lambda **_: [],
    )
    monkeypatch.setattr(
        "playground.scripts.run_phase3_walk_forward.export_phase3_monitoring_snapshot",
        lambda **_: types.SimpleNamespace(path=output_dir / "snapshot.json", payload={}),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_phase3_walk_forward.py",
            "--dataset-path",
            str(dataset_path),
            "--output-dir",
            str(output_dir),
            "--heatmap-specs",
            "turnover-vs-liquidity-multipliers",
        ],
    )

    run_phase3_main()

    assert calls["refresh_dataset_path"] == dataset_path
    assert calls["refresh_output_dir"] == output_dir
    assert calls["heatmap_spec_slugs"] == ("turnover-vs-liquidity-multipliers",)
