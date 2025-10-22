"""
Tests for the Phase 3 walk-forward CLI helpers.

These tests focus on verifying that the script wires shared configuration
defaults through to the backtest harness so all entrypoints stay consistent.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from ml.config.playground import ThreeDRiskBacktestDefaults
from playground.scripts.run_phase3_walk_forward import refresh_phase3_walk_forward


def test_refresh_phase3_walk_forward_uses_shared_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the walk-forward refresh uses the canonical defaults for liquidity and turnover."""
    defaults = ThreeDRiskBacktestDefaults()
    captured: dict[str, object] = {}

    def fake_run_walk_forward_backtest_suite(
        *,
        dataset_path: Path,
        output_dir: Path,
        walk_forward_config,
        config_overrides=None,
        liquidity_config=None,
        turnover_overrides=None,
    ) -> object:
        captured["dataset_path"] = dataset_path
        captured["output_dir"] = output_dir
        captured["walk_forward_config"] = walk_forward_config
        captured["liquidity_config"] = liquidity_config
        captured["turnover_overrides"] = turnover_overrides
        # Return a lightweight object compatible with downstream expectations.
        return SimpleNamespace(summarize_metrics=lambda: pl.DataFrame())

    monkeypatch.setattr(
        "playground.scripts.run_phase3_walk_forward.run_walk_forward_backtest_suite",
        fake_run_walk_forward_backtest_suite,
    )
    # Prevent accidental execution of liquidity experiments when the flag is disabled.
    monkeypatch.setattr(
        "playground.scripts.run_phase3_walk_forward.run_liquidity_mitigation_experiments",
        lambda **_: [],
    )

    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    output_dir = tmp_path / "reports"

    refresh_phase3_walk_forward(
        dataset_path=dataset_path,
        output_dir=output_dir,
    )

    assert captured["dataset_path"] == dataset_path
    assert captured["output_dir"] == output_dir
    liquidity_config = captured["liquidity_config"]
    assert liquidity_config is not None
    assert liquidity_config.severe_threshold == pytest.approx(defaults.liquidity_scaling.severe_threshold)
    assert liquidity_config.moderate_threshold == pytest.approx(defaults.liquidity_scaling.moderate_threshold)

    turnover_overrides = captured["turnover_overrides"]
    assert turnover_overrides == {
        "3d_factor_stable": defaults.stable_turnover_smoothing,
        "3d_factor_rolling": defaults.rolling_turnover_smoothing,
    }
