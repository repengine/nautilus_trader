"""
Tests for Phase 3 visual export script.

The script operates on CSV artefacts so the tests synthesise a minimal dataset to
exercise each chart path without relying on the full backtest output.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from playground.scripts.export_phase3_visuals import export_phase3_visuals


def _write_csv(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(path)


def test_export_phase3_visuals_creates_expected_pngs(tmp_path: Path) -> None:
    data_dir = tmp_path / "reports"
    attr_dir = data_dir / "attribution"
    regime_dir = attr_dir / "regime"

    performance = pl.DataFrame(
        {
            "strategy": [
                "60/40 Portfolio",
                "3D Factor (Rolling Betas)",
                "3D Factor (Stable Betas)",
                "Equal Weight",
            ],
            "annualized_return": [16.0, 15.5, 15.0, 14.5],
            "annualized_volatility": [20.0, 19.0, 18.5, 18.0],
            "sharpe_ratio": [0.75, 0.72, 0.70, 0.68],
            "sortino_ratio": [0.90, 0.88, 0.85, 0.83],
            "calmar_ratio": [0.50, 0.48, 0.46, 0.44],
            "max_drawdown": [-30.0, -32.0, -35.0, -34.0],
            "information_ratio": [0.05, 0.04, 0.03, 0.02],
            "num_rebalances": [70, 70, 70, 70],
            "transaction_costs": [4_000.0, 15_000.0, 12_000.0, 5_000.0],
        }
    )

    train_vs_test = pl.DataFrame(
        {
            "strategy": [
                "3D Factor (Rolling Betas)",
                "3D Factor (Rolling Betas)",
                "3D Factor (Stable Betas)",
                "3D Factor (Stable Betas)",
                "60/40 Portfolio",
                "60/40 Portfolio",
                "Equal Weight",
                "Equal Weight",
            ],
            "period": ["train", "test"] * 4,
            "annualized_return": [10.0, 15.5, 9.8, 14.8, 11.0, 16.0, 9.5, 14.5],
            "annualized_volatility": [15.0, 19.0, 15.5, 18.5, 16.0, 20.0, 16.5, 18.0],
            "sharpe_ratio": [0.65, 0.72, 0.64, 0.70, 0.60, 0.70, 0.58, 0.68],
            "calmar_ratio": [0.40, 0.48, 0.38, 0.46, 0.42, 0.50, 0.39, 0.44],
            "max_drawdown": [-20.0, -32.0, -21.0, -35.0, -19.0, -30.0, -22.0, -34.0],
            "cumulative_return": [1.2, 1.3, 1.15, 1.25, 1.1, 1.2, 1.05, 1.15],
            "num_rebalances": [100, 70, 100, 70, 100, 70, 100, 70],
        }
    )

    regime_summary = pl.DataFrame(
        {
            "strategy": ["3D Factor (Rolling Betas)", "3D Factor (Rolling Betas)"],
            "regime_name": ["Bull Market", "Rate Hike"],
            "sharpe_ratio": [0.9, 0.4],
            "annualized_return": [18.0, 10.0],
            "annualized_volatility": [20.0, 18.0],
            "max_drawdown": [-15.0, -25.0],
            "calmar_ratio": [1.2, 0.5],
            "win_rate": [0.6, 0.4],
            "num_observations": [120, 80],
            "status": ["Passed", "Failed"],
        }
    )

    regime_attr = pl.DataFrame(
        {
            "regime": ["Bull Market", "Bull Market", "Bull Market", "Rate Hike", "Rate Hike", "Rate Hike"],
            "factor": ["factor_credit", "factor_duration", "factor_liquidity"] * 2,
            "beta": [0.1, 0.2, 0.05, 0.1, 0.2, 0.05],
            "annualized_contribution": [0.02, 0.01, 0.005, 0.015, 0.008, -0.02],
            "alpha": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "alpha_annualized": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )

    rolling_attr = pl.DataFrame(
        {
            "factor": ["factor_credit", "factor_duration", "factor_liquidity", "alpha"],
            "beta": [0.1, 0.2, 0.05, 0.3],
            "annualized_contribution": [0.03, 0.01, -0.01, 0.05],
        }
    )

    sixty_forty_attr = pl.DataFrame(
        {
            "factor": ["factor_credit", "factor_duration", "factor_liquidity", "alpha"],
            "beta": [0.05, 0.1, 0.02, 0.25],
            "annualized_contribution": [0.02, 0.005, -0.015, 0.04],
        }
    )

    _write_csv(data_dir / "performance_comparison_table.csv", performance)
    _write_csv(data_dir / "train_vs_test_metrics.csv", train_vs_test)
    _write_csv(data_dir / "regime_summary.csv", regime_summary)
    _write_csv(regime_dir / "3d_factor_rolling_betas_regime_attribution.csv", regime_attr)
    _write_csv(attr_dir / "3d_factor_rolling_betas_attribution.csv", rolling_attr)
    _write_csv(attr_dir / "60_40_portfolio_attribution.csv", sixty_forty_attr)

    output_dir = data_dir / "visuals"
    export_phase3_visuals(data_dir=data_dir, output_dir=output_dir)

    expected_files = {
        "rolling_vs_benchmark_sharpe.png",
        "regime_contributions.png",
        "liquidity_stress_panel.png",
        "sharpe_vs_tc.png",
        "attribution_waterfall.png",
    }

    created = {path.name for path in output_dir.iterdir() if path.suffix == ".png"}
    missing = expected_files.difference(created)
    assert not missing, f"Missing visuals: {sorted(missing)}"
    for file_name in expected_files:
        assert (output_dir / file_name).stat().st_size > 0


def test_export_phase3_visuals_writes_metadata_summary(tmp_path: Path) -> None:
    data_dir = tmp_path / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    # minimal CSVs required by function
    _write_csv(
        data_dir / "performance_comparison_table.csv",
        pl.DataFrame({"strategy": ["60/40 Portfolio"], "annualized_return": [10.0], "annualized_volatility": [15.0], "sharpe_ratio": [0.6], "sortino_ratio": [0.7], "calmar_ratio": [0.4], "max_drawdown": [-20.0], "information_ratio": [0.02], "num_rebalances": [12], "transaction_costs": [1000.0]}),
    )
    _write_csv(
        data_dir / "train_vs_test_metrics.csv",
        pl.DataFrame({"strategy": ["60/40 Portfolio"], "period": ["test"], "annualized_return": [10.0], "annualized_volatility": [15.0], "sharpe_ratio": [0.6], "calmar_ratio": [0.4], "max_drawdown": [-20.0], "cumulative_return": [1.1], "num_rebalances": [12]}),
    )
    _write_csv(
        data_dir / "regime_summary.csv",
        pl.DataFrame({"strategy": ["3D Factor (Rolling Betas)"], "regime_name": ["Sample"], "sharpe_ratio": [0.5], "annualized_return": [12.0], "annualized_volatility": [18.0], "max_drawdown": [-25.0], "calmar_ratio": [0.4], "win_rate": [0.6], "num_observations": [60], "status": ["Passed"]}),
    )
    attr_dir = data_dir / "attribution"
    regime_dir = attr_dir / "regime"
    _write_csv(
        regime_dir / "3d_factor_rolling_betas_regime_attribution.csv",
        pl.DataFrame({"regime": ["Sample"], "factor": ["factor_liquidity"], "beta": [0.1], "annualized_contribution": [-0.01], "alpha": [0.0], "alpha_annualized": [0.0]}),
    )
    _write_csv(
        attr_dir / "3d_factor_rolling_betas_attribution.csv",
        pl.DataFrame({"factor": ["factor_liquidity"], "beta": [0.1], "annualized_contribution": [-0.01]}),
    )
    _write_csv(
        attr_dir / "60_40_portfolio_attribution.csv",
        pl.DataFrame({"factor": ["factor_liquidity"], "beta": [0.05], "annualized_contribution": [-0.005]}),
    )

    metadata_dir = data_dir / "walk_forward"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.joinpath("metadata.json").write_text(
        """{
  "risk_free_rate": 0.02,
  "turnover_smoothing": {"stable": 0.3, "rolling": 0.4},
  "liquidity_config": {
    "severe_threshold": -0.02,
    "moderate_threshold": -0.01,
    "severe_liquidity_multiplier": 0.55,
    "moderate_liquidity_multiplier": 0.70,
    "neutral_liquidity_multiplier": 1.0
  },
  "baseline_strategies": ["Equal Weight", "60/40 Portfolio"]
}""",
        encoding="utf-8",
    )

    output_dir = data_dir / "visuals"
    export_phase3_visuals(data_dir=data_dir, output_dir=output_dir)

    summary_path = output_dir / "walk_forward_metadata.txt"
    assert summary_path.exists()
    contents = summary_path.read_text()
    assert "Risk-free rate: 0.0200" in contents
    assert "Turnover smoothing" in contents
    assert "Baseline strategies" in contents
