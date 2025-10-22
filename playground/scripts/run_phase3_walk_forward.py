"""
Utility to regenerate Phase 3 walk-forward artefacts.

The script executes the canonical walk-forward configuration (5y train / 1y test
with a 1y stride) alongside additional validation permutations defined in
``ThreeDRiskBacktestDefaults``. Outputs are written to
``playground/reports/backtesting/walk_forward/`` with per-permutation summaries.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.config.playground import ThreeDRiskBacktestDefaults  # noqa: E402
from ml.config.playground import WalkForwardPermutationDefaults  # noqa: E402
from playground.backtest.runner import get_liquidity_mitigation_scenarios  # noqa: E402
from playground.backtest.runner import run_liquidity_mitigation_experiments  # noqa: E402
from playground.backtest.runner import run_monte_carlo_stress_suite  # noqa: E402
from playground.backtest.runner import run_multi_horizon_walk_forward_analysis  # noqa: E402
from playground.backtest.splits import WalkForwardConfig  # noqa: E402


DEFAULT_DATASET_PATH = Path("playground/data/sector_dataset")
DEFAULT_OUTPUT_DIR = Path("playground/reports/backtesting")
PLAYGROUND_DEFAULTS = ThreeDRiskBacktestDefaults()


def refresh_phase3_walk_forward(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    config: WalkForwardConfig | None = None,
) -> None:
    """
    Refresh walk-forward outputs for the Phase 3 backtest suite.

    Args:
        dataset_path: Path to the sector dataset used for backtesting.
        output_dir: Directory receiving walk-forward artefacts.
        config: Optional custom walk-forward configuration for the primary permutation.
    """
    resolved_config = config or WalkForwardConfig(
        start_date=datetime(2010, 1, 1, tzinfo=UTC),
        end_date=datetime(2024, 12, 31, tzinfo=UTC),
        train_years=5,
        test_years=1,
        step_years=1,
    )
    resolved_defaults = PLAYGROUND_DEFAULTS
    liquidity_config = resolved_defaults.build_liquidity_config()
    turnover_overrides = {
        "3d_factor_stable": resolved_defaults.stable_turnover_smoothing,
        "3d_factor_rolling": resolved_defaults.rolling_turnover_smoothing,
    }

    permutations: list[WalkForwardPermutationDefaults] = list(resolved_defaults.walk_forward_permutations)
    if permutations:
        primary = permutations[0]
        nested = primary.nested
        if nested is not None and nested.train_years >= resolved_config.train_years:
            nested = None
        permutations[0] = WalkForwardPermutationDefaults(
            name=primary.name,
            description=primary.description,
            train_years=resolved_config.train_years,
            test_years=resolved_config.test_years,
            step_years=resolved_config.step_years,
            nested=nested,
        )

    result = run_multi_horizon_walk_forward_analysis(
        dataset_path=dataset_path,
        output_dir=output_dir,
        start_date=resolved_config.start_date,
        end_date=resolved_config.end_date,
        permutations=tuple(permutations),
        config_overrides=None,
        liquidity_config=liquidity_config,
        turnover_overrides=turnover_overrides,
        include_primary_root=True,
    )
    result.write_summary()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Refresh Phase 3 walk-forward outputs.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the sector dataset root (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store walk-forward artefacts (default: %(default)s)",
    )
    parser.add_argument(
        "--liquidity-experiments",
        action="store_true",
        help="Also regenerate liquidity mitigation scenarios with walk-forward summaries.",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="",
        help="Comma separated liquidity scenario names (defaults to all).",
    )
    parser.add_argument(
        "--monte-carlo-stress",
        action="store_true",
        help="Execute Monte Carlo regime stress sweep using default configuration.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for walk-forward refresh routines."""
    args = parse_args()
    dataset_path = args.dataset_path
    output_dir = args.output_dir

    walk_forward_config = WalkForwardConfig(
        start_date=datetime(2010, 1, 1, tzinfo=UTC),
        end_date=datetime(2024, 12, 31, tzinfo=UTC),
        train_years=5,
        test_years=1,
        step_years=1,
    )

    refresh_phase3_walk_forward(
        dataset_path=dataset_path,
        output_dir=output_dir,
        config=walk_forward_config,
    )

    if args.monte_carlo_stress:
        run_monte_carlo_stress_suite(
            dataset_path=dataset_path,
            output_dir=output_dir,
        )

    if args.liquidity_experiments:
        experiments_dir = output_dir / "experiments" / "liquidity_mitigation"
        scenario_names = [name.strip() for name in args.scenarios.split(",") if name.strip()]
        scenario_list = get_liquidity_mitigation_scenarios(scenario_names or None)
        run_liquidity_mitigation_experiments(
            dataset_path=dataset_path,
            output_dir=experiments_dir,
            scenarios=scenario_list,
            run_walk_forward=True,
            walk_forward_config=walk_forward_config,
        )


if __name__ == "__main__":
    main()
