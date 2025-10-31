"""
Utility to regenerate Phase 3 walk-forward artefacts.

The script executes the canonical walk-forward configuration (5y train / 1y test
with a 1y stride) alongside additional validation permutations defined in
``ThreeDRiskBacktestDefaults``. Outputs are written to
``playground/reports/backtesting/walk_forward/`` with per-permutation summaries.
"""

from __future__ import annotations

import argparse
import cProfile
import sys
from collections.abc import Sequence
from datetime import UTC
from datetime import datetime
from pathlib import Path
from time import perf_counter

import structlog


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.common.metrics_bootstrap import get_histogram  # noqa: E402
from ml.config.playground import ThreeDRiskBacktestDefaults  # noqa: E402
from ml.config.playground import WalkForwardPermutationDefaults  # noqa: E402
from playground.backtest.runner import MultiHorizonWalkForwardResult  # noqa: E402
from playground.backtest.runner import export_phase3_monitoring_snapshot  # noqa: E402
from playground.backtest.runner import get_liquidity_mitigation_scenarios  # noqa: E402
from playground.backtest.runner import run_extended_diagnostics  # noqa: E402
from playground.backtest.runner import run_full_backtest_suite  # noqa: E402
from playground.backtest.runner import run_liquidity_mitigation_experiments  # noqa: E402
from playground.backtest.runner import run_monte_carlo_stress_suite  # noqa: E402
from playground.backtest.runner import run_multi_horizon_walk_forward_analysis  # noqa: E402
from playground.backtest.runner import run_parameter_heatmap_suite  # noqa: E402
from playground.backtest.runner import run_proxy_dataset_validation  # noqa: E402
from playground.backtest.runner import run_vintage_simulation_suite  # noqa: E402
from playground.backtest.splits import WalkForwardConfig  # noqa: E402
from playground.monitoring.integrations import persist_monitoring_integrations  # noqa: E402


DEFAULT_DATASET_PATH = Path("playground/data/sector_dataset")
DEFAULT_OUTPUT_DIR = Path("playground/reports/backtesting")
PLAYGROUND_DEFAULTS = ThreeDRiskBacktestDefaults()
LOGGER = structlog.get_logger(__name__)
BATTERY_RUNTIME_HIST = get_histogram(
    "phase3_validation_battery_runtime_seconds",
    "Runtime distribution for the Phase 3 validation battery executions.",
    labelnames=("mode",),
    buckets=(600.0, 900.0, 1200.0, 1500.0, 1800.0, 2400.0),
)


def _parse_comma_separated(value: str) -> tuple[str, ...]:
    """Parse comma-separated CLI input into a tuple of unique entries preserving order."""
    entries: list[str] = []
    seen: set[str] = set()
    for part in value.split(","):
        normalized = part.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        entries.append(normalized)
    return tuple(entries)


def _positive_int(value: str) -> int:
    """Argparse helper to ensure integer options remain positive."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def refresh_phase3_walk_forward(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    config: WalkForwardConfig | None = None,
) -> MultiHorizonWalkForwardResult:
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
    return result


def execute_phase3_suite(
    *,
    dataset_path: Path,
    output_dir: Path,
    heatmap_slugs: tuple[str, ...],
    scenario_names: tuple[str, ...],
    run_liquidity_experiments: bool,
    run_monte_carlo: bool,
    run_heatmaps: bool,
    run_diagnostics: bool,
    run_proxy: bool,
    run_vintage: bool,
    run_monitoring: bool,
) -> None:
    """
    Execute the configured Phase 3 validation suites.

    Args:
        dataset_path: Root path containing the canonical Phase 3 dataset.
        output_dir: Directory receiving all generated artefacts.
        heatmap_slugs: Optional heatmap specifications to target.
        scenario_names: Optional liquidity mitigation scenarios to run.
        run_liquidity_experiments: Whether liquidity mitigation experiments should run.
        run_monte_carlo: Whether to execute the Monte Carlo stress suite.
        run_heatmaps: Whether parameter heatmaps should run.
        run_diagnostics: Whether to execute extended diagnostics.
        run_proxy: Whether proxy dataset validation should run.
        run_vintage: Whether vintage simulations should run.
        run_monitoring: Whether to emit the monitoring snapshot and integration payloads.
    """
    walk_forward_config = WalkForwardConfig(
        start_date=datetime(2010, 1, 1, tzinfo=UTC),
        end_date=datetime(2024, 12, 31, tzinfo=UTC),
        train_years=5,
        test_years=1,
        step_years=1,
    )

    walk_forward_result = refresh_phase3_walk_forward(
        dataset_path=dataset_path,
        output_dir=output_dir,
        config=walk_forward_config,
    )
    LOGGER.info(
        "walk_forward_refresh_completed",
        summary_path=str((output_dir / "walk_forward" / "permutation_summary.csv").resolve()),
    )

    monte_carlo_result = None
    if run_monte_carlo:
        monte_carlo_result = run_monte_carlo_stress_suite(
            dataset_path=dataset_path,
            output_dir=output_dir,
        )
        monte_carlo_summary = output_dir / "stress" / "monte_carlo" / "summary.csv"
        overlay_events = sum(len(path.overlay_events) for path in monte_carlo_result.paths)
        LOGGER.info(
            "monte_carlo_stress_suite_completed",
            summary_path=str(monte_carlo_summary.resolve()),
            overlay_events=overlay_events,
        )

    heatmap_result = None
    if run_heatmaps:
        heatmap_result = run_parameter_heatmap_suite(
            dataset_path=dataset_path,
            output_dir=output_dir,
            spec_slugs=heatmap_slugs if heatmap_slugs else None,
        )
        LOGGER.info(
            "parameter_heatmaps_completed",
            specs=[run.spec.slug for run in heatmap_result.runs],
            requested_specs=list(heatmap_slugs),
            summary_path=str((output_dir / heatmap_result.output_dirname / "summary.csv").resolve()),
        )

    diagnostics_result = None
    if run_diagnostics:
        baseline_dir = output_dir / "baseline"
        baseline_suite = run_full_backtest_suite(
            dataset_path=dataset_path,
            output_dir=baseline_dir,
            split=None,
        )
        diagnostics_result = run_extended_diagnostics(
            suite=baseline_suite,
            output_dir=output_dir,
        )
        LOGGER.info(
            "extended_diagnostics_completed",
            output_directory=str((output_dir / "diagnostics").resolve()),
        )

    proxy_result = None
    if run_proxy:
        proxy_result = run_proxy_dataset_validation(
            dataset_path=dataset_path,
            output_dir=output_dir,
        )
        LOGGER.info(
            "proxy_dataset_validation_completed",
            datasets=[run.spec.slug for run in proxy_result.runs],
            summary_path=str((output_dir / "proxy_datasets" / "summary.csv").resolve()),
        )

    vintage_result = None
    if run_vintage:
        vintage_result = run_vintage_simulation_suite(
            dataset_path=dataset_path,
            output_dir=output_dir,
        )
        LOGGER.info(
            "vintage_simulation_suite_completed",
            summary_path=str((output_dir / "vintage" / "summary.csv").resolve()),
        )

    if run_liquidity_experiments:
        experiments_dir = output_dir / "experiments" / "liquidity_mitigation"
        scenario_list = get_liquidity_mitigation_scenarios(list(scenario_names) or None)
        run_liquidity_mitigation_experiments(
            dataset_path=dataset_path,
            output_dir=experiments_dir,
            scenarios=scenario_list,
            run_walk_forward=True,
            walk_forward_config=walk_forward_config,
        )
        LOGGER.info(
            "liquidity_mitigation_experiments_completed",
            scenarios=[scenario.name for scenario in scenario_list],
            output_directory=str(experiments_dir.resolve()),
        )

    if run_monitoring:
        snapshot = export_phase3_monitoring_snapshot(
            output_dir=output_dir,
            walk_forward=walk_forward_result,
            monte_carlo=monte_carlo_result,
            heatmaps=heatmap_result,
            diagnostics=diagnostics_result,
            proxy_datasets=proxy_result,
            vintage=vintage_result,
        )
        integration_artifacts = persist_monitoring_integrations(
            snapshot=snapshot,
            output_dir=output_dir,
        )
        LOGGER.info(
            "monitoring_snapshot_emitted",
            path=str(snapshot.path.resolve()),
            grafana_payload=str(integration_artifacts.grafana_payload_path.resolve()),
            pagerduty_payload=str(integration_artifacts.pagerduty_payload_path.resolve()),
        )


def _resolve_profile_path(base: Path, run_index: int, total_runs: int) -> Path:
    """Return a per-run profile path to avoid overwriting stats when stressing the battery."""
    if total_runs <= 1:
        return base
    suffix = base.suffix or ".prof"
    stem = base.stem or "phase3_battery"
    return base.with_name(f"{stem}_run{run_index + 1}{suffix}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--parameter-heatmaps",
        action="store_true",
        help="Generate parameter heatmaps using configured defaults.",
    )
    parser.add_argument(
        "--heatmap-specs",
        type=str,
        default="",
        help="Comma separated heatmap spec slugs to execute (default: all configured).",
    )
    parser.add_argument(
        "--extended-diagnostics",
        action="store_true",
        help="Run extended diagnostics on the baseline backtest suite.",
    )
    parser.add_argument(
        "--proxy-validation",
        action="store_true",
        help="Validate proxy datasets specified in configuration defaults.",
    )
    parser.add_argument(
        "--vintage-simulations",
        action="store_true",
        help="Execute vintage walk-forward simulations across configured windows.",
    )
    parser.add_argument(
        "--monitoring-export",
        action="store_true",
        help="Emit consolidated monitoring snapshot referencing generated artefacts.",
    )
    parser.add_argument(
        "--phase3-battery",
        action="store_true",
        help="Execute the full Phase 3 validation battery (enables all suite toggles).",
    )
    parser.add_argument(
        "--stress-runs",
        type=_positive_int,
        default=1,
        help="Repeat the selected suites multiple times to stress runtime limits (default: 1).",
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=None,
        help="Optional path to write cProfile data for Phase 3 battery runs.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """CLI entrypoint for walk-forward refresh routines."""
    args = parse_args()
    dataset_path = args.dataset_path
    output_dir = args.output_dir
    heatmap_slugs = _parse_comma_separated(args.heatmap_specs)
    scenario_names = _parse_comma_separated(args.scenarios)

    if args.phase3_battery:
        LOGGER.info("phase3_battery_enabled")

    stress_runs = args.stress_runs
    profile_output: Path | None = args.profile_output

    run_liquidity_experiments = args.liquidity_experiments or args.phase3_battery
    run_monte_carlo = args.monte_carlo_stress or args.phase3_battery
    run_heatmaps = args.parameter_heatmaps or args.phase3_battery or bool(heatmap_slugs)
    if heatmap_slugs and not args.parameter_heatmaps:
        LOGGER.info("parameter_heatmaps_auto_enabled", spec_slugs=heatmap_slugs)
    run_diagnostics = args.extended_diagnostics or args.phase3_battery
    run_proxy = args.proxy_validation or args.phase3_battery
    run_vintage = args.vintage_simulations or args.phase3_battery
    run_monitoring = args.monitoring_export or args.phase3_battery

    mode_label = "full" if args.phase3_battery else "custom"
    run_durations: list[float] = []

    for run_index in range(stress_runs):
        profiler: cProfile.Profile | None = None
        profile_path: Path | None = None
        start_time = perf_counter()

        if profile_output is not None:
            profile_path = _resolve_profile_path(profile_output, run_index, stress_runs)
            profiler = cProfile.Profile()
            profiler.enable()

        try:
            execute_phase3_suite(
                dataset_path=dataset_path,
                output_dir=output_dir,
                heatmap_slugs=heatmap_slugs,
                scenario_names=scenario_names,
                run_liquidity_experiments=run_liquidity_experiments,
                run_monte_carlo=run_monte_carlo,
                run_heatmaps=run_heatmaps,
                run_diagnostics=run_diagnostics,
                run_proxy=run_proxy,
                run_vintage=run_vintage,
                run_monitoring=run_monitoring,
            )
        finally:
            if profiler is not None and profile_path is not None:
                profiler.disable()
                profile_path.parent.mkdir(parents=True, exist_ok=True)
                profiler.dump_stats(str(profile_path))

        runtime_seconds = perf_counter() - start_time
        BATTERY_RUNTIME_HIST.labels(mode=mode_label).observe(runtime_seconds)
        LOGGER.info(
            "phase3_suite_run_completed",
            run_index=run_index + 1,
            total_runs=stress_runs,
            runtime_seconds=round(runtime_seconds, 3),
            profile_path=str(profile_path.resolve()) if profile_path is not None else None,
            profiling_enabled=profile_path is not None,
        )
        run_durations.append(runtime_seconds)

    if stress_runs > 1:
        total_runtime = sum(run_durations)
        LOGGER.info(
            "phase3_suite_stress_summary",
            total_runs=stress_runs,
            total_runtime_seconds=round(total_runtime, 3),
            average_runtime_seconds=round(total_runtime / stress_runs, 3),
            max_runtime_seconds=round(max(run_durations), 3),
            min_runtime_seconds=round(min(run_durations), 3),
        )


if __name__ == "__main__":
    main()
