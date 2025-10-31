"""
CLI utilities to run Phase 2 regression diagnostics for the 3D risk model.

The script loads persisted sector/factor datasets, computes per-sector
regression diagnostics, validates acceptance thresholds, and writes the outputs
under ``playground/reports/phase2/diagnostics`` (or a user-specified directory).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC
from datetime import datetime
from pathlib import Path

import polars as pl
import structlog


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.config.playground import PhaseTwoValidationDefaults  # noqa: E402
from playground.risk_model.diagnostics import RegressionDiagnostics  # noqa: E402
from playground.risk_model.diagnostics import compute_regression_diagnostics  # noqa: E402
from playground.risk_model.diagnostics import create_diagnostics_summary  # noqa: E402


LOGGER = structlog.get_logger(__name__)
DEFAULT_DATASET_PATH = Path("playground/data/sector_dataset")
DEFAULT_OUTPUT_DIR = Path("playground/reports/phase2/diagnostics")
DEFAULT_SECTORS = (
    "XLF",
    "XLK",
    "XLE",
    "XLU",
    "XLY",
    "XLC",
    "XLI",
    "XLB",
    "XLV",
)
DEFAULT_FACTORS = ("factor_duration", "factor_credit", "factor_liquidity")
SECTOR_RETURNS_FILENAME = "sector_returns.parquet"
FACTOR_RETURNS_FILENAME = "factor_returns.parquet"


def _parse_iso_datetime(value: str | None) -> datetime | None:
    """Parse ISO-8601 timestamps, defaulting to UTC when tzinfo is missing."""
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _load_frame(path: Path) -> pl.DataFrame:
    """Load a parquet dataset with error handling."""
    if not path.exists():
        msg = f"Required dataset file missing: {path}"
        raise FileNotFoundError(msg)
    return pl.read_parquet(path)


def _filter_time_range(frame: pl.DataFrame, start: datetime | None, end: datetime | None) -> pl.DataFrame:
    """Filter a DataFrame by timestamp bounds."""
    filtered = frame
    if start is not None:
        filtered = filtered.filter(pl.col("timestamp") >= start)
    if end is not None:
        filtered = filtered.filter(pl.col("timestamp") <= end)
    return filtered


def _filter_sectors(frame: pl.DataFrame, sectors: Sequence[str]) -> pl.DataFrame:
    """Restrict the sector DataFrame to the requested tickers."""
    if not sectors:
        msg = "At least one sector must be provided"
        raise ValueError(msg)
    return frame.filter(pl.col("symbol").is_in(list(sectors)))


def _normalize_run_tag(run_tag: str | None) -> str:
    """Return a filesystem-safe run tag."""
    if not run_tag:
        return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in run_tag.strip())
    if not cleaned:
        msg = "Run tag must contain at least one valid character"
        raise ValueError(msg)
    return cleaned


def _diagnostics_records(diagnostics: dict[str, RegressionDiagnostics]) -> list[dict[str, object]]:
    """Convert diagnostics dataclasses into serialisable dictionaries."""
    records: list[dict[str, object]] = []
    for sector_id in sorted(diagnostics.keys()):
        diag = diagnostics[sector_id]
        records.append(
            {
                "sector_id": diag.sector_id,
                "n_observations": diag.n_observations,
                "date_range_start": diag.date_range_start.isoformat(),
                "date_range_end": diag.date_range_end.isoformat(),
                "r_squared": diag.r_squared,
                "adj_r_squared": diag.adj_r_squared,
                "f_statistic": diag.f_statistic,
                "f_pvalue": diag.f_pvalue,
                "durbin_watson": diag.durbin_watson,
                "beta_duration": diag.beta_duration,
                "beta_credit": diag.beta_credit,
                "beta_liquidity": diag.beta_liquidity,
                "alpha": diag.alpha,
                "p_value_duration": diag.p_value_duration,
                "p_value_credit": diag.p_value_credit,
                "p_value_liquidity": diag.p_value_liquidity,
                "vif_duration": diag.vif_duration,
                "vif_credit": diag.vif_credit,
                "vif_liquidity": diag.vif_liquidity,
                "bp_test_statistic": diag.bp_test_statistic,
                "bp_p_value": diag.bp_p_value,
                "residual_std": diag.residual_std,
            }
        )
    return records


def run_phase2_regression_diagnostics(
    dataset_path: Path,
    output_dir: Path,
    *,
    start: datetime | None,
    end: datetime | None,
    sectors: Sequence[str],
    factor_columns: Sequence[str],
    run_tag: str | None = None,
) -> Path:
    """
    Execute regression diagnostics and persist outputs.

    Returns
    -------
    Path
        Directory containing the generated artefacts.
    """
    sector_path = dataset_path / SECTOR_RETURNS_FILENAME
    factor_path = dataset_path / FACTOR_RETURNS_FILENAME
    sector_returns = _filter_sectors(_load_frame(sector_path), sectors)
    factor_returns = _load_frame(factor_path)

    sector_returns = _filter_time_range(sector_returns, start, end)
    factor_returns = _filter_time_range(factor_returns, start, end)

    if sector_returns.is_empty():
        msg = "Sector dataset is empty after applying filters"
        raise ValueError(msg)
    if factor_returns.is_empty():
        msg = "Factor dataset is empty after applying filters"
        raise ValueError(msg)

    thresholds = PhaseTwoValidationDefaults()
    diagnostics = compute_regression_diagnostics(
        sector_returns,
        factor_returns,
        factor_columns=tuple(factor_columns),
    )
    summary = create_diagnostics_summary(
        diagnostics,
        r2_threshold=thresholds.r2_threshold,
        p_value_threshold=thresholds.significance_level,
        vif_threshold=thresholds.vif_threshold,
        dw_lower=thresholds.durbin_watson_lower,
        dw_upper=thresholds.durbin_watson_upper,
        r2_pass_rate=thresholds.min_sector_pass_rate,
        significant_beta_pass_rate=thresholds.min_significant_beta_pass_rate,
        dw_pass_rate=thresholds.min_durbin_watson_pass_rate,
    )

    run_directory = output_dir / _normalize_run_tag(run_tag)
    run_directory.mkdir(parents=True, exist_ok=True)

    diagnostics_records = _diagnostics_records(diagnostics)
    diagnostics_frame = pl.DataFrame(diagnostics_records)
    diagnostics_csv = run_directory / "sector_regression_diagnostics.csv"
    diagnostics_parquet = run_directory / "sector_regression_diagnostics.parquet"
    diagnostics_frame.write_csv(diagnostics_csv)
    diagnostics_frame.write_parquet(diagnostics_parquet, compression="zstd")

    summary_payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "config": {
            "dataset_path": str(dataset_path.resolve()),
            "start": start.isoformat() if start else None,
            "end": end.isoformat() if end else None,
            "sectors": list(sectors),
            "factor_columns": list(factor_columns),
            "thresholds": asdict(thresholds),
        },
        "summary_stats": summary.summary_stats,
        "acceptance_status": summary.acceptance_status,
        "n_sectors": len(diagnostics_records),
    }

    summary_path = run_directory / "phase2_regression_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    LOGGER.info(
        "phase2_regression_diagnostics_completed",
        run_directory=str(run_directory.resolve()),
        diagnostics_path=str(diagnostics_csv.resolve()),
        summary_path=str(summary_path.resolve()),
        overall_pass=summary.acceptance_status.get("overall"),
    )
    return run_directory


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the diagnostics runner."""
    parser = argparse.ArgumentParser(description="Run Phase 2 regression diagnostics for the 3D risk model.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path containing sector_returns.parquet and factor_returns.parquet (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store diagnostics artefacts (default: %(default)s)",
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Optional inclusive start date (ISO-8601). Defaults to earliest available.",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="Optional inclusive end date (ISO-8601). Defaults to latest available.",
    )
    parser.add_argument(
        "--sectors",
        nargs="+",
        default=DEFAULT_SECTORS,
        help="Sector tickers to evaluate (default: canonical Phase 2 universe).",
    )
    parser.add_argument(
        "--factor-columns",
        nargs="+",
        default=DEFAULT_FACTORS,
        help="Factor column names to regress against (default: duration, credit, liquidity).",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        help="Optional identifier for the output directory. Defaults to current UTC timestamp.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint."""
    args = parse_args(argv)
    start = _parse_iso_datetime(args.start)
    end = _parse_iso_datetime(args.end)
    run_tag = args.run_tag
    try:
        run_phase2_regression_diagnostics(
            args.dataset_path,
            args.output_dir,
            start=start,
            end=end,
            sectors=tuple(args.sectors),
            factor_columns=tuple(args.factor_columns),
            run_tag=run_tag,
        )
    except Exception:
        LOGGER.exception(
            "phase2_regression_diagnostics_failed",
            dataset_path=str(args.dataset_path.resolve()),
            output_dir=str(args.output_dir.resolve()),
        )
        raise


if __name__ == "__main__":
    main()
