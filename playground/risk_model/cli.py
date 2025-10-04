"""Command-line runner for the playground 3D risk pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from collections.abc import Mapping
from datetime import UTC
from datetime import datetime
from pathlib import Path

from playground.exposure.persistence import CrossAssetBetaPersistenceConfig
from playground.risk_model import RiskPipelineConfig
from playground.risk_model import run_risk_pipeline
from playground.risk_model.dataset import CoverageSummary
from playground.risk_model.fetchers import SectorFetcherConfig


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

DEFAULT_FACTORS = (
    "factor_duration",
    "factor_credit",
    "factor_liquidity",
)


def _parse_date(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    return dt


def _parse_sector_overrides(values: Iterable[str]) -> dict[str, tuple[str, ...]]:
    overrides: dict[str, tuple[str, ...]] = {}
    for raw in values:
        if "=" not in raw:
            msg = f"Invalid sector proxy specification '{raw}'. Expected format SECTOR=TICK1,TICK2"
            raise ValueError(msg)
        sector, tickers_raw = raw.split("=", 1)
        tickers = tuple(ticker.strip() for ticker in tickers_raw.split(",") if ticker.strip())
        if not tickers:
            msg = f"No tickers provided for sector '{sector}'"
            raise ValueError(msg)
        overrides[sector.strip()] = tickers
    return overrides


def _parse_weight_caps(values: Iterable[str]) -> dict[str, float]:
    caps: dict[str, float] = {}
    for raw in values:
        if "=" not in raw:
            msg = f"Invalid weight cap specification '{raw}'. Expected SECTOR=0.25"
            raise ValueError(msg)
        sector, value = raw.split("=", 1)
        sector_key = sector.strip()
        try:
            cap = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid weight cap '{value}' for sector '{sector_key}'") from exc
        if cap <= 0:
            raise ValueError(f"Weight cap must be positive for sector '{sector_key}'")
        caps[sector_key] = cap
    return caps


def _format_ratio(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def _format_alert_bucket(name: str, bucket: Mapping[str, float]) -> str:
    if not bucket:
        return f"{name}=ok"
    parts = [
        f"{series} ({_format_ratio(ratio)})"
        for series, ratio in sorted(bucket.items())
    ]
    joined = ", ".join(parts)
    return f"{name}: {joined}"


def _format_coverage_snapshot(summary: CoverageSummary) -> str:
    composite_count = len(summary.composite_coverage)
    return (
        "[coverage] "
        f"calendar={summary.calendar_name}, "
        f"sector_days={summary.sector_expected_days}, "
        f"factor_days={summary.factor_expected_days}, "
        f"composites_tracked={composite_count}"
    )


def _format_alert_snapshot(alerts: Mapping[str, Mapping[str, float]]) -> str:
    sector_bucket = _format_alert_bucket("sector", alerts.get("sector", {}))
    factor_bucket = _format_alert_bucket("factor", alerts.get("factor", {}))
    composite_bucket = _format_alert_bucket("composite", alerts.get("composite", {}))
    return f"[alerts] {sector_bucket} | {factor_bucket} | {composite_bucket}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the playground 3D risk pipeline")
    parser.add_argument("--start", default="1970-01-01", help="Start date (ISO-8601, default 1970-01-01)")
    parser.add_argument("--end", default=datetime.now(tz=UTC).date().isoformat(), help="End date (ISO-8601, default today UTC)")
    parser.add_argument("--sectors", nargs="+", default=DEFAULT_SECTORS, help="One or more sector tickers")
    parser.add_argument(
        "--factor-columns",
        nargs="+",
        default=DEFAULT_FACTORS,
        help="Factor column names (default duration, credit, liquidity)",
    )
    parser.add_argument("--feature-set-id", default="playground-sector-risk", help="Feature set identifier")
    parser.add_argument("--min-weight", type=float, default=0.01, help="Minimum weight clamp applied during optimisation")
    parser.add_argument("--max-weight", type=float, help="Optional maximum weight applied to each sector")
    parser.add_argument("--persist-dir", type=Path, default=Path("playground/data/sector_dataset"))
    parser.add_argument("--cache-dir", type=Path, default=Path("playground/data/cache"))
    parser.add_argument("--visualization-dir", type=Path, default=Path("playground/data/visualizations"))
    parser.add_argument(
        "--sector-proxy",
        action="append",
        default=[],
        help="Sector proxy mapping in the form SECTOR=TICK1,TICK2 (can be supplied multiple times)",
    )
    parser.add_argument(
        "--weight-cap",
        action="append",
        default=[],
        help="Per-sector weight cap in the form SECTOR=0.25 (can be supplied multiple times)",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.85,
        help="Minimum acceptable coverage ratio before falling back to proxy tickers",
    )
    parser.add_argument(
        "--calendar",
        default="XNYS",
        help="Trading calendar identifier used for expected session counts (default XNYS)",
    )
    parser.add_argument(
        "--min-sector-coverage",
        type=float,
        default=0.8,
        help="Minimum required sector coverage ratio (0-1, default 0.8)",
    )
    parser.add_argument(
        "--min-factor-coverage",
        type=float,
        default=0.8,
        help="Minimum required factor coverage ratio (0-1, default 0.8)",
    )
    parser.add_argument(
        "--coverage-report",
        type=Path,
        help="Optional path to write coverage summary and alert diagnostics as JSON",
    )
    parser.add_argument(
        "--persist-betas",
        action="store_true",
        help="Persist EWMA betas into ml_cross_asset_betas",
    )
    parser.add_argument(
        "--beta-connection",
        help="Explicit PostgreSQL connection used for beta persistence",
    )
    parser.add_argument(
        "--beta-chunk-size",
        type=int,
        default=1000,
        help="Chunk size used when persisting betas",
    )

    args = parser.parse_args()

    overrides = _parse_sector_overrides(args.sector_proxy)
    weight_caps = _parse_weight_caps(args.weight_cap)
    sector_config = SectorFetcherConfig(
        ticker_overrides=overrides or None,
        min_coverage_ratio=args.min_coverage,
    )

    beta_persistence = CrossAssetBetaPersistenceConfig(
        enabled=args.persist_betas,
        connection_url=args.beta_connection,
        chunk_size=max(args.beta_chunk_size, 1),
    )

    config = RiskPipelineConfig(
        sectors=tuple(args.sectors),
        factor_columns=tuple(args.factor_columns),
        start=_parse_date(args.start),
        end=_parse_date(args.end),
        feature_set_id=args.feature_set_id,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        weight_caps=weight_caps or None,
        persist_dir=args.persist_dir,
        cache_dir=args.cache_dir,
        visualization_dir=args.visualization_dir,
        sector_fetcher_config=sector_config,
        calendar=args.calendar,
        min_sector_coverage=args.min_sector_coverage,
        min_factor_coverage=args.min_factor_coverage,
        beta_persistence=beta_persistence,
    )

    result = run_risk_pipeline(config)
    summary_payload = {
        "coverage": result.coverage_summary.to_dict(),
        "coverage_alerts": result.coverage_alerts,
    }

    if args.coverage_report is not None:
        args.coverage_report.parent.mkdir(parents=True, exist_ok=True)
        args.coverage_report.write_text(
            json.dumps(summary_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    print(
        "Completed pipeline",
        {
            "profiles": len(result.profiles),
            "coverage": summary_payload["coverage"],
            "coverage_alerts": summary_payload["coverage_alerts"],
            "eigenvalue_trends": result.eigenvalue_trends,
        },
    )
    print(_format_coverage_snapshot(result.coverage_summary))
    print(_format_alert_snapshot(result.coverage_alerts))

    if result.optimizer_recommendations:
        latest_year = max(result.optimizer_recommendations)
        weights = result.optimizer_recommendations[latest_year]
        formatted = ", ".join(
            f"{sector}={weight:.2%}" for sector, weight in sorted(weights.items())
        )
        print(
            "[optimizer]",
            {
                "year": latest_year,
                "weights": formatted,
                "beta_rows": result.beta_persisted_rows,
            },
        )
    else:
        print("[optimizer] no profiles")


if __name__ == "__main__":
    main()
