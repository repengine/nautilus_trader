#!/usr/bin/env python3
"""
Configuration definitions for earnings ingestion workflows.

This module centralizes tunables used by the automated earnings ingestion
pipeline so that operational scripts and services can be configured without
hard-coded constants. The defaults are aligned with Tier-1 production usage
but callers are expected to construct an :class:`EarningsIngestionConfig`
explicitly (e.g., from CLI arguments or environment variables).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Final

from ml.config.ingestion_windows import WatermarkWindowConfig
from ml.config.ingestion_windows import earnings_window_defaults
from ml.config.sec_identity import SecIdentityConfig
from ml.config.universes import TIER1_FULL_95


DEFAULT_PARQUET_ROOT: Final[Path] = Path("data/features/earnings_raw")
DEFAULT_SKIP_ACTUALS_TICKERS: Final[tuple[str, ...]] = (
    "DIA",
    "EEM",
    "EFA",
    "FXE",
    "GLD",
    "IWM",
    "QQQ",
    "SLV",
    "SPY",
    "TLT",
    "UNG",
    "USO",
    "UUP",
    "VIXY",
    "VNQ",
    "VNQI",
    "VTI",
    "VWO",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLV",
)


@dataclass(frozen=True)
class EarningsIngestionConfig:
    """
    Runtime configuration for the earnings ingestion pipeline.

    Attributes
    ----------
    postgres_dsn:
        Connection string used for the primary PostgreSQL store.
    parquet_root:
        Root directory used for Parquet mirror writes. The writer will create
        sub-directories per dataset type underneath this root.
    universe_mode:
        Source for the ingestion universe. ``"postgres"`` queries the live
        instrument metadata tables, while ``"tier1_full"`` reuses the static
        Tier-1 universe baked into :mod:`ml.config.universes`.
    fallback_symbols:
        Symbols used when universe discovery fails. Defaults to the Tier-1
        production universe.
    override_symbols:
        Optional explicit override of the ingestion universe. When provided,
        ``universe_mode`` is ignored.
    skip_actuals:
        Tickers that should skip EDGAR fetching (e.g., ETFs lacking filings).
    edgar_quarters:
        Number of quarters to request per ticker from EDGAR.
    edgar_rate_limit:
        Delay in seconds between EDGAR API calls.
    edgar_max_retries:
        Maximum retries for EDGAR requests.
    yahoo_rate_limit:
        Delay in seconds between Yahoo Finance requests.
    yahoo_max_retries:
        Maximum retries for Yahoo requests.
    enable_yahoo:
        Whether to fetch Yahoo consensus estimates.
    estimate_match_window_days:
        Max days between earnings date and filing date used to align estimates
        to actual period_end values.
    sec_identity:
        Optional SEC identity string passed to the edgartools client.
    parquet_partition_keys:
        Logical partition keys used by the Parquet raw writer.
    watermark_config:
        Watermark window configuration used to scope incremental ingestion.
    edgar_min_quarters:
        Minimum number of EDGAR quarters to request when using watermarks.
    edgar_max_quarters:
        Maximum number of EDGAR quarters to request when using watermarks.
    edgar_quarter_days:
        Days per quarter used to convert watermark ranges into quarters.
    """

    postgres_dsn: str
    parquet_root: Path = DEFAULT_PARQUET_ROOT
    universe_mode: str = "postgres"
    fallback_symbols: tuple[str, ...] = field(default_factory=lambda: tuple(TIER1_FULL_95))
    override_symbols: tuple[str, ...] | None = None
    skip_actuals: tuple[str, ...] = DEFAULT_SKIP_ACTUALS_TICKERS
    edgar_quarters: int = 8
    edgar_rate_limit: float = 1.0
    edgar_max_retries: int = 3
    yahoo_rate_limit: float = 0.5
    yahoo_max_retries: int = 3
    enable_yahoo: bool = True
    estimate_match_window_days: int = 14
    sec_identity: str | None = None
    parquet_partition_keys: tuple[str, ...] = ("ticker",)
    watermark_config: WatermarkWindowConfig = field(default_factory=earnings_window_defaults)
    edgar_min_quarters: int = 1
    edgar_max_quarters: int | None = None
    edgar_quarter_days: int = 90

    def __post_init__(self) -> None:
        """
        Validate and normalize ingestion configuration.
        """
        if self.edgar_quarters < 1:
            raise ValueError("edgar_quarters must be >= 1")
        if self.edgar_min_quarters < 1:
            raise ValueError("edgar_min_quarters must be >= 1")
        if self.edgar_quarter_days < 1:
            raise ValueError("edgar_quarter_days must be >= 1")
        max_quarters = self.edgar_max_quarters
        if max_quarters is None:
            max_quarters = self.edgar_quarters
            object.__setattr__(self, "edgar_max_quarters", max_quarters)
        if max_quarters < self.edgar_min_quarters:
            raise ValueError("edgar_max_quarters must be >= edgar_min_quarters")

    def resolved_sec_identity(self) -> str | None:
        """
        Resolve the SEC identity string from config or environment.
        """
        if self.sec_identity is not None and self.sec_identity.strip():
            return self.sec_identity.strip()
        return SecIdentityConfig.from_env().resolved_identity()

    def resolved_override(self) -> tuple[str, ...] | None:
        """Return override symbols with normalized casing, if supplied."""
        if self.override_symbols is None:
            return None
        return tuple(symbol.upper() for symbol in self.override_symbols if symbol)

    def normalized_skip_set(self) -> tuple[str, ...]:
        """Return the skip list normalized to uppercase symbols."""
        return tuple(ticker.upper() for ticker in self.skip_actuals)


__all__: tuple[str, ...] = (
    "DEFAULT_PARQUET_ROOT",
    "DEFAULT_SKIP_ACTUALS_TICKERS",
    "EarningsIngestionConfig",
)
