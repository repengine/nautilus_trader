from __future__ import annotations

"""
Centralized symbol universes for ML workflows.

This module collects commonly used universes (Tier1/2/3, supplementary ETFs,
etc.) to avoid duplication across CLIs and scripts. Callers should import from
here rather than hardcoding lists locally.
"""

# NOTE: Initial seed; populate/extend as needed. Keeping minimal in Phase 1 to
# avoid changing behavior in existing CLIs/tests.

TIER1_CORE: list[str] = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "VTI",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "TSLA",
    "AMD",
]

SUPPLEMENTARY_ETFS: dict[str, list[str]] = {
    "sectors": [
        "XLK",
        "XLF",
        "XLV",
        "XLE",
        "XLI",
        "XLY",
        "XLP",
        "XLB",
        "XLRE",
        "XLU",
        "XLC",
    ],
    "bonds": ["SHY", "IEF", "TLT", "TIP", "LQD", "HYG", "EMB", "AGG"],
    "commodities": ["GLD", "SLV", "USO", "UNG", "DBA", "DBB", "DBC"],
}

__all__ = [
    "SUPPLEMENTARY_ETFS",
    "TIER1_CORE",
]
