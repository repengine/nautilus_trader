"""
CoveragePolicy — subscription-bound lookback windows for backfill.

Environment overrides:
- ML_L0_LOOKBACK_DAYS (default 2555 ~ 7y)
- ML_L1_LOOKBACK_DAYS (default 365)
- ML_L2_LOOKBACK_DAYS (default 30)
- ML_L3_LOOKBACK_DAYS (default 30)

Use get_max_lookback_days(dataset_id_or_type) to derive the appropriate window.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


__all__ = ["CoveragePolicy", "get_max_lookback_days"]


@dataclass(slots=True, frozen=True)
class CoveragePolicy:
    """Subscription coverage windows in days."""

    l0_max_lookback_days: int = 365 * 7
    l1_max_lookback_days: int = 365
    l2_max_lookback_days: int = 30
    l3_max_lookback_days: int = 30

    @staticmethod
    def from_env() -> CoveragePolicy:
        def _get(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except Exception:
                return default

        return CoveragePolicy(
            l0_max_lookback_days=_get("ML_L0_LOOKBACK_DAYS", 365 * 7),
            l1_max_lookback_days=_get("ML_L1_LOOKBACK_DAYS", 365),
            l2_max_lookback_days=_get("ML_L2_LOOKBACK_DAYS", 30),
            l3_max_lookback_days=_get("ML_L3_LOOKBACK_DAYS", 30),
        )


def get_max_lookback_days(dataset_id_or_type: str, policy: CoveragePolicy | None = None) -> int:
    """
    Map dataset identifier/type to a subscription-bound lookback window (days).

    Mapping:
    - L0: bars → 7 years
    - L1: quotes, trades → 1 year
    - L2/L3: mbp/tbbo/orderbook → 30 days
    Unknown types default to L2 window.
    """
    p = policy or CoveragePolicy.from_env()
    key = dataset_id_or_type.strip().lower()
    if key == "bars":
        return p.l0_max_lookback_days
    if key in {"quotes", "trades"}:
        return p.l1_max_lookback_days
    # L2/L3 families (keep conservative by default)
    if key in {"mbp", "mbp1", "tbbo", "orderbook", "l2", "l3"}:
        return p.l2_max_lookback_days
    # Default conservative window
    return p.l2_max_lookback_days

