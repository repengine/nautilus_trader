"""
CoveragePolicy — subscription-bound lookback windows for backfill.

DEPRECATED: This module has been consolidated into ml.data.ingest.subscription.
Use SubscriptionPolicy from ml.data.ingest.subscription instead.

This module remains for backwards compatibility and will redirect all calls
to the new unified subscription module.

Environment overrides:
- ML_L0_LOOKBACK_DAYS (default 2555 ~ 7y)
- ML_L1_LOOKBACK_DAYS (default 365)
- ML_L2_LOOKBACK_DAYS (default 30)
- ML_L3_LOOKBACK_DAYS (default 30)

Use get_max_lookback_days(dataset_id_or_type) to derive the appropriate window.
"""

from __future__ import annotations

import warnings

from ml.data.ingest.subscription import SubscriptionPolicy as _SubscriptionPolicy
from ml.data.ingest.subscription import get_max_lookback_days as _get_max_lookback_days


__all__ = ["CoveragePolicy", "get_max_lookback_days"]


# Emit deprecation warning on module import
warnings.warn(
    "ml.config.coverage is deprecated. Use ml.data.ingest.subscription.SubscriptionPolicy instead.",
    DeprecationWarning,
    stacklevel=2,
)


# Type alias for backwards compatibility
CoveragePolicy = _SubscriptionPolicy


def get_max_lookback_days(
    dataset_id_or_type: str,
    policy: CoveragePolicy | None = None,
) -> int:
    """
    Map dataset identifier/type to a subscription-bound lookback window (days).

    DEPRECATED: Use ml.data.ingest.subscription.get_max_lookback_days instead.

    Mapping:
    - L0: bars → 7 years
    - L1: quotes, trades → 1 year
    - L2/L3: mbp/tbbo/orderbook → 30 days
    Unknown types default to L2 window.

    Parameters
    ----------
    dataset_id_or_type : str
        Dataset identifier or type (bars, quotes, trades, etc.).
    policy : CoveragePolicy | None, optional
        Policy to use. If None, constructs from environment.

    Returns
    -------
    int
        Maximum lookback days for the specified level.

    """
    return _get_max_lookback_days(dataset_id_or_type, policy)
