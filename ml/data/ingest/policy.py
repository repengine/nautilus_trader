"""
Databento coverage policy guard.

DEPRECATED: This module has been consolidated into ml.data.ingest.subscription.
Use SubscriptionPolicy from ml.data.ingest.subscription instead.

This module remains for backwards compatibility and will redirect all calls
to the new unified subscription module.

Enable via environment variables (all optional; disabled by default):

- DATABENTO_ALLOWED_DATASETS: comma-separated dataset names (e.g., "EQUS.MINI,XNAS.ITCH")
- DATABENTO_ALLOWED_SCHEMAS: comma-separated schemas (e.g., "ohlcv-1m,mbp-1,tbbo,trades")
- DATABENTO_ALLOWED_SYMBOLS: comma-separated symbols (optional strict allowlist)
- DATABENTO_MAX_DAYS: maximum days per request window (integer)
- DATABENTO_MAX_DAYS_BY_SCHEMA: per-schema max days CSV (e.g.,
  "ohlcv-1m:3650,tbbo:365,trades:365,bbo:365,mbp-1:365,mbp-10:31,mbo:31,imbalance:31")
- DATABENTO_EARLIEST_DATE: ISO date (YYYY-MM-DD) earliest allowed start
- DATABENTO_LATEST_DATE: ISO date (YYYY-MM-DD) latest allowed end
- DATABENTO_MAX_SYMBOLS: maximum number of symbols per request (integer)
- DATABENTO_POLICY_STRICT: "1" to raise on violations, "0" to clamp/filter (default "0")

This module is cold-path only and safe to import from CLIs or ingest helpers.

"""

from __future__ import annotations

import warnings

from ml.data.ingest.subscription import SubscriptionPolicy as _SubscriptionPolicy


__all__ = ["DatabentoCoveragePolicy"]


# Emit deprecation warning on module import
warnings.warn(
    "ml.data.ingest.policy.DatabentoCoveragePolicy is deprecated. "
    "Use ml.data.ingest.subscription.SubscriptionPolicy instead.",
    DeprecationWarning,
    stacklevel=2,
)


# Type alias for backwards compatibility
DatabentoCoveragePolicy = _SubscriptionPolicy
