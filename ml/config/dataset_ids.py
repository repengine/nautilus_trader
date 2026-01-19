"""
Dataset ID constants for ML module.

These constants define canonical dataset identifiers used across the ML infrastructure.
All dataset references should use these constants rather than hardcoded strings to ensure
consistency and prevent typos. Constants are marked as Final to prevent reassignment.

Constants
---------
EARNINGS_ACTUALS_DATASET_ID : str
    Dataset ID for earnings actuals data.
EARNINGS_ESTIMATES_DATASET_ID : str
    Dataset ID for earnings estimates data.

Usage
-----
Import from ml.config for public API access:

    from ml.config import EARNINGS_ACTUALS_DATASET_ID, EARNINGS_ESTIMATES_DATASET_ID

    manifest = registry.get_manifest(EARNINGS_ACTUALS_DATASET_ID)

"""

from __future__ import annotations

from typing import Final


__all__ = [
    "EARNINGS_ACTUALS_DATASET_ID",
    "EARNINGS_ESTIMATES_DATASET_ID",
    "EQUS_MINI_DATASET_ID",
    "EVENTS_CALENDAR_DATASET_ID",
    "FEATURE_VALUES_DATASET_ID",
    "L2_MINUTE_DATASET_ID",
    "MACRO_OBSERVATIONS_DATASET_ID",
    "MACRO_RELEASES_DATASET_ID",
    "MICRO_MINUTE_DATASET_ID",
]

# =============================================================================
# EARNINGS DATASET IDs
# =============================================================================

EARNINGS_ACTUALS_DATASET_ID: Final[str] = "ml.earnings_actuals"
"""Dataset identifier for earnings actuals data."""

EARNINGS_ESTIMATES_DATASET_ID: Final[str] = "ml.earnings_estimates"
"""Dataset identifier for earnings estimates data."""

MACRO_RELEASES_DATASET_ID: Final[str] = "ml.macro_release_calendar"
"""Dataset identifier for macro release calendar data."""

MACRO_OBSERVATIONS_DATASET_ID: Final[str] = "ml.macro_observations"
"""Dataset identifier for macro observation long-format data."""

EVENTS_CALENDAR_DATASET_ID: Final[str] = "ml.events_calendar"
"""Dataset identifier for normalized events/calendar features."""

FEATURE_VALUES_DATASET_ID: Final[str] = "ml.feature_values"
"""Dataset identifier for computed FeatureStore values."""

MICRO_MINUTE_DATASET_ID: Final[str] = "ml.microstructure_minute"
"""Dataset identifier for aggregated microstructure per-minute features."""

L2_MINUTE_DATASET_ID: Final[str] = "ml.l2_minute"
"""Dataset identifier for aggregated L2 depth per-minute features."""

EQUS_MINI_DATASET_ID: Final[str] = "EQUS.MINI"
"""Dataset identifier for Databento EQUS aggregated minute bars."""
