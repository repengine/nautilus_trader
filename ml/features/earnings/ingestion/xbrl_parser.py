#!/usr/bin/env python3
"""
XBRL Parser utilities for SEC EDGAR earnings data.

Provides utilities for parsing XBRL (eXtensible Business Reporting Language) financial
data from SEC filings. Handles common XBRL tags and fallback strategies for non-standard
implementations.

Performance targets: Parsing <100ms per filing
Hot/Cold path separation: All parsing is cold-path only

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Final


# ===== Constants =====
# Standard US-GAAP XBRL tags for earnings data
XBRL_TAGS: Final[dict[str, str]] = {
    "eps_basic": "us-gaap:EarningsPerShareBasic",
    "eps_diluted": "us-gaap:EarningsPerShareDiluted",
    "revenue": "us-gaap:Revenues",
    "net_income": "us-gaap:NetIncomeLoss",
    "operating_income": "us-gaap:OperatingIncomeLoss",
    "shares_outstanding": "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding",
    "shares_basic": "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic",
}

# Alternative XBRL tags (fallbacks for non-standard filers)
ALTERNATIVE_TAGS: Final[dict[str, list[str]]] = {
    "revenue": [
        "us-gaap:Revenues",
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap:SalesRevenueNet",
        "us-gaap:SalesRevenueGoodsNet",
    ],
    "net_income": [
        "us-gaap:NetIncomeLoss",
        "us-gaap:ProfitLoss",
        "us-gaap:NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class XBRLParser:
    """
    Parser for XBRL financial data from SEC filings.

    This class provides static methods for extracting financial metrics from
    XBRL-formatted documents, with fallback strategies for non-standard implementations.

    Methods
    -------
    extract_eps
        Extract EPS (basic and diluted) from XBRL facts
    extract_revenue
        Extract revenue from XBRL facts with fallback tags
    extract_net_income
        Extract net income from XBRL facts with fallback tags
    extract_shares_outstanding
        Extract shares outstanding from XBRL facts
    safe_extract
        Safely extract a value with type conversion and error handling

    """

    @staticmethod
    def extract_eps(facts: dict[str, Any], prefer_diluted: bool = True) -> float | None:
        """
        Extract earnings per share from XBRL facts.

        Parameters
        ----------
        facts : dict[str, Any]
            XBRL facts dictionary from edgartools
        prefer_diluted : bool, default=True
            Whether to prefer diluted EPS over basic EPS

        Returns
        -------
        float | None
            EPS value or None if not found

        """
        if prefer_diluted:
            eps_diluted = XBRLParser.safe_extract(
                facts,
                XBRL_TAGS["eps_diluted"],
                float,
            )
            if eps_diluted is not None:
                return eps_diluted

        # Fallback to basic EPS
        eps_basic = XBRLParser.safe_extract(
            facts,
            XBRL_TAGS["eps_basic"],
            float,
        )
        return eps_basic

    @staticmethod
    def extract_revenue(facts: dict[str, Any]) -> float | None:
        """
        Extract revenue from XBRL facts with fallback tags.

        Parameters
        ----------
        facts : dict[str, Any]
            XBRL facts dictionary from edgartools

        Returns
        -------
        float | None
            Revenue value or None if not found

        """
        # Try all alternative tags
        for tag in ALTERNATIVE_TAGS["revenue"]:
            revenue = XBRLParser.safe_extract(facts, tag, float)
            if revenue is not None:
                return revenue

        logger.debug("Revenue not found in XBRL facts with standard tags")
        return None

    @staticmethod
    def extract_net_income(facts: dict[str, Any]) -> float | None:
        """
        Extract net income from XBRL facts with fallback tags.

        Parameters
        ----------
        facts : dict[str, Any]
            XBRL facts dictionary from edgartools

        Returns
        -------
        float | None
            Net income value or None if not found

        """
        # Try all alternative tags
        for tag in ALTERNATIVE_TAGS["net_income"]:
            net_income = XBRLParser.safe_extract(facts, tag, float)
            if net_income is not None:
                return net_income

        logger.debug("Net income not found in XBRL facts with standard tags")
        return None

    @staticmethod
    def extract_shares_outstanding(facts: dict[str, Any]) -> int | None:
        """
        Extract shares outstanding from XBRL facts.

        Parameters
        ----------
        facts : dict[str, Any]
            XBRL facts dictionary from edgartools

        Returns
        -------
        int | None
            Shares outstanding or None if not found

        """
        # Try diluted shares first
        shares = XBRLParser.safe_extract(
            facts,
            XBRL_TAGS["shares_outstanding"],
            int,
        )
        if shares is not None and isinstance(shares, int):
            return shares

        # Fallback to basic shares
        shares_basic = XBRLParser.safe_extract(
            facts,
            XBRL_TAGS["shares_basic"],
            int,
        )
        if shares_basic is not None and isinstance(shares_basic, int):
            return shares_basic
        return None

    @staticmethod
    def safe_extract(
        facts: dict[str, Any],
        tag: str,
        target_type: type[float | int],
    ) -> float | int | None:
        """
        Safely extract a value from XBRL facts with type conversion.

        Parameters
        ----------
        facts : dict[str, Any]
            XBRL facts dictionary
        tag : str
            XBRL tag to extract
        target_type : type[float] | type[int]
            Target type for conversion

        Returns
        -------
        float | int | None
            Extracted and converted value, or None if not found or conversion fails

        """
        try:
            value = facts.get(tag)
            if value is None:
                return None

            # Handle list of values (take most recent)
            if isinstance(value, list) and len(value) > 0:
                value = value[-1]

            # Handle dict with 'value' key
            if isinstance(value, dict):
                value = value.get("value")

            if value is None:
                return None

            # Type conversion
            if target_type is float:
                return float(value)
            elif target_type is int:
                return int(value)
            else:
                # Unexpected type, return None
                return None

        except (ValueError, TypeError, KeyError) as e:
            logger.debug(
                "Failed to extract XBRL tag %s: %s",
                tag,
                e,
                exc_info=True,
            )
            return None


__all__ = [
    "ALTERNATIVE_TAGS",
    "XBRL_TAGS",
    "XBRLParser",
]
