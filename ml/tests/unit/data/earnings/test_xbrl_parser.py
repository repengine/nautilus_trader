#!/usr/bin/env python3
"""
Unit tests for XBRL parser utilities.

Tests cover:
- Extraction of EPS (basic and diluted)
- Extraction of revenue with fallback tags
- Extraction of net income with fallback tags
- Extraction of shares outstanding
- Safe extraction with type conversion
- Error handling for missing/invalid data

"""

from __future__ import annotations

import pytest

from ml.data.earnings.xbrl_parser import XBRL_TAGS
from ml.data.earnings.xbrl_parser import XBRLParser


class TestXBRLParser:
    """Test suite for XBRLParser."""

    def test_extract_eps_diluted_preferred(self) -> None:
        """Test EPS extraction prefers diluted over basic."""
        facts = {
            XBRL_TAGS["eps_diluted"]: 2.52,
            XBRL_TAGS["eps_basic"]: 2.55,
        }

        eps = XBRLParser.extract_eps(facts, prefer_diluted=True)

        assert eps == 2.52  # Should prefer diluted

    def test_extract_eps_basic_fallback(self) -> None:
        """Test EPS extraction falls back to basic when diluted missing."""
        facts = {
            XBRL_TAGS["eps_basic"]: 2.55,
        }

        eps = XBRLParser.extract_eps(facts, prefer_diluted=True)

        assert eps == 2.55  # Should use basic as fallback

    def test_extract_eps_prefer_basic(self) -> None:
        """Test EPS extraction can prefer basic."""
        facts = {
            XBRL_TAGS["eps_diluted"]: 2.52,
            XBRL_TAGS["eps_basic"]: 2.55,
        }

        eps = XBRLParser.extract_eps(facts, prefer_diluted=False)

        assert eps == 2.55  # Should use basic

    def test_extract_eps_missing(self) -> None:
        """Test EPS extraction returns None when missing."""
        facts = {}

        eps = XBRLParser.extract_eps(facts, prefer_diluted=True)

        assert eps is None

    def test_extract_revenue_standard_tag(self) -> None:
        """Test revenue extraction with standard tag."""
        facts = {
            "us-gaap:Revenues": 94.9e9,
        }

        revenue = XBRLParser.extract_revenue(facts)

        assert revenue == 94.9e9

    def test_extract_revenue_alternative_tag(self) -> None:
        """Test revenue extraction with alternative tag."""
        facts = {
            "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax": 94.9e9,
        }

        revenue = XBRLParser.extract_revenue(facts)

        assert revenue == 94.9e9

    def test_extract_revenue_missing(self) -> None:
        """Test revenue extraction returns None when missing."""
        facts = {}

        revenue = XBRLParser.extract_revenue(facts)

        assert revenue is None

    def test_extract_net_income_standard_tag(self) -> None:
        """Test net income extraction with standard tag."""
        facts = {
            "us-gaap:NetIncomeLoss": 22.9e9,
        }

        net_income = XBRLParser.extract_net_income(facts)

        assert net_income == 22.9e9

    def test_extract_net_income_alternative_tag(self) -> None:
        """Test net income extraction with alternative tag."""
        facts = {
            "us-gaap:ProfitLoss": 22.9e9,
        }

        net_income = XBRLParser.extract_net_income(facts)

        assert net_income == 22.9e9

    def test_extract_net_income_missing(self) -> None:
        """Test net income extraction returns None when missing."""
        facts = {}

        net_income = XBRLParser.extract_net_income(facts)

        assert net_income is None

    def test_extract_shares_outstanding_diluted(self) -> None:
        """Test shares outstanding extraction (diluted)."""
        facts = {
            XBRL_TAGS["shares_outstanding"]: 15000000000,
        }

        shares = XBRLParser.extract_shares_outstanding(facts)

        assert shares == 15000000000

    def test_extract_shares_outstanding_basic_fallback(self) -> None:
        """Test shares outstanding extraction falls back to basic."""
        facts = {
            XBRL_TAGS["shares_basic"]: 15500000000,
        }

        shares = XBRLParser.extract_shares_outstanding(facts)

        assert shares == 15500000000

    def test_extract_shares_outstanding_missing(self) -> None:
        """Test shares outstanding extraction returns None when missing."""
        facts = {}

        shares = XBRLParser.extract_shares_outstanding(facts)

        assert shares is None

    def test_safe_extract_float(self) -> None:
        """Test safe extraction with float conversion."""
        facts = {
            "test_tag": "2.52",
        }

        value = XBRLParser.safe_extract(facts, "test_tag", float)

        assert value == 2.52
        assert isinstance(value, float)

    def test_safe_extract_int(self) -> None:
        """Test safe extraction with int conversion."""
        facts = {
            "test_tag": "15000000000",
        }

        value = XBRLParser.safe_extract(facts, "test_tag", int)

        assert value == 15000000000
        assert isinstance(value, int)

    def test_safe_extract_list_value(self) -> None:
        """Test safe extraction handles list (takes last value)."""
        facts = {
            "test_tag": [2.40, 2.45, 2.52],
        }

        value = XBRLParser.safe_extract(facts, "test_tag", float)

        assert value == 2.52  # Should take last value

    def test_safe_extract_dict_value(self) -> None:
        """Test safe extraction handles dict with 'value' key."""
        facts = {
            "test_tag": {"value": 2.52, "unit": "USD"},
        }

        value = XBRLParser.safe_extract(facts, "test_tag", float)

        assert value == 2.52

    def test_safe_extract_missing_tag(self) -> None:
        """Test safe extraction returns None for missing tag."""
        facts = {}

        value = XBRLParser.safe_extract(facts, "missing_tag", float)

        assert value is None

    def test_safe_extract_invalid_conversion(self) -> None:
        """Test safe extraction returns None for invalid conversion."""
        facts = {
            "test_tag": "not_a_number",
        }

        value = XBRLParser.safe_extract(facts, "test_tag", float)

        assert value is None  # Should handle ValueError gracefully

    def test_safe_extract_none_value(self) -> None:
        """Test safe extraction handles None values."""
        facts = {
            "test_tag": None,
        }

        value = XBRLParser.safe_extract(facts, "test_tag", float)

        assert value is None
