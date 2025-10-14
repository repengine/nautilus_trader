"""
Trading day calculator for DataScheduler.

This module provides business day calculation logic for determining the previous trading
day based on current date.

"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import Protocol


class TradingDayCalculatorProtocol(Protocol):
    """
    Protocol for trading day calculation operations.
    """

    def get_previous_trading_day(
        self,
        reference_date: datetime | None = None,
    ) -> datetime:
        """
        Get the previous trading day.

        Parameters
        ----------
        reference_date : datetime | None
            Reference date for calculation. If None, uses current time.

        Returns
        -------
        datetime
            Previous trading day

        """
        ...


class TradingDayCalculator:
    """
    Calculate trading days and business day logic.

    Implements Pattern 3: Hot/Cold Path Separation
    This is a cold path utility - no hot path performance requirements.

    This component is responsible ONLY for trading day calculations.

    """

    def get_previous_trading_day(
        self,
        reference_date: datetime | None = None,
    ) -> datetime:
        """
        Get the previous trading day based on reference date.

        Logic:
        - Monday: Returns previous Friday (3 days back)
        - Sunday: Returns previous Friday (2 days back)
        - Other days: Returns previous day (1 day back)

        Parameters
        ----------
        reference_date : datetime | None
            Reference date for calculation. If None, uses datetime.now().

        Returns
        -------
        datetime
            Previous trading day

        """
        today = reference_date or datetime.now(tz=UTC)

        if today.weekday() == 0:  # Monday
            # Get Friday's data
            return today - timedelta(days=3)
        elif today.weekday() == 6:  # Sunday
            # Get Friday's data
            return today - timedelta(days=2)
        else:
            # Get previous day's data
            return today - timedelta(days=1)
