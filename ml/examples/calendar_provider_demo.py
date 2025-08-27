#!/usr/bin/env python3
"""
Demonstration of real market calendar provider with pandas_market_calendars.

This script shows how to use the PandasCalendarSource to get accurate market schedules
including trading hours, holidays, and pre/after market sessions.

"""

from __future__ import annotations

from datetime import datetime

from ml._imports import HAS_PANDAS_MARKET_CALENDARS
from ml.data.providers.factory import ProviderFactory
from ml.data.sources.calendar import PandasCalendarSource


def main() -> None:
    """
    Run calendar provider demonstration.
    """
    print("=" * 60)
    print("Market Calendar Provider Demonstration")
    print("=" * 60)

    # Create factory - will automatically use PandasCalendarSource if available
    factory = ProviderFactory()
    _calendar_provider = factory.get_calendar_provider()

    print(f"\nUsing calendar source: {factory._calendar_source.__class__.__name__}")
    print(f"pandas_market_calendars available: {HAS_PANDAS_MARKET_CALENDARS}")

    # Test various scenarios
    test_dates = [
        ("Regular trading day (Tuesday)", datetime(2024, 1, 16, 10, 30)),
        ("Pre-market hours", datetime(2024, 1, 16, 5, 0)),
        ("After-hours trading", datetime(2024, 1, 16, 17, 0)),
        ("Weekend (Saturday)", datetime(2024, 1, 20, 10, 30)),
        ("New Year's Day", datetime(2024, 1, 1, 10, 30)),
        ("Independence Day", datetime(2024, 7, 4, 10, 30)),
        ("Christmas", datetime(2024, 12, 25, 10, 30)),
    ]

    print("\n" + "=" * 60)
    print("NYSE Market Schedule Tests")
    print("=" * 60)

    for description, dt in test_dates:
        print(f"\n{description}: {dt}")
        print("-" * 40)

        # Get schedule directly from source if using PandasCalendarSource
        if isinstance(factory._calendar_source, PandasCalendarSource):
            schedule = factory._calendar_source.get_schedule(dt, "NYSE")

            print(f"  Trading Day: {schedule.is_trading_day}")
            print(f"  Holiday: {schedule.is_holiday}")
            print(f"  Market Hours: {schedule.is_market_hours}")
            print(f"  Pre-Market: {schedule.is_pre_market}")
            print(f"  After-Hours: {schedule.is_after_hours}")

            if schedule.is_trading_day:
                print(f"  Market Open: {schedule.market_open.time()}")
                print(f"  Market Close: {schedule.market_close.time()}")
                if schedule.is_market_hours:
                    print(f"  Minutes to Close: {schedule.minutes_to_close}")

    # Test different exchanges
    print("\n" + "=" * 60)
    print("Multi-Exchange Support")
    print("=" * 60)

    if isinstance(factory._calendar_source, PandasCalendarSource):
        exchanges = ["NYSE", "NASDAQ", "LSE", "JPX", "CRYPTO"]
        dt = datetime(2024, 1, 16, 10, 30)  # Tuesday

        print(f"\nTesting time: {dt}")
        print("-" * 40)

        for exchange in exchanges:
            try:
                schedule = factory._calendar_source.get_schedule(dt, exchange)
                print(
                    f"\n{exchange:8} - Trading: {schedule.is_trading_day}, "
                    f"Market Hours: {schedule.is_market_hours}",
                )
            except Exception as e:
                print(f"\n{exchange:8} - Error: {e}")

    # Show supported exchanges
    if isinstance(factory._calendar_source, PandasCalendarSource):
        print("\n" + "=" * 60)
        print("Supported Exchange Codes")
        print("=" * 60)

        supported = factory._calendar_source.get_supported_exchanges()
        print(f"\nTotal supported exchanges: {len(supported)}")

        # Group by region
        us_exchanges = [
            e for e in supported if e in ["NYSE", "NASDAQ", "CME", "CBOT", "ICE", "CBOE"]
        ]
        eu_exchanges = [e for e in supported if e in ["LSE", "EUREX", "XETR"]]
        asia_exchanges = [e for e in supported if e in ["JPX", "HKEX", "SSE", "ASX"]]
        crypto_exchanges = [
            e for e in supported if "CRYPTO" in e or "BINANCE" in e or "COINBASE" in e
        ]

        print(f"\nUS Exchanges: {', '.join(us_exchanges)}")
        print(f"European Exchanges: {', '.join(eu_exchanges)}")
        print(f"Asian Exchanges: {', '.join(asia_exchanges)}")
        print(f"Crypto Exchanges: {', '.join(crypto_exchanges)}")

    # Test holiday detection
    if isinstance(factory._calendar_source, PandasCalendarSource) and HAS_PANDAS_MARKET_CALENDARS:
        print("\n" + "=" * 60)
        print("Holiday Detection for NYSE in 2024")
        print("=" * 60)

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)

        holidays = factory._calendar_source.get_holidays("NYSE", start_date, end_date)

        if holidays:
            print(f"\nFound {len(holidays)} holidays in 2024:")
            for i, holiday in enumerate(holidays[:10]):  # Show first 10
                print(f"  {i+1}. {holiday.date()}")
            if len(holidays) > 10:
                print(f"  ... and {len(holidays) - 10} more")
        else:
            print("\nNo holidays found (may be using fallback source)")

    print("\n" + "=" * 60)
    print("Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
