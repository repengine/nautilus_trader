#!/usr/bin/env python3
"""
Mock Databento data source for testing ML pipeline without market hours.

Generates synthetic OHLCV bars with realistic price movements for testing
feature computation, model inference, and signal generation.
"""

import asyncio
import time
from collections.abc import AsyncIterator
from datetime import UTC
from datetime import datetime
from typing import Any

import numpy as np
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity

from nautilus_trader.core import nautilus_pyo3
from nautilus_trader.model.enums import AggregationSource
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import PriceType


_MOCK_RANDOM = np.random.default_rng()
"""Module-level generator to drive synthetic variability for mock data."""


def reseed_mock_random(seed: int | None = None) -> None:
    """
    Reseed the module-level random generator for deterministic test runs.

    Parameters
    ----------
    seed : int | None
        Seed value to initialize the generator. Uses current time when None.
    """
    global _MOCK_RANDOM
    actual_seed = seed if seed is not None else int(time.time() * 1_000)
    _MOCK_RANDOM = np.random.default_rng(actual_seed)


class MockDatabentoGenerator:
    """
    Generate synthetic market data for testing.

    Creates realistic OHLCV bars with:
    - Brownian motion price movements
    - Volume patterns
    - Occasional volatility spikes
    - Support for multiple instruments
    """

    def __init__(
        self,
        instrument_id: InstrumentId,
        bar_type: BarType,
        initial_price: float = 650.0,
        volatility: float = 0.002,
        trend: float = 0.0001,
        volume_mean: float = 1_000_000,
        volume_std: float = 200_000,
    ):
        """
        Initialize the mock data generator.

        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument to generate data for
        bar_type : BarType
            The bar type specification
        initial_price : float
            Starting price (default: 650.0 for SPY-like)
        volatility : float
            Price volatility per bar (default: 0.2%)
        trend : float
            Directional bias (default: 0.01% upward)
        volume_mean : float
            Average volume per bar
        volume_std : float
            Volume standard deviation
        """
        self.instrument_id = instrument_id
        self.bar_type = bar_type
        self.current_price = initial_price
        self.volatility = volatility
        self.trend = trend
        self.volume_mean = volume_mean
        self.volume_std = volume_std
        self.bar_count = 0

        # Track last bar time for proper sequencing
        self.last_ts_event = nautilus_pyo3.millis_to_nanos(
            int(datetime.now(UTC).timestamp() * 1000)
        )

    def generate_bar(self) -> Bar:
        """
        Generate a single synthetic OHLCV bar.

        Returns
        -------
        Bar
            A synthetic bar with realistic price movements
        """
        # Generate price movements
        returns = np.random.normal(self.trend, self.volatility)

        # Occasionally add volatility spikes (5% chance)
        if _MOCK_RANDOM.random() < 0.05:
            returns *= 3.0

        # Calculate OHLC prices
        open_price = self.current_price
        close_price = open_price * (1 + returns)

        # High/Low with realistic wicks
        wick_size = abs(returns) * _MOCK_RANDOM.uniform(0.5, 1.5)
        high_price = max(open_price, close_price) + abs(open_price * wick_size * _MOCK_RANDOM.random())
        low_price = min(open_price, close_price) - abs(open_price * wick_size * _MOCK_RANDOM.random())

        # Generate volume
        volume = max(100, int(np.random.normal(self.volume_mean, self.volume_std)))

        # Update current price for next bar
        self.current_price = close_price

        # Generate timestamps (1 minute apart for 1-MINUTE bars)
        if self.bar_type.spec.aggregation == BarAggregation.MINUTE:
            time_delta_ns = 60_000_000_000  # 60 seconds in nanoseconds
        else:
            time_delta_ns = 1_000_000_000  # 1 second default

        ts_event = self.last_ts_event + time_delta_ns
        ts_init = ts_event + int(_MOCK_RANDOM.integers(1000, 5001))  # Small processing delay

        self.last_ts_event = ts_event
        self.bar_count += 1

        # Create the bar
        bar = Bar(
            bar_type=self.bar_type,
            open=Price.from_str(f"{open_price:.2f}"),
            high=Price.from_str(f"{high_price:.2f}"),
            low=Price.from_str(f"{low_price:.2f}"),
            close=Price.from_str(f"{close_price:.2f}"),
            volume=Quantity.from_int(volume),
            ts_event=ts_event,
            ts_init=ts_init,
        )

        return bar

    async def generate_stream(self, rate_hz: float = 1.0, duration_seconds: float = 60.0) -> AsyncIterator[Bar]:
        """
        Generate a stream of bars at specified rate.

        Parameters
        ----------
        rate_hz : float
            Bars per second generation rate
        duration_seconds : float
            How long to generate data for

        Yields
        ------
        Bar
            Synthetic bars at the specified rate
        """
        interval = 1.0 / rate_hz
        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            yield self.generate_bar()
            await asyncio.sleep(interval)


class MockDatabentoClient:
    """
    Mock Databento client for testing without real market data.

    Mimics the Databento Live API but generates synthetic data.
    """

    def __init__(self, instrument_id: str = "SPY.EQUS", enable_logging: bool = True):
        """
        Initialize mock Databento client.

        Parameters
        ----------
        instrument_id : str
            Instrument to generate data for
        enable_logging : bool
            Whether to log generated bars
        """
        self.instrument_id = InstrumentId.from_str(instrument_id)
        self.enable_logging = enable_logging

        # Create bar type for 1-minute bars
        from nautilus_trader.model.data import BarSpecification

        self.bar_type = BarType(
            instrument_id=self.instrument_id,
            bar_spec=BarSpecification(
                step=1,
                aggregation=BarAggregation.MINUTE,
                price_type=PriceType.LAST,
            ),
            aggregation_source=AggregationSource.EXTERNAL,
        )

        # Initialize generator with SPY-like characteristics
        self.generator = MockDatabentoGenerator(
            instrument_id=self.instrument_id,
            bar_type=self.bar_type,
            initial_price=650.0 + _MOCK_RANDOM.uniform(-5, 5),  # Random starting point
            volatility=0.002,  # 0.2% per minute
            trend=0.00005,  # Slight upward bias
            volume_mean=1_000_000,
            volume_std=200_000,
        )

    async def subscribe_bars(self, callback: Any, rate_hz: float = 1.0) -> None:
        """
        Subscribe to synthetic bar stream.

        Parameters
        ----------
        callback : callable
            Function to call with each generated bar
        rate_hz : float
            Rate to generate bars (bars per second)
        """
        print(f"📊 Mock Databento: Starting synthetic data stream for {self.instrument_id}")
        print(f"   Generation rate: {rate_hz} bars/second")

        async for bar in self.generator.generate_stream(rate_hz=rate_hz, duration_seconds=3600):
            if self.enable_logging and self.generator.bar_count <= 5:
                print(f"   Bar #{self.generator.bar_count}: {bar.close} (V: {bar.volume})")

            # Call the callback with the synthetic bar
            if callback:
                callback(bar)

            # Log milestones
            if self.generator.bar_count % 100 == 0:
                print(f"   Generated {self.generator.bar_count} bars...")


def create_test_environment() -> dict[str, str]:
    """
    Create a complete test environment configuration.

    Returns
    -------
    dict
        Environment variables for test mode
    """
    return {
        "USE_MOCK_DATA": "true",
        "MOCK_DATA_RATE": "10",  # 10 bars per second for fast testing
        "USE_TEST_DATABASE": "true",
        "TEST_DB_NAME": "nautilus_test",
        "CLEAR_TEST_DB": "true",  # Start fresh each run
        "LOG_PREDICTIONS": "true",
        "LOG_FEATURES": "true",
        "FEATURE_COMPUTATION_MODE": "fast",  # Skip some expensive features
    }


async def main() -> None:
    """
    Standalone test of mock data generation.
    """
    print("=" * 60)
    print("MOCK DATABENTO DATA GENERATOR TEST")
    print("=" * 60)

    # Create mock client
    mock_client = MockDatabentoClient(instrument_id="SPY.EQUS")

    # Counter for received bars
    bars_received = [0]

    def handle_bar(bar: Bar) -> None:
        bars_received[0] += 1
        if bars_received[0] <= 3:
            print(f"\n✅ Received Bar #{bars_received[0]}:")
            print(f"   OHLC: {bar.open} / {bar.high} / {bar.low} / {bar.close}")
            print(f"   Volume: {bar.volume}")
            print(f"   Timestamp: {datetime.fromtimestamp(bar.ts_event / 1e9)}")

    # Subscribe and generate for 10 seconds
    try:
        await asyncio.wait_for(
            mock_client.subscribe_bars(callback=handle_bar, rate_hz=2.0),
            timeout=10.0
        )
    except TimeoutError:
        pass

    print("\n" + "=" * 60)
    print(f"Test complete! Generated {bars_received[0]} bars in 10 seconds")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
