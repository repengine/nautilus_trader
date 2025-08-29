#!/usr/bin/env python3

"""
Mock service implementations for ML module testing.

This module provides mock implementations of external services:
- Databento API mock
- FRED API mock
- Yahoo Finance mock
- Redis mock
- PostgreSQL mock for unit tests
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import Mock

import numpy as np
import pandas as pd


class MockDatabentoClient:
    """Mock Databento client for testing."""

    def __init__(self, api_key: str = "test_key", fail_on_request: bool = False):
        """
        Initialize mock Databento client.

        Parameters
        ----------
        api_key : str
            Mock API key
        fail_on_request : bool
            If True, simulate API failures

        """
        self.api_key = api_key
        self.fail_on_request = fail_on_request
        self.request_count = 0
        self.rate_limit_hits = 0

        # Mock datasets
        self.datasets = {
            "XNAS.ITCH": {
                "start": "2022-01-01T00:00:00Z",
                "end": "2024-12-31T23:59:59Z",
                "symbols": ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA"],
            },
            "GLBX.MDP3": {
                "start": "2022-01-01T00:00:00Z",
                "end": "2024-12-31T23:59:59Z",
                "symbols": ["ES", "NQ", "RTY", "CL", "GC", "ZB"],
            },
        }

    def list_datasets(self) -> list[str]:
        """List available datasets."""
        if self.fail_on_request:
            raise ConnectionError("Mock connection error")
        self.request_count += 1
        return list(self.datasets.keys())

    def get_dataset_range(self, dataset: str) -> dict[str, str]:
        """Get date range for dataset."""
        if self.fail_on_request:
            raise ConnectionError("Mock connection error")
        if dataset not in self.datasets:
            raise ValueError(f"Dataset {dataset} not found")

        self.request_count += 1
        return {
            "start": self.datasets[dataset]["start"],
            "end": self.datasets[dataset]["end"],
        }

    def get_data(
        self,
        dataset: str,
        symbols: list[str],
        schema: str,
        start: str | datetime,
        end: str | datetime,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Get mock market data."""
        if self.fail_on_request:
            raise ConnectionError("Mock connection error")

        # Simulate rate limiting
        self.request_count += 1
        if self.request_count > 100:
            self.rate_limit_hits += 1
            raise Exception("Rate limit exceeded")

        # Generate mock data based on schema
        if schema == "ohlcv-1m":
            return self._generate_ohlcv_data(symbols, start, end)
        elif schema == "trades":
            return self._generate_trades_data(symbols, start, end)
        elif schema == "mbp-1":
            return self._generate_l2_data(symbols, start, end)
        elif schema == "tbbo":
            return self._generate_tbbo_data(symbols, start, end)
        else:
            raise ValueError(f"Unsupported schema: {schema}")

    def _generate_ohlcv_data(
        self,
        symbols: list[str],
        start: str | datetime,
        end: str | datetime,
    ) -> pd.DataFrame:
        """Generate mock OHLCV data."""
        if isinstance(start, str):
            start = pd.Timestamp(start)
        if isinstance(end, str):
            end = pd.Timestamp(end)

        # Generate time index
        time_index = pd.date_range(start, end, freq="1min")

        data = []
        for symbol in symbols:
            base_price = 100.0 + hash(symbol) % 400  # Deterministic base price

            for ts in time_index:
                # Random walk with mean reversion
                price_change = np.random.normal(0, 0.001)
                open_price = base_price * (1 + price_change)
                high_price = open_price * (1 + abs(np.random.normal(0, 0.0005)))
                low_price = open_price * (1 - abs(np.random.normal(0, 0.0005)))
                close_price = open_price * (1 + np.random.normal(0, 0.0003))

                data.append({
                    "symbol": symbol,
                    "timestamp": ts,
                    "open": open_price,
                    "high": max(high_price, open_price, close_price),
                    "low": min(low_price, open_price, close_price),
                    "close": close_price,
                    "volume": np.random.uniform(1000, 10000),
                })

                base_price = close_price  # Update for next iteration

        return pd.DataFrame(data)

    def _generate_trades_data(
        self,
        symbols: list[str],
        start: str | datetime,
        end: str | datetime,
    ) -> pd.DataFrame:
        """Generate mock trades data."""
        if isinstance(start, str):
            start = pd.Timestamp(start)
        if isinstance(end, str):
            end = pd.Timestamp(end)

        data = []
        for symbol in symbols:
            base_price = 100.0 + hash(symbol) % 400

            # Generate random trades
            n_trades = np.random.randint(100, 500)
            timestamps = pd.to_datetime(
                np.random.uniform(start.value, end.value, n_trades),
                unit="ns",
            ).sort_values()

            for ts in timestamps:
                price = base_price * (1 + np.random.normal(0, 0.001))
                size = np.random.uniform(1, 1000)
                side = np.random.choice(["buy", "sell"])

                data.append({
                    "symbol": symbol,
                    "timestamp": ts,
                    "price": price,
                    "size": size,
                    "side": side,
                })

        return pd.DataFrame(data)

    def _generate_l2_data(
        self,
        symbols: list[str],
        start: str | datetime,
        end: str | datetime,
    ) -> pd.DataFrame:
        """Generate mock L2 market depth data."""
        if isinstance(start, str):
            start = pd.Timestamp(start)
        if isinstance(end, str):
            end = pd.Timestamp(end)

        data = []
        time_index = pd.date_range(start, end, freq="1s")  # L2 updates every second

        for symbol in symbols:
            base_price = 100.0 + hash(symbol) % 400
            tick_size = 0.01

            for ts in time_index:
                mid_price = base_price * (1 + np.random.normal(0, 0.0005))

                # Generate 5 levels of bids and asks
                for level in range(1, 6):
                    bid_price = mid_price - level * tick_size
                    ask_price = mid_price + level * tick_size
                    bid_size = np.random.uniform(100, 1000) * (6 - level)  # Larger at better prices
                    ask_size = np.random.uniform(100, 1000) * (6 - level)

                    data.append({
                        "symbol": symbol,
                        "timestamp": ts,
                        "level": level,
                        "bid_price": bid_price,
                        "bid_size": bid_size,
                        "ask_price": ask_price,
                        "ask_size": ask_size,
                    })

        return pd.DataFrame(data)

    def _generate_tbbo_data(
        self,
        symbols: list[str],
        start: str | datetime,
        end: str | datetime,
    ) -> pd.DataFrame:
        """Generate mock top-of-book data."""
        if isinstance(start, str):
            start = pd.Timestamp(start)
        if isinstance(end, str):
            end = pd.Timestamp(end)

        data = []
        time_index = pd.date_range(start, end, freq="100ms")  # TBBO updates frequently

        for symbol in symbols:
            base_price = 100.0 + hash(symbol) % 400

            for ts in time_index:
                mid_price = base_price * (1 + np.random.normal(0, 0.0002))
                spread = np.random.uniform(0.01, 0.05)

                data.append({
                    "symbol": symbol,
                    "timestamp": ts,
                    "bid_price": mid_price - spread / 2,
                    "bid_size": np.random.uniform(100, 5000),
                    "ask_price": mid_price + spread / 2,
                    "ask_size": np.random.uniform(100, 5000),
                })

                base_price = mid_price

        return pd.DataFrame(data)


class MockFredClient:
    """Mock FRED API client for testing."""

    def __init__(self, api_key: str = "test_fred_key"):
        """Initialize mock FRED client."""
        self.api_key = api_key
        self.request_count = 0

        # Mock economic series data
        self.series_data = {
            "DGS10": self._generate_treasury_data(),  # 10-Year Treasury
            "DEXUSEU": self._generate_exchange_rate_data(),  # USD/EUR
            "VIXCLS": self._generate_vix_data(),  # VIX
            "UNRATE": self._generate_unemployment_data(),  # Unemployment Rate
            "CPIAUCSL": self._generate_cpi_data(),  # CPI
        }

    def get_series(self, series_id: str, **kwargs: Any) -> pd.DataFrame:
        """Get economic series data."""
        self.request_count += 1

        if series_id not in self.series_data:
            raise ValueError(f"Series {series_id} not found")

        data = self.series_data[series_id]

        # Apply date filters if provided
        if "start_date" in kwargs:
            data = data[data.index >= pd.Timestamp(kwargs["start_date"])]
        if "end_date" in kwargs:
            data = data[data.index <= pd.Timestamp(kwargs["end_date"])]

        return data

    def _generate_treasury_data(self) -> pd.DataFrame:
        """Generate mock 10-year treasury rate data."""
        dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")
        # Simulate realistic treasury rates with trend
        base_rate = 2.0
        trend = np.linspace(0, 2, len(dates))
        noise = np.random.normal(0, 0.1, len(dates))
        rates = base_rate + trend + noise
        rates = np.clip(rates, 0.5, 5.0)  # Keep in realistic range

        return pd.DataFrame({"value": rates}, index=dates)

    def _generate_exchange_rate_data(self) -> pd.DataFrame:
        """Generate mock USD/EUR exchange rate data."""
        dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")
        # Simulate exchange rate around 1.10-1.20
        base_rate = 1.15
        walk = np.random.normal(0, 0.005, len(dates)).cumsum()
        rates = base_rate + walk * 0.01
        rates = np.clip(rates, 1.05, 1.25)

        return pd.DataFrame({"value": rates}, index=dates)

    def _generate_vix_data(self) -> pd.DataFrame:
        """Generate mock VIX volatility data."""
        dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")
        # VIX typically between 10-40, occasionally spikes
        base_vix = 20
        noise = np.random.gamma(2, 2, len(dates))
        # Add occasional spikes
        spikes = np.random.choice([0, 20], len(dates), p=[0.95, 0.05])
        vix = base_vix + noise + spikes
        vix = np.clip(vix, 10, 80)

        return pd.DataFrame({"value": vix}, index=dates)

    def _generate_unemployment_data(self) -> pd.DataFrame:
        """Generate mock unemployment rate data."""
        dates = pd.date_range("2020-01-01", "2024-12-31", freq="MS")  # Monthly
        # Simulate unemployment rate 3-10%
        base_rate = 5.0
        trend = np.sin(np.linspace(0, 4 * np.pi, len(dates))) * 2
        noise = np.random.normal(0, 0.3, len(dates))
        rates = base_rate + trend + noise
        rates = np.clip(rates, 3.0, 10.0)

        return pd.DataFrame({"value": rates}, index=dates)

    def _generate_cpi_data(self) -> pd.DataFrame:
        """Generate mock CPI data."""
        dates = pd.date_range("2020-01-01", "2024-12-31", freq="MS")  # Monthly
        # Simulate CPI with upward trend
        base_cpi = 250
        trend = np.linspace(0, 30, len(dates))
        seasonal = np.sin(np.linspace(0, 20 * np.pi, len(dates))) * 2
        noise = np.random.normal(0, 0.5, len(dates))
        cpi = base_cpi + trend + seasonal + noise

        return pd.DataFrame({"value": cpi}, index=dates)


class MockYahooClient:
    """Mock Yahoo Finance client for testing."""

    def __init__(self):
        """Initialize mock Yahoo client."""
        self.request_count = 0
        self.symbols = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]

    def get_history(
        self,
        symbol: str,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Get historical price data."""
        self.request_count += 1

        if symbol not in self.symbols:
            raise ValueError(f"Symbol {symbol} not found")

        # Generate daily data by default
        if start is None:
            start = datetime.now() - timedelta(days=365)
        if end is None:
            end = datetime.now()

        if isinstance(start, str):
            start = pd.Timestamp(start)
        if isinstance(end, str):
            end = pd.Timestamp(end)

        # Generate price data
        dates = pd.date_range(start, end, freq="D")
        base_price = 100.0 + hash(symbol) % 400

        # Random walk with drift
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))

        data = pd.DataFrame({
            "Open": prices * (1 + np.random.normal(0, 0.005, len(dates))),
            "High": prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            "Low": prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            "Close": prices,
            "Volume": np.random.uniform(1e6, 1e8, len(dates)),
            "Adj Close": prices,
        }, index=dates)

        # Ensure high >= max(open, close) and low <= min(open, close)
        data["High"] = data[["Open", "High", "Close"]].max(axis=1)
        data["Low"] = data[["Open", "Low", "Close"]].min(axis=1)

        return data

    def get_info(self, symbol: str) -> dict[str, Any]:
        """Get symbol information."""
        self.request_count += 1

        if symbol not in self.symbols:
            raise ValueError(f"Symbol {symbol} not found")

        return {
            "symbol": symbol,
            "longName": f"Mock Company {symbol}",
            "sector": random.choice(["Technology", "Finance", "Healthcare", "Consumer"]),
            "marketCap": random.uniform(1e9, 1e12),
            "trailingPE": random.uniform(10, 40),
            "dividendYield": random.uniform(0, 0.05),
            "beta": random.uniform(0.5, 2.0),
        }


class MockRedis:
    """Mock Redis client for testing."""

    def __init__(self):
        """Initialize mock Redis."""
        self.data: dict[str, Any] = {}
        self.expiry: dict[str, datetime] = {}
        self.pubsub_channels: dict[str, list[Callable]] = defaultdict(list)

    def get(self, key: str) -> bytes | None:
        """Get value by key."""
        if key in self.expiry and datetime.now() > self.expiry[key]:
            del self.data[key]
            del self.expiry[key]
            return None
        return self.data.get(key)

    def set(
        self,
        key: str,
        value: bytes | str,
        ex: int | None = None,
        px: int | None = None,
    ) -> bool:
        """Set key-value pair."""
        if isinstance(value, str):
            value = value.encode()
        self.data[key] = value

        if ex:
            self.expiry[key] = datetime.now() + timedelta(seconds=ex)
        elif px:
            self.expiry[key] = datetime.now() + timedelta(milliseconds=px)

        return True

    def delete(self, *keys: str) -> int:
        """Delete keys."""
        count = 0
        for key in keys:
            if key in self.data:
                del self.data[key]
                if key in self.expiry:
                    del self.expiry[key]
                count += 1
        return count

    def exists(self, *keys: str) -> int:
        """Check if keys exist."""
        count = 0
        for key in keys:
            if key in self.data:
                if key in self.expiry and datetime.now() > self.expiry[key]:
                    del self.data[key]
                    del self.expiry[key]
                else:
                    count += 1
        return count

    def lpush(self, key: str, *values: Any) -> int:
        """Push values to list."""
        if key not in self.data:
            self.data[key] = []
        for value in reversed(values):
            self.data[key].insert(0, value)
        return len(self.data[key])

    def rpop(self, key: str) -> bytes | None:
        """Pop from right of list."""
        if self.data.get(key):
            return self.data[key].pop()
        return None

    def hset(self, name: str, key: str, value: Any) -> int:
        """Set hash field."""
        if name not in self.data:
            self.data[name] = {}
        is_new = key not in self.data[name]
        self.data[name][key] = value
        return 1 if is_new else 0

    def hget(self, name: str, key: str) -> Any:
        """Get hash field."""
        if name in self.data and isinstance(self.data[name], dict):
            return self.data[name].get(key)
        return None

    def publish(self, channel: str, message: Any) -> int:
        """Publish message to channel."""
        subscribers = self.pubsub_channels.get(channel, [])
        for callback in subscribers:
            callback(message)
        return len(subscribers)

    def subscribe(self, channel: str, callback: Callable) -> None:
        """Subscribe to channel."""
        self.pubsub_channels[channel].append(callback)

    def ping(self) -> bool:
        """Ping Redis."""
        return True

    def flushdb(self) -> bool:
        """Clear all data."""
        self.data.clear()
        self.expiry.clear()
        return True


class MockPostgreSQL:
    """Mock PostgreSQL for unit tests (simulates basic operations)."""

    def __init__(self):
        """Initialize mock PostgreSQL."""
        self.tables: dict[str, pd.DataFrame] = {}
        self.sequences: dict[str, int] = defaultdict(int)
        self.transaction_active = False
        self.transaction_data: dict[str, pd.DataFrame] = {}

    def execute(self, query: str, params: tuple | None = None) -> Mock:
        """Execute SQL query."""
        query_lower = query.lower().strip()

        if query_lower.startswith("create table"):
            return self._handle_create_table(query)
        elif query_lower.startswith("insert into"):
            return self._handle_insert(query, params)
        elif query_lower.startswith("select"):
            return self._handle_select(query, params)
        elif query_lower.startswith("update"):
            return self._handle_update(query, params)
        elif query_lower.startswith("delete"):
            return self._handle_delete(query, params)
        elif query_lower == "begin":
            self.transaction_active = True
            self.transaction_data = self.tables.copy()
        elif query_lower == "commit":
            self.transaction_active = False
            self.transaction_data.clear()
        elif query_lower == "rollback":
            if self.transaction_active:
                self.tables = self.transaction_data.copy()
                self.transaction_active = False

        # Return mock result
        result = Mock()
        result.fetchall = Mock(return_value=[])
        result.fetchone = Mock(return_value=None)
        result.rowcount = 0
        return result

    def _handle_create_table(self, query: str) -> Mock:
        """Handle CREATE TABLE query."""
        # Extract table name (simplified)
        import re
        match = re.search(r"create\s+table\s+(?:if\s+not\s+exists\s+)?(\w+)", query, re.IGNORECASE)
        if match:
            table_name = match.group(1)
            self.tables[table_name] = pd.DataFrame()

        result = Mock()
        result.rowcount = 0
        return result

    def _handle_insert(self, query: str, params: tuple | None) -> Mock:
        """Handle INSERT query."""
        # Simplified insert handling
        import re
        match = re.search(r"insert\s+into\s+(\w+)", query, re.IGNORECASE)
        if match:
            table_name = match.group(1)
            if table_name not in self.tables:
                self.tables[table_name] = pd.DataFrame()
            # Would need to parse columns and values properly in real implementation

        result = Mock()
        result.rowcount = 1
        return result

    def _handle_select(self, query: str, params: tuple | None) -> Mock:
        """Handle SELECT query."""
        # Very simplified select handling
        result = Mock()
        result.fetchall = Mock(return_value=[])
        result.fetchone = Mock(return_value=None)
        result.rowcount = 0
        return result

    def _handle_update(self, query: str, params: tuple | None) -> Mock:
        """Handle UPDATE query."""
        result = Mock()
        result.rowcount = 0
        return result

    def _handle_delete(self, query: str, params: tuple | None) -> Mock:
        """Handle DELETE query."""
        result = Mock()
        result.rowcount = 0
        return result


def create_mock_databento_client(**kwargs: Any) -> MockDatabentoClient:
    """Factory function to create mock Databento client."""
    return MockDatabentoClient(**kwargs)


def create_mock_fred_client(**kwargs: Any) -> MockFredClient:
    """Factory function to create mock FRED client."""
    return MockFredClient(**kwargs)


def create_mock_yahoo_client(**kwargs: Any) -> MockYahooClient:
    """Factory function to create mock Yahoo client."""
    return MockYahooClient(**kwargs)


def create_mock_redis(**kwargs: Any) -> MockRedis:
    """Factory function to create mock Redis."""
    return MockRedis(**kwargs)


def create_mock_postgresql(**kwargs: Any) -> MockPostgreSQL:
    """Factory function to create mock PostgreSQL."""
    return MockPostgreSQL(**kwargs)
