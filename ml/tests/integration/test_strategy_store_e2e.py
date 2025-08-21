"""
End-to-end integration test for StrategyStore with PostgreSQL.

Tests the complete flow from ML signal generation through strategy decision persistence
to the PostgreSQL database.

"""

import os
import time
from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy import text

from ml.actors.base import MLSignal
from ml.config.base import MLStrategyConfig
from ml.stores.strategy_store import StrategyStore
from ml.strategies.ml_strategy import MLTradingStrategy
from nautilus_trader.common.component import MessageBus
from nautilus_trader.common.component import TestClock
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.portfolio.portfolio import Portfolio
from nautilus_trader.test_kit.stubs.component import TestComponentStubs


# Skip if PostgreSQL is not available
POSTGRES_AVAILABLE = os.environ.get("POSTGRES_TEST_URL") is not None
POSTGRES_URL = os.environ.get(
    "POSTGRES_TEST_URL",
    "postgresql://postgres:postgres@localhost:5432/nautilus",
)


@pytest.mark.skipif(not POSTGRES_AVAILABLE, reason="PostgreSQL not available")
class TestStrategyStoreE2E:
    """
    End-to-end tests for StrategyStore with real PostgreSQL.
    """

    engine: Any

    @classmethod
    def setup_class(cls) -> None:
        """
        Set up database connection and ensure tables exist.
        """
        try:
            cls.engine = create_engine(POSTGRES_URL)

            # Ensure tables exist (simplified schema for testing)
            with cls.engine.connect() as conn:
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS ml_strategy_signals (
                        strategy_id VARCHAR(255) NOT NULL,
                        instrument_id VARCHAR(100) NOT NULL,
                        ts_event BIGINT NOT NULL,
                        ts_init BIGINT,
                        signal_type VARCHAR(20) NOT NULL,
                        strength FLOAT NOT NULL,
                        model_predictions JSONB,
                        risk_metrics JSONB,
                        execution_params JSONB,
                        is_live BOOLEAN DEFAULT FALSE,
                        created_at BIGINT,
                        PRIMARY KEY (strategy_id, instrument_id, ts_event)
                    )
                """,
                    ),
                )
                conn.commit()
        except Exception as e:
            pytest.skip(f"Cannot connect to PostgreSQL: {e}")

    def setup_method(self) -> None:
        """
        Set up test fixtures.
        """
        self.clock = TestClock()
        self.trader_id = TraderId("E2E-TESTER")
        self.msgbus = MessageBus(
            trader_id=self.trader_id,
            clock=self.clock,
        )
        self.cache = TestComponentStubs.cache()
        self.portfolio = Portfolio(
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )

        # Create test instrument
        self.instrument_id = InstrumentId.from_str("BTC/USDT.BINANCE")

        # Create unique strategy ID for each test
        self.strategy_id = f"E2E-{int(time.time() * 1000)}"

        # Clean up any existing test data
        self._cleanup_test_data()

    def teardown_method(self) -> None:
        """
        Clean up test data after each test.
        """
        self._cleanup_test_data()

    def _cleanup_test_data(self) -> None:
        """
        Remove test data from database.
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text("DELETE FROM ml_strategy_signals WHERE strategy_id = :sid"),
                    {"sid": self.strategy_id},
                )
                conn.commit()
        except Exception:
            pass  # Ignore cleanup errors

    def test_full_pipeline_with_real_database(self) -> None:
        """
        Test complete flow from signal to database persistence.
        """
        # Create strategy with real database connection
        config = MLStrategyConfig(
            strategy_id=self.strategy_id,
            instrument_id=self.instrument_id,
            ml_signal_source="E2E_TEST",
            use_strategy_store=True,
            strategy_store_config={
                "connection_string": POSTGRES_URL,
                "batch_size": 10,
                "flush_interval_ms": 100,
            },
            persist_all_signals=True,
        )

        strategy = MLTradingStrategy(config)
        strategy.register_base(
            portfolio=self.portfolio,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )

        # Process multiple signals
        signals_processed = []

        for i in range(5):
            signal = MLSignal(
                instrument_id=self.instrument_id,
                model_id=f"model_{i}",
                prediction=0.4 + i * 0.1,  # 0.4, 0.5, 0.6, 0.7, 0.8
                confidence=0.8 + i * 0.02,
                metadata={"iteration": i},
                ts_event=dt_to_unix_nanos(self.clock.utc_now()) + i * 1000000000,
                ts_init=dt_to_unix_nanos(self.clock.utc_now()) + i * 1000000000,
            )

            strategy._process_ml_signal(signal)
            signals_processed.append(signal)

            # Advance clock
            self.clock.advance_time(1000000000)  # 1 second

        # Flush the store to ensure all data is written
        if strategy.strategy_store:
            strategy.strategy_store.flush()

        # Verify data in database
        with self.engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT strategy_id, instrument_id, signal_type, strength,
                           model_predictions, risk_metrics, execution_params
                    FROM ml_strategy_signals
                    WHERE strategy_id = :sid
                    ORDER BY ts_event
                """,
                ),
                {"sid": self.strategy_id},
            )

            rows = result.fetchall()

            # Should have 5 decisions persisted
            assert len(rows) == 5

            # Verify each decision
            for i, row in enumerate(rows):
                assert row[0] == self.strategy_id  # strategy_id
                assert row[1] == str(self.instrument_id)  # instrument_id

                # Signal type based on prediction threshold
                prediction = 0.4 + i * 0.1
                expected_type = "SELL" if prediction < 0.5 else "BUY"
                # First signal might be SELL, rest should follow pattern
                if i == 0:
                    assert row[2] in ["SELL", "BUY", "HOLD"]
                else:
                    # After first position, might be HOLD if same direction
                    assert row[2] in [expected_type, "HOLD"]

                # Verify strength matches confidence
                assert abs(row[3] - (0.8 + i * 0.02)) < 0.001

                # Verify model predictions
                assert f"model_{i}" in row[4]  # model_predictions JSON

    def test_batch_persistence(self) -> None:
        """
        Test that batching works correctly with real database.
        """
        # Create store with small batch size
        store = StrategyStore(
            connection_string=POSTGRES_URL,
            batch_size=3,  # Small batch for testing
            flush_interval_ms=10000,  # Long interval so we control flushing
            clock=self.clock,
        )

        # Write signals without flushing
        for i in range(5):
            store.write_signal(
                strategy_id=f"BATCH-{self.strategy_id}",
                instrument_id=str(self.instrument_id),
                signal_type="BUY" if i % 2 == 0 else "SELL",
                strength=0.5 + i * 0.1,
                model_predictions={f"model_{i}": 0.6 + i * 0.05},
                risk_metrics={"risk": i * 0.1},
                execution_params={"size": 100 * (i + 1)},
                ts_event=dt_to_unix_nanos(self.clock.utc_now()) + i * 1000000000,
                is_live=False,
            )

        # After 5 writes with batch_size=3, should have auto-flushed once
        # Buffer should have 2 items (5 % 3 = 2)
        assert len(store._write_buffer) == 2

        # Verify first batch (3 items) was written
        with self.engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT COUNT(*) FROM ml_strategy_signals
                    WHERE strategy_id = :sid
                """,
                ),
                {"sid": f"BATCH-{self.strategy_id}"},
            )
            count_before_flush = result.scalar()
            assert count_before_flush == 3  # First batch auto-flushed

        # Manual flush for remaining items
        store.flush()

        # Verify all 5 items are now in database
        with self.engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT COUNT(*) FROM ml_strategy_signals
                    WHERE strategy_id = :sid
                """,
                ),
                {"sid": f"BATCH-{self.strategy_id}"},
            )
            count_after_flush = result.scalar()
            assert count_after_flush == 5

        # Clean up
        with self.engine.connect() as conn:
            conn.execute(
                text("DELETE FROM ml_strategy_signals WHERE strategy_id = :sid"),
                {"sid": f"BATCH-{self.strategy_id}"},
            )
            conn.commit()

    def test_error_recovery(self) -> None:
        """
        Test that the system recovers from database errors.
        """
        config = MLStrategyConfig(
            strategy_id=self.strategy_id,
            instrument_id=self.instrument_id,
            ml_signal_source="ERROR_TEST",
            use_strategy_store=True,
            strategy_store_config={
                "connection_string": POSTGRES_URL,
                "batch_size": 10,
                "flush_interval_ms": 100,
            },
        )

        strategy = MLTradingStrategy(config)
        strategy.register_base(
            portfolio=self.portfolio,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )

        # Temporarily break the connection
        original_engine = strategy.strategy_store.engine if strategy.strategy_store else None
        if strategy.strategy_store:
            strategy.strategy_store.engine = MagicMock()
            strategy.strategy_store.engine.connect.side_effect = Exception("Connection lost")

        # Process signal - should handle error gracefully
        signal = MLSignal(
            instrument_id=self.instrument_id,
            model_id="error_test",
            prediction=0.7,
            confidence=0.8,
            metadata={},
            ts_event=dt_to_unix_nanos(self.clock.utc_now()),
            ts_init=dt_to_unix_nanos(self.clock.utc_now()),
        )

        # Should not raise exception
        strategy._process_ml_signal(signal)

        # Restore connection
        if strategy.strategy_store and original_engine:
            strategy.strategy_store.engine = original_engine

        # Process another signal - should work
        signal2 = MLSignal(
            instrument_id=self.instrument_id,
            model_id="recovery_test",
            prediction=0.6,
            confidence=0.9,
            metadata={},
            ts_event=dt_to_unix_nanos(self.clock.utc_now()) + 1000000000,
            ts_init=dt_to_unix_nanos(self.clock.utc_now()) + 1000000000,
        )

        strategy._process_ml_signal(signal2)

        # Flush to ensure write
        if strategy.strategy_store:
            strategy.strategy_store.flush()

        # Verify second signal was persisted
        with self.engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT COUNT(*) FROM ml_strategy_signals
                    WHERE strategy_id = :sid
                """,
                ),
                {"sid": self.strategy_id},
            )
            count = result.scalar()
            # At least the recovery signal should be there
            assert count >= 1

    def test_concurrent_strategies(self) -> None:
        """
        Test multiple strategies writing to the same database.
        """
        strategies = []

        # Create multiple strategies
        for i in range(3):
            config = MLStrategyConfig(
                strategy_id=f"{self.strategy_id}-{i}",
                instrument_id=self.instrument_id,
                ml_signal_source=f"CONCURRENT_{i}",
                use_strategy_store=True,
                strategy_store_config={
                    "connection_string": POSTGRES_URL,
                    "batch_size": 5,
                    "flush_interval_ms": 100,
                },
            )

            strategy = MLTradingStrategy(config)
            strategy.register_base(
                portfolio=self.portfolio,
                msgbus=self.msgbus,
                cache=self.cache,
                clock=self.clock,
            )
            strategies.append(strategy)

        # Each strategy processes signals
        for i, strategy in enumerate(strategies):
            signal = MLSignal(
                instrument_id=self.instrument_id,
                model_id=f"concurrent_{i}",
                prediction=0.5 + i * 0.1,
                confidence=0.7 + i * 0.05,
                metadata={"strategy_index": i},
                ts_event=dt_to_unix_nanos(self.clock.utc_now()) + i * 1000000000,
                ts_init=dt_to_unix_nanos(self.clock.utc_now()) + i * 1000000000,
            )

            strategy._process_ml_signal(signal)

            # Flush to ensure write
            if strategy.strategy_store:
                strategy.strategy_store.flush()

        # Verify all strategies wrote their data
        with self.engine.connect() as conn:
            for i in range(3):
                result = conn.execute(
                    text(
                        """
                        SELECT COUNT(*) FROM ml_strategy_signals
                        WHERE strategy_id = :sid
                    """,
                    ),
                    {"sid": f"{self.strategy_id}-{i}"},
                )
                count = result.scalar()
                assert count >= 1  # Each strategy should have at least 1 decision

        # Clean up
        for i in range(3):
            with self.engine.connect() as conn:
                conn.execute(
                    text("DELETE FROM ml_strategy_signals WHERE strategy_id = :sid"),
                    {"sid": f"{self.strategy_id}-{i}"},
                )
                conn.commit()


if __name__ == "__main__":
    # Run with PostgreSQL if available
    if POSTGRES_AVAILABLE:
        pytest.main([__file__, "-xvs"])
    else:
        print("PostgreSQL not available. Set POSTGRES_TEST_URL environment variable to run tests.")
        print(
            "Example: export POSTGRES_TEST_URL='postgresql://postgres:postgres@localhost:5432/nautilus'",
        )
