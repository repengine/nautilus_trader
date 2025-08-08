# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Integration tests for ML actors with Nautilus Trader.

Tests that ML actors properly integrate with the Nautilus Trader system using the
configuration adapter pattern.

"""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.config.base import MLActorConfig
from ml.examples.simple_ml_actor import SimpleMLActor
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.engine import BacktestEngineConfig
from nautilus_trader.common.component import MessageBus
from nautilus_trader.common.component import TestClock
from nautilus_trader.data.engine import DataEngine
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.identifiers import AccountId
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money

# from nautilus_trader.msgbus.bus import MessageBus as MsgBus  # Not needed
from nautilus_trader.portfolio.portfolio import Portfolio
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.test_kit.stubs.component import TestComponentStubs


class DummyTestModel:
    """
    Dummy model for testing.
    """

    def predict(self, X):
        import numpy as np

        return np.array([0.8])


class TestMLActorIntegration:
    """
    Integration tests for ML actors.
    """

    def setup_method(self):
        """
        Set up test environment.
        """
        # Create test components
        self.clock = TestClock()
        self.trader_id = TraderId("TESTER-001")
        self.account_id = AccountId("SIM-001")
        self.venue = Venue("SIM")

        # Create message bus
        self.msgbus = MessageBus(
            trader_id=self.trader_id,
            clock=self.clock,
        )

        # Create cache
        self.cache = TestComponentStubs.cache()

        # Create portfolio
        self.portfolio = Portfolio(
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )

        # Create data engine
        self.data_engine = DataEngine(
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )

    def test_simple_ml_actor_creation_with_config(self) -> None:
        """
        Test creating a SimpleMLActor with ML configuration.
        """
        # Arrange
        instrument_id = InstrumentId.from_str("EURUSD.IDEALPRO")
        bar_type = BarType.from_str("EURUSD.IDEALPRO-1-MINUTE-MID-EXTERNAL")

        config = MLActorConfig(
            model_path=str(Path.home() / ".nautilus" / "dummy_model.pkl"),
            bar_type=bar_type,
            instrument_id=instrument_id,
            prediction_threshold=0.6,
            warm_up_period=20,
        )

        # Act
        actor = SimpleMLActor(config)

        # Assert
        assert actor is not None
        assert actor._config == config
        assert actor._config.prediction_threshold == 0.6
        assert actor._config.warm_up_period == 20

    def test_ml_actor_initialization_with_msgbus(self) -> None:
        """
        Test ML actor initialization with message bus.
        """
        # Arrange
        instrument_id = InstrumentId.from_str("EURUSD.IDEALPRO")
        bar_type = BarType.from_str("EURUSD.IDEALPRO-1-MINUTE-MID-EXTERNAL")

        config = MLActorConfig(
            model_path=str(Path.home() / ".nautilus" / "dummy_model.pkl"),
            bar_type=bar_type,
            instrument_id=instrument_id,
        )

        actor = SimpleMLActor(config)

        # Act - Register with message bus
        actor.register_base(
            portfolio=self.portfolio,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )

        # Assert
        assert actor.id is not None
        assert actor.is_initialized
        assert actor.trader_id == self.trader_id

    def test_ml_actor_processes_bars(self) -> None:
        """
        Test that ML actor properly processes bar data.
        """
        # Arrange
        instrument_id = InstrumentId.from_str("EURUSD.IDEALPRO")
        bar_type = BarType.from_str("EURUSD.IDEALPRO-1-MINUTE-MID-EXTERNAL")

        # Create dummy model file
        import pickle
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(DummyTestModel(), f)
            model_path = f.name

        # Create ML actor
        config = MLActorConfig(
            model_path=model_path,
            bar_type=bar_type,
            instrument_id=instrument_id,
            warm_up_period=5,
            log_predictions=True,
        )

        actor = SimpleMLActor(config)
        actor.register_base(
            portfolio=self.portfolio,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )

        # Create test bars
        from nautilus_trader.model.data import Bar
        from nautilus_trader.model.objects import Price
        from nautilus_trader.model.objects import Quantity

        bars = []
        for i in range(30):
            bar = Bar(
                bar_type=bar_type,
                open=Price.from_str(f"1.{1000 + i:04d}"),
                high=Price.from_str(f"1.{1010 + i:04d}"),
                low=Price.from_str(f"1.{990 + i:04d}"),
                close=Price.from_str(f"1.{1005 + i:04d}"),
                volume=Quantity.from_int(1000000 + i * 1000),
                ts_event=i * 60_000_000_000,  # 1 minute intervals
                ts_init=i * 60_000_000_000,
            )
            bars.append(bar)

        # Act - Start actor and process bars
        actor.start()

        for bar in bars:
            actor.on_bar(bar)

        # Assert
        assert actor._bars_processed == 30
        assert actor._is_warmed_up is True
        assert actor._prediction_count > 0  # Should have made predictions after warm-up

        # Cleanup
        import os

        os.unlink(model_path)

    @pytest.mark.skipif(
        not (Path.home() / ".nautilus" / "test_model.pkl").exists(),
        reason="Requires a test model file",
    )
    def test_ml_actor_in_backtest_engine(self) -> None:
        """
        Test ML actor running in a full backtest engine.
        """
        # Arrange
        config = BacktestEngineConfig(
            trader_id="BACKTESTER-001",
        )

        engine = BacktestEngine(config=config)

        # Add venue
        engine.add_venue(
            venue=Venue("SIM"),
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=USD,
            starting_balances=[Money(1_000_000, USD)],
        )

        # Load data
        instrument = TestInstrumentProvider.default_fx_ccy("EUR/USD")
        engine.add_instrument(instrument)

        # Create ML actor
        bar_type = BarType.from_str("EURUSD.SIM-1-MINUTE-MID-EXTERNAL")
        ml_config = MLActorConfig(
            model_path=str(Path.home() / ".nautilus" / "test_model.pkl"),
            bar_type=bar_type,
            instrument_id=instrument.id,
            warm_up_period=20,
            publish_signals=True,
        )

        actor = SimpleMLActor(ml_config)
        engine.add_actor(actor)

        # Load test data (would need actual data)
        # engine.add_data(bars)

        # Act
        # engine.run()

        # Assert
        # assert engine.iteration == expected_iterations
        # assert actor._prediction_count > 0
