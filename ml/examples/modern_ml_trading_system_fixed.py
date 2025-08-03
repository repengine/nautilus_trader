#!/usr/bin/env python3
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
Modern ML Trading System Example (Fixed).

This demonstrates the three-layer architecture:
1. Multiple ML inference actors (different models)
2. Portfolio construction actor (combines signals)
3. Execution strategy (manages orders)
"""

import numpy as np
from decimal import Decimal
from typing import Optional

from nautilus_trader.common.actor import Actor
from nautilus_trader.common.enums import LogColor
from nautilus_trader.config import ActorConfig
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.data import Data
from nautilus_trader.core.message import Event
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.data import DataType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy


# Custom message types for ML communication
# Note: In production, these would inherit from Data properly
# For now, using simple classes that work with msgspec

class MLSignal:
    """ML model prediction signal."""
    
    def __init__(
        self,
        trader_id: TraderId,
        instrument_id: InstrumentId,
        model_id: str,
        prediction: float,  # -1 to 1, strength of signal
        confidence: float,  # 0 to 1, model confidence
        features: dict,  # For debugging/analysis
        ts_event: int,
        ts_init: int,
    ):
        self.trader_id = trader_id
        self.instrument_id = instrument_id
        self.model_id = model_id
        self.prediction = prediction
        self.confidence = confidence
        self.features = features
        self.ts_event = ts_event
        self.ts_init = ts_init


class PortfolioTarget:
    """Portfolio optimization output."""
    
    def __init__(
        self,
        trader_id: TraderId,
        targets: dict[InstrumentId, float],  # Instrument -> target weight
        signal_sources: dict[str, float],  # Model contributions
        risk_score: float,
        ts_event: int,
        ts_init: int,
    ):
        self.trader_id = trader_id
        self.targets = targets
        self.signal_sources = signal_sources
        self.risk_score = risk_score
        self.ts_event = ts_event
        self.ts_init = ts_init


# Layer 1: ML Inference Actors
class MLInferenceActorConfig(ActorConfig, frozen=True):
    """Configuration for ML inference actors."""
    
    model_id: str
    model_path: str
    bar_type_str: str  # String representation to avoid BarType in config
    lookback_periods: int = 20
    prediction_threshold: float = 0.3


class MLInferenceActor(Actor):
    """
    Base ML inference actor - generates signals from market data.
    
    In production, you'd have multiple of these with different models:
    - MomentumMLActor
    - MeanReversionMLActor
    - SentimentMLActor
    - VolumeMLActor
    """
    
    def __init__(self, config: MLInferenceActorConfig) -> None:
        super().__init__(config)
        
        # Configuration
        self.model_id = config.model_id
        self.model_path = config.model_path
        self.bar_type = BarType.from_str(config.bar_type_str)
        self.lookback = config.lookback_periods
        self.threshold = config.prediction_threshold
        
        # State
        self.bars: list[Bar] = []
        self.model = self._load_model()
        
        # Performance tracking
        self.predictions_made = 0
        self.signals_sent = 0
        
    def on_start(self) -> None:
        """Subscribe to market data on start."""
        self.subscribe_bars(self.bar_type)
        self.log.info(f"ML Actor {self.model_id} started", LogColor.GREEN)
        
    def on_bar(self, bar: Bar) -> None:
        """Process new bar data and generate signals."""
        self.bars.append(bar)
        
        # Keep rolling window
        if len(self.bars) > self.lookback:
            self.bars.pop(0)
            
        # Need enough data
        if len(self.bars) < self.lookback:
            return
            
        # Compute features
        features = self._compute_features()
        
        # Generate prediction
        prediction = self.model.predict(features)
        confidence = self._calculate_confidence(prediction)
        
        self.predictions_made += 1
        
        # Only send strong signals
        if abs(prediction) > self.threshold:
            signal = MLSignal(
                trader_id=self.trader_id,
                instrument_id=bar.instrument_id,
                model_id=self.model_id,
                prediction=float(prediction),
                confidence=float(confidence),
                features=features,
                ts_event=bar.ts_event,
                ts_init=self.clock.timestamp_ns(),
            )
            
            # In production, you'd publish this properly
            # For now, just log it
            self.signals_sent += 1
            
            self.log.info(
                f"{self.model_id} signal: {prediction:.3f} (conf: {confidence:.2f})",
                LogColor.BLUE,
            )
    
    def _load_model(self):
        """Load model from disk or MLflow."""
        # Simplified - in production use MLflow or model server
        # return joblib.load(self.model_path)
        
        # Mock model for example
        class MockModel:
            def predict(self, features):
                # Simulate momentum model
                return np.tanh(features["momentum"] * 2 + np.random.normal(0, 0.1))
                
        return MockModel()
    
    def _compute_features(self) -> dict:
        """Compute features from price data."""
        closes = [bar.close.as_double() for bar in self.bars]
        
        # Simple features for example
        return {
            "returns": (closes[-1] - closes[-2]) / closes[-2],
            "momentum": (closes[-1] - closes[0]) / closes[0],
            "volatility": np.std(closes) / np.mean(closes),
            "rsi": self._calculate_rsi(closes),
        }
    
    def _calculate_rsi(self, prices: list[float], period: int = 14) -> float:
        """Simple RSI calculation."""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 100
        return 100 - (100 / (1 + rs))
    
    def _calculate_confidence(self, prediction: float) -> float:
        """Calculate model confidence (mock)."""
        # In production: use prediction probabilities, ensemble agreement, etc.
        return min(abs(prediction), 1.0)


# Layer 2: Portfolio Construction
class PortfolioConstructorConfig(ActorConfig, frozen=True):
    """Configuration for portfolio construction."""
    
    model_weights_str: str  # JSON string of model weights
    rebalance_interval_seconds: int = 300
    min_signal_agreement: float = 0.5
    max_position_size: float = 0.3
    risk_limit: float = 0.02


class PortfolioConstructorActor(Actor):
    """
    Combines ML signals into portfolio targets.
    
    Modern approaches:
    - Hierarchical Risk Parity
    - Black-Litterman with ML views
    - Reinforcement Learning for allocation
    """
    
    def __init__(self, config: PortfolioConstructorConfig) -> None:
        super().__init__(config)
        
        import json
        self.model_weights = json.loads(config.model_weights_str)
        self.rebalance_interval = config.rebalance_interval_seconds
        self.min_agreement = config.min_signal_agreement
        self.max_position = config.max_position_size
        self.risk_limit = config.risk_limit
        
        # Signal buffer
        self.latest_signals: dict[str, MLSignal] = {}
        self.last_rebalance = None
        
    def on_start(self) -> None:
        """Set up rebalance timer."""
        # Set up rebalance timer
        self.clock.set_timer(
            name="rebalance",
            interval=self.rebalance_interval * 1_000_000_000,
        )
        
        self.log.info("Portfolio Constructor started", LogColor.GREEN)
        
    def on_event(self, event: Event) -> None:
        """Handle timer events."""
        # In production, handle timer events for rebalancing
        pass


# Layer 3: Execution Strategy
class MLPortfolioStrategyConfig(StrategyConfig, frozen=True):
    """Configuration for ML portfolio execution."""
    
    instrument_id_str: str  # String representation
    portfolio_actor_id: str
    order_size_usd: float = 10000.0
    max_slippage_pct: float = 0.001
    stop_loss_pct: float = 0.02


class MLPortfolioStrategy(Strategy):
    """
    Executes portfolio targets from ML system.
    
    Handles:
    - Order generation and execution
    - Risk management
    - Position tracking
    - Performance attribution
    """
    
    def __init__(self, config: MLPortfolioStrategyConfig) -> None:
        super().__init__(config)
        
        self.instrument_id = InstrumentId.from_str(config.instrument_id_str)
        self.portfolio_actor_id = config.portfolio_actor_id
        self.order_size_usd = config.order_size_usd
        self.max_slippage = config.max_slippage_pct
        self.stop_loss = config.stop_loss_pct
        
        # State
        self.latest_targets: Optional[PortfolioTarget] = None
        self.target_positions: dict[InstrumentId, float] = {}
        
    def on_start(self) -> None:
        """Subscribe to instruments."""
        # Subscribe to bars for the instrument
        bar_type = BarType.from_str(f"{self.instrument_id}-1-MINUTE-LAST-EXTERNAL")
        self.subscribe_bars(bar_type)
        
        self.log.info("ML Portfolio Strategy started", LogColor.GREEN)
        
    def on_bar(self, bar: Bar) -> None:
        """Monitor positions on each bar."""
        # In production, implement position monitoring
        pass
        
    def execute_trade_example(self, instrument_id: InstrumentId, side: OrderSide, size: Decimal) -> None:
        """Example of how to execute a trade."""
        # Create market order
        order = self.order_factory.market(
            instrument_id=instrument_id,
            order_side=side,
            quantity=Quantity.from_str(str(size)),
            time_in_force=TimeInForce.GTC,
        )
        
        # Submit order
        self.submit_order(order)
        
        self.log.info(
            f"Executing {side} {size} {instrument_id}",
            LogColor.BLUE,
        )


# Example usage
def create_ml_trading_components():
    """Create ML trading system components."""
    # Note: These would be added to a BacktestEngine or TradingNode
    
    # Create ML inference actor
    ml_actor_config = MLInferenceActorConfig(
        actor_id="ML-MOMENTUM",
        model_id="momentum_v2",
        model_path="/models/momentum.pkl",
        bar_type_str="AAPL.NASDAQ-1-MINUTE-LAST-EXTERNAL",
    )
    ml_actor = MLInferenceActor(ml_actor_config)
    
    # Create portfolio constructor
    portfolio_config = PortfolioConstructorConfig(
        actor_id="PORTFOLIO-OPT",
        model_weights_str='{"momentum_v2": 0.6, "mean_reversion_v1": 0.4}',
        rebalance_interval_seconds=300,
    )
    portfolio_actor = PortfolioConstructorActor(portfolio_config)
    
    # Create execution strategy
    strategy_config = MLPortfolioStrategyConfig(
        strategy_id="ML-EXEC",
        instrument_id_str="AAPL.NASDAQ",
        portfolio_actor_id="PORTFOLIO-OPT",
    )
    strategy = MLPortfolioStrategy(strategy_config)
    
    return ml_actor, portfolio_actor, strategy


if __name__ == "__main__":
    print("ML Trading System Example")
    print("This shows the structure - add components to BacktestEngine to run")
    
    # Create components
    ml_actor, portfolio_actor, strategy = create_ml_trading_components()
    print(f"Created: {ml_actor}, {portfolio_actor}, {strategy}")