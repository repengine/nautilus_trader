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
Modern ML Trading System Example.

This demonstrates the three-layer architecture:
1. Multiple ML inference actors (different models)
2. Portfolio construction actor (combines signals)
3. Execution strategy (manages orders)
"""

import numpy as np
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
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

# Custom message types for ML communication
from dataclasses import dataclass
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.identifiers import TraderId


@dataclass
class MLSignal(Data):
    """ML model prediction signal."""
    
    trader_id: TraderId
    instrument_id: InstrumentId
    model_id: str
    prediction: float  # -1 to 1, strength of signal
    confidence: float  # 0 to 1, model confidence
    features: dict  # For debugging/analysis
    ts_event: int
    ts_init: int
    
    @staticmethod
    def from_dict(values: dict) -> "MLSignal":
        return MLSignal(**values)


@dataclass
class PortfolioTarget(Data):
    """Portfolio optimization output."""
    
    trader_id: TraderId
    targets: dict[InstrumentId, float]  # Instrument -> target weight
    signal_sources: dict[str, float]  # Model contributions
    risk_score: float
    ts_event: int
    ts_init: int
    
    @staticmethod
    def from_dict(values: dict) -> "PortfolioTarget":
        return PortfolioTarget(**values)


# Layer 1: ML Inference Actors
class MLInferenceActorConfig(ActorConfig, frozen=True):
    """Configuration for ML inference actors."""
    
    model_id: str
    model_path: str
    bar_type: BarType
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
        self.bar_type = config.bar_type
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
            
            # Publish to message bus
            self.publish_data(DataType(MLSignal, metadata={"model_id": self.model_id}), signal)
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
    
    model_weights: dict[str, float]  # Model ID -> weight
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
        
        self.model_weights = config.model_weights
        self.rebalance_interval = config.rebalance_interval_seconds
        self.min_agreement = config.min_signal_agreement
        self.max_position = config.max_position_size
        self.risk_limit = config.risk_limit
        
        # Signal buffer
        self.latest_signals: dict[str, MLSignal] = {}
        self.last_rebalance = None
        
    def on_start(self) -> None:
        """Subscribe to ML signals."""
        # Subscribe to all ML signals
        for model_id in self.model_weights:
            self.subscribe_data(
                DataType(MLSignal, metadata={"model_id": model_id})
            )
            
        # Set up rebalance timer
        self.clock.set_timer(
            name="rebalance",
            interval_ns=self.rebalance_interval * 1_000_000_000,
            start_time_ns=self.clock.timestamp_ns(),
            stop_time_ns=0,  # Run indefinitely
        )
        
        self.log.info("Portfolio Constructor started", LogColor.GREEN)
        
    def on_data(self, data: Data) -> None:
        """Handle ML signals."""
        if isinstance(data, MLSignal):
            self.latest_signals[data.model_id] = data
            
            # Check if we should rebalance
            if self._should_rebalance():
                self._construct_portfolio()
                
    def on_event(self, event: Event) -> None:
        """Handle timer events."""
        if hasattr(event, 'name') and event.name == "rebalance":
            self._construct_portfolio()
            
    def _should_rebalance(self) -> bool:
        """Check if we should rebalance now."""
        # Rebalance if:
        # 1. We have signals from all models
        # 2. Signals are recent (< 60 seconds old)
        # 3. Haven't rebalanced recently
        
        if len(self.latest_signals) < len(self.model_weights):
            return False
            
        current_time = self.clock.timestamp_ns()
        for signal in self.latest_signals.values():
            age_seconds = (current_time - signal.ts_event) / 1_000_000_000
            if age_seconds > 60:
                return False
                
        return True
        
    def _construct_portfolio(self) -> None:
        """Combine signals into portfolio targets."""
        if not self.latest_signals:
            return
            
        # Weighted ensemble of predictions
        weighted_predictions = {}
        
        for model_id, signal in self.latest_signals.items():
            weight = self.model_weights.get(model_id, 0)
            
            if signal.instrument_id not in weighted_predictions:
                weighted_predictions[signal.instrument_id] = 0
                
            weighted_predictions[signal.instrument_id] += (
                weight * signal.prediction * signal.confidence
            )
        
        # Normalize and apply risk limits
        targets = {}
        for instrument_id, raw_prediction in weighted_predictions.items():
            # Convert prediction to position size
            target = np.tanh(raw_prediction) * self.max_position
            
            # Apply risk limits
            target = np.clip(target, -self.risk_limit, self.risk_limit)
            
            targets[instrument_id] = target
            
        # Calculate risk score
        risk_score = self._calculate_risk_score(targets)
        
        # Create portfolio target
        portfolio_target = PortfolioTarget(
            trader_id=self.trader_id,
            targets=targets,
            signal_sources={
                model_id: signal.prediction 
                for model_id, signal in self.latest_signals.items()
            },
            risk_score=risk_score,
            ts_event=self.clock.timestamp_ns(),
            ts_init=self.clock.timestamp_ns(),
        )
        
        # Publish to strategies
        self.publish_data(DataType(PortfolioTarget), portfolio_target)
        
        self.log.info(
            f"Portfolio target: {targets}, risk: {risk_score:.3f}",
            LogColor.CYAN,
        )
        
        self.last_rebalance = self.clock.timestamp_ns()
        
    def _calculate_risk_score(self, targets: dict) -> float:
        """Calculate portfolio risk score."""
        # Simplified - in production use proper risk models
        total_exposure = sum(abs(t) for t in targets.values())
        concentration = max(abs(t) for t in targets.values()) if targets else 0
        
        return (total_exposure + concentration) / 2


# Layer 3: Execution Strategy
class MLPortfolioStrategyConfig(StrategyConfig, frozen=True):
    """Configuration for ML portfolio execution."""
    
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
        
        self.portfolio_actor_id = config.portfolio_actor_id
        self.order_size_usd = config.order_size_usd
        self.max_slippage = config.max_slippage_pct
        self.stop_loss = config.stop_loss_pct
        
        # State
        self.latest_targets: Optional[PortfolioTarget] = None
        self.target_positions: dict[InstrumentId, float] = {}
        
    def on_start(self) -> None:
        """Subscribe to portfolio targets."""
        self.subscribe_data(DataType(PortfolioTarget))
        self.log.info("ML Portfolio Strategy started", LogColor.GREEN)
        
    def on_data(self, data: Data) -> None:
        """Handle portfolio targets."""
        if isinstance(data, PortfolioTarget):
            self.latest_targets = data
            self._execute_portfolio_targets(data)
            
    def _execute_portfolio_targets(self, targets: PortfolioTarget) -> None:
        """Convert targets to orders."""
        for instrument_id, target_weight in targets.targets.items():
            
            # Get current position
            position = self.cache.position_for_order(
                self.order_factory.client_order_id(),
                instrument_id,
            )
            
            current_weight = 0.0
            if position:
                # Calculate current weight (simplified)
                current_weight = position.quantity.as_double() / self.order_size_usd
                
            # Calculate required adjustment
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) < 0.01:  # 1% threshold
                continue
                
            # Generate order
            if weight_diff > 0:
                order = self.order_factory.market(
                    instrument_id=instrument_id,
                    order_side=OrderSide.BUY,
                    quantity=Quantity.from_int(int(abs(weight_diff) * self.order_size_usd)),
                )
            else:
                order = self.order_factory.market(
                    instrument_id=instrument_id,
                    order_side=OrderSide.SELL,
                    quantity=Quantity.from_int(int(abs(weight_diff) * self.order_size_usd)),
                )
                
            self.submit_order(order)
            
            self.log.info(
                f"Executing {order.side} {order.quantity} {instrument_id} "
                f"(target: {target_weight:.2%})",
                LogColor.BLUE,
            )


# Example usage in backtest
def create_ml_trading_system():
    """Create a modern ML trading system."""
    from nautilus_trader.backtest.engine import BacktestEngine
    from nautilus_trader.config import BacktestEngineConfig
    from nautilus_trader.model.identifiers import Venue
    
    # Create actors for different models
    momentum_actor = MLInferenceActor(
        config=MLInferenceActorConfig(
            actor_id="MLActor-MOMENTUM",
            model_id="momentum_v2",
            model_path="/models/momentum.pkl",
            bar_type=BarType.from_str("AAPL.NASDAQ-1-MINUTE-LAST-EXTERNAL"),
        )
    )
    
    mean_reversion_actor = MLInferenceActor(
        config=MLInferenceActorConfig(
            actor_id="MLActor-MEANREV",
            model_id="mean_reversion_v1", 
            model_path="/models/mean_reversion.pkl",
            bar_type=BarType.from_str("AAPL.NASDAQ-1-MINUTE-LAST-EXTERNAL"),
        )
    )
    
    # Portfolio constructor
    portfolio_actor = PortfolioConstructorActor(
        config=PortfolioConstructorConfig(
            actor_id="PORTFOLIO-OPT",
            model_weights={
                "momentum_v2": 0.6,
                "mean_reversion_v1": 0.4,
            },
            rebalance_interval_seconds=300,
        )
    )
    
    # Execution strategy
    strategy = MLPortfolioStrategy(
        config=MLPortfolioStrategyConfig(
            strategy_id="ML-EXEC",
            portfolio_actor_id="PORTFOLIO-OPT",
        )
    )
    
    # Create engine and add components
    engine = BacktestEngine(config=BacktestEngineConfig())
    
    engine.add_actor(momentum_actor)
    engine.add_actor(mean_reversion_actor) 
    engine.add_actor(portfolio_actor)
    engine.add_strategy(strategy)
    
    return engine


if __name__ == "__main__":
    # This would be run in a backtest or live environment
    engine = create_ml_trading_system()
    # engine.run()