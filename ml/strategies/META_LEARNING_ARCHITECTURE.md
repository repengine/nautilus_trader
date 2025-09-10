# Meta-Learning Architecture for Multi-Model Orchestration

## Overview

This document outlines the architecture for using machine learning to dynamically orchestrate multiple trading models, moving beyond static aggregation to adaptive, intelligent model combination.

## Current Architecture

### Signal Flow

```
Actor 1 (Model A) ──┐
Actor 2 (Model B) ──┼──> Strategy (Static Aggregation) ──> Trade Decision
Actor 3 (Model C) ──┘
```

### Current Limitations

- **Static Weights**: Model weights don't adapt to market conditions
- **Simple Aggregation**: Basic voting or weighted average
- **No Learning**: Doesn't improve aggregation over time
- **Regime-Agnostic**: Same weights in trending vs volatile markets

## Proposed Meta-Learning Architecture

### Enhanced Signal Flow

```
Actor 1 (Model A) ──┐
Actor 2 (Model B) ──┼──> Meta-Strategy (ML Orchestrator) ──> Adaptive Trade Decision
Actor 3 (Model C) ──┘         ↑
                              │
                    Meta-Model (Learns Optimal Aggregation)
```

## Implementation Options

### Option 1: Meta-Learning Strategy (Recommended)

```python
class MetaMLStrategy(MLTradingStrategy):
    """
    Uses ML to learn optimal aggregation of multiple model signals.

    The meta-model learns:
    - Dynamic weights for each model based on market regime
    - Optimal position sizing based on model agreement
    - When to trust specific models
    - Risk allocation across signals
    """

    def __init__(self, config: MetaStrategyConfig):
        super().__init__(config)
        self.meta_model = self._load_meta_model()  # Trained offline
        self.model_performance = ModelPerformanceTracker()
        self.regime_detector = MarketRegimeDetector()

    def _aggregate_signal(self, signal: MLSignal):
        # Collect signals
        self._model_signals[signal.model_id] = signal

        if self._ready_for_decision():
            # Extract meta-features
            meta_features = self._extract_meta_features()

            # Meta-model decides weights and action
            meta_decision = self.meta_model.predict(meta_features)

            # Execute with learned parameters
            self._execute_meta_trade(meta_decision)

    def _extract_meta_features(self) -> dict:
        """
        Extract features for meta-model decision making.
        """
        return {
            # Model agreement metrics
            "prediction_mean": np.mean([s.prediction for s in signals]),
            "prediction_std": np.std([s.prediction for s in signals]),
            "confidence_mean": np.mean([s.confidence for s in signals]),
            "confidence_std": np.std([s.confidence for s in signals]),

            # Model performance tracking
            "model_1_recent_accuracy": self.performance_tracker["model_1"]["accuracy"],
            "model_1_recent_sharpe": self.performance_tracker["model_1"]["sharpe"],
            "model_2_recent_accuracy": self.performance_tracker["model_2"]["accuracy"],
            "model_2_recent_sharpe": self.performance_tracker["model_2"]["sharpe"],

            # Market regime features
            "volatility_regime": self.regime_detector.current_volatility_regime(),
            "trend_strength": self.regime_detector.trend_strength(),
            "market_microstructure": self.regime_detector.microstructure_regime(),

            # Time-based features
            "hour_of_day": datetime.now().hour,
            "day_of_week": datetime.now().weekday(),
            "minutes_to_close": self._minutes_to_market_close(),

            # Cross-model correlation
            "model_correlation_matrix": self._calculate_model_correlations(),
            "model_disagreement_score": self._calculate_disagreement(),
        }
```

### Option 2: Reinforcement Learning Orchestrator

```python
class RLOrchestrator(BaseMLStrategy):
    """
    Uses reinforcement learning to learn optimal trading policy.

    State: Model predictions + Market features
    Action: Trade decision + Position size
    Reward: Risk-adjusted returns
    """

    def __init__(self, config: RLOrchestratorConfig):
        self.rl_agent = PPOAgent(
            state_dim=self._calculate_state_dim(),
            action_space={
                "direction": ["BUY", "HOLD", "SELL"],
                "size": [0.25, 0.5, 0.75, 1.0],  # Position size multiplier
                "confidence_threshold": [0.5, 0.6, 0.7, 0.8],
            }
        )
        self.experience_buffer = ExperienceReplay(capacity=10000)

    def _process_signals(self, signals: list[MLSignal]):
        # Encode current state
        state = self._encode_state(signals)

        # RL agent decides action
        action = self.rl_agent.act(state)

        # Execute trade
        reward = self._execute_and_measure(action)

        # Store experience
        self.experience_buffer.add(state, action, reward)

        # Periodic learning
        if len(self.experience_buffer) >= self.batch_size:
            self.rl_agent.learn(self.experience_buffer.sample())
```

### Option 3: Bayesian Model Combination

```python
class BayesianEnsembleStrategy(BaseMLStrategy):
    """
    Uses Bayesian inference to update model weights based on performance.
    """

    def __init__(self, config: BayesianConfig):
        self.model_priors = config.model_priors  # Prior beliefs about models
        self.performance_window = deque(maxlen=100)
        self.model_posteriors = {}

    def update_beliefs(self, model_id: str, prediction: float, outcome: float):
        """
        Bayesian update of model reliability.
        """
        # Calculate likelihood of outcome given prediction
        likelihood = self._calculate_likelihood(prediction, outcome)

        # Update posterior
        prior = self.model_posteriors.get(model_id, self.model_priors[model_id])
        posterior = (likelihood * prior) / self._calculate_evidence()

        self.model_posteriors[model_id] = posterior

    def _aggregate_with_posteriors(self, signals: dict[str, MLSignal]) -> float:
        """
        Weight predictions by posterior probabilities.
        """
        weighted_sum = 0.0
        total_weight = 0.0

        for model_id, signal in signals.items():
            weight = self.model_posteriors.get(model_id, 0.5)
            weighted_sum += weight * signal.prediction
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.5
```

## Meta-Model Training Pipeline

### 1. Data Collection Phase

```python
def collect_meta_training_data():
    """
    Run backtest with individual models to collect training data.
    """
    meta_data = []

    for market_condition in ["trending", "volatile", "ranging"]:
        for timeframe in backtest_periods:
            # Run each model individually
            model_results = {}
            for model_id in model_list:
                results = backtest_model(model_id, timeframe)
                model_results[model_id] = results

            # Find optimal combination (oracle)
            optimal_weights = find_optimal_weights(model_results)

            # Store training example
            meta_data.append({
                "features": extract_meta_features(model_results, market_condition),
                "target": {
                    "optimal_weights": optimal_weights,
                    "expected_sharpe": calculate_sharpe(optimal_weights, model_results),
                    "max_drawdown": calculate_drawdown(optimal_weights, model_results),
                }
            })

    return meta_data
```

### 2. Meta-Model Training

```python
def train_meta_model(meta_data: list):
    """
    Train the meta-model to predict optimal weights.
    """
    # Prepare features and targets
    X = np.array([d["features"] for d in meta_data])
    y_weights = np.array([d["target"]["optimal_weights"] for d in meta_data])
    y_sharpe = np.array([d["target"]["expected_sharpe"] for d in meta_data])

    # Multi-output model
    meta_model = MultiOutputRegressor(
        LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=5,
        )
    )

    # Train with cross-validation
    cv_scores = cross_val_score(meta_model, X, y_weights, cv=PurgedKFold(n_splits=5))

    # Final training
    meta_model.fit(X, y_weights)

    return meta_model
```

### 3. Online Learning Component

```python
class OnlineMetaLearner:
    """
    Continuously updates meta-model based on realized performance.
    """

    def __init__(self, base_meta_model):
        self.meta_model = base_meta_model
        self.performance_buffer = deque(maxlen=1000)
        self.update_frequency = 100  # Update every 100 trades

    def record_outcome(self, meta_features: dict, weights_used: dict, realized_pnl: float):
        """
        Record actual outcomes for online learning.
        """
        self.performance_buffer.append({
            "features": meta_features,
            "weights": weights_used,
            "pnl": realized_pnl,
            "timestamp": time.time(),
        })

        if len(self.performance_buffer) >= self.update_frequency:
            self._update_meta_model()

    def _update_meta_model(self):
        """
        Incremental learning on recent performance.
        """
        recent_data = list(self.performance_buffer)
        X = np.array([d["features"] for d in recent_data])
        y = np.array([d["weights"] for d in recent_data])

        # Incremental update (partial_fit for models that support it)
        self.meta_model.partial_fit(X, y)
```

## Market Regime Detection

```python
class MarketRegimeDetector:
    """
    Identifies current market regime for meta-model features.
    """

    def __init__(self):
        self.lookback_window = 100
        self.regime_model = self._load_regime_model()

    def current_regime(self) -> dict:
        """
        Detect current market regime.
        """
        return {
            "volatility_regime": self._detect_volatility_regime(),  # Low/Medium/High
            "trend_regime": self._detect_trend_regime(),  # Trending/Ranging/Reversal
            "liquidity_regime": self._detect_liquidity_regime(),  # Liquid/Illiquid
            "correlation_regime": self._detect_correlation_regime(),  # High/Low correlation
        }

    def _detect_volatility_regime(self) -> str:
        # Use GARCH or regime-switching model
        current_vol = self._calculate_current_volatility()
        if current_vol < self.vol_thresholds["low"]:
            return "LOW_VOL"
        elif current_vol > self.vol_thresholds["high"]:
            return "HIGH_VOL"
        return "MEDIUM_VOL"

    def _detect_trend_regime(self) -> str:
        # Use trend strength indicators
        trend_strength = self._calculate_trend_strength()
        if abs(trend_strength) > 0.7:
            return "STRONG_TREND"
        elif abs(trend_strength) < 0.3:
            return "RANGING"
        return "WEAK_TREND"
```

## Performance Metrics for Meta-Learning

```python
class MetaPerformanceTracker:
    """
    Tracks performance of meta-model decisions.
    """

    def __init__(self):
        self.metrics = {
            "weight_accuracy": [],  # How close to optimal weights
            "sharpe_improvement": [],  # Improvement over equal weighting
            "drawdown_reduction": [],  # Reduction in max drawdown
            "regime_adaptation": {},  # Performance by regime
        }

    def evaluate_meta_decision(self, meta_weights: dict, individual_results: dict):
        """
        Evaluate quality of meta-model weight allocation.
        """
        # Calculate portfolio performance with meta weights
        portfolio_return = sum(
            meta_weights[model_id] * individual_results[model_id]["return"]
            for model_id in meta_weights
        )

        # Compare to benchmarks
        equal_weight_return = np.mean([r["return"] for r in individual_results.values()])
        best_single_return = max(r["return"] for r in individual_results.values())

        # Track improvements
        self.metrics["sharpe_improvement"].append(
            calculate_sharpe(portfolio_return) - calculate_sharpe(equal_weight_return)
        )
```

## Integration with Existing System

### 1. Configuration

```python
# ml/config/meta_strategy.py
@dataclass
class MetaStrategyConfig(MLStrategyConfig):
    """Configuration for meta-learning strategy."""

    # Meta-model settings
    meta_model_path: str
    meta_model_type: Literal["ensemble", "rl", "bayesian"]

    # Online learning
    enable_online_learning: bool = True
    online_update_frequency: int = 100

    # Regime detection
    use_regime_detection: bool = True
    regime_lookback_window: int = 100

    # Performance tracking
    track_meta_performance: bool = True
    performance_window: int = 500
```

### 2. Deployment

```python
# ml/deployment/deploy_meta_strategy.py
def deploy_meta_strategy():
    """Deploy meta-learning strategy with multiple actors."""

    # Load meta-model
    meta_model = load_model("models/meta_orchestrator.onnx")

    # Initialize actors with different models
    actors = [
        MLSignalActor(model_path="models/momentum.onnx", model_id="momentum"),
        MLSignalActor(model_path="models/mean_revert.onnx", model_id="mean_revert"),
        MLSignalActor(model_path="models/microstructure.onnx", model_id="microstructure"),
    ]

    # Initialize meta-strategy
    strategy = MetaMLStrategy(
        config=MetaStrategyConfig(
            meta_model_path="models/meta_orchestrator.onnx",
            target_model_ids=["momentum", "mean_revert", "microstructure"],
            enable_online_learning=True,
        )
    )

    # Connect actors to strategy
    for actor in actors:
        actor.register_strategy(strategy)

    return strategy, actors
```

## Benefits of Meta-Learning Approach

1. **Adaptive Weighting**: Model weights adjust to market conditions
2. **Regime Awareness**: Different models excel in different regimes
3. **Risk Management**: Position sizing based on model agreement
4. **Continuous Improvement**: Online learning from outcomes
5. **Explainability**: Can analyze why certain models are weighted higher
6. **Robust Performance**: Reduces reliance on single model

## Next Steps

1. **Phase 1**: Implement basic meta-model with static features
2. **Phase 2**: Add regime detection and dynamic features
3. **Phase 3**: Implement online learning component
4. **Phase 4**: Add reinforcement learning for position sizing
5. **Phase 5**: Production deployment with A/B testing

## Research References

- "Dynamic Model Combination for Quantitative Trading" (2023)
- "Meta-Learning for Financial Time Series" (2022)
- "Adaptive Ensemble Methods in Trading" (2021)
- "Online Learning for Portfolio Management" (2020)

## Notes

This architecture represents the next evolution of the ML trading system, moving from static model aggregation to intelligent, adaptive orchestration. The meta-learning layer learns not just what to trade, but how to best combine multiple trading signals for optimal performance.
