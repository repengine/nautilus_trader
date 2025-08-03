# ML Inference Architecture for Nautilus Trader

## Modern Best Practices for Trading ML Systems

### 1. Multi-Layer Architecture (You're Correct!)

The modern consensus is a **three-layer architecture**:

```
┌─────────────────────────────────────────────────────────────────┐
│                        LAYER 3: EXECUTION                         │
│                    PortfolioExecutionActor                        │
│            (Risk checks, position sizing, order routing)          │
└────────────────────────────┬────────────────────────────────────┘
                             │ PortfolioTarget
┌─────────────────────────────┴────────────────────────────────────┐
│                    LAYER 2: PORTFOLIO OPTIMIZATION                │
│                    PortfolioConstructionActor                     │
│        (Combines signals, optimizes allocation, manages risk)     │
└────────────────────────────┬────────────────────────────────────┘
                             │ MLSignals
┌─────────────────────────────┴────────────────────────────────────┐
│                    LAYER 1: SIGNAL GENERATION                     │
│     MLInferenceActor1    MLInferenceActor2    MLInferenceActor3  │
│        (XGBoost)           (LightGBM)         (Neural Net)        │
└──────────────────────────────────────────────────────────────────┘
```

### 2. Why This Architecture Works

**Benefits:**
- **Ensemble Power**: Multiple models vote, reducing single-model risk
- **Specialization**: Each model can focus on different patterns
- **Risk Distribution**: No single point of ML failure
- **A/B Testing**: Easy to test new models in production
- **Gradual Rollout**: Can weight new models lower initially

**Modern Wisdom:**
- Netflix/Uber use similar ensemble approaches for recommendations
- Renaissance Technologies reportedly uses 100s of weak predictors
- Two Sigma emphasizes ensemble diversity over single complex models

### 3. Docker Deployment Architecture

```yaml
# docker-compose.ml.yml
version: '3.8'

services:
  # Layer 1: ML Inference Fleet
  ml-inference-momentum:
    image: nautilus-ml:latest
    environment:
      - MODEL_TYPE=momentum
      - MODEL_URI=mlflow://models/momentum-v2/production
      - INFERENCE_MODE=realtime
    volumes:
      - ./models:/models:ro
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'

  ml-inference-mean-reversion:
    image: nautilus-ml:latest
    environment:
      - MODEL_TYPE=mean_reversion
      - MODEL_URI=mlflow://models/mean-reversion-v3/production
    volumes:
      - ./models:/models:ro
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'

  ml-inference-sentiment:
    image: nautilus-ml:latest
    environment:
      - MODEL_TYPE=sentiment
      - MODEL_URI=mlflow://models/sentiment-v1/production
    volumes:
      - ./models:/models:ro
    deploy:
      resources:
        limits:
          memory: 4G  # NLP models need more memory
          cpus: '2.0'

  # Layer 2: Portfolio Construction
  portfolio-constructor:
    image: nautilus-ml:latest
    environment:
      - ACTOR_TYPE=portfolio_constructor
      - OPTIMIZATION_METHOD=hierarchical_risk_parity
      - REBALANCE_FREQUENCY=300  # seconds
    depends_on:
      - ml-inference-momentum
      - ml-inference-mean-reversion
      - ml-inference-sentiment
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'

  # Layer 3: Main Trading Node
  trading-node:
    image: nautilus-trader:latest
    environment:
      - STRATEGY=MLPortfolioStrategy
      - RISK_LIMIT=0.02
    depends_on:
      - portfolio-constructor
    ports:
      - "5555:5555"  # API
      - "5556:5556"  # WebSocket
```

### 4. Inference Optimization Strategies

#### A. Model Serving Options

**Option 1: Embedded Models (Current Approach)**
```python
class MLInferenceActor(Actor):
    def __init__(self, config):
        self.model = joblib.load(config.model_path)  # Fast, simple
```

**Option 2: Model Server (Modern Approach)**
```python
class MLInferenceActor(Actor):
    def __init__(self, config):
        self.model_server = ModelServerClient(
            url="http://triton-server:8000",
            model_name="momentum_model",
            version="2"
        )
    
    async def predict(self, features):
        return await self.model_server.predict_async(features)
```

**Benefits of Model Servers:**
- **Hot Swapping**: Update models without restarting
- **GPU Sharing**: Multiple models share GPU resources
- **Optimization**: TensorRT, ONNX optimizations
- **Monitoring**: Built-in metrics and logging

**Popular Model Servers:**
- NVIDIA Triton (supports all frameworks)
- TorchServe (PyTorch specific)
- TensorFlow Serving
- Seldon Core (Kubernetes native)

#### B. Feature Computation Optimization

**1. Incremental Updates (Critical for Latency)**
```python
class IncrementalFeatureEngine:
    def __init__(self):
        self.sma_20 = SimpleMovingAverage(20)
        self.rsi = RelativeStrengthIndex(14)
        
    def update(self, bar: Bar) -> dict:
        # O(1) updates instead of O(n) recalculation
        self.sma_20.update(bar.close)
        self.rsi.update(bar.close)
        
        return {
            'sma_20': self.sma_20.value,
            'rsi': self.rsi.value,
            'price_sma_ratio': bar.close / self.sma_20.value
        }
```

**2. Feature Store Pattern**
```python
# Shared feature computation service
class FeatureStoreActor(Actor):
    """Computes features once, broadcasts to all models."""
    
    def on_bar(self, bar: Bar):
        features = self.compute_features(bar)
        
        # Broadcast to all inference actors
        self.publish_data(
            DataType(MLFeatures),
            MLFeatures(
                instrument_id=bar.instrument_id,
                features=features,
                ts_event=bar.ts_event
            )
        )
```

#### C. Ensemble Strategies

**1. Simple Voting**
```python
class PortfolioConstructionActor(Actor):
    def aggregate_signals(self, signals: list[MLSignal]) -> float:
        # Simple average
        return sum(s.prediction for s in signals) / len(signals)
```

**2. Weighted Ensemble**
```python
def aggregate_signals(self, signals: list[MLSignal]) -> float:
    # Weight by recent performance
    weights = self.get_dynamic_weights()  # Based on trailing Sharpe
    return sum(s.prediction * weights[s.model_id] for s in signals)
```

**3. Stacking/Meta-Learning**
```python
def aggregate_signals(self, signals: list[MLSignal]) -> float:
    # Use another model to combine predictions
    meta_features = np.array([s.prediction for s in signals])
    return self.meta_model.predict(meta_features.reshape(1, -1))[0]
```

### 5. Production Best Practices

#### A. Model Versioning & Rollout

```python
class MLInferenceActor(Actor):
    def __init__(self, config):
        self.model_a = self.load_model(config.model_a_uri)  # 80% weight
        self.model_b = self.load_model(config.model_b_uri)  # 20% weight (new)
        
    def predict(self, features):
        pred_a = self.model_a.predict(features)
        pred_b = self.model_b.predict(features)
        
        # Gradual rollout
        return 0.8 * pred_a + 0.2 * pred_b
```

#### B. Circuit Breakers

```python
class MLInferenceActor(Actor):
    def __init__(self, config):
        self.prediction_bounds = (-3.0, 3.0)  # 3 sigma
        self.daily_loss_limit = -0.05  # -5%
        
    def validate_prediction(self, pred: float) -> float:
        # Sanity checks
        if not self.prediction_bounds[0] <= pred <= self.prediction_bounds[1]:
            self.log.warning(f"Prediction {pred} out of bounds!")
            return 0.0  # Neutral
            
        # Circuit breaker
        if self.daily_pnl < self.daily_loss_limit:
            self.log.warning("Daily loss limit hit, going neutral")
            return 0.0
            
        return pred
```

#### C. A/B Testing Framework

```python
class ABTestPortfolioActor(Actor):
    def __init__(self, config):
        self.control_allocation = 0.7
        self.experiment_allocation = 0.3
        
    def construct_portfolio(self, signals):
        control_weights = self.control_strategy(signals)
        experiment_weights = self.experiment_strategy(signals)
        
        # Track separately for analysis
        self.track_performance('control', control_weights)
        self.track_performance('experiment', experiment_weights)
        
        # Blend allocations
        return blend_weights(
            control_weights, self.control_allocation,
            experiment_weights, self.experiment_allocation
        )
```

### 6. Monitoring & Observability

#### A. Key Metrics to Track

```python
class MLMonitoringActor(Actor):
    def track_metrics(self):
        # Model Performance
        self.track("prediction_latency_ms", self.last_inference_time)
        self.track("feature_computation_ms", self.feature_time)
        
        # Model Quality
        self.track("prediction_mean", np.mean(self.recent_predictions))
        self.track("prediction_std", np.std(self.recent_predictions))
        self.track("signal_hit_rate", self.calculate_hit_rate())
        
        # Business Metrics
        self.track("signal_to_trade_ratio", self.signals_sent / self.trades_executed)
        self.track("model_pnl_attribution", self.calculate_model_pnl())
```

#### B. Grafana Dashboard Example

```
┌─────────────────────────────────────────────────────────────┐
│                    ML Trading Dashboard                      │
├─────────────────────┬───────────────────┬──────────────────┤
│  Inference Latency  │  Signal Quality   │  Model PnL       │
│  ┌─────────────┐    │  ┌────────────┐   │  ┌────────────┐ │
│  │  P99: 12ms  │    │  │ Hit: 54.2% │   │  │ Today: +2% │ │
│  │  P95: 8ms   │    │  │ Sharp: 1.8  │   │  │ Week: +5%  │ │
│  └─────────────┘    │  └────────────┘   │  └────────────┘ │
├─────────────────────┴───────────────────┴──────────────────┤
│                    Model Predictions                         │
│  [Real-time chart showing predictions vs actual]            │
├─────────────────────────────────────────────────────────────┤
│                    Feature Importance                        │
│  [Bar chart of top 10 features by model]                    │
└─────────────────────────────────────────────────────────────┘
```

### 7. Advanced Patterns

#### A. Online Learning Integration

```python
class AdaptiveMLActor(Actor):
    def __init__(self, config):
        self.base_model = self.load_model(config.model_uri)
        self.online_adjuster = OnlineGradientBooster()
        
    def on_trade_fill(self, fill):
        # Learn from outcomes
        features = self.get_features_at_entry(fill)
        outcome = self.calculate_trade_outcome(fill)
        
        # Update online component
        self.online_adjuster.partial_fit(features, outcome)
        
    def predict(self, features):
        base_pred = self.base_model.predict(features)
        adjustment = self.online_adjuster.predict(features)
        return base_pred + 0.1 * adjustment  # Small online adjustment
```

#### B. Multi-Timeframe Ensemble

```python
class MultiTimeframeMLActor(Actor):
    def __init__(self, config):
        self.models = {
            '1min': self.load_model('models/hft_1min'),
            '5min': self.load_model('models/momentum_5min'),
            '1hour': self.load_model('models/trend_1hour'),
        }
        
    def generate_signals(self):
        signals = {}
        for timeframe, model in self.models.items():
            features = self.feature_stores[timeframe].get_latest()
            signals[timeframe] = model.predict(features)
            
        # Combine with time-decay weights
        return self.combine_timeframes(signals)
```

### 8. Common Pitfalls to Avoid

1. **Over-Engineering**: Start simple, add complexity based on evidence
2. **Ignoring Latency**: Every millisecond counts in HFT
3. **Poor Model Isolation**: One bad model shouldn't crash the system
4. **Insufficient Monitoring**: Can't improve what you don't measure
5. **Rigid Architecture**: Build for experimentation and evolution

### 9. Recommended Implementation Path

1. **Start Simple** (Week 1-2)
   - Single inference actor
   - Basic portfolio construction
   - Simple equal weighting

2. **Add Robustness** (Week 3-4)
   - Multiple models
   - Circuit breakers
   - Performance tracking

3. **Optimize** (Week 5-6)
   - Model server integration
   - Feature store pattern
   - Advanced ensembling

4. **Scale** (Month 2+)
   - A/B testing framework
   - Online learning
   - Multi-timeframe models

This architecture provides a solid foundation that can start simple and evolve based on your specific needs and performance requirements.