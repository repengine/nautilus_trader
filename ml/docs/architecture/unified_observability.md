# Unified Observability Architecture: The Power Stack

## Core Components Integration

### 1. 📚 Four Domain Bookkeepers
**Role**: Authoritative record keepers for each domain

- **DataRegistry/Store**: Raw market data
- **FeatureRegistry/Store**: Feature engineering
- **ModelRegistry/Store**: ML models and predictions  
- **StrategyRegistry/Store**: Trading signals and decisions

### 2. 📊 Prometheus
**Role**: Real-time metrics and alerting

- Scrapes metrics from all registries/stores
- Provides time-series data for performance analysis
- Triggers alerts on anomalies

### 3. 🚌 Nautilus Message Bus
**Role**: Real-time event distribution

- Distributes market data events
- Propagates predictions and signals
- Enables event-driven architecture

## 🔄 The Synergy: How They Work Together

```python
from nautilus_trader.msgbus.bus import MessageBus
from prometheus_client import Counter, Histogram, Gauge
from ml.registry.data_registry import DataRegistry
from typing import Any


class UnifiedObservabilityPipeline:
    """
    Combines all 5 systems for complete pipeline observability.
    """
    
    def __init__(self, msgbus: MessageBus):
        # Core message bus for event distribution
        self.msgbus = msgbus
        
        # Domain bookkeepers
        self.data_bookkeeper = DataBookkeeper()
        self.feature_bookkeeper = FeatureBookkeeper()
        self.model_bookkeeper = ModelBookkeeper()
        self.strategy_bookkeeper = StrategyBookkeeper()
        
        # Prometheus metrics
        self.event_counter = Counter(
            'ml_pipeline_events_total',
            'Total events processed',
            ['domain', 'stage', 'status']
        )
        self.latency_histogram = Histogram(
            'ml_pipeline_latency_seconds',
            'Processing latency by stage',
            ['domain', 'stage']
        )
        self.pipeline_gauge = Gauge(
            'ml_pipeline_health',
            'Pipeline health score',
            ['domain']
        )
        
        # Wire everything together
        self._setup_subscriptions()
        self._setup_metrics_export()
    
    def _setup_subscriptions(self):
        """Subscribe to all relevant message bus topics."""
        
        # Data domain events
        self.msgbus.subscribe(
            topic="data.quote.*",
            handler=self._handle_quote_event
        )
        self.msgbus.subscribe(
            topic="data.trade.*",
            handler=self._handle_trade_event
        )
        
        # Feature domain events
        self.msgbus.subscribe(
            topic="features.computed.*",
            handler=self._handle_feature_event
        )
        
        # Model domain events
        self.msgbus.subscribe(
            topic="model.prediction.*",
            handler=self._handle_prediction_event
        )
        
        # Strategy domain events
        self.msgbus.subscribe(
            topic="strategy.signal.*",
            handler=self._handle_signal_event
        )
    
    def _handle_quote_event(self, event: Any):
        """Process quote through entire pipeline."""
        with self.latency_histogram.labels(domain='data', stage='ingestion').time():
            # 1. Record in Data bookkeeper
            self.data_bookkeeper.record_event({
                'type': 'QUOTE_RECEIVED',
                'instrument': event.instrument_id,
                'timestamp': event.ts_event
            })
            
            # 2. Update Prometheus metrics
            self.event_counter.labels(
                domain='data',
                stage='ingestion',
                status='success'
            ).inc()
            
            # 3. Publish to feature computation
            self.msgbus.publish(
                topic=f"compute.features.{event.instrument_id}",
                msg=event
            )
    
    def _handle_feature_event(self, event: Any):
        """Handle computed features."""
        with self.latency_histogram.labels(domain='features', stage='computation').time():
            # 1. Record in Feature bookkeeper
            feature_id = self.feature_bookkeeper.record_event({
                'type': 'FEATURE_COMPUTED',
                'feature_set': event.feature_set_id,
                'timestamp': event.ts_event
            })
            
            # 2. Update metrics
            self.event_counter.labels(
                domain='features',
                stage='computation',
                status='success'
            ).inc()
            
            # 3. Trigger model inference
            self.msgbus.publish(
                topic=f"model.inference.{event.instrument_id}",
                msg={'features': event.features, 'feature_id': feature_id}
            )
    
    def _handle_prediction_event(self, event: Any):
        """Handle model predictions."""
        with self.latency_histogram.labels(domain='model', stage='inference').time():
            # 1. Record in Model bookkeeper
            prediction_id = self.model_bookkeeper.record_event({
                'type': 'PREDICTION_MADE',
                'model_id': event.model_id,
                'prediction': event.prediction,
                'confidence': event.confidence
            })
            
            # 2. Update metrics
            self.event_counter.labels(
                domain='model',
                stage='inference',
                status='success'
            ).inc()
            
            # 3. Forward to strategy
            self.msgbus.publish(
                topic=f"strategy.evaluate.{event.instrument_id}",
                msg={'prediction': event.prediction, 'prediction_id': prediction_id}
            )
    
    def _handle_signal_event(self, event: Any):
        """Handle strategy signals."""
        with self.latency_histogram.labels(domain='strategy', stage='signal').time():
            # 1. Record in Strategy bookkeeper
            self.strategy_bookkeeper.record_event({
                'type': 'SIGNAL_GENERATED',
                'strategy_id': event.strategy_id,
                'signal': event.signal,
                'strength': event.strength
            })
            
            # 2. Update metrics
            self.event_counter.labels(
                domain='strategy',
                stage='signal',
                status='success'
            ).inc()
            
            # 3. Execute if actionable
            if event.signal != 0:
                self.msgbus.publish(
                    topic="execution.order.create",
                    msg=event
                )
    
    def get_end_to_end_latency(self, signal_id: str) -> float:
        """
        Trace signal back to original data and calculate total latency.
        """
        # Use bookkeepers to trace lineage
        signal = self.strategy_bookkeeper.get_event(signal_id)
        prediction = self.model_bookkeeper.get_event(signal.prediction_id)
        features = self.feature_bookkeeper.get_event(prediction.feature_id)
        data = self.data_bookkeeper.get_event(features.data_id)
        
        # Calculate end-to-end latency
        return (signal.timestamp - data.timestamp) / 1e9  # Convert to seconds
```

## 📈 Real-World Power Use Cases

### 1. **Automatic Issue Detection & Recovery**

```python
class AutoRecoverySystem:
    """Automatically detects and recovers from issues."""
    
    def __init__(self, pipeline: UnifiedObservabilityPipeline):
        self.pipeline = pipeline
        
        # Monitor Prometheus alerts
        self.alert_rules = [
            {
                'name': 'DataGap',
                'query': 'rate(ml_pipeline_events_total{domain="data"}[1m]) == 0',
                'action': self.handle_data_gap
            },
            {
                'name': 'FeatureDrift',
                'query': 'ml_feature_drift_score > 0.3',
                'action': self.trigger_model_retrain
            },
            {
                'name': 'ModelDegradation',
                'query': 'ml_model_accuracy < 0.6',
                'action': self.rollback_model
            }
        ]
    
    def handle_data_gap(self, alert):
        """Automatically trigger backfill when data gap detected."""
        # Use DataRegistry to identify gap
        gap_info = self.pipeline.data_bookkeeper.identify_gap()
        
        # Trigger backfill via message bus
        self.pipeline.msgbus.publish(
            topic="data.backfill.start",
            msg={'dataset': gap_info.dataset, 'period': gap_info.period}
        )
    
    def trigger_model_retrain(self, alert):
        """Automatically retrain model on drift detection."""
        self.pipeline.msgbus.publish(
            topic="model.retrain.start",
            msg={'reason': 'feature_drift', 'urgency': 'high'}
        )
```

### 2. **Real-Time Performance Attribution**

```python
class PerformanceAttributor:
    """Attribute trading performance to specific components."""
    
    def attribute_pnl(self, trade_id: str) -> dict:
        """Break down PnL contribution by component."""
        
        # Trace through all systems
        lineage = self.trace_trade_lineage(trade_id)
        
        return {
            'data_quality_impact': self.calculate_data_impact(lineage),
            'feature_contribution': self.calculate_feature_impact(lineage),
            'model_accuracy_impact': self.calculate_model_impact(lineage),
            'strategy_timing_impact': self.calculate_strategy_impact(lineage),
            'total_pnl': lineage.trade.pnl
        }
```

### 3. **Intelligent Circuit Breakers**

```python
class IntelligentCircuitBreaker:
    """Smart circuit breakers using all 5 systems."""
    
    def should_halt_trading(self) -> bool:
        """Decide if trading should be halted."""
        
        # Check multiple dimensions via Prometheus
        metrics = {
            'data_quality': prometheus.query('avg(ml_data_quality_score[5m])'),
            'feature_stability': prometheus.query('avg(ml_feature_drift_score[5m])'),
            'model_confidence': prometheus.query('avg(ml_model_confidence[5m])'),
            'strategy_performance': prometheus.query('sum(ml_strategy_pnl[1h])')
        }
        
        # Check event patterns via Message Bus
        recent_events = self.msgbus.get_recent_events(minutes=5)
        anomaly_score = self.detect_anomalies(recent_events)
        
        # Check bookkeeper health
        health_scores = {
            'data': self.data_bookkeeper.get_health(),
            'features': self.feature_bookkeeper.get_health(),
            'models': self.model_bookkeeper.get_health(),
            'strategies': self.strategy_bookkeeper.get_health()
        }
        
        # Intelligent decision
        if (metrics['data_quality'] < 0.8 or 
            metrics['model_confidence'] < 0.5 or
            anomaly_score > 0.7 or
            any(h['score'] < 0.6 for h in health_scores.values())):
            
            # Publish halt event
            self.msgbus.publish(
                topic="risk.trading.halt",
                msg={'reason': self.get_halt_reason(metrics, health_scores)}
            )
            return True
        
        return False
```

## 🎯 The Ultimate Benefits

### 1. **Complete Observability**
- Every event tracked (Registries)
- Every metric measured (Prometheus)  
- Every message traced (Message Bus)

### 2. **Intelligent Automation**
- Self-healing pipelines
- Auto-scaling based on load
- Automatic model retraining

### 3. **Real-Time Decision Making**
- Circuit breakers with context
- Dynamic risk adjustment
- Performance attribution

### 4. **Time Travel Debugging**
```python
# Reconstruct exact state at any moment
state = pipeline.reconstruct_state(
    timestamp="2024-01-15T14:30:00Z"
)
print(f"Market data: {state.data}")
print(f"Features: {state.features}")
print(f"Model state: {state.model}")
print(f"Strategy state: {state.strategy}")
print(f"Metrics: {state.prometheus}")
print(f"Messages in flight: {state.msgbus}")
```

## 🚀 Implementation Roadmap

### Phase 1: Core Integration (Current)
- ✅ Domain bookkeepers implemented
- ✅ Basic Prometheus metrics
- ⚠️ Partial message bus integration

### Phase 2: Full Wiring
- Wire all bookkeepers to message bus
- Add comprehensive Prometheus metrics
- Implement event correlation

### Phase 3: Intelligence Layer
- Add anomaly detection
- Implement auto-recovery
- Build circuit breakers

### Phase 4: Advanced Features
- Real-time performance attribution
- Predictive maintenance
- Self-optimizing pipelines

## 📊 Monitoring Dashboard Example

```yaml
Unified ML Pipeline Dashboard:
  Row 1 - Pipeline Flow:
    - Events/sec by domain (line chart)
    - End-to-end latency (histogram)
    - Message bus throughput (gauge)
    
  Row 2 - Domain Health:
    - Data quality score (gauge)
    - Feature drift score (gauge)
    - Model accuracy (gauge)
    - Strategy Sharpe ratio (gauge)
    
  Row 3 - Alerts & Actions:
    - Active alerts (table)
    - Auto-recovery actions (log)
    - Circuit breaker status (status panel)
    
  Row 4 - Lineage & Attribution:
    - Recent trades with full lineage (table)
    - PnL attribution breakdown (pie chart)
    - Component contribution over time (stacked area)
```

The combination of these 5 systems gives you:
- **Eyes** (Prometheus metrics)
- **Memory** (Registry/Store bookkeepers)
- **Nervous System** (Message Bus)
- **Intelligence** (Correlation & automation)
- **Action** (Automated responses)

This is enterprise-grade ML infrastructure at its finest!