# ML Migration Risk Assessment & Mitigation Plan

## Overview

This document identifies key risks in migrating the ML system from OLD/trade/nautilus_ml to the new ml/ directory and provides mitigation strategies.

## Risk Categories

### 1. Technical Risks

#### Risk: Feature Parity Violation
**Severity**: CRITICAL
**Probability**: HIGH
**Impact**: Trading losses due to misaligned features between training and inference

**Indicators**:

- Test failures in `test_feature_parity.py`
- Unexpected model predictions
- Degraded backtest performance

**Mitigation**:

1. **Automated Testing**

   ```python
   # ml/tests/test_feature_parity.py
   def test_feature_parity_comprehensive():
       # Test with various market conditions
       for scenario in ['trending', 'ranging', 'volatile']:
           train_features = compute_training_features(scenario_data)
           inference_features = compute_inference_features(scenario_data)

           np.testing.assert_allclose(
               train_features,
               inference_features,
               rtol=1e-10,
               atol=1e-10,
               err_msg=f"Feature parity failed for {scenario}"
           )
   ```

2. **Feature Version Control**

   ```python
   # ml/features/versioning.py
   FEATURE_VERSION = "2.0.0"

   def validate_feature_version(model_metadata: dict) -> bool:
       return model_metadata.get('feature_version') == FEATURE_VERSION
   ```

3. **Continuous Monitoring**
   - Log feature distributions in production
   - Alert on statistical anomalies
   - Daily parity validation jobs

#### Risk: Message Bus Performance Degradation
**Severity**: HIGH
**Probability**: MEDIUM
**Impact**: Increased latency, missed trading opportunities

**Indicators**:

- Message queue depth > 1000
- Latency > 10ms
- Dropped messages

**Mitigation**:

1. **Message Batching**

   ```python
   class BatchedFeatureActor(Actor):
       def __init__(self, config):
           super().__init__(config)
           self.batch_size = 10
           self.batch_timeout = 100  # ms

       def publish_features_batch(self, features_list):
           # Publish as single message
           batch = MLFeaturesBatch(features_list)
           self.publish_data(DataType(MLFeaturesBatch), batch)
   ```

2. **Priority Queues**
   - High priority: Trading signals
   - Medium priority: Features
   - Low priority: Monitoring

3. **Circuit Breakers**

   ```python
   if self.msgbus.queue_depth > self.max_queue_depth:
       self.log.error("Message bus overloaded, halting ML signals")
       self.halt_predictions = True
   ```

#### Risk: Model Loading Failures
**Severity**: HIGH
**Probability**: LOW
**Impact**: No ML signals, manual trading only

**Mitigation**:

1. **Fallback Models**

   ```python
   def _load_model_with_fallback(self):
       try:
           self.model = self._load_primary_model()
       except Exception as e:
           self.log.error(f"Primary model failed: {e}")
           self.model = self._load_fallback_model()
   ```

2. **Model Validation**

   ```python
   def _validate_model(self, model) -> bool:
       # Test with known inputs
       test_features = np.array([[0.1, 0.2, 0.3, 0.4]])
       try:
           prediction = model.predict(test_features)
           return prediction.shape == (1,)
       except:
           return False
   ```

### 2. Data Risks

#### Risk: Historical Data Incompatibility
**Severity**: MEDIUM
**Probability**: HIGH
**Impact**: Cannot train new models on existing data

**Mitigation**:

1. **Data Migration Scripts**

   ```python
   # ml/scripts/migrate_training_data.py
   def migrate_parquet_catalog():
       old_catalog = ParquetDataCatalog(old_path)
       new_catalog = ParquetDataCatalog(new_path)

       for instrument in old_catalog.list_instruments():
           bars = old_catalog.bars(instrument)
           new_catalog.write_bars(bars, instrument)
   ```

2. **Backward Compatibility Layer**

   ```python
   class DataLoaderCompat:
       def load_bars(self, instrument: str, source: str = "auto"):
           if source == "auto":
               # Try new format first, fall back to old
               try:
                   return self._load_new_format(instrument)
               except:
                   return self._load_old_format(instrument)
   ```

### 3. Integration Risks

#### Risk: Incompatible Message Types
**Severity**: HIGH
**Probability**: MEDIUM
**Impact**: Actors cannot communicate

**Mitigation**:

1. **Message Adapters**

   ```python
   class MessageAdapter:
       @staticmethod
       def adapt_old_to_new(old_msg: OldMLPrediction) -> MLSignal:
           return MLSignal(
               instrument_id=old_msg.instrument_id,
               prediction=1 if old_msg.prediction > 0.5 else -1,
               probability=old_msg.confidence,
               confidence=old_msg.features_quality,
               ts_event=old_msg.timestamp_ns,
               ts_init=old_msg.timestamp_ns,
           )
   ```

2. **Gradual Migration**
   - Run old and new systems in parallel
   - Compare outputs
   - Switch over when validated

### 4. Performance Risks

#### Risk: Increased Latency
**Severity**: HIGH
**Probability**: MEDIUM
**Impact**: Missed trades, reduced profitability

**Mitigation**:

1. **Performance Benchmarks**

   ```python
   # ml/tests/performance/benchmarks.py
   def test_inference_latency():
       actor = MLInferenceActor(config)
       features = create_test_features()

       latencies = []
       for _ in range(1000):
           start = time.perf_counter_ns()
           actor._generate_prediction(features)
           latencies.append(time.perf_counter_ns() - start)

       p99_latency = np.percentile(latencies, 99) / 1e6  # ms
       assert p99_latency < 5.0, f"P99 latency {p99_latency}ms exceeds 5ms"
   ```

2. **Profiling Integration**

   ```python
   # ml/utils/profiling.py
   from functools import wraps
   import cProfile

   def profile_critical_path(func):
       @wraps(func)
       def wrapper(*args, **kwargs):
           if os.getenv('ML_PROFILING') == '1':
               profiler = cProfile.Profile()
               profiler.enable()
               result = func(*args, **kwargs)
               profiler.disable()
               profiler.dump_stats(f"{func.__name__}.prof")
               return result
           return func(*args, **kwargs)
       return wrapper
   ```

### 5. Operational Risks

#### Risk: Insufficient Test Coverage
**Severity**: HIGH
**Probability**: MEDIUM
**Impact**: Bugs in production, trading losses

**Mitigation**:

1. **Coverage Gates**

   ```yaml
   # .github/workflows/ml-tests.yml
   - name: Check ML Coverage
     run: |
       coverage run -m pytest ml/tests/
       coverage report --fail-under=90
   ```

2. **Test Categories**
   - Unit tests: Individual components
   - Integration tests: Message flows
   - System tests: Full pipeline
   - Performance tests: Latency/throughput

#### Risk: Configuration Errors
**Severity**: MEDIUM
**Probability**: HIGH
**Impact**: Incorrect trading behavior

**Mitigation**:

1. **Config Validation**

   ```python
   from msgspec import ValidationError

   def validate_config(config_dict: dict, config_class: type) -> Any:
       try:
           return config_class(**config_dict)
       except ValidationError as e:
           raise ValueError(f"Invalid config: {e}")
   ```

2. **Config Testing**

   ```python
   def test_config_compatibility():
       # Test all production configs load correctly
       for config_file in Path("ml/config/production").glob("*.yaml"):
           config = load_config(config_file)
           assert config.validate()
   ```

## Risk Matrix

| Risk | Probability | Impact | Mitigation Priority |
|------|------------|--------|-------------------|
| Feature Parity | HIGH | CRITICAL | IMMEDIATE |
| Message Performance | MEDIUM | HIGH | HIGH |
| Model Loading | LOW | HIGH | MEDIUM |
| Data Compatibility | HIGH | MEDIUM | HIGH |
| Latency Increase | MEDIUM | HIGH | HIGH |
| Test Coverage | MEDIUM | HIGH | HIGH |

## Migration Phases with Risk Mitigation

### Phase 1: Foundation (Low Risk)

- Set up directory structure
- Create base classes
- No production impact

### Phase 2: Feature Engineering (CRITICAL Risk)

- Implement with extensive testing
- Run parallel validation
- Daily parity checks

### Phase 3: Actor Implementation (Medium Risk)

- Start with single actor
- Monitor performance closely
- Have rollback plan

### Phase 4: Integration (High Risk)

- Gradual rollout
- A/B testing
- Continuous monitoring

## Rollback Plan

### Immediate Rollback Triggers

1. Feature parity test failures
2. Latency > 10ms sustained
3. Message bus errors > 1%
4. Model prediction anomalies

### Rollback Procedure

```bash
# 1. Stop new system
supervisorctl stop ml-actors

# 2. Restart old system
supervisorctl start nautilus-ml-old

# 3. Verify old system healthy
./scripts/health_check.py --system=old

# 4. Investigate issues
./scripts/analyze_ml_failure.py --timestamp=$(date -u +%s)
```

## Monitoring Dashboard

### Key Metrics

```python
# ml/monitoring/metrics.py
CRITICAL_METRICS = {
    'feature_parity_score': {
        'threshold': 0.99999,  # 1e-5 tolerance
        'alert': 'PagerDuty',
    },
    'inference_latency_p99': {
        'threshold': 5.0,  # ms
        'alert': 'Slack',
    },
    'message_queue_depth': {
        'threshold': 1000,
        'alert': 'Email',
    },
    'model_prediction_drift': {
        'threshold': 0.1,  # 10% drift
        'alert': 'Slack',
    },
}
```

## Success Criteria

### Go/No-Go Decision Points

#### After Phase 2 (Features)

- [ ] Feature parity tests pass 100%
- [ ] Performance benchmarks meet targets
- [ ] Integration tests pass

#### After Phase 3 (Actors)

- [ ] Message flow validated
- [ ] Latency < 5ms P99
- [ ] Memory usage stable

#### After Phase 4 (Integration)

- [ ] Parallel run matches old system
- [ ] All monitoring green
- [ ] Rollback tested successfully

## Summary

This risk assessment identifies the critical areas that could impact the ML migration:

1. **Feature parity** is the highest risk and must be validated continuously
2. **Performance** must be monitored at every stage
3. **Gradual migration** with parallel running reduces risk
4. **Comprehensive testing** is non-negotiable
5. **Rollback capability** must be maintained throughout

By following this mitigation plan, we can safely migrate the ML system while maintaining trading system integrity.
