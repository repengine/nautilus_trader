# Migration Guide: Mandatory Store Integration

## Overview

We are migrating ALL ML actors to use **mandatory store integration**. This ensures:
- **100% data persistence** - No data is ever lost
- **Feature parity** - Training and inference use identical features
- **Audit trail** - Complete history of all predictions and signals
- **Performance monitoring** - Track every model's performance
- **Reliability** - No optional components that might fail silently

## What's Changing

### Before (BAD ❌)
```python
class MyMLActor(Actor):
    def __init__(self, config, model_store=None):  # Optional store!
        self._model_store = model_store  # Might be None!
        
    def on_bar(self, bar):
        prediction = self.model.predict(features)
        
        # Data might not be stored!
        if self._model_store:
            self._model_store.write_prediction(...)
```

### After (GOOD ✅)
```python
class MyMLActor(MLActorBase):  # Inherits integrated base
    def __init__(self, config):
        super().__init__(config)  # Stores auto-initialized!
        
    def on_bar(self, bar):
        # Parent class handles ALL storage automatically
        super().on_bar(bar)
```

## Migration Steps

### Step 1: Update Your Actor Class

#### Change inheritance:
```python
# OLD
from nautilus_trader.common.actor import Actor

class MyMLActor(Actor):
    ...

# NEW
from ml.actors.base_integrated import MLActorBase

class MyMLActor(MLActorBase):
    ...
```

#### Remove store parameters:
```python
# OLD
def __init__(self, config, feature_store=None, model_store=None):
    self._feature_store = feature_store
    self._model_store = model_store

# NEW
def __init__(self, config):
    super().__init__(config)  # Stores initialized automatically!
```

#### Implement required methods:
```python
class MyMLActor(MLActorBase):
    
    def compute_features(self, data: Data) -> np.ndarray:
        """Compute features from market data."""
        # Your feature computation logic
        return features
    
    def run_inference(self, features: np.ndarray) -> tuple[float, float]:
        """Run model inference."""
        prediction = self.model.predict(features)
        confidence = self.calculate_confidence(prediction)
        return prediction, confidence
    
    def generate_signal(self, prediction: float, confidence: float, data: Data) -> Data | None:
        """Generate trading signal."""
        if abs(prediction) > self.threshold:
            return Signal(...)
        return None
```

### Step 2: Update Configuration

Add database connection to your config:
in config/actors.py:
    db_connection: "postgresql://postgres:postgres@localhost:5432/nautilus"
    feature_set_id: "my_features_v1"
    model_id: "xgboost_v2"
    strategy_id: "trend_following"
```

### Step 3: Start Infrastructure

Use Docker Compose to start everything:
```bash
cd ml/
docker-compose up -d
```

Or use the integration manager:
```python
from ml.core.integration import MLIntegrationManager

# This starts PostgreSQL and runs migrations automatically
integration = MLIntegrationManager(auto_start_postgres=True)
```

### Step 4: Update Tests

```python
# OLD TEST
def test_actor_without_store():
    actor = MyMLActor(config)  # No store
    actor.on_bar(bar)
    # No data was stored!

# NEW TEST
def test_actor_with_automatic_store():
    actor = MyMLActor(config)  # Stores auto-connected
    actor.on_bar(bar)
    
    # Verify data was stored
    assert actor._feature_store.get_latest(...) is not None
    assert actor._model_store.get_latest(...) is not None
```

## Benefits After Migration

### 1. Automatic Data Persistence
```python
# BEFORE: Developer must remember to store
if self._model_store and should_store:
    self._model_store.write_prediction(...)  # Might not happen!

# AFTER: Always happens automatically
# No code needed - parent class handles it!
```

### 2. Guaranteed Feature Parity
```python
# Training uses stored features
features = feature_store.load_features(...)
model.train(features)

# Inference stores identical features
actor.on_bar(bar)  # Features automatically stored
# Guaranteed to match training!
```

### 3. Complete Audit Trail
```sql
-- Every prediction is stored
SELECT * FROM ml_model_predictions 
WHERE model_id = 'xgboost_v2'
ORDER BY ts_event DESC;

-- Every signal is tracked
SELECT * FROM ml_strategy_signals
WHERE strategy_id = 'trend_following';
```

### 4. Performance Monitoring
```python
# Automatic metrics collection
health = actor.get_health_status()
print(f"Total predictions: {health['total_predictions']}")
print(f"Failed predictions: {health['failed_predictions']}")
print(f"Latency violations: {health['latency_violations']}")
```

## Common Issues and Solutions

### Issue 1: "Connection refused" to PostgreSQL

**Solution**: Start PostgreSQL with Docker Compose
```bash
cd ml/
docker-compose up -d
```

### Issue 2: "Table does not exist" errors

**Solution**: Run migrations
```python
from ml.core.integration import MLIntegrationManager
integration = MLIntegrationManager(auto_migrate=True)
```

### Issue 3: Performance concerns

**Solution**: Batching is automatic
```python
# Stores batch writes automatically
# No performance impact on hot path
# Configurable batch size in config
```

## Rollback Plan

If you need to temporarily disable stores (NOT RECOMMENDED):

```python
class MyMLActor(MLActorBase):
    def _init_stores(self):
        # Override to create dummy stores
        self._feature_store = DummyStore()
        self._model_store = DummyStore()
        self._strategy_store = DummyStore()
```

**WARNING**: This defeats the entire purpose and should only be used for debugging!

## Timeline

1. **Week 1**: Update all actor base classes
2. **Week 2**: Migrate existing actors to new base
3. **Week 3**: Update tests and documentation
4. **Week 4**: Deploy to staging
5. **Week 5**: Production rollout

## Checklist

- [ ] Update actor inheritance to `MLActorBase`
- [ ] Remove optional store parameters
- [ ] Implement required abstract methods
- [ ] Add database connection to config
- [ ] Start PostgreSQL with Docker Compose
- [ ] Run migrations
- [ ] Update tests to verify storage
- [ ] Test in development environment
- [ ] Deploy to staging
- [ ] Monitor metrics and health
- [ ] Production deployment

## Support

For issues or questions:
1. Check health status: `actor.get_health_status()`
2. Verify PostgreSQL is running: `docker ps`
3. Check migrations: `psql -U postgres -d nautilus -c '\dt'`
4. Review logs: `actor.log.info(...)` statements

## Conclusion

This migration ensures that **EVERY piece of data is ALWAYS stored**. No more optional stores, no more missing data, no more manual wiring. The system becomes:

- **Reliable**: Data is always persisted
- **Auditable**: Complete history available
- **Monitorable**: Performance metrics automatic
- **Consistent**: Training and inference use same features

The migration requires minimal code changes but provides massive reliability improvements.