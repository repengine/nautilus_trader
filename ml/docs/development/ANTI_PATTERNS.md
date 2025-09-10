# ML Anti-Patterns and Risk Mitigation Guide

## Overview

This document identifies common anti-patterns in ML system development and provides concrete examples of what to avoid and how to implement correct patterns. Following these guidelines prevents performance issues, security vulnerabilities, and architectural problems in the Nautilus Trader ML system.

**Target Audience**: ML Engineers, Software Engineers, DevOps Engineers working with the ML system.

## Table of Contents

- [Performance Anti-Patterns](#performance-anti-patterns)
- [Security Anti-Patterns](#security-anti-patterns)
- [Architecture Anti-Patterns](#architecture-anti-patterns)
- [Data Pipeline Anti-Patterns](#data-pipeline-anti-patterns)
- [Model Deployment Anti-Patterns](#model-deployment-anti-patterns)
- [Testing Anti-Patterns](#testing-anti-patterns)
- [Monitoring Anti-Patterns](#monitoring-anti-patterns)
- [Configuration Anti-Patterns](#configuration-anti-patterns)
- [Risk Mitigation Strategies](#risk-mitigation-strategies)

---

## Performance Anti-Patterns

### Anti-Pattern 1: Hot Path Heavy Computations

**❌ ANTI-PATTERN**: Performing heavy computations in real-time inference paths

```python
# BAD: Heavy operations in hot path
class BadMLActor(BaseMLInferenceActor):
    def on_bar(self, bar: Bar) -> None:
        # ❌ DataFrame creation in hot path
        df = pd.DataFrame([{
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        }])

        # ❌ Complex feature engineering in hot path
        df['rsi'] = ta.rsi(df['close'], window=14)
        df['macd'] = ta.macd(df['close'])
        df['bollinger'] = ta.bollinger_bands(df['close'])

        # ❌ Model training in hot path
        if len(self.historical_data) > 1000:
            self.model.fit(self.historical_data)

        # ❌ File I/O in hot path
        with open('predictions.csv', 'a') as f:
            f.write(f"{bar.ts_event},{prediction}\n")
```

**✅ CORRECT PATTERN**: Pre-allocated arrays and minimal hot path operations

```python
class GoodMLActor(BaseMLInferenceActor):
    def __init__(self, config: MLActorConfig):
        super().__init__(config)

        # Pre-allocate arrays during initialization
        self.price_buffer = np.zeros(20, dtype=np.float32)
        self.volume_buffer = np.zeros(20, dtype=np.float32)
        self.feature_array = np.zeros(5, dtype=np.float32)
        self.buffer_index = 0

        # Load model once during initialization
        self.model = self._load_model_once()

    def on_bar(self, bar: Bar) -> None:
        # ✅ Zero allocations, reuse buffers
        idx = self.buffer_index % 20
        self.price_buffer[idx] = bar.close
        self.volume_buffer[idx] = bar.volume
        self.buffer_index += 1

        # ✅ Minimal feature computation
        if self.buffer_index >= 5:
            self.feature_array[0] = bar.close
            self.feature_array[1] = self.price_buffer.mean()  # Vectorized
            # ... other minimal computations

        # ✅ Pre-loaded model inference
        prediction = self.model.predict(self.feature_array.reshape(1, -1))[0]

        # ✅ No I/O in hot path - use async logging
        self._log_prediction_async(bar.ts_event, prediction)
```

**Performance Impact**: Hot path violations can increase latency from <1ms to >50ms.

### Anti-Pattern 2: Memory Allocations in Tight Loops

**❌ ANTI-PATTERN**: Creating new objects in performance-critical loops

```python
# BAD: Allocations in loops
def compute_features_bad(bars: list[Bar]) -> list[dict]:
    results = []
    for bar in bars:
        # ❌ New dictionary allocation for each bar
        features = {
            'price': bar.close,
            'volume': bar.volume,
            'timestamp': bar.ts_event
        }
        # ❌ List append causes reallocations
        results.append(features)
    return results
```

**✅ CORRECT PATTERN**: Pre-allocated arrays with vectorized operations

```python
def compute_features_good(bars: list[Bar]) -> np.ndarray:
    n_bars = len(bars)
    # ✅ Single allocation for all data
    features = np.zeros((n_bars, 3), dtype=np.float64)

    # ✅ Vectorized assignment
    features[:, 0] = [bar.close for bar in bars]
    features[:, 1] = [bar.volume for bar in bars]
    features[:, 2] = [bar.ts_event for bar in bars]

    return features
```

### Anti-Pattern 3: Inefficient Data Structures

**❌ ANTI-PATTERN**: Using inappropriate data structures for the use case

```python
# BAD: Using lists for frequent lookups
class BadFeatureStore:
    def __init__(self):
        self.features = []  # ❌ O(n) lookup time

    def get_feature(self, instrument_id: str) -> dict:
        # ❌ Linear search through list
        for feature in self.features:
            if feature['instrument_id'] == instrument_id:
                return feature
        return None
```

**✅ CORRECT PATTERN**: Using appropriate data structures

```python
class GoodFeatureStore:
    def __init__(self):
        self.features = {}  # ✅ O(1) lookup time
        self.lru_cache = {}  # ✅ LRU cache for frequent access

    def get_feature(self, instrument_id: str) -> dict:
        # ✅ Constant time lookup
        return self.features.get(instrument_id)
```

---

## Security Anti-Patterns

### Anti-Pattern 4: Insecure Model Loading

**❌ ANTI-PATTERN**: Using pickle for model serialization

```python
# BAD: Pickle allows arbitrary code execution
import pickle

class InsecureModelLoader:
    def load_model(self, model_path: str):
        # ❌ SECURITY RISK: Pickle can execute arbitrary code
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    def save_model(self, model, model_path: str):
        # ❌ SECURITY RISK: Creates exploitable files
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
```

**✅ CORRECT PATTERN**: Using secure serialization formats

```python
import onnx
import onnxruntime as rt
from ml.registry.model_registry import ModelRegistry

class SecureModelLoader:
    def load_model(self, model_path: str) -> rt.InferenceSession:
        # ✅ ONNX is safe and performant
        if not model_path.endswith('.onnx'):
            raise SecurityError("Only ONNX models allowed in production")

        # ✅ Validate model before loading
        model = onnx.load(model_path)
        onnx.checker.check_model(model)

        # ✅ Create secure inference session
        session = rt.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']  # Explicit provider
        )
        return session
```

### Anti-Pattern 5: Hardcoded Credentials

**❌ ANTI-PATTERN**: Embedding credentials in code

```python
# BAD: Hardcoded secrets
class BadDataConnector:
    def __init__(self):
        # ❌ SECURITY RISK: Credentials in code
        self.api_key = "sk-abcd1234..."
        self.db_password = "admin123"
        self.connection_string = "postgresql://user:password@localhost/db"
```

**✅ CORRECT PATTERN**: Environment-based configuration

```python
import os
from ml.config.security import SecureConfig

class SecureDataConnector:
    def __init__(self):
        # ✅ Credentials from environment
        self.api_key = os.getenv('DATABENTO_API_KEY')
        self.db_password = os.getenv('DB_PASSWORD')

        if not self.api_key:
            raise ValueError("DATABENTO_API_KEY environment variable required")

        # ✅ Use secure configuration management
        self.config = SecureConfig.load_from_env()
```

---

## Architecture Anti-Patterns

### Anti-Pattern 6: Bypassing Mandatory Store Pattern

**❌ ANTI-PATTERN**: Creating custom storage layers instead of using mandatory stores

```python
# BAD: Custom storage bypasses architecture patterns
class BadMLActor:
    def __init__(self, config):
        # ❌ Bypasses mandatory 4-store pattern
        self.custom_db = sqlite3.connect('custom.db')
        self.redis_client = redis.Redis()
        self.file_storage = open('data.txt', 'a')

    def save_prediction(self, prediction):
        # ❌ Inconsistent storage patterns
        self.custom_db.execute("INSERT INTO predictions VALUES (?)", (prediction,))
        self.redis_client.set(f"pred_{time.time()}", prediction)
        self.file_storage.write(f"{prediction}\n")
```

**✅ CORRECT PATTERN**: Using mandatory store architecture

```python
class GoodMLActor(BaseMLInferenceActor):
    def __init__(self, config: MLActorConfig):
        # ✅ Automatic initialization of all 4 stores
        super().__init__(config)

        # Stores available automatically:
        # - self.feature_store
        # - self.model_store
        # - self.strategy_store
        # - self.data_store

    def save_prediction(self, prediction, metadata):
        # ✅ Consistent storage through mandatory stores
        self.model_store.record_prediction(
            model_id=self.config.model_id,
            prediction=prediction,
            metadata=metadata,
            ts_event=time.time_ns()
        )
```

### Anti-Pattern 7: Direct Protocol Implementation

**❌ ANTI-PATTERN**: Using inheritance instead of protocol-first design

```python
# BAD: Tight coupling through inheritance
from abc import ABC, abstractmethod

class BadStoreBase(ABC):
    @abstractmethod
    def write_data(self, data): pass

    @abstractmethod
    def read_data(self, key): pass

class BadPostgreSQLStore(BadStoreBase):
    # ❌ Forced inheritance, tight coupling
    def write_data(self, data):
        pass

    def read_data(self, key):
        pass

class BadDummyStore(BadStoreBase):
    # ❌ Must inherit even for testing
    def write_data(self, data):
        pass

    def read_data(self, key):
        pass
```

**✅ CORRECT PATTERN**: Protocol-first design

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class StoreProtocol(Protocol):
    """Protocol for store implementations."""

    def write_data(self, data) -> None: ...
    def read_data(self, key) -> dict | None: ...
    def get_health_status(self) -> dict: ...

# ✅ Implementation without inheritance
class PostgreSQLStore:
    def write_data(self, data) -> None:
        # Implementation
        pass

    def read_data(self, key) -> dict | None:
        # Implementation
        pass

    def get_health_status(self) -> dict:
        return {"status": "healthy"}

# ✅ Test implementation without inheritance
class DummyStore:
    def __init__(self):
        self.data = {}

    def write_data(self, data) -> None:
        self.data.update(data)

    def read_data(self, key) -> dict | None:
        return self.data.get(key)

    def get_health_status(self) -> dict:
        return {"status": "dummy"}

# ✅ Both implement the protocol automatically
assert isinstance(PostgreSQLStore(), StoreProtocol)
assert isinstance(DummyStore(), StoreProtocol)
```

---

## Data Pipeline Anti-Patterns

### Anti-Pattern 8: Synchronous I/O in Hot Paths

**❌ ANTI-PATTERN**: Blocking I/O operations in real-time processing

```python
# BAD: Synchronous I/O blocks processing
class BadDataProcessor:
    def process_bar(self, bar: Bar) -> None:
        # ❌ Synchronous database write blocks processing
        connection = psycopg2.connect(connection_string)
        cursor = connection.cursor()
        cursor.execute("INSERT INTO bars VALUES (%s, %s, %s)",
                      (bar.instrument_id, bar.close, bar.ts_event))
        connection.commit()  # ❌ Blocks until database confirms
        connection.close()

        # ❌ Synchronous external API call
        response = requests.get(f"https://api.external.com/validate/{bar.instrument_id}")
        if response.status_code == 200:
            self.process_validated_bar(bar)
```

**✅ CORRECT PATTERN**: Asynchronous processing with queues

```python
import asyncio
from asyncio import Queue

class GoodDataProcessor:
    def __init__(self):
        self.write_queue: Queue = Queue(maxsize=1000)
        self.validation_queue: Queue = Queue(maxsize=500)

        # Start background workers
        asyncio.create_task(self._batch_writer())
        asyncio.create_task(self._async_validator())

    def process_bar(self, bar: Bar) -> None:
        # ✅ Non-blocking queue operations
        try:
            self.write_queue.put_nowait(bar)
        except asyncio.QueueFull:
            # ✅ Graceful handling of backpressure
            logger.warning("Write queue full, dropping bar")

    async def _batch_writer(self):
        batch = []
        while True:
            # ✅ Batch writes for efficiency
            try:
                bar = await asyncio.wait_for(self.write_queue.get(), timeout=0.1)
                batch.append(bar)

                if len(batch) >= 100:  # Batch size
                    await self._write_batch(batch)
                    batch.clear()

            except asyncio.TimeoutError:
                if batch:  # Write partial batch
                    await self._write_batch(batch)
                    batch.clear()
```

### Anti-Pattern 9: Inefficient Data Transformation

**❌ ANTI-PATTERN**: Row-by-row processing instead of vectorized operations

```python
# BAD: Row-by-row processing
def bad_feature_computation(bars: pd.DataFrame) -> pd.DataFrame:
    results = []
    for index, row in bars.iterrows():  # ❌ Extremely slow
        # ❌ Individual computations for each row
        sma = bars.iloc[max(0, index-19):index+1]['close'].mean()
        volatility = bars.iloc[max(0, index-19):index+1]['close'].std()

        results.append({
            'ts_event': row['ts_event'],
            'sma_20': sma,
            'volatility_20': volatility
        })

    return pd.DataFrame(results)
```

**✅ CORRECT PATTERN**: Vectorized operations

```python
def good_feature_computation(bars: pd.DataFrame) -> pd.DataFrame:
    # ✅ Vectorized operations - much faster
    features = bars.copy()

    # ✅ Built-in pandas rolling operations
    features['sma_20'] = bars['close'].rolling(window=20).mean()
    features['volatility_20'] = bars['close'].rolling(window=20).std()

    # ✅ Vectorized numpy operations
    features['log_returns'] = np.log(bars['close'] / bars['close'].shift(1))

    return features
```

---

## Model Deployment Anti-Patterns

### Anti-Pattern 10: Runtime Model Training

**❌ ANTI-PATTERN**: Training models during inference

```python
# BAD: Training during inference
class BadPredictiveActor(BaseMLInferenceActor):
    def __init__(self, config):
        super().__init__(config)
        self.model = xgb.XGBClassifier()
        self.training_data = []

    def on_bar(self, bar: Bar) -> None:
        # ❌ Collecting training data in hot path
        self.training_data.append([bar.open, bar.high, bar.low, bar.close])

        # ❌ CRITICAL ERROR: Training in inference path
        if len(self.training_data) % 1000 == 0:
            X = np.array(self.training_data)
            y = self._generate_labels(X)  # ❌ Expensive operation
            self.model.fit(X, y)  # ❌ Blocks inference for minutes

        # Inference after potentially long training delay
        prediction = self.model.predict([[bar.open, bar.high, bar.low, bar.close]])
        self.emit_signal(prediction[0])
```

**✅ CORRECT PATTERN**: Pre-trained models with scheduled retraining

```python
class GoodPredictiveActor(BaseMLInferenceActor):
    def __init__(self, config):
        super().__init__(config)
        # ✅ Load pre-trained model from registry
        self.model = self.model_registry.load_model(config.model_id)

        # ✅ Schedule periodic retraining (cold path)
        self._schedule_retraining()

    def on_bar(self, bar: Bar) -> None:
        # ✅ Inference only - no training
        features = self._compute_features(bar)
        prediction = self.model.predict(features.reshape(1, -1))[0]

        # ✅ Log for offline analysis
        self._log_for_retraining(features, bar.ts_event)

        self.emit_signal(prediction)

    def _schedule_retraining(self):
        # ✅ Scheduled retraining in background
        asyncio.create_task(self._retrain_periodically())

    async def _retrain_periodically(self):
        while True:
            await asyncio.sleep(3600)  # ✅ Retrain hourly
            await self._retrain_model_background()
```

### Anti-Pattern 11: Missing Model Validation

**❌ ANTI-PATTERN**: Deploying models without validation

```python
# BAD: No model validation
class BadModelDeployment:
    def deploy_model(self, model_path: str):
        # ❌ No validation before deployment
        self.current_model = joblib.load(model_path)
        logger.info("Model deployed")  # ❌ Assuming success

    def predict(self, features):
        # ❌ No error handling for model failures
        return self.current_model.predict(features)
```

**✅ CORRECT PATTERN**: Comprehensive model validation

```python
class GoodModelDeployment:
    def deploy_model(self, model_path: str) -> bool:
        try:
            # ✅ Load and validate model
            new_model = self._load_and_validate(model_path)

            # ✅ Performance validation
            if not self._validate_performance(new_model):
                logger.error("Model failed performance validation")
                return False

            # ✅ Schema compatibility check
            if not self._validate_schema_compatibility(new_model):
                logger.error("Model incompatible with current schema")
                return False

            # ✅ Safe deployment with rollback capability
            old_model = self.current_model
            self.current_model = new_model

            # ✅ Validation in production
            if not self._validate_in_production(timeout=30):
                logger.error("Production validation failed, rolling back")
                self.current_model = old_model
                return False

            logger.info("Model successfully deployed and validated")
            return True

        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            return False

    def predict(self, features):
        # ✅ Error handling and circuit breaker
        try:
            return self.current_model.predict(features)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # ✅ Fallback to default prediction
            return self._fallback_prediction(features)
```

---

## Testing Anti-Patterns

### Anti-Pattern 12: Non-Deterministic Tests

**❌ ANTI-PATTERN**: Tests that randomly fail due to non-deterministic behavior

```python
# BAD: Non-deterministic test
class BadTestMLActor:
    def test_prediction_accuracy(self):
        actor = MLSignalActor(config)

        # ❌ Random data leads to random test results
        random_bars = [self._generate_random_bar() for _ in range(100)]

        predictions = []
        for bar in random_bars:
            prediction = actor.on_bar(bar)
            predictions.append(prediction)

        # ❌ Non-deterministic assertion
        accuracy = calculate_accuracy(predictions)
        assert accuracy > 0.8  # ❌ May randomly fail
```

**✅ CORRECT PATTERN**: Deterministic tests with fixed seeds

```python
class GoodTestMLActor:
    def test_prediction_accuracy(self):
        # ✅ Fixed random seed for reproducibility
        np.random.seed(42)
        random.seed(42)

        actor = MLSignalActor(config)

        # ✅ Fixed test data for deterministic results
        test_bars = self._load_fixed_test_data('test_bars.json')

        predictions = []
        for bar in test_bars:
            prediction = actor.on_bar(bar)
            predictions.append(prediction)

        # ✅ Deterministic assertion with known expected results
        expected_predictions = self._load_expected_predictions('expected.json')
        np.testing.assert_array_almost_equal(predictions, expected_predictions, decimal=6)
```

### Anti-Pattern 13: Missing Performance Tests

**❌ ANTI-PATTERN**: No performance validation in tests

```python
# BAD: No performance testing
class BadPerformanceTest:
    def test_actor_functionality(self):
        actor = MLSignalActor(config)
        bar = create_test_bar()

        # ❌ Only testing functionality, not performance
        result = actor.on_bar(bar)
        assert result is not None
        # ❌ No latency or memory usage validation
```

**✅ CORRECT PATTERN**: Performance-aware testing

```python
import time
import tracemalloc

class GoodPerformanceTest:
    def test_actor_performance_sla(self):
        actor = MLSignalActor(config)
        test_bar = create_test_bar()

        # ✅ Warmup to eliminate JIT effects
        for _ in range(100):
            actor.on_bar(test_bar)

        # ✅ Memory usage tracking
        tracemalloc.start()
        memory_start = tracemalloc.get_traced_memory()[0]

        # ✅ Latency measurement
        latencies = []
        for _ in range(1000):
            start_time = time.perf_counter_ns()
            result = actor.on_bar(test_bar)
            end_time = time.perf_counter_ns()
            latencies.append(end_time - start_time)

        memory_end = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        # ✅ Performance assertions
        p99_latency_ms = np.percentile(latencies, 99) / 1_000_000
        assert p99_latency_ms < 5.0, f"P99 latency {p99_latency_ms}ms exceeds 5ms SLA"

        # ✅ Memory allocation validation
        memory_allocated = memory_end - memory_start
        assert memory_allocated == 0, f"Hot path allocated {memory_allocated} bytes"

        # ✅ Functionality validation
        assert result is not None
```

---

## Monitoring Anti-Patterns

### Anti-Pattern 14: Direct Prometheus Client Usage

**❌ ANTI-PATTERN**: Importing prometheus_client directly

```python
# BAD: Direct prometheus usage causes conflicts
from prometheus_client import Counter, Histogram, Gauge

class BadMonitoredActor:
    def __init__(self):
        # ❌ Direct prometheus usage
        self.prediction_counter = Counter('predictions_total', 'Total predictions')
        self.latency_histogram = Histogram('latency_seconds', 'Latency')

        # ❌ Registry conflicts possible
        from prometheus_client import CollectorRegistry, REGISTRY
        self.custom_registry = CollectorRegistry()
```

**✅ CORRECT PATTERN**: Using centralized metrics bootstrap

```python
# ✅ Use centralized metrics bootstrap
from ml.common.metrics_bootstrap import get_counter, get_histogram, get_gauge

class GoodMonitoredActor:
    def __init__(self):
        # ✅ Centralized metrics management
        self.prediction_counter = get_counter(
            'ml_predictions_total',
            'Total ML predictions made',
            labels=['model_id', 'instrument_id']
        )

        self.latency_histogram = get_histogram(
            'ml_inference_latency_seconds',
            'ML inference latency distribution',
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1],
            labels=['model_id']
        )

        # ✅ No registry conflicts
```

### Anti-Pattern 15: Missing Metric Labels

**❌ ANTI-PATTERN**: Metrics without proper labeling

```python
# BAD: Metrics without context
class BadMetrics:
    def __init__(self):
        # ❌ No labels - can't distinguish between different models/instruments
        self.counter = get_counter('predictions_total', 'Predictions')
        self.errors = get_counter('errors_total', 'Errors')

    def record_prediction(self, model_id: str, instrument_id: str, prediction: float):
        # ❌ No context in metrics
        self.counter.inc()

    def record_error(self, error_type: str, component: str):
        # ❌ No error classification
        self.errors.inc()
```

**✅ CORRECT PATTERN**: Well-labeled metrics

```python
class GoodMetrics:
    def __init__(self):
        # ✅ Comprehensive labels for filtering and alerting
        self.prediction_counter = get_counter(
            'ml_predictions_total',
            'Total predictions made',
            labels=['model_id', 'instrument_id', 'prediction_class', 'confidence_level']
        )

        self.error_counter = get_counter(
            'ml_errors_total',
            'Total errors encountered',
            labels=['error_type', 'component', 'severity']
        )

    def record_prediction(self, model_id: str, instrument_id: str,
                         prediction: float, confidence: float):
        # ✅ Rich contextual information
        prediction_class = 'buy' if prediction > 0.5 else 'sell'
        confidence_level = 'high' if confidence > 0.8 else 'medium' if confidence > 0.6 else 'low'

        self.prediction_counter.inc(labels={
            'model_id': model_id,
            'instrument_id': instrument_id,
            'prediction_class': prediction_class,
            'confidence_level': confidence_level
        })

    def record_error(self, error_type: str, component: str, severity: str = 'error'):
        # ✅ Detailed error classification
        self.error_counter.inc(labels={
            'error_type': error_type,
            'component': component,
            'severity': severity
        })
```

---

## Configuration Anti-Patterns

### Anti-Pattern 16: Hardcoded Configuration

**❌ ANTI-PATTERN**: Embedding configuration values in code

```python
# BAD: Hardcoded configuration
class BadMLActor:
    def __init__(self):
        # ❌ Configuration embedded in code
        self.lookback_period = 20
        self.confidence_threshold = 0.75
        self.batch_size = 100
        self.model_path = "/models/xgb_v1.2.onnx"
        self.db_host = "localhost"
        self.db_port = 5432

        # ❌ Environment-specific hardcoding
        if os.getenv('ENV') == 'production':
            self.db_host = "prod-db-server"
        elif os.getenv('ENV') == 'staging':
            self.db_host = "staging-db-server"
```

**✅ CORRECT PATTERN**: Configuration-driven development

```python
from dataclasses import dataclass, field
from ml.config.base import BaseMLConfig

@dataclass(frozen=True)
class MLActorConfig(BaseMLConfig):
    # ✅ All configuration externalized
    lookback_period: int = 20
    confidence_threshold: float = 0.75
    batch_size: int = 100
    model_path: str = ""

    # ✅ Database configuration
    db_host: str = field(default_factory=lambda: os.getenv('DB_HOST', 'localhost'))
    db_port: int = field(default_factory=lambda: int(os.getenv('DB_PORT', '5432')))

    def __post_init__(self):
        # ✅ Validation in configuration
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.lookback_period <= 0:
            raise ValueError("lookback_period must be positive")

class GoodMLActor:
    def __init__(self, config: MLActorConfig):
        # ✅ Configuration injected as dependency
        self.config = config

        # ✅ All behavior driven by configuration
        self.lookback_period = config.lookback_period
        self.threshold = config.confidence_threshold
```

---

## Risk Mitigation Strategies

### Production System Risks

#### Risk 1: Model Performance Degradation
**Symptoms**: Gradual decrease in prediction accuracy over time
**Mitigation Strategies**:

```python
class ModelDriftMonitor:
    def __init__(self):
        self.accuracy_threshold = 0.70
        self.drift_threshold = 0.05

    def monitor_model_performance(self, predictions, actuals):
        # ✅ Track accuracy over time
        current_accuracy = self._calculate_accuracy(predictions, actuals)

        if current_accuracy < self.accuracy_threshold:
            # ✅ Automatic model rollback
            self.trigger_model_rollback()

        # ✅ Statistical drift detection
        drift_score = self._calculate_drift_score(predictions)
        if drift_score > self.drift_threshold:
            # ✅ Schedule model retraining
            self.schedule_model_retraining()
```

#### Risk 2: Data Pipeline Failures
**Symptoms**: Missing or corrupted input data
**Mitigation Strategies**:

```python
class ResilientDataPipeline:
    def __init__(self):
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=60)
        self.fallback_sources = ['primary_api', 'backup_api', 'cached_data']

    @circuit_breaker
    def fetch_data(self, symbol: str) -> pd.DataFrame:
        for source in self.fallback_sources:
            try:
                data = self._fetch_from_source(source, symbol)
                self._validate_data_quality(data)
                return data
            except Exception as e:
                logger.warning(f"Source {source} failed: {e}")
                continue

        # ✅ Graceful degradation
        return self._get_cached_data(symbol)
```

#### Risk 3: Memory Leaks in Long-Running Processes
**Symptoms**: Gradual memory usage increase over time
**Mitigation Strategies**:

```python
import tracemalloc
from functools import wraps

def memory_leak_detector(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # ✅ Track memory allocations
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        result = func(*args, **kwargs)

        snapshot_after = tracemalloc.take_snapshot()
        top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')

        # ✅ Alert on significant memory growth
        if top_stats and top_stats[0].size_diff > 1024 * 1024:  # 1MB threshold
            logger.warning(f"Potential memory leak in {func.__name__}: {top_stats[0]}")

        tracemalloc.stop()
        return result
    return wrapper
```

### Development Process Risks

#### Risk 4: Inconsistent Development Practices
**Mitigation**: Automated Quality Gates

```bash
#!/bin/bash
# ✅ Pre-commit quality gates
set -e

echo "Running quality checks..."

# Type checking
mypy ml/ --strict

# Linting
ruff check ml/

# Formatting
black --check ml/
isort --check-only ml/

# Testing
pytest ml/ --cov=ml --cov-fail-under=90

# Performance testing
pytest ml/tests/performance/ -v

echo "All quality checks passed!"
```

#### Risk 5: Configuration Drift Between Environments
**Mitigation**: Configuration Validation

```python
class EnvironmentValidator:
    def validate_production_config(self, config: MLActorConfig) -> list[str]:
        issues = []

        # ✅ Validate critical settings
        if config.confidence_threshold < 0.7:
            issues.append("Production confidence_threshold should be >= 0.7")

        if 'test' in config.model_path.lower():
            issues.append("Test model detected in production config")

        if config.db_host == 'localhost':
            issues.append("localhost database not allowed in production")

        return issues
```

## Best Practices Summary

### Development Best Practices

1. **Always use BaseMLInferenceActor** for ML components
2. **Pre-allocate arrays** for hot path operations
3. **Use ONNX models** for secure and performant inference
4. **Implement circuit breakers** for external dependencies
5. **Use protocol-first design** for flexibility and testing

### Security Best Practices

1. **Never use pickle** for model serialization
2. **Externalize all credentials** via environment variables
3. **Validate all inputs** at system boundaries
4. **Implement proper error handling** without information leakage
5. **Use secure defaults** for all configurations

### Performance Best Practices

1. **Measure P99 latency** for all hot path operations
2. **Use vectorized operations** instead of loops where possible
3. **Implement memory leak detection** for long-running processes
4. **Cache frequently accessed data** appropriately
5. **Profile regularly** to identify bottlenecks

### Testing Best Practices

1. **Use deterministic test data** for reproducible results
2. **Test performance characteristics** not just functionality
3. **Include error condition testing** in all test suites
4. **Use property-based testing** for complex algorithms
5. **Validate against production data** regularly

## Conclusion

Following these anti-pattern guidelines prevents common pitfalls that can severely impact the performance, security, and maintainability of ML systems. The patterns identified here are based on real-world experience and analysis of the Nautilus Trader ML codebase.

**Key Takeaways**:

- Performance issues often stem from inappropriate data structures and hot path violations
- Security vulnerabilities frequently arise from insecure serialization and credential management
- Architecture problems typically result from bypassing established patterns
- Testing issues usually involve non-deterministic behavior and missing performance validation

Regular review of these patterns during development and code review processes will help maintain the high quality and reliability of the ML system.

---
**Document Version**: 1.0
**Last Updated**: 2025-09-03
**Next Review**: Quarterly
**Status**: Active Reference Guide
