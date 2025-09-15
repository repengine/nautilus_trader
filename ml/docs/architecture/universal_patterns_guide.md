# Universal ML Architecture Patterns - Implementation Guide

## Overview

This guide provides detailed implementation instructions for the Universal ML Architecture Patterns that ALL ML components in Nautilus Trader MUST follow. These patterns ensure consistency, reliability, and performance across the entire ML pipeline.

## Fundamental Principle: Public API via `__init__.py`

### The `__init__.py` Contract

**CRITICAL**: Every ML module's public API is defined exclusively through its `__init__.py` file. This is the single source of truth for what a module exports.

#### Core Rules

1. **Start with `__init__.py`**: Before implementing any feature, first update the module's `__init__.py` to define the public API
2. **End with `__init__.py`**: After implementation, verify the `__init__.py` accurately reflects what was built
3. **Only Export via `__all__`**: The `__all__` list is the authoritative API contract
4. **Alphabetical Order**: All `__all__` lists MUST be alphabetically sorted
5. **No Direct Imports**: Consumers should import from the module, not from internal files

#### Implementation Workflow

```python
# STEP 1: Define the API in __init__.py
# ml/features/__init__.py
__all__ = [
    "FeatureConfig",          # Configuration class
    "FeatureEngineer",        # Main computation engine
    "validate_feature_parity", # Validation function
]

# STEP 2: Implement in internal modules
# ml/features/engineering.py (internal, not imported directly)
class FeatureConfig:
    ...

class FeatureEngineer:
    ...

# ml/features/validation.py (internal, not imported directly)
def validate_feature_parity():
    ...

# STEP 3: Import in __init__.py
from ml.features.engineering import FeatureConfig, FeatureEngineer
from ml.features.validation import validate_feature_parity

# STEP 4: Consumer usage (ONLY through public API)
# ✅ CORRECT
from ml.features import FeatureConfig, FeatureEngineer

# ❌ INCORRECT - Never import from internal modules
from ml.features.engineering import FeatureConfig  # WRONG!
```

#### API Design Principles

1. **Minimal Surface**: Export only what's necessary for external use
2. **Hide Implementation**: Internal classes/functions should not be in `__all__`
3. **Stable Contracts**: Changes to `__all__` are breaking changes
4. **Clear Documentation**: Module docstring explains the API's purpose
5. **Lazy Imports**: Use `__getattr__` for circular dependency resolution when needed

#### Example: Complete `__init__.py` Structure

```python
"""
ML Feature Engineering Module.

This module provides feature computation with guaranteed hot/cold path parity.
The public API focuses on configuration, computation, and validation.
"""

# ============================================================================
# IMPORTS (Organized by category)
# ============================================================================
from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.features.validation import FeatureParityError
from ml.features.validation import validate_feature_parity

# ============================================================================
# PUBLIC API (Alphabetically sorted)
# ============================================================================
__all__ = [
    "FeatureConfig",
    "FeatureEngineer",
    "FeatureParityError",
    "validate_feature_parity",
]
```

#### Validation Checklist

Before any PR, verify:
- [ ] `__init__.py` updated BEFORE implementation
- [ ] `__all__` list is alphabetically sorted
- [ ] Only intended public APIs are exported
- [ ] Internal implementation details are hidden
- [ ] Module docstring describes the API
- [ ] No direct imports from internal modules in tests/examples
- [ ] `__init__.py` updated AFTER implementation to match reality

## Pattern 1: Mandatory 4-Store + 4-Registry Integration

### Pattern Description
Every ML actor MUST use all 4 stores and 4 registries via `BaseMLInferenceActor` inheritance. This ensures consistent data lifecycle management and automatic component initialization.

### Implementation Requirements

#### Base Actor Inheritance

```python
from ml.actors.base import BaseMLInferenceActor

class YourCustomActor(BaseMLInferenceActor):
    """Custom ML actor with mandatory store integration."""

    def __init__(self, config: YourCustomActorConfig):
        # REQUIRED: Call super().__init__ first
        super().__init__(config)

        # Stores and registries are now automatically available:
        # - self.feature_store
        # - self.model_store
        # - self.strategy_store
        # - self.data_store
        # - self.feature_registry
        # - self.model_registry
        # - self.strategy_registry
        # - self.data_registry

        # Your custom initialization here
        self.custom_logic = self._initialize_custom_logic()
```

#### Store Access Pattern

```python
def on_bar(self, bar: Bar) -> None:
    """Example hot path implementation."""
    # ✅ CORRECT: Use pre-initialized stores
    features = self.feature_store.get_latest_features(
        instrument_id=bar.instrument_id,
        ts_event=bar.ts_event
    )

    prediction = self.model.predict(features)

    self.model_store.record_prediction(
        model_id=self.config.model_id,
        prediction=prediction,
        ts_event=bar.ts_event,
        instrument_id=bar.instrument_id
    )

    # ❌ INCORRECT: Don't create new store instances
    # feature_store = FeatureStore(connection_string=...)  # WRONG
```

#### Progressive Fallback Implementation

```python
class BaseMLInferenceActor:
    """Base implementation with progressive fallback."""

    def _init_stores(self) -> None:
        """Initialize stores with fallback strategy."""
        try:
            # Primary: Full PostgreSQL-backed stores
            self.feature_store = FeatureStore(
                connection_string=self.config.db_connection
            )
            self.model_store = ModelStore(
                persistence_config=self._get_persistence_config()
            )
            logger.info("Initialized full ML stores with PostgreSQL")

        except ConnectionError:
            # Fallback: Dummy stores with warnings
            logger.warning("PostgreSQL unavailable, using dummy stores")
            self.feature_store = DummyFeatureStore()
            self.model_store = DummyModelStore()

            # Emit metrics for monitoring
            from ml.common.metrics_manager import MetricsManager
            mm = MetricsManager.default()
            mm.inc(
                "ml_fallback_activations_total",
                "Store fallback activation events",
                labels={"store_type": "feature"},
                labelnames=("store_type",),
            )
```

#### Validation Checklist

Before deploying any ML actor, ensure:

1. ✅ Inherits from `BaseMLInferenceActor`
2. ✅ Calls `super().__init__(config)` as first line
3. ✅ Uses `self.{store_name}` for all store access
4. ✅ Does not create additional store instances
5. ✅ Handles fallback scenarios gracefully
6. ✅ Includes health monitoring for all stores

```python
def validate_store_integration(actor_instance: BaseMLInferenceActor) -> list[str]:
    """Validation function for Pattern 1 compliance."""
    issues = []

    required_stores = ["feature_store", "model_store", "strategy_store", "data_store"]
    required_registries = ["feature_registry", "model_registry", "strategy_registry", "data_registry"]

    for store_name in required_stores:
        if not hasattr(actor_instance, store_name):
            issues.append(f"Missing required store: {store_name}")
        elif getattr(actor_instance, store_name) is None:
            issues.append(f"Store {store_name} is None")

    for registry_name in required_registries:
        if not hasattr(actor_instance, registry_name):
            issues.append(f"Missing required registry: {registry_name}")

    return issues
```

## Pattern 2: Protocol-First Interface Design

### Pattern Description
Use `typing.Protocol` for all component interfaces to enable structural typing without implementation coupling. This supports duck typing for testing and clear contracts.

### Implementation Requirements

#### Protocol Definition

```python
from typing import Protocol, runtime_checkable, Any

@runtime_checkable
class FeatureStoreProtocol(Protocol):
    """Protocol for feature store implementations."""

    def write_features(
        self,
        instrument_id: str,
        features: dict[str, float],
        ts_event: int,
        ts_init: int
    ) -> None:
        """Write features to store."""
        ...

    def get_latest_features(
        self,
        instrument_id: str,
        ts_event: int
    ) -> dict[str, float] | None:
        """Get most recent features for instrument."""
        ...

    def get_health_status(self) -> dict[str, Any]:
        """Get component health information."""
        ...
```

#### Component Implementation

```python
# ✅ CORRECT: Implement protocol without inheritance
class PostgreSQLFeatureStore:
    """PostgreSQL implementation of FeatureStoreProtocol."""

    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)

    def write_features(self, instrument_id: str, features: dict[str, float],
                      ts_event: int, ts_init: int) -> None:
        # Implementation here
        pass

    def get_latest_features(self, instrument_id: str, ts_event: int) -> dict[str, float] | None:
        # Implementation here
        pass

    def get_health_status(self) -> dict[str, Any]:
        return {"status": "ok", "connection": "active"}

# ✅ CORRECT: Test implementation conforming to protocol
class DummyFeatureStore:
    """Test/fallback implementation of FeatureStoreProtocol."""

    def __init__(self):
        self.features_cache: dict[str, dict[str, float]] = {}

    def write_features(self, instrument_id: str, features: dict[str, float],
                      ts_event: int, ts_init: int) -> None:
        self.features_cache[instrument_id] = features

    def get_latest_features(self, instrument_id: str, ts_event: int) -> dict[str, float] | None:
        return self.features_cache.get(instrument_id)

    def get_health_status(self) -> dict[str, Any]:
        return {"status": "dummy", "cached_instruments": len(self.features_cache)}

# Protocol compliance is checked at runtime
assert isinstance(PostgreSQLFeatureStore(...), FeatureStoreProtocol)
assert isinstance(DummyFeatureStore(), FeatureStoreProtocol)
```

#### Consumer Implementation

```python
def process_features(store: FeatureStoreProtocol, instrument_id: str) -> bool:
    """Function using protocol-based typing."""
    try:
        # Works with any conforming implementation
        features = store.get_latest_features(instrument_id, time.time_ns())
        return features is not None
    except Exception:
        return False

# ✅ Works with both implementations
postgres_store = PostgreSQLFeatureStore("postgresql://...")
dummy_store = DummyFeatureStore()

assert process_features(postgres_store, "EUR/USD")
assert process_features(dummy_store, "EUR/USD")
```

#### Protocol Validation

```python
from typing import get_type_hints

def validate_protocol_compliance(instance: Any, protocol: type) -> list[str]:
    """Validate that an instance conforms to a protocol."""
    issues = []

    # Check runtime protocol compliance
    if not isinstance(instance, protocol):
        issues.append(f"Instance does not conform to {protocol.__name__}")
        return issues

    # Check method signatures
    protocol_hints = get_type_hints(protocol)
    for method_name in dir(protocol):
        if method_name.startswith('_'):
            continue

        if not hasattr(instance, method_name):
            issues.append(f"Missing method: {method_name}")
            continue

        method = getattr(instance, method_name)
        if not callable(method):
            issues.append(f"Attribute {method_name} is not callable")

    return issues

# Usage
issues = validate_protocol_compliance(my_store, FeatureStoreProtocol)
if issues:
    raise ValueError(f"Protocol compliance issues: {issues}")
```

## Pattern 3: Hot/Cold Path Separation

### Pattern Description
Enforce strict performance budgets by separating hot path operations (real-time, <5ms P99) from cold path operations (training, analytics, I/O).

### Implementation Requirements

#### Hot Path Implementation

```python
import numpy as np
from typing import Final

class HotPathFeatureComputer:
    """Hot path optimized feature computation."""

    # Pre-allocate all arrays during initialization
    def __init__(self, max_lookback: int = 20):
        self.MAX_LOOKBACK: Final = max_lookback
        # Pre-allocated arrays (zero allocations in hot path)
        self.price_buffer = np.zeros(max_lookback, dtype=np.float32)
        self.volume_buffer = np.zeros(max_lookback, dtype=np.float32)
        self.feature_output = np.zeros(5, dtype=np.float32)  # Fixed feature count
        self.buffer_index = 0

    def compute_features(self, price: float, volume: float) -> np.ndarray:
        """Hot path feature computation - ZERO allocations."""
        # Update circular buffer (no new allocations)
        idx = self.buffer_index % self.MAX_LOOKBACK
        self.price_buffer[idx] = price
        self.volume_buffer[idx] = volume
        self.buffer_index += 1

        # Compute features in-place
        self.feature_output[0] = price  # Current price
        self.feature_output[1] = volume  # Current volume

        # Technical indicators (vectorized, no allocations)
        if self.buffer_index >= 5:
            start_idx = max(0, (self.buffer_index - 5) % self.MAX_LOOKBACK)
            end_idx = self.buffer_index % self.MAX_LOOKBACK

            # SMA calculation using pre-allocated buffer
            self.feature_output[2] = np.mean(self.price_buffer[start_idx:end_idx])

        return self.feature_output  # Return view, no copy

# ✅ CORRECT: Hot path usage
feature_computer = HotPathFeatureComputer()

def on_bar_hot_path(bar: Bar) -> None:
    """Example hot path bar handler."""
    # Zero allocations, <1ms P99 latency
    features = feature_computer.compute_features(bar.close, bar.volume)

    # Use pre-loaded model (loaded once at startup)
    prediction = global_model.predict_proba(features.reshape(1, -1))[0, 1]

    if prediction > 0.7:  # Pre-configured threshold
        emit_signal(bar.instrument_id, "BUY", confidence=prediction)
```

#### Cold Path Implementation

```python
import pandas as pd
from pathlib import Path

class ColdPathAnalyzer:
    """Cold path operations - can use expensive operations."""

    def train_model(self, data_path: Path) -> dict[str, float]:
        """Cold path model training - can take hours."""
        # ✅ OK in cold path: Heavy I/O, large DataFrames
        df = pd.read_parquet(data_path)  # Large allocation OK

        # ✅ OK in cold path: Complex feature engineering
        features = self._compute_complex_features(df)

        # ✅ OK in cold path: Model training
        model = self._train_xgboost(features)

        # ✅ OK in cold path: Evaluation and metrics
        metrics = self._evaluate_model(model, features)

        # Save model for hot path loading
        self._save_model_for_hot_path(model)

        return metrics

    def backfill_features(self, start_date: str, end_date: str) -> None:
        """Cold path feature backfill - can use batch processing."""
        # ✅ OK in cold path: Database queries
        raw_data = self._load_historical_data(start_date, end_date)

        # ✅ OK in cold path: Batch processing
        for chunk in self._chunk_data(raw_data, chunk_size=10000):
            features = self._compute_batch_features(chunk)
            self._save_features_batch(features)
```

## Pattern 5: Timestamp Normalization

### Pattern Description
All persisted timestamps, time-window bounds, and event times are nanoseconds (ns). Any
external or ambiguous inputs must be normalized before persistence, event emission, or
bus publish to prevent unit drift and subtle bugs.

### Implementation Requirements

- Always normalize via `ml.common.timestamps.sanitize_timestamp_ns`.
- Prefer `self.clock.timestamp_ns()` for “now” in components that have a clock, falling
  back to `time.time_ns()` in non-hot paths.
- Provide a contextual `context` string and optional `logger` when normalizing.
- Keep imports lazy inside functions; never perform normalization at import time.

#### Example

```python
from typing import Optional
from ml.common.timestamps import sanitize_timestamp_ns

def compute_bounds(start_dt: Optional[datetime], end_dt: Optional[datetime]) -> tuple[int, int]:
    """Return normalized [start_ns, end_ns) for queries."""
    start_ns = sanitize_timestamp_ns(
        int(start_dt.timestamp() * 1e9),
        context="feature_store.query:start",
        logger=logger,
    ) if start_dt else 0
    end_ns = sanitize_timestamp_ns(
        int(end_dt.timestamp() * 1e9),
        context="feature_store.query:end",
        logger=logger,
    ) if end_dt else start_ns
    return start_ns, end_ns
```

See `ml/docs/development/TIMESTAMP_GUIDE.md` for details and validation commands.

#### Performance Monitoring

```python
from ml.common.metrics_bootstrap import get_histogram

# Separate metrics for hot and cold paths
hot_path_latency = get_histogram(
    "ml_hot_path_latency_seconds",
    "Hot path operation latency",
    buckets=[0.0005, 0.001, 0.002, 0.005, 0.01],  # Sub-millisecond buckets
)

cold_path_duration = get_histogram(
    "ml_cold_path_duration_seconds",
    "Cold path operation duration",
    buckets=[1, 10, 60, 300, 1800, 3600],  # Second to hour buckets
)

@hot_path_latency.time()
def hot_path_operation():
    """Monitored hot path operation."""
    pass

@cold_path_duration.time()
def cold_path_operation():
    """Monitored cold path operation."""
    pass
```

#### Validation and Testing

```python
import time
import pytest

class TestHotPathPerformance:
    """Performance tests for hot path compliance."""

    def test_hot_path_latency_sla(self):
        """Ensure hot path operations meet latency SLA."""
        feature_computer = HotPathFeatureComputer()

        # Warmup to eliminate JIT compilation effects
        for _ in range(100):
            feature_computer.compute_features(100.0, 1000.0)

        # Measure P99 latency over many iterations
        latencies = []
        for _ in range(10000):
            start = time.perf_counter_ns()
            features = feature_computer.compute_features(100.0, 1000.0)
            end = time.perf_counter_ns()
            latencies.append(end - start)

        p99_latency_ns = np.percentile(latencies, 99)
        p99_latency_ms = p99_latency_ns / 1_000_000

        # ✅ REQUIRED: P99 < 5ms for hot path
        assert p99_latency_ms < 5.0, f"Hot path P99 latency {p99_latency_ms}ms exceeds 5ms SLA"

    def test_hot_path_zero_allocations(self):
        """Ensure hot path has zero allocations after warmup."""
        import tracemalloc

        feature_computer = HotPathFeatureComputer()

        # Warmup
        for _ in range(100):
            feature_computer.compute_features(100.0, 1000.0)

        # Start memory tracking
        tracemalloc.start()
        current, peak = tracemalloc.get_traced_memory()

        # Execute hot path operations
        for _ in range(1000):
            features = feature_computer.compute_features(100.0, 1000.0)

        current_after, peak_after = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # ✅ REQUIRED: Zero allocations in hot path
        allocation_bytes = current_after - current
        assert allocation_bytes == 0, f"Hot path allocated {allocation_bytes} bytes"
```

## Pattern 4: Progressive Fallback Chains

### Pattern Description
All external dependencies MUST have fallback strategies to ensure system resilience. Implement graceful degradation rather than hard failures.

### Implementation Requirements

#### Database Fallback Chain

```python
from enum import Enum
from typing import Protocol

class FallbackLevel(Enum):
    PRIMARY = "primary"           # Full PostgreSQL functionality
    CACHED = "cached"            # Local cache with periodic sync
    FILE_BASED = "file_based"    # Local file storage
    DUMMY = "dummy"              # In-memory only, no persistence

class ResilientFeatureStore:
    """Feature store with progressive fallback chain."""

    def __init__(self, connection_string: str, cache_dir: Path):
        self.fallback_level = FallbackLevel.DUMMY
        self.store_impl = None

        # Try fallback chain in order
        self._init_with_fallback(connection_string, cache_dir)

    def _init_with_fallback(self, connection_string: str, cache_dir: Path) -> None:
        """Initialize with progressive fallback."""

        # Level 1: Try full PostgreSQL
        try:
            from ml.stores.feature_store import FeatureStore
            self.store_impl = FeatureStore(connection_string=connection_string)
            self._test_connection()
            self.fallback_level = FallbackLevel.PRIMARY
            logger.info("Initialized PRIMARY feature store (PostgreSQL)")
            return
        except Exception as e:
            logger.warning(f"PRIMARY store failed: {e}")

        # Level 2: Try cached store with periodic sync
        try:
            self.store_impl = CachedFeatureStore(
                primary_connection=connection_string,
                cache_dir=cache_dir,
                sync_interval_seconds=300
            )
            self.fallback_level = FallbackLevel.CACHED
            logger.info("Initialized CACHED feature store")
            return
        except Exception as e:
            logger.warning(f"CACHED store failed: {e}")

        # Level 3: Try file-based store
        try:
            self.store_impl = FileFeatureStore(cache_dir)
            self.fallback_level = FallbackLevel.FILE_BASED
            logger.warning("Initialized FILE_BASED feature store (degraded)")
            return
        except Exception as e:
            logger.warning(f"FILE_BASED store failed: {e}")

        # Level 4: Dummy store (last resort)
        self.store_impl = DummyFeatureStore()
        self.fallback_level = FallbackLevel.DUMMY
        logger.error("Initialized DUMMY feature store (no persistence)")

        # Emit metrics for monitoring
        from ml.common.metrics_bootstrap import get_counter
        fallback_counter = get_counter("ml_fallback_activations_total", "Fallback activations")
        fallback_counter.inc(labels={"component": "feature_store", "level": self.fallback_level.value})

    def write_features(self, *args, **kwargs) -> None:
        """Write with fallback handling."""
        try:
            return self.store_impl.write_features(*args, **kwargs)
        except Exception as e:
            if self.fallback_level != FallbackLevel.DUMMY:
                logger.error(f"Store operation failed, attempting fallback: {e}")
                self._attempt_fallback()
                return self.store_impl.write_features(*args, **kwargs)
            raise

    def _attempt_fallback(self) -> None:
        """Attempt to fallback to next level."""
        current_level = list(FallbackLevel).index(self.fallback_level)
        if current_level < len(FallbackLevel) - 1:
            next_level = list(FallbackLevel)[current_level + 1]
            logger.warning(f"Falling back from {self.fallback_level.value} to {next_level.value}")
            self.fallback_level = next_level
            # Re-initialize with next fallback level
            # Implementation depends on specific fallback logic
```

#### Model Loading Fallback

```python
class ResilientModelLoader:
    """Model loader with multiple fallback strategies."""

    def __init__(self, primary_registry: ModelRegistry, fallback_paths: list[Path]):
        self.primary_registry = primary_registry
        self.fallback_paths = fallback_paths
        self.loaded_models: dict[str, Any] = {}

    def load_model(self, model_id: str) -> Any:
        """Load model with fallback chain."""

        # Try cache first
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]

        # Strategy 1: Primary registry
        try:
            model = self.primary_registry.load_model(model_id)
            self.loaded_models[model_id] = model
            logger.info(f"Loaded model {model_id} from primary registry")
            return model
        except ModelNotFoundError:
            logger.warning(f"Model {model_id} not found in primary registry")
        except Exception as e:
            logger.error(f"Primary registry error for {model_id}: {e}")

        # Strategy 2: Fallback file paths
        for fallback_path in self.fallback_paths:
            try:
                model_file = fallback_path / f"{model_id}.onnx"
                if model_file.exists():
                    model = self._load_onnx_model(model_file)
                    self.loaded_models[model_id] = model
                    logger.warning(f"Loaded model {model_id} from fallback path {fallback_path}")
                    return model
            except Exception as e:
                logger.warning(f"Fallback path {fallback_path} failed for {model_id}: {e}")

        # Strategy 3: Default model
        try:
            default_model = self._load_default_model()
            logger.error(f"Using default model for {model_id} - all fallbacks failed")
            return default_model
        except Exception as e:
            raise RuntimeError(f"All fallback strategies failed for model {model_id}: {e}")
```

#### Circuit Breaker Implementation

```python
from enum import Enum
from datetime import datetime, timedelta

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    """Circuit breaker for external service calls."""

    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout_seconds: int = 60,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = timedelta(seconds=recovery_timeout_seconds)
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: datetime | None = None
        self.state = CircuitState.CLOSED

    def __call__(self, func):
        """Decorator for circuit breaker protection."""
        def wrapper(*args, **kwargs):
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker open for {func.__name__}")

            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result

            except self.expected_exception as e:
                self._on_failure()
                raise e

        return wrapper

    def _on_success(self) -> None:
        """Handle successful call."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return datetime.now() - self.last_failure_time > self.recovery_timeout

# Usage example
class ExternalDataProvider:
    def __init__(self):
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout_seconds=30,
            expected_exception=ConnectionError
        )

    @circuit_breaker
    def fetch_market_data(self, instrument_id: str) -> dict:
        """Fetch data with circuit breaker protection."""
        # Potentially failing external call
        response = requests.get(f"https://api.example.com/data/{instrument_id}")
        response.raise_for_status()
        return response.json()
```

## Pattern 5: Centralized Metrics Bootstrap

### Pattern Description
NEVER import prometheus_client directly. Use `ml.common.metrics_bootstrap` to prevent metric registry conflicts and ensure consistent naming.

### Implementation Requirements

#### Metrics Bootstrap Usage

```python
# ✅ CORRECT: Use centralized metrics bootstrap
from ml.common.metrics_bootstrap import get_counter, get_histogram, get_gauge

# Component-specific metrics
predictions_counter = get_counter(
    name="ml_predictions_total",
    documentation="Total ML predictions made",
    labels=["model_id", "instrument_id", "prediction_class"]
)

inference_latency = get_histogram(
    name="ml_inference_latency_seconds",
    documentation="ML inference latency distribution",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
    labels=["model_id", "model_type"]
)

model_accuracy = get_gauge(
    name="ml_model_accuracy_ratio",
    documentation="Current model accuracy ratio",
    labels=["model_id", "evaluation_window"]
)

# ❌ INCORRECT: Direct prometheus_client import
# from prometheus_client import Counter, Histogram  # NEVER DO THIS
```

#### Metrics Integration in Components

```python
class MLSignalActor(BaseMLInferenceActor):
    """Example actor with proper metrics integration."""

    def __init__(self, config: MLSignalActorConfig):
        super().__init__(config)

        # Initialize metrics during component setup
        self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize component-specific metrics."""
        from ml.common.metrics_bootstrap import get_counter, get_histogram

        self.predictions_counter = get_counter(
            "ml_signal_predictions_total",
            "Total signal predictions made",
            labels=["instrument_id", "signal_type", "confidence_band"]
        )

        self.feature_latency = get_histogram(
            "ml_signal_feature_latency_seconds",
            "Feature computation latency for signals",
            buckets=[0.0005, 0.001, 0.002, 0.005, 0.01],
            labels=["instrument_id", "feature_count"]
        )

        self.signal_strength = get_gauge(
            "ml_signal_strength_ratio",
            "Current signal strength",
            labels=["instrument_id", "signal_direction"]
        )

    def on_bar(self, bar: Bar) -> None:
        """Bar handler with metrics tracking."""
        # Time feature computation
        with self.feature_latency.time(labels={
            "instrument_id": bar.instrument_id.value,
            "feature_count": str(len(self.feature_config.feature_names))
        }):
            features = self.compute_features(bar)

        # Make prediction
        prediction = self.model.predict_proba(features)[0, 1]
        confidence_band = self._classify_confidence(prediction)

        # Record prediction
        self.predictions_counter.inc(labels={
            "instrument_id": bar.instrument_id.value,
            "signal_type": "BUY" if prediction > 0.5 else "SELL",
            "confidence_band": confidence_band
        })

        # Update current signal strength
        signal_direction = "long" if prediction > 0.5 else "short"
        self.signal_strength.set(
            prediction if prediction > 0.5 else 1 - prediction,
            labels={
                "instrument_id": bar.instrument_id.value,
                "signal_direction": signal_direction
            }
        )
```

#### Custom Metrics Classes

```python
from ml.common.metrics_bootstrap import MetricsCollection

class ModelPerformanceMetrics(MetricsCollection):
    """Custom metrics collection for model performance tracking."""

    def __init__(self, model_id: str):
        super().__init__(prefix="ml_model_performance")
        self.model_id = model_id

        # Initialize related metrics as a group
        self.accuracy = self.get_gauge(
            "accuracy_ratio",
            "Model accuracy over evaluation window",
            labels=["model_id", "time_window"]
        )

        self.precision = self.get_gauge(
            "precision_ratio",
            "Model precision over evaluation window",
            labels=["model_id", "time_window", "class"]
        )

        self.recall = self.get_gauge(
            "recall_ratio",
            "Model recall over evaluation window",
            labels=["model_id", "time_window", "class"]
        )

        self.drift_score = self.get_gauge(
            "drift_score",
            "Statistical drift score for model features",
            labels=["model_id", "feature_group"]
        )

    def update_evaluation_metrics(self, evaluation_results: dict) -> None:
        """Update all evaluation metrics atomically."""
        labels = {"model_id": self.model_id, "time_window": "1h"}

        self.accuracy.set(evaluation_results["accuracy"], labels=labels)

        for class_name, metrics in evaluation_results["per_class"].items():
            class_labels = {**labels, "class": class_name}
            self.precision.set(metrics["precision"], labels=class_labels)
            self.recall.set(metrics["recall"], labels=class_labels)

    def update_drift_scores(self, drift_analysis: dict[str, float]) -> None:
        """Update feature drift scores."""
        for feature_group, score in drift_analysis.items():
            self.drift_score.set(score, labels={
                "model_id": self.model_id,
                "feature_group": feature_group
            })

# Usage
model_metrics = ModelPerformanceMetrics(model_id="lgb_v1_2_3")
model_metrics.update_evaluation_metrics(evaluation_results)
```

#### Metrics Naming Conventions

```python
# ✅ CORRECT: Follow naming conventions

# Counters: Always end with _total
requests_total = get_counter("ml_api_requests_total", "Total API requests")
errors_total = get_counter("ml_processing_errors_total", "Total processing errors")

# Histograms: Use appropriate units
latency_seconds = get_histogram("ml_operation_latency_seconds", "Operation latency")
duration_seconds = get_histogram("ml_training_duration_seconds", "Training duration")

# Gauges: Use clear units
memory_bytes = get_gauge("ml_memory_usage_bytes", "Memory usage")
accuracy_ratio = get_gauge("ml_accuracy_ratio", "Model accuracy as ratio")
temperature_celsius = get_gauge("ml_gpu_temperature_celsius", "GPU temperature")

# ❌ INCORRECT: Poor naming
# bad_counter = get_counter("requests", "requests")  # Missing _total, no context
# bad_histogram = get_histogram("latency", "latency")  # No units
# bad_gauge = get_gauge("acc", "acc")  # Abbreviations, no units
```

## Pattern 6: Events, Topics, and Watermarks

### Core Rules
- Always use enums: `ml.config.events.{Stage, Source, EventStatus}` (no raw strings).
- Build topics with `ml.common.message_topics.build_topic_for_stage` and honor scheme/prefix from config.
- Emit events and update registry watermarks via the façade helper (monotonic watermarks):
  `ml.common.events_util.emit_dataset_event_and_watermark`.
- Publishing is best‑effort; wrap message bus calls in `try/except` and keep them off hot paths.

#### Minimal Example
```python
from ml.common.message_topics import build_topic_for_stage
from ml.common.events_util import emit_dataset_event_and_watermark
from ml.config.events import Stage, Source, EventStatus

# Emit event + watermark (registry)
emit_dataset_event_and_watermark(
    registry,
    dataset_id="features",
    instrument_id="EUR/USD",
    stage=Stage.FEATURE_COMPUTED,
    source=Source.HISTORICAL,
    run_id=run_id,
    ts_min=start_ns,
    ts_max=end_ns,
    count=len(rows),
    status=EventStatus.SUCCESS,
    dataset_type="features",
    component="feature_store",
)

# Publish (optional, best‑effort)
try:
    topic = build_topic_for_stage(Stage.FEATURE_COMPUTED, "EUR/USD", scheme=scheme, prefix=prefix)
    publisher.publish(topic, {"dataset_id": "features", "run_id": run_id})
except Exception:
    logger.debug("bus publish failed", exc_info=True)
```

## Pattern 7: Observability (Cold Path Only)

### Core Rules
- Use DTO builders + service; never instantiate Prometheus collectors directly.
- Persist metrics/logs off the hot path; use `MLIntegrationManager` helpers to schedule flushes.
- Emit fallback/degradation metrics with labels (e.g., `ml_fallback_activations_total`).

#### Minimal Example
```python
from ml.common.metrics_manager import MetricsManager

mm = MetricsManager.default()  # no server in unit tests
mm.inc(
    "ml_fallback_activations_total",
    "Fallback activations",
    labels={"store_type": "feature"},
    labelnames=("store_type",),
)
```

---

## Core Conventions (Quick Reference)
- Hot path: P99 < 5ms; no DataFrame/file I/O/network/training; avoid allocations; push publish/metrics to cold paths.
- Protocol‑first: type against `Protocol`; avoid concrete coupling; prefer adapters.
- DB engines: only via `EngineManager.get_engine(...)`; safe SQL; partition helpers centralized.
- Security: no `pickle`/`joblib` for model artifacts; prefer ONNX + validation; never hardcode secrets.
- Config/flags: environment‑gated toggles for gradual rollout; keep rollbacks easy.
- Tests: property/contract/metamorphic/pairwise; deterministic profiles; mark serial integration.
- Timestamps: nanoseconds only; normalize via `sanitize_timestamp_ns` with context.

## Pattern Compliance Validation

### Automated Compliance Checking

```python
import ast
import inspect
from typing import Any, Protocol, get_type_hints, runtime_checkable
from ml.common.protocols import MLComponentProtocol
from ml.actors.base import BaseMLInferenceActor

class ProtocolComplianceAnalyzer(ast.NodeVisitor):
    """AST visitor for deep protocol compliance analysis."""
    
    def __init__(self):
        self.has_protocol_imports = False
        self.implemented_protocols = []
        self.method_signatures = {}
        self.used_imports = set()
        self.try_except_blocks = []
        self.hot_path_violations = []
        self.current_method = None
        
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track imports for validation."""
        if node.module:
            self.used_imports.add(node.module)
            if node.module == "typing":
                for alias in node.names:
                    if isinstance(alias, ast.alias) and alias.name in ["Protocol", "runtime_checkable"]:
                        self.has_protocol_imports = True
            elif node.module == "prometheus_client":
                # Track direct prometheus imports (violation)
                self.used_imports.add("VIOLATION:prometheus_client")
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Analyze class definitions for protocol compliance."""
        # Check if class implements protocols
        for base in node.bases:
            if isinstance(base, ast.Name) and "Protocol" in base.id:
                self.implemented_protocols.append(base.id)
        
        # Analyze methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                self.method_signatures[item.name] = {
                    "args": [arg.arg for arg in item.args.args],
                    "returns": self._get_return_annotation(item),
                    "has_docstring": ast.get_docstring(item) is not None
                }
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Analyze function definitions for pattern compliance."""
        old_method = self.current_method
        self.current_method = node.name
        
        # Check for hot path methods
        if node.name in ["on_bar", "on_quote", "on_trade", "on_order_book"]:
            self._check_hot_path_compliance(node)
        
        self.generic_visit(node)
        self.current_method = old_method
    
    def visit_Try(self, node: ast.Try) -> None:
        """Track try/except blocks for fallback validation."""
        self.try_except_blocks.append({
            "method": self.current_method,
            "handlers": len(node.handlers),
            "has_finally": node.finalbody is not None
        })
        self.generic_visit(node)
    
    def _check_hot_path_compliance(self, node: ast.FunctionDef) -> None:
        """Check hot path methods for performance violations."""
        for child in ast.walk(node):
            # Check for pandas usage
            if isinstance(child, ast.Attribute):
                if isinstance(child.value, ast.Name) and child.value.id in ["pd", "pandas"]:
                    self.hot_path_violations.append(f"{node.name}: pandas usage in hot path")
            
            # Check for file I/O
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id in ["open", "read", "write"]:
                    self.hot_path_violations.append(f"{node.name}: file I/O in hot path")
                
                # Check for training calls
                if isinstance(child.func, ast.Attribute):
                    if child.func.attr in ["fit", "train", "compile"]:
                        self.hot_path_violations.append(f"{node.name}: model training in hot path")
    
    def _get_return_annotation(self, node: ast.FunctionDef) -> str | None:
        """Extract return type annotation."""
        if node.returns:
            return ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)
        return None


class UniversalPatternValidator:
    """
    Enhanced validator for Universal ML Architecture Patterns with AST-based analysis.
    
    This validator performs deep structural analysis using Abstract Syntax Tree (AST)
    parsing combined with runtime introspection to ensure comprehensive pattern compliance.
    """

    def validate_actor_compliance(self, actor_class: type) -> dict[str, list[str]]:
        """
        Validate actor compliance with all Universal ML Architecture Patterns.
        
        Parameters
        ----------
        actor_class : type
            The actor class to validate
            
        Returns
        -------
        dict[str, list[str]]
            Dictionary mapping pattern names to lists of compliance issues
        """
        issues = {}

        # Pattern 1: 4-Store + 4-Registry Integration
        issues["pattern_1"] = self._validate_store_integration(actor_class)

        # Pattern 2: Protocol-First Interface Design (Enhanced)
        issues["pattern_2"] = self._validate_protocol_compliance(actor_class)

        # Pattern 3: Hot/Cold Path Separation (Enhanced)
        issues["pattern_3"] = self._validate_path_separation(actor_class)

        # Pattern 4: Progressive Fallback Chains (Enhanced)
        issues["pattern_4"] = self._validate_fallback_implementation(actor_class)

        # Pattern 5: Centralized Metrics Bootstrap
        issues["pattern_5"] = self._validate_metrics_usage(actor_class)

        return {k: v for k, v in issues.items() if v}  # Only return non-empty issues

    def _validate_store_integration(self, actor_class: type) -> list[str]:
        """
        Validate Pattern 1: Mandatory 4-Store + 4-Registry Integration.
        
        Ensures actor inherits from BaseMLInferenceActor and has access to all
        required stores and registries.
        """
        issues = []

        # Check inheritance
        if not issubclass(actor_class, BaseMLInferenceActor):
            issues.append("Must inherit from BaseMLInferenceActor")
            return issues  # No point checking further if inheritance is wrong

        # Check required attributes via AST
        required_stores = ["feature_store", "model_store", "strategy_store", "data_store"]
        required_registries = ["feature_registry", "model_registry", "strategy_registry", "data_registry"]
        
        try:
            # Attempt runtime instantiation check
            test_config = {}  # Minimal config for testing
            instance = actor_class(test_config)
            
            for store in required_stores:
                if not hasattr(instance, store):
                    issues.append(f"Missing required store: {store}")
                elif getattr(instance, store) is None:
                    issues.append(f"Store {store} is None - initialization may have failed")
            
            for registry in required_registries:
                if not hasattr(instance, registry):
                    issues.append(f"Missing required registry: {registry}")
                elif getattr(instance, registry) is None:
                    issues.append(f"Registry {registry} is None - initialization may have failed")
                    
        except Exception as e:
            # Fall back to static analysis if instantiation fails
            issues.append(f"Cannot instantiate for runtime check: {e}")
            
            # Use AST to check for store access patterns
            module = inspect.getmodule(actor_class)
            if module:
                source = inspect.getsource(module)
                tree = ast.parse(source)
                
                for store in required_stores:
                    if f"self.{store}" not in source:
                        issues.append(f"No usage of required store: {store}")
                
                for registry in required_registries:
                    if f"self.{registry}" not in source:
                        issues.append(f"No usage of required registry: {registry}")

        return issues

    def _validate_protocol_compliance(self, actor_class: type) -> list[str]:
        """
        Validate Pattern 2: Protocol-First Interface Design with deep AST analysis.
        
        This enhanced method performs:
        1. AST-based structural analysis of protocol implementations
        2. Runtime protocol compliance checking
        3. Method signature validation
        4. Type hint verification
        """
        issues = []
        
        # Get module source for AST analysis
        module = inspect.getmodule(actor_class)
        if not module:
            issues.append("Cannot access module source for analysis")
            return issues
        
        try:
            source = inspect.getsource(module)
            tree = ast.parse(source)
            
            # Analyze with AST visitor
            analyzer = ProtocolComplianceAnalyzer()
            analyzer.visit(tree)
            
            # Check for protocol imports
            if not analyzer.has_protocol_imports:
                issues.append("No Protocol imports from typing module detected")
            
            # Validate MLComponentProtocol compliance if applicable
            if any(store in analyzer.method_signatures for store in 
                   ["feature_store", "model_store", "strategy_store", "data_store"]):
                # This actor uses stores, should implement MLComponentProtocol
                required_methods = ["get_health_status", "get_performance_metrics", "validate_configuration"]
                
                for method in required_methods:
                    if method not in analyzer.method_signatures:
                        issues.append(f"Missing MLComponentProtocol method: {method}")
                    else:
                        # Validate method signature
                        sig = analyzer.method_signatures[method]
                        if method == "get_health_status" and sig["returns"] != "dict[str, Any]":
                            issues.append(f"{method} should return dict[str, Any]")
                        elif method == "get_performance_metrics" and sig["returns"] != "dict[str, float]":
                            issues.append(f"{method} should return dict[str, float]")
                        elif method == "validate_configuration" and sig["returns"] != "list[str]":
                            issues.append(f"{method} should return list[str]")
            
            # Runtime protocol check for instances
            try:
                # Check if the class or its stores implement MLComponentProtocol
                test_instance = actor_class.__new__(actor_class)
                if hasattr(test_instance, "feature_store"):
                    if not isinstance(getattr(test_instance, "feature_store", None), MLComponentProtocol):
                        issues.append("feature_store does not implement MLComponentProtocol")
            except Exception:
                # Runtime instantiation not possible, rely on static analysis
                pass
                
        except Exception as e:
            issues.append(f"AST analysis failed: {e}")

        return issues

    def _validate_path_separation(self, actor_class: type) -> list[str]:
        """
        Validate Pattern 3: Hot/Cold Path Separation with AST-based analysis.
        
        Ensures hot path methods avoid heavy operations and maintain <5ms P99 latency.
        """
        issues = []
        
        module = inspect.getmodule(actor_class)
        if not module:
            return ["Cannot access module source for hot path analysis"]
        
        try:
            source = inspect.getsource(actor_class)
            tree = ast.parse(source)
            
            # Analyze with AST visitor
            analyzer = ProtocolComplianceAnalyzer()
            analyzer.visit(tree)
            
            # Report hot path violations
            issues.extend(analyzer.hot_path_violations)
            
            # Additional checks for hot path methods
            hot_path_methods = ["on_bar", "on_quote", "on_trade", "on_order_book"]
            
            for method_name in hot_path_methods:
                if hasattr(actor_class, method_name):
                    method = getattr(actor_class, method_name)
                    if method:
                        method_source = inspect.getsource(method)
                        
                        # Check for specific anti-patterns
                        if "time.sleep" in method_source:
                            issues.append(f"{method_name}: contains blocking sleep call")
                        
                        if ".to_pandas()" in method_source or "DataFrame" in method_source:
                            issues.append(f"{method_name}: uses pandas DataFrame (use numpy arrays)")
                        
                        if "requests." in method_source or "urllib" in method_source:
                            issues.append(f"{method_name}: contains network I/O")
                        
                        if not any(pattern in method_source for pattern in 
                                  ["pre_allocated", "self._buffer", "np.zeros", "np.empty"]):
                            issues.append(f"{method_name}: no evidence of pre-allocated arrays")
                            
        except Exception as e:
            issues.append(f"Hot path analysis failed: {e}")

        return issues

    def _validate_fallback_implementation(self, actor_class: type) -> list[str]:
        """
        Validate Pattern 4: Progressive Fallback Chains with AST analysis.
        
        Ensures proper error handling and fallback strategies for external dependencies.
        """
        issues = []
        
        module = inspect.getmodule(actor_class)
        if not module:
            return ["Cannot access module source for fallback analysis"]
        
        try:
            source = inspect.getsource(actor_class)
            tree = ast.parse(source)
            
            # Analyze with AST visitor
            analyzer = ProtocolComplianceAnalyzer()
            analyzer.visit(tree)
            
            # Check for try/except blocks in critical methods
            critical_methods = ["__init__", "_init_stores", "on_start", "on_stop"]
            
            for method_name in critical_methods:
                if method_name in analyzer.method_signatures:
                    # Check if method has error handling
                    method_has_handling = any(
                        block["method"] == method_name 
                        for block in analyzer.try_except_blocks
                    )
                    
                    if not method_has_handling and method_name in ["__init__", "_init_stores"]:
                        issues.append(f"{method_name}: missing error handling for initialization")
            
            # Check for fallback patterns
            if "DummyStore" not in source and "fallback" not in source.lower():
                issues.append("No evidence of fallback implementation (DummyStore or fallback logic)")
            
            # Check for circuit breaker or retry logic
            if not any(pattern in source for pattern in ["CircuitBreaker", "retry", "exponential_backoff"]):
                issues.append("No circuit breaker or retry logic for external dependencies")
                
        except Exception as e:
            issues.append(f"Fallback analysis failed: {e}")

        return issues

    def _validate_metrics_usage(self, actor_class: type) -> list[str]:
        """
        Validate Pattern 5: Centralized Metrics Bootstrap.
        
        Ensures no direct prometheus_client imports and proper use of metrics_bootstrap.
        """
        issues = []
        
        module = inspect.getmodule(actor_class)
        if not module:
            return ["Cannot access module source for metrics analysis"]
        
        try:
            source = inspect.getsource(module)
            tree = ast.parse(source)
            
            # Analyze with AST visitor
            analyzer = ProtocolComplianceAnalyzer()
            analyzer.visit(tree)
            
            # Check for prometheus_client violations
            if "VIOLATION:prometheus_client" in analyzer.used_imports:
                issues.append("Direct prometheus_client import detected - use ml.common.metrics_bootstrap")
            
            # Check for proper metrics bootstrap usage
            if "ml.common.metrics_bootstrap" not in analyzer.used_imports:
                if any(metric in source for metric in ["counter", "histogram", "gauge", "metrics"]):
                    issues.append("Uses metrics but doesn't import from ml.common.metrics_bootstrap")
            
            # Check for metrics in hot path (should use pre-initialized metrics)
            hot_path_methods = ["on_bar", "on_quote", "on_trade"]
            for method_name in hot_path_methods:
                if hasattr(actor_class, method_name):
                    method_source = inspect.getsource(getattr(actor_class, method_name))
                    if "get_counter(" in method_source or "get_histogram(" in method_source:
                        issues.append(f"{method_name}: creates metrics in hot path (should pre-initialize)")
                        
        except Exception as e:
            issues.append(f"Metrics analysis failed: {e}")

        return issues

# Enhanced usage example with detailed reporting
def validate_with_report(actor_class: type) -> None:
    """
    Validate an actor class and generate a detailed compliance report.
    
    Parameters
    ----------
    actor_class : type
        The actor class to validate
    """
    validator = UniversalPatternValidator()
    compliance_issues = validator.validate_actor_compliance(actor_class)
    
    if not compliance_issues:
        print(f"✅ {actor_class.__name__} is fully compliant with all Universal ML Architecture Patterns!")
        return
    
    print(f"❌ {actor_class.__name__} has compliance issues:\n")
    
    pattern_names = {
        "pattern_1": "4-Store + 4-Registry Integration",
        "pattern_2": "Protocol-First Interface Design",
        "pattern_3": "Hot/Cold Path Separation", 
        "pattern_4": "Progressive Fallback Chains",
        "pattern_5": "Centralized Metrics Bootstrap"
    }
    
    for pattern_key, issues in compliance_issues.items():
        pattern_name = pattern_names.get(pattern_key, pattern_key)
        print(f"  {pattern_name}:")
        for issue in issues:
            print(f"    • {issue}")
        print()

# Example usage
if __name__ == "__main__":
    from ml.actors.signal import MLSignalActor
    validate_with_report(MLSignalActor)
```

### Integration Testing

```python
import pytest
from ml.core.integration import MLIntegrationManager

class TestUniversalPatternCompliance:
    """Integration tests for pattern compliance."""

    @pytest.fixture
    def integration_manager(self):
        """Fixture providing integration manager."""
        return MLIntegrationManager(auto_start_postgres=True, auto_migrate=True)

    def test_pattern_1_store_integration(self, integration_manager):
        """Test Pattern 1: 4-Store + 4-Registry Integration."""
        # Verify all stores are initialized
        assert integration_manager.feature_store is not None
        assert integration_manager.model_store is not None
        assert integration_manager.strategy_store is not None
        assert integration_manager.data_store is not None

        # Verify all registries are initialized
        assert integration_manager.feature_registry is not None
        assert integration_manager.model_registry is not None
        assert integration_manager.strategy_registry is not None
        assert integration_manager.data_registry is not None

        # Test fallback behavior
        integration_manager.shutdown()
        # After shutdown, should fallback gracefully

    def test_pattern_2_protocol_compliance(self, integration_manager):
        """Test Pattern 2: Protocol-First Interface Design."""
        from ml.common.protocols import MLComponentProtocol

        # All stores should implement the protocol
        assert isinstance(integration_manager.feature_store, MLComponentProtocol)
        assert isinstance(integration_manager.model_store, MLComponentProtocol)

        # Protocol methods should work
        health = integration_manager.feature_store.get_health_status()
        assert "status" in health

        metrics = integration_manager.feature_store.get_performance_metrics()
        assert isinstance(metrics, dict)

    def test_pattern_3_hot_path_performance(self):
        """Test Pattern 3: Hot/Cold Path Separation."""
        from ml.actors.signal import MLSignalActor

        # Create test actor
        config = MLSignalActorConfig(
            model_path="test_model.onnx",
            instrument_id=InstrumentId.from_str("EUR/USD.SIM")
        )
        actor = MLSignalActor(config)

        # Test hot path performance
        test_bar = TestDataStubs.bar_5decimal()

        # Warmup
        for _ in range(100):
            actor.on_bar(test_bar)

        # Measure performance
        latencies = []
        for _ in range(1000):
            start = time.perf_counter_ns()
            actor.on_bar(test_bar)
            end = time.perf_counter_ns()
            latencies.append(end - start)

        p99_latency_ms = np.percentile(latencies, 99) / 1_000_000
        assert p99_latency_ms < 5.0, f"P99 latency {p99_latency_ms}ms exceeds SLA"

    def test_pattern_4_fallback_resilience(self, integration_manager):
        """Test Pattern 4: Progressive Fallback Chains."""
        # Simulate database failure
        original_connection = integration_manager.db_connection
        integration_manager.db_connection = "invalid://connection"

        # Should gracefully fallback
        try:
            # This should not raise but should use fallback
            health = integration_manager.check_health()
            # Some components should report as unhealthy but system continues
            assert health["postgres"] is False
        except Exception as e:
            pytest.fail(f"Should gracefully fallback but raised: {e}")
        finally:
            integration_manager.db_connection = original_connection

    def test_pattern_5_metrics_bootstrap(self):
        """Test Pattern 5: Centralized Metrics Bootstrap."""
        from ml.common.metrics_bootstrap import get_counter, get_histogram

        # Should create metrics without conflicts
        counter1 = get_counter("test_counter_total", "Test counter")
        counter2 = get_counter("test_counter_total", "Test counter")  # Same name

        # Should return same instance (no conflicts)
        assert counter1 is counter2

        # Metrics should work
        counter1.inc()
        # Should not raise registry conflicts
```

This comprehensive implementation guide ensures that all ML components in Nautilus Trader follow the Universal ML Architecture Patterns consistently, providing reliability, performance, and maintainability across the entire system.

## Public API Facades (Cold Path)

### The Sacred Contract of `__init__.py`

**FUNDAMENTAL RULE**: All work in the ML module begins and ends with `__init__.py` files. They are the sacred contracts that define what each module provides to the world.

#### Development Workflow

1. **START**: Read the module's `__init__.py` to understand its public API
2. **PLAN**: Update `__init__.py` with new exports BEFORE implementation
3. **IMPLEMENT**: Build the functionality in internal modules
4. **VERIFY**: Ensure `__init__.py` accurately reflects the implementation
5. **END**: Confirm all consumers use ONLY the public API from `__init__.py`

### Overview

- Provide a small, typed, domain-focused facade via each domain package's `__init__.py`.
- Orchestrators and CLIs import these domain packages; they delegate to focused implementation modules.
- Actors and other hot-path code must not import domain facades.

### Design Rules

- Facade per domain with stable, minimal entrypoints (≤ 5–7 functions):
  - `ml/data/__init__.py` — dataset building and related utilities
  - `ml/features/__init__.py` — feature registration, listing, export
  - `ml/models/__init__.py` — training, evaluation, promotion
  - `ml/evaluation/__init__.py` — dataset/prediction evaluation reports
  - `ml/monitoring/__init__.py` — metrics bootstrap + snapshots
  - `ml/deployment/__init__.py` — pipeline run/plan + validators
  - `ml/stores/__init__.py` — store health/partitions/migrations
  - `ml/strategies/__init__.py` — read-side strategy summaries
  - `ml/actors/__init__.py` — actor integration helpers
  - `ml/preprocessing/__init__.py` — dataset preprocessing helpers
  - `ml/observability/__init__.py` — observability aggregation/flush
  - `ml/registry/__init__.py` — listings of datasets/models/features/watermarks
  - `ml/consumers/__init__.py` — cold-path consumer planning/running
- Facades are cold-path only. No DataFrame creation, heavy I/O, or training in hot paths.
- Facades rely on: `ml.config.events` enums, `ml.common.message_topics`, and `ml.common.metrics_bootstrap`.
- CLIs remain thin wrappers: parse args → build config → call facade → print/save.

Layering

- CLI → `ml.<domain>` → domain services/stores/registries
- Orchestrators/Schedulers → `ml.<domain>` → domain services/stores/registries
- Hot path (actors/on_*) → pre-initialized stores; must not depend on facades

Import Examples

```python
from ml.data import DatasetBuildConfig, build_tft_dataset

cfg = DatasetBuildConfig(
    data_dir=Path("catalog"),
    out_dir=Path("ml_out/datasets/spy"),
    symbols=["SPY"],
)
result = build_tft_dataset(cfg)
print(result.dataset_parquet)
```

Contracts & TDD

- Each facade has a comment-only stub describing Definition of Done (DoD).
- Agents implement functions using TDD: write contract/property tests, then add thin wrappers.
- Keep strict typing (`mypy --strict`) and ruff clean on changed files.

### Enforcement

#### `__init__.py` Rules
- **MANDATORY**: All `__all__` lists must be alphabetically sorted
- **MANDATORY**: Work starts by reading `__init__.py` to understand the API
- **MANDATORY**: Work ends by verifying `__init__.py` reflects the implementation
- **FORBIDDEN**: Direct imports from internal modules (e.g., `from ml.features.engineering import X`)
- **REQUIRED**: All public APIs must be exported through `__all__`

#### Automated Checks
- Import-linter rule: `ml/*` may import `ml/api/*`, but domain code must not import `ml/cli/*`.
- Validators: `make validate-metrics`, `make validate-events`, and the advisory `make validate-nautilus-patterns`.
- Pre-commit hooks verify `__all__` lists are alphabetically sorted
- MyPy strict mode ensures proper typing of all exports
