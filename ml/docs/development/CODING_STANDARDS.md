# ML Coding Standards

This document establishes comprehensive coding standards for the `ml/` directory in Nautilus Trader. These standards ensure high-quality, maintainable, and performant ML code that integrates seamlessly with Nautilus Trader's architecture.

## Table of Contents

- [A. Code Structure & Organization](#a-code-structure--organization)
- [B. Type Annotations & Static Analysis](#b-type-annotations--static-analysis)
- [C. Testing Requirements](#c-testing-requirements)
- [D. Performance Requirements](#d-performance-requirements)
- [E. Configuration Management](#e-configuration-management)
- [F. Error Handling](#f-error-handling)
- [G. Documentation Standards](#g-documentation-standards)
- [H. Quality Gates](#h-quality-gates)
- [I. Nautilus-Specific Requirements](#i-nautilus-specific-requirements)
- [J. ML-Specific Standards](#j-ml-specific-standards)

---

## A. Code Structure & Organization

### File Naming Conventions

- **Python modules**: Use lowercase with underscores (`feature_engineering.py`, `model_store.py`)
- **Configuration files**: End with `_config.py` or place in `config/` directory
- **Test files**: Prefix with `test_` (`test_signal_actor.py`)
- **CLI scripts**: End with `_cli.py` (`feature_cli.py`, `tft_cli.py`)

### Module Organization

```
ml/
├── actors/          # ML inference actors
├── config/          # Configuration classes
├── data/            # Data loading and processing
├── features/        # Feature engineering
├── registry/        # Model and feature registries
├── stores/          # Data persistence layer
├── strategies/      # ML trading strategies
├── training/        # Model training components
└── monitoring/      # Metrics and monitoring
```

### Import Management

**✅ MANDATORY**: Never import ML libraries directly. Use centralized imports:

```python
# ❌ Wrong
import xgboost as xgb
import lightgbm as lgb

# ✅ Correct
from ml._imports import HAS_XGBOOST, xgb, check_ml_dependencies

if not HAS_XGBOOST:
    check_ml_dependencies(["xgboost"])
```

**Import order** (enforced by Ruff):

1. Standard library imports
2. Third-party imports
3. Nautilus core imports
4. ML module imports

```python
from __future__ import annotations

import time
from typing import Any

import numpy as np

from nautilus_trader.common.actor import Actor
from nautilus_trader.model.data import Bar

from ml._imports import HAS_ONNX, ort
from ml.config.base import MLActorConfig
```

### No Versioned Filenames

**✅ RULE**: Never create versioned files (`tft_model_v2.py`, `strategy_v3.py`)

```python
# ❌ Wrong
tft_model_v2.py
strategy_v3.py

# ✅ Correct - Use semantic versioning in registries
tft_model.py  # Version tracked in ModelRegistry
strategy.py   # Backward compatibility in same file
```

---

## B. Type Annotations & Static Analysis

### Strict Type Annotations

**✅ MANDATORY**: Complete type annotations for all functions and methods:

```python
from typing import Self
from collections.abc import Sequence

def process_features(
    self,
    data: np.ndarray,
    feature_names: list[str],
    normalize: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """Process features with normalization."""
    # Implementation
    return processed_data, stats

# Method chaining with Self
def configure(self, **kwargs: Any) -> Self:
    """Configure instance."""
    # Implementation
    return self
```

### Python 3.11+ Features

Use modern typing features:

```python
# ✅ Preferred
list[str]           # Not List[str]
dict[str, float]    # Not Dict[str, float]
tuple[int, ...]     # Variable-length tuples
str | None          # Union syntax
```

### Type Guards for Conditional Imports

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import lightgbm as lgb
    import xgboost as xgb

def train_model(
    model_type: str,
    data: np.ndarray,
) -> lgb.Booster | xgb.Booster:
    """Train model with proper typing."""
    if model_type == "lightgbm":
        if not HAS_LIGHTGBM:
            check_ml_dependencies(["lightgbm"])
        return lgb.train(...)
    # Implementation
```

### MyPy Strict Mode Compliance

**✅ MANDATORY**: All code must pass `mypy ml --strict` with zero errors:

```python
# ✅ Good - Explicit Any with justification
def handle_external_data(data: Any) -> dict[str, float]:
    """Handle data from external API with unknown structure."""
    # Justified use of Any for external interface

# ❌ Avoid implicit Any
def process_data(data):  # Missing type annotation
    return data.values   # Implicit Any
```

### Protocols and Interfaces

- Define structural interfaces with `typing.Protocol` (see `ml/stores/protocols.py`).
- Type higher layers (actors/strategies) against protocols instead of concrete stores to prevent interface drift.
- Dummy and alternate implementations must conform (e.g., implement `get_statistics`, not ad‑hoc `get_stats`).

---

## C. Testing Requirements

### Coverage Requirements

- **General Python code**: ≥80% coverage
- **ML modules**: ≥90% coverage
- **Critical hot path**: ≥95% coverage

### Test Naming Conventions

```python
def test_{function}_when_{condition}_returns_{expected}():
    """Descriptive test naming pattern."""

# Examples:
def test_feature_engineering_when_missing_data_returns_filled_values():
    """Test feature engineering handles missing data correctly."""

def test_ml_actor_when_model_fails_returns_circuit_breaker_open():
    """Test circuit breaker activates on model failures."""
```

### Property-Based Testing with Hypothesis

**✅ MANDATORY** for ML components:

```python
from hypothesis import given, strategies as st
import numpy.testing as npt

@given(
    data=st.lists(st.floats(min_value=0.01, max_value=100.0), min_size=10),
    window=st.integers(min_value=2, max_value=20),
)
def test_moving_average_properties(data: list[float], window: int) -> None:
    """Test moving average mathematical properties."""
    result = calculate_moving_average(np.array(data), window)

    # Property: Result length should be data length - window + 1
    assert len(result) == len(data) - window + 1

    # Property: All values should be within data min/max
    assert np.all(result >= min(data))
    assert np.all(result <= max(data))
```

### Test Profiles & Hooks

- Profiles:
  - PR gate: `make pytest-ml-fast`
  - Integration-fast (DB‑backed subset): `make pytest-ml-db FAST=1`
  - Full suite: run locally or in generous CI runners
- Heavy tests should patch the provided hooks instead of internals:
  - FeatureStore: `_execute_write`, `_execute_query`, `_get_connection`
  - ModelStore: `_execute_write`
  - StrategyStore: `_execute_write`, `_get_connection`
  - DataStore: `_begin_transaction`, `_update_watermark`
- Hypothesis: For function‑scoped fixtures, use `@settings(..., suppress_health_check=[HealthCheck.function_scoped_fixture])`.

### Parity Testing

**✅ MANDATORY** for feature engineering:

```python
def test_feature_parity_batch_vs_online():
    """Test batch and online feature computation parity."""
    # Batch computation
    batch_features = feature_engineer.compute_batch(data)

    # Online computation
    online_features = []
    for bar in data:
        features = feature_engineer.compute_online(bar)
        online_features.append(features)

    # Verify parity with tight tolerance
    npt.assert_allclose(
        batch_features,
        np.array(online_features),
        rtol=1e-10,
        err_msg="Batch and online features must be identical"
    )
```

### Error Condition Testing

```python
def test_ml_actor_handles_invalid_model_gracefully():
    """Test actor graceful degradation on model errors."""
    with pytest.raises(ModelLoadError):
        MLSignalActor(config_with_invalid_model_path)

def test_circuit_breaker_opens_after_threshold_failures():
    """Test circuit breaker fault tolerance."""
    actor = create_test_actor()

    # Trigger failures beyond threshold
    for _ in range(6):  # Threshold is 5
        actor.handle_prediction_failure()

    assert actor.circuit_breaker.state == CircuitBreakerState.OPEN
```

---

## D. Performance Requirements

### Hot Path vs Cold Path Separation

**Hot Path** (<5ms P99 latency):

- Real-time inference
- Feature computation during trading
- Signal generation
- Order management

**Cold Path** (training/offline):

- Model training
- Batch feature engineering
- Data validation
- Model export

```python
# ✅ Hot path - Pre-allocated arrays
class MLSignalActor:
    def __init__(self, config: MLActorConfig) -> None:
        # Pre-allocate feature buffer
        self._feature_buffer = np.zeros(
            (config.batch_size, config.n_features),
            dtype=np.float32
        )

    def on_bar(self, bar: Bar) -> None:
        """Hot path - no allocations."""
        # Reuse pre-allocated buffer
        self._compute_features_inplace(bar, self._feature_buffer[0])
```

### Zero-Allocation Patterns

```python
# ✅ Reuse arrays
def compute_features_inplace(
    self,
    input_data: np.ndarray,
    output_buffer: np.ndarray,
) -> None:
    """Compute features without allocating new arrays."""
    # Write directly to output buffer
    np.mean(input_data, axis=1, out=output_buffer[:, 0])

# ❌ Avoid in hot path
def compute_features(self, input_data: np.ndarray) -> list[float]:
    """Creates new list every call."""
    return [float(x) for x in np.mean(input_data, axis=1)]
```

### Latency Budgets

```python
import time

def validate_latency_budget(func):
    """Decorator to validate latency requirements."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if elapsed_ms > 5.0:  # Hot path budget
            logger.warning(f"{func.__name__} exceeded budget: {elapsed_ms:.2f}ms")

        return result
    return wrapper

@validate_latency_budget
def compute_features(self, bar: Bar) -> np.ndarray:
    """Must complete within 5ms."""
    # Implementation
```

---

## E. Configuration Management

### No Hardcoded Values

**✅ MANDATORY**: All constants in configuration classes:

```python
# ❌ Wrong
def compute_rsi(prices: np.ndarray) -> np.ndarray:
    window = 14  # Hardcoded
    return talib.RSI(prices, timeperiod=window)

# ✅ Correct
@dataclass(frozen=True)
class TechnicalIndicatorConfig:
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26

def compute_rsi(
    prices: np.ndarray,
    config: TechnicalIndicatorConfig,
) -> np.ndarray:
    return talib.RSI(prices, timeperiod=config.rsi_window)
```

### Configuration Class Patterns

Use frozen dataclasses extending NautilusConfig:

```python
from nautilus_trader.common.config import NautilusConfig
from nautilus_trader.common.config import PositiveInt, PositiveFloat

class MLFeatureConfig(NautilusConfig, kw_only=True, frozen=True):
    """Configuration for ML feature engineering."""

    lookback_window: PositiveInt = 100
    normalize_features: bool = True
    fill_missing_with: float = 0.0
    average_volume: PositiveFloat = 1000000.0

    def __post_init__(self) -> None:
        """Validate configuration constraints."""
        if self.lookback_window > 1000:
            raise ValidationError("lookback_window must be <= 1000")
```

### Environment Variables

```python
import os
from typing import ClassVar

class DatabaseConfig(NautilusConfig, kw_only=True, frozen=True):
    """Database connection configuration."""

    connection_string: str = "postgresql://localhost:5432/nautilus"
    pool_size: PositiveInt = 10

    # Environment variable overrides
    _ENV_MAPPING: ClassVar[dict[str, str]] = {
        "connection_string": "NAUTILUS_DB_URL",
        "pool_size": "NAUTILUS_DB_POOL_SIZE",
    }

    def __post_init__(self) -> None:
        """Apply environment variable overrides."""
        for field, env_var in self._ENV_MAPPING.items():
            if env_value := os.getenv(env_var):
                object.__setattr__(self, field, type(getattr(self, field))(env_value))

### Database & Migrations

- Canonical migrations: use `ml/stores/migrations/*.sql`; avoid legacy `ml/schema/*.sql`.
- Preflight: run `ml/stores/db_preflight.check_db_prereqs()` at startup and in CI to verify required DB functions and current‑month partitions.
- Pooling: obtain engines exclusively via `ml.core.db_engine.EngineManager.get_engine()`; do not instantiate engines directly.
- SQL: parameterize all dynamic queries via `sqlalchemy.text()` with bind parameters; avoid f‑strings in SQL.
 - Indexes: for large range scans, add BRIN on `ts_event` in addition to composite BTREE lookup indexes; see `ml/stores/migrations/007_brin_indexes.sql`.

### Timestamp Policy

- All timestamps are UNIX nanoseconds.
- Use `ml/common/timestamps.{normalize_timestamp_ns,sanitize_timestamp_ns}` in write paths.
- Policy via env: `ML_TS_NORMALIZATION_MODE` in {`warn` (default), `normalize`, `reject`}.
```

---

## F. Error Handling

### Validation Patterns

**✅ Validate early and aggressively**:

```python
def process_market_data(
    data: pl.DataFrame,
    instrument_ids: list[InstrumentId],
) -> pl.DataFrame:
    """Process market data with comprehensive validation."""
    # Input validation
    if data.is_empty():
        raise ValueError("Input data cannot be empty")

    required_columns = {"instrument_id", "ts_event", "ts_init"}
    if not required_columns.issubset(data.columns):
        missing = required_columns - set(data.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # Timestamp validation
    if not data["ts_event"].is_sorted():
        raise ValueError("Data must be sorted by ts_event")

    # Implementation
```

### Exception Hierarchies

```python
class MLError(Exception):
    """Base exception for ML module."""

class ModelLoadError(MLError):
    """Model loading failed."""

class FeatureComputationError(MLError):
    """Feature computation failed."""

class InferenceError(MLError):
    """Model inference failed."""
```

### Graceful Degradation

```python
class CircuitBreaker:
    """Fault tolerance for ML components."""

    def call(self, func: Callable[..., Any], *args: Any) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            raise CircuitBreakerOpenError("Circuit breaker is open")

        try:
            result = func(*args)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise InferenceError(f"Function failed: {e}") from e
```

### Logging Standards

```python
import logging

logger = logging.getLogger(__name__)

def train_model(config: MLTrainingConfig) -> None:
    """Train ML model with comprehensive logging."""
    logger.info(
        "Starting model training",
        extra={
            "model_type": config.model_type,
            "data_source": config.data_source,
            "hyperparameters": config.model_params,
        }
    )

    try:
        # Training logic
        logger.info("Model training completed successfully")
    except Exception as e:
        logger.error(
            "Model training failed",
            extra={"error": str(e), "config": config},
            exc_info=True
        )
        raise
```

---

## G. Documentation Standards

### Docstring Format

Use Google-style docstrings consistently:

```python
def compute_microstructure_features(
    quotes: pl.DataFrame,
    trades: pl.DataFrame,
    window_ms: int = 1000,
) -> dict[str, np.ndarray]:
    """
    Compute microstructure features from L2/L3 market data.

    This function calculates order book imbalance, price impact, and flow toxicity
    metrics using high-frequency quotes and trades data.

    Args:
        quotes: Level 2 quotes with columns [timestamp, bid, ask, bid_size, ask_size].
        trades: Trade ticks with columns [timestamp, price, size, aggressor_side].
        window_ms: Rolling window size in milliseconds for feature computation.

    Returns:
        Dictionary mapping feature names to computed values:
        - "order_flow_imbalance": Buy vs sell pressure ratio
        - "price_impact": Price change per unit volume
        - "bid_ask_spread": Relative spread values

    Raises:
        ValueError: If input data is empty or missing required columns.
        FeatureComputationError: If feature computation fails.

    Example:
        >>> quotes = pl.DataFrame({"timestamp": [1, 2], "bid": [1.0, 1.1], ...})
        >>> trades = pl.DataFrame({"timestamp": [1, 2], "price": [1.05, 1.15], ...})
        >>> features = compute_microstructure_features(quotes, trades)
        >>> assert "order_flow_imbalance" in features
    """
```

### Code Examples in Docstrings

Include practical examples:

```python
def register_model(
    self,
    model_id: str,
    model_path: Path,
    metadata: dict[str, Any],
) -> None:
    """
    Register a trained model in the registry.

    Example:
        >>> registry = ModelRegistry()
        >>> registry.register_model(
        ...     model_id="xgb_eurusd_v1",
        ...     model_path=Path("models/xgb_eurusd.onnx"),
        ...     metadata={
        ...         "training_data": "2023-01-01_to_2023-12-31",
        ...         "features": ["rsi", "macd", "volume_ratio"],
        ...         "target_metric": "sharpe_ratio",
        ...         "performance": {"accuracy": 0.67, "sharpe": 1.42}
        ...     }
        ... )
        >>> assert registry.get_model("xgb_eurusd_v1") is not None
    """
```

### Inline Comments

```python
def purged_cross_validation(
    data: pl.DataFrame,
    n_splits: int = 5,
    purge_pct: float = 0.01,
) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
    """Purged cross-validation for time series."""
    splits = []
    n_samples = len(data)

    # Calculate fold size accounting for purge buffer
    purge_samples = int(n_samples * purge_pct)
    fold_size = (n_samples - (n_splits - 1) * purge_samples) // n_splits

    for i in range(n_splits):
        # Training data: everything before test fold (minus purge)
        train_end = i * (fold_size + purge_samples)
        train_data = data[:train_end] if train_end > 0 else pl.DataFrame()

        # Test data: current fold
        test_start = train_end + purge_samples
        test_end = test_start + fold_size
        test_data = data[test_start:test_end]

        splits.append((train_data, test_data))

    return splits
```

---

## H. Quality Gates

### Pre-commit Requirements

**✅ MANDATORY** before any commit:

```bash
# Format code
make format

# Run linter
ruff check ml/

# Type checking
mypy ml/ --strict

# Run tests with coverage
pytest ml/ --cov=ml --cov-report=term-missing

# Verify coverage threshold
coverage report --fail-under=90  # For ML modules
```

### Sanity Sweep (Advisory)

- `make sanity` runs fast advisory checks:
  - ruff (S608/C901 on `ml/`)
  - mypy strict (`ml/`)
  - legacy schema refs (`ml/schema/*`)
  - SQL f‑strings and broad `except`
  - layering (stores importing actors)
- Included as a non‑blocking pre‑commit hook to surface issues early.

### Ruff Configuration

Key rules enforced:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "N",    # pep8-naming
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]  # Allow unused imports in __init__.py
"test_*.py" = ["S101"]    # Allow assert in tests
```

### MyPy Configuration

```toml
[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_any_generics = true
```

### Test Execution

```bash
# Unit tests only
pytest ml/tests/unit/ -v

# Property-based tests
pytest ml/tests/unit/ -k hypothesis --hypothesis-show-statistics

# Integration tests (slower)
pytest ml/tests/integration/ -v

# Performance benchmarks
pytest ml/tests/performance/ -v --benchmark-only
```

---

## I. Nautilus-Specific Requirements

### Timestamp Handling

**✅ MANDATORY**: All timestamps in nanoseconds since epoch:

```python
from nautilus_trader.core.datetime import nanos_to_secs

def process_bar_data(bar: Bar) -> dict[str, Any]:
    """Process bar with proper timestamp handling."""
    return {
        "instrument_id": str(bar.bar_type.instrument_id),
        "ts_event": bar.ts_event,  # Already in nanoseconds
        "ts_init": bar.ts_init,    # Already in nanoseconds
        "open": float(bar.open),
        "high": float(bar.high),
        "low": float(bar.low),
        "close": float(bar.close),
        "volume": float(bar.volume),
    }

# Database storage
def store_features(features: dict[str, float], bar: Bar) -> None:
    """Store features with Nautilus timestamp format."""
    query = """
    INSERT INTO features (instrument_id, ts_event, ts_init, features)
    VALUES (%(instrument_id)s, %(ts_event)s, %(ts_init)s, %(features)s)
    """

    params = {
        "instrument_id": str(bar.bar_type.instrument_id),
        "ts_event": bar.ts_event,  # nanoseconds
        "ts_init": bar.ts_init,    # nanoseconds
        "features": features,
    }
```

### Domain Types Usage

```python
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity

# ✅ Use domain types
def validate_trade(
    instrument_id: InstrumentId,  # Not str
    price: Price,                 # Not float
    quantity: Quantity,           # Not float
) -> bool:
    """Validate trade using Nautilus domain types."""
    return (
        price.precision == instrument_id.symbol.price_precision and
        quantity.precision == instrument_id.symbol.size_precision
    )

# ❌ Avoid raw primitives
def validate_trade_wrong(
    instrument: str,     # Raw string
    price: float,        # Raw float
    quantity: float,     # Raw float
) -> bool:
    """Wrong - uses primitives instead of domain types."""
```

### Event Patterns

```python
from nautilus_trader.core.data import Data
from nautilus_trader.model.data import DataType

class MLSignal(Data):
    """ML signal data type."""

    def __init__(
        self,
        instrument_id: InstrumentId,
        signal: float,
        confidence: float,
        ts_event: int,
        ts_init: int,
    ) -> None:
        """Initialize ML signal."""
        super().__init__(ts_event, ts_init)
        self.instrument_id = instrument_id
        self.signal = signal
        self.confidence = confidence

# Register data type
ML_SIGNAL = DataType(MLSignal, metadata={"type": "MLSignal"})

# Subscribe in actor
def on_start(self) -> None:
    """Subscribe to required data."""
    self.subscribe_data(ML_SIGNAL)
```

### Store Integration

**✅ MANDATORY**: Use the three required stores:

```python
from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategyStore

class BaseMLInferenceActor(Actor):
    """Base for all ML actors with mandatory stores."""

    def __init__(self, config: MLActorConfig) -> None:
        super().__init__()

        # Initialize mandatory stores
        self.feature_store = FeatureStore(config.db_connection)
        self.model_store = ModelStore(config.db_connection)
        self.strategy_store = StrategyStore(config.db_connection)

    async def store_prediction(
        self,
        prediction: float,
        confidence: float,
        features: dict[str, float],
    ) -> None:
        """Store prediction with all required metadata."""
        # Store features for training/inference parity
        await self.feature_store.store_features(
            instrument_id=self.instrument_id,
            ts_event=self._clock.timestamp_ns(),
            features=features,
        )

        # Store model predictions
        await self.model_store.store_prediction(
            model_id=self.model_id,
            prediction=prediction,
            confidence=confidence,
            ts_event=self._clock.timestamp_ns(),
        )
```

---

## J. ML-Specific Standards

### Feature Engineering Patterns

```python
from ml.features.pipeline import PipelineSpec, TransformSpec

# ✅ Use declarative pipeline specification
pipeline_spec = PipelineSpec(
    transforms=[
        TransformSpec(
            name="returns",
            function="compute_returns",
            params={"periods": [1, 5, 20]},
            data_requirements=["L1_ONLY"],
        ),
        TransformSpec(
            name="volatility",
            function="compute_realized_volatility",
            params={"window": 20},
            data_requirements=["L1_ONLY"],
        ),
        TransformSpec(
            name="order_flow",
            function="compute_order_flow_imbalance",
            params={"window_ms": 1000},
            data_requirements=["L2_REQUIRED"],
        ),
    ]
)

# Register with feature registry
feature_registry.register_pipeline(
    pipeline_id="microstructure_v1",
    spec=pipeline_spec,
)
```

### Model Registry Integration

```python
from ml.registry.model_registry import ModelRegistry

# ✅ Register models with semantic versioning
model_registry = ModelRegistry()

model_registry.register_model(
    model_id="xgb_eurusd",
    version="1.2.0",  # Semantic versioning
    model_path=Path("models/xgb_eurusd.onnx"),
    feature_manifest={
        "features": ["rsi_14", "macd_signal", "volume_ratio"],
        "schema_hash": compute_schema_hash(feature_spec),
    },
    metadata={
        "training_period": "2023-01-01_to_2023-12-31",
        "validation_metrics": {
            "accuracy": 0.67,
            "precision": 0.72,
            "recall": 0.63,
            "sharpe_ratio": 1.42,
        },
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
        },
    },
)
```

### Training/Inference Parity

**✅ MANDATORY**: Ensure identical feature computation:

```python
def test_training_inference_parity():
    """Test that training and inference produce identical features."""
    # Training-time feature computation (batch)
    training_features = feature_engineer.compute_batch(historical_data)

    # Inference-time feature computation (online)
    inference_features = []
    feature_engineer.reset()  # Reset internal state

    for bar in historical_data:
        features = feature_engineer.compute_online(bar)
        inference_features.append(features)

    # Verify exact parity
    np.testing.assert_allclose(
        training_features,
        np.array(inference_features),
        rtol=1e-10,
        err_msg="Training and inference features must be identical"
    )
```

### Monitoring Integration

```python
from ml.common.metrics_bootstrap import get_counter, get_histogram

# ✅ Acquire metrics via bootstrap (idempotent)
PREDICTIONS_TOTAL = get_counter(
    "ml_predictions_total",
    "Total ML predictions made",
    ["actor_id", "model_id", "instrument"],
)

INFERENCE_LATENCY = get_histogram(
    "ml_inference_latency_seconds",
    "ML inference latency",
    ["actor_id", "model_id"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1],  # Focus on low latency
)

class MLSignalActor(BaseMLInferenceActor):
    """ML signal actor with monitoring."""

    def _make_prediction(self, features: np.ndarray) -> float:
        """Make prediction with monitoring."""
        start_time = time.perf_counter()

        try:
            prediction = self._model.predict(features)[0]

            # Record successful prediction
            PREDICTIONS_TOTAL.labels(
                actor_id=self.id.value,
                model_id=self._model_id,
                instrument=str(self._instrument_id),
            ).inc()

            return prediction

        finally:
            # Always record latency
            latency = time.perf_counter() - start_time
            INFERENCE_LATENCY.labels(
                actor_id=self.id.value,
                model_id=self._model_id,
            ).observe(latency)
```

---

## Quality Checklist

Before submitting any pull request, verify:

- [ ] **Imports**: All ML dependencies imported via `ml._imports`
- [ ] **Types**: Complete type annotations, passes `mypy --strict`
- [ ] **Tests**: ≥90% coverage for ML modules, includes property tests
- [ ] **Performance**: Hot path functions validated <5ms
- [ ] **Config**: No hardcoded values, all constants in config classes
- [ ] **Metrics**: Use bootstrap (`ml.common.metrics_bootstrap`) for collectors
- [ ] **Events**: Use canonical event constants (`ml.config.events.Stage`); no raw literals

### CI/Validation Helpers

- Run `make validate-metrics` to ensure no direct prometheus instantiation in code.
- Run `make validate-events` to ensure event stage constants are used (Stage.*.value).
- [ ] **Errors**: Comprehensive validation and graceful error handling
- [ ] **Docs**: Google-style docstrings with examples
- [ ] **Standards**: Passes Ruff, Black formatting
- [ ] **Nautilus**: Uses domain types, nanosecond timestamps
- [ ] **Stores**: Integrates with FeatureStore, ModelStore, StrategyStore
- [ ] **Monitoring**: Exposes Prometheus metrics
- [ ] **Parity**: Training/inference feature parity validated

---

## Tools and Commands

```bash
# Code quality checks
make format                    # Format with Black + isort
ruff check ml/                # Lint with Ruff
mypy ml/ --strict             # Type check

# Testing
pytest ml/tests/unit/         # Unit tests
pytest ml/ --cov=ml --cov-report=term-missing  # Coverage
pytest ml/tests/performance/  # Performance tests

# Property testing
pytest ml/ -k hypothesis --hypothesis-show-statistics

# Pre-commit (run all checks)
make pre-commit
```

This document establishes the foundation for maintaining high-quality, performant, and maintainable ML code within the Nautilus Trader ecosystem. All contributors must adhere to these standards to ensure consistency and reliability across the ML codebase.
