# Context: Root Module

## Overview

The ml/ root module provides the foundational infrastructure for the Nautilus ML package, managing package initialization, optional dependency handling, type definitions, and testing configuration. This module establishes the core patterns that enable the ML package to integrate seamlessly with Nautilus Trader's high-performance trading infrastructure while maintaining strict separation between optional ML dependencies and core functionality.

**Key Features:**
- Centralized dependency management with graceful fallbacks
- Type-only imports for static analysis without runtime overhead
- Hot/cold path performance architecture documentation
- Comprehensive pytest configuration for ML-specific testing needs
- Package version and metadata management

## Architecture

### Module Structure

```
ml/
├── __init__.py          # Package documentation and metadata
├── _imports.py          # Centralized optional dependency management
├── typing.py            # Type aliases for optional dependencies
└── conftest.py          # Pytest configuration and collection control
```

### Design Principles

1. **Optional Dependency Isolation**: All ML dependencies are optional with graceful fallbacks
2. **Type Safety**: Complete type annotations available regardless of runtime dependencies
3. **Performance First**: Clear documentation of hot/cold path separation requirements
4. **Testing Infrastructure**: Robust test configuration preventing common pitfalls
5. **Security**: Explicit deprecation of unsafe formats (pickle) with clear error messages

## Key Components

### Package Initialization (`__init__.py`)

**Purpose**: Provides package-level documentation and versioning information.

```python
__version__ = "0.1.0"
```

**Key Documentation Sections:**
- **Cold Path (Training)**: Polars-based data loading, XGBoost/LightGBM training, Optuna optimization, MLflow registry
- **Hot Path (Inference)**: Real-time numpy computation, <5ms latency requirements, actor-based signals, message bus integration

**Performance Requirements Documented:**
- Hot path: <5ms end-to-end latency requirement
- Real-time feature computation with numpy
- Actor-based signal generation with message bus integration

### Dependency Management (`_imports.py`)

**Purpose**: Centralized management of all optional ML dependencies with proper error handling and availability flags.

**Core Pattern:**
```python
try:
    import optional_package as pkg
    HAS_PACKAGE = True
    PACKAGE_IMPORT_ERROR = None
except ImportError as e:
    HAS_PACKAGE = False
    PACKAGE_IMPORT_ERROR = e
    pkg = None
```

**Supported Dependencies:**
- **Core ML**: `onnxruntime`, `onnx`, `polars`, `xgboost`, `lightgbm`, `sklearn`
- **Optimization**: `optuna`, `torch`
- **Data Sources**: `pandas`, `fredapi`, `databento`, `pandas_market_calendars`
- **Monitoring**: `prometheus_client` with dummy implementations
- **Export Tools**: `onnxmltools`, `skl2onnx`

**Security Features:**
- **MLflow Deprecation**: Intentionally disabled to prevent telemetry activation
- **Databento Conditional**: Only imported when API key present or explicitly enabled
- **Pickle Security**: Not managed here but mentioned in architecture guidance

**Dummy Implementations:**
- Complete Prometheus client fallbacks (`Counter`, `Gauge`, `Histogram`)
- Registry and metrics generation with no-op implementations
- Method chaining support for transparent operation

**Public API:**
```python
def check_ml_dependencies(required: list[str]) -> None:
    """Validate required dependencies with helpful error messages."""

def generate_latest(registry: object | None = None) -> bytes:
    """Unified Prometheus metrics export."""

REGISTRY: object  # Prometheus registry (real or dummy)
```

### Type Definitions (`typing.py`)

**Purpose**: Provides type aliases for optional dependencies without runtime overhead.

**Type-Only Imports Pattern:**
```python
if TYPE_CHECKING:
    import pandas as _pd
    import polars as _pl
    from sklearn.preprocessing import StandardScaler as _StandardScaler
else:
    # Runtime stubs with no functionality
    class _pd:
        class DataFrame: pass
        class Series: pass
```

**Public Type Aliases:**
```python
PandasDF = _pd.DataFrame
PandasSeries = _pd.Series
PolarsDF = _pl.DataFrame
PolarsSeries = _pl.Series

# Union types for flexibility
DataFrameLike = PandasDF | PolarsDF
SeriesLike = PandasSeries | PolarsSeries

# ML component types
StandardScaler = _StandardScaler
```

**Benefits:**
- Zero runtime cost when dependencies absent
- Full static type checking with mypy/ruff
- Clean public API without underscore prefixes
- Union types enable flexible function signatures

### Test Configuration (`conftest.py`)

**Purpose**: Configures pytest to handle ML-specific testing requirements and prevent common conflicts.

**Primary Function:**
```python
def pytest_ignore_collect(collection_path: Any, config: Any) -> bool:
    """Ignore training modules during test collection to avoid naming conflicts."""
```

**Ignored Patterns:**
- `ml/training/non_distilled/lightgbm.py`
- `ml/training/non_distilled/xgboost.py`
- `ml/training/student/lightgbm.py`
- `ml/training/student/lightgbm_student.py`
- Compatibility shims: `ml/training/lightgbm.py`, `ml/training/xgboost.py`

**Purpose of Exclusions:**
Training modules have same names as installed packages (lightgbm, xgboost), causing import conflicts when pytest tries to collect them as test modules.

## Dependencies

### Internal Dependencies
- **Nautilus Core**: Integrates with core Nautilus infrastructure (not imported directly)
- **Type System**: Uses modern Python typing features (`typing`, `TYPE_CHECKING`)

### External Dependencies (All Optional)
- **ML Frameworks**: `xgboost`, `lightgbm`, `sklearn`, `torch`
- **Data Processing**: `polars`, `pandas`
- **Model Runtime**: `onnxruntime`, `onnx`
- **Data Sources**: `fredapi`, `databento`, `pandas_market_calendars`
- **Monitoring**: `prometheus_client`
- **Optimization**: `optuna`
- **Export**: `onnxmltools`, `skl2onnx`

### Environment Variables
- `ML_ENABLE_DATABENTO`: Enable Databento import without API key
- `DATABENTO_API_KEY`: Triggers automatic Databento import
- `ML_STRICT_PROTOCOL_VALIDATION`: Enables strict protocol validation (referenced in architecture)

## Usage Patterns

### Dependency Checking Pattern
```python
from ml._imports import HAS_XGBOOST, xgb, check_ml_dependencies

if not HAS_XGBOOST:
    check_ml_dependencies(["xgboost"])

# Safe to use xgb here
model = xgb.XGBClassifier()
```

### Type Annotation Pattern
```python
from ml.typing import DataFrameLike, StandardScaler

def process_data(df: DataFrameLike) -> DataFrameLike:
    """Function works with both Pandas and Polars DataFrames."""
    # Implementation handles both types
    pass

def create_scaler() -> StandardScaler:
    """Type-safe scaler creation."""
    from ml._imports import sklearn
    return sklearn.preprocessing.StandardScaler()
```

### Prometheus Metrics Pattern
```python
from ml._imports import Counter, Gauge, generate_latest, REGISTRY

# Works regardless of prometheus_client availability
counter = Counter("ml_operations_total", "Total ML operations")
counter.inc()

# Export metrics (returns empty bytes if prometheus unavailable)
metrics_data = generate_latest(REGISTRY)
```

### Testing with Dummy Implementations
```python
# Tests work even without prometheus_client installed
from ml._imports import Counter

def test_metrics():
    counter = Counter("test_counter", "Test counter")
    counter.inc()  # No-op if prometheus unavailable
    assert True  # Test passes regardless
```

## Integration Points

### Nautilus Trader Integration
- **Performance Requirements**: Documents <5ms hot path latency requirement
- **Message Bus**: References actor-based signal generation integration
- **Data Model**: Emphasizes nanosecond timestamp requirements (ts_event, ts_init)

### ML Package Integration
- **Store Protocols**: Type definitions support structural typing in stores
- **Feature Engineering**: DataFrameLike types enable flexible data processing
- **Model Registry**: Optional dependency management supports model deployment
- **Training Pipeline**: Dependency flags control training module availability

### Testing Integration
- **Pytest Configuration**: Prevents collection conflicts with training modules
- **Mock Support**: Dummy implementations enable testing without dependencies
- **CI/CD**: Environment variable controls enable stable test environments

## Implementation Notes

### Hot/Cold Path Separation
The package documentation establishes critical performance boundaries:
- **Hot Path**: Real-time inference with <5ms requirement, numpy-based
- **Cold Path**: Training operations, heavy I/O, Polars-based data processing

### Security Considerations
- **MLflow Deprecation**: Explicitly disabled to prevent unwanted telemetry
- **Pickle Avoidance**: Architecture guidance discourages pickle usage
- **Conditional Imports**: Network-dependent packages only imported when safe

### Error Handling Philosophy
- **Descriptive Errors**: `check_ml_dependencies()` provides installation hints
- **Graceful Degradation**: Dummy implementations allow operation without dependencies
- **Early Validation**: Import-time checks prevent runtime surprises

### Type Safety Strategy
- **TYPE_CHECKING Guards**: Zero runtime cost for type-only imports
- **Union Types**: Modern Python typing for flexible APIs
- **Structural Typing**: Enables protocol-based interfaces in stores

### Testing Philosophy
- **Name Conflict Prevention**: Explicit pytest exclusions for problematic modules
- **Environment Isolation**: Conditional imports prevent network dependencies in tests
- **Mock Support**: Dummy implementations ensure tests pass in any environment

### Performance Considerations
- **Lazy Imports**: All ML dependencies imported on-demand
- **Memory Efficiency**: Type stubs have minimal memory footprint
- **Import Time**: Fast package initialization with deferred heavy imports

## Critical Implementation Details

### Import Error Preservation
All import errors are captured and stored for diagnostic purposes:
```python
except ImportError as e:
    HAS_PACKAGE = False
    PACKAGE_IMPORT_ERROR = e
```

This enables `check_ml_dependencies()` to provide detailed error information including original ImportError details.

### Prometheus Registry Handling
The module provides both real and dummy Prometheus registries with identical interfaces:
```python
# Real implementation
from prometheus_client import REGISTRY as _REAL_REGISTRY

# Dummy implementation
class _DummyRegistry:
    def __init__(self) -> None:
        self._names_to_collectors: dict[str, Any] = {}
```

### Type Stub Completeness
Runtime type stubs provide just enough structure to prevent AttributeError during imports while maintaining zero functionality.

### Configuration Compatibility
Environment variable handling supports various deployment scenarios:
- Development: Manual enablement via `ML_ENABLE_DATABENTO`
- Production: Automatic detection via `DATABENTO_API_KEY`
- Testing: Default disabled state for stable CI/CD

This root module architecture provides the essential foundation for the entire ML package, ensuring reliable operation across diverse deployment environments while maintaining strict performance requirements and comprehensive type safety.