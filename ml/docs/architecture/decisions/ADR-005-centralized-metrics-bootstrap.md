# ADR-005: Centralized Metrics Bootstrap Pattern

## Status
**ACCEPTED** - 2024-01-15

## Context

The Nautilus Trader ML system requires comprehensive metrics collection for monitoring, alerting, and performance analysis. However, directly using prometheus_client creates several problems:

**Registry Conflicts**:
- Multiple components importing prometheus_client create duplicate metric registrations
- Module reloading in development/testing causes registry conflicts
- Different parts of the system may use different metric registries
- Unit tests fail when same metrics are registered multiple times

**Inconsistent Naming**:
- No standard naming conventions across components
- Metrics with similar purposes use different names/labels
- Difficulty aggregating and correlating metrics across system

**Memory Leaks**:
- Prometheus metrics accumulate in global registry indefinitely
- High-cardinality metrics (e.g., per-trade-ID labels) cause memory growth
- No built-in cleanup mechanisms for dynamic metrics

**Testing Difficulties**:
- Tests pollute global metric registry
- No easy way to isolate metrics between test runs
- Mock metrics difficult to implement with direct prometheus_client usage

**Hot Path Performance**:
- Metric collection must not impact trading performance
- Need efficient metric updates in tight loops
- Thread-safety required for concurrent access

## Decision

**NEVER import prometheus_client directly. Use centralized `ml.common.metrics_bootstrap` for ALL metrics operations.**

### Core Principles
- **Single Source**: All metrics go through centralized bootstrap system
- **Registry Management**: Automatic handling of registry conflicts and cleanup
- **Consistent Naming**: Enforced naming conventions and validation
- **Performance Optimized**: Hot path optimizations and efficient collection
- **Test Friendly**: Easy mocking and isolation for tests

### Bootstrap System Features
- Singleton metric registry with conflict resolution
- Automatic naming convention enforcement
- Memory leak prevention with metric lifecycle management
- Hot path optimization with pre-allocated metrics
- Test isolation and cleanup utilities

## Consequences

### Positive
- **No Registry Conflicts**: Centralized management prevents duplicate registrations
- **Consistent Metrics**: Enforced naming and labeling conventions
- **Memory Safety**: Automatic cleanup prevents metric-related memory leaks
- **Testing Support**: Easy isolation and mocking for unit tests
- **Performance**: Hot path optimizations for trading operations
- **Maintainability**: Single place to manage all metrics configuration

### Negative
- **Indirection**: Additional layer between components and metrics
- **Learning Curve**: Developers must learn bootstrap API instead of direct prometheus_client
- **Dependency**: All components depend on bootstrap system
- **Flexibility**: Some advanced prometheus_client features may not be exposed

### Risks
- **Single Point of Failure**: If bootstrap system has bugs, affects all metrics
- **Performance Bottleneck**: Centralized system could become bottleneck under high load
- **Feature Limitations**: Bootstrap may not support all prometheus use cases

## Implementation Details

### Core Bootstrap API
```python
# ml/common/metrics_bootstrap.py
from typing import Dict, List, Optional, Any
from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge
from prometheus_client import multiprocess
import threading
import os
import re

class MetricsBootstrap:
    """Centralized metrics management system."""
    
    _instance: Optional['MetricsBootstrap'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'MetricsBootstrap':
        """Singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.registry = self._create_registry()
        self.metrics_cache: Dict[str, Any] = {}
        self.naming_validator = MetricsNamingValidator()
        self._initialized = True
    
    def _create_registry(self) -> CollectorRegistry:
        """Create appropriate registry based on environment."""
        if os.environ.get('PROMETHEUS_MULTIPROC_DIR'):
            # Multi-process environment (production)
            return CollectorRegistry()
        else:
            # Single process (development/testing)
            return CollectorRegistry()
    
    def get_counter(self, name: str, documentation: str, 
                   labels: List[str] = None) -> Counter:
        """Get or create counter metric."""
        labels = labels or []
        cache_key = f"counter:{name}:{':'.join(labels)}"
        
        if cache_key in self.metrics_cache:
            return self.metrics_cache[cache_key]
        
        # Validate naming convention
        self.naming_validator.validate_counter_name(name)
        self.naming_validator.validate_labels(labels)
        
        with self._lock:
            # Double-check after acquiring lock
            if cache_key in self.metrics_cache:
                return self.metrics_cache[cache_key]
            
            counter = Counter(
                name=name,
                documentation=documentation,
                labelnames=labels,
                registry=self.registry
            )
            
            self.metrics_cache[cache_key] = counter
            return counter
    
    def get_histogram(self, name: str, documentation: str,
                     buckets: List[float] = None, labels: List[str] = None) -> Histogram:
        """Get or create histogram metric."""
        labels = labels or []
        buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        
        cache_key = f"histogram:{name}:{':'.join(labels)}:{':'.join(map(str, buckets))}"
        
        if cache_key in self.metrics_cache:
            return self.metrics_cache[cache_key]
        
        # Validate naming convention  
        self.naming_validator.validate_histogram_name(name)
        self.naming_validator.validate_labels(labels)
        
        with self._lock:
            if cache_key in self.metrics_cache:
                return self.metrics_cache[cache_key]
            
            histogram = Histogram(
                name=name,
                documentation=documentation,
                labelnames=labels,
                buckets=buckets,
                registry=self.registry
            )
            
            self.metrics_cache[cache_key] = histogram
            return histogram
    
    def get_gauge(self, name: str, documentation: str,
                 labels: List[str] = None) -> Gauge:
        """Get or create gauge metric."""
        labels = labels or []
        cache_key = f"gauge:{name}:{':'.join(labels)}"
        
        if cache_key in self.metrics_cache:
            return self.metrics_cache[cache_key]
        
        # Validate naming convention
        self.naming_validator.validate_gauge_name(name)
        self.naming_validator.validate_labels(labels)
        
        with self._lock:
            if cache_key in self.metrics_cache:
                return self.metrics_cache[cache_key]
            
            gauge = Gauge(
                name=name,
                documentation=documentation,
                labelnames=labels,
                registry=self.registry
            )
            
            self.metrics_cache[cache_key] = gauge
            return gauge
    
    def clear_metrics(self) -> None:
        """Clear all metrics (for testing)."""
        with self._lock:
            self.registry = self._create_registry()
            self.metrics_cache.clear()

class MetricsNamingValidator:
    """Validates metric naming conventions."""
    
    COUNTER_SUFFIX = "_total"
    HISTOGRAM_SUFFIXES = ["_seconds", "_bytes", "_ratio"]
    GAUGE_SUFFIXES = ["_ratio", "_percent", "_bytes", "_seconds", "_celsius", "_count"]
    
    VALID_NAME_PATTERN = re.compile(r'^[a-zA-Z_:][a-zA-Z0-9_:]*$')
    VALID_LABEL_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
    
    def validate_counter_name(self, name: str) -> None:
        """Validate counter naming conventions."""
        if not self.VALID_NAME_PATTERN.match(name):
            raise ValueError(f"Invalid counter name: {name}")
        
        if not name.endswith(self.COUNTER_SUFFIX):
            raise ValueError(f"Counter names must end with '{self.COUNTER_SUFFIX}': {name}")
        
        if not name.startswith("ml_"):
            raise ValueError(f"ML metrics must start with 'ml_': {name}")
    
    def validate_histogram_name(self, name: str) -> None:
        """Validate histogram naming conventions."""
        if not self.VALID_NAME_PATTERN.match(name):
            raise ValueError(f"Invalid histogram name: {name}")
        
        if not any(name.endswith(suffix) for suffix in self.HISTOGRAM_SUFFIXES):
            raise ValueError(f"Histogram names must end with one of {self.HISTOGRAM_SUFFIXES}: {name}")
        
        if not name.startswith("ml_"):
            raise ValueError(f"ML metrics must start with 'ml_': {name}")
    
    def validate_gauge_name(self, name: str) -> None:
        """Validate gauge naming conventions."""
        if not self.VALID_NAME_PATTERN.match(name):
            raise ValueError(f"Invalid gauge name: {name}")
        
        if not name.startswith("ml_"):
            raise ValueError(f"ML metrics must start with 'ml_': {name}")
    
    def validate_labels(self, labels: List[str]) -> None:
        """Validate label naming conventions."""
        for label in labels:
            if not self.VALID_LABEL_PATTERN.match(label):
                raise ValueError(f"Invalid label name: {label}")
            
            if label.startswith("__"):
                raise ValueError(f"Labels cannot start with '__': {label}")

# Global singleton instance
_bootstrap = MetricsBootstrap()

# Public API functions
def get_counter(name: str, documentation: str, labels: List[str] = None) -> Counter:
    """Get or create counter metric."""
    return _bootstrap.get_counter(name, documentation, labels)

def get_histogram(name: str, documentation: str, 
                 buckets: List[float] = None, labels: List[str] = None) -> Histogram:
    """Get or create histogram metric."""
    return _bootstrap.get_histogram(name, documentation, buckets, labels)

def get_gauge(name: str, documentation: str, labels: List[str] = None) -> Gauge:
    """Get or create gauge metric."""
    return _bootstrap.get_gauge(name, documentation, labels)

def clear_metrics() -> None:
    """Clear all metrics (for testing)."""
    _bootstrap.clear_metrics()

def get_registry() -> CollectorRegistry:
    """Get the metrics registry."""
    return _bootstrap.registry
```

### Usage Patterns

#### Component Metrics
```python
# ✅ CORRECT: Use bootstrap system
from ml.common.metrics_bootstrap import get_counter, get_histogram

class MLSignalActor:
    def __init__(self, config):
        # Initialize metrics during component setup
        self.predictions_counter = get_counter(
            "ml_signal_predictions_total",
            "Total signal predictions made",
            labels=["instrument_id", "signal_type", "confidence_band"]
        )
        
        self.feature_latency = get_histogram(
            "ml_signal_feature_latency_seconds",
            "Feature computation latency",
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1],
            labels=["instrument_id"]
        )
    
    def on_bar(self, bar):
        # Hot path metric updates
        with self.feature_latency.time(labels={"instrument_id": bar.instrument_id.value}):
            features = self.compute_features(bar)
        
        prediction = self.model.predict(features)
        
        self.predictions_counter.inc(labels={
            "instrument_id": bar.instrument_id.value,
            "signal_type": "BUY" if prediction > 0.5 else "SELL",
            "confidence_band": self._get_confidence_band(prediction)
        })

# ❌ INCORRECT: Direct prometheus_client import
# from prometheus_client import Counter, Histogram  # NEVER DO THIS
```

#### Custom Metrics Collections
```python
from ml.common.metrics_bootstrap import get_counter, get_histogram, get_gauge

class MLPipelineMetrics:
    """Centralized metrics collection for ML pipeline."""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        
        # Pipeline performance metrics
        self.operations_total = get_counter(
            "ml_pipeline_operations_total",
            "Total ML pipeline operations",
            labels=["component", "operation", "status"]
        )
        
        self.operation_duration = get_histogram(
            "ml_pipeline_operation_duration_seconds",
            "ML pipeline operation duration",
            buckets=[0.001, 0.01, 0.1, 1.0, 10.0],
            labels=["component", "operation"]
        )
        
        self.queue_size = get_gauge(
            "ml_pipeline_queue_size_count",
            "Current queue size",
            labels=["component", "queue_type"]
        )
        
        self.memory_usage = get_gauge(
            "ml_pipeline_memory_usage_bytes",
            "Memory usage in bytes", 
            labels=["component"]
        )
    
    def record_operation(self, operation: str, duration: float, status: str) -> None:
        """Record pipeline operation metrics."""
        self.operations_total.inc(labels={
            "component": self.component_name,
            "operation": operation,
            "status": status
        })
        
        self.operation_duration.observe(
            duration,
            labels={"component": self.component_name, "operation": operation}
        )
    
    def update_queue_size(self, queue_type: str, size: int) -> None:
        """Update queue size metric."""
        self.queue_size.set(
            size,
            labels={"component": self.component_name, "queue_type": queue_type}
        )
```

#### Hot Path Optimizations
```python
class HotPathMetricsOptimizer:
    """Optimizations for hot path metric collection."""
    
    def __init__(self):
        # Pre-allocate metrics for hot path
        self.inference_latency = get_histogram(
            "ml_inference_latency_seconds",
            "Model inference latency",
            buckets=[0.0005, 0.001, 0.002, 0.005, 0.01],  # Sub-millisecond buckets
            labels=["model_id"]
        )
        
        # Pre-allocate label dictionaries to avoid dict creation in hot path
        self.model_labels = {}  # Cache for label dictionaries
    
    def get_model_labels(self, model_id: str) -> dict:
        """Get cached label dictionary for model."""
        if model_id not in self.model_labels:
            self.model_labels[model_id] = {"model_id": model_id}
        return self.model_labels[model_id]
    
    def record_inference(self, model_id: str, latency: float) -> None:
        """Record inference with optimized label lookup."""
        # Use cached labels to avoid dictionary creation
        labels = self.get_model_labels(model_id)
        self.inference_latency.observe(latency, labels=labels)
```

### Testing Integration

#### Test Utilities
```python
# ml/testing/metrics.py
import pytest
from ml.common.metrics_bootstrap import clear_metrics, get_registry

class MetricsTestCase:
    """Base test case for metrics testing."""
    
    def setup_method(self):
        """Clear metrics before each test."""
        clear_metrics()
    
    def teardown_method(self):
        """Clean up after each test."""
        clear_metrics()
    
    def assert_metric_exists(self, metric_name: str) -> None:
        """Assert that metric exists in registry."""
        registry = get_registry()
        metric_names = [metric.describe()[0].name for metric in registry.collect()]
        assert metric_name in metric_names, f"Metric {metric_name} not found"
    
    def get_metric_value(self, metric_name: str, labels: dict = None) -> float:
        """Get current metric value."""
        registry = get_registry()
        for metric in registry.collect():
            for sample in metric.samples:
                if sample.name == metric_name:
                    if labels is None or self._labels_match(sample.labels, labels):
                        return sample.value
        
        raise ValueError(f"Metric {metric_name} with labels {labels} not found")
    
    def _labels_match(self, sample_labels: dict, expected_labels: dict) -> bool:
        """Check if sample labels match expected labels."""
        for key, value in expected_labels.items():
            if sample_labels.get(key) != value:
                return False
        return True

# Test usage
class TestMLActorMetrics(MetricsTestCase):
    def test_predictions_counter_increments(self):
        """Test that predictions counter increments correctly."""
        actor = MLSignalActor(test_config)
        
        # Execute prediction
        actor.on_bar(test_bar)
        
        # Verify metric was incremented
        self.assert_metric_exists("ml_signal_predictions_total")
        
        counter_value = self.get_metric_value(
            "ml_signal_predictions_total",
            labels={"instrument_id": "EUR/USD", "signal_type": "BUY"}
        )
        assert counter_value == 1.0
```

#### Mock Metrics for Testing
```python
from unittest.mock import Mock

class MockMetricsBootstrap:
    """Mock bootstrap for unit testing."""
    
    def __init__(self):
        self.counters = {}
        self.histograms = {}
        self.gauges = {}
    
    def get_counter(self, name: str, documentation: str, labels: list = None):
        if name not in self.counters:
            mock_counter = Mock()
            mock_counter.inc = Mock()
            self.counters[name] = mock_counter
        return self.counters[name]
    
    def get_histogram(self, name: str, documentation: str, buckets: list = None, labels: list = None):
        if name not in self.histograms:
            mock_histogram = Mock()
            mock_histogram.observe = Mock()
            mock_histogram.time = Mock(return_value=Mock(__enter__=Mock(), __exit__=Mock()))
            self.histograms[name] = mock_histogram
        return self.histograms[name]
    
    def get_gauge(self, name: str, documentation: str, labels: list = None):
        if name not in self.gauges:
            mock_gauge = Mock()
            mock_gauge.set = Mock()
            self.gauges[name] = mock_gauge
        return self.gauges[name]

# Use in tests
@pytest.fixture
def mock_metrics(monkeypatch):
    """Mock metrics bootstrap for testing."""
    mock_bootstrap = MockMetricsBootstrap()
    monkeypatch.setattr("ml.common.metrics_bootstrap._bootstrap", mock_bootstrap)
    return mock_bootstrap
```

## Performance Optimization

### Metric Collection Efficiency
```python
class PerformantMetricsCollector:
    """High-performance metrics collection for trading systems."""
    
    def __init__(self):
        # Pre-allocate all metrics and label combinations for hot path
        self.trade_metrics = {}
        self._initialize_trade_metrics()
    
    def _initialize_trade_metrics(self):
        """Pre-allocate metrics for known instruments."""
        instruments = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]  # Common pairs
        
        for instrument in instruments:
            self.trade_metrics[instrument] = {
                'latency': get_histogram(
                    "ml_trade_execution_latency_seconds",
                    "Trade execution latency",
                    buckets=[0.001, 0.005, 0.01, 0.05, 0.1],
                    labels=["instrument_id"]
                ),
                'labels': {"instrument_id": instrument}  # Pre-allocated labels
            }
    
    def record_trade_latency(self, instrument_id: str, latency: float):
        """Record trade latency with minimal overhead."""
        if instrument_id in self.trade_metrics:
            # Use pre-allocated metric and labels
            metric_info = self.trade_metrics[instrument_id]
            metric_info['latency'].observe(latency, labels=metric_info['labels'])
        else:
            # Fallback for unknown instruments (slower path)
            latency_metric = get_histogram(
                "ml_trade_execution_latency_seconds",
                "Trade execution latency",
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1],
                labels=["instrument_id"]
            )
            latency_metric.observe(latency, labels={"instrument_id": instrument_id})
```

### Memory Management
```python
class MemoryEfficientMetrics:
    """Memory-efficient metrics with cleanup."""
    
    def __init__(self, max_cardinality: int = 10000):
        self.max_cardinality = max_cardinality
        self.label_combinations = {}
        self.cleanup_threshold = 0.8
    
    def get_safe_labels(self, base_labels: dict, dynamic_key: str, dynamic_value: str) -> dict:
        """Get labels with cardinality protection."""
        # Limit cardinality to prevent memory explosion
        if len(self.label_combinations) > self.max_cardinality * self.cleanup_threshold:
            self._cleanup_old_labels()
        
        # Use hash of labels as key to limit memory
        labels = {**base_labels, dynamic_key: dynamic_value}
        labels_key = hash(frozenset(labels.items()))
        
        if labels_key not in self.label_combinations:
            if len(self.label_combinations) < self.max_cardinality:
                self.label_combinations[labels_key] = labels
            else:
                # Use generic label to prevent unbounded growth
                labels = {**base_labels, dynamic_key: "other"}
        else:
            labels = self.label_combinations[labels_key]
        
        return labels
    
    def _cleanup_old_labels(self):
        """Clean up old label combinations to prevent memory leak."""
        # Remove oldest 20% of label combinations
        items_to_remove = len(self.label_combinations) // 5
        keys_to_remove = list(self.label_combinations.keys())[:items_to_remove]
        
        for key in keys_to_remove:
            del self.label_combinations[key]
```

## Migration Strategy

### Phase 1: Bootstrap Infrastructure
- Implement core `MetricsBootstrap` class with singleton pattern
- Add naming validation and registry management
- Create test utilities and mock implementations

### Phase 2: Component Migration
- Replace direct prometheus_client imports with bootstrap calls
- Update existing metrics to follow naming conventions
- Add validation rules to prevent direct imports

### Phase 3: Advanced Features
- Implement hot path optimizations and memory management
- Add cardinality protection and cleanup mechanisms
- Create metrics collections for common use cases

### Phase 4: Enforcement and Testing
- Add linting rules to prevent direct prometheus_client usage
- Comprehensive testing of bootstrap system under load
- Performance benchmarks and optimization

### Phase 5: Monitoring and Operations
- Deploy metrics collection infrastructure
- Create dashboards and alerting based on standardized metrics
- Documentation and training for development teams

## Compliance Validation

### Static Analysis Rules
```python
# .pylintrc addition
[MESSAGES CONTROL]
disable = ...

[DESIGN]
# Prevent direct prometheus import
forbidden-import-names = prometheus_client

# Custom checker for metrics bootstrap compliance
load-plugins = ml.tools.metrics_compliance_checker
```

### Automated Testing
```python
import ast
import os
from pathlib import Path

class MetricsComplianceChecker:
    """Check code compliance with metrics bootstrap requirements."""
    
    def check_directory(self, directory: Path) -> list[str]:
        """Check all Python files in directory for compliance."""
        violations = []
        
        for py_file in directory.rglob("*.py"):
            file_violations = self.check_file(py_file)
            violations.extend(file_violations)
        
        return violations
    
    def check_file(self, file_path: Path) -> list[str]:
        """Check single file for metrics compliance."""
        violations = []
        
        try:
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module == "prometheus_client":
                        violations.append(f"{file_path}:{node.lineno}: Direct prometheus_client import")
                
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "prometheus_client":
                            violations.append(f"{file_path}:{node.lineno}: Direct prometheus_client import")
        
        except Exception as e:
            violations.append(f"{file_path}: Parse error: {e}")
        
        return violations

# CI integration
def test_metrics_compliance():
    """Ensure all ML code uses metrics bootstrap."""
    checker = MetricsComplianceChecker()
    violations = checker.check_directory(Path("ml/"))
    
    assert len(violations) == 0, f"Metrics compliance violations:\n" + "\n".join(violations)
```

## Related ADRs
- ADR-001: 4-Store + 4-Registry Mandatory Pattern
- ADR-002: Protocol-First Interface Design
- ADR-003: Hot/Cold Path Separation Strategy
- ADR-004: Progressive Fallback Implementation

## References
- [Prometheus Client Library](https://github.com/prometheus/client_python)
- [Metrics Bootstrap Implementation](../../common/metrics_bootstrap.py)
- [Performance Testing Documentation](../integration_testing_strategy.md#performance-integration-tests)