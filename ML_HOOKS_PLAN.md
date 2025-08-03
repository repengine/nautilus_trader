# ML Pre-commit Hooks Plan

## Current Hooks Summary

### General Hooks

- **check_tests_pass.py**: Runs tests for changed files, ALL ML tests if ML files changed
- **check_test_coverage_general.py**: Ensures >80% coverage for all Python
- **check_tests_clean.py**: Ensures 0 failures/errors/warnings
- **check_mypy_clean.py**: Ensures 0 mypy errors

### ML-Specific Hooks

- **check_test_coverage.py**: Ensures ≥90% coverage for ML modules
- **check_ml_performance.py**: Ensures no regression >20% tolerance
- **check_feature_parity.py**: Validates training/inference compute identical features (1e-10 tolerance)

## Additional Hooks Needed

### 1. Docstring Validation Hook
**File**: `.pre-commit-hooks/check_docstrings.py`
**Purpose**: Ensure all public symbols have NumPy-style docstrings
**Checks**:

- All public classes, functions, methods have docstrings
- Docstrings follow NumPy format
- Include Parameters, Returns, Examples sections where appropriate
- Imperative mood for Python docstrings

### 2. Prometheus Monitoring Hook
**File**: `.pre-commit-hooks/check_prometheus_metrics.py`
**Purpose**: Ensure ML components have proper monitoring
**Checks**:

- ML actors expose required metrics (latency, throughput, errors)
- Metric names follow prometheus conventions
- Grafana dashboards updated for new metrics

### 3. Nautilus Pattern Validation Hook
**File**: `.pre-commit-hooks/check_nautilus_patterns.py`
**Purpose**: Ensure ML code follows Nautilus patterns
**Checks**:

- Actors inherit from correct base classes
- Strategies use proper initialization patterns (no clock/logger in __init__)
- Configuration classes use frozen=True
- Proper use of Nautilus types (Price, Quantity, etc.)
- Hot/cold path separation

## Pattern Detection Strategy

### Identifiable Patterns from CLAUDE.md and ML docs

1. **Import Patterns**
   - Check imports match standard patterns
   - Verify no pandas in hot path
   - Ensure Polars used for cold path

2. **Actor Patterns**
   - Must inherit from Actor base class
   - on_start() for subscriptions
   - Proper message bus usage
   - Pre-allocated numpy arrays

3. **Strategy Patterns**
   - Must inherit from Strategy
   - Configuration with frozen=True
   - No clock/logger access in __init__
   - Use on_start() for initialization

4. **Data Type Patterns**
   - Custom Data types for ML signals
   - Proper ts_event/ts_init handling
   - msgspec for configuration

5. **Performance Patterns**
   - No blocking operations in event handlers
   - Model loaded once at startup
   - Incremental feature updates
   - Bounded collections

## Implementation Priority

1. **Phase 1 (Immediate)**: Docstring validation - critical for code quality
2. **Phase 2 (Week 1)**: Nautilus pattern validation - ensures correct integration
3. **Phase 3 (Week 2)**: Prometheus monitoring - needed for production

## Hook Configuration

Add to `.pre-commit-config.yaml`:

```yaml
- id: check-docstrings
  name: check docstrings
  description: Ensures all public symbols have proper docstrings
  entry: .pre-commit-hooks/check_docstrings.py
  language: script
  types: [python]
  files: 'ml/.*\.py$'
  pass_filenames: true
  stages: [pre-commit]

- id: check-nautilus-patterns
  name: check Nautilus patterns
  description: Validates ML code follows Nautilus architectural patterns
  entry: .pre-commit-hooks/check_nautilus_patterns.py
  language: script
  types: [python]
  files: 'ml/.*\.py$'
  pass_filenames: true
  stages: [pre-commit]

- id: check-prometheus-metrics
  name: check Prometheus metrics
  description: Ensures ML components have proper monitoring
  entry: .pre-commit-hooks/check_prometheus_metrics.py
  language: script
  types: [python]
  files: 'ml/(actors|strategies)/.*\.py$'
  pass_filenames: true
  stages: [pre-commit]
```

## Testing Strategy

Each hook should be tested with:

1. Valid code that should pass
2. Invalid code that should fail
3. Edge cases (empty files, test files, etc.)

## Integration with CI/CD

These hooks will run:

1. Locally on pre-commit
2. In CI pipeline
3. As part of PR validation

This ensures consistent code quality and pattern adherence throughout the ML codebase.
