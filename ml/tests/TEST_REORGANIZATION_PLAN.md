# ML Test Reorganization Plan

## Executive Summary

This document provides a comprehensive plan to reorganize the ML test suite for improved clarity, maintainability, and adherence to CLAUDE.md testing standards. The reorganization separates tests by type (unit, integration, e2e), mirrors source code structure, and establishes clear naming conventions and guidelines.

## Current State Analysis

### Issues Identified

1. **Mixed Test Types**: Unit, integration, and e2e tests are scattered across directories
2. **Inconsistent Naming**: Some tests at root level should be categorized (e.g., `test_feature_parity.py`)
3. **Redundant Coverage**: Multiple test files testing similar functionality
4. **Unclear Organization**: Tests don't consistently mirror source structure
5. **Missing Categories**: No dedicated e2e or system test directories
6. **Fixture Sprawl**: Test fixtures and data scattered across multiple locations

### Strengths to Preserve

1. **Contract Testing**: Well-established contract test pattern in `contracts/`
2. **Performance Testing**: Dedicated performance benchmarking in `performance/`
3. **Property Testing**: Good use of Hypothesis in property tests
4. **Test Protocol**: Comprehensive TESTING_PROTOCOL.md documentation

## Proposed Directory Structure

```
ml/tests/
├── README.md                      # Test suite overview and quick start guide
├── TESTING_PROTOCOL.md           # Existing comprehensive testing guidelines
├── conftest.py                   # Root-level pytest configuration
│
├── unit/                         # Fast, isolated component tests
│   ├── __init__.py
│   ├── actors/
│   │   ├── __init__.py
│   │   ├── test_base_actor.py
│   │   ├── test_signal_actor.py
│   │   └── test_signal_actor_features.py
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── test_coverage_cli.py
│   │   └── test_feature_cli.py
│   ├── common/
│   │   ├── __init__.py
│   │   └── test_metrics.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── test_actors_config.py
│   │   └── test_base_config.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── test_cache.py
│   │   └── test_integration_core.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── test_catalog_utils.py
│   │   ├── test_collector.py
│   │   ├── test_scheduler.py
│   │   ├── test_tft_dataset_builder.py
│   │   ├── loaders/
│   │   │   ├── __init__.py
│   │   │   └── test_fred_loader.py
│   │   ├── providers/
│   │   │   ├── __init__.py
│   │   │   ├── test_base_provider.py
│   │   │   ├── test_calendar_provider.py
│   │   │   ├── test_events_provider.py
│   │   │   ├── test_factory.py
│   │   │   ├── test_metadata_provider.py
│   │   │   └── test_utils.py
│   │   └── sources/
│   │       ├── __init__.py
│   │       ├── test_calendar_source.py
│   │       ├── test_events_source.py
│   │       └── test_metadata_source.py
│   ├── deployment/
│   │   ├── __init__.py
│   │   └── test_health_check.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── test_engineering.py
│   │   ├── test_feature_export.py
│   │   ├── test_materialize.py
│   │   ├── test_microstructure.py
│   │   ├── test_pipeline.py
│   │   └── test_validation.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── test_collectors.py
│   │   ├── test_dashboard_factory.py
│   │   ├── test_grafana_client.py
│   │   └── test_server.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── test_joins.py
│   │   └── test_stationarity.py
│   ├── registry/
│   │   ├── __init__.py
│   │   ├── test_base_registry.py
│   │   ├── test_data_registry.py
│   │   ├── test_feature_registry.py
│   │   ├── test_model_registry.py
│   │   ├── test_persistence.py
│   │   ├── test_statistics.py
│   │   ├── test_strategy_registry.py
│   │   └── test_utils.py
│   ├── stores/
│   │   ├── __init__.py
│   │   ├── test_base_store.py
│   │   ├── test_data_processor.py
│   │   ├── test_data_store.py
│   │   ├── test_feature_store.py
│   │   ├── test_model_store.py
│   │   ├── test_partition_manager.py
│   │   └── test_strategy_store.py
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── test_base_strategy.py
│   │   └── test_ml_strategy.py
│   └── training/
│       ├── __init__.py
│       ├── test_base_trainer.py
│       ├── test_export.py
│       ├── test_optuna_optimizer.py
│       ├── distillation/
│       │   ├── __init__.py
│       │   └── test_distillation_cli.py
│       ├── non_distilled/
│       │   ├── __init__.py
│       │   ├── test_lightgbm.py
│       │   └── test_xgboost.py
│       ├── student/
│       │   ├── __init__.py
│       │   └── test_lightgbm_student.py
│       └── teacher/
│           ├── __init__.py
│           ├── test_tft_teacher.py
│           └── test_torchscript.py
│
├── integration/                  # Component interaction tests
│   ├── __init__.py
│   ├── conftest.py              # Integration-specific fixtures
│   ├── actors/
│   │   ├── __init__.py
│   │   └── test_actor_pipeline.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── test_data_pipeline.py
│   │   ├── test_databento_integration.py
│   │   └── test_scheduler_integration.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── test_feature_parity.py
│   │   └── test_feature_pipeline.py
│   ├── registry/
│   │   ├── __init__.py
│   │   ├── test_registry_gates.py
│   │   └── test_registry_rollout.py
│   ├── stores/
│   │   ├── __init__.py
│   │   ├── test_store_interactions.py
│   │   └── test_store_persistence.py
│   └── strategies/
│       ├── __init__.py
│       └── test_strategy_pipeline.py
│
├── e2e/                          # End-to-end system tests
│   ├── __init__.py
│   ├── conftest.py              # E2E-specific fixtures
│   ├── test_data_registry_flow.py
│   ├── test_ml_pipeline_complete.py
│   ├── test_signal_generation_flow.py
│   ├── test_strategy_execution_flow.py
│   └── test_training_deployment_flow.py
│
├── system/                       # System-level tests
│   ├── __init__.py
│   ├── test_dry_run_system.py
│   ├── test_infrastructure.py
│   └── test_ml_backtest_system.py
│
├── contracts/                    # Behavioral contract tests (preserved)
│   ├── README.md
│   ├── __init__.py
│   ├── test_actor_contracts.py
│   ├── test_registry_behavioral.py
│   └── test_strategy_contracts.py
│
├── performance/                  # Performance benchmarks (preserved)
│   ├── README.md
│   ├── __init__.py
│   ├── benchmark_hot_path.py
│   ├── test_hot_path_latency.py
│   └── test_zero_allocation.py
│
├── property/                     # Property-based tests
│   ├── __init__.py
│   ├── test_feature_properties.py
│   ├── test_numerical_stability.py
│   └── test_temporal_consistency.py
│
├── fixtures/                     # Shared test utilities
│   ├── __init__.py
│   ├── model_factory.py         # Centralized model creation
│   ├── data_generators.py       # Test data generation
│   ├── mock_actors.py           # Reusable mock components
│   └── test_configs.py          # Standard test configurations
│
├── data/                         # Test data files (preserved)
│   ├── README.md
│   ├── __init__.py
│   ├── model_registry/          # Test model registry data
│   └── model_registry_rollout/  # Rollout test data
│
└── benchmarks/                   # Benchmark results storage
    ├── __init__.py
    └── README.md
```

## Migration Plan

### Phase 1: Preparation (No Breaking Changes)

#### Step 1.1: Create New Directory Structure
```bash
# Create all new directories without moving files
mkdir -p ml/tests/{e2e,system,property}
mkdir -p ml/tests/unit/{cli,common,deployment,monitoring,preprocessing}
mkdir -p ml/tests/unit/training/{distillation,non_distilled,student,teacher}
mkdir -p ml/tests/integration/{actors,features,registry,strategies}
```

#### Step 1.2: Create Directory README Files
Create README.md in each major directory explaining:
- Purpose of tests in this directory
- When to add tests here
- Example test patterns
- Coverage requirements

### Phase 2: File Migration (Automated)

#### Step 2.1: Unit Test Migration

**From Root Level to Appropriate Unit Subdirectories:**

| Current Location | New Location | Rationale |
|-----------------|--------------|-----------|
| `test_cli_coverage_backfill.py` | `unit/cli/test_coverage_cli.py` | CLI functionality unit test |
| `test_data_store_validation.py` | `unit/stores/test_data_store.py` | Store validation unit test |
| `test_stores_simple.py` | `unit/stores/test_stores_basic.py` | Basic store unit tests |
| `test_strategy_registry.py` | `unit/registry/test_strategy_registry.py` | Registry unit test |

**From features/ to unit/features/:**

| Current Location | New Location | Rationale |
|-----------------|--------------|-----------|
| `features/test_materialize_cli.py` | `unit/features/test_materialize.py` | Feature materialization unit test |

**From training/ to unit/training/:**

| Current Location | New Location | Rationale |
|-----------------|--------------|-----------|
| `training/test_torchscript_export.py` | `unit/training/test_export.py` | Export functionality unit test |
| `training/teacher/test_tft_cli.py` | `unit/training/teacher/test_tft_teacher.py` | Teacher model unit test |
| `training/teacher/test_tft_teacher_smoke.py` | Remove - covered by `test_tft_teacher.py` | Redundant smoke test |

#### Step 2.2: Integration Test Migration

**From Root Level to Integration:**

| Current Location | New Location | Rationale |
|-----------------|--------------|-----------|
| `test_feature_parity.py` | `integration/features/test_feature_parity.py` | Feature parity integration test |
| `test_feature_store_integration.py` | `integration/stores/test_feature_store_integration.py` | Store integration test |
| `test_stores_integration.py` | `integration/stores/test_store_interactions.py` | Store interaction test |
| `test_registry_gates.py` | `integration/registry/test_registry_gates.py` | Registry gating integration |
| `test_registry_student_integration.py` | `integration/registry/test_registry_student.py` | Student registry integration |

**Keep in integration/ (already well-placed):**
- All files currently in `integration/` directory

#### Step 2.3: E2E Test Migration

**From Root Level to E2E:**

| Current Location | New Location | Rationale |
|-----------------|--------------|-----------|
| `test_data_registry_e2e.py` | `e2e/test_data_registry_flow.py` | Complete data registry flow |
| `test_strategy_store_events.py` | `e2e/test_strategy_event_flow.py` | Strategy event flow |

**From Integration to E2E:**

| Current Location | New Location | Rationale |
|-----------------|--------------|-----------|
| `integration/test_end_to_end_pipeline.py` | `e2e/test_ml_pipeline_complete.py` | Complete pipeline e2e test |
| `integration/test_e2e_signal_actor_featurestore.py` | `e2e/test_signal_generation_flow.py` | Signal generation e2e |
| `integration/test_strategy_store_e2e.py` | `e2e/test_strategy_execution_flow.py` | Strategy execution e2e |

#### Step 2.4: System Test Migration

**From Integration to System:**

| Current Location | New Location | Rationale |
|-----------------|--------------|-----------|
| `integration/test_infrastructure.py` | `system/test_infrastructure.py` | Infrastructure system test |
| `integration/test_dry_run_integration.py` | `system/test_dry_run_system.py` | Dry run system test |
| `integration/test_ml_strategy_backtest.py` | `system/test_ml_backtest_system.py` | Backtest system test |

#### Step 2.5: Property Test Creation

**New Property Tests (extract from existing):**

| Source File | New Property Test | Focus |
|------------|------------------|-------|
| Various hypothesis tests | `property/test_numerical_stability.py` | Numerical properties |
| Actor hypothesis tests | `property/test_temporal_consistency.py` | Time-based properties |
| Feature hypothesis tests | `property/test_feature_properties.py` | Feature computation properties |

### Phase 3: Consolidation and Cleanup

#### Step 3.1: Merge Redundant Tests

**Tests to Consolidate:**

1. **Registry Tests**: Combine similar registry tests into comprehensive suites
   - Merge: `test_registry_performance.py`, `test_registry_statistics.py`, `test_registry_hypothesis_cleaned.py`
   - Into: `unit/registry/test_registry_comprehensive.py`

2. **Feature Tests**: Consolidate feature engineering tests
   - Merge: `test_feature_engineering.py`, `test_feature_engineering_hypothesis.py`
   - Into: `unit/features/test_engineering.py` with property tests in `property/`

3. **Actor Tests**: Combine actor test variants
   - Merge: `test_signal_actor.py`, `test_signal_actor_hypothesis.py`, `test_signal_actor_feature_manifest.py`
   - Into: `unit/actors/test_signal_actor.py` with comprehensive coverage

#### Step 3.2: Remove Obsolete Files

**Files to Remove:**
- `unit/collectors/COVERAGE_REFACTORING_NEEDED.md` - Outdated documentation
- `training/teacher/test_tft_teacher_smoke.py` - Redundant with main test
- Empty `__init__.py` files that don't contain fixtures

### Phase 4: Configuration and Documentation

#### Step 4.1: Update pytest Configuration

**Update `conftest.py`:**

```python
# ml/tests/conftest.py
import pytest
from pathlib import Path

# Configure pytest markers
def pytest_configure(config):
    config.addinivalue_line("markers", "unit: Mark test as unit test")
    config.addinivalue_line("markers", "integration: Mark test as integration test")
    config.addinivalue_line("markers", "e2e: Mark test as end-to-end test")
    config.addinivalue_line("markers", "system: Mark test as system test")
    config.addinivalue_line("markers", "slow: Mark test as slow running")
    config.addinivalue_line("markers", "performance: Mark test as performance benchmark")
    config.addinivalue_line("markers", "property: Mark test as property-based test")
    config.addinivalue_line("markers", "contract: Mark test as contract test")

# Test categorization by directory
def pytest_collection_modifyitems(config, items):
    for item in items:
        # Auto-mark based on location
        test_path = Path(item.fspath)
        if "unit" in test_path.parts:
            item.add_marker(pytest.mark.unit)
        elif "integration" in test_path.parts:
            item.add_marker(pytest.mark.integration)
        elif "e2e" in test_path.parts:
            item.add_marker(pytest.mark.e2e)
        elif "system" in test_path.parts:
            item.add_marker(pytest.mark.system)
        elif "performance" in test_path.parts:
            item.add_marker(pytest.mark.performance)
        elif "property" in test_path.parts:
            item.add_marker(pytest.mark.property)
        elif "contracts" in test_path.parts:
            item.add_marker(pytest.mark.contract)
```

#### Step 4.2: Create pytest.ini

```ini
# ml/tests/pytest.ini
[tool:pytest]
testpaths = ml/tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Markers for test categorization
markers =
    unit: Unit tests - fast, isolated component tests
    integration: Integration tests - component interaction tests
    e2e: End-to-end tests - complete workflow tests
    system: System tests - full system behavior tests
    slow: Slow running tests (>1s)
    performance: Performance benchmark tests
    property: Property-based tests using Hypothesis
    contract: Behavioral contract tests

# Coverage settings
addopts = 
    --strict-markers
    --tb=short
    --cov-branch

# Test execution aliases
[aliases]
test-unit = pytest -m unit
test-integration = pytest -m integration
test-e2e = pytest -m e2e
test-fast = pytest -m "not slow"
test-all = pytest
```

## Naming Conventions

### File Naming

```
Format: test_<module_name>_<aspect>.py

Examples:
- test_signal_actor.py          # Main unit tests
- test_signal_actor_features.py  # Feature-specific tests
- test_feature_parity.py        # Parity testing
- test_ml_pipeline_complete.py  # Complete e2e flow
```

### Class Naming

```python
# Unit tests
class TestSignalActor:
    """Unit tests for SignalActor."""

# Integration tests  
class TestSignalActorIntegration:
    """Integration tests for SignalActor with dependencies."""

# Contract tests
class TestActorContracts:
    """Behavioral contracts all actors must satisfy."""
```

### Method Naming

```python
def test_<action>_<condition>_<expected_result>():
    """Test that <action> when <condition> produces <expected_result>."""
    
Examples:
- test_compute_features_with_missing_data_returns_default()
- test_hot_swap_model_during_inference_maintains_consistency()
- test_signal_generation_after_warmup_produces_valid_signals()
```

## Test Categorization Strategy

### Unit Tests (Target: 95% Coverage)
- **Scope**: Single component/class
- **Dependencies**: Mocked
- **Execution Time**: <100ms per test
- **Database**: In-memory or mocked
- **External Services**: Mocked
- **Example**: Testing FeatureEngineer.compute_features()

### Integration Tests (Target: 85% Coverage)
- **Scope**: Multiple components
- **Dependencies**: Real components
- **Execution Time**: <1s per test
- **Database**: Test database
- **External Services**: Test doubles
- **Example**: Testing FeatureStore + FeatureEngineer integration

### E2E Tests (Target: 70% Coverage)
- **Scope**: Complete user workflow
- **Dependencies**: Full stack
- **Execution Time**: <10s per test
- **Database**: Test database with data
- **External Services**: Test environment
- **Example**: Data ingestion → Feature computation → Model training → Deployment

### System Tests (Target: Key Paths Only)
- **Scope**: Full system behavior
- **Dependencies**: Production-like setup
- **Execution Time**: Can be slow
- **Database**: Production-like data
- **External Services**: Staging environment
- **Example**: Complete backtest with real data

## Guidelines for Future Test Placement

### Decision Tree for Test Placement

```
Is it testing a single component in isolation?
├─ YES → unit/<module>/ 
└─ NO → Is it testing component interactions?
    ├─ YES → Is it a complete user workflow?
    │   ├─ YES → e2e/
    │   └─ NO → integration/<module>/
    └─ NO → Is it testing system behavior?
        ├─ YES → system/
        └─ NO → Is it testing invariants/properties?
            ├─ YES → property/
            └─ NO → Is it a performance test?
                ├─ YES → performance/
                └─ NO → contracts/
```

### Module-Specific Guidelines

**Actors**: 
- Unit: Individual actor methods
- Integration: Actor with stores/registry
- E2E: Complete signal generation flow

**Stores**:
- Unit: CRUD operations
- Integration: Store interactions
- E2E: Data persistence workflow

**Registry**:
- Unit: Registration/retrieval
- Integration: Registry with stores
- Contract: Registry behavioral guarantees

**Features**:
- Unit: Individual feature calculations
- Integration: Feature pipeline
- Property: Numerical stability

**Strategies**:
- Unit: Strategy logic
- Integration: Strategy with actors
- System: Complete backtest

## Pytest Markers Usage

### Required Markers

Every test MUST have at least one category marker:

```python
@pytest.mark.unit
def test_feature_calculation():
    pass

@pytest.mark.integration
@pytest.mark.slow  # Additional marker for slow tests
def test_pipeline_integration():
    pass
```

### Marker Combinations

```python
# Common combinations
@pytest.mark.unit
@pytest.mark.performance
def test_hot_path_latency():
    """Unit test with performance requirements."""

@pytest.mark.integration  
@pytest.mark.property
def test_feature_parity_property():
    """Integration test using property-based testing."""

@pytest.mark.e2e
@pytest.mark.slow
def test_complete_ml_pipeline():
    """End-to-end test that takes time."""
```

## Coverage Targets by Test Type

### Mandatory Coverage Requirements

| Test Type | ML Modules | Utilities | Experimental |
|-----------|------------|-----------|--------------|
| Unit | 95% | 80% | 70% |
| Integration | 85% | 70% | 60% |
| E2E | 70% | - | - |
| Overall | **90%** | 80% | 70% |

### Critical Path Coverage

These components MUST have ≥95% coverage:

1. **Hot Path Components**:
   - `actors/signal.py`
   - `features/engineering.py` (online computation)
   - `stores/feature_store.py` (read operations)

2. **Data Integrity**:
   - `stores/data_processor.py`
   - `registry/` (all registry operations)
   - `features/validation.py`

3. **Trading Logic**:
   - `strategies/ml_strategy.py`
   - `actors/base.py`

## Implementation Timeline

### Week 1: Setup and Migration
- Day 1-2: Create directory structure, update configurations
- Day 3-4: Migrate unit tests
- Day 5: Migrate integration tests

### Week 2: Consolidation
- Day 1-2: Migrate e2e and system tests
- Day 3-4: Consolidate redundant tests
- Day 5: Update documentation

### Week 3: Validation
- Day 1-2: Run full test suite, fix issues
- Day 3-4: Update CI/CD configurations
- Day 5: Team training and documentation

## Migration Script

```python
#!/usr/bin/env python3
"""Automated test migration script."""

import shutil
from pathlib import Path

# Define migrations as (source, destination) tuples
MIGRATIONS = [
    ("test_cli_coverage_backfill.py", "unit/cli/test_coverage_cli.py"),
    ("test_data_store_validation.py", "unit/stores/test_data_store.py"),
    # ... all migrations from the table above
]

def migrate_tests():
    """Execute test migrations."""
    base_path = Path("ml/tests")
    
    for source, dest in MIGRATIONS:
        source_path = base_path / source
        dest_path = base_path / dest
        
        # Create destination directory
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move file
        if source_path.exists():
            print(f"Moving {source} → {dest}")
            shutil.move(source_path, dest_path)
        else:
            print(f"Warning: {source} not found")

if __name__ == "__main__":
    migrate_tests()
```

## Success Metrics

### Quantitative Metrics
- Test execution time reduced by 30%
- Test discovery time <1s
- Coverage maintained at ≥90%
- Zero test failures after migration

### Qualitative Metrics
- Clear test organization understood by team
- Reduced time to find relevant tests
- Easier test maintenance
- Clear guidelines prevent future sprawl

## Risk Mitigation

### Risks and Mitigations

1. **Risk**: Breaking CI/CD during migration
   - **Mitigation**: Create parallel test configuration during transition

2. **Risk**: Lost test coverage
   - **Mitigation**: Run coverage before/after comparison

3. **Risk**: Team confusion during transition
   - **Mitigation**: Comprehensive documentation and training

4. **Risk**: Git history fragmentation
   - **Mitigation**: Use `git mv` to preserve history

## Appendix: Test Organization Examples

### Example: Testing a New Feature

When adding a new `AdvancedFeatureEngineer`:

1. **Unit Test**: `unit/features/test_advanced_engineering.py`
   - Test individual feature calculations
   - Mock data inputs
   - Verify mathematical correctness

2. **Integration Test**: `integration/features/test_advanced_pipeline.py`
   - Test with FeatureStore
   - Verify pipeline integration
   - Test with real data flow

3. **Property Test**: `property/test_advanced_features.py`
   - Numerical stability properties
   - Invariant checking
   - Edge case generation

4. **E2E Test**: Update `e2e/test_ml_pipeline_complete.py`
   - Include new features in complete flow
   - Verify end-to-end correctness

### Example: Testing File Organization

```python
# unit/features/test_advanced_engineering.py
class TestAdvancedFeatureEngineer:
    """Unit tests for AdvancedFeatureEngineer."""
    
    def test_compute_momentum_features(self):
        """Test momentum feature calculation."""
        
    def test_compute_volatility_features(self):
        """Test volatility feature calculation."""

# integration/features/test_advanced_pipeline.py  
class TestAdvancedFeaturePipeline:
    """Integration tests for advanced feature pipeline."""
    
    def test_pipeline_with_feature_store(self):
        """Test complete pipeline with persistence."""
        
    def test_feature_parity_online_offline(self):
        """Verify online/offline computation parity."""

# property/test_advanced_features.py
class TestAdvancedFeatureProperties:
    """Property-based tests for advanced features."""
    
    @given(prices=price_series_strategy())
    def test_momentum_monotonicity(self, prices):
        """Property: Momentum preserves direction."""
```

---

*Document Version: 1.0*
*Last Updated: 2025*
*Next Review: After Phase 1 Implementation*