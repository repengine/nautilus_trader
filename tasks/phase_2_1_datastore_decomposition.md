# Task: [Phase 2.1] DataStore Decomposition

## Context
**Phase:** 2 - Core Store Refactoring (God Classes)
**Task ID:** 2.1
**Depends On:** Phase 1 (DRY Violations - All Tasks Complete)
**Estimated Effort:** 20 hours
**Impact Score:** High - Refactoring 3,730-line god class into 5 components

## Scope
Decompose the monolithic `DataStore` class (3,730 lines) into 5 focused components using the Strangler Fig pattern. This refactoring improves testability, maintainability, and reduces cognitive load while maintaining 100% backward compatibility through a facade.

**Current State:** `ml/stores/data_store.py` (3,730 lines)
**Target State:** 5 specialized components + 1 facade (~800 lines total in facade)

**Components to Extract:**
1. **SchemaValidator** - Type checking, validation rules, quality enforcement
2. **DataReader** - Read operations for features, predictions, signals, earnings
3. **DataWriter** - Write operations for all data types with validation
4. **ContractEnforcer** - Contract retrieval, validation, quality reporting
5. **DataStoreFacade** - Public API that delegates to specialized components

## Required Reading
- [x] REFACTORING_PLAN.md (Phase 2.1, lines 177-195)
- [x] AGENT_TASK_FRAMEWORK.md (Task Template Structure section)
- [x] ml/docs/development/CODING_STANDARDS.md
- [x] ml/docs/architecture/universal_patterns_guide.md
- [x] CLAUDE.md (Universal ML Architecture Patterns)
- [x] ml/stores/data_store.py (entire file - understand current structure)
- [x] tasks/phase_1_2_table_schema_factory.md (reference example)

## Definition of Done
- [ ] All 5 components extracted with clear single responsibilities
- [ ] DataStoreFacade maintains 100% backward compatibility
- [ ] All public APIs preserved (no breaking changes)
- [ ] Feature flag `ML_USE_LEGACY_DATA_STORE` implemented and tested
- [ ] All existing tests pass without modification
- [ ] New unit tests for each component (≥90% coverage per component)
- [ ] Integration tests verify facade behavior matches original
- [ ] Zero new circular dependencies introduced
- [ ] Ruff check passes (zero violations)
- [ ] MyPy --strict passes (zero errors)
- [ ] make validate-nautilus-patterns passes
- [ ] Documentation updated with architecture diagrams
- [ ] Rollback plan tested and documented

## Files to Modify

### Create New Components (5 files)
- [ ] `ml/stores/schema_validator.py` - Validation logic (~400 lines)
- [ ] `ml/stores/data_reader.py` - Read operations (~350 lines)
- [ ] `ml/stores/data_writer.py` - Write operations (~600 lines)
- [ ] `ml/stores/contract_enforcer.py` - Contract validation (~450 lines)
- [ ] `ml/stores/data_store_facade.py` - Public facade (~800 lines)

### Create New Tests (5 files)
- [ ] `ml/tests/unit/stores/test_schema_validator.py`
- [ ] `ml/tests/unit/stores/test_data_reader.py`
- [ ] `ml/tests/unit/stores/test_data_writer.py`
- [ ] `ml/tests/unit/stores/test_contract_enforcer.py`
- [ ] `ml/tests/integration/stores/test_data_store_facade.py`

### Modify Existing (3 files)
- [ ] `ml/stores/data_store.py` - Replace with facade or legacy toggle
- [ ] `ml/stores/__init__.py` - Export new components + facade
- [ ] `ml/stores/protocols.py` - Add component protocols if needed

## Implementation Steps

### Week 3: Extract Independent Components

#### Day 1-2: Extract SchemaValidator (lines 2800-3370)

**Responsibility:** All validation logic including type checking, range validation, regex, nullability, uniqueness, monotonicity, lateness.

**Line Ranges to Extract:**
- `_apply_validation_rule()` (lines 2800-2840)
- `_validate_types()` (lines 2842-2876)
- `_validate_regex()` (lines 2878-2951)
- `_validate_range()` (lines 2953-3037)
- `_validate_uniqueness()` (lines 3038-3114)
- `_validate_monotonicity()` (lines 3115-3189)
- `_validate_nullability()` (lines 3190-3243)
- `_validate_lateness()` (lines 3244-3282)
- `_types_compatible()` (lines 3283-3301)
- `_format_violations()` (lines 3302-3317)
- `_enforce_quality_report()` (lines 3318-3370)

**Steps:**
1. Create `ml/stores/schema_validator.py`
2. Define `SchemaValidatorProtocol` with all validation methods
3. Extract validation methods into `SchemaValidator` class
4. Add proper type annotations and docstrings
5. Remove dependencies on DataStore instance state (pass as parameters)
6. Create comprehensive unit tests
7. Verify metrics are using `ml.common.metrics_bootstrap` (not direct prometheus imports)

**Example Structure:**
```python
"""Schema validation for ML data contracts."""

from __future__ import annotations

from typing import Any, Protocol, cast
import logging

from ml.registry.dataclasses import (
    DataContract,
    DatasetManifest,
    QualityFlag,
    ValidationRule,
    ValidationRuleType,
)
from ml.ml_types import DataFrameLike

logger = logging.getLogger(__name__)


class SchemaValidatorProtocol(Protocol):
    """Protocol for schema validation operations."""

    def validate_batch(
        self,
        data: DataFrameLike,
        manifest: DatasetManifest,
        contract: DataContract,
        strict_mode: bool = False,
    ) -> QualityReport: ...

    def apply_validation_rule(
        self,
        rule: ValidationRule,
        data_frame: object,
        manifest: DatasetManifest,
    ) -> ValidationViolation | None: ...


class SchemaValidator:
    """
    Validates data against schema contracts.

    Performs comprehensive validation including type checking, range validation,
    regex patterns, nullability, uniqueness, monotonicity, and lateness checks.
    """

    def __init__(self) -> None:
        """Initialize schema validator."""
        self._initialize_metrics()

    def _initialize_metrics(self) -> None:
        """Initialize Prometheus metrics using centralized bootstrap."""
        from ml.common.metrics_bootstrap import get_counter, get_histogram

        self.validation_violations_counter = get_counter(
            "ml_validation_violations_total",
            "Total validation violations by type",
        )
        # ... other metrics

    def validate_batch(
        self,
        data: DataFrameLike,
        manifest: DatasetManifest,
        contract: DataContract,
        strict_mode: bool = False,
    ) -> QualityReport:
        """
        Validate a batch of data against contract.

        Parameters
        ----------
        data : DataFrameLike
            Data to validate
        manifest : DatasetManifest
            Dataset manifest with schema
        contract : DataContract
            Data contract with validation rules
        strict_mode : bool
            If True, treat warnings as failures

        Returns
        -------
        QualityReport
            Validation results with quality score and violations
        """
        # Implementation extracted from data_store.py:validate_batch()
        ...

    # All validation methods here...
```

#### Day 3-4: Extract DataReader (lines 533-707)

**Responsibility:** All read operations - features, predictions, signals, earnings.

**Line Ranges to Extract:**
- `get_features_at_or_before()` (lines 533-547)
- `get_latest_prediction_at_or_before()` (lines 549-601)
- `get_latest_signal_at_or_before()` (lines 603-653)
- `get_earnings_actuals_at_or_before()` (lines 655-683)
- `get_earnings_estimate_at_or_before()` (lines 685-705)
- `read_range()` (lines 2310-2408)

**Steps:**
1. Create `ml/stores/data_reader.py`
2. Define `DataReaderProtocol` with all read methods
3. Extract read methods into `DataReader` class
4. Take FeatureStore, ModelStore, StrategyStore, EarningsStore as dependencies (constructor injection)
5. Add proper type annotations and docstrings
6. Create comprehensive unit tests with mocked stores
7. Test cold-path performance (should remain <5ms P99)

**Example Structure:**
```python
"""Read operations for ML data stores."""

from __future__ import annotations

from typing import Protocol
import logging

from ml.stores.protocols import (
    EarningsStoreProtocol,
    PredictionRecord,
    SignalRecord,
)

logger = logging.getLogger(__name__)


class DataReaderProtocol(Protocol):
    """Protocol for data read operations."""

    def get_features_at_or_before(
        self,
        instrument_id: str,
        ts_event: int,
    ) -> dict[str, float] | None: ...

    def get_latest_prediction_at_or_before(
        self,
        instrument_id: str,
        ts_event: int,
        model_id: str | None = None,
    ) -> PredictionRecord | None: ...


class DataReader:
    """
    Performs read operations across ML stores.

    Provides typed read facades over FeatureStore, ModelStore, StrategyStore,
    and EarningsStore for cold-path queries.
    """

    def __init__(
        self,
        feature_store: Any,
        model_store: Any,
        strategy_store: Any,
        earnings_store: EarningsStoreProtocol,
    ) -> None:
        """
        Initialize data reader with store dependencies.

        Parameters
        ----------
        feature_store : FeatureStore
            Feature store instance
        model_store : ModelStore
            Model store instance
        strategy_store : StrategyStore
            Strategy store instance
        earnings_store : EarningsStoreProtocol
            Earnings store instance
        """
        self.feature_store = feature_store
        self.model_store = model_store
        self.strategy_store = strategy_store
        self.earnings_store = earnings_store

    def get_features_at_or_before(
        self,
        instrument_id: str,
        ts_event: int,
    ) -> dict[str, float] | None:
        """
        Return latest feature values at or before timestamp.

        Parameters
        ----------
        instrument_id : str
            Instrument identifier
        ts_event : int
            Event timestamp in nanoseconds

        Returns
        -------
        dict[str, float] | None
            Feature values or None if not found
        """
        return self.feature_store.get_latest_at_or_before(
            instrument_id,
            int(ts_event),
        )

    # All read methods here...
```

### Week 4: Extract Writers and Contract Enforcement

#### Day 1-3: Extract DataWriter (lines 1201-2220)

**Responsibility:** All write operations with validation, event emission, and watermark updates.

**Line Ranges to Extract:**
- `write_ingestion()` (lines 1201-1688)
- `write_features()` (lines 1706-1823)
- `write_predictions()` (lines 1824-1920)
- `write_signals()` (lines 1921-2014)
- `write_earnings_actual()` (lines 2015-2124)
- `write_earnings_estimate()` (lines 2125-2219)
- `_emit_success_event_and_update()` (lines 2220-2309)
- Helper methods for data conversion (lines 3372-3557)

**Steps:**
1. Create `ml/stores/data_writer.py`
2. Define `DataWriterProtocol` with all write methods
3. Extract write methods into `DataWriter` class
4. Inject ContractEnforcer, SchemaValidator, and underlying stores
5. Preserve event emission and watermark update logic
6. Add proper type annotations and docstrings
7. Create comprehensive unit tests
8. Test integration with message bus

**Example Structure:**
```python
"""Write operations for ML data stores."""

from __future__ import annotations

from typing import Any, Protocol
import logging
import time

from ml.stores.base import FeatureData, ModelPrediction, StrategySignal
from ml.stores.contract_enforcer import ContractEnforcer
from ml.stores.schema_validator import SchemaValidator

logger = logging.getLogger(__name__)


class DataWriterProtocol(Protocol):
    """Protocol for data write operations."""

    def write_ingestion(
        self,
        dataset_id: str,
        records: Any,
        source: str,
        run_id: str,
        stage: str | None = None,
    ) -> DataEvent: ...

    def write_features(
        self,
        instrument_id: str,
        features: list[FeatureData],
        source: str = "computed",
        run_id: str | None = None,
    ) -> DataEvent: ...


class DataWriter:
    """
    Performs write operations with validation and event emission.

    Wraps underlying stores with contract validation, quality enforcement,
    event emission, and watermark tracking.
    """

    def __init__(
        self,
        feature_store: Any,
        model_store: Any,
        strategy_store: Any,
        earnings_store: Any,
        contract_enforcer: ContractEnforcer,
        schema_validator: SchemaValidator,
        registry: Any,
        publisher: Any | None = None,
        enable_publishing: bool = False,
        fail_on_validation_error: bool = True,
        batch_size: int = 10000,
    ) -> None:
        """
        Initialize data writer with dependencies.

        Parameters
        ----------
        feature_store : FeatureStore
            Feature store instance
        model_store : ModelStore
            Model store instance
        strategy_store : StrategyStore
            Strategy store instance
        earnings_store : EarningsStoreProtocol
            Earnings store instance
        contract_enforcer : ContractEnforcer
            Contract enforcement component
        schema_validator : SchemaValidator
            Schema validation component
        registry : RegistryProtocol
            Data registry for manifest/contract retrieval
        publisher : MessagePublisherProtocol | None
            Message bus publisher (optional)
        enable_publishing : bool
            Enable event publishing
        fail_on_validation_error : bool
            If True, fail writes on validation errors
        batch_size : int
            Batch size for write operations
        """
        self.feature_store = feature_store
        self.model_store = model_store
        self.strategy_store = strategy_store
        self.earnings_store = earnings_store
        self.contract_enforcer = contract_enforcer
        self.schema_validator = schema_validator
        self.registry = registry
        self.publisher = publisher
        self.enable_publishing = enable_publishing
        self.fail_on_validation_error = fail_on_validation_error
        self.batch_size = batch_size
        self._initialize_metrics()

    # All write methods here...
```

#### Day 4-5: Extract ContractEnforcer (lines 2567-2745 + validation logic)

**Responsibility:** Contract retrieval, caching, quality enforcement, migration window management.

**Line Ranges to Extract:**
- `preflight_check()` (lines 968-1175)
- `validate_batch()` (lines 2410-2561) - delegates to SchemaValidator
- `_get_manifest()` (lines 2567-2589)
- `_get_contract()` (lines 2591-2608)
- `_ensure_dataset_registered()` (lines 3579-3609)
- `_compute_schema_hash()` (lines 3610-3646)
- `_is_in_migration_window()` (lines 3647-3674)
- `_start_migration_window()` (lines 3675-3698)
- Schema migration state management

**Steps:**
1. Create `ml/stores/contract_enforcer.py`
2. Define `ContractEnforcerProtocol`
3. Extract contract management methods
4. Take SchemaValidator as dependency
5. Maintain caching logic for manifests and contracts
6. Preserve migration window state management
7. Add proper type annotations and docstrings
8. Create comprehensive unit tests
9. Test migration window scenarios

**Example Structure:**
```python
"""Contract enforcement for ML data operations."""

from __future__ import annotations

from typing import Any, Protocol
import logging
import time

from ml.registry.dataclasses import DataContract, DatasetManifest
from ml.stores.schema_validator import SchemaValidator

logger = logging.getLogger(__name__)


class ContractEnforcerProtocol(Protocol):
    """Protocol for contract enforcement operations."""

    def preflight_check(
        self,
        dataset_id: str,
        data: Any,
        strict: bool = True,
    ) -> tuple[bool, str | None, dict[str, Any]]: ...

    def get_manifest(self, dataset_id: str) -> DatasetManifest: ...

    def get_contract(self, dataset_id: str) -> DataContract: ...


class ContractEnforcer:
    """
    Enforces data contracts and manages schema versions.

    Retrieves and caches manifests/contracts, validates data against contracts,
    performs preflight checks, and manages schema migration windows.
    """

    def __init__(
        self,
        registry: Any,
        schema_validator: SchemaValidator,
        allow_schema_migration: bool = False,
        schema_migration_window_hours: int = 24,
    ) -> None:
        """
        Initialize contract enforcer.

        Parameters
        ----------
        registry : RegistryProtocol
            Data registry for manifest/contract retrieval
        schema_validator : SchemaValidator
            Schema validation component
        allow_schema_migration : bool
            Allow dual-write during schema migration
        schema_migration_window_hours : int
            Hours to allow dual-write during migration
        """
        self.registry = registry
        self.schema_validator = schema_validator
        self.allow_schema_migration = allow_schema_migration
        self.schema_migration_window_hours = schema_migration_window_hours

        # Caches
        self._manifest_cache: dict[str, DatasetManifest] = {}
        self._contract_cache: dict[str, DataContract] = {}
        self._schema_migration_state: dict[str, dict[str, Any]] = {}

    def preflight_check(
        self,
        dataset_id: str,
        data: Any,
        strict: bool = True,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        """
        Perform preflight schema validation before processing.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        data : DataFrame | list
            Data to validate
        strict : bool
            Require exact schema match

        Returns
        -------
        tuple[bool, str | None, dict[str, Any]]
            (success, error_message, validation_details)
        """
        # Implementation extracted from data_store.py:preflight_check()
        ...

    # All contract methods here...
```

### Week 4: Create Facade and Feature Flag

#### Day 5: Create DataStoreFacade

**Responsibility:** Maintain backward-compatible public API by delegating to specialized components.

**Steps:**
1. Create `ml/stores/data_store_facade.py`
2. Replicate all public methods from original DataStore
3. Initialize all 5 components in constructor
4. Delegate each method to appropriate component
5. Preserve MLComponentMixin, BusPublisherMixin, DataRegistryMixin inheritance
6. Add feature flag check: `ML_USE_LEGACY_DATA_STORE`
7. If flag is set, use original implementation; otherwise use new components
8. Add proper type annotations and docstrings

**Example Structure:**
```python
"""
DataStore facade maintaining backward compatibility.

This facade delegates to specialized components while preserving the original
public API. Feature flag ML_USE_LEGACY_DATA_STORE controls legacy vs new path.
"""

from __future__ import annotations

import os
import logging
from typing import Any

from ml.common.protocols import MLComponentMixin
from ml.common.message_bus import BusPublisherMixin
from ml.stores.mixins import DataRegistryMixin
from ml.stores.schema_validator import SchemaValidator
from ml.stores.data_reader import DataReader
from ml.stores.data_writer import DataWriter
from ml.stores.contract_enforcer import ContractEnforcer

logger = logging.getLogger(__name__)


class DataStore(MLComponentMixin, BusPublisherMixin, DataRegistryMixin):
    """
    Unified interface for ML data operations with contract validation.

    This facade delegates to specialized components while maintaining
    100% backward compatibility with the original DataStore API.
    """

    def __init__(
        self,
        connection_string: str,
        registry: Any | None = None,
        feature_store: Any | None = None,
        model_store: Any | None = None,
        strategy_store: Any | None = None,
        earnings_store: Any | None = None,
        data_processor: Any | None = None,
        publisher: Any | None = None,
        enable_publishing: bool = False,
        fail_on_validation_error: bool = True,
        batch_size: int = 10000,
        allow_schema_migration: bool = False,
        schema_migration_window_hours: int = 24,
        raw_writer: Any | None = None,
        raw_reader: Any | None = None,
        circuit_breaker: Any | None = None,
    ) -> None:
        """
        Initialize DataStore with registry and underlying stores.

        Parameters match original DataStore constructor for compatibility.
        """
        # Check feature flag
        use_legacy = os.getenv("ML_USE_LEGACY_DATA_STORE", "0") == "1"

        if use_legacy:
            # Use original implementation
            logger.info("Using legacy DataStore implementation")
            from ml.stores.data_store_legacy import DataStoreLegacy
            self._impl = DataStoreLegacy(
                connection_string=connection_string,
                registry=registry,
                feature_store=feature_store,
                model_store=model_store,
                strategy_store=strategy_store,
                earnings_store=earnings_store,
                data_processor=data_processor,
                publisher=publisher,
                enable_publishing=enable_publishing,
                fail_on_validation_error=fail_on_validation_error,
                batch_size=batch_size,
                allow_schema_migration=allow_schema_migration,
                schema_migration_window_hours=schema_migration_window_hours,
                raw_writer=raw_writer,
                raw_reader=raw_reader,
                circuit_breaker=circuit_breaker,
            )
            self._use_legacy = True
        else:
            # Use new component-based implementation
            logger.info("Using component-based DataStore implementation")

            # Initialize all stores (same as original)
            # ... store initialization code ...

            # Initialize components
            self._schema_validator = SchemaValidator()
            self._contract_enforcer = ContractEnforcer(
                registry=self.registry,
                schema_validator=self._schema_validator,
                allow_schema_migration=allow_schema_migration,
                schema_migration_window_hours=schema_migration_window_hours,
            )
            self._data_reader = DataReader(
                feature_store=self.feature_store,
                model_store=self.model_store,
                strategy_store=self.strategy_store,
                earnings_store=self.earnings_store,
            )
            self._data_writer = DataWriter(
                feature_store=self.feature_store,
                model_store=self.model_store,
                strategy_store=self.strategy_store,
                earnings_store=self.earnings_store,
                contract_enforcer=self._contract_enforcer,
                schema_validator=self._schema_validator,
                registry=self.registry,
                publisher=publisher,
                enable_publishing=enable_publishing,
                fail_on_validation_error=fail_on_validation_error,
                batch_size=batch_size,
            )
            self._use_legacy = False

    # Delegate all public methods
    def write_ingestion(self, *args, **kwargs):
        """Write ingestion data with validation."""
        if self._use_legacy:
            return self._impl.write_ingestion(*args, **kwargs)
        return self._data_writer.write_ingestion(*args, **kwargs)

    def write_features(self, *args, **kwargs):
        """Write features with validation."""
        if self._use_legacy:
            return self._impl.write_features(*args, **kwargs)
        return self._data_writer.write_features(*args, **kwargs)

    def get_features_at_or_before(self, *args, **kwargs):
        """Get latest features at or before timestamp."""
        if self._use_legacy:
            return self._impl.get_features_at_or_before(*args, **kwargs)
        return self._data_reader.get_features_at_or_before(*args, **kwargs)

    def validate_batch(self, *args, **kwargs):
        """Validate batch against contract."""
        if self._use_legacy:
            return self._impl.validate_batch(*args, **kwargs)
        manifest = self._contract_enforcer.get_manifest(args[0])
        contract = self._contract_enforcer.get_contract(args[0])
        return self._schema_validator.validate_batch(
            args[1],
            manifest,
            contract,
            kwargs.get('strict_mode', False),
        )

    # ... all other public methods ...

    def get_health_status(self) -> dict[str, Any]:
        """Get health status from all components."""
        if self._use_legacy:
            return self._impl.get_health_status()

        return {
            "schema_validator": "healthy",
            "contract_enforcer": "healthy",
            "data_reader": "healthy",
            "data_writer": "healthy",
            "feature_store": self.feature_store.get_health_status(),
            "model_store": self.model_store.get_health_status(),
            "strategy_store": self.strategy_store.get_health_status(),
        }
```

#### Day 6-7: Integration Testing and Validation

**Steps:**
1. Move original `data_store.py` to `data_store_legacy.py`
2. Update `ml/stores/__init__.py` exports
3. Create integration test suite:
   - Test facade with legacy flag ON - verify original behavior
   - Test facade with legacy flag OFF - verify new behavior
   - Compare outputs between legacy and new implementations
   - Test all public APIs for backward compatibility
4. Run full test suite
5. Fix any failures
6. Performance benchmarking (ensure no regression)
7. Update documentation

### Testing Strategy

**Unit Tests (Per Component):**
```python
# ml/tests/unit/stores/test_schema_validator.py
def test_schema_validator_type_validation():
    """Test type validation against manifest schema."""
    ...

def test_schema_validator_range_validation():
    """Test range validation rules."""
    ...

def test_schema_validator_quality_score_calculation():
    """Test quality score calculation."""
    ...


# ml/tests/unit/stores/test_data_reader.py
def test_data_reader_get_features():
    """Test feature retrieval with mocked store."""
    ...

def test_data_reader_get_predictions():
    """Test prediction retrieval with mocked store."""
    ...


# ml/tests/unit/stores/test_data_writer.py
def test_data_writer_write_features_success():
    """Test successful feature write."""
    ...

def test_data_writer_write_features_validation_failure():
    """Test feature write with validation failure."""
    ...


# ml/tests/unit/stores/test_contract_enforcer.py
def test_contract_enforcer_preflight_check():
    """Test preflight validation."""
    ...

def test_contract_enforcer_schema_migration():
    """Test schema migration window."""
    ...
```

**Integration Tests:**
```python
# ml/tests/integration/stores/test_data_store_facade.py
def test_facade_backward_compatibility():
    """Verify facade matches original API."""
    legacy = DataStoreLegacy(...)
    facade = DataStore(...)  # Feature flag OFF

    # Compare outputs for same operations
    assert facade.write_features(...) == legacy.write_features(...)
    ...

def test_facade_feature_flag_legacy():
    """Test facade with ML_USE_LEGACY_DATA_STORE=1."""
    os.environ["ML_USE_LEGACY_DATA_STORE"] = "1"
    store = DataStore(...)
    # Verify using legacy path
    ...

def test_facade_feature_flag_new():
    """Test facade with ML_USE_LEGACY_DATA_STORE=0."""
    os.environ["ML_USE_LEGACY_DATA_STORE"] = "0"
    store = DataStore(...)
    # Verify using new component path
    ...
```

## Testing Requirements

### Unit Tests (≥90% coverage per component)
- [ ] test_schema_validator.py - All validation methods
- [ ] test_data_reader.py - All read operations
- [ ] test_data_writer.py - All write operations
- [ ] test_contract_enforcer.py - Contract management

### Integration Tests
- [ ] test_data_store_facade.py - Backward compatibility
- [ ] test_feature_flag_toggle.py - Legacy vs new behavior
- [ ] test_component_integration.py - Components work together

### Performance Tests
- [ ] Benchmark read operations (target: <5ms P99)
- [ ] Benchmark write operations (should not regress)
- [ ] Benchmark validation operations (should not regress)

### Regression Tests
- [ ] All existing DataStore tests pass unchanged
- [ ] No behavioral changes in public APIs
- [ ] Event emission unchanged
- [ ] Watermark updates unchanged

## Rollback Plan

### Immediate Rollback (Production Issue)
```bash
# Set feature flag to use legacy implementation
export ML_USE_LEGACY_DATA_STORE=1

# Restart services
kubectl rollout restart deployment/ml-service
```

### Code Rollback (Development Issue)
```bash
# Revert all new files
git checkout ml/stores/schema_validator.py
git checkout ml/stores/data_reader.py
git checkout ml/stores/data_writer.py
git checkout ml/stores/contract_enforcer.py
git checkout ml/stores/data_store_facade.py
git checkout ml/stores/data_store_legacy.py

# Restore original
git checkout ml/stores/data_store.py
git checkout ml/stores/__init__.py

# Revert tests
git checkout ml/tests/unit/stores/test_schema_validator.py
git checkout ml/tests/unit/stores/test_data_reader.py
git checkout ml/tests/unit/stores/test_data_writer.py
git checkout ml/tests/unit/stores/test_contract_enforcer.py
git checkout ml/tests/integration/stores/test_data_store_facade.py
```

### Verification After Rollback
```bash
# Run tests
pytest ml/tests/unit/stores/ -v
pytest ml/tests/integration/stores/ -v

# Verify imports
python -c "from ml.stores import DataStore; print('OK')"
```

## Success Metrics

### Code Quality Metrics
- Lines reduced: 3,730 → ~2,600 (5 components + facade)
- Average file size: 3,730 → ~400 lines (85% reduction per file)
- Cyclomatic complexity: Reduced by ~70% (smaller focused functions)
- Test coverage: ≥90% per component (up from ~75% for monolith)

### Architecture Metrics
- Number of responsibilities: 1 god class → 5 focused components
- Files affected: 1 → 6 (5 components + 1 facade)
- Circular dependencies: 0 (no new cycles introduced)
- Protocol conformance: 100% (all components implement protocols)

### Performance Metrics
- Read operation latency: <5ms P99 (no regression)
- Write operation latency: <50ms P99 (no regression)
- Validation latency: <100ms P99 (no regression)
- Memory usage: ≤10% increase (acceptable for better structure)

### Testing Metrics
- Unit tests: +5 test files, ~200 new tests
- Integration tests: +1 test file, ~20 new tests
- Test execution time: <10% increase (parallel execution)
- Coverage: 75% → 90% (improved testability)

### Maintainability Metrics
- Cognitive load: Reduced (smaller focused classes)
- Onboarding time: Faster (clearer separation of concerns)
- Change impact: Localized (changes affect single component)
- Documentation: Improved (clear component boundaries)

## Notes

### Critical Requirements
- **100% Backward Compatibility:** All public APIs must work identically
- **Feature Flag:** `ML_USE_LEGACY_DATA_STORE` must work for safe rollout
- **No Breaking Changes:** Existing code using DataStore must work unchanged
- **Zero New Cycles:** No circular dependencies introduced
- **Strangler Fig Pattern:** New code alongside old, then switch via flag

### Strangler Fig Pattern Benefits
- Safe incremental migration
- Easy rollback via feature flag
- Low risk to production
- Allows parallel testing of old vs new
- Confidence in correctness before full cutover

### Component Interaction Flow
```
┌─────────────────────────────────────────────────────┐
│              DataStoreFacade (Public API)            │
│  - Maintains backward compatibility                 │
│  - Feature flag toggle (legacy vs new)             │
└──────────────┬──────────────────────────────────────┘
               │
               ├──> SchemaValidator
               │    - Type checking
               │    - Validation rules
               │    - Quality enforcement
               │
               ├──> ContractEnforcer
               │    - Manifest retrieval
               │    - Contract caching
               │    - Migration windows
               │    └──> SchemaValidator (composition)
               │
               ├──> DataReader
               │    - Feature queries
               │    - Prediction queries
               │    - Signal queries
               │    - Earnings queries
               │
               └──> DataWriter
                    - Write with validation
                    - Event emission
                    - Watermark updates
                    └──> ContractEnforcer (composition)
                    └──> SchemaValidator (composition)
```

### Architecture Decisions
1. **Protocol-First:** All components implement protocols for testability
2. **Dependency Injection:** Components receive dependencies in constructor
3. **Composition Over Inheritance:** Components compose each other, not inherit
4. **Single Responsibility:** Each component has one clear purpose
5. **Metrics Bootstrap:** Use `ml.common.metrics_bootstrap` for all metrics

### Migration Path
1. **Week 3:** Extract independent components (SchemaValidator, DataReader)
2. **Week 4:** Extract dependent components (DataWriter, ContractEnforcer)
3. **Week 4:** Create facade with feature flag
4. **Week 5:** Integration testing and validation
5. **Week 6:** Deploy with flag ON (legacy mode)
6. **Week 7:** Gradual rollout with flag OFF (new mode)
7. **Week 8:** Remove legacy code if new mode stable

### Known Challenges
- **State Management:** Original DataStore has internal state (caches, migration state) - must be preserved
- **Event Emission:** Must maintain identical event emission behavior
- **Watermark Updates:** Must preserve watermark update logic
- **Error Handling:** Must maintain identical error behavior
- **Metrics:** Must maintain identical metric names and labels for dashboard compatibility
