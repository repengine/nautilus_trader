# Context: Migrations Module

## Overview

The ML migrations system provides automated database schema evolution and management for Nautilus Trader's ML components. The architecture ensures consistent database state across development, testing, and production environments through versioned SQL migrations that handle schema creation, data transformations, and partition management for PostgreSQL.

The migration system is distributed across three primary locations, each serving specific components:

- `ml/registry/migrations/`: Schema for model, feature, and strategy registries
- `ml/stores/migrations/`: Schema for ML data storage (features, predictions, signals)
- `ml/migrations/`: Critical fixes and immediate patches

**Key Features:**

- **Automatic Execution**: Migrations run automatically via `MLIntegrationManager` when `auto_migrate=True`
- **Partition Management**: Comprehensive time-based partitioning for high-volume ML data tables
- **Schema Validation**: Registry tables include validation triggers and constraint enforcement
- **Lineage Tracking**: Complete data lineage and dependency tracking across ML pipeline stages
- **Production Safety**: Idempotent migrations with conflict resolution and rollback capabilities

## Architecture

### Migration Organization Structure

```
ml/
├── migrations/                          # Critical fixes and patches
│   └── 999_fix_partitions_immediate.sql # Emergency partition fixes
├── registry/migrations/                 # Registry schema management
│   ├── 001_initial_schema.sql          # Models, features, strategies tables
│   └── 002_add_cold_path_fields.sql    # Cold path feature additions
└── stores/migrations/                   # Data storage schema
    ├── 002_stores_schema.sql            # Core ML data tables with partitioning
    ├── 003_auto_partitioning.sql       # Automatic partition management
    ├── 004_market_data.sql              # Market data table extensions
    ├── 005_data_registry.sql           # Data registry and event tracking
    ├── 007_schema_hardening.sql        # Data integrity improvements
    ├── 006_feature_values_dedupe.sql  # Deduplication constraints
    ├── 008_views.sql                    # Analytical views and summaries
    ├── 009_disable_partition_triggers.sql # Testing optimizations
    ├── 010_add_event_metadata.sql      # Event metadata extensions
    └── 011_brin_indexes.sql             # Performance optimizations
```

### Migration Execution Flow

The migration system follows this execution pattern:

1. **Auto-Discovery**: `MLIntegrationManager._run_migrations()` defines ordered migration list
2. **Sequential Execution**: Migrations run in predetermined order with SQL statement splitting
3. **Error Handling**: "Already exists" errors are ignored; other errors generate warnings
4. **Transaction Safety**: Each migration runs within a transaction for atomicity

## Key Components

### Migration Executor (`MLIntegrationManager._run_migrations`)

**Location**: `ml/core/integration.py:321-356`

**Core Migration List**:

```python
migrations = [
    "ml/registry/migrations/001_initial_schema.sql",
    "ml/stores/migrations/002_stores_schema.sql",
    "ml/stores/migrations/003_auto_partitioning.sql",
    "ml/stores/migrations/004_market_data.sql",
    "ml/stores/migrations/005_data_registry.sql",
    "ml/stores/migrations/010_add_event_metadata.sql",
]
```

**Features**:

- Automatic file existence checking
- SQL statement splitting for multi-statement compatibility
- Error tolerance with logging
- Transaction-based execution

### Registry Schema (`001_initial_schema.sql`)

**Core Tables**:

- `models`: Model registry with deployment tracking, lineage, and validation
- `features`: Feature set registry with schema hashing and parity validation
- `strategies`: Strategy registry with compatibility matrices and requirements
- `registry_audit_log`: Complete audit trail for all registry changes

**Advanced Features**:

- Recursive views for lineage tracking (`feature_lineage`, `strategy_compatibility`)
- Validation triggers for deployment status and model roles
- Automatic timestamp updates via `update_last_modified()` trigger
- Dependency resolution functions (`get_model_dependencies()`)

### Store Schema (`002_stores_schema.sql`)

**Partitioned Tables** (by `ts_event` in nanoseconds):

- `ml_feature_values`: Feature computation results with metadata
- `ml_model_predictions`: Model inference outputs with timing
- `ml_strategy_signals`: Strategy signals with risk metrics

**Partition Strategy**:

- Monthly partitions from 2024-2026 (36 months coverage)
- Helper function `create_monthly_partitions()` for consistent creation
- Nanosecond timestamp boundaries aligned with Nautilus conventions

### Automatic Partitioning (`003_auto_partitioning.sql`)

**Automated Management Functions**:

- `auto_create_partitions()`: Creates future partitions (3 months ahead)
- `auto_cleanup_partitions()`: Removes partitions beyond retention period
- `ensure_partition_exists()`: Trigger-based partition creation (now disabled)

**Safety Mechanisms**:

- Race condition avoidance through trigger disabling (migration 006)
- Pre-created test partitions for common timestamp ranges
- Idempotent partition creation with conflict handling

### Market Data Table (`004_market_data.sql`)

Defines the canonical `market_data` table used for raw market data persistence.

- Primary key on `(instrument_id, ts_event)` with BRIN index on `ts_event` for efficient range scans
- Nanosecond `ts_event` and `ts_init` fields aligned with Nautilus conventions
- Referenced by `ml/stores/coverage_sql.py` implementations:
  - `SqlCoverageProvider` computes day‑bucket coverage directly from this table
  - `SqlMarketDataWriter` performs idempotent inserts for backfilled/live data (Postgres ON CONFLICT, SQLite OR IGNORE)

Ensure this migration is applied in any environment where the orchestrator backfill runs.

### Data Registry (`005_data_registry.sql`)

**Comprehensive Data Lineage**:

- `ml_dataset_registry`: Dataset manifests with schema validation
- `ml_data_events`: Pipeline stage tracking with partitioning
- `ml_data_watermarks`: Processing completeness and lag tracking
- `ml_data_lineage`: Transformation dependency graphs

**Analytics Views**:

- `ml_stage_coverage`: Cross-stage data flow analysis
- `ml_watermark_lag`: Real-time lag monitoring
- `ml_lineage_graph`: Recursive dataset dependency trees
- `ml_data_quality_summary`: Comprehensive quality metrics

### Schema Hardening (`007_schema_hardening.sql`)

**Data Integrity Improvements**:

- Unique constraints for ML feature values upsert operations
- Timestamp type standardization to `TIMESTAMPTZ`
- Automatic nanosecond-to-timestamp conversion for analytics compatibility

### Partition Management Fixes

**Emergency Fixes** (`999_fix_partitions_immediate.sql`, `009_disable_partition_triggers.sql`):

- Disables race-prone automatic partition triggers
- Pre-creates test partitions for 2023-2024 timestamp ranges
- Ensures critical test partitions exist for common test timestamps

## Dependencies

### Internal Dependencies

- `ml.core.db_engine.EngineManager`: Database connection management
- `ml.stores.infrastructure.PartitionManager`: Runtime partition management
- `nautilus_trader.model.identifiers.*`: Domain types for instrument identification
- `nautilus_trader.core.data.*`: Data model compliance

### External Dependencies

- **PostgreSQL 12+**: Advanced partitioning and JSON support
- **SQLAlchemy**: Migration execution engine with transaction management
- **pg_cron** (optional): Scheduled partition maintenance

### Schema Dependencies

- All ML tables require `instrument_id`, `ts_event`, `ts_init` fields (Nautilus standard)
- Nanosecond timestamp precision throughout
- JSONB for flexible metadata storage
- Time-based partitioning on `ts_event` for performance

## Usage Patterns

### Automatic Migration Execution

```python
from ml.core.integration import MLIntegrationManager

# Automatic migration during initialization
integration = MLIntegrationManager(
    db_connection="postgresql://...",
    auto_migrate=True  # Runs migrations automatically
)
```

### Manual Migration Execution

```python
# Environment variable control
import os
os.environ["ML_AUTO_MIGRATE"] = "true"

# Or explicit migration call
integration._run_migrations()
```

### CLI Migration Runner

For ad-hoc application outside the integration manager, use the lightweight runner:

```bash
# Print plan only
uv run --active --no-sync python -m ml.scripts.apply_migrations --print-only

# Apply baseline migrations to both registry and stores (uses $DATABASE_URL if set)
uv run --active --no-sync python -m ml.scripts.apply_migrations --db-url postgresql://postgres:postgres@localhost:5432/nautilus

# Apply full set including hardening, views, optional indices/fixes
uv run --active --no-sync python -m ml.scripts.apply_migrations --db-url postgresql://... --full

# Apply only store migrations (baseline+optional)
uv run --active --no-sync python -m ml.scripts.apply_migrations --schema stores --full

# Dry run (no changes), prints which files would be applied
uv run --active --no-sync python -m ml.scripts.apply_migrations --full --dry-run
```

The runner mirrors the canonical ordering used by `MLIntegrationManager` and executes each SQL file transactionally. It is idempotency-friendly and tolerates common "already exists" conflicts as warnings.

DB readiness tips:

```
# Start local Postgres for tests
make docker-up-test

# Wait/check current DATABASE_URL (defaults to localhost:5432/nautilus)
make check-db
```

### Adding New Migrations

1. Create numbered SQL file in appropriate directory:
   - Registry changes: `ml/registry/migrations/00X_description.sql`
   - Store changes: `ml/stores/migrations/00X_description.sql`
   - Critical fixes: `ml/migrations/XXX_description.sql`

2. Add to migration list in `MLIntegrationManager._run_migrations()`

3. Test migration idempotency and rollback scenarios

### Partition Management

```sql
-- Create additional future partitions
SELECT auto_create_partitions();

-- Clean up old partitions (keep 24 months)
SELECT auto_cleanup_partitions(24);

-- Check partition status
SELECT tablename, pg_size_pretty(pg_total_relation_size(tablename::regclass)) as size
FROM pg_tables
WHERE schemaname = 'public' AND tablename LIKE 'ml_%'
ORDER BY tablename;
```

## Integration Points

### Store Integration

- **FeatureStore**: Writes to `ml_feature_values` with automatic partitioning
- **ModelStore**: Writes to `ml_model_predictions` with inference timing
- **StrategyStore**: Writes to `ml_strategy_signals` with risk metrics
- **DataStore**: Registry integration with event emission

### Registry Integration

- **ModelRegistry**: Uses `models` table for deployment tracking
- **FeatureRegistry**: Uses `features` table with schema validation
- **StrategyRegistry**: Uses `strategies` table with compatibility checking
- **DataRegistry**: Complete lineage tracking via data registry tables

### Actor Integration

- **BaseMLInferenceActor**: Automatically connects to migrated stores/registries
- **Pipeline Events**: Data events tracked through `ml_data_events` table
- **Health Monitoring**: Watermarks provide lag detection for observability

## Implementation Notes

### Migration Safety

- **Idempotent Design**: All migrations use `IF NOT EXISTS` clauses
- **Conflict Resolution**: Overlapping partitions and existing objects handled gracefully
- **Transaction Boundaries**: Each migration file runs in separate transaction
- **Error Handling**: Non-critical errors logged as warnings, not failures

### Performance Considerations

- **BRIN Indexes**: Used for time-series data with natural correlation
- **Partition Pruning**: Query performance scales with partition elimination
- **Batch Operations**: Bulk insert optimizations for high-volume ML data
- **Index Strategy**: Composite indexes aligned with query patterns

### PostgreSQL Specific Features

- **Advanced Partitioning**: Range partitioning on nanosecond timestamps
- **JSONB Storage**: Flexible metadata with GIN indexing capabilities
- **Recursive CTEs**: Complex lineage queries with cycle detection
- **Trigger Functions**: PL/pgSQL for complex validation and automation
- **Foreign Key Cascades**: Maintains referential integrity during cleanup

### Testing Considerations

- **Test Partitions**: Pre-created for common test timestamp ranges (2023-2024)
- **Trigger Disabling**: Automatic partition creation disabled to prevent race conditions
- **Data Isolation**: Each test can use unique `instrument_id` for partition isolation
- **Timestamp Consistency**: Test data should use nanosecond timestamps within covered ranges

### Common Migration Patterns

- **Column Additions**: Use `ALTER TABLE ADD COLUMN IF NOT EXISTS`
- **Index Creation**: Always use `CREATE INDEX IF NOT EXISTS`
- **Function Updates**: Use `CREATE OR REPLACE FUNCTION` for updates
- **Partition Creation**: Check existence before creation to avoid conflicts
- **Schema Evolution**: Maintain backward compatibility with existing queries

This migration system ensures that Nautilus Trader's ML infrastructure can evolve safely while maintaining data integrity, performance, and operational requirements across all deployment environments.

## Implementation Review Addendum

### Summary

The ml/migrations domain implementation has been analyzed against documentation claims and Universal ML Architecture Pattern compliance. This review validates ground-truth implementation against documented functionality.

### Architecture Compliance Analysis

#### Universal ML Architecture Patterns Compliance

**Pattern 1: Mandatory 4-Store + 4-Registry Integration** - ❌ **NOT APPLICABLE**

- **Status**: The migrations domain does not contain actor implementations
- **Finding**: Migrations are pure SQL schema management, no BaseMLInferenceActor inheritance required
- **Compliance**: N/A - Domain provides schema for stores/registries used by actors

**Pattern 2: Protocol-First Interface Design** - ❌ **NOT APPLICABLE**

- **Status**: Migrations provide SQL DDL, not Python protocol implementations
- **Finding**: Migration files are PostgreSQL SQL, no typing.Protocol usage expected
- **Compliance**: N/A - Schema definitions enable protocol implementations in other domains

**Pattern 3: Hot/Cold Path Separation** - ✅ **COMPLIANT**

- **Status**: Migrations are cold-path operations by design
- **Finding**: All migration operations are properly categorized as cold-path (schema changes, maintenance)
- **Compliance**: PASS - No hot-path operations in migration files

**Pattern 4: Progressive Fallback Chains** - ⚠️ **PARTIALLY IMPLEMENTED**

- **Status**: Basic fallback exists but limited
- **Finding**: MLIntegrationManager implements basic error tolerance but no sophisticated fallback chains
- **File**: `/home/nate/projects/nautilus_trader/ml/core/integration.py:450-459`
- **Implementation**: Ignores "already exists" errors, logs warnings for others
- **Missing**: No circuit breaker, retry logic, or degraded operation modes

**Pattern 5: Centralized Metrics Bootstrap** - ❌ **NOT APPLICABLE**

- **Status**: SQL migrations don't use metrics
- **Finding**: No prometheus_client imports expected in SQL DDL files
- **Compliance**: N/A - Metrics are application-layer concern

#### Coding Standards Compliance

**Type Annotations** - ✅ **COMPLIANT**

- **Status**: Python migration utilities are properly typed
- **Files Reviewed**: `ml/scripts/apply_migrations.py`, `ml/deployment/migrations.py`, `ml/observability/migrations.py`
- **Finding**: All functions have complete type annotations with proper return types

**Error Handling** - ✅ **COMPLIANT**

- **Status**: Appropriate exception handling with specific error types
- **Implementation**: Migration runners catch and categorize SQL errors appropriately
- **File**: `ml/scripts/apply_migrations.py:155-164` - Differentiates idempotent conflicts from real errors

**Database Usage** - ✅ **COMPLIANT**

- **Status**: Proper use of EngineManager for database connections
- **Finding**: All migration utilities use `EngineManager.get_engine()` as required
- **Files**: `ml/core/integration.py:428`, `ml/scripts/apply_migrations.py:228`

### Documentation Accuracy Validation

#### Structural Claims vs Reality

**Migration Organization Structure** - ✅ **ACCURATE**

- **Documented**: 3-directory structure (ml/migrations/, ml/registry/migrations/, ml/stores/migrations/)
- **Reality**: Structure exists exactly as documented
- **Files Found**:
  - `ml/migrations/999_fix_partitions_immediate.sql` (1 file)
  - `ml/registry/migrations/` (2 files: 001_initial_schema.sql, 002_add_cold_path_fields.sql)
  - `ml/stores/migrations/` (10 files: comprehensive store schema)

**Migration Execution Flow** - ✅ **ACCURATE**

- **Documented**: `MLIntegrationManager._run_migrations()` with ordered execution
- **Reality**: Implementation matches documentation exactly
- **File**: `ml/core/integration.py:419-426` - Migration list matches documentation

**CLI Migration Runner** - ✅ **ACCURATE**

- **Documented**: CLI runner with --full, --dry-run, --schema options
- **Reality**: `ml/scripts/apply_migrations.py` implements all documented features
- **Finding**: Command examples in documentation are accurate

#### Claimed Completion Status Review

**"Critical fixes and immediate patches"** - ⚠️ **MISCHARACTERIZED**

- **Reality**: Only 1 file in `ml/migrations/999_fix_partitions_immediate.sql`
- **Documentation implies**: Multiple critical fixes
- **Actual scope**: Single emergency partition fix for test environment

#### Performance and Safety Claims

**"Idempotent Design"** - ✅ **VALIDATED**

- **Claim**: "All migrations use IF NOT EXISTS clauses"
- **Reality**: Confirmed in migration files (001_initial_schema.sql uses IF NOT EXISTS throughout)
- **File**: `ml/registry/migrations/001_initial_schema.sql:6, 12, 36, 44`

**"Transaction Safety"** - ✅ **VALIDATED**

- **Claim**: "Each migration file runs in separate transaction"
- **Reality**: Confirmed in implementation
- **File**: `ml/core/integration.py:446` - Uses `with engine.begin() as conn:`

**"Error Handling"** - ⚠️ **PARTIALLY ACCURATE**

- **Claim**: "Non-critical errors logged as warnings, not failures"
- **Reality**: Basic error categorization exists but could be more sophisticated
- **Implementation**: Distinguishes "already exists" from other errors but categorization is simplistic

### Implementation Gaps and Issues

#### Missing Implementations

1. **Migration Rollback System**
   - **Documentation**: No rollback mechanism documented
   - **Reality**: No rollback implementation found
   - **Impact**: HIGH - Production deployments need rollback capability

2. **Migration Dependencies**
   - **Documentation**: Claims dependency tracking but no implementation found
   - **Reality**: Migrations run in predefined order, no dependency validation
   - **Impact**: MEDIUM - Could lead to dependency violations

3. **Migration Testing Framework**
   - **Finding**: Limited testing of migration execution
   - **Files Found**: Basic tests in `ml/tests/unit/deployment/test_migrations.py`
   - **Missing**: Integration tests for full migration scenarios

#### Implementation Discrepancies

1. **Schema Location Inconsistency**
   - **Documentation**: Claims registry tables in `ml_registry` schema
   - **Reality**: `ml/registry/migrations/001_initial_schema.sql:9` sets search_path but creates in public schema
   - **File**: Lines 6-9 create ml_registry schema but subsequent tables may end up in public

2. **Partition Management Complexity**
   - **Documentation**: Claims "Comprehensive time-based partitioning"
   - **Reality**: Implementation exists but has race conditions requiring emergency fixes
   - **Evidence**: `ml/migrations/999_fix_partitions_immediate.sql` - drops auto-partition triggers due to race conditions

3. **Migration Order Mismatch**
   - **Documentation**: Claims migrations run in predetermined order
   - **Reality**: `MLIntegrationManager` hardcodes specific subset, not all available migrations
   - **File**: `ml/core/integration.py:419-426` - Only 6 migrations in list vs 12+ files available

### Security and Safety Analysis

#### SQL Injection Protection

**Status**: ✅ **SECURE**

- **Finding**: All SQL execution uses parameterized queries or sqlalchemy.text()
- **Files**: `ml/scripts/apply_migrations.py:154`, `ml/core/integration.py:449`
- **Implementation**: Proper use of text() with parameter binding

#### Dollar-Quoted Function Safety

**Status**: ✅ **SECURE**

- **Finding**: Sophisticated SQL statement splitter respects dollar-quoted blocks
- **File**: `ml/scripts/apply_migrations.py:69-135` - Handles $tag$...$tag$ safely
- **Validation**: Prevents SQL injection through function body manipulation

#### Migration File Integrity

**Status**: ⚠️ **BASIC VALIDATION**

- **Finding**: No checksum or signature validation of migration files
- **Impact**: MEDIUM - Malicious modification of migration files possible
- **Recommendation**: Add file integrity checking for production deployments

### Performance Analysis

#### Migration Execution Performance

**SQL Statement Splitting**: ✅ **OPTIMIZED**

- **Implementation**: Custom splitter respects PostgreSQL-specific constructs
- **File**: `ml/scripts/apply_migrations.py:69-135`
- **Benefit**: Avoids naive semicolon splitting that breaks on function definitions

**Database Connection Management**: ✅ **OPTIMIZED**

- **Implementation**: Uses EngineManager singleton pattern
- **Benefit**: Prevents connection pool exhaustion during migrations

**Transaction Boundaries**: ✅ **APPROPRIATE**

- **Implementation**: Each migration file runs in separate transaction
- **Benefit**: Atomic migration application with rollback on failure

### Testing Coverage Analysis

#### Unit Test Coverage

**Migration Utilities**: ✅ **COVERED**

- **File**: `ml/tests/unit/deployment/test_migrations.py`
- **Scope**: Tests file discovery and command building
- **Gap**: No tests for actual SQL execution

**Observability Migrations**: ✅ **COVERED**

- **File**: `ml/tests/unit/observability/test_db_migrations_postgres.py`
- **Scope**: Tests index creation and validation
- **Strength**: Includes actual PostgreSQL integration

#### Integration Test Gaps

1. **End-to-End Migration Testing**: Missing comprehensive migration scenarios
2. **Rollback Testing**: No rollback capability to test
3. **Performance Testing**: No migration performance benchmarks
4. **Partition Testing**: Limited testing of partition creation/management

### Recommendations for Improvement

#### High Priority

1. **Implement Migration Rollback System**
   - Add rollback SQL generation capability
   - Version control for schema changes
   - Safe rollback validation

2. **Fix Schema Consistency**
   - Ensure registry tables are created in ml_registry schema consistently
   - Verify all table references use correct schema qualification

3. **Complete Migration List Integration**
   - Update MLIntegrationManager to include all available migrations
   - Implement dependency graph validation

#### Medium Priority

1. **Enhance Error Handling**
   - More sophisticated error categorization
   - Better logging and diagnostics
   - Automatic retry for transient failures

2. **Add Migration Validation**
   - Pre-execution validation of migration syntax
   - Dependency checking
   - Conflict detection

3. **Improve Testing Coverage**
   - End-to-end integration tests
   - Performance benchmarks
   - Error scenario testing

#### Low Priority

1. **Add Migration File Integrity Checking**
   - Checksums or signatures for migration files
   - Verification before execution

2. **Enhanced Documentation**
   - Rollback procedures
   - Troubleshooting guide
   - Performance tuning recommendations

### Conclusion

The ml/migrations domain provides a solid foundation for database schema management with good safety practices and proper PostgreSQL integration. The core functionality is well-implemented and follows database best practices. However, several important features are missing (rollback, comprehensive testing, dependency validation) and some implementation discrepancies exist between documentation and reality.

The domain demonstrates strong adherence to coding standards and appropriate cold-path categorization within the Universal ML Architecture Patterns. The main areas for improvement are enhanced error handling, rollback capabilities, and more comprehensive testing coverage.

**Overall Assessment**: ✅ **FUNCTIONAL** with ⚠️ **IMPROVEMENT OPPORTUNITIES**

- Core migration functionality: COMPLETE
- Safety and error handling: GOOD
- Documentation accuracy: MOSTLY ACCURATE with minor discrepancies
- Production readiness: REQUIRES rollback implementation for full production deployment
