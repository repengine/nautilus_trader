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
    ├── 001_stores_schema.sql            # Core ML data tables with partitioning
    ├── 002_auto_partitioning.sql       # Automatic partition management
    ├── 003_market_data.sql              # Market data table extensions
    ├── 004_data_registry.sql           # Data registry and event tracking
    ├── 005_schema_hardening.sql        # Data integrity improvements
    ├── 005a_feature_values_dedupe.sql  # Deduplication constraints
    ├── 005_views.sql                    # Analytical views and summaries
    ├── 006_disable_partition_triggers.sql # Testing optimizations
    ├── 007_add_event_metadata.sql      # Event metadata extensions
    └── 007_brin_indexes.sql             # Performance optimizations
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
    "ml/stores/migrations/001_stores_schema.sql", 
    "ml/stores/migrations/002_auto_partitioning.sql",
    "ml/stores/migrations/003_market_data.sql",
    "ml/stores/migrations/004_data_registry.sql",
    "ml/stores/migrations/007_add_event_metadata.sql",
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

### Store Schema (`001_stores_schema.sql`)

**Partitioned Tables** (by `ts_event` in nanoseconds):
- `ml_feature_values`: Feature computation results with metadata
- `ml_model_predictions`: Model inference outputs with timing
- `ml_strategy_signals`: Strategy signals with risk metrics

**Partition Strategy**:
- Monthly partitions from 2024-2026 (36 months coverage)
- Helper function `create_monthly_partitions()` for consistent creation
- Nanosecond timestamp boundaries aligned with Nautilus conventions

### Automatic Partitioning (`002_auto_partitioning.sql`)

**Automated Management Functions**:
- `auto_create_partitions()`: Creates future partitions (3 months ahead)
- `auto_cleanup_partitions()`: Removes partitions beyond retention period
- `ensure_partition_exists()`: Trigger-based partition creation (now disabled)

**Safety Mechanisms**:
- Race condition avoidance through trigger disabling (migration 006)
- Pre-created test partitions for common timestamp ranges
- Idempotent partition creation with conflict handling

### Data Registry (`004_data_registry.sql`)

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

### Schema Hardening (`005_schema_hardening.sql`)

**Data Integrity Improvements**:
- Unique constraints for ML feature values upsert operations
- Timestamp type standardization to `TIMESTAMPTZ`
- Automatic nanosecond-to-timestamp conversion for analytics compatibility

### Partition Management Fixes

**Emergency Fixes** (`999_fix_partitions_immediate.sql`, `006_disable_partition_triggers.sql`):
- Disables race-prone automatic partition triggers
- Pre-creates test partitions for 2023-2024 timestamp ranges
- Ensures critical test partitions exist for common test timestamps

## Dependencies

### Internal Dependencies
- `ml.core.db_engine.EngineManager`: Database connection management
- `ml.stores.partition_manager.PartitionManager`: Runtime partition management
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