# Pipeline Orchestration Context

**Last Updated:** 2025-11-19
**Target:** ml/orchestration/
**Status:** Facade-only (legacy removed)

---

## Executive Summary

The `ml/orchestration/` directory implements cold-path ML pipeline orchestration. It manages the complete lifecycle: ingestion → dataset building → training → promotion.

**Current State:**
- Component facade is canonical (legacy orchestrator removed; no feature flags).
- CLI glue lives in `pipeline_orchestrator_cli.py`, facade wiring in `pipeline_orchestrator_facade.py`.
- Ingestion, dataset, training, and promotion flow through component stack.

---

## File Inventory

```
ml/orchestration/
├── __init__.py                         106 lines   # Public API
├── pipeline_orchestrator_cli.py      2,214 lines   # CLI glue + orchestration helpers
├── pipeline_orchestrator_facade.py     330 lines   # Facade wiring (component stack)
│
├── config_types.py                     235 lines   # Typed config dataclasses
├── config_loader.py                    680 lines   # TOML/JSON loader
├── config_resolver.py                  707 lines   # Config resolution component
│
├── discovery_client.py                 468 lines   # Dataset discovery
├── binding_resolver.py                 616 lines   # Market binding resolution
├── ingestion_coordinator.py          1,188 lines   # Backfill coordination
├── dataset_builder.py                1,128 lines   # Dataset construction
│
├── stage2_engine.py                    608 lines   # Promotion validation
├── promotions.py                       757 lines   # Model promotion helpers
└── scheduler.py                        362 lines   # Pipeline scheduler

Total: ~10,900 lines
```

---

## Architecture: Facade Composition

```
MLPipelineOrchestratorFacade
├── ConfigResolver          # Market inputs, window bounds, symbol maps
├── DiscoveryClient         # Dataset discovery via ingestion service
├── BindingResolver         # Market binding + coverage validation
├── IngestionCoordinator    # Backfill, auto-fill, pre-ingestion
└── DatasetBuilder          # Dataset construction + validation
```

---

## Core Components

### 1. ConfigResolver

**File:** `ml/orchestration/config_resolver.py`
**Protocol:** `ConfigResolverProtocol` (lines 39-162)

**Purpose:** Configuration resolution, symbol mapping, window bounds calculation.

**Key Methods:**
- `apply_default_market_inputs(cfg)` → DatasetBuildConfig
  Seeds market inputs from descriptor files (lines 275-335)

- `collect_symbol_map(ds_cfg, symbols, instruments, instrument_ids, market_inputs)` → dict[str, tuple[str, ...]]
  Unifies symbol → instrument_id mappings from multiple sources (lines 337-434)

- `resolve_window_bounds_ns(cfg)` → tuple[int, int]
  Converts ISO dates to nanosecond timestamps (lines 467-509)

- `prepare_dataset_config(cfg, resolved_inputs, bindings)` → DatasetBuildConfig
  Merges resolved market inputs back into config (lines 512-559)

**Example:**
```python
from ml.orchestration import ConfigResolver

resolver = ConfigResolver()
cfg = resolver.apply_default_market_inputs(dataset_config)
symbol_map = resolver.collect_symbol_map(ds_cfg, symbols=("SPY",), ...)
start_ns, end_ns = resolver.resolve_window_bounds_ns(cfg)
```

---

### 2. DiscoveryClient

**File:** `ml/orchestration/discovery_client.py`
**Protocol:** `DiscoveryClientProtocol` (lines 40-125)

**Purpose:** Dataset discovery, service health checks.

**Key Methods:**
- `discover_market_inputs(symbol_map, schema, start_ns, end_ns, dataset_hint)` → tuple[MarketDatasetInput, ...]
  Discovers available datasets via ingestion service (lines 160-246)

- `discover_binding_for_symbol(symbol, instrument_ids, schema, start_ns, end_ns)` → ResolvedMarketBinding | None
  Finds binding for specific symbol (lines 248-297)

**Metrics:**
- `ml_discovery_operations_total{status="success|error"}`
- `ml_discovery_latency_seconds`

**Integration:** Depends on `DatasetDiscoveryService` from `ml.data.ingest.discovery`.

---

### 3. BindingResolver

**File:** `ml/orchestration/binding_resolver.py`
**Protocol:** `BindingResolverProtocol` (lines 40-151)

**Purpose:** Market binding resolution with coverage validation.

**Key Methods:**
- `resolve_market_inputs(cfg, symbol_map, start_ns, end_ns)` → tuple[tuple[MarketDatasetInput, ...] | None, tuple[ResolvedMarketBinding, ...]]
  Full resolution pipeline (lines 199-333)

- `filter_candidate_bindings(candidates, start_ns, end_ns, symbol, default_schema)` → tuple[ResolvedMarketBinding, ...]
  Filters by storage availability and schema (lines 335-385)

- `select_binding_with_coverage(candidates, start_ns, end_ns, symbol)` → ResolvedMarketBinding | None
  Priority selection: SQL > Parquet > Catalog (lines 387-448)

**Priority Logic (lines 451-474):**
```python
def _binding_priority_key(binding: ResolvedMarketBinding) -> tuple[int, str]:
    """Lower number = higher priority."""
    storage_priority = {
        StorageKind.SQL: 0,
        StorageKind.PARQUET: 1,
        StorageKind.CATALOG: 2,
    }
    return (storage_priority.get(binding.storage, 99), binding.dataset_id)
```

**Metrics:**
- `ml_binding_resolutions_total{status="success|error"}`
- `ml_binding_resolution_latency_seconds`

---

### 4. IngestionCoordinator

**File:** `ml/orchestration/ingestion_coordinator.py`
**Protocol:** `IngestionCoordinatorProtocol` (lines 92-206)

**Purpose:** Backfill management, auto-fill universe, pre-ingestion.

**Key Methods:**
- `run_pre_ingestion(catalog_path, scheduler_cfg, options)` → None
  Executes pre-ingestion tasks from scheduler (lines 244-313)

- `backfill(dataset_id, schema, instrument_id, lookback_days)` → BackfillWindowList
  Single-instrument backfill (lines 315-375)

- `backfill_binding(binding, lookback_days, symbol)` → BackfillWindowList
  Backfill via resolved binding (lines 377-425)

- `backfill_coverage(cfg, start_ns, end_ns, bindings)` → None
  Multi-binding coverage backfill (lines 427-518)

- `auto_fill_universe(cfg)` → None
  Populates universe from catalog (lines 520-784)

**Auto-Fill Flow (lines 615-784):**
```python
def auto_fill_universe(self, cfg: AutoFillUniverseConfig) -> None:
    # 1. Discover catalog datasets
    manifest = build_auto_dataset_manifest(cfg.dataset_id, cfg.include_bars, ...)

    # 2. Filter by instrument_ids if specified
    if cfg.instrument_ids:
        filtered_instruments = set(cfg.instrument_ids) & catalog_instruments

    # 3. Backfill each schema (bars, tbbo, trades, l2, l3)
    for schema in schemas:
        self._ingest_schema_for_universe(schema, instruments, cfg)

    # 4. Emit metrics
    self._metrics.operations_total.labels(schema=schema, status="success").inc()
```

**L2 Integration (lines 785-896):**
Efficient L2 data population via `populate_l2_efficient` task.

**Metrics:**
- `ml_auto_fill_operations_total{schema, status}`
- `ml_auto_fill_latency_seconds{schema}`

---

### 5. DatasetBuilder

**File:** `ml/orchestration/dataset_builder.py`
**Protocol:** `DatasetBuilderProtocol` (lines 84-132)

**Purpose:** Dataset construction, validation, metadata management.

**Key Methods:**
- `build_dataset(cfg)` → int
  Builds dataset from config (lines 185-465)

- `validate_dataset(dataset_path, expectations, validation_config)` → tuple[bool, DatasetMetadata | None]
  **NOT IMPLEMENTED** (line 131: `raise NotImplementedError`)

**Build Flow (lines 185-465):**
```python
def build_dataset(self, cfg: DatasetBuildConfig) -> int:
    # 1. Delegate to CLI builder (tft_dataset_builder)
    from ml.pipelines.build_runner import main as build_main
    args = self._build_cli_args(cfg)
    exit_code = build_main(args)

    # 2. Load metadata
    metadata = load_dataset_metadata(out_dir / "dataset_metadata.json")

    # 3. Convert vintage→age if requested
    if cfg.convert_vintage_to_age:
        convert_vintage_timestamps_to_age(dataset_csv, metadata_path, ...)

    # 4. Validate (if enabled)
    if cfg.validation:
        validate_dataset_metadata_expectations(metadata, expectations)

    return exit_code
```

Metadata helpers for this flow live in `ml/data/metadata.py`, and the macro/vintage
refresh + validation utilities are centralized in `ml/data/common/macro_vintage.py`.

**Build Artifacts (lines 52-61):**
```python
@dataclass(slots=True, frozen=True)
class BuildArtifacts:
    out_dir: Path
    feature_set_id: str | None = None
    feature_names: tuple[str, ...] = ()
    feature_registry_dir: str | None = None
    dataset_metadata: DatasetMetadata | None = None
```

**Gap:** `validate_dataset` method is defined in protocol but not implemented (raises `NotImplementedError`).

---

## Configuration System

### Config Types

**File:** `ml/orchestration/config_types.py`

**Key Dataclasses:**
- `DatasetBuildConfig` (lines 46-89): Dataset construction params
- `AutoFillUniverseConfig` (lines 91-113): Catalog auto-fill
- `HPOConfig` (lines 115-134): Hyperparameter optimization
- `TeacherTrainConfig` (lines 136-147): Teacher model training
- `StudentDistillConfig` (lines 149-166): Student distillation
- `IntegrationConfig` (lines 168-181): MLIntegrationManager attachment
- `PreIngestionOptions` (lines 183-193): Pre-ingestion scheduler stage
- `PromotionsConfig` (lines 195-208): Model promotion config
- `OrchestratorConfig` (lines 210-225): Composite config consumed by legacy runner

**All configs use `@dataclass(slots=True, frozen=True)` for immutability.**

### Config Loader

**File:** `ml/orchestration/config_loader.py`

**Stages (lines 61-70):**
```python
class Stage(StrEnum):
    INGEST = "ingest"      # Backfill only
    DATASET = "dataset"    # Backfill + dataset build
    TRAIN = "train"        # Training only (dataset must exist)
    FULL = "full"          # Ingest → dataset → train → promote
```

**Config Loading (lines 240-290):**
```python
def load_orchestrator_run_config(
    path: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> OrchestratorRunConfig:
    """Load TOML or JSON orchestrator config with env overrides."""
    suffix = path.suffix.lower()

    if suffix == ".toml":
        import tomllib
        data = tomllib.loads(path.read_text())
    elif suffix == ".json":
        data = json.loads(path.read_text())
    else:
        raise ValueError(f"Unsupported config format: {suffix}")

    # Apply environment overrides
    if env:
        data = _apply_env_overrides(data, env)

    return _build_run_config_from_dict(data)
```

**TOML Example:** `ml/config/orchestrator/production_full.toml`

---

## Promotion System

### Stage 2 Engines

**File:** `ml/orchestration/stage2_engine.py`

**Purpose:** Compute trading metrics for promotion gate validation (cold-path only).

**Engines:**
1. **ReturnsStage2Engine** (lines 99-285):
   Computes realized forward returns from catalog bars aligned to teacher validation tail.

2. **BacktestStage2EngineRunner** (lines 287-608):
   Advisory hook for Nautilus Trader BacktestEngine. Falls back to returns engine if backtest stack unavailable.

**Result Type (lines 36-40):**
```python
@dataclass(slots=True, frozen=True)
class Stage2Result:
    status: Literal["passed", "failed", "skipped"]
    metrics: dict[str, float]
    reason: str | None = None
```

**Protocol (lines 42-43):**
```python
class Stage2Engine(Protocol):
    def run(self, cfg: Stage2Config) -> Stage2Result: ...
```

### Promotion Helpers

**File:** `ml/orchestration/promotions.py`

**Key Functions:**
- `register_and_promote_model(...)` (lines 200-420):
  Registers model with ModelRegistry and promotes if gates pass.

- `register_or_refresh_features(...)` (lines 422-600):
  Registers feature set with FeatureRegistry or refreshes existing.

**Integration:** Uses `ModelRegistry`, `FeatureRegistry`, and `emit_dataset_event` from `ml.common.event_emitter`.

---

## Scheduler

**File:** `ml/orchestration/scheduler.py`

**Purpose:** Time-based or interval-based pipeline scheduling (cold-path).

**Key Functions:**
- `compute_next_run(schedule_time, interval_min, now)` → datetime
  Computes next run from daily UTC schedule or interval (lines 93-152)

- `run_forever(config_path, ...)` → None
  Infinite scheduling loop with event emission (lines 154-362)

**Scheduling Logic (lines 93-152):**
```python
def compute_next_run(
    schedule_time: str | None,      # e.g., "02:00" (UTC)
    interval_min: int | None,        # e.g., 60 (minutes)
    now: datetime,
) -> datetime:
    """
    Compute next run time from a daily UTC schedule or interval.

    Precedence:
    1. schedule_time (daily UTC)
    2. interval_min (periodic)
    3. Default: 1 hour from now
    """
    if schedule_time:
        # Parse HH:MM and compute next occurrence
        return next_scheduled_time(schedule_time, now)
    elif interval_min:
        return now + timedelta(minutes=interval_min)
    else:
        return now + timedelta(hours=1)
```

**Metrics:**
- `nautilus_ml_orch_runs_total{status}`
- `nautilus_ml_orch_phase_latency_seconds{phase}`

---

## Integration Points

### CLI Entry Point

**File:** `ml/cli/pipeline_orchestrator.py`

```python
from ml.orchestration.pipeline_orchestrator_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

**Invocation:**
```bash
python -m ml.cli.pipeline_orchestrator \
    --config ml/config/orchestrator/production_full.toml \
    --stage full
```

### Tests

**Key Test Files:**
- `ml/tests/e2e/test_pipeline_orchestrator_e2e.py` - E2E tests for facade-only orchestration
- `ml/tests/integration/orchestration/test_ml_pipeline_orchestrator_facade.py` - Facade compatibility tests
- `ml/tests/unit/orchestration/test_config_resolver.py` - ConfigResolver unit tests
- `ml/tests/unit/orchestration/test_binding_resolver.py` - BindingResolver unit tests
- `ml/tests/unit/orchestration/test_discovery_client.py` - DiscoveryClient unit tests

### Dashboard Integration

**File:** `ml/dashboard/services/pipelines_service.py`

Uses `load_orchestrator_config` to display pipeline status and configuration in dashboard.

---

## Known Gaps & Migration Status

### ✅ Migrated to Components

1. **Configuration Resolution** - ConfigResolver (100% complete)
2. **Dataset Discovery** - DiscoveryClient (100% complete)
3. **Market Binding Resolution** - BindingResolver (100% complete)
4. **Ingestion Coordination** - IngestionCoordinator (100% complete)
5. **Dataset Building** - DatasetBuilder (90% complete, validation method stub)

### ⚠️ Follow-ups

1. Validate HPO/teacher/student flows end-to-end under facade-only orchestrator.
2. Keep parity coverage focused on training + promotions for regressions.

**Reason:** Training-related operations are deferred to Phase 2+ of the strangler fig migration.

### 🐛 Incomplete Implementations

1. **DatasetBuilder.validate_dataset** (line 131):
   ```python
   def validate_dataset(...) -> tuple[bool, DatasetMetadata | None]:
       raise NotImplementedError
   ```
   Protocol defines method but implementation missing. Dataset validation currently happens inline during `build_dataset`.

2. **Facade-only**: Component implementation is the only path; legacy switches are removed.

3. **Error Recovery**: Circuit breakers and backpressure skeletons exist but not fully implemented for all failure modes.

4. **Monitoring Gaps**: Some cold-path operations lack comprehensive telemetry.

---

## Code Patterns

### Protocol-First Design

**Every component defines a Protocol:**
```python
from typing import Protocol

class ConfigResolverProtocol(Protocol):
    def apply_default_market_inputs(self, cfg: DatasetBuildConfig) -> DatasetBuildConfig: ...
    def collect_symbol_map(...) -> dict[str, tuple[str, ...]]: ...
```

**Benefits:**
- Structural typing without implementation coupling
- Duck typing support for testing (DummyStore conforms)
- Type safety without circular dependencies

### Frozen Dataclasses for Configs

**All config types use `@dataclass(slots=True, frozen=True)`:**
```python
@dataclass(slots=True, frozen=True)
class DatasetBuildConfig:
    data_dir: str
    symbols: str
    out_dir: str
    dataset_id: str = "tft_dataset"
    # ...
```

**Validation in `__post_init__`:**
```python
def __post_init__(self) -> None:
    if self.lookback_window < 1:
        raise ValueError("lookback_window must be >= 1")
```

### Centralized Metrics Bootstrap

**Never import `prometheus_client` directly:**
```python
from ml.common.metrics_bootstrap import get_counter, get_histogram

counter = get_counter(
    "ml_discovery_operations_total",
    "Discovery operations by status",
    labelnames=("status",),
)
```

### Progressive Fallback Chains

**Pattern:** PRIMARY → CACHED → FILE → DUMMY

**Example from IngestionCoordinator (lines 315-375):**
```python
def backfill(self, dataset_id, schema, instrument_id, lookback_days):
    try:
        # PRIMARY: Use ingestion orchestrator
        return self._orchestrator.backfill(...)
    except Exception as exc:
        logger.warning("Backfill failed, attempting fallback", exc_info=True)
        # FALLBACK: Use ingestor directly
        return self._ingestor.backfill_single(...)
```

---

## Developer Guidelines

### Orchestrator Implementation

- Component facade is the canonical path for orchestration.
- Legacy pipeline switches are removed; use facade components for all changes.

### Adding New Configuration Options

1. Add field to appropriate config dataclass in `config_types.py`
2. Update TOML schema examples in `ml/config/orchestrator/`
3. Update `config_loader.py` if new top-level section
4. Add validation in `__post_init__` if needed
5. Update tests in `ml/tests/unit/orchestration/test_config_loader.py`

### Extending Components

**Example: Adding new DiscoveryClient method:**

1. Update protocol (`discovery_client.py` lines 40-125):
   ```python
   class DiscoveryClientProtocol(Protocol):
       def new_discovery_method(self, ...) -> ...: ...
   ```

2. Implement in DiscoveryClient class (lines 127+):
   ```python
   def new_discovery_method(self, ...) -> ...:
       # Implementation
   ```

3. Add corresponding method to legacy orchestrator if needed for backward compat

4. Write unit tests in `ml/tests/unit/orchestration/test_discovery_client.py`

### Testing Best Practices

**Unit Tests:**
- Test each component in isolation with mocked dependencies
- Use `pytest.mark.serial` for database-heavy tests
- Verify metrics are emitted correctly

**Integration Tests:**
- Test component interactions via facade
- Validate orchestration flows remain backward-compatible at the API level

**E2E Tests:**
- Full pipeline runs from config → artifacts
- Validate dataset metadata, feature registration, model promotion

**Property Tests:**
- Window bounds calculations (monotonicity)
- Symbol map merging (associativity)
- Priority selection (determinism)

---

## Quick Reference

### Common Imports

```python
# Components
from ml.orchestration import (
    ConfigResolver,
    DiscoveryClient,
    BindingResolver,
    IngestionCoordinator,
    DatasetBuilder,
)

# Protocols
from ml.orchestration import (
    ConfigResolverProtocol,
    DiscoveryClientProtocol,
    BindingResolverProtocol,
    IngestionCoordinatorProtocol,
    DatasetBuilderProtocol,
)

# Config Types
from ml.orchestration import (
    DatasetBuildConfig,
    AutoFillUniverseConfig,
    HPOConfig,
    TeacherTrainConfig,
    StudentDistillConfig,
    OrchestratorConfig,
)

# Config Loading
from ml.orchestration import (
    load_orchestrator_config,
    load_orchestrator_run_config,
    Stage,
)

# Scheduling
from ml.orchestration import compute_next_run, run_forever

# Promotions
from ml.orchestration import (
    register_and_promote_model,
    register_or_refresh_features,
)
```

### Running Pipeline

```bash
# Via CLI
python -m ml.cli.pipeline_orchestrator \
    --config ml/config/orchestrator/production_full.toml \
    --stage full

# Via scheduler
python -m ml.cli.pipeline_scheduler \
    --config ml/config/orchestrator/production_full.toml \
    --schedule-time "02:00"
```

---

## Related Documentation

- `ml/docs/context/context_data.md` - Data ingestion and storage
- `ml/docs/context/context_registry.md` - Feature and model registries
- `ml/docs/context/context_training.md` - Training pipelines
- `ml/docs/context/context_cli.md` - CLI tooling
- `ml/docs/ROADMAP.md` - Overall ML system roadmap
