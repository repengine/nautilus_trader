# ML Tasks & Pipelines Context

## Overview

Canonical domain modules under `ml/<domain>/` and orchestration modules under `ml/orchestration/` provide cold-path ML workflow orchestration. CLI entry points import typed services from these canonical owners. No hot-path code should import orchestration modules.

### Key Distinction
- **ml/<domain>/**: Function-based helpers (config + business logic). Consumed by CLI entry points in `ml/cli/`.
- **ml/pipelines/**: Higher-level pipeline composition (dataset → training → distillation). Thin wrappers over orchestration.
- **ml.data**: Underlying data processing APIs that tasks wrap for CLI integration.

**Architecture Principle**: Tasks provide a **configuration façade** over `ml.data` APIs, converting CLI arguments into typed configs and managing result presentation.

---

## Directory Structure

### Canonical Domain Services (Cold-Path)

```
ml/<domain>/
├── __init__.py                 # Lazy module loader (41 lines)
├── registry.py                 # Feature registration & promotion (97 lines)
├── db.py                        # Database migrations (346 lines)
├── datasets/
│   ├── __init__.py             # Re-exports TFT, Report; optional ProductionDataset
│   ├── tft.py                  # TFTDatasetTaskConfig & build_tft_dataset (137 lines)
│   ├── report.py               # Dataset reporting (200+ lines)
│   ├── splits.py               # Cross-validation splits
│   └── production.py           # Optional production dataset builder
├── ingest/
│   ├── __init__.py             # Re-exports all ingest tasks (37 lines)
│   ├── recent.py               # Recent OHLCV backfill (68 lines)
│   ├── backfill.py             # Full backfill runner
│   ├── l2.py                   # L2 book population
│   ├── supplementary.py        # Macro data ingestion
│   ├── alternative.py          # Alternative data sources
│   └── yahoo.py                # Yahoo Finance loader
├── training/
│   ├── __init__.py             # Re-exports QuickTFTTrain*
│   └── quick.py                # QuickTFTTrainConfig & train_tft_quick (210 lines)
├── monitoring/
│   ├── __init__.py             # Re-exports health & coverage
│   ├── health.py               # Health check queries (150+ lines)
│   └── coverage.py             # Data coverage analysis
├── observability/
│   ├── __init__.py             # Re-exports flush & backfill
│   ├── flush.py                # JSONL/CSV export for metrics & traces
│   └── backfill.py             # Observability backfill tasks
├── dev/
│   ├── __init__.py
│   └── sanity_check.py         # Development sanity checks
└── pipelines/
    ├── __init__.py             # Re-exports runner & scheduler
    ├── runner.py               # MLPipelineRunner & PipelineRunConfig (325 lines)
    └── scheduler.py            # PipelineScheduleConfig & run_pipeline_schedule (65 lines)
```

**Total**: ~3,697 lines across datasets/, monitoring/, and observability/ subdirectories.

### ml/pipelines/ (High-Level Composition)

```
ml/pipelines/
├── __init__.py                 # Public API exports with extensive docstrings (172 lines)
├── build_runner.py             # BuildConfig, BuildTask, BuildWindow (476 lines)
└── tft_train_distill.py        # Compatibility wrapper → orchestrator_main (150 lines)
```

**Purpose**: `ml/pipelines/` provides **multi-step orchestration** that composes CLI and API components into complete workflows (e.g., dataset build → teacher train → student distill).

---

## Key Modules

### Registry Tasks (ml/registry/feature_operations.py)

**Purpose**: Feature promotion, registration, deprecation.

**Key exports**:
- `FeaturePromotionGate`: Typed validation gate (metric_name, threshold, comparison)
- `register_default_feature_set()`: Register default FeatureConfig manifest
- `promote_feature_set()`: Validate + promote via quality gates
- `deprecate_feature_set()`: Mark feature set deprecated

**Integration**: Used by `ml/cli/feature_cli.py` and feature promotion workflows.

**Code pattern** (lines 40-59):
```python
def register_default_feature_set(...) -> str:
    engineer = FeatureEngineer(FeatureConfig())
    manifest = engineer.generate_feature_manifest(...)
    registry = FeatureRegistry(registry_path)
    return registry.register_feature_set(manifest)
```

### Database Tasks (ml/stores/migrations_runner.py)

**Purpose**: SQL migration planning & execution. Canonical baseline consolidated 2025-10-01 from 18 fragmented migrations.

**Key exports**:
- `MigrationSchema`: Enum (STORES, REGISTRY, BOTH)
- `MigrationPlan`: Immutable (files: tuple[Path, ...])
- `MigrationResult`: Mutable result state (applied, skipped, warnings, errors)
- `build_migration_plan()`: Filter base + optional migrations by schema
- `apply_migration_files()`: Execute via SQLAlchemy with idempotent error handling
- `split_sql_statements()`: Parse SQL respecting dollar-quoted & string literals (lines 144-201)

**Base migrations** (lines 37-42):
```python
_BASE_MIGRATIONS = (
    "ml/registry/migrations/001_initial_schema.sql",
    "ml/registry/migrations/002_add_cold_path_fields.sql",
    "ml/registry/migrations/003_add_artifact_digest.sql",
    "ml/stores/migrations/001_bootstrap_schema.sql",
)
```

**Idempotent error handling** (lines 29-34):
```python
IDEMPOTENT_ERROR_PHRASES = (
    "already exists",
    "does not exist",
    "duplicate key",
    "is not partitioned",
)
```

**Used by**: `ml/cli/apply_migrations.py` and deployment scripts.

### Dataset Building (ml/data/)

**TFT Dataset Config** (ml/data/build.py):
- **`TFTDatasetTaskConfig`**: 30-field frozen dataclass (lines 29-69)
  - Core params: `horizon_minutes`, `threshold`, `lookback_periods`
  - Feature toggles: `include_macro/micro/l2/events/calendar/earnings`
  - Registry integration: `register_features`, `feature_registry_dir`, `feature_role`
  - Vintage control: `vintage_policy`, `vintage_as_of`, `convert_vintage_to_age`
  - Validation: `validation: DatasetValidationConfig | None`

- **`build_tft_dataset()`** (lines 72-129):
  - Wraps `ml.data.DatasetBuildConfig` + calls internal `_build_tft_dataset()`
  - Supports vintage age conversion via `ml.preprocessing.vintage_age`
  - Returns `BuildResult` from `ml.data`

**Key integration pattern** (lines 76-111):
```python
dataset_cfg = DatasetBuildConfig(
    data_dir=cfg.data_dir,
    out_dir=cfg.out_dir,
    symbols=[symbol.upper() for symbol in cfg.symbols],
    # ... map all 30 fields from TFTDatasetTaskConfig → DatasetBuildConfig
)
result = _build_tft_dataset(dataset_cfg)
```

**Dataset Report** (ml/data/validation.py):
- `DatasetReportConfig`: Report generation config (lines 25-29)
- `DatasetReport`: Mutable result with JSON/markdown outputs (lines 32-38)
- `generate_dataset_report()`: Analyze dataset statistics (macro columns, null rates, feature counts)
- **Macro column inference** (lines 41-69): Pattern matching for known FRED series (DGS1, FEDFUNDS, VIXCLS, etc.)

**Integration**:
- `ml/cli/build_tft_dataset.py` imports `TFTDatasetTaskConfig` & `build_tft_dataset`
- `ml/cli/dataset_report.py` uses reporting
- CLI converts argparse args → task config → calls task function

### Data Ingestion (ml/data/ingest/)

**Coordinated ingestion helpers** wrapping `ml.data.loaders.*` and `ml.data.ingest.*`:

- **`backfill_recent_ohlcv()`** (ml/data/loaders/ohlcv_recent.py):
  - Wraps `ml.data.loaders.ohlcv_recent.backfill_recent_ohlcv`
  - Returns `OhlcvRecentBackfillResult` with per-symbol status
  - Uses `DatabentoCoveragePolicy.from_env()` for coverage rules

- **`populate_l2_efficient()`** (ml/data/ingest/l2_efficient.py):
  - Efficient L2 book population via Databento
  - Wraps `ml.data.ingest.l2` loaders

- **`populate_supplementary_data()`** (ml/data/loaders/supplementary.py):
  - Macro data (FRED series, earnings)
  - Wraps `ml.data.loaders.fred_loader` and `ml.data.loaders.alternative`

- **`populate_alternative_data_task()`** (ml/data/loaders/alternative.py):
  - Alternative sources (Databento, custom providers)

- **`ingest_backfill_main()`** (ml/cli/ingest_backfill.py):
  - Main CLI entry for historical backfill
  - Multi-symbol coordination with progress tracking

**Each task has**:
- Frozen `*TaskConfig` dataclass (typed parameters)
- Main function returning typed result (`*Result` dataclass)
- Wrapper around `ml.data.*` API with CLI-friendly interface

**Used by**: 13 CLI entry points (backfill_ohlcv_recent.py, populate_l2_efficient.py, populate_supplementary_simple.py, etc.)

**Pattern example** (ml/data/loaders/ohlcv_recent.py, lines 38-59):
```python
def backfill_recent_ohlcv(config: BackfillRecentOhlcvTaskConfig) -> OhlcvRecentBackfillResult:
    service = ensure_service()  # Get Databento service
    policy = DatabentoCoveragePolicy.from_env()
    domain_config = OhlcvRecentBackfillConfig(...)  # Convert task config → data config
    return _backfill_recent_ohlcv(domain_config, service=service, policy=policy)
```

### Training Tasks (ml/training/teacher/)

**Quick TFT Training** (ml/training/teacher/quick.py):
- **`QuickTFTTrainConfig`** (lines 36-48): Data dirs, symbols, horizon_minutes, thresholds, sample_prediction_count
- **`QuickTFTTrainResult`** (lines 51-62): Result summary (dataset_parquet, dataset_csv, dataset_shape, target_distribution_json, trained, sample_predictions)
- **`train_tft_quick()`** (lines 92-202): Quick dataset + optional teacher training
  - Uses `TFTDatasetBuilder` from `ml.data.tft_dataset_builder_facade`
  - Optional TFT teacher training via `ml.training.teacher.tft_teacher.TFTTeacher`
  - Returns structured result with sample predictions

**Integration**:
- Used by `ml/cli/train_tft_quick.py`
- Demonstrates full pipeline: catalog → dataset → training → predictions

**Default data locations** (lines 28-33):
```python
_DEFAULT_DATA_DIRS = (
    Path("data/tier1"),
    Path("/home/nate/projects/nautilus_trader/data/tier1"),
    Path("data"),
)
```

### Pipeline Runners (ml/orchestration/)

**ml/orchestration/pipeline_runner.py** (Cold-path ML pipeline):
- **`MLPipelineRunner`** (lines 37-225): Encapsulates backfill/daily/realtime modes
  - `setup_ml_system()`: Initialize catalog, scheduler, collector, feature_engineer
  - `run_backfill()`: Process historical data with date ranges
  - `run_daily()`: Single daily update
  - `run_realtime()`: Long-running real-time mode
  - Signal handlers for graceful shutdown (SIGINT, SIGTERM)

- **`PipelineRunConfig`** (lines 298-305): Config dataclass (mode, start_date, end_date, config_path, dry_run, verbose)

- **`run_pipeline()`** (lines 308-315): Main entry point
  - Loads config → creates runner → setup ML system → execute mode

**Dependencies**:
- Uses `DataScheduler` from `ml.data.scheduler_facade` for scheduling
- Uses `DataCollector` from `ml.data.collector` for data collection
- Uses `FeatureEngineer` from `ml.features` (optional)
- Requires `DATABENTO_API_KEY` environment variable (unless dry_run)
- Uses `DB_CONNECTION` for feature store (defaults to localhost PostgreSQL)

**Environment validation** (lines 88-104):
- Checks `DATABENTO_API_KEY` (required for live collection)
- Checks `DB_CONNECTION` (optional, defaults to local PostgreSQL)
- Validates Polars and Databento dependencies

**ml/orchestration/scheduler.py** (Pipeline scheduling):
- **`PipelineScheduleConfig`** (lines 14-22): Scheduling config (schedule_time, interval_minutes, config_path, dry_run, force)
- **`run_pipeline_schedule()`** (lines 25-61): Wrapper around `ml.orchestration.scheduler.run_forever()`
  - Maps config fields to environment variables (`ORCH_SCHEDULE_TIME`, `ORCH_INTERVAL_MIN`, etc.)
  - Accepts explicit `invoke_pipeline` and `sleep_fn` for testability

**Used by**:
- `ml/cli/run_ml_pipeline.py` (backfill/daily/realtime modes)
- `ml/cli/pipeline_scheduler.py` (scheduled execution)

### Monitoring & Observability (ml/monitoring/, ml/observability/)

**Health Monitoring** (ml/monitoring/health.py):
- `PipelineHealthChecker`: Queries health views from database
- `aggregate_integration_health()`: Aggregate health across components
- Human-readable or JSON output formats
- Integration with Grafana dashboards

**Coverage Monitoring** (ml/cli/coverage.py):
- `CoverageReporter`: Analyze data coverage gaps
- `plan_backfill()`: Generate backfill plans based on gaps
- Supports tier-based symbol grouping

**Observability Flush** (ml/cli/observability.py):
- Exports metrics & traces to JSONL/CSV
- Seeds sample data for testing
- Async support for non-blocking flushes

**Observability Backfill** (ml/cli/observability_backfill.py):
- Historical observability data backfill
- Coordinates with existing data ingestion

---

## Integration Points

### CLI Layer (ml/cli/)

**Pattern**: CLI entry points in `ml/cli/` import tasks and wrap them with argparse.

CLI entry points import from canonical domain modules, following this standard pattern:

Example (`ml/cli/build_tft_dataset.py`, lines 1-24):
```python
from ml.data import TFTDatasetTaskConfig, build_tft_dataset

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    # ... add arguments
    args = parser.parse_args(argv)

    # Convert argparse namespace → typed task config
    config = TFTDatasetTaskConfig(
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
        symbols=_parse_symbols(args.symbols),
        # ... map all fields
    )

    # Call task function
    result = build_tft_dataset(config)

    # Present result
    print(f"Dataset built: {result.dataset_parquet}")
    return 0
```

**Benefits of this pattern**:
1. **Type safety**: Task configs are fully typed; CLIs are thin wrappers
2. **Testability**: Tasks can be tested without argparse
3. **Reusability**: Tasks can be called from other orchestration layers
4. **Separation of concerns**: CLI handles I/O, tasks handle business logic

### Data Layer (ml.data/)

**Critical relationship**: Tasks are **configuration façades** over `ml.data` APIs.

**Example flow** (TFT dataset building):
```
CLI args (argparse)
  ↓
TFTDatasetTaskConfig (ml/data/build.py)
  ↓
DatasetBuildConfig (ml/data/build.py; re-exported via ml/data/__init__.py)
  ↓
build_tft_dataset() (ml/data/build.py; re-exported via ml/data/__init__.py)
  ↓
TFTDatasetBuilder (ml/data/tft_dataset_builder_facade.py)
  ↓
BuildResult
```

**Why this layering?**
- **ml.data**: Public API for programmatic use (typed, documented, stable)
- **Canonical domain modules**: CLI-friendly typed services with field mapping and result presentation
- **ml/cli**: Thin argparse wrappers for command-line usage

**Other examples**:
- `ml/data/loaders/ohlcv_recent.py` wraps `ml.data.loaders.ohlcv_recent`
- `ml/training/teacher/quick.py` wraps `ml.data.tft_dataset_builder_facade.TFTDatasetBuilder`

### Orchestration Layer (ml.orchestration/)

**Critical integration**: `ml/pipelines/tft_train_distill.py` forwards to `ml.orchestration.pipeline_orchestrator.main()`.

This is a **compatibility wrapper** (lines 1-17):
```python
"""
Compatibility wrapper around the unified pipeline orchestrator for TFT teacher + student runs.

The legacy pipeline staged dataset build, teacher training, and student distillation
manually. This wrapper now forwards arguments to ml.orchestration.pipeline_orchestrator
so that a single orchestrator path handles registration, promotions, and validation.
"""
from ml.orchestration.pipeline_orchestrator import main as orchestrator_main
```

**Function `build_orchestrator_args()`** (lines 22-98):
- Converts TFT-specific args → unified orchestrator args
- Maps dataset flags (`--include_macro`, `--include_l2`) → orchestrator equivalents
- Maps training flags (`--train_teacher`, `--student_model_id`) → orchestrator workflow stages

**Why this matters**:
- **Backward compatibility**: Existing TFT pipelines continue to work
- **Unified path**: All pipelines eventually go through `ml.orchestration.pipeline_orchestrator`
- **Migration signal**: This wrapper shows the canonical path for new pipelines

**Also**: `ml/orchestration/scheduler.py` wraps `ml.orchestration.scheduler.run_forever()` for scheduled execution.

### Cold vs Hot Path

**Never import orchestration helpers in**:
- Actors (`ml/actors/`)
- Strategies (`ml/strategies/`)
- Any hot-path code (< 5ms P99 latency requirement)

**Always import from canonical domain modules in**:
- CLI entry points (`ml/cli/`)
- Background tasks (scheduled jobs, cron)
- Batch jobs (dataset builds, backfills)
- Orchestration components (cold-path only)

**Rationale**:
- Tasks involve I/O (file reads, database queries, network calls)
- Tasks create DataFrames and allocate large buffers
- Tasks are not optimized for latency

---

## Configuration Patterns

All task configs use `@dataclass(frozen=True)` immutable pattern:

```python
@dataclass(frozen=True)
class TFTDatasetTaskConfig:
    data_dir: Path
    out_dir: Path
    symbols: Sequence[str]
    horizon_minutes: int = 15
    threshold: float = 0.001
    # ... 26 more fields
```

**Validation**: In `__post_init__()` where needed (not always present).

**Result dataclasses**: Often mutable (`@dataclass(slots=True)` without `frozen=True`) to accumulate state:

```python
@dataclass(slots=True)
class MigrationResult:
    applied: int = 0
    skipped: int = 0
    warnings: int = 0
    errors: int = 0
    files_applied: list[Path] = field(default_factory=list)
```

---

## Metrics & Observability

### Metrics Bootstrap

Tasks use **optional** `MetricsManager` from `ml.common.metrics_manager`:

Example from `ml/pipelines/build_runner.py` (lines 54-70):
```python
try:
    from ml.common.metrics_manager import MetricsManager
    _MM = MetricsManager.default()
    _RUNS_TOTAL = _MM.counter(
        "nautilus_ml_build_runner_runs_total",
        "Total dataset build tasks executed",
        ["status"],
    )
    _RUN_DURATION = _MM.histogram(
        "nautilus_ml_build_runner_task_duration_seconds",
        "Duration of per-symbol dataset build tasks",
        ["symbol"],
    )
except Exception:  # Metrics optional
    _RUNS_TOTAL = None
    _RUN_DURATION = None
```

**Counters**: `nautilus_ml_build_runner_runs_total` (labels: status)
**Histograms**: `nautilus_ml_build_runner_task_duration_seconds` (labels: symbol)

### Message Bus

Task modules use `MessageBusConfig.from_env()` where applicable. No hard-coded topics.

**Pattern**: Respect environment-driven configuration (`scheme`, `topic_prefix`) from `ml.config.events`.

---

## Build Runner Deep Dive (ml/pipelines/build_runner.py)

**Purpose**: Multi-symbol dataset build orchestration with local concurrency.

### Key Components

**`BuildWindow`** (lines 82-86):
```python
@dataclass(frozen=True)
class BuildWindow:
    start: str | None = None  # ISO date (inclusive)
    end: str | None = None    # ISO date (inclusive)
    days_back: int | None = None  # Overrides start/end
```

**`BuildConfig`** (lines 89-155):
- 15 fields including `data_dir`, `symbols`, `window`, feature toggles, worker count
- `workers: int = 1` (controls ProcessPoolExecutor pool size)
- `use_subprocess: bool = False` (if True, spawns `uv run python -m ml.scripts.build_tft_dataset`)
- `prefer_api: bool = False` (if True, calls `ml.data.build_tft_dataset` directly instead of CLI)
- `convert_vintage_to_age: bool = False` (post-processing vintage timestamps)
- `@staticmethod from_mapping()` to parse JSON/TOML configs

**`BuildTask`** (lines 174-177):
```python
@dataclass(frozen=True)
class BuildTask:
    symbol: str
    # Future: year/month partitioning fields
```

### Execution Modes

**1. Sequential execution** (`workers <= 1`, lines 386-412):
- Iterate tasks, run `_run_single()` for each
- Log progress events to `out_dir/progress.jsonl`
- Record metrics (optional)

**2. Parallel execution** (`workers > 1`, lines 413-452):
- Use `ProcessPoolExecutor` with `max_workers=cfg.workers`
- Submit all tasks, collect results via `as_completed()`
- Same progress logging and metrics

### Build Invocation Modes

**Mode 1: Subprocess** (`use_subprocess=True`, lines 265-298):
```python
cmd = ["uv", "run", "--active", "--no-sync", "python", "-m", "ml.scripts.build_tft_dataset", *args]
proc = run_command(cmd, capture_output=True, timeout=cfg.subprocess_timeout)
```
- Uses `ml.common.subprocess_utils.run_command()`
- Logs subprocess output tail to progress JSONL
- Returns subprocess exit code

**Mode 2: API** (`prefer_api=True`, lines 301-371):
```python
from ml.data import DatasetBuildConfig as APICfg, build_tft_dataset as api_build
api_cfg = APICfg(...)
result = api_build(api_cfg)
if cfg.convert_vintage_to_age:
    _apply_vintage_conversion(result.dataset_parquet)
return 0
```
- Directly calls `ml.data.build_tft_dataset()` without subprocess overhead
- Falls back to CLI path on exception (lines 342-371)

**Mode 3: CLI import** (default, lines 373-376):
```python
from ml.cli.build_tft_dataset import main as build_main
return int(build_main(args))
```
- Imports and calls CLI `main()` directly (no subprocess)
- Suitable for testing with monkeypatching

### Progress Tracking

**Progress log format** (JSONL at `out_dir/progress.jsonl`):
```json
{"event": "start", "symbol": "SPY"}
{"event": "success", "symbol": "SPY", "rc": 0}
{"event": "failure", "symbol": "QQQ", "rc": 1}
{"event": "exception", "symbol": "AAPL", "error": "..."}
{"event": "subprocess_log", "symbol": "MSFT", "output": "..."}
```

**Resumability**: Progress log can be parsed to identify completed symbols and resume builds.

### Vintage Age Conversion

**`_apply_vintage_conversion()`** (lines 195-216):
- Reads `dataset.parquet` and `dataset_metadata.json`
- Calls `ml.preprocessing.vintage_age.convert_vintage_timestamps_to_age()`
- Writes `dataset_with_vintage_age.parquet` and updates metadata
- Used for point-in-time correctness with macro data vintages

---

## Test Coverage Analysis

### Current State

**Unit tests** (7 files in `ml/tests/unit/tasks/`):
- `test_alternative_task.py`
- `test_l2_task.py`
- `test_pipeline_runner.py`
- `test_pipeline_scheduler.py`
- `test_purged_splits.py`
- `test_supplementary_task.py`
- `test_yahoo_task.py`

**Unit tests for pipelines** (1 file):
- `ml/tests/unit/pipelines/test_build_runner.py` (62 lines, covers load_config, plan_tasks, execute with monkeypatching)

**Integration tests**:
- `ml/tests/integration/earnings/test_tft_task_dataset.py`
- `ml/tests/integration/pipeline/test_tft_train_distill_pipeline.py`
- `ml/tests/integration/pipeline/test_pipeline_orchestrator_runtime.py`
- 15+ more pipeline integration tests

**Test quality**:
- **Good**: Ingest tasks have dedicated unit tests with monkeypatching
- **Good**: `test_build_runner.py` covers config loading and execution modes
- **Gap**: No dedicated tests for `ml/data/validation.py` (200+ lines)
- **Gap**: No dedicated tests for `ml/registry/feature_operations.py` feature promotion logic
- **Gap**: Limited coverage for `ml/stores/migrations_runner.py` migration plan building (346 lines)

---

## Critical Gaps & Issues

### 1. **Incomplete Documentation for Two-Tier Dataset Building**

**Current state**:
- `ml/pipelines/build_runner.py`: Local orchestration via `concurrent.futures`, JSON/TOML config
- `ml.orchestration.pipeline_orchestrator`: Unified orchestrator with distributed execution

**Gap**: No clear guidance on:
- When to use `build_runner` vs `pipeline_orchestrator`
- Migration path for existing `build_runner` users
- Feature parity comparison

**Impact**: Users may choose the wrong tool, leading to performance issues or missing features.

**Recommendation**: Add decision matrix to docs comparing:
- Concurrency model (ProcessPoolExecutor vs distributed)
- Configuration format (JSON/TOML vs orchestrator config)
- Progress tracking and resumability
- Integration with registry and promotion workflows

### 2. **Optional Dependencies Without Clear Guidance**

**Examples**:
- `ml/data/__init__.py` (lines 18-29): Tries to import `ProductionDatasetConfig`, swallows `ModuleNotFoundError`
- `ml/orchestration/pipeline_runner.py` checks `HAS_POLARS`, `HAS_DATABENTO` but doesn't document which tasks require which dependencies

**Gap**: No `pyproject.toml` extra groups for task-specific dependencies.

**Impact**: Users encounter runtime errors when optional deps are missing.

**Recommendation**:
- Add `[tool.poetry.extras]` groups: `databento`, `production-datasets`, `full-tasks`
- Document which tasks require which extras
- Provide clear error messages pointing to installation commands

### 3. **Test Coverage Gaps**

**Missing tests**:
- `ml/data/validation.py`: No dedicated unit tests (200+ lines)
- `ml/registry/feature_operations.py`: No tests for `promote_feature_set()` quality gate validation
- `ml/stores/migrations_runner.py`: Limited tests for `split_sql_statements()` (complex dollar-quote parsing)

**Missing contract tests**:
- Task config serialization/deserialization (JSON/TOML roundtrip)
- Task result schemas (ensure backward compatibility)
- Migration idempotency (applying same migration twice should succeed)

**Impact**: Regressions may go undetected, especially for complex SQL parsing and config validation.

**Recommendation**: Add property tests using `hypothesis` for:
- Config serialization roundtrips
- SQL statement splitting (dollar quotes, string literals, comments)
- Migration idempotency (using in-memory SQLite)

### 4. **Logging Inconsistencies**

**Current state**:
- `ml/stores/migrations_runner.py`: Uses `structlog.get_logger(__name__)`
- `ml/orchestration/pipeline_runner.py`: Uses `logging.getLogger(__name__)`
- `ml/data/loaders/ohlcv_recent.py`: Uses `logging.getLogger(__name__)`

**Gap**: No centralized task logger prefix; hard to filter task logs from hot-path logs.

**Impact**: Log aggregation and filtering are difficult in production.

**Recommendation**:
- Standardize on `structlog` for all task modules (aligns with CLAUDE.md)
- Add context binding: `bind_log_context(task="dataset_build", symbol="SPY")`
- Use `configure_logging()` from `ml.common.logging_config` for consistent setup

### 5. **Vintage Age Conversion Not Integrated in All Paths**

**Current state**:
- `ml/pipelines/build_runner.py` supports `convert_vintage_to_age` (lines 339-340)
- `ml/data/build.py` supports it (lines 113-129)
- `ml.data.build_tft_dataset()` does **not** support it directly

**Gap**: Vintage conversion is a post-processing step, not part of the core data API.

**Impact**: Users calling `ml.data.build_tft_dataset()` directly won't get vintage age conversion.

**Recommendation**:
- Add `convert_vintage_to_age: bool = False` to `DatasetBuildConfig` in `ml/data/build.py` (re-exported via `ml.data`)
- Integrate conversion into `build_tft_dataset()` before returning `BuildResult`
- Remove duplication from task and pipeline layers

---

## Key File Locations

| File | Lines | Purpose |
|------|-------|---------|
| ml/<domain>/__init__.py | 41 | Lazy module loader |
| ml/registry/feature_operations.py | 97 | Feature promotion |
| ml/stores/migrations_runner.py | 346 | SQL migrations with idempotent error handling |
| ml/data/build.py | 137 | TFT dataset config wrapper over ml.data |
| ml/data/validation.py | 200+ | Dataset quality reporting |
| ml/data/ingest/__init__.py | 37 | Ingest task re-exports |
| ml/data/loaders/ohlcv_recent.py | 68 | Recent OHLCV backfill wrapper |
| ml/training/teacher/quick.py | 210 | Quick TFT training with sample predictions |
| ml/orchestration/pipeline_runner.py | 325 | MLPipelineRunner (backfill/daily/realtime) |
| ml/orchestration/scheduler.py | 65 | Scheduled pipeline execution |
| ml/pipelines/__init__.py | 172 | Public API with extensive docstrings |
| ml/pipelines/build_runner.py | 476 | Multi-symbol dataset build orchestration |
| ml/pipelines/tft_train_distill.py | 150 | Compatibility wrapper → orchestrator |

**Total task code**: ~1,604 lines (datasets/ + ingest/)
**Total monitoring/observability**: ~2,093 lines (additional)

---

## Conventions

1. **Naming**: `*TaskConfig` for configs, `*Result` for results, `*Protocol` for interfaces
2. **Immutability**: All configs frozen (`@dataclass(frozen=True)`), results mutable where state accumulates
3. **Errors**: Log with `exc_info=True` in exception handlers (CLAUDE.md requirement)
4. **Metrics**: Optional; use `MetricsManager` or `ml.common.metrics_bootstrap`
5. **Type Hints**: Complete annotations required; no `Any` without justification
6. **Imports**: Tasks import from `ml.data`, `ml.features`, `ml.registry`; never from `ml.actors` or `ml.strategies`

---

## Performance Characteristics

**All tasks are cold-path**:
- No P99 latency requirements
- File I/O, network calls, DataFrame operations allowed
- Heavy allocations (Polars/Pandas DataFrames) permitted
- Database transactions with multi-second duration OK

**DO NOT**:
- Import tasks in hot-path actors/strategies
- Call task functions from `on_bar()`, `on_quote()`, or signal generation loops
- Use tasks for real-time feature computation (use `ml.features` instead)

---

## See Also

- **ml/docs/context/context_orchestration.md**: Higher-level orchestration (scheduler, distributed execution)
- **ml/docs/context/context_cli.md**: CLI layer that wraps tasks
- **ml/docs/context/context_data.md**: Data APIs that tasks wrap (`ml.data.*`)
- **CLAUDE.md**: Performance rules, error handling, testing standards, Universal ML Architecture Patterns
