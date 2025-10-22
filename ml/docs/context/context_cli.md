# Context: CLI Module

## Overview

The ML CLI module provides 42+ command-line interfaces for managing the Nautilus Trader ML pipeline, spanning dataset building, training, data ingestion, registry management, system health monitoring, and observability. The CLI layer follows a **thin wrapper delegation pattern**: most CLIs are lightweight entry points (10-150 lines) that parse arguments and delegate to task functions in `ml/tasks/`, ensuring separation of concerns and testability.

**Total CLI Module Size**: 5,807 lines across 42 files (as of October 2025)

**Key Architectural Patterns:**

1. **Thin Wrapper Pattern**: CLIs parse args → call task functions in `ml/tasks/`
2. **Standard ArgParse Flow**: `parse_args()` → `main()` → `raise SystemExit(main())`
3. **Progressive Fallback**: PostgreSQL → JSON backends with automatic detection
4. **Config-Driven**: Environment variables + CLI overrides via dataclass configs

## Architecture

### Module Structure

```
ml/cli/
├── Dataset Building (5 CLIs, ~650 lines)
│   ├── build_tft_dataset.py           # TFT dataset builder with macro/micro/events (232 lines)
│   ├── build_production_dataset.py    # Production dataset builder (68 lines)
│   ├── dataset_report.py              # Dataset quality reports (42 lines)
│   ├── convert_vintage_age.py         # Macro vintage timestamp → age conversion (141 lines)
│   └── validate_training_claims.py    # Dataset validation claims (397 lines)
│
├── Training & Model Management (5 CLIs, ~870 lines)
│   ├── train_tft_quick.py             # Quick TFT training wrapper (100 lines)
│   ├── hpo_tft.py                     # Hyperparameter optimization (536 lines)
│   ├── emit_teacher_predictions.py    # Teacher model prediction export (47 lines)
│   ├── ensemble_teacher_preds.py      # Multi-model ensembling (106 lines)
│   └── evaluate_predictions.py        # Prediction metrics (AUC, PR-AUC) (118 lines)
│
├── Data Ingestion (8 CLIs, ~1,500 lines)
│   ├── populate_universe.py           # Unified L0/L1/L2 data population (1,001 lines)
│   ├── ingest_dbn_archive.py          # Databento DBN archive ingestion (135 lines)
│   ├── backfill_ohlcv_recent.py       # Recent OHLCV backfilling (88 lines)
│   ├── populate_yahoo_data.py         # Yahoo Finance data ingestion (95 lines)
│   ├── populate_l2_efficient.py       # L2 market depth ingestion (142 lines)
│   ├── populate_alternative_data.py   # Alternative data sources (84 lines)
│   ├── populate_supplementary_simple.py # Supplementary data (81 lines)
│   └── ingest_backfill.py             # Gap backfill orchestration (21 lines, delegates)
│
├── FRED Macro Data (3 CLIs, ~480 lines)
│   ├── fred_integration_bridge.py     # FRED updater → ML pipeline bridge (296 lines)
│   ├── fred_export_ml_parquet.py      # FRED data export to parquet (42 lines)
│   └── check_databento_subscription.py # Databento subscription checker (207 lines)
│
├── Registry & Feature Management (4 CLIs, ~440 lines)
│   ├── feature_cli.py                 # Feature lifecycle (register/promote/deprecate) (147 lines)
│   ├── feature_backfill_cli.py        # Parallel feature backfilling (113 lines)
│   ├── promote_features.py            # Feature promotion with gates (139 lines)
│   ├── promote_model_if_metrics_pass.py # Model promotion gates (87 lines)
│   └── update_artifact.py             # Artifact registry updates (148 lines)
│
├── Monitoring & Health (6 CLIs, ~350 lines)
│   ├── coverage.py                    # Pipeline coverage reporting (56 lines, delegates)
│   ├── health.py                      # System health aggregation (48 lines)
│   ├── check_pipeline_health.py       # Pipeline health checks (21 lines, delegates)
│   ├── check_symbol_datasets.py       # Symbol dataset validation (123 lines)
│   ├── sanity_check.py                # Dev sanity checks (23 lines, delegates)
│   └── compare_databento_spy_ohlcv.py # Data quality comparison (164 lines)
│
├── Observability & Events (4 CLIs, ~270 lines)
│   ├── observability.py               # Observability flush (21 lines, delegates)
│   ├── observability_backfill.py      # Observability backfill (21 lines, delegates)
│   ├── events_consumer.py             # Redis streams event consumer (107 lines)
│   └── streaming_persistence_worker.py # Streaming training worker (127 lines)
│
├── Pipeline Orchestration (4 CLIs, ~200 lines)
│   ├── pipeline_orchestrator.py       # Main pipeline orchestrator (11 lines, delegates)
│   ├── pipeline_scheduler.py          # Pipeline scheduling (71 lines)
│   ├── run_ml_pipeline.py             # Pipeline runner (58 lines)
│   └── scheduler_smoke.py             # Scheduler smoke tests (121 lines)
│
├── Database & Migrations (1 CLI, 121 lines)
│   └── apply_migrations.py            # Database migration executor (121 lines)
│
└── Dashboard (1 CLI, 101 lines)
    └── dashboard_welcome.py           # Dashboard bootstrap & welcome (101 lines)
```

**Total**: 42 CLI files, 5,807 lines

### Command Invocation Pattern

All CLI tools follow the Python module execution pattern:

```bash
# Dataset building
python -m ml.cli.build_tft_dataset --symbols SPY,QQQ --out_dir ./output

# Training
python -m ml.cli.train_tft_quick --symbols SPY --horizon-minutes 15

# Data ingestion
python -m ml.cli.populate_universe --level L0 --tier 1

# Feature management
python -m ml.cli.feature_cli register-default ~/.nautilus/ml/features

# Monitoring
python -m ml.cli.health --db-connection postgresql://...
python -m ml.cli.coverage report --dataset BARS --start 2024-01-01

# Observability
python -m ml.cli.observability flush-jsonl --base-path ./observability
python -m ml.cli.events_consumer --stream ml-events --pattern events.ml.#

# Database
python -m ml.cli.apply_migrations --schema both --full

# Dashboard
python -m ml.cli.dashboard_welcome --compose-file docker-compose.yml
```

### Standard CLI Pattern (Thin Wrapper Delegation)

**Pattern**: CLIs are thin wrappers (10-150 lines) that delegate to `ml/tasks/`:

```python
#!/usr/bin/env python3
"""
Thin wrapper delegating to :mod:`ml.tasks.{area}.{task}`.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence

from ml.tasks.{area} import {task_function}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task description")
    parser.add_argument("--option", type=str, help="Option description")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = {task_function}(args.option)
    print(result)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
```

**Key Elements:**

1. **parse_args()**: Uses argparse, accepts `argv` for testability
2. **main()**: Delegates to task function, returns int exit code
3. **Entrypoint**: `if __name__ == "__main__": raise SystemExit(main())`
4. **Type Annotations**: All args/returns typed (`argv: Sequence[str] | None`, `-> int`)
5. **Docstring**: Module docstring explains purpose and delegation target

**Examples of Thin Wrappers:**

- `ml/cli/health.py` (48 lines): Delegates to `ml.tasks.monitoring.aggregate_integration_health`
- `ml/cli/observability.py` (21 lines): Delegates to `ml.tasks.observability.flush.main`
- `ml/cli/coverage.py` (56 lines): Delegates to `ml.tasks.monitoring.coverage.main`
- `ml/cli/train_tft_quick.py` (100 lines): Delegates to `ml.tasks.training.train_tft_quick`
- `ml/cli/pipeline_orchestrator.py` (11 lines): Pure delegation to `ml.orchestration.pipeline_orchestrator.main`

### Backend Configuration Strategy

The CLI tools implement a consistent dual-backend approach:

- **Primary**: PostgreSQL backend via `NAUTILUS_REGISTRY_DB_URL` or `--db-connection`
- **Fallback**: JSON file backend with configurable path (default: `ml_registry/`)
- **Auto-detection**: Attempts PostgreSQL first, gracefully falls back to JSON if unavailable

**Environment Variables:**

```bash
# Database connection
export NAUTILUS_REGISTRY_DB_URL="postgresql://user:pass@host:port/db"
export DATABASE_URL="postgresql://..."  # Alternative
export DB_CONNECTION="postgresql://..."  # Alternative

# Databento integration (for backfill)
export DATABENTO_API_KEY="your_api_key_here"

# Catalog path (for data storage)
export NAUTILUS_CATALOG_PATH="./catalog"

# Redis event streaming (for events_consumer)
export ML_BUS_REDIS_URL="redis://localhost:6379/0"
export ML_BUS_REDIS_STREAM="ml-events"

# Streaming training persistence
export ML_STREAMING_PERSISTENCE_ENABLED="true"
export ML_STREAMING_PERSISTENCE_STATE_PATH="./checkpoints/streaming_state.json"
```

## Key Components by Category

### 1. Dataset Building CLIs

#### build_tft_dataset.py (232 lines)

**Purpose**: Build TFT (Temporal Fusion Transformer) datasets with macro/micro/L2/events features

**Core Functionality:**

- Delegates to `ml.tasks.datasets.build_tft_dataset`
- Configures via `TFTDatasetTaskConfig` dataclass
- Supports macro (FRED), micro (OHLCV), L2 (order book), events, earnings data
- Optional feature registry registration
- Vintage policy support (real-time vs final macro revisions)
- Optional vintage → age conversion for macro features

**Key Arguments:**

```bash
--data_dir        # Source data directory (default: data/tier1)
--symbols         # Comma-separated symbols (required)
--out_dir         # Output directory (required)
--horizon_minutes # Prediction horizon (default: 15)
--threshold       # Min return threshold for labeling (default: 0.001)
--lookback_periods # Feature lookback (default: 30)
--start / --end   # Date range (YYYY-MM-DD)
--chunk_days      # Chunk size for large datasets (default: 0 = no chunking)
--macro_lag_days  # Macro feature lag (default: 1)
--include_micro   # Include microstructure features
--include_l2      # Include L2 order book features
--include_events  # Include event features
--include_calendar # Include calendar features
--include_earnings # Include earnings events
--earnings_lag_days # Earnings lag days (default: 1)
--student_mode    # Student mode (exclude future-peeking features)
--emit_dataset_events # Emit dataset events to message bus
--fred_vintage_dir # FRED vintage data directory
--events_dir      # Events data directory
--register_features # Register features in feature registry
--feature_registry_dir # Feature registry path
--feature_role    # Feature role (teacher/student/inference_support)
--market_dataset_id # Market dataset identifier
--market_inputs_json # JSON payload for market feed inputs
--vintage_policy  # Vintage policy (real_time/final)
--vintage_as_of   # ISO8601 timestamp limiting macro revisions
--convert-vintage-age # Convert vintage timestamps to age features
--verbose         # Enable debug logging
--no_macro        # Disable FRED macro join
```

**Example:**

```bash
python -m ml.cli.build_tft_dataset \
  --data_dir data/tier1 \
  --symbols SPY,QQQ,IWM \
  --out_dir ./output/tft_dataset \
  --horizon_minutes 15 \
  --threshold 0.002 \
  --lookback_periods 50 \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --include_macro \
  --include_micro \
  --include_events \
  --register_features \
  --feature_registry_dir ~/.nautilus/ml/features \
  --feature_role teacher \
  --vintage_policy real_time \
  --convert-vintage-age
```

**Integration Points:**

- `ml/tasks/datasets/tft.py`: Dataset building logic
- `ml/config/market_data.py`: Market data input configuration
- `ml/data/vintage.py`: Vintage policy handling
- `ml/preprocessing/vintage_age.py`: Vintage → age conversion
- `ml/registry/feature_registry.py`: Feature registration

**File References:**

- `/home/nate/projects/nautilus_trader/ml/cli/build_tft_dataset.py` (lines 1-232)
- `/home/nate/projects/nautilus_trader/ml/tasks/datasets/tft.py`

---

#### build_production_dataset.py (68 lines)

**Purpose**: Build production datasets via `ml.tasks.datasets.build_production_dataset`

**Delegates to**: `ml.tasks.datasets.ProductionDatasetConfig` + `build_production_dataset`

**Example:**

```bash
python -m ml.cli.build_production_dataset \
  --config production_config.toml \
  --output ./production_dataset
```

---

#### dataset_report.py (42 lines)

**Purpose**: Generate dataset quality reports

**Delegates to**: `ml.tasks.datasets.generate_dataset_report`

**Example:**

```bash
python -m ml.cli.dataset_report \
  --dataset ./output/dataset.parquet \
  --report ./output/report.json
```

---

#### convert_vintage_age.py (141 lines)

**Purpose**: Convert macro vintage timestamps (`*_value_vintage_ts`) to age features (`*_vintage_age_minutes`)

**Core Functionality:**

- Streams parquet dataset in batches (default: 32,768 rows)
- Replaces `*_value_vintage_ts` columns with numeric age-in-minutes
- Updates accompanying `dataset_metadata.json`
- Uses `ml.preprocessing.vintage_age` module

**Key Arguments:**

```bash
--source           # Input parquet dataset (required)
--destination      # Output parquet (defaults to <stem>_with_vintage_age.parquet)
--metadata         # Metadata JSON path (defaults to dataset_metadata.json)
--timestamp-column # Event timestamp column (default: timestamp)
--batch-size       # Rows per batch (default: 32768)
--compression      # Parquet compression (default: snappy)
--overwrite        # Overwrite destination if exists
```

**Example:**

```bash
python -m ml.cli.convert_vintage_age \
  --source ml_out/full_tft_95/dataset.parquet \
  --metadata ml_out/full_tft_95/dataset_metadata.json \
  --batch-size 65536 \
  --overwrite
```

**Integration Points:**

- `ml/preprocessing/vintage_age.py`: Conversion logic
  - `convert_vintage_timestamps_to_age()`
  - `update_metadata_with_vintage_age()`

**File References:**

- `/home/nate/projects/nautilus_trader/ml/cli/convert_vintage_age.py` (lines 1-141)
- `/home/nate/projects/nautilus_trader/ml/preprocessing/vintage_age.py`

---

#### validate_training_claims.py (397 lines)

**Purpose**: Validate dataset quality claims (schema, stats, feature coverage)

**Example:**

```bash
python -m ml.cli.validate_training_claims \
  --dataset ./dataset.parquet \
  --claims ./claims.json
```

---

### 2. Training & Model Management CLIs

#### train_tft_quick.py (100 lines)

**Purpose**: Quick TFT training wrapper for rapid prototyping

**Core Functionality:**

- Delegates to `ml.tasks.training.train_tft_quick`
- Configures via `QuickTFTTrainConfig` dataclass
- Outputs JSON summary with dataset shape, target distribution, sample predictions

**Key Arguments:**

```bash
--data-dir         # Candidate data directories (repeatable)
--output-dir       # Output directory (CSV + Parquet)
--symbols          # Comma-separated symbols
--horizon-minutes  # Prediction horizon (default: 15)
--min-return-threshold # Min return for labeling (default: 0.002)
--lookback-periods # Feature lookback (default: 50)
--sample-predictions # Sample predictions to display (default: 10)
```

**Example:**

```bash
python -m ml.cli.train_tft_quick \
  --data-dir data/tier1 \
  --data-dir data/tier2 \
  --symbols SPY,QQQ \
  --horizon-minutes 15 \
  --min-return-threshold 0.002 \
  --lookback-periods 50 \
  --sample-predictions 10
```

**Output (JSON):**

```json
{
  "dataset_csv": "./output/dataset.csv",
  "dataset_parquet": "./output/dataset.parquet",
  "dataset_shape": [10000, 50],
  "target_distribution": {"0": 0.52, "1": 0.48},
  "trained": true,
  "sample_predictions": [0.45, 0.62, 0.38, ...]
}
```

**Integration Points:**

- `ml/tasks/training/quick.py`: Quick training logic
- `ml/common/logging_config.py`: Structured logging setup

**File References:**

- `/home/nate/projects/nautilus_trader/ml/cli/train_tft_quick.py` (lines 1-100)
- `/home/nate/projects/nautilus_trader/ml/tasks/training/quick.py`

---

#### hpo_tft.py (536 lines)

**Purpose**: Hyperparameter optimization sweep for TFT teacher models

**Core Functionality:**

- Grid search over TFT hyperparameters (hidden_dim, num_heads, dropout, etc.)
- Evaluates validation metrics from `teacher_preds.npz`
- Computes AUC, PR-AUC, PRx, LogLoss, Brier, ECE
- Outputs JSON summary with best configuration
- Optional Optuna integration (if `HAS_OPTUNA`)

**Key Arguments:**

```bash
--dataset_csv       # TFT dataset CSV (required)
--out_dir           # Output directory (required)
--feature_registry_dir # Feature registry path
--feature_set_id    # Feature set ID
--epochs            # Training epochs (default: 2)
--workers           # Parallel workers (default: 4)
```

**Example:**

```bash
python -m ml.cli.hpo_tft \
  --dataset_csv /tmp/tft_universe_60d/merged/dataset.csv \
  --out_dir /tmp/tft_universe_60d/hpo \
  --feature_registry_dir ~/.nautilus/ml/features \
  --feature_set_id <fid> \
  --epochs 5 \
  --workers 8
```

**Metrics Computed:**

- **AUC**: ROC AUC score
- **PR_AUC**: Precision-Recall AUC
- **PRx**: PR-AUC multiple (PR-AUC / prevalence baseline)
- **LogLoss**: Cross-entropy loss
- **Brier**: Brier score
- **ECE**: Expected Calibration Error (10 bins)
- **Prevalence**: Positive class prevalence

**Integration Points:**

- `ml/training/teacher/tft_cli.py`: Teacher CLI invocation
- `ml/common/subprocess_utils.py`: Subprocess execution
- `ml._imports`: Optuna optional dependency check

**File References:**

- `/home/nate/projects/nautilus_trader/ml/cli/hpo_tft.py` (lines 1-536)

---

#### evaluate_predictions.py (118 lines)

**Purpose**: Compute evaluation metrics (ROC AUC, PR AUC, logloss) from predictions

**Core Functionality:**

- Reads NPZ file with probabilities or logits + true labels
- Computes metrics: `roc_auc`, `pr_auc`, `logloss`
- Writes JSON summary for promotion gates
- Supports sigmoid transformation for logits

**Key Arguments:**

```bash
--preds       # NPZ path with predictions (required)
--probs_key   # Key for probabilities (default: q_val)
--logits_key  # Key for logits (default: z_val)
--y_key       # Key for true labels (default: y_val_true)
--out_json    # Output JSON path (required)
```

**Examples:**

```bash
# Evaluate from probabilities
python -m ml.cli.evaluate_predictions \
  --preds /tmp/teacher_preds.npz \
  --out_json /tmp/metrics.json

# Evaluate from logits
python -m ml.cli.evaluate_predictions \
  --preds /tmp/logits.npz \
  --logits_key z \
  --y_key y \
  --out_json /tmp/metrics.json
```

**Output (JSON):**

```json
{
  "logloss": 0.45,
  "pr_auc": 0.72,
  "roc_auc": 0.68
}
```

**Integration Points:**

- `sklearn.metrics`: Metric computation (requires `HAS_SKLEARN`)
- Feeds `promote_features.py` via `--metrics_json`

**File References:**

- `/home/nate/projects/nautilus_trader/ml/cli/evaluate_predictions.py` (lines 1-118)

---

#### promote_model_if_metrics_pass.py (87 lines)

**Purpose**: Promotion gate for teacher predictions based on metrics

**Core Functionality:**

- Reads `teacher_preds.npz` (probabilities + labels)
- Evaluates AUC, PR-AUC, LogLoss, Brier
- Checks against configured gates
- Exits with code 0 (pass) or 2 (fail)

**Key Arguments:**

```bash
--teacher_npz          # Path to teacher_preds.npz (required)
--min_auc              # Minimum AUC threshold (default: 0.56)
--min_pr_auc_multiple  # Minimum PR-AUC multiple of prevalence (default: 1.5)
```

**Example:**

```bash
python -m ml.cli.promote_model_if_metrics_pass \
  --teacher_npz /tmp/run/teacher_preds.npz \
  --min_auc 0.60 \
  --min_pr_auc_multiple 2.0
```

**Exit Codes:**

- `0`: Metrics pass gates → promote
- `2`: Metrics fail gates → reject

**File References:**

- `/home/nate/projects/nautilus_trader/ml/cli/promote_model_if_metrics_pass.py` (lines 1-87)

---

#### emit_teacher_predictions.py (47 lines)

**Purpose**: Export teacher model predictions to NPZ format

**Example:**

```bash
python -m ml.cli.emit_teacher_predictions \
  --model ./teacher_model.onnx \
  --dataset ./dataset.parquet \
  --output ./teacher_preds.npz
```

---

#### ensemble_teacher_preds.py (106 lines)

**Purpose**: Ensemble multiple teacher model predictions

**Example:**

```bash
python -m ml.cli.ensemble_teacher_preds \
  --preds /tmp/model1_preds.npz /tmp/model2_preds.npz \
  --weights 0.6 0.4 \
  --output /tmp/ensemble_preds.npz
```

---

### 3. Data Ingestion CLIs

#### populate_universe.py (1,001 lines)

**Purpose**: Unified data population script for ML universe (L0/L1/L2/L3 data)

**Core Functionality:**

- Populates L0 (OHLCV bars, 7 years), L1 (quotes/trades, 1 year), L2/L3 (order books, 30 days)
- Cost estimation and safeguards
- Progress tracking and resume capability
- Configurable date ranges and symbols
- Parallel downloads with rate limiting
- Comprehensive error handling

**Key Arguments:**

```bash
--estimate-only  # Estimate costs only (no download)
--level          # Data level (L0/L1/L2/L3)
--tier           # Universe tier (1/2/3)
--resume         # Resume from progress
--force          # Force restart (ignore progress)
```

**Examples:**

```bash
# Estimate costs only
python -m ml.cli.populate_universe --estimate-only

# Populate specific data level
python -m ml.cli.populate_universe --level L0
python -m ml.cli.populate_universe --level L1
python -m ml.cli.populate_universe --level L2

# Populate specific tier
python -m ml.cli.populate_universe --tier 1 --level L1

# Resume from progress
python -m ml.cli.populate_universe --resume

# Force restart (ignore progress)
python -m ml.cli.populate_universe --force
```

**Integration Points:**

- `ml/config/universes.py`: Tier 1 universe definitions (TIER1_FULL_95, TIER1_CORE_12, TIER1_SYMBOL_SETS)
- `ml/data/ingest/api.py`: `ensure_service()`, `fetch_symbol_data()`
- `ml/data/ingest/service.py`: `DatabentoIngestionService`
- `ml/data/ingest/common.py`: Progress tracking

**File References:**

- `/home/nate/projects/nautilus_trader/ml/cli/populate_universe.py` (lines 1-1001)

---

#### ingest_dbn_archive.py (135 lines)

**Purpose**: Ingest Databento DBN archives from disk into market data store

**Core Functionality:**

- Processes `.zip` bundles containing `.dbn.zst` members
- Decodes DBN format and writes to canonical SQL `market_data` table
- Optional DataStore mirror
- Handles metadata overrides (dataset, schema, source_dataset, instrument_suffix)

**Key Arguments:**

```bash
PATH              # Path to .zip archive or directory (positional, required)
--db-url          # PostgreSQL connection URL (required)
--dataset         # Override dataset identifier
--schema          # Override schema identifier
--source-dataset  # Source dataset tag for provenance
--instrument-suffix # Optional suffix for instrument IDs
--mirror-data-store # Also write to ML DataStore
--table-name      # Market data table name (default: market_data)
```

**Example:**

```bash
python -m ml.cli.ingest_dbn_archive \
  data/batch/EQUS.MINI_2024-01-01.zip \
  --db-url postgresql://postgres:postgres@localhost:5433/nautilus \
  --dataset EQUS.MINI \
  --schema bars \
  --mirror-data-store \
  --table-name market_data
```

**Integration Points:**

- `ml/data/ingest/dbn_archive.py`: `DBNArchiveIngestor`, `DBNArchiveIngestionConfig`
- `ml/stores/providers.py`: `SqlMarketDataWriter`
- `ml/stores/writers.py`: `DataStoreMarketDataWriter`

**File References:**

- `/home/nate/projects/nautilus_trader/ml/cli/ingest_dbn_archive.py` (lines 1-135)

---

#### backfill_ohlcv_recent.py (88 lines)

**Purpose**: Backfill recent OHLCV data for symbols

**Delegates to**: `ml.tasks.ingest.backfill_recent_ohlcv`

**Example:**

```bash
python -m ml.cli.backfill_ohlcv_recent \
  --symbols SPY,QQQ \
  --lookback-days 7 \
  --output ./data/recent
```

---

#### populate_yahoo_data.py (95 lines)

**Purpose**: Ingest Yahoo Finance data

**Delegates to**: `ml.tasks.ingest.populate_yahoo_data`

**Example:**

```bash
python -m ml.cli.populate_yahoo_data \
  --symbols SPY,QQQ \
  --start 2024-01-01 \
  --output ./data/yahoo
```

---

#### populate_l2_efficient.py (142 lines)

**Purpose**: Ingest L2 market depth data efficiently

**Delegates to**: `ml.tasks.ingest.populate_l2_efficient`

**Example:**

```bash
python -m ml.cli.populate_l2_efficient \
  --symbols SPY,QQQ \
  --days 30 \
  --output ./data/l2
```

---

#### populate_alternative_data.py (84 lines)

**Purpose**: Ingest alternative data sources

**Delegates to**: `ml.tasks.ingest.populate_alternative_data_task`

**Example:**

```bash
python -m ml.cli.populate_alternative_data \
  --source sentiment \
  --symbols SPY,QQQ \
  --output ./data/alternative
```

---

#### populate_supplementary_simple.py (81 lines)

**Purpose**: Ingest supplementary data (simple mode)

**Delegates to**: `ml.tasks.ingest.populate_supplementary_data`

**Example:**

```bash
python -m ml.cli.populate_supplementary_simple \
  --config supplementary_config.toml
```

---

#### ingest_backfill.py (21 lines)

**Purpose**: Gap backfill orchestration with pluggable coverage and writers

**Delegates to**: `ml.tasks.ingest.backfill.main`

**Options:**

```bash
--db                # Postgres URL (defaults DB_CONNECTION)
--dataset-id        # e.g., EQUS.MINI
--schema            # bars|tbbo|trades (bars default for catalog client)
--instruments       # Comma list or file path
--lookback-days     # Default 7 (env BACKFILL_LOOKBACK_DAYS)
--coverage-mode     # sql|catalog (default sql)
--write-mode        # sql (default sql; parquet not implemented by default)
--catalog-path      # Required for catalog coverage/client
--table-name        # Target table (default market_data)
--state-path        # State JSON path (default checkpoints/ingest_state.json)
--client-mode       # catalog|databento|noop (default catalog)
--api-key           # Databento API key (for client-mode=databento)
--dry-run           # Plan only (no ingestion/writes)
```

**Examples:**

```bash
# Plan gaps against SQL store, do not write
python -m ml.cli.ingest_backfill \
  --db postgresql://postgres:postgres@localhost:5433/nautilus \
  --dataset-id EQUS.MINI --schema bars \
  --instruments SPY.XNAS,QQQ.XNAS \
  --lookback-days 7 \
  --dry-run

# Use Parquet catalog for coverage and ingestion client, write to SQL canonical store
python -m ml.cli.ingest_backfill \
  --db postgresql://postgres:postgres@localhost:5433/nautilus \
  --dataset-id EQUS.MINI --schema bars \
  --instruments SPY.XNAS \
  --coverage-mode catalog --client-mode catalog \
  --catalog-path /abs/path/to/catalog \
  --lookback-days 14

# Use Databento client for ingestion (still writing to SQL)
python -m ml.cli.ingest_backfill \
  --db postgresql://postgres:postgres@localhost:5433/nautilus \
  --dataset-id EQUS.MINI --schema bars \
  --instruments SPY.XNAS \
  --client-mode databento --api-key "$DATABENTO_API_KEY" \
  --lookback-days 7
```

**Notes:**

- Canonical writes are SQL; registry events/watermarks reflect DB persistence
- Catalog coverage/client is intended for historical workflows; for live backfills, use SQL coverage and a real Databento client

**File References:**

- `/home/nate/projects/nautilus_trader/ml/cli/ingest_backfill.py` (lines 1-21)
- `/home/nate/projects/nautilus_trader/ml/tasks/ingest/backfill.py`

---

### 4. FRED Macro Data CLIs

#### fred_integration_bridge.py (296 lines)

**Purpose**: Integration bridge between simple FRED updater and ML pipeline

**Core Functionality:**

- Converts `fred_indicators_updated.parquet` (wide format) to ML pipeline format (long format)
- Transforms timestamp to nanoseconds
- Converts to Polars for ML pipeline compatibility
- Saves in ML-compatible parquet format

**Key Functions:**

- `convert_simple_to_ml_format()`: Wide → long format conversion
- Reads: `data/fred/fred_indicators_updated.parquet`
- Writes: `data/fred/fred_indicators_ml_format.parquet`

**Output Format:**

```python
# ML-format DataFrame (long format)
{
    "timestamp": datetime,
    "timestamp_ns": int64,
    "series_id": str,  # e.g., "GDP", "CPI", "UNRATE"
    "value": float,
}
```

**File References:**

- `/home/nate/projects/nautilus_trader/ml/cli/fred_integration_bridge.py` (lines 1-296)

---

#### fred_export_ml_parquet.py (42 lines)

**Purpose**: Export FRED data to ML-compatible parquet format

**Example:**

```bash
python -m ml.cli.fred_export_ml_parquet \
  --input data/fred/raw \
  --output data/fred/ml_format.parquet
```

---

#### check_databento_subscription.py (207 lines)

**Purpose**: Check Databento subscription status and entitlements

**Example:**

```bash
python -m ml.cli.check_databento_subscription \
  --api-key "$DATABENTO_API_KEY"
```

---

### 5. Registry & Feature Management CLIs

#### feature_cli.py (147 lines)

**Purpose**: Feature registry lifecycle management and operations

**Core Functionality:**

- Delegates to `ml.tasks.registry` functions
- Three subcommands: `register-default`, `promote`, `deprecate`

**Subcommands:**

**1. register-default**: Register default FeatureConfig as a feature set

```bash
python -m ml.cli.feature_cli register-default \
  ~/.nautilus/ml/features \
  --name default \
  --version v1.0.0 \
  --role student \
  --data-requirements l1_only
```

**2. promote**: Promote a feature set with quality gates

```bash
python -m ml.cli.feature_cli promote \
  ~/.nautilus/ml/features \
  <feature_set_id> \
  --gate roc_auc gte 0.65 \
  --gate pr_auc gte 0.70
```

**3. deprecate**: Deprecate a feature set

```bash
python -m ml.cli.feature_cli deprecate \
  ~/.nautilus/ml/features \
  <feature_set_id> \
  --reason "Replaced by v2.0.0 with improved macro features"
```

**Integration Points:**

- `ml/tasks/registry.py`: Task functions
  - `register_default_feature_set()`
  - `promote_feature_set()`
  - `deprecate_feature_set()`
- `ml/registry/feature_registry.py`: `FeatureRegistry`, `FeatureRole`
- `ml/registry/base.py`: `DataRequirements`

**File References:**

- `/home/nate/projects/nautilus_trader/ml/cli/feature_cli.py` (lines 1-147)
- `/home/nate/projects/nautilus_trader/ml/tasks/registry.py`

---

#### feature_backfill_cli.py (113 lines)

**Purpose**: Parallel feature computation and historical backfilling

**Core Functionality:**

- Delegates to `FeatureStore.compute_historical_parallel()`
- Supports file-based or comma-separated instrument lists
- Flexible date range specification (ISO 8601 format)
- Configurable worker threads for optimal resource utilization
- Force recompute option for data refresh scenarios

**Key Arguments:**

```bash
--db            # PostgreSQL connection string (required)
--instruments   # Comma-separated instruments or file path (required)
--start         # Start ISO date/time (optional)
--end           # End ISO date/time (optional)
--force         # Force recompute even if data present
--max-workers   # Max parallel workers (default 4)
```

**Example:**

```bash
# Backfill from comma-separated list
python -m ml.cli.feature_backfill_cli \
  --db "postgresql://localhost/nautilus" \
  --instruments EUR/USD,GBP/USD \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --max-workers 8 \
  --force

# Backfill from file
echo "EUR/USD" > instruments.txt
echo "GBP/USD" >> instruments.txt
python -m ml.cli.feature_backfill_cli \
  --db "postgresql://localhost/nautilus" \
  --instruments instruments.txt \
  --max-workers 4
```

**Output:**

```
Completed: 2, Failed: 0, Total rows: 150000
  EUR/USD: 75000
  GBP/USD: 75000
```

**Integration Points:**

- `ml/stores/feature_store.py`: `FeatureStore.compute_historical_parallel()`
- `ml/features/engineering.py`: `FeatureConfig`

**File References:**

- `/home/nate/projects/nautilus_trader/ml/cli/feature_backfill_cli.py` (lines 1-113)

---

#### promote_features.py (139 lines)

**Purpose**: Feature set promotion with quality gates

**Example:**

```bash
python -m ml.cli.promote_features \
  --feature_set_id <fid> \
  --metrics_json /tmp/metrics.json \
  --registry ~/.nautilus/ml/features
```

---

#### update_artifact.py (148 lines)

**Purpose**: Update artifact registry entries

**Example:**

```bash
python -m ml.cli.update_artifact \
  --artifact_id <aid> \
  --metadata ./metadata.json \
  --registry ~/.nautilus/ml/artifacts
```

---

### 6. Monitoring & Health CLIs

#### coverage.py (56 lines)

**Purpose**: Comprehensive data coverage analysis and automated backfill orchestration

**Delegates to**: `ml.tasks.monitoring.coverage.main`

**Commands:**

- `report`: Generate coverage reports showing data flow through pipeline stages
- `plan-backfill`: Identify gaps and create backfill job specifications
- `apply-backfill`: Execute backfill jobs with rate limiting and retry logic

**Key Features:**

- **Stage Coverage Analysis**: Tracks percentage coverage across all pipeline stages
- **Lag Monitoring**: Measures time since last successful processing per instrument
- **Gap Detection**: Identifies missing data where source exists but target is missing
- **Backfill Planning**: Creates JSON job specifications for missing data gaps
- **Production Execution**: Rate-limited API calls with exponential backoff retry
- **Databento Integration**: Native support for fetching historical data via Databento API

**Examples:**

```bash
# Generate coverage report for dataset
python -m ml.cli.coverage report \
  --dataset BARS \
  --start 2024-01-01 \
  --end 2024-01-07

# Identify gaps and plan backfill
python -m ml.cli.coverage plan-backfill \
  --from BARS \
  --to FEATURES \
  --date 2024-01-15

# Execute backfill job with safety measures
python -m ml.cli.coverage apply-backfill \
  --job-file backfill_job.json \
  --dry-run
python -m ml.cli.coverage apply-backfill \
  --job-file backfill_job.json
```

**Pipeline Stages Tracked:**

```
Raw Data → CATALOG_WRITTEN → FEATURE_COMPUTED → PREDICTION_EMITTED → SIGNAL_EMITTED
```

**File References:**

- `/home/nate/projects/nautilus_trader/ml/cli/coverage.py` (lines 1-56)
- `/home/nate/projects/nautilus_trader/ml/tasks/monitoring/coverage.py`

---

#### health.py (48 lines)

**Purpose**: System health aggregation and monitoring dashboard integration

**Core Functionality:**

- Delegates to `ml.tasks.monitoring.aggregate_integration_health`
- Utilizes `MLIntegrationManager` for comprehensive health checks
- Aggregates health status across all ML components (stores, registries, actors)
- Outputs JSON-formatted health summaries suitable for monitoring dashboards
- Supports strict protocol validation mode for enhanced error detection

**Key Arguments:**

```bash
--db-connection  # PostgreSQL connection string (optional)
--strict         # Raise on protocol validation failures
```

**Example:**

```bash
python -m ml.cli.health \
  --db-connection "postgresql://localhost/nautilus" \
  --strict
```

**Output (JSON):**

```json
{
  "healthy": true,
  "components": {
    "feature_store": {"status": "healthy", "checks": 5},
    "model_store": {"status": "healthy", "checks": 3},
    "data_store": {"status": "healthy", "checks": 7},
    "strategy_store": {"status": "healthy", "checks": 4}
  },
  "timestamp": "2025-10-19T12:34:56Z"
}
```

**Integration Points:**

- `ml/tasks/monitoring/__init__.py`: `aggregate_integration_health()`
- `ml/core/integration.py`: `MLIntegrationManager`

**File References:**

- `/home/nate/projects/nautilus_trader/ml/cli/health.py` (lines 1-48)
- `/home/nate/projects/nautilus_trader/ml/tasks/monitoring/__init__.py`

---

#### check_pipeline_health.py (21 lines)

**Purpose**: Pipeline health checks

**Delegates to**: `ml.tasks.monitoring.health.main`

**Example:**

```bash
python -m ml.cli.check_pipeline_health \
  --db-connection "postgresql://localhost/nautilus"
```

---

#### check_symbol_datasets.py (123 lines)

**Purpose**: Validate symbol dataset availability and quality

**Example:**

```bash
python -m ml.cli.check_symbol_datasets \
  --symbols SPY,QQQ \
  --dataset BARS \
  --start 2024-01-01
```

---

#### sanity_check.py (23 lines)

**Purpose**: Dev sanity checks for rapid validation

**Delegates to**: `ml.tasks.dev.sanity_check.main`

**Example:**

```bash
python -m ml.cli.sanity_check
```

---

#### compare_databento_spy_ohlcv.py (164 lines)

**Purpose**: Compare Databento vs catalog OHLCV data for quality assurance

**Example:**

```bash
python -m ml.cli.compare_databento_spy_ohlcv \
  --symbol SPY.XNAS \
  --start 2024-01-01 \
  --end 2024-12-31
```

---

### 7. Observability & Events CLIs

#### observability.py (21 lines)

**Purpose**: Observability data management with multiple sinks and background processing

**Delegates to**: `ml.tasks.observability.flush.main`

**Commands:**

- `flush-jsonl`: Export observability data to JSONL/CSV files
- `flush-db`: Flush observability data to PostgreSQL database
- `start`: Start background observability data collection with periodic flushing

**Examples:**

```bash
# Flush current observability data to files
python -m ml.cli.observability flush-jsonl \
  --base-path ./observability \
  --format jsonl \
  --seed-sample

# Start background collection
python -m ml.cli.observability start \
  --sink db \
  --db-url postgresql://user:pass@host/db \
  --interval 30.0 \
  --duration 3600.0

# Flush to database
python -m ml.cli.observability flush-db \
  --db-url postgresql://user:pass@host/db
```

**Integration Points:**

- `ml/tasks/observability/flush.py`: Flush logic
- `ml/core/integration.py`: `MLIntegrationManager`

**File References:**

- `/home/nate/projects/nautilus_trader/ml/cli/observability.py` (lines 1-21)
- `/home/nate/projects/nautilus_trader/ml/tasks/observability/flush.py`

---

#### observability_backfill.py (21 lines)

**Purpose**: Backfill observability data from historical sources

**Delegates to**: `ml.tasks.observability.backfill.main`

**Example:**

```bash
python -m ml.cli.observability_backfill \
  --start 2024-01-01 \
  --end 2024-12-31
```

---

#### events_consumer.py (107 lines)

**Purpose**: Redis streams event consumption with topic filtering and idempotent processing

**Core Functionality:**

- **Redis Streams Integration**: Subscribes to Redis streams with configurable stream names
- **Topic Pattern Filtering**: Wildcard pattern matching using `*` and `#` semantics
- **Idempotent Processing**: Built-in watermark gating to prevent duplicate processing
- **JSON Event Handling**: Processes events with `topic` and `payload` fields
- **Configurable Polling**: Supports blocking/non-blocking reads with iteration control

**Key Arguments:**

```bash
--redis-url    # Redis connection URL (default: env ML_BUS_REDIS_URL or redis://localhost:6379/0)
--stream       # Redis stream name (default: env ML_BUS_REDIS_STREAM or ml-events)
--pattern      # Topic pattern to filter (wildcards: * and #). May be repeated.
--count        # Max messages to read per iteration (default: 100)
--block-ms     # XREAD block duration in ms (default: 0 = non-blocking)
--iterations   # Number of poll iterations (default: 1)
```

**Examples:**

```bash
# Subscribe to feature computation events
python -m ml.cli.events_consumer \
  --redis-url redis://localhost:6379/0 \
  --stream ml-events \
  --pattern events.ml.FEATURE_COMPUTED.# \
  --iterations 1 --count 100

# Multiple pattern filtering
python -m ml.cli.events_consumer \
  --pattern events.ml.FEATURE_* \
  --pattern events.ml.PREDICTION_* \
  --block-ms 5000
```

**Integration Points:**

- `ml/consumers/redis_streams_consumer.py`: `RedisStreamsConsumer`
- `ml/common/topic_filters.py`: `match_topic()` for wildcard matching

**File References:**

- `/home/nate/projects/nautilus_trader/ml/cli/events_consumer.py` (lines 1-107)

---

#### streaming_persistence_worker.py (127 lines)

**Purpose**: Streaming training persistence worker against Redis Streams

**Core Functionality:**

- Long-running worker that polls Redis Streams for training events
- Persists streaming training snapshots to disk
- Configurable batch size, block duration, poll interval
- Graceful shutdown via signal handlers (SIGINT, SIGTERM)

**Key Arguments:**

```bash
--state-path      # Override persistence snapshot path (optional)
--batch-size      # Maximum entries to process per poll (optional)
--block-ms        # Block duration in ms for Redis XREAD (optional)
--poll-interval   # Idle interval in seconds between polls (optional)
--enable          # Force-enable worker regardless of environment
--disable         # Disable worker regardless of environment
```

**Example:**

```bash
python -m ml.cli.streaming_persistence_worker \
  --state-path ./checkpoints/streaming_state.json \
  --batch-size 50 \
  --block-ms 1000 \
  --poll-interval 2.0 \
  --enable
```

**Configuration:**

- Reads `StreamingPersistenceConfig.from_env()`
- Environment variables:
  - `ML_STREAMING_PERSISTENCE_ENABLED`
  - `ML_STREAMING_PERSISTENCE_STATE_PATH`
  - `ML_STREAMING_PERSISTENCE_BATCH_SIZE`
  - `ML_STREAMING_PERSISTENCE_BLOCK_MS`
  - `ML_STREAMING_PERSISTENCE_POLL_INTERVAL_SECONDS`

**Integration Points:**

- `ml/config/streaming_pipeline.py`: `StreamingPersistenceConfig`
- `ml/config/bus.py`: `MessageBusConfig`
- `ml/consumers/streaming_training_worker.py`: `StreamingTrainingPersistenceWorker`

**File References:**

- `/home/nate/projects/nautilus_trader/ml/cli/streaming_persistence_worker.py` (lines 1-127)

---

### 8. Pipeline Orchestration CLIs

#### pipeline_orchestrator.py (11 lines)

**Purpose**: Main pipeline orchestrator (pure delegation)

**Delegates to**: `ml.orchestration.pipeline_orchestrator.main`

**Example:**

```bash
python -m ml.cli.pipeline_orchestrator \
  --config production_full.toml
```

**File References:**

- `/home/nate/projects/nautilus_trader/ml/cli/pipeline_orchestrator.py` (lines 1-11)
- `/home/nate/projects/nautilus_trader/ml/orchestration/pipeline_orchestrator.py`

---

#### pipeline_scheduler.py (71 lines)

**Purpose**: Pipeline scheduling and cron-like execution

**Delegates to**: `ml.tasks.pipelines.run_pipeline_schedule`

**Example:**

```bash
python -m ml.cli.pipeline_scheduler \
  --schedule daily_pipeline.toml
```

---

#### run_ml_pipeline.py (58 lines)

**Purpose**: Pipeline runner for ad-hoc execution

**Delegates to**: `ml.tasks.pipelines.runner.run_pipeline`

**Example:**

```bash
python -m ml.cli.run_ml_pipeline \
  --config quick_run.toml \
  --symbols SPY,QQQ
```

---

#### scheduler_smoke.py (121 lines)

**Purpose**: Scheduler smoke tests for rapid validation

**Example:**

```bash
python -m ml.cli.scheduler_smoke
```

---

### 9. Database & Migrations CLI

#### apply_migrations.py (121 lines)

**Purpose**: Apply ML database migrations with planning and execution

**Core Functionality:**

- Delegates to `ml.tasks.db` module
- Supports schema selection (market_data, ml_observability, both)
- Optional migrations (hardening, views, fixes)
- Dry-run and print-only modes
- Progressive fallback for DB connection

**Key Arguments:**

```bash
--db-url      # PostgreSQL connection URL (optional, uses env if not provided)
--schema      # Schema selection (market_data/ml_observability/both)
--full        # Include optional migrations (hardening, views, fixes)
--dry-run     # List files that would be applied without executing
--print-only  # Print migration plan and exit
```

**Examples:**

```bash
# Apply core migrations to both schemas
python -m ml.cli.apply_migrations \
  --db-url postgresql://localhost/nautilus \
  --schema both

# Apply full migrations (including optional) with dry-run
python -m ml.cli.apply_migrations \
  --db-url postgresql://localhost/nautilus \
  --schema both \
  --full \
  --dry-run

# Print migration plan only
python -m ml.cli.apply_migrations \
  --schema ml_observability \
  --print-only
```

**Output:**

```
Migration plan:
 - ml/schema/observability/001_initial.sql
 - ml/schema/observability/002_add_indices.sql
 - ml/schema/observability/003_partitioning.sql

Migration Summary
=================
Applied: 3
Skipped: 0
Warnings: 0
Errors: 0

Files Applied:
 - ml/schema/observability/001_initial.sql
 - ml/schema/observability/002_add_indices.sql
 - ml/schema/observability/003_partitioning.sql
```

**Integration Points:**

- `ml/tasks/db.py`: Migration logic
  - `apply_database_migrations()`
  - `build_migration_plan()`
  - `split_sql_statements()`
- `ml/common/db_connections.py`: Connection management
  - `collect_postgres_candidates()`
  - `select_first_working_connection()`

**File References:**

- `/home/nate/projects/nautilus_trader/ml/cli/apply_migrations.py` (lines 1-121)
- `/home/nate/projects/nautilus_trader/ml/tasks/db.py`

---

### 10. Dashboard CLI

#### dashboard_welcome.py (101 lines)

**Purpose**: Bootstrap the Nautilus Trader ML dashboard stack and display a welcome screen

**Core Functionality:**

- Starts Docker Compose services (PostgreSQL, Redis, Grafana, Prometheus)
- Performs health checks on all services
- Displays welcome screen with service status
- Configurable health probes (timeout, retries, interval)

**Key Arguments:**

```bash
--compose-file    # docker compose file to use (default: deployment/docker-compose.yml)
--service         # explicit service list (repeatable)
--timeout         # health probe timeout in seconds (default: 5.0)
--retries         # health probe retry attempts (default: 5)
--retry-interval  # seconds between health retries (default: 2.0)
--status-only     # skip docker compose up and only display status
--checks          # custom health check in the form name=url (repeatable)
```

**Example:**

```bash
# Start all services with health checks
python -m ml.cli.dashboard_welcome \
  --compose-file deployment/docker-compose.yml \
  --timeout 10.0 \
  --retries 10

# Status check only (no docker compose up)
python -m ml.cli.dashboard_welcome \
  --status-only

# Custom health checks
python -m ml.cli.dashboard_welcome \
  --checks "postgres=http://localhost:5432/health" \
  --checks "grafana=http://localhost:3000/api/health"
```

**Integration Points:**

- `ml/dashboard_bootstrap/__init__.py`: Bootstrap logic
  - `build_welcome_summary()`
  - `DEFAULT_COMPOSE_FILE`
  - `DEFAULT_SERVICES`
  - `DEFAULT_HEALTH_CHECKS`

**File References:**

- `/home/nate/projects/nautilus_trader/ml/cli/dashboard_welcome.py` (lines 1-101)
- `/home/nate/projects/nautilus_trader/ml/dashboard_bootstrap/__init__.py`

---

## Integration Points

### Registry System Integration

All CLI tools integrate with the universal registry system:

- **Data Registry**: Event tracking and watermark management
- **Feature Registry**: Feature set lifecycle and validation
- **Model Registry**: Model deployment tracking
- **Strategy Registry**: Strategy compatibility validation

**Example**: `feature_cli.py` uses `FeatureRegistry` for registration/promotion/deprecation

### Store Integration

CLI tools leverage the mandatory 4-store pattern:

- **FeatureStore**: Historical feature computation and retrieval
- **DataStore**: Unified data access with contract validation
- **ModelStore**: Prediction storage and performance tracking
- **StrategyStore**: Trading decision persistence

**Example**: `feature_backfill_cli.py` uses `FeatureStore.compute_historical_parallel()`

### Pipeline Integration

The coverage CLI provides visibility into the complete ML pipeline:

```
Raw Data → CATALOG_WRITTEN → FEATURE_COMPUTED → PREDICTION_EMITTED → SIGNAL_EMITTED
```

**Example**: `coverage.py` tracks data flow through all pipeline stages

### External System Integration

- **Databento API**: Historical market data fetching with rate limiting
  - `populate_universe.py`, `ingest_dbn_archive.py`, `check_databento_subscription.py`
- **PostgreSQL**: Primary persistence layer with connection pooling
  - All CLIs with `--db-url` or `--db-connection` options
- **Redis Streams**: Event streaming and message bus integration
  - `events_consumer.py`, `streaming_persistence_worker.py`
- **Docker/Compose**: Health monitoring integration for containerized deployments
  - `dashboard_welcome.py`
- **File System**: Observability data export to JSONL/CSV for external analytics
  - `observability.py`, `convert_vintage_age.py`

### Tasks Layer Integration

**Critical Pattern**: CLIs are **thin wrappers** that delegate to `ml/tasks/`:

```
ml/cli/{name}.py (10-150 lines)
    ↓ delegates to
ml/tasks/{area}/{task}.py (100-1000+ lines)
    ↓ uses
ml/stores/, ml/registry/, ml/features/, ml/data/, etc.
```

**Examples:**

| CLI | Lines | Delegates To | Task Module |
|-----|-------|--------------|-------------|
| `health.py` | 48 | `ml.tasks.monitoring.aggregate_integration_health` | `ml/tasks/monitoring/__init__.py` |
| `coverage.py` | 56 | `ml.tasks.monitoring.coverage.main` | `ml/tasks/monitoring/coverage.py` |
| `observability.py` | 21 | `ml.tasks.observability.flush.main` | `ml/tasks/observability/flush.py` |
| `train_tft_quick.py` | 100 | `ml.tasks.training.train_tft_quick` | `ml/tasks/training/quick.py` |
| `build_tft_dataset.py` | 232 | `ml.tasks.datasets.build_tft_dataset` | `ml/tasks/datasets/tft.py` |
| `feature_cli.py` | 147 | `ml.tasks.registry.*` | `ml/tasks/registry.py` |
| `apply_migrations.py` | 121 | `ml.tasks.db.*` | `ml/tasks/db.py` |
| `ingest_backfill.py` | 21 | `ml.tasks.ingest.backfill.main` | `ml/tasks/ingest/backfill.py` |

**Benefits:**

1. **Testability**: Task functions testable without CLI layer
2. **Reusability**: Task functions callable from other modules (orchestrators, services)
3. **Separation of Concerns**: CLIs handle arg parsing, tasks handle business logic
4. **Type Safety**: Task functions fully typed, CLIs convert strings to typed objects

## Error Handling Patterns

### Standard Exit Codes

All CLIs follow this convention:

- `0`: Success
- `1`: General error (file not found, invalid args, runtime exception)
- `2`: Business logic failure (promotion gate failed, metrics below threshold)

**Example**:

```python
def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        result = task_function(args)
        if result.success:
            return 0
        else:
            print(f"Task failed: {result.message}", file=sys.stderr)
            return 2
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
```

### Progressive Fallback

**Pattern**: PostgreSQL → JSON backend with warning messages

```python
# Database connection with fallback
candidates = collect_postgres_candidates(
    ConnectionRole.MIGRATION,
    explicit=args.db_url,
)
if not candidates.urls:
    raise SystemExit("No PostgreSQL connection candidates found. Set --db-url or DATABASE_URL.")

try:
    db_url = select_first_working_connection(candidates.urls)
except RuntimeError as exc:
    db_url = candidates.urls[0]
    LOGGER.warning(
        "PostgreSQL connectivity probe failed; using first candidate",
        connection=db_url,
        error=str(exc),
    )
```

### Validation

**Pattern**: Input validation with descriptive error messages and early exit

```python
def _parse_symbols(value: str) -> list[str]:
    symbols = [item.strip() for item in value.split(",") if item.strip()]
    if not symbols:
        msg = "At least one symbol is required"
        raise argparse.ArgumentTypeError(msg)
    return symbols
```

### Resource Management

**Pattern**: Proper cleanup in finally blocks

```python
def run_ingestion(config: IngestionConfig) -> None:
    ingestor = Ingestor(config)
    try:
        ingestor.run()
    finally:
        ingestor.cleanup()
```

## Performance Considerations

### Parallel Processing

- **feature_backfill_cli.py**: Configurable thread pools (default: 4 workers)
- **populate_universe.py**: Parallel downloads with rate limiting

### Memory Management

- **convert_vintage_age.py**: Streaming processing for large datasets (batch size: 32,768 rows)
- **build_tft_dataset.py**: Chunking support (`--chunk_days`) to prevent OOM

### Rate Limiting

- **populate_universe.py**: Rate-limited API calls to prevent quota exhaustion
- **coverage.py**: Exponential backoff retry for API failures

## Security Considerations

### API Key Management

- **Never hardcode secrets**: All API keys loaded from environment
- **Databento**: `DATABENTO_API_KEY` environment variable
- **FRED**: `FRED_API_KEY` environment variable

### SQL Injection

- **Parameterized Queries**: All SQL uses SQLAlchemy `text()` with bound parameters
- **No String Interpolation**: Never construct SQL via f-strings or string concatenation

### File Path Validation

- **Path Validation**: All file paths resolved via `Path.resolve()` and validated
- **Overwrite Protection**: `--overwrite` flags required for destructive operations

### Connection Security

- **Database URLs**: Connection string validation and secure defaults
- **TLS Support**: Redis and PostgreSQL connections support TLS via connection string

## Implementation Notes

### Testing Strategy

All CLIs designed for testability:

1. **parse_args()** accepts `argv` parameter for testing
2. **main()** returns int exit code (testable via `assert main(["--arg", "value"]) == 0`)
3. **Pure delegation** to task functions (task functions fully unit tested)

**Example Test:**

```python
def test_health_cli():
    result = health.main(["--db-connection", "postgresql://test"])
    assert result == 0
```

### Logging Strategy

CLIs use structured logging:

```python
from ml.common.logging_config import bind_log_context, configure_logging

configure_logging(level="DEBUG" if args.verbose else "INFO")
run_id = f"cli_build_tft_dataset_{_uuid.uuid4().hex[:8]}"
bind_log_context(run_id=run_id, component="ml.cli.build_tft_dataset")
```

### Config-Driven Design

CLIs use dataclass configs for all tunable values:

```python
from ml.tasks.datasets import TFTDatasetTaskConfig

cfg = TFTDatasetTaskConfig(
    data_dir=Path(args.data_dir),
    out_dir=out_dir,
    symbols=args.symbols,
    horizon_minutes=args.horizon_minutes,
    threshold=args.threshold,
    # ... all config from args
)

result = build_tft_dataset(cfg)
```

**Benefits:**

1. **Type Safety**: Config validated at construction
2. **Testability**: Config objects fully testable
3. **Documentation**: Dataclass fields self-documenting
4. **Immutability**: `frozen=True` prevents accidental mutation

## Known Gaps and Incomplete Work

### No TODOs Found

As of October 2025, **zero TODOs/FIXMEs/HACKs** found in ml/cli/ codebase (verified via grep).

### Potential Enhancements

1. **Unified CLI Framework**: Consider consolidating CLIs into a single entrypoint with subcommands (e.g., `ml <command> <subcommand>`)
2. **Progress Bars**: Add progress bars for long-running operations (dataset building, ingestion)
3. **Config File Support**: Support loading CLI args from TOML/YAML config files
4. **Shell Completion**: Add bash/zsh completion scripts for better UX
5. **Docker Integration**: Add `--docker` flag to run CLIs inside containers

### Documentation Gaps

1. **CLI Reference**: No centralized CLI reference documentation (this file is a start!)
2. **Cookbook**: Missing cookbook with common workflow examples
3. **Troubleshooting**: No troubleshooting guide for common CLI errors

## Usage Patterns

### Coverage Reporting Workflow

```bash
# 1. Generate coverage report for dataset
python -m ml.cli.coverage report --dataset BARS --start 2024-01-01 --end 2024-01-07

# 2. Identify gaps and plan backfill
python -m ml.cli.coverage plan-backfill --from BARS --to FEATURES --date 2024-01-15

# 3. Execute backfill job with safety measures
python -m ml.cli.coverage apply-backfill --job-file backfill_job.json --dry-run
python -m ml.cli.coverage apply-backfill --job-file backfill_job.json
```

### Feature Management Workflow

```bash
# 1. Backfill historical features in parallel
python -m ml.cli.feature_backfill_cli \
  --db "postgresql://localhost/nautilus" \
  --instruments EUR/USD,GBP/USD \
  --max-workers 8

# 2. Register feature set
python -m ml.cli.feature_cli register-default \
  ~/.nautilus/ml/features \
  --name default \
  --version v1.0.0

# 3. Monitor system health
python -m ml.cli.health \
  --db-connection "postgresql://localhost/nautilus" \
  --strict

# 4. Stream event consumption
python -m ml.cli.events_consumer \
  --redis-url redis://localhost:6379/0 \
  --stream ml-events \
  --pattern events.ml.#

# 5. Observability data management
python -m ml.cli.observability start \
  --sink file \
  --interval 60.0 \
  --duration 3600.0
```

### Dataset Building Workflow

```bash
# 1. Build TFT dataset with all features
python -m ml.cli.build_tft_dataset \
  --data_dir data/tier1 \
  --symbols SPY,QQQ \
  --out_dir ./output/tft_dataset \
  --horizon_minutes 15 \
  --include_macro \
  --include_micro \
  --include_events \
  --register_features

# 2. Convert vintage timestamps to age features
python -m ml.cli.convert_vintage_age \
  --source ./output/tft_dataset/dataset.parquet \
  --metadata ./output/tft_dataset/dataset_metadata.json \
  --overwrite

# 3. Validate dataset
python -m ml.cli.validate_training_claims \
  --dataset ./output/tft_dataset/dataset_with_vintage_age.parquet \
  --claims ./claims.json

# 4. Generate dataset report
python -m ml.cli.dataset_report \
  --dataset ./output/tft_dataset/dataset_with_vintage_age.parquet \
  --report ./output/tft_dataset/report.json
```

### Training Workflow

```bash
# 1. Quick training for prototyping
python -m ml.cli.train_tft_quick \
  --data-dir data/tier1 \
  --symbols SPY \
  --horizon-minutes 15

# 2. Hyperparameter optimization
python -m ml.cli.hpo_tft \
  --dataset_csv ./output/dataset.csv \
  --out_dir ./output/hpo \
  --epochs 10 \
  --workers 8

# 3. Evaluate predictions
python -m ml.cli.evaluate_predictions \
  --preds ./output/hpo/best/teacher_preds.npz \
  --out_json ./output/hpo/best/metrics.json

# 4. Promote model if metrics pass gates
python -m ml.cli.promote_model_if_metrics_pass \
  --teacher_npz ./output/hpo/best/teacher_preds.npz \
  --min_auc 0.60 \
  --min_pr_auc_multiple 2.0
```

### Data Ingestion Workflow

```bash
# 1. Estimate costs for L0/L1/L2 data
python -m ml.cli.populate_universe --estimate-only

# 2. Populate L0 data (7 years OHLCV)
python -m ml.cli.populate_universe --level L0 --tier 1

# 3. Populate L1 data (1 year quotes/trades)
python -m ml.cli.populate_universe --level L1 --tier 1

# 4. Ingest Databento archives
python -m ml.cli.ingest_dbn_archive \
  data/batch/EQUS.MINI_2024-01-01.zip \
  --db-url postgresql://localhost/nautilus \
  --dataset EQUS.MINI \
  --schema bars

# 5. Backfill gaps
python -m ml.cli.ingest_backfill \
  --db postgresql://localhost/nautilus \
  --dataset-id EQUS.MINI \
  --schema bars \
  --instruments SPY.XNAS,QQQ.XNAS \
  --lookback-days 7
```

### Database & Migrations Workflow

```bash
# 1. Print migration plan
python -m ml.cli.apply_migrations \
  --schema both \
  --full \
  --print-only

# 2. Dry-run migrations
python -m ml.cli.apply_migrations \
  --db-url postgresql://localhost/nautilus \
  --schema both \
  --full \
  --dry-run

# 3. Apply migrations
python -m ml.cli.apply_migrations \
  --db-url postgresql://localhost/nautilus \
  --schema both \
  --full
```

### Dashboard Workflow

```bash
# 1. Start dashboard stack
python -m ml.cli.dashboard_welcome \
  --compose-file deployment/docker-compose.yml

# 2. Check dashboard status
python -m ml.cli.dashboard_welcome \
  --status-only

# 3. Custom health checks
python -m ml.cli.dashboard_welcome \
  --checks "postgres=http://localhost:5432/health" \
  --checks "grafana=http://localhost:3000/api/health"
```

## Environment Configuration

```bash
# ========================================
# Database Connections
# ========================================
export NAUTILUS_REGISTRY_DB_URL="postgresql://user:pass@host:port/db"
export DATABASE_URL="postgresql://user:pass@host:port/db"
export DB_CONNECTION="postgresql://user:pass@host:port/db"

# ========================================
# API Keys
# ========================================
export DATABENTO_API_KEY="your_databento_api_key"
export FRED_API_KEY="your_fred_api_key"

# ========================================
# Data Paths
# ========================================
export NAUTILUS_CATALOG_PATH="./catalog"
export ML_REGISTRY_PATH="~/.nautilus/ml"

# ========================================
# Redis Event Streaming
# ========================================
export ML_BUS_REDIS_URL="redis://localhost:6379/0"
export ML_BUS_REDIS_STREAM="ml-events"

# ========================================
# Streaming Training Persistence
# ========================================
export ML_STREAMING_PERSISTENCE_ENABLED="true"
export ML_STREAMING_PERSISTENCE_STATE_PATH="./checkpoints/streaming_state.json"
export ML_STREAMING_PERSISTENCE_BATCH_SIZE="50"
export ML_STREAMING_PERSISTENCE_BLOCK_MS="1000"
export ML_STREAMING_PERSISTENCE_POLL_INTERVAL_SECONDS="2.0"

# ========================================
# Backfill Configuration
# ========================================
export BACKFILL_LOOKBACK_DAYS="7"

# ========================================
# Logging
# ========================================
export ML_DEBUG="1"  # Enable debug logging
```

## Quick Reference: CLI by Use Case

### Dataset Building
- `build_tft_dataset.py`: Build TFT datasets with macro/micro/events
- `build_production_dataset.py`: Build production datasets
- `convert_vintage_age.py`: Convert vintage timestamps to age features
- `validate_training_claims.py`: Validate dataset quality

### Training
- `train_tft_quick.py`: Quick TFT training
- `hpo_tft.py`: Hyperparameter optimization

### Data Ingestion
- `populate_universe.py`: Unified L0/L1/L2 data population
- `ingest_dbn_archive.py`: Databento archive ingestion
- `backfill_ohlcv_recent.py`: Recent OHLCV backfill
- `ingest_backfill.py`: Gap backfill orchestration

### Registry & Features
- `feature_cli.py`: Feature lifecycle (register/promote/deprecate)
- `feature_backfill_cli.py`: Parallel feature backfilling
- `promote_features.py`: Feature promotion with gates
- `promote_model_if_metrics_pass.py`: Model promotion gates

### Monitoring & Health
- `health.py`: System health aggregation
- `coverage.py`: Pipeline coverage reporting
- `check_pipeline_health.py`: Pipeline health checks

### Observability
- `observability.py`: Observability flush
- `events_consumer.py`: Redis streams event consumer
- `streaming_persistence_worker.py`: Streaming training worker

### Database
- `apply_migrations.py`: Database migration executor

### Dashboard
- `dashboard_welcome.py`: Dashboard bootstrap

## Summary

The ML CLI module provides 42+ command-line interfaces totaling 5,807 lines of code. The architecture follows a **thin wrapper delegation pattern** where CLIs (10-150 lines) parse arguments and delegate to task functions in `ml/tasks/` (100-1000+ lines), ensuring separation of concerns, testability, and reusability.

**Key Strengths:**

1. **Comprehensive Coverage**: 42 CLIs covering all ML pipeline operations
2. **Consistent Patterns**: Standard argparse → main → task delegation flow
3. **Type Safety**: Complete type annotations on all functions
4. **Testability**: All CLIs accept `argv` parameter for testing
5. **Progressive Fallback**: PostgreSQL → JSON backends with graceful degradation
6. **Config-Driven**: Dataclass configs for all tunable values
7. **No Technical Debt**: Zero TODOs/FIXMEs/HACKs in codebase

**Architecture Highlights:**

- **Thin Wrapper Pattern**: CLIs are lightweight entry points, business logic in `ml/tasks/`
- **Standard Exit Codes**: 0 (success), 1 (error), 2 (business failure)
- **Structured Logging**: All CLIs use `ml.common.logging_config` for consistent logging
- **Error Handling**: Input validation, progressive fallback, resource cleanup

**Integration Points:**

- 4-store pattern (FeatureStore, ModelStore, DataStore, StrategyStore)
- 4-registry pattern (FeatureRegistry, ModelRegistry, DataRegistry, StrategyRegistry)
- External systems (Databento, PostgreSQL, Redis, Docker)
- Tasks layer (`ml/tasks/`) for all business logic
