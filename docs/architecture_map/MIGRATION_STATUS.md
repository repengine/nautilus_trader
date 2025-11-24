# Migration & Technical Debt Status

**Status:** Verified Code Audit
**Date:** 2025-11-19

## 1. Overview

The ML subsystem is in a **Hybrid State**. The "Universal ML Architecture" is present in design but compromised by legacy implementation details. A comprehensive code audit has confirmed 4 Critical and 5 Major issues that must be resolved to achieve production readiness.

## 2. Subsystem Status

| Subsystem | Migration Phase | Status | Key Issues Identified |
| :--- | :--- | :--- | :--- |
| **Registry** | Phase 1 | ✅ **Done** | Scalability bottleneck (Loads all models on init). |
| **Orchestrator** | Phase 2 | ⚠️ **Pending** | God Class Wrapper pattern obscures logic. |
| **Stores** | Phase 3 | ⚠️ **Pending** | Synchronous I/O used in Actor fallback. |
| **Actors** | Phase 4 | ❌ **Legacy** | Blocking I/O in Hot Path. Memory Leaks. |
| **Features** | Phase 5 | ❌ **Legacy** | Fake Vectorization (Python loops). Hidden Allocations. |

## 3. Confirmed Technical Debt (Audit Results)

### A. Critical Severity (Must Fix for Production)

1.  **Fake Vectorization (`ml/features/engineering.py`):** `update_batch_vectorized` uses a Python `for` loop over bars. This effectively makes offline training single-threaded and slow.
2.  **Blocking Hot Path (`ml/actors/base.py`):** The `_generate_prediction_protected` method falls back to synchronous `write_features` (SQLAlchemy) if the async worker fails. This will cause massive latency spikes (>10ms).
3.  **Hidden Allocations (`ml/features/engineering.py`):** `IndicatorManager` creates a new list copy (`price_history[...]`) on *every* bar update. This generates O(N) garbage per tick.
4.  **Synchronous Store (`ml/stores/feature_persistence.py`):** The persistence layer is designed for Cold Path (blocking) but is being accessed by the Actor fallback.
5.  **Fake Distillation (`ml/training/distillation/emit.py`):** The function `generate_teacher_targets` generates "teacher logits" by taking the mean of the input features (`logits = arr.mean(axis=1)`). It **does not** load or query a trained TFT model.

### B. Major Severity (Stability/Correctness)

1.  **Memory Leak (`ml/actors/signal.py`):** `_prediction_history` grows unbounded in some code paths.
2.  **Registry Scalability (`ml/registry/model_registry.py`):** `_load_registry` loads *all* model metadata into RAM on startup. This will crash with >10k models.
3.  **Calendar Stubs (`ml/data/tft_dataset_builder.py`):** `is_trading_day` always returns `True`.
4.  **Silent Null Writers (`ml/orchestration/pipeline_orchestrator.py`):** Missing configuration results in data being silently discarded.
5.  **Dual Ingestion Paths (`ml/data/ingest/`):** Two parallel implementations for downloading data: `l2_efficient.py` vs `orchestrator.py`. `l2_efficient.py` is robust (streaming) but standalone; `orchestrator.py` is integrated but uses older logic.
6.  **Microstructure Loop (`ml/features/microstructure.py`):** `compute_all_features` iterates `for i in range(...)` over every timestamp to calculate rolling window features.

### C. Moderate Severity (Optimization)

1.  **Implicit Logit Conversion (`ml/training/teacher/tft_teacher.py`):** Targets in [0, 1] are automatically logit-transformed, which may be incorrect for regression tasks.
2.  **N+1 Query Pattern (`ml/data/tft_dataset_builder.py`):** Earnings features are fetched one-by-one.
3.  **God Class Wrapper (`ml/orchestration/pipeline_orchestrator.py`):** Inherits from the massive Legacy implementation while mixing in new components.
4.  **Synchronous Migrations (`ml/core/integration.py`):** Synchronous execution of DB migrations on process startup.

### D. Minor Severity (Hygiene)

1.  **Spaghetti Logic (`ml/data/tft_dataset_builder.py`):** Deeply nested loops and try/except blocks for handling fallback paths.
2.  **Singleton State (`ml/core/integration.py`):** Relies on global module-level state.
3.  **Path Traversal Risk (`ml/registry/model_registry.py`):** Relies on string prefix matching (`startswith`) after `resolve()`.
4.  **Manual Metadata Parsing (`ml/training/teacher/tft_teacher.py`):** Complex manual logic to align streaming batch inputs with row identifiers.

## 4. Next Steps (Remediation Plan)

1.  **Phase 4 (Actors):** Rewrite `MLSignalActor` to remove `_prediction_history` list and enforce `MLPersistenceWorker` usage (disable synchronous fallback).
2.  **Phase 5 (Features):** Rewrite `IndicatorManager` to use `collections.deque` for history and implement true Polars vectorization for `update_batch_vectorized`.
3.  **Phase 6 (Cleanup):** Implement pagination for `ModelRegistry` and consolidate ingestion paths.
