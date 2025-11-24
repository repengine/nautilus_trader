# Nautilus ML Architecture Map

**Status:** Living Document
**Root:** `ml/`
**Context:** Event-driven Machine Learning layer sitting atop Nautilus Trader.

## 1. System Overview

The `ml` package is designed around a strict separation of concerns to satisfy high-frequency trading requirements:

-   **Cold Path (Training/Research):** Focus on throughput, data volume, and experimental flexibility. Uses heavy libraries (Polars, PyTorch, MLflow).
-   **Hot Path (Inference/Trading):** Focus on latency (P99 < 5ms) and allocation-free execution. Uses lightweight structures (Numpy, C++ bridges).

## 2. Top-Level Directory Structure

### A. Core Foundations

-   `config/` - **[CRITICAL]** Centralized configuration definitions (dataclasses). No magic numbers allowed in code.
-   `core/` - Base classes, interfaces, and shared domain logic.
-   `common/` - Shared utilities (logging, timestamps, simple helpers).
-   `ml_types.py` - Global type definitions.

### B. Data & Feature Engineering

-   `data/` - Data ingestion, loading, and dataset building (Parquet/Polars).
-   `features/` - Feature definitions and generators. Bridges the gap between offline calculation and online streaming.
-   `preprocessing/` - Data cleaning and transformation logic.

### C. State & Registry (The "Brains")

-   `registry/` - Central management for artifacts:
-   **Model Registry:** Tracks trained models (MLflow backend).
-   **Feature Registry:** Tracks feature definitions.
-   **Strategy Registry:** Tracks strategy configurations.
-   `stores/` - Persistence layers (Redis, Postgres, FileSystem).
-   `schema/` - Data contracts (Pandera schemas) ensuring data quality.

### D. Training (Cold Path)

-   `training/` - Training loops, trainer implementations, and offline experiments.
-   `models/` - Model wrapper definitions and architectures.
-   `evaluation/` - Backtesting and metric calculation tools.

### E. Inference & Execution (Hot Path)

-   `actors/` - **[Nautilus Integration]** Nautilus Actors that host models for real-time inference.
-   `strategies/` - ML-driven strategies that consume signals from Actors.
-   `consumers/` - Event bus consumers for signal distribution.

### F. Orchestration & Operations

-   `orchestration/` - Workflow management for complex pipelines.
-   `pipelines/` - Concrete pipeline definitions.
-   `tasks/` - Async tasks (likely Celery or similar).
-   `cli/` - Command-line entry points for human interaction.
-   `deployment/` - Infrastructure-as-Code and deployment scripts.

### G. Observability

-   `observability/` - Tracing, structured logging, and telemetry hooks.
-   `monitoring/` - Metrics collection (Prometheus) and dashboarding.

## 3. Architectural Invariants (from GEMINI.md)

1.  **Hot-Path Latency:** No I/O, no Pandas, no heavy allocations in `actors/` or `strategies/`.
2.  **Config-Driven:** All tunable parameters must live in `config/`.
3.  **Type Safety:** Strict typing (MyPy) is enforced across the board.
4.  **Dependencies:** ML libraries (XGBoost, Torch) are imported lazily via `ml._imports`.
