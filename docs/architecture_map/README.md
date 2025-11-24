# Nautilus Trader ML Architecture Map

**Status:** Generated Artifact
**Scope:** `ml/` package and integration points.

## 1. Introduction

This directory contains a "System Cartography" of the Machine Learning layer in Nautilus Trader. It maps the codebase not just by file structure, but by **flow**, **lifecycle**, and **architectural constraints**.

The system is designed around a strict **Hot Path / Cold Path** separation to enable high-frequency trading (HFT) capabilities while supporting complex ML workflows.

## 2. System Navigation

| Subsystem | Directory | Responsibility | Key Pattern |
| :--- | :--- | :--- | :--- |
| **[Root Overview](./ml/README.md)** | `ml/` | Configuration & Common Utilities | *Configuration-as-Code* |
| **[Core Integration](./ml/core/README.md)** | `ml/core/` | Component Wiring & Optimization | *Progressive Fallback* |
| **[Stores](./ml/stores/README.md)** | `ml/stores/` | Persistence Facades | *DataStore/FeatureStore Facades* |
| **[Data Layer](./ml/data/README.md)** | `ml/data/` | Ingestion, Catalog, & Datasets | *Direct Catalog Access* |
| **[Feature Engineering](./ml/features/README.md)** | `ml/features/` | Batch & Online Calculation | *Dual-Path Parity* |
| **[Training](./ml/training/README.md)** | `ml/training/` | Model Training & Distillation | *Teacher-Student* |
| **[Registry](./ml/registry/README.md)** | `ml/registry/` | Artifact Lifecycle | *4-Pillars (Model, Feature, Strategy, Data)* |
| **[Orchestration](./ml/orchestration/README.md)** | `ml/orchestration/` | Pipeline Coordination | *Gap-Free Pipeline* |
| **[Actors (Hot Path)](./ml/actors/README.md)** | `ml/actors/` | Real-time Inference | *Zero-Allocation Hot Path* |

## 3. Data Flow Architecture

```mermaid
graph TD
    %% Cold Path
    subgraph "Cold Path (Training & Research)"
        Raw[Data Sources] -->|ml.data| Catalog[Parquet Catalog]
        Catalog -->|ml.features (Batch)| Dataset[Training Dataset]
        Dataset -->|ml.training| Trainer[XGBoost/TFT Trainer]
        Trainer -->|Export ONNX| Artifact[Model Artifact]
        Artifact -->|Register| Reg[Model Registry]
    end

    %% Registry Bridge
    subgraph "Registry Bridge"
        Reg -->|Load| Actor
        FeatReg[Feature Registry] -->|Schema| Actor
    end

    %% Hot Path
    subgraph "Hot Path (Trading)"
        Market[Market Data] -->|on_bar| Actor[MLSignalActor]
        Actor -->|ml.features (Online)| Features[Feature Vector]
        Features -->|Inference| Signal[MLSignal]
        Signal -->|Event Bus| Strategy
    end
```

## 4. Universal ML Architecture Patterns

The codebase rigorously adheres to 5 Universal Patterns:

1.  **4-Store + 4-Registry:** No ad-hoc state. Everything lives in a Store (Data/Feature/Model/Strategy) or Registry.
2.  **Protocol-First:** Components interact via abstract Interfaces (Protocols), enabling dependency injection and mocking.
3.  **Hot/Cold Separation:**
    -   *Cold Path:* Heavy I/O, Pandas/Polars, Python objects.
    -   *Hot Path:* Zero I/O, Numpy arrays, Pre-allocated C++ objects.
4.  **Progressive Fallback:** Systems degrade gracefully (e.g., `CircuitBreaker` in Actors, `DummyRegistry` in tests).
5.  **Centralized Metrics:** All components emit telemetry to a unified Prometheus sink.

## 5. How to Use This Map

-   **New Developer:** Start with **[Root Overview](./ml/README.md)** to understand the package layout, then read **[Actors](./ml/actors/README.md)** to see how code runs in production.
-   **Adding a Feature:** Read **[Feature Engineering](./ml/features/README.md)** to understand how to add logic that works in both training and inference.
-   **Debugging Production:** Check **[Actors](./ml/actors/README.md)** for circuit breaker logic and **[Registry](./ml/registry/README.md)** for artifact security.
