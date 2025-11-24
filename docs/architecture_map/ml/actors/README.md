# ML Actors Architecture

**Status:** Living Document
**Root:** `ml/actors/`
**Context:** The Hot Path execution layer.

## 1. System Overview

The `ml/actors` module contains the **Actors** that run inside the Nautilus Trader event loop. They are responsible for receiving market data, computing features, running model inference, and emitting signals—all within strict latency budgets (<5ms).

**Core Pattern:** `BaseMLInferenceActor` implements the Universal Pattern 1 (4-Store + 4-Registry Integration).

## 2. Core Components

### A. Base Actor (`base.py`)

-   **`BaseMLInferenceActor`**: The foundation.
-   **Lifecycle:** `on_start` loads the model and indicators. `on_bar` executes the hot path.
-   **Protection:** Includes `CircuitBreaker` and `HealthMonitor` to degrade gracefully under load or error.
-   **Persistence:** Writes predictions asynchronously to the `ModelStore` via `MLPersistenceWorker`.

### B. Signal Actor (`signal.py`)

-   **`MLSignalActor`**: The primary production actor.
-   **Strategy:** Configurable `SignalStrategy` (e.g., Threshold, Adaptive).
-   **Output:** Emits `MLSignal` events which Strategies subscribe to.

### C. Feature Computation

-   **Zero Allocation:** Actors use pre-allocated Numpy arrays (`self._feature_buffer`) to avoid GC pauses during `on_bar`.
-   **Stateful Indicators:** Uses the same `IndicatorManager` logic as the offline feature engineering, ensuring parity.

## 3. Data Flow (Hot Path)

```mermaid
graph TD
    A[Market Data (Bar)] -->|on_bar| B[BaseMLInferenceActor]
    B -->|Update| C[Indicators (C++)]
    C -->|Read| D[Feature Buffer (Numpy)]
    D -->|Inference (ONNX)| E[Model]
    E -->|Prediction| F[Signal Policy]
    F -->|Publish| G[MLSignal]
    G --> H[Trading Strategy]

    subgraph "Async Sidecar"
    B -.->|Enqueue| I[MLPersistenceWorker]
    I -.->|Batch Write| J[Postgres/Redis]
    end
```

## 4. Important Files

-   `ml/actors/base.py`: **[HEAVY]** Contains the `BaseMLInferenceActor` logic, circuit breakers, and hot-reload mechanisms.
-   `ml/actors/signal.py`: The concrete implementation used in production.
-   `ml/actors/ml_domain_events.py`: Internal event definitions.

## 5. Key Invariants

1.  **Latency Budget:** P99 < 5ms. No blocking I/O allowed in `on_bar`.
2.  **Safety:** `CircuitBreaker` must trip if error rates spike, preventing cascade failures.
3.  **Parity:** The Actor *must* load the exact Feature Schema defined in the Registry to ensure inputs match the trained model.

## 6. Code Audit Findings (2025-11-19)

### A. Blocking I/O Fallback (`base.py`)

-   **Severity:** **CRITICAL**
-   **Location:** `BaseMLInferenceActor._generate_prediction_protected` (Line ~1430)
-   **Issue:** If `_persistence_worker` is None (or fails), the code falls back to synchronous `_feature_store.write_features()`.
-   **Impact:** This writes to Postgres (SQLAlchemy) directly on the Actor thread, blocking the Hot Path for milliseconds to seconds (if DB is slow).

### B. Memory Leak Risk (`signal.py`)

-   **Severity:** **MAJOR**
-   **Location:** `MLSignalActor` (State management)
-   **Issue:** `self._prediction_history` is initialized as a standard `list` and potentially appended to without a strict `maxlen` constraint in all code paths (unlike `deque`).
-   **Impact:** Long-running actors may slowly OOM.

### C. Hidden Allocations in Strategies (`signal.py`)

-   **Severity:** **MODERATE**
-   **Location:** `MomentumStrategy.generate_signal` (Line ~786)
-   **Issue:** `recent_predictions = history[-look:]` creates a list slice copy on every bar.
-   **Impact:** Unnecessary GC pressure in the Hot Path.
