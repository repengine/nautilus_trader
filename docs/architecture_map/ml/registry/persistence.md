# Registry Persistence

**Status:** Living Document
**Root:** `ml/registry/persistence.py`
**Key Class:** `PersistenceManager`

## 1. System Overview

The Persistence Layer abstracts the storage mechanism for the ML Registry system. It supports two modes:

1.  **JSON (Development):** Human-readable files stored on disk. Ideal for local testing and debugging.
2.  **PostgreSQL (Production):** Robust, concurrent, transactional storage using SQLAlchemy. Ideal for distributed deployment.

## 2. Data Models (SQLAlchemy)

The module defines the database schema for the 4 pillars:

-   **`ModelTable`**: Mirrors `ModelManifest`. Includes `artifact_sha256_digest` and `deployment_status`.
-   **`FeatureTable`**: Mirrors `FeatureManifest`. Includes `schema_hash` and `pipeline_signature`.
-   **`StrategyTable`**: Stores strategy configuration schemas and backtest metrics.
-   **`AuditLogTable`**: Immutable log of all registry actions (`registered`, `deployed`, `retired`).

## 3. Persistence Manager

The `PersistenceManager` acts as the gateway.

-   **Initialization:** Checks `PersistenceConfig.backend` (`JSON` or `POSTGRES`).
-   **Connection Pooling:** Uses `ml.common.db_utils.get_or_create_engine` to manage SQL connections efficiently.
-   **Audit Logging:** Automatically writes to `audit_log.jsonl` (JSON mode) or `registry_audit_log` table (Postgres mode).

## 4. Configuration

Configuration is controlled by `PersistenceConfig`:

-   `connection_string`: Postgres URL (e.g., `postgresql://user:pass@host/db`). Defaults to env `NAUTILUS_REGISTRY_DB_URL`.
-   `json_path`: Root directory for JSON files (e.g., `ml_registry/`).

## 5. Data Flow

```mermaid
graph TD
    A[Registry Client] -->|Save Manifest| B[PersistenceManager]
    B -->|Check Config| C{Backend?}
    C -->|JSON| D[Write .json File]
    C -->|Postgres| E[SQLAlchemy Session]
    E -->|Commit| F[(PostgreSQL DB)]
    B -->|Log Action| G[Audit Log]
```
