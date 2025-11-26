# ML Concerns – Duplicate Basename Inventory

Generated from `ml/py_files_non_tests_by_name_described.json` (non-test `.py` files). Use this to pick canonical owners per concern and retire duplicates.

| Basename | Count | Examples (path: summary) |
|---|---|---|
| `base.py` | 9 | ml/actors/base.py: Base class for ML inference actors.; ml/config/base.py: Base configuration classes for ML components using msgspec. |
| `protocols.py` | 6 | ml/common/protocols.py: Universal ML component protocol and mixin.; ml/consumers/protocols.py: Consumer protocols and event envelope type for cross-domain processing. |
| `registry.py` | 5 | ml/actors/common/registry.py: Registry Operations Component.; ml/config/registry.py: Registry-related configuration classes. |
| `scheduler.py` | 4 | ml/data/scheduler.py: Data Scheduler for automated daily data collection and processing.; ml/observability/scheduler.py: Background flusher for observability (off hot-path). |
| `metrics.py` | 4 | ml/common/metrics.py: Centralized Prometheus metrics for the ML system.; ml/dashboard/blueprints/metrics.py: Metrics Blueprint for Dashboard API. |
| `service.py` | 3 | ml/dashboard/service.py: Dashboard service implementation providing health aggregation and control actions.; ml/data/ingest/service.py: Databento historical ingestion service. |
| `pipeline.py` | 3 | ml/dashboard/blueprints/pipeline.py: Pipeline Blueprint for Dashboard API.; ml/features/pipeline.py: Declarative feature pipeline scaffolding. |
| `persistence.py` | 3 | ml/observability/persistence.py: Observability persistence adapters (off hot-path).; ml/registry/persistence.py: Persistence layer for registry with configurable backends (JSON or PostgreSQL). |
| `observability.py` | 3 | ml/cli/observability.py: Thin wrapper delegating observability flush tasks.; ml/config/observability.py: Module for ml / config / observability.py. Description not found. |
| `lightgbm.py` | 3 | ml/config/lightgbm.py: Configuration for LightGBM model training.; ml/training/non_distilled/lightgbm.py: LightGBM trainer for financial time series prediction (non-distilled). |
| `health.py` | 3 | ml/cli/health.py: Thin wrapper delegating ML integration health aggregation to tasks.; ml/dashboard/blueprints/health.py: Health and Services Blueprint for Dashboard API. |
| `features.py` | 3 | ml/actors/common/features.py: Features Component.; ml/dashboard/blueprints/features.py: Features Blueprint for Dashboard API. |
| `feature_computation.py` | 3 | ml/data/common/feature_computation.py: Feature computation component extracted from DataScheduler.; ml/stores/common/feature_computation.py: Feature computation component for FeatureStore. |
| `events.py` | 3 | ml/config/events.py: Canonical event constants for ML pipeline stages and sources.; ml/data/providers/events.py: Event schedule provider for economic and earnings calendars. |
| `coverage.py` | 3 | ml/cli/coverage.py: Thin wrapper delegating to :mod:`ml.tasks.monitoring.coverage`.; ml/config/coverage.py: CoveragePolicy — subscription-bound lookback windows for backfill. |
| `correlation.py` | 3 | ml/common/correlation.py: Correlation utilities for event tracing.; ml/features/cross_asset/correlation.py: Rolling Correlation Computation - Cross-Asset Relationship Features. |
| `adapters.py` | 3 | ml/actors/adapters.py: Signal policy adapter protocol and example implementations.; ml/config/adapters.py: Configuration utilities for ML actors. |

## Notes
- Events/Calendar/Earnings concerns show multiple implementations; align on one DTO/protocol and provider per concern.
- Scheduler/Orchestrator/Backfill/Coverage split across CLI/tasks/data/orchestration; designate canonical entrypoints and mark others as legacy or wrappers.
- Metrics/Observability/Persistence modules are fragmented; prefer `ml.common.metrics_bootstrap` and a single observability persistence API.
- Keep this inventory updated when retiring duplicates or adding canonical facades.