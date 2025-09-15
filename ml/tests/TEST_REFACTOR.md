Test Refactor Checklist

- [x] Phase 0 — Inventory & Clustering
  - [x] Enumerate and classify tests (unit/property/contracts/metamorphic/integration/performance)
  - [x] Identify DB-bound “unit” tests and flakey/serial usage
  - [x] Maintain migration map (TEST_MIGRATION_MAP.csv)

- [x] Phase 1 — Infrastructure & Profiles
  - [x] Standardize pytest config (markers, addopts, env defaults, strict markers)
  - [x] Provide Make/UV targets for dev-fast/integration/validators
  - [x] Sanctioned fakes (JSON DataRegistry, in-memory publisher) and deterministic Hypothesis profile
  - [x] DB cleanup strategy (allowlist TRUNCATE; statement_timeout; TEST_DB_SKIP_TRUNCATE=1)

- [x] Phase 2 — Migration & Normalization
  - [x] Move DB-bound unit tests to integration with module-scoped cleanup
  - [x] Normalize remaining unit tests to be DB-free; drop @database/@serial
  - [x] Refactor registry fallback unit test to assert POSTGRES→JSON without touching DB
  - [x] Update TEST_MIGRATION_MAP.csv with moves/splits

- [x] Phase 3 — Contracts & Boundaries
  - [x] Contracts assert enums (Stage/Source/EventStatus) and topics via build_topic_for_stage
  - [x] Align/introduce Pandera-like schema assertions where missing (event payload schemas)
  - [x] Idempotent consumer contracts (duplicates by correlation_id; watermark gating)
  - [x] Actor-bus mutual exclusion (store-level publishers disabled when actor bus enabled)
  - [x] Event watermark progression schema at event boundary (monotonic per key)
  - [x] Unit tests use metrics_bootstrap DTOs only (no HTTP server)

- [ ] Phase 4 — Properties & Metamorphic (in progress)
  - [x] Property-based idempotent consumer invariants (duplicates + watermarks)
  - [x] Metamorphic tests for store event publishing modes (batch/row/both)
  - [x] Property-based invariants for ModelStore/StrategyStore (bounds, ordering, timestamps)
  - [x] Throttler properties (burst limit, steady-state refill)
  - [x] Properties for topic scheme parity (instrument normalization across schemes)

- [x] Integration Hygiene
  - [x] Prefer module/class-scoped cleanup; avoid per-test TRUNCATE
  - [x] Group DB tests; minimize serial

- [x] Observability in Unit
  - [x] No Prometheus HTTP server; metrics_bootstrap only

- [ ] Validators & Reports
  - [x] validate-events passes for contracts
  - [x] validate-metrics passes
  - [ ] Coverage/JUnit under ml/tests/validation_reports/

Changes Completed in This Tranche

- Moved DB-bound tests to integration with module-scoped cleanup:
  - test_strategy_performance_agg → integration/test_stores_strategy_performance_agg.py
  - Split Postgres smoke out of unit registry conformance → integration/registry/test_data_registry_postgres_backend_smoke.py
- Normalized unit tests to be DB-free:
  - Dropped @database/@serial markers in unit suites (hypothesis comprehensive, monitoring collectors, data_store_validation, deployment check_health, config)
  - Refactored registry_fallback to DataStore + mock stores with DataRegistry monkeypatch (POSTGRES→JSON)
  - Switched DSNs to sqlite:///:memory: where DataStore constructed in unit tests
- Updated TEST_MIGRATION_MAP.csv accordingly

- Added event-driven contract tests (Phase 3 completion):
  - contracts/test_event_bus_contracts.py (updated)
    - Enums-only (Stage/Source/EventStatus.value), metadata.correlation_id
    - Topics via build_topic_for_stage (no hardcoded strings)
  - contracts/test_consumer_idempotency_contracts.py (new)
    - Uses IdempotentConsumer; validates duplicate correlation drops and watermark gating
  - contracts/test_actor_bus_mutual_exclusion_contracts.py (new)
    - Actor-side bus enablement disables store-level publishers (mutual exclusion)
  - contracts/test_watermark_event_contracts.py (new)
    - Pandera schema for event watermark progression (monotonic per dataset/instrument/source)
  - contracts/test_store_env_topic_config_contracts.py (new)
    - Stores honor env-driven scheme/prefix via MessageBusConfig.from_env in BusPublisherMixin
  - metamorphic/test_store_publishing_modes.py (new)
    - Row vs batch vs both preserves counts/ranges; scheme/prefix respected
  - contracts/test_fallback_metrics_contracts.py (new)
    - Fallback activation increments ml_fallback_activations_total with labels
  - property/test_consumer_idempotent_properties.py (new)
    - Hypothesis-based invariants for idempotent consumer duplicate/watermark behavior
  - property/test_model_store_invariants.py (new)
    - Predictions/confidence bounds; ts_init>=ts_event; non-decreasing ordering preserved
  - property/test_strategy_store_invariants.py (new)
    - BUY/SELL direction bounds; ts_init>=ts_event; non-decreasing ordering
  - property/test_topic_scheme_parity_properties.py (new)
    - Instrument normalization parity across domain_op/stage_first schemes

Validation Commands

- Types: uv run --active --no-sync mypy ml --strict
- Lint: make ruff
- Fast tests: make test-fast
- Integration (subset): uv run --active --no-sync pytest -q -m integration -n 1 ml/tests/integration -k "registry or strategy_performance_agg"
- Validators: make validate-events; make validate-metrics

Notes

- Running mypy in this environment may fail due to external site-packages (e.g., "can't read .../google"). Use the standard dev venv (make install-debug) and run `uv run --active --no-sync mypy ml --strict`.
- `make test-fast` uses coverage with Cython.Coverage plugin; install Cython locally or run without coverage if speed only is needed.
