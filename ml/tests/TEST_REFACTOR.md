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
  - [x] Strategy PnL directional properties (monotonic trend sanity)
  - [x] Throttler/backpressure extensions (rate/burst combos; watermark immutability)

- [x] Integration Hygiene
  - [x] Prefer module/class-scoped cleanup; avoid per-test TRUNCATE
  - [x] Group DB tests; minimize serial

- [x] Observability in Unit
  - [x] No Prometheus HTTP server; metrics_bootstrap only

Organization by Category (per Testing Strategy)

- Property-Based (`ml/tests/property/`)
  - New: `test_model_store_invariants.py`, `test_strategy_store_invariants.py`, `test_topic_scheme_parity_properties.py`, `test_strategy_pnl_properties.py`
  - Extended: `test_topic_scheme_parity_properties.py` adds stage/prefix/mapping checks and batch parity across all Stage values
  - Existing highlights: `test_consumer_idempotent_properties.py`, `test_throttler_properties.py`, `test_store_invariants.py`, `test_event_ordering_invariants.py`, `test_domain_bookkeeping_stateful.py` (stateful properties)
- Contract/Schema (`ml/tests/contracts/`)
  - Highlights: `test_store_env_topic_config_contracts.py`, `test_fallback_metrics_contracts.py`, `test_event_bus_contracts.py`, `test_consumer_idempotency_contracts.py`, `test_actor_bus_mutual_exclusion_contracts.py`, `test_watermark_event_contracts.py`, `test_store_schemas.py`
- Metamorphic (`ml/tests/metamorphic/`)
  - Highlights: `test_store_publishing_modes.py`, `test_signal_predictions.py`, `test_feature_transforms.py`, `test_event_publishing_metamorphic.py`, `test_store_time_shift_and_permutation_metamorphic.py`
- Pairwise/Combinatorial (`ml/tests/combinatorial/`)
  - New: `test_topic_scheme_parity_pairwise.py` — Stage × instrument × prefix pairwise coverage for topic parity
  - Use small curated sets to keep runs fast; expand as needed
- Integration (`ml/tests/integration/`)
  - DB-backed and end-to-end checks; marked `@pytest.mark.integration` and prefer module/class-scoped cleanup.
- Performance (`ml/tests/performance/`)
  - Micro-bench and guardrails: `test_observability_perf.py`, `test_hot_path_fixes.py`, `test_parity_buffer_guardrails.py` (respect hot-path budgets and zero allocations).

Profiles and Commands (aligned with Testing Strategy)

- Quick Validation (green lane)
  - Property: `pytest -q ml/tests/property -x`
  - Contracts: `pytest -q ml/tests/contracts -x`
  - Metamorphic: `pytest -q ml/tests/metamorphic -x`
  - Pairwise: `pytest -q ml/tests/combinatorial -x` (when present)
- dev-fast profile (no integration/serial)
  - `TEST_DB_SKIP_TRUNCATE=1 ML_DISABLE_METRICS_SERVER=1 HYPOTHESIS_PROFILE=ci \
     pytest -q -m "not integration and not serial" -n auto --dist=loadscope \
     ml/tests/unit ml/tests/property ml/tests/contracts`
- dev-medium (add integration)
  - `pytest -q -m integration ml/tests/integration`
- Integration subset
  - `uv run --active --no-sync pytest -q -m integration -n 1 ml/tests/integration`
- Validators
  - `make validate-events` and `make validate-metrics`

Environment Controls & Isolation

- `TEST_DB_SKIP_TRUNCATE=1` — skip per-test TRUNCATE; prefer module/class-scoped cleanup
- `ML_DISABLE_METRICS_SERVER=1` — disable Prometheus HTTP server in unit tests
- `HYPOTHESIS_PROFILE=ci` — deterministic, bounded property tests
- `PYTHONHASHSEED` — pin for reproducible dict ordering when needed

Database Strategy

- Unit/Contract: use JSON-backed registry/fakes; avoid DB I/O
- Integration: real Postgres via EngineManager; mark `@pytest.mark.integration`
- Cleanup: prefer transaction rollback; otherwise allowlisted TRUNCATE with short `statement_timeout`

CI Pipeline (suggested stages)

1. Lint + Types: ruff, `mypy ml --strict`, import-linter
2. Unit/Contract/Property: parallel (deterministic profiles); produce coverage + JUnit
3. Integration: spin up Postgres; run `@integration` (serial where marked)
4. Extended: metamorphic + validators; duplication/security scans

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
  - metamorphic/test_store_time_shift_and_permutation_metamorphic.py (new)
    - Time shift relation: ts_min/ts_max shift by constant; non-ts fields unchanged
    - Permutation invariance: key set (id, instrument, ts_event) preserved under reversal
    - Duplication invariance (advisory): unique key set unchanged with in-batch duplicates (upsert contract)
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
    - Prefix behavior and domain/op mapping validated; batch parity across all stages
  - property/test_strategy_pnl_properties.py (new)
    - Directional PnL sanity under monotonic up/down trends for BUY/SELL-only sequences
  - property/test_throttler_properties.py (updated)
    - Additional properties: burst cap at a single tick; accumulation bound over time
  - unit/actors/test_domain_event_bridge.py (updated)
    - Ensures payload watermark fields (ts_min/ts_max) are not mutated under throttled/published paths
  - unit/actors/test_circuit_breaker_unit.py (new)
    - Validates CircuitBreaker transitions: CLOSED→OPEN on threshold, OPEN→HALF_OPEN on timeout, HALF_OPEN→CLOSED on success_threshold, HALF_OPEN failure re-opens

Validation Commands

- Types: uv run --active --no-sync mypy ml --strict
- Lint: make ruff
- Fast tests: make test-fast
- Integration (subset): uv run --active --no-sync pytest -q -m integration -n 1 ml/tests/integration -k "registry or strategy_performance_agg"
- Validators: make validate-events; make validate-metrics

Notes

- Running mypy in this environment may fail due to external site-packages (e.g., "can't read .../google"). Use the standard dev venv (make install-debug) and run `uv run --active --no-sync mypy ml --strict`.
- `make test-fast` uses coverage with Cython.Coverage plugin; install Cython locally or run without coverage if speed only is needed.
