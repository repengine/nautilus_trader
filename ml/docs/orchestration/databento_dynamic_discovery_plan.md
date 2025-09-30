# Databento Dynamic Discovery Refactor Plan

## Objectives

- Replace static Databento dataset descriptors and safety allowlists with a runtime discovery mechanism that selects the optimal dataset per symbol and schema.
- Preserve guardrails for coverage, cost, and schema compatibility while supporting stage-aware orchestration across ingestion, dataset build, and depth ingestion.
- Keep all integrations fully typed, observable, and compliant with existing ML architectural standards (structlog logging, metrics via `ml.common.metrics_bootstrap`, and strict mypy/ruff gates).

## Current Constraints (Research)

1. `ml/orchestration/pipeline_orchestrator.py:163-188` injects every entry from `ml/config/market_feed_descriptors.json`, forcing symbols to the hard-coded dataset IDs.
2. `ml/config/databento_safe_config.json` enumerates allowed datasets and enforces a global cost limit, causing the ingestion service (`ml/data/ingest/service.py:469-471`) to reject any dataset not on the list.
3. `IngestionOrchestrator.resolve_market_bindings` only consults the provided `MarketDatasetInput` descriptors, so runtime discovery never occurs.
4. Coverage providers and manifest builders assume static mappings for storage kind and schemas, so mirrored writes break when new datasets appear.

## Analysis

- Dynamic discovery must query Databento metadata (`Historical.metadata`) to retrieve schema availability, coverage windows, and estimated costs per symbol.
- Guardrails must be enforced via policy objects rather than static JSON: e.g., compute per-request cost estimates and compare against environment-configured ceilings.
- We need typed data structures to describe discovery results (dataset ID, schema, storage kind, instrument templates, coverage bounds, cost estimates).
- Orchestrator workflow needs to accept discovery results, convert them into `MarketDatasetInput` equivalents, and reuse current binding selection logic without regressing metrics or event emission.

## Plan (Design + Implementation Steps)

1. **Discovery Module**
   - Create `ml/data/ingest/discovery.py` with a typed `DatasetDiscoveryService` that
     - Accepts `DatabentoHistoricalClient` plus policy config
     - Exposes `discover(symbols: Sequence[str], schema: str) -> tuple[DiscoveredDataset, ...]`
     - Calculates per-symbol cost via `get_cost`, respects coverage windows, and raises typed errors for violations
   - Instrument with structlog + metrics histogram (`discovery_latency_seconds`).

2. **Policy Refactor**
   - Introduce a typed policy dataclass (`DynamicSafetyPolicy`) that reads env overrides instead of hard-coded JSON.
   - Deprecate or downgrade `databento_safe_config.json` to optional seed defaults; allow discovery to proceed when datasets are not pre-listed but pass policy checks.

3. **Orchestrator Integration**
   - Update `_apply_default_market_inputs` to call the discovery service when explicit inputs are absent. Remove blind cloning of `market_feed_descriptors.json`.
   - Extend `_resolve_market_inputs` to merge explicit descriptors with dynamic discoveries, ensuring per-symbol selection and schema bucketing remain backwards compatible.
   - Maintain full typing by introducing `DiscoveredMarketInput` (Protocol) convertible to `MarketDatasetInput`.

4. **Service + Safety Updates**
   - Adjust `DatabentoIngestionService` to use the new policy object. Replace `_validate_dataset` static set check with a call that verifies policy constraints (e.g., `policy.allow(dataset, schema)`), integrating cost/tier logic.
   - Ensure structured logging captures dataset, schema, symbol, cost, and decision outcome.

5. **Tests**
   - Add unit tests for discovery (mock Databento client) covering:
     - Successful discovery with multiple schemas
     - Cost/coverage rejection
     - Schema compatibility mapping
   - Add orchestrator tests verifying dynamic inputs are injected and bindings are resolved per symbol without explicit descriptors.
   - Update integration smoke tests to use discovery path and assert ingestion proceeds with datasets not in the static list.

6. **Validation + Docs**
   - Run `uv run --active --no-sync mypy ml --strict`, `make ruff`, and targeted pytest suites (`make pytest -k discovery`).
   - Document behaviour in `ml/docs/orchestration_runbook.md` and add examples showing configuration-free discovery.

## Implementation Checklist

- [ ] Discovery module with typed dataclasses, logging, and metrics
- [ ] Dynamic safety policy replacing static allowlist
- [ ] Orchestrator refactor to call discovery
- [ ] Updated ingestion service validation
- [ ] Unit/integration tests
- [ ] Documentation updates
- [ ] Tooling checks (mypy, ruff, pytest)

