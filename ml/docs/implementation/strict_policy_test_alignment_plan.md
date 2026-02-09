# Strict Policy Test Alignment Plan

Date: 2026-02-08

## Objective

Preserve strict-by-default production safety while aligning tests to validate the strict contracts directly, not bypass them.

## Non-Negotiables

- Do not relax strict production defaults in runtime code.
- Do not add global permissive test defaults.
- Keep strict compatibility gates active by default in test runs.
- Use permissive policy only in explicitly scoped legacy-compatibility tests.

## Strict Contracts To Enforce In Tests

- Serveable model registration must include `feature_set_id`.
- `feature_set_id` must resolve to a registry entry with matching `feature_schema_hash`.
- Strict model loading requires SHA-256 digest metadata.
- Serveable model registration/load requires output semantics metadata (`output_schema`, calibration metadata when applicable).

## Execution Plan

1. Audit failing tests by cluster and classify intent.
- Policy-contract tests: should assert strict failures/successes explicitly.
- Behavior tests unrelated to policy: must provide strict-valid manifests/artifacts.
- Legacy/migration compatibility tests: keep permissive behavior but scope it explicitly.

2. Update test data/builders for strict-valid defaults where intent is non-policy behavior.
- Ensure serveable manifests include `feature_set_id`.
- Register matching feature sets in test setup.
- Provide artifact digests via manifest or sidecar metadata.
- Provide valid output semantics for serveable manifests.

3. Keep policy behavior explicit at test call sites.
- Strict-mode assertions must set strict policy/env explicitly.
- Permissive-mode assertions must set permissive policy/env explicitly and be marked as legacy/migration compatibility checks.
- Avoid relying on ambient environment for policy expectations.

4. Add fixture helpers for policy isolation, not policy relaxation.
- `isolated_registry_policy_env`: clears policy env keys to prevent cross-test leakage.
- `strict_registry_policy_env`: explicit strict env for tests that must enforce strict behavior.
- Optional permissive fixture allowed only for explicitly marked legacy/migration tests.

5. Update fixture documentation and examples.
- Add strict-policy testing guidance to `ml/tests/fixtures/FIXTURE_GUIDE.md`.
- Document when permissive policy is acceptable and require explicit marker usage.
- Include examples for strict-valid manifest construction and feature registry linkage.

6. Validate and gate.
- Run focused suites for registry, loader, and integration clusters.
- Run full `poetry run pytest -n auto ml` and compare failure clusters before/after.
- Keep strict lanes passing: `make pytest-ml-registry-hardening` and `make pytest-ml-strict-policy`.

## Deliverables

- Updated failing tests aligned with strict contracts.
- Shared fixture helpers for policy isolation and explicit strict/permissive intent.
- Updated `ml/tests/fixtures/FIXTURE_GUIDE.md` with strict-policy standards.
- Reduced or eliminated strict-policy fallout in full-suite runs without weakening production behavior.

## Acceptance Criteria

- Production strict defaults remain unchanged.
- Full suite no longer fails due to implicit legacy assumptions about permissive policy.
- Strict-policy contract tests still fail when required fields are absent.
- Any permissive behavior is explicit, localized, and documented.

## Progress Update (2026-02-08)

### Completed Implementation Items

- Added strict-first registry policy fixture helpers under `ml/tests/fixtures/registry_policy.py`:
  - `isolated_registry_policy_env`
  - `strict_registry_policy_env`
  - `permissive_registry_policy_env` (explicit opt-in only)
- Updated fixture export guard to include the new policy fixture module (`ml/tests/fixtures/test_exports.py`).
- Remediated strict-fallout registry tests to use strict-valid serveable manifests:
  - matching `feature_set_id` and feature registry linkage
  - matching `feature_schema_hash`
  - explicit output semantics metadata
  - digest metadata when strict load tests require it
- Kept permissive behavior only in explicitly-scoped tests that intentionally validate permissive compatibility (`test_load_model_missing_digest_warning` in security integration suite).
- Updated strict-first testing documentation:
  - `ml/tests/fixtures/FIXTURE_GUIDE.md`
  - `ml/tests/docs/TESTING_STRATEGY.md`

### Evidence Snapshot

- Focused strict-fallout cluster (registry facade + security + policy files):
  - `94 passed` (`poetry run pytest -n auto -q ml/tests/unit/registry/test_model_registry_facade.py ml/tests/contracts/test_registry_behavioral.py ml/tests/e2e/test_model_registry_e2e.py ml/tests/integration/registry/test_model_registry_security.py ml/tests/unit/common/test_model_load_policy.py ml/tests/unit/actors/common/test_model_load_policy_gating.py`)
- Expanded registry/model-loader integration slice:
  - `593 passed, 1 skipped` (`poetry run pytest -n auto -q ml/tests/unit/registry ml/tests/integration/registry ml/tests/contracts/test_registry_behavioral.py ml/tests/unit/common/test_model_load_policy.py ml/tests/unit/actors/common/test_model_load_policy_gating.py`)

### Invariants Preserved

- Production strict defaults remain unchanged in:
  - `ml/config/registry.py`
  - `ml/registry/model_registry_facade.py`
  - `ml/common/model_load_policy.py`
- No global permissive test defaults were introduced.

## Progress Update (2026-02-08, Cluster A/B/C Continuation)

### Completed Implementation Items

- Extended shared strict-valid artifact helpers in `ml/tests/utils/model_artifacts.py`:
  - `ensure_strict_onnx_sidecar` for existing ONNX files
  - `write_stub_onnx_artifact` now supports calibration sidecar payloads
- Remediated actor direct-load strict digest fallout by wiring strict-valid ONNX sidecars into:
  - `ml/tests/fixtures/common.py`
  - `ml/tests/unit/actors/test_model_loader.py`
  - `ml/tests/unit/actors/test_facade_parity.py`
  - `ml/tests/unit/actors/common/test_model.py`
  - `ml/tests/unit/actors/common/test_model_warmup.py`
- Remediated strict serveable registration fallout in training/property suites:
  - `ml/tests/unit/training/test_registry_first_export.py`
  - `ml/tests/unit/test_ml_hypothesis_comprehensive.py`
  - `ml/tests/unit/test_ml_property_comprehensive.py`
  - ensured strict-valid `feature_set_id` linkage, matching `feature_schema_hash`, and output semantics metadata via manifest/sidecar
- Fixed the warning-prone timestamp normalization path in `ml/data/ingest/orchestrator.py` by preferring numeric coercion before datetime inference.

### Evidence Snapshot

- Cluster A focused actor suites:
  - `127 passed` (`poetry run pytest -q ml/tests/unit/actors/test_model_loader.py ml/tests/unit/actors/test_base_ml_inference_actor_facade.py ml/tests/unit/actors/test_facade_parity.py ml/tests/unit/actors/common/test_model.py ml/tests/unit/actors/common/test_model_warmup.py --maxfail=0`)
- Cluster B focused training/property suites:
  - `33 passed` (`poetry run pytest -q ml/tests/unit/training/test_registry_first_export.py ml/tests/unit/test_ml_hypothesis_comprehensive.py ml/tests/unit/test_ml_property_comprehensive.py --maxfail=0`)
- Cluster C targeted warning path:
  - `1 passed` (`poetry run pytest -q ml/tests/unit/ingest/test_orchestrator_additional.py::test_normalize_time_columns_numeric_and_ts_init_paths`)
  - datetime-inference warnings for that path no longer emitted.
- Strict lanes remained green:
  - `make pytest-ml-registry-hardening`: `19 passed`
  - `make pytest-ml-strict-policy`: runtime `39 passed` + registry `19 passed`

### Full-Suite Baseline After This Pass

- `poetry run pytest -n auto ml --maxfail=0`
- Result: `1 failed, 8157 passed, 85 skipped, 1 xfailed`
- Remaining failure is unrelated to strict-policy compatibility clusters:
  - `ml/tests/unit/data/ingest/test_symbology.py::test_resolver_raises_after_retry_budget_exhausted` (`nautilus_ml_symbology_retry_total` expected `1.0`, observed `2.0`)

## Progress Update (2026-02-08, Final Closeout Pass)

### Completed Implementation Items

- Remediated the final full-suite failure in `ml/tests/unit/data/ingest/test_symbology.py` by replacing brittle absolute metric assertions with delta assertions against pre/post counter samples.
- Kept runtime strict-policy behavior unchanged; the fix is test-isolation only.

### Evidence Snapshot

- Focused failing file under xdist:
  - `4 passed` (`poetry run pytest -q -n auto ml/tests/unit/data/ingest/test_symbology.py --maxfail=1`)
- Expanded ingest-related shard:
  - `33 passed` (`poetry run pytest -q -n auto ml/tests/unit/data/ingest/test_symbology.py ml/tests/unit/data/ingest/test_discovery.py ml/tests/unit/ingest/test_ingestion_service.py ml/tests/unit/ingest/test_ingestion_service_branches.py --maxfail=1`)
- Strict lanes remained green:
  - `make pytest-ml-registry-hardening`: `19 passed`
  - `make pytest-ml-strict-policy`: runtime `39 passed` + registry `19 passed`
- Full validation gates:
  - `poetry run mypy ml --strict`: `Success: no issues found in 1799 source files`
  - `poetry run ruff check ml`: `All checks passed!`
  - `make validate-fixtures`: passed

### Full-Suite Baseline After Final Closeout Pass

- `poetry run pytest -n auto ml --maxfail=0`
- Result: `8158 passed, 85 skipped, 1 xfailed` (0 failures)

### Strict-Policy Alignment Status

- Strict-policy fallout clusters A/B/C remain resolved.
- No strict-policy regressions observed in focused suites, strict lanes, or full-suite execution.
