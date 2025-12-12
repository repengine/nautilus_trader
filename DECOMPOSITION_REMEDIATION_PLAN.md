# Decomposition Remediation Plan (Updated)

## Current Gaps (evidence)
- FeatureEngineer legacy shim remains for `compute_features` compatibility; main batch/online paths are component-backed and gated by the feature flag.

## Remediation Steps (prioritized)
1) **Promote orchestrator components to primary paths and remove structural stubs**
   - DONE: Delete dataset builder Phase0 helper stubs so API/CLI logic is the only path; unit tests now cover API success, CLI fallback, and metadata validation.  
   - DONE: Config resolver structural stubs removed; real window bounds, instrument resolution, and market input logic active.  
   - DONE: Registry synchronizer placeholders removed; real manifest sync/guardrails now primary.  
   - DONE: Runtime attacher structural shims removed; validators enforced via real flow.
2) **Cut over FeatureEngineer to components**
   - DONE: Gate legacy instantiation behind the feature flag; batch/online computation flows use the component calculator path with parity tests relying on standalone legacy instances.  
   - DONE: Legacy compatibility retained only for `compute_features` shim; legacy used lazily for parity/performance while primary APIs stay component-backed.
3) **Orchestrator facade default behavior**
   - DONE: Default export stays on component-backed facade; legacy path only when `ML_USE_LEGACY_ORCHESTRATOR` is set.  
   - DONE: Integration smoke asserts component alias by default and feature-flag switch behavior without legacy delegation leaks.
4) **Validation and cleanup**
   - Run `poetry run mypy ml --strict` and `poetry ruff check ml` after removing placeholders to catch interface drift.  
   - Re-run orchestration unit/integration suites with feature flags off/on; add coverage targets for new component code to keep ML ≥90%.

## Owner Handoff Notes
- When replacing placeholder helpers, delete or relocate the Phase0 “structural compatibility” comments to avoid future shadowing.  
- Keep legacy imports behind feature flags only; remove incidental legacy instantiation from facades once parity is validated.  
- Track progress in `reports/DECOMPOSITION_AUDIT_FINDINGS.md` and the migration runbook once each component flips to primary.
