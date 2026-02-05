# Public API Surface Compliance Plan

Goal: enforce domain-level public APIs via `__init__.py` while avoiding circular imports and optional-dependency failures.

## Checklist
- [x] Add lazy + typed re-exports in `ml/common/__init__.py` for:
  - decision metadata helpers
  - prediction surface helpers
  - resource monitor
  - symbol utils
  - watermark window helpers
- [x] Expand `ml/config/__init__.py` public surface:
  - export ingestion window configs/defaults
  - add lazy export for `FeatureDatasetMirrorConfig` (avoid eager `ml.orchestration`/`ml.stores` import)
- [x] Extend domain `__init__.py` surfaces:
  - `ml/actors/common/__init__.py` exports signal metadata helpers
  - `ml/strategies/common/__init__.py` lazy exports returns updater types
  - `ml/registry/common/__init__.py` exports `resolve_primary_keys` + `set_instrumentation_search_path`
  - `ml/stores/__init__.py` lazy exports mirror utilities
- [x] Normalize cross-domain imports to package-level APIs (no internal-package self-imports):
  - `ml.common.*` consumers → `from ml.common import …`
  - `ml.config.ingestion_windows` consumers → `from ml.config import …`
  - mirror config/utility consumers → `from ml.config import …` / `from ml.stores import …`
  - `ml.actors.common.signal_metadata` consumers → `from ml.actors.common import …`
  - `ml.strategies.common.returns_updater` consumers → `from ml.strategies.common import …`
  - `ml.registry.common.{sql_utils,manifest_defaults}` consumers → `from ml.registry.common import …`
- [x] Update affected tests to use the package-level APIs.
- [ ] (Optional) Decide whether to add `ml/training/event_driven/__init__.py` to formalize its public API.

## Validation (optional)
- [x] Run a focused import smoke check for the updated packages.
- [ ] Run any affected unit tests if import behavior changes.
