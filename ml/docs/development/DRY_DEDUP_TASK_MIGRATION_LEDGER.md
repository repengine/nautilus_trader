# DRY Task Migration Ledger

## Purpose
Module-by-module ownership map for retiring `ml/tasks/**` as a business-logic layer.

Rules applied:
- Domain and shared logic move to `ml/<domain>/**`, `ml/<domain>/common/**`, or `ml/common/**`.
- `ml/cli/**` remains the only argument-parsing/process-boundary layer.
- `ml/tasks/**` Python module shims are fully retired.

Status legend:
- `planned`: destination agreed, move not started.
- `in_progress`: actively migrating/splitting.
- `done`: canonical module in place and task module reduced to shim.

Ownership mode legend:
- `move`: relocate to a new canonical module with no strong existing owner.
- `merge`: absorb into existing domain/common owner.
- `split+merge`: split CLI/process logic from service logic, then merge service into existing owner(s).
- `merge/retire`: merge if still needed, otherwise deprecate and remove.
- `retire`: remove after import migration.

## Ledger (All Python Modules Under `ml/tasks/`)
| Source module | Canonical owner(s) | Ownership mode | Reasoning (move vs merge/takeover) | Transitional shim plan | Status |
| --- | --- | --- | --- | --- | --- |
No remaining `ml/tasks/**/*.py` modules.

## Notes
- This ledger is the Phase 0 ownership baseline and must be updated in every migration PR.
- If a destination module name changes during implementation, update this ledger first in the same PR.
- Guardrail enforcement is implemented in `ml/tests/unit/common/test_tasks_migration_guardrails.py`; update that test when module statuses change.
- Retired package namespace shims removed in Phase 5: `ml/tasks/{__init__,caches,datasets,dev,ingest,monitoring,observability,pipelines,training}/__init__.py`.
- Retired leaf module shims removed in Phase 5: `ml/tasks/{caches/hydration,datasets/production,datasets/report,datasets/splits,datasets/tft,datasets/tft_cli,db,dev/sanity_check,ingest/alternative,ingest/backfill,ingest/l2,ingest/recent,ingest/supplementary,ingest/yahoo,monitoring/coverage,monitoring/health,observability/backfill,observability/flush,pipelines/runner,pipelines/scheduler,registry,training/hpo_tft,training/quick}.py`.
