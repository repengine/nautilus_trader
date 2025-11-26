# Remaining validate-nautilus-patterns follow-ups (non-god-class scope)

Source: `validate-nautilus-patterns.log` (latest run in repo). This lists only items **not** covered by the god-class refactors in `REFACTORING_PLAN.md` and ignores `ml/actors/base.py` / `ml/actors/base_legacy.py`.

## Scope exclusions
- God classes slated for replacement (per `REFACTORING_PLAN.md`): `pipeline_orchestrator*`, `config_loader.py`, `dataset_builder*`, `ingestion_coordinator.py`, `data_store*`, `data_writer.py`, `data_processor.py`, `data_scheduler*`, `tft_dataset_builder*`, `model_registry*`, `data_registry*`, `feature_registry.py`, `feature_store*`, `entrypoint_pipeline.py`, `entrypoint_actor.py`, `entrypoint_strategy.py`.
- Actors base classes: `ml/actors/base.py`, `ml/actors/base_legacy.py`.

## Completed in this pass
- `ml/actors/components/model.py` — removed joblib support, eliminated `open()` usage in `_load_json_model`, and switched version hashing to metadata-only (no file I/O).
- Config immutability: `ml/cli/streaming_training_runner.py:111` (`RunnerConfig`) now `frozen=True`.
- Strict store protocols: `ml/stores/components/store_operations.py` constructor uses strict store protocols.
- Dashboard EventStatus compliance: switched `success` strings to `EventStatus` in `ml/dashboard/metrics_snapshot.py`, `ml/dashboard/service.py`, and `ml/dashboard/components/pipeline_integration.py`.
- Frozen config: `TopicThrottleConfig` in `ml/actors/ml_domain_events.py` is now an immutable dataclass.

## Latest run
- Command: `make validate-nautilus-patterns > patterns-iter2.log 2>&1`
- Remaining pattern errors (excluding bases/god-classes): **none**; only `ml/actors/base.py` and `ml/actors/base_legacy.py` remain.

## Remaining high-priority fixes
- Re-run `make validate-nautilus-patterns` to confirm the security rule in `ml/actors/components/model.py` is cleared and adjust if any new hits appear.

## Logging setup (ml-no-basicConfig-outside-entrypoints)
Replace `logging.basicConfig` with the shared logging configuration in:
- `ml/cli/streaming_training_runner.py:2372`
- `ml/examples/scheduler_with_features.py:29`
- `ml/examples/scheduler_with_metrics.py:76`
- `ml/examples/tft_with_feature_store.py:31`
- `ml/monitoring/integration_examples_updated.py:491`
- `ml/monitoring/scripts/export_dashboards.py:36`
- `ml/monitoring/scripts/import_dashboards.py:51`
- `ml/monitoring/scripts/validate_config.py:40`
- `ml/monitoring/scripts/validate_dashboards.py:36`
- `ml/scripts/ensure_peer_logits_metadata.py:173`
- `ml/scripts/migrate_tier1_to_catalog.py:18`
- `ml/stores/schedule_partitions.py:48`
(Ignore `ml/data/scheduler.py` and other excluded god-class files.)

## Exception logging hygiene (ml-require-exc-info-in-except-logs)
Add `exc_info=True` (and relevant context) inside except blocks for:
- Actors components: `ml/actors/components/features.py`, `ml/actors/components/model.py`, `ml/actors/components/model_warmup.py`, `ml/actors/components/registry.py`, `ml/actors/components/store_operations.py`
- Actors: `ml/actors/multi_signal.py`, `ml/actors/signal.py`, `ml/actors/signal_facade_impl.py`
- CLI: `ml/cli/populate_universe.py`
- Common/core: `ml/common/security.py`, `ml/core/db_engine.py`
- Dashboard: `ml/dashboard/services/features_service.py`, `ml/dashboard/services/terminal_service.py`
- Data layer: `ml/data/collection_coordinator.py`, `ml/data/earnings/edgar_fetcher.py`, `ml/data/loaders/fred_loader.py`, `ml/data/providers/{calendar,events,factory,metadata}.py`, `ml/data/sources/{calendar,metadata}.py`
- Deployment: `ml/deployment/entrypoint_mock.py`
- Examples: `ml/examples/scheduler_with_features.py`, `ml/examples/simple_ml_actor.py`, `ml/examples/tft_with_feature_store.py`
- Monitoring: `ml/monitoring/grafana_client.py`, `ml/monitoring/integration_examples_updated.py`, `ml/monitoring/scripts/{export_dashboards,import_dashboards,validate_config,validate_dashboards}.py`, `ml/monitoring/server.py`
- Orchestration (non-god-class helpers): `ml/orchestration/scheduler.py`
- Stores: `ml/stores/components/store_operations.py`, `ml/stores/feature_computation.py`, `ml/stores/feature_table_manager.py`, `ml/stores/infrastructure.py`, `ml/stores/mixins.py`, `ml/stores/schedule_partitions.py`, `ml/stores/services/model_services.py`
- Strategies/tasks: `ml/strategies/base.py`, `ml/tasks/monitoring/coverage.py`, `ml/tasks/pipelines/runner.py`, `ml/tasks/training/quick.py`

## Silent excepts (ml-no-except-pass)
Replace `except Exception: pass/...` with at least debug logging and safe fallbacks in:
- `ml/common/circuit_breaker.py`
- `ml/orchestration/registry_synchronizer.py`
- `ml/orchestration/runtime_attacher.py`
- `ml/registry/artifacts.py`
- `ml/stores/services/strategy_services.py`

## Bandit B110 (try/except/pass)
- `ml/training/event_driven/worker.py:569` — log the fallback path (with context) instead of `pass`.

## Small cleanup (Vulture)
- Remove unused symbols flagged at: `ml/_imports.py:215`, `ml/data/loaders/alfred_loader.py:75-76`, `ml/training/event_driven/worker.py:582`.

## Helper: script to rescan the log with exclusions applied
Run from repo root to regenerate the above lists without the god-class files/base actors:

```bash
python - <<'PY'
import re
from pathlib import Path

log = Path("validate-nautilus-patterns.log").read_text().splitlines()
ignore = {
    "ml/actors/base.py",
    "ml/actors/base_legacy.py",
    "ml/orchestration/pipeline_orchestrator.py",
    "ml/orchestration/pipeline_orchestrator_facade.py",
    "ml/orchestration/config_loader.py",
    "ml/orchestration/dataset_builder.py",
    "ml/orchestration/ingestion_coordinator.py",
    "ml/data/tft_dataset_builder.py",
    "ml/data/tft_dataset_builder_legacy.py",
    "ml/data/scheduler.py",
    "ml/data/scheduler_legacy.py",
    "ml/stores/data_store.py",
    "ml/stores/data_store_facade.py",
    "ml/stores/data_writer.py",
    "ml/stores/data_processor.py",
    "ml/stores/feature_store.py",
    "ml/stores/feature_store_legacy.py",
    "ml/registry/model_registry.py",
    "ml/registry/model_registry_facade.py",
    "ml/registry/data_registry.py",
    "ml/registry/data_registry_legacy.py",
    "ml/registry/feature_registry.py",
    "ml/registry/manifest_manager.py",
    "ml/registry/strategy_registry.py",
    "ml/registry/model_persistence.py",
    "ml/deployment/entrypoint_pipeline.py",
    "ml/deployment/entrypoint_actor.py",
    "ml/deployment/entrypoint_strategy.py",
}

file_rules: dict[str, set[str]] = {}
current_file: str | None = None
for line in log:
    file_match = re.match(r"\\s+(ml/[^\\s]+\\.py)", line)
    if file_match:
        current_file = file_match.group(1)
        continue
    rule_match = re.search(r"tools\\.semgrep\\.(ml-[^\\s]+)", line)
    if rule_match and current_file and not any(p in current_file for p in ignore):
        file_rules.setdefault(current_file, set()).add(rule_match.group(1))

for path in sorted(file_rules):
    print(f\"{path}: {', '.join(sorted(file_rules[path]))}\")
PY
```

Edit the `ignore` set if more files graduate to the refactored stack.
