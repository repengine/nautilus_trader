# ML API Analysis Report

## 🔴 Undocumented Public APIs (Priority 1)
These should have docstrings:
- `ml._imports.generate_latest`
- `ml.actors.actor_services.ActorServices`
- `ml.actors.base.NautilusActor`
- `ml.actors.base.NautilusData`
- `ml.cli.events_consumer.main`
- `ml.cli.events_consumer.parse_args`
- `ml.cli.feature_backfill_cli.main`
- `ml.cli.feature_backfill_cli.run`
- `ml.cli.health.main`
- `ml.cli.ingest_backfill.main`

## 🟡 Refactoring Candidates (Priority 2)
- **ml.core.integration.MLIntegrationManager**: 24 methods
  - Consider splitting into smaller classes or using composition
- **ml.registry.model_registry.ModelRegistry**: 34 methods
  - Consider splitting into smaller classes or using composition

## 🟠 API Stability Issues
APIs with version indicators:
- `ml.scripts.populate_alternative_data.NewsSentimentLoader.fetch_news_sentiment`
- `ml.scripts.populate_l2_efficient.merge_new_with_existing`
- `ml.scripts.sanity_check.check_legacy_schema_refs`
- `ml.stores.partition_manager.PartitionManager.cleanup_old_partitions`
- `ml.tests.contracts.test_store_schemas.TestSchemaEvolution.test_feature_schema_allows_new_columns`
- `ml.tests.integration.test_scheduler_databento.TestDataSchedulerIntegration.test_clean_old_data`
- `ml.tests.tools.archive.conftest_backup.xgb_v1_model_path`
- `ml.tests.tools.archive.conftest_backup.xgb_v2_model_path`
- `ml.tests.tools.archive.conftest_old.xgb_v1_model_path`
- `ml.tests.tools.archive.conftest_old.xgb_v2_model_path`

## 💡 Potential Property Conversions
Consider using @property for these getters/setters:
- `ml.actors.base.BaseMLInferenceActor.get_health_status`
- `ml.actors.base.CircuitBreaker.get_stats`
- `ml.actors.base.HealthMonitor.get_success_rate`
- `ml.actors.base.HealthMonitor.get_uptime_seconds`
- `ml.actors.base.HealthMonitor.set_indicators_initialized`
- `ml.actors.base.HealthMonitor.set_model_loaded`
- `ml.actors.base.ModelLoader.get_model_version`
- `ml.actors.base.ONNXModelLoader.get_model_version`
- `ml.actors.signal.MLSignalActor.get_signal_statistics`
- `ml.actors.signal.ModelSwapper.set_current`
