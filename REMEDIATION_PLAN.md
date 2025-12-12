# ML Remediation Plan

Working plan for stabilizing the ML test suite and fixtures. Keep this file updated with progress and new findings.

## Guardrails & References
- Follow typing/lint/testing mandates (`poetry run mypy ml --strict`, `poetry ruff check ml`, focused pytest shards) [AGENTS.md].
- Prefer property/contract/metamorphic/pairwise tests; avoid config identity and enum isinstance/identity checks; add `@pytest.mark.serial` for DB tests; keep mocks test-scoped [ml/tests/docs/TESTING_STRATEGY.md], [ml/tests/docs/TEST_ANTI_PATTERNS.md].
- Use shared fixtures via pytest plug-in; lean on `fresh_store_bundle`, `cloned_test_database`, and isolation guidance [ml/tests/fixtures/FIXTURE_GUIDE.md], [db-fixture-optimizations.md].
- Schema registry, identifier templates, and dataset type defaults live in `ml/schema.py`; subscriptable type shims in `ml/ml_types.py` are the safe place for typing adjustments [ml/schema.py], [ml/ml_types.py].

## Work Plan (check off as completed)
- [x] **Fixture/DB harness stabilization**: Verify Postgres test instance on :5434, fix `fresh_store_bundle` UnboundLocal `engine`, ensure store/registry tests use cloned DB fixtures (no template writes) [ml/tests/fixtures/FIXTURE_GUIDE.md], [db-fixture-optimizations.md].
- [x] **Pandera schema/type error wave**: Resolve `TypeError: type 'Series' is not subscriptable` across contract/observability tests by adjusting Series typing/shims (via `ml/tests/fixtures/pandera.py` and schema imports) [ml/ml_types.py].
- [x] **Config/constructor gaps**: Restore `DatasetBuildConfig` defaults/overloads for orchestrator/discovery tests; ensure schema/workflow functions accept expected kwargs and maintain strict typing [AGENTS.md].
- [x] **Orchestrator/public API parity**: Add or expose orchestration methods (`apply_default_market_inputs`, `resolve_window_bounds_ns`, etc.) and align parity/feature-flag behavior; address missing mocks/fixtures in parity E2Es [ml/tests/docs/TESTING_STRATEGY.md].
- [x] **Registry/store integration fixes**: Address Postgres auth failures in feature/data store facades and ensure event payloads honor enums/topics (registry synchronizer hooks implemented: `_emit_feature_refresh_event`, `_ensure_dataset_registered`) [ml/tests/fixtures/FIXTURE_GUIDE.md].
- [x] **DataStore + Registry contract convergence**: Standardize DataStore stage/source/status coercion and bus payloads via centralized enums/helpers, fix Postgres lineage persistence to use a single atomic session, and extend schema registry to cover Tier‑1 feature-family schemas (earnings/macro/events/micro/L2) so coverage/config loaders validate. Validated with focused contracts/config shards plus `mypy`/`ruff` [ml/stores/data_store.py], [ml/registry/data_registry.py], [ml/schema.py].
- [x] **Message bus/topic standardization**: Ensure stores, actors, and dashboard routes build topics via `build_topic_for_stage`, honor env scheme/topic_prefix, publish best‑effort off hot paths with `exc_info=True`, and avoid raw Stage/Source/EventStatus strings. Validated with contracts, store publishing suites, and dashboard parity shards.
- [ ] **Feature parity & performance guardrails**: Fix `n_features`/feature count mismatches, zero-allocation buffer/view requirements, and perf benchmarks; keep hot-path allocations near-zero and wrap bus publish in try/except [AGENTS.md].
- [ ] **Targeted validation loop**: After each cluster, run `poetry run mypy ml --strict`, `poetry ruff check ml`, and focused pytest shards (contracts, fixtures, orchestrator, perf) to confirm regression scope [AGENTS.md], [ml/tests/docs/TESTING_STRATEGY.md].

## Immediate Next Steps (execution order)
- [x] Patch `fresh_store_bundle` UnboundLocal `engine` bug so DB fixtures are usable; re-run `pytest ml/tests/unit/fixtures/test_fresh_store_bundle.py -x` to confirm [ml/tests/fixtures/FIXTURE_GUIDE.md].
- [ ] Verify Postgres test credentials/connection on :5434; confirm `ml/tests/unit/stores/test_feature_store_facade.py` passes under `cloned_test_database` in aggregate runs, and migrate remaining store/registry/perf suites to clones or `EngineManager` patching (no template writes) [db-fixture-optimizations.md], [AGENTS.md].
- [x] Fix the Pandera `Series` typing issue by adjusting the shared typing shim (`ml/ml_types.py`) or schema imports so `Series[...]` is subscriptable; validate with `pytest ml/tests/contracts -x` (now green) [ml/ml_types.py].
- [x] Restore `DatasetBuildConfig` constructor defaults/overloads expected by orchestrator/discovery workflows; rerun `pytest ml/tests/integration/orchestration/test_schema_autofill.py -x` [AGENTS.md].
- [x] Add missing orchestrator facade/public methods (`apply_default_market_inputs`, `resolve_window_bounds_ns`, etc.) or expose wrappers to satisfy parity/e2e tests; validate with `pytest ml/tests/e2e/test_pipeline_orchestrator_e2e.py -k e2e_apply_default_market_inputs -x` [ml/tests/docs/TESTING_STRATEGY.md].
- [x] Triage scheduler/lookback helper and coverage config constant failures (DataScheduler binding, coverage dataset IDs) ahead of feature parity/perf guardrail fixes; restored lookback/ts helpers in `DataScheduler` and verified focused shards plus `mypy`/`ruff`.
- [ ] After the above, run `poetry run mypy ml --strict` and `poetry ruff check ml` to keep the baseline green before tackling the next cluster; follow with targeted pytest shards (stores/registry/perf) [AGENTS.md].

## Revised Next Steps
- Align Postgres-backed store/reg tests to `cloned_test_database`; fix auth failures in `ml/tests/unit/stores/test_feature_store_facade.py` and related integration suites on port 5434. DataStore facade integration now uses `cloned_test_database`, exposes earnings point-in-time helpers, and matches health/batch-size validation expectations.
- ✅ Registry synchronizer hooks `_emit_feature_refresh_event` and `_ensure_dataset_registered`; validate via `pytest ml/tests/integration/orchestration/test_registry_events.py -x`.
- Address feature parity/perf guardrails (feature counts, buffer views, zero-allocation) plus the scheduler/lookback/coverage constant cluster, and rerun hot-path/perf shards.
- Run full `poetry run mypy ml --strict` and `poetry ruff check ml` once the above clusters are addressed, followed by targeted pytest shards (stores/registry/perf).

## Baseline Pytest Output (for reference)
```
============================================================================= slowest 10 durations =============================================================================
22.11s call     tests/contracts/test_observability_pipeline_schemas.py::TestObservabilityPipelineIntegrationContracts::test_end_to_end_observability_contract
12.70s call     tests/unit/data/providers/test_factory.py::TestTransformProviderAdapter::test_adapter_handles_arbitrary_data_sizes
12.50s call     tests/integration/test_transform_provider_integration.py::TestTransformProviderIntegration::test_provider_scalability
10.61s call     tests/unit/data/providers/test_factory.py::TestProviderFactory::test_factory_handles_multiple_custom_providers
10.01s call     tests/unit/data/test_scheduler_targeted.py::test_collect_symbol_data_handles_timezone_aware_target
8.51s call     dashboard/tests/test_terminal_service.py::TestCommandHistory::test_history_max_size_enforced
7.74s call     tests/integration/deployment/test_deployment_integration.py::TestDeploymentIntegration::test_pipeline_with_stores_initialization
7.50s call     tests/unit/data/test_tft_dataset_builder_facade.py::TestFacadeHappyPath::test_e7_facade_pandas_output_mode
6.47s call     tests/contracts/test_data_store_routing_advanced.py::TestDataStoreValidation::test_prediction_bounds_property
5.78s call     tests/unit/stores/test_data_store_validation.py::TestPropertyBased::test_validation_consistency
=========================================================================== short test summary info ============================================================================
SKIPPED [1] ml/tests/unit/actors/test_facade_parity.py:1100: Test has structural issues: tries to manipulate _feature_store.schema directly and query custom schemas without proper table setup. Needs redesign with fresh_store_bundle fixture.
SKIPPED [1] ml/tests/integration/features/test_feature_calculator_facade_integration.py:41: Test design - implementation pending
SKIPPED [1] ml/tests/integration/features/test_feature_calculator_facade_integration.py:85: Test design - implementation pending
SKIPPED [1] ml/tests/integration/features/test_feature_calculator_facade_integration.py:116: Test design - implementation pending
SKIPPED [1] ml/tests/integration/features/test_feature_calculator_facade_integration.py:170: Test design - implementation pending
SKIPPED [1] ml/tests/integration/features/test_feature_calculator_facade_integration.py:236: Test design - implementation pending
SKIPPED [1] ml/tests/integration/features/test_feature_calculator_facade_integration.py:295: Test design - implementation pending
SKIPPED [3] ml/tests/integration/features/test_feature_calculator_facade_integration.py:376: Test design - implementation pending
SKIPPED [1] ml/tests/performance/test_feature_calculator_microbench.py:526: compute_features may not support DataFrames directly
SKIPPED [1] ml/tests/unit/actors/common/test_features.py:862: Requires PostgreSQL integration
SKIPPED [1] ml/tests/unit/actors/common/test_features.py:867: Requires PostgreSQL integration
SKIPPED [1] ml/tests/unit/actors/common/test_features.py:872: Requires PostgreSQL integration
SKIPPED [1] ml/tests/unit/actors/common/test_features.py:877: Requires PostgreSQL integration
SKIPPED [1] ml/tests/unit/actors/common/test_features.py:882: Requires PostgreSQL integration
SKIPPED [1] ml/tests/unit/features/common/test_feature_calculator.py:644: compute_features may not support DataFrames directly
SKIPPED [1] ml/tests/unit/training/teacher/test_tft_teacher_streaming.py:115: CUDA device required to verify device alignment
SKIPPED [1] ml/tests/fixtures/test_pollution_detection.py:121: PostgreSQL required for pool statistics test
XFAIL ml/tests/unit/features/test_feature_parity_fix.py::TestFeatureParityFix::test_trade_flow_features_computed_online - Legacy feature engineer does not support online trade flow features (WIP)
XPASS ml/tests/test_enum_comparison_patterns.py::TestEnumIsinstanceAntiPattern::test_isinstance_may_fail_in_parallel isinstance() fails in pytest-xdist parallel execution
ERROR ml/tests/metamorphic/test_feature_transforms.py::TestFeatureTransformMetamorphic::test_trade_flow_missing_trades_matches_ohlcv
ERROR ml/tests/integration/orchestration/test_schema_autofill.py::test_auto_fill_universe_populates_all_schemas - TypeError: DatasetBuildConfig.__init__() missing 3 required positional arguments: 'data_dir', 'symbols', and 'out_dir'
ERROR ml/tests/integration/orchestration/test_schema_autofill.py::test_auto_fill_schema_triggers_ingestion_workflow - TypeError: DatasetBuildConfig.__init__() missing 3 required positional arguments: 'data_dir', 'symbols', and 'out_dir'
ERROR ml/tests/integration/orchestration/test_schema_autofill.py::test_auto_fill_l2_populates_depth_and_mbp_schemas - TypeError: DatasetBuildConfig.__init__() missing 3 required positional arguments: 'data_dir', 'symbols', and 'out_dir'
ERROR ml/tests/unit/fixtures/test_fresh_store_bundle.py::test_fresh_store_bundle_isolation - UnboundLocalError: cannot access local variable 'engine' where it is not associated with a value
ERROR ml/tests/unit/fixtures/test_fresh_store_bundle.py::test_fresh_store_bundle_no_pollution - UnboundLocalError: cannot access local variable 'engine' where it is not associated with a value
ERROR ml/tests/unit/fixtures/test_fresh_store_bundle.py::test_fresh_store_bundle_consistency[1] - UnboundLocalError: cannot access local variable 'engine' where it is not associated with a value
ERROR ml/tests/unit/fixtures/test_fresh_store_bundle.py::test_fresh_store_bundle_consistency[2] - UnboundLocalError: cannot access local variable 'engine' where it is not associated with a value
ERROR ml/tests/unit/fixtures/test_fresh_store_bundle.py::test_fresh_store_bundle_consistency[3] - UnboundLocalError: cannot access local variable 'engine' where it is not associated with a value
ERROR ml/tests/unit/fixtures/test_fresh_store_bundle.py::test_fresh_store_bundle_attributes - UnboundLocalError: cannot access local variable 'engine' where it is not associated with a value
ERROR ml/tests/unit/stores/test_feature_store_facade.py::TestFacadeComponentWiring::test_facade_wires_all_components - sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection to server at "localhost" (127.0.0.1), port 5434 failed: FATAL:  password authentication failed for ...
ERROR ml/tests/unit/stores/test_feature_store_facade.py::TestFacadeComponentWiring::test_facade_initializes_engine - sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection to server at "localhost" (127.0.0.1), port 5434 failed: FATAL:  password authentication failed for ...
ERROR ml/tests/unit/stores/test_feature_store_facade.py::TestFacadeComponentWiring::test_facade_initializes_feature_engineer - sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection to server at "localhost" (127.0.0.1), port 5434 failed: FATAL:  password authentication failed for ...
ERROR ml/tests/unit/stores/test_feature_store_facade.py::TestFacadePublicAPI::test_facade_exposes_required_attributes - sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection to server at "localhost" (127.0.0.1), port 5434 failed: FATAL:  password authentication failed for ...
ERROR ml/tests/unit/stores/test_feature_store_facade.py::TestFacadeDelegation::test_facade_delegates_write_features_to_writer - sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection to server at "localhost" (127.0.0.1), port 5434 failed: FATAL:  password authentication failed for ...
ERROR ml/tests/unit/stores/test_feature_store_facade.py::TestFacadeDelegation::test_facade_delegates_get_training_data_to_reader - sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection to server at "localhost" (127.0.0.1), port 5434 failed: FATAL:  password authentication failed for ...
ERROR ml/tests/unit/stores/test_feature_store_facade.py::TestFacadeDelegation::test_facade_delegates_compute_realtime_to_computation - sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection to server at "localhost" (127.0.0.1), port 5434 failed: FATAL:  password authentication failed for ...
ERROR ml/tests/unit/stores/test_feature_store_facade.py::TestFacadeDelegation::test_facade_delegates_is_healthy_to_health - sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection to server at "localhost" (127.0.0.1), port 5434 failed: FATAL:  password authentication failed for ...
ERROR ml/tests/unit/stores/test_feature_store_facade.py::TestFacadeDelegation::test_facade_delegates_clear_features_to_health - sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection to server at "localhost" (127.0.0.1), port 5434 failed: FATAL:  password authentication failed for ...
ERROR ml/tests/unit/stores/test_feature_store_facade.py::TestFacadeDelegation::test_facade_delegates_flush - sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection to server at "localhost" (127.0.0.1), port 5434 failed: FATAL:  password authentication failed for ...
ERROR ml/tests/unit/stores/test_feature_store_facade.py::TestFacadeDelegation::test_facade_delegates_write_batch_to_writer - sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection to server at "localhost" (127.0.0.1), port 5434 failed: FATAL:  password authentication failed for ...
ERROR ml/tests/unit/stores/test_feature_store_facade.py::TestFacadeDelegation::test_facade_delegates_store_features_to_writer - sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection to server at "localhost" (127.0.0.1), port 5434 failed: FATAL:  password authentication failed for ...
ERROR ml/tests/unit/stores/test_feature_store_facade.py::TestFacadeDelegation::test_facade_delegates_get_latest_at_or_before_to_reader - sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection to server at "localhost" (127.0.0.1), port 5434 failed: FATAL:  password authentication failed for ...
ERROR ml/tests/unit/stores/test_feature_store_facade.py::TestFacadeDelegation::test_facade_delegates_compute_and_store_historical_to_computation - sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection to server at "localhost" (127.0.0.1), port 5434 failed: FATAL:  password authentication failed for ...
ERROR ml/tests/unit/stores/test_feature_store_facade.py::TestFacadeDelegation::test_facade_delegates_compute_historical_parallel_to_computation - sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection to server at "localhost" (127.0.0.1), port 5434 failed: FATAL:  password authentication failed for ...
ERROR ml/tests/unit/stores/test_feature_store_facade.py::TestRegistryIntegration::test_facade_set_data_registry - sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection to server at "localhost" (127.0.0.1), port 5434 failed: FATAL:  password authentication failed for ...
ERROR ml/tests/unit/stores/test_feature_store_facade.py::TestRegistryIntegration::test_facade_get_data_registry - sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection to server at "localhost" (127.0.0.1), port 5434 failed: FATAL:  password authentication failed for ...
ERROR ml/tests/unit/stores/test_feature_store_facade.py::TestEdgeCases::test_facade_buffer_alias - sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection to server at "localhost" (127.0.0.1), port 5434 failed: FATAL:  password authentication failed for ...
ERROR ml/tests/integration/orchestration/test_facade_integration.py::TestDataStoreIntegration::test_data_store_write_read_cycle
ERROR ml/tests/integration/orchestration/test_facade_integration.py::TestDataStoreIntegration::test_data_store_time_range_query
ERROR ml/tests/integration/orchestration/test_discovery_workflows.py::test_service_discovery_workflow - TypeError: Unexpected keyword argument 'schema'
ERROR ml/tests/integration/orchestration/test_discovery_workflows.py::test_resource_discovery_workflow - TypeError: DatasetBuildConfig.__init__() missing 2 required positional arguments: 'data_dir' and 'out_dir'
ERROR ml/tests/integration/orchestration/test_discovery_workflows.py::test_schema_discovery_and_mapping_workflow - TypeError: DatasetBuildConfig.__init__() missing 2 required positional arguments: 'data_dir' and 'out_dir'
ERROR ml/tests/unit/stores/test_feature_store_facade.py::TestEdgeCases::test_facade_with_none_feature_config - sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection to server at "localhost" (127.0.0.1), port 5434 failed: FATAL:  password authentication failed for ...
ERROR ml/tests/unit/stores/test_feature_store_facade.py::TestEdgeCases::test_facade_with_ml_feature_config - sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection to server at "localhost" (127.0.0.1), port 5434 failed: FATAL:  password authentication failed for ...
FAILED ml/tests/performance/test_ml_hot_path_benchmarks.py::TestEndToEndBenchmarks::test_signal_generation_e2e_latency - NameError: name 'test_database' is not defined
FAILED ml/tests/integration/test_registry_store_l2_integration.py::TestL2L3RegistryStoreIntegration::test_feature_store_computes_l2_features - AssertionError: assert False is True
FAILED ml/tests/integration/test_end_to_end_pipeline.py::TestEndToEndPipeline::test_online_feature_parity - AttributeError: 'FeatureEngineer' object has no attribute 'n_features'
FAILED ml/tests/integration/test_feature_parity.py::TestFeatureParity::test_feature_engineer_batch_vs_online_parity - AssertionError: Feature parity violation! Max difference: 1.2110353708267212
FAILED ml/tests/integration/test_feature_parity.py::TestFeatureParity::test_feature_versioning - AssertionError: Feature versions must change when configuration changes
FAILED ml/tests/integration/registry/test_data_registry_postgres_backend_smoke.py::test_data_registry_postgres_backend_smoke - sqlalchemy.exc.ProgrammingError: (psycopg2.errors.UndefinedTable) relation "ml_data_events" does not exist
FAILED ml/tests/integration/test_postgres_integration.py::test_migrations_applied - AssertionError: Table ml_feature_values should exist after migrations
FAILED ml/tests/integration/test_feature_store_integration.py::TestFeatureStoreIntegration::test_feature_store_config_propagation - AssertionError: FeatureConfig mismatch: {'lookback_window': 120, 'indicators': None, 'feature_names': None, 'normalize_features': True, 'fill_missing_with': 0.0, 'average_...
FAILED ml/tests/contracts/test_store_schemas.py::TestStoreSchemaContracts::test_watermark_schema_validation - TypeError: type 'Series' is not subscriptable
FAILED ml/tests/contracts/test_store_schemas.py::TestSchemaEvolution::test_metric_schema_extensibility - TypeError: type 'Series' is not subscriptable
FAILED ml/tests/contracts/test_store_schemas.py::TestStoreSchemaContracts::test_feature_input_schema_validation - TypeError: type 'Series' is not subscriptable
FAILED ml/tests/contracts/test_store_schemas.py::TestStoreSchemaContracts::test_prediction_schema_validation - TypeError: type 'Series' is not subscriptable
FAILED ml/tests/contracts/test_store_schemas.py::TestStoreSchemaContracts::test_signal_schema_validation - TypeError: type 'Series' is not subscriptable
FAILED ml/tests/contracts/test_store_schemas.py::TestStoreSchemaContracts::test_cross_store_consistency - TypeError: type 'Series' is not subscriptable
FAILED ml/tests/contracts/test_observability_persisted_schemas.py::test_persisted_jsonl_conforms_to_contracts - TypeError: type 'Series' is not subscriptable
FAILED ml/tests/contracts/test_databento_fixtures_contracts.py::test_tbbo_fixture_contract - TypeError: type 'Series' is not subscriptable
FAILED ml/tests/contracts/test_databento_fixtures_contracts.py::test_trades_fixture_contract - TypeError: type 'Series' is not subscriptable
FAILED ml/tests/contracts/test_databento_fixtures_contracts.py::test_mbp10_fixture_contract - TypeError: type 'Series' is not subscriptable
FAILED ml/tests/contracts/test_streaming_payloads.py::test_calendar_event_payload_schema - TypeError: type 'Series' is not subscriptable
FAILED ml/tests/contracts/test_watermark_event_contracts.py::test_watermark_progression_valid - TypeError: type 'Series' is not subscriptable
FAILED ml/tests/contracts/test_watermark_event_contracts.py::test_watermark_progression_regression_fails - TypeError: type 'Series' is not subscriptable
FAILED ml/tests/contracts/test_observability_pipeline_schemas.py::TestPipelineLineageContracts::test_pipeline_lineage_schema_validation - TypeError: type 'Series' is not subscriptable
FAILED ml/tests/contracts/test_observability_pipeline_schemas.py::TestPipelineLineageContracts::test_completion_timestamp_consistency_validation - TypeError: type 'Series' is not subscriptable
FAILED ml/tests/contracts/test_observability_pipeline_schemas.py::TestLatencyTrackingContracts::test_latency_watermark_schema_validation - TypeError: type 'Series' is not subscriptable
FAILED ml/tests/contracts/test_observability_pipeline_schemas.py::TestLatencyTrackingContracts::test_invalid_latency_timestamps_rejected - TypeError: type 'Series' is not subscriptable
FAILED ml/tests/contracts/test_observability_pipeline_schemas.py::TestLatencyTrackingContracts::test_stage_latency_consistency_check - TypeError: type 'Series' is not subscriptable
FAILED ml/tests/contracts/test_observability_pipeline_schemas.py::TestMetricsCollectionContracts::test_metrics_collection_schema_validation - TypeError: type 'Series' is not subscriptable
FAILED ml/tests/contracts/test_observability_pipeline_schemas.py::TestMetricsCollectionContracts::test_metric_type_consistency_validation - TypeError: type 'Series' is not subscriptable
FAILED ml/tests/contracts/test_observability_pipeline_schemas.py::TestMetricsCollectionContracts::test_negative_metric_values_rejected - TypeError: type 'Series' is not subscriptable
FAILED ml/tests/contracts/test_observability_pipeline_schemas.py::TestEventCorrelationContracts::test_event_correlation_schema_validation - TypeError: type 'Series' is not subscriptable
FAILED ml/tests/contracts/test_observability_pipeline_schemas.py::TestEventCorrelationContracts::test_root_event_consistency_validation - TypeError: type 'Series' is not subscriptable
FAILED ml/tests/contracts/test_observability_pipeline_schemas.py::TestHealthScoreContracts::test_health_score_schema_validation - TypeError: type 'Series' is not subscriptable
FAILED ml/tests/contracts/test_observability_pipeline_schemas.py::TestHealthScoreContracts::test_invalid_health_score_range_rejected - TypeError: type 'Series' is not subscriptable
FAILED ml/tests/contracts/test_observability_pipeline_schemas.py::TestObservabilityPipelineIntegrationContracts::test_end_to_end_observability_contract - TypeError: type 'Series' is not subscriptable
FAILED ml/tests/contracts/test_event_bus_contracts.py::TestEventBusContracts::test_ml_data_event_schema_validation - TypeError: type 'Series' is not subscriptable
FAILED ml/tests/contracts/test_event_bus_contracts.py::TestEventBusContracts::test_ml_registry_event_schema_validation - TypeError: type 'Series' is not subscriptable
FAILED ml/tests/unit/orchestration/test_pipeline_orchestrator_facade.py::TestComponentDelegation::test_run_pre_ingestion_delegates_to_ingestion_coordinator - TypeError: object of type 'Mock' has no len()
FAILED ml/tests/unit/orchestration/test_pipeline_orchestrator_facade.py::TestComponentDelegation::test_backfill_binding_delegates_to_ingestion_coordinator - TypeError: 'Mock' object is not iterable
FAILED ml/tests/unit/orchestration/test_pipeline_orchestrator_facade.py::TestComponentDelegation::test_backfill_coverage_delegates_with_policy - TypeError: int() argument must be a string, a bytes-like object or a real number, not 'Mock'
FAILED ml/tests/unit/orchestration/test_pipeline_orchestrator_facade.py::TestComponentDelegation::test_train_teacher_delegates_to_training_coordinator - ValueError: Dataset metadata must include dataset_id before teacher training
FAILED ml/tests/unit/orchestration/test_pipeline_orchestrator_facade.py::TestComponentDelegation::test_distill_student_delegates_to_training_coordinator - KeyError: 'X_train is not a file in the archive'
FAILED ml/tests/unit/orchestration/test_pipeline_orchestrator_facade.py::TestComponentDelegation::test_run_delegates_pipeline_stages - FileNotFoundError: Dataset metadata missing at /tmp/test_output/dataset_metadata.json
FAILED ml/tests/unit/orchestration/test_pipeline_orchestrator_facade.py::TestComponentDelegation::test_run_training_only_delegates_correctly - ValueError: Invalid vintage_policy in metadata: REAL_TIME
FAILED ml/tests/unit/orchestration/test_orchestrator_parity.py::TestMethodParity::test_distill_student_disabled_parity - TypeError: MLPipelineOrchestrator.distill_student() takes 2 positional arguments but 4 were given
FAILED ml/tests/unit/orchestration/test_orchestrator_parity.py::TestFeatureFlagBehavior::test_feature_flag_default_uses_component_mode - AssertionError: assert 'component-based' == 'component_based'
FAILED ml/tests/property/test_feature_calculator_properties.py::TestFeatureCalculatorProperties::test_features_bounded_invariant - AssertionError: Volume ratios should be positive after warmup
FAILED ml/tests/facades/test_feature_engineer_error_handling.py::TestModeValidationErrors::test_calculate_features_with_invalid_mode_raises - Failed: DID NOT RAISE <class 'ValueError'>
FAILED ml/tests/facades/test_feature_engineer_error_handling.py::TestModeValidationErrors::test_calculate_features_with_empty_mode_raises - Failed: DID NOT RAISE <class 'ValueError'>
FAILED ml/tests/e2e/test_orchestrator_smoke.py::TestFeatureFlagParity::test_build_dataset_parity - AttributeError: 'MLPipelineOrchestrator' object attribute '_capture_cli_build_artifacts' is read-only
FAILED ml/dashboard/tests/test_streaming_monitor.py::test_dashboard_streaming_monitor_tracks_events - KeyError: 'telemetry'
FAILED ml/dashboard/tests/test_streaming_monitor.py::test_streaming_state_endpoint - assert 404 == 200
FAILED ml/tests/performance/test_feature_calculator_microbench.py::TestFeatureCalculatorPerformance::test_calculate_mid_return_features_microbench - AssertionError: P99 1651.7 μs exceeds 800 μs threshold
FAILED ml/tests/validation/system_validation_smoke_test.py::test_component_progressive_fallback - AssertionError: assert 5 == 4
FAILED ml/tests/unit/core/test_health_typeddict.py::test_typeddict_enables_ide_autocomplete - ImportError: cannot import name 'ComponentHealthStatus' from 'ml.core.integration' (/home/nate/projects/nautilus_trader/ml/core/integration.py)
FAILED ml/tests/unit/core/test_health_typeddict.py::test_aggregate_health_with_unhealthy_components - assert True is False
FAILED ml/tests/unit/actors/common/test_model.py::test_model_loading_from_registry - AssertionError: assert False
FAILED ml/tests/unit/actors/common/test_model.py::test_hot_reload_timer_scheduling - AssertionError: assert False
FAILED ml/tests/unit/actors/common/test_model.py::test_manifest_warm_up_when_enabled - AttributeError: immutable type: 'MLActorConfig'
FAILED ml/tests/integration/orchestration/test_registry_events.py::test_feature_refresh_event_emitted_to_message_bus - AttributeError: 'RegistrySynchronizer' object has no attribute '_emit_feature_refresh_event'
FAILED ml/tests/integration/orchestration/test_registry_events.py::test_dataset_registered_event_emitted - AttributeError: 'RegistrySynchronizer' object has no attribute '_ensure_dataset_registered'
FAILED ml/tests/integration/orchestration/test_registry_events.py::test_event_payload_contains_metadata - AttributeError: 'RegistrySynchronizer' object has no attribute '_emit_feature_refresh_event'
FAILED ml/tests/unit/orchestration/common/test_stage_controller.py::TestStageControllerPromotions::test_promotions_called_after_successful_training - AssertionError: assert (False or False)
FAILED ml/tests/e2e/test_pipeline_orchestrator_e2e.py::TestE2EConfigResolution::test_e2e_apply_default_market_inputs - AttributeError: 'MLPipelineOrchestrator' object has no attribute 'apply_default_market_inputs'
FAILED ml/tests/e2e/test_pipeline_orchestrator_e2e.py::TestE2EConfigResolution::test_e2e_collect_symbol_map - AttributeError: 'MLPipelineOrchestrator' object has no attribute 'collect_symbol_map'
FAILED ml/tests/e2e/test_pipeline_orchestrator_e2e.py::TestE2EConfigResolution::test_e2e_compute_window_start - AttributeError: 'MLPipelineOrchestrator' object has no attribute 'compute_window_start_iso'
FAILED ml/tests/e2e/test_pipeline_orchestrator_e2e.py::TestE2EConfigResolution::test_e2e_resolve_window_bounds - AttributeError: 'MLPipelineOrchestrator' object has no attribute 'resolve_window_bounds_ns'. Did you mean: '_resolve_window_bounds_ns'?
FAILED ml/tests/e2e/test_pipeline_orchestrator_e2e.py::TestE2EConfigResolution::test_e2e_discover_market_inputs - AttributeError: 'MLPipelineOrchestrator' object has no attribute 'resolve_window_bounds_ns'. Did you mean: '_resolve_window_bounds_ns'?
FAILED ml/tests/e2e/test_pipeline_orchestrator_e2e.py::TestE2EConfigResolution::test_e2e_binding_resolution - AttributeError: 'MLPipelineOrchestrator' object has no attribute 'resolve_window_bounds_ns'. Did you mean: '_resolve_window_bounds_ns'?
FAILED ml/tests/e2e/test_pipeline_orchestrator_e2e.py::TestE2EConfigResolution::test_e2e_binding_resolution_with_config - AttributeError: 'MLPipelineOrchestrator' object has no attribute 'filter_candidate_bindings'. Did you mean: '_filter_candidate_bindings'?
FAILED ml/tests/e2e/test_pipeline_orchestrator_e2e.py::TestE2EConfigResolution::test_e2e_prepare_dataset_config - AttributeError: 'MLPipelineOrchestrator' object has no attribute 'prepare_dataset_config'. Did you mean: '_prepare_dataset_config'?
FAILED ml/tests/e2e/test_pipeline_orchestrator_e2e.py::TestE2EComponentIntegration::test_e2e_full_configuration_pipeline - AttributeError: 'MLPipelineOrchestrator' object has no attribute 'apply_default_market_inputs'
FAILED ml/tests/e2e/test_pipeline_orchestrator_e2e.py::TestE2EHealthMonitoring::test_e2e_health_status_all_components - AssertionError: assert 'legacy' == 'component_based'
FAILED ml/tests/e2e/test_pipeline_orchestrator_e2e.py::TestE2ELegacyComponentParity::test_e2e_config_resolution_parity - NameError: name 'mock_data_registry' is not defined
FAILED ml/tests/e2e/test_pipeline_orchestrator_e2e.py::TestE2ELegacyComponentParity::test_e2e_window_bounds_parity - NameError: name 'mock_data_registry' is not defined
FAILED ml/tests/e2e/test_pipeline_orchestrator_e2e.py::TestE2ELegacyComponentParity::test_e2e_health_status_structure_parity - NameError: name 'mock_data_registry' is not defined
FAILED ml/tests/e2e/test_pipeline_orchestrator_e2e.py::TestE2EErrorHandling::test_e2e_invalid_window_bounds_handled - AttributeError: 'MLPipelineOrchestrator' object has no attribute 'resolve_window_bounds_ns'. Did you mean: '_resolve_window_bounds_ns'?
FAILED ml/tests/unit/dashboard/common/test_pipeline_integration.py::test_get_pipeline_progress_not_found - AssertionError: assert 'deferred' == 'not_found'
FAILED ml/tests/unit/dashboard/common/test_pipeline_integration.py::test_get_pipeline_progress_unavailable - AssertionError: assert 'deferred' == 'unavailable'
FAILED ml/tests/unit/dashboard/common/test_pipeline_integration.py::test_get_pipeline_progress_error - AssertionError: assert 'failed' == 'error'
FAILED ml/tests/unit/core/test_create_data_store_signature.py::TestCreateDataStoreImportStyle::test_no_dynamic_import - AssertionError: Function should use direct import of DataStore. Expected pattern: from ml.stores.data_store import DataStore
FAILED ml/tests/unit/core/test_create_integrated_actor_generic.py::TestCreateIntegratedActorSignature::test_return_type_not_object - AssertionError: Return type should not be generic object type, got object. Expected generic TypeVar (e.g., ActorT). Full type: <class 'object'>
FAILED ml/tests/unit/core/test_create_integrated_actor_generic.py::TestCreateIntegratedActorSignature::test_actor_class_param_not_type_any - AssertionError: actor_class should use generic TypeVar (ActorT), not Any. Got type[typing.Any]
FAILED ml/tests/unit/core/test_create_integrated_actor_generic.py::TestCreateIntegratedActorSignature::test_typevar_defined_at_module_level - AssertionError: ActorT TypeVar should be defined at module level. Expected: ActorT = TypeVar('ActorT') after imports.
FAILED ml/tests/unit/core/test_create_integrated_actor_generic.py::TestCreateIntegratedActorSignature::test_typevar_imported_from_typing - AssertionError: TypeVar should be imported from typing module. Expected pattern: from typing import ..., TypeVar, ...
FAILED ml/tests/integration/stores/test_data_store_facade.py::TestDelegationMapping::test_read_methods_delegate_to_data_reader - AttributeError: 'DataStoreFacade' object has no attribute 'get_earnings_actuals_at_or_before'
FAILED ml/tests/integration/stores/test_data_store_facade.py::TestHealthAndMetrics::test_get_health_status_includes_all_components - AssertionError: assert 'feature_store' in {'checked_at': 1765303409141001744, 'circuit_breakers_open': [], 'components': {'data_registry': {'details': <MagicMoc...ble'}, '...
FAILED ml/tests/integration/stores/test_data_store_facade.py::TestConfigurationValidation::test_validate_configuration_checks_batch_size - AssertionError: assert 'batch_size must be positive' in []
FAILED ml/tests/contracts/test_fallback_metrics_contracts.py::test_fallback_activation_emits_metric - UnboundLocalError: cannot access local variable 'Path' where it is not associated with a value
FAILED ml/tests/performance/test_streaming_persistence_microbench.py::test_streaming_scaling_regression - KeyError: 'summary'
FAILED ml/tests/performance/test_hot_path_fixes.py::TestHotPathFixes::test_feature_engineer_returns_view_not_copy - AssertionError: calculate_features_online should return a view of feature_buffer for zero allocation
FAILED ml/tests/performance/test_parity_buffer_guardrails.py::TestFeatureComputationGuardrails::test_feature_computation_zero_allocations - AssertionError: ❌ Feature computation allocated 16424 bytes (164.2 per call), expected near-zero. Top allocations: [<StatisticDiff traceback=<Traceback (<Frame filename='...
FAILED ml/tests/performance/test_parity_buffer_guardrails.py::TestBufferReuseGuardrails::test_feature_buffer_reuse - AssertionError: Features should be a view of the pre-allocated buffer
FAILED ml/tests/performance/test_zero_allocation.py::TestZeroAllocationHotPath::test_feature_engineer_returns_buffer_view - AssertionError: calculate_features_online should return a view of feature_buffer
FAILED ml/tests/performance/test_zero_allocation.py::TestZeroAllocationHotPath::test_hot_path_memory_stability - AssertionError: Hot path allocated 17016 bytes, expected near zero. Allocations: [<StatisticDiff traceback=<Traceback (<Frame filename='/home/nate/projects/nautilus_trader...
FAILED ml/tests/unit_tests/actors/test_multi_signal_actor.py::test_infer_batch_falls_back_to_per_row_on_ort_error - TypeError: debug() got an unexpected keyword argument 'exc_info'
FAILED ml/tests/unit/tasks/caches/test_hydration.py::test_hydrate_micro_caches_builds_partition - AssertionError: assert 'midprice' in ['timestamp']
FAILED ml/tests/unit/tasks/caches/test_hydration.py::test_hydrate_l2_caches_handles_symbol_with_venue - AssertionError: assert 'midprice' in ['timestamp']
FAILED ml/tests/unit/core/test_integration_event_ingestion.py::test_ingest_events_uses_data_store_when_provided - assert []
FAILED ml/tests/unit/observability/test_db_persistor.py::test_db_persistor_writes_and_validates - TypeError: type 'Series' is not subscriptable
FAILED ml/tests/unit/features/test_earnings_features_metrics.py::test_compute_surprise_batch_with_metrics - assert 0 > 0
FAILED ml/tests/unit/features/test_feature_parity_fix.py::TestFeatureParityFix::test_feature_count_parity_microstructure_config - AssertionError: Expected 33 features for microstructure config, got 26
FAILED ml/tests/unit/features/test_feature_parity_fix.py::TestFeatureParityFix::test_feature_count_parity_full_config - AssertionError: Expected 37 features for full config, got 26
FAILED ml/tests/unit/features/test_feature_parity_fix.py::TestFeatureParityFix::test_microstructure_features_computed_online - AssertionError: Expected 33 features, got 26
FAILED ml/tests/unit/features/test_feature_parity_fix.py::TestFeatureParityFix::test_regression_scaler_dimension_mismatch - AssertionError: Expected 37 training features, got 26
FAILED ml/tests/unit/features/test_l2_aggregate.py::test_aggregate_l2_minute_pl_empty_schema - AssertionError: assert ('timestamp',) == ('timestamp',...ce_top3', ...)
FAILED ml/tests/unit/features/test_microstructure.py::test_aggregate_microstructure_minute_pl_sorts_inputs - polars.exceptions.InvalidOperationError: argument in operation 'group_by_dynamic' is not sorted, please sort the 'expr/series/column' first
FAILED ml/tests/unit/features/test_microstructure.py::test_aggregate_microstructure_drops_null_timestamps - polars.exceptions.InvalidOperationError: argument in operation 'group_by_dynamic' is not sorted, please sort the 'expr/series/column' first
FAILED ml/tests/unit/features/test_microstructure.py::test_micro_aggregator_resolves_symbol_with_venue - assert not True
FAILED ml/tests/unit/features/test_microstructure.py::test_micro_aggregator_prefers_latest_file - AssertionError: assert 10.0 > 20.0
FAILED ml/tests/unit/features/test_feature_validation.py::TestFeatureParityValidator::test_validator_initialization_default - AssertionError: assert False
FAILED ml/tests/unit/features/test_feature_validation.py::TestUnifiedFeatureCalculation::test_unified_method_invalid_mode - KeyError: 'close'
FAILED ml/tests/unit/features/test_feature_validation.py::TestUnifiedFeatureCalculation::test_unified_method_online_without_manager - AssertionError: Regex pattern did not match.
FAILED ml/tests/unit/features/common/test_feature_calculator.py::TestComputeFeatures::test_compute_features_empty_input - Failed: DID NOT RAISE (<class 'ValueError'>, <class 'IndexError'>, <class 'TypeError'>)
FAILED ml/tests/unit/features/earnings/test_parity.py::test_validate_known_future_effective_times_raises_for_earnings_leakage - ml.features.validation.KnownFutureError: Known-future detected in earnings_lag: 2 violations
FAILED ml/tests/unit/features/test_known_future_transforms.py::test_validate_known_future_effective_times_raises_on_leakage - ml.features.validation.KnownFutureError: Known-future detected in unit_test_macro: 1 violations
FAILED ml/tests/unit/training/teacher/test_tft_teacher_streaming.py::test_fit_streaming_returns_logits - AssertionError: assert 0 > 0
FAILED ml/tests/unit/training/teacher/test_tft_cli_outputs.py::test_persist_teacher_outputs_writes_all_files - TypeError: Object of type StreamingRunTelemetry is not JSON serializable
FAILED ml/tests/unit/training/event_driven/test_worker.py::test_lightning_worker_runs_with_real_teacher - AssertionError: assert <EventStatus.DEFERRED: 'deferred'> in {<EventStatus.PARTIAL: 'partial'>, <EventStatus.SUCCESS: 'success'>}
FAILED ml/tests/unit/dashboard/common/test_registry_manager.py::test_list_datasets_empty - AssertionError: assert 1 == 0
FAILED ml/tests/unit/data/test_alfred_loader.py::test_alfred_loader_windowed_fetch_normalizes_datetimes - TypeError: can't compare offset-naive and offset-aware datetimes
FAILED ml/tests/unit/data/test_alfred_loader.py::test_alfred_loader_falls_back_to_fred_series - TypeError: ALFREDConfig.__init__() got an unexpected keyword argument 'fallback_to_fred_series'
FAILED ml/tests/unit/data/test_alfred_loader.py::test_alfred_loader_fallback_handles_missing_series - TypeError: ALFREDConfig.__init__() got an unexpected keyword argument 'fallback_to_fred_series'
FAILED ml/tests/unit/data/test_feature_restorer.py::test_restorer_replays_earnings_actuals - AssertionError: assert 0 == 1
FAILED ml/tests/unit/data/test_feature_restorer.py::test_restorer_filters_to_requested_buckets - AssertionError: assert 0 == 1
FAILED ml/tests/unit/data/test_feature_restorer.py::test_restorer_replays_macro_release_dataset - AssertionError: assert 0 == 1
FAILED ml/tests/unit/data/test_feature_restorer.py::test_restorer_filters_file_backed_events - AssertionError: assert 0 == 1
FAILED ml/tests/unit/data/test_feature_restorer.py::test_restorer_replays_micro_dataset - AssertionError: assert 0 == 1
FAILED ml/tests/unit/data/test_feature_restorer.py::test_restorer_replays_l2_dataset - AssertionError: assert 0 == 1
FAILED ml/tests/unit/data/test_tft_dataset_builder_store.py::test_prepare_training_data_from_store_uses_datastore - RuntimeError: No features loaded from FeatureStore for any instrument
FAILED ml/tests/unit/data/test_tft_dataset_builder_store.py::test_component_builder_respects_include_flags - assert False is True
FAILED ml/tests/unit/data/test_tft_dataset_builder_store.py::test_restrict_df_to_window_trims_rows - AttributeError: 'TFTDatasetBuilder' object has no attribute '_restrict_df_to_window'
FAILED ml/tests/unit/data/test_tft_dataset_builder_store.py::test_restrict_df_to_window_without_timestamp_returns_input - AttributeError: 'TFTDatasetBuilder' object has no attribute '_restrict_df_to_window'
FAILED ml/tests/unit/data/test_dataset_build_macro.py::test_build_tft_dataset_invokes_macro_refresh - KeyError: 'data_store'
FAILED ml/tests/unit/data/test_dataset_build_macro.py::test_build_tft_dataset_marks_capabilities_for_earnings - AttributeError: 'DatasetMetadata' object has no attribute 'capability_flags'
FAILED ml/tests/unit/data/test_dataset_build_macro.py::test_build_tft_dataset_registers_capability_flags - assert False is True
FAILED ml/tests/unit/data/test_dataset_build_macro.py::test_tft_dataset_task_config_overrides_base_dirs - AssertionError: assert '/home/nate/t...fig_o0/source' == '/home/nate/t...icro_override'
FAILED ml/tests/unit/data/test_scheduler_lookback.py::test_binding_lookback_clamped_to_license_start - AttributeError: type object 'DataScheduler' has no attribute '_binding_lookback_days'
FAILED ml/tests/unit/data/test_scheduler_lookback.py::test_binding_lookback_defaults_when_no_license - AttributeError: type object 'DataScheduler' has no attribute '_binding_lookback_days'
FAILED ml/tests/unit/data/test_scheduler_lookback.py::test_binding_lookback_handles_expired_dataset - AttributeError: type object 'DataScheduler' has no attribute '_binding_lookback_days'
FAILED ml/tests/unit/data/test_scheduler_lookback.py::test_compute_dynamic_lookbacks_uses_sql_staleness - AttributeError: type object 'DataScheduler' has no attribute '_compute_dynamic_lookbacks'
FAILED ml/tests/unit/data/test_scheduler_lookback.py::test_binding_dynamic_base_prefers_instrument_map - AttributeError: type object 'DataScheduler' has no attribute '_binding_dynamic_base'
FAILED ml/tests/unit/data/test_l2_cache.py::test_ensure_day_rebuilds_placeholder_partition - assert 0 == 1
FAILED ml/tests/unit/data/test_scheduler_ts_extraction.py::test_extract_ts_bounds_mixed_sources - AttributeError: type object 'DataScheduler' has no attribute '_extract_ts_bounds'
FAILED ml/tests/unit/data/test_scheduler_ts_extraction.py::test_extract_ts_bounds_handles_pandas_timestamp - AttributeError: type object 'DataScheduler' has no attribute '_extract_ts_bounds'
FAILED ml/tests/unit/data/test_scheduler_ts_extraction.py::test_extract_ts_bounds_handles_callable_returning_datetime - AttributeError: type object 'DataScheduler' has no attribute '_extract_ts_bounds'
FAILED ml/tests/unit/data/test_scheduler_ts_extraction.py::test_extract_ts_bounds_returns_zero_when_missing - AttributeError: type object 'DataScheduler' has no attribute '_extract_ts_bounds'
FAILED ml/tests/unit/data/test_tft_dataset_builder_phase_one.py::test_append_macro_delta_features_polars_computes_differences - AttributeError: 'TFTDatasetBuilder' object has no attribute '_append_macro_delta_features_polars'
FAILED ml/tests/unit/data/test_tft_dataset_builder_phase_one.py::test_event_features_join_when_calendar_lags_enabled - TypeError: TFTDatasetBuilder.__init__() got an unexpected keyword argument 'include_clustering_tags'
FAILED ml/tests/unit/data/test_fred_join_validation.py::test_iter_vintage_series_dirs_raises_when_directory_missing - Failed: DID NOT RAISE <class 'FileNotFoundError'>
FAILED ml/tests/unit/data/test_fred_join_validation.py::test_load_vintage_release_pl_normalizes_schema - polars.exceptions.ShapeError: unable to append to a DataFrame of width 4 with a DataFrame of width 5
FAILED ml/tests/unit/data/common/test_time_series_windowing.py::TestPropertyBased::test_property_coerce_roundtrip - assert 1e-06 < 1e-06
FAILED ml/tests/unit/data/test_scheduler_targeted.py::test_run_targeted_update_invokes_collection - assert 0 == 1
FAILED ml/tests/unit/data/test_scheduler_targeted.py::test_run_targeted_update_requires_api_key - Failed: DID NOT RAISE <class 'ValueError'>
FAILED ml/tests/unit/data/test_scheduler_targeted.py::test_run_targeted_update_uses_orchestrator_when_enabled - AssertionError: Expected 'backfill_gaps' to have been called once. Called 0 times.
FAILED ml/tests/unit/data/test_scheduler_targeted.py::test_run_targeted_update_expands_orchestrator_lookback - AttributeError: 'NoneType' object has no attribute 'kwargs'
FAILED ml/tests/unit/data/test_scheduler_targeted.py::test_collect_symbol_data_handles_timezone_aware_target - assert False is True
FAILED ml/tests/unit/data/test_scheduler_targeted.py::test_apply_trading_day_padding_weekday - AttributeError: 'DataScheduler' object has no attribute '_apply_trading_day_padding'
FAILED ml/tests/unit/data/test_scheduler_targeted.py::test_apply_trading_day_padding_sunday - AttributeError: 'DataScheduler' object has no attribute '_apply_trading_day_padding'
FAILED ml/tests/unit/data/test_scheduler_targeted.py::test_apply_trading_day_padding_monday - AttributeError: 'DataScheduler' object has no attribute '_apply_trading_day_padding'
FAILED ml/tests/unit/data/test_scheduler_targeted.py::test_derive_catalog_lookback_days_uses_catalog - AttributeError: 'DataScheduler' object has no attribute '_derive_catalog_lookback_days'
FAILED ml/tests/unit/data/test_scheduler_targeted.py::test_derive_catalog_lookback_days_without_provider - AttributeError: 'DataScheduler' object has no attribute '_derive_catalog_lookback_days'
FAILED ml/tests/unit/data/test_tft_dataset_builder_facade.py::TestFacadeHappyPath::test_e2_facade_prepare_training_data - AssertionError: Expected 'prepare_training_data' to have been called once. Called 0 times.
FAILED ml/tests/unit/data/test_tft_dataset_builder_facade.py::TestFacadeHappyPath::test_e8_facade_threshold_bps_alias - AssertionError: Expected 'build_training_dataset' to have been called once. Called 0 times.
FAILED ml/tests/unit/data/test_tft_dataset_builder_facade.py::TestErrorConditions::test_e13_facade_invalid_catalog_error - assert 1113128 == 0
FAILED ml/tests/unit/data/ingest/test_symbology.py::test_resolver_uses_alias_for_brk - ml.data.ingest.symbology.SymbologyResolutionError: Symbol BRK not found
FAILED ml/tests/unit/data/ingest/test_symbology.py::test_resolver_retries_server_error_then_succeeds - TypeError: DatabentoSymbologyResolver.__init__() got an unexpected keyword argument 'retry_attempts'
FAILED ml/tests/unit/data/ingest/test_symbology.py::test_resolver_raises_after_retry_budget_exhausted - TypeError: DatabentoSymbologyResolver.__init__() got an unexpected keyword argument 'retry_attempts'
FAILED ml/tests/unit/data/earnings/test_edgar_fetcher.py::TestEdgarFetcher::test_fetch_earnings_success - AttributeError: <module 'ml.data.earnings.edgar_fetcher' from '/home/nate/projects/nautilus_trader/ml/data/earnings/edgar_fetcher.py'> does not have the attribute 'edgarto...
FAILED ml/tests/unit/data/earnings/test_edgar_fetcher.py::TestEdgarFetcher::test_fetch_earnings_invalid_ticker - AttributeError: <module 'ml.data.earnings.edgar_fetcher' from '/home/nate/projects/nautilus_trader/ml/data/earnings/edgar_fetcher.py'> does not have the attribute 'edgarto...
FAILED ml/tests/unit/data/earnings/test_edgar_fetcher.py::TestEdgarFetcher::test_fetch_earnings_no_filings - AttributeError: <module 'ml.data.earnings.edgar_fetcher' from '/home/nate/projects/nautilus_trader/ml/data/earnings/edgar_fetcher.py'> does not have the attribute 'edgarto...
FAILED ml/tests/unit/data/earnings/test_edgar_fetcher.py::TestEdgarFetcher::test_fetch_earnings_multiple_quarters - AttributeError: <module 'ml.data.earnings.edgar_fetcher' from '/home/nate/projects/nautilus_trader/ml/data/earnings/edgar_fetcher.py'> does not have the attribute 'edgarto...
FAILED ml/tests/unit/data/earnings/test_edgar_fetcher.py::TestEdgarFetcher::test_extract_fiscal_period_fallback_to_filing_date - assert 1 == 2024
FAILED ml/tests/unit/data/test_feature_npz_stream.py::test_compute_metadata_from_polars - AttributeError: 'DatasetMetadata' object has no attribute 'capability_flags'
FAILED ml/tests/unit/data/sources/test_calendar_pandas.py::TestPandasCalendarSource::test_get_schedule_uses_fallback_when_disabled - AssertionError: assert None == datetime.timezone.utc
FAILED ml/tests/unit/data/sources/test_calendar_pandas.py::TestPandasCalendarSource::test_get_24_7_schedule - AssertionError: assert None == datetime.timezone.utc
FAILED ml/tests/unit/data/sources/test_calendar_pandas.py::TestPandasCalendarSource::test_build_schedule_normalizes_timezone - AssertionError: assert <DstTzInfo 'America/New_York' EST-1 day, 19:00:00 STD> == datetime.timezone.utc
FAILED ml/tests/unit/cli/test_streaming_training_runner_adaptive.py::test_runner_azure_notice_triggers_checkpoint - dataclasses.FrozenInstanceError: cannot assign to field 'scheduled_events'
FAILED ml/tests/unit/config/test_coverage_config.py::test_load_dataset_coverage_entries_parses_aliases - ValueError: Unknown schema 'earnings'
FAILED ml/tests/unit/config/test_coverage_config.py::test_load_dataset_coverage_entries_preserves_blank_partition_template - ValueError: Unknown schema 'events_calendar'
FAILED ml/tests/unit/config/test_dataset_ids.py::test_dataset_ids_accessible_from_config - AttributeError: module 'ml.config' has no attribute 'EQUS_MINI_DATASET_ID'. Did you mean: 'L2_MINUTE_DATASET_ID'?
FAILED ml/tests/unit/config/test_dataset_ids.py::test_dataset_ids_in_public_api - AssertionError: assert 'EQUS_MINI_DATASET_ID' in ['EARNINGS_ACTUALS_DATASET_ID', 'EARNINGS_ESTIMATES_DATASET_ID', 'EVENTS_CALENDAR_DATASET_ID', 'L2_MINUTE_DATASET_ID', 'MA...
FAILED ml/tests/unit/config/test_dataset_ids.py::test_constants_imported_consistently - AttributeError: module 'ml.config' has no attribute 'EQUS_MINI_DATASET_ID'. Did you mean: 'L2_MINUTE_DATASET_ID'?
FAILED ml/tests/unit/common/test_message_bus_config.py::test_publisher_from_config_redis_backend - assert False
FAILED ml/tests/unit/stores/test_migrations_runner.py::test_verify_market_data_schema_success - sqlalchemy.exc.NoInspectionAvailable: No inspection system is available for object of type <class 'ml.tests.unit.stores.test_migrations_runner._DummyEngine'>
FAILED ml/tests/unit/stores/test_migrations_runner.py::test_verify_market_data_schema_missing_column - sqlalchemy.exc.NoInspectionAvailable: No inspection system is available for object of type <class 'ml.tests.unit.stores.test_migrations_runner._DummyEngine'>
FAILED ml/tests/unit/stores/test_migrations_runner.py::test_verify_instrumentation_tables_success - sqlalchemy.exc.NoInspectionAvailable: No inspection system is available for object of type <class 'ml.tests.unit.stores.test_migrations_runner._DummyEngine'>
FAILED ml/tests/unit/stores/test_migrations_runner.py::test_verify_instrumentation_tables_missing - sqlalchemy.exc.NoInspectionAvailable: No inspection system is available for object of type <class 'ml.tests.unit.stores.test_migrations_runner._DummyEngine'>
FAILED ml/tests/unit/stores/common/test_feature_health_component.py::TestIsHealthy::test_is_healthy_logs_warning_on_failure - AssertionError: Expected 'warning' to have been called once. Called 0 times.
FAILED ml/tests/unit/stores/common/test_feature_health_component.py::TestIsHealthy::test_is_healthy_emits_metrics_on_success - AssertionError: Expected 'labels' to be called once. Called 0 times.
FAILED ml/tests/unit/stores/common/test_feature_health_component.py::TestIsHealthy::test_is_healthy_emits_metrics_on_failure - AssertionError: Expected 'labels' to be called once. Called 0 times.
FAILED ml/tests/unit/stores/common/test_feature_health_component.py::TestClearFeatures::test_clear_features_emits_metrics_for_instrument_scope - AssertionError: Expected 'labels' to be called once. Called 0 times.
FAILED ml/tests/unit/stores/common/test_feature_health_component.py::TestClearFeatures::test_clear_features_emits_metrics_for_version_scope - AssertionError: Expected 'labels' to be called once. Called 0 times.
FAILED ml/tests/unit/stores/common/test_feature_health_component.py::TestClearFeatures::test_clear_features_emits_metrics_for_all_scope - AssertionError: Expected 'labels' to be called once. Called 0 times.
FAILED ml/tests/unit/stores/common/test_feature_health_component.py::TestClearFeatures::test_clear_features_emits_metrics_for_combined_scope - AssertionError: Expected 'labels' to be called once. Called 0 times.
FAILED ml/tests/unit/stores/common/test_feature_health_component.py::TestClearFeatures::test_clear_features_logs_debug_message - AssertionError: Expected 'debug' to have been called once. Called 0 times.
FAILED ml/tests/unit/stores/common/test_feature_health_component.py::TestFlush::test_flush_logs_debug_message - AssertionError: Expected 'debug' to have been called once. Called 0 times.
FAILED ml/tests/unit/stores/common/test_feature_computation_component.py::TestComputeRealtime::test_compute_realtime_creates_internal_indicator_manager - AssertionError: Expected '_get_or_create_indicator_manager' to have been called once. Called 0 times.
FAILED ml/tests/unit/stores/common/test_feature_computation_component.py::TestComputeHistoricalParallel::test_compute_historical_parallel_caps_workers - assert 0 == 1
FAILED ml/tests/unit/stores/common/test_feature_computation_component.py::TestEventEmission::test_emit_realtime_event_calls_registry - AssertionError: Expected 'emit_dataset_event_and_watermark' to have been called once. Called 0 times.
FAILED ml/tests/unit/stores/common/test_feature_computation_component.py::TestEventEmission::test_emit_historical_event_calls_registry - AssertionError: Expected 'emit_dataset_event_and_watermark' to have been called once. Called 0 times.
FAILED ml/tests/unit/stores/common/test_feature_schema_component.py::TestSetupTables::test_setup_tables_reflects_existing_table - AssertionError: assert Table('ml_feature_values', MetaData(), schema='public') == <ml.tests.unit.stores.common.test_feature_schema_component.MockTable object at 0x70973957...
FAILED ml/tests/unit/stores/common/test_feature_schema_component.py::TestSetupTables::test_setup_tables_creates_fallback_when_no_migration - AssertionError: assert Table('ml_feature_values', MetaData(), schema='public') == <ml.tests.unit.stores.common.test_feature_schema_component.MockTable object at 0x7097274c...
FAILED ml/tests/unit/stores/common/test_feature_schema_component.py::TestSetupTables::test_setup_tables_uses_schema_name_from_config - TypeError: 'NoneType' object is not subscriptable
FAILED ml/tests/unit/stores/common/test_feature_schema_component.py::TestGetFeatureNamesOnline::test_get_feature_names_online_uses_feature_engineer - AssertionError: assert [] == ['online_feature_1']
FAILED ml/tests/unit/stores/common/test_data_writer.py::test_write_ingestion_emits_rejection_metric - AssertionError: Expected 'labels' to be called once. Called 0 times.
FAILED ml/tests/unit/stores/test_data_store_validation.py::TestContractValidation::test_type_validation - AssertionError: assert 1.0 < 1.0
FAILED ml/tests/unit/stores/test_data_store_validation.py::TestContractValidation::test_null_validation - AssertionError: assert 1.0 < 1.0
FAILED ml/tests/unit/stores/test_data_store_validation.py::TestContractValidation::test_range_validation - AssertionError: assert 1.0 < 1.0
FAILED ml/tests/unit/stores/test_data_store_validation.py::TestContractValidation::test_uniqueness_validation - AssertionError: assert 1.0 < 1.0
FAILED ml/tests/unit/stores/test_data_store_validation.py::TestContractValidation::test_monotonicity_validation - AssertionError: assert 1.0 < 1.0
FAILED ml/tests/unit/stores/test_data_store_validation.py::TestContractValidation::test_lateness_validation - assert 0 > 0
FAILED ml/tests/unit/stores/test_data_store_validation.py::TestFailClosedWrites::test_write_rejected_on_validation_failure - Failed: DID NOT RAISE <class 'ValueError'>
FAILED ml/tests/unit/stores/test_data_store_validation.py::TestFailClosedWrites::test_strict_mode_enforcement - AssertionError: assert 1.0 < 1.0
FAILED ml/tests/unit/stores/test_data_store_validation.py::TestPropertyBased::test_validation_consistency - AssertionError: assert 1.0 < 1.0
FAILED ml/tests/unit/stores/test_data_store_validation.py::TestPrometheusMetrics::test_validation_metrics_emitted - AssertionError: assert False
FAILED ml/tests/unit/stores/test_feature_store_facade.py::TestEdgeCases::test_facade_with_none_feature_config - sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection to server at "localhost" (127.0.0.1), port 5434 failed: FATAL:  password authentication failed for ...
FAILED ml/tests/unit/stores/test_feature_store_facade.py::TestEdgeCases::test_facade_with_ml_feature_config - sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection to server at "localhost" (127.0.0.1), port 5434 failed: FATAL:  password authentication failed for ...
FAILED ml/tests/unit/deployment/test_check_health.py::TestMainFunction::test_main_all_healthy - assert 1 == 0
FAILED ml/tests/unit/deployment/test_check_health.py::TestMainFunction::test_main_all_unhealthy - AssertionError: assert '[✓]' not in '===========...vice_name>\n'
FAILED ml/tests/unit/deployment/test_entrypoint_actor_node.py::test_setup_falls_back_to_mock_data_when_key_invalid - TypeError: cannot unpack non-iterable NoneType object
FAILED ml/tests/unit/deployment/test_entrypoint_actor_node.py::test_strategy_does_not_register_databento_with_invalid_key - TypeError: cannot unpack non-iterable NoneType object
FAILED ml/tests/unit/deployment/test_entrypoint_pipeline.py::test_load_feature_coverage_entries_uses_env_manifest - assert ()
FAILED ml/tests/unit/deployment/test_entrypoint_pipeline.py::test_load_feature_coverage_entries_includes_feature_datasets - AssertionError: assert False
FAILED ml/tests/unit/deployment/test_entrypoint_pipeline.py::test_run_coverage_restoration_once_initializes_scheduler - TypeError: test_run_coverage_restoration_once_initializes_scheduler.<locals>._SchedulerStub.__init__() got an unexpected keyword argument 'dual_write_dataset_types'
FAILED ml/tests/unit/ingest/test_orchestrator_backfill.py::test_normalize_time_columns_derives_ts_event_from_timestamp_column - KeyError: 'ts_event'
FAILED ml/tests/unit/ingest/test_orchestrator_backfill.py::test_normalize_time_columns_uses_ts_exchange_fallback - KeyError: 'ts_event'
FAILED ml/tests/unit/registry/test_bootstrap_datasets_earnings.py::test_bootstrap_datasets_postgres_registers_earnings - sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) could not translate host name "registry" to address: Name or service not known
FAILED ml/tests/unit/registry/test_model_registry_facade.py::TestPersistence::test_backend_property - AssertionError: assert <BackendType.JSON: 'json'> == <BackendType.JSON: 'json'>
FAILED ml/tests/unit/registry/test_feature_registry.py::test_register_feature_set_records_capability_flag_diff - AssertionError: assert None == {'include_l2': {'current': True, 'previous': False}}
FAILED ml/tests/unit/registry/test_model_persistence.py::test_load_model_with_caching - AssertionError: assert None == <Mock name='InferenceSession()' id='123794634594240'>
FAILED ml/tests/unit/registry/test_model_persistence.py::test_load_model_lru_eviction - assert 0 == 3
FAILED ml/tests/unit/registry/test_bootstrap_datasets_market_data.py::test_equs_mini_manifest_registered - AssertionError: Manifest for EQUS.MINI not found
FAILED ml/tests/unit/registry/test_bootstrap_datasets_market_data.py::test_equs_mini_contract_present - KeyError: 'EQUS.MINI'
FAILED ml/tests/unit/registry/test_data_registry_facade.py::TestDelegation::test_facade_delegates_emit_event - assert 0 >= 1
FAILED ml/tests/unit/registry/test_data_registry_facade.py::TestDelegation::test_facade_delegates_update_watermark - assert None is not None
FAILED ml/tests/unit/registry/test_data_registry_facade.py::TestDelegation::test_facade_delegates_get_watermark - assert None is not None
FAILED ml/tests/unit/registry/test_data_registry_facade.py::TestDelegation::test_facade_delegates_iter_watermarks - RuntimeError: Failed to get database session
FAILED ml/tests/unit/registry/test_data_registry_facade.py::TestDelegation::test_facade_delegates_link_lineage - RuntimeError: Failed to get database session
FAILED ml/tests/unit/registry/test_data_registry_facade.py::TestDelegation::test_facade_delegates_flush - AssertionError: assert False
FAILED ml/tests/unit/registry/test_data_registry_facade.py::TestPublicAPI::test_facade_inherits_ml_component_mixin - assert False
FAILED ml/tests/unit/registry/test_data_registry_facade.py::TestThreadSafetyAndCleanup::test_facade_destructor_flushes - AssertionError: assert False
FAILED ml/tests/unit/registry/common/test_data_persistence.py::TestLoadRegistry::test_load_registry_from_existing_json - assert 0 == 1
FAILED ml/tests/unit/registry/common/test_data_persistence.py::TestSaveRegistry::test_save_registry_immediate_mode - AssertionError: assert False
FAILED ml/tests/unit/registry/common/test_data_persistence.py::TestSaveRegistry::test_flush_forces_immediate_save - AssertionError: assert False
FAILED ml/tests/unit/registry/common/test_data_persistence.py::TestSaveRegistry::test_batch_save_interval_zero_saves_immediately - AssertionError: assert False
FAILED ml/tests/unit/registry/common/test_manifest_serialization.py::TestManifestSerialization::test_dict_to_manifest_valid_data - AssertionError: assert False
FAILED ml/tests/unit/registry/common/test_manifest_serialization.py::TestManifestSerialization::test_dict_to_manifest_enum_case_insensitive - AssertionError: assert <DatasetType.FEATURES: 'features'> == <DatasetType.FEATURES: 'features'>
FAILED ml/tests/unit/registry/common/test_manifest_serialization.py::TestManifestSerialization::test_manifest_to_dict_roundtrip - AssertionError: assert <DatasetType.FEATURES: 'features'> == <DatasetType.FEATURES: 'features'>
FAILED ml/tests/unit/registry/common/test_contract_serialization.py::TestContractSerialization::test_dict_to_contract_valid_data - AssertionError: assert False
FAILED ml/tests/unit/registry/common/test_watermark_manager.py::TestUpdateWatermark::test_update_watermark_success - assert None is not None
FAILED ml/tests/unit/registry/common/test_watermark_manager.py::TestUpdateWatermark::test_update_watermark_creates_key - AssertionError: assert 'test_dataset:EUR/USD:live' in {}
FAILED ml/tests/unit/registry/common/test_watermark_manager.py::TestUpdateWatermark::test_update_watermark_normalizes_source_enum - AttributeError: 'NoneType' object has no attribute 'source'
FAILED ml/tests/unit/registry/common/test_watermark_manager.py::TestUpdateWatermark::test_update_watermark_sets_updated_at - AttributeError: 'NoneType' object has no attribute 'updated_at'
FAILED ml/tests/unit/registry/common/test_watermark_manager.py::TestUpdateWatermark::test_update_watermark_saves_immediately_json - FileNotFoundError: [Errno 2] No such file or directory: '/home/nate/tmp/pytest-of-nate/pytest-5627/test_update_watermark_saves_im0/registry/data_registry.json'
FAILED ml/tests/unit/registry/common/test_watermark_manager.py::TestGetWatermark::test_get_watermark_success - assert None is not None
FAILED ml/tests/unit/registry/common/test_watermark_manager.py::TestGetWatermark::test_get_watermark_accepts_source_enum - assert None is not None
FAILED ml/tests/unit/registry/common/test_watermark_manager.py::TestGetWatermark::test_get_watermark_accepts_source_string - assert None is not None
FAILED ml/tests/unit/registry/common/test_watermark_manager.py::TestIterWatermarks::test_iter_watermarks_no_filter - RuntimeError: Failed to get database session
FAILED ml/tests/unit/registry/common/test_watermark_manager.py::TestIterWatermarks::test_iter_watermarks_filter_by_dataset_id - RuntimeError: Failed to get database session
FAILED ml/tests/unit/registry/common/test_watermark_manager.py::TestIterWatermarks::test_iter_watermarks_filter_by_instrument_id - RuntimeError: Failed to get database session
FAILED ml/tests/unit/registry/common/test_watermark_manager.py::TestIterWatermarks::test_iter_watermarks_filter_by_source - RuntimeError: Failed to get database session
FAILED ml/tests/unit/registry/common/test_watermark_manager.py::TestIterWatermarks::test_iter_watermarks_with_limit - RuntimeError: Failed to get database session
FAILED ml/tests/unit/registry/common/test_watermark_manager.py::TestIterWatermarks::test_iter_watermarks_sorted_by_updated_at_desc - RuntimeError: Failed to get database session
FAILED ml/tests/unit/registry/common/test_watermark_manager.py::TestPostgresBackend::test_get_watermark_postgres_checks_cache_first - assert None is not None
FAILED ml/tests/unit/registry/common/test_watermark_manager.py::TestPostgresBackend::test_update_watermark_postgres_session_failure_raises - Failed: DID NOT RAISE <class 'RuntimeError'>
FAILED ml/tests/unit/registry/common/test_lineage_tracker.py::TestLinkLineage::test_link_lineage_single_parent - assert 0 == 1
FAILED ml/tests/unit/registry/common/test_lineage_tracker.py::TestLinkLineage::test_link_lineage_multiple_parents - assert 0 == 3
FAILED ml/tests/unit/registry/common/test_lineage_tracker.py::TestLinkLineage::test_link_lineage_stores_ts_range - IndexError: list index out of range
FAILED ml/tests/unit/registry/common/test_lineage_tracker.py::TestLinkLineage::test_link_lineage_stores_parameters - IndexError: list index out of range
FAILED ml/tests/unit/registry/common/test_lineage_tracker.py::TestLinkLineage::test_link_lineage_sets_created_at - IndexError: list index out of range
FAILED ml/tests/unit/registry/common/test_lineage_tracker.py::TestLinkLineage::test_link_lineage_trims_old_entries_json - assert 0 == 5000
FAILED ml/tests/unit/registry/common/test_lineage_tracker.py::TestIterLineage::test_iter_lineage_no_filter - RuntimeError: Failed to get database session
FAILED ml/tests/unit/registry/common/test_lineage_tracker.py::TestIterLineage::test_iter_lineage_filter_by_child - RuntimeError: Failed to get database session
FAILED ml/tests/unit/registry/common/test_lineage_tracker.py::TestIterLineage::test_iter_lineage_filter_by_parent - RuntimeError: Failed to get database session
FAILED ml/tests/unit/registry/common/test_lineage_tracker.py::TestIterLineage::test_iter_lineage_with_limit - RuntimeError: Failed to get database session
FAILED ml/tests/unit/registry/common/test_lineage_tracker.py::TestIterLineage::test_iter_lineage_sorted_by_created_at_desc - RuntimeError: Failed to get database session
FAILED ml/tests/unit/registry/common/test_manifest_manager.py::TestContractCreation::test_create_contract_from_manifest_with_ranges - assert 0 >= 1
FAILED ml/tests/unit/registry/common/test_manifest_manager.py::TestContractCreation::test_create_contract_from_manifest_with_nullability - assert 0 >= 1
FAILED ml/tests/unit/registry/common/test_manifest_manager.py::TestContractCreation::test_create_contract_from_manifest_with_regex - assert 0 == 1
FAILED ml/tests/unit/registry/common/test_manifest_manager.py::TestContractCreation::test_create_contract_from_manifest_default_rule - AssertionError: assert <ValidationRuleType.TYPE_CHECK: 'type_check'> == <ValidationRuleType.TYPE_CHECK: 'type_check'>
FAILED ml/tests/unit/registry/common/test_manifest_manager.py::TestContractCreation::test_get_contract_creates_if_missing - AssertionError: assert False
FAILED ml/tests/unit/registry/common/test_event_emission.py::TestEmitEvent::test_emit_event_success - assert 0 >= 1
FAILED ml/tests/unit/registry/common/test_event_emission.py::TestEmitEvent::test_emit_event_normalizes_enums - IndexError: list index out of range
FAILED ml/tests/unit/registry/common/test_event_emission.py::TestEmitEvent::test_emit_event_with_metadata - IndexError: list index out of range
FAILED ml/tests/unit/registry/common/test_event_emission.py::TestEmitEvent::test_emit_event_with_error - IndexError: list index out of range
FAILED ml/tests/unit/registry/common/test_event_emission.py::TestEventTrimming::test_emit_event_trims_old_events_json - AssertionError: assert 9999 == 10000
FAILED ml/tests/unit/registry/common/test_event_emission.py::TestEventPersistence::test_emit_event_saves_immediately_json - FileNotFoundError: [Errno 2] No such file or directory: '/home/nate/tmp/pytest-of-nate/pytest-5627/test_emit_event_saves_immediat0/registry/data_registry.json'
FAILED ml/tests/unit/registry/common/test_event_emission.py::TestEdgeCases::test_emit_event_zero_counts_allowed - IndexError: list index out of range
FAILED ml/tests/unit/registry/common/test_event_emission.py::TestEdgeCases::test_emit_event_with_correlation_id - IndexError: list index out of range
FAILED ml/tests/unit/registry/common/test_event_emission.py::TestThreadSafety::test_emit_event_thread_safe - assert 0 == 50
FAILED ml/tests/unit/registry/common/test_event_emission.py::TestPostgresBackend::test_emit_event_postgres_session_failure_raises - Failed: DID NOT RAISE <class 'RuntimeError'>
FAILED ml/tests/fixtures/test_exports.py::test_fixtures_module_is_importable - AssertionError: assert <module 'ml.tests.fixtures' from '/home/nate/projects/nautilus_trader/ml/tests/fixtures/__init__.py'> is fixtures
FAILED ml/tests/integration/test_dashboard_ml_integration.py::TestDashboardMLIntegration::test_telemetry_emission[start_actor-ml_dashboard_actions_total] - AssertionError: Expected 'labels' to have been called.
FAILED ml/tests/integration/test_dashboard_ml_integration.py::TestDashboardMLIntegration::test_telemetry_emission[trigger_pipeline-ml_dashboard_actions_total] - AssertionError: Expected 'labels' to have been called.
FAILED ml/tests/integration/test_dashboard_ml_integration.py::TestDashboardMLIntegration::test_telemetry_emission[emergency_stop-ml_dashboard_actions_total] - AssertionError: Expected 'labels' to have been called.
FAILED ml/tests/integration/pipelines/test_earnings_pipeline.py::TestEarningsPipelineIntegration::test_pipeline_spec_serialization - _pickle.PicklingError: Can't pickle <class 'ml.features.earnings.earnings_transforms.EarningsSurpriseTransformSpec'>: it's not the same object as ml.features.earnings.earn...
FAILED ml/tests/integration/actors/test_actor_circuit_breaker_integration.py::test_actor_bus_scheme_prefix_integration - ValueError: The actor has not been registered
FAILED ml/tests/integration/features/test_feature_calculator_integration.py::TestFeatureCalculatorIntegration::test_feature_calculator_with_realistic_market_data - AssertionError: Volume ratios should be positive
FAILED ml/tests/integration/features/test_feature_calculator_integration.py::TestFeatureCalculatorIntegration::test_feature_calculator_batch_online_parity - AssertionError: 
FAILED ml/tests/integration/test_smoke.py::test_can_create_feature_engineer - AssertionError: Config mismatch: {'lookback_window': 120, 'indicators': None, 'feature_names': None, 'normalize_features': True, 'fill_missing_with': 0.0, 'average_volume'...
FAILED ml/tests/integration/test_observability_tracing.py::TestTracingWithOpenTelemetry::test_trace_context_with_mocked_otel - AssertionError: assert None == '00-b9cef277c31d4a97807bc31ffe1d3c24-9db0e48fe0f04a8e-01'
FAILED ml/tests/integration/test_observability_tracing.py::TestTracingWithOpenTelemetry::test_inject_trace_context_with_mocked_otel - AssertionError: assert 'trace_context' in {'correlation_id': 'test123'}
FAILED ml/tests/integration/test_observability_tracing.py::TestTracingWithOpenTelemetry::test_trace_cold_path_with_mocked_otel - IndexError: list index out of range
FAILED ml/tests/integration/test_observability_tracing.py::TestTracingWithOpenTelemetry::test_trace_cold_path_decorator_with_mocked_otel - IndexError: list index out of range
FAILED ml/tests/integration/test_observability_tracing.py::TestTracingWithOpenTelemetry::test_trace_inference_decorator_with_mocked_otel - IndexError: list index out of range
FAILED ml/tests/integration/test_observability_tracing.py::TestTracingGracefulFallback::test_graceful_fallback_when_otel_unavailable - assert not True
FAILED ml/tests/integration/dashboard/test_streaming_state_endpoint.py::test_dashboard_streaming_state_snapshot - AttributeError: 'module' object at ml.dashboard.service has no attribute 'ObservabilityService'
FAILED ml/tests/integration/data/test_tft_builder_integration.py::TestComponentIntegration::test_components_initialized_correctly - AssertionError: assert False
FAILED ml/tests/integration/pipeline/test_tft_pipeline_sidecar.py::test_pipeline_reads_sidecar_when_args_missing - ml.data.validation.DatasetValidationError: Dataset has 0 rows; minimum required is 1
FAILED ml/tests/integration/pipeline/test_tft_train_distill_pipeline.py::test_pipeline_registers_features_and_passes_ids - SystemExit: 2
FAILED ml/tests/integration/deployment/test_pipeline_rehydration.py::test_pipeline_runner_performs_catalog_rehydration - SystemExit: 1
FAILED ml/tests/integration/earnings/test_tft_task_dataset.py::test_task_builds_dataset_with_earnings_columns - ml.data.validation.DatasetValidationError: Dataset has 0 rows; minimum required is 1
FAILED ml/tests/integration/registry/test_model_registry_security.py::TestModelRegistryIntegrity::test_load_model_verifies_integrity - AssertionError: assert None == <MagicMock name='InferenceSession()' id='123796486074544'>
FAILED ml/tests/integration/registry/test_model_registry_security.py::TestModelRegistryIntegrity::test_load_model_missing_digest_warning - AssertionError: assert None == <MagicMock name='InferenceSession()' id='123803787236688'>
FAILED ml/tests/integration/registry/test_model_registry_security.py::TestModelRegistryIntegrity::test_register_model_permission_error - Failed: DID NOT RAISE <class 'ValueError'>
FAILED ml/tests/e2e/test_tft_dataset_builder_e2e.py::TestE2EBasicDatasetBuilding::test_e2e_build_dataset_with_technical_features - AssertionError: Missing price column
FAILED ml/tests/e2e/test_tft_dataset_builder_e2e.py::TestE2ESaveLoadDatasets::test_e2e_save_and_load_dataset - AttributeError: 'TFTDatasetBuilder' object has no attribute '_dataset_serializer'
FAILED ml/tests/e2e/test_tft_dataset_builder_e2e.py::TestE2EValidationSplits::test_e2e_split_dataset - AttributeError: 'TFTDatasetBuilder' object has no attribute '_validation_splitter'
FAILED ml/tests/e2e/test_tft_dataset_builder_e2e.py::TestE2EErrorHandling::test_e2e_empty_catalog_handled_gracefully - AssertionError: Unexpected error: assert (10994 == 0 or 10994 < 30)
FAILED ml/tests/e2e/test_data_scheduler_e2e.py::test_01_scheduler_component_initialization_e2e - AssertionError: TradingDayCalculator missing
FAILED ml/tests/e2e/test_data_scheduler_e2e.py::test_14_legacy_component_parity_e2e[0] - assert False is True
FAILED ml/tests/e2e/test_data_scheduler_e2e.py::test_15_feature_flag_toggle_e2e - AssertionError: assert False
FAILED ml/tests/e2e/test_datastore_e2e.py::TestE2EHealthAndConfiguration::test_e2e_health_status_reports_all_components - KeyError: 'implementation'
FAILED ml/tests/e2e/test_datastore_e2e.py::TestE2EHealthAndConfiguration::test_e2e_configuration_validation_catches_errors - assert 0 >= 2
FAILED ml/tests/e2e/test_feature_store_e2e.py::test_02_batch_write - TypeError: FeatureData.__init__() got an unexpected keyword argument 'feature_values'
FAILED ml/tests/e2e/test_feature_store_e2e.py::test_05_config_hashing - AssertionError: assert '550dec97ce216230' != '550dec97ce216230'
FAILED ml/tests/e2e/test_feature_store_e2e.py::test_10_health_check - sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection to server at "localhost" (127.0.0.1), port 5434 failed: FATAL:  password authentication failed for ...
FAILED ml/tests/data/ingest/test_discovery.py::test_discover_tracks_symbology_rejection - assert None == 1.0
FAILED ml/tests/facades/test_model_registry_parity.py::TestRollbackParity::test_parity_rollback - AssertionError: assert <DeploymentStatus.ACTIVE: 'active'> == <DeploymentStatus.ACTIVE: 'active'>
FAILED ml/tests/contracts/test_store_env_topic_config_contracts.py::test_feature_store_honors_env_topic_scheme_and_prefix - sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) could not translate host name "ignored" to address: Name or service not known
FAILED ml/tests/contracts/test_data_store_routing_advanced.py::TestDataStoreRouting::test_features_routing_contract - TypeError: Any cannot be instantiated
FAILED ml/tests/contracts/test_data_store_routing_advanced.py::TestDataStoreRouting::test_predictions_routing_contract - TypeError: Any cannot be instantiated
FAILED ml/tests/contracts/test_data_store_routing_advanced.py::TestDataStoreRouting::test_signals_routing_contract - TypeError: Any cannot be instantiated
FAILED ml/tests/contracts/test_data_store_routing_advanced.py::TestDataStoreRouting::test_read_routing_by_dataset_type - TypeError: Any cannot be instantiated
FAILED ml/tests/contracts/test_data_store_routing_advanced.py::TestDataStoreValidation::test_range_validation_contract - TypeError: Any cannot be instantiated
FAILED ml/tests/contracts/test_data_store_routing_advanced.py::TestDataStoreValidation::test_nullability_validation_contract - TypeError: Any cannot be instantiated
FAILED ml/tests/contracts/test_data_store_routing_advanced.py::TestDataStoreValidation::test_prediction_bounds_property - TypeError: Any cannot be instantiated
FAILED ml/tests/contracts/test_data_store_routing_advanced.py::TestDataStoreEvents::test_event_emission_contract - TypeError: Any cannot be instantiated
FAILED ml/tests/contracts/test_data_store_routing_advanced.py::TestDataStoreEvents::test_error_event_contract - TypeError: Any cannot be instantiated
FAILED ml/tests/contracts/test_data_store_routing_advanced.py::TestDataStoreEvents::test_batch_event_aggregation_contract - TypeError: Any cannot be instantiated
FAILED ml/tests/contracts/test_data_store_routing_advanced.py::TestDataStoreWatermarks::test_watermark_progression_contract - TypeError: Any cannot be instantiated
FAILED ml/tests/contracts/test_data_store_routing_advanced.py::TestDataStoreWatermarks::test_cross_store_watermark_consistency - TypeError: Any cannot be instantiated
FAILED ml/tests/contracts/test_data_store_routing_advanced.py::TestDataStoreTransactions::test_multi_store_consistency_contract - TypeError: Any cannot be instantiated
FAILED ml/tests/contracts/test_data_store_routing_advanced.py::TestDataStoreTransactions::test_flush_coordination_contract - TypeError: Any cannot be instantiated
FAILED ml/tests/contracts/test_data_store_routing_advanced.py::TestDataStoreErrorPropagation::test_store_error_context_contract - TypeError: Any cannot be instantiated
FAILED ml/tests/contracts/test_data_store_routing_advanced.py::TestDataStoreErrorPropagation::test_validation_error_detail_contract - TypeError: Any cannot be instantiated
FAILED ml/tests/contracts/test_data_store_routing_advanced.py::TestDataStoreIntegrationContracts::test_complete_workflow_contract - TypeError: Any cannot be instantiated
FAILED ml/tests/contracts/test_data_store_routing_advanced.py::TestDataStoreIntegrationContracts::test_concurrent_operations_contract - TypeError: Any cannot be instantiated
FAILED ml/tests/contracts/test_data_store_routing_advanced.py::TestDataStoreCircuitBreaker::test_circuit_breaker_activation_contract - TypeError: Any cannot be instantiated
FAILED ml/tests/contracts/test_dataset_event_contracts.py::TestDatasetEventContracts::test_data_store_emit_event_uses_centralized_helper - ValueError: 'STAGE.FEATURE_COMPUTED' is not a valid Stage
FAILED ml/tests/contracts/test_base_actor_initialization.py::TestBaseMLInferenceActorInitialization::test_model_loader_initialization_with_default_loader - AssertionError: assert False
FAILED ml/tests/contracts/stores/test_store_event_contracts.py::test_strategy_store_registry_event_contracts - AssertionError: assert 'live' == 'historical'
FAILED ml/tests/contracts/stores/test_store_event_contracts.py::test_model_store_registry_event_contracts - AssertionError: assert 'live' == 'historical'
FAILED ml/dashboard/tests/test_pipelines_routes.py::TestPipelinesBuildDatasetEndpoint::test_build_dataset_success - assert 503 == 202
FAILED ml/dashboard/tests/test_pipelines_routes.py::TestPipelinesBuildDatasetEndpoint::test_build_dataset_invalid_config - assert 503 == 400
FAILED ml/dashboard/tests/test_pipelines_routes.py::TestPipelinesTrainModelEndpoint::test_train_model_success - assert 503 == 202
FAILED ml/dashboard/tests/test_pipelines_routes.py::TestPipelinesRunHpoEndpoint::test_run_hpo_success - assert 503 == 202
FAILED ml/dashboard/tests/test_pipelines_routes.py::TestPipelinesProgressEndpoint::test_get_progress_success - assert 500 == 200
FAILED ml/dashboard/tests/test_pipelines_routes.py::TestPipelinesProgressEndpoint::test_get_progress_not_found - assert 500 == 404
FAILED ml/dashboard/tests/test_pipelines_routes.py::TestPipelinesProgressEndpoint::test_get_progress_unavailable - assert 500 == 503
FAILED ml/dashboard/tests/test_pipelines_routes.py::TestPipelinesCancelEndpoint::test_cancel_job_success - assert 404 == 200
FAILED ml/dashboard/tests/test_pipelines_routes.py::TestPipelinesCancelEndpoint::test_cancel_job_unavailable - assert 404 == 503
FAILED ml/dashboard/tests/test_pipelines_routes.py::TestPipelinesIntegration::test_full_pipeline_workflow - assert 503 == 202
============================ 370 failed, 6463 passed, 19 skipped, 122 deselected, 1 xfailed, 1 xpassed, 4 warnings, 33 errors in 2773.67s (0:46:13) ============================
{"event": "Test session completed with exit status: 1", "level": "info", "timestamp": "2025-12-09T18:40:37.957039Z", "logger": "ml.tests.conftest"}
```
