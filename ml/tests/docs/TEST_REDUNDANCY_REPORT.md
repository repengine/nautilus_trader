# Test Redundancy Analysis Report
Analyzed 130 test files
Found 1320 test functions

## Similar Test Names (Potential Duplicates)

### Similar group

- test_clean_db_fixture (ml/tests/test_postgres_fixes.py:26)
- test_clean_postgres_db_fixture (ml/tests/test_postgres_integration.py:56)

### Similar group

- test_postgres_connection (ml/tests/test_postgres_integration.py:8)
- test_postgres_connection (ml/tests/test_postgres_simple.py:12)

### Similar group

- TestTestConfiguration.test_config_initialization (ml/tests/test_configuration.py:24)
- TestTestConfiguration.test_test_database_initialization (ml/tests/test_configuration.py:48)

### Similar group

- TestTestConfiguration.test_database_seed_data (ml/tests/test_configuration.py:59)
- TestTestConfiguration.test_database_rollback (ml/tests/test_configuration.py:72)

### Similar group

- TestMockServices.test_mock_databento_client (ml/tests/test_configuration.py:95)
- TestMockServices.test_mock_fred_client (ml/tests/test_configuration.py:118)
- TestMockServices.test_mock_yahoo_client (ml/tests/test_configuration.py:129)

### Similar group

- TestStrategyContracts.test_strategy_receives_ml_signals (ml/tests/contracts/test_strategy_contracts.py:32)
- TestStrategyContracts.test_strategy_handles_multiple_model_signals (ml/tests/contracts/test_strategy_contracts.py:132)
- TestStrategyContracts.test_strategy_handles_conflicting_signals (ml/tests/contracts/test_strategy_contracts.py:253)

### Similar group

- TestStoreSchemaContracts.test_feature_input_schema_validation (ml/tests/contracts/test_store_schemas.py:290)
- TestStoreSchemaContracts.test_prediction_schema_validation (ml/tests/contracts/test_store_schemas.py:326)
- TestStoreSchemaContracts.test_signal_schema_validation (ml/tests/contracts/test_store_schemas.py:367)
- TestStoreSchemaContracts.test_watermark_schema_validation (ml/tests/contracts/test_store_schemas.py:395)

### Similar group

- TestConfigurationCombinations.test_feature_config_pairwise (ml/tests/combinatorial/test_config_combinations.py:29)
- TestConfigurationCombinations.test_store_configuration_pairwise (ml/tests/combinatorial/test_config_combinations.py:147)

### Similar group

- TestDataRegistryE2E.test_full_day_pipeline_json (ml/tests/e2e/test_data_registry_e2e.py:407)
- TestDataRegistryE2E.test_full_day_pipeline_postgres (ml/tests/e2e/test_data_registry_e2e.py:733)

### Similar group

- TestDataRegistryE2E.test_concurrent_access_json (ml/tests/e2e/test_data_registry_e2e.py:493)
- TestDataRegistryE2E.test_data_contracts_json (ml/tests/e2e/test_data_registry_e2e.py:597)

### Similar group

- TestFeatureStoreInvariants.test_timestamp_monotonicity_invariant (ml/tests/property/test_store_invariants.py:91)
- TestFeatureStoreInvariants.test_feature_immutability_invariant (ml/tests/property/test_store_invariants.py:138)

### Similar group

- TestStrategyStoreInvariants.test_signal_ordering_invariant (ml/tests/property/test_store_invariants.py:290)
- TestDataStoreInvariants.test_event_ordering_invariant (ml/tests/property/test_store_invariants.py:401)

### Similar group

- TestFeatureTransformMetamorphic.test_price_scaling_invariance (ml/tests/metamorphic/test_feature_transforms.py:39)
- TestFeatureTransformMetamorphic.test_data_duplication_invariance (ml/tests/metamorphic/test_feature_transforms.py:197)

### Similar group

- TestFeatureCompositionMetamorphic.test_feature_subset_consistency (ml/tests/metamorphic/test_feature_transforms.py:268)
- TestFeatureCompositionMetamorphic.test_concatenation_consistency (ml/tests/metamorphic/test_feature_transforms.py:323)

### Similar group

- TestInfrastructure.test_ml_signals (ml/tests/integration/test_infrastructure.py:55)
- TestInfrastructure.test_ml_config (ml/tests/integration/test_infrastructure.py:72)

### Similar group

- TestEngineManagerIntegration.test_test_environment_detection (ml/tests/integration/test_engine_manager_stores.py:131)
- TestEngineManager.test_test_environment_detection (ml/tests/unit/core/test_db_engine.py:142)

### Similar group

- TestSchedulerMetrics.test_metrics_server_disabled (ml/tests/integration/test_scheduler_metrics.py:87)
- TestSchedulerMetrics.test_metrics_server_port_conflict (ml/tests/integration/test_scheduler_metrics.py:225)

### Similar group

- TestSchedulerMetrics.test_collection_error_metrics (ml/tests/integration/test_scheduler_metrics.py:127)
- TestSchedulerMetrics.test_cleanup_metrics (ml/tests/integration/test_scheduler_metrics.py:277)

### Similar group

- TestModelStore.test_write_prediction (ml/tests/integration/test_stores_integration.py:183)
- TestModelStore.test_read_latest_predictions (ml/tests/integration/test_stores_integration.py:206)

### Similar group

- TestStrategyStore.test_write_signal (ml/tests/integration/test_stores_integration.py:258)
- TestStrategyStore.test_read_active_signals (ml/tests/integration/test_stores_integration.py:281)

### Similar group

- TestDataProcessor.test_process_market_data (ml/tests/integration/test_stores_integration.py:314)
- TestDataProcessor.test_process_market_data_with_crossed_market (ml/tests/integration/test_stores_integration.py:339)
- TestDataProcessor.test_process_prediction (ml/tests/integration/test_stores_integration.py:385)
- TestDataProcessorSimple.test_process_market_data_simple (ml/tests/integration/test_stores_integration.py:577)

### Similar group

- TestDataProcessor.test_process_features_with_nan (ml/tests/integration/test_stores_integration.py:362)
- TestDataProcessorSimple.test_process_features_with_nan (ml/tests/integration/test_stores_integration.py:640)

### Similar group

- TestIntegration.test_end_to_end_flow (ml/tests/integration/test_stores_integration.py:473)
- TestIntegration.test_end_to_end_error_recovery (ml/tests/unit/data/test_error_handling_comprehensive.py:1326)

### Similar group

- TestSchedulerFeatureStoreIntegration.test_feature_computation_with_catalog_data (ml/tests/integration/test_scheduler_feature_store.py:152)
- TestSchedulerFeatureStoreIntegration.test_feature_computation_disabled (ml/tests/integration/test_scheduler_feature_store.py:195)
- TestSchedulerFeatureStoreIntegration.test_feature_computation_without_engineer (ml/tests/integration/test_scheduler_feature_store.py:218)

### Similar group

- TestSchedulerFeatureStoreIntegration.test_feature_store_initialization_failure (ml/tests/integration/test_scheduler_feature_store.py:236)
- TestSchedulerFeatureStoreIntegration.test_feature_store_connection_from_env (ml/tests/integration/test_scheduler_feature_store.py:288)
- TestSchedulerResilience.test_feature_store_initialization_failures (ml/tests/unit/data/test_scheduler_resilience.py:149)

### Similar group

- TestTransformProviderIntegration.test_calendar_transform_integration (ml/tests/integration/test_transform_provider_integration.py:30)
- TestTransformProviderIntegration.test_metadata_transform_integration (ml/tests/integration/test_transform_provider_integration.py:73)
- TestTransformProviderIntegration.test_event_transform_integration (ml/tests/integration/test_transform_provider_integration.py:106)

### Similar group

- TestTransformProviderIntegration.test_provider_scalability (ml/tests/integration/test_transform_provider_integration.py:244)
- TestTransformProviderIntegration.test_provider_caching_efficiency (ml/tests/integration/test_transform_provider_integration.py:273)

### Similar group

- TestCalendarProviderIntegration.test_factory_creates_pandas_calendar_by_default (ml/tests/integration/test_calendar_provider_integration.py:28)
- TestCalendarProviderIntegration.test_factory_get_calendar_provider (ml/tests/integration/test_calendar_provider_integration.py:122)

### Similar group

- TestCalendarProviderIntegration.test_provider_with_real_pandas_source (ml/tests/integration/test_calendar_provider_integration.py:67)
- TestCalendarProviderIntegration.test_provider_with_mock_source (ml/tests/integration/test_calendar_provider_integration.py:93)
- TestCalendarProviderIntegration.test_provider_month_boundaries (ml/tests/integration/test_calendar_provider_integration.py:195)
- TestCalendarProviderIntegration.test_provider_with_empty_timestamps (ml/tests/integration/test_calendar_provider_integration.py:219)

### Similar group

- TestCalendarProviderIntegration.test_provider_handles_multiple_exchanges (ml/tests/integration/test_calendar_provider_integration.py:134)
- TestCalendarProviderIntegration.test_provider_cyclic_encodings (ml/tests/integration/test_calendar_provider_integration.py:158)

### Similar group

- TestConcurrentWrites.test_feature_store_concurrent_writes (ml/tests/integration/test_stores_concurrency.py:205)
- TestConcurrentWrites.test_model_store_concurrent_predictions (ml/tests/integration/test_stores_concurrency.py:296)
- TestConcurrentWrites.test_strategy_store_concurrent_signals (ml/tests/integration/test_stores_concurrency.py:367)

### Similar group

- TestEdgeCases.test_empty_batch_handling (ml/tests/integration/test_stores_concurrency.py:1457)
- TestEdgeCases.test_empty_dataset_handling (ml/tests/unit/data/test_error_handling_comprehensive.py:833)

### Similar group

- TestEndToEndPipeline.test_pipeline_with_real_databento_data (ml/tests/integration/test_end_to_end_pipeline.py:210)
- TestEndToEndPipeline.test_pipeline_with_mock_data (ml/tests/integration/test_end_to_end_pipeline.py:241)

### Similar group

- TestEndToEndPipeline.test_signal_generation_from_features (ml/tests/integration/test_end_to_end_pipeline.py:324)
- TestEndToEndProperties.test_signal_generation_properties (ml/tests/unit/test_ml_hypothesis_comprehensive.py:226)

### Similar group

- TestEndToEndPipeline.test_tft_dataset_integration (ml/tests/integration/test_end_to_end_pipeline.py:501)
- TestEndToEndPipeline.test_provider_integration (ml/tests/integration/test_end_to_end_pipeline.py:545)

### Similar group

- TestStorePersistence.test_feature_store_persistence (ml/tests/integration/test_store_persistence.py:21)
- TestStorePersistence.test_model_store_persistence (ml/tests/integration/test_store_persistence.py:70)
- TestStorePersistence.test_strategy_store_persistence (ml/tests/integration/test_store_persistence.py:122)

### Similar group

- TestStorePersistence.test_store_failure_handling (ml/tests/integration/test_store_persistence.py:176)
- TestStorePersistence.test_store_is_healthy (ml/tests/integration/test_store_persistence.py:187)

### Similar group

- TestDataSchedulerIntegration.test_scheduler_initialization (ml/tests/integration/test_scheduler_databento.py:36)
- TestDataSchedulerIntegration.test_scheduler_status (ml/tests/integration/test_scheduler_databento.py:100)

### Similar group

- TestDataSchedulerIntegration.test_collect_symbol_data_success (ml/tests/integration/test_scheduler_databento.py:134)
- TestDataSchedulerIntegration.test_collect_symbol_data_retry_logic (ml/tests/integration/test_scheduler_databento.py:183)

### Similar group

- TestL2L3RegistryStoreIntegration.test_feature_registry_manifest_with_l2_features (ml/tests/integration/test_registry_store_l2_integration.py:118)
- TestL2L3RegistryStoreIntegration.test_feature_store_computes_l2_features (ml/tests/integration/test_registry_store_l2_integration.py:193)

### Similar group

- TestFeatureStoreIntegration.test_ml_signal_actor_with_feature_store (ml/tests/integration/test_feature_store_integration.py:41)
- TestFeatureStoreIntegration.test_ml_signal_actor_without_feature_store (ml/tests/integration/test_feature_store_integration.py:97)
- TestFeatureStoreIntegration.test_training_with_feature_store (ml/tests/integration/test_feature_store_integration.py:127)

### Similar group

- TestBackwardCompatibility.test_ml_signal_actor_backward_compatibility (ml/tests/integration/test_feature_store_integration.py:331)
- TestBackwardCompatibility.test_training_backward_compatibility (ml/tests/integration/test_feature_store_integration.py:359)

### Similar group

- TestIndicatorManagerProperties.test_indicator_initialization (ml/tests/unit/test_ml_property_comprehensive.py:180)
- TestIndicatorManagerProperties.test_indicator_determinism (ml/tests/unit/test_ml_property_comprehensive.py:213)
- TestIndicatorManager.test_indicator_manager_initialization (ml/tests/unit/features/test_feature_engineering.py:304)

### Similar group

- TestModelRegistryProperties.test_registry_version_ordering (ml/tests/unit/test_ml_property_comprehensive.py:282)
- TestModelRegistryProperties.test_registry_isolation (ml/tests/unit/test_ml_property_comprehensive.py:330)

### Similar group

- TestFeatureEngineeringExtended.test_feature_shape_consistency (ml/tests/unit/test_ml_property_comprehensive.py:473)
- TestFeatureEngineeringExtended.test_feature_nan_handling (ml/tests/unit/test_ml_property_comprehensive.py:512)

### Similar group

- TestEndToEndProperties.test_metric_validity (ml/tests/unit/test_ml_hypothesis_comprehensive.py:313)
- TestEndToEndProperties.test_state_machine_validity (ml/tests/unit/test_ml_hypothesis_comprehensive.py:356)

### Similar group

- TestZeroAllocationHotPath.test_ring_buffer_get_last_returns_view (ml/tests/performance/test_zero_allocation.py:68)
- TestZeroAllocationHotPath.test_ring_buffer_get_window_returns_view (ml/tests/performance/test_zero_allocation.py:88)
- TestZeroAllocationHotPath.test_reservoir_sampler_get_sample_returns_view (ml/tests/performance/test_zero_allocation.py:127)
- TestZeroAllocationHotPath.test_feature_cache_returns_views (ml/tests/performance/test_zero_allocation.py:144)

### Similar group

- TestFeatureComputationBenchmarks.test_feature_computation_p99_latency (ml/tests/performance/test_ml_hot_path_benchmarks.py:207)
- TestFeatureComputationBenchmarks.test_feature_computation_throughput (ml/tests/performance/test_ml_hot_path_benchmarks.py:270)

### Similar group

- TestModelInferenceBenchmarks.test_onnx_inference_p99_latency (ml/tests/performance/test_ml_hot_path_benchmarks.py:407)
- TestModelInferenceBenchmarks.test_model_swap_latency (ml/tests/performance/test_ml_hot_path_benchmarks.py:485)

### Similar group

- TestEnhancedModelRegistry.test_register_with_quality_gates_pass (ml/tests/unit/registry/test_enhanced_registry.py:83)
- TestEnhancedModelRegistry.test_register_with_quality_gates_fail (ml/tests/unit/registry/test_enhanced_registry.py:110)

### Similar group

- TestEnhancedModelRegistry.test_start_canary_deployment (ml/tests/unit/registry/test_enhanced_registry.py:163)
- TestEnhancedModelRegistry.test_update_canary_metrics (ml/tests/unit/registry/test_enhanced_registry.py:206)

### Similar group

- TestEnhancedModelRegistry.test_evaluate_canary_for_promotion (ml/tests/unit/registry/test_enhanced_registry.py:245)
- TestEnhancedModelRegistry.test_evaluate_canary_for_rollback (ml/tests/unit/registry/test_enhanced_registry.py:295)

### Similar group

- TestStatisticalUtilities.test_welch_t_test_detects_significant_difference (ml/tests/unit/registry/test_registry_statistics.py:18)
- TestStatisticalUtilities.test_welch_t_test_no_difference (ml/tests/unit/registry/test_registry_statistics.py:34)

### Similar group

- TestStatisticalUtilities.test_welch_t_test_handles_small_samples (ml/tests/unit/registry/test_registry_statistics.py:47)
- TestStatisticalUtilities.test_welch_t_test_handles_insufficient_samples (ml/tests/unit/registry/test_registry_statistics.py:58)
- TestStatisticalUtilities.test_welch_t_test_handles_zero_variance (ml/tests/unit/registry/test_registry_statistics.py:69)

### Similar group

- TestStatisticalUtilities.test_calculate_sample_size_standard_case (ml/tests/unit/registry/test_registry_statistics.py:79)
- TestStatisticalUtilities.test_calculate_sample_size_small_effect (ml/tests/unit/registry/test_registry_statistics.py:88)
- TestStatisticalUtilities.test_calculate_sample_size_zero_effect (ml/tests/unit/registry/test_registry_statistics.py:95)
- TestStatisticalUtilities.test_calculate_sample_size_high_power (ml/tests/unit/registry/test_registry_statistics.py:101)

### Similar group

- TestStatisticalUtilities.test_relative_improvement_calculation (ml/tests/unit/registry/test_registry_statistics.py:109)
- TestStatisticalUtilities.test_relative_improvement_handles_zero_baseline (ml/tests/unit/registry/test_registry_statistics.py:121)

### Similar group

- TestModelRegistryBackends.test_json_backend_register_and_retrieve (ml/tests/unit/registry/test_postgres_backend.py:89)
- TestModelRegistryBackends.test_postgres_backend_register_and_retrieve (ml/tests/unit/registry/test_postgres_backend.py:130)
- TestFeatureRegistryBackends.test_json_backend_register_and_retrieve (ml/tests/unit/registry/test_postgres_backend.py:190)
- TestStrategyRegistryBackends.test_json_backend_register_and_retrieve (ml/tests/unit/registry/test_postgres_backend.py:285)

### Similar group

- TestFeatureRegistryBackends.test_postgres_backend_with_lifecycle (ml/tests/unit/registry/test_postgres_backend.py:231)
- TestStrategyRegistryBackends.test_postgres_backend_with_compatibility (ml/tests/unit/registry/test_postgres_backend.py:342)

### Similar group

- TestUnifiedRegistry.test_register_student_model_with_lineage (ml/tests/unit/registry/test_unified_registry.py:99)
- TestUnifiedRegistry.test_get_model_lineage (ml/tests/unit/registry/test_unified_registry.py:193)

### Similar group

- TestUnifiedRegistry.test_get_models_by_role (ml/tests/unit/registry/test_unified_registry.py:136)
- TestUnifiedRegistry.test_get_models_by_data_requirements (ml/tests/unit/registry/test_unified_registry.py:164)

### Similar group

- TestModelContracts.test_valid_teacher_contract (ml/tests/unit/registry/test_model_contracts.py:462)
- TestModelContracts.test_valid_student_contract (ml/tests/unit/registry/test_model_contracts.py:473)

### Similar group

- TestModelContracts.test_invalid_student_with_l2_data (ml/tests/unit/registry/test_model_contracts.py:491)
- TestModelContracts.test_invalid_student_high_latency (ml/tests/unit/registry/test_model_contracts.py:504)

### Similar group

- TestRegistryPerformance.test_registry_bulk_registration_performance (ml/tests/unit/registry/test_registry_performance.py:30)
- TestRegistryPerformance.test_registry_concurrent_read_performance (ml/tests/unit/registry/test_registry_performance.py:75)
- TestRegistryPerformance.test_registry_path_validation_performance (ml/tests/unit/registry/test_registry_performance.py:139)
- TestRegistryPerformance.test_registry_security_rejection_performance (ml/tests/unit/registry/test_registry_performance.py:173)
- TestRegistryPerformance.test_registry_persistence_performance (ml/tests/unit/registry/test_registry_performance.py:199)

### Similar group

- TestRegistryDeployment.test_registry_basic_deploy (ml/tests/unit/registry/test_deployment_manager.py:29)
- TestRegistryDeployment.test_registry_hot_reload (ml/tests/unit/registry/test_deployment_manager.py:80)
- TestRegistryDeployment.test_registry_gradual_rollout (ml/tests/unit/registry/test_deployment_manager.py:148)

### Similar group

- TestCanaryDeployment.test_record_metric_success (ml/tests/unit/registry/test_registry_canary.py:51)
- TestCanaryDeployment.test_record_metric_error (ml/tests/unit/registry/test_registry_canary.py:73)
- TestCanaryDeployment.test_should_promote_success (ml/tests/unit/registry/test_registry_canary.py:152)

### Similar group

- TestCanaryDeployment.test_should_rollback_high_error_rate (ml/tests/unit/registry/test_registry_canary.py:95)
- TestCanaryDeployment.test_should_rollback_performance_degradation (ml/tests/unit/registry/test_registry_canary.py:115)
- TestCanaryDeployment.test_should_promote_high_error_rate (ml/tests/unit/registry/test_registry_canary.py:206)

### Similar group

- TestCanaryDeployment.test_should_rollback_insufficient_samples (ml/tests/unit/registry/test_registry_canary.py:134)
- TestCanaryDeployment.test_should_promote_insufficient_samples (ml/tests/unit/registry/test_registry_canary.py:186)

### Similar group

- TestStrategyRegistry.test_registry_initialization (ml/tests/unit/registry/test_strategy_registry.py:231)
- TestStrategyRegistry.test_register_strategy (ml/tests/unit/registry/test_strategy_registry.py:242)

### Similar group

- TestStrategyRegistry.test_get_strategy (ml/tests/unit/registry/test_strategy_registry.py:267)
- TestStrategyRegistry.test_get_strategy_lineage (ml/tests/unit/registry/test_strategy_registry.py:534)

### Similar group

- TestStrategyRegistry.test_filter_by_regime (ml/tests/unit/registry/test_strategy_registry.py:293)
- TestStrategyRegistry.test_filter_by_instrument_type (ml/tests/unit/registry/test_strategy_registry.py:326)

### Similar group

- TestStrategyRegistry.test_update_performance_metrics (ml/tests/unit/registry/test_strategy_registry.py:348)
- TestStrategyRegistry.test_rank_by_performance (ml/tests/unit/registry/test_strategy_registry.py:383)

### Similar group

- TestMLStrategyNode.test_setup_with_dry_run_mode (ml/tests/unit/deployment/test_entrypoint_strategy.py:56)
- TestMLStrategyNode.test_setup_with_live_mode (ml/tests/unit/deployment/test_entrypoint_strategy.py:78)
- TestMLStrategyNode.test_setup_with_strategy_store (ml/tests/unit/deployment/test_entrypoint_strategy.py:116)
- TestMLStrategyNode.test_setup_without_strategy_store (ml/tests/unit/deployment/test_entrypoint_strategy.py:136)
- TestMLStrategyNode.test_setup_with_dry_run_mode (ml/tests/unit/deployment/test_entrypoint_strategy_backup.py:49)

### Similar group

- TestMLStrategyNode.test_setup_risk_parameters (ml/tests/unit/deployment/test_entrypoint_strategy.py:97)
- TestMLStrategyNode.test_setup_risk_parameters (ml/tests/unit/deployment/test_entrypoint_strategy_backup.py:90)

### Similar group

- TestMLStrategyNode.test_setup_with_databento_api_key (ml/tests/unit/deployment/test_entrypoint_strategy.py:154)
- TestMLStrategyNode.test_setup_without_databento_api_key (ml/tests/unit/deployment/test_entrypoint_strategy.py:173)
- TestMLStrategyNode.test_setup_with_databento_api_key (ml/tests/unit/deployment/test_entrypoint_strategy_backup.py:147)

### Similar group

- TestMLStrategyNode.test_environment_variable_parsing (ml/tests/unit/deployment/test_entrypoint_strategy.py:322)
- TestMLSignalActorNode.test_environment_variable_parsing (ml/tests/unit/deployment/test_entrypoint_actor.py:248)
- TestMLStrategyNode.test_environment_variable_parsing (ml/tests/unit/deployment/test_entrypoint_strategy_backup.py:289)

### Similar group

- TestMLStrategyNode.test_dry_run_mode_output (ml/tests/unit/deployment/test_entrypoint_strategy.py:356)
- TestMLStrategyNode.test_dry_run_mode_output (ml/tests/unit/deployment/test_entrypoint_strategy_backup.py:323)

### Similar group

- TestMainFunction.test_main_successful_run (ml/tests/unit/deployment/test_entrypoint_strategy.py:402)
- TestMainFunction.test_main_successful_run (ml/tests/unit/deployment/test_entrypoint_actor.py:342)

### Similar group

- TestMainFunction.test_main_handles_keyboard_interrupt (ml/tests/unit/deployment/test_entrypoint_strategy.py:416)
- TestMainFunction.test_main_handles_fatal_error (ml/tests/unit/deployment/test_entrypoint_strategy.py:428)
- TestMainFunction.test_main_handles_keyboard_interrupt (ml/tests/unit/deployment/test_entrypoint_actor.py:356)

### Similar group

- TestMainFunction.test_main_prints_startup_info (ml/tests/unit/deployment/test_entrypoint_strategy.py:440)
- TestMainFunction.test_main_prints_startup_info (ml/tests/unit/deployment/test_entrypoint_actor.py:380)

### Similar group

- TestHealthEndpoint.test_health_check_healthy (ml/tests/unit/deployment/test_entrypoint_pipeline.py:40)
- TestHealthEndpoint.test_health_check_unhealthy (ml/tests/unit/deployment/test_entrypoint_pipeline.py:55)

### Similar group

- TestPipelineRunner.test_init (ml/tests/unit/deployment/test_entrypoint_pipeline.py:97)
- TestPipelineRunner.test_initialize_stores (ml/tests/unit/deployment/test_entrypoint_pipeline.py:156)

### Similar group

- TestPipelineRunner.test_signal_handler (ml/tests/unit/deployment/test_entrypoint_pipeline.py:104)
- TestPipelineRunner.test_signal_handler_no_scheduler (ml/tests/unit/deployment/test_entrypoint_pipeline.py:120)
- TestPipelineRunner.test_signal_registration (ml/tests/unit/deployment/test_entrypoint_pipeline.py:369)
- TestMLPipelineRunner.test_signal_handlers_setup (ml/tests/unit/scripts/test_run_ml_pipeline.py:58)

### Similar group

- TestPipelineRunner.test_run_backfill_mode (ml/tests/unit/deployment/test_entrypoint_pipeline.py:197)
- TestPipelineRunner.test_run_daily_mode (ml/tests/unit/deployment/test_entrypoint_pipeline.py:215)
- TestPipelineRunner.test_run_realtime_mode (ml/tests/unit/deployment/test_entrypoint_pipeline.py:232)
- TestPipelineRunner.test_run_invalid_mode (ml/tests/unit/deployment/test_entrypoint_pipeline.py:249)
- TestMLPipelineRunner.test_run_backfill_dry_run_mode (ml/tests/unit/scripts/test_run_ml_pipeline.py:217)
- TestMLPipelineRunner.test_run_daily_dry_run_mode (ml/tests/unit/scripts/test_run_ml_pipeline.py:266)

### Similar group

- TestPipelineRunner.test_run_backfill_executes_daily_update (ml/tests/unit/deployment/test_entrypoint_pipeline.py:279)
- TestMLPipelineRunner.test_run_daily_executes_update (ml/tests/unit/scripts/test_run_ml_pipeline.py:280)

### Similar group

- TestPipelineRunner.test_run_daily_schedules_updates (ml/tests/unit/deployment/test_entrypoint_pipeline.py:291)
- TestMLPipelineRunner.test_run_daily_handles_scheduler_none (ml/tests/unit/scripts/test_run_ml_pipeline.py:291)

### Similar group

- TestPipelineRunner.test_run_realtime_handles_errors (ml/tests/unit/deployment/test_entrypoint_pipeline.py:329)
- TestMLPipelineRunner.test_run_realtime_handles_keyboard_interrupt (ml/tests/unit/scripts/test_run_ml_pipeline.py:312)

### Similar group

- TestMainFunction.test_main_starts_health_server (ml/tests/unit/deployment/test_entrypoint_pipeline.py:397)
- TestMainFunction.test_main_all_healthy (ml/tests/unit/deployment/test_check_health.py:293)
- TestMainFunction.test_main_all_unhealthy (ml/tests/unit/deployment/test_check_health.py:332)

### Similar group

- TestMLSignalActorNode.test_setup_with_valid_config (ml/tests/unit/deployment/test_entrypoint_actor.py:54)
- TestMLSignalActorNode.test_setup_without_api_key_exits (ml/tests/unit/deployment/test_entrypoint_actor.py:85)
- TestMLSignalActorNode.test_setup_with_database_connection (ml/tests/unit/deployment/test_entrypoint_actor.py:111)

### Similar group

- TestMLSignalActorNode.test_feature_config_creation (ml/tests/unit/deployment/test_entrypoint_actor.py:280)
- TestMLSignalActorNode.test_databento_config_creation (ml/tests/unit/deployment/test_entrypoint_actor.py:303)
- TestMLSignalActor.test_feature_config_initialization (ml/tests/unit/actors/test_signal_actor.py:1189)

### Similar group

- TestServiceHealthChecks.test_check_service_health_success (ml/tests/unit/deployment/test_check_health.py:31)
- TestServiceHealthChecks.test_check_service_health_failure (ml/tests/unit/deployment/test_check_health.py:40)
- TestServiceHealthChecks.test_check_service_health_exception (ml/tests/unit/deployment/test_check_health.py:49)
- TestServiceHealthChecks.test_check_postgres_healthy (ml/tests/unit/deployment/test_check_health.py:59)
- TestServiceHealthChecks.test_check_redis_healthy (ml/tests/unit/deployment/test_check_health.py:86)
- TestServiceHealthChecks.test_check_redis_unhealthy (ml/tests/unit/deployment/test_check_health.py:103)
- TestServiceHealthChecks.test_check_ml_pipeline_healthy (ml/tests/unit/deployment/test_check_health.py:115)
- TestServiceHealthChecks.test_check_grafana_healthy (ml/tests/unit/deployment/test_check_health.py:179)

### Similar group

- TestServiceHealthChecks.test_check_postgres_unhealthy (ml/tests/unit/deployment/test_check_health.py:75)
- TestServiceHealthChecks.test_check_ml_pipeline_unhealthy (ml/tests/unit/deployment/test_check_health.py:127)
- TestServiceHealthChecks.test_check_prometheus_healthy (ml/tests/unit/deployment/test_check_health.py:156)
- TestServiceHealthChecks.test_check_prometheus_unhealthy (ml/tests/unit/deployment/test_check_health.py:168)
- TestServiceHealthChecks.test_check_grafana_unhealthy (ml/tests/unit/deployment/test_check_health.py:191)

### Similar group

- TestServiceHealthChecks.test_check_ml_pipeline_connection_error (ml/tests/unit/deployment/test_check_health.py:138)
- TestServiceHealthChecks.test_check_ml_pipeline_timeout (ml/tests/unit/deployment/test_check_health.py:147)

### Similar group

- TestServiceHealthChecks.test_check_docker_compose_all_running (ml/tests/unit/deployment/test_check_health.py:202)
- TestServiceHealthChecks.test_check_docker_compose_missing_service (ml/tests/unit/deployment/test_check_health.py:225)
- TestServiceHealthChecks.test_check_docker_compose_service_not_running (ml/tests/unit/deployment/test_check_health.py:243)
- TestServiceHealthChecks.test_check_docker_compose_command_error (ml/tests/unit/deployment/test_check_health.py:261)
- TestServiceHealthChecks.test_check_docker_compose_invalid_json (ml/tests/unit/deployment/test_check_health.py:272)

### Similar group

- TestPreflightCheck.test_preflight_check_valid_data (ml/tests/unit/stores/test_data_store_validation.py:199)
- TestPreflightCheck.test_preflight_check_type_mismatch (ml/tests/unit/stores/test_data_store_validation.py:237)

### Similar group

- TestContractValidation.test_type_validation (ml/tests/unit/stores/test_data_store_validation.py:307)
- TestContractValidation.test_null_validation (ml/tests/unit/stores/test_data_store_validation.py:330)
- TestContractValidation.test_range_validation (ml/tests/unit/stores/test_data_store_validation.py:356)
- TestContractValidation.test_uniqueness_validation (ml/tests/unit/stores/test_data_store_validation.py:383)
- TestContractValidation.test_monotonicity_validation (ml/tests/unit/stores/test_data_store_validation.py:407)
- TestContractValidation.test_lateness_validation (ml/tests/unit/stores/test_data_store_validation.py:431)

### Similar group

- TestFailClosedWrites.test_write_rejected_on_validation_failure (ml/tests/unit/stores/test_data_store_validation.py:495)
- TestFailClosedWrites.test_write_rejected_on_preflight_failure (ml/tests/unit/stores/test_data_store_validation.py:517)

### Similar group

- TestIntegration.test_full_validation_pipeline (ml/tests/unit/stores/test_data_store_validation.py:1215)
- TestIntegration.test_validation_performance (ml/tests/unit/stores/test_data_store_validation.py:1266)

### Similar group

- TestModelRegistryConfig.test_default_values (ml/tests/unit/config/test_config_classes.py:22)
- TestModelRegistryConfig.test_custom_values (ml/tests/unit/config/test_config_classes.py:34)

### Similar group

- TestModelDeploymentConfig.test_immediate_deployment (ml/tests/unit/config/test_config_classes.py:119)
- TestModelDeploymentConfig.test_gradual_deployment (ml/tests/unit/config/test_config_classes.py:134)

### Similar group

- TestCanaryDeploymentConfig.test_default_canary_config (ml/tests/unit/config/test_config_classes.py:166)
- TestCanaryDeploymentConfig.test_custom_canary_config (ml/tests/unit/config/test_config_classes.py:180)

### Similar group

- TestMLActorConfiguration.test_ml_actor_config_creation (ml/tests/unit/config/test_ml_actor_config.py:27)
- TestMLActorConfiguration.test_simple_ml_actor_initialization (ml/tests/unit/config/test_ml_actor_config.py:72)

### Similar group

- TestMLActorConfiguration.test_configuration_helper_get_bar_type (ml/tests/unit/config/test_ml_actor_config.py:97)
- TestMLActorConfiguration.test_configuration_helper_get_instrument_id (ml/tests/unit/config/test_ml_actor_config.py:116)
- TestMLActorConfiguration.test_configuration_helper_get_model_path (ml/tests/unit/config/test_ml_actor_config.py:135)
- TestMLActorConfiguration.test_configuration_helper_missing_attribute_raises (ml/tests/unit/config/test_ml_actor_config.py:154)

### Similar group

- test_plan_backfill_with_gaps (ml/tests/unit/cli/test_cli_coverage_backfill.py:25)
- test_plan_backfill_no_gaps (ml/tests/unit/cli/test_cli_coverage_backfill.py:175)

### Similar group

- TestTFTDatasetBuilderErrors.test_builder_without_feature_store (ml/tests/unit/data/test_tft_and_fred_error_handling.py:92)
- TestTFTDatasetBuilderErrors.test_empty_features_from_store (ml/tests/unit/data/test_tft_and_fred_error_handling.py:109)
- TestTFTDatasetBuilderWithFeatureStore.test_init_without_feature_store (ml/tests/unit/data/test_tft_dataset_builder_store.py:88)

### Similar group

- TestTFTDatasetBuilderErrors.test_mismatched_feature_dimensions (ml/tests/unit/data/test_tft_and_fred_error_handling.py:128)
- TestTFTDatasetBuilderErrors.test_static_features_generation_errors (ml/tests/unit/data/test_tft_and_fred_error_handling.py:239)

### Similar group

- TestFREDDataLoaderErrors.test_fredapi_import_failure (ml/tests/unit/data/test_tft_and_fred_error_handling.py:366)
- TestFREDDataLoaderErrors.test_api_connection_failures (ml/tests/unit/data/test_tft_and_fred_error_handling.py:374)

### Similar group

- TestFREDDataLoaderErrors.test_api_rate_limiting (ml/tests/unit/data/test_tft_and_fred_error_handling.py:396)
- TestFREDDataLoader.test_rate_limiting (ml/tests/unit/data/test_fred_loader.py:244)

### Similar group

- TestFREDDataLoaderErrors.test_cache_ttl_expiration (ml/tests/unit/data/test_tft_and_fred_error_handling.py:455)
- TestFREDDataLoader.test_cache_expiry (ml/tests/unit/data/test_fred_loader.py:223)

### Similar group

- TestFREDDataLoaderErrors.test_empty_api_response (ml/tests/unit/data/test_tft_and_fred_error_handling.py:489)
- TestFREDDataLoaderErrors.test_malformed_api_response (ml/tests/unit/data/test_tft_and_fred_error_handling.py:505)

### Similar group

- TestFREDDataLoaderErrors.test_data_store_integration_errors (ml/tests/unit/data/test_tft_and_fred_error_handling.py:561)
- TestFREDDataLoaderErrors.test_registry_integration_errors (ml/tests/unit/data/test_tft_and_fred_error_handling.py:586)

### Similar group

- TestSchedulerResilience.test_data_registry_initialization_failures (ml/tests/unit/data/test_scheduler_resilience.py:126)
- TestSchedulerResilience.test_data_registry_event_emission_failures (ml/tests/unit/data/test_scheduler_resilience.py:405)

### Similar group

- TestSchedulerResilience.test_temporary_file_handling_errors (ml/tests/unit/data/test_scheduler_resilience.py:270)
- TestSchedulerResilience.test_dbn_file_loading_errors (ml/tests/unit/data/test_scheduler_resilience.py:343)

### Similar group

- TestCatalogUtils.test_bars_to_dataframe_empty (ml/tests/unit/data/test_catalog_utils.py:135)
- TestCatalogUtils.test_bars_to_dataframe_with_data (ml/tests/unit/data/test_catalog_utils.py:162)
- TestCatalogUtils.test_quotes_to_dataframe_empty (ml/tests/unit/data/test_catalog_utils.py:215)
- TestCatalogUtils.test_trades_to_dataframe_empty (ml/tests/unit/data/test_catalog_utils.py:290)
- TestCatalogUtils.test_trades_to_dataframe_with_data (ml/tests/unit/data/test_catalog_utils.py:315)
- TestCatalogUtils.test_bars_to_dataframe_basic (ml/tests/unit/data/test_data_loader.py:107)
- TestCatalogUtils.test_bars_to_dataframe_empty (ml/tests/unit/data/test_data_loader.py:146)
- TestCatalogUtils.test_quotes_to_dataframe_basic (ml/tests/unit/data/test_data_loader.py:166)
- TestCatalogUtils.test_trades_to_dataframe_basic (ml/tests/unit/data/test_data_loader.py:217)

### Similar group

- TestCatalogUtils.test_invalid_instrument_id_format (ml/tests/unit/data/test_catalog_utils.py:397)
- TestCatalogUtils.test_invalid_instrument_id (ml/tests/unit/data/test_data_loader.py:280)

### Similar group

- TestFREDIndicator.test_indicator_creation (ml/tests/unit/data/test_fred_loader.py:71)
- TestFREDIndicator.test_indicator_defaults (ml/tests/unit/data/test_fred_loader.py:87)

### Similar group

- TestFREDDataLoader.test_custom_indicators (ml/tests/unit/data/test_fred_loader.py:151)
- TestFREDDataLoader.test_fetch_indicator (ml/tests/unit/data/test_fred_loader.py:175)
- TestFREDDataLoader.test_fetch_all_indicators (ml/tests/unit/data/test_fred_loader.py:276)
- TestFREDDataLoader.test_combine_indicators (ml/tests/unit/data/test_fred_loader.py:303)
- TestFREDDataLoader.test_combine_indicators_with_gaps (ml/tests/unit/data/test_fred_loader.py:328)
- TestFREDDataLoader.test_store_indicators (ml/tests/unit/data/test_fred_loader.py:362)

### Similar group

- TestDataCollectorErrorRecovery.test_storage_calculation_with_inaccessible_files (ml/tests/unit/data/test_collector_error_recovery.py:80)
- TestDataCollectorErrorRecovery.test_spread_calculation_with_invalid_data (ml/tests/unit/data/test_collector_error_recovery.py:279)

### Similar group

- TestDataCollectorErrorRecovery.test_l2_depth_collection_with_api_failures (ml/tests/unit/data/test_collector_error_recovery.py:108)
- TestDataCollectorErrorRecovery.test_tbbo_collection_with_empty_responses (ml/tests/unit/data/test_collector_error_recovery.py:193)
- TestDataCollectorErrorRecovery.test_enhanced_collection_pipeline_failures (ml/tests/unit/data/test_collector_error_recovery.py:373)

### Similar group

- TestDataCollectorErrorRecovery.test_l1_trades_collection_storage_limit_handling (ml/tests/unit/data/test_collector_error_recovery.py:136)
- TestDataCollectorErrorRecovery.test_minute_bars_collection_rate_limiting (ml/tests/unit/data/test_collector_error_recovery.py:214)

### Similar group

- TestDataCollectorErrorRecovery.test_file_write_failures_during_collection (ml/tests/unit/data/test_collector_error_recovery.py:249)
- TestDataCollectorErrorRecovery.test_storage_limit_enforcement_during_collection (ml/tests/unit/data/test_collector_error_recovery.py:419)

### Similar group

- TestConcurrentCollectionConflicts.test_collector_handles_concurrent_writes (ml/tests/unit/data/test_error_handling_comprehensive.py:563)
- TestConcurrentCollectionConflicts.test_feature_store_concurrent_access (ml/tests/unit/data/test_error_handling_comprehensive.py:625)

### Similar group

- TestCircuitBreaker.test_circuit_breaker_activation (ml/tests/unit/data/test_error_handling_comprehensive.py:1126)
- TestCircuitBreaker.test_circuit_breaker_recovery (ml/tests/unit/data/test_error_handling_comprehensive.py:1166)

### Similar group

- TestTFTDatasetBuilderWithFeatureStore.test_init_with_feature_store (ml/tests/unit/data/test_tft_dataset_builder_store.py:64)
- TestTFTDatasetBuilderWithFeatureStore.test_build_training_dataset_uses_feature_store (ml/tests/unit/data/test_tft_dataset_builder_store.py:268)
- TestTFTDatasetBuilderWithFeatureStore.test_logging_feature_source (ml/tests/unit/data/test_tft_dataset_builder_store.py:351)

### Similar group

- TestTFTDatasetBuilderWithFeatureStore.test_prepare_training_data_from_store_no_store (ml/tests/unit/data/test_tft_dataset_builder_store.py:109)
- TestTFTDatasetBuilderWithFeatureStore.test_prepare_training_data_from_store_success (ml/tests/unit/data/test_tft_dataset_builder_store.py:126)
- TestTFTDatasetBuilderWithFeatureStore.test_prepare_training_data_from_store_no_features (ml/tests/unit/data/test_tft_dataset_builder_store.py:176)
- TestTFTDatasetBuilderWithFeatureStore.test_prepare_training_data_auto_selection_with_store (ml/tests/unit/data/test_tft_dataset_builder_store.py:207)
- TestTFTDatasetBuilderWithFeatureStore.test_prepare_training_data_fallback_to_direct (ml/tests/unit/data/test_tft_dataset_builder_store.py:243)

### Similar group

- TestDataStructure.test_test_data_dir_exists (ml/tests/unit/data/test_data_structure.py:13)
- TestDataStructure.test_model_registry_dir_exists (ml/tests/unit/data/test_data_structure.py:19)

### Similar group

- TestDataStructure.test_xgboost_test_models_exist (ml/tests/unit/data/test_data_structure.py:36)
- TestDataStructure.test_onnx_test_models_exist (ml/tests/unit/data/test_data_structure.py:50)

### Similar group

- TestXGBoostOptunaOptimizer.test_optimizer_initialization (ml/tests/unit/training/test_optuna_optimizer.py:69)
- TestXGBoostOptunaOptimizer.test_optimize (ml/tests/unit/training/test_optuna_optimizer.py:546)

### Similar group

- TestXGBoostOptunaOptimizer.test_ensure_optuna (ml/tests/unit/training/test_optuna_optimizer.py:81)
- TestXGBoostOptunaOptimizer.test_ensure_optuna_not_available (ml/tests/unit/training/test_optuna_optimizer.py:92)
- TestXGBoostOptunaOptimizer.test_create_study (ml/tests/unit/training/test_optuna_optimizer.py:103)
- TestXGBoostOptunaOptimizer.test_create_pruner (ml/tests/unit/training/test_optuna_optimizer.py:214)

### Similar group

- TestXGBoostOptunaOptimizer.test_create_study_with_storage (ml/tests/unit/training/test_optuna_optimizer.py:125)
- TestXGBoostOptunaOptimizer.test_get_study_summary (ml/tests/unit/training/test_optuna_optimizer.py:635)

### Similar group

- TestXGBoostOptunaOptimizer.test_sample_xgboost_params (ml/tests/unit/training/test_optuna_optimizer.py:250)
- TestXGBoostOptunaOptimizer.test_sample_xgboost_params_gpu (ml/tests/unit/training/test_optuna_optimizer.py:295)
- TestXGBoostOptunaOptimizer.test_sample_xgboost_params_regression (ml/tests/unit/training/test_optuna_optimizer.py:331)

### Similar group

- TestXGBoostOptunaOptimizer.test_create_objective_function (ml/tests/unit/training/test_optuna_optimizer.py:370)
- TestXGBoostOptunaOptimizer.test_create_objective_function_regression (ml/tests/unit/training/test_optuna_optimizer.py:438)

### Similar group

- TestXGBoostOptunaOptimizer.test_get_study_summary_no_study (ml/tests/unit/training/test_optuna_optimizer.py:690)
- TestXGBoostOptunaOptimizer.test_get_study_summary_with_values (ml/tests/unit/training/test_optuna_optimizer.py:701)

### Similar group

- TestGrafanaPanelFactory.test_create_stat_panel_basic (ml/tests/unit/monitoring/test_dashboard_factory.py:25)
- TestGrafanaPanelFactory.test_create_stat_panel_with_custom_unit (ml/tests/unit/monitoring/test_dashboard_factory.py:43)
- TestGrafanaPanelFactory.test_create_stat_panel_with_alert_config (ml/tests/unit/monitoring/test_dashboard_factory.py:77)
- TestGrafanaPanelFactory.test_create_stat_panel_datasource_config (ml/tests/unit/monitoring/test_dashboard_factory.py:97)
- TestGrafanaPanelFactory.test_create_timeseries_panel_basic (ml/tests/unit/monitoring/test_dashboard_factory.py:111)
- TestGrafanaPanelFactory.test_create_table_panel_basic (ml/tests/unit/monitoring/test_dashboard_factory.py:197)
- TestGrafanaPanelFactory.test_create_heatmap_panel_basic (ml/tests/unit/monitoring/test_dashboard_factory.py:250)
- TestGrafanaPanelFactory.test_create_row_panel_basic (ml/tests/unit/monitoring/test_dashboard_factory.py:295)
- TestGrafanaPanelFactory.test_create_row_panel_collapsed (ml/tests/unit/monitoring/test_dashboard_factory.py:312)

### Similar group

- TestGrafanaPanelFactory.test_create_stat_panel_with_custom_thresholds (ml/tests/unit/monitoring/test_dashboard_factory.py:57)
- TestGrafanaPanelFactory.test_create_timeseries_panel_with_custom_unit (ml/tests/unit/monitoring/test_dashboard_factory.py:138)
- TestGrafanaPanelFactory.test_create_timeseries_panel_with_custom_legend (ml/tests/unit/monitoring/test_dashboard_factory.py:154)
- TestGrafanaPanelFactory.test_create_table_panel_without_transformations (ml/tests/unit/monitoring/test_dashboard_factory.py:237)
- TestGrafanaPanelFactory.test_create_heatmap_panel_with_custom_color_scheme (ml/tests/unit/monitoring/test_dashboard_factory.py:268)

### Similar group

- TestGrafanaPanelFactory.test_create_timeseries_panel_default_legend (ml/tests/unit/monitoring/test_dashboard_factory.py:176)
- TestGrafanaPanelFactory.test_create_heatmap_panel_default_color_scheme (ml/tests/unit/monitoring/test_dashboard_factory.py:282)

### Similar group

- TestGrafanaDashboardFactory.test_create_base_dashboard_basic (ml/tests/unit/monitoring/test_dashboard_factory.py:339)
- TestGrafanaDashboardFactory.test_create_base_dashboard_with_custom_settings (ml/tests/unit/monitoring/test_dashboard_factory.py:360)
- TestGrafanaDashboardFactory.test_create_base_dashboard_annotations (ml/tests/unit/monitoring/test_dashboard_factory.py:379)
- TestGrafanaDashboardFactory.test_create_base_dashboard_links (ml/tests/unit/monitoring/test_dashboard_factory.py:396)
- TestGrafanaDashboardFactory.test_create_alert_config_basic (ml/tests/unit/monitoring/test_dashboard_factory.py:444)
- TestGrafanaDashboardFactory.test_save_dashboard (ml/tests/unit/monitoring/test_dashboard_factory.py:508)

### Similar group

- TestGrafanaDashboardFactory.test_create_alert_config_with_custom_parameters (ml/tests/unit/monitoring/test_dashboard_factory.py:462)
- TestGrafanaDashboardFactory.test_create_alert_config_structure (ml/tests/unit/monitoring/test_dashboard_factory.py:484)

### Similar group

- test_main_function (ml/tests/unit/monitoring/test_dashboard_factory.py:700)
- test_main_function (ml/tests/unit/monitoring/test_grafana_client.py:828)

### Similar group

- TestGrafanaClient.test_init_with_api_token (ml/tests/unit/monitoring/test_grafana_client.py:36)
- TestGrafanaClient.test_init_with_basic_auth (ml/tests/unit/monitoring/test_grafana_client.py:47)

### Similar group

- TestGrafanaClient.test_init_invalid_url (ml/tests/unit/monitoring/test_grafana_client.py:62)
- TestGrafanaClient.test_init_no_auth (ml/tests/unit/monitoring/test_grafana_client.py:69)

### Similar group

- TestGrafanaClient.test_make_request_success_200 (ml/tests/unit/monitoring/test_grafana_client.py:84)
- TestGrafanaClient.test_make_request_success_200_no_content (ml/tests/unit/monitoring/test_grafana_client.py:101)
- TestGrafanaClient.test_make_request_success_201 (ml/tests/unit/monitoring/test_grafana_client.py:116)
- TestGrafanaClient.test_make_request_success_201_no_content (ml/tests/unit/monitoring/test_grafana_client.py:132)
- TestGrafanaClient.test_make_request_success_204 (ml/tests/unit/monitoring/test_grafana_client.py:147)
- TestGrafanaClient.test_make_request_not_found_404 (ml/tests/unit/monitoring/test_grafana_client.py:161)

### Similar group

- TestGrafanaClient.test_make_request_error_with_json_response (ml/tests/unit/monitoring/test_grafana_client.py:175)
- TestGrafanaClient.test_make_request_error_without_json_response (ml/tests/unit/monitoring/test_grafana_client.py:193)

### Similar group

- TestGrafanaClient.test_health_check_success (ml/tests/unit/monitoring/test_grafana_client.py:224)
- TestGrafanaClient.test_health_check_failure (ml/tests/unit/monitoring/test_grafana_client.py:237)
- TestGrafanaClient.test_create_folder_success (ml/tests/unit/monitoring/test_grafana_client.py:576)
- TestGrafanaClient.test_get_folder_success (ml/tests/unit/monitoring/test_grafana_client.py:603)
- TestGrafanaClient.test_get_datasources_success (ml/tests/unit/monitoring/test_grafana_client.py:641)
- TestGrafanaClient.test_test_datasource_success (ml/tests/unit/monitoring/test_grafana_client.py:670)

### Similar group

- TestGrafanaClient.test_get_server_info_success (ml/tests/unit/monitoring/test_grafana_client.py:249)
- TestGrafanaClient.test_get_server_info_failure (ml/tests/unit/monitoring/test_grafana_client.py:262)
- TestGrafanaClient.test_get_server_time_success (ml/tests/unit/monitoring/test_grafana_client.py:276)
- TestGrafanaClient.test_get_server_time_failure (ml/tests/unit/monitoring/test_grafana_client.py:295)
- TestGrafanaClient.test_get_dashboard_success (ml/tests/unit/monitoring/test_grafana_client.py:365)
- TestGrafanaClient.test_get_folders_success (ml/tests/unit/monitoring/test_grafana_client.py:535)

### Similar group

- TestGrafanaClient.test_search_dashboards_success (ml/tests/unit/monitoring/test_grafana_client.py:307)
- TestGrafanaClient.test_search_dashboards_empty_params (ml/tests/unit/monitoring/test_grafana_client.py:328)
- TestGrafanaClient.test_search_dashboards_failure (ml/tests/unit/monitoring/test_grafana_client.py:341)
- TestGrafanaClient.test_search_dashboards_non_list_response (ml/tests/unit/monitoring/test_grafana_client.py:353)
- TestGrafanaClient.test_create_dashboard_success (ml/tests/unit/monitoring/test_grafana_client.py:406)
- TestGrafanaClient.test_update_dashboard_success (ml/tests/unit/monitoring/test_grafana_client.py:437)
- TestGrafanaClient.test_delete_dashboard_success (ml/tests/unit/monitoring/test_grafana_client.py:467)
- TestGrafanaClient.test_import_dashboard_success (ml/tests/unit/monitoring/test_grafana_client.py:504)

### Similar group

- TestGrafanaClient.test_get_dashboard_not_found (ml/tests/unit/monitoring/test_grafana_client.py:382)
- TestGrafanaClient.test_get_dashboard_other_error (ml/tests/unit/monitoring/test_grafana_client.py:394)
- TestGrafanaClient.test_create_dashboard_failure (ml/tests/unit/monitoring/test_grafana_client.py:425)
- TestGrafanaClient.test_delete_dashboard_not_found (ml/tests/unit/monitoring/test_grafana_client.py:480)
- TestGrafanaClient.test_delete_dashboard_other_error (ml/tests/unit/monitoring/test_grafana_client.py:492)
- TestGrafanaClient.test_get_folder_not_found (ml/tests/unit/monitoring/test_grafana_client.py:617)

### Similar group

- TestGrafanaClient.test_update_dashboard_failure (ml/tests/unit/monitoring/test_grafana_client.py:455)
- TestGrafanaClient.test_import_dashboard_failure (ml/tests/unit/monitoring/test_grafana_client.py:523)
- TestGrafanaClient.test_create_folder_failure (ml/tests/unit/monitoring/test_grafana_client.py:591)
- TestGrafanaClient.test_get_datasources_failure (ml/tests/unit/monitoring/test_grafana_client.py:658)
- TestGrafanaClient.test_test_datasource_failure (ml/tests/unit/monitoring/test_grafana_client.py:684)

### Similar group

- TestGrafanaClient.test_get_folders_failure (ml/tests/unit/monitoring/test_grafana_client.py:552)
- TestGrafanaClient.test_get_folders_non_list_response (ml/tests/unit/monitoring/test_grafana_client.py:564)
- TestGrafanaClient.test_get_folder_other_error (ml/tests/unit/monitoring/test_grafana_client.py:629)
- TestGrafanaClient.test_get_annotations_failure (ml/tests/unit/monitoring/test_grafana_client.py:748)

### Similar group

- TestGrafanaClient.test_get_annotations_success (ml/tests/unit/monitoring/test_grafana_client.py:696)
- TestGrafanaClient.test_get_annotations_minimal_params (ml/tests/unit/monitoring/test_grafana_client.py:731)

### Similar group

- TestGrafanaAPIError.test_init_basic (ml/tests/unit/monitoring/test_grafana_client.py:784)
- TestGrafanaAPIError.test_init_with_status_code (ml/tests/unit/monitoring/test_grafana_client.py:794)

### Similar group

- TestMLPipelineRunner.test_initialize_feature_engineer_success (ml/tests/unit/scripts/test_run_ml_pipeline.py:172)
- TestMLPipelineRunner.test_initialize_feature_engineer_import_error (ml/tests/unit/scripts/test_run_ml_pipeline.py:186)

### Similar group

- TestConfigurationLoading.test_load_config_yaml_file (ml/tests/unit/scripts/test_run_ml_pipeline.py:329)
- TestConfigurationLoading.test_load_config_json_file (ml/tests/unit/scripts/test_run_ml_pipeline.py:356)
- TestConfigurationLoading.test_load_config_file_not_found (ml/tests/unit/scripts/test_run_ml_pipeline.py:375)
- TestConfigurationLoading.test_load_config_yaml_import_error (ml/tests/unit/scripts/test_run_ml_pipeline.py:395)

### Similar group

- TestLoggingSetup.test_setup_logging_default_level (ml/tests/unit/scripts/test_run_ml_pipeline.py:440)
- TestLoggingSetup.test_setup_logging_verbose_level (ml/tests/unit/scripts/test_run_ml_pipeline.py:451)

### Similar group

- TestValidationFunctions.test_validate_backfill_dates_valid_range (ml/tests/unit/scripts/test_run_ml_pipeline.py:481)
- TestValidationFunctions.test_validate_backfill_dates_missing_start (ml/tests/unit/scripts/test_run_ml_pipeline.py:490)
- TestValidationFunctions.test_validate_backfill_dates_missing_end (ml/tests/unit/scripts/test_run_ml_pipeline.py:498)
- TestValidationFunctions.test_validate_backfill_dates_invalid_format (ml/tests/unit/scripts/test_run_ml_pipeline.py:506)
- TestValidationFunctions.test_validate_backfill_dates_start_after_end (ml/tests/unit/scripts/test_run_ml_pipeline.py:514)

### Similar group

- TestPipelineModeExecution.test_execute_pipeline_mode_backfill (ml/tests/unit/scripts/test_run_ml_pipeline.py:528)
- TestPipelineModeExecution.test_execute_pipeline_mode_daily (ml/tests/unit/scripts/test_run_ml_pipeline.py:541)
- TestPipelineModeExecution.test_execute_pipeline_mode_realtime (ml/tests/unit/scripts/test_run_ml_pipeline.py:551)
- TestPipelineModeExecution.test_execute_pipeline_mode_keyboard_interrupt (ml/tests/unit/scripts/test_run_ml_pipeline.py:561)
- TestPipelineModeExecution.test_execute_pipeline_mode_exception (ml/tests/unit/scripts/test_run_ml_pipeline.py:572)

### Similar group

- TestCalendarTransform.test_calendar_transform_cyclic_encoding (ml/tests/unit/features/test_known_future_transforms.py:28)
- TestCalendarTransform.test_calendar_transform_fourier_encoding (ml/tests/unit/features/test_known_future_transforms.py:52)
- TestCalendarTransform.test_calendar_transform_onehot_encoding (ml/tests/unit/features/test_known_future_transforms.py:69)

### Similar group

- TestEventScheduleTransform.test_event_schedule_default_params (ml/tests/unit/features/test_known_future_transforms.py:142)
- TestEventScheduleTransform.test_event_schedule_custom_events (ml/tests/unit/features/test_known_future_transforms.py:166)
- TestEventScheduleTransform.test_event_schedule_requires_l1_only (ml/tests/unit/features/test_known_future_transforms.py:188)

### Similar group

- TestMacroIndicatorsTransform.test_macro_indicators_default_params (ml/tests/unit/features/test_known_future_transforms.py:233)
- TestMacroIndicatorsTransform.test_macro_indicators_custom_indicators (ml/tests/unit/features/test_known_future_transforms.py:257)
- TestMacroIndicatorsTransform.test_macro_indicators_requires_l1_only (ml/tests/unit/features/test_known_future_transforms.py:280)

### Similar group

- TestPipelineIntegration.test_pipeline_with_known_future_transforms (ml/tests/unit/features/test_known_future_transforms.py:326)
- TestPipelineIntegration.test_pipeline_signature_with_known_future (ml/tests/unit/features/test_known_future_transforms.py:348)

### Similar group

- TestStaticCovariatesTransform.test_static_covariates_default_params (ml/tests/unit/features/test_known_future_transforms.py:395)
- TestStaticCovariatesTransform.test_static_covariates_custom_params (ml/tests/unit/features/test_known_future_transforms.py:416)
- TestStaticCovariatesTransform.test_static_covariates_requires_l1_only (ml/tests/unit/features/test_known_future_transforms.py:430)

### Similar group

- TestFeatureParityValidator.test_validator_initialization_default (ml/tests/unit/features/test_feature_validation.py:81)
- TestFeatureParityValidator.test_validator_initialization_custom (ml/tests/unit/features/test_feature_validation.py:91)

### Similar group

- TestFeatureParityValidator.test_validate_parity_success (ml/tests/unit/features/test_feature_validation.py:103)
- TestFeatureParityValidator.test_validate_parity_detailed_report (ml/tests/unit/features/test_feature_validation.py:137)
- TestFeatureParityValidator.test_validate_parity_no_detailed_report (ml/tests/unit/features/test_feature_validation.py:171)
- TestFeatureParityValidator.test_validate_parity_full_range (ml/tests/unit/features/test_feature_validation.py:188)
- TestFeatureParityValidator.test_validate_parity_invalid_range (ml/tests/unit/features/test_feature_validation.py:204)
- TestFeatureParityValidator.test_validate_parity_edge_case_small_data (ml/tests/unit/features/test_feature_validation.py:219)
- TestFeatureParityValidator.test_validate_performance (ml/tests/unit/features/test_feature_validation.py:234)

### Similar group

- TestFeatureParityValidator.test_generate_test_data (ml/tests/unit/features/test_feature_validation.py:287)
- TestFeatureParityValidator.test_generate_test_data_reproducible (ml/tests/unit/features/test_feature_validation.py:315)

### Similar group

- TestFeatureParityValidator.test_create_bar_from_row (ml/tests/unit/features/test_feature_validation.py:341)
- TestFeatureParityValidator.test_create_bar_from_row_missing_columns (ml/tests/unit/features/test_feature_validation.py:372)

### Similar group

- TestValidateFeatureParityFunction.test_validate_feature_parity_function (ml/tests/unit/features/test_feature_validation.py:441)
- TestValidateFeatureParityFunction.test_validate_feature_parity_function_custom_params (ml/tests/unit/features/test_feature_validation.py:460)

### Similar group

- TestUnifiedFeatureCalculation.test_unified_method_batch_mode (ml/tests/unit/features/test_feature_validation.py:488)
- TestUnifiedFeatureCalculation.test_unified_method_online_mode (ml/tests/unit/features/test_feature_validation.py:525)
- TestUnifiedFeatureCalculation.test_unified_method_invalid_mode (ml/tests/unit/features/test_feature_validation.py:573)
- TestUnifiedFeatureCalculation.test_unified_method_online_without_manager (ml/tests/unit/features/test_feature_validation.py:586)

### Similar group

- TestL2MicrostructureFeatures.test_spread_features_basic (ml/tests/unit/features/test_microstructure.py:32)
- TestL2MicrostructureFeatures.test_imbalance_features_basic (ml/tests/unit/features/test_microstructure.py:65)
- TestL2MicrostructureFeatures.test_depth_features_basic (ml/tests/unit/features/test_microstructure.py:87)
- TestL2MicrostructureFeatures.test_shape_features_basic (ml/tests/unit/features/test_microstructure.py:108)
- TestL2MicrostructureFeatures.test_compute_all_features_polars (ml/tests/unit/features/test_microstructure.py:128)
- TestL2MicrostructureFeatures.test_spread_features_property_positive (ml/tests/unit/features/test_microstructure.py:167)

### Similar group

- TestL3TradeFlowFeatures.test_trade_imbalance_basic (ml/tests/unit/features/test_microstructure.py:227)
- TestL3TradeFlowFeatures.test_price_impact_basic (ml/tests/unit/features/test_microstructure.py:284)

### Similar group

- TestL3TradeFlowFeatures.test_vwap_features_basic (ml/tests/unit/features/test_microstructure.py:248)
- TestL3TradeFlowFeatures.test_intensity_features_basic (ml/tests/unit/features/test_microstructure.py:266)
- TestL3TradeFlowFeatures.test_compute_all_features_polars (ml/tests/unit/features/test_microstructure.py:300)

### Similar group

- TestL3TradeFlowFeatures.test_imbalance_property_range (ml/tests/unit/features/test_microstructure.py:331)
- TestL3TradeFlowFeatures.test_vwap_property_in_price_range (ml/tests/unit/features/test_microstructure.py:360)

### Similar group

- TestFeatureEngineerProperties.test_feature_count_consistency (ml/tests/unit/features/test_feature_engineering_hypothesis.py:44)
- TestFeatureEngineerProperties.test_feature_scaling_invariance (ml/tests/unit/features/test_feature_engineering_hypothesis.py:244)
- TestFeatureEngineerProperties.test_feature_determinism (ml/tests/unit/features/test_feature_engineering_hypothesis.py:369)

### Similar group

- TestFeatureEngineerProperties.test_rsi_bounds_property (ml/tests/unit/features/test_feature_engineering_hypothesis.py:123)
- TestFeatureEngineerProperties.test_feature_buffer_reuse_property (ml/tests/unit/features/test_feature_engineering_hypothesis.py:315)

### Similar group

- TestSafeDivide.test_safe_divide_zero_denominator (ml/tests/unit/features/test_feature_engineering.py:49)
- TestSafeDivide.test_safe_divide_none_denominator (ml/tests/unit/features/test_feature_engineering.py:57)

### Similar group

- TestFeatureConfig.test_default_config_creation (ml/tests/unit/features/test_feature_engineering.py:70)
- TestFeatureConfig.test_custom_config_creation (ml/tests/unit/features/test_feature_engineering.py:81)

### Similar group

- TestFeatureConfig.test_config_validation_ema_periods (ml/tests/unit/features/test_feature_engineering.py:104)
- TestFeatureConfig.test_config_validation_rsi_period (ml/tests/unit/features/test_feature_engineering.py:114)
- TestFeatureConfig.test_config_validation_bb_period (ml/tests/unit/features/test_feature_engineering.py:124)
- TestFeatureConfig.test_config_validation_bb_std (ml/tests/unit/features/test_feature_engineering.py:134)
- TestFeatureConfig.test_config_validation_atr_period (ml/tests/unit/features/test_feature_engineering.py:144)
- TestFeatureConfig.test_config_validation_ema_fast (ml/tests/unit/features/test_feature_engineering.py:154)
- TestFeatureConfig.test_config_validation_ema_slow (ml/tests/unit/features/test_feature_engineering.py:165)
- TestFeatureConfig.test_config_validation_macd_signal (ml/tests/unit/features/test_feature_engineering.py:176)

### Similar group

- TestFeatureConfig.test_get_feature_names (ml/tests/unit/features/test_feature_engineering.py:186)
- TestFeatureEngineer.test_get_feature_names (ml/tests/unit/features/test_feature_engineering.py:678)

### Similar group

- TestIndicatorManager.test_get_values (ml/tests/unit/features/test_feature_engineering.py:393)
- TestIndicatorManager.test_reset (ml/tests/unit/features/test_feature_engineering.py:423)

### Similar group

- TestFeatureEngineer.test_feature_engineer_initialization (ml/tests/unit/features/test_feature_engineering.py:491)
- TestFeatureEngineer.test_feature_engineer_with_custom_config (ml/tests/unit/features/test_feature_engineering.py:505)

### Similar group

- TestFeatureEngineer.test_calculate_features_batch_basic (ml/tests/unit/features/test_feature_engineering.py:518)
- TestFeatureEngineer.test_calculate_features_batch_with_scaling (ml/tests/unit/features/test_feature_engineering.py:540)
- TestFeatureEngineer.test_calculate_features_batch_missing_columns (ml/tests/unit/features/test_feature_engineering.py:576)
- TestFeatureEngineer.test_calculate_features_online (ml/tests/unit/features/test_feature_engineering.py:598)
- TestFeatureEngineer.test_calculate_features_online_with_scaler (ml/tests/unit/features/test_feature_engineering.py:625)

### Similar group

- TestFeatureEngineer.test_return_features_calculation (ml/tests/unit/features/test_feature_engineering.py:705)
- TestFeatureEngineer.test_momentum_features_calculation (ml/tests/unit/features/test_feature_engineering.py:749)
- TestFeatureEngineer.test_volatility_features_calculation (ml/tests/unit/features/test_feature_engineering.py:787)
- TestFeatureEngineer.test_rsi_features_calculation (ml/tests/unit/features/test_feature_engineering.py:823)

### Similar group

- TestFeatureEngineer.test_edge_case_empty_dataframe (ml/tests/unit/features/test_feature_engineering.py:938)
- TestFeatureEngineer.test_edge_case_single_row_dataframe (ml/tests/unit/features/test_feature_engineering.py:950)

### Similar group

- test_feature_names_parity_default (ml/tests/unit/features/test_feature_schema_parity.py:25)
- test_feature_names_parity_with_trade_flow (ml/tests/unit/features/test_feature_schema_parity.py:39)

### Similar group

- test_model_manifest_feature_dtype_mismatch_raises (ml/tests/unit/actors/test_signal_actor_model_manifest_parity.py:32)
- test_model_manifest_feature_dtype_match_passes (ml/tests/unit/actors/test_signal_actor_model_manifest_parity.py:64)

### Similar group

- TestMLSignalActorProperties.test_warmup_property (ml/tests/unit/actors/test_signal_actor_hypothesis.py:53)
- TestMLSignalActorProperties.test_signal_threshold_property (ml/tests/unit/actors/test_signal_actor_hypothesis.py:131)
- TestMLSignalActorProperties.test_latency_monitoring_property (ml/tests/unit/actors/test_signal_actor_hypothesis.py:232)

### Similar group

- TestMLSignalActor.test_actor_initialization (ml/tests/unit/actors/test_signal_actor.py:241)
- TestMLSignalActor.test_ensemble_weights_initialization (ml/tests/unit/actors/test_signal_actor.py:1020)

### Similar group

- TestMLSignalActor.test_threshold_signal_generation (ml/tests/unit/actors/test_signal_actor.py:260)
- TestMLSignalActor.test_extremes_signal_generation (ml/tests/unit/actors/test_signal_actor.py:316)
- TestMLSignalActor.test_momentum_signal_generation (ml/tests/unit/actors/test_signal_actor.py:365)
- TestMLSignalActor.test_ensemble_signal_generation (ml/tests/unit/actors/test_signal_actor.py:397)
- TestMLSignalActor.test_adaptive_signal_generation (ml/tests/unit/actors/test_signal_actor.py:433)

### Similar group

- TestMLSignalActor.test_market_regime_detection (ml/tests/unit/actors/test_signal_actor.py:494)
- TestMLSignalActor.test_market_regime_detection_scenarios (ml/tests/unit/actors/test_signal_actor.py:929)

### Similar group

- TestMLSignalActor.test_feature_computation_performance (ml/tests/unit/actors/test_signal_actor.py:529)
- TestMLSignalActor.test_slow_feature_computation_warning (ml/tests/unit/actors/test_signal_actor.py:1266)

### Similar group

- TestMLSignalActor.test_get_signal_statistics (ml/tests/unit/actors/test_signal_actor.py:681)
- TestMLSignalActor.test_reset_signal_state (ml/tests/unit/actors/test_signal_actor.py:706)

### Similar group

- TestMLSignalActor.test_onnx_model_prediction (ml/tests/unit/actors/test_signal_actor.py:762)
- TestMLSignalActor.test_sklearn_proba_model_prediction (ml/tests/unit/actors/test_signal_actor.py:786)
- TestMLSignalActor.test_sklearn_basic_model_prediction (ml/tests/unit/actors/test_signal_actor.py:809)

### Similar group

- TestMLSignalActor.test_compute_features_without_indicator_manager (ml/tests/unit/actors/test_signal_actor.py:1071)
- TestMLSignalActor.test_compute_features_before_indicators_ready (ml/tests/unit/actors/test_signal_actor.py:1086)

### Similar group

- TestMLSignalActor.test_indicator_state_backup_without_manager (ml/tests/unit/actors/test_signal_actor.py:1287)
- TestMLSignalActor.test_indicator_state_restore_without_backup (ml/tests/unit/actors/test_signal_actor.py:1312)

### Similar group

- TestMLSignalActor.test_momentum_signal_insufficient_history (ml/tests/unit/actors/test_signal_actor.py:1368)
- TestMLSignalActor.test_extremes_signal_insufficient_history (ml/tests/unit/actors/test_signal_actor.py:1392)

### Similar group

- TestMLSignalActor.test_generate_prediction_protected_success (ml/tests/unit/actors/test_signal_actor.py:1573)
- TestMLSignalActor.test_generate_prediction_protected_failure (ml/tests/unit/actors/test_signal_actor.py:1597)

### Similar group

- TestAsofJoin.test_asof_join_basic_polars (ml/tests/unit/preprocessing/test_joins.py:43)
- TestAsofJoin.test_asof_join_basic_pandas (ml/tests/unit/preprocessing/test_joins.py:69)
- TestAsofJoin.test_asof_join_with_tolerance_polars (ml/tests/unit/preprocessing/test_joins.py:96)

### Similar group

- TestValidateNoLookahead.test_validate_no_lookahead_pass (ml/tests/unit/preprocessing/test_joins.py:267)
- TestValidateNoLookahead.test_validate_no_lookahead_fail (ml/tests/unit/preprocessing/test_joins.py:282)
- TestValidateNoLookahead.test_validate_no_lookahead_property (ml/tests/unit/preprocessing/test_joins.py:302)

### Similar group

- TestCreateLagFeatures.test_create_lag_features_basic (ml/tests/unit/preprocessing/test_joins.py:334)
- TestCreateLagFeatures.test_create_lag_features_grouped (ml/tests/unit/preprocessing/test_joins.py:359)
- TestCreateLagFeatures.test_create_lag_features_property_consistency (ml/tests/unit/preprocessing/test_joins.py:395)

### Similar group

- TestPurgedCrossValidator.test_init_valid_params (ml/tests/unit/preprocessing/test_purged_cv.py:22)
- TestPurgedCrossValidator.test_init_invalid_n_splits (ml/tests/unit/preprocessing/test_purged_cv.py:29)
- TestPurgedCrossValidator.test_init_invalid_purge_gap (ml/tests/unit/preprocessing/test_purged_cv.py:34)
- TestPurgedCrossValidator.test_init_invalid_embargo_pct (ml/tests/unit/preprocessing/test_purged_cv.py:39)

### Similar group

- TestPurgedCrossValidator.test_basic_split_no_purge_no_embargo (ml/tests/unit/preprocessing/test_purged_cv.py:44)
- TestPurgedCrossValidator.test_split_with_embargo (ml/tests/unit/preprocessing/test_purged_cv.py:87)

### Similar group

- TestPurgedCrossValidator.test_split_with_purge_gap (ml/tests/unit/preprocessing/test_purged_cv.py:65)
- TestPurgedCrossValidator.test_split_property_purge_gap_respected (ml/tests/unit/preprocessing/test_purged_cv.py:146)
- TestPurgedCrossValidator.test_split_coverage (ml/tests/unit/preprocessing/test_purged_cv.py:171)

### Similar group

- TestLockFreeRingBuffer.test_init_with_valid_size (ml/tests/unit/core/test_cache.py:22)
- TestLockFreeRingBuffer.test_init_with_invalid_size_raises (ml/tests/unit/core/test_cache.py:31)
- TestLockFreeRingBuffer.test_get_last_with_various_sizes (ml/tests/unit/core/test_cache.py:99)

### Similar group

- TestLockFreeRingBuffer.test_append_single_value (ml/tests/unit/core/test_cache.py:41)
- TestLockFreeRingBuffer.test_append_array (ml/tests/unit/core/test_cache.py:86)

### Similar group

- TestLockFreeRingBuffer.test_get_window (ml/tests/unit/core/test_cache.py:133)
- TestLockFreeRingBuffer.test_get_window_edge_cases (ml/tests/unit/core/test_cache.py:155)
- TestLockFreeRingBuffer.test_get_all_and_reset (ml/tests/unit/core/test_cache.py:181)
- TestLockFreeRingBuffer.test_mean_and_std (ml/tests/unit/core/test_cache.py:223)

### Similar group

- TestReservoirSampler.test_init_with_valid_size (ml/tests/unit/core/test_cache.py:247)
- TestReservoirSampler.test_init_with_invalid_size_raises (ml/tests/unit/core/test_cache.py:255)

### Similar group

- TestReservoirSampler.test_add_multiple_samples (ml/tests/unit/core/test_cache.py:296)
- TestReservoirSampler.test_get_multiple_percentiles (ml/tests/unit/core/test_cache.py:327)

### Similar group

- TestReservoirSampler.test_get_percentile (ml/tests/unit/core/test_cache.py:309)
- TestReservoirSampler.test_reset_sampler (ml/tests/unit/core/test_cache.py:344)

### Similar group

- TestPreAllocatedFeatureCache.test_init_cache (ml/tests/unit/core/test_cache.py:381)
- TestPreAllocatedFeatureCache.test_cache_reset (ml/tests/unit/core/test_cache.py:504)

### Similar group

- TestPreAllocatedFeatureCache.test_prepare_onnx_input (ml/tests/unit/core/test_cache.py:472)
- TestPreAllocatedFeatureCache.test_get_onnx_input_buffer (ml/tests/unit/core/test_cache.py:491)

### Similar group

- TestEngineManager.test_dispose_engine (ml/tests/unit/core/test_db_engine.py:114)
- TestEngineManager.test_dispose_all (ml/tests/unit/core/test_db_engine.py:128)

### Similar group

- TestEngineManager.test_empty_connection_string_raises_error (ml/tests/unit/core/test_db_engine.py:158)
- TestEngineManager.test_none_connection_string_raises_error (ml/tests/unit/core/test_db_engine.py:164)

### Similar group

- TestEngineManager.test_pool_status_for_existing_engine (ml/tests/unit/core/test_db_engine.py:170)
- TestEngineManager.test_pool_status_for_nonexistent_engine (ml/tests/unit/core/test_db_engine.py:183)

### Similar group

- TestBaseMLStrategy.test_on_data_processes_ml_signal (ml/tests/unit/strategies/test_base_strategy.py:219)
- TestBaseMLStrategy.test_on_data_ignores_non_ml_signals (ml/tests/unit/strategies/test_base_strategy.py:239)

### Similar group

- TestBaseMLStrategy.test_handle_ml_signal_filters_wrong_instrument (ml/tests/unit/strategies/test_base_strategy.py:266)
- TestBaseMLStrategy.test_handle_ml_signal_filters_low_confidence (ml/tests/unit/strategies/test_base_strategy.py:291)

### Similar group

- TestBaseMLStrategy.test_calculate_position_size_with_account (ml/tests/unit/strategies/test_base_strategy.py:340)
- TestBaseMLStrategy.test_calculate_position_size_no_account (ml/tests/unit/strategies/test_base_strategy.py:379)
- TestBaseMLStrategy.test_calculate_position_size_no_instrument (ml/tests/unit/strategies/test_base_strategy.py:398)
- TestBaseMLStrategy.test_calculate_position_size_with_quote_tick_fallback (ml/tests/unit/strategies/test_base_strategy.py:419)
- TestBaseMLStrategy.test_calculate_position_size_no_price_data (ml/tests/unit/strategies/test_base_strategy.py:457)

### Similar group

- TestBaseMLStrategy.test_place_market_order (ml/tests/unit/strategies/test_base_strategy.py:484)
- TestBaseMLStrategy.test_place_market_order_reduce_only (ml/tests/unit/strategies/test_base_strategy.py:512)
- TestBaseMLStrategy.test_place_stop_loss (ml/tests/unit/strategies/test_base_strategy.py:532)

### Similar group

- TestBaseMLStrategy.test_get_current_position_returns_first_open (ml/tests/unit/strategies/test_base_strategy.py:560)
- TestBaseMLStrategy.test_get_current_position_returns_none_when_no_positions (ml/tests/unit/strategies/test_base_strategy.py:585)

### Similar group

- TestSimpleMLStrategy.test_process_ml_signal_opens_long_position (ml/tests/unit/strategies/test_base_strategy.py:715)
- TestSimpleMLStrategy.test_process_ml_signal_opens_short_position (ml/tests/unit/strategies/test_base_strategy.py:744)
- TestSimpleMLStrategy.test_process_ml_signal_reverses_position (ml/tests/unit/strategies/test_base_strategy.py:773)
- TestSimpleMLStrategy.test_process_ml_signal_keeps_aligned_position (ml/tests/unit/strategies/test_base_strategy.py:816)
- TestSimpleMLStrategy.test_process_ml_signal_no_entry_when_position_sizing_fails (ml/tests/unit/strategies/test_base_strategy.py:849)

### Similar group

- TestStrategyStoreIntegration.test_strategy_store_initialization_with_config (ml/tests/unit/strategies/test_strategy_store_integration.py:59)
- TestStrategyStoreIntegration.test_strategy_store_disabled_by_config (ml/tests/unit/strategies/test_strategy_store_integration.py:95)

### Similar group

- TestStrategyStoreIntegration.test_persist_buy_decision (ml/tests/unit/strategies/test_strategy_store_integration.py:114)
- TestStrategyStoreIntegration.test_persist_sell_decision (ml/tests/unit/strategies/test_strategy_store_integration.py:165)
- TestStrategyStoreIntegration.test_persist_hold_decision_with_config (ml/tests/unit/strategies/test_strategy_store_integration.py:215)

### Similar group

- TestSignalStrategies.test_threshold_strategy (ml/tests/unit/strategies/test_signal_strategies.py:113)
- TestSignalStrategies.test_extremes_strategy (ml/tests/unit/strategies/test_signal_strategies.py:135)
- TestSignalStrategies.test_momentum_strategy (ml/tests/unit/strategies/test_signal_strategies.py:161)
- TestSignalStrategies.test_ensemble_strategy (ml/tests/unit/strategies/test_signal_strategies.py:189)
- TestSignalStrategies.test_adaptive_strategy (ml/tests/unit/strategies/test_signal_strategies.py:215)
- TestSignalStrategies.test_custom_strategy_plugin (ml/tests/unit/strategies/test_signal_strategies.py:249)

### Similar group

- TestOptimizationLevels.test_standard_optimization_level (ml/tests/unit/strategies/test_signal_strategies.py:309)
- TestOptimizationLevels.test_actor_with_standard_optimization (ml/tests/unit/strategies/test_signal_strategies.py:341)
- TestOptimizationLevels.test_actor_with_optimized_level (ml/tests/unit/strategies/test_signal_strategies.py:361)

### Similar group

- TestConfigurationSystem.test_strategy_config_defaults (ml/tests/unit/strategies/test_signal_strategies.py:390)
- TestConfigurationSystem.test_strategy_config_custom_values (ml/tests/unit/strategies/test_signal_strategies.py:404)

### Similar group

- TestConfigurationSystem.test_threshold_strategy_enum (ml/tests/unit/strategies/test_signal_strategies.py:427)
- TestConfigurationSystem.test_signal_strategy_enum (ml/tests/unit/strategies/test_signal_strategies.py:435)

### Similar group

- TestPerformanceMonitoring.test_performance_monitor_initialization (ml/tests/unit/strategies/test_signal_strategies.py:458)
- TestPerformanceMonitoring.test_performance_monitor_timing_recording (ml/tests/unit/strategies/test_signal_strategies.py:470)
- TestPerformanceMonitoring.test_performance_monitor_latency_percentiles (ml/tests/unit/strategies/test_signal_strategies.py:490)

### Similar group

- TestPandasCalendarSource.test_init_with_pandas_market_calendars_available (ml/tests/unit/data/sources/test_calendar_pandas.py:31)
- TestPandasCalendarSource.test_init_without_pandas_market_calendars (ml/tests/unit/data/sources/test_calendar_pandas.py:41)

### Similar group

- TestPandasCalendarSource.test_init_with_custom_fallback (ml/tests/unit/data/sources/test_calendar_pandas.py:50)
- TestPandasCalendarSource.test_init_with_custom_cache_ttl (ml/tests/unit/data/sources/test_calendar_pandas.py:60)
- TestPandasCalendarSource.test_get_holidays_with_fallback (ml/tests/unit/data/sources/test_calendar_pandas.py:227)

### Similar group

- TestPandasCalendarSource.test_cache_functionality (ml/tests/unit/data/sources/test_calendar_pandas.py:152)
- TestPandasCalendarSource.test_cache_expiration (ml/tests/unit/data/sources/test_calendar_pandas.py:315)

### Similar group

- TestPandasCalendarSource.test_build_schedule_pre_market (ml/tests/unit/data/sources/test_calendar_pandas.py:263)
- TestPandasCalendarSource.test_build_schedule_after_hours (ml/tests/unit/data/sources/test_calendar_pandas.py:280)

### Similar group

- TestCyclicEncode.test_cyclic_encode_basic (ml/tests/unit/data/providers/test_utils.py:29)
- TestCyclicEncode.test_cyclic_encode_range_property (ml/tests/unit/data/providers/test_utils.py:64)
- TestCyclicEncode.test_cyclic_encode_periodicity (ml/tests/unit/data/providers/test_utils.py:76)

### Similar group

- TestTimeToEvent.test_time_to_event_hours (ml/tests/unit/data/providers/test_utils.py:112)
- TestTimeToEvent.test_time_to_event_days (ml/tests/unit/data/providers/test_utils.py:120)
- TestTimeToEvent.test_time_to_event_minutes (ml/tests/unit/data/providers/test_utils.py:128)
- TestTimeToEvent.test_time_to_event_negative (ml/tests/unit/data/providers/test_utils.py:136)
- TestTimeToEvent.test_time_to_event_invalid_unit (ml/tests/unit/data/providers/test_utils.py:144)
- TestTimeToEvent.test_time_to_event_consistency (ml/tests/unit/data/providers/test_utils.py:156)

### Similar group

- TestValidateTimestamps.test_validate_timestamps_valid (ml/tests/unit/data/providers/test_utils.py:174)
- TestValidateTimestamps.test_validate_timestamps_with_nulls (ml/tests/unit/data/providers/test_utils.py:183)
- TestValidateTimestamps.test_validate_timestamps_unsorted (ml/tests/unit/data/providers/test_utils.py:191)
- TestValidateTimestamps.test_validate_timestamps_negative (ml/tests/unit/data/providers/test_utils.py:199)
- TestValidateTimestamps.test_validate_timestamps_future (ml/tests/unit/data/providers/test_utils.py:207)
- TestValidateTimestamps.test_validate_timestamps_property (ml/tests/unit/data/providers/test_utils.py:225)

### Similar group

- TestAlignTimeseries.test_align_timeseries_inner (ml/tests/unit/data/providers/test_utils.py:238)
- TestAlignTimeseries.test_align_timeseries_left (ml/tests/unit/data/providers/test_utils.py:261)
- TestAlignTimeseries.test_align_timeseries_outer (ml/tests/unit/data/providers/test_utils.py:286)
- TestAlignTimeseries.test_align_timeseries_property (ml/tests/unit/data/providers/test_utils.py:318)

### Similar group

- TestMockCalendarSource.test_mock_source_generates_schedule (ml/tests/unit/data/providers/test_calendar.py:30)
- TestMockEventSource.test_mock_source_generates_events (ml/tests/unit/data/providers/test_events.py:139)
- TestMockMetadataSource.test_mock_source_generates_metadata (ml/tests/unit/data/providers/test_metadata.py:32)

### Similar group

- TestMockCalendarSource.test_mock_source_weekend_detection (ml/tests/unit/data/providers/test_calendar.py:44)
- TestMockCalendarSource.test_mock_source_handles_any_date (ml/tests/unit/data/providers/test_calendar.py:111)
- TestMockMetadataSource.test_mock_source_etf_detection (ml/tests/unit/data/providers/test_metadata.py:76)

### Similar group

- TestSimpleCalendarSource.test_simple_source_basic_schedule (ml/tests/unit/data/providers/test_calendar.py:130)
- TestSimpleCalendarSource.test_simple_source_minutes_to_close (ml/tests/unit/data/providers/test_calendar.py:146)
- TestSimpleCalendarSource.test_simple_source_different_exchanges (ml/tests/unit/data/providers/test_calendar.py:167)

### Similar group

- TestMarketCalendarProvider.test_provider_computes_features (ml/tests/unit/data/providers/test_calendar.py:194)
- TestEventScheduleProvider.test_provider_computes_features (ml/tests/unit/data/providers/test_events.py:256)

### Similar group

- TestMarketCalendarProvider.test_provider_cyclic_encoding (ml/tests/unit/data/providers/test_calendar.py:245)
- TestMarketCalendarProvider.test_provider_days_to_month_end (ml/tests/unit/data/providers/test_calendar.py:308)

### Similar group

- TestMarketCalendarProvider.test_provider_month_boundaries (ml/tests/unit/data/providers/test_calendar.py:274)
- TestMarketCalendarProvider.test_provider_handles_source_errors (ml/tests/unit/data/providers/test_calendar.py:361)

### Similar group

- TestMarketCalendarProvider.test_provider_handles_any_timestamps (ml/tests/unit/data/providers/test_calendar.py:342)
- TestEventScheduleProvider.test_provider_handles_any_timestamps (ml/tests/unit/data/providers/test_events.py:388)

### Similar group

- TestProviderProtocols.test_data_provider_protocol_enforcement (ml/tests/unit/data/providers/test_base.py:28)
- TestProviderProtocols.test_static_provider_protocol (ml/tests/unit/data/providers/test_base.py:59)
- TestProviderProtocols.test_cacheable_provider_protocol (ml/tests/unit/data/providers/test_base.py:71)

### Similar group

- TestBaseDataProvider.test_base_provider_initialization (ml/tests/unit/data/providers/test_base.py:92)
- TestBaseDataProvider.test_base_provider_validation (ml/tests/unit/data/providers/test_base.py:100)
- TestCachedDataProvider.test_cached_provider_initialization (ml/tests/unit/data/providers/test_base.py:145)

### Similar group

- TestCachedDataProvider.test_cache_key_generation (ml/tests/unit/data/providers/test_base.py:162)
- TestCachedDataProvider.test_cache_ttl_property (ml/tests/unit/data/providers/test_base.py:201)

### Similar group

- TestProviderFactory.test_factory_creates_default_providers (ml/tests/unit/data/providers/test_factory.py:38)
- TestProviderFactory.test_factory_register_custom_provider (ml/tests/unit/data/providers/test_factory.py:113)
- TestProviderFactory.test_factory_handles_multiple_custom_providers (ml/tests/unit/data/providers/test_factory.py:133)

### Similar group

- TestTransformProviderAdapter.test_adapter_maps_transform_to_provider (ml/tests/unit/data/providers/test_factory.py:156)
- TestTransformProviderAdapter.test_adapter_loads_transform_data (ml/tests/unit/data/providers/test_factory.py:178)
- TestTransformProviderAdapter.test_adapter_caches_providers (ml/tests/unit/data/providers/test_factory.py:226)
- TestTransformProviderAdapter.test_adapter_with_custom_provider (ml/tests/unit/data/providers/test_factory.py:324)

### Similar group

- TestTransformProviderAdapter.test_adapter_handles_static_data (ml/tests/unit/data/providers/test_factory.py:206)
- TestTransformProviderAdapter.test_adapter_merges_multi_instrument_data (ml/tests/unit/data/providers/test_factory.py:261)
- TestTransformProviderAdapter.test_adapter_handles_arbitrary_data_sizes (ml/tests/unit/data/providers/test_factory.py:292)

### Similar group

- TestEconomicEvent.test_economic_event_creation (ml/tests/unit/data/providers/test_events.py:33)
- TestEconomicEvent.test_economic_event_with_actual (ml/tests/unit/data/providers/test_events.py:54)

### Similar group

- TestMockEventSource.test_mock_source_earnings_events (ml/tests/unit/data/providers/test_events.py:159)
- TestMockEventSource.test_mock_source_deterministic (ml/tests/unit/data/providers/test_events.py:182)

### Similar group

- TestSimpleEventSource.test_simple_source_fed_meetings (ml/tests/unit/data/providers/test_events.py:207)
- TestSimpleEventSource.test_simple_source_earnings_calendar (ml/tests/unit/data/providers/test_events.py:226)

### Similar group

- TestEventScheduleProvider.test_provider_handles_multiple_instruments (ml/tests/unit/data/providers/test_events.py:299)
- TestEventScheduleProvider.test_provider_caches_events (ml/tests/unit/data/providers/test_events.py:409)
- TestEventScheduleProvider.test_provider_handles_source_errors (ml/tests/unit/data/providers/test_events.py:451)

### Similar group

- TestCSVMetadataSource.test_csv_source_loads_from_file (ml/tests/unit/data/providers/test_metadata.py:125)
- TestCSVMetadataSource.test_csv_source_handles_missing_file (ml/tests/unit/data/providers/test_metadata.py:156)

### Similar group

- TestDatabentoMetadataSource.test_databento_source_without_key (ml/tests/unit/data/providers/test_metadata.py:204)
- TestDatabentoMetadataSource.test_databento_source_with_api_key (ml/tests/unit/data/providers/test_metadata.py:216)

### Similar group

- TestInstrumentMetadataProvider.test_provider_loads_and_caches (ml/tests/unit/data/providers/test_metadata.py:272)
- TestInstrumentMetadataProvider.test_provider_validates_data (ml/tests/unit/data/providers/test_metadata.py:291)
- TestInstrumentMetadataProvider.test_provider_handles_empty_data (ml/tests/unit/data/providers/test_metadata.py:314)
- TestInstrumentMetadataProvider.test_provider_schema (ml/tests/unit/data/providers/test_metadata.py:342)

### Similar group

- TestInstrumentMetadataProvider.test_provider_handles_source_errors (ml/tests/unit/data/providers/test_metadata.py:328)
- TestInstrumentMetadataProvider.test_provider_handles_arbitrary_instruments (ml/tests/unit/data/providers/test_metadata.py:392)

### Similar group

- TestResourceUtilizationCollector.test_initialization_with_disabled_config (ml/tests/unit/monitoring/collectors/test_resource_collector.py:35)
- TestResourceUtilizationCollector.test_initialization_with_enabled_config (ml/tests/unit/monitoring/collectors/test_resource_collector.py:48)
- TestBaseMetricsCollector.test_initialization_with_disabled_config (ml/tests/unit/monitoring/collectors/test_base.py:26)

### Similar group

- TestResourceUtilizationCollector.test_record_model_memory_usage (ml/tests/unit/monitoring/collectors/test_resource_collector.py:60)
- TestResourceUtilizationCollector.test_record_cpu_usage (ml/tests/unit/monitoring/collectors/test_resource_collector.py:77)
- TestResourceUtilizationCollector.test_record_gpu_metrics (ml/tests/unit/monitoring/collectors/test_resource_collector.py:94)
- TestResourceUtilizationCollector.test_record_feature_store_size (ml/tests/unit/monitoring/collectors/test_resource_collector.py:124)
- TestResourceUtilizationCollector.test_record_disk_usage (ml/tests/unit/monitoring/collectors/test_resource_collector.py:159)
- TestResourceUtilizationCollector.test_reset_metrics (ml/tests/unit/monitoring/collectors/test_resource_collector.py:280)

### Similar group

- TestResourceUtilizationCollector.test_record_data_io (ml/tests/unit/monitoring/collectors/test_resource_collector.py:176)
- TestResourceUtilizationCollector.test_record_inference_batch_size (ml/tests/unit/monitoring/collectors/test_resource_collector.py:193)
- TestResourceUtilizationCollector.test_record_training_data_processed (ml/tests/unit/monitoring/collectors/test_resource_collector.py:210)

### Similar group

- TestBaseMetricsCollector.test_health_check_basic_functionality (ml/tests/unit/monitoring/collectors/test_base.py:70)
- TestBaseMetricsCollector.test_reset_metrics_functionality (ml/tests/unit/monitoring/collectors/test_base.py:171)

## Tests with Overlapping Coverage

### Target: np
  Tested by 342 tests:

- test_can_compute_basic_features
- TestActorContracts.test_actor_publishes_ml_signal_on_bar
- TestActorContracts.test_actor_includes_model_id_in_signal
- TestActorContracts.test_actor_handles_multiple_instruments
- TestActorContracts.test_actor_gracefully_handles_inference_failure
  ... and 337 more

### Target: st
  Tested by 254 tests:

- TestFeatureStoreInvariants.test_timestamp_monotonicity_invariant
- TestFeatureStoreInvariants.test_timestamp_monotonicity_invariant
- TestFeatureStoreInvariants.test_feature_immutability_invariant
- TestFeatureStoreInvariants.test_partition_consistency_invariant
- TestFeatureStoreInvariants.test_partition_consistency_invariant
  ... and 249 more

### Target: registry
  Tested by 179 tests:

- TestRegistryBehaviors.test_thread_safety_concurrent_operations
- TestRegistryBehaviors.test_thread_safety_concurrent_operations
- TestRegistryBehaviors.test_rollback_restores_previous_model
- TestRegistryBehaviors.test_rollback_restores_previous_model
- TestRegistryBehaviors.test_rollback_restores_previous_model
  ... and 174 more

### Target: pytest
  Tested by 178 tests:

- test_assert_features_compatible_names_and_types
- test_assert_features_compatible_names_and_types
- TestStoreSchemaContracts.test_feature_input_schema_validation
- TestStoreSchemaContracts.test_feature_input_schema_validation
- TestStoreSchemaContracts.test_prediction_schema_validation
  ... and 173 more

### Target: time
  Tested by 174 tests:

- TestRegistryBehaviors.test_rollback_restores_previous_model
- TestRegistryBehaviors.test_rollback_restores_previous_model
- TestRegistryBehaviors.test_rollback_restores_previous_model
- TestRegistryBehaviors.test_rollback_restores_previous_model
- TestRegistryBehaviors.test_rollback_restores_previous_model
  ... and 169 more

### Target: self
  Tested by 151 tests:

- TestStrategyContracts.test_strategy_receives_ml_signals
- TestStrategyContracts.test_strategy_filters_by_model_id
- TestStrategyContracts.test_strategy_handles_multiple_model_signals
- TestStrategyContracts.test_strategy_respects_signal_confidence_threshold
- TestStrategyContracts.test_strategy_handles_conflicting_signals
  ... and 146 more

### Target: pl
  Tested by 139 tests:

- TestNautilusDataPipeline.test_batch_vs_online_feature_parity
- TestNautilusDataPipeline.test_feature_scaling_consistency
- TestTransformProviderIntegration.test_calendar_transform_integration
- TestTransformProviderIntegration.test_event_transform_integration
- TestTransformProviderIntegration.test_pipeline_with_providers
  ... and 134 more

### Target: patch
  Tested by 131 tests:

- TestMLPipelineIntegration.test_partial_system_failure_resilience
- TestMLPipelineIntegration.test_partial_system_failure_resilience
- TestSchedulerFeatureStoreIntegration.test_feature_computation_with_catalog_data
- TestSchedulerFeatureStoreIntegration.test_feature_store_connection_from_env
- TestSchedulerFeatureStoreIntegration.test_metrics_tracking
  ... and 126 more

### Target: strategy
  Tested by 106 tests:

- TestStrategyContracts.test_strategy_receives_ml_signals
- TestStrategyContracts.test_strategy_filters_by_model_id
- TestStrategyContracts.test_strategy_handles_multiple_model_signals
- TestStrategyContracts.test_strategy_respects_signal_confidence_threshold
- TestStrategyContracts.test_strategy_handles_conflicting_signals
  ... and 101 more

### Target: datetime
  Tested by 83 tests:

- TestMLPipelineIntegration.test_multi_provider_failover
- TestMLPipelineIntegration.test_multi_provider_failover
- TestMLPipelineIntegration.test_multi_provider_failover
- TestMLPipelineIntegration.test_multi_provider_failover
- TestMLPipelineIntegration.test_multi_provider_failover
  ... and 78 more

### Target: pd
  Tested by 72 tests:

- TestStoreSchemaContracts.test_feature_input_schema_validation
- TestStoreSchemaContracts.test_feature_input_schema_validation
- TestStoreSchemaContracts.test_feature_input_schema_validation
- TestStoreSchemaContracts.test_prediction_schema_validation
- TestStoreSchemaContracts.test_prediction_schema_validation
  ... and 67 more

### Target: collector
  Tested by 70 tests:

- TestMLPipelineIntegration.test_multi_provider_failover
- TestMLPipelineIntegration.test_multi_provider_failover
- TestMLPipelineIntegration.test_multi_provider_failover
- TestDataCollectorErrorRecovery.test_collector_initialization_without_api_key
- TestDataCollectorErrorRecovery.test_storage_calculation_with_inaccessible_files
  ... and 65 more

### Target: monkeypatch
  Tested by 64 tests:

- test_feature_store_realtime_event_and_jsonb
- test_feature_store_realtime_event_and_jsonb
- test_feature_store_realtime_event_and_jsonb
- test_model_store_events_and_jsonb_and_reads
- test_registry_registers_teacher_and_student_and_loads_onnx
  ... and 59 more

### Target: actor
  Tested by 63 tests:

- TestActorContracts.test_actor_publishes_ml_signal_on_bar
- TestActorContracts.test_actor_includes_model_id_in_signal
- TestActorContracts.test_actor_handles_multiple_instruments
- TestActorContracts.test_actor_handles_multiple_instruments
- TestActorContracts.test_actor_handles_multiple_instruments
  ... and 58 more

### Target: Price
  Tested by 61 tests:

- TestMLPipelineIntegration.test_pipeline_scalability
- TestMLPipelineIntegration.test_pipeline_scalability
- TestMLPipelineIntegration.test_pipeline_scalability
- TestMLPipelineIntegration.test_pipeline_scalability
- TestDataEventTracing.test_full_pipeline_data_flow
  ... and 56 more

### Target: scheduler
  Tested by 60 tests:

- TestSchedulerMetrics.test_scheduler_initialization_with_metrics
- TestSchedulerMetrics.test_pipeline_metrics_recording
- TestSchedulerMetrics.test_collection_error_metrics
- TestSchedulerMetrics.test_feature_computation_metrics
- TestSchedulerMetrics.test_cleanup_metrics
  ... and 55 more

### Target: engineer
  Tested by 55 tests:

- test_can_compute_basic_features
- TestFeatureTransformMetamorphic.test_price_scaling_invariance
- TestFeatureTransformMetamorphic.test_price_scaling_invariance
- TestFeatureTransformMetamorphic.test_time_reversal_relationships
- TestFeatureTransformMetamorphic.test_time_reversal_relationships
  ... and 50 more

### Target: tempfile
  Tested by 54 tests:

- test_registry_can_initialize
- TestRegistryBehaviors.test_thread_safety_concurrent_operations
- TestRegistryBehaviors.test_rollback_restores_previous_model
- TestRegistryBehaviors.test_ab_test_splits_traffic
- TestRegistryBehaviors.test_hot_reload_updates_without_downtime
  ... and 49 more

### Target: InstrumentId
  Tested by 52 tests:

- TestStrategyContracts.test_strategy_receives_ml_signals
- TestStrategyContracts.test_strategy_filters_by_model_id
- TestStrategyContracts.test_strategy_filters_by_model_id
- TestStrategyContracts.test_strategy_filters_by_model_id
- TestStrategyContracts.test_strategy_handles_multiple_model_signals
  ... and 47 more

### Target: runner
  Tested by 52 tests:

- TestL2L3RegistryStoreIntegration.test_pipeline_integration_with_l2_transforms
- TestL2L3RegistryStoreIntegration.test_pipeline_integration_with_l2_transforms
- TestPipelineRunner.test_signal_handler
- TestPipelineRunner.test_signal_handler_no_scheduler
- TestPipelineRunner.test_create_config
  ... and 47 more

### Target: engine
  Tested by 51 tests:

- test_postgres_connection
- test_postgres_connection
- TestMLPipelineIntegration.test_docker_compose_stack_integration
- TestMLPipelineIntegration.test_system_recovery_from_database_failure
- TestMLPipelineIntegration.test_system_recovery_from_database_failure
  ... and 46 more

### Target: tracer
  Tested by 50 tests:

- TestDataEventTracing.test_full_pipeline_data_flow
- TestDataEventTracing.test_full_pipeline_data_flow
- TestDataEventTracing.test_full_pipeline_data_flow
- TestDataEventTracing.test_full_pipeline_data_flow
- TestDataEventTracing.test_full_pipeline_data_flow
  ... and 45 more

### Target: client
  Tested by 50 tests:

- TestHealthEndpoint.test_health_check_healthy
- TestHealthEndpoint.test_health_check_unhealthy
- TestDeploymentIntegration.test_health_check_endpoint_availability
- TestGrafanaClient.test_make_request_success_200
- TestGrafanaClient.test_make_request_success_200_no_content
  ... and 45 more

### Target: provider
  Tested by 49 tests:

- TestCalendarProviderIntegration.test_provider_with_real_pandas_source
- TestCalendarProviderIntegration.test_provider_with_mock_source
- TestCalendarProviderIntegration.test_provider_handles_multiple_exchanges
- TestCalendarProviderIntegration.test_provider_handles_multiple_exchanges
- TestCalendarProviderIntegration.test_provider_cyclic_encodings
  ... and 44 more

### Target: EngineManager
  Tested by 47 tests:

- TestEngineManagerIntegration.test_stores_share_engine_instance
- TestEngineManagerIntegration.test_stores_share_engine_instance
- TestEngineManagerIntegration.test_connection_pool_limits_enforced
- TestEngineManagerIntegration.test_connection_pool_limits_enforced
- TestEngineManagerIntegration.test_dispose_cleans_up_properly
  ... and 42 more

### Target: buffer
  Tested by 45 tests:

- TestZeroAllocationHotPath.test_ring_buffer_get_last_returns_view
- TestZeroAllocationHotPath.test_ring_buffer_get_last_returns_view
- TestZeroAllocationHotPath.test_ring_buffer_get_window_returns_view
- TestZeroAllocationHotPath.test_ring_buffer_get_window_returns_view
- TestZeroAllocationHotPath.test_ring_buffer_wraparound_requires_allocation
  ... and 40 more

### Target: source
  Tested by 43 tests:

- TestPandasCalendarSource.test_get_schedule_uses_fallback_when_disabled
- TestPandasCalendarSource.test_get_24_7_schedule
- TestPandasCalendarSource.test_get_schedule_with_real_calendar
- TestPandasCalendarSource.test_cache_functionality
- TestPandasCalendarSource.test_cache_functionality
  ... and 38 more

### Target: store
  Tested by 40 tests:

- test_can_initialize_feature_store
- test_feature_store_realtime_event_and_jsonb
- test_model_store_events_and_jsonb_and_reads
- test_model_store_events_and_jsonb_and_reads
- test_model_store_events_and_jsonb_and_reads
  ... and 35 more

### Target: factory
  Tested by 38 tests:

- TestCalendarProviderIntegration.test_factory_get_calendar_provider
- TestCalendarProviderIntegration.test_factory_get_calendar_provider
- TestEndToEndPipeline.test_provider_integration
- TestEndToEndPipeline.test_provider_integration
- TestGrafanaDashboardFactory.test_create_base_dashboard_basic
  ... and 33 more

### Target: result
  Tested by 36 tests:

- test_clean_db_fixture
- test_database_fixture_works
- test_database_cleanup
- test_database_cleanup
- test_database_cleanup
  ... and 31 more

### Target: json
  Tested by 36 tests:

- TestMLPipelineIntegration.test_message_queue_failure_handling
- TestFeatureStore.test_read_range
- TestModelStore.test_read_latest_predictions
- TestStrategyStore.test_read_active_signals
- TestStrategyStore.test_read_active_signals
  ... and 31 more

### Target: Quantity
  Tested by 33 tests:

- TestMLPipelineIntegration.test_pipeline_scalability
- TestDataEventTracing.test_full_pipeline_data_flow
- TestDataEventTracing.test_full_pipeline_data_flow
- TestDataEventTracing.test_pipeline_with_deliberate_failures
- TestFeatureParity.test_parity_at_different_sequence_lengths
  ... and 28 more

### Target: reg
  Tested by 32 tests:

- test_register_non_serveable_teacher_artifact
- test_register_non_serveable_teacher_artifact
- test_register_non_serveable_teacher_artifact
- test_register_student_requires_onnx
- test_validate_and_promote
  ... and 27 more

### Target: node
  Tested by 27 tests:

- TestMLStrategyNode.test_setup_with_dry_run_mode
- TestMLStrategyNode.test_setup_with_dry_run_mode
- TestMLStrategyNode.test_setup_with_live_mode
- TestMLStrategyNode.test_setup_with_live_mode
- TestMLStrategyNode.test_setup_risk_parameters
  ... and 22 more

### Target: conn
  Tested by 26 tests:

- test_database_fixture_works
- test_database_cleanup
- test_database_cleanup
- test_database_cleanup
- test_database_cleanup
  ... and 21 more

### Target: threading
  Tested by 26 tests:

- TestRegistryBehaviors.test_thread_safety_concurrent_operations
- TestRegistryBehaviors.test_thread_safety_concurrent_operations
- TestDataRegistryE2E.test_concurrent_access_json
- TestConcurrentWrites.test_feature_store_concurrent_writes
- TestReadWriteConflicts.test_feature_store_read_during_write
  ... and 21 more

### Target: model_path
  Tested by 26 tests:

- TestRegistryBehaviors.test_deployment_validates_constraints
- TestModelRegistryProperties.test_registry_version_ordering
- TestModelRegistryProperties.test_registry_isolation
- TestEndToEndProperties.test_model_selection_consistency
- TestEnhancedModelRegistry.test_full_deployment_pipeline
  ... and 21 more

### Target: data_store
  Tested by 26 tests:

- TestMLPipelineIntegration.test_data_consistency_across_pipeline_stages
- TestMLPipelineIntegration.test_data_consistency_across_pipeline_stages
- TestPreflightCheck.test_preflight_check_valid_data
- TestPreflightCheck.test_preflight_check_missing_required_columns
- TestPreflightCheck.test_preflight_check_type_mismatch
  ... and 21 more

### Target: validator
  Tested by 25 tests:

- TestModelContracts.test_valid_teacher_contract
- TestModelContracts.test_valid_student_contract
- TestModelContracts.test_valid_student_contract
- TestModelContracts.test_invalid_student_with_l2_data
- TestModelContracts.test_invalid_student_high_latency
  ... and 20 more

### Target: rng
  Tested by 24 tests:

- TestInfrastructure.test_feature_parity_validation
- TestInfrastructure.test_feature_parity_validation
- test_tft_teacher_fit_predict_smoke
- test_tft_teacher_fit_predict_smoke
- test_tft_teacher_fit_predict_smoke
  ... and 19 more

### Target: adapter
  Tested by 24 tests:

- TestTransformProviderIntegration.test_calendar_transform_integration
- TestTransformProviderIntegration.test_metadata_transform_integration
- TestTransformProviderIntegration.test_event_transform_integration
- TestTransformProviderIntegration.test_pipeline_with_providers
- TestTransformProviderIntegration.test_transform_feature_consistency
  ... and 19 more

### Target: builder
  Tested by 24 tests:

- TestEndToEndPipeline.test_tft_dataset_integration
- TestTFTDatasetBuilderErrors.test_builder_without_feature_store
- TestTFTDatasetBuilderErrors.test_empty_features_from_store
- TestTFTDatasetBuilderErrors.test_mismatched_feature_dimensions
- TestTFTDatasetBuilderErrors.test_nan_and_inf_values_in_features
  ... and 19 more

### Target: feature_store
  Tested by 23 tests:

- TestMLPipelineIntegration.test_e2e_ml_pipeline_with_real_data
- TestMLPipelineIntegration.test_e2e_ml_pipeline_with_real_data
- TestMLPipelineIntegration.test_system_recovery_from_database_failure
- TestMLPipelineIntegration.test_system_recovery_from_database_failure
- TestMLPipelineIntegration.test_system_recovery_from_database_failure
  ... and 18 more

### Target: cache
  Tested by 23 tests:

- TestZeroAllocationHotPath.test_feature_cache_returns_views
- TestZeroAllocationHotPath.test_feature_cache_returns_views
- TestZeroAllocationHotPath.test_feature_cache_returns_views
- TestZeroAllocationHotPath.test_feature_cache_history_returns_view_when_contiguous
- TestZeroAllocationHotPath.test_feature_cache_history_returns_view_when_contiguous
  ... and 18 more

### Target: deployment
  Tested by 22 tests:

- TestCanaryDeployment.test_record_metric_success
- TestCanaryDeployment.test_record_metric_error
- TestCanaryDeployment.test_should_rollback_high_error_rate
- TestCanaryDeployment.test_should_rollback_high_error_rate
- TestCanaryDeployment.test_should_rollback_high_error_rate
  ... and 17 more

### Target: executor
  Tested by 20 tests:

- TestConcurrentWrites.test_feature_store_concurrent_writes
- TestConcurrentWrites.test_model_store_concurrent_predictions
- TestConcurrentWrites.test_strategy_store_concurrent_signals
- TestReadWriteConflicts.test_model_store_hot_swap
- TestTransactionIntegrity.test_atomic_batch_writes
  ... and 15 more

### Target: features
  Tested by 19 tests:

- TestFeatureCombinations.test_feature_compatibility_pairwise
- TestMLPipelineIntegration.test_data_consistency_across_pipeline_stages
- TestMLPipelineIntegration.test_data_consistency_across_pipeline_stages
- TestMLPipelineIntegration.test_data_consistency_across_pipeline_stages
- TestSignalPredictionMetamorphic.test_time_shift_invariance
  ... and 14 more

### Target: sampler
  Tested by 19 tests:

- TestZeroAllocationHotPath.test_reservoir_sampler_get_sample_returns_view
- TestZeroAllocationHotPath.test_reservoir_sampler_get_sample_returns_view
- TestReservoirSampler.test_add_samples_up_to_size
- TestReservoirSampler.test_add_samples_up_to_size
- TestReservoirSampler.test_reservoir_sampling_maintains_size
  ... and 14 more

### Target: thread
  Tested by 18 tests:

- TestRegistryBehaviors.test_thread_safety_concurrent_operations
- TestRegistryBehaviors.test_thread_safety_concurrent_operations
- TestDataRegistryE2E.test_concurrent_access_json
- TestDataRegistryE2E.test_concurrent_access_json
- TestReadWriteConflicts.test_feature_store_read_during_write
  ... and 13 more

### Target: calculator
  Tested by 18 tests:

- TestL2L3RegistryStoreIntegration.test_l2_feature_computation_with_real_data
- TestL2L3RegistryStoreIntegration.test_l2_feature_computation_with_real_data
- TestL2L3RegistryStoreIntegration.test_l2_feature_computation_with_real_data
- TestL2MicrostructureFeatures.test_spread_features_basic
- TestL2MicrostructureFeatures.test_imbalance_features_basic
  ... and 13 more

### Target: optimizer
  Tested by 18 tests:

- TestXGBoostOptunaOptimizer.test_ensure_optuna
- TestXGBoostOptunaOptimizer.test_ensure_optuna_not_available
- TestXGBoostOptunaOptimizer.test_create_study
- TestXGBoostOptunaOptimizer.test_create_study_with_storage
- TestXGBoostOptunaOptimizer.test_create_sampler
  ... and 13 more

### Target: transform
  Tested by 18 tests:

- TestCalendarTransform.test_calendar_transform_cyclic_encoding
- TestCalendarTransform.test_calendar_transform_fourier_encoding
- TestCalendarTransform.test_calendar_transform_onehot_encoding
- TestCalendarTransform.test_calendar_transform_minute_granularity
- TestCalendarTransform.test_calendar_transform_requires_l1_only
  ... and 13 more

### Target: os
  Tested by 17 tests:

- test_postgres_connection
- test_postgres_connection
- test_postgres_connection
- test_postgres_connection
- test_postgres_connection
  ... and 12 more

### Target: future
  Tested by 17 tests:

- TestConcurrentWrites.test_feature_store_concurrent_writes
- TestConcurrentWrites.test_model_store_concurrent_predictions
- TestConcurrentWrites.test_strategy_store_concurrent_signals
- TestReadWriteConflicts.test_model_store_hot_swap
- TestTransactionIntegrity.test_atomic_batch_writes
  ... and 12 more

### Target: GrafanaPanelFactory
  Tested by 17 tests:

- TestGrafanaPanelFactory.test_create_stat_panel_basic
- TestGrafanaPanelFactory.test_create_stat_panel_with_custom_unit
- TestGrafanaPanelFactory.test_create_stat_panel_with_custom_thresholds
- TestGrafanaPanelFactory.test_create_stat_panel_with_alert_config
- TestGrafanaPanelFactory.test_create_stat_panel_datasource_config
  ... and 12 more

### Target: mock_make_request
  Tested by 17 tests:

- TestGrafanaClient.test_health_check_success
- TestGrafanaClient.test_get_server_info_success
- TestGrafanaClient.test_get_server_time_success
- TestGrafanaClient.test_search_dashboards_success
- TestGrafanaClient.test_search_dashboards_empty_params
  ... and 12 more

### Target: df
  Tested by 16 tests:

- TestTransformProviderIntegration.test_pipeline_with_providers
- TestTransformProviderIntegration.test_pipeline_with_providers
- TestTransformProviderIntegration.test_provider_scalability
- TestEndToEndPipeline.test_pipeline_with_mock_data
- TestEndToEndPipeline.test_pipeline_error_recovery
  ... and 11 more

### Target: model_store
  Tested by 15 tests:

- TestMLPipelineIntegration.test_e2e_ml_pipeline_with_real_data
- TestMLPipelineIntegration.test_e2e_ml_pipeline_with_real_data
- TestMLPipelineIntegration.test_data_consistency_across_pipeline_stages
- TestMLPipelineIntegration.test_data_consistency_across_pipeline_stages
- test_isolated_component_failures
  ... and 10 more

### Target: strategy_store
  Tested by 15 tests:

- TestMLPipelineIntegration.test_e2e_ml_pipeline_with_real_data
- TestMLPipelineIntegration.test_e2e_ml_pipeline_with_real_data
- TestMLPipelineIntegration.test_partial_system_failure_resilience
- TestMLPipelineIntegration.test_partial_system_failure_resilience
- TestMLPipelineIntegration.test_data_consistency_across_pipeline_stages
  ... and 10 more

### Target: clock
  Tested by 15 tests:

- test_strategy_store_integration_demo
- test_strategy_store_integration_demo
- test_strategy_store_integration_demo
- test_strategy_store_integration_demo
- test_strategy_store_integration_demo
  ... and 10 more

### Target: data_processor
  Tested by 15 tests:

- TestDataProcessor.test_process_market_data
- TestDataProcessor.test_process_market_data_with_crossed_market
- TestDataProcessor.test_process_features_with_nan
- TestDataProcessor.test_process_prediction
- TestDataProcessor.test_process_signal_with_risk_limits
  ... and 10 more

### Target: indicator_mgr
  Tested by 15 tests:

- TestEndToEndPipeline.test_online_feature_parity
- TestEndToEndPipeline.test_online_feature_parity
- TestZeroAllocationHotPath.test_feature_engineer_returns_buffer_view
- TestZeroAllocationHotPath.test_hot_path_memory_stability
- TestZeroAllocationHotPath.test_feature_parity_with_views
  ... and 10 more

### Target: config
  Tested by 15 tests:

- TestL2L3RegistryStoreIntegration.test_feature_config_includes_microstructure
- TestL2L3RegistryStoreIntegration.test_feature_registry_manifest_with_l2_features
- TestL2L3RegistryStoreIntegration.test_end_to_end_l2_feature_persistence
- TestFeatureParityValidator.test_validate_parity_detailed_report
- TestFeatureParityValidator.test_validate_performance
  ... and 10 more

### Target: loader
  Tested by 15 tests:

- test_onnx_runtime_session_load_and_run
- TestFREDDataLoaderErrors.test_api_connection_failures
- TestFREDDataLoaderErrors.test_api_rate_limiting
- TestFREDDataLoaderErrors.test_invalid_series_id
- TestFREDDataLoaderErrors.test_cache_corruption_recovery
  ... and 10 more

### Target: fe
  Tested by 15 tests:

- TestFeatureEngineer.test_calculate_features_batch_basic
- TestFeatureEngineer.test_calculate_features_batch_with_scaling
- TestFeatureEngineer.test_calculate_features_batch_missing_columns
- TestFeatureEngineer.test_calculate_features_online
- TestFeatureEngineer.test_calculate_features_online_with_scaler
  ... and 10 more

### Target: BarType
  Tested by 14 tests:

- TestMLPipelineIntegration.test_pipeline_scalability
- test_pipeline_smoke_test
- TestStrategyProperties.test_strategy_warmup
- TestMLActorConfig.test_model_id_required
- TestConfigIntegration.test_multi_model_with_actor_configs
  ... and 9 more

### Target: results
  Tested by 14 tests:

- TestConcurrentWrites.test_feature_store_concurrent_writes
- TestConcurrentWrites.test_feature_store_concurrent_writes
- TestConcurrentWrites.test_feature_store_concurrent_writes
- TestConcurrentWrites.test_model_store_concurrent_predictions
- TestConcurrentWrites.test_model_store_concurrent_predictions
  ... and 9 more

### Target: fred_loader
  Tested by 14 tests:

- TestFREDDataLoader.test_fetch_indicator
- TestFREDDataLoader.test_fetch_indicator_with_dates
- TestFREDDataLoader.test_cache_functionality
- TestFREDDataLoader.test_cache_functionality
- TestFREDDataLoader.test_cache_expiry
  ... and 9 more

### Target: session
  Tested by 13 tests:

- test_clean_db_fixture
- TestTestConfiguration.test_test_database_initialization
- TestTestConfiguration.test_database_seed_data
- TestTestConfiguration.test_database_rollback
- TestTestConfiguration.test_database_rollback
  ... and 8 more

### Target: TestDataStubs
  Tested by 13 tests:

- TestActorContracts.test_actor_publishes_ml_signal_on_bar
- TestActorContracts.test_actor_includes_model_id_in_signal
- TestActorContracts.test_actor_gracefully_handles_inference_failure
- TestActorContracts.test_actor_gracefully_handles_inference_failure
- TestActorContracts.test_actor_gracefully_handles_inference_failure
  ... and 8 more

### Target: feature_engineer
  Tested by 12 tests:

- TestMLPipelineIntegration.test_e2e_ml_pipeline_with_real_data
- TestMLPipelineIntegration.test_system_recovery_from_database_failure
- TestMLPipelineIntegration.test_system_recovery_from_database_failure
- TestMLPipelineIntegration.test_system_recovery_from_database_failure
- TestMLPipelineIntegration.test_partial_system_failure_resilience
  ... and 7 more

### Target: helper
  Tested by 12 tests:

- test_model_registry_register_load_deploy_lineage
- test_model_registry_register_load_deploy_lineage
- test_model_registry_register_load_deploy_lineage
- test_model_registry_register_load_deploy_lineage
- test_model_registry_register_load_deploy_lineage
  ... and 7 more

### Target: mreg
  Tested by 11 tests:

- test_register_serveable_requires_onnx_and_schema
- test_register_serveable_requires_onnx_and_schema
- test_feature_set_id_linkage_and_hash_validation
- test_feature_set_id_linkage_and_hash_validation
- test_feature_set_id_linkage_and_hash_validation
  ... and 6 more

### Target: futures
  Tested by 11 tests:

- TestRaceConditions.test_feature_computation_race
- TestRaceConditions.test_event_ordering_race
- TestRaceConditions.test_watermark_update_race
- TestPerformanceUnderLoad.test_stress_test_all_stores
- TestPerformanceUnderLoad.test_stress_test_all_stores
  ... and 6 more

### Target: strategy_path
  Tested by 11 tests:

- TestStrategyRegistryBackends.test_json_backend_register_and_retrieve
- TestStrategyRegistryBackends.test_postgres_backend_with_compatibility
- TestStrategyRegistry.test_register_strategy
- TestStrategyRegistry.test_get_strategy
- TestStrategyRegistry.test_filter_by_regime
  ... and 6 more

### Target: simulator
  Tested by 10 tests:

- TestDataRegistryE2E.test_full_day_pipeline_json
- TestDataRegistryE2E.test_failure_recovery_json
- TestDataRegistryE2E.test_failure_recovery_json
- TestDataRegistryE2E.test_gap_detection_json
- TestDataRegistryE2E.test_gap_detection_json
  ... and 5 more

### Target: json_registry
  Tested by 10 tests:

- TestDataRegistryE2E.test_full_day_pipeline_json
- TestDataRegistryE2E.test_data_contracts_json
- TestDataRegistryE2E.test_data_contracts_json
- TestDataRegistryE2E.test_performance_benchmarks_json
- TestDataRegistryE2E.test_idempotent_writes_json
  ... and 5 more

### Target: capsys
  Tested by 10 tests:

- TestMLStrategyNode.test_dry_run_mode_output
- TestMLStrategyNode.test_dry_run_mode_output
- TestMainFunction.test_main_prints_startup_info
- TestMainFunction.test_main_prints_startup_info
- TestMainFunction.test_main_prints_startup_info
  ... and 5 more

### Target: threads
  Tested by 9 tests:

- TestRegistryBehaviors.test_thread_safety_concurrent_operations
- TestDataRegistryE2E.test_concurrent_access_json
- TestDeadlockPrevention.test_lock_ordering
- TestDeadlockPrevention.test_lock_ordering
- TestRegistryPerformance.test_registry_concurrent_read_performance
  ... and 4 more

### Target: indicator_manager
  Tested by 9 tests:

- TestNautilusDataPipeline.test_parquet_catalog_to_ml_features
- TestNautilusDataPipeline.test_batch_vs_online_feature_parity
- TestNautilusDataPipeline.test_multi_instrument_feature_engineering
- TestNautilusDataPipeline.test_feature_engineering_with_gaps
- TestNautilusDataPipeline.test_feature_scaling_consistency
  ... and 4 more

### Target: features_df
  Tested by 9 tests:

- TestEndToEndPipeline.test_feature_computation_and_storage
- TestEndToEndPipeline.test_signal_generation_from_features
- TestEndToEndPipeline.test_pipeline_error_recovery
- TestFeatureEngineer.test_return_features_calculation
- TestFeatureEngineer.test_return_features_calculation
  ... and 4 more

### Target: mgr
  Tested by 9 tests:

- TestIndicatorManagerProperties.test_indicator_initialization
- TestIndicatorManagerProperties.test_indicator_initialization
- TestEndToEndBenchmarks.test_concurrent_signal_generation
- TestIndicatorManager.test_update_from_bar
- TestIndicatorManager.test_price_history_memory_management
  ... and 4 more

### Target: cv
  Tested by 9 tests:

- TestPurgedCrossValidator.test_basic_split_no_purge_no_embargo
- TestPurgedCrossValidator.test_split_with_purge_gap
- TestPurgedCrossValidator.test_split_with_embargo
- TestPurgedCrossValidator.test_get_n_splits
- TestPurgedCrossValidator.test_get_n_splits
  ... and 4 more

### Target: test_database
  Tested by 8 tests:

- test_clean_db_fixture
- TestTestConfiguration.test_test_database_initialization
- TestTestConfiguration.test_database_seed_data
- TestTestConfiguration.test_database_seed_data
- TestTestConfiguration.test_database_rollback
  ... and 3 more

### Target: catalog
  Tested by 8 tests:

- TestEndToEndPipeline.test_pipeline_with_mock_data
- TestEndToEndPipeline.test_feature_computation_and_storage
- TestEndToEndPipeline.test_signal_generation_from_features
- TestEndToEndPipeline.test_pipeline_scalability
- TestEndToEndPipeline.test_tft_dataset_integration
  ... and 3 more

### Target: tracemalloc
  Tested by 8 tests:

- TestZeroAllocationHotPath.test_hot_path_memory_stability
- TestZeroAllocationHotPath.test_hot_path_memory_stability
- TestZeroAllocationHotPath.test_hot_path_memory_stability
- TestZeroAllocationHotPath.test_hot_path_memory_stability
- TestFeatureComputationBenchmarks.test_feature_memory_allocation
  ... and 3 more

### Target: benchmark
  Tested by 8 tests:

- TestFeatureComputationBenchmarks.test_feature_computation_p99_latency
- TestModelInferenceBenchmarks.test_onnx_inference_p99_latency
- TestModelInferenceBenchmarks.test_model_swap_latency
- TestStoreBenchmarks.test_feature_store_read_latency
- TestStoreBenchmarks.test_store_write_buffering
  ... and 3 more

### Target: model
  Tested by 7 tests:

- TestMLPipelineIntegration.test_e2e_ml_pipeline_with_real_data
- TestMLPipelineIntegration.test_e2e_ml_pipeline_with_real_data
- TestMLPipelineIntegration.test_data_consistency_across_pipeline_stages
- TestMLPipelineIntegration.test_data_consistency_across_pipeline_stages
- TestEndToEndPipeline.test_signal_generation_from_features
  ... and 2 more

### Target: online_features
  Tested by 7 tests:

- TestNautilusDataPipeline.test_parquet_catalog_to_ml_features
- TestNautilusDataPipeline.test_batch_vs_online_feature_parity
- TestNautilusDataPipeline.test_feature_scaling_consistency
- TestEndToEndPipeline.test_online_feature_parity
- TestFeatureParity.test_feature_engineer_batch_vs_online_parity
  ... and 2 more

### Target: df1
  Tested by 7 tests:

- TestTransformProviderIntegration.test_provider_caching_efficiency
- TestFREDDataLoader.test_cache_functionality
- TestFeatureParityValidator.test_generate_test_data_reproducible
- TestFeatureParityValidator.test_generate_test_data_reproducible
- TestEventScheduleProvider.test_provider_caches_events
  ... and 2 more

### Target: gc
  Tested by 7 tests:

- TestZeroAllocationHotPath.test_hot_path_memory_stability
- TestFeatureComputationBenchmarks.test_feature_memory_allocation
- TestPerformanceRegression.test_memory_leak_detection
- TestPerformanceRegression.test_memory_leak_detection
- TestMemoryPressure.test_garbage_collection_triggers
  ... and 2 more

### Target: values
  Tested by 7 tests:

- TestIndicatorManager.test_get_values
- TestIndicatorManager.test_get_values
- TestIndicatorManager.test_get_values
- TestIndicatorManager.test_get_values
- TestIndicatorManager.test_get_values
  ... and 2 more

### Target: monitor
  Tested by 7 tests:

- TestPerformanceMonitoring.test_performance_monitor_timing_recording
- TestPerformanceMonitoring.test_performance_monitor_timing_recording
- TestPerformanceMonitoring.test_performance_monitor_timing_recording
- TestPerformanceMonitoring.test_performance_monitor_timing_recording
- TestPerformanceMonitoring.test_performance_monitor_latency_percentiles
  ... and 2 more

### Target: bars
  Tested by 6 tests:

- TestMLPipelineIntegration.test_pipeline_scalability
- TestFeatureParity.test_parity_at_different_sequence_lengths
- test_bars
- TestCatalogUtils.test_bars_to_dataframe_with_data
- TestCatalogUtils.test_multiple_instruments
  ... and 1 more

### Target: batch_features
  Tested by 6 tests:

- TestInfrastructure.test_feature_parity_validation
- TestNautilusDataPipeline.test_parquet_catalog_to_ml_features
- TestNautilusDataPipeline.test_batch_vs_online_feature_parity
- TestNautilusDataPipeline.test_feature_scaling_consistency
- TestEndToEndPipeline.test_online_feature_parity
  ... and 1 more

### Target: online_engineer
  Tested by 6 tests:

- TestNautilusDataPipeline.test_batch_vs_online_feature_parity
- TestNautilusDataPipeline.test_feature_scaling_consistency
- TestNautilusDataPipeline.test_feature_scaling_consistency
- TestNautilusDataPipeline.test_feature_scaling_consistency
- TestFeatureParity.test_feature_engineer_batch_vs_online_parity
  ... and 1 more

### Target: hashlib
  Tested by 6 tests:

- TestL2L3RegistryStoreIntegration.test_end_to_end_l2_feature_persistence
- TestUnifiedRegistry.test_inference_model_registration
- TestModelContracts.test_teacher_without_l2_features
- TestRegistryPerformance.test_registry_bulk_registration_performance
- TestPropertyBased.test_validation_consistency
  ... and 1 more

### Target: ConfigurationHelper
  Tested by 6 tests:

- TestMLActorConfiguration.test_configuration_helper_get_bar_type
- TestMLActorConfiguration.test_configuration_helper_get_instrument_id
- TestMLActorConfiguration.test_configuration_helper_get_model_path
- TestMLActorConfiguration.test_configuration_helper_missing_attribute_raises
- TestMLActorConfiguration.test_configuration_helper_missing_attribute_raises
  ... and 1 more

### Target: cfg
  Tested by 6 tests:

- test_generate_and_register_manifest
- test_feature_names_parity_default
- test_feature_names_parity_with_microstructure
- test_feature_names_parity_with_trade_flow
- test_manifest_schema_matches_config_names
  ... and 1 more

### Target: signal_stats
  Tested by 6 tests:

- TestMLSignalActor.test_threshold_signal_generation
- TestMLSignalActor.test_ensemble_signal_generation
- TestMLSignalActor.test_adaptive_signal_generation
- TestMLSignalActor.test_adaptive_signal_generation
- TestMLSignalActor.test_feature_computation_performance
  ... and 1 more

### Target: published_signals
  Tested by 5 tests:

- TestActorContracts.test_actor_publishes_ml_signal_on_bar
- TestActorContracts.test_actor_includes_model_id_in_signal
- TestActorContracts.test_actor_handles_multiple_instruments
- TestActorContracts.test_actor_gracefully_handles_inference_failure
- TestActorContracts.test_actor_respects_hot_path_constraints

### Target: requests
  Tested by 5 tests:

- TestMLPipelineIntegration.test_docker_compose_stack_integration
- TestMLPipelineIntegration.test_docker_compose_stack_integration
- TestServiceHealthChecks.test_check_ml_pipeline_connection_error
- TestServiceHealthChecks.test_check_ml_pipeline_timeout
- TestGrafanaClient.test_make_request_connection_error

### Target: response
  Tested by 5 tests:

- TestMLPipelineIntegration.test_docker_compose_stack_integration
- TestMLPipelineIntegration.test_docker_compose_stack_integration
- TestHealthEndpoint.test_health_check_healthy
- TestHealthEndpoint.test_health_check_unhealthy
- TestDeploymentIntegration.test_health_check_endpoint_availability

### Target: dt
  Tested by 5 tests:

- TestTransformProviderIntegration.test_provider_scalability
- TestPandasCalendarSource.test_cache_functionality
- TestPandasCalendarSource.test_cache_expiration
- TestTransformProviderAdapter.test_adapter_handles_arbitrary_data_sizes
- TestEventScheduleProvider.test_provider_handles_any_timestamps

### Target: store2
  Tested by 5 tests:

- TestStorePersistence.test_feature_store_persistence
- TestStorePersistence.test_model_store_persistence
- TestStorePersistence.test_model_store_persistence
- TestStorePersistence.test_strategy_store_persistence
- TestStorePersistence.test_strategy_store_persistence

### Target: data
  Tested by 5 tests:

- test_tft_cli_training_with_registration
- TestPropertyBased.test_validation_consistency
- TestPropertyBased.test_range_validation_fuzzing
- TestIntegration.test_full_validation_pipeline
- TestIntegration.test_validation_performance

### Target: actor_node
  Tested by 5 tests:

- TestDeploymentIntegration.test_actor_to_strategy_communication
- TestDeploymentIntegration.test_environment_variable_propagation
- TestDeploymentIntegration.test_prometheus_metrics_exposure
- TestDeploymentIntegration.test_configuration_validation
- TestDeploymentIntegration.test_configuration_validation

### Target: col
  Tested by 5 tests:

- TestFeatureEngineerProperties.test_rsi_bounds_property
- TestFeatureEngineerProperties.test_moving_average_monotonicity_skip
- TestFeatureEngineerProperties.test_moving_average_monotonicity_skip
- TestFeatureEngineerProperties.test_moving_average_monotonicity_skip
- TestFeatureEngineerProperties.test_feature_scaling_invariance

### Target: mock_redis
  Tested by 4 tests:

- TestMockServices.test_mock_redis
- TestMockServices.test_mock_redis
- TestMockServices.test_mock_redis
- TestMockServices.test_mock_redis

### Target: f_schema
  Tested by 4 tests:

- test_register_serveable_requires_onnx_and_schema
- test_register_serveable_requires_onnx_and_schema
- test_resolve_latest_and_list_compatible
- test_resolve_latest_and_list_compatible

### Target: FeatureInputSchema
  Tested by 4 tests:

- TestStoreSchemaContracts.test_feature_input_schema_validation
- TestStoreSchemaContracts.test_feature_input_schema_validation
- TestStoreSchemaContracts.test_feature_input_schema_validation
- TestStoreSchemaContracts.test_cross_store_consistency

### Target: PredictionSchema
  Tested by 4 tests:

- TestStoreSchemaContracts.test_prediction_schema_validation
- TestStoreSchemaContracts.test_prediction_schema_validation
- TestStoreSchemaContracts.test_prediction_schema_validation
- TestStoreSchemaContracts.test_cross_store_consistency

### Target: all_features
  Tested by 4 tests:

- TestFeatureCombinations.test_feature_compatibility_pairwise
- TestNautilusDataPipeline.test_multi_instrument_feature_engineering
- TestNautilusDataPipeline.test_multi_instrument_feature_engineering
- TestTransformProviderIntegration.test_pipeline_with_providers

### Target: torch
  Tested by 4 tests:

- test_torchscript_export_parity
- test_torchscript_export_parity
- test_torchscript_export_parity
- test_torchscript_export_parity

### Target: date
  Tested by 4 tests:

- TestDataRegistryE2E.test_full_day_pipeline_json
- TestDataRegistryE2E.test_full_day_pipeline_json
- TestDataRegistryE2E.test_gap_detection_json
- TestDataRegistryE2E.test_gap_detection_json

### Target: mock_model
  Tested by 4 tests:

- TestSignalPredictionMetamorphic.test_time_shift_invariance
- TestSignalPredictionMetamorphic.test_time_shift_invariance
- TestSignalPredictionMetamorphic.test_noise_robustness
- TestSignalPredictionMetamorphic.test_noise_robustness

### Target: e
  Tested by 4 tests:

- test_feature_store_realtime_event_and_jsonb
- test_data_store_canonical_ids_for_events
- test_data_store_canonical_ids_for_events
- test_data_store_canonical_ids_for_events

### Target: teacher_path
  Tested by 4 tests:

- test_registry_registers_teacher_and_student_and_loads_onnx
- TestUnifiedRegistry.test_register_student_model_with_lineage
- TestUnifiedRegistry.test_get_models_by_role
- TestUnifiedRegistry.test_auto_deploy_with_validation

### Target: student_path
  Tested by 4 tests:

- test_registry_registers_teacher_and_student_and_loads_onnx
- TestUnifiedRegistry.test_register_student_model_with_lineage
- TestUnifiedRegistry.test_get_models_by_role
- TestUnifiedRegistry.test_auto_deploy_with_validation

### Target: mock_node_class
  Tested by 4 tests:

- TestMainFunction.test_main_successful_run
- TestMainFunction.test_main_successful_run
- TestMainFunction.test_main_successful_run
- TestMLSignalActorNode.test_setup_with_valid_config

### Target: mock_get
  Tested by 4 tests:

- TestServiceHealthChecks.test_check_ml_pipeline_healthy
- TestServiceHealthChecks.test_check_prometheus_healthy
- TestServiceHealthChecks.test_check_grafana_healthy
- TestPandasCalendarSource.test_cache_expiration

### Target: caplog
  Tested by 4 tests:

- TestTFTDatasetBuilderWithFeatureStore.test_logging_feature_source
- TestTFTDatasetBuilderWithFeatureStore.test_logging_feature_source
- TestTFTDatasetBuilderWithFeatureStore.test_logging_fallback
- TestTFTDatasetBuilderWithFeatureStore.test_logging_fallback

### Target: prices
  Tested by 4 tests:

- TestL3TradeFlowFeatures.test_vwap_property_in_price_range
- TestL3TradeFlowFeatures.test_vwap_property_in_price_range
- TestFeatureEngineer.test_volatility_features_calculation
- TestFeatureEngineer.test_rsi_features_calculation

### Target: test_idx
  Tested by 4 tests:

- TestPurgedCrossValidator.test_split_with_purge_gap
- TestPurgedCrossValidator.test_split_with_purge_gap
- TestPurgedCrossValidator.test_split_with_embargo
- TestPurgedCrossValidator.test_split_temporal_order

### Target: mock_databento_client
  Tested by 3 tests:

- TestMockServices.test_mock_databento_client
- TestMockServices.test_mock_databento_client
- TestFixtureIntegration.test_database_with_mocks

### Target: freg
  Tested by 3 tests:

- test_register_serveable_requires_onnx_and_schema
- test_resolve_latest_and_list_compatible
- test_registry_registers_teacher_and_student_and_loads_onnx

### Target: onnx_path
  Tested by 3 tests:

- test_register_serveable_requires_onnx_and_schema
- test_feature_set_id_linkage_and_hash_validation
- test_resolve_latest_and_list_compatible

### Target: p
  Tested by 3 tests:

- test_register_non_serveable_teacher_artifact
- TestMultiprocessing.test_multiprocess_writes
- TestMultiprocessing.test_multiprocess_writes

### Target: model_v1_path
  Tested by 3 tests:

- TestRegistryBehaviors.test_rollback_restores_previous_model
- TestRegistryBehaviors.test_hot_reload_updates_without_downtime
- TestRegistryDeployment.test_registry_hot_reload

### Target: model_v2_path
  Tested by 3 tests:

- TestRegistryBehaviors.test_rollback_restores_previous_model
- TestRegistryBehaviors.test_hot_reload_updates_without_downtime
- TestRegistryDeployment.test_registry_hot_reload

### Target: SignalSchema
  Tested by 3 tests:

- TestStoreSchemaContracts.test_signal_schema_validation
- TestStoreSchemaContracts.test_signal_schema_validation
- TestStoreSchemaContracts.test_cross_store_consistency

### Target: xgb
  Tested by 3 tests:

- TestMLPipelineIntegration.test_e2e_ml_pipeline_with_real_data
- TestMLPipelineIntegration.test_data_consistency_across_pipeline_stages
- TestEndToEndPipeline.test_signal_generation_from_features

### Target: subprocess
  Tested by 3 tests:

- TestMLPipelineIntegration.test_docker_compose_stack_integration
- TestMLPipelineIntegration.test_docker_compose_stack_integration
- TestMLPipelineIntegration.test_docker_compose_stack_integration

### Target: redis
  Tested by 3 tests:

- TestMLPipelineIntegration.test_docker_compose_stack_integration
- TestMLPipelineIntegration.test_message_queue_failure_handling
- TestMLPipelineIntegration.test_message_queue_failure_handling

### Target: redis_client
  Tested by 3 tests:

- TestMLPipelineIntegration.test_docker_compose_stack_integration
- TestMLPipelineIntegration.test_message_queue_failure_handling
- TestMLPipelineIntegration.test_message_queue_failure_handling

### Target: postgres_registry
  Tested by 3 tests:

- TestDataRegistryE2E.test_full_day_pipeline_postgres
- test_backend_switching
- test_backend_switching

### Target: features1
  Tested by 3 tests:

- TestFeatureStoreInvariants.test_feature_immutability_invariant
- TestFeatureStoreInvariants.test_feature_immutability_invariant
- TestFeatureEngineerProperties.test_feature_determinism

### Target: processor
  Tested by 3 tests:

- TestDataProcessorSimple.test_process_market_data_simple
- TestDataProcessorSimple.test_process_market_data_with_crossed_market
- TestDataProcessorSimple.test_process_features_with_nan

### Target: bars_data
  Tested by 3 tests:

- TestNautilusDataPipeline.test_batch_vs_online_feature_parity
- TestNautilusDataPipeline.test_feature_scaling_consistency
- TestFeatureParity.test_feature_engineer_batch_vs_online_parity

### Target: batch_engineer
  Tested by 3 tests:

- TestNautilusDataPipeline.test_batch_vs_online_feature_parity
- TestNautilusDataPipeline.test_feature_scaling_consistency
- TestFeatureParity.test_feature_engineer_batch_vs_online_parity

### Target: bars_with_gaps
  Tested by 3 tests:

- TestNautilusDataPipeline.test_feature_engineering_with_gaps
- TestNautilusDataPipeline.test_feature_engineering_with_gaps
- test_edge_cases_and_circuit_breaker

### Target: timestamps
  Tested by 3 tests:

- TestTransformProviderIntegration.test_provider_scalability
- TestTransformProviderAdapter.test_adapter_handles_arbitrary_data_sizes
- TestEventScheduleProvider.test_provider_handles_any_timestamps

### Target: TestInstrumentProvider
  Tested by 3 tests:

- TestMLStrategyBacktest.test_multi_instrument_ml_portfolio
- TestMLStrategyBacktest.test_multi_instrument_ml_portfolio
- TestMLStrategyBacktest.test_multi_instrument_ml_portfolio

### Target: _rng
  Tested by 3 tests:

- TestL2L3RegistryStoreIntegration.test_l2_feature_computation_with_real_data
- TestL2L3RegistryStoreIntegration.test_l2_feature_computation_with_real_data
- TestL2L3RegistryStoreIntegration.test_l2_feature_computation_with_real_data

### Target: valid_transitions
  Tested by 3 tests:

- TestEndToEndProperties.test_state_machine_validity
- TestEndToEndProperties.test_state_machine_validity
- TestEndToEndProperties.test_state_machine_validity

### Target: meta_path
  Tested by 3 tests:

- test_tft_cli_registry_calibration_with_z_val
- test_tft_cli_registry_calibration_with_z_val
- test_tft_cli_training_minimal_flow

### Target: registry2
  Tested by 3 tests:

- TestModelRegistryBackends.test_postgres_backend_register_and_retrieve
- TestRegistryPerformance.test_registry_persistence_performance
- TestRegistryProperties.test_metrics_persistence

### Target: schema_json
  Tested by 3 tests:

- TestUnifiedRegistry.test_inference_model_registration
- TestModelContracts.test_teacher_without_l2_features
- TestRegistryPerformance.test_registry_bulk_registration_performance

### Target: registry1
  Tested by 3 tests:

- TestRegistryProperties.test_metrics_persistence
- TestRegistryProperties.test_metrics_persistence
- TestRegistryProperties.test_metrics_persistence

### Target: mock_asyncio_run
  Tested by 3 tests:

- TestMainFunction.test_main_successful_run
- TestMainFunction.test_main_successful_run
- TestMainFunction.test_main_successful_run

### Target: app
  Tested by 3 tests:

- TestHealthEndpoint.test_health_check_healthy
- TestHealthEndpoint.test_health_check_unhealthy
- TestDeploymentIntegration.test_health_check_endpoint_availability

### Target: mock_run
  Tested by 3 tests:

- TestServiceHealthChecks.test_check_postgres_healthy
- TestServiceHealthChecks.test_check_redis_healthy
- TestServiceHealthChecks.test_check_docker_compose_all_running

### Target: details
  Tested by 3 tests:

- TestPreflightCheck.test_preflight_check_type_mismatch
- TestPreflightCheck.test_preflight_check_schema_hash_mismatch
- TestSchemaMigration.test_dual_write_during_migration

### Target: interceptor
  Tested by 3 tests:

- test_interceptor_routes_to_recorder
- test_interceptor_routes_to_recorder
- test_interceptor_routes_to_recorder

### Target: registry_path
  Tested by 3 tests:

- test_plan_backfill_with_gaps
- test_plan_backfill_no_gaps
- test_plan_backfill_dataset_shortcuts

### Target: file_path
  Tested by 3 tests:

- TestSchedulerResilience.test_dbn_file_loading_errors
- TestDataCorruptionHandling.test_various_data_corruption_scenarios
- TestDataCorruptionHandling.test_various_data_corruption_scenarios

### Target: lock2
  Tested by 3 tests:

- TestSchedulerResilience.test_concurrent_scheduler_runs_prevention
- TestSchedulerResilience.test_concurrent_scheduler_runs_prevention
- TestSchedulerResilience.test_concurrent_scheduler_runs_prevention

### Target: f
  Tested by 3 tests:

- TestConfigurationLoading.test_load_config_yaml_file
- TestConfigurationLoading.test_load_config_yaml_file
- TestConfigurationLoading.test_load_config_yaml_file

### Target: ensemble
  Tested by 3 tests:

- TestMLSignalActor.test_ensemble_signal_with_no_component_signals
- TestMLSignalActor.test_ensemble_signal_partial_strategies
- TestSignalStrategies.test_ensemble_strategy

### Target: confidence_buffer
  Tested by 3 tests:

- TestCacheIntegration.test_ring_buffer_and_percentile_tracking
- TestCacheIntegration.test_ring_buffer_and_percentile_tracking
- TestCacheIntegration.test_ring_buffer_and_percentile_tracking

### Target: feature_cache
  Tested by 3 tests:

- TestCacheIntegration.test_feature_cache_with_ring_buffer_history
- TestCacheIntegration.test_feature_cache_with_ring_buffer_history
- TestCacheIntegration.test_feature_cache_with_ring_buffer_history

## Example Tests with Property Test Coverage
These example tests might be redundant with property tests:

- TestStrategyRegistry.test_get_strategy_lineage (covered by TestStrategyRegistry.test_get_strategy)
- TestValidateNoLookahead.test_validate_no_lookahead_pass (covered by TestValidateNoLookahead.test_validate_no_lookahead_property)
- TestValidateNoLookahead.test_validate_no_lookahead_fail (covered by TestValidateNoLookahead.test_validate_no_lookahead_property)
- TestValidateTimestamps.test_validate_timestamps_valid (covered by TestValidateTimestamps.test_validate_timestamps_property)
- TestValidateTimestamps.test_validate_timestamps_with_nulls (covered by TestValidateTimestamps.test_validate_timestamps_property)
- TestValidateTimestamps.test_validate_timestamps_unsorted (covered by TestValidateTimestamps.test_validate_timestamps_property)
- TestValidateTimestamps.test_validate_timestamps_negative (covered by TestValidateTimestamps.test_validate_timestamps_property)
- TestValidateTimestamps.test_validate_timestamps_future (covered by TestValidateTimestamps.test_validate_timestamps_property)
- TestAlignTimeseries.test_align_timeseries_inner (covered by TestAlignTimeseries.test_align_timeseries_property)
- TestAlignTimeseries.test_align_timeseries_left (covered by TestAlignTimeseries.test_align_timeseries_property)
- TestAlignTimeseries.test_align_timeseries_outer (covered by TestAlignTimeseries.test_align_timeseries_property)

## Tests with Duplicate Assertion Patterns

### Pattern: ('assert_allclose',)

- TestSignalPredictionMetamorphic.test_time_shift_invariance
- TestFeatureCompositionMetamorphic.test_feature_subset_consistency
- TestEndToEndPipeline.test_online_feature_parity
- TestIndicatorManagerProperties.test_indicator_determinism
- TestZeroAllocationHotPath.test_feature_parity_with_views
- TestUnifiedFeatureCalculation.test_unified_method_batch_mode
- TestUnifiedFeatureCalculation.test_unified_method_online_mode
- TestFeatureEngineerProperties.test_feature_scaling_invariance
- TestMLSignalActorProperties.test_prediction_distribution_property
- TestLockFreeRingBuffer.test_mean_and_std
- TestCacheIntegration.test_feature_cache_with_ring_buffer_history

## Tests That Could Be Parameterized
These test families could potentially use pytest.mark.parametrize:

### Base: TestMainFunction.test_main_successful_run

- TestMainFunction.test_main_successful_run
- TestMainFunction.test_main_successful_run
- TestMainFunction.test_main_successful_run

### Base: TestMainFunction.test_main_handles_keyboard_interrupt

- TestMainFunction.test_main_handles_keyboard_interrupt
- TestMainFunction.test_main_handles_keyboard_interrupt
- TestMainFunction.test_main_handles_keyboard_interrupt

### Base: TestMainFunction.test_main_handles_fatal_error

- TestMainFunction.test_main_handles_fatal_error
- TestMainFunction.test_main_handles_fatal_error
- TestMainFunction.test_main_handles_fatal_error

### Base: TestMainFunction.test_main_prints_startup_info

- TestMainFunction.test_main_prints_startup_info
- TestMainFunction.test_main_prints_startup_info
- TestMainFunction.test_main_prints_startup_info

### Base: TestGrafanaClient.test_make_request_success

- TestGrafanaClient.test_make_request_success_200
- TestGrafanaClient.test_make_request_success_201
- TestGrafanaClient.test_make_request_success_204

## Statistics

- Total tests: 1320
- Tests using fixtures: 650
- Tests with decorators: 296
- Unique test targets: 428

## Recommendations

1. **Review similar test names** - These might be duplicates
2. **Remove redundant example tests** - Property tests already cover these
3. **Use parametrized tests** - Reduce code duplication with @pytest.mark.parametrize
4. **Consolidate overlapping tests** - 124 targets are tested by 4+ tests
