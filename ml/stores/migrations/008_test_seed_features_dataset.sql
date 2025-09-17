-- Test-only seed for DataRegistry: ensure 'features' dataset exists in the test DB.
-- This script is mounted only in docker-compose environments which use
-- ml/stores/migrations for initialization. To avoid affecting production,
-- the insert executes only when the current database is 'nautilus_test'.

DO $$
BEGIN
    IF current_database() = 'nautilus_test' THEN
        INSERT INTO ml_dataset_registry (
            dataset_id,
            name,
            version,
            dataset_type,
            storage_kind,
            location,
            partitioning,
            retention_days,
            schema,
            schema_hash,
            constraints,
            parents,
            pipeline_signature,
            metadata
        ) VALUES (
            'features',
            'Runtime Features',
            '1.0.0',
            'FEATURES',
            'postgres',
            'ml_feature_values',
            '{"by": "ts_event", "interval": "monthly"}'::jsonb,
            180,
            '{"feature_set_id": "str", "instrument_id": "str", "ts_event": "int64", "ts_init": "int64", "values": "jsonb"}'::jsonb,
            '',  -- schema_hash optional in seed
            '{}'::jsonb,
            '[]'::jsonb,
            'feature_runtime_auto',
            '{"ts_field": "ts_event", "primary_keys": ["feature_set_id", "instrument_id", "ts_event"], "auto_registered": true}'::jsonb
        )
        ON CONFLICT (dataset_id) DO NOTHING;

        -- Ensure a generic 'predictions' dataset exists for test-mode event/watermark flows
        INSERT INTO ml_dataset_registry (
            dataset_id,
            name,
            version,
            dataset_type,
            storage_kind,
            location,
            partitioning,
            retention_days,
            schema,
            schema_hash,
            constraints,
            parents,
            pipeline_signature,
            metadata
        ) VALUES (
            'predictions',
            'Model Predictions',
            '1.0.0',
            'PREDICTIONS',
            'postgres',
            'ml_model_predictions',
            '{"by": "ts_event", "interval": "monthly"}'::jsonb,
            90,
            '{"model_id": "str", "instrument_id": "str", "ts_event": "int64", "ts_init": "int64", "prediction": "float64", "confidence": "float64", "features_used": "json", "inference_time_ms": "float64", "is_live": "bool"}'::jsonb,
            '',
            '{}'::jsonb,
            '[]'::jsonb,
            'model_inference_auto',
            '{"ts_field": "ts_event", "primary_keys": ["model_id", "instrument_id", "ts_event"], "auto_registered": true}'::jsonb
        )
        ON CONFLICT (dataset_id) DO NOTHING;
    END IF;
END
$$;
