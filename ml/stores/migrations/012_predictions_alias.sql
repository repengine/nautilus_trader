-- Migration: Register generic predictions dataset and deprecate legacy alias.
-- Rollback: DELETE FROM ml_dataset_registry WHERE dataset_id = 'predictions'; UPDATE metadata for 'predictions_xgboost' to remove deprecated flag.

-- Add a generic 'predictions' dataset manifest and deprecate 'predictions_xgboost'.
-- Safe to run multiple times (uses ON CONFLICT and guarded updates)

-- Insert generic predictions dataset if missing
INSERT INTO ml_dataset_registry (
    dataset_id, name, version, dataset_type, storage_kind,
    location, partitioning, retention_days, schema, schema_hash,
    constraints, parents, pipeline_signature, metadata
) VALUES (
    'predictions', 'Model Predictions', '1.0.0', 'PREDICTIONS', 'postgres',
    'ml_model_predictions', '{"by": "ts_event", "interval": "monthly"}'::jsonb, 90,
    '{"model_id": "str", "instrument_id": "str", "ts_event": "int64", "ts_init": "int64", "prediction": "float64", "confidence": "float64", "features_used": "json", "inference_time_ms": "float64", "is_live": "bool"}'::jsonb,
    '', '{}'::jsonb, '[]'::jsonb, 'model_inference_auto',
    '{"ts_field": "ts_event", "primary_keys": ["model_id", "instrument_id", "ts_event"], "auto_registered": true}'::jsonb
)
ON CONFLICT (dataset_id) DO NOTHING;

-- Deprecate the old predictions_xgboost manifest if it exists
UPDATE ml_dataset_registry
SET metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object('deprecated', true),
    last_modified = NOW()
WHERE dataset_id = 'predictions_xgboost'
  AND (metadata IS NULL OR (metadata->>'deprecated') IS DISTINCT FROM 'true');
