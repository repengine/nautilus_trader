-- Migration: add cold-path fields and feature linkage to models table
ALTER TABLE models ADD COLUMN IF NOT EXISTS serveable BOOLEAN DEFAULT TRUE;
ALTER TABLE models ADD COLUMN IF NOT EXISTS artifact_format TEXT DEFAULT 'onnx';
ALTER TABLE models ADD COLUMN IF NOT EXISTS feature_set_id TEXT;
ALTER TABLE models ADD COLUMN IF NOT EXISTS pipeline_signature TEXT;
ALTER TABLE models ADD COLUMN IF NOT EXISTS pipeline_version TEXT;
CREATE INDEX IF NOT EXISTS idx_models_feature_set_id ON models(feature_set_id);
