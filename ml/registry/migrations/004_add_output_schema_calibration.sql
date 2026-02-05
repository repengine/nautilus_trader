-- Migration: add output schema and calibration metadata to models table
ALTER TABLE models ADD COLUMN IF NOT EXISTS output_schema JSONB;
ALTER TABLE models ADD COLUMN IF NOT EXISTS calibration JSONB;
