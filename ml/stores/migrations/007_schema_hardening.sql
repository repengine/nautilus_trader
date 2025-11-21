-- Migration: Harden ML store schemas via unique indexes and timestamp normalization.
-- Rollback: DROP INDEX IF EXISTS uq_ml_feature_values_key; revert column type changes manually via ALTER TABLE.

-- Schema Hardening Migration (007)
-- - Adds unique upsert key for features
-- - Standardizes created_at types (TIMESTAMPTZ with DEFAULT NOW())
-- - Qualifies helper views are moved separately (008_views.sql)

-- Ensure unique upsert key for ml_feature_values
-- Postgres requires partition key to be part of unique index
-- (ts_event included explicitly in the definition below)
CREATE UNIQUE INDEX IF NOT EXISTS uq_ml_feature_values_key
    ON public.ml_feature_values (feature_set_id, instrument_id, ts_event);

-- Standardize created_at types to TIMESTAMPTZ for analytics friendliness
-- Convert existing BIGINT nanoseconds to timestamptz if needed
DO $$
BEGIN
    -- ml_model_predictions
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'ml_model_predictions' AND column_name = 'created_at'
              AND data_type IN ('bigint', 'integer')
    ) THEN
        EXECUTE 'ALTER TABLE public.ml_model_predictions
                 ALTER COLUMN created_at TYPE TIMESTAMPTZ
                 USING to_timestamp(created_at::double precision / 1000000000)';
        EXECUTE 'ALTER TABLE public.ml_model_predictions
                 ALTER COLUMN created_at SET DEFAULT NOW()';
    END IF;

    -- ml_strategy_signals
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'ml_strategy_signals' AND column_name = 'created_at'
              AND data_type IN ('bigint', 'integer')
    ) THEN
        EXECUTE 'ALTER TABLE public.ml_strategy_signals
                 ALTER COLUMN created_at TYPE TIMESTAMPTZ
                 USING to_timestamp(created_at::double precision / 1000000000)';
        EXECUTE 'ALTER TABLE public.ml_strategy_signals
                 ALTER COLUMN created_at SET DEFAULT NOW()';
    END IF;
END$$;
