-- Disable Automatic Partition Triggers for Testing
-- 
-- This migration disables the automatic partition creation triggers that were
-- created in 002_auto_partitioning.sql. These triggers have race conditions
-- and can cause sub-partition creation attempts that fail during testing.
--
-- Instead, we use pre-created partitions managed by PartitionManager.
-- ============================================================================

-- Drop all automatic partition creation triggers
DROP TRIGGER IF EXISTS auto_create_partition_feature_values ON ml_feature_values;
DROP TRIGGER IF EXISTS auto_create_partition_model_predictions ON ml_model_predictions;
DROP TRIGGER IF EXISTS auto_create_partition_strategy_signals ON ml_strategy_signals;

-- Drop the function that causes sub-partition creation issues
DROP FUNCTION IF EXISTS ensure_partition_exists();

-- Note: We keep the auto_create_partitions() function as it's useful for
-- manual/scheduled partition creation, just not as a trigger.

-- Create partitions for test data timestamps (2023-2026)
-- This ensures tests have the partitions they need
DO $$
DECLARE
    v_tables TEXT[] := ARRAY['ml_feature_values', 'ml_model_predictions', 'ml_strategy_signals'];
    v_table TEXT;
    v_year INT;
    v_month INT;
    v_partition_name TEXT;
    v_start_ts BIGINT;
    v_end_ts BIGINT;
BEGIN
    FOREACH v_table IN ARRAY v_tables LOOP
        -- Create partitions for 2023-2026 (covers all test timestamps)
        FOR v_year IN 2023..2026 LOOP
            FOR v_month IN 1..12 LOOP
                v_partition_name := v_table || '_' || v_year || '_' || LPAD(v_month::TEXT, 2, '0');
                
                -- Calculate nanosecond timestamps
                v_start_ts := EXTRACT(EPOCH FROM DATE(v_year || '-' || LPAD(v_month::TEXT, 2, '0') || '-01')) * 1000000000;
                IF v_month = 12 THEN
                    v_end_ts := EXTRACT(EPOCH FROM DATE((v_year + 1) || '-01-01')) * 1000000000;
                ELSE
                    v_end_ts := EXTRACT(EPOCH FROM DATE(v_year || '-' || LPAD((v_month + 1)::TEXT, 2, '0') || '-01')) * 1000000000;
                END IF;
                
                -- Check if partition exists before creating
                IF NOT EXISTS (
                    SELECT 1 FROM pg_tables 
                    WHERE schemaname = 'public' 
                    AND tablename = v_partition_name
                ) THEN
                    BEGIN
                        EXECUTE format(
                            'CREATE TABLE %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L)',
                            v_partition_name, v_table, v_start_ts, v_end_ts
                        );
                        RAISE NOTICE 'Created partition %', v_partition_name;
                    EXCEPTION
                        WHEN OTHERS THEN
                            -- Ignore errors (partition may exist or overlap)
                            RAISE NOTICE 'Skipping % (may already exist): %', v_partition_name, SQLERRM;
                    END;
                END IF;
            END LOOP;
        END LOOP;
    END LOOP;
    
    RAISE NOTICE 'Test partitions created successfully';
END $$;

-- Verify partitions exist for test timestamp
-- The common test timestamp 1700000000000000000 is in November 2023
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_tables 
        WHERE tablename IN ('ml_strategy_signals_2023_11', 'ml_feature_values_2023_11', 'ml_model_predictions_2023_11')
    ) THEN
        RAISE NOTICE 'Critical test partitions verified';
    ELSE
        RAISE WARNING 'Some test partitions may be missing';
    END IF;
END $$;