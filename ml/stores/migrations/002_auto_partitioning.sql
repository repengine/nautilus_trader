-- Automatic Partition Management for ML Stores
-- This migration sets up automatic partition creation and maintenance

-- ============================================================================
-- Automatic Partition Creation Function
-- ============================================================================

-- Function to automatically create partitions ahead of time
CREATE OR REPLACE FUNCTION auto_create_partitions()
RETURNS VOID AS $$
DECLARE
    table_names TEXT[] := ARRAY['ml_feature_values', 'ml_model_predictions', 'ml_strategy_signals'];
    table_name TEXT;
    cur_date DATE;
    last_partition_date DATE;
    partition_name TEXT;
    start_ns BIGINT;
    end_ns BIGINT;
    months_ahead INTEGER := 3;  -- Always maintain 3 months ahead
BEGIN
    cur_date := CURRENT_DATE;

    FOREACH table_name IN ARRAY table_names
    LOOP
        -- Find the last partition for this table
        SELECT MAX(TO_DATE(substring(tablename from length(table_name) + 2 for 7), 'YYYY_MM'))
        INTO last_partition_date
        FROM pg_tables
        WHERE schemaname = 'public'
        AND tablename LIKE table_name || '_%';

        -- If no partitions exist, start from current month
        IF last_partition_date IS NULL THEN
            last_partition_date := DATE_TRUNC('month', cur_date);
        END IF;

        -- Create partitions up to 3 months in the future
        WHILE last_partition_date <= cur_date + INTERVAL '3 months' LOOP
            partition_name := table_name || '_' || TO_CHAR(last_partition_date, 'YYYY_MM');
            start_ns := EXTRACT(EPOCH FROM last_partition_date) * 1000000000;
            end_ns := EXTRACT(EPOCH FROM last_partition_date + INTERVAL '1 month') * 1000000000;

            -- Check if partition already exists
            IF NOT EXISTS (
                SELECT 1 FROM pg_tables
                WHERE schemaname = 'public'
                AND tablename = partition_name
            ) THEN
                EXECUTE format('
                    CREATE TABLE IF NOT EXISTS %I PARTITION OF %I
                    FOR VALUES FROM (%L) TO (%L)',
                    partition_name, table_name, start_ns, end_ns
                );

                RAISE NOTICE 'Created partition % for table %', partition_name, table_name;
            END IF;

            last_partition_date := last_partition_date + INTERVAL '1 month';
        END LOOP;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Automatic Partition Cleanup Function (Optional)
-- ============================================================================

-- Function to drop old partitions beyond retention period
CREATE OR REPLACE FUNCTION auto_cleanup_partitions(retention_months INTEGER DEFAULT 24)
RETURNS VOID AS $$
DECLARE
    table_names TEXT[] := ARRAY['ml_feature_values', 'ml_model_predictions', 'ml_strategy_signals'];
    table_name TEXT;
    partition_name TEXT;
    partition_date DATE;
    cutoff_date DATE;
BEGIN
    cutoff_date := CURRENT_DATE - (retention_months || ' months')::INTERVAL;

    FOREACH table_name IN ARRAY table_names
    LOOP
        -- Find old partitions to drop
        FOR partition_name IN
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = 'public'
            AND tablename LIKE table_name || '_%'
        LOOP
            -- Extract date from partition name
            BEGIN
                partition_date := TO_DATE(substring(partition_name from length(table_name) + 2 for 7), 'YYYY_MM');

                IF partition_date < cutoff_date THEN
                    EXECUTE format('DROP TABLE IF EXISTS %I CASCADE', partition_name);
                    RAISE NOTICE 'Dropped old partition %', partition_name;
                END IF;
            EXCEPTION
                WHEN OTHERS THEN
                    -- Skip if date parsing fails
                    CONTINUE;
            END;
        END LOOP;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Trigger-Based Automatic Partition Creation
-- ============================================================================

-- Function to check and create partition on insert
-- Note: Triggers on partitioned tables are expanded to child partitions by PostgreSQL.
--       Use TG_RELID to resolve the root partitioned table, not the child name.
CREATE OR REPLACE FUNCTION ensure_partition_exists()
RETURNS TRIGGER AS $$
DECLARE
    partition_date DATE;
    partition_name TEXT;
    root_table REGCLASS;
    table_name TEXT;
    start_ns BIGINT;
    end_ns BIGINT;
BEGIN
    -- Resolve the root partitioned table if this trigger fired on a child partition
    SELECT inhparent::REGCLASS
      INTO root_table
      FROM pg_inherits
     WHERE inhrelid = TG_RELID
     LIMIT 1;

    IF root_table IS NULL THEN
        table_name := TG_TABLE_NAME;  -- Trigger on root table
    ELSE
        table_name := root_table::TEXT;  -- Trigger fired on a partition; use root
    END IF;

    -- Calculate partition date from ts_event
    partition_date := DATE_TRUNC('month', TO_TIMESTAMP(NEW.ts_event / 1000000000.0));

    -- Generate partition name under the root table
    partition_name := table_name || '_' || TO_CHAR(partition_date, 'YYYY_MM');

    -- Check if partition exists (in current schema)
    IF NOT EXISTS (
        SELECT 1
          FROM pg_tables
         WHERE schemaname = TG_TABLE_SCHEMA
           AND tablename = partition_name
    ) THEN
        -- Calculate boundaries
        start_ns := EXTRACT(EPOCH FROM partition_date) * 1000000000;
        end_ns := EXTRACT(EPOCH FROM (partition_date + INTERVAL '1 month')) * 1000000000;

        -- Create partition on the root table
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L)',
            partition_name, table_name, start_ns, end_ns
        );

        RAISE NOTICE 'Auto-created partition % for table %', partition_name, table_name;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for automatic partition creation on insert
CREATE OR REPLACE TRIGGER auto_create_partition_feature_values
    BEFORE INSERT ON ml_feature_values
    FOR EACH ROW
    EXECUTE FUNCTION ensure_partition_exists();

CREATE OR REPLACE TRIGGER auto_create_partition_model_predictions
    BEFORE INSERT ON ml_model_predictions
    FOR EACH ROW
    EXECUTE FUNCTION ensure_partition_exists();

CREATE OR REPLACE TRIGGER auto_create_partition_strategy_signals
    BEFORE INSERT ON ml_strategy_signals
    FOR EACH ROW
    EXECUTE FUNCTION ensure_partition_exists();

-- ============================================================================
-- Scheduled Maintenance with pg_cron (Optional - requires pg_cron extension)
-- ============================================================================

-- Uncomment if pg_cron is available:
/*
-- Install pg_cron extension if not already installed
CREATE EXTENSION IF NOT EXISTS pg_cron;

-- Schedule daily partition maintenance at 2 AM
SELECT cron.schedule(
    'ml_partition_maintenance',
    '0 2 * * *',  -- Daily at 2 AM
    $$
    SELECT auto_create_partitions();
    SELECT auto_cleanup_partitions(24);  -- Keep 24 months of data
    $$
);
*/

-- ============================================================================
-- Alternative: PostgreSQL 14+ Native Partitioning (if available)
-- ============================================================================

-- For PostgreSQL 14+, you can use automatic list/range partition creation
-- This is a more modern approach but requires PostgreSQL 14 or later

/*
-- Example for PostgreSQL 14+ with automatic partition creation
-- This would replace the manual partition creation

-- Drop existing tables first (BE CAREFUL - this loses data!)
-- DROP TABLE IF EXISTS ml_feature_values CASCADE;

-- Recreate with automatic partitioning
CREATE TABLE ml_feature_values (
    id BIGSERIAL,
    feature_set_id VARCHAR(255) NOT NULL,
    instrument_id VARCHAR(100) NOT NULL,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,
    values JSONB NOT NULL,
    is_live BOOLEAN DEFAULT FALSE,
    source VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (id, ts_event)
) PARTITION BY RANGE (ts_event);

-- Enable automatic partition creation (PostgreSQL 14+)
ALTER TABLE ml_feature_values SET (
    partition_auto_create = on,
    partition_auto_create_interval = '1 month'
);
*/

-- ============================================================================
-- Manual Execution Commands
-- ============================================================================

-- Run these manually or schedule with your preferred scheduler:
-- SELECT auto_create_partitions();  -- Creates future partitions
-- SELECT auto_cleanup_partitions(24);  -- Removes partitions older than 24 months

-- Check existing partitions:
-- SELECT tablename, pg_size_pretty(pg_total_relation_size(tablename::regclass)) as size
-- FROM pg_tables
-- WHERE schemaname = 'public'
-- AND tablename LIKE 'ml_%'
-- ORDER BY tablename;
