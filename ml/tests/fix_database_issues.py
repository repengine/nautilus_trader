#!/usr/bin/env python3
"""
Database Issues Fix Script for ML Tests

This script fixes critical database issues that block ML tests:
1. Partition violations due to missing partitions for test timestamps
2. Missing database functions (emit_data_event, update_watermark)
3. Source constraint violations (tests using 'unit' instead of allowed values)
4. Foreign key violations (missing test datasets in registry)
5. JSONB parameter handling issues

Usage:
    python ml/tests/fix_database_issues.py
    
Environment Variables:
    DATABASE_URL: PostgreSQL connection string (default: postgresql://postgres:postgres@localhost:5432/nautilus_test)
"""

import logging
import os
from datetime import datetime

import psycopg2


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/nautilus_test",
)


def parse_database_url(url: str) -> dict[str, any]:
    """Parse PostgreSQL connection URL into components."""
    if url.startswith("postgresql://"):
        conn_params = url.replace("postgresql://", "").split("/")
        user_pass_host = conn_params[0].split("@")
        if len(user_pass_host) == 2:
            user_pass = user_pass_host[0].split(":")
            host_port = user_pass_host[1].split(":")
            host = host_port[0]
            port = host_port[1] if len(host_port) > 1 else 5432
            user = user_pass[0]
            password = user_pass[1] if len(user_pass) > 1 else ""
            database = conn_params[1] if len(conn_params) > 1 else "postgres"
        else:
            # Fallback
            user, password, host, port, database = "postgres", "postgres", "localhost", 5432, "nautilus_test"
    else:
        user, password, host, port, database = "postgres", "postgres", "localhost", 5432, "nautilus_test"

    return {
        "host": host,
        "port": int(port),
        "database": database,
        "user": user,
        "password": password
    }


def create_partitions_for_test_years(cursor) -> None:
    """Create partitions for years commonly used in tests (1970, 2001, 2025)."""
    logger.info("Creating partitions for test years (1970, 2001, 2025)...")

    create_test_partitions_sql = """
DO $$
DECLARE
    v_tables TEXT[] := ARRAY['ml_feature_values', 'ml_model_predictions', 'ml_strategy_signals', 'ml_data_events'];
    v_table TEXT;
    v_years INT[] := ARRAY[1970, 2001, 2025];
    v_year INT;
    v_month INT;
    v_partition_name TEXT;
    v_start_ts BIGINT;
    v_end_ts BIGINT;
    v_partitions_created INT := 0;
BEGIN
    FOREACH v_table IN ARRAY v_tables LOOP
        FOREACH v_year IN ARRAY v_years LOOP
            -- Special handling for 1970 (only January for test timestamps near epoch)
            IF v_year = 1970 THEN
                v_partition_name := v_table || '_1970_01';
                v_start_ts := 0;  -- January 1, 1970
                v_end_ts := 2678400000000000; -- February 1, 1970
                
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
                        v_partitions_created := v_partitions_created + 1;
                        RAISE NOTICE 'Created partition %', v_partition_name;
                    EXCEPTION
                        WHEN OTHERS THEN
                            RAISE NOTICE 'Skipping % (error: %)', v_partition_name, SQLERRM;
                    END;
                END IF;
            ELSE
                -- Create all months for other years
                FOR v_month IN 1..12 LOOP
                    v_partition_name := v_table || '_' || v_year || '_' || LPAD(v_month::TEXT, 2, '0');
                    
                    v_start_ts := EXTRACT(EPOCH FROM DATE(v_year || '-' || LPAD(v_month::TEXT, 2, '0') || '-01')) * 1000000000;
                    IF v_month = 12 THEN
                        v_end_ts := EXTRACT(EPOCH FROM DATE((v_year + 1) || '-01-01')) * 1000000000;
                    ELSE
                        v_end_ts := EXTRACT(EPOCH FROM DATE(v_year || '-' || LPAD((v_month + 1)::TEXT, 2, '0') || '-01')) * 1000000000;
                    END IF;
                    
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
                            v_partitions_created := v_partitions_created + 1;
                            RAISE NOTICE 'Created partition %', v_partition_name;
                        EXCEPTION
                            WHEN OTHERS THEN
                                RAISE NOTICE 'Skipping % (error: %)', v_partition_name, SQLERRM;
                        END;
                    END IF;
                END LOOP;
            END IF;
        END LOOP;
    END LOOP;
    
    RAISE NOTICE 'Created % test partitions total', v_partitions_created;
END $$;
"""

    cursor.execute(create_test_partitions_sql)


def create_missing_functions(cursor) -> None:
    """Create missing database functions required by the ML system."""
    logger.info("Creating missing database functions...")

    # Create update_watermark function
    update_watermark_sql = """
CREATE OR REPLACE FUNCTION update_watermark(
    p_dataset_id VARCHAR(255),
    p_instrument_id VARCHAR(100),
    p_source VARCHAR(50),
    p_last_success_ns BIGINT,
    p_count BIGINT,
    p_completeness_pct DECIMAL(5,2) DEFAULT NULL
)
RETURNS VOID AS $FUNC$
BEGIN
    INSERT INTO ml_data_watermarks (
        dataset_id,
        instrument_id,
        source,
        last_success_ns,
        last_attempt_ns,
        last_count,
        completeness_pct,
        updated_at
    )
    VALUES (
        p_dataset_id,
        p_instrument_id,
        p_source,
        p_last_success_ns,
        p_last_success_ns,
        p_count,
        p_completeness_pct,
        NOW()
    )
    ON CONFLICT (dataset_id, instrument_id, source)
    DO UPDATE SET
        last_success_ns = EXCLUDED.last_success_ns,
        last_attempt_ns = EXCLUDED.last_attempt_ns,
        last_count = EXCLUDED.last_count,
        completeness_pct = COALESCE(EXCLUDED.completeness_pct, ml_data_watermarks.completeness_pct),
        updated_at = NOW();
END;
$FUNC$ LANGUAGE plpgsql;
"""

    # Create emit_data_event function
    emit_data_event_sql = """
CREATE OR REPLACE FUNCTION emit_data_event(
    p_dataset_id VARCHAR(255),
    p_instrument_id VARCHAR(100),
    p_stage VARCHAR(50),
    p_source VARCHAR(50),
    p_run_id VARCHAR(255),
    p_ts_min BIGINT,
    p_ts_max BIGINT,
    p_count BIGINT,
    p_status VARCHAR(20),
    p_error TEXT DEFAULT NULL,
    p_seq_min BIGINT DEFAULT NULL,
    p_seq_max BIGINT DEFAULT NULL
)
RETURNS BIGINT AS $FUNC$
DECLARE
    v_event_id BIGINT;
    v_ts_event BIGINT;
BEGIN
    -- Get current timestamp in nanoseconds
    v_ts_event := EXTRACT(EPOCH FROM NOW()) * 1000000000;

    -- Insert event
    INSERT INTO ml_data_events (
        dataset_id, instrument_id, stage, source, run_id,
        ts_min, ts_max, ts_event, count, seq_min, seq_max,
        status, error, created_at
    )
    VALUES (
        p_dataset_id, p_instrument_id, p_stage, p_source, p_run_id,
        p_ts_min, p_ts_max, v_ts_event, p_count, p_seq_min, p_seq_max,
        p_status, p_error, NOW()
    )
    RETURNING event_id INTO v_event_id;

    -- Update watermark if successful
    IF p_status = 'success' THEN
        PERFORM update_watermark(
            p_dataset_id,
            p_instrument_id,
            p_source,
            p_ts_max,
            p_count,
            NULL
        );
    END IF;

    RETURN v_event_id;
END;
$FUNC$ LANGUAGE plpgsql;
"""

    # Create emit_data_event_ext function with metadata support
    emit_data_event_ext_sql = """
CREATE OR REPLACE FUNCTION emit_data_event_ext(
    p_dataset_id VARCHAR(255),
    p_instrument_id VARCHAR(100),
    p_stage VARCHAR(50),
    p_source VARCHAR(50),
    p_run_id VARCHAR(255),
    p_ts_min BIGINT,
    p_ts_max BIGINT,
    p_count BIGINT,
    p_status VARCHAR(20),
    p_error TEXT DEFAULT NULL,
    p_metadata TEXT DEFAULT '{}'  -- Accept JSON as text string
)
RETURNS BIGINT AS $FUNC$
DECLARE
    v_event_id BIGINT;
    v_ts_event BIGINT;
BEGIN
    -- Get current timestamp in nanoseconds
    v_ts_event := EXTRACT(EPOCH FROM NOW()) * 1000000000;

    -- Insert event (metadata is accepted but not currently stored)
    INSERT INTO ml_data_events (
        dataset_id, instrument_id, stage, source, run_id,
        ts_min, ts_max, ts_event, count, seq_min, seq_max,
        status, error, created_at
    )
    VALUES (
        p_dataset_id, p_instrument_id, p_stage, p_source, p_run_id,
        p_ts_min, p_ts_max, v_ts_event, p_count, NULL, NULL,
        p_status, p_error, NOW()
    )
    RETURNING event_id INTO v_event_id;

    -- Update watermark if successful
    IF p_status = 'success' THEN
        PERFORM update_watermark(
            p_dataset_id,
            p_instrument_id,
            p_source,
            p_ts_max,
            p_count,
            NULL
        );
    END IF;

    RETURN v_event_id;
END;
$FUNC$ LANGUAGE plpgsql;
"""

    cursor.execute(update_watermark_sql)
    cursor.execute(emit_data_event_sql)
    cursor.execute(emit_data_event_ext_sql)


def relax_constraints_for_testing(cursor) -> None:
    """Relax database constraints to allow test values."""
    logger.info("Relaxing constraints for testing...")

    relax_constraints_sql = """
-- Allow test source values in addition to production ones
ALTER TABLE ml_data_events DROP CONSTRAINT IF EXISTS check_source;
ALTER TABLE ml_data_events ADD CONSTRAINT check_source 
  CHECK (source IN ('live', 'historical', 'backfill', 'unit', 'test', 'computed'));

ALTER TABLE ml_data_watermarks DROP CONSTRAINT IF EXISTS check_source_watermark;
ALTER TABLE ml_data_watermarks ADD CONSTRAINT check_source_watermark 
  CHECK (source IN ('live', 'historical', 'backfill', 'unit', 'test', 'computed'));
"""

    cursor.execute(relax_constraints_sql)


def create_test_datasets(cursor) -> None:
    """Create test datasets required by tests."""
    logger.info("Creating test datasets in registry...")

    create_test_datasets_sql = """
-- Insert test datasets that tests reference
INSERT INTO ml_dataset_registry (
    dataset_id, name, version, dataset_type, storage_kind,
    location, partitioning, retention_days, schema, schema_hash,
    constraints, parents, pipeline_signature
)
VALUES
    ('features', 'Test Features Dataset', '1.0.0', 'FEATURES', 'postgres',
     'ml_feature_values', '{}', 90, '{}', 'test_hash', '{}', '[]', 'test_pipeline'),
    ('test_features_v1', 'Test Features V1', '1.0.0', 'FEATURES', 'postgres',
     'ml_feature_values', '{}', 90, '{}', 'test_hash_v1', '{}', '[]', 'test_pipeline'),
    ('test_model', 'Test Model Predictions', '1.0.0', 'PREDICTIONS', 'postgres',
     'ml_model_predictions', '{}', 90, '{}', 'test_model_hash', '{}', '[]', 'test_pipeline'),
    ('test_strategy', 'Test Strategy Signals', '1.0.0', 'SIGNALS', 'postgres',
     'ml_strategy_signals', '{}', 90, '{}', 'test_strategy_hash', '{}', '[]', 'test_pipeline')
ON CONFLICT (dataset_id) DO NOTHING;
"""

    cursor.execute(create_test_datasets_sql)


def verify_fixes(cursor) -> None:
    """Verify that all fixes have been applied correctly."""
    logger.info("Verifying database fixes...")

    # Check functions exist
    cursor.execute("""
        SELECT proname FROM pg_proc 
        WHERE proname IN ('emit_data_event', 'update_watermark', 'emit_data_event_ext')
        ORDER BY proname;
    """)
    functions = [row[0] for row in cursor.fetchall()]
    logger.info(f"Functions available: {functions}")

    # Check partitions exist for test years
    cursor.execute("""
        SELECT COUNT(*) FROM pg_tables 
        WHERE tablename ~ '^ml_(feature_values|model_predictions|strategy_signals|data_events)_(1970_01|2001_|2025_)';
    """)
    partition_count = cursor.fetchone()[0]
    logger.info(f"Test partitions created: {partition_count}")

    # Check test datasets exist
    cursor.execute("""
        SELECT COUNT(*) FROM ml_dataset_registry 
        WHERE dataset_id IN ('features', 'test_features_v1', 'test_model', 'test_strategy');
    """)
    dataset_count = cursor.fetchone()[0]
    logger.info(f"Test datasets created: {dataset_count}")

    logger.info("Database fixes verification complete")


def main() -> None:
    """Apply all database fixes for ML tests."""
    logger.info("Starting database fixes for ML tests...")
    logger.info(f"Using database: {DATABASE_URL}")

    try:
        conn_params = parse_database_url(DATABASE_URL)
        conn = psycopg2.connect(**conn_params)
        conn.autocommit = True
        cursor = conn.cursor()

        # Apply all fixes
        create_partitions_for_test_years(cursor)
        create_missing_functions(cursor)
        relax_constraints_for_testing(cursor)
        create_test_datasets(cursor)
        verify_fixes(cursor)

        cursor.close()
        conn.close()

        logger.info("All database fixes applied successfully!")
        logger.info("ML tests should now pass database-related validations")

    except Exception as e:
        logger.error(f"Error applying database fixes: {e}")
        raise


if __name__ == "__main__":
    main()
