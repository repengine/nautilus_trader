#!/usr/bin/env python3
"""
Test script for ML Pipeline Health monitoring.

This script:
1. Creates the monitoring views if they don't exist
2. Inserts sample data for testing
3. Runs the health check script
4. Cleans up test data

Usage:
    python test_pipeline_health.py --setup    # Create views and test data
    python test_pipeline_health.py --check    # Run health check
    python test_pipeline_health.py --cleanup  # Remove test data

"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def get_connection_string() -> str:
    """
    Get database connection string from environment or default.
    """
    return os.environ.get(
        "ML_DB_CONNECTION",
        "postgresql://postgres:postgres@localhost:5432/nautilus",
    )


def run_sql_file(sql_file: Path) -> bool:
    """
    Execute SQL file using psql.

    Args:
        sql_file: Path to SQL file

    Returns:
        True if successful, False otherwise

    """
    connection_string = get_connection_string()

    # Parse connection string for psql
    # Format: postgresql://user:pass@host:port/database
    parts = connection_string.replace("postgresql://", "").split("@")
    if len(parts) != 2:
        print(f"Invalid connection string format: {connection_string}")
        return False

    user_pass = parts[0].split(":")
    host_db = parts[1].split("/")
    host_port = host_db[0].split(":")

    user = user_pass[0]
    password = user_pass[1] if len(user_pass) > 1 else ""
    host = host_port[0]
    port = host_port[1] if len(host_port) > 1 else "5432"
    database = host_db[1] if len(host_db) > 1 else "nautilus"

    # Build psql command
    env = os.environ.copy()
    if password:
        env["PGPASSWORD"] = password

    cmd = [
        "psql",
        "-h",
        host,
        "-p",
        port,
        "-U",
        user,
        "-d",
        database,
        "-f",
        str(sql_file),
        "-v",
        "ON_ERROR_STOP=1",
    ]

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error executing SQL file: {result.stderr}")
            return False
        print(f"Successfully executed {sql_file.name}")
        return True
    except FileNotFoundError:
        print("Error: psql not found. Please install PostgreSQL client tools.")
        return False
    except Exception as e:
        print(f"Error executing SQL file: {e}")
        return False


def setup_test_environment() -> bool:
    """
    Set up test environment with monitoring views and sample data.

    Returns:
        True if successful, False otherwise

    """
    print("Setting up test environment...")

    # Get paths
    base_dir = Path(__file__).parent.parent
    schema_dir = base_dir / "schema"
    health_sql = schema_dir / "pipeline_health.sql"

    if not health_sql.exists():
        print(f"Error: {health_sql} not found")
        return False

    # Execute pipeline health views
    if not run_sql_file(health_sql):
        return False

    # Create sample test data
    test_data_sql = schema_dir / "test_pipeline_data.sql"
    test_data_content = """
-- Insert sample test data for health monitoring views
-- This creates realistic test data to validate the monitoring system

-- Helper function to get current timestamp in nanoseconds
CREATE OR REPLACE FUNCTION current_timestamp_ns() RETURNS BIGINT AS $$
BEGIN
    RETURN (EXTRACT(EPOCH FROM NOW()) * 1000000000)::BIGINT;
END;
$$ LANGUAGE plpgsql;

-- Ensure schema exists
CREATE SCHEMA IF NOT EXISTS ml;

-- Insert sample feature values (last 24 hours)
INSERT INTO ml_feature_values (feature_set_id, instrument_id, ts_event, ts_init, values, is_live, source)
SELECT
    'technical_indicators' as feature_set_id,
    instrument as instrument_id,
    current_timestamp_ns() - (hour * 3600 * 1000000000) as ts_event,
    current_timestamp_ns() - (hour * 3600 * 1000000000) + 1000000 as ts_init,
    jsonb_build_object(
        'rsi', 50 + random() * 50,
        'macd', random() * 2 - 1,
        'bollinger_upper', 100 + random() * 10,
        'bollinger_lower', 90 + random() * 10
    ) as values,
    CASE WHEN hour < 2 THEN true ELSE false END as is_live,
    CASE WHEN hour < 2 THEN 'live' ELSE 'historical' END as source
FROM
    generate_series(0, 23) as hour,
    (VALUES ('EURUSD.IDEALPRO'), ('GBPUSD.IDEALPRO'), ('USDJPY.IDEALPRO')) as t(instrument);

-- Insert sample feature computation stats
INSERT INTO ml_feature_computation_stats (feature_set_id, instrument_id, computation_time_ms, num_features, ts_event, ts_init)
SELECT
    'technical_indicators' as feature_set_id,
    instrument as instrument_id,
    50 + random() * 200 as computation_time_ms,  -- 50-250ms
    4 as num_features,
    current_timestamp_ns() - (hour * 3600 * 1000000000) as ts_event,
    current_timestamp_ns() - (hour * 3600 * 1000000000) + 1000000 as ts_init
FROM
    generate_series(0, 23) as hour,
    (VALUES ('EURUSD.IDEALPRO'), ('GBPUSD.IDEALPRO'), ('USDJPY.IDEALPRO')) as t(instrument);

-- Insert sample model predictions
INSERT INTO ml_model_predictions (model_id, instrument_id, ts_event, ts_init, prediction, confidence, features_used, inference_time_ms)
SELECT
    'xgboost_v1' as model_id,
    instrument as instrument_id,
    current_timestamp_ns() - (hour * 3600 * 1000000000) as ts_event,
    current_timestamp_ns() - (hour * 3600 * 1000000000) + 1000000 as ts_init,
    random() * 2 - 1 as prediction,  -- -1 to 1
    0.5 + random() * 0.5 as confidence,  -- 0.5 to 1.0
    jsonb_build_object('rsi', 50, 'macd', 0.1) as features_used,
    10 + random() * 90 as inference_time_ms  -- 10-100ms
FROM
    generate_series(0, 23) as hour,
    (VALUES ('EURUSD.IDEALPRO'), ('GBPUSD.IDEALPRO'), ('USDJPY.IDEALPRO')) as t(instrument);

-- Insert sample strategy signals
INSERT INTO ml_strategy_signals (strategy_id, instrument_id, signal_type, strength, ts_event, ts_init, model_predictions, risk_metrics, execution_params)
SELECT
    'momentum_v1' as strategy_id,
    instrument as instrument_id,
    CASE
        WHEN random() > 0.7 THEN 'BUY'
        WHEN random() > 0.3 THEN 'HOLD'
        ELSE 'SELL'
    END as signal_type,
    random() as strength,
    current_timestamp_ns() - (hour * 3600 * 1000000000) as ts_event,
    current_timestamp_ns() - (hour * 3600 * 1000000000) + 1000000 as ts_init,
    jsonb_build_object('xgboost_v1', 0.5) as model_predictions,
    jsonb_build_object('sharpe_ratio', 1.5, 'max_drawdown', 0.1) as risk_metrics,
    jsonb_build_object('stop_loss', 0.02, 'take_profit', 0.05) as execution_params
FROM
    generate_series(0, 23) as hour,
    (VALUES ('EURUSD.IDEALPRO'), ('GBPUSD.IDEALPRO'), ('USDJPY.IDEALPRO')) as t(instrument)
WHERE random() > 0.3;  -- Create some gaps

-- Simulate some errors for testing
INSERT INTO ml_model_predictions (model_id, instrument_id, ts_event, ts_init, prediction, confidence, features_used, inference_time_ms)
SELECT
    'xgboost_v1' as model_id,
    'BROKEN.TEST' as instrument_id,
    current_timestamp_ns() - (i * 60 * 1000000000) as ts_event,
    current_timestamp_ns() - (i * 60 * 1000000000) + 1000000 as ts_init,
    0 as prediction,
    0.3 as confidence,  -- Low confidence to trigger warnings
    jsonb_build_object() as features_used,
    1500 as inference_time_ms  -- High latency to trigger alerts
FROM generate_series(1, 5) as i;

SELECT 'Test data inserted successfully' as status;
"""

    test_data_sql.write_text(test_data_content)

    # Execute test data SQL
    if not run_sql_file(test_data_sql):
        return False

    # Clean up temp file
    test_data_sql.unlink()

    print("Test environment setup complete!")
    return True


def run_health_check() -> int:
    """
    Run the health check script.

    Returns:
        Exit code from health check

    """
    print("Running health check...")

    script_path = Path(__file__).parent / "check_pipeline_health.py"
    if not script_path.exists():
        print(f"Error: {script_path} not found")
        return 2

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
        )

        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr, file=sys.stderr)

        return result.returncode
    except Exception as e:
        print(f"Error running health check: {e}")
        return 2


def cleanup_test_data() -> bool:
    """
    Clean up test data.

    Returns:
        True if successful, False otherwise

    """
    print("Cleaning up test data...")

    cleanup_sql = Path(__file__).parent.parent / "schema" / "cleanup_test.sql"
    cleanup_content = """
-- Clean up test data
DELETE FROM ml_feature_values WHERE source IN ('live', 'historical') AND instrument_id LIKE '%.IDEALPRO';
DELETE FROM ml_feature_computation_stats WHERE instrument_id LIKE '%.IDEALPRO';
DELETE FROM ml_model_predictions WHERE instrument_id LIKE '%.IDEALPRO' OR instrument_id = 'BROKEN.TEST';
DELETE FROM ml_strategy_signals WHERE instrument_id LIKE '%.IDEALPRO';

SELECT 'Test data cleaned up' as status;
"""

    cleanup_sql.write_text(cleanup_content)

    # Execute cleanup SQL
    success = run_sql_file(cleanup_sql)

    # Clean up temp file
    cleanup_sql.unlink()

    if success:
        print("Test data cleanup complete!")

    return success


def main() -> int:
    """
    Main entry point for test script.

    Returns:
        Exit code

    """
    parser = argparse.ArgumentParser(description="Test ML Pipeline Health monitoring")
    parser.add_argument("--setup", action="store_true", help="Set up test environment")
    parser.add_argument("--check", action="store_true", help="Run health check")
    parser.add_argument("--cleanup", action="store_true", help="Clean up test data")
    parser.add_argument("--all", action="store_true", help="Run all steps")

    args = parser.parse_args()

    # Default to running all if no specific option
    if not any([args.setup, args.check, args.cleanup, args.all]):
        args.all = True

    exit_code = 0

    try:
        if args.setup or args.all:
            if not setup_test_environment():
                return 2

        if args.check or args.all:
            exit_code = run_health_check()

        if args.cleanup or args.all:
            if not cleanup_test_data():
                return 2

        return exit_code

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"Test failed: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
