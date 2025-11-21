-- Migration: Add BRIN indexes to partitioned ML tables for faster scans.
-- Rollback: DROP INDEX IF EXISTS brin_ml_feature_values_ts; DROP INDEX IF EXISTS brin_ml_model_predictions_ts; DROP INDEX IF EXISTS brin_ml_strategy_signals_ts.

-- BRIN indexes for time-range efficiency on large partitioned tables
-- Safe to run multiple times (uses IF NOT EXISTS)

-- Feature values (optional BRIN on ts_event for range scans across partitions)
CREATE INDEX IF NOT EXISTS brin_ml_feature_values_ts
    ON ml_feature_values USING BRIN (ts_event);

-- Model predictions
CREATE INDEX IF NOT EXISTS brin_ml_model_predictions_ts
    ON ml_model_predictions USING BRIN (ts_event);

-- Strategy signals
CREATE INDEX IF NOT EXISTS brin_ml_strategy_signals_ts
    ON ml_strategy_signals USING BRIN (ts_event);
