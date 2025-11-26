-- ML Stores Schema Migration
-- Creates tables for storing ML features, predictions, and signals

-- ============================================================================
-- Helper Functions (must be defined before use)
-- ============================================================================

-- Function to create partitions for specified months
CREATE OR REPLACE FUNCTION create_monthly_partitions(
    table_name TEXT,
    start_date DATE,
    num_months INTEGER
)
RETURNS VOID AS $$
DECLARE
    partition_date DATE;
    partition_name TEXT;
    start_ns BIGINT;
    end_ns BIGINT;
BEGIN
    FOR i IN 0..num_months-1 LOOP
        partition_date := start_date + (i || ' months')::INTERVAL;
        partition_name := table_name || '_' || TO_CHAR(partition_date, 'YYYY_MM');
        start_ns := EXTRACT(EPOCH FROM partition_date) * 1000000000;
        end_ns := EXTRACT(EPOCH FROM partition_date + '1 month'::INTERVAL) * 1000000000;

        EXECUTE format('
            CREATE TABLE IF NOT EXISTS %I PARTITION OF %I
            FOR VALUES FROM (%L) TO (%L)',
            partition_name, table_name, start_ns, end_ns
        );
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Feature Store Tables
-- ============================================================================

-- Feature values table (partitioned by time)
CREATE TABLE IF NOT EXISTS ml_feature_values (
    id BIGSERIAL,
    feature_set_id VARCHAR(255) NOT NULL,
    instrument_id VARCHAR(100) NOT NULL,
    ts_event BIGINT NOT NULL,  -- Nautilus convention: nanoseconds
    ts_init BIGINT NOT NULL,
    values JSONB NOT NULL,      -- Feature name -> value mapping
    is_live BOOLEAN DEFAULT FALSE,
    source VARCHAR(50),         -- 'historical', 'live', 'backfill'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    PRIMARY KEY (id, ts_event)
) PARTITION BY RANGE (ts_event);

-- Create partitions for recent past and future (2024-2026)
-- Using the helper function to create all partitions
SELECT create_monthly_partitions('ml_feature_values', '2024-01-01'::DATE, 36);

-- Default partition to catch rows outside pre-created monthly ranges
CREATE TABLE IF NOT EXISTS ml_feature_values_default
    PARTITION OF ml_feature_values DEFAULT;

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_ml_feature_values_lookup
    ON ml_feature_values (feature_set_id, instrument_id, ts_event);
CREATE INDEX IF NOT EXISTS idx_ml_feature_values_live
    ON ml_feature_values (is_live) WHERE is_live = TRUE;

-- Ensure upsert key exists for ON CONFLICT in FeatureStore
CREATE UNIQUE INDEX IF NOT EXISTS uq_ml_feature_values_key
    ON ml_feature_values (feature_set_id, instrument_id, ts_event);

-- Feature computation metadata
CREATE TABLE IF NOT EXISTS ml_feature_computation_stats (
    id BIGSERIAL PRIMARY KEY,
    feature_set_id VARCHAR(255) NOT NULL,
    instrument_id VARCHAR(100) NOT NULL,
    computation_time_ms FLOAT NOT NULL,
    num_features INTEGER NOT NULL,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Feature lineage tracking
CREATE TABLE IF NOT EXISTS ml_feature_lineage (
    id BIGSERIAL PRIMARY KEY,
    feature_set_id VARCHAR(255) NOT NULL,
    parent_feature_set_id VARCHAR(255),
    transformation_applied TEXT,
    parameters JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- Model Store Tables
-- ============================================================================

-- Model predictions table (partitioned by time)
CREATE TABLE IF NOT EXISTS ml_model_predictions (
    model_id VARCHAR(255) NOT NULL,
    instrument_id VARCHAR(100) NOT NULL,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,
    prediction FLOAT NOT NULL,
    confidence FLOAT,
    features_used JSONB,  -- Feature values at prediction time
    inference_time_ms FLOAT,
    is_live BOOLEAN DEFAULT FALSE,
    created_at BIGINT,

    PRIMARY KEY (model_id, instrument_id, ts_event)
) PARTITION BY RANGE (ts_event);

-- Create partitions for model predictions
SELECT create_monthly_partitions('ml_model_predictions', '2024-01-01'::DATE, 36);

-- Default partition for out-of-range or missing monthly partitions
CREATE TABLE IF NOT EXISTS ml_model_predictions_default
    PARTITION OF ml_model_predictions DEFAULT;

-- Indexes for model predictions
CREATE INDEX IF NOT EXISTS idx_ml_model_predictions_lookup
    ON ml_model_predictions (model_id, instrument_id, ts_event);
CREATE INDEX IF NOT EXISTS idx_ml_model_predictions_live
    ON ml_model_predictions (is_live) WHERE is_live = TRUE;

-- ============================================================================
-- Strategy Store Tables
-- ============================================================================

-- Strategy signals table (partitioned by time)
CREATE TABLE IF NOT EXISTS ml_strategy_signals (
    strategy_id VARCHAR(255) NOT NULL,
    instrument_id VARCHAR(100) NOT NULL,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,
    signal_type VARCHAR(20) NOT NULL,  -- BUY, SELL, HOLD
    strength FLOAT NOT NULL,
    model_predictions JSONB,  -- Model ID -> prediction mapping
    risk_metrics JSONB,  -- Risk calculations
    execution_params JSONB,  -- Stop loss, take profit, etc.
    is_live BOOLEAN DEFAULT FALSE,
    created_at BIGINT,

    PRIMARY KEY (strategy_id, instrument_id, ts_event)
) PARTITION BY RANGE (ts_event);

-- Create partitions for strategy signals
SELECT create_monthly_partitions('ml_strategy_signals', '2024-01-01'::DATE, 36);

-- Default partition for out-of-range or missing monthly partitions
CREATE TABLE IF NOT EXISTS ml_strategy_signals_default
    PARTITION OF ml_strategy_signals DEFAULT;

-- Indexes for strategy signals
CREATE INDEX IF NOT EXISTS idx_ml_strategy_signals_lookup
    ON ml_strategy_signals (strategy_id, instrument_id, ts_event);
CREATE INDEX IF NOT EXISTS idx_ml_strategy_signals_type
    ON ml_strategy_signals (signal_type);
CREATE INDEX IF NOT EXISTS idx_ml_strategy_signals_live
    ON ml_strategy_signals (is_live) WHERE is_live = TRUE;

-- Strategy performance tracking table
CREATE TABLE IF NOT EXISTS ml_strategy_performance (
    strategy_id VARCHAR(255) NOT NULL,
    period_start BIGINT NOT NULL,
    period_end BIGINT,
    signal_count BIGINT,
    buy_count BIGINT,
    sell_count BIGINT,
    hold_count BIGINT,
    avg_strength FLOAT,
    avg_risk_score FLOAT,
    created_at BIGINT,

    PRIMARY KEY (strategy_id, period_start)
);

CREATE INDEX IF NOT EXISTS idx_ml_strategy_performance
    ON ml_strategy_performance (strategy_id, period_start);

-- ============================================================================
-- Usage Examples
-- ============================================================================

-- To create additional partitions for future months:
-- SELECT create_monthly_partitions('ml_feature_values', '2027-01-01'::DATE, 12);
-- SELECT create_monthly_partitions('ml_model_predictions', '2027-01-01'::DATE, 12);
-- SELECT create_monthly_partitions('ml_strategy_signals', '2027-01-01'::DATE, 12);
