-- Market Data Schema Migration
-- Creates tables for market data storage and metadata
-- This integrates with the ML processing pipeline

-- ============================================================================
-- Market Data Tables
-- ============================================================================

-- Market data table (partitioned by time)
CREATE TABLE IF NOT EXISTS market_data (
    instrument_id VARCHAR(100) NOT NULL,
    ts_event BIGINT NOT NULL,  -- Event timestamp (nanoseconds)
    ts_init BIGINT NOT NULL,   -- Initialization timestamp (nanoseconds)

    -- Price data
    bid DECIMAL(20,10),
    ask DECIMAL(20,10),
    bid_size DECIMAL(20,10),
    ask_size DECIMAL(20,10),
    last DECIMAL(20,10),

    -- OHLCV data
    open DECIMAL(20,10),
    high DECIMAL(20,10),
    low DECIMAL(20,10),
    close DECIMAL(20,10),
    volume DECIMAL(20,10),

    -- Additional fields
    trade_count INTEGER,
    vwap DECIMAL(20,10),  -- Volume-weighted average price
    spread DECIMAL(20,10) GENERATED ALWAYS AS (ask - bid) STORED,
    mid_price DECIMAL(20,10) GENERATED ALWAYS AS ((bid + ask) / 2) STORED,

    -- Metadata
    source VARCHAR(50),  -- 'exchange', 'aggregator', 'synthetic'
    quality_flags INTEGER DEFAULT 0,  -- Bit flags for data quality
    source_dataset VARCHAR(100),

    PRIMARY KEY (instrument_id, ts_event)
) PARTITION BY RANGE (ts_event);

-- Pre-create partitions covering historical ingest windows (10 years)
SELECT create_monthly_partitions('market_data', '2018-01-01'::DATE, 120);

-- Default partition to capture out-of-range rows during fallback ingest
CREATE TABLE IF NOT EXISTS market_data_default
    PARTITION OF market_data DEFAULT;

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_market_data_time ON market_data USING BRIN (ts_event);
CREATE INDEX IF NOT EXISTS idx_market_data_instrument ON market_data (instrument_id, ts_event DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_quality ON market_data (quality_flags) WHERE quality_flags > 0;

-- ============================================================================
-- Market Data Metadata
-- ============================================================================

-- Instrument metadata table
CREATE TABLE IF NOT EXISTS market_data_metadata (
    instrument_id VARCHAR(100) PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL,
    exchange VARCHAR(50),
    asset_class VARCHAR(50),  -- 'equity', 'forex', 'crypto', 'commodity', 'bond'

    -- Trading specifications
    tick_size DECIMAL(10,8),
    lot_size DECIMAL(10,4),
    min_volume DECIMAL(10,4),
    max_volume DECIMAL(10,4),

    -- Contract specifications (for derivatives)
    contract_size DECIMAL(10,4),
    contract_multiplier DECIMAL(10,4),
    expiry_date DATE,

    -- Reference data
    currency VARCHAR(10),
    timezone VARCHAR(50),
    trading_hours JSONB,  -- Trading session times

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    is_tradable BOOLEAN DEFAULT TRUE,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB  -- Additional flexible metadata
);

-- Indexes for metadata
CREATE INDEX IF NOT EXISTS idx_metadata_symbol ON market_data_metadata (symbol);
CREATE INDEX IF NOT EXISTS idx_metadata_exchange ON market_data_metadata (exchange);
CREATE INDEX IF NOT EXISTS idx_metadata_asset_class ON market_data_metadata (asset_class);
CREATE INDEX IF NOT EXISTS idx_metadata_active ON market_data_metadata (is_active) WHERE is_active = TRUE;

-- ============================================================================
-- Market Data Statistics (for outlier detection)
-- ============================================================================

-- Rolling statistics for data quality checks
CREATE TABLE IF NOT EXISTS market_data_statistics (
    instrument_id VARCHAR(100) NOT NULL,
    period_start BIGINT NOT NULL,
    period_end BIGINT NOT NULL,

    -- Price statistics
    mean_price DECIMAL(20,10),
    std_price DECIMAL(20,10),
    min_price DECIMAL(20,10),
    max_price DECIMAL(20,10),

    -- Volume statistics
    mean_volume DECIMAL(20,10),
    std_volume DECIMAL(20,10),
    total_volume DECIMAL(20,10),

    -- Spread statistics
    mean_spread DECIMAL(20,10),
    std_spread DECIMAL(20,10),

    -- Data quality
    total_records BIGINT,
    clean_records BIGINT,
    outlier_count BIGINT,
    missing_count BIGINT,

    PRIMARY KEY (instrument_id, period_start)
);

CREATE INDEX IF NOT EXISTS idx_stats_lookup ON market_data_statistics (instrument_id, period_start DESC);

-- ============================================================================
-- Position and Risk Tracking (referenced by DataProcessor)
-- ============================================================================

-- Current positions table
CREATE TABLE IF NOT EXISTS ml_positions (
    strategy_id VARCHAR(255) NOT NULL,
    instrument_id VARCHAR(100) NOT NULL,

    -- Position details
    quantity DECIMAL(20,10) NOT NULL,
    side VARCHAR(10) NOT NULL,  -- 'LONG', 'SHORT'
    entry_price DECIMAL(20,10),
    current_price DECIMAL(20,10),

    -- P&L
    unrealized_pnl DECIMAL(20,10),
    realized_pnl DECIMAL(20,10),

    -- Risk metrics
    position_value DECIMAL(20,10),
    exposure DECIMAL(20,10),
    var_95 DECIMAL(20,10),  -- Value at Risk

    -- Timestamps
    entry_time BIGINT NOT NULL,
    last_update BIGINT NOT NULL,

    PRIMARY KEY (strategy_id, instrument_id)
);

CREATE INDEX IF NOT EXISTS idx_positions_strategy ON ml_positions (strategy_id);
CREATE INDEX IF NOT EXISTS idx_positions_instrument ON ml_positions (instrument_id);

-- Risk limits configuration
CREATE TABLE IF NOT EXISTS ml_risk_limits (
    strategy_id VARCHAR(255) PRIMARY KEY,

    -- Exposure limits
    max_exposure DECIMAL(20,10),
    max_position_size DECIMAL(20,10),
    max_positions INTEGER,

    -- Risk limits
    max_drawdown DECIMAL(10,4),
    max_var DECIMAL(20,10),
    max_leverage DECIMAL(10,2),

    -- Trading limits
    max_daily_trades INTEGER,
    max_order_size DECIMAL(20,10),

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- Helper Functions
-- ============================================================================

-- Function to calculate current exposure
CREATE OR REPLACE FUNCTION get_current_exposure(
    p_strategy_id VARCHAR,
    p_instrument_id VARCHAR DEFAULT NULL
)
RETURNS DECIMAL AS $$
DECLARE
    total_exposure DECIMAL;
BEGIN
    SELECT COALESCE(SUM(ABS(exposure)), 0)
    INTO total_exposure
    FROM ml_positions
    WHERE strategy_id = p_strategy_id
    AND (p_instrument_id IS NULL OR instrument_id = p_instrument_id);

    RETURN total_exposure;
END;
$$ LANGUAGE plpgsql;

-- Function to check risk limits
CREATE OR REPLACE FUNCTION check_risk_limits(
    p_strategy_id VARCHAR,
    p_new_exposure DECIMAL
)
RETURNS BOOLEAN AS $$
DECLARE
    current_exp DECIMAL;
    max_exp DECIMAL;
BEGIN
    -- Get current exposure
    current_exp := get_current_exposure(p_strategy_id);

    -- Get max exposure limit
    SELECT max_exposure INTO max_exp
    FROM ml_risk_limits
    WHERE strategy_id = p_strategy_id
    AND is_active = TRUE;

    -- Check if new exposure would exceed limit
    RETURN (current_exp + p_new_exposure) <= COALESCE(max_exp, 999999999);
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Triggers for Data Quality
-- ============================================================================

-- Trigger to update statistics on market data insert
CREATE OR REPLACE FUNCTION update_market_statistics()
RETURNS TRIGGER AS $$
BEGIN
    -- Update rolling statistics (simplified version)
    -- In production, this would be done asynchronously
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS market_data_stats_trigger ON market_data;
CREATE TRIGGER market_data_stats_trigger
    AFTER INSERT ON market_data
    FOR EACH STATEMENT
    EXECUTE FUNCTION update_market_statistics();

-- ============================================================================
-- Views for Common Queries
-- ============================================================================

-- Latest market data view
DROP VIEW IF EXISTS latest_market_data CASCADE;
CREATE VIEW latest_market_data AS
SELECT DISTINCT ON (instrument_id)
    instrument_id,
    ts_event,
    bid,
    ask,
    last,
    volume,
    spread,
    mid_price
FROM market_data
ORDER BY instrument_id, ts_event DESC;

DROP VIEW IF EXISTS position_summary CASCADE;
CREATE VIEW position_summary AS
SELECT
    p.strategy_id,
    COUNT(*) as position_count,
    SUM(ABS(p.exposure)) as total_exposure,
    SUM(p.unrealized_pnl) as total_unrealized_pnl,
    SUM(p.realized_pnl) as total_realized_pnl,
    MAX(p.last_update) as last_update
FROM ml_positions p
GROUP BY p.strategy_id;

-- ============================================================================
-- Maintenance Functions
-- ============================================================================

-- Function to clean old market data
CREATE OR REPLACE FUNCTION cleanup_old_market_data(
    retention_days INTEGER DEFAULT 90
)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
    cutoff_ns BIGINT;
BEGIN
    -- Calculate cutoff timestamp in nanoseconds
    cutoff_ns := EXTRACT(EPOCH FROM (NOW() - (retention_days || ' days')::INTERVAL)) * 1000000000;

    -- Delete old data
    DELETE FROM market_data
    WHERE ts_event < cutoff_ns;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Permissions (adjust as needed)
-- ============================================================================

-- Grant appropriate permissions
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO ml_reader;
-- GRANT INSERT, UPDATE ON market_data, ml_positions TO ml_writer;
-- GRANT ALL ON ALL TABLES IN SCHEMA public TO ml_admin;
