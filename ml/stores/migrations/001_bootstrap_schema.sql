-- ============================================================================
-- Nautilus ML Bootstrap Schema
-- Consolidated from fragmented migration history (greenfield deployment)
-- ============================================================================
-- Created: 2025-10-01
-- Consolidates: 18 previous migrations into single bootstrap
-- Notes: No production data existed - clean slate deployment
-- ============================================================================

-- ============================================================================
-- Helper Functions
-- ============================================================================

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

-- Feature values (partitioned by ts_event)
CREATE TABLE IF NOT EXISTS ml_feature_values (
    id BIGSERIAL,
    feature_set_id VARCHAR(255) NOT NULL,
    instrument_id VARCHAR(100) NOT NULL,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,
    values JSONB NOT NULL,
    is_live BOOLEAN DEFAULT FALSE,
    source VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (feature_set_id, instrument_id, ts_event)
) PARTITION BY RANGE (ts_event);

CREATE TABLE IF NOT EXISTS ml_feature_values_default PARTITION OF ml_feature_values DEFAULT;

-- Feature computation stats
CREATE TABLE IF NOT EXISTS ml_feature_computation_stats (
    computation_id BIGSERIAL PRIMARY KEY,
    feature_set_id VARCHAR(255) NOT NULL,
    instrument_id VARCHAR(100) NOT NULL,
    ts_start BIGINT NOT NULL,
    ts_end BIGINT NOT NULL,
    computation_time_ms DOUBLE PRECISION,
    feature_count INTEGER,
    error_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Feature lineage tracking
CREATE TABLE IF NOT EXISTS ml_feature_lineage (
    lineage_id BIGSERIAL PRIMARY KEY,
    feature_set_id VARCHAR(255) NOT NULL,
    parent_dataset_id VARCHAR(255),
    transform_type VARCHAR(100),
    transform_params JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- Model Store Tables
-- ============================================================================

-- Model predictions (partitioned by ts_event)
CREATE TABLE IF NOT EXISTS ml_model_predictions (
    prediction_id BIGSERIAL,
    model_name VARCHAR(255) NOT NULL,
    instrument_id VARCHAR(100) NOT NULL,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,
    prediction DOUBLE PRECISION NOT NULL,
    confidence DOUBLE PRECISION,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (model_name, instrument_id, ts_event)
) PARTITION BY RANGE (ts_event);

CREATE TABLE IF NOT EXISTS ml_model_predictions_default PARTITION OF ml_model_predictions DEFAULT;

-- ============================================================================
-- Strategy Store Tables
-- ============================================================================

-- Strategy signals (partitioned by ts_event)
CREATE TABLE IF NOT EXISTS ml_strategy_signals (
    signal_id BIGSERIAL,
    strategy_id VARCHAR(255) NOT NULL,
    instrument_id VARCHAR(100) NOT NULL,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,
    signal_type VARCHAR(50) NOT NULL,
    signal_value DOUBLE PRECISION,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (strategy_id, instrument_id, ts_event)
) PARTITION BY RANGE (ts_event);

CREATE TABLE IF NOT EXISTS ml_strategy_signals_default PARTITION OF ml_strategy_signals DEFAULT;

-- ============================================================================
-- Market Data Tables
-- ============================================================================

CREATE TABLE IF NOT EXISTS market_data (
    instrument_id VARCHAR(100) NOT NULL,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,

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
    vwap DECIMAL(20,10),
    spread DECIMAL(20,10) GENERATED ALWAYS AS (ask - bid) STORED,
    mid_price DECIMAL(20,10) GENERATED ALWAYS AS ((bid + ask) / 2) STORED,

    -- Metadata
    source VARCHAR(50),
    quality_flags INTEGER DEFAULT 0,
    source_dataset VARCHAR(100),  -- Provenance tracking

    PRIMARY KEY (instrument_id, ts_event)
) PARTITION BY RANGE (ts_event);

CREATE TABLE IF NOT EXISTS market_data_default PARTITION OF market_data DEFAULT;

-- Market data metadata
CREATE TABLE IF NOT EXISTS market_data_metadata (
    instrument_id VARCHAR(100) PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL,
    exchange VARCHAR(50),
    asset_class VARCHAR(50),
    tick_size DECIMAL(10,8),
    lot_size DECIMAL(10,4),
    min_volume DECIMAL(10,4),
    max_volume DECIMAL(10,4),
    contract_size DECIMAL(10,4),
    contract_multiplier DECIMAL(10,4),
    expiry_date DATE,
    currency VARCHAR(10),
    timezone VARCHAR(50),
    trading_hours JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    is_tradable BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB
);

-- Market data statistics
CREATE TABLE IF NOT EXISTS market_data_statistics (
    instrument_id VARCHAR(100) NOT NULL,
    period_start BIGINT NOT NULL,
    period_end BIGINT NOT NULL,
    mean_price DECIMAL(20,10),
    std_price DECIMAL(20,10),
    min_price DECIMAL(20,10),
    max_price DECIMAL(20,10),
    mean_volume DECIMAL(20,10),
    std_volume DECIMAL(20,10),
    total_volume DECIMAL(20,10),
    mean_spread DECIMAL(20,10),
    std_spread DECIMAL(20,10),
    total_records BIGINT,
    clean_records BIGINT,
    outlier_count BIGINT,
    missing_count BIGINT,
    PRIMARY KEY (instrument_id, period_start)
);

-- Positions tracking
CREATE TABLE IF NOT EXISTS ml_positions (
    strategy_id VARCHAR(255) NOT NULL,
    instrument_id VARCHAR(100) NOT NULL,
    quantity DECIMAL(20,10) NOT NULL,
    side VARCHAR(10) NOT NULL,
    entry_price DECIMAL(20,10),
    current_price DECIMAL(20,10),
    unrealized_pnl DECIMAL(20,10),
    realized_pnl DECIMAL(20,10),
    position_value DECIMAL(20,10),
    exposure DECIMAL(20,10),
    var_95 DECIMAL(20,10),
    entry_time BIGINT NOT NULL,
    last_update BIGINT NOT NULL,
    PRIMARY KEY (strategy_id, instrument_id)
);

-- Risk limits
CREATE TABLE IF NOT EXISTS ml_risk_limits (
    strategy_id VARCHAR(255) PRIMARY KEY,
    max_exposure DECIMAL(20,10),
    max_position_size DECIMAL(20,10),
    max_positions INTEGER,
    max_drawdown DECIMAL(10,4),
    max_var DECIMAL(20,10),
    max_leverage DECIMAL(10,2),
    max_daily_trades INTEGER,
    max_order_size DECIMAL(20,10),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- Data Registry & Event Tracking
-- ============================================================================

CREATE TABLE IF NOT EXISTS ml_dataset_registry (
    dataset_id VARCHAR(255) PRIMARY KEY,
    dataset_type VARCHAR(50) NOT NULL,
    schema_version VARCHAR(50),
    storage_kind VARCHAR(50),
    catalog_uri TEXT,
    table_name VARCHAR(255),
    instrument_ids TEXT[],
    ts_event_start BIGINT,
    ts_event_end BIGINT,
    feature_names TEXT[],
    feature_schema_hash VARCHAR(64),
    parent_dataset_ids TEXT[],
    tags JSONB,
    status VARCHAR(50) DEFAULT 'registered',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS ml_data_events (
    event_id BIGSERIAL PRIMARY KEY,
    dataset_id VARCHAR(255) NOT NULL,
    instrument_id VARCHAR(100),
    stage VARCHAR(50) NOT NULL,
    source VARCHAR(50) NOT NULL,
    run_id VARCHAR(255),
    ts_min BIGINT,
    ts_max BIGINT,
    ts_event BIGINT NOT NULL,
    count BIGINT DEFAULT 0,
    seq_min BIGINT,
    seq_max BIGINT,
    status VARCHAR(20) NOT NULL,
    error TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ml_data_watermarks (
    watermark_id BIGSERIAL PRIMARY KEY,
    dataset_id VARCHAR(255) NOT NULL,
    instrument_id VARCHAR(100),
    source VARCHAR(50) NOT NULL,
    ts_max BIGINT NOT NULL,
    count BIGINT DEFAULT 0,
    last_seq BIGINT,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (dataset_id, instrument_id, source)
);

CREATE TABLE IF NOT EXISTS ml_data_lineage (
    lineage_id BIGSERIAL PRIMARY KEY,
    dataset_id VARCHAR(255) NOT NULL,
    parent_dataset_id VARCHAR(255) NOT NULL,
    lineage_type VARCHAR(50) NOT NULL,
    transform_metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- Indexes
-- ============================================================================

-- BRIN indexes for time-range efficiency on partitioned tables
CREATE INDEX IF NOT EXISTS brin_ml_feature_values_ts ON ml_feature_values USING BRIN (ts_event);
CREATE INDEX IF NOT EXISTS brin_ml_model_predictions_ts ON ml_model_predictions USING BRIN (ts_event);
CREATE INDEX IF NOT EXISTS brin_ml_strategy_signals_ts ON ml_strategy_signals USING BRIN (ts_event);
CREATE INDEX IF NOT EXISTS brin_market_data_ts ON market_data USING BRIN (ts_event);

-- Standard indexes
CREATE INDEX IF NOT EXISTS idx_market_data_instrument ON market_data (instrument_id, ts_event DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_quality ON market_data (quality_flags) WHERE quality_flags > 0;
CREATE INDEX IF NOT EXISTS idx_metadata_symbol ON market_data_metadata (symbol);
CREATE INDEX IF NOT EXISTS idx_metadata_exchange ON market_data_metadata (exchange);
CREATE INDEX IF NOT EXISTS idx_metadata_asset_class ON market_data_metadata (asset_class);
CREATE INDEX IF NOT EXISTS idx_metadata_active ON market_data_metadata (is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_stats_lookup ON market_data_statistics (instrument_id, period_start DESC);
CREATE INDEX IF NOT EXISTS idx_positions_strategy ON ml_positions (strategy_id);
CREATE INDEX IF NOT EXISTS idx_positions_instrument ON ml_positions (instrument_id);
CREATE INDEX IF NOT EXISTS idx_data_events_run ON ml_data_events (run_id);
CREATE INDEX IF NOT EXISTS idx_data_events_stage ON ml_data_events (stage, dataset_id);
CREATE INDEX IF NOT EXISTS idx_data_events_status ON ml_data_events (status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_watermarks_lookup ON ml_data_watermarks (dataset_id, instrument_id, source);
CREATE INDEX IF NOT EXISTS idx_lineage_dataset ON ml_data_lineage (dataset_id);
CREATE INDEX IF NOT EXISTS idx_lineage_parent ON ml_data_lineage (parent_dataset_id);

-- ============================================================================
-- Helper Functions for Event Emission
-- ============================================================================

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
    p_metadata JSONB DEFAULT NULL,
    p_seq_min BIGINT DEFAULT NULL,
    p_seq_max BIGINT DEFAULT NULL
)
RETURNS BIGINT AS $$
DECLARE
    v_event_id BIGINT;
    v_ts_event BIGINT;
BEGIN
    v_ts_event := EXTRACT(EPOCH FROM NOW()) * 1000000000;

    INSERT INTO ml_data_events (
        dataset_id, instrument_id, stage, source, run_id,
        ts_min, ts_max, ts_event, count, seq_min, seq_max,
        status, error, metadata, created_at
    )
    VALUES (
        p_dataset_id, p_instrument_id, p_stage, p_source, p_run_id,
        p_ts_min, p_ts_max, v_ts_event, p_count, p_seq_min, p_seq_max,
        p_status, p_error, p_metadata, NOW()
    )
    RETURNING event_id INTO v_event_id;

    IF p_status = 'success' THEN
        INSERT INTO ml_data_watermarks (
            dataset_id, instrument_id, source, ts_max, count, updated_at
        )
        VALUES (
            p_dataset_id, p_instrument_id, p_source, p_ts_max, p_count, NOW()
        )
        ON CONFLICT (dataset_id, instrument_id, source)
        DO UPDATE SET
            ts_max = GREATEST(ml_data_watermarks.ts_max, EXCLUDED.ts_max),
            count = ml_data_watermarks.count + EXCLUDED.count,
            updated_at = NOW();
    END IF;

    RETURN v_event_id;
END;
$$ LANGUAGE plpgsql;

-- Simpler emit function for backward compatibility
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
    p_error TEXT DEFAULT NULL
)
RETURNS BIGINT AS $$
BEGIN
    RETURN emit_data_event_ext(
        p_dataset_id, p_instrument_id, p_stage, p_source, p_run_id,
        p_ts_min, p_ts_max, p_count, p_status, p_error, NULL, NULL, NULL
    );
END;
$$ LANGUAGE plpgsql;

-- Update watermark helper
CREATE OR REPLACE FUNCTION update_watermark(
    p_dataset_id VARCHAR(255),
    p_instrument_id VARCHAR(100),
    p_source VARCHAR(50),
    p_ts_max BIGINT,
    p_count BIGINT,
    p_last_seq BIGINT DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO ml_data_watermarks (
        dataset_id, instrument_id, source, ts_max, count, last_seq, updated_at
    )
    VALUES (
        p_dataset_id, p_instrument_id, p_source, p_ts_max, p_count, p_last_seq, NOW()
    )
    ON CONFLICT (dataset_id, instrument_id, source)
    DO UPDATE SET
        ts_max = GREATEST(ml_data_watermarks.ts_max, EXCLUDED.ts_max),
        count = ml_data_watermarks.count + EXCLUDED.count,
        last_seq = COALESCE(EXCLUDED.last_seq, ml_data_watermarks.last_seq),
        updated_at = NOW();
END;
$$ LANGUAGE plpgsql;

-- Risk limit check helper
CREATE OR REPLACE FUNCTION check_risk_limits(
    p_strategy_id VARCHAR,
    p_new_exposure DECIMAL
)
RETURNS BOOLEAN AS $$
DECLARE
    current_exp DECIMAL;
    max_exp DECIMAL;
BEGIN
    SELECT COALESCE(SUM(ABS(exposure)), 0)
    INTO current_exp
    FROM ml_positions
    WHERE strategy_id = p_strategy_id;

    SELECT max_exposure INTO max_exp
    FROM ml_risk_limits
    WHERE strategy_id = p_strategy_id AND is_active = TRUE;

    RETURN (current_exp + p_new_exposure) <= COALESCE(max_exp, 999999999);
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Views for Common Queries
-- ============================================================================

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
    mid_price,
    source_dataset
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
-- Pre-create Partitions (2023-2027 for testing + initial production)
-- ============================================================================

SELECT create_monthly_partitions('ml_feature_values', '2023-01-01'::DATE, 60);
SELECT create_monthly_partitions('ml_model_predictions', '2023-01-01'::DATE, 60);
SELECT create_monthly_partitions('ml_strategy_signals', '2023-01-01'::DATE, 60);
SELECT create_monthly_partitions('market_data', '2023-01-01'::DATE, 60);

-- ============================================================================
-- Verification
-- ============================================================================

-- Verify partitions exist
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_tables
        WHERE tablename LIKE 'ml_feature_values_2023%'
    ) THEN
        RAISE NOTICE 'Bootstrap schema applied successfully - partitions created';
    ELSE
        RAISE WARNING 'Partition creation may have failed';
    END IF;
END$$;
