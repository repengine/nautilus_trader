-- ============================================================================
-- Nautilus ML Bootstrap Schema
-- Consolidated from fragmented migration history (greenfield deployment)
-- ============================================================================
-- Created: 2025-10-01
-- Consolidates: 18 previous migrations into single bootstrap
-- Notes: No production data existed - clean slate deployment
-- ============================================================================

-- Ensure store objects land in the public schema by default.
SET search_path TO public, pg_catalog, ml_registry;

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
    model_id VARCHAR(255) NOT NULL,
    instrument_id VARCHAR(100) NOT NULL,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,
    prediction DOUBLE PRECISION NOT NULL,
    confidence DOUBLE PRECISION,
    features_used JSONB,
    inference_time_ms DOUBLE PRECISION,
    is_live BOOLEAN DEFAULT FALSE,
    created_at BIGINT,
    PRIMARY KEY (model_id, instrument_id, ts_event)
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
    strength DOUBLE PRECISION NOT NULL,
    model_predictions JSONB,
    risk_metrics JSONB,
    execution_params JSONB,
    is_live BOOLEAN DEFAULT FALSE,
    created_at BIGINT,
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
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    dataset_type VARCHAR(50) NOT NULL,
    storage_kind VARCHAR(20) NOT NULL,
    location TEXT NOT NULL,
    partitioning JSONB,
    retention_days INTEGER NOT NULL,
    schema JSONB NOT NULL,
    schema_hash VARCHAR(64) NOT NULL,
    constraints JSONB,
    parents JSONB,
    pipeline_signature VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_modified TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB,
    CONSTRAINT check_dataset_type CHECK (
        dataset_type IN (
            'BARS',
            'TRADES',
            'QUOTES',
            'MBP1',
            'TBBO',
            'FEATURES',
            'PREDICTIONS',
            'SIGNALS',
            'EARNINGS_ACTUALS',
            'EARNINGS_ESTIMATES'
        )
    ),
    CONSTRAINT check_storage_kind CHECK (storage_kind IN ('parquet', 'postgres')),
    CONSTRAINT check_retention_positive CHECK (retention_days > 0)
);

CREATE TABLE IF NOT EXISTS ml_data_events (
    event_id BIGSERIAL,
    dataset_id VARCHAR(255) NOT NULL,
    instrument_id VARCHAR(100) NOT NULL,
    stage VARCHAR(50) NOT NULL,
    source VARCHAR(50) NOT NULL,
    run_id VARCHAR(255),
    ts_min BIGINT NOT NULL,
    ts_max BIGINT NOT NULL,
    ts_event BIGINT NOT NULL,
    count BIGINT NOT NULL,
    seq_min BIGINT,
    seq_max BIGINT,
    status VARCHAR(20) NOT NULL,
    error TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (event_id, ts_event),
    CONSTRAINT check_stage CHECK (
        stage IN (
            'INGESTED',
            'CATALOG_WRITTEN',
            'FEATURE_COMPUTED',
            'PREDICTION_EMITTED',
            'SIGNAL_EMITTED',
            'MODEL_INFERRED'
        )
    ),
    CONSTRAINT check_source CHECK (
        source IN ('live', 'historical', 'backfill', 'batch')
    ),
    CONSTRAINT check_status CHECK (status IN ('success', 'failed', 'partial')),
    CONSTRAINT check_time_range CHECK (ts_min <= ts_max),
    CONSTRAINT check_seq_range CHECK (
        seq_min IS NULL OR seq_max IS NULL OR seq_min <= seq_max
    )
) PARTITION BY RANGE (ts_event);

CREATE TABLE IF NOT EXISTS ml_data_events_default PARTITION OF ml_data_events DEFAULT;
CREATE INDEX IF NOT EXISTS idx_ml_data_events_time ON ml_data_events USING BRIN (ts_event);
CREATE INDEX IF NOT EXISTS idx_ml_data_events_lookup ON ml_data_events (dataset_id, instrument_id, ts_event DESC);
CREATE INDEX IF NOT EXISTS idx_ml_data_events_stage ON ml_data_events (stage, ts_event DESC);

CREATE TABLE IF NOT EXISTS ml_data_watermarks (
    dataset_id VARCHAR(255) NOT NULL,
    instrument_id VARCHAR(100) NOT NULL,
    source VARCHAR(50) NOT NULL,
    last_success_ns BIGINT,
    last_attempt_ns BIGINT,
    last_count BIGINT DEFAULT 0,
    completeness_pct NUMERIC,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (dataset_id, instrument_id, source),
    CONSTRAINT check_source_watermark CHECK (source IN ('live', 'historical', 'backfill')),
    CONSTRAINT check_completeness CHECK (
        completeness_pct IS NULL OR (completeness_pct >= 0 AND completeness_pct <= 100)
    ),
    CONSTRAINT fk_watermark_dataset FOREIGN KEY (dataset_id)
        REFERENCES ml_dataset_registry(dataset_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS ml_data_lineage (
    lineage_id BIGSERIAL PRIMARY KEY,
    transform_id VARCHAR(255) NOT NULL,
    child_dataset_id VARCHAR(255) NOT NULL,
    parent_dataset_id VARCHAR(255) NOT NULL,
    ts_range JSONB,
    parameters JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT fk_lineage_child FOREIGN KEY (child_dataset_id)
        REFERENCES ml_dataset_registry(dataset_id) ON DELETE CASCADE,
    CONSTRAINT fk_lineage_parent FOREIGN KEY (parent_dataset_id)
        REFERENCES ml_dataset_registry(dataset_id) ON DELETE CASCADE
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
CREATE INDEX IF NOT EXISTS idx_ml_data_lineage_child ON ml_data_lineage (child_dataset_id);
CREATE INDEX IF NOT EXISTS idx_ml_data_lineage_parent ON ml_data_lineage (parent_dataset_id);
CREATE INDEX IF NOT EXISTS idx_ml_data_lineage_transform ON ml_data_lineage (transform_id);

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
            p_ts_max,
            p_ts_max,
            p_count,
            100.0,
            NOW()
        )
        ON CONFLICT (dataset_id, instrument_id, source)
        DO UPDATE SET
            last_success_ns = GREATEST(
                COALESCE(ml_data_watermarks.last_success_ns, 0),
                EXCLUDED.last_success_ns
            ),
            last_attempt_ns = GREATEST(
                COALESCE(ml_data_watermarks.last_attempt_ns, 0),
                EXCLUDED.last_attempt_ns
            ),
            last_count = EXCLUDED.last_count,
            completeness_pct = COALESCE(EXCLUDED.completeness_pct, ml_data_watermarks.completeness_pct),
            updated_at = NOW();
    ELSE
        INSERT INTO ml_data_watermarks (
            dataset_id,
            instrument_id,
            source,
            last_attempt_ns,
            updated_at
        )
        VALUES (
            p_dataset_id,
            p_instrument_id,
            p_source,
            p_ts_max,
            NOW()
        )
        ON CONFLICT (dataset_id, instrument_id, source)
        DO UPDATE SET
            last_attempt_ns = GREATEST(
                COALESCE(ml_data_watermarks.last_attempt_ns, 0),
                EXCLUDED.last_attempt_ns
            ),
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
    p_last_success_ns BIGINT,
    p_count BIGINT,
    p_completeness_pct NUMERIC DEFAULT NULL
)
RETURNS VOID AS $$
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
        last_success_ns = GREATEST(
            COALESCE(ml_data_watermarks.last_success_ns, 0),
            EXCLUDED.last_success_ns
        ),
        last_attempt_ns = GREATEST(
            COALESCE(ml_data_watermarks.last_attempt_ns, 0),
            EXCLUDED.last_attempt_ns
        ),
        last_count = EXCLUDED.last_count,
        completeness_pct = COALESCE(EXCLUDED.completeness_pct, ml_data_watermarks.completeness_pct),
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
