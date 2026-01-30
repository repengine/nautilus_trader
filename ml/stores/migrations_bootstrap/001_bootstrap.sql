-- ============================================================================
-- Nautilus ML Stores Bootstrap Schema
-- Consolidated canonical schema for greenfield databases.
-- ============================================================================
-- Created: 2026-01-22
-- Notes:
-- - Legacy incremental migrations live under ml/stores/migrations_legacy.
-- - This file omits test-only seeds and disabled partition triggers.
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS ml;

-- ============================================================================
-- Partition Helpers
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
    FOR i IN 0..num_months - 1 LOOP
        partition_date := start_date + (i || ' months')::INTERVAL;
        partition_name := table_name || '_' || TO_CHAR(partition_date, 'YYYY_MM');
        start_ns := EXTRACT(EPOCH FROM partition_date) * 1000000000;
        end_ns := EXTRACT(EPOCH FROM partition_date + '1 month'::INTERVAL) * 1000000000;

        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L)',
            partition_name,
            table_name,
            start_ns,
            end_ns
        );
    END LOOP;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION create_event_partitions(
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
    FOR i IN 0..num_months - 1 LOOP
        partition_date := start_date + (i || ' months')::INTERVAL;
        partition_name := table_name || '_' || TO_CHAR(partition_date, 'YYYY_MM');
        start_ns := EXTRACT(EPOCH FROM partition_date) * 1000000000;
        end_ns := EXTRACT(EPOCH FROM partition_date + '1 month'::INTERVAL) * 1000000000;

        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L)',
            partition_name,
            table_name,
            start_ns,
            end_ns
        );
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Optional manual partition maintenance helpers (triggers intentionally omitted)
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
    months_ahead INTEGER := 3;
BEGIN
    cur_date := CURRENT_DATE;

    FOREACH table_name IN ARRAY table_names
    LOOP
        SELECT MAX(TO_DATE(substring(tablename from length(table_name) + 2 for 7), 'YYYY_MM'))
        INTO last_partition_date
        FROM pg_tables
        WHERE schemaname = 'public'
          AND tablename LIKE table_name || '_%';

        IF last_partition_date IS NULL THEN
            last_partition_date := DATE_TRUNC('month', cur_date);
        END IF;

        WHILE last_partition_date <= cur_date + INTERVAL '3 months' LOOP
            partition_name := table_name || '_' || TO_CHAR(last_partition_date, 'YYYY_MM');
            start_ns := EXTRACT(EPOCH FROM last_partition_date) * 1000000000;
            end_ns := EXTRACT(EPOCH FROM last_partition_date + INTERVAL '1 month') * 1000000000;

            IF NOT EXISTS (
                SELECT 1 FROM pg_tables
                WHERE schemaname = 'public'
                  AND tablename = partition_name
            ) THEN
                EXECUTE format(
                    'CREATE TABLE IF NOT EXISTS %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L)',
                    partition_name,
                    table_name,
                    start_ns,
                    end_ns
                );
            END IF;

            last_partition_date := last_partition_date + INTERVAL '1 month';
        END LOOP;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

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
        FOR partition_name IN
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = 'public'
              AND tablename LIKE table_name || '_%'
        LOOP
            BEGIN
                partition_date := TO_DATE(substring(partition_name from length(table_name) + 2 for 7), 'YYYY_MM');
                IF partition_date < cutoff_date THEN
                    EXECUTE format('DROP TABLE IF EXISTS %I CASCADE', partition_name);
                END IF;
            EXCEPTION
                WHEN OTHERS THEN
                    CONTINUE;
            END;
        END LOOP;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Feature Store Tables
-- ============================================================================

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
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (model_id, instrument_id, ts_event)
) PARTITION BY RANGE (ts_event);

CREATE TABLE IF NOT EXISTS ml_model_predictions_default PARTITION OF ml_model_predictions DEFAULT;

-- ============================================================================
-- Strategy Store Tables
-- ============================================================================

CREATE TABLE IF NOT EXISTS ml_strategy_signals (
    signal_id BIGSERIAL,
    strategy_id VARCHAR(255) NOT NULL,
    instrument_id VARCHAR(100) NOT NULL,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,
    run_id VARCHAR(255),
    ingested_at_ns BIGINT,
    signal_type VARCHAR(50) NOT NULL,
    strength DOUBLE PRECISION NOT NULL,
    model_predictions JSONB,
    risk_metrics JSONB,
    execution_params JSONB,
    is_live BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (strategy_id, instrument_id, ts_event)
) PARTITION BY RANGE (ts_event);

CREATE TABLE IF NOT EXISTS ml_strategy_signals_default PARTITION OF ml_strategy_signals DEFAULT;

CREATE TABLE IF NOT EXISTS ml_strategy_performance (
    strategy_id VARCHAR(255) NOT NULL,
    period_start BIGINT NOT NULL,
    period_end BIGINT,
    signal_count BIGINT,
    buy_count BIGINT,
    sell_count BIGINT,
    hold_count BIGINT,
    avg_strength DOUBLE PRECISION,
    avg_risk_score DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (strategy_id, period_start)
);

CREATE TABLE IF NOT EXISTS ml_strategy_risk_halt_events (
    event_id VARCHAR(64) NOT NULL,
    strategy_id VARCHAR(255) NOT NULL,
    instrument_id VARCHAR(100) NOT NULL,
    event_type VARCHAR(32) NOT NULL,
    reason VARCHAR(255) NOT NULL,
    detail TEXT,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT,
    is_live BOOLEAN DEFAULT FALSE,
    run_id VARCHAR(255),
    ingested_at_ns BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (event_id)
);

CREATE TABLE IF NOT EXISTS ml_strategy_replay_summary (
    run_id VARCHAR(255) NOT NULL,
    instrument_ids JSONB,
    started_ns BIGINT,
    finished_ns BIGINT,
    total_orders BIGINT,
    total_fills BIGINT,
    total_halts BIGINT,
    total_sizing_rejects BIGINT,
    total_positions BIGINT,
    ts_event BIGINT,
    ts_init BIGINT,
    ingested_at_ns BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (run_id)
);

-- ============================================================================
-- Market Data Tables
-- ============================================================================

CREATE TABLE IF NOT EXISTS market_data_bar (
    instrument_id VARCHAR(100) NOT NULL,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,
    open DECIMAL(20,10),
    high DECIMAL(20,10),
    low DECIMAL(20,10),
    close DECIMAL(20,10),
    volume DECIMAL(20,10),
    vwap DECIMAL(20,10),
    trade_count INTEGER,
    source VARCHAR(50),
    quality_flags INTEGER DEFAULT 0,
    source_dataset VARCHAR(100),
    PRIMARY KEY (instrument_id, ts_event)
) PARTITION BY RANGE (ts_event);

CREATE TABLE IF NOT EXISTS market_data_bar_default
    PARTITION OF market_data_bar DEFAULT;

CREATE TABLE IF NOT EXISTS market_data_quote_tick (
    instrument_id VARCHAR(100) NOT NULL,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,
    bid DECIMAL(20,10),
    ask DECIMAL(20,10),
    bid_size DECIMAL(20,10),
    ask_size DECIMAL(20,10),
    spread DECIMAL(20,10) GENERATED ALWAYS AS (ask - bid) STORED,
    mid_price DECIMAL(20,10) GENERATED ALWAYS AS ((bid + ask) / 2) STORED,
    source VARCHAR(50),
    quality_flags INTEGER DEFAULT 0,
    source_dataset VARCHAR(100),
    PRIMARY KEY (instrument_id, ts_event)
) PARTITION BY RANGE (ts_event);

CREATE TABLE IF NOT EXISTS market_data_quote_tick_default
    PARTITION OF market_data_quote_tick DEFAULT;

CREATE TABLE IF NOT EXISTS market_data_tbbo (
    instrument_id VARCHAR(100) NOT NULL,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,
    bid DECIMAL(20,10),
    ask DECIMAL(20,10),
    bid_size DECIMAL(20,10),
    ask_size DECIMAL(20,10),
    spread DECIMAL(20,10) GENERATED ALWAYS AS (ask - bid) STORED,
    mid_price DECIMAL(20,10) GENERATED ALWAYS AS ((bid + ask) / 2) STORED,
    source VARCHAR(50),
    quality_flags INTEGER DEFAULT 0,
    source_dataset VARCHAR(100),
    PRIMARY KEY (instrument_id, ts_event)
) PARTITION BY RANGE (ts_event);

CREATE TABLE IF NOT EXISTS market_data_tbbo_default
    PARTITION OF market_data_tbbo DEFAULT;

CREATE TABLE IF NOT EXISTS market_data_mbp1 (
    instrument_id VARCHAR(100) NOT NULL,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,
    bid DECIMAL(20,10),
    ask DECIMAL(20,10),
    bid_size DECIMAL(20,10),
    ask_size DECIMAL(20,10),
    spread DECIMAL(20,10) GENERATED ALWAYS AS (ask - bid) STORED,
    mid_price DECIMAL(20,10) GENERATED ALWAYS AS ((bid + ask) / 2) STORED,
    source VARCHAR(50),
    quality_flags INTEGER DEFAULT 0,
    source_dataset VARCHAR(100),
    PRIMARY KEY (instrument_id, ts_event)
) PARTITION BY RANGE (ts_event);

CREATE TABLE IF NOT EXISTS market_data_mbp1_default
    PARTITION OF market_data_mbp1 DEFAULT;

CREATE TABLE IF NOT EXISTS market_data_trade_tick (
    instrument_id VARCHAR(100) NOT NULL,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,
    last DECIMAL(20,10),
    trade_count INTEGER,
    vwap DECIMAL(20,10),
    volume DECIMAL(20,10),
    source VARCHAR(50),
    quality_flags INTEGER DEFAULT 0,
    source_dataset VARCHAR(100),
    PRIMARY KEY (instrument_id, ts_event)
) PARTITION BY RANGE (ts_event);

CREATE TABLE IF NOT EXISTS market_data_trade_tick_default
    PARTITION OF market_data_trade_tick DEFAULT;

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

CREATE OR REPLACE FUNCTION check_risk_limits(
    p_strategy_id VARCHAR,
    p_new_exposure DECIMAL
)
RETURNS BOOLEAN AS $$
DECLARE
    current_exp DECIMAL;
    max_exp DECIMAL;
BEGIN
    current_exp := get_current_exposure(p_strategy_id);

    SELECT max_exposure INTO max_exp
    FROM ml_risk_limits
    WHERE strategy_id = p_strategy_id AND is_active = TRUE;

    RETURN (current_exp + p_new_exposure) <= COALESCE(max_exp, 999999999);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION update_market_statistics()
RETURNS TRIGGER AS $$
BEGIN
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Data Registry and Event Tracking
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
            'ORDER_EVENTS',
            'RISK_HALT_EVENTS',
            'REPLAY_SUMMARY',
            'EARNINGS_ACTUALS',
            'EARNINGS_ESTIMATES',
            'MACRO_RELEASES',
            'MACRO_OBSERVATIONS',
            'EVENTS_CALENDAR',
            'MICRO_MINUTE_FEATURES',
            'L2_MINUTE_FEATURES'
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
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (event_id, ts_event),
    CONSTRAINT check_stage CHECK (
        stage IN (
            'INGESTED',
            'CATALOG_WRITTEN',
            'FEATURE_COMPUTED',
            'PREDICTION_EMITTED',
            'SIGNAL_EMITTED',
            'MODEL_INFERRED',
            'ORDER_EVENT_EMITTED',
            'RISK_HALT_EMITTED',
            'REPLAY_SUMMARY_EMITTED'
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

CREATE TABLE IF NOT EXISTS ml_data_watermarks (
    dataset_id VARCHAR(255) NOT NULL,
    instrument_id VARCHAR(100) NOT NULL,
    source VARCHAR(50) NOT NULL,
    last_success_ns BIGINT,
    last_attempt_ns BIGINT,
    last_count BIGINT DEFAULT 0,
    completeness_pct DECIMAL(5,2),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (dataset_id, instrument_id, source),
    CONSTRAINT check_source_watermark CHECK (
        source IN ('live', 'historical', 'backfill')
    ),
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

CREATE OR REPLACE FUNCTION update_watermark(
    p_dataset_id VARCHAR(255),
    p_instrument_id VARCHAR(100),
    p_source VARCHAR(50),
    p_last_success_ns BIGINT,
    p_count BIGINT,
    p_completeness_pct DECIMAL(5,2) DEFAULT NULL
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
        last_success_ns = EXCLUDED.last_success_ns,
        last_attempt_ns = EXCLUDED.last_attempt_ns,
        last_count = EXCLUDED.last_count,
        completeness_pct = COALESCE(EXCLUDED.completeness_pct, ml_data_watermarks.completeness_pct),
        updated_at = NOW();
END;
$$ LANGUAGE plpgsql;

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
$$ LANGUAGE plpgsql;

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

CREATE OR REPLACE FUNCTION update_dataset_modified()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_modified = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_dataset_modified ON ml_dataset_registry;
CREATE TRIGGER trigger_update_dataset_modified
    BEFORE UPDATE ON ml_dataset_registry
    FOR EACH ROW
    EXECUTE FUNCTION update_dataset_modified();

-- ============================================================================
-- Feature Family Tables (Schema: ml)
-- ============================================================================

CREATE TABLE IF NOT EXISTS ml.macro_release_calendar (
    series_id VARCHAR(64) NOT NULL,
    observation_ts BIGINT NOT NULL,
    release_ts BIGINT NOT NULL,
    release_end_ts BIGINT,
    value DOUBLE PRECISION,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,
    source VARCHAR(32),
    run_id VARCHAR(64),
    PRIMARY KEY (series_id, observation_ts, release_ts, ts_event)
) PARTITION BY RANGE (ts_event);

CREATE TABLE IF NOT EXISTS ml.macro_release_calendar_default
    PARTITION OF ml.macro_release_calendar DEFAULT;

CREATE TABLE IF NOT EXISTS ml.macro_observations (
    series_id VARCHAR(64) NOT NULL,
    observation_ts BIGINT NOT NULL,
    value DOUBLE PRECISION,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,
    source VARCHAR(32),
    run_id VARCHAR(64),
    PRIMARY KEY (series_id, observation_ts, ts_event)
) PARTITION BY RANGE (ts_event);

CREATE TABLE IF NOT EXISTS ml.macro_observations_default
    PARTITION OF ml.macro_observations DEFAULT;

CREATE TABLE IF NOT EXISTS ml.events_calendar (
    event_timestamp BIGINT NOT NULL,
    event_type VARCHAR(64) NOT NULL,
    name VARCHAR(255) NOT NULL,
    instrument_id VARCHAR(64) NOT NULL,
    importance VARCHAR(32),
    source VARCHAR(64),
    metadata JSONB,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,
    PRIMARY KEY (event_type, event_timestamp, instrument_id, name, ts_event)
) PARTITION BY RANGE (ts_event);

CREATE TABLE IF NOT EXISTS ml.events_calendar_default
    PARTITION OF ml.events_calendar DEFAULT;

CREATE TABLE IF NOT EXISTS ml.microstructure_minute (
    instrument_id VARCHAR(32) NOT NULL,
    "timestamp" BIGINT NOT NULL,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,
    midprice DOUBLE PRECISION,
    spread_bps DOUBLE PRECISION,
    quote_imbalance DOUBLE PRECISION,
    trade_imbalance DOUBLE PRECISION,
    realized_vol DOUBLE PRECISION,
    PRIMARY KEY (instrument_id, "timestamp", ts_event)
) PARTITION BY RANGE (ts_event);

CREATE TABLE IF NOT EXISTS ml.microstructure_minute_default
    PARTITION OF ml.microstructure_minute DEFAULT;

CREATE TABLE IF NOT EXISTS ml.l2_minute (
    instrument_id VARCHAR(32) NOT NULL,
    "timestamp" BIGINT NOT NULL,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,
    midprice DOUBLE PRECISION,
    spread_bps DOUBLE PRECISION,
    microprice_bps DOUBLE PRECISION,
    depth_imbalance_top1 DOUBLE PRECISION,
    depth_imbalance_top3 DOUBLE PRECISION,
    depth_imbalance_top5 DOUBLE PRECISION,
    depth_imbalance_top10 DOUBLE PRECISION,
    dwp_bps_top1 DOUBLE PRECISION,
    dwp_bps_top3 DOUBLE PRECISION,
    dwp_bps_top5 DOUBLE PRECISION,
    dwp_bps_top10 DOUBLE PRECISION,
    bid_slope_top1 DOUBLE PRECISION,
    bid_slope_top3 DOUBLE PRECISION,
    bid_slope_top5 DOUBLE PRECISION,
    bid_slope_top10 DOUBLE PRECISION,
    ask_slope_top1 DOUBLE PRECISION,
    ask_slope_top3 DOUBLE PRECISION,
    ask_slope_top5 DOUBLE PRECISION,
    ask_slope_top10 DOUBLE PRECISION,
    PRIMARY KEY (instrument_id, "timestamp", ts_event)
) PARTITION BY RANGE (ts_event);

CREATE TABLE IF NOT EXISTS ml.l2_minute_default
    PARTITION OF ml.l2_minute DEFAULT;

-- ============================================================================
-- Views and Helper Functions
-- ============================================================================

CREATE OR REPLACE FUNCTION ns_to_timestamp(ns BIGINT)
RETURNS TIMESTAMPTZ AS $$
BEGIN
    RETURN to_timestamp(ns::DOUBLE PRECISION / 1000000000);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

CREATE OR REPLACE FUNCTION timestamp_to_ns(ts TIMESTAMPTZ)
RETURNS BIGINT AS $$
BEGIN
    RETURN (EXTRACT(EPOCH FROM ts) * 1000000000)::BIGINT;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

DROP VIEW IF EXISTS market_data CASCADE;
CREATE VIEW market_data AS
SELECT
    instrument_id,
    ts_event,
    ts_init,
    bid,
    ask,
    bid_size,
    ask_size,
    NULL::DECIMAL(20,10) AS last,
    NULL::DECIMAL(20,10) AS open,
    NULL::DECIMAL(20,10) AS high,
    NULL::DECIMAL(20,10) AS low,
    NULL::DECIMAL(20,10) AS close,
    NULL::DECIMAL(20,10) AS volume,
    NULL::INTEGER AS trade_count,
    NULL::DECIMAL(20,10) AS vwap,
    spread,
    mid_price,
    source,
    quality_flags,
    source_dataset
FROM market_data_tbbo
UNION ALL
SELECT
    instrument_id,
    ts_event,
    ts_init,
    bid,
    ask,
    bid_size,
    ask_size,
    NULL::DECIMAL(20,10) AS last,
    NULL::DECIMAL(20,10) AS open,
    NULL::DECIMAL(20,10) AS high,
    NULL::DECIMAL(20,10) AS low,
    NULL::DECIMAL(20,10) AS close,
    NULL::DECIMAL(20,10) AS volume,
    NULL::INTEGER AS trade_count,
    NULL::DECIMAL(20,10) AS vwap,
    spread,
    mid_price,
    source,
    quality_flags,
    source_dataset
FROM market_data_quote_tick
UNION ALL
SELECT
    instrument_id,
    ts_event,
    ts_init,
    bid,
    ask,
    bid_size,
    ask_size,
    NULL::DECIMAL(20,10) AS last,
    NULL::DECIMAL(20,10) AS open,
    NULL::DECIMAL(20,10) AS high,
    NULL::DECIMAL(20,10) AS low,
    NULL::DECIMAL(20,10) AS close,
    NULL::DECIMAL(20,10) AS volume,
    NULL::INTEGER AS trade_count,
    NULL::DECIMAL(20,10) AS vwap,
    spread,
    mid_price,
    source,
    quality_flags,
    source_dataset
FROM market_data_mbp1
UNION ALL
SELECT
    instrument_id,
    ts_event,
    ts_init,
    NULL::DECIMAL(20,10) AS bid,
    NULL::DECIMAL(20,10) AS ask,
    NULL::DECIMAL(20,10) AS bid_size,
    NULL::DECIMAL(20,10) AS ask_size,
    last,
    NULL::DECIMAL(20,10) AS open,
    NULL::DECIMAL(20,10) AS high,
    NULL::DECIMAL(20,10) AS low,
    NULL::DECIMAL(20,10) AS close,
    volume,
    trade_count,
    vwap,
    NULL::DECIMAL(20,10) AS spread,
    NULL::DECIMAL(20,10) AS mid_price,
    source,
    quality_flags,
    source_dataset
FROM market_data_trade_tick
UNION ALL
SELECT
    instrument_id,
    ts_event,
    ts_init,
    NULL::DECIMAL(20,10) AS bid,
    NULL::DECIMAL(20,10) AS ask,
    NULL::DECIMAL(20,10) AS bid_size,
    NULL::DECIMAL(20,10) AS ask_size,
    NULL::DECIMAL(20,10) AS last,
    open,
    high,
    low,
    close,
    volume,
    trade_count,
    vwap,
    NULL::DECIMAL(20,10) AS spread,
    NULL::DECIMAL(20,10) AS mid_price,
    source,
    quality_flags,
    source_dataset
FROM market_data_bar;

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

CREATE OR REPLACE VIEW ml.pipeline_health AS
SELECT
    DATE(ns_to_timestamp(f.ts_event)) as date,
    COUNT(DISTINCT f.instrument_id) as instruments_processed,
    COUNT(*) as total_features,
    MAX(f.ts_init) as last_update_ns,
    ns_to_timestamp(MAX(f.ts_init)) as last_update_time,
    EXTRACT(EPOCH FROM NOW() - ns_to_timestamp(MAX(f.ts_init))) as staleness_seconds,
    CASE
        WHEN EXTRACT(EPOCH FROM NOW() - ns_to_timestamp(MAX(f.ts_init))) > 86400 THEN 0
        WHEN EXTRACT(EPOCH FROM NOW() - ns_to_timestamp(MAX(f.ts_init))) > 3600 THEN 50
        ELSE 100
    END as health_score
FROM public.ml_feature_values f
WHERE f.ts_event > timestamp_to_ns(NOW() - INTERVAL '7 days')
GROUP BY DATE(ns_to_timestamp(f.ts_event))
ORDER BY date DESC;

CREATE OR REPLACE VIEW ml.data_collection_stats AS
WITH hourly_data AS (
    SELECT
        instrument_id,
        DATE_TRUNC('hour', ns_to_timestamp(ts_event)) as hour,
        COUNT(*) as records_collected,
        MIN(ts_event) as first_record_ns,
        MAX(ts_event) as last_record_ns,
        COUNT(DISTINCT feature_set_id) as feature_sets
    FROM public.ml_feature_values
    WHERE ts_event > timestamp_to_ns(NOW() - INTERVAL '24 hours')
    GROUP BY instrument_id, DATE_TRUNC('hour', ns_to_timestamp(ts_event))
)
SELECT
    instrument_id,
    hour,
    records_collected,
    feature_sets,
    ns_to_timestamp(first_record_ns) as first_record_time,
    ns_to_timestamp(last_record_ns) as last_record_time,
    ROUND(records_collected::NUMERIC / 60, 2) as records_per_minute,
    CASE
        WHEN (last_record_ns - first_record_ns) / 1000000000 < 3300 THEN 'continuous'
        ELSE 'gaps_detected'
    END as collection_status
FROM hourly_data
ORDER BY hour DESC, instrument_id;

CREATE OR REPLACE VIEW ml.model_performance_summary AS
WITH model_stats AS (
    SELECT
        model_id,
        instrument_id,
        DATE(ns_to_timestamp(ts_event)) as date,
        COUNT(*) as predictions,
        AVG(prediction) as avg_prediction,
        STDDEV(prediction) as prediction_stddev,
        AVG(confidence) as avg_confidence,
        MIN(confidence) as min_confidence,
        AVG(inference_time_ms) as avg_inference_ms,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY inference_time_ms) as p95_inference_ms,
        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY inference_time_ms) as p99_inference_ms
    FROM public.ml_model_predictions
    WHERE ts_event > timestamp_to_ns(NOW() - INTERVAL '7 days')
    GROUP BY model_id, instrument_id, DATE(ns_to_timestamp(ts_event))
)
SELECT
    model_id,
    instrument_id,
    date,
    predictions,
    ROUND(avg_prediction::NUMERIC, 4) as avg_prediction,
    ROUND(prediction_stddev::NUMERIC, 4) as prediction_stddev,
    ROUND(avg_confidence::NUMERIC, 3) as avg_confidence,
    ROUND(min_confidence::NUMERIC, 3) as min_confidence,
    ROUND(avg_inference_ms::NUMERIC, 2) as avg_inference_ms,
    ROUND(p95_inference_ms::NUMERIC, 2) as p95_inference_ms,
    ROUND(p99_inference_ms::NUMERIC, 2) as p99_inference_ms,
    CASE
        WHEN avg_confidence < 0.5 THEN 'low_confidence'
        WHEN p99_inference_ms > 1000 THEN 'slow_inference'
        WHEN prediction_stddev < 0.01 THEN 'low_variance'
        ELSE 'healthy'
    END as health_status
FROM model_stats
ORDER BY date DESC, model_id, instrument_id;

CREATE OR REPLACE VIEW ml.strategy_signal_summary AS
WITH signal_stats AS (
    SELECT
        strategy_id,
        instrument_id,
        signal_type,
        DATE(ns_to_timestamp(ts_event)) as date,
        COUNT(*) as signal_count,
        AVG(strength) as avg_strength,
        MIN(strength) as min_strength,
        MAX(strength) as max_strength,
        AVG((risk_metrics->>'sharpe_ratio')::FLOAT) as avg_sharpe,
        AVG((risk_metrics->>'max_drawdown')::FLOAT) as avg_max_drawdown
    FROM public.ml_strategy_signals
    WHERE ts_event > timestamp_to_ns(NOW() - INTERVAL '7 days')
    GROUP BY strategy_id, instrument_id, signal_type, DATE(ns_to_timestamp(ts_event))
)
SELECT
    strategy_id,
    instrument_id,
    signal_type,
    date,
    signal_count,
    ROUND(avg_strength::NUMERIC, 3) as avg_strength,
    ROUND(min_strength::NUMERIC, 3) as min_strength,
    ROUND(max_strength::NUMERIC, 3) as max_strength,
    ROUND(COALESCE(avg_sharpe, 0)::NUMERIC, 3) as avg_sharpe_ratio,
    ROUND(COALESCE(avg_max_drawdown, 0)::NUMERIC, 3) as avg_max_drawdown,
    CASE
        WHEN avg_strength < 0.3 THEN 'weak_signals'
        WHEN signal_count < 10 THEN 'low_activity'
        WHEN COALESCE(avg_sharpe, 0) < 0 THEN 'negative_sharpe'
        ELSE 'healthy'
    END as signal_quality
FROM signal_stats
ORDER BY date DESC, strategy_id, instrument_id;

CREATE OR REPLACE VIEW ml_stage_coverage AS
WITH stage_counts AS (
    SELECT
        dataset_id,
        instrument_id,
        stage,
        source,
        DATE(to_timestamp(ts_event / 1000000000)) as event_date,
        COUNT(*) as event_count,
        SUM(count) as record_count,
        COUNT(CASE WHEN status = 'success' THEN 1 END) as success_count,
        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failure_count
    FROM ml_data_events
    WHERE ts_event >= EXTRACT(EPOCH FROM (CURRENT_DATE - INTERVAL '30 days')) * 1000000000
    GROUP BY dataset_id, instrument_id, stage, source, event_date
),
feature_coverage AS (
    SELECT
        feature_set_id as dataset_id,
        instrument_id,
        DATE(to_timestamp(ts_event / 1000000000)) as event_date,
        COUNT(*) as feature_count
    FROM ml_feature_values
    WHERE ts_event >= EXTRACT(EPOCH FROM (CURRENT_DATE - INTERVAL '30 days')) * 1000000000
    GROUP BY feature_set_id, instrument_id, event_date
),
prediction_coverage AS (
    SELECT
        model_id as dataset_id,
        instrument_id,
        DATE(to_timestamp(ts_event / 1000000000)) as event_date,
        COUNT(*) as prediction_count
    FROM ml_model_predictions
    WHERE ts_event >= EXTRACT(EPOCH FROM (CURRENT_DATE - INTERVAL '30 days')) * 1000000000
    GROUP BY model_id, instrument_id, event_date
),
signal_coverage AS (
    SELECT
        strategy_id as dataset_id,
        instrument_id,
        DATE(to_timestamp(ts_event / 1000000000)) as event_date,
        COUNT(*) as signal_count
    FROM ml_strategy_signals
    WHERE ts_event >= EXTRACT(EPOCH FROM (CURRENT_DATE - INTERVAL '30 days')) * 1000000000
    GROUP BY strategy_id, instrument_id, event_date
)
SELECT
    COALESCE(sc.dataset_id, fc.dataset_id, pc.dataset_id, sgc.dataset_id) as dataset_id,
    COALESCE(sc.instrument_id, fc.instrument_id, pc.instrument_id, sgc.instrument_id) as instrument_id,
    COALESCE(sc.event_date, fc.event_date, pc.event_date, sgc.event_date) as event_date,
    sc.stage,
    sc.source,
    sc.event_count,
    sc.record_count,
    sc.success_count,
    sc.failure_count,
    fc.feature_count,
    pc.prediction_count,
    sgc.signal_count,
    CASE
        WHEN sc.stage = 'FEATURE_COMPUTED' AND fc.feature_count > 0 THEN 100.0
        WHEN sc.stage = 'PREDICTION_EMITTED' AND pc.prediction_count > 0 THEN 100.0
        WHEN sc.stage = 'SIGNAL_EMITTED' AND sgc.signal_count > 0 THEN 100.0
        WHEN sc.record_count > 0 THEN
            CASE
                WHEN fc.feature_count IS NOT NULL THEN (fc.feature_count::DECIMAL / sc.record_count) * 100
                WHEN pc.prediction_count IS NOT NULL THEN (pc.prediction_count::DECIMAL / sc.record_count) * 100
                WHEN sgc.signal_count IS NOT NULL THEN (sgc.signal_count::DECIMAL / sc.record_count) * 100
                ELSE 0
            END
        ELSE 0
    END as coverage_pct
FROM stage_counts sc
LEFT JOIN feature_coverage fc
    ON sc.dataset_id = fc.dataset_id
    AND sc.instrument_id = fc.instrument_id
    AND sc.event_date = fc.event_date
LEFT JOIN prediction_coverage pc
    ON sc.dataset_id = pc.dataset_id
    AND sc.instrument_id = pc.instrument_id
    AND sc.event_date = pc.event_date
LEFT JOIN signal_coverage sgc
    ON sc.dataset_id = sgc.dataset_id
    AND sc.instrument_id = sgc.instrument_id
    AND sc.event_date = sgc.event_date;

CREATE OR REPLACE VIEW ml_watermark_lag AS
SELECT
    w.dataset_id,
    w.instrument_id,
    w.source,
    w.last_success_ns,
    w.last_attempt_ns,
    w.completeness_pct,
    EXTRACT(EPOCH FROM NOW()) * 1000000000 - w.last_success_ns as lag_ns,
    (EXTRACT(EPOCH FROM NOW()) * 1000000000 - w.last_success_ns) / 1000000000.0 as lag_seconds,
    r.dataset_type,
    r.storage_kind,
    w.updated_at
FROM ml_data_watermarks w
JOIN ml_dataset_registry r ON w.dataset_id = r.dataset_id
ORDER BY lag_seconds DESC NULLS LAST;

CREATE OR REPLACE VIEW ml_lineage_graph AS
WITH RECURSIVE lineage_tree AS (
    SELECT
        dataset_id,
        dataset_id as root_dataset,
        0 as depth,
        ARRAY[dataset_id]::varchar[] as path
    FROM ml_dataset_registry
    WHERE parents IS NULL OR parents = '[]'::jsonb

    UNION ALL

    SELECT
        l.child_dataset_id as dataset_id,
        lt.root_dataset as root_dataset,
        lt.depth + 1 as depth,
        lt.path || l.child_dataset_id as path
    FROM ml_data_lineage l
    JOIN lineage_tree lt ON l.parent_dataset_id = lt.dataset_id
    WHERE NOT l.child_dataset_id = ANY(lt.path)
)
SELECT
    dataset_id,
    root_dataset,
    depth,
    path
FROM lineage_tree
ORDER BY root_dataset, depth, dataset_id;

CREATE OR REPLACE VIEW ml_data_quality_summary AS
SELECT
    e.dataset_id,
    e.instrument_id,
    DATE(to_timestamp(e.ts_event / 1000000000)) as date,
    COUNT(*) as total_events,
    COUNT(CASE WHEN e.status = 'success' THEN 1 END) as successful_events,
    COUNT(CASE WHEN e.status = 'failed' THEN 1 END) as failed_events,
    COUNT(CASE WHEN e.status = 'partial' THEN 1 END) as partial_events,
    ROUND(100.0 * COUNT(CASE WHEN e.status = 'success' THEN 1 END) / NULLIF(COUNT(*), 0), 2) as success_rate,
    SUM(e.count) as total_records,
    MAX(w.completeness_pct) as max_completeness,
    AVG(w.completeness_pct) as avg_completeness
FROM ml_data_events e
LEFT JOIN ml_data_watermarks w
    ON e.dataset_id = w.dataset_id
    AND e.instrument_id = w.instrument_id
WHERE e.ts_event >= EXTRACT(EPOCH FROM (CURRENT_DATE - INTERVAL '7 days')) * 1000000000
GROUP BY e.dataset_id, e.instrument_id, date
ORDER BY date DESC, dataset_id, instrument_id;

-- ============================================================================
-- Indexes
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_ml_feature_values_lookup
    ON ml_feature_values (feature_set_id, instrument_id, ts_event);
CREATE INDEX IF NOT EXISTS idx_ml_feature_values_live
    ON ml_feature_values (is_live) WHERE is_live = TRUE;
CREATE UNIQUE INDEX IF NOT EXISTS uq_ml_feature_values_key
    ON ml_feature_values (feature_set_id, instrument_id, ts_event);
CREATE INDEX IF NOT EXISTS brin_ml_feature_values_ts
    ON ml_feature_values USING BRIN (ts_event);

CREATE INDEX IF NOT EXISTS idx_ml_model_predictions_lookup
    ON ml_model_predictions (model_id, instrument_id, ts_event);
CREATE INDEX IF NOT EXISTS idx_ml_model_predictions_live
    ON ml_model_predictions (is_live) WHERE is_live = TRUE;
CREATE INDEX IF NOT EXISTS brin_ml_model_predictions_ts
    ON ml_model_predictions USING BRIN (ts_event);

CREATE INDEX IF NOT EXISTS idx_ml_strategy_signals_lookup
    ON ml_strategy_signals (strategy_id, instrument_id, ts_event);
CREATE INDEX IF NOT EXISTS idx_ml_strategy_signals_type
    ON ml_strategy_signals (signal_type);
CREATE INDEX IF NOT EXISTS idx_ml_strategy_signals_live
    ON ml_strategy_signals (is_live) WHERE is_live = TRUE;
CREATE INDEX IF NOT EXISTS brin_ml_strategy_signals_ts
    ON ml_strategy_signals USING BRIN (ts_event);

CREATE INDEX IF NOT EXISTS idx_ml_strategy_performance
    ON ml_strategy_performance (strategy_id, period_start);
CREATE INDEX IF NOT EXISTS idx_ml_strategy_risk_halt_events_lookup
    ON ml_strategy_risk_halt_events (strategy_id, instrument_id, ts_event);
CREATE INDEX IF NOT EXISTS idx_ml_strategy_risk_halt_events_type
    ON ml_strategy_risk_halt_events (event_type);
CREATE INDEX IF NOT EXISTS idx_ml_strategy_replay_summary
    ON ml_strategy_replay_summary (run_id);

CREATE INDEX IF NOT EXISTS idx_market_data_bar_time
    ON market_data_bar USING BRIN (ts_event);
CREATE INDEX IF NOT EXISTS idx_market_data_bar_instrument
    ON market_data_bar (instrument_id, ts_event DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_bar_quality
    ON market_data_bar (quality_flags) WHERE quality_flags > 0;

CREATE INDEX IF NOT EXISTS idx_market_data_quote_tick_time
    ON market_data_quote_tick USING BRIN (ts_event);
CREATE INDEX IF NOT EXISTS idx_market_data_quote_tick_instrument
    ON market_data_quote_tick (instrument_id, ts_event DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_quote_tick_quality
    ON market_data_quote_tick (quality_flags) WHERE quality_flags > 0;

CREATE INDEX IF NOT EXISTS idx_market_data_tbbo_time
    ON market_data_tbbo USING BRIN (ts_event);
CREATE INDEX IF NOT EXISTS idx_market_data_tbbo_instrument
    ON market_data_tbbo (instrument_id, ts_event DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_tbbo_quality
    ON market_data_tbbo (quality_flags) WHERE quality_flags > 0;

CREATE INDEX IF NOT EXISTS idx_market_data_mbp1_time
    ON market_data_mbp1 USING BRIN (ts_event);
CREATE INDEX IF NOT EXISTS idx_market_data_mbp1_instrument
    ON market_data_mbp1 (instrument_id, ts_event DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_mbp1_quality
    ON market_data_mbp1 (quality_flags) WHERE quality_flags > 0;

CREATE INDEX IF NOT EXISTS idx_market_data_trade_tick_time
    ON market_data_trade_tick USING BRIN (ts_event);
CREATE INDEX IF NOT EXISTS idx_market_data_trade_tick_instrument
    ON market_data_trade_tick (instrument_id, ts_event DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_trade_tick_quality
    ON market_data_trade_tick (quality_flags) WHERE quality_flags > 0;

CREATE INDEX IF NOT EXISTS idx_metadata_symbol
    ON market_data_metadata (symbol);
CREATE INDEX IF NOT EXISTS idx_metadata_exchange
    ON market_data_metadata (exchange);
CREATE INDEX IF NOT EXISTS idx_metadata_asset_class
    ON market_data_metadata (asset_class);
CREATE INDEX IF NOT EXISTS idx_metadata_active
    ON market_data_metadata (is_active) WHERE is_active = TRUE;

CREATE INDEX IF NOT EXISTS idx_stats_lookup
    ON market_data_statistics (instrument_id, period_start DESC);

CREATE INDEX IF NOT EXISTS idx_positions_strategy
    ON ml_positions (strategy_id);
CREATE INDEX IF NOT EXISTS idx_positions_instrument
    ON ml_positions (instrument_id);

CREATE INDEX IF NOT EXISTS idx_ml_dataset_registry_name
    ON ml_dataset_registry (name);
CREATE INDEX IF NOT EXISTS idx_ml_dataset_registry_type
    ON ml_dataset_registry (dataset_type);
CREATE INDEX IF NOT EXISTS idx_ml_dataset_registry_created
    ON ml_dataset_registry (created_at);

CREATE INDEX IF NOT EXISTS idx_ml_data_events_time
    ON ml_data_events USING BRIN (ts_event);
CREATE INDEX IF NOT EXISTS idx_ml_data_events_lookup
    ON ml_data_events (dataset_id, instrument_id, ts_event DESC);
CREATE INDEX IF NOT EXISTS idx_ml_data_events_stage
    ON ml_data_events (stage, ts_event DESC);
CREATE INDEX IF NOT EXISTS idx_ml_data_events_failures
    ON ml_data_events (status, created_at DESC) WHERE status = 'failed';
CREATE INDEX IF NOT EXISTS idx_ml_data_events_run
    ON ml_data_events (run_id);
CREATE INDEX IF NOT EXISTS idx_ml_data_events_status
    ON ml_data_events (status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_ml_data_watermarks_dataset
    ON ml_data_watermarks (dataset_id);
CREATE INDEX IF NOT EXISTS idx_ml_data_watermarks_updated
    ON ml_data_watermarks (updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_ml_data_watermarks_incomplete
    ON ml_data_watermarks (completeness_pct) WHERE completeness_pct < 95;
CREATE INDEX IF NOT EXISTS idx_ml_data_watermarks_lookup
    ON ml_data_watermarks (dataset_id, instrument_id, source);

CREATE INDEX IF NOT EXISTS idx_ml_data_lineage_child
    ON ml_data_lineage (child_dataset_id);
CREATE INDEX IF NOT EXISTS idx_ml_data_lineage_parent
    ON ml_data_lineage (parent_dataset_id);
CREATE INDEX IF NOT EXISTS idx_ml_data_lineage_transform
    ON ml_data_lineage (transform_id);
CREATE INDEX IF NOT EXISTS idx_ml_data_lineage_created
    ON ml_data_lineage (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_macro_release_series
    ON ml.macro_release_calendar (series_id);
CREATE INDEX IF NOT EXISTS idx_macro_release_ts_event
    ON ml.macro_release_calendar USING BRIN (ts_event);

CREATE INDEX IF NOT EXISTS idx_macro_observations_series
    ON ml.macro_observations (series_id);
CREATE INDEX IF NOT EXISTS idx_macro_observations_ts_event
    ON ml.macro_observations USING BRIN (ts_event);

CREATE INDEX IF NOT EXISTS idx_events_calendar_ts_event
    ON ml.events_calendar USING BRIN (ts_event);

CREATE INDEX IF NOT EXISTS idx_microstructure_minute_brin
    ON ml.microstructure_minute USING BRIN (ts_event);
CREATE INDEX IF NOT EXISTS idx_microstructure_minute_instrument
    ON ml.microstructure_minute (instrument_id);

CREATE INDEX IF NOT EXISTS idx_l2_minute_brin
    ON ml.l2_minute USING BRIN (ts_event);
CREATE INDEX IF NOT EXISTS idx_l2_minute_instrument
    ON ml.l2_minute (instrument_id);

-- ============================================================================
-- Partition Bootstrap (2023-2027)
-- ============================================================================

SELECT create_monthly_partitions('ml_feature_values', '2023-01-01'::DATE, 60);
SELECT create_monthly_partitions('ml_model_predictions', '2023-01-01'::DATE, 60);
SELECT create_monthly_partitions('ml_strategy_signals', '2023-01-01'::DATE, 60);
SELECT create_monthly_partitions('market_data_bar', '2023-01-01'::DATE, 60);
SELECT create_monthly_partitions('market_data_quote_tick', '2023-01-01'::DATE, 60);
SELECT create_monthly_partitions('market_data_tbbo', '2023-01-01'::DATE, 60);
SELECT create_monthly_partitions('market_data_mbp1', '2023-01-01'::DATE, 60);
SELECT create_monthly_partitions('market_data_trade_tick', '2023-01-01'::DATE, 60);
SELECT create_event_partitions('ml_data_events', '2023-01-01'::DATE, 60);
