-- Data Registry Schema Migration
-- Creates tables for dataset manifests, contracts, lineage, events, and watermarks
-- This migration implements Phase 0 of the Data Registry hardening plan

-- ============================================================================
-- Helper Functions (must be defined before use)
-- ============================================================================

-- Function to create monthly partitions for event tables
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
-- Dataset Registry Table
-- ============================================================================

-- Main registry table for dataset manifests
CREATE TABLE IF NOT EXISTS ml_dataset_registry (
    -- Identity
    dataset_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    dataset_type VARCHAR(50) NOT NULL,  -- BARS, TRADES, QUOTES, MBP1, TBBO, FEATURES, PREDICTIONS, SIGNALS

    -- Storage configuration
    storage_kind VARCHAR(20) NOT NULL,  -- 'parquet' or 'postgres'
    location TEXT NOT NULL,              -- File path or table name
    partitioning JSONB,                  -- {"by": "ts_event", "interval": "monthly"}
    retention_days INTEGER NOT NULL,

    -- Schema information
    schema JSONB NOT NULL,               -- {"instrument_id": "str", "ts_event": "int64", ...}
    schema_hash VARCHAR(64) NOT NULL,    -- SHA256 hash of schema for validation

    -- Validation and constraints
    constraints JSONB,                   -- {"ranges": {...}, "nullability": {...}}

    -- Lineage tracking
    parents JSONB,                       -- List of parent dataset IDs
    pipeline_signature VARCHAR(255),     -- Signature of pipeline that creates this dataset

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_modified TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Metadata
    metadata JSONB,                      -- Additional flexible metadata

    -- Constraints
    CONSTRAINT check_dataset_type CHECK (
        dataset_type IN ('BARS', 'TRADES', 'QUOTES', 'MBP1', 'TBBO', 'FEATURES', 'PREDICTIONS', 'SIGNALS')
    ),
    CONSTRAINT check_storage_kind CHECK (
        storage_kind IN ('parquet', 'postgres')
    ),
    CONSTRAINT check_retention_positive CHECK (
        retention_days > 0
    )
);

-- Indexes for dataset registry
CREATE INDEX IF NOT EXISTS idx_ml_dataset_registry_name
    ON ml_dataset_registry (name);
CREATE INDEX IF NOT EXISTS idx_ml_dataset_registry_type
    ON ml_dataset_registry (dataset_type);
CREATE INDEX IF NOT EXISTS idx_ml_dataset_registry_created
    ON ml_dataset_registry (created_at);

-- ============================================================================
-- Data Events Table (Partitioned by ts_event)
-- ============================================================================

-- Track data processing events across all pipeline stages
CREATE TABLE IF NOT EXISTS ml_data_events (
    -- Event identification
    event_id BIGSERIAL,
    dataset_id VARCHAR(255) NOT NULL,
    instrument_id VARCHAR(100) NOT NULL,

    -- Event details
    stage VARCHAR(50) NOT NULL,         -- INGESTED, CATALOG_WRITTEN, FEATURE_COMPUTED, PREDICTION_EMITTED, SIGNAL_EMITTED
    source VARCHAR(50) NOT NULL,        -- 'live', 'historical', 'backfill'
    run_id VARCHAR(255),                -- Unique identifier for this processing run

    -- Time range of data in this event
    ts_min BIGINT NOT NULL,             -- Minimum timestamp in nanoseconds
    ts_max BIGINT NOT NULL,             -- Maximum timestamp in nanoseconds
    ts_event BIGINT NOT NULL,           -- When this event occurred (for partitioning)

    -- Data statistics
    count BIGINT NOT NULL,              -- Number of records processed
    seq_min BIGINT,                     -- Minimum sequence number (optional)
    seq_max BIGINT,                     -- Maximum sequence number (optional)

    -- Processing status
    status VARCHAR(20) NOT NULL,        -- 'success', 'failed', 'partial'
    error TEXT,                         -- Error message if status is 'failed'

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    PRIMARY KEY (event_id, ts_event),

    -- Constraints
    CONSTRAINT check_stage CHECK (
        stage IN ('INGESTED', 'CATALOG_WRITTEN', 'FEATURE_COMPUTED', 'PREDICTION_EMITTED', 'SIGNAL_EMITTED')
    ),
    CONSTRAINT check_source CHECK (
        source IN ('live', 'historical', 'backfill')
    ),
    CONSTRAINT check_status CHECK (
        status IN ('success', 'failed', 'partial')
    ),
    CONSTRAINT check_time_range CHECK (
        ts_min <= ts_max
    ),
    CONSTRAINT check_seq_range CHECK (
        seq_min IS NULL OR seq_max IS NULL OR seq_min <= seq_max
    )
) PARTITION BY RANGE (ts_event);

-- Create partitions for data events (36 months)
SELECT create_event_partitions('ml_data_events', '2024-01-01'::DATE, 36);

-- Indexes for data events
-- BRIN index for time-based queries
CREATE INDEX IF NOT EXISTS idx_ml_data_events_time
    ON ml_data_events USING BRIN (ts_event);

-- Composite index for common lookups
CREATE INDEX IF NOT EXISTS idx_ml_data_events_lookup
    ON ml_data_events (dataset_id, instrument_id, ts_event DESC);

-- Index for stage-based queries
CREATE INDEX IF NOT EXISTS idx_ml_data_events_stage
    ON ml_data_events (stage, ts_event DESC);

-- Partial index for failed events
CREATE INDEX IF NOT EXISTS idx_ml_data_events_failures
    ON ml_data_events (status, created_at DESC)
    WHERE status = 'failed';

-- ============================================================================
-- Data Watermarks Table
-- ============================================================================

-- Track processing watermarks and completeness metrics
CREATE TABLE IF NOT EXISTS ml_data_watermarks (
    -- Identification
    dataset_id VARCHAR(255) NOT NULL,
    instrument_id VARCHAR(100) NOT NULL,
    source VARCHAR(50) NOT NULL,        -- 'live', 'historical', 'backfill'

    -- Watermark tracking
    last_success_ns BIGINT,             -- Last successful processing timestamp
    last_attempt_ns BIGINT,             -- Last attempted processing timestamp

    -- Statistics
    last_count BIGINT DEFAULT 0,        -- Count from last successful processing
    completeness_pct DECIMAL(5,2),      -- Percentage of expected data received (0-100)

    -- Metadata
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Primary key and constraints
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

-- Indexes for watermarks
CREATE INDEX IF NOT EXISTS idx_ml_data_watermarks_dataset
    ON ml_data_watermarks (dataset_id);
CREATE INDEX IF NOT EXISTS idx_ml_data_watermarks_updated
    ON ml_data_watermarks (updated_at DESC);

-- Partial index for incomplete data
CREATE INDEX IF NOT EXISTS idx_ml_data_watermarks_incomplete
    ON ml_data_watermarks (completeness_pct)
    WHERE completeness_pct < 95;

-- ============================================================================
-- Data Lineage Table
-- ============================================================================

-- Track dataset transformation lineage
CREATE TABLE IF NOT EXISTS ml_data_lineage (
    -- Lineage identification
    lineage_id BIGSERIAL PRIMARY KEY,
    transform_id VARCHAR(255) NOT NULL,  -- Unique identifier for this transformation

    -- Dataset relationships
    child_dataset_id VARCHAR(255) NOT NULL,
    parent_dataset_id VARCHAR(255) NOT NULL,

    -- Transformation details
    ts_range JSONB,                     -- {"start_ns": ..., "end_ns": ...}
    parameters JSONB,                    -- Transformation parameters used

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT fk_lineage_child FOREIGN KEY (child_dataset_id)
        REFERENCES ml_dataset_registry(dataset_id) ON DELETE CASCADE,
    CONSTRAINT fk_lineage_parent FOREIGN KEY (parent_dataset_id)
        REFERENCES ml_dataset_registry(dataset_id) ON DELETE CASCADE
);

-- Indexes for lineage
CREATE INDEX IF NOT EXISTS idx_ml_data_lineage_child
    ON ml_data_lineage (child_dataset_id);
CREATE INDEX IF NOT EXISTS idx_ml_data_lineage_parent
    ON ml_data_lineage (parent_dataset_id);
CREATE INDEX IF NOT EXISTS idx_ml_data_lineage_transform
    ON ml_data_lineage (transform_id);
CREATE INDEX IF NOT EXISTS idx_ml_data_lineage_created
    ON ml_data_lineage (created_at DESC);

-- ============================================================================
-- Views for Monitoring and Analysis
-- ============================================================================

-- View for stage coverage analysis
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

-- View for watermark lag analysis
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

-- View for lineage graph (useful for visualizations)
CREATE OR REPLACE VIEW ml_lineage_graph AS
WITH RECURSIVE lineage_tree AS (
    -- Base case: datasets with no parents
    SELECT
        dataset_id,
        dataset_id as root_dataset,
        0 as depth,
        ARRAY[dataset_id] as path
    FROM ml_dataset_registry
    WHERE parents IS NULL OR parents = '[]'::jsonb

    UNION ALL

    -- Recursive case: follow lineage relationships
    SELECT
        l.child_dataset_id as dataset_id,
        lt.root_dataset as root_dataset,
        lt.depth + 1 as depth,
        lt.path || l.child_dataset_id as path
    FROM ml_data_lineage l
    JOIN lineage_tree lt ON l.parent_dataset_id = lt.dataset_id
    WHERE NOT l.child_dataset_id = ANY(lt.path)  -- Prevent cycles
)
SELECT
    dataset_id,
    root_dataset,
    depth,
    path
FROM lineage_tree
ORDER BY root_dataset, depth, dataset_id;

-- View for data quality summary
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
-- Functions for Data Registry Operations
-- ============================================================================

-- Function to update watermark with automatic completeness calculation
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

-- Function to emit data event with automatic watermark update
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
RETURNS BIGINT AS $$
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
            NULL  -- Let the application calculate completeness
        );
    END IF;

    RETURN v_event_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Triggers for Automatic Updates
-- ============================================================================

-- Trigger to update last_modified on dataset registry changes
CREATE OR REPLACE FUNCTION update_dataset_modified()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_modified = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_dataset_modified
    BEFORE UPDATE ON ml_dataset_registry
    FOR EACH ROW
    EXECUTE FUNCTION update_dataset_modified();

-- ============================================================================
-- Initial Data Population (Optional)
-- ============================================================================

-- Insert common dataset definitions (can be customized per deployment)
INSERT INTO ml_dataset_registry (
    dataset_id, name, version, dataset_type, storage_kind,
    location, partitioning, retention_days, schema, schema_hash,
    constraints, parents, pipeline_signature
)
VALUES
    -- Market data datasets
    ('bars_1m', 'One Minute Bars', '1.0.0', 'BARS', 'postgres',
     'market_data', '{"by": "ts_event", "interval": "monthly"}'::jsonb, 365,
     '{"instrument_id": "str", "ts_event": "int64", "ts_init": "int64", "open": "float64", "high": "float64", "low": "float64", "close": "float64", "volume": "float64"}'::jsonb,
     '', '{"ranges": {"open": {"min": 0}, "high": {"min": 0}, "low": {"min": 0}, "close": {"min": 0}, "volume": {"min": 0}}}'::jsonb,
     '[]'::jsonb, 'data_scheduler_v1'),

    -- Feature datasets
    ('features_microstructure', 'Microstructure Features', '1.0.0', 'FEATURES', 'postgres',
     'ml_feature_values', '{"by": "ts_event", "interval": "monthly"}'::jsonb, 180,
     '{"feature_set_id": "str", "instrument_id": "str", "ts_event": "int64", "ts_init": "int64", "values": "jsonb"}'::jsonb,
     '', '{}'::jsonb, '["bars_1m"]'::jsonb, 'feature_engineer_v1'),

    -- Prediction datasets
    ('predictions_xgboost', 'XGBoost Model Predictions', '1.0.0', 'PREDICTIONS', 'postgres',
     'ml_model_predictions', '{"by": "ts_event", "interval": "monthly"}'::jsonb, 90,
     '{"model_id": "str", "instrument_id": "str", "ts_event": "int64", "ts_init": "int64", "prediction": "float64", "confidence": "float64"}'::jsonb,
     '', '{}'::jsonb, '["features_microstructure"]'::jsonb, 'model_inference_v1'),

    -- Signal datasets
    ('signals_momentum', 'Momentum Strategy Signals', '1.0.0', 'SIGNALS', 'postgres',
     'ml_strategy_signals', '{"by": "ts_event", "interval": "monthly"}'::jsonb, 90,
     '{"strategy_id": "str", "instrument_id": "str", "ts_event": "int64", "ts_init": "int64", "signal_type": "str", "strength": "float64"}'::jsonb,
     '', '{}'::jsonb, '["predictions_xgboost"]'::jsonb, 'strategy_executor_v1')
ON CONFLICT (dataset_id) DO NOTHING;

-- ============================================================================
-- Usage Examples and Comments
-- ============================================================================

-- Example: Register a new dataset
-- INSERT INTO ml_dataset_registry (dataset_id, name, version, dataset_type, ...)
-- VALUES ('my_dataset', 'My Dataset', '1.0.0', 'FEATURES', ...);

-- Example: Emit a data event
-- SELECT emit_data_event('bars_1m', 'EUR/USD', 'CATALOG_WRITTEN', 'historical',
--                        'run_123', 1234567890000000000, 1234567900000000000, 1000, 'success');

-- Example: Update watermark
-- SELECT update_watermark('bars_1m', 'EUR/USD', 'live', 1234567900000000000, 1000, 98.5);

-- Example: Query stage coverage
-- SELECT * FROM ml_stage_coverage WHERE dataset_id = 'bars_1m' AND event_date >= CURRENT_DATE - 7;

-- Example: Check watermark lag
-- SELECT * FROM ml_watermark_lag WHERE lag_seconds > 3600;  -- More than 1 hour behind

-- Example: View lineage tree
-- SELECT * FROM ml_lineage_graph WHERE root_dataset = 'bars_1m';