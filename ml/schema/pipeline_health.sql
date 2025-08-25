-- NOTE: These views have been moved into migrations (ml/stores/migrations/005_views.sql)
-- to ensure consistent application order and proper schema qualification.
-- This file is retained for reference only.
-- Provides comprehensive health check views for monitoring the ML pipeline
-- All timestamps follow Nautilus convention (nanoseconds since epoch)

-- ============================================================================
-- Helper Functions
-- ============================================================================

-- Convert nanoseconds to timestamp for display
CREATE OR REPLACE FUNCTION ns_to_timestamp(ns BIGINT)
RETURNS TIMESTAMP WITH TIME ZONE AS $$
BEGIN
    RETURN to_timestamp(ns::DOUBLE PRECISION / 1000000000);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Convert timestamp to nanoseconds
CREATE OR REPLACE FUNCTION timestamp_to_ns(ts TIMESTAMP WITH TIME ZONE)
RETURNS BIGINT AS $$
BEGIN
    RETURN (EXTRACT(EPOCH FROM ts) * 1000000000)::BIGINT;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ============================================================================
-- Pipeline Health Overview
-- ============================================================================

-- Overall pipeline health status
CREATE OR REPLACE VIEW ml.pipeline_health AS
SELECT
    DATE(ns_to_timestamp(ts_event)) as date,
    COUNT(DISTINCT instrument_id) as instruments_processed,
    COUNT(*) as total_features,
    MAX(ts_init) as last_update_ns,
    ns_to_timestamp(MAX(ts_init)) as last_update_time,
    -- Data freshness in seconds
    EXTRACT(EPOCH FROM NOW() - ns_to_timestamp(MAX(ts_init))) as staleness_seconds,
    -- Health score (0-100)
    CASE
        WHEN EXTRACT(EPOCH FROM NOW() - ns_to_timestamp(MAX(ts_init))) > 86400 THEN 0  -- >24h stale
        WHEN EXTRACT(EPOCH FROM NOW() - ns_to_timestamp(MAX(ts_init))) > 3600 THEN 50  -- >1h stale
        ELSE 100
    END as health_score
FROM ml_feature_values
WHERE ts_event > timestamp_to_ns(NOW() - INTERVAL '7 days')
GROUP BY DATE(ns_to_timestamp(ts_event))
ORDER BY date DESC;

-- ============================================================================
-- Data Collection Statistics
-- ============================================================================

-- Data collection metrics per instrument
CREATE OR REPLACE VIEW ml.data_collection_stats AS
WITH hourly_data AS (
    SELECT
        instrument_id,
        DATE_TRUNC('hour', ns_to_timestamp(ts_event)) as hour,
        COUNT(*) as records_collected,
        MIN(ts_event) as first_record_ns,
        MAX(ts_event) as last_record_ns,
        COUNT(DISTINCT feature_set_id) as feature_sets
    FROM ml_feature_values
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
    -- Calculate collection rate (records per minute)
    ROUND(records_collected::NUMERIC / 60, 2) as records_per_minute,
    -- Check if collection is continuous
    CASE
        WHEN (last_record_ns - first_record_ns) / 1000000000 < 3300 THEN 'continuous'  -- <55 min gap
        ELSE 'gaps_detected'
    END as collection_status
FROM hourly_data
ORDER BY hour DESC, instrument_id;

-- ============================================================================
-- Feature Computation Statistics
-- ============================================================================

-- Feature computation health and performance
CREATE OR REPLACE VIEW ml.feature_computation_stats AS
WITH feature_stats AS (
    SELECT
        fcs.feature_set_id,
        fcs.instrument_id,
        DATE(ns_to_timestamp(fcs.ts_event)) as computation_date,
        COUNT(*) as computations,
        AVG(fcs.computation_time_ms) as avg_computation_ms,
        MAX(fcs.computation_time_ms) as max_computation_ms,
        MIN(fcs.computation_time_ms) as min_computation_ms,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY fcs.computation_time_ms) as p95_computation_ms,
        SUM(fcs.num_features) as total_features_computed
    FROM ml_feature_computation_stats fcs
    WHERE fcs.ts_event > timestamp_to_ns(NOW() - INTERVAL '7 days')
    GROUP BY fcs.feature_set_id, fcs.instrument_id, DATE(ns_to_timestamp(fcs.ts_event))
),
feature_quality AS (
    SELECT
        feature_set_id,
        instrument_id,
        DATE(ns_to_timestamp(ts_event)) as date,
        COUNT(*) as total_records,
        -- Check for null values in JSONB
        SUM(CASE WHEN jsonb_typeof(values) = 'null' THEN 1 ELSE 0 END) as null_records,
        -- Check for empty feature sets
        SUM(CASE WHEN jsonb_array_length(values::jsonb) = 0 THEN 1 ELSE 0 END) as empty_records
    FROM ml_feature_values
    WHERE ts_event > timestamp_to_ns(NOW() - INTERVAL '7 days')
    GROUP BY feature_set_id, instrument_id, DATE(ns_to_timestamp(ts_event))
)
SELECT
    fs.feature_set_id,
    fs.instrument_id,
    fs.computation_date,
    fs.computations,
    fs.total_features_computed,
    ROUND(fs.avg_computation_ms::NUMERIC, 2) as avg_computation_ms,
    ROUND(fs.p95_computation_ms::NUMERIC, 2) as p95_computation_ms,
    fs.max_computation_ms,
    COALESCE(fq.total_records, 0) as stored_records,
    COALESCE(fq.null_records, 0) as null_records,
    COALESCE(fq.empty_records, 0) as empty_records,
    -- Quality score (0-100)
    CASE
        WHEN COALESCE(fq.total_records, 0) = 0 THEN 0
        ELSE ROUND(((fq.total_records - COALESCE(fq.null_records, 0) - COALESCE(fq.empty_records, 0))::NUMERIC / fq.total_records) * 100, 2)
    END as quality_score,
    -- Performance health
    CASE
        WHEN fs.p95_computation_ms > 500 THEN 'critical'
        WHEN fs.p95_computation_ms > 200 THEN 'warning'
        ELSE 'healthy'
    END as performance_status
FROM feature_stats fs
LEFT JOIN feature_quality fq
    ON fs.feature_set_id = fq.feature_set_id
    AND fs.instrument_id = fq.instrument_id
    AND fs.computation_date = fq.date
ORDER BY fs.computation_date DESC, fs.instrument_id;

-- ============================================================================
-- Data Freshness Monitoring
-- ============================================================================

-- Monitor data staleness per instrument
CREATE OR REPLACE VIEW ml.data_freshness AS
WITH latest_data AS (
    SELECT
        instrument_id,
        feature_set_id,
        MAX(ts_event) as last_event_ns,
        MAX(ts_init) as last_update_ns,
        COUNT(*) as records_last_hour
    FROM ml_feature_values
    WHERE ts_event > timestamp_to_ns(NOW() - INTERVAL '1 hour')
    GROUP BY instrument_id, feature_set_id
),
all_instruments AS (
    SELECT DISTINCT
        instrument_id,
        feature_set_id
    FROM ml_feature_values
    WHERE ts_event > timestamp_to_ns(NOW() - INTERVAL '7 days')
)
SELECT
    ai.instrument_id,
    ai.feature_set_id,
    ns_to_timestamp(COALESCE(ld.last_event_ns, 0)) as last_event_time,
    ns_to_timestamp(COALESCE(ld.last_update_ns, 0)) as last_update_time,
    COALESCE(ld.records_last_hour, 0) as records_last_hour,
    -- Calculate staleness in seconds
    CASE
        WHEN ld.last_update_ns IS NULL THEN 999999
        ELSE EXTRACT(EPOCH FROM NOW() - ns_to_timestamp(ld.last_update_ns))
    END as staleness_seconds,
    -- Freshness status
    CASE
        WHEN ld.last_update_ns IS NULL THEN 'no_data'
        WHEN EXTRACT(EPOCH FROM NOW() - ns_to_timestamp(ld.last_update_ns)) > 86400 THEN 'stale_critical'  -- >24h
        WHEN EXTRACT(EPOCH FROM NOW() - ns_to_timestamp(ld.last_update_ns)) > 3600 THEN 'stale_warning'   -- >1h
        WHEN EXTRACT(EPOCH FROM NOW() - ns_to_timestamp(ld.last_update_ns)) > 300 THEN 'delayed'          -- >5min
        ELSE 'fresh'
    END as freshness_status,
    -- Is this instrument being actively processed?
    CASE
        WHEN ld.records_last_hour > 0 THEN TRUE
        ELSE FALSE
    END as is_active
FROM all_instruments ai
LEFT JOIN latest_data ld
    ON ai.instrument_id = ld.instrument_id
    AND ai.feature_set_id = ld.feature_set_id
ORDER BY staleness_seconds DESC, ai.instrument_id;

-- ============================================================================
-- Error and Issue Summary
-- ============================================================================

-- Track errors and data quality issues
CREATE OR REPLACE VIEW ml.error_summary AS
WITH prediction_errors AS (
    SELECT
        'prediction' as error_type,
        model_id as component,
        DATE(ns_to_timestamp(ts_event)) as error_date,
        COUNT(*) as error_count,
        AVG(inference_time_ms) as avg_latency_ms
    FROM ml_model_predictions
    WHERE ts_event > timestamp_to_ns(NOW() - INTERVAL '24 hours')
        AND (confidence < 0.5 OR inference_time_ms > 1000)
    GROUP BY model_id, DATE(ns_to_timestamp(ts_event))
),
signal_errors AS (
    SELECT
        'signal' as error_type,
        strategy_id as component,
        DATE(ns_to_timestamp(ts_event)) as error_date,
        COUNT(*) as error_count,
        0 as avg_latency_ms
    FROM ml_strategy_signals
    WHERE ts_event > timestamp_to_ns(NOW() - INTERVAL '24 hours')
        AND strength < 0.3  -- Weak signals might indicate issues
    GROUP BY strategy_id, DATE(ns_to_timestamp(ts_event))
),
feature_errors AS (
    SELECT
        'feature' as error_type,
        feature_set_id as component,
        DATE(ns_to_timestamp(ts_event)) as error_date,
        COUNT(*) as error_count,
        0 as avg_latency_ms
    FROM ml_feature_values
    WHERE ts_event > timestamp_to_ns(NOW() - INTERVAL '24 hours')
        AND jsonb_typeof(values) = 'null'
    GROUP BY feature_set_id, DATE(ns_to_timestamp(ts_event))
)
SELECT
    error_type,
    component,
    error_date,
    error_count,
    avg_latency_ms,
    -- Calculate error rate trend
    CASE
        WHEN error_count > 100 THEN 'critical'
        WHEN error_count > 10 THEN 'warning'
        ELSE 'normal'
    END as severity
FROM (
    SELECT * FROM prediction_errors
    UNION ALL
    SELECT * FROM signal_errors
    UNION ALL
    SELECT * FROM feature_errors
) combined_errors
ORDER BY error_date DESC, error_count DESC;

-- ============================================================================
-- Model Performance Summary
-- ============================================================================

-- Track model inference performance and health
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
    FROM ml_model_predictions
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
    -- Model health assessment
    CASE
        WHEN avg_confidence < 0.5 THEN 'low_confidence'
        WHEN p99_inference_ms > 1000 THEN 'slow_inference'
        WHEN prediction_stddev < 0.01 THEN 'low_variance'  -- Model might be stuck
        ELSE 'healthy'
    END as health_status
FROM model_stats
ORDER BY date DESC, model_id, instrument_id;

-- ============================================================================
-- Strategy Signal Summary
-- ============================================================================

-- Track strategy signal generation and quality
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
        -- Extract risk metrics from JSONB
        AVG((risk_metrics->>'sharpe_ratio')::FLOAT) as avg_sharpe,
        AVG((risk_metrics->>'max_drawdown')::FLOAT) as avg_max_drawdown
    FROM ml_strategy_signals
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
    -- Signal quality assessment
    CASE
        WHEN avg_strength < 0.3 THEN 'weak_signals'
        WHEN signal_count < 10 THEN 'low_activity'
        WHEN COALESCE(avg_sharpe, 0) < 0 THEN 'negative_sharpe'
        ELSE 'healthy'
    END as signal_quality
FROM signal_stats
ORDER BY date DESC, strategy_id, instrument_id;

-- ============================================================================
-- Pipeline Processing Summary
-- ============================================================================

-- Overall pipeline processing statistics
CREATE OR REPLACE VIEW ml.pipeline_processing_summary AS
WITH daily_stats AS (
    SELECT
        DATE(ns_to_timestamp(ts_event)) as date,
        'features' as stage,
        COUNT(*) as records_processed,
        COUNT(DISTINCT instrument_id) as unique_instruments,
        COUNT(DISTINCT feature_set_id) as unique_components
    FROM ml_feature_values
    WHERE ts_event > timestamp_to_ns(NOW() - INTERVAL '7 days')
    GROUP BY DATE(ns_to_timestamp(ts_event))

    UNION ALL

    SELECT
        DATE(ns_to_timestamp(ts_event)) as date,
        'predictions' as stage,
        COUNT(*) as records_processed,
        COUNT(DISTINCT instrument_id) as unique_instruments,
        COUNT(DISTINCT model_id) as unique_components
    FROM ml_model_predictions
    WHERE ts_event > timestamp_to_ns(NOW() - INTERVAL '7 days')
    GROUP BY DATE(ns_to_timestamp(ts_event))

    UNION ALL

    SELECT
        DATE(ns_to_timestamp(ts_event)) as date,
        'signals' as stage,
        COUNT(*) as records_processed,
        COUNT(DISTINCT instrument_id) as unique_instruments,
        COUNT(DISTINCT strategy_id) as unique_components
    FROM ml_strategy_signals
    WHERE ts_event > timestamp_to_ns(NOW() - INTERVAL '7 days')
    GROUP BY DATE(ns_to_timestamp(ts_event))
)
SELECT
    date,
    stage,
    records_processed,
    unique_instruments,
    unique_components,
    -- Calculate daily processing rate
    ROUND(records_processed::NUMERIC / 86400, 2) as records_per_second,
    -- Stage health
    CASE
        WHEN records_processed = 0 THEN 'no_data'
        WHEN records_processed < 1000 THEN 'low_volume'
        ELSE 'normal'
    END as processing_status
FROM daily_stats
ORDER BY date DESC, stage;

-- ============================================================================
-- Data Quality Metrics
-- ============================================================================

-- Comprehensive data quality tracking
CREATE OR REPLACE VIEW ml.data_quality_metrics AS
WITH quality_checks AS (
    SELECT
        'features' as data_type,
        feature_set_id as component_id,
        instrument_id,
        DATE(ns_to_timestamp(ts_event)) as date,
        COUNT(*) as total_records,
        SUM(CASE WHEN jsonb_typeof(values) = 'null' THEN 1 ELSE 0 END) as null_count,
        SUM(CASE WHEN NOT is_live THEN 1 ELSE 0 END) as historical_count,
        SUM(CASE WHEN is_live THEN 1 ELSE 0 END) as live_count
    FROM ml_feature_values
    WHERE ts_event > timestamp_to_ns(NOW() - INTERVAL '24 hours')
    GROUP BY feature_set_id, instrument_id, DATE(ns_to_timestamp(ts_event))
)
SELECT
    data_type,
    component_id,
    instrument_id,
    date,
    total_records,
    null_count,
    historical_count,
    live_count,
    -- Calculate quality score
    ROUND(((total_records - null_count)::NUMERIC / NULLIF(total_records, 0)) * 100, 2) as quality_percentage,
    -- Data mix ratio
    ROUND((live_count::NUMERIC / NULLIF(total_records, 0)) * 100, 2) as live_data_percentage,
    -- Quality status
    CASE
        WHEN (null_count::NUMERIC / NULLIF(total_records, 0)) > 0.1 THEN 'poor_quality'
        WHEN (null_count::NUMERIC / NULLIF(total_records, 0)) > 0.05 THEN 'needs_attention'
        ELSE 'good_quality'
    END as quality_status
FROM quality_checks
ORDER BY date DESC, quality_percentage ASC;

-- ============================================================================
-- Indexes for Performance
-- ============================================================================

-- Create indexes if they don't exist for optimal view performance
CREATE INDEX IF NOT EXISTS idx_ml_feature_values_health
    ON ml_feature_values(ts_event, instrument_id, feature_set_id)
    WHERE ts_event > timestamp_to_ns(NOW() - INTERVAL '7 days');

CREATE INDEX IF NOT EXISTS idx_ml_model_predictions_health
    ON ml_model_predictions(ts_event, model_id, instrument_id)
    WHERE ts_event > timestamp_to_ns(NOW() - INTERVAL '7 days');

CREATE INDEX IF NOT EXISTS idx_ml_strategy_signals_health
    ON ml_strategy_signals(ts_event, strategy_id, instrument_id)
    WHERE ts_event > timestamp_to_ns(NOW() - INTERVAL '7 days');

-- ============================================================================
-- View Documentation
-- ============================================================================

COMMENT ON VIEW ml.pipeline_health IS 'Overall ML pipeline health status with daily aggregates';
COMMENT ON VIEW ml.data_collection_stats IS 'Data collection metrics per instrument and hour';
COMMENT ON VIEW ml.feature_computation_stats IS 'Feature computation performance and quality metrics';
COMMENT ON VIEW ml.data_freshness IS 'Monitor data staleness and freshness per instrument';
COMMENT ON VIEW ml.error_summary IS 'Summary of errors and issues across all ML components';
COMMENT ON VIEW ml.model_performance_summary IS 'Model inference performance and health metrics';
COMMENT ON VIEW ml.strategy_signal_summary IS 'Strategy signal generation and quality metrics';
COMMENT ON VIEW ml.pipeline_processing_summary IS 'Overall pipeline processing statistics by stage';
COMMENT ON VIEW ml.data_quality_metrics IS 'Comprehensive data quality tracking and scoring';
