-- Migration: Materialize monitoring views and helper functions for ML schemas.
-- Rollback: DROP VIEW/ FUNCTION statements for the objects created in this file.

-- Views Migration (008)
-- Moves monitoring views from ml/schema/pipeline_health.sql into migrations
-- and qualifies base tables with public schema explicitly.

-- Helper functions for time conversion
CREATE SCHEMA IF NOT EXISTS ml;

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

-- Pipeline health overview
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

-- Data collection stats (hourly)
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

-- Model performance summary
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

-- Strategy signal summary
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
