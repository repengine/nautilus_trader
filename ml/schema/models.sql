-- Model predictions and metadata schema for ML module
-- Stores model predictions, performance metrics, and metadata

-- Switch to ML schema
SET search_path TO ml, public;

-- Model predictions table
CREATE TABLE IF NOT EXISTS ml_model_predictions (
    id BIGSERIAL PRIMARY KEY,
    model_id VARCHAR(64) NOT NULL,
    instrument_id VARCHAR(64) NOT NULL,
    prediction FLOAT NOT NULL,
    confidence FLOAT,
    features JSONB,  -- Feature values used for this prediction
    metadata JSONB,  -- Additional metadata
    ts_event BIGINT NOT NULL,  -- Event timestamp (nanoseconds)
    ts_init BIGINT NOT NULL,    -- Initialization timestamp
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for model predictions
CREATE INDEX IF NOT EXISTS idx_ml_predictions_model_instrument 
    ON ml_model_predictions(model_id, instrument_id, ts_event DESC);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_time 
    ON ml_model_predictions(ts_event DESC);

-- Model metadata table
CREATE TABLE IF NOT EXISTS ml_model_metadata (
    model_id VARCHAR(64) PRIMARY KEY,
    model_version VARCHAR(32),
    model_type VARCHAR(64),  -- xgboost, lightgbm, neural_network, etc.
    feature_schema JSONB,     -- Expected feature names and types
    training_metadata JSONB,  -- Training parameters, metrics, etc.
    deployment_status VARCHAR(32) DEFAULT 'inactive',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Model performance metrics
CREATE TABLE IF NOT EXISTS ml_model_performance (
    id BIGSERIAL PRIMARY KEY,
    model_id VARCHAR(64) NOT NULL,
    instrument_id VARCHAR(64),
    metric_name VARCHAR(64) NOT NULL,  -- accuracy, sharpe_ratio, etc.
    metric_value FLOAT NOT NULL,
    evaluation_period_start TIMESTAMP WITH TIME ZONE,
    evaluation_period_end TIMESTAMP WITH TIME ZONE,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES ml_model_metadata(model_id)
);

-- Comments
COMMENT ON TABLE ml_model_predictions IS 'Stores all model predictions for auditing and analysis';
COMMENT ON TABLE ml_model_metadata IS 'Model registry with metadata and configuration';
COMMENT ON TABLE ml_model_performance IS 'Model performance metrics over time';

-- Grant permissions
GRANT ALL ON ALL TABLES IN SCHEMA ml TO postgres;
GRANT ALL ON ALL SEQUENCES IN SCHEMA ml TO postgres;