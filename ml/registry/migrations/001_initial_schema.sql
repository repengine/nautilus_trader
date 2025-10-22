-- Nautilus Trader ML Registry Database Schema
-- Version: 1.0.0
-- Description: Initial schema for model, feature, and strategy registries

-- Create schema if not exists
CREATE SCHEMA IF NOT EXISTS ml_registry;

-- Set search path
SET search_path TO ml_registry, public;

-- Models table
CREATE TABLE IF NOT EXISTS models (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(255) UNIQUE NOT NULL,
    role VARCHAR(50) NOT NULL,
    data_requirements VARCHAR(50) NOT NULL,
    architecture VARCHAR(100) NOT NULL,
    feature_schema JSONB NOT NULL,
    feature_schema_hash VARCHAR(64) NOT NULL,
    parent_id VARCHAR(255),
    children_ids TEXT[],
    training_config JSONB,
    performance_metrics JSONB,
    deployment_constraints JSONB,
    deployment_status VARCHAR(50) NOT NULL,
    deployed_to TEXT[],
    version VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_modified TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB,
    model_path TEXT NOT NULL,
    performance_history JSONB
);

-- Create indexes for models
CREATE INDEX IF NOT EXISTS idx_model_role ON models(role);
CREATE INDEX IF NOT EXISTS idx_model_parent ON models(parent_id);
CREATE INDEX IF NOT EXISTS idx_model_created ON models(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_model_deployment_status ON models(deployment_status);
CREATE INDEX IF NOT EXISTS idx_model_architecture ON models(architecture);
CREATE INDEX IF NOT EXISTS idx_model_feature_schema_hash ON models(feature_schema_hash);

-- Features table
CREATE TABLE IF NOT EXISTS features (
    id SERIAL PRIMARY KEY,
    feature_set_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    role VARCHAR(50) NOT NULL,
    data_requirements VARCHAR(50) NOT NULL,
    feature_names TEXT[],
    feature_dtypes TEXT[],
    schema_hash VARCHAR(64) NOT NULL,
    pipeline_signature VARCHAR(255),
    pipeline_version VARCHAR(50),
    capability_flags JSONB,
    constraints JSONB,
    parity_tolerance FLOAT,
    parity_digest JSONB,
    perf_digest JSONB,
    parent_feature_set_id VARCHAR(255),
    stage VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_modified TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

-- Create indexes for features
CREATE INDEX IF NOT EXISTS idx_feature_stage ON features(stage);
CREATE INDEX IF NOT EXISTS idx_feature_role ON features(role);
CREATE INDEX IF NOT EXISTS idx_feature_created ON features(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feature_schema_hash ON features(schema_hash);
CREATE INDEX IF NOT EXISTS idx_feature_parent ON features(parent_feature_set_id);

-- Strategies table
CREATE TABLE IF NOT EXISTS strategies (
    id SERIAL PRIMARY KEY,
    strategy_id VARCHAR(255) UNIQUE NOT NULL,
    strategy_type VARCHAR(50) NOT NULL,
    version VARCHAR(50) NOT NULL,
    required_models TEXT[],
    required_features TEXT[],
    suitable_regimes TEXT[],
    instrument_types TEXT[],
    timeframe_range VARCHAR(100),
    max_position_size FLOAT,
    max_leverage FLOAT,
    max_drawdown FLOAT,
    stop_loss_type VARCHAR(50),
    min_sharpe_ratio FLOAT,
    min_win_rate FLOAT,
    max_correlation_with_portfolio FLOAT,
    parent_strategy_id VARCHAR(255),
    incompatible_strategies TEXT[],
    config_schema JSONB,
    default_config JSONB,
    backtest_metrics JSONB,
    live_metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_modified TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    author VARCHAR(255),
    description TEXT
);

-- Create indexes for strategies
CREATE INDEX IF NOT EXISTS idx_strategy_type ON strategies(strategy_type);
CREATE INDEX IF NOT EXISTS idx_strategy_created ON strategies(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_strategy_parent ON strategies(parent_strategy_id);
CREATE INDEX IF NOT EXISTS idx_strategy_author ON strategies(author);

-- Audit log for all changes
CREATE TABLE IF NOT EXISTS registry_audit_log (
    id SERIAL PRIMARY KEY,
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(255) NOT NULL,
    action VARCHAR(50) NOT NULL,
    changes JSONB,
    user_id VARCHAR(255),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for audit log
CREATE INDEX IF NOT EXISTS idx_audit_entity ON registry_audit_log(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON registry_audit_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_action ON registry_audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_user ON registry_audit_log(user_id);

-- Create views for common queries

DROP VIEW IF EXISTS active_models CASCADE;
CREATE VIEW active_models AS
SELECT
    m.*,
    COUNT(DISTINCT c.model_id) as num_children,
    p.model_id as parent_model_id,
    p.role as parent_role
FROM models m
LEFT JOIN models c ON m.model_id = ANY(c.children_ids)
LEFT JOIN models p ON m.parent_id = p.model_id
WHERE m.deployment_status = 'active'
GROUP BY m.id, p.model_id, p.role;

DROP VIEW IF EXISTS feature_lineage CASCADE;
CREATE VIEW feature_lineage AS
WITH RECURSIVE feature_tree AS (
    SELECT
        feature_set_id,
        name,
        version,
        parent_feature_set_id,
        0::integer AS level,
        ARRAY[feature_set_id]::text[] AS path
    FROM features
    WHERE parent_feature_set_id IS NULL

    UNION ALL

    SELECT
        f.feature_set_id,
        f.name,
        f.version,
        f.parent_feature_set_id,
        ft.level + 1,
        array_append(ft.path, f.feature_set_id)
    FROM features f
    JOIN feature_tree ft ON f.parent_feature_set_id = ft.feature_set_id
)
SELECT
    feature_set_id,
    name,
    version,
    parent_feature_set_id,
    level,
    path
FROM feature_tree;

-- Strategy compatibility view
DROP VIEW IF EXISTS strategy_compatibility CASCADE;
CREATE VIEW strategy_compatibility AS
SELECT
    s1.strategy_id as strategy_1,
    s2.strategy_id as strategy_2,
    CASE
        WHEN s2.strategy_id = ANY(s1.incompatible_strategies) THEN 'incompatible'
        WHEN s1.strategy_id = ANY(s2.incompatible_strategies) THEN 'incompatible'
        ELSE 'compatible'
    END as compatibility
FROM strategies s1
CROSS JOIN strategies s2
WHERE s1.strategy_id != s2.strategy_id;

-- Create functions for common operations

-- Function to update last_modified timestamp
CREATE OR REPLACE FUNCTION update_last_modified()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_modified = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for automatic timestamp updates
DROP TRIGGER IF EXISTS update_models_last_modified ON models;
CREATE TRIGGER update_models_last_modified
    BEFORE UPDATE ON models
    FOR EACH ROW
    EXECUTE FUNCTION update_last_modified();

DROP TRIGGER IF EXISTS update_features_last_modified ON features;
CREATE TRIGGER update_features_last_modified
    BEFORE UPDATE ON features
    FOR EACH ROW
    EXECUTE FUNCTION update_last_modified();

DROP TRIGGER IF EXISTS update_strategies_last_modified ON strategies;
CREATE TRIGGER update_strategies_last_modified
    BEFORE UPDATE ON strategies
    FOR EACH ROW
    EXECUTE FUNCTION update_last_modified();

-- Function to get model dependencies
CREATE OR REPLACE FUNCTION get_model_dependencies(p_model_id VARCHAR(255))
RETURNS TABLE(
    dependency_type VARCHAR(50),
    dependency_id VARCHAR(255),
    dependency_name VARCHAR(255)
) AS $$
BEGIN
    -- Get parent model
    RETURN QUERY
    SELECT
        'parent_model'::VARCHAR(50) as dependency_type,
        m.parent_id as dependency_id,
        p.architecture as dependency_name
    FROM models m
    LEFT JOIN models p ON m.parent_id = p.model_id
    WHERE m.model_id = p_model_id AND m.parent_id IS NOT NULL;

    -- Get required features (from strategies that use this model)
    RETURN QUERY
    SELECT DISTINCT
        'feature'::VARCHAR(50) as dependency_type,
        unnest(s.required_features) as dependency_id,
        f.name as dependency_name
    FROM strategies s
    LEFT JOIN features f ON f.feature_set_id = ANY(s.required_features)
    WHERE p_model_id = ANY(s.required_models);
END;
$$ LANGUAGE plpgsql;

-- Function to validate model registration
CREATE OR REPLACE FUNCTION validate_model_registration()
RETURNS TRIGGER AS $$
BEGIN
    -- Check if parent exists when specified
    IF NEW.parent_id IS NOT NULL THEN
        IF NOT EXISTS (SELECT 1 FROM models WHERE model_id = NEW.parent_id) THEN
            RAISE EXCEPTION 'Parent model % does not exist', NEW.parent_id;
        END IF;
    END IF;

    -- Validate deployment status
    IF NEW.deployment_status NOT IN ('inactive', 'active', 'testing', 'retired', 'failed') THEN
        RAISE EXCEPTION 'Invalid deployment status: %', NEW.deployment_status;
    END IF;

    -- Validate role
    IF NEW.role NOT IN ('teacher', 'student', 'inference', 'ensemble', 'feature') THEN
        RAISE EXCEPTION 'Invalid model role: %', NEW.role;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create validation trigger for models
DROP TRIGGER IF EXISTS validate_model_before_insert ON models;
CREATE TRIGGER validate_model_before_insert
    BEFORE INSERT ON models
    FOR EACH ROW
    EXECUTE FUNCTION validate_model_registration();

-- Grant permissions (adjust as needed for your setup)
GRANT ALL ON SCHEMA ml_registry TO postgres;
GRANT ALL ON ALL TABLES IN SCHEMA ml_registry TO postgres;
GRANT ALL ON ALL SEQUENCES IN SCHEMA ml_registry TO postgres;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA ml_registry TO postgres;

-- Restore canonical search path for subsequent migrations.
SET search_path TO public, pg_catalog, ml_registry;
