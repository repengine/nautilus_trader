-- Cross-Asset Relationship Features Schema
-- Storage for cross-asset beta, spreads, and correlation features
--
-- Design principles:
-- - All timestamps in nanoseconds (ts_event, ts_init) per Nautilus convention
-- - Partitioned by ts_event for time-range queries
-- - instrument_id fields for joinability with bar/quote data
-- - JSONB for flexible feature storage
-- - Indexes optimized for hot path queries

-- ============================================================================
-- Cross-Asset Beta Features Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS ml_cross_asset_betas (
    id BIGSERIAL,
    feature_set_id VARCHAR(255) NOT NULL,
    asset_id VARCHAR(100) NOT NULL,       -- Primary asset instrument_id
    benchmark_id VARCHAR(100) NOT NULL,   -- Benchmark/market instrument_id
    ts_event BIGINT NOT NULL,             -- Event timestamp (nanoseconds)
    ts_init BIGINT NOT NULL,              -- Initialization timestamp (nanoseconds)

    -- Beta metrics
    ewma_beta DOUBLE PRECISION NOT NULL,
    ewma_cov DOUBLE PRECISION NOT NULL,
    ewma_var_market DOUBLE PRECISION NOT NULL,

    -- State metadata
    n_observations INTEGER NOT NULL,      -- Number of observations used
    alpha DOUBLE PRECISION NOT NULL,      -- EWMA decay factor

    -- Additional features (optional)
    features JSONB,                       -- Extended features as key-value

    -- Metadata
    is_live BOOLEAN DEFAULT FALSE,
    source VARCHAR(50),                   -- 'historical', 'live', 'backfill'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    PRIMARY KEY (id, ts_event)
) PARTITION BY RANGE (ts_event);

-- Create partitions for 2024-2026 (36 months)
-- Note: create_monthly_partitions function should be available from main schema migration
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_proc WHERE proname = 'create_monthly_partitions'
    ) THEN
        PERFORM create_monthly_partitions('ml_cross_asset_betas', '2024-01-01'::DATE, 36);
    END IF;
END $$;

-- Default partition for dates outside pre-created ranges
CREATE TABLE IF NOT EXISTS ml_cross_asset_betas_default
    PARTITION OF ml_cross_asset_betas DEFAULT;

-- Indexes for efficient lookups
CREATE INDEX IF NOT EXISTS idx_ml_cross_asset_betas_lookup
    ON ml_cross_asset_betas (asset_id, benchmark_id, ts_event);

CREATE INDEX IF NOT EXISTS idx_ml_cross_asset_betas_feature_set
    ON ml_cross_asset_betas (feature_set_id, ts_event);

CREATE INDEX IF NOT EXISTS idx_ml_cross_asset_betas_live
    ON ml_cross_asset_betas (is_live) WHERE is_live = TRUE;

-- Unique constraint for upserts
CREATE UNIQUE INDEX IF NOT EXISTS uq_ml_cross_asset_betas_key
    ON ml_cross_asset_betas (feature_set_id, asset_id, benchmark_id, ts_event);

-- ============================================================================
-- Cross-Asset Spread Features Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS ml_cross_asset_spreads (
    id BIGSERIAL,
    feature_set_id VARCHAR(255) NOT NULL,
    asset_a_id VARCHAR(100) NOT NULL,     -- First asset instrument_id
    asset_b_id VARCHAR(100) NOT NULL,     -- Second asset instrument_id
    ts_event BIGINT NOT NULL,             -- Event timestamp (nanoseconds)
    ts_init BIGINT NOT NULL,              -- Initialization timestamp (nanoseconds)

    -- Spread metrics
    spread DOUBLE PRECISION NOT NULL,     -- Raw spread (price_a - price_b)
    zscore DOUBLE PRECISION NOT NULL,     -- Z-scored spread
    spread_mean DOUBLE PRECISION NOT NULL,
    spread_std DOUBLE PRECISION NOT NULL,

    -- State metadata
    n_observations INTEGER NOT NULL,      -- Number of observations used

    -- Additional features (optional)
    features JSONB,                       -- Extended features as key-value

    -- Metadata
    is_live BOOLEAN DEFAULT FALSE,
    source VARCHAR(50),                   -- 'historical', 'live', 'backfill'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    PRIMARY KEY (id, ts_event)
) PARTITION BY RANGE (ts_event);

-- Create partitions for 2024-2026 (36 months)
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_proc WHERE proname = 'create_monthly_partitions'
    ) THEN
        PERFORM create_monthly_partitions('ml_cross_asset_spreads', '2024-01-01'::DATE, 36);
    END IF;
END $$;

-- Default partition for dates outside pre-created ranges
CREATE TABLE IF NOT EXISTS ml_cross_asset_spreads_default
    PARTITION OF ml_cross_asset_spreads DEFAULT;

-- Indexes for efficient lookups
CREATE INDEX IF NOT EXISTS idx_ml_cross_asset_spreads_lookup
    ON ml_cross_asset_spreads (asset_a_id, asset_b_id, ts_event);

CREATE INDEX IF NOT EXISTS idx_ml_cross_asset_spreads_feature_set
    ON ml_cross_asset_spreads (feature_set_id, ts_event);

CREATE INDEX IF NOT EXISTS idx_ml_cross_asset_spreads_live
    ON ml_cross_asset_spreads (is_live) WHERE is_live = TRUE;

-- Unique constraint for upserts
CREATE UNIQUE INDEX IF NOT EXISTS uq_ml_cross_asset_spreads_key
    ON ml_cross_asset_spreads (feature_set_id, asset_a_id, asset_b_id, ts_event);

-- ============================================================================
-- Cross-Asset Correlation Features Table (Optional)
-- ============================================================================

CREATE TABLE IF NOT EXISTS ml_cross_asset_correlations (
    id BIGSERIAL,
    feature_set_id VARCHAR(255) NOT NULL,
    asset_a_id VARCHAR(100) NOT NULL,     -- First asset instrument_id
    asset_b_id VARCHAR(100) NOT NULL,     -- Second asset instrument_id
    ts_event BIGINT NOT NULL,             -- Event timestamp (nanoseconds)
    ts_init BIGINT NOT NULL,              -- Initialization timestamp (nanoseconds)

    -- Correlation metrics
    correlation DOUBLE PRECISION NOT NULL, -- Rolling correlation coefficient

    -- State metadata
    n_observations INTEGER NOT NULL,       -- Window size used
    window_type VARCHAR(50) NOT NULL,     -- 'expanding', 'rolling', 'ewma'

    -- Additional features (optional)
    features JSONB,                        -- Extended features as key-value

    -- Metadata
    is_live BOOLEAN DEFAULT FALSE,
    source VARCHAR(50),                    -- 'historical', 'live', 'backfill'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    PRIMARY KEY (id, ts_event)
) PARTITION BY RANGE (ts_event);

-- Create partitions for 2024-2026 (36 months)
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_proc WHERE proname = 'create_monthly_partitions'
    ) THEN
        PERFORM create_monthly_partitions('ml_cross_asset_correlations', '2024-01-01'::DATE, 36);
    END IF;
END $$;

-- Default partition for dates outside pre-created ranges
CREATE TABLE IF NOT EXISTS ml_cross_asset_correlations_default
    PARTITION OF ml_cross_asset_correlations DEFAULT;

-- Indexes for efficient lookups
CREATE INDEX IF NOT EXISTS idx_ml_cross_asset_correlations_lookup
    ON ml_cross_asset_correlations (asset_a_id, asset_b_id, ts_event);

CREATE INDEX IF NOT EXISTS idx_ml_cross_asset_correlations_feature_set
    ON ml_cross_asset_correlations (feature_set_id, ts_event);

CREATE INDEX IF NOT EXISTS idx_ml_cross_asset_correlations_live
    ON ml_cross_asset_correlations (is_live) WHERE is_live = TRUE;

-- Unique constraint for upserts
CREATE UNIQUE INDEX IF NOT EXISTS uq_ml_cross_asset_correlations_key
    ON ml_cross_asset_correlations (feature_set_id, asset_a_id, asset_b_id, ts_event);

-- ============================================================================
-- Comments for Documentation
-- ============================================================================

COMMENT ON TABLE ml_cross_asset_betas IS
'Stores EWMA beta features for cross-asset relationships. Beta measures systematic risk exposure of an asset relative to a benchmark/market.';

COMMENT ON COLUMN ml_cross_asset_betas.ewma_beta IS
'Exponentially weighted moving average beta (cov / var_market)';

COMMENT ON COLUMN ml_cross_asset_betas.alpha IS
'EWMA decay factor. Common values: 0.94 (RiskMetrics), 0.97';

COMMENT ON TABLE ml_cross_asset_spreads IS
'Stores z-scored spread features for pairs trading and statistical arbitrage.';

COMMENT ON COLUMN ml_cross_asset_spreads.zscore IS
'Standardized spread: (spread - mean) / std. Values >2 or <-2 indicate extreme divergence.';

COMMENT ON TABLE ml_cross_asset_correlations IS
'Stores rolling correlation features for cross-asset relationship monitoring.';
