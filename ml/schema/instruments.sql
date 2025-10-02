-- Instrument Metadata Schema
--
-- This schema provides temporal versioning of instrument metadata for factor mapping,
-- enabling dynamic assignment of instruments to duration buckets, issuer types, and
-- liquidity tiers based on research findings.
--
-- Key features:
-- - Temporal versioning via ts_event and ts_init (nanoseconds since epoch)
-- - Factor mappings: duration bucket, issuer type, liquidity tier
-- - Efficient joins with market data via instrument_id, ts_event, ts_init
-- - Partitioned by month for scalability
-- - BRIN indexes for time-series access patterns

-- =============================================================================
-- Core Metadata Table
-- =============================================================================

CREATE TABLE IF NOT EXISTS ml.instrument_metadata (
    instrument_id TEXT NOT NULL,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,

    -- Factor mappings (research-driven categorization)
    duration_bucket SMALLINT NOT NULL,  -- 0=Short (0-2Y), 1=Medium (2-7Y), 2=Long (7Y+)
    issuer_type SMALLINT NOT NULL,      -- 0=SOVEREIGN, 1=QUASI_SOVEREIGN, 2=CORPORATE_IG, 3=CORPORATE_HY
    liquidity_tier SMALLINT NOT NULL,   -- 1=High, 2=Medium, 3=Low

    -- Additional metadata (optional, extensible)
    region TEXT,                        -- e.g., 'US', 'EU', 'ASIA'
    sector TEXT,                        -- e.g., 'TREASURY', 'AGENCY', 'CORPORATE'
    rating TEXT,                        -- Credit rating if applicable

    -- Temporal metadata
    valid_from_ns BIGINT NOT NULL,     -- Start of validity period (nanoseconds)
    valid_until_ns BIGINT,             -- End of validity period (NULL = current)

    -- Audit fields
    created_at_ns BIGINT NOT NULL,
    updated_at_ns BIGINT NOT NULL,

    -- Primary key: unique per instrument and event time
    PRIMARY KEY (instrument_id, ts_event)
);

-- =============================================================================
-- Comments for Schema Documentation
-- =============================================================================

COMMENT ON TABLE ml.instrument_metadata IS
'Temporal instrument metadata for factor-based portfolio construction. Each row represents a snapshot of instrument characteristics at a specific point in time (ts_event).';

COMMENT ON COLUMN ml.instrument_metadata.instrument_id IS
'Nautilus InstrumentId (e.g., "US10Y.BOND", "AAPL.NASDAQ"). Joins to market data tables.';

COMMENT ON COLUMN ml.instrument_metadata.ts_event IS
'Event timestamp in nanoseconds (when metadata became effective). Part of temporal join key.';

COMMENT ON COLUMN ml.instrument_metadata.ts_init IS
'Initialization timestamp in nanoseconds (when metadata was first created/ingested). Part of temporal join key.';

COMMENT ON COLUMN ml.instrument_metadata.duration_bucket IS
'Duration classification: 0=Short (0-2 years), 1=Medium (2-7 years), 2=Long (7+ years). Used for factor assignment.';

COMMENT ON COLUMN ml.instrument_metadata.issuer_type IS
'Issuer classification: 0=SOVEREIGN, 1=QUASI_SOVEREIGN (e.g., agencies), 2=CORPORATE_IG (investment grade), 3=CORPORATE_HY (high yield). Used for credit factor exposure.';

COMMENT ON COLUMN ml.instrument_metadata.liquidity_tier IS
'Liquidity classification: 1=High (on-the-run, benchmark bonds), 2=Medium (regular trading), 3=Low (infrequent trading). Used for liquidity factor exposure.';

COMMENT ON COLUMN ml.instrument_metadata.valid_from_ns IS
'Start of metadata validity period (nanoseconds). Use for point-in-time queries.';

COMMENT ON COLUMN ml.instrument_metadata.valid_until_ns IS
'End of metadata validity period (nanoseconds). NULL indicates currently valid metadata.';

-- =============================================================================
-- Indexes for Time-Series Access Patterns
-- =============================================================================

-- BRIN index on ts_event for efficient time-range scans
-- BRIN is ideal for time-series data as it clusters naturally by insertion order
CREATE INDEX IF NOT EXISTS idx_instrument_metadata_ts_event
ON ml.instrument_metadata USING BRIN (ts_event);

-- BRIN index on ts_init for initialization time queries
CREATE INDEX IF NOT EXISTS idx_instrument_metadata_ts_init
ON ml.instrument_metadata USING BRIN (ts_init);

-- Composite index for point-in-time lookups (instrument + time range)
CREATE INDEX IF NOT EXISTS idx_instrument_metadata_instrument_ts
ON ml.instrument_metadata (instrument_id, ts_event DESC);

-- Index for validity period queries (find currently valid metadata)
CREATE INDEX IF NOT EXISTS idx_instrument_metadata_validity
ON ml.instrument_metadata (instrument_id, valid_from_ns, valid_until_ns)
WHERE valid_until_ns IS NULL;

-- Factor-based indexes for analytics
CREATE INDEX IF NOT EXISTS idx_instrument_metadata_duration
ON ml.instrument_metadata (duration_bucket, ts_event);

CREATE INDEX IF NOT EXISTS idx_instrument_metadata_issuer
ON ml.instrument_metadata (issuer_type, ts_event);

CREATE INDEX IF NOT EXISTS idx_instrument_metadata_liquidity
ON ml.instrument_metadata (liquidity_tier, ts_event);

-- =============================================================================
-- Partitioning Strategy (Monthly Range Partitions)
-- =============================================================================

-- Note: Partitioning will be managed by PartitionManager in ml/stores/infrastructure.py
-- The base table is created as a regular table; partitions are added dynamically.
--
-- Partition naming convention: ml.instrument_metadata_YYYY_MM
-- Example: ml.instrument_metadata_2024_01, ml.instrument_metadata_2024_02
--
-- Partitions are created for:
-- - Current month
-- - Next 3 months (to handle future-dated metadata)
-- - Historical months as needed
--
-- See ml/stores/infrastructure.py:PartitionManager for implementation details.

-- =============================================================================
-- Example Queries (Documentation)
-- =============================================================================

-- Query 1: Get current metadata for a specific instrument
-- SELECT * FROM ml.instrument_metadata
-- WHERE instrument_id = 'US10Y.BOND'
--   AND valid_until_ns IS NULL
-- ORDER BY ts_event DESC
-- LIMIT 1;

-- Query 2: Get metadata at a specific point in time
-- SELECT * FROM ml.instrument_metadata
-- WHERE instrument_id = 'US10Y.BOND'
--   AND ts_event <= :query_time_ns
--   AND (valid_until_ns IS NULL OR valid_until_ns > :query_time_ns)
-- ORDER BY ts_event DESC
-- LIMIT 1;

-- Query 3: Get all high-liquidity sovereign instruments
-- SELECT DISTINCT instrument_id
-- FROM ml.instrument_metadata
-- WHERE liquidity_tier = 1
--   AND issuer_type = 0
--   AND valid_until_ns IS NULL;

-- Query 4: Join with market data for factor analysis
-- SELECT
--     b.instrument_id,
--     b.ts_event,
--     b.close,
--     m.duration_bucket,
--     m.issuer_type,
--     m.liquidity_tier
-- FROM ml.bars b
-- INNER JOIN ml.instrument_metadata m
--   ON b.instrument_id = m.instrument_id
--   AND b.ts_event >= m.valid_from_ns
--   AND (m.valid_until_ns IS NULL OR b.ts_event < m.valid_until_ns)
-- WHERE b.ts_event BETWEEN :start_ns AND :end_ns;

-- =============================================================================
-- Migration Compatibility
-- =============================================================================

-- This schema is designed to be idempotent and can be run multiple times.
-- CREATE TABLE IF NOT EXISTS ensures no error on re-run.
-- Indexes are also created with IF NOT EXISTS for safety.
