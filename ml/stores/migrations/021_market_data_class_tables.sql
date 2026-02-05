-- Add per-data-class market data tables (bars, quotes, trades, mbp1, tbbo).
-- Existing market_data table is preserved for legacy compatibility.

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

CREATE TABLE IF NOT EXISTS market_data_mbp10 (
    instrument_id VARCHAR(100) NOT NULL,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,
    bids JSONB,
    asks JSONB,
    bid_counts JSONB,
    ask_counts JSONB,
    flags INTEGER DEFAULT 0,
    sequence BIGINT NOT NULL DEFAULT 0,
    source VARCHAR(50),
    quality_flags INTEGER DEFAULT 0,
    source_dataset VARCHAR(100),
    PRIMARY KEY (instrument_id, ts_event, sequence)
) PARTITION BY RANGE (ts_event);

CREATE TABLE IF NOT EXISTS market_data_mbp10_default
    PARTITION OF market_data_mbp10 DEFAULT;

CREATE TABLE IF NOT EXISTS market_data_mbo (
    instrument_id VARCHAR(100) NOT NULL,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,
    action VARCHAR(16),
    order_payload JSONB,
    flags INTEGER DEFAULT 0,
    sequence BIGINT NOT NULL DEFAULT 0,
    source VARCHAR(50),
    quality_flags INTEGER DEFAULT 0,
    source_dataset VARCHAR(100),
    PRIMARY KEY (instrument_id, ts_event, sequence)
) PARTITION BY RANGE (ts_event);

CREATE TABLE IF NOT EXISTS market_data_mbo_default
    PARTITION OF market_data_mbo DEFAULT;

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

CREATE INDEX IF NOT EXISTS idx_market_data_mbp10_time
    ON market_data_mbp10 USING BRIN (ts_event);
CREATE INDEX IF NOT EXISTS idx_market_data_mbp10_instrument
    ON market_data_mbp10 (instrument_id, ts_event DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_mbp10_quality
    ON market_data_mbp10 (quality_flags) WHERE quality_flags > 0;

CREATE INDEX IF NOT EXISTS idx_market_data_mbo_time
    ON market_data_mbo USING BRIN (ts_event);
CREATE INDEX IF NOT EXISTS idx_market_data_mbo_instrument
    ON market_data_mbo (instrument_id, ts_event DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_mbo_quality
    ON market_data_mbo (quality_flags) WHERE quality_flags > 0;

CREATE INDEX IF NOT EXISTS idx_market_data_trade_tick_time
    ON market_data_trade_tick USING BRIN (ts_event);
CREATE INDEX IF NOT EXISTS idx_market_data_trade_tick_instrument
    ON market_data_trade_tick (instrument_id, ts_event DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_trade_tick_quality
    ON market_data_trade_tick (quality_flags) WHERE quality_flags > 0;

SELECT create_monthly_partitions('market_data_bar', '2023-01-01'::DATE, 60);
SELECT create_monthly_partitions('market_data_quote_tick', '2023-01-01'::DATE, 60);
SELECT create_monthly_partitions('market_data_tbbo', '2023-01-01'::DATE, 60);
SELECT create_monthly_partitions('market_data_mbp1', '2023-01-01'::DATE, 60);
SELECT create_monthly_partitions('market_data_mbp10', '2023-01-01'::DATE, 60);
SELECT create_monthly_partitions('market_data_mbo', '2023-01-01'::DATE, 60);
SELECT create_monthly_partitions('market_data_trade_tick', '2023-01-01'::DATE, 60);
