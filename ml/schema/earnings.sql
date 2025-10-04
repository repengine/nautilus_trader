-- Earnings data schema for Nautilus Trader ML
-- This schema supports SEC EDGAR actuals, Yahoo Finance estimates, and earnings calendar
-- All tables include instrument_id, ts_event, ts_init for Nautilus compatibility

-- Table 1: Actual Earnings (from SEC EDGAR)
CREATE TABLE IF NOT EXISTS ml.earnings_actuals (
    ticker VARCHAR(20) NOT NULL,
    period_end DATE NOT NULL,           -- Quarter end date (e.g., 2024-09-30)
    filing_date DATE NOT NULL,          -- Date 10-Q was filed
    ts_event BIGINT NOT NULL,           -- Filing date in nanoseconds (for point-in-time)
    ts_init BIGINT NOT NULL,            -- Record creation timestamp

    -- Actual results (from EDGAR XBRL)
    eps_basic DOUBLE PRECISION,
    eps_diluted DOUBLE PRECISION,
    revenue DOUBLE PRECISION,           -- In dollars
    net_income DOUBLE PRECISION,
    operating_income DOUBLE PRECISION,
    shares_outstanding BIGINT,

    -- Metadata
    filing_type VARCHAR(10),            -- '10-Q' or '10-K'
    fiscal_year INTEGER,
    fiscal_quarter INTEGER,             -- 1, 2, 3, 4
    data_source VARCHAR(20) DEFAULT 'EDGAR',

    PRIMARY KEY (ticker, period_end)
);

CREATE INDEX IF NOT EXISTS idx_earnings_actuals_ts_event ON ml.earnings_actuals(ts_event);
CREATE INDEX IF NOT EXISTS idx_earnings_actuals_ticker ON ml.earnings_actuals(ticker);
CREATE INDEX IF NOT EXISTS idx_earnings_actuals_filing_date ON ml.earnings_actuals(filing_date);

COMMENT ON TABLE ml.earnings_actuals IS 'Historical earnings actuals from SEC EDGAR filings';
COMMENT ON COLUMN ml.earnings_actuals.ts_event IS 'Filing date in nanoseconds for point-in-time queries';

-- Table 2: Earnings Estimates (from Yahoo Finance)
CREATE TABLE IF NOT EXISTS ml.earnings_estimates (
    ticker VARCHAR(20) NOT NULL,
    estimate_date DATE NOT NULL,        -- Date estimate was recorded
    period_end DATE NOT NULL,           -- Quarter being estimated
    ts_event BIGINT NOT NULL,           -- Estimate date in nanoseconds
    ts_init BIGINT NOT NULL,

    -- Consensus estimates
    eps_consensus DOUBLE PRECISION,
    revenue_consensus DOUBLE PRECISION,
    num_analysts INTEGER,

    -- Metadata
    data_source VARCHAR(20) DEFAULT 'YAHOO',

    PRIMARY KEY (ticker, estimate_date, period_end)
);

CREATE INDEX IF NOT EXISTS idx_earnings_estimates_ts_event ON ml.earnings_estimates(ts_event);
CREATE INDEX IF NOT EXISTS idx_earnings_estimates_ticker ON ml.earnings_estimates(ticker);
CREATE INDEX IF NOT EXISTS idx_earnings_estimates_period ON ml.earnings_estimates(period_end);

COMMENT ON TABLE ml.earnings_estimates IS 'Consensus earnings estimates from Yahoo Finance';

-- Table 3: Earnings Calendar (upcoming announcements)
CREATE TABLE IF NOT EXISTS ml.earnings_calendar (
    ticker VARCHAR(20) NOT NULL,
    earnings_date TIMESTAMP NOT NULL,   -- Scheduled announcement date/time
    period_end DATE NOT NULL,           -- Quarter being reported
    ts_event BIGINT NOT NULL,           -- Calendar update time in nanoseconds
    ts_init BIGINT NOT NULL,

    -- Estimates for upcoming earnings
    eps_consensus DOUBLE PRECISION,
    revenue_consensus DOUBLE PRECISION,
    num_analysts INTEGER,

    -- Status
    is_confirmed BOOLEAN DEFAULT FALSE, -- Whether date is confirmed
    time_of_day VARCHAR(20),            -- 'BMO' (before market), 'AMC' (after market)

    PRIMARY KEY (ticker, earnings_date)
);

CREATE INDEX IF NOT EXISTS idx_earnings_calendar_date ON ml.earnings_calendar(earnings_date);
CREATE INDEX IF NOT EXISTS idx_earnings_calendar_ticker ON ml.earnings_calendar(ticker);

COMMENT ON TABLE ml.earnings_calendar IS 'Upcoming earnings announcements calendar';

-- View: Combined Earnings (actuals + estimates)
CREATE OR REPLACE VIEW ml.earnings_combined AS
SELECT
    a.ticker,
    a.period_end,
    a.filing_date,
    a.eps_diluted AS eps_actual,
    a.revenue AS revenue_actual,
    e.eps_consensus AS eps_estimate,
    e.revenue_consensus AS revenue_estimate,
    -- Calculate surprises
    (a.eps_diluted - e.eps_consensus) AS eps_surprise,
    ((a.eps_diluted - e.eps_consensus) / NULLIF(e.eps_consensus, 0) * 100) AS eps_surprise_pct,
    a.fiscal_year,
    a.fiscal_quarter
FROM ml.earnings_actuals a
LEFT JOIN ml.earnings_estimates e
    ON a.ticker = e.ticker
    AND a.period_end = e.period_end
    AND e.estimate_date <= a.filing_date  -- Point-in-time estimate
ORDER BY a.ticker, a.period_end DESC;

COMMENT ON VIEW ml.earnings_combined IS 'Actuals joined with estimates for surprise calculation';
