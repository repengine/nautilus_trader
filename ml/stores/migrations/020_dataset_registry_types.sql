-- Migration: Expand dataset registry types for feature datasets.
-- Rollback: Restore previous check_dataset_type constraint (manual).

ALTER TABLE ml_dataset_registry
    DROP CONSTRAINT IF EXISTS check_dataset_type;

ALTER TABLE ml_dataset_registry
    ADD CONSTRAINT check_dataset_type CHECK (
        dataset_type IN (
            'BARS',
            'TRADES',
            'QUOTES',
            'MBP1',
            'TBBO',
            'FEATURES',
            'PREDICTIONS',
            'SIGNALS',
            'EARNINGS_ACTUALS',
            'EARNINGS_ESTIMATES',
            'MACRO_RELEASES',
            'MACRO_OBSERVATIONS',
            'EVENTS_CALENDAR',
            'MICRO_MINUTE_FEATURES',
            'L2_MINUTE_FEATURES'
        )
    );
