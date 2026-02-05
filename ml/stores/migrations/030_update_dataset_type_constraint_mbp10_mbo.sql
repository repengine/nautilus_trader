-- Extend dataset type constraint to include MBP10 and MBO.

ALTER TABLE public.ml_dataset_registry
    DROP CONSTRAINT IF EXISTS check_dataset_type;

ALTER TABLE public.ml_dataset_registry
    ADD CONSTRAINT check_dataset_type
    CHECK (
        (dataset_type)::text = ANY (
            ARRAY[
                'BARS',
                'TRADES',
                'QUOTES',
                'MBP1',
                'MBP10',
                'MBO',
                'TBBO',
                'FEATURES',
                'PREDICTIONS',
                'SIGNALS',
                'ORDER_EVENTS',
                'RISK_HALT_EVENTS',
                'REPLAY_SUMMARY',
                'EARNINGS_ACTUALS',
                'EARNINGS_ESTIMATES',
                'MACRO_RELEASES',
                'MACRO_OBSERVATIONS',
                'EVENTS_CALENDAR',
                'MICRO_MINUTE_FEATURES',
                'L2_MINUTE_FEATURES'
            ]::text[]
        )
    );
