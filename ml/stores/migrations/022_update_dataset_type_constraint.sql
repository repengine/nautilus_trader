-- Update dataset type constraint to include macro and feature dataset types.
-- This keeps postgres registry bootstrap aligned with current DatasetType enums.

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
            ]::text[]
        )
    );
