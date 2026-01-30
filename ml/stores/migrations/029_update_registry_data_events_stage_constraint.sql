-- Align ml_registry.ml_data_events stage constraint with new event stages.

DO $$
DECLARE
    partition REGCLASS;
    is_local BOOLEAN;
BEGIN
    IF to_regclass('ml_registry.ml_data_events') IS NULL THEN
        RETURN;
    END IF;

    EXECUTE 'ALTER TABLE ml_registry.ml_data_events DROP CONSTRAINT IF EXISTS check_stage';
    EXECUTE $sql$
        ALTER TABLE ml_registry.ml_data_events
        ADD CONSTRAINT check_stage
        CHECK (
            (stage)::text = ANY (
                ARRAY[
                    'INGESTED',
                    'CATALOG_WRITTEN',
                    'FEATURE_COMPUTED',
                    'PREDICTION_EMITTED',
                    'SIGNAL_EMITTED',
                    'MODEL_INFERRED',
                    'ORDER_EVENT_EMITTED',
                    'RISK_HALT_EMITTED',
                    'REPLAY_SUMMARY_EMITTED'
                ]::text[]
            )
        )
    $sql$;

    FOR partition IN
        SELECT inhrelid::regclass
        FROM pg_inherits
        WHERE inhparent = 'ml_registry.ml_data_events'::regclass
    LOOP
        SELECT conislocal
            INTO is_local
            FROM pg_constraint
            WHERE conrelid = partition
              AND conname = 'check_stage';

        IF is_local THEN
            EXECUTE format('ALTER TABLE %s DROP CONSTRAINT check_stage', partition);
            EXECUTE format($sql$
                ALTER TABLE %s
                ADD CONSTRAINT check_stage
                CHECK (
                    (stage)::text = ANY (
                        ARRAY[
                            'INGESTED',
                            'CATALOG_WRITTEN',
                            'FEATURE_COMPUTED',
                            'PREDICTION_EMITTED',
                            'SIGNAL_EMITTED',
                            'MODEL_INFERRED',
                            'ORDER_EVENT_EMITTED',
                            'RISK_HALT_EMITTED',
                            'REPLAY_SUMMARY_EMITTED'
                        ]::text[]
                    )
                )
            $sql$, partition);
        END IF;
    END LOOP;
END $$;
