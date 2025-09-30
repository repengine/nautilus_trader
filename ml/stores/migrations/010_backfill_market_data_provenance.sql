-- Backfill provenance metadata fields for market data

ALTER TABLE market_data
    ALTER COLUMN source_dataset SET DEFAULT 'UNKNOWN',
    ALTER COLUMN aggregation_mode SET DEFAULT 'native';

UPDATE market_data
SET
    source_dataset = COALESCE(source_dataset, 'UNKNOWN'),
    aggregation_mode = COALESCE(aggregation_mode, 'native')
WHERE source_dataset IS NULL
   OR aggregation_mode IS NULL;
