-- Migration: add artifact SHA-256 digest field to models table for integrity checks
ALTER TABLE models ADD COLUMN IF NOT EXISTS artifact_sha256_digest VARCHAR(64);

