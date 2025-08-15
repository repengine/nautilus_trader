-- Initialize ML database schemas
-- This file is auto-executed when PostgreSQL container starts

-- Create database if not exists (run as superuser)
SELECT 'CREATE DATABASE nautilus'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'nautilus')\gexec

-- Connect to nautilus database
\c nautilus;

-- Create ML schema
CREATE SCHEMA IF NOT EXISTS ml;

-- Grant permissions
GRANT ALL ON SCHEMA ml TO postgres;

-- Set search path
SET search_path TO ml, public;

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE 'ML Schema initialization complete';
END
$$;