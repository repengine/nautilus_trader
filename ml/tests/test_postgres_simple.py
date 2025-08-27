#!/usr/bin/env python3
"""
Simple test to verify PostgreSQL connection works.
"""

import os
from sqlalchemy import create_engine, text


def test_postgres_connection():
    """Test direct PostgreSQL connection."""
    # Get connection parameters from environment or defaults
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    database = os.getenv("POSTGRES_DB", "nautilus_test")
    
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    print(f"Testing connection: {connection_string}")
    
    # Create engine and test connection
    engine = create_engine(connection_string)
    
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        assert result.scalar() == 1
        print("✓ Connection successful")
        
        # Check tables
        result = conn.execute(text("""
            SELECT tablename 
            FROM pg_tables 
            WHERE schemaname = 'public'
            LIMIT 5
        """))
        tables = [row[0] for row in result]
        print(f"✓ Found {len(tables)} tables: {tables}")
    
    engine.dispose()
    print("✓ Test passed")


if __name__ == "__main__":
    test_postgres_connection()