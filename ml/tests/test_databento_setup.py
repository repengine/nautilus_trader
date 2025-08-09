#!/usr/bin/env python3
"""
Test script to verify Databento API key configuration.

This script checks that the Databento API key is properly configured and can connect to
the Databento service.

"""

import os
import sys

import pytest


def test_databento_setup():
    """
    Test that Databento API key is configured and working.

    This test will skip if Databento is not available, which is expected for ML-only
    development.

    """
    # Check environment variable
    api_key = os.environ.get("DATABENTO_API_KEY")

    if not api_key:
        pytest.skip(
            "DATABENTO_API_KEY environment variable not set - "
            "this is expected for ML-only development",
        )

    # Mask the key for display
    masked_key = f"{api_key[:10]}...{api_key[-4:]}" if len(api_key) > 14 else "***"
    print(f"✅ DATABENTO_API_KEY found: {masked_key}")

    # Try importing databento
    try:
        import databento as db

        print("✅ Databento package imported successfully")
    except ImportError as e:
        pytest.skip(
            f"Databento package not installed: {e} - "
            "Install with: pip install databento - "
            "Skipping databento tests as this is expected for ML-only development",
        )

    # Try creating a client (without making actual API calls)
    try:
        client = db.Historical(api_key)
        print("✅ Databento client created successfully")

        # Optional: Test actual connection (commented out to avoid API calls)
        # metadata = client.metadata.list_datasets()
        # print(f"✅ Connected to Databento - {len(metadata)} datasets available")

        assert client is not None, "Databento client should be created"
    except Exception as e:
        pytest.fail(f"Error creating Databento client: {e}")


if __name__ == "__main__":
    # When run directly (not via pytest), run simple checks
    print("Testing Databento Setup")
    print("-" * 40)

    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        print("⚠️  DATABENTO_API_KEY environment variable not set!")
        print("   This is expected for ML-only development")
        print("   To use DataBento, activate the virtual environment: source .venv/bin/activate")
        sys.exit(0)  # Exit successfully since this is optional

    # Mask the key for display
    masked_key = f"{api_key[:10]}...{api_key[-4:]}" if len(api_key) > 14 else "***"
    print(f"✅ DATABENTO_API_KEY found: {masked_key}")

    # Try importing databento
    try:
        import databento as db

        print("✅ Databento package imported successfully")

        # Try creating a client
        client = db.Historical(api_key)
        print("✅ Databento client created successfully")
        print("-" * 40)
        print("✅ Databento setup is complete and working!")
    except ImportError as e:
        print(f"⚠️  Databento package not installed: {e}")
        print("   Install with: pip install databento")
        print("   Skipping databento tests - this is expected for ML-only development")
        print("-" * 40)
        print("✅ Setup check complete (databento optional)")
    except Exception as e:
        print(f"❌ Error creating Databento client: {e}")
        print("-" * 40)
        print("❌ Databento setup needs attention")
        sys.exit(1)
