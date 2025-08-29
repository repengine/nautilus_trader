#!/usr/bin/env python3
"""
Script to update conftest.py with improved connection management.

This applies the critical fixes identified in the health report
to resolve the 45-50% test pass rate issue.
"""

import os
import sys
from pathlib import Path


def update_conftest():
    """Update the existing conftest.py with critical improvements."""
    conftest_path = Path(__file__).parent / "conftest.py"

    # Read current conftest
    with open(conftest_path) as f:
        content = f.read()

    # Key improvements to add
    improvements = []

    # 1. Add session-scoped engine fixture if not present
    if '@pytest.fixture(scope="session")' not in content or "database_engine" not in content:
        improvements.append("""
# Session-scoped engine to prevent connection exhaustion
@pytest.fixture(scope="session")
def session_engine():
    \"\"\"Single engine for entire test session to prevent connection exhaustion.\"\"\"
    from ml.core.db_engine import EngineManager

    engine = EngineManager.get_engine(
        DATABASE_URL,
        pool_size=2,  # Conservative for tests
        max_overflow=3,  # Limited overflow
        pool_pre_ping=True,  # Test connections
    )
    yield engine
    EngineManager.dispose_all()
""")

    # 2. Improve cleanup_engines fixture
    if "cleanup_engines" in content:
        # Find and enhance the cleanup_engines fixture
        print("✓ cleanup_engines fixture already exists")
    else:
        improvements.append("""
@pytest.fixture(autouse=True)
def cleanup_engines():
    \"\"\"Clean up database engines after each test to prevent leaks.\"\"\"
    yield
    from ml.core.db_engine import EngineManager
    EngineManager.dispose_all()
""")

    # 3. Add connection monitoring
    if "connection_monitor" not in content:
        improvements.append("""
@pytest.fixture
def connection_monitor():
    \"\"\"Monitor database connections to detect leaks.\"\"\"
    from ml.core.db_engine import EngineManager
    import logging

    logger = logging.getLogger(__name__)

    # Log initial state
    status = EngineManager.get_pool_status(DATABASE_URL)
    initial = status.get("checked_out", 0) if status else 0

    yield

    # Check for leaks
    status = EngineManager.get_pool_status(DATABASE_URL)
    final = status.get("checked_out", 0) if status else 0

    if final > initial + 2:
        logger.warning(f"Connection leak: {initial} -> {final}")
""")

    # 4. Add parallel execution configuration
    if "pytest_configure" not in content:
        improvements.append("""
def pytest_configure(config):
    \"\"\"Configure pytest for optimal parallel execution.\"\"\"
    try:
        import xdist
        import multiprocessing

        # Use conservative parallelism to avoid overwhelming DB
        cpu_count = multiprocessing.cpu_count()
        optimal_workers = min(4, max(1, cpu_count // 2))

        if not config.getoption("--numprocesses", default=None):
            config.option.numprocesses = optimal_workers
    except ImportError:
        pass  # xdist not installed
""")

    # 5. Ensure proper test database initialization in pytest_sessionstart
    if "pytest_sessionstart" in content:
        print("✓ pytest_sessionstart already configured")

        # Check if it's using EngineManager properly
        if "EngineManager.get_engine" not in content:
            print("⚠ Update pytest_sessionstart to use EngineManager.get_engine")

    # Write improvements to a patch file
    if improvements:
        patch_path = Path(__file__).parent / "conftest_patch.py"
        with open(patch_path, "w") as f:
            f.write("# Add these improvements to conftest.py\n\n")
            f.write("\n".join(improvements))
        print(f"✓ Created patch file: {patch_path}")
        print(f"  Contains {len(improvements)} improvements to add")
    else:
        print("✓ All critical improvements already present")

    # Check for potential issues
    issues = []

    # Check if using create_engine directly instead of EngineManager
    if "from sqlalchemy import create_engine" in content:
        direct_usage = content.count("create_engine(")
        manager_usage = content.count("EngineManager.get_engine(")
        if direct_usage > manager_usage:
            issues.append(
                f"⚠ Found {direct_usage} direct create_engine calls vs "
                f"{manager_usage} EngineManager calls"
            )

    # Check pool settings
    if "pool_size=" in content:
        import re
        pool_sizes = re.findall(r"pool_size=(\d+)", content)
        for size in pool_sizes:
            if int(size) > 5:
                issues.append(f"⚠ pool_size={size} may be too large for tests")

    # Check for missing autouse on cleanup
    if "def cleanup" in content and "autouse=True" not in content:
        issues.append("⚠ Cleanup fixtures should use autouse=True")

    if issues:
        print("\nPotential issues found:")
        for issue in issues:
            print(f"  {issue}")

    print("\n✅ Analysis complete")
    print("\nNext steps:")
    print("1. Review conftest_patch.py for improvements to add")
    print("2. Manually integrate improvements into conftest.py")
    print("3. Run: pytest ml/tests/test_smoke.py -xvs")
    print("4. Check connection usage: watch -n1 \"psql -c 'SELECT count(*) FROM pg_stat_activity;'\"")


if __name__ == "__main__":
    update_conftest()
