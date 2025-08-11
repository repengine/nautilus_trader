"""
Hypothesis configuration for ML property tests.

This file sets up profiles for different testing environments.
"""

from hypothesis import settings

# Register CI profile for faster tests in CI
settings.register_profile(
    "ci",
    max_examples=50,  # Reduced from default 100
    deadline=5000,  # 5 seconds
    print_blob=True,
    report_multiple_bugs=True,
    derandomize=True,  # Reproducible in CI
)

# Register dev profile for local development
settings.register_profile(
    "dev",
    max_examples=10,  # Quick feedback during development
    deadline=2000,  # 2 seconds
    print_blob=True,
)

# Register debug profile for finding bugs
settings.register_profile(
    "debug",
    max_examples=1000,  # Thorough testing
    deadline=None,  # No deadline
    print_blob=True,
    report_multiple_bugs=True,
    verbosity=2,  # Verbose output
)

# Load profile from environment
import os
profile = os.getenv("HYPOTHESIS_PROFILE", "default")
settings.load_profile(profile)
