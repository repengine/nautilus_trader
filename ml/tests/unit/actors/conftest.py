"""
Pytest configuration for actors unit tests.

These facade tests are actually integration tests that take 60+ seconds each.
They should be moved to integration tests, but for now we exclude them from
unit test collection to keep the unit test suite fast.
"""

# Exclude slow facade/integration tests from unit test collection
collect_ignore = [
    "test_base_ml_inference_actor_facade.py",
    "test_mlsignal_actor_facade.py",
    "test_facade_parity.py",
]
