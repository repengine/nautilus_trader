#!/usr/bin/env python3
"""
CORRECTED Analysis of Event-Driven Architecture.

After discovering that the topic filtering implementation is actually correct and the
docstring examples were wrong, this provides the corrected analysis.

"""

import os
import sys
import time
import uuid
from typing import Dict, List, Any

# Add the project root to sys.path
project_root = "/home/nate/projects/nautilus_trader"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.common.topic_filters import match_topic
from ml.common.in_memory_bus import InMemoryPublisher
from ml.actors.ml_domain_events import DomainEventBridge
from ml.consumers.idempotent import IdempotentConsumer
from ml.common.throttler import Throttler


def test_corrected_topic_filtering():
    """
    Test topic filtering with CORRECT expectations.
    """
    print("🔍 Testing Topic Filtering (CORRECTED)...")

    # These should all work according to actual MQTT semantics
    test_cases = [
        # Exact matching
        ("ml.features.updated.EURUSD.SIM", "ml.features.updated.EURUSD.SIM", True, "Exact match"),
        (
            "ml.features.updated.EURUSD.SIM",
            "ml.features.updated.BTCUSDT.SIM",
            False,
            "No match different instrument",
        ),
        # Star wildcards - each * matches exactly ONE token
        (
            "ml.features.updated.*.*",
            "ml.features.updated.EURUSD.SIM",
            True,
            "Two stars match two tokens",
        ),
        ("ml.features.updated.*", "ml.features.updated.EURUSD", True, "One star matches one token"),
        (
            "ml.features.updated.*",
            "ml.features.updated.EURUSD.SIM",
            False,
            "One star cannot match two tokens",
        ),
        ("ml.*.*.*.*", "ml.features.updated.EURUSD.SIM", True, "Four stars match four tokens"),
        # Hash wildcards - # matches zero or more tokens
        ("ml.features.updated.#", "ml.features.updated", True, "Hash matches zero tokens"),
        ("ml.features.updated.#", "ml.features.updated.EURUSD", True, "Hash matches one token"),
        (
            "ml.features.updated.#",
            "ml.features.updated.EURUSD.SIM",
            True,
            "Hash matches two tokens",
        ),
        # Mixed patterns
        ("ml.*.updated.#", "ml.features.updated", True, "Star + hash with zero"),
        ("ml.*.updated.#", "ml.features.updated.EURUSD.SIM", True, "Star + hash with two"),
        ("*.ml.*", "events.ml.FEATURE_COMPUTED", True, "Star at start and middle"),
    ]

    passed = 0
    failed = 0

    for pattern, topic, expected, description in test_cases:
        result = match_topic(pattern, topic)
        success = result == expected
        status = "✅" if success else "❌"

        print(f"  {status} {description}")
        print(
            f"      Pattern: '{pattern}' | Topic: '{topic}' | Expected: {expected} | Got: {result}"
        )

        if success:
            passed += 1
        else:
            failed += 1

    print(f"\n  Result: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = test_corrected_topic_filtering()
    raise SystemExit(0 if ok else 1)
