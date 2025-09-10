#!/usr/bin/env python3
"""
Detailed analysis of the event-driven architecture implementation.

This analysis examines the gap between documented claims and actual implementation,
providing concrete evidence and recommendations.

"""

import os
import sys
from typing import Dict, List, Any

# Add the project root to sys.path
project_root = "/home/nate/projects/nautilus_trader"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import all event system components
from ml.config.events import Stage, Source, EventStatus
from ml.config.bus import MessageBusConfig
from ml.config.actor_bus import ActorBusConfig
from ml.common.message_bus import NoopPublisher, RedisStreamsPublisher, publisher_from_config
from ml.common.message_topics import (
    build_topic,
    build_stage_topic,
    map_stage_to_topic_segments,
    _normalize_instrument_id,
    build_topic_for_stage,
)
from ml.common.topic_filters import match_topic
from ml.common.in_memory_bus import InMemoryPublisher
from ml.common.throttler import Throttler
from ml.actors.ml_domain_events import DomainEventBridge
from ml.consumers.idempotent import IdempotentConsumer
from ml.consumers.redis_streams_consumer import RedisStreamsConsumer


class ArchitectureAnalyzer:
    """
    Analyze the event-driven architecture implementation.
    """

    def __init__(self):
        self.findings = {
            "working_features": [],
            "broken_features": [],
            "missing_features": [],
            "performance_issues": [],
            "integration_gaps": [],
        }

    def analyze_topic_filtering(self) -> Dict[str, Any]:
        """
        Analyze topic filtering functionality.
        """
        print("🔍 Analyzing Topic Filtering Implementation...")

        results = {
            "exact_matching": False,
            "star_wildcard": False,
            "hash_wildcard": False,
            "complex_patterns": False,
            "issues": [],
        }

        # Test exact matching
        try:
            exact_result = match_topic(
                "ml.features.computed.EURUSD.SIM", "ml.features.computed.EURUSD.SIM"
            )
            results["exact_matching"] = exact_result
            if exact_result:
                print("  ✅ Exact matching works")
            else:
                print("  ❌ Exact matching failed")
                results["issues"].append("Exact matching returns False for identical strings")
        except Exception as e:
            results["issues"].append(f"Exact matching error: {e}")
            print(f"  ❌ Exact matching error: {e}")

        # Test star wildcard - debug the implementation
        try:
            print("\n  🔍 Debugging star wildcard matching:")
            test_cases = [
                ("ml.features.computed.*", "ml.features.computed.EURUSD.SIM"),
                ("ml.*.emitted.*", "ml.signals.emitted.BTCUSDT.BINANCE"),
                ("*", "single"),
                ("*.test", "hello.test"),
            ]

            for pattern, topic in test_cases:
                result = match_topic(pattern, topic)
                print(f"    Pattern: '{pattern}' vs Topic: '{topic}' -> {result}")

                # Debug the tokenization
                p_parts = pattern.split(".")
                t_parts = topic.split(".")
                print(f"    Pattern parts: {p_parts}, Topic parts: {t_parts}")

            # Check if any star wildcard works
            star_works = match_topic("ml.features.computed.*", "ml.features.computed.EURUSD.SIM")
            results["star_wildcard"] = star_works

        except Exception as e:
            results["issues"].append(f"Star wildcard error: {e}")
            print(f"  ❌ Star wildcard error: {e}")

        # Test hash wildcard
        try:
            hash_result = match_topic("ml.features.#", "ml.features.computed.EURUSD.SIM")
            results["hash_wildcard"] = hash_result
            if hash_result:
                print("  ✅ Hash wildcard works")
            else:
                print("  ❌ Hash wildcard failed")
                results["issues"].append("Hash wildcard not working properly")
        except Exception as e:
            results["issues"].append(f"Hash wildcard error: {e}")
            print(f"  ❌ Hash wildcard error: {e}")

        return results

    def analyze_message_flow(self) -> Dict[str, Any]:
        """
        Analyze end-to-end message flow.
        """
        print("\n🔍 Analyzing Message Flow...")

        results = {
            "in_memory_pub_sub": False,
            "event_delivery": False,
        }
        # ... truncated for brevity (same as original script) ...
        return results


if __name__ == "__main__":
    analyzer = ArchitectureAnalyzer()
    analyzer.analyze_topic_filtering()
    analyzer.analyze_message_flow()
