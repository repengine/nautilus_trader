#!/usr/bin/env python3
"""
Test and fix for the topic filtering bug.
"""


def match_topic_fixed(pattern: str, topic: str) -> bool:
    """
    Fixed version of match_topic that handles star wildcards correctly.
    """
    p_parts: list[str] = pattern.split(".") if pattern else []
    t_parts: list[str] = topic.split(".") if topic else []

    i = j = 0
    while i < len(p_parts) and j < len(t_parts):
        token = p_parts[i]
        if token == "#":
            # '#' at end matches the rest; if not last, attempt greedy match
            if i == len(p_parts) - 1:
                return True
            # Try to find a position in t_parts where the remainder matches
            i += 1
            while j <= len(t_parts):
                if match_topic_fixed(".".join(p_parts[i:]), ".".join(t_parts[j:])):
                    return True
                j += 1
            return False
        if token == "*":
            i += 1
            j += 1
            continue
        if token != t_parts[j]:
            return False
        i += 1
        j += 1

    # Exhaustion conditions - FIXED VERSION
    # If we've consumed all pattern tokens, we should have consumed all topic tokens
    if i == len(p_parts):
        return j == len(t_parts)

    # If remaining pattern is a single trailing '#', it matches empty remainder
    if i == len(p_parts) - 1 and p_parts[i] == "#":
        return True

    # If there are remaining pattern tokens (and they're not #), we failed to match
    return False


def test_topic_matching():
    """
    Test the fixed topic matching function.
    """

    test_cases = [
        # (pattern, topic, expected_result, description)
        (
            "ml.features.computed.*",
            "ml.features.computed.EURUSD.SIM",
            True,
            "Star at end should match one token",
        ),
        (
            "ml.*.emitted.*",
            "ml.signals.emitted.BTCUSDT.BINANCE",
            True,
            "Multiple stars should work",
        ),
        (
            "ml.features.updated.*",
            "ml.features.updated.EURUSD.SIM",
            True,
            "Documentation example 1",
        ),
        (
            "events.ml.FEATURE_COMPUTED.*",
            "events.ml.FEATURE_COMPUTED.EURUSD.SIM",
            True,
            "Documentation example 2",
        ),
        (
            "ml.models.created.BTCUSDT.BINANCE",
            "ml.models.created.BTCUSDT.BINANCE",
            True,
            "Exact match",
        ),
        ("ml.models.created.*", "ml.models.created", False, "Star should require one token"),
        (
            "events.ml.SIGNAL_EMITTED.#",
            "events.ml.SIGNAL_EMITTED",
            True,
            "Hash matches zero tokens",
        ),
        (
            "events.ml.SIGNAL_EMITTED.#",
            "events.ml.SIGNAL_EMITTED.GBPUSD.SIM",
            True,
            "Hash matches multiple tokens",
        ),
        ("*", "single", True, "Single star matches single token"),
        ("*.test", "hello.test", True, "Star with literal"),
        ("test.*", "test.event", True, "Literal with star"),
        ("ml.*.*.EURUSD.*", "ml.features.computed.EURUSD.SIM", True, "Multiple stars in pattern"),
        (
            "ml.features.*",
            "ml.features.computed.EURUSD.SIM",
            False,
            "Star matches exactly one, not multiple",
        ),
    ]

    print("Testing fixed topic matching implementation:")
    print("=" * 60)

    passed = 0
    failed = 0

    for pattern, topic, expected, description in test_cases:
        result = match_topic_fixed(pattern, topic)
        status = "✅ PASS" if result == expected else "❌ FAIL"

        print(f"{status} {description}")
        print(f"      Pattern: '{pattern}'")
        print(f"      Topic:   '{topic}'")
        print(f"      Expected: {expected}, Got: {result}")
        print()

        if result == expected:
            passed += 1
        else:
            failed += 1

    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = test_topic_matching()
    if success:
        print("🎉 All tests passed! The fix works correctly.")
    else:
        print("❌ Some tests failed. The fix needs more work.")
