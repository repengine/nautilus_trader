#!/usr/bin/env python3
"""
Validate statistical correctness of Chow test results.

This script performs comprehensive validation of the Chow test results
stored in chow_test_results.json to ensure statistical validity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def validate_f_statistic(result: dict[str, Any]) -> tuple[bool, str]:
    """Validate F-statistic is non-negative."""
    f_stat = result["f_statistic"]
    if f_stat < 0:
        return False, f"F-statistic is negative: {f_stat}"
    return True, "F-statistic valid"


def validate_p_value(result: dict[str, Any]) -> tuple[bool, str]:
    """Validate p-value is in [0, 1] range."""
    p_value = result["p_value"]
    if not (0 <= p_value <= 1):
        return False, f"p-value out of range: {p_value}"
    return True, "p-value valid"


def validate_break_detection(result: dict[str, Any]) -> tuple[bool, str]:
    """Validate break detection logic matches p-value."""
    p_value = result["p_value"]
    detected = result["structural_break_detected"]
    expected = p_value < 0.05

    if detected != expected:
        return False, f"Break detection mismatch: detected={detected}, p-value={p_value}"
    return True, "Break detection consistent"


def validate_betas(result: dict[str, Any]) -> tuple[bool, str]:
    """Validate betas are finite and reasonable."""
    pre_betas = result["pre_break_betas"]
    post_betas = result["post_break_betas"]

    all_betas = list(pre_betas.values()) + list(post_betas.values())

    for beta in all_betas:
        if not np.isfinite(beta):
            return False, f"Beta is not finite: {beta}"
        if abs(beta) > 5:  # Extreme beta threshold
            return False, f"Beta magnitude too large: {beta}"

    return True, "Betas valid"


def validate_sample_sizes(result: dict[str, Any]) -> tuple[bool, str]:
    """Validate sample sizes are reasonable."""
    pre_n = result["pre_break_n"]
    post_n = result["post_break_n"]

    if pre_n < 20:
        return False, f"Pre-break sample too small: {pre_n}"
    if post_n < 20:
        return False, f"Post-break sample too small: {post_n}"

    return True, "Sample sizes valid"


def validate_r_squared(result: dict[str, Any]) -> tuple[bool, str]:
    """Validate R-squared values are in valid range."""
    r_squared_values = [
        result["pre_break_r_squared"],
        result["post_break_r_squared"],
        result["pooled_r_squared"],
    ]

    for r2 in r_squared_values:
        if r2 < -1 or r2 > 1:
            return False, f"R-squared out of range: {r2}"

    return True, "R-squared values valid"


def validate_critical_value(result: dict[str, Any]) -> tuple[bool, str]:
    """Validate critical value is reasonable for F-distribution."""
    critical = result["critical_value_5pct"]

    # For F(4, n) distribution at alpha=0.05, critical value typically 2.3-2.5
    if critical < 2.0 or critical > 3.0:
        return False, f"Critical value unexpected: {critical}"

    return True, "Critical value valid"


def main() -> None:
    """Run all validations and generate report."""
    json_path = Path("/home/nate/projects/nautilus_trader/playground/data/chow_test_results.json")

    if not json_path.exists():
        print(f"ERROR: Results file not found: {json_path}")
        return

    with json_path.open() as f:
        data = json.load(f)

    results = data["results"]

    print("=" * 80)
    print("CHOW TEST STATISTICAL VALIDATION")
    print("=" * 80)
    print()

    validators = [
        ("F-statistic >= 0", validate_f_statistic),
        ("p-value in [0, 1]", validate_p_value),
        ("Break detection consistent", validate_break_detection),
        ("Betas finite and reasonable", validate_betas),
        ("Sample sizes >= 20", validate_sample_sizes),
        ("R-squared in valid range", validate_r_squared),
        ("Critical value reasonable", validate_critical_value),
    ]

    all_passed = True

    for test_name, validator in validators:
        failures = []
        for result in results:
            sector = result["sector_id"]
            date = result["break_date"]
            passed, message = validator(result)

            if not passed:
                failures.append(f"  {sector} @ {date}: {message}")
                all_passed = False

        if failures:
            print(f"❌ {test_name}: FAILED")
            for failure in failures:
                print(failure)
        else:
            print(f"✅ {test_name}: PASSED")

    print()
    print("=" * 80)
    print("DETAILED STATISTICS")
    print("=" * 80)
    print()

    # Extract statistics
    f_stats = [r["f_statistic"] for r in results]
    p_values = [r["p_value"] for r in results]
    breaks = [r["structural_break_detected"] for r in results]

    print(f"Total tests: {len(results)}")
    print(f"Breaks detected: {sum(breaks)} ({sum(breaks)/len(results)*100:.1f}%)")
    print()
    print(f"F-statistic range: [{min(f_stats):.4f}, {max(f_stats):.4f}]")
    print(f"F-statistic mean: {np.mean(f_stats):.4f}")
    print()
    print(f"p-value range: [{min(p_values):.4f}, {max(p_values):.4f}]")
    print(f"p-value mean: {np.mean(p_values):.4f}")
    print()

    # Sector with break
    breaks_by_sector = {}
    for r in results:
        if r["structural_break_detected"]:
            sector = r["sector_id"]
            if sector not in breaks_by_sector:
                breaks_by_sector[sector] = []
            breaks_by_sector[sector].append({
                "date": r["break_date"],
                "f_stat": r["f_statistic"],
                "p_value": r["p_value"],
            })

    if breaks_by_sector:
        print("Sectors with breaks:")
        for sector, info_list in breaks_by_sector.items():
            for info in info_list:
                print(f"  {sector}: F={info['f_stat']:.2f}, p={info['p_value']:.4f}")
    else:
        print("No structural breaks detected")

    print()
    print("=" * 80)

    if all_passed:
        print("✅ ALL VALIDATIONS PASSED")
    else:
        print("❌ SOME VALIDATIONS FAILED")

    print("=" * 80)


if __name__ == "__main__":
    main()
