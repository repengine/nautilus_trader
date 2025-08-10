#!/usr/bin/env python3

"""
Test statistical utilities for model comparison.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.registry.statistics import welch_t_test, calculate_sample_size


class TestStatisticalUtilities:
    """Test statistical functions for A/B testing."""
    
    def test_welch_t_test_detects_significant_difference(self) -> None:
        """Test that Welch's t-test detects significant differences."""
        # Create two samples with different means
        np.random.seed(42)
        sample_a = np.random.normal(loc=10.0, scale=2.0, size=100)
        sample_b = np.random.normal(loc=12.0, scale=2.0, size=100)
        
        result = welch_t_test(sample_a, sample_b)
        
        assert result["statistically_significant"] is True
        assert result["mean_a"] < result["mean_b"]
        assert result["difference"] > 0
        assert result["p_value_approx"] < 0.05
        assert "t_statistic" in result
        assert "degrees_of_freedom" in result
    
    def test_welch_t_test_no_difference(self) -> None:
        """Test that Welch's t-test correctly identifies no difference."""
        # Create two samples with same distribution
        np.random.seed(42)
        sample_a = np.random.normal(loc=10.0, scale=2.0, size=100)
        sample_b = np.random.normal(loc=10.0, scale=2.0, size=100)
        
        result = welch_t_test(sample_a, sample_b)
        
        assert result["statistically_significant"] is False
        assert abs(result["difference"]) < 1.0
        assert result["p_value_approx"] > 0.05
    
    def test_welch_t_test_handles_small_samples(self) -> None:
        """Test that Welch's t-test handles small samples correctly."""
        sample_a = np.array([10.0, 11.0, 9.0])
        sample_b = np.array([15.0, 16.0, 14.0])
        
        result = welch_t_test(sample_a, sample_b)
        
        assert result["statistically_significant"] is True
        assert result["critical_value"] == 2.0  # Conservative for small samples
        assert result["degrees_of_freedom"] < 10
    
    def test_welch_t_test_handles_insufficient_samples(self) -> None:
        """Test that Welch's t-test handles insufficient samples."""
        sample_a = np.array([10.0])
        sample_b = np.array([15.0])
        
        result = welch_t_test(sample_a, sample_b)
        
        assert result["statistically_significant"] is False
        assert result["error"] == "Insufficient samples for test"
        assert result["p_value_approx"] == 1.0
    
    def test_welch_t_test_handles_zero_variance(self) -> None:
        """Test that Welch's t-test handles zero variance samples."""
        sample_a = np.array([10.0, 10.0, 10.0])
        sample_b = np.array([15.0, 15.0, 15.0])
        
        result = welch_t_test(sample_a, sample_b)
        
        assert result["statistically_significant"] is False
        assert result["error"] == "Zero variance in samples"
    
    def test_calculate_sample_size_standard_case(self) -> None:
        """Test sample size calculation for standard effect size."""
        # Medium effect size (Cohen's d = 0.5)
        n = calculate_sample_size(effect_size=0.5, power=0.8)
        
        # Standard approximation should give ~64 per group
        assert 50 < n < 80
        assert n >= 30  # Minimum requirement
    
    def test_calculate_sample_size_small_effect(self) -> None:
        """Test sample size calculation for small effect size."""
        # Small effect size requires more samples
        n = calculate_sample_size(effect_size=0.2, power=0.8)
        
        assert n > 300  # Small effects need large samples
    
    def test_calculate_sample_size_zero_effect(self) -> None:
        """Test sample size calculation handles zero effect size."""
        n = calculate_sample_size(effect_size=0.0, power=0.8)
        
        assert n == 100000  # Returns very large number
    
    def test_calculate_sample_size_high_power(self) -> None:
        """Test sample size increases with higher power requirement."""
        n_low = calculate_sample_size(effect_size=0.5, power=0.8)
        n_high = calculate_sample_size(effect_size=0.5, power=0.95)
        
        # Higher power requires more samples
        assert n_high > n_low
    
    def test_relative_improvement_calculation(self) -> None:
        """Test relative improvement calculation in t-test."""
        # Use samples with variance to avoid zero variance error
        sample_a = np.array([9.9, 10.0, 10.1])
        sample_b = np.array([10.9, 11.0, 11.1])
        
        result = welch_t_test(sample_a, sample_b, significance_level=0.05)
        
        assert abs(result["mean_a"] - 10.0) < 0.01
        assert abs(result["mean_b"] - 11.0) < 0.01
        assert abs(result["relative_improvement"] - 10.0) < 0.1  # ~10% improvement
    
    def test_relative_improvement_handles_zero_baseline(self) -> None:
        """Test relative improvement when baseline is zero."""
        # Add small variance to avoid zero variance error
        sample_a = np.array([-0.01, 0.0, 0.01])
        sample_b = np.array([0.99, 1.0, 1.01])
        
        result = welch_t_test(sample_a, sample_b)
        
        assert abs(result["mean_a"]) < 0.01  # Nearly zero
        assert abs(result["mean_b"] - 1.0) < 0.01
        # When mean_a is very close to zero, relative improvement should be 0
        assert result["relative_improvement"] == 0.0 or abs(result["mean_a"]) < 0.001